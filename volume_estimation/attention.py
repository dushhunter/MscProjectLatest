"""RAP-inspired multi-view attention for point cloud fusion.

Implements part-wise (within-view) and global (cross-view) attention,
with sinusoidal 3D position encoding and learnable view embeddings,
following the DiTLayer design from RAP's PointCloudDiT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding3D(nn.Module):
    """Sinusoidal encoding of 3D coordinates, as used in RAP.

    Maps each (x, y, z) coordinate to a higher-dimensional feature
    using sinusoidal functions at multiple frequencies.
    """

    def __init__(self, embed_dim: int, max_freq_log2: float = 6.0, n_freq: int = 0):
        super().__init__()
        if n_freq == 0:
            n_freq = embed_dim // 6
        self.n_freq = n_freq
        self.out_dim = 3 * 2 * n_freq

        freqs = 2.0 ** torch.linspace(0, max_freq_log2, n_freq)
        self.register_buffer("freqs", freqs)

        self.proj = nn.Linear(self.out_dim, embed_dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (..., 3) coordinates.

        Returns:
            (..., embed_dim) position encodings.
        """
        shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(-1, 3)

        scaled = xyz_flat.unsqueeze(-1) * self.freqs.unsqueeze(0)
        sin_feat = torch.sin(scaled).reshape(-1, 3 * self.n_freq)
        cos_feat = torch.cos(scaled).reshape(-1, 3 * self.n_freq)
        enc = torch.cat([sin_feat, cos_feat], dim=-1)

        out = self.proj(enc)
        return out.reshape(*shape, -1)


class PartWiseAttention(nn.Module):
    """Self-attention within each view's points (part-wise, from RAP DiTLayer).

    Points from the same view attend only to each other, respecting
    the multi-view structure. Uses multi-head attention with QK-norm.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, qk_norm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, view_ids: torch.Tensor, pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) point features.
            view_ids: (B, N) integer view assignment for each point.
            pad_mask: (B, N) True for padded positions.

        Returns:
            (B, N, D) attended features.
        """
        residual = x
        x = self.norm(x)

        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        view_mask = view_ids.unsqueeze(1) != view_ids.unsqueeze(2)
        view_mask = view_mask | pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        attn = attn.masked_fill(view_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)
        return residual + out


class GlobalCrossAttention(nn.Module):
    """Global attention across all views (cross-view, from RAP DiTLayer).

    All points attend to all other points regardless of view assignment,
    enabling cross-view information exchange for implicit registration.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, qk_norm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) point features.
            pad_mask: (B, N) True for padded positions.

        Returns:
            (B, N, D) attended features.
        """
        residual = x
        x = self.norm(x)

        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        key_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)
        return residual + out


class GEGLUFFN(nn.Module):
    """Gated GELU feed-forward network (from RAP DiTLayer / diffusers)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim == 0:
            hidden_dim = 4 * embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(gate) * value
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class MultiViewAttentionBlock(nn.Module):
    """One attention block combining part-wise + global + FFN (RAP DiTLayer pattern)."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        ffn_hidden_dim: int = 0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.part_attn = PartWiseAttention(embed_dim, n_heads, qk_norm)
        self.global_attn = GlobalCrossAttention(embed_dim, n_heads, qk_norm)
        self.ffn = GEGLUFFN(embed_dim, ffn_hidden_dim, dropout)

    def forward(
        self, x: torch.Tensor, view_ids: torch.Tensor, pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.part_attn(x, view_ids, pad_mask)
        x = self.global_attn(x, pad_mask)
        x = self.ffn(x)
        return x


class MultiViewAttention(nn.Module):
    """Full multi-view attention stack (multiple DiTLayer-style blocks).

    Combines sinusoidal 3D position encoding + learnable view embeddings
    with a stack of part-wise/global attention blocks.
    """

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_views: int = 32,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.pos_enc = SinusoidalPositionEncoding3D(embed_dim)
        self.view_embed = nn.Embedding(max_views, embed_dim)

        self.blocks = nn.ModuleList([
            MultiViewAttentionBlock(embed_dim, n_heads, 0, qk_norm, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        view_ids: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, N, C_in) point features from encoder.
            xyz: (B, N, 3) point positions.
            view_ids: (B, N) view assignment per point.
            pad_mask: (B, N) True for padded positions.

        Returns:
            (B, N, embed_dim) fused multi-view features.
        """
        x = self.input_proj(features)

        pos = self.pos_enc(xyz)
        x = x + pos

        view_emb = self.view_embed(view_ids.clamp(0, self.view_embed.num_embeddings - 1))
        x = x + view_emb

        for block in self.blocks:
            x = block(x, view_ids, pad_mask)

        x = self.final_norm(x)
        return x
