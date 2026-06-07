"""StoneCompletionNet -- PointSea-inspired segmentation + completion model.

Architecture:
  1. PointNet++ encoder: hierarchical point feature extraction
  2. Segmentation head: per-point stone/floor classification
  3. SVFNet: self-view fusion (3-view depth projection + ResNet-18 + PointNet++)
  4. SDG: self-structure dual-generator for coarse-to-fine completion

References:
  - PointSea (IJCV 2025): Self-view fusion, SDG dual-path refinement
  - SVDFormer (ICCV 2023): Original architecture
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .pcviews import PCViews

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CompletionConfig:
    input_points: int = 2048
    coarse_points: int = 512
    fine1_points: int = 2048
    fine2_points: int = 8192
    feat_dim: int = 256
    view_feat_dim: int = 256
    num_sa_heads: int = 4
    sdg_heads: int = 4
    pc_resolution: int = 224
    camera_dist: float = 1.5
    seg_feat_dim: int = 128
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# PointNet++ building blocks
# ---------------------------------------------------------------------------

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(B,N,C), (B,M,C) -> (B,N,M) squared distances."""
    return (
        torch.sum(src ** 2, dim=-1, keepdim=True)
        + torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(-1, -2)
        - 2 * torch.matmul(src, dst.transpose(-1, -2))
    )


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling. (B,N,3) -> (B,npoint) indices."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_idx, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. (B,N,C), (B,S) -> (B,S,C)."""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, device=points.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def _knn(src: torch.Tensor, dst: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN indices. (B,N,C), (B,M,C) -> (B,N,k)."""
    dists = _square_distance(src, dst)
    _, indices = dists.topk(k, dim=-1, largest=False)
    return indices


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction with FPS + kNN + shared MLP."""

    def __init__(self, npoint: int, k: int, in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers = []
        last_ch = in_channel + 3
        for out_ch in mlp:
            layers.append(nn.Conv1d(last_ch, out_ch, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_dim = mlp[-1]

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C) or None
        Returns:
            new_xyz: (B, npoint, 3)
            new_features: (B, npoint, out_dim)
        """
        fps_idx = _farthest_point_sample(xyz, self.npoint)
        new_xyz = _index_points(xyz, fps_idx)

        knn_idx = _knn(new_xyz, xyz, self.k)

        grouped_xyz = _index_points(xyz, knn_idx.view(xyz.shape[0], -1)).view(
            xyz.shape[0], self.npoint, self.k, 3
        )
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_feat = _index_points(features, knn_idx.view(xyz.shape[0], -1)).view(
                xyz.shape[0], self.npoint, self.k, -1
            )
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_xyz

        grouped = grouped.view(xyz.shape[0], self.npoint, self.k, -1)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()
        B, C, S, K = grouped.shape
        grouped = grouped.view(B, C, S * K)
        grouped = self.mlp(grouped)
        grouped = grouped.view(B, -1, S, K)
        new_features = grouped.max(dim=-1)[0]
        new_features = new_features.permute(0, 2, 1).contiguous()

        return new_xyz, new_features


class PointNetPPEncoder(nn.Module):
    """3-level PointNet++ encoder."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.sa1 = SetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 32, 128, [128, 128, feat_dim])
        self.sa3 = SetAbstraction(1, 128, feat_dim, [feat_dim, feat_dim * 2, feat_dim * 2])
        self.out_dim = feat_dim * 2

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_feat: (B, out_dim)
            sa1_feat: (B, 512, 128) -- for segmentation
            sa1_xyz: (B, 512, 3) -- for segmentation
        """
        sa1_xyz, sa1_feat = self.sa1(xyz)
        sa2_xyz, sa2_feat = self.sa2(sa1_xyz, sa1_feat)
        _, sa3_feat = self.sa3(sa2_xyz, sa2_feat)
        global_feat = sa3_feat.squeeze(1)
        return global_feat, sa1_feat, sa1_xyz


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Per-point binary segmentation (stone vs floor).

    Propagates global features back to the original point resolution
    using nearest-neighbor interpolation, then classifies each point.
    """

    def __init__(self, global_dim: int, local_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim + local_dim + 3, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        global_feat: torch.Tensor,
        sa1_feat: torch.Tensor,
        sa1_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) original points
            global_feat: (B, D_g)
            sa1_feat: (B, 512, D_l)
            sa1_xyz: (B, 512, 3)
        Returns:
            logits: (B, N) per-point logits
        """
        B, N, _ = xyz.shape

        knn_idx = _knn(xyz, sa1_xyz, 3)
        neighbor_feat = _index_points(sa1_feat, knn_idx.view(B, -1)).view(B, N, 3, -1)
        local_interp = neighbor_feat.mean(dim=2)

        global_exp = global_feat.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([xyz, global_exp, local_interp], dim=-1)

        B, N, C = combined.shape
        logits = self.mlp(combined.view(B * N, C)).view(B, N)
        return logits


# ---------------------------------------------------------------------------
# 2D feature extractor (ResNet-18 on depth images)
# ---------------------------------------------------------------------------

class DepthResNet(nn.Module):
    """ResNet-18 adapted for single-channel depth images."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B*V, 1, H, W) -> (B*V, out_dim)."""
        feat = self.backbone(x).flatten(1)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# SVFNet: Self-View Fusion Network
# ---------------------------------------------------------------------------

class ViewAttention(nn.Module):
    """Multi-head attention to fuse 3D and 2D view features."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, S, D) with self-attention."""
        out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + out)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens


class SVFNet(nn.Module):
    """Self-View Fusion Network.

    Combines 3D point features with 2D depth features from self-projected
    views, then decodes a coarse point cloud.
    """

    def __init__(self, cfg: CompletionConfig, global_feat_dim: int = 512):
        super().__init__()
        self.pcviews = PCViews(cfg.pc_resolution, cfg.camera_dist)
        self.depth_cnn = DepthResNet(cfg.view_feat_dim)
        self.global_feat_dim = global_feat_dim

        self.proj_3d = nn.Linear(global_feat_dim, cfg.feat_dim)
        self.proj_2d = nn.Linear(cfg.view_feat_dim, cfg.feat_dim)
        self.view_attn = ViewAttention(cfg.feat_dim, cfg.num_sa_heads)

        self.coarse_decoder = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.feat_dim),
            nn.GELU(),
            nn.Linear(cfg.feat_dim, cfg.feat_dim),
            nn.GELU(),
            nn.Linear(cfg.feat_dim, cfg.coarse_points * 3),
        )
        self.coarse_points = cfg.coarse_points

    def forward(
        self, partial: torch.Tensor, global_feat_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            partial: (B, N, 3) partial stone cloud (centered)
            global_feat_3d: (B, D_enc) from PointNet++ encoder

        Returns:
            fused_feat: (B, feat_dim) global fused feature
            coarse: (B, coarse_points, 3) coarse completion
        """
        B = partial.shape[0]

        global_proj = self.proj_3d(global_feat_3d)

        depth_imgs = self.pcviews.get_img(partial)
        view_feats = self.depth_cnn(depth_imgs)
        view_feats = view_feats.view(PCViews.NUM_VIEWS, B, -1).permute(1, 0, 2)
        view_feats = self.proj_2d(view_feats)

        tokens = torch.cat(
            [global_proj.unsqueeze(1), view_feats], dim=1
        )
        fused = self.view_attn(tokens)
        fused_feat = fused[:, 0, :]

        coarse = self.coarse_decoder(fused_feat).view(B, self.coarse_points, 3)
        return fused_feat, coarse


# ---------------------------------------------------------------------------
# SDG: Self-structure Dual Generator
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for scalar incompleteness scores."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N) -> (B, N, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=x.device, dtype=x.dtype)
            / half
        )
        args = x.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class StructureAnalysis(nn.Module):
    """Path A: incompleteness-aware self-attention for shape prior."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.inc_pe = SinusoidalPE(dim)
        self.inc_proj = nn.Linear(dim, dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, coarse_feat: torch.Tensor, incompleteness: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coarse_feat: (B, M, D)
            incompleteness: (B, M) scalar distance to nearest partial point
        Returns:
            (B, M, D)
        """
        inc_emb = self.inc_proj(self.inc_pe(incompleteness))
        q = coarse_feat + inc_emb
        out, _ = self.self_attn(q, q, q)
        x = self.norm(coarse_feat + out)
        x = self.norm2(x + self.ff(x))
        return x


class SimilarityAlignment(nn.Module):
    """Path B: cross-attention finding similar structures in partial input."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, query_feat: torch.Tensor, local_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_feat: (B, M, D) from structure analysis
            local_feat: (B, K, D) from partial input local encoder
        Returns:
            (B, M, D)
        """
        out, _ = self.cross_attn(query_feat, local_feat, local_feat)
        x = self.norm(query_feat + out)
        x = self.norm2(x + self.ff(x))
        return x


class LocalEncoder(nn.Module):
    """EdgeConv-style local feature extractor for the partial input."""

    def __init__(self, in_dim: int = 3, out_dim: int = 256, k: int = 16):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim * 2, 64, 1), nn.BatchNorm1d(64), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64 * 2, 128, 1), nn.BatchNorm1d(128), nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128 * 2, out_dim, 1), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    def _edge_conv(
        self, x: torch.Tensor, feat: torch.Tensor, conv: nn.Module
    ) -> torch.Tensor:
        """Single EdgeConv layer."""
        B, N, C = feat.shape
        knn_idx = _knn(x, x, self.k)
        neighbors = _index_points(feat, knn_idx.view(B, -1)).view(B, N, self.k, C)
        center = feat.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([center, neighbors - center], dim=-1)

        edge = edge.view(B, N * self.k, -1).permute(0, 2, 1)
        edge = conv(edge)
        edge = edge.view(B, -1, N, self.k).max(dim=-1)[0]
        return edge.permute(0, 2, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) -> (B, N, out_dim)."""
        f1 = self._edge_conv(xyz, xyz, self.conv1)
        f2 = self._edge_conv(xyz, f1, self.conv2)
        f3 = self._edge_conv(xyz, f2, self.conv3)
        return f3


class SDGStage(nn.Module):
    """Single Self-structure Dual Generator stage.

    Dual-path refinement:
      Path A (StructureAnalysis): learned shape prior for missing regions
      Path B (SimilarityAlignment): copy existing local structures
      Fusion: adaptive per-point gate
    """

    def __init__(self, feat_dim: int, global_dim: int, ratio: int, heads: int = 4):
        super().__init__()
        self.ratio = ratio
        self.point_embed = nn.Linear(3, feat_dim)
        self.global_proj = nn.Linear(global_dim, feat_dim)

        self.structure_analysis = StructureAnalysis(feat_dim, heads)
        self.similarity_alignment = SimilarityAlignment(feat_dim, heads)

        self.fusion_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

        self.offset_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )

    def forward(
        self,
        coarse: torch.Tensor,
        global_feat: torch.Tensor,
        partial: torch.Tensor,
        local_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coarse: (B, M, 3) input points to refine
            global_feat: (B, D_g) global feature
            partial: (B, N, 3) original partial cloud
            local_feat: (B, N, D) local features of partial
        Returns:
            fine: (B, M*ratio, 3)
        """
        B, M, _ = coarse.shape

        coarse_feat = self.point_embed(coarse) + self.global_proj(global_feat).unsqueeze(1)

        dists = _square_distance(coarse, partial).min(dim=-1)[0]
        sigma = 0.2
        incompleteness = dists / (sigma ** 2)

        path_a = self.structure_analysis(coarse_feat, incompleteness)
        path_b = self.similarity_alignment(path_a, local_feat)

        gate = self.fusion_gate(torch.cat([path_a, path_b], dim=-1))
        fused = gate * path_a + (1 - gate) * path_b

        expanded = coarse.unsqueeze(2).expand(-1, -1, self.ratio, -1).reshape(B, M * self.ratio, 3)
        fused_expanded = fused.unsqueeze(2).expand(-1, -1, self.ratio, -1).reshape(B, M * self.ratio, -1)

        offsets = self.offset_head(fused_expanded)
        fine = expanded + offsets

        return fine


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class StoneCompletionNet(nn.Module):
    """End-to-end stone segmentation + completion network.

    Forward pass:
      1. PointNet++ encoder -> global + local features
      2. Segmentation head -> per-point stone/floor labels
      3. SVFNet (self-view depth fusion) -> fused global feat + coarse completion
      4. SDG stage 1 -> fine1 completion
      5. SDG stage 2 -> fine2 completion (final output)

    At inference, the completed cloud is passed to NKSR for mesh generation.
    """

    def __init__(self, cfg: Optional[CompletionConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = CompletionConfig()
        self.cfg = cfg

        self.encoder = PointNetPPEncoder(cfg.feat_dim)

        self.seg_head = SegmentationHead(
            global_dim=self.encoder.out_dim,
            local_dim=self.encoder.sa1.out_dim,
            hidden=cfg.seg_feat_dim,
        )

        self.svfnet = SVFNet(cfg, global_feat_dim=self.encoder.out_dim)

        self.local_encoder = LocalEncoder(in_dim=3, out_dim=cfg.feat_dim, k=16)

        step1 = cfg.fine1_points // cfg.coarse_points
        step2 = cfg.fine2_points // cfg.fine1_points

        self.sdg1 = SDGStage(cfg.feat_dim, self.encoder.out_dim, ratio=step1, heads=cfg.sdg_heads)
        self.sdg2 = SDGStage(cfg.feat_dim, self.encoder.out_dim, ratio=step2, heads=cfg.sdg_heads)

    def forward(
        self,
        points: torch.Tensor,
        partial_stone: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) full scene points (stone + floor) for segmentation.
            partial_stone: (B, M, 3) pre-segmented stone points for completion.
                           During training, this uses GT mask. At inference,
                           uses predicted segmentation.
        Returns:
            dict with keys:
              seg_logits: (B, N)
              coarse: (B, C, 3)
              fine1: (B, F1, 3)
              fine2: (B, F2, 3)
        """
        global_feat, sa1_feat, sa1_xyz = self.encoder(points)
        seg_logits = self.seg_head(points, global_feat, sa1_feat, sa1_xyz)

        if partial_stone is None:
            seg_mask = (seg_logits > 0).float()
            partial_stone = points * seg_mask.unsqueeze(-1)

        local_feat = self.local_encoder(partial_stone)
        fused_feat, coarse = self.svfnet(partial_stone, global_feat)

        merge_xyz = torch.cat([partial_stone, coarse], dim=1)
        if merge_xyz.shape[1] > self.cfg.coarse_points:
            fps_idx = _farthest_point_sample(merge_xyz, self.cfg.coarse_points)
            merge_xyz = _index_points(merge_xyz, fps_idx)

        fine1 = self.sdg1(merge_xyz, global_feat, partial_stone, local_feat)
        fine2 = self.sdg2(fine1, global_feat, partial_stone, local_feat)

        return {
            "seg_logits": seg_logits,
            "coarse": coarse,
            "fine1": fine1,
            "fine2": fine2,
        }

    @torch.no_grad()
    def complete(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only: segment + complete.

        Returns:
            seg_mask: (B, N) boolean mask
            completed: (B, fine2_points, 3) completed stone cloud
        """
        self.eval()
        out = self.forward(points)
        seg_mask = out["seg_logits"] > 0
        return seg_mask, out["fine2"]
