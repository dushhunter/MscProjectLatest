"""StoneReconNet: neural multi-view stone segmentation and registration.

Replaces the classical pipeline (RANSAC + ICP + pose graph) with a trained
neural model:
  1. PointNet++ encoder extracts per-view features.
  2. Segmentation head separates stone from floor/background.
  3. Multi-view attention fuses features across views (RAP-inspired DiTLayer).
  4. RPF-style rectified flow learns to register point clouds.

At inference, Euler ODE integration produces a registered stone point cloud.
Poisson surface reconstruction then creates a watertight mesh for volume
measurement -- no classical registration or TSDF needed.

Adapted from Rectified Point Flow (RPF), NeurIPS 2025 Spotlight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiViewAttention
from .encoder import PointNetPPEncoder, SegmentationHead


@dataclass
class StoneReconNetConfig:
    """Configuration for StoneReconNet."""

    sa1_npoint: int = 2048
    sa1_radius: float = 0.01
    sa1_nsample: int = 32
    sa1_mlp: List[int] = field(default_factory=lambda: [64, 64, 128])

    sa2_npoint: int = 512
    sa2_radius: float = 0.02
    sa2_nsample: int = 32
    sa2_mlp: List[int] = field(default_factory=lambda: [128, 128, 256])

    sa3_npoint: int = 128
    sa3_radius: float = 0.04
    sa3_nsample: int = 32
    sa3_mlp: List[int] = field(default_factory=lambda: [256, 256, 256])

    feature_dim: int = 256

    attn_embed_dim: int = 256
    attn_n_layers: int = 4
    attn_n_heads: int = 8
    attn_max_views: int = 32
    attn_qk_norm: bool = True
    attn_dropout: float = 0.0

    seg_hidden_dim: int = 128

    # RPF flow parameters
    flow_loss_type: str = "mse"
    timestep_sampling: str = "u_shaped"
    inference_sampling_steps: int = 10
    t_embed_dim: int = 64


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a learnable MLP (DDPM / RPF style).

    Maps a scalar timestep t in [0, 1] to a high-dimensional vector that the
    flow head can use to modulate its velocity prediction at different points
    along the ODE trajectory.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar timesteps -> (B, dim) embeddings."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FlowHead(nn.Module):
    """Predicts 3D velocity field conditioned on position, timestep, and
    attention features (proper RPF-style).

    Unlike a naive flow head that only sees encoded features, this module
    receives three inputs at each ODE step:
      - features: (B, M, D) fused multi-view attention features (constant
        across steps -- computed once by the encoder + attention stack).
      - x_t: (B, M, 3) current point positions along the flow trajectory.
        At t=1 this is pure noise; at t=0 it should be the registered cloud.
      - t: (B,) scalar timestep in [0, 1].

    This lets the model learn position-dependent velocity: given where points
    ARE right now (x_t) and how far along the trajectory we are (t), predict
    which direction they should move. Multi-step Euler integration then
    iteratively refines the positions from noise toward registration.
    """

    def __init__(self, embed_dim: int, t_embed_dim: int = 64):
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(t_embed_dim)
        input_dim = embed_dim + 3 + t_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )

    def forward(
        self,
        features: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, M, D) fused multi-view features.
            x_t: (B, M, 3) current flow positions.
            t: (B,) timestep scalars in [0, 1].

        Returns:
            (B, M, 3) predicted velocity at this (x_t, t).
        """
        B, M, _ = features.shape
        t_emb = self.t_embed(t).unsqueeze(1).expand(-1, M, -1)
        combined = torch.cat([features, x_t, t_emb], dim=-1)
        return self.net(combined)


class StoneReconNet(nn.Module):
    """Neural multi-view stone reconstruction with RPF-style flow registration.

    Pipeline:
      1. PointNet++ encodes each view's point cloud (shared weights).
      2. Segmentation head classifies stone vs floor/background per point.
      3. Multi-view attention fuses features across views.
      4. Flow head predicts per-point velocity field (RPF rectified flow).

    Training: segmentation BCE + flow velocity MSE.
    Inference: segment stone -> Euler ODE registration -> Poisson mesh -> volume.
    """

    def __init__(self, cfg: Optional[StoneReconNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = StoneReconNetConfig()
        self.cfg = cfg

        self.encoder = PointNetPPEncoder(
            sa1_npoint=cfg.sa1_npoint,
            sa1_radius=cfg.sa1_radius,
            sa1_nsample=cfg.sa1_nsample,
            sa1_mlp=cfg.sa1_mlp,
            sa2_npoint=cfg.sa2_npoint,
            sa2_radius=cfg.sa2_radius,
            sa2_nsample=cfg.sa2_nsample,
            sa2_mlp=cfg.sa2_mlp,
            sa3_npoint=cfg.sa3_npoint,
            sa3_radius=cfg.sa3_radius,
            sa3_nsample=cfg.sa3_nsample,
            sa3_mlp=cfg.sa3_mlp,
            feature_dim=cfg.feature_dim,
        )

        self.seg_head = SegmentationHead(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.seg_hidden_dim,
        )

        self.multi_view_attn = MultiViewAttention(
            input_dim=cfg.feature_dim,
            embed_dim=cfg.attn_embed_dim,
            n_layers=cfg.attn_n_layers,
            n_heads=cfg.attn_n_heads,
            max_views=cfg.attn_max_views,
            qk_norm=cfg.attn_qk_norm,
            dropout=cfg.attn_dropout,
        )

        self.flow_head = FlowHead(
            embed_dim=cfg.attn_embed_dim,
            t_embed_dim=cfg.t_embed_dim,
        )

        self.timestep_sampling = cfg.timestep_sampling
        self.flow_loss_type = cfg.flow_loss_type
        self.inference_sampling_steps = cfg.inference_sampling_steps

    # ------------------------------------------------------------------
    # RPF rectified flow methods (adapted from RPF modeling.py)
    # ------------------------------------------------------------------

    def _sample_timesteps(
        self, batch_size: int, device: torch.device,
        a: float = 4.0, eps: float = 0.01,
    ) -> torch.Tensor:
        """Sample timesteps with a U-shaped distribution (from RPF)."""
        if self.timestep_sampling == "u_shaped":
            u = torch.rand(batch_size, device=device) * 2 - 1
            u = torch.asinh(u * math.sinh(a)) / a
            u = (u + 1) / 2
        elif self.timestep_sampling == "uniform":
            u = torch.rand(batch_size, device=device)
        else:
            raise ValueError(f"Unknown timestep sampling: {self.timestep_sampling}")
        return u.clamp(eps, 1.0)

    @staticmethod
    def _compute_flow_target(
        x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rectified flow interpolation and velocity target (from RPF).

        Args:
            x_0: (B, M, 3) GT registered point positions.
            x_1: (B, M, 3) Gaussian noise.
            t: (B,) timesteps in [0, 1].

        Returns:
            x_t: (B, M, 3) interpolated positions.
            v_t: (B, M, 3) target velocity field.
        """
        t = t.view(-1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        v_t = x_1 - x_0
        return x_t, v_t

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward pass: encode, segment, fuse, and compute flow.

        Args:
            batch: dict with keys:
                - points: (B, N, 3) padded point positions
                - view_ids: (B, N) view assignment per point
                - pad_mask: (B, N) True for padded positions
                - n_points: (B,) actual number of points per sample
                - gt_points_registered: (B, N, 3) clean GT positions (flow target)
        """
        points = batch["points"]
        view_ids = batch["view_ids"]
        pad_mask = batch["pad_mask"]
        B, N, _ = points.shape

        sa_xyz, sa_feat, global_feat = self.encoder(points, mask=pad_mask)

        seg_logits = self.seg_head(points, sa_xyz, sa_feat)
        seg_logits = seg_logits.masked_fill(pad_mask, 0.0)

        sa_view_ids = self._downsample_view_ids(points, sa_xyz, view_ids)
        M = sa_xyz.shape[1]
        sa_pad_mask = torch.zeros(B, M, dtype=torch.bool, device=points.device)

        fused = self.multi_view_attn(sa_feat, sa_xyz, sa_view_ids, sa_pad_mask)

        output = {
            "seg_logits": seg_logits,
            "fused_features": fused,
            "sa_xyz": sa_xyz,
        }

        if "gt_points_registered" in batch:
            gt_reg = batch["gt_points_registered"]
            x_0_sa = self._downsample_gt_points(points, sa_xyz, gt_reg)

            timesteps = self._sample_timesteps(B, points.device)
            x_1 = torch.randn_like(x_0_sa)
            x_t, v_t = self._compute_flow_target(x_0_sa, x_1, timesteps)

            v_pred = self.flow_head(fused, x_t, timesteps)

            output["v_pred"] = v_pred
            output["v_t"] = v_t
            output["t"] = timesteps
            output["x_0"] = x_0_sa
            output["x_1"] = x_1
            output["x_t"] = x_t

        return output

    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inference forward: encode, segment, fuse features."""
        points = batch["points"]
        view_ids = batch["view_ids"]
        pad_mask = batch["pad_mask"]
        B, N, _ = points.shape

        sa_xyz, sa_feat, global_feat = self.encoder(points, mask=pad_mask)

        seg_logits = self.seg_head(points, sa_xyz, sa_feat)
        seg_logits = seg_logits.masked_fill(pad_mask, 0.0)

        sa_view_ids = self._downsample_view_ids(points, sa_xyz, view_ids)
        M = sa_xyz.shape[1]
        sa_pad_mask = torch.zeros(B, M, dtype=torch.bool, device=points.device)

        fused = self.multi_view_attn(sa_feat, sa_xyz, sa_view_ids, sa_pad_mask)

        return {
            "seg_logits": seg_logits,
            "fused_features": fused,
            "sa_xyz": sa_xyz,
            "sa_feat": sa_feat,
            "sa_view_ids": sa_view_ids,
        }

    @torch.inference_mode()
    def sample_rectified_flow(
        self, batch: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler ODE integration for flow-based registration (from RPF).

        Integrates from t=1 (noise) toward t=0 (registered) to produce
        the registered point cloud at the SA level.

        At each step the flow head sees the CURRENT position x_t and the
        CURRENT timestep t, allowing position-dependent velocity prediction
        that iteratively refines the point cloud toward registration.

        Returns:
            registered_points: (B, M, 3) registered point positions.
            seg_logits: (B, N) per-point segmentation logits.
        """
        if num_steps is None:
            num_steps = self.inference_sampling_steps

        inf_out = self.forward_inference(batch)
        fused = inf_out["fused_features"]
        sa_xyz = inf_out["sa_xyz"]
        B, M, _ = sa_xyz.shape

        x_t = torch.randn(B, M, 3, device=sa_xyz.device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t_tensor = torch.full((B,), t_val, device=sa_xyz.device)
            v_pred = self.flow_head(fused, x_t, t_tensor)
            x_t = x_t - v_pred * dt

        return x_t, inf_out["seg_logits"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _downsample_view_ids(
        self, xyz_full: torch.Tensor, xyz_sa: torch.Tensor, view_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assign view IDs to SA-level points via nearest neighbor."""
        dists = torch.cdist(xyz_sa, xyz_full)
        nn_idx = dists.argmin(dim=-1)
        return view_ids.gather(1, nn_idx)

    def _downsample_gt_points(
        self, xyz_full: torch.Tensor, xyz_sa: torch.Tensor, gt_full: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample GT registered points to SA resolution via nearest neighbor."""
        dists = torch.cdist(xyz_sa, xyz_full)
        nn_idx = dists.argmin(dim=-1)
        B, M = nn_idx.shape
        batch_idx = torch.arange(B, device=nn_idx.device).unsqueeze(1).expand(B, M)
        return gt_full[batch_idx, nn_idx]

    def get_stone_points(
        self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """Extract per-sample stone-only point clouds using segmentation."""
        points = batch["points"]
        seg_probs = torch.sigmoid(output["seg_logits"])
        pad_mask = batch["pad_mask"]
        n_pts = batch["n_points"]

        result = []
        B = points.shape[0]
        for i in range(B):
            n = int(n_pts[i].item())
            mask = (seg_probs[i, :n] > 0.5) & (~pad_mask[i, :n])
            stone_pts = points[i, :n][mask]
            result.append(stone_pts)
        return result

    def freeze_encoder(self):
        """Freeze the PointNet++ encoder (RPF-style frozen encoder support)."""
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the PointNet++ encoder."""
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per module."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "encoder": _count(self.encoder),
            "seg_head": _count(self.seg_head),
            "multi_view_attn": _count(self.multi_view_attn),
            "flow_head": _count(self.flow_head),
            "total": _count(self),
        }
