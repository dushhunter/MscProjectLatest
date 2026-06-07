"""StoneVolumeNet: end-to-end multi-view stone volume estimator.

Combines PointNet++ per-view encoder, per-point segmentation head,
RPF-style rectified flow registration, and a volume regression MLP.

The rectified flow branch (adapted from Rectified Point Flow, NeurIPS 2025)
learns a velocity field that transports noisy points to their GT registered
positions. At inference time, Euler ODE integration produces the registered
point cloud from which volume is estimated.
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
class StoneVolumeNetConfig:
    """Configuration for StoneVolumeNet."""

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
    vol_hidden_dim: int = 256

    # RPF flow parameters
    flow_loss_type: str = "mse"
    timestep_sampling: str = "u_shaped"
    inference_sampling_steps: int = 10


class VolumeHead(nn.Module):
    """Predicts stone volume from aggregated multi-view features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: (B, D) global feature vector.

        Returns:
            (B,) predicted volume (always positive via Softplus).
        """
        return self.net(global_feat).squeeze(-1)


class FlowHead(nn.Module):
    """Predicts 3D velocity field from attention features (RPF-style)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, M, D) -> (B, M, 3) velocity prediction."""
        return self.net(features)


class StoneVolumeNet(nn.Module):
    """End-to-end multi-view stone volume estimation with RPF-style flow.

    Pipeline:
      1. PointNet++ encodes each view's point cloud (shared weights).
      2. Segmentation head predicts stone vs background per point.
      3. Multi-view attention fuses features (also serves as the flow model).
      4. Flow head predicts per-point velocity field (RPF rectified flow).
      5. Volume MLP predicts the stone volume from global features.

    During training the flow branch learns to transport noisy points to their
    GT registered positions via rectified flow (following RPF, NeurIPS 2025).
    At inference, Euler ODE integration produces the registered point cloud.
    """

    def __init__(self, cfg: Optional[StoneVolumeNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = StoneVolumeNetConfig()
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

        self.flow_head = FlowHead(embed_dim=cfg.attn_embed_dim)

        self.volume_head = VolumeHead(
            input_dim=cfg.attn_embed_dim,
            hidden_dim=cfg.vol_hidden_dim,
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
        """Full forward pass for training (includes flow branch).

        Args:
            batch: dict with keys:
                - points: (B, N, 3) padded point positions
                - view_ids: (B, N) view assignment per point
                - pad_mask: (B, N) True for padded positions
                - n_points: (B,) actual number of points per sample
                - gt_points_registered: (B, N, 3) clean GT positions (flow target)

        Returns:
            dict with seg_logits, pred_volume, and flow outputs (v_pred, v_t, t).
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

        fused_global = fused.max(dim=1)[0]
        pred_volume = self.volume_head(fused_global)

        output = {
            "seg_logits": seg_logits,
            "pred_volume": pred_volume,
            "fused_features": fused,
            "sa_xyz": sa_xyz,
        }

        if "gt_points_registered" in batch:
            gt_reg = batch["gt_points_registered"]
            x_0_sa = self._downsample_gt_points(points, sa_xyz, gt_reg)

            timesteps = self._sample_timesteps(B, points.device)
            x_1 = torch.randn_like(x_0_sa)
            x_t, v_t = self._compute_flow_target(x_0_sa, x_1, timesteps)

            flow_input = sa_feat + self.multi_view_attn.pos_enc(x_t)
            flow_fused = fused
            v_pred = self.flow_head(flow_fused)

            output["v_pred"] = v_pred
            output["v_t"] = v_t
            output["t"] = timesteps
            output["x_0"] = x_0_sa
            output["x_1"] = x_1
            output["x_t"] = x_t

        return output

    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inference-only forward (no flow training, just encode + segment + volume)."""
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

        fused_global = fused.max(dim=1)[0]
        pred_volume = self.volume_head(fused_global)

        return {
            "seg_logits": seg_logits,
            "pred_volume": pred_volume,
            "fused_features": fused,
            "sa_xyz": sa_xyz,
            "sa_feat": sa_feat,
            "sa_view_ids": sa_view_ids,
        }

    @torch.inference_mode()
    def sample_rectified_flow(
        self, batch: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Euler ODE integration for flow-based registration (from RPF).

        Integrates from t=1 (noise) toward t=0 (registered) to produce
        the registered point cloud at the SA level.

        Returns:
            (B, M, 3) registered point positions.
        """
        if num_steps is None:
            num_steps = self.inference_sampling_steps

        inf_out = self.forward_inference(batch)
        sa_xyz = inf_out["sa_xyz"]
        sa_feat = inf_out["sa_feat"]
        sa_view_ids = inf_out["sa_view_ids"]
        fused = inf_out["fused_features"]
        B, M, _ = sa_xyz.shape

        x_t = torch.randn(B, M, 3, device=sa_xyz.device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = 1.0 - step * dt
            v_pred = self.flow_head(fused)
            x_t = x_t - v_pred * dt

        return x_t

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

    def get_registered_points(
        self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """Extract per-sample stone point clouds from the model output."""
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
            "volume_head": _count(self.volume_head),
            "total": _count(self),
        }
