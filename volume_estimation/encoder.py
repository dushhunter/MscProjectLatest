"""PointNet++ encoder for per-view point cloud feature extraction.

Pure PyTorch implementation (no external pointnet2-ops dependency).
Uses farthest-point sampling, ball-query grouping, and shared-MLP
set abstraction layers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _farthest_point_sample(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Farthest point sampling on a batch of point clouds.

    Args:
        xyz: (B, N, 3) input point positions.
        n_sample: number of points to sample.

    Returns:
        (B, n_sample) indices of sampled points.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    n_sample = min(n_sample, N)

    centroids = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_sample):
        centroids[:, i] = farthest
        centroid_xyz = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def _ball_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    n_sample: int,
) -> torch.Tensor:
    """Ball query: for each point in new_xyz, find up to n_sample neighbors in xyz.

    Args:
        xyz: (B, N, 3) all points.
        new_xyz: (B, S, 3) query centers.
        radius: search radius.
        n_sample: max neighbors per query.

    Returns:
        (B, S, n_sample) indices into xyz.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device

    dists = torch.cdist(new_xyz, xyz)
    dists[dists > radius] = 1e10

    _, idx = dists.topk(min(n_sample, N), dim=-1, largest=False)
    if idx.shape[-1] < n_sample:
        pad = idx[:, :, :1].expand(-1, -1, n_sample - idx.shape[-1])
        idx = torch.cat([idx, pad], dim=-1)

    first = idx[:, :, 0:1].expand_as(idx)
    mask = dists.gather(-1, idx) > radius
    idx[mask] = first[mask]

    return idx


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx]


class SharedMLP(nn.Module):
    """Shared MLP applied point-wise (1x1 convolution equivalent)."""

    def __init__(self, channels: List[int], bn: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self._has_bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., C_in) -> (..., C_out)"""
        shape = x.shape
        if len(shape) > 2:
            x_flat = x.reshape(-1, shape[-1])
            out = self.mlp(x_flat)
            return out.reshape(*shape[:-1], -1)
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer with FPS + ball query + shared MLP."""

    def __init__(
        self,
        n_point: int,
        radius: float,
        n_sample: int,
        in_channel: int,
        mlp_channels: List[int],
    ):
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample

        full_channels = [in_channel + 3] + mlp_channels
        self.mlp = SharedMLP(full_channels)

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) point positions.
            features: (B, N, C) per-point features (or None for first layer).

        Returns:
            new_xyz: (B, n_point, 3)
            new_features: (B, n_point, C_out)
        """
        B, N, _ = xyz.shape
        n_pt = min(self.n_point, N)

        fps_idx = _farthest_point_sample(xyz, n_pt)
        new_xyz = _index_points(xyz, fps_idx)

        group_idx = _ball_query(xyz, new_xyz, self.radius, self.n_sample)
        grouped_xyz = _index_points(xyz, group_idx)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_feat = _index_points(features, group_idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_xyz

        grouped = self.mlp(grouped)
        new_features = grouped.max(dim=2)[0]

        return new_xyz, new_features


class PointNetPPEncoder(nn.Module):
    """PointNet++ encoder with 3 set-abstraction levels.

    Produces per-point features at multiple scales and a global feature
    by max-pooling the final layer.
    """

    def __init__(
        self,
        sa1_npoint: int = 4096,
        sa1_radius: float = 0.01,
        sa1_nsample: int = 32,
        sa1_mlp: Optional[List[int]] = None,
        sa2_npoint: int = 1024,
        sa2_radius: float = 0.02,
        sa2_nsample: int = 32,
        sa2_mlp: Optional[List[int]] = None,
        sa3_npoint: int = 256,
        sa3_radius: float = 0.04,
        sa3_nsample: int = 32,
        sa3_mlp: Optional[List[int]] = None,
        feature_dim: int = 256,
    ):
        super().__init__()

        sa1_mlp = sa1_mlp or [64, 64, 128]
        sa2_mlp = sa2_mlp or [128, 128, 256]
        sa3_mlp = sa3_mlp or [256, 256, feature_dim]

        self.sa1 = SetAbstraction(sa1_npoint, sa1_radius, sa1_nsample, 0, sa1_mlp)
        self.sa2 = SetAbstraction(sa2_npoint, sa2_radius, sa2_nsample, sa1_mlp[-1], sa2_mlp)
        self.sa3 = SetAbstraction(sa3_npoint, sa3_radius, sa3_nsample, sa2_mlp[-1], sa3_mlp)

        self.feature_dim = feature_dim

    def forward(
        self, xyz: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) point positions.
            mask: (B, N) optional padding mask (True = padded, ignore).

        Returns:
            sa3_xyz: (B, 256, 3) positions at final SA level.
            sa3_feat: (B, 256, feature_dim) features at final SA level.
            global_feat: (B, feature_dim) max-pooled global feature.
        """
        if mask is not None:
            xyz = xyz.clone()
            xyz[mask] = 0.0

        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)

        global_feat = feat3.max(dim=1)[0]

        return xyz3, feat3, global_feat


class SegmentationHead(nn.Module):
    """Per-point binary segmentation head.

    Takes the coarse SA3-level features and propagates scores back
    to the original point resolution using nearest-neighbor interpolation.
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, xyz_full: torch.Tensor, xyz_sa: torch.Tensor, feat_sa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz_full: (B, N, 3) original point positions.
            xyz_sa: (B, M, 3) SA-level positions (M < N).
            feat_sa: (B, M, C) SA-level features.

        Returns:
            logits: (B, N) per-point segmentation logit.
        """
        B, N, _ = xyz_full.shape

        dists = torch.cdist(xyz_full, xyz_sa)
        _, nn_idx = dists.topk(3, dim=-1, largest=False)

        nn_dist = dists.gather(-1, nn_idx).clamp(min=1e-8)
        weights = 1.0 / nn_dist
        weights = weights / weights.sum(dim=-1, keepdim=True)

        nn_feat = _index_points(feat_sa, nn_idx)
        interp_feat = (nn_feat * weights.unsqueeze(-1)).sum(dim=2)

        flat = interp_feat.reshape(B * N, -1)
        logits = self.mlp(flat).reshape(B, N)
        return logits
