"""Self-view depth projection for point clouds (from PointSea / SVDFormer).

Projects a 3D point cloud into depth images from 3 orthogonal viewpoints
(front, side, top). The resulting depth images provide global shape
understanding that helps the completion network identify missing regions.

Reference: PointSea (IJCV 2025), SVDFormer (ICCV 2023)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _rotation_matrix(axis: str, angle_deg: float, device: torch.device) -> torch.Tensor:
    """3x3 rotation matrix around the given axis."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    if axis == "y":
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device)
    elif axis == "x":
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=device)
    else:
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device)
    return R.float()


class PCViews:
    """Project a point cloud into depth images from 3 orthogonal viewpoints.

    Following PointSea, we use front (0 deg), side (90 deg Y), and top
    (90 deg X) views. Points are first normalized to [-0.5, 0.5], then
    perspective-projected onto a 2D grid where each pixel stores the
    nearest depth value.

    The output is a (B*3, 1, H, W) tensor suitable for a 2D CNN backbone.
    """

    NUM_VIEWS = 3
    RESOLUTION = 224

    ROTATIONS = [
        ("y", 0.0),
        ("y", 90.0),
        ("x", 90.0),
    ]

    CAMERA_DIST = 1.5

    def __init__(self, resolution: int = 224, camera_dist: float = 1.5):
        self.resolution = resolution
        self.camera_dist = camera_dist

    def get_img(self, points: torch.Tensor) -> torch.Tensor:
        """Project points to 3-view depth images.

        Args:
            points: (B, N, 3) point cloud, centered around origin.

        Returns:
            depth_imgs: (B*3, 1, H, W) depth images.
        """
        B, N, _ = points.shape
        device = points.device
        H = W = self.resolution

        all_views = []

        for axis, angle in self.ROTATIONS:
            R = _rotation_matrix(axis, angle, device)
            rotated = torch.matmul(points, R.T)

            rotated[..., 2] += self.camera_dist

            z = rotated[..., 2].clamp(min=1e-6)
            u = rotated[..., 0] / z
            v = rotated[..., 1] / z

            u = ((u + 0.5) * (W - 1)).long().clamp(0, W - 1)
            v = ((v + 0.5) * (H - 1)).long().clamp(0, H - 1)

            depth_map = torch.full((B, H, W), fill_value=1e6, device=device)

            flat_idx = v * W + u
            for b in range(B):
                depth_flat = depth_map[b].view(-1)
                z_b = z[b]
                idx_b = flat_idx[b]
                depth_flat.scatter_reduce_(0, idx_b, z_b, reduce="amin")
                depth_map[b] = depth_flat.view(H, W)

            bg_mask = depth_map >= 1e5
            depth_map[bg_mask] = 0.0

            fg_mask = ~bg_mask
            if fg_mask.any():
                d_min = depth_map[fg_mask].min()
                d_max = depth_map[fg_mask].max()
                d_range = d_max - d_min
                if d_range > 1e-8:
                    depth_map[fg_mask] = (depth_map[fg_mask] - d_min) / d_range
                else:
                    depth_map[fg_mask] = 0.5

            all_views.append(depth_map.unsqueeze(1))

        return torch.cat(all_views, dim=0)
