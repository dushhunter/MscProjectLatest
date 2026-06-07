"""Dataset for stone completion training.

Each sample provides:
  - Full scene points (stone + floor) for segmentation training
  - Partial stone cloud (GT-masked, turntable-rotated) for completion training
  - GT complete stone cloud (from Blender mesh) for Chamfer Distance loss
  - GT segmentation labels (from masks)

Compatible with existing stone_syn_dataset/ layout.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained, no imports from volume_estimation)
# ---------------------------------------------------------------------------

def _backproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Back-project depth image to 3D camera-space points.

    Returns (N, 3) array of valid (finite, positive-depth) points.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _turntable_rotation_y(frame_index: int, angle_per_frame_deg: float = 3.0) -> np.ndarray:
    """4x4 Y-rotation for a turntable frame."""
    theta = math.radians(frame_index * angle_per_frame_deg)
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c;  T[0, 2] = s
    T[2, 0] = -s; T[2, 2] = c
    return T


def _load_mask(mask_path: str, H: int, W: int) -> np.ndarray:
    """Load binary mask (stone=True)."""
    img = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
    return np.array(img) > 127


def _extract_frame_index(filepath: str) -> int:
    """Extract numeric frame index from filename like depth_0042.npy."""
    digits = "".join(c for c in Path(filepath).stem if c.isdigit())
    return int(digits) if digits else 0


def _parse_intrinsics(path: str, stone_id: str, W: int, H: int) -> Tuple[float, float, float, float]:
    """Parse normalized intrinsics file. Returns (fx, fy, cx, cy) in pixels."""
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == stone_id:
                fn, fyn, cxn, cyn = (float(x) for x in parts[1:5])
                return fn * W, fyn * H, cxn * W, cyn * H
    raise KeyError(f"Stone '{stone_id}' not found in {path}")


def _farthest_point_sample_np(pts: np.ndarray, n: int) -> np.ndarray:
    """Numpy FPS. Returns (n, 3) subsampled points."""
    if pts.shape[0] <= n:
        if pts.shape[0] == 0:
            return np.zeros((n, 3), dtype=np.float32)
        choice = np.random.choice(pts.shape[0], n, replace=True)
        return pts[choice]

    selected = [np.random.randint(pts.shape[0])]
    dists = np.full(pts.shape[0], np.inf)

    for _ in range(n - 1):
        last = pts[selected[-1]]
        d = np.sum((pts - last) ** 2, axis=-1)
        dists = np.minimum(dists, d)
        selected.append(np.argmax(dists))

    return pts[np.array(selected)]


def _load_gt_complete(path: str, n_points: int) -> np.ndarray:
    """Load GT complete stone cloud (.ply or .npy), FPS to n_points."""
    if path.endswith(".npy"):
        pts = np.load(path).astype(np.float32)
    else:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)

    return _farthest_point_sample_np(pts, n_points)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StoneCompletionDataset(Dataset):
    """Dataset yielding (scene_points, partial_stone, gt_complete, seg_labels).

    For each sample, picks a random stone, selects K random views,
    back-projects and applies turntable rotation + mask segmentation.

    Args:
        dataset_dir: Root dir containing stone_XX/ and stone_XX_depth_npy/
        intrinsics_file: Path to intrinsics.txt
        stone_ids: List of stone IDs to use (e.g. ["stone_01", ..., "stone_10"])
        gt_complete_dir: Dir containing stone_XX_gt_complete.ply files
        width, height: Depth map resolution
        input_points: Fixed number of scene points per sample
        completion_points: Fixed number of partial stone points
        gt_points: Number of points in GT complete cloud
        min_views, max_views: Range of views to sample per stone
        max_points_per_view: Max points to keep from each view
        augment: Whether to apply data augmentation
        samples_per_epoch: Virtual epoch length
    """

    def __init__(
        self,
        dataset_dir: str,
        intrinsics_file: str,
        stone_ids: List[str],
        gt_complete_dir: str,
        width: int = 1024,
        height: int = 576,
        input_points: int = 4096,
        completion_points: int = 2048,
        gt_points: int = 8192,
        min_views: int = 4,
        max_views: int = 18,
        max_points_per_view: int = 4096,
        augment: bool = True,
        samples_per_epoch: int = 500,
        angle_per_frame_deg: float = 3.0,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.input_points = input_points
        self.completion_points = completion_points
        self.gt_points = gt_points
        self.min_views = min_views
        self.max_views = max_views
        self.max_points_per_view = max_points_per_view
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.angle_per_frame = angle_per_frame_deg
        self.width = width
        self.height = height

        self.stone_ids: List[str] = []
        self._depth_files: Dict[str, List[str]] = {}
        self._mask_dirs: Dict[str, str] = {}
        self._intrinsics: Dict[str, Tuple[float, float, float, float]] = {}
        self._gt_complete_paths: Dict[str, str] = {}

        for sid in stone_ids:
            depth_dir = os.path.join(dataset_dir, f"{sid}_depth_npy")
            mask_dir = os.path.join(dataset_dir, sid, "masks")
            if not os.path.isdir(depth_dir):
                LOG.warning("Depth dir not found: %s, skipping %s", depth_dir, sid)
                continue

            depth_files = sorted([
                os.path.join(depth_dir, f)
                for f in os.listdir(depth_dir)
                if f.lower().endswith(".npy")
            ])
            if not depth_files:
                LOG.warning("No .npy files in %s, skipping %s", depth_dir, sid)
                continue

            gt_path = None
            for suffix in ("_gt_complete.ply", "_gt_complete.npy", "_gt_pointcloud.ply"):
                candidate = os.path.join(gt_complete_dir, f"{sid}{suffix}")
                if os.path.isfile(candidate):
                    gt_path = candidate
                    break
            if gt_path is None:
                LOG.warning(
                    "No GT complete cloud for %s in %s "
                    "(expected %s_gt_complete.ply or %s_gt_pointcloud.ply). "
                    "Run: python -m stone_completion.prepare_gt --dataset_dir %s --output_dir %s",
                    sid, gt_complete_dir, sid, sid,
                    dataset_dir, gt_complete_dir,
                )
                continue

            try:
                intr = _parse_intrinsics(intrinsics_file, sid, width, height)
            except KeyError:
                LOG.warning("Intrinsics not found for %s", sid)
                continue

            self.stone_ids.append(sid)
            self._depth_files[sid] = depth_files
            self._mask_dirs[sid] = mask_dir
            self._intrinsics[sid] = intr
            self._gt_complete_paths[sid] = gt_path

        if not self.stone_ids:
            raise RuntimeError(
                f"No valid stones found in {dataset_dir} with GT in {gt_complete_dir}. "
                "Ensure stone_XX_gt_complete.ply files exist."
            )

        LOG.info("StoneCompletionDataset: %d stones ready: %s", len(self.stone_ids), self.stone_ids)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = random.choice(self.stone_ids)
        depth_files = self._depth_files[sid]
        fx, fy, cx, cy = self._intrinsics[sid]
        mask_dir = self._mask_dirs[sid]

        n_views = random.randint(self.min_views, min(self.max_views, len(depth_files)))
        selected = random.sample(depth_files, n_views)

        all_scene_pts = []
        all_stone_pts = []
        all_seg_labels = []

        for dpath in selected:
            depth = np.load(dpath).astype(np.float32)
            if depth.shape != (self.height, self.width):
                continue

            pts_cam = _backproject(depth, fx, fy, cx, cy)
            if pts_cam.shape[0] == 0:
                continue

            frame_idx = _extract_frame_index(dpath)
            T = _turntable_rotation_y(frame_idx, self.angle_per_frame)
            R = T[:3, :3].astype(np.float32)
            pts_world = (R @ pts_cam.T).T

            mask_file = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
            if os.path.isfile(mask_file):
                mask_2d = _load_mask(mask_file, self.height, self.width)
                valid = np.isfinite(depth) & (depth > 0)
                valid_indices = np.where(valid.ravel())[0]
                mask_flat = mask_2d.ravel()[valid_indices]
                seg = mask_flat.astype(np.float32)
            else:
                seg = np.ones(pts_world.shape[0], dtype=np.float32)

            if pts_world.shape[0] > self.max_points_per_view:
                choice = np.random.choice(pts_world.shape[0], self.max_points_per_view, replace=False)
                pts_world = pts_world[choice]
                seg = seg[choice]

            all_scene_pts.append(pts_world)
            all_seg_labels.append(seg)

            stone_mask = seg > 0.5
            if stone_mask.any():
                all_stone_pts.append(pts_world[stone_mask])

        if not all_scene_pts:
            return self._empty_sample()

        scene_pts = np.concatenate(all_scene_pts, axis=0)
        seg_labels = np.concatenate(all_seg_labels, axis=0)

        if all_stone_pts:
            stone_pts = np.concatenate(all_stone_pts, axis=0)
        else:
            stone_pts = np.zeros((1, 3), dtype=np.float32)

        centroid = scene_pts.mean(axis=0)
        scene_pts = scene_pts - centroid
        stone_pts = stone_pts - centroid

        scene_pts = _farthest_point_sample_np(scene_pts, self.input_points)
        seg_labels_sampled = np.zeros(self.input_points, dtype=np.float32)

        if seg_labels.shape[0] > 0:
            all_pts_orig = np.concatenate(all_scene_pts, axis=0) - centroid
            from scipy.spatial import cKDTree
            tree = cKDTree(all_pts_orig)
            _, nn_idx = tree.query(scene_pts, k=1)
            nn_idx = np.clip(nn_idx, 0, seg_labels.shape[0] - 1)
            seg_labels_sampled = seg_labels[nn_idx]

        partial_stone = _farthest_point_sample_np(stone_pts, self.completion_points)

        gt_complete = _load_gt_complete(self._gt_complete_paths[sid], self.gt_points)
        gt_complete = gt_complete - centroid

        if self.augment:
            angle = np.random.uniform(-180, 180)
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            R_aug = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            scene_pts = (R_aug @ scene_pts.T).T
            partial_stone = (R_aug @ partial_stone.T).T
            gt_complete = (R_aug @ gt_complete.T).T

            scale = 1.0 + np.random.uniform(-0.05, 0.05)
            scene_pts *= scale
            partial_stone *= scale
            gt_complete *= scale

        return {
            "scene_points": torch.from_numpy(scene_pts).float(),
            "seg_labels": torch.from_numpy(seg_labels_sampled).float(),
            "partial_stone": torch.from_numpy(partial_stone).float(),
            "gt_complete": torch.from_numpy(gt_complete).float(),
        }

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        return {
            "scene_points": torch.zeros(self.input_points, 3),
            "seg_labels": torch.zeros(self.input_points),
            "partial_stone": torch.zeros(self.completion_points, 3),
            "gt_complete": torch.zeros(self.gt_points, 3),
        }
