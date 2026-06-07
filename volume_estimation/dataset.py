"""PyTorch dataset for multi-view stone volume estimation.

Each sample is a random subset of K depth views from a single stone,
along with ground-truth segmentation masks and the GT volume.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare_gt import _turntable_rotation_y


def _backproject_np(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project depth to 3D points. Returns (points_Nx3, valid_flat_indices)."""
    H, W = depth.shape
    yy, xx = np.indices((H, W))
    ys = yy.ravel()
    xs = xx.ravel()
    zs = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(zs) & (zs > 0)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    pts = np.stack([X, Y, zs], axis=1).astype(np.float32)
    flat_idx = (ys * W + xs).astype(np.int64)
    return pts, flat_idx


def _load_mask_np(path: str, H: int, W: int) -> np.ndarray:
    """Load a binary mask PNG as a flat boolean array of size H*W."""
    from PIL import Image
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (H, W):
        img = img.resize((W, H), Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)
    return (arr > 127).ravel()


def _random_rotation_matrix(max_angle_deg: float) -> np.ndarray:
    """Small random 3x3 rotation for augmentation."""
    angles = np.random.uniform(
        -math.radians(max_angle_deg), math.radians(max_angle_deg), size=3
    )
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


class StoneVolumeDataset(Dataset):
    """Dataset that yields random multi-view samples for stone volume training.

    Each __getitem__ call picks a stone at random, selects K random views,
    back-projects them to 3D, applies GT masks and turntable transforms,
    and returns the points with segmentation labels and volume.
    """

    def __init__(
        self,
        dataset_dir: str,
        volumes_json: str,
        intrinsics_path: str,
        stone_ids: List[str],
        width: int = 1024,
        height: int = 576,
        min_views: int = 4,
        max_views: int = 24,
        max_points_per_view: int = 4096,
        angle_per_frame_deg: float = 3.0,
        augment: bool = True,
        depth_noise_sigma_m: float = 0.002,
        rotation_perturb_deg: float = 5.0,
        scale_jitter: float = 0.05,
        point_dropout_rate: float = 0.1,
        samples_per_epoch: int = 500,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.stone_ids = list(stone_ids)
        self.width = width
        self.height = height
        self.min_views = min_views
        self.max_views = max_views
        self.max_points_per_view = max_points_per_view
        self.angle_per_frame_deg = angle_per_frame_deg
        self.augment = augment
        self.depth_noise_sigma = depth_noise_sigma_m
        self.rotation_perturb_deg = rotation_perturb_deg
        self.scale_jitter = scale_jitter
        self.point_dropout_rate = point_dropout_rate
        self.samples_per_epoch = samples_per_epoch

        with open(volumes_json, "r") as f:
            vol_data = json.load(f)
        self.volumes: Dict[str, float] = {}
        for sid in self.stone_ids:
            if sid in vol_data:
                self.volumes[sid] = float(vol_data[sid]["volume_cm3"])
            else:
                raise KeyError(f"Stone '{sid}' not found in {volumes_json}")

        from neural_pipeline.geometry import load_intrinsics
        self._intrinsics_cache: Dict[str, Tuple[float, float, float, float]] = {}
        for sid in self.stone_ids:
            K = load_intrinsics(intrinsics_path, sid, width, height)
            self._intrinsics_cache[sid] = (K.fx, K.fy, K.cx, K.cy)

        self._depth_files: Dict[str, List[Tuple[int, str]]] = {}
        self._mask_files: Dict[str, Dict[int, str]] = {}
        for sid in self.stone_ids:
            depth_dir = os.path.join(dataset_dir, f"{sid}_depth_npy")
            mask_dir = os.path.join(dataset_dir, sid, "masks")

            npy_files = []
            for f in sorted(os.listdir(depth_dir)):
                if f.lower().endswith(".npy"):
                    digits = "".join(c for c in Path(f).stem if c.isdigit())
                    if digits:
                        npy_files.append((int(digits), os.path.join(depth_dir, f)))
            self._depth_files[sid] = npy_files

            mask_map = {}
            if os.path.isdir(mask_dir):
                for f in sorted(os.listdir(mask_dir)):
                    if f.lower().endswith(".png"):
                        digits = "".join(c for c in Path(f).stem if c.isdigit())
                        if digits:
                            mask_map[int(digits)] = os.path.join(mask_dir, f)
            self._mask_files[sid] = mask_map

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stone_id = random.choice(self.stone_ids)
        depth_files = self._depth_files[stone_id]
        masks = self._mask_files[stone_id]
        fx, fy, cx, cy = self._intrinsics_cache[stone_id]

        n_available = len(depth_files)
        K = random.randint(self.min_views, min(self.max_views, n_available))
        selected = random.sample(depth_files, K)

        all_pts = []
        all_seg_labels = []
        all_view_ids = []

        for view_i, (frame_idx, npy_path) in enumerate(selected):
            depth = np.load(npy_path).astype(np.float32)

            if self.augment and self.depth_noise_sigma > 0:
                noise = np.random.normal(0, self.depth_noise_sigma, depth.shape).astype(np.float32)
                valid_mask = np.isfinite(depth) & (depth > 0)
                depth[valid_mask] += noise[valid_mask]

            pts_cam, flat_idx = _backproject_np(depth, fx, fy, cx, cy)
            if pts_cam.shape[0] == 0:
                continue

            has_mask = frame_idx in masks
            if has_mask:
                mask_flat = _load_mask_np(masks[frame_idx], self.height, self.width)
                seg_labels = mask_flat[flat_idx].astype(np.float32)
            else:
                seg_labels = np.ones(pts_cam.shape[0], dtype=np.float32)

            T = _turntable_rotation_y(frame_idx, self.angle_per_frame_deg)
            R = T[:3, :3].astype(np.float32)
            pts_world = (R @ pts_cam.T).T

            if self.augment and self.point_dropout_rate > 0:
                keep = np.random.rand(pts_world.shape[0]) > self.point_dropout_rate
                pts_world = pts_world[keep]
                seg_labels = seg_labels[keep]

            if pts_world.shape[0] > self.max_points_per_view:
                choice = np.random.choice(
                    pts_world.shape[0], self.max_points_per_view, replace=False
                )
                pts_world = pts_world[choice]
                seg_labels = seg_labels[choice]

            view_id = np.full(pts_world.shape[0], view_i, dtype=np.int64)
            all_pts.append(pts_world)
            all_seg_labels.append(seg_labels)
            all_view_ids.append(view_id)

        if not all_pts:
            return self._empty_sample()

        points = np.concatenate(all_pts, axis=0)
        seg_labels = np.concatenate(all_seg_labels, axis=0)
        view_ids = np.concatenate(all_view_ids, axis=0)

        # Store pre-augmentation GT registered positions for flow target x_0.
        # These are the clean turntable-aligned positions before any
        # augmentation noise, and they share the same centroid subtraction.
        centroid = points.mean(axis=0)
        gt_points_registered = points - centroid

        if self.augment:
            if self.rotation_perturb_deg > 0:
                R_aug = _random_rotation_matrix(self.rotation_perturb_deg)
                points = (R_aug @ points.T).T
            if self.scale_jitter > 0:
                scale = 1.0 + random.uniform(-self.scale_jitter, self.scale_jitter)
                points = points * scale

        points = points - points.mean(axis=0)

        gt_volume = self.volumes[stone_id]

        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "seg_labels": torch.from_numpy(seg_labels),
            "view_ids": torch.from_numpy(view_ids),
            "gt_volume": torch.tensor(gt_volume, dtype=torch.float32),
            "n_views": torch.tensor(K, dtype=torch.int64),
            "stone_id": stone_id,
            "gt_points_registered": torch.from_numpy(
                gt_points_registered.astype(np.float32)
            ),
        }

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Fallback for degenerate cases."""
        return {
            "points": torch.zeros(1, 3),
            "seg_labels": torch.zeros(1),
            "view_ids": torch.zeros(1, dtype=torch.int64),
            "gt_volume": torch.tensor(0.0),
            "n_views": torch.tensor(0, dtype=torch.int64),
            "stone_id": "",
            "gt_points_registered": torch.zeros(1, 3),
        }


def collate_variable_points(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for variable-length point clouds.

    Pads all point clouds to the max size in the batch and creates
    a padding mask. Scalar values are simply stacked.
    """
    max_pts = max(b["points"].shape[0] for b in batch)
    B = len(batch)

    points_padded = torch.zeros(B, max_pts, 3)
    gt_reg_padded = torch.zeros(B, max_pts, 3)
    seg_padded = torch.zeros(B, max_pts)
    view_padded = torch.zeros(B, max_pts, dtype=torch.int64)
    pad_mask = torch.ones(B, max_pts, dtype=torch.bool)

    gt_volumes = []
    n_views_list = []
    stone_ids = []
    n_points = []

    for i, b in enumerate(batch):
        n = b["points"].shape[0]
        points_padded[i, :n] = b["points"]
        gt_reg_padded[i, :n] = b["gt_points_registered"]
        seg_padded[i, :n] = b["seg_labels"]
        view_padded[i, :n] = b["view_ids"]
        pad_mask[i, :n] = False
        gt_volumes.append(b["gt_volume"])
        n_views_list.append(b["n_views"])
        stone_ids.append(b["stone_id"])
        n_points.append(n)

    return {
        "points": points_padded,
        "gt_points_registered": gt_reg_padded,
        "seg_labels": seg_padded,
        "view_ids": view_padded,
        "pad_mask": pad_mask,
        "gt_volume": torch.stack(gt_volumes),
        "n_views": torch.stack(n_views_list),
        "stone_ids": stone_ids,
        "n_points": torch.tensor(n_points, dtype=torch.int64),
    }
