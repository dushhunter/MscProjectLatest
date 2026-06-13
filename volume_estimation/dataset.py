"""PyTorch dataset for stone shape completion via registered point clouds.

Pipeline (alignment decoupled from learning):
  1. Back-project each depth map to 3D (camera space).
  2. Segment stone pixels using GT masks.
  3. Register each view to a common frame using known turntable rotation
     (params saved by prepare_gt.py).
  4. Merge all registered stone views into a single partial cloud.
  5. Center using the GT centroid.

The neural model receives the registered, stone-only partial cloud and
learns only shape completion via flow matching -- the task RAP/RPF solve.
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
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)

ViewEntry = Tuple[int, str, Optional[np.ndarray], Optional[str]]


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


def _load_gt_cloud(path: str) -> Optional[np.ndarray]:
    """Load a GT point cloud (.ply or .npy). Returns (N, 3) or None."""
    if not os.path.isfile(path):
        return None
    try:
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts if pts.shape[0] > 0 else None
    except Exception:
        return None


def _farthest_point_sample_np(pts: np.ndarray, n: int) -> np.ndarray:
    """Downsample to *n* points with good spatial coverage via greedy FPS."""
    if pts.shape[0] <= n:
        if pts.shape[0] == 0:
            return np.zeros((n, 3), dtype=np.float32)
        choice = np.random.choice(pts.shape[0], n, replace=True)
        return pts[choice]
    cap = min(pts.shape[0], n * 4)
    if pts.shape[0] > cap:
        idx = np.random.choice(pts.shape[0], cap, replace=False)
        pts = pts[idx]
    selected = [np.random.randint(pts.shape[0])]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(n - 1):
        d = np.sum((pts - pts[selected[-1]]) ** 2, axis=-1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[np.array(selected)]


def _rotation_y(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix around the Y (up/gravity) axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _register_view(
    pts_cam: np.ndarray,
    frame_idx: int,
    R_floor_up: Optional[np.ndarray],
    turntable_center: Optional[np.ndarray],
    angle_per_frame_deg: float,
) -> np.ndarray:
    """Register camera-space points to the common turntable frame.

    Replicates the exact registration from prepare_gt.py:
      1. Apply floor-up rotation to align gravity with +Y.
      2. Subtract optimised turntable center.
      3. Apply inverse turntable Y-rotation for this frame index.
    """
    pts = pts_cam.astype(np.float64)
    if R_floor_up is not None:
        pts = (R_floor_up @ pts.T).T
    if turntable_center is not None:
        pts = pts - turntable_center
    theta = math.radians(frame_idx * (-angle_per_frame_deg))
    c, s = math.cos(theta), math.sin(theta)
    R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    pts = (R_yaw @ pts.T).T
    return pts.astype(np.float32)


def _scan_depth_dir(
    depth_dir: str,
    external_poses: Optional[Dict[int, np.ndarray]] = None,
    mask_dir: Optional[str] = None,
) -> List[ViewEntry]:
    """Scan a depth directory and return ViewEntry tuples."""
    mask_by_idx: Dict[int, str] = {}
    if mask_dir and os.path.isdir(mask_dir):
        for f in sorted(os.listdir(mask_dir)):
            if f.lower().endswith(".png"):
                digits = "".join(c for c in Path(f).stem if c.isdigit())
                if digits:
                    mask_by_idx[int(digits)] = os.path.join(mask_dir, f)

    entries: List[ViewEntry] = []
    for f in sorted(os.listdir(depth_dir)):
        if not f.lower().endswith(".npy"):
            continue
        digits = "".join(c for c in Path(f).stem if c.isdigit())
        if not digits:
            continue
        frame_idx = int(digits)
        pose = external_poses.get(frame_idx) if external_poses else None
        mask_path = mask_by_idx.get(frame_idx)
        entries.append((frame_idx, os.path.join(depth_dir, f), pose, mask_path))
    return entries


class StoneReconDataset(Dataset):
    """Dataset for stone shape completion with pre-registered point clouds.

    Each __getitem__ call:
      1. Picks a random stone and K random views.
      2. Back-projects each view, masks stone pixels, registers to common frame.
      3. Merges views into a single partial stone cloud (FPS to fixed size).
      4. Pairs with the depth-merge GT cloud (same coordinate frame).

    Registration uses analytical turntable poses; parameters are loaded from
    ``{gt_cloud_dir}/{stone_id}_registration.npz`` (generated by prepare_gt.py).
    """

    def __init__(
        self,
        dataset_dir: str,
        volumes_json: Optional[str],
        intrinsics_path: str,
        stone_ids: List[str],
        width: int = 1024,
        height: int = 576,
        min_views: int = 4,
        max_views: int = 24,
        max_points_per_view: int = 4096,
        merged_cloud_points: int = 8192,
        angle_per_frame_deg: float = 3.0,
        augment: bool = True,
        depth_noise_sigma_m: float = 0.002,
        scale_jitter: float = 0.05,
        point_dropout_rate: float = 0.1,
        xyz_jitter_sigma: float = 0.001,
        samples_per_epoch: int = 500,
        random_views_suffix: str = "_random_npy",
        gt_cloud_dir: Optional[str] = None,
        gt_cloud_points: int = 8192,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.stone_ids = list(stone_ids)
        self.width = width
        self.height = height
        self.min_views = min_views
        self.max_views = max_views
        self.max_points_per_view = max_points_per_view
        self.merged_cloud_points = merged_cloud_points
        self.angle_per_frame_deg = angle_per_frame_deg
        self.augment = augment
        self.depth_noise_sigma = depth_noise_sigma_m
        self.scale_jitter = scale_jitter
        self.point_dropout_rate = point_dropout_rate
        self.xyz_jitter_sigma = xyz_jitter_sigma
        self.samples_per_epoch = samples_per_epoch
        self.gt_cloud_points = gt_cloud_points

        self.volumes: Dict[str, float] = {}
        if volumes_json is not None:
            with open(volumes_json, "r") as f:
                vol_data = json.load(f)
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

        self._registration_params: Dict[str, Dict[str, np.ndarray]] = {}
        self._gt_clouds_cached: Dict[str, Optional[np.ndarray]] = {}

        self._view_entries: Dict[str, List[ViewEntry]] = {}
        for sid in self.stone_ids:
            all_views: List[ViewEntry] = []

            turntable_dir = os.path.join(dataset_dir, f"{sid}_depth_npy")
            turntable_mask_dir = os.path.join(dataset_dir, sid, "masks")
            if os.path.isdir(turntable_dir):
                all_views.extend(_scan_depth_dir(
                    turntable_dir, None, turntable_mask_dir,
                ))

            random_dir = os.path.join(dataset_dir, f"{sid}{random_views_suffix}")
            if os.path.isdir(random_dir):
                random_mask_dir = os.path.join(random_dir, "masks")
                all_views.extend(_scan_depth_dir(
                    random_dir, None, random_mask_dir,
                ))

            self._view_entries[sid] = all_views

            if gt_cloud_dir:
                reg_path = os.path.join(gt_cloud_dir, f"{sid}_registration.npz")
                if os.path.isfile(reg_path):
                    data = dict(np.load(reg_path))
                    self._registration_params[sid] = data
                    _LOG.info("Registration params %s: loaded", sid)
                else:
                    _LOG.warning(
                        "No registration params for %s at %s — "
                        "run prepare_gt.py first", sid, reg_path,
                    )

            gt_cached = None
            if gt_cloud_dir:
                cache_npy = os.path.join(
                    gt_cloud_dir, f"{sid}_cached_{gt_cloud_points}.npy",
                )
                if os.path.isfile(cache_npy):
                    gt_cached = np.load(cache_npy).astype(np.float32)
                    _LOG.info("GT cloud %s: loaded from cache (%d pts)",
                              sid, gt_cached.shape[0])
                else:
                    for suffix in ("_gt_pointcloud.ply", "_gt_complete.ply", "_gt.ply"):
                        candidate = os.path.join(gt_cloud_dir, f"{sid}{suffix}")
                        if os.path.isfile(candidate):
                            _LOG.info("GT cloud %s: loading %s ...", sid, candidate)
                            raw = _load_gt_cloud(candidate)
                            if raw is not None:
                                _LOG.info("GT cloud %s: FPS %d -> %d ...",
                                          sid, raw.shape[0], gt_cloud_points)
                                gt_cached = _farthest_point_sample_np(raw, gt_cloud_points)
                                gt_cached = (gt_cached - gt_cached.mean(axis=0)).astype(
                                    np.float32
                                )
                                np.save(cache_npy, gt_cached)
                                _LOG.info("GT cloud %s: cached to %s", sid, cache_npy)
                            break
            self._gt_clouds_cached[sid] = gt_cached

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stone_id = random.choice(self.stone_ids)
        view_entries = self._view_entries[stone_id]
        fx, fy, cx, cy = self._intrinsics_cache[stone_id]
        reg = self._registration_params.get(stone_id, {})
        R_floor_up = reg.get("R_floor_up")
        turntable_center = reg.get("turntable_center")
        gt_centroid = reg.get("gt_centroid")

        n_available = len(view_entries)
        K = random.randint(self.min_views, min(self.max_views, n_available))
        selected = random.sample(view_entries, K)

        all_pts: List[np.ndarray] = []

        for frame_idx, npy_path, _pose, mask_path in selected:
            depth = np.load(npy_path).astype(np.float32)

            if self.augment and self.depth_noise_sigma > 0:
                noise = np.random.normal(
                    0, self.depth_noise_sigma, depth.shape
                ).astype(np.float32)
                valid_mask = np.isfinite(depth) & (depth > 0)
                depth[valid_mask] += noise[valid_mask]

            pts_cam, flat_idx = _backproject_np(depth, fx, fy, cx, cy)
            if pts_cam.shape[0] == 0:
                continue

            if mask_path is not None:
                mask_flat = _load_mask_np(mask_path, self.height, self.width)
                stone_sel = mask_flat[flat_idx]
                pts_cam = pts_cam[stone_sel]

            if pts_cam.shape[0] < 10:
                continue

            if self.augment and self.point_dropout_rate > 0:
                keep = np.random.rand(pts_cam.shape[0]) > self.point_dropout_rate
                pts_cam = pts_cam[keep]

            if pts_cam.shape[0] > self.max_points_per_view:
                choice = np.random.choice(
                    pts_cam.shape[0], self.max_points_per_view, replace=False,
                )
                pts_cam = pts_cam[choice]

            pts_reg = _register_view(
                pts_cam, frame_idx, R_floor_up,
                turntable_center, self.angle_per_frame_deg,
            )
            all_pts.append(pts_reg)

        if not all_pts:
            return self._empty_sample()

        points = np.concatenate(all_pts, axis=0)

        if gt_centroid is not None:
            points = points - gt_centroid.astype(np.float32)
        else:
            points = points - points.mean(axis=0)

        points = _farthest_point_sample_np(points, self.merged_cloud_points)

        gt_cloud = self._gt_clouds_cached.get(stone_id)
        if gt_cloud is not None:
            gt_cloud = gt_cloud.copy()

        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            R = _rotation_y(angle)
            points = (points @ R.T).astype(np.float32)
            if gt_cloud is not None:
                gt_cloud = (gt_cloud @ R.T).astype(np.float32)

            if self.scale_jitter > 0:
                scale = 1.0 + random.uniform(-self.scale_jitter, self.scale_jitter)
                points = (points * scale).astype(np.float32)
                if gt_cloud is not None:
                    gt_cloud = (gt_cloud * scale).astype(np.float32)

            if self.xyz_jitter_sigma > 0:
                noise = np.random.normal(
                    0, self.xyz_jitter_sigma, points.shape
                ).astype(np.float32)
                points = points + noise

        seg_labels = np.ones(points.shape[0], dtype=np.float32)
        view_ids = np.zeros(points.shape[0], dtype=np.int64)

        sample: Dict[str, Any] = {
            "points": torch.from_numpy(points.astype(np.float32)),
            "seg_labels": torch.from_numpy(seg_labels),
            "view_ids": torch.from_numpy(view_ids),
            "n_views": torch.tensor(K, dtype=torch.int64),
            "stone_id": stone_id,
        }

        if gt_cloud is not None:
            sample["gt_cloud"] = torch.from_numpy(gt_cloud.astype(np.float32))

        if stone_id in self.volumes:
            sample["gt_volume"] = torch.tensor(
                self.volumes[stone_id], dtype=torch.float32,
            )

        return sample

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Fallback for degenerate cases."""
        n = self.merged_cloud_points
        sample: Dict[str, Any] = {
            "points": torch.zeros(n, 3),
            "seg_labels": torch.ones(n),
            "view_ids": torch.zeros(n, dtype=torch.int64),
            "n_views": torch.tensor(0, dtype=torch.int64),
            "stone_id": "",
        }
        if self.volumes:
            sample["gt_volume"] = torch.tensor(0.0, dtype=torch.float32)
        return sample


def collate_variable_points(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for variable-length point clouds.

    Pads all point clouds to the max size in the batch and creates
    a padding mask.  Scalar values are simply stacked.
    """
    max_pts = max(b["points"].shape[0] for b in batch)
    B = len(batch)

    points_padded = torch.zeros(B, max_pts, 3)
    seg_padded = torch.zeros(B, max_pts)
    view_padded = torch.zeros(B, max_pts, dtype=torch.int64)
    pad_mask = torch.ones(B, max_pts, dtype=torch.bool)

    n_views_list = []
    stone_ids = []
    n_points = []
    gt_volumes = []
    gt_clouds = []
    has_volume = "gt_volume" in batch[0]
    has_gt_cloud = "gt_cloud" in batch[0]

    for i, b in enumerate(batch):
        n = b["points"].shape[0]
        points_padded[i, :n] = b["points"]
        seg_padded[i, :n] = b["seg_labels"]
        view_padded[i, :n] = b["view_ids"]
        pad_mask[i, :n] = False
        n_views_list.append(b["n_views"])
        stone_ids.append(b["stone_id"])
        n_points.append(n)
        if has_volume:
            gt_volumes.append(b["gt_volume"])
        if has_gt_cloud:
            gt_clouds.append(b["gt_cloud"])

    result: Dict[str, Any] = {
        "points": points_padded,
        "seg_labels": seg_padded,
        "view_ids": view_padded,
        "pad_mask": pad_mask,
        "n_views": torch.stack(n_views_list),
        "stone_ids": stone_ids,
        "n_points": torch.tensor(n_points, dtype=torch.int64),
    }

    if has_volume:
        result["gt_volume"] = torch.stack(gt_volumes)

    if has_gt_cloud and gt_clouds:
        result["gt_cloud"] = torch.stack(gt_clouds)

    return result
