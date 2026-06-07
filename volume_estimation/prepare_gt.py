#!/usr/bin/env python3
"""Generate ground-truth registered point clouds and volumes for training.

For each stone:
  1. Load all depth maps (.npy) from turntable + random view directories.
  2. Back-project each depth map to 3D using camera intrinsics.
  3. Apply the known pose:
     - Turntable views: analytical rotation (frame_i * 3° about Y).
     - Random views:    4x4 extrinsic from poses.json.
  4. Keep only stone pixels (from Blender mask).
  5. Merge all views into a single dense registered point cloud.
  6. Compute the ground-truth volume via voxelization.
  7. Save gt_pointcloud.ply and update stone_volumes_gt.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

from neural_pipeline.geometry import (  # noqa: E402
    Intrinsics,
    _backproject_full,
    load_intrinsics,
    make_pcd,
)

LOG = logging.getLogger("prepare_gt")


def _turntable_rotation_y(frame_index: int, angle_per_frame_deg: float = 3.0) -> np.ndarray:
    """4x4 rotation matrix for a turntable frame (rotation about Y through origin)."""
    theta = math.radians(frame_index * angle_per_frame_deg)
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c;  T[0, 2] = s
    T[2, 0] = -s; T[2, 2] = c
    return T


def _load_mask(mask_path: str, height: int, width: int) -> np.ndarray:
    """Load a binary stone mask from a PNG file."""
    from PIL import Image
    img = Image.open(mask_path).convert("L")
    mask = np.array(img, dtype=np.uint8)
    if mask.shape != (height, width):
        img_resized = img.resize((width, height), Image.NEAREST)
        mask = np.array(img_resized, dtype=np.uint8)
    return mask > 127


def _compute_volume_voxel(points: np.ndarray, voxel_size_mm: float = 0.5) -> float:
    """Estimate volume (mm^3) by counting occupied voxels in a fine grid."""
    if points.shape[0] < 100:
        return 0.0

    pts_mm = points * 1000.0
    voxel = voxel_size_mm

    mins = pts_mm.min(axis=0)
    indices = ((pts_mm - mins) / voxel).astype(np.int64)

    unique_voxels = np.unique(indices, axis=0)
    volume_mm3 = len(unique_voxels) * (voxel ** 3)
    return float(volume_mm3)


def _compute_volume_convex_hull(points: np.ndarray) -> float:
    """Estimate volume (mm^3) via convex hull."""
    if points.shape[0] < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        pts_mm = points * 1000.0
        hull = ConvexHull(pts_mm)
        return float(hull.volume)
    except Exception:
        return 0.0


def _load_poses_json(poses_path: str) -> Dict[int, np.ndarray]:
    """Load per-view 4x4 camera extrinsic matrices from poses.json."""
    with open(poses_path, "r") as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float64) for k, v in data.items()}


def _process_depth_dir(
    depth_dir: str,
    mask_by_index: Dict[int, str],
    intrinsics: Intrinsics,
    poses: Optional[Dict[int, np.ndarray]],
    angle_per_frame_deg: float,
) -> Tuple[List[np.ndarray], int]:
    """Process all depth files in a directory, returning stone points and count.

    Args:
        poses: If provided, use explicit 4x4 matrices; otherwise turntable rotation.
    """
    npy_files = sorted(
        f for f in os.listdir(depth_dir)
        if f.lower().endswith(".npy")
    )

    all_stone_pts: List[np.ndarray] = []
    n_views = 0

    for npy_file in npy_files:
        stem = Path(npy_file).stem
        digits = "".join(c for c in stem if c.isdigit())
        if not digits:
            continue
        frame_idx = int(digits)

        depth = np.load(os.path.join(depth_dir, npy_file)).astype(np.float32)
        if depth.shape != (intrinsics.height, intrinsics.width):
            LOG.warning("  Skipping %s: shape %s != expected", npy_file, depth.shape)
            continue

        pts_cam, flat_idx = _backproject_full(depth, intrinsics, stride=1)
        if pts_cam.shape[0] == 0:
            continue

        if frame_idx in mask_by_index:
            mask = _load_mask(
                mask_by_index[frame_idx], intrinsics.height, intrinsics.width
            )
            mask_flat = mask.ravel()
            stone_sel = mask_flat[flat_idx.astype(np.int64)]
            pts_stone = pts_cam[stone_sel]
        else:
            pts_stone = pts_cam

        if pts_stone.shape[0] < 10:
            continue

        if poses is not None and frame_idx in poses:
            R = poses[frame_idx][:3, :3]
            t = poses[frame_idx][:3, 3]
            pts_world = (R @ pts_stone.T).T + t
        else:
            T = _turntable_rotation_y(frame_idx, angle_per_frame_deg)
            R = T[:3, :3]
            pts_world = (R @ pts_stone.T).T

        all_stone_pts.append(pts_world)
        n_views += 1

    return all_stone_pts, n_views


def process_one_stone(
    stone_id: str,
    depth_npy_dir: str,
    mask_dir: str,
    intrinsics: Intrinsics,
    output_dir: str,
    angle_per_frame_deg: float = 3.0,
    voxel_downsample_m: float = 0.0005,
    voxel_volume_mm: float = 0.5,
    random_views_dir: Optional[str] = None,
    random_masks_dir: Optional[str] = None,
) -> Dict:
    """Process one stone: register all views and compute GT volume.

    Supports turntable views (analytical pose) and optional random views
    (explicit poses from poses.json in the random views directory).
    """
    LOG.info("Processing %s ...", stone_id)

    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ) if os.path.isdir(mask_dir) else []

    mask_by_index: Dict[int, str] = {}
    for mf in mask_files:
        stem = Path(mf).stem
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            mask_by_index[int(digits)] = os.path.join(mask_dir, mf)

    # Also load masks from random views directory
    if random_masks_dir and os.path.isdir(random_masks_dir):
        for mf in sorted(os.listdir(random_masks_dir)):
            if mf.lower().endswith(".png"):
                stem = Path(mf).stem
                digits = "".join(c for c in stem if c.isdigit())
                if digits:
                    mask_by_index[int(digits)] = os.path.join(random_masks_dir, mf)

    all_stone_pts: List[np.ndarray] = []
    n_views_turntable = 0
    n_views_random = 0

    # 1) Turntable views
    if os.path.isdir(depth_npy_dir):
        turntable_poses_path = os.path.join(depth_npy_dir, "poses.json")
        tt_poses = _load_poses_json(turntable_poses_path) if os.path.isfile(turntable_poses_path) else None
        pts_list, n = _process_depth_dir(
            depth_npy_dir, mask_by_index, intrinsics, tt_poses, angle_per_frame_deg,
        )
        all_stone_pts.extend(pts_list)
        n_views_turntable = n
        LOG.info("  Turntable: %d views processed", n)

    # 2) Random views
    if random_views_dir and os.path.isdir(random_views_dir):
        random_poses_path = os.path.join(random_views_dir, "poses.json")
        if os.path.isfile(random_poses_path):
            rand_poses = _load_poses_json(random_poses_path)
            pts_list, n = _process_depth_dir(
                random_views_dir, mask_by_index, intrinsics, rand_poses, angle_per_frame_deg,
            )
            all_stone_pts.extend(pts_list)
            n_views_random = n
            LOG.info("  Random: %d views processed", n)
        else:
            LOG.warning("  Random views dir %s has no poses.json — skipping", random_views_dir)

    n_views_used = n_views_turntable + n_views_random

    if not all_stone_pts:
        LOG.warning("  No valid views for %s", stone_id)
        return {"stone_id": stone_id, "volume_mm3": 0.0, "volume_cm3": 0.0, "n_points": 0}

    merged_pts = np.concatenate(all_stone_pts, axis=0)
    LOG.info("  Merged %d points from %d views (turntable=%d, random=%d)",
             merged_pts.shape[0], n_views_used, n_views_turntable, n_views_random)

    pcd = make_pcd(merged_pts, estimate_normals=False)
    pcd = pcd.voxel_down_sample(voxel_downsample_m)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    final_pts = np.asarray(pcd.points)
    LOG.info("  After downsampling + outlier removal: %d points", final_pts.shape[0])

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=6 * voxel_downsample_m, max_nn=40
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"{stone_id}_gt_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    LOG.info("  Saved: %s", ply_path)

    vol_voxel = _compute_volume_voxel(final_pts, voxel_size_mm=voxel_volume_mm)
    vol_hull = _compute_volume_convex_hull(final_pts)
    volume_mm3 = (vol_voxel + vol_hull) / 2.0
    volume_cm3 = volume_mm3 / 1000.0

    LOG.info(
        "  Volume: voxel=%.2f mm3, hull=%.2f mm3, avg=%.2f mm3 (%.4f cm3)",
        vol_voxel, vol_hull, volume_mm3, volume_cm3,
    )

    return {
        "stone_id": stone_id,
        "volume_mm3": round(volume_mm3, 2),
        "volume_cm3": round(volume_cm3, 6),
        "volume_voxel_mm3": round(vol_voxel, 2),
        "volume_hull_mm3": round(vol_hull, 2),
        "n_points": final_pts.shape[0],
        "n_views": n_views_used,
        "n_views_turntable": n_views_turntable,
        "n_views_random": n_views_random,
        "ply_path": ply_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth point clouds and volumes for stone training data"
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Root of stone_syn_dataset (contains stone_XX/ and stone_XX_depth_npy/)",
    )
    parser.add_argument(
        "--intrinsics", required=True,
        help="Path to intrinsics.txt (e.g. splits/stone/intrinsics.txt)",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument(
        "--output_dir", default="volume_estimation/gt_data",
        help="Directory for output point clouds and volumes JSON",
    )
    parser.add_argument(
        "--stones", nargs="*", default=None,
        help="Stone IDs to process (e.g. stone_01 stone_02). Default: auto-detect all.",
    )
    parser.add_argument("--angle_deg", type=float, default=3.0)
    parser.add_argument("--voxel_downsample_mm", type=float, default=0.5)
    parser.add_argument("--voxel_volume_mm", type=float, default=0.5)
    parser.add_argument(
        "--random_views_suffix", default="_random_npy",
        help="Suffix for random-views directories (e.g. stone_01_random_npy). "
             "Each must contain poses.json with per-view 4x4 extrinsics.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    dataset_dir = args.dataset_dir

    if args.stones:
        stone_ids = args.stones
    else:
        stone_ids = []
        for name in sorted(os.listdir(dataset_dir)):
            if name.startswith("stone_") and "_depth_npy" not in name and "_sparse" not in name and "_random" not in name:
                full = os.path.join(dataset_dir, name)
                if os.path.isdir(full):
                    stone_ids.append(name)
        LOG.info("Auto-detected stones: %s", stone_ids)

    if not stone_ids:
        LOG.error("No stones found in %s", dataset_dir)
        sys.exit(1)

    results = {}
    for sid in stone_ids:
        depth_dir = os.path.join(dataset_dir, f"{sid}_depth_npy")
        mask_dir = os.path.join(dataset_dir, sid, "masks")
        random_dir = os.path.join(dataset_dir, f"{sid}{args.random_views_suffix}")
        random_masks = os.path.join(random_dir, "masks")

        if not os.path.isdir(depth_dir):
            LOG.warning("Depth dir not found: %s — skipping %s", depth_dir, sid)
            continue

        has_random = os.path.isdir(random_dir)
        if has_random:
            LOG.info("Found random views directory: %s", random_dir)

        K = load_intrinsics(args.intrinsics, sid, args.width, args.height)
        info = process_one_stone(
            stone_id=sid,
            depth_npy_dir=depth_dir,
            mask_dir=mask_dir,
            intrinsics=K,
            output_dir=args.output_dir,
            angle_per_frame_deg=args.angle_deg,
            voxel_downsample_m=args.voxel_downsample_mm * 1e-3,
            voxel_volume_mm=args.voxel_volume_mm,
            random_views_dir=random_dir if has_random else None,
            random_masks_dir=random_masks if has_random and os.path.isdir(random_masks) else None,
        )
        results[sid] = info

    json_path = os.path.join(args.output_dir, "stone_volumes_gt.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Saved volumes JSON: %s", json_path)
    LOG.info("Done — processed %d stones.", len(results))


if __name__ == "__main__":
    main()
