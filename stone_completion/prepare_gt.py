"""Generate GT complete stone point clouds from depth .npy files + masks.

Same approach as volume_estimation/prepare_gt.py but self-contained:
  1. Load ALL 120 turntable depth maps (+ optional random views)
  2. Back-project to 3D using intrinsics
  3. Apply turntable rotation (frame_idx * 3 deg about Y)
  4. Keep only stone pixels (from masks)
  5. Merge into dense registered point cloud
  6. Voxel downsample + outlier removal
  7. Save as stone_XX_gt_complete.ply

No Blender mesh export needed -- uses only the existing .npy depth data.

Usage:
    python -m stone_completion.prepare_gt \
        --dataset_dir stone_syn_dataset \
        --intrinsics splits/stone/intrinsics.txt \
        --output_dir stone_syn_dataset/gt_complete \
        --stones stone_01 stone_02 ... stone_12
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

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

LOG = logging.getLogger("stone_completion.prepare_gt")


# ---------------------------------------------------------------------------
# Geometry (self-contained, no imports from other packages)
# ---------------------------------------------------------------------------

def _backproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project depth to 3D. Returns (points (N,3), flat_indices (N,))."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(depth) & (depth > 0)
    flat_idx = np.where(valid.ravel())[0]
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pts, flat_idx


def _turntable_rotation_y(frame_index: int, angle_per_frame_deg: float = 3.0) -> np.ndarray:
    theta = math.radians(frame_index * angle_per_frame_deg)
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c;  T[0, 2] = s
    T[2, 0] = -s; T[2, 2] = c
    return T


def _extract_frame_index(filepath: str) -> int:
    digits = "".join(c for c in Path(filepath).stem if c.isdigit())
    return int(digits) if digits else 0


def _load_mask(mask_path: str, H: int, W: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(mask_path).convert("L")
    mask = np.array(img, dtype=np.uint8)
    if mask.shape != (H, W):
        img = img.resize((W, H), Image.NEAREST)
        mask = np.array(img, dtype=np.uint8)
    return mask > 127


def _parse_intrinsics(
    path: str, stone_id: str, W: int, H: int
) -> Tuple[float, float, float, float]:
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == stone_id:
                fn, fyn, cxn, cyn = (float(x) for x in parts[1:5])
                return fn * W, fyn * H, cxn * W, cyn * H
    raise KeyError(f"Stone '{stone_id}' not found in {path}")


def _load_poses_json(path: str) -> Dict[int, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float64) for k, v in data.items()}


def _compute_volume_voxel(points: np.ndarray, voxel_mm: float = 0.5) -> float:
    if points.shape[0] < 100:
        return 0.0
    pts_mm = points * 1000.0
    mins = pts_mm.min(axis=0)
    indices = ((pts_mm - mins) / voxel_mm).astype(np.int64)
    unique = np.unique(indices, axis=0)
    return float(len(unique) * (voxel_mm ** 3))


def _compute_volume_hull(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points * 1000.0)
        return float(hull.volume)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_stone(
    stone_id: str,
    dataset_dir: str,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    output_dir: str,
    angle_deg: float = 3.0,
    voxel_mm: float = 0.5,
    random_suffix: str = "_random_npy",
) -> Dict:
    """Process one stone: merge all views into GT complete cloud."""
    depth_dir = os.path.join(dataset_dir, f"{stone_id}_depth_npy")
    mask_dir = os.path.join(dataset_dir, stone_id, "masks")
    random_dir = os.path.join(dataset_dir, f"{stone_id}{random_suffix}")

    if not os.path.isdir(depth_dir):
        LOG.warning("Depth dir not found: %s", depth_dir)
        return {"stone_id": stone_id, "n_points": 0, "volume_cm3": 0.0}

    mask_by_idx: Dict[int, str] = {}
    if os.path.isdir(mask_dir):
        for f in os.listdir(mask_dir):
            if f.lower().endswith(".png"):
                idx = _extract_frame_index(f)
                mask_by_idx[idx] = os.path.join(mask_dir, f)

    random_mask_dir = os.path.join(random_dir, "masks")
    if os.path.isdir(random_mask_dir):
        for f in os.listdir(random_mask_dir):
            if f.lower().endswith(".png"):
                idx = _extract_frame_index(f)
                mask_by_idx[idx] = os.path.join(random_mask_dir, f)

    all_pts: List[np.ndarray] = []
    n_tt, n_rand = 0, 0

    def _process_dir(
        ddir: str, poses: Optional[Dict[int, np.ndarray]]
    ) -> int:
        count = 0
        for fname in sorted(os.listdir(ddir)):
            if not fname.lower().endswith(".npy"):
                continue
            frame_idx = _extract_frame_index(fname)

            depth = np.load(os.path.join(ddir, fname)).astype(np.float32)
            if depth.shape != (height, width):
                continue

            pts_cam, flat_idx = _backproject(depth, fx, fy, cx, cy)
            if pts_cam.shape[0] == 0:
                continue

            if frame_idx in mask_by_idx:
                mask = _load_mask(mask_by_idx[frame_idx], height, width)
                stone_sel = mask.ravel()[flat_idx.astype(np.int64)]
                pts_cam = pts_cam[stone_sel]
            if pts_cam.shape[0] < 10:
                continue

            if poses is not None and frame_idx in poses:
                R = poses[frame_idx][:3, :3]
                t = poses[frame_idx][:3, 3]
                pts_world = (R @ pts_cam.T).T + t
            else:
                view_center = pts_cam.mean(axis=0)
                pts_centered = pts_cam - view_center
                T = _turntable_rotation_y(frame_idx, angle_deg)
                R = T[:3, :3]
                pts_world = (R @ pts_centered.T).T

            all_pts.append(pts_world)
            count += 1
        return count

    n_tt = _process_dir(depth_dir, None)

    if os.path.isdir(random_dir):
        poses_path = os.path.join(random_dir, "poses.json")
        if os.path.isfile(poses_path):
            rand_poses = _load_poses_json(poses_path)
            n_rand = _process_dir(random_dir, rand_poses)

    if not all_pts:
        LOG.warning("No valid points for %s", stone_id)
        return {"stone_id": stone_id, "n_points": 0, "volume_cm3": 0.0}

    merged = np.concatenate(all_pts, axis=0)
    LOG.info(
        "%s: %d points from %d views (tt=%d, rand=%d)",
        stone_id, merged.shape[0], n_tt + n_rand, n_tt, n_rand,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_mm * 1e-3)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=6 * voxel_mm * 1e-3, max_nn=40
        )
    )

    final_pts = np.asarray(pcd.points)
    LOG.info("  After cleanup: %d points", final_pts.shape[0])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stone_id}_gt_complete.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    LOG.info("  Saved: %s", out_path)

    vol_vox = _compute_volume_voxel(final_pts, voxel_mm)
    vol_hull = _compute_volume_hull(final_pts)
    vol_mm3 = (vol_vox + vol_hull) / 2.0
    vol_cm3 = vol_mm3 / 1000.0

    LOG.info("  Volume: %.2f mm3 (%.4f cm3)", vol_mm3, vol_cm3)

    return {
        "stone_id": stone_id,
        "n_points": final_pts.shape[0],
        "n_views": n_tt + n_rand,
        "volume_mm3": round(vol_mm3, 2),
        "volume_cm3": round(vol_cm3, 6),
        "ply_path": out_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GT complete stone point clouds from depth .npy + masks"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_dir", default="stone_syn_dataset/gt_complete")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--stones", nargs="*", default=None,
                        help="Stone IDs (default: auto-detect)")
    parser.add_argument("--angle_deg", type=float, default=3.0)
    parser.add_argument("--voxel_mm", type=float, default=0.5)
    parser.add_argument("--random_suffix", default="_random_npy")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if args.stones:
        stone_ids = args.stones
    else:
        stone_ids = sorted([
            d for d in os.listdir(args.dataset_dir)
            if d.startswith("stone_") and os.path.isdir(os.path.join(args.dataset_dir, d))
            and "_depth_npy" not in d and "_sparse" not in d and "_random" not in d
        ])
        LOG.info("Auto-detected: %s", stone_ids)

    if not stone_ids:
        LOG.error("No stones found")
        return

    results = {}
    for sid in stone_ids:
        fx, fy, cx, cy = _parse_intrinsics(
            args.intrinsics, sid, args.width, args.height
        )
        info = process_stone(
            sid, args.dataset_dir, fx, fy, cx, cy,
            args.width, args.height, args.output_dir,
            args.angle_deg, args.voxel_mm, args.random_suffix,
        )
        results[sid] = info

    json_path = os.path.join(args.output_dir, "stone_volumes_gt.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Saved: %s", json_path)
    LOG.info("Done -- %d stones processed", len(results))


if __name__ == "__main__":
    main()
