#!/usr/bin/env python3
"""Predict stone volume using StoneCompletionNet + NKSR.

Pipeline:
  1. Load sparse depth views, back-project with turntable rotation.
  2. StoneCompletionNet segments stone and completes the surface.
  3. NKSR (pre-trained) converts completed cloud to watertight mesh.
  4. Volume is computed geometrically from the mesh.

Usage:
    python predict_stone_v2.py \
        --depth_dir stone_syn_dataset/stone_01_sparse_npy_n18 \
        --intrinsics splits/stone/intrinsics.txt \
        --sequence stone_01 \
        --checkpoint stone_completion_output/stone_completion_net.pt \
        --output_dir volume_output_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

from stone_completion.model import (  # noqa: E402
    CompletionConfig,
    StoneCompletionNet,
    _farthest_point_sample,
    _index_points,
)

LOG = logging.getLogger("predict_stone_v2")


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained)
# ---------------------------------------------------------------------------

def _backproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _turntable_rotation_y(frame_index: int, angle_deg: float = 3.0) -> np.ndarray:
    theta = math.radians(frame_index * angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c;  T[0, 2] = s
    T[2, 0] = -s; T[2, 2] = c
    return T


def _extract_frame_index(filepath: str) -> int:
    digits = "".join(c for c in Path(filepath).stem if c.isdigit())
    return int(digits) if digits else 0


def _parse_intrinsics(
    path: str, seq: str, W: int, H: int
) -> Tuple[float, float, float, float]:
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == seq:
                fn, fyn, cxn, cyn = (float(x) for x in parts[1:5])
                return fn * W, fyn * H, cxn * W, cyn * H
    raise KeyError(f"Sequence '{seq}' not found in {path}")


def _fps_np(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] <= n:
        if pts.shape[0] == 0:
            return np.zeros((n, 3), dtype=np.float32)
        choice = np.random.choice(pts.shape[0], n, replace=True)
        return pts[choice]
    selected = [np.random.randint(pts.shape[0])]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(n - 1):
        d = np.sum((pts - pts[selected[-1]]) ** 2, axis=-1)
        dists = np.minimum(dists, d)
        selected.append(np.argmax(dists))
    return pts[np.array(selected)]


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _poisson_mesh(pts: np.ndarray, depth: int = 9) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Poisson surface reconstruction from points with estimated normals."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities_np = np.asarray(densities)
    if len(densities_np) > 0:
        threshold = np.quantile(densities_np, 0.05)
        remove_mask = densities_np < threshold
        mesh.remove_vertices_by_mask(remove_mask)
    mesh.compute_vertex_normals()
    return mesh, pcd


def _try_nksr(pts: np.ndarray, device: str = "cuda") -> Optional[o3d.geometry.TriangleMesh]:
    """Attempt NKSR reconstruction. Returns None if nksr not available."""
    try:
        from neural_pipeline.nksr_loader import load_nksr
    except ImportError:
        LOG.info("NKSR not available, falling back to Poisson")
        return None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        nksr_model = load_nksr(weights="", device=device)
        mesh = nksr_model.reconstruct(pcd, voxel_m=0.0005)
        LOG.info("NKSR reconstruction: %d verts, %d tris",
                 len(mesh.vertices), len(mesh.triangles))
        return mesh
    except Exception as e:
        LOG.warning("NKSR failed: %s. Falling back to Poisson.", e)
        return None


def _mesh_volume(mesh: o3d.geometry.TriangleMesh) -> Tuple[float, bool]:
    """Compute volume from mesh. Returns (volume_m3, is_watertight)."""
    watertight = mesh.is_watertight()
    if watertight:
        vol = mesh.get_volume()
    else:
        try:
            hull, _ = mesh.compute_convex_hull()
            vol = hull.get_volume()
        except Exception:
            vol = 0.0
    return vol, watertight


# ---------------------------------------------------------------------------
# Main prediction
# ---------------------------------------------------------------------------

def predict(
    depth_dir: str,
    intrinsics_path: str,
    sequence: str,
    checkpoint: str,
    output_dir: str,
    width: int = 1024,
    height: int = 576,
    device: str = "cuda",
    input_points: int = 4096,
    completion_points: int = 2048,
    use_nksr: bool = True,
    poisson_depth: int = 9,
) -> Dict:
    """Run full prediction pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    fx, fy, cx, cy = _parse_intrinsics(intrinsics_path, sequence, width, height)

    depth_files = sorted([
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.lower().endswith(".npy")
    ])
    if not depth_files:
        raise RuntimeError(f"No .npy files in {depth_dir}")

    LOG.info("Loading %d depth views from %s", len(depth_files), depth_dir)

    all_pts = []
    for dpath in depth_files:
        depth = np.load(dpath).astype(np.float32)
        if depth.shape != (height, width):
            LOG.warning("Skipping %s: shape %s", dpath, depth.shape)
            continue

        pts_cam = _backproject(depth, fx, fy, cx, cy)
        if pts_cam.shape[0] == 0:
            continue

        frame_idx = _extract_frame_index(dpath)
        view_center = pts_cam.mean(axis=0)
        pts_centered = pts_cam - view_center
        T = _turntable_rotation_y(frame_idx, 3.0)
        R = T[:3, :3].astype(np.float32)
        pts_world = (R @ pts_centered.T).T

        if pts_world.shape[0] > 4096:
            choice = np.random.choice(pts_world.shape[0], 4096, replace=False)
            pts_world = pts_world[choice]

        all_pts.append(pts_world)

    if not all_pts:
        raise RuntimeError("No valid points from depth files")

    scene_pts = np.concatenate(all_pts, axis=0)
    centroid = scene_pts.mean(axis=0)
    scene_pts = scene_pts - centroid

    LOG.info("Total scene points: %d", scene_pts.shape[0])

    scene_fps = _fps_np(scene_pts, input_points)

    LOG.info("Loading model from %s", checkpoint)
    cfg = CompletionConfig(
        input_points=input_points,
        fine1_points=completion_points,
        fine2_points=8192,
    )
    model = StoneCompletionNet(cfg)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    scene_tensor = torch.from_numpy(scene_fps).unsqueeze(0).float().to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        seg_mask, completed = model.complete(scene_tensor)

    completed_pts = completed[0].cpu().numpy()
    seg_mask_np = seg_mask[0].cpu().numpy()

    seg_pts = scene_fps[seg_mask_np]
    n_stone = seg_mask_np.sum()
    LOG.info("Segmented %d/%d points as stone (%.1f%%)",
             n_stone, input_points, 100 * n_stone / input_points)
    LOG.info("Completion output: %d points", completed_pts.shape[0])

    completed_pts = completed_pts + centroid

    mesh = None
    if use_nksr:
        mesh = _try_nksr(completed_pts, device=device)

    if mesh is None:
        LOG.info("Using Poisson reconstruction (depth=%d)", poisson_depth)
        mesh, _ = _poisson_mesh(completed_pts, depth=poisson_depth)

    volume_m3, watertight = _mesh_volume(mesh)
    volume_cm3 = volume_m3 * 1e6
    volume_mm3 = volume_m3 * 1e9
    elapsed = time.perf_counter() - t0

    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points = o3d.utility.Vector3dVector((seg_pts + centroid).astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(output_dir, "stone_segmented.ply"), seg_pcd)

    comp_pcd = o3d.geometry.PointCloud()
    comp_pcd.points = o3d.utility.Vector3dVector(completed_pts.astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(output_dir, "stone_completed.ply"), comp_pcd)

    o3d.io.write_triangle_mesh(os.path.join(output_dir, "stone_mesh.ply"), mesh)

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)

    report = f"""\
============================================================
StoneCompletionNet + NKSR -- Volume Prediction Report
============================================================

Input:            {depth_dir}
Views:            {len(depth_files)}
Scene points:     {scene_pts.shape[0]}
Segmented stone:  {n_stone} ({100 * n_stone / input_points:.1f}%)
Completed points: {completed_pts.shape[0]}

--- Mesh Reconstruction ---
Method:           {"NKSR" if use_nksr else "Poisson"}
Mesh vertices:    {n_verts}
Mesh triangles:   {n_tris}
Watertight:       {watertight}

Volume:           {volume_cm3:.6f} cm3
                  {volume_mm3:.2f} mm3

Inference time:   {elapsed:.3f} s

Output files:
  Segmented PC:   stone_segmented.ply
  Completed PC:   stone_completed.ply
  Mesh:           stone_mesh.ply
  Report:         volume_report.txt
============================================================
"""

    print(report)
    with open(os.path.join(output_dir, "volume_report.txt"), "w") as f:
        f.write(report)

    result = {
        "sequence": sequence,
        "n_views": len(depth_files),
        "n_scene_points": int(scene_pts.shape[0]),
        "n_stone_points": int(n_stone),
        "n_completed_points": int(completed_pts.shape[0]),
        "n_mesh_verts": n_verts,
        "n_mesh_tris": n_tris,
        "watertight": watertight,
        "volume_cm3": float(volume_cm3),
        "volume_mm3": float(volume_mm3),
        "inference_time_s": float(elapsed),
        "mesh_method": "NKSR" if use_nksr else "Poisson",
    }

    with open(os.path.join(output_dir, "prediction_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict stone volume using StoneCompletionNet + NKSR"
    )
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="volume_output_v2")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_points", type=int, default=4096)
    parser.add_argument("--completion_points", type=int, default=2048)
    parser.add_argument("--no_nksr", action="store_true",
                        help="Force Poisson reconstruction instead of NKSR")
    parser.add_argument("--poisson_depth", type=int, default=9)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    predict(
        depth_dir=args.depth_dir,
        intrinsics_path=args.intrinsics,
        sequence=args.sequence,
        checkpoint=args.checkpoint,
        output_dir=os.path.join(args.output_dir, args.sequence),
        width=args.width,
        height=args.height,
        device=args.device,
        input_points=args.input_points,
        completion_points=args.completion_points,
        use_nksr=not args.no_nksr,
        poisson_depth=args.poisson_depth,
    )


if __name__ == "__main__":
    main()
