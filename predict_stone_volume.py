#!/usr/bin/env python3
"""Predict stone volume from sparse depth maps using StoneReconNet.

Pipeline:
  1. Model segments stone points and registers them via RPF flow (Euler ODE).
  2. Poisson surface reconstruction creates a watertight mesh from the
     registered stone point cloud.
  3. Volume is computed geometrically from the mesh.

No neural volume regression -- all volume computation is geometric.

Supports both turntable views (fixed camera) and random views (arbitrary
camera positions with poses loaded from poses.json).

Usage:
    # Turntable views only:
    python predict_stone_volume.py \
        --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
        --intrinsics splits/stone/intrinsics.txt \
        --sequence stone_01 \
        --checkpoint models/stone_recon_net.pt

    # With additional random views:
    python predict_stone_volume.py \
        --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
        --random_depth_dir stone_syn_dataset/stone_01_random_npy \
        --intrinsics splits/stone/intrinsics.txt \
        --sequence stone_01 \
        --checkpoint models/stone_recon_net.pt

Output:
    - stone_registered.ply        (registered stone point cloud)
    - stone_mesh.ply              (watertight Poisson mesh)
    - volume_report.txt           (geometric volume and statistics)
    - prediction_result.json      (machine-readable results)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

from neural_pipeline.geometry import (  # noqa: E402
    Intrinsics,
    _backproject_full,
    load_intrinsics,
    make_pcd,
)
from volume_estimation.model import StoneReconNet, StoneReconNetConfig  # noqa: E402

LOG = logging.getLogger("predict_stone_volume")


def _load_depth_files(depth_dir: str) -> List[str]:
    """Find and sort .npy depth files (excludes poses.json)."""
    files = sorted(
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.lower().endswith(".npy")
    )
    if not files:
        raise FileNotFoundError(f"No .npy files in {depth_dir}")
    return files


def _load_poses_for_inference(depth_dir: str) -> Optional[Dict[int, np.ndarray]]:
    """Load poses.json from a depth directory if it exists."""
    poses_path = os.path.join(depth_dir, "poses.json")
    if not os.path.isfile(poses_path):
        return None
    with open(poses_path, "r") as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float64) for k, v in data.items()}


def _prepare_input(
    depth_files: List[str],
    intrinsics: Intrinsics,
    max_points_per_view: int = 4096,
    device: str = "cuda",
    random_depth_files: Optional[List[str]] = None,
    random_poses: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, int, int]:
    """Load depth files, back-project, and prepare model input.

    Returns:
        batch: Model input dict.
        centroid: Point cloud centroid for de-centering.
        n_turntable: Number of turntable views loaded.
        n_random: Number of random views loaded.
    """
    all_pts = []
    all_view_ids = []
    view_counter = 0

    for path in depth_files:
        depth = np.load(path).astype(np.float32)
        if depth.shape != (intrinsics.height, intrinsics.width):
            LOG.warning("Skipping %s: shape %s", path, depth.shape)
            continue

        pts_cam, _ = _backproject_full(depth, intrinsics, stride=1)
        if pts_cam.shape[0] == 0:
            continue

        pts = pts_cam.astype(np.float32)
        if pts.shape[0] > max_points_per_view:
            choice = np.random.choice(pts.shape[0], max_points_per_view, replace=False)
            pts = pts[choice]

        view_id = np.full(pts.shape[0], view_counter, dtype=np.int64)
        all_pts.append(pts)
        all_view_ids.append(view_id)
        view_counter += 1

    n_turntable = view_counter

    n_random = 0
    if random_depth_files:
        for path in random_depth_files:
            depth = np.load(path).astype(np.float32)
            if depth.shape != (intrinsics.height, intrinsics.width):
                LOG.warning("Skipping random %s: shape %s", path, depth.shape)
                continue

            pts_cam, _ = _backproject_full(depth, intrinsics, stride=1)
            if pts_cam.shape[0] == 0:
                continue

            pts = pts_cam.astype(np.float32)
            if pts.shape[0] > max_points_per_view:
                choice = np.random.choice(pts.shape[0], max_points_per_view, replace=False)
                pts = pts[choice]

            view_id = np.full(pts.shape[0], view_counter, dtype=np.int64)
            all_pts.append(pts)
            all_view_ids.append(view_id)
            view_counter += 1
            n_random += 1

    if not all_pts:
        raise RuntimeError("No valid points from any depth file")

    points = np.concatenate(all_pts, axis=0)
    view_ids = np.concatenate(all_view_ids, axis=0)

    centroid = points.mean(axis=0)
    points = points - centroid

    N = points.shape[0]
    batch = {
        "points": torch.from_numpy(points).unsqueeze(0).to(device),
        "view_ids": torch.from_numpy(view_ids).unsqueeze(0).to(device),
        "pad_mask": torch.zeros(1, N, dtype=torch.bool, device=device),
        "n_points": torch.tensor([N], dtype=torch.int64, device=device),
    }

    return batch, centroid, n_turntable, n_random


def poisson_mesh(
    points: np.ndarray,
    depth: int = 9,
    density_quantile: float = 0.01,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Poisson surface reconstruction from a point cloud.

    Args:
        points: (N, 3) point positions.
        depth: Octree depth for Poisson reconstruction (higher = finer detail).
        density_quantile: Remove low-density vertices (trims outlier geometry).

    Returns:
        mesh: Triangle mesh (watertight when possible).
        pcd: Point cloud with estimated normals (used for reconstruction).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array(points.mean(axis=0), dtype=np.float64)
    )
    # Flip normals to point outward (orient_normals_towards_camera puts them
    # toward the centroid, so we invert)
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True,
    )

    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    return mesh, pcd


def _mesh_volume_safe(mesh: o3d.geometry.TriangleMesh) -> float:
    """Compute mesh volume, returning 0 if the mesh is not watertight."""
    if mesh.is_watertight():
        return abs(mesh.get_volume())

    LOG.warning("Mesh is not watertight; attempting to close holes")
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy = mesh_copy.remove_degenerate_triangles()
    mesh_copy = mesh_copy.remove_duplicated_triangles()
    mesh_copy = mesh_copy.remove_duplicated_vertices()
    mesh_copy = mesh_copy.remove_non_manifold_edges()

    if mesh_copy.is_watertight():
        return abs(mesh_copy.get_volume())

    LOG.warning("Could not make mesh watertight; estimating volume from convex hull")
    try:
        hull, _ = mesh_copy.compute_convex_hull()
        if hull.is_watertight():
            return abs(hull.get_volume())
    except Exception:
        pass

    return 0.0


@torch.no_grad()
def predict(
    model: StoneReconNet,
    batch: Dict[str, torch.Tensor],
    centroid: np.ndarray,
    flow_steps: int = 10,
    poisson_depth: int = 9,
) -> Dict:
    """Full inference: segment -> RPF registration -> Poisson mesh -> volume.

    Returns dict with registered points, mesh, and geometric volume.
    """
    model.eval()

    registered_pts, seg_logits = model.sample_rectified_flow(
        batch, num_steps=flow_steps,
    )

    seg_probs = torch.sigmoid(seg_logits[0]).cpu().numpy()
    stone_mask_full = seg_probs > 0.5

    flow_pts = registered_pts[0].cpu().numpy()

    input_pts = batch["points"][0].cpu().numpy()
    stone_pts_input = input_pts[stone_mask_full] + centroid

    n_stone = int(stone_mask_full.sum())
    n_total = input_pts.shape[0]
    seg_ratio = n_stone / max(n_total, 1)

    results = {
        "flow_registered_points": flow_pts,
        "stone_points_input": stone_pts_input,
        "seg_probs": seg_probs,
        "n_stone_points": n_stone,
        "n_total_points": n_total,
        "n_flow_points": flow_pts.shape[0],
        "seg_ratio": seg_ratio,
        "flow_steps": flow_steps,
    }

    if flow_pts.shape[0] < 100:
        LOG.warning("Too few registered points (%d) for mesh reconstruction", flow_pts.shape[0])
        results["volume_cm3"] = 0.0
        results["volume_mm3"] = 0.0
        results["mesh"] = None
        return results

    mesh, pcd = poisson_mesh(flow_pts, depth=poisson_depth)

    volume_cm3 = _mesh_volume_safe(mesh)

    results["volume_cm3"] = volume_cm3
    results["volume_mm3"] = volume_cm3 * 1000.0
    results["mesh"] = mesh
    results["mesh_vertices"] = len(mesh.vertices)
    results["mesh_triangles"] = len(mesh.triangles)
    results["mesh_watertight"] = mesh.is_watertight()

    return results


def save_results(
    results: Dict,
    output_dir: str,
    depth_dir: str,
    n_turntable: int,
    n_random: int,
    elapsed_s: float,
    random_depth_dir: Optional[str] = None,
):
    """Save mesh, point cloud(s), and reports."""
    os.makedirs(output_dir, exist_ok=True)
    n_views = n_turntable + n_random

    flow_pts = results["flow_registered_points"]
    if flow_pts.shape[0] > 0:
        pcd = make_pcd(flow_pts, estimate_normals=True)
        ply_path = os.path.join(output_dir, "stone_registered.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        LOG.info("Saved registered cloud: %s (%d pts)", ply_path, flow_pts.shape[0])

    stone_pts = results["stone_points_input"]
    if stone_pts.shape[0] > 0:
        stone_pcd = make_pcd(stone_pts, estimate_normals=True)
        stone_ply = os.path.join(output_dir, "stone_segmented.ply")
        o3d.io.write_point_cloud(stone_ply, stone_pcd)

    mesh = results.get("mesh")
    if mesh is not None:
        mesh_path = os.path.join(output_dir, "stone_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        LOG.info("Saved mesh: %s (%d vertices, %d triangles)",
                 mesh_path, len(mesh.vertices), len(mesh.triangles))

    report_lines = [
        "=" * 60,
        "StoneReconNet -- Volume Prediction Report",
        "=" * 60,
        "",
        f"Input (turntable): {depth_dir}",
    ]
    if random_depth_dir:
        report_lines.append(f"Input (random):    {random_depth_dir}")
    report_lines += [
        f"Views:            {n_views} (turntable={n_turntable}, random={n_random})",
        f"Total points:     {results['n_total_points']}",
        f"Stone points:     {results['n_stone_points']} ({results['seg_ratio']:.1%})",
        f"Flow points:      {results['n_flow_points']}",
        f"Flow ODE steps:   {results['flow_steps']}",
        "",
        "--- Mesh Reconstruction ---",
    ]

    if mesh is not None:
        report_lines += [
            f"Mesh vertices:    {results['mesh_vertices']}",
            f"Mesh triangles:   {results['mesh_triangles']}",
            f"Watertight:       {results['mesh_watertight']}",
            "",
            f"Volume:           {results['volume_cm3']:.6f} cm3",
            f"                  {results['volume_mm3']:.2f} mm3",
        ]
    else:
        report_lines.append("Mesh: FAILED (too few points)")

    report_lines += [
        "",
        f"Inference time:   {elapsed_s:.3f} s",
        "",
        "Output files:",
        "  Registered PC:  stone_registered.ply",
        "  Segmented PC:   stone_segmented.ply",
    ]
    if mesh is not None:
        report_lines.append("  Mesh:           stone_mesh.ply")
    report_lines += [
        "  Report:         volume_report.txt",
        "=" * 60,
    ]

    report_path = os.path.join(output_dir, "volume_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    LOG.info("Saved report: %s", report_path)

    for line in report_lines:
        print(line)

    result_json = {
        "volume_cm3": results["volume_cm3"],
        "volume_mm3": results["volume_mm3"],
        "n_stone_points": results["n_stone_points"],
        "n_total_points": results["n_total_points"],
        "n_flow_points": results["n_flow_points"],
        "seg_ratio": results["seg_ratio"],
        "flow_steps": results["flow_steps"],
        "n_views": n_views,
        "n_turntable_views": n_turntable,
        "n_random_views": n_random,
        "inference_time_s": elapsed_s,
        "input_dir": depth_dir,
    }
    if random_depth_dir:
        result_json["random_input_dir"] = random_depth_dir
    if mesh is not None:
        result_json["mesh_vertices"] = results["mesh_vertices"]
        result_json["mesh_triangles"] = results["mesh_triangles"]
        result_json["mesh_watertight"] = results["mesh_watertight"]

    with open(os.path.join(output_dir, "prediction_result.json"), "w") as f:
        json.dump(result_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Predict stone volume: neural segmentation + RPF flow registration + Poisson mesh"
    )
    parser.add_argument("--depth_dir", required=True,
                        help="Directory with turntable .npy depth files")
    parser.add_argument("--random_depth_dir", default=None,
                        help="Optional directory with random-view .npy depth files "
                             "(must contain poses.json with per-view 4x4 extrinsics)")
    parser.add_argument("--intrinsics", required=True,
                        help="Path to intrinsics.txt")
    parser.add_argument("--sequence", required=True,
                        help="Stone sequence ID (e.g. stone_01)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--output_dir", default="volume_output",
                        help="Output directory")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--max_points_per_view", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flow_steps", type=int, default=10,
                        help="Number of Euler ODE steps for RPF flow registration")
    parser.add_argument("--poisson_depth", type=int, default=9,
                        help="Octree depth for Poisson surface reconstruction")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    LOG.info("Loading model from %s", args.checkpoint)
    cfg = StoneReconNetConfig()
    model = StoneReconNet(cfg)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(args.device)
    model.eval()
    LOG.info("Model loaded (%s)", args.device)

    K = load_intrinsics(args.intrinsics, args.sequence, args.width, args.height)

    depth_files = _load_depth_files(args.depth_dir)
    LOG.info("Found %d turntable depth files in %s", len(depth_files), args.depth_dir)

    random_depth_files = None
    random_poses = None
    if args.random_depth_dir:
        random_depth_files = _load_depth_files(args.random_depth_dir)
        random_poses = _load_poses_for_inference(args.random_depth_dir)
        LOG.info("Found %d random-view depth files in %s",
                 len(random_depth_files), args.random_depth_dir)
        if random_poses:
            LOG.info("Loaded %d poses from poses.json", len(random_poses))
        else:
            LOG.warning("No poses.json found in %s — random views will be treated "
                        "as unposed (model must handle registration)", args.random_depth_dir)

    LOG.info("Preparing input...")
    batch, centroid, n_turntable, n_random = _prepare_input(
        depth_files, K,
        max_points_per_view=args.max_points_per_view,
        device=args.device,
        random_depth_files=random_depth_files,
        random_poses=random_poses,
    )

    LOG.info("Running inference (RPF flow + Poisson mesh)...")
    t0 = time.perf_counter()
    results = predict(
        model, batch, centroid,
        flow_steps=args.flow_steps,
        poisson_depth=args.poisson_depth,
    )
    elapsed = time.perf_counter() - t0

    save_results(
        results, args.output_dir, args.depth_dir,
        n_turntable, n_random, elapsed,
        random_depth_dir=args.random_depth_dir,
    )
    LOG.info("Done.")


if __name__ == "__main__":
    main()
