#!/usr/bin/env python3
"""Predict stone volume from depth maps using the registered pipeline.

Pipeline (matching training):
  1. Back-project each depth map to 3D (camera space).
  2. Segment stone pixels using masks.
  3. Register each view using known turntable rotation (from prepare_gt.py).
  4. Merge views into a single registered partial cloud.
  5. Feed the registered stone cloud to the flow model.
  6. Euler ODE integration produces the complete stone shape.
  7. Poisson surface reconstruction creates a watertight mesh.
  8. Volume is computed geometrically from the mesh.

Usage:
    python predict_stone_volume.py \\
        --depth_dir stone_syn_dataset/stone_01_depth_npy \\
        --mask_dir stone_syn_dataset/stone_01/masks \\
        --intrinsics splits/stone/intrinsics.txt \\
        --sequence stone_01 \\
        --checkpoint volume_training_output/stone_recon_net.pt \\
        --gt_cloud_dir volume_estimation/gt_data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
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


def _load_depth_files(depth_dir: str) -> List[Tuple[int, str]]:
    """Find .npy depth files and extract frame indices.

    Returns list of (frame_idx, file_path) sorted by frame index.
    """
    results = []
    for f in sorted(os.listdir(depth_dir)):
        if not f.lower().endswith(".npy"):
            continue
        digits = "".join(c for c in Path(f).stem if c.isdigit())
        if not digits:
            continue
        results.append((int(digits), os.path.join(depth_dir, f)))
    if not results:
        raise FileNotFoundError(f"No .npy files in {depth_dir}")
    return results


def _load_mask(mask_dir: str, frame_idx: int, H: int, W: int) -> Optional[np.ndarray]:
    """Try to load a mask PNG matching the frame index. Returns flat bool array."""
    if not mask_dir or not os.path.isdir(mask_dir):
        return None
    for f in os.listdir(mask_dir):
        if not f.lower().endswith(".png"):
            continue
        digits = "".join(c for c in Path(f).stem if c.isdigit())
        if digits and int(digits) == frame_idx:
            from PIL import Image
            img = Image.open(os.path.join(mask_dir, f)).convert("L")
            arr = np.array(img, dtype=np.uint8)
            if arr.shape != (H, W):
                img = img.resize((W, H), Image.NEAREST)
                arr = np.array(img, dtype=np.uint8)
            return (arr > 127).ravel()
    return None


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Greedy FPS to downsample to n points."""
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


def _prepare_input(
    depth_files: List[Tuple[int, str]],
    intrinsics: Intrinsics,
    mask_dir: Optional[str],
    registration: Dict[str, np.ndarray],
    angle_per_frame_deg: float = 3.0,
    max_points_per_view: int = 4096,
    merged_cloud_points: int = 8192,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, int]:
    """Load depth files, apply masks, register to common frame, and merge.

    Returns:
        batch: Model input dict with registered stone-only points.
        gt_centroid: The centroid used for centering (for de-centering output).
        n_views: Number of views loaded.
    """
    R_floor_up = registration.get("R_floor_up")
    turntable_center = registration.get("turntable_center")
    gt_centroid = registration.get("gt_centroid", np.zeros(3, dtype=np.float64))

    all_pts: List[np.ndarray] = []

    for frame_idx, npy_path in depth_files:
        depth = np.load(npy_path).astype(np.float32)
        if depth.shape != (intrinsics.height, intrinsics.width):
            LOG.warning("Skipping %s: shape %s", npy_path, depth.shape)
            continue

        pts_cam, flat_idx = _backproject_full(depth, intrinsics, stride=1)
        if pts_cam.shape[0] == 0:
            continue

        mask = _load_mask(mask_dir, frame_idx, intrinsics.height, intrinsics.width)
        if mask is not None:
            stone_sel = mask[flat_idx.astype(np.int64)]
            pts_cam = pts_cam[stone_sel]

        if pts_cam.shape[0] < 10:
            continue

        if pts_cam.shape[0] > max_points_per_view:
            choice = np.random.choice(pts_cam.shape[0], max_points_per_view, replace=False)
            pts_cam = pts_cam[choice]

        pts = pts_cam.astype(np.float64)
        if R_floor_up is not None:
            pts = (R_floor_up @ pts.T).T
        if turntable_center is not None:
            pts = pts - turntable_center
        theta = math.radians(frame_idx * (-angle_per_frame_deg))
        c, s = math.cos(theta), math.sin(theta)
        R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        pts = (R_yaw @ pts.T).T

        all_pts.append(pts.astype(np.float32))

    if not all_pts:
        raise RuntimeError("No valid stone points from any depth file")

    n_views = len(all_pts)
    points = np.concatenate(all_pts, axis=0)
    LOG.info("Merged %d stone points from %d views", points.shape[0], n_views)

    points = (points - gt_centroid.astype(np.float32))
    points = _farthest_point_sample(points, merged_cloud_points)

    N = points.shape[0]
    batch = {
        "points": torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(device),
        "view_ids": torch.zeros(1, N, dtype=torch.long, device=device),
        "pad_mask": torch.zeros(1, N, dtype=torch.bool, device=device),
        "n_points": torch.tensor([N], dtype=torch.int64, device=device),
    }

    return batch, gt_centroid, n_views


def _clean_flow_cloud(points: np.ndarray, sigma_thresh: float = 2.0) -> np.ndarray:
    """Remove statistical outlier points from flow output."""
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    threshold = dists.mean() + sigma_thresh * dists.std()
    mask = dists < threshold
    cleaned = points[mask]
    n_removed = points.shape[0] - cleaned.shape[0]
    if n_removed > 0:
        LOG.info("Outlier removal: %d/%d points removed (%.1f%%)",
                 n_removed, points.shape[0], 100 * n_removed / points.shape[0])
    return cleaned


def poisson_mesh(
    points: np.ndarray,
    depth: int = 9,
    density_quantile: float = 0.01,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Poisson surface reconstruction from a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    LOG.info("After Open3D statistical outlier removal: %d points", len(pcd.points))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array(
            np.asarray(pcd.points).mean(axis=0), dtype=np.float64
        )
    )
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
    flow_steps: int = 20,
    poisson_depth: int = 9,
) -> Dict:
    """Inference: flow-based shape completion -> mesh -> volume.

    The model receives a registered, stone-only partial cloud and
    generates the complete stone shape via Euler ODE integration.
    """
    model.eval()

    flow_pts_raw, upsampled_pts = model.sample_rectified_flow(
        batch, num_steps=flow_steps,
    )

    flow_pts = flow_pts_raw[0].cpu().numpy()
    upsampled = upsampled_pts[0].cpu().numpy()

    input_pts = batch["points"][0].cpu().numpy()

    results = {
        "flow_points": flow_pts,
        "upsampled_points": upsampled,
        "input_points": input_pts,
        "centroid": centroid,
        "n_input_points": input_pts.shape[0],
        "n_flow_points": flow_pts.shape[0],
        "n_upsampled_points": upsampled.shape[0],
        "flow_steps": flow_steps,
    }

    mesh_pts = _clean_flow_cloud(upsampled, sigma_thresh=1.5)
    if mesh_pts.shape[0] < 50:
        LOG.warning("Too few upsampled points (%d) for mesh reconstruction",
                     mesh_pts.shape[0])
        results["volume_cm3"] = 0.0
        results["volume_mm3"] = 0.0
        results["mesh"] = None
        return results

    LOG.info("Building Poisson mesh from %d upsampled points "
             "(flow: %d -> upsampled: %d)",
             mesh_pts.shape[0], flow_pts.shape[0], upsampled.shape[0])

    mesh, pcd = poisson_mesh(mesh_pts, depth=poisson_depth)

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
    n_views: int,
    elapsed_s: float,
):
    """Save mesh, point cloud(s), and reports."""
    os.makedirs(output_dir, exist_ok=True)

    flow_pts = results["flow_points"]
    if flow_pts.shape[0] > 0:
        pcd = make_pcd(flow_pts, estimate_normals=True)
        ply_path = os.path.join(output_dir, "stone_flow.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        LOG.info("Saved flow cloud: %s (%d pts)", ply_path, flow_pts.shape[0])

    upsampled = results.get("upsampled_points")
    if upsampled is not None and upsampled.shape[0] > 0:
        up_pcd = make_pcd(upsampled, estimate_normals=True)
        up_path = os.path.join(output_dir, "stone_upsampled.ply")
        o3d.io.write_point_cloud(up_path, up_pcd)
        LOG.info("Saved upsampled cloud: %s (%d pts)", up_path, upsampled.shape[0])

    input_pts = results["input_points"]
    if input_pts.shape[0] > 0:
        in_pcd = make_pcd(input_pts, estimate_normals=True)
        in_ply = os.path.join(output_dir, "stone_input_registered.ply")
        o3d.io.write_point_cloud(in_ply, in_pcd)

    mesh = results.get("mesh")
    if mesh is not None:
        mesh_path = os.path.join(output_dir, "stone_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        LOG.info("Saved mesh: %s (%d vertices, %d triangles)",
                 mesh_path, len(mesh.vertices), len(mesh.triangles))

    report_lines = [
        "=" * 60,
        "StoneReconNet -- Volume Prediction Report (Registered Pipeline)",
        "=" * 60,
        "",
        f"Input:            {depth_dir}",
        f"Views:            {n_views}",
        f"Input points:     {results['n_input_points']} (registered, stone-only)",
        f"Flow points:      {results['n_flow_points']}",
        f"Upsampled points: {results['n_upsampled_points']}",
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
        "n_input_points": results["n_input_points"],
        "n_flow_points": results["n_flow_points"],
        "n_upsampled_points": results["n_upsampled_points"],
        "flow_steps": results["flow_steps"],
        "n_views": n_views,
        "inference_time_s": elapsed_s,
        "input_dir": depth_dir,
    }
    if mesh is not None:
        result_json["mesh_vertices"] = results["mesh_vertices"]
        result_json["mesh_triangles"] = results["mesh_triangles"]
        result_json["mesh_watertight"] = results["mesh_watertight"]

    with open(os.path.join(output_dir, "prediction_result.json"), "w") as f:
        json.dump(result_json, f, indent=2)


def _chamfer_distance_np(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Chamfer distance between two point clouds: (a->b, b->a, avg)."""
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a)
    db, _ = ta.query(b)
    return float(da.mean()), float(db.mean()), float((da.mean() + db.mean()) / 2)


def save_diagnostics(
    results: Dict,
    output_dir: str,
    gt_cloud_path: Optional[str] = None,
):
    """Save comprehensive diagnostics for evaluating model quality."""
    diag: Dict = {}

    flow = results["flow_points"]
    up = results["upsampled_points"]

    diag["flow_center"] = flow.mean(axis=0).tolist()
    diag["flow_std"] = flow.std(axis=0).tolist()
    diag["flow_span"] = (flow.max(axis=0) - flow.min(axis=0)).tolist()
    diag["flow_n_points"] = flow.shape[0]

    diag["upsampled_center"] = up.mean(axis=0).tolist()
    diag["upsampled_std"] = up.std(axis=0).tolist()
    diag["upsampled_span"] = (up.max(axis=0) - up.min(axis=0)).tolist()

    diag["volume_cm3"] = results.get("volume_cm3", 0.0)
    diag["mesh_watertight"] = results.get("mesh_watertight", False)

    if gt_cloud_path and os.path.isfile(gt_cloud_path):
        if gt_cloud_path.endswith(".npy"):
            gt = np.load(gt_cloud_path).astype(np.float32)
        else:
            gt_pcd = o3d.io.read_point_cloud(gt_cloud_path)
            gt = np.asarray(gt_pcd.points, dtype=np.float32)

        diag["gt_n_points"] = gt.shape[0]
        diag["gt_center"] = gt.mean(axis=0).tolist()
        diag["gt_std"] = gt.std(axis=0).tolist()
        diag["gt_span"] = (gt.max(axis=0) - gt.min(axis=0)).tolist()

        cd_flow = _chamfer_distance_np(flow, gt)
        diag["chamfer_flow_to_gt"] = cd_flow[0]
        diag["chamfer_gt_to_flow"] = cd_flow[1]
        diag["chamfer_flow_avg"] = cd_flow[2]

        cd_up = _chamfer_distance_np(up, gt)
        diag["chamfer_up_to_gt"] = cd_up[0]
        diag["chamfer_gt_to_up"] = cd_up[1]
        diag["chamfer_up_avg"] = cd_up[2]

        gt_span_norm = float(np.linalg.norm(gt.max(0) - gt.min(0)))
        diag["chamfer_flow_pct"] = (
            cd_flow[2] / gt_span_norm * 100 if gt_span_norm > 0 else 0
        )
        diag["chamfer_up_pct"] = (
            cd_up[2] / gt_span_norm * 100 if gt_span_norm > 0 else 0
        )

        scale_ratio = float(
            np.linalg.norm(flow.std(0)) /
            max(np.linalg.norm(gt.std(0)), 1e-9)
        )
        diag["scale_ratio_flow_vs_gt"] = scale_ratio

        diag["PASS_chamfer_under_10pct"] = diag["chamfer_flow_pct"] < 10.0
        diag["PASS_scale_ratio_0.8_1.2"] = 0.8 < scale_ratio < 1.2
        diag["PASS_mesh_watertight"] = diag["mesh_watertight"]
        diag["PASS_volume_nonzero"] = diag["volume_cm3"] > 1e-6

        np.save(os.path.join(output_dir, "gt_cloud_used.npy"), gt)

    path = os.path.join(output_dir, "diagnostics.json")
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)
    LOG.info("Saved diagnostics: %s", path)

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS SUMMARY")
    print("=" * 60)
    for k, v in diag.items():
        if k.startswith("PASS_"):
            status = "PASS" if v else "FAIL"
            print(f"  [{status}] {k[5:]}")
    if "chamfer_flow_pct" in diag:
        print(f"\n  Chamfer (flow->GT): {diag['chamfer_flow_pct']:.1f}% of GT span")
        print(f"  Chamfer (up->GT):   {diag['chamfer_up_pct']:.1f}% of GT span")
        print(f"  Scale ratio:       {diag['scale_ratio_flow_vs_gt']:.2f}x")
    print(f"  Volume:            {diag['volume_cm3']:.6f} cm3")
    print("=" * 60 + "\n")

    return diag


def main():
    parser = argparse.ArgumentParser(
        description="Predict stone volume using registered pipeline"
    )
    parser.add_argument("--depth_dir", required=True,
                        help="Directory with turntable .npy depth files")
    parser.add_argument("--mask_dir", default=None,
                        help="Directory with stone mask .png files")
    parser.add_argument("--gt_cloud_dir", default=None,
                        help="Directory with registration params and GT clouds "
                             "(from prepare_gt.py). Contains "
                             "{stone_id}_registration.npz files.")
    parser.add_argument("--intrinsics", required=True,
                        help="Path to intrinsics.txt")
    parser.add_argument("--sequence", required=True,
                        help="Stone sequence ID (e.g. stone_01)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--output_dir", default="volume_output",
                        help="Output directory")
    parser.add_argument("--gt_cloud", default=None,
                        help="Path to GT point cloud (.ply or .npy) for diagnostics")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--max_points_per_view", type=int, default=4096)
    parser.add_argument("--merged_cloud_points", type=int, default=8192,
                        help="FPS target for merged registered cloud")
    parser.add_argument("--angle_per_frame_deg", type=float, default=3.0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flow_steps", type=int, default=20,
                        help="Number of Euler ODE steps for flow generation")
    parser.add_argument("--poisson_depth", type=int, default=9,
                        help="Octree depth for Poisson surface reconstruction")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    registration: Dict[str, np.ndarray] = {}
    if args.gt_cloud_dir:
        reg_path = os.path.join(
            args.gt_cloud_dir, f"{args.sequence}_registration.npz"
        )
        if os.path.isfile(reg_path):
            registration = dict(np.load(reg_path))
            LOG.info("Loaded registration params from %s", reg_path)
        else:
            LOG.warning("No registration params at %s — "
                        "using identity registration", reg_path)
    else:
        LOG.warning("No --gt_cloud_dir specified — using identity registration")

    mask_dir = args.mask_dir
    if mask_dir is None:
        candidate = os.path.join(
            os.path.dirname(args.depth_dir),
            args.sequence, "masks",
        )
        if os.path.isdir(candidate):
            mask_dir = candidate
            LOG.info("Auto-detected mask_dir: %s", mask_dir)
        else:
            LOG.warning("No --mask_dir and could not auto-detect. "
                        "Using ALL depth points (no stone segmentation).")

    LOG.info("Loading model from %s", args.checkpoint)
    cfg = StoneReconNetConfig()
    model = StoneReconNet(cfg)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(args.device)
    model.eval()
    LOG.info("Model loaded (%s)", args.device)

    K = load_intrinsics(args.intrinsics, args.sequence, args.width, args.height)

    depth_files = _load_depth_files(args.depth_dir)
    LOG.info("Found %d depth files in %s", len(depth_files), args.depth_dir)

    LOG.info("Preparing input (registering + masking stone)...")
    batch, centroid, n_views = _prepare_input(
        depth_files, K, mask_dir, registration,
        angle_per_frame_deg=args.angle_per_frame_deg,
        max_points_per_view=args.max_points_per_view,
        merged_cloud_points=args.merged_cloud_points,
        device=args.device,
    )

    LOG.info("Running inference (flow -> mesh)...")
    t0 = time.perf_counter()
    results = predict(
        model, batch, centroid,
        flow_steps=args.flow_steps,
        poisson_depth=args.poisson_depth,
    )
    elapsed = time.perf_counter() - t0

    save_results(results, args.output_dir, args.depth_dir, n_views, elapsed)

    gt_cloud_path = args.gt_cloud
    if gt_cloud_path is None and args.gt_cloud_dir:
        for suffix in ("_gt_aligned.ply", "_gt_pointcloud.ply", "_gt.ply"):
            candidate = os.path.join(
                args.gt_cloud_dir, f"{args.sequence}{suffix}"
            )
            if os.path.isfile(candidate):
                gt_cloud_path = candidate
                break

    save_diagnostics(results, args.output_dir, gt_cloud_path=gt_cloud_path)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
