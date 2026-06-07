#!/usr/bin/env python3
"""Predict stone volume from sparse depth maps using a trained StoneVolumeNet.

Supports two registration modes:
  - Direct: segmentation + volume head (fast, single forward pass)
  - RPF Flow: Euler ODE integration for flow-based registration (from RPF)

Usage:
    python predict_stone_volume.py \
        --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
        --intrinsics splits/stone/intrinsics.txt \
        --sequence stone_01 \
        --checkpoint models/stone_volume_net.pt \
        --output_dir volume_output/ \
        --use_flow --flow_steps 10

Output:
    - stone_pointcloud.ply         (segmented registered point cloud)
    - flow_registered.ply          (RPF flow-registered point cloud, if --use_flow)
    - volume_report.txt            (predicted volume and statistics)
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
from typing import Dict, List

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
from volume_estimation.model import StoneVolumeNet, StoneVolumeNetConfig  # noqa: E402

LOG = logging.getLogger("predict_stone_volume")


def _load_depth_files(depth_dir: str) -> List[str]:
    """Find and sort .npy depth files."""
    files = sorted(
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.lower().endswith(".npy")
    )
    if not files:
        raise FileNotFoundError(f"No .npy files in {depth_dir}")
    return files


def _prepare_input(
    depth_files: List[str],
    intrinsics: Intrinsics,
    max_points_per_view: int = 4096,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Load depth files, back-project, and prepare model input."""
    all_pts = []
    all_view_ids = []

    for view_i, path in enumerate(depth_files):
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

        view_id = np.full(pts.shape[0], view_i, dtype=np.int64)
        all_pts.append(pts)
        all_view_ids.append(view_id)

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

    return batch, centroid


@torch.no_grad()
def predict(
    model: StoneVolumeNet,
    batch: Dict[str, torch.Tensor],
    centroid: np.ndarray,
    use_flow: bool = False,
    flow_steps: int = 10,
) -> Dict:
    """Run inference with optional RPF flow-based Euler ODE registration.

    When use_flow=True, runs Euler ODE integration from t=1 (noise) to t=0
    to produce a flow-registered point cloud alongside the direct prediction.
    """
    model.eval()

    output = model.forward_inference(batch)

    pred_volume = float(output["pred_volume"].item())

    points = batch["points"][0].cpu().numpy()
    seg_probs = torch.sigmoid(output["seg_logits"][0]).cpu().numpy()
    stone_mask = seg_probs > 0.5

    stone_pts = points[stone_mask] + centroid
    all_pts = points + centroid

    n_stone = int(stone_mask.sum())
    n_total = points.shape[0]
    seg_ratio = n_stone / max(n_total, 1)

    results = {
        "pred_volume_cm3": pred_volume,
        "pred_volume_mm3": pred_volume * 1000.0,
        "stone_points": stone_pts,
        "all_points": all_pts,
        "seg_probs": seg_probs,
        "n_stone_points": n_stone,
        "n_total_points": n_total,
        "seg_ratio": seg_ratio,
    }

    if use_flow:
        registered = model.sample_rectified_flow(batch, num_steps=flow_steps)
        flow_pts = registered[0].cpu().numpy() + centroid
        results["flow_registered_points"] = flow_pts
        results["flow_steps"] = flow_steps

    return results


def save_results(
    results: Dict,
    output_dir: str,
    depth_dir: str,
    n_views: int,
    elapsed_s: float,
):
    """Save point cloud(s) and text report."""
    os.makedirs(output_dir, exist_ok=True)

    stone_pts = results["stone_points"]
    if stone_pts.shape[0] > 0:
        pcd = make_pcd(stone_pts, estimate_normals=True)
        ply_path = os.path.join(output_dir, "stone_pointcloud.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        LOG.info("Saved point cloud: %s (%d points)", ply_path, stone_pts.shape[0])

    all_pcd = make_pcd(results["all_points"], estimate_normals=False)
    all_ply = os.path.join(output_dir, "full_pointcloud.ply")
    o3d.io.write_point_cloud(all_ply, all_pcd)

    if "flow_registered_points" in results:
        flow_pts = results["flow_registered_points"]
        if flow_pts.shape[0] > 0:
            flow_pcd = make_pcd(flow_pts, estimate_normals=True)
            flow_ply = os.path.join(output_dir, "flow_registered.ply")
            o3d.io.write_point_cloud(flow_ply, flow_pcd)
            LOG.info("Saved flow-registered cloud: %s (%d pts)", flow_ply, flow_pts.shape[0])

    report_lines = [
        "=" * 60,
        "StoneVolumeNet — Prediction Report",
        "=" * 60,
        "",
        f"Input:            {depth_dir}",
        f"Views:            {n_views}",
        f"Total points:     {results['n_total_points']}",
        f"Stone points:     {results['n_stone_points']} ({results['seg_ratio']:.1%})",
        "",
        f"Predicted Volume: {results['pred_volume_cm3']:.6f} cm³",
        f"                  {results['pred_volume_mm3']:.2f} mm³",
        "",
        f"Inference time:   {elapsed_s:.3f} s",
    ]

    if "flow_steps" in results:
        report_lines.append(f"Flow ODE steps:   {results['flow_steps']}")
        report_lines.append(f"Flow points:      {results['flow_registered_points'].shape[0]}")

    report_lines += [
        "",
        "Output files:",
        f"  Point cloud:    stone_pointcloud.ply",
        f"  Full cloud:     full_pointcloud.ply",
    ]
    if "flow_registered_points" in results:
        report_lines.append(f"  Flow cloud:     flow_registered.ply")
    report_lines += [
        f"  Report:         volume_report.txt",
        "=" * 60,
    ]

    report_path = os.path.join(output_dir, "volume_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    LOG.info("Saved report: %s", report_path)

    for line in report_lines:
        print(line)

    result_json = {
        "pred_volume_cm3": results["pred_volume_cm3"],
        "pred_volume_mm3": results["pred_volume_mm3"],
        "n_stone_points": results["n_stone_points"],
        "n_total_points": results["n_total_points"],
        "seg_ratio": results["seg_ratio"],
        "n_views": n_views,
        "inference_time_s": elapsed_s,
        "input_dir": depth_dir,
    }
    if "flow_steps" in results:
        result_json["flow_steps"] = results["flow_steps"]
        result_json["flow_registered_n_points"] = int(results["flow_registered_points"].shape[0])
    with open(os.path.join(output_dir, "prediction_result.json"), "w") as f:
        json.dump(result_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Predict stone volume from sparse depth maps"
    )
    parser.add_argument("--depth_dir", required=True,
                        help="Directory with sparse .npy depth files")
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
    parser.add_argument("--use_flow", action="store_true",
                        help="Use RPF Euler ODE integration for flow-based registration")
    parser.add_argument("--flow_steps", type=int, default=10,
                        help="Number of Euler ODE steps for flow registration")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    LOG.info("Loading model from %s", args.checkpoint)
    cfg = StoneVolumeNetConfig()
    model = StoneVolumeNet(cfg)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(args.device)
    model.eval()
    LOG.info("Model loaded (%s)", args.device)

    K = load_intrinsics(args.intrinsics, args.sequence, args.width, args.height)

    depth_files = _load_depth_files(args.depth_dir)
    LOG.info("Found %d depth files in %s", len(depth_files), args.depth_dir)

    LOG.info("Preparing input...")
    batch, centroid = _prepare_input(
        depth_files, K,
        max_points_per_view=args.max_points_per_view,
        device=args.device,
    )

    mode = "RPF flow registration" if args.use_flow else "direct"
    LOG.info("Running inference (%s)...", mode)
    t0 = time.perf_counter()
    results = predict(
        model, batch, centroid,
        use_flow=args.use_flow,
        flow_steps=args.flow_steps,
    )
    elapsed = time.perf_counter() - t0

    save_results(results, args.output_dir, args.depth_dir, len(depth_files), elapsed)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
