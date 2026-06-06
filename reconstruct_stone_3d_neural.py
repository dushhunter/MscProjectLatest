#!/usr/bin/env python3
"""Neural sparse depth-only stone 3D reconstruction (pure ML, CUDA required).

Depth-only upgrade of :mod:`reconstruct_stone_3d_sparse`. Every learnable
stage runs a 2024-2026 deep-learning module on a CUDA GPU. There is no
classical fallback; use :mod:`reconstruct_stone_3d_sparse` for the
handcrafted baseline.

Pipeline:
  1. Load all `.npy` depth frames + camera intrinsics.
  2. Per-frame stone segmentation (PointTransformerV3 binary head).
  3. Floor plane fit on PTv3 floor points + floor-up Rodrigues transform.
  4. Per-frame stone point-cloud extraction in the floor-up frame.
  5. Pairwise registration (PARE-Net + ICP; GeoTransformer if a pair fails).
  6. Multi-view alignment (RAP flow-matching + Procrustes).
  7. Surface reconstruction (NKSR or NoKSR).
  8. Watertight closure if the neural mesh has a floor-side hole.
  9. Optional chamfer / F-score against a dense reference mesh.

References (2024-2026):
  - PointTransformerV3 (Wu et al., CVPR 2024).
  - PARE-Net (Yao et al., ECCV 2024).
  - GeoTransformer (Qin et al., PAMI 2023).
  - RAP (Pan et al., NeurIPS 2025).
  - NKSR (Huang et al., CVPR 2023 Highlight).
  - NoKSR (Yi et al., arXiv 2025).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List

import numpy as np

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

from neural_pipeline.geometry import (  # noqa: E402
    Intrinsics, load_intrinsics, make_pcd, merge_pointclouds, render_preview,
    floor_up_transform, load_depth_only_frames, write_segmentation_preview,
)
from neural_pipeline import (  # noqa: E402
    NeuralConfig,
    StoneSegmenter, PairwiseRegistrar, MultiViewRegistrar,
    SurfaceReconstructor, write_neural_report,
)
from neural_pipeline.config import require_cuda  # noqa: E402
from neural_pipeline.report import compute_chamfer_fscore  # noqa: E402


LOG = logging.getLogger("stone3d_neural")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--depth_dir", required=True,
                   help="Folder with .npy depth files (one per view).")
    p.add_argument("--intrinsics", default="splits/stone/intrinsics.txt")
    p.add_argument("--sequence", default="stone_01")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)

    p.add_argument("--surface_model", choices=("nksr", "noksr"), default="nksr",
                   help="Neural surface model (NKSR or NoKSR ablation).")
    p.add_argument("--models_dir", default="models",
                   help="Directory containing pretrained weights.")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda).")

    p.add_argument("--voxel_mm", type=float, default=0.5)
    p.add_argument("--sdf_trunc_mm", type=float, default=3.0)
    p.add_argument("--icp_voxel_mm", type=float, default=0.6)
    p.add_argument("--icp_max_corr_mm", type=float, default=4.0)
    p.add_argument("--min_edge_fitness", type=float, default=0.40)
    p.add_argument("--refinement_iters", type=int, default=2)
    p.add_argument("--component_keep_fraction", type=float, default=0.05)

    p.add_argument("--reference_mesh", default=None,
                   help="Optional dense reference mesh (.ply) for chamfer / F-score.")

    p.add_argument("--output_dir", default="reconstruction_output_neural")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    require_cuda("reconstruct_stone_3d_neural")

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = NeuralConfig(
        surface_model=args.surface_model,
        models_dir=args.models_dir,
        device=args.device,
    )
    LOG.info("Configuration: %s", cfg)

    K = load_intrinsics(args.intrinsics, args.sequence, args.width, args.height)

    frames = load_depth_only_frames(args.depth_dir, expected_size=(args.height, args.width))
    LOG.info("Loaded %d depth-only frames", len(frames))

    segmenter = StoneSegmenter(cfg)
    seg_result = segmenter.segment(frames, K)
    for frame, mask in zip(frames, seg_result.masks):
        frame.mask = mask
    floors = seg_result.floors

    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames with non-empty stone masks.")
    seg_mask_counts = [int(f.mask.sum()) for f in frames]

    icp_voxel = args.icp_voxel_mm * 1e-3
    pcds_cam = []
    for f in frames:
        ys, xs = np.where(f.mask)
        zs = f.depth[ys, xs].astype(np.float64)
        valid = np.isfinite(zs) & (zs > 0)
        ys, xs, zs = ys[valid], xs[valid], zs[valid]
        if zs.size == 0:
            raise RuntimeError("Stone mask produced no valid depth points")
        X = (xs - K.cx) * zs / K.fx
        Y = (ys - K.cy) * zs / K.fy
        pts = np.stack([X, Y, zs], axis=1)
        pcd = make_pcd(pts).voxel_down_sample(icp_voxel)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=4 * icp_voxel, max_nn=30,
            )
        )
        pcds_cam.append(pcd)

    floor_up_T = [floor_up_transform(fl.normal, fl.d) for fl in floors]
    pcds_floor_up: List[o3d.geometry.PointCloud] = []
    for pcd, T in zip(pcds_cam, floor_up_T):
        cp = o3d.geometry.PointCloud(pcd)
        cp.transform(T)
        cp.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=4 * icp_voxel, max_nn=30,
            )
        )
        pcds_floor_up.append(cp)

    pair_registrar = PairwiseRegistrar(cfg)
    pair_results, pair_status = pair_registrar.register_all_pairs(
        pcds_floor_up,
        voxel_m=icp_voxel,
        min_edge_fitness=args.min_edge_fitness,
    )

    multi = MultiViewRegistrar(cfg, pair_registrar)
    refined_world, pg_summary, multi_status = multi.solve(
        pcds_floor_up, pair_results,
        voxel_m=icp_voxel,
        max_corr_m=args.icp_max_corr_mm * 1e-3,
        min_edge_fitness=args.min_edge_fitness,
        refinement_iters=args.refinement_iters,
    )

    poses_world_from_cam = [Tw @ Tf for Tw, Tf in zip(refined_world, floor_up_T)]
    merged = merge_pointclouds(pcds_cam, poses_world_from_cam, voxel=icp_voxel)

    surface = SurfaceReconstructor(cfg)
    surf_result = surface.reconstruct(
        frames, poses_world_from_cam, merged, K,
        voxel_mm=args.voxel_mm,
        sdf_trunc_mm=args.sdf_trunc_mm,
        depth_trunc_m=1.0,
        component_keep_fraction=args.component_keep_fraction,
    )
    mesh_water = surf_result.mesh
    mesh_pre = surf_result.mesh_pre_closure

    chamfer = None
    if args.reference_mesh and os.path.exists(args.reference_mesh):
        try:
            ref = o3d.io.read_triangle_mesh(args.reference_mesh)
            if len(ref.triangles):
                chamfer = compute_chamfer_fscore(mesh_water, ref)
                LOG.info(
                    "Chamfer vs %s: %.3f mm (F@1mm=%.3f)",
                    args.reference_mesh, chamfer["chamfer_mm"], chamfer["f_score_1mm"],
                )
        except Exception as e:
            LOG.warning("Chamfer evaluation failed: %s", e)

    out = args.output_dir
    pre_path = os.path.join(out, "stone_mesh_pre_closure.ply")
    water_ply = os.path.join(out, "stone_mesh_watertight.ply")
    water_obj = os.path.join(out, "stone_mesh_watertight.obj")
    pcd_path = os.path.join(out, "stone_pointcloud.ply")
    preview_path = os.path.join(out, "stone_3d_views_composite.png")
    seg_path = os.path.join(out, "auto_segmentation_preview.png")
    report_path = os.path.join(out, "reconstruction_report.txt")

    o3d.io.write_triangle_mesh(pre_path, mesh_pre, write_ascii=False)
    o3d.io.write_triangle_mesh(water_ply, mesh_water, write_ascii=False)
    o3d.io.write_triangle_mesh(water_obj, mesh_water, write_ascii=True)
    o3d.io.write_point_cloud(pcd_path, merged, write_ascii=False)
    LOG.info("Wrote: %s", pre_path)
    LOG.info("Wrote: %s", water_ply)
    LOG.info("Wrote: %s", water_obj)
    LOG.info("Wrote: %s", pcd_path)

    write_segmentation_preview(frames, floors, seg_path)
    try:
        render_preview(mesh_water, preview_path,
                       up_axis_world=np.array([0.0, 1.0, 0.0]))
    except Exception as e:
        LOG.warning("Preview render failed: %s", e)

    edge_diag = [
        (s, t, pr.fitness, pr.rmse, pr.yaw_margin)
        for (s, t), pr in pair_results.items()
    ]
    write_neural_report(
        report_path, cfg, K, floors, poses_world_from_cam,
        voxel_mm=args.voxel_mm, sdf_trunc_mm=args.sdf_trunc_mm,
        edge_diagnostics=edge_diag,
        mesh_pre_closure=mesh_pre, mesh_water=mesh_water,
        pointcloud=merged,
        elapsed_s=time.time() - t0,
        n_frames=len(frames),
        pair_results=pair_results,
        pg_summary=pg_summary,
        kept_component_sizes=surf_result.kept_component_sizes,
        min_edge_fitness=args.min_edge_fitness,
        stage_statuses=[seg_result.status, pair_status, multi_status, surf_result.status],
        seg_mask_pixel_counts=seg_mask_counts,
        chamfer_metrics=chamfer,
    )
    LOG.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
