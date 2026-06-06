"""Extended report writer for the neural-only pipeline.

Includes:
  - per-stage latency and model details
  - per-frame stone-point counts
  - RAP chamfer diagnostic (when RAP ran)
  - chamfer distance + F-score against the dense 120-frame reference mesh
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .config import NeuralConfig, StageStatus
from .geometry import Intrinsics, FloorFit


LOG = logging.getLogger("stone3d_neural.report")


def compute_chamfer_fscore(
    mesh_pred: o3d.geometry.TriangleMesh,
    mesh_ref: o3d.geometry.TriangleMesh,
    n_samples: int = 30000,
    f_score_threshold_m: float = 1.0e-3,
    align: bool = True,
) -> Dict[str, float]:
    """Bidirectional chamfer distance + F-score on point samples.

    The sparse reconstruction lives in a "floor-up frame 0" coordinate
    system; the dense reference lives in a turntable-axis frame. The two
    can differ by an arbitrary horizontal yaw and translation even when
    they encode the exact same stone shape. We therefore first centre
    each cloud at its bottom-AABB anchor, do a coarse yaw sweep, then
    refine with a tight point-to-point ICP before measuring chamfer.
    This makes the metric a *shape* comparison (which is what the
    dissertation cares about), not a pose comparison.

    The F-score reports the fraction of points within
    ``f_score_threshold_m`` of the other surface, harmonic-mean style:
    F = 2PR/(P+R).
    """
    if len(mesh_pred.triangles) == 0 or len(mesh_ref.triangles) == 0:
        return {"chamfer_mm": float("nan"), "f_score_1mm": float("nan"),
                "precision_1mm": float("nan"), "recall_1mm": float("nan"),
                "alignment_rmse_mm": float("nan")}

    pcd_pred = mesh_pred.sample_points_uniformly(number_of_points=n_samples)
    pcd_ref = mesh_ref.sample_points_uniformly(number_of_points=n_samples)
    pts_pred = np.asarray(pcd_pred.points)
    pts_ref = np.asarray(pcd_ref.points)

    align_rmse_mm = float("nan")
    if align:
        T_align, align_rmse_mm = _shape_align(pcd_pred, pcd_ref)
        pcd_pred.transform(T_align)
        pts_pred = np.asarray(pcd_pred.points)

    tree_ref = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_ref)))
    tree_pred = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_pred)))

    def nn_distances(query_pts, tree):
        d = np.zeros(query_pts.shape[0])
        for i, p in enumerate(query_pts):
            _, _idx, dist_sq = tree.search_knn_vector_3d(p, 1)
            d[i] = float(np.sqrt(dist_sq[0]))
        return d

    d_pred_to_ref = nn_distances(pts_pred, tree_ref)
    d_ref_to_pred = nn_distances(pts_ref, tree_pred)

    chamfer_mm = 0.5 * (d_pred_to_ref.mean() + d_ref_to_pred.mean()) * 1000.0
    precision = float((d_pred_to_ref < f_score_threshold_m).mean())
    recall = float((d_ref_to_pred < f_score_threshold_m).mean())
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0

    return {
        "chamfer_mm": float(chamfer_mm),
        "f_score_1mm": float(f_score),
        "precision_1mm": float(precision),
        "recall_1mm": float(recall),
        "alignment_rmse_mm": align_rmse_mm,
    }


def _shape_align(
    pcd_pred: o3d.geometry.PointCloud,
    pcd_ref: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, float]:
    """Centre + yaw-sweep + ICP. Returns (T_align, final_rmse_mm)."""
    pred_c = pcd_pred.get_center()
    ref_c = pcd_ref.get_center()
    T_center = np.eye(4)
    T_center[:3, 3] = ref_c - pred_c

    pred_d = pcd_pred.voxel_down_sample(1.0e-3)
    ref_d = pcd_ref.voxel_down_sample(1.0e-3)

    best = (float("inf"), np.eye(4))
    for ang_deg in np.arange(0.0, 360.0, 5.0):
        c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        Tyaw = np.eye(4); Tyaw[:3, :3] = R
        Tyaw[:3, 3] = ref_c - R @ ref_c
        T_try = Tyaw @ T_center
        ev = o3d.pipelines.registration.evaluate_registration(
            pred_d, ref_d, max_correspondence_distance=3.0e-3, transformation=T_try,
        )
        if ev.inlier_rmse > 0 and ev.fitness > 0.05 and ev.inlier_rmse < best[0]:
            best = (float(ev.inlier_rmse), T_try)
    T_init = best[1]

    res = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_ref, max_correspondence_distance=5.0e-3,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    final_rmse_mm = float(res.inlier_rmse) * 1000.0
    return np.asarray(res.transformation), final_rmse_mm


def write_neural_report(
    out_path: str,
    cfg: NeuralConfig,
    intrinsics: Intrinsics,
    floors: List[FloorFit],
    poses_world_from_cam: List[np.ndarray],
    voxel_mm: float,
    sdf_trunc_mm: float,
    edge_diagnostics: List[Tuple[int, int, float, float, float]],
    mesh_pre_closure: o3d.geometry.TriangleMesh,
    mesh_water: o3d.geometry.TriangleMesh,
    pointcloud: o3d.geometry.PointCloud,
    elapsed_s: float,
    n_frames: int,
    pair_results: dict,
    pg_summary: dict,
    kept_component_sizes: List[int],
    min_edge_fitness: float,
    stage_statuses: List[StageStatus],
    seg_mask_pixel_counts: List[int],
    chamfer_metrics: Optional[Dict[str, float]] = None,
) -> None:
    bbox = mesh_water.get_axis_aligned_bounding_box() if len(mesh_water.triangles) else None
    extent = np.asarray(bbox.get_extent()) if bbox is not None else np.zeros(3)

    is_em = mesh_water.is_edge_manifold() if len(mesh_water.triangles) else False
    is_vm = mesh_water.is_vertex_manifold() if len(mesh_water.triangles) else False
    is_wt = mesh_water.is_watertight() if len(mesh_water.triangles) else False
    surface_area_mm2 = mesh_water.get_surface_area() * 1e6 if len(mesh_water.triangles) else 0.0
    try:
        volume_mm3 = mesh_water.get_volume() * 1e9 if is_wt else float("nan")
    except RuntimeError:
        volume_mm3 = float("nan")

    tilts = []
    cam_axis = np.array([0.0, 0.0, 1.0])
    for fl in floors:
        n_unit = fl.normal / np.linalg.norm(fl.normal)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(n_unit @ cam_axis)))))
        tilts.append(90.0 - ang)
    tilts = np.array(tilts)

    with open(out_path, "w") as f:
        f.write("Neural Sparse Stone Reconstruction Report\n")
        f.write("=========================================\n\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed_s:.1f}s\n")
        f.write(f"Frames input: {n_frames}\n")
        f.write(f"Frames with non-empty stone mask: "
                f"{sum(1 for c in seg_mask_pixel_counts if c > 0)}/{n_frames}\n\n")

        # -- Stage timing ---------------------------------------------------
        f.write("Pipeline stages (neural-only)\n")
        f.write("-----------------------------\n")
        f.write(f"  {'stage':<25} {'backend':<12} {'latency':>10}\n")
        for st in stage_statuses:
            f.write(
                f"  {st.stage:<25} {st.backend_used:<12} {st.latency_s:>8.2f}s\n"
            )
            if st.stage == "surface" and "model" in st.extra:
                f.write(f"      model: {st.extra['model']}\n")
            if st.stage == "pairwise_registration":
                f.write(
                    f"      pare pairs: {st.extra.get('n_pare_pairs', '?')}  "
                    f"geotr pairs: {st.extra.get('n_geotr_pairs', '?')}\n"
                )
        f.write(f"  surface_model config: {cfg.surface_model}\n\n")

        # -- Intrinsics -----------------------------------------------------
        f.write("Camera intrinsics (sequence-specific)\n")
        f.write("-------------------------------------\n")
        f.write(f"  fx={intrinsics.fx:.4f}\n  fy={intrinsics.fy:.4f}\n")
        f.write(f"  cx={intrinsics.cx:.4f}\n  cy={intrinsics.cy:.4f}\n")
        f.write(f"  width={intrinsics.width}\n  height={intrinsics.height}\n\n")

        # -- Floor + tilt ---------------------------------------------------
        f.write("Per-frame floor fit + camera tilt\n")
        f.write("---------------------------------\n")
        f.write(f"  Median inlier ratio: {np.median([fl.inlier_ratio for fl in floors]):.2f}\n")
        f.write(f"  Camera tilt (deg from horizontal): min={tilts.min():.1f} "
                f"med={np.median(tilts):.1f} max={tilts.max():.1f}\n\n")

        # -- Per-frame stats ------------------------------------------------
        f.write("Per-frame stone-point counts (after auto-segmentation)\n")
        f.write("------------------------------------------------------\n")
        for i, c in enumerate(seg_mask_pixel_counts):
            f.write(f"  #{i+1:02d}: {c} px (floor_inl={floors[i].inlier_ratio:.2f}, "
                    f"tilt={tilts[i]:.1f} deg)\n")
        f.write("\n")

        # -- Pairwise / pose graph ----------------------------------------
        f.write("Pairwise registration\n")
        f.write("---------------------\n")
        fits = np.array([pr.fitness for pr in pair_results.values()])
        rmses = np.array([pr.rmse for pr in pair_results.values()])
        f.write(f"  total pairs: {len(pair_results)}\n")
        f.write(f"  fitness:  min={fits.min():.3f} med={np.median(fits):.3f} max={fits.max():.3f}\n")
        f.write(f"  rmse mm:  min={rmses.min()*1000:.3f} med={np.median(rmses)*1000:.3f} "
                f"max={rmses.max()*1000:.3f}\n")
        f.write(f"  ambiguous yaw pairs: {pg_summary.get('ambiguous', 0)}\n\n")

        f.write("Multi-view solver\n")
        f.write("-----------------\n")
        f.write(f"  method: {pg_summary.get('method', '?')}\n")
        f.write(f"  edges kept per iter: {pg_summary.get('edges_kept_per_iter', [])}\n")
        f.write(f"  isolated frames:     {pg_summary.get('iso_frames', [])}\n")
        if "irls_residual" in pg_summary:
            f.write(f"  irls residual:       {pg_summary['irls_residual']:.4g}\n")
        if "rap_chamfer_d" in pg_summary:
            f.write(f"  rap chamfer_d:       {pg_summary['rap_chamfer_d']:.4g}\n")
        f.write("\n")

        # -- Mesh stats -----------------------------------------------------
        f.write("Output mesh\n")
        f.write("-----------\n")
        f.write(f"  Vertices:        {np.asarray(mesh_water.vertices).shape[0]}\n")
        f.write(f"  Triangles:       {np.asarray(mesh_water.triangles).shape[0]}\n")
        f.write(f"  Edge manifold:   {is_em}\n")
        f.write(f"  Vertex manifold: {is_vm}\n")
        f.write(f"  Watertight:      {is_wt}\n")
        f.write(f"  Bounding box (mm): x={extent[0]*1000:.2f} "
                f"y={extent[1]*1000:.2f} z={extent[2]*1000:.2f}\n")
        f.write(f"  Surface area (mm^2): {surface_area_mm2:.2f}\n")
        if not math.isnan(volume_mm3):
            f.write(f"  Volume (mm^3):       {volume_mm3:.2f}\n")
        else:
            f.write("  Volume (mm^3):       (mesh not watertight; volume undefined)\n")
        f.write(f"  Kept component sizes: {kept_component_sizes[:8]}\n\n")

        if chamfer_metrics is not None:
            f.write("Comparison to dense 120-frame reference (shape-aligned)\n")
            f.write("-------------------------------------------------------\n")
            f.write(f"  Pred-vs-ref ICP RMSE (mm):  {chamfer_metrics.get('alignment_rmse_mm', float('nan')):.3f}\n")
            f.write(f"  Chamfer distance (mm):      {chamfer_metrics['chamfer_mm']:.3f}\n")
            f.write(f"  F-score @ 1mm:              {chamfer_metrics['f_score_1mm']:.3f}\n")
            f.write(f"    precision @ 1mm:          {chamfer_metrics['precision_1mm']:.3f}\n")
            f.write(f"    recall    @ 1mm:          {chamfer_metrics['recall_1mm']:.3f}\n\n")

        # -- Files ---------------------------------------------------------
        f.write("Outputs written\n")
        f.write("---------------\n")
        f.write("  stone_mesh_pre_closure.ply   - Pre-closure mesh (NKSR/NoKSR output)\n")
        f.write("  stone_mesh_watertight.ply    - Watertight mesh (after closure if needed)\n")
        f.write("  stone_mesh_watertight.obj    - Same, OBJ format\n")
        f.write("  stone_pointcloud.ply         - Merged point cloud (post pose-graph)\n")
        f.write("  stone_3d_views_composite.png - 6-viewpoint preview render\n")
        f.write("  auto_segmentation_preview.png- Per-frame depth + auto-mask overlay\n")
    LOG.info("Wrote neural report: %s", out_path)
