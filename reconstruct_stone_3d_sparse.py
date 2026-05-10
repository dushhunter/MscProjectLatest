#!/usr/bin/env python3
"""Sparse depth-only stone 3D reconstruction.

Reconstructs a watertight 3D mesh of a stone from a small set (typically
10-15) of metric depth `.npy` frames captured at *unknown* camera angles.
The only inputs are the depth maps; no RGB images, no foreground masks, and
no per-frame turntable angles are required.

Pipeline:
  1. Load all `.npy` depth files from `--depth_dir`.
  2. Per-frame: RANSAC the floor plane in camera coordinates, then auto-
     segment the stone as the largest above-plane connected cluster.
  3. Per-frame: build a "floor-up" rigid transform that rotates the floor
     normal onto +Y_world and translates the floor onto Y=0. This fixes
     three of six pose DoF, leaving only yaw (rotation about Y) and 2-D
     translation in the XZ plane unknown per frame.
  4. Pairwise registration: 1-D coarse yaw sweep (cheap
     `evaluate_registration`) followed by point-to-plane ICP refinement.
  5. All-pairs pose-graph optimisation with a line process to drop bad
     edges.
  6. TSDF volumetric depth fusion (reused from reconstruct_stone_3d.py)
     using the auto-derived per-frame mask to ignore floor pixels.
  7. Polygon-respecting flat cap on the floor plane to make the mesh
     watertight (also reused).

References:
  - Curless and Levoy 1996 (TSDF fusion).
  - Choi, Zhou, Koltun 2015 (multiway pose-graph optimisation, line process).
  - The dense-view companion script: reconstruct_stone_3d.py.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")

import open3d as o3d  # noqa: E402

# Reuse pose-agnostic stages from the dense-view script.
from reconstruct_stone_3d import (  # noqa: E402
    Frame,
    Intrinsics,
    axis_angle_to_matrix,
    integrate_tsdf,
    load_intrinsics,
    make_pcd,
    make_watertight_mesh,
    merge_pointclouds,
    render_preview,
)

LOG = logging.getLogger("stone3d_sparse")


# ---------------------------------------------------------------------------
# Loaders (depth only)
# ---------------------------------------------------------------------------
def list_depth_files(depth_dir: str) -> List[str]:
    """Return sorted list of .npy depth files in `depth_dir` (non-recursive)."""
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"depth_dir does not exist: {depth_dir}")
    files = [
        os.path.join(depth_dir, f)
        for f in sorted(os.listdir(depth_dir))
        if f.lower().endswith(".npy")
    ]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {depth_dir}")
    return files


def load_depth_only_frame(
    path: str,
    index: int,
    expected_size: Tuple[int, int],
    color_value: int = 180,
) -> Frame:
    """Load one depth-only frame as the same `Frame` dataclass used elsewhere.

    Open3D's TSDF integrate requires an RGBDImage with both colour and depth,
    so we synthesize a uniform grey colour image. The mask is initialised to
    all-False here and replaced by the auto-segmentation step.
    """
    H_exp, W_exp = expected_size
    depth = np.load(path).astype(np.float32)
    if depth.shape != (H_exp, W_exp):
        raise ValueError(
            f"Depth {path} shape {depth.shape} != expected {(H_exp, W_exp)}"
        )
    color = np.full((H_exp, W_exp, 3), color_value, dtype=np.uint8)
    mask = np.zeros((H_exp, W_exp), dtype=bool)
    return Frame(index=index, depth=depth, mask=mask, color=color)


def load_depth_only_frames(
    depth_dir: str, expected_size: Tuple[int, int]
) -> List[Frame]:
    files = list_depth_files(depth_dir)
    LOG.info("Found %d .npy depth files in %s", len(files), depth_dir)
    frames: List[Frame] = []
    for i, path in enumerate(files, start=1):
        frame = load_depth_only_frame(path, index=i, expected_size=expected_size)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Per-frame floor segmentation + auto stone mask
# ---------------------------------------------------------------------------
def _backproject_full(
    depth: np.ndarray, K: Intrinsics, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project (optionally subsampled) finite depth pixels to 3D.

    Returns ``(points_3d, pixel_indices)`` where ``pixel_indices`` is the
    flat (y * W + x) index for each returned point so we can rasterize a
    per-pixel mask later.
    """
    H, W = depth.shape
    if stride > 1:
        ys, xs = np.meshgrid(
            np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij"
        )
        ys = ys.ravel(); xs = xs.ravel()
    else:
        yy, xx = np.indices((H, W))
        ys = yy.ravel(); xs = xx.ravel()
    zs = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(zs) & (zs > 0)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]
    X = (xs - K.cx) * zs / K.fx
    Y = (ys - K.cy) * zs / K.fy
    pts = np.stack([X, Y, zs], axis=1)
    flat_idx = ys * W + xs
    return pts, flat_idx


@dataclass
class FloorFit:
    normal: np.ndarray   # unit, points toward camera (negative z)
    d: float             # plane offset: n . X + d = 0
    inlier_ratio: float  # |inliers| / |full cloud|


def _fit_floor_plane(
    pts: np.ndarray,
    distance_threshold: float = 3e-4,
    num_iterations: int = 3000,
) -> Tuple[FloorFit, np.ndarray]:
    """RANSAC plane fit. Returns plane params plus boolean inlier mask of `pts`."""
    pcd = make_pcd(pts)
    plane, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    a, b, c, d_off = plane
    n = np.array([a, b, c], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm == 0:
        raise RuntimeError("Degenerate floor plane (zero normal).")
    n /= norm
    d_off = float(d_off) / norm
    if n[2] > 0:  # ensure normal points toward camera (+z is into scene)
        n = -n
        d_off = -d_off
    inlier_mask = np.zeros(len(pts), dtype=bool)
    inlier_mask[np.asarray(inliers, dtype=np.int64)] = True
    return FloorFit(n, d_off, float(inlier_mask.mean())), inlier_mask


def auto_segment_stone(
    depth: np.ndarray,
    K: Intrinsics,
    stone_height_thresh_m: float = 1.0e-3,
    cluster_eps_m: float = 2.0e-3,
    cluster_min_points: int = 30,
    floor_inlier_min_ratio: float = 0.4,
    pixel_stride: int = 2,
    max_stone_fraction: float = 0.4,
) -> Tuple[np.ndarray, FloorFit]:
    """Find the floor plane and the stone region from a depth image.

    Returns ``(stone_pixel_mask, floor_fit)`` where ``stone_pixel_mask`` is
    a boolean array of shape ``depth.shape``.
    """
    H, W = depth.shape

    # Subsample for fast plane fitting.
    pts_sub, _ = _backproject_full(depth, K, stride=pixel_stride)

    # Try a tight threshold first; relax if the plane has too few inliers.
    floor: Optional[FloorFit] = None
    for thresh in (3e-4, 6e-4, 1e-3):
        fit, _ = _fit_floor_plane(pts_sub, distance_threshold=thresh)
        if fit.inlier_ratio >= floor_inlier_min_ratio:
            floor = fit
            break
    if floor is None:
        # Best-effort: keep the loosest fit and warn.
        LOG.warning(
            "Floor RANSAC inlier ratio low (%.2f); accepting anyway.",
            fit.inlier_ratio,
        )
        floor = fit
    LOG.info(
        "Floor plane: n=[%.4f, %.4f, %.4f] d=%.5f (inlier_ratio=%.2f)",
        floor.normal[0], floor.normal[1], floor.normal[2], floor.d, floor.inlier_ratio,
    )

    # Now back-project every pixel and pick those above the floor.
    pts_full, idx_full = _backproject_full(depth, K, stride=1)
    above = pts_full @ floor.normal + floor.d
    cand = above >= stone_height_thresh_m
    if cand.sum() < cluster_min_points:
        LOG.warning("Auto-segment found no stone candidates above %.2f mm",
                    stone_height_thresh_m * 1000.0)
        return np.zeros((H, W), dtype=bool), floor

    cand_pts = pts_full[cand]
    cand_idx = idx_full[cand]
    pcd_cand = make_pcd(cand_pts)
    labels = np.asarray(
        pcd_cand.cluster_dbscan(eps=cluster_eps_m, min_points=cluster_min_points,
                                print_progress=False)
    )

    if labels.size == 0 or labels.max() < 0:
        LOG.warning("DBSCAN found no clusters in stone candidates")
        return np.zeros((H, W), dtype=bool), floor

    # Pick the largest non-noise cluster.
    n_clusters = int(labels.max() + 1)
    sizes = [int((labels == k).sum()) for k in range(n_clusters)]
    largest = int(np.argmax(sizes))
    sel = labels == largest
    stone_idx = cand_idx[sel]

    # Sanity: stone shouldn't fill the image.
    stone_fraction = stone_idx.size / float(H * W)
    if stone_fraction > max_stone_fraction:
        LOG.warning(
            "Auto-segment cluster covers %.1f%% of pixels (>%.0f%%); "
            "treating as failure.",
            100 * stone_fraction, 100 * max_stone_fraction,
        )
        return np.zeros((H, W), dtype=bool), floor

    mask = np.zeros(H * W, dtype=bool)
    mask[stone_idx] = True
    mask = mask.reshape(H, W)
    LOG.info(
        "Auto-segment: %d stone pixels (%.2f%% of image), cluster=%d/%d",
        int(mask.sum()), 100 * stone_fraction, largest, n_clusters,
    )
    return mask, floor


# ---------------------------------------------------------------------------
# Floor-up alignment
# ---------------------------------------------------------------------------
def floor_up_transform(n_floor_cam: np.ndarray, d_floor_cam: float) -> np.ndarray:
    """Build the camera-frame -> "floor-up" world transform.

    After applying this transform:
      - The floor plane lies on Y=0.
      - The floor normal points to +Y (so "up" is +Y in this frame).
      - The X and Z axes lie in the floor plane (yaw is the only free
        rotational DoF).
    """
    n = n_floor_cam / np.linalg.norm(n_floor_cam)
    target = np.array([0.0, 1.0, 0.0])  # +Y_world
    # `n` in camera frame currently points toward camera (-z-ish). The world
    # convention here is that +Y points away from the floor (i.e. the side
    # the stone sits on). The camera-frame normal also points away from the
    # floor (we forced n_z < 0), so we want R such that R @ n = +Y.
    axis = np.cross(n, target)
    s = np.linalg.norm(axis)
    c = float(n @ target)
    if s < 1e-9:
        # n is already +/-Y in camera frame: identity or 180-deg flip.
        if c > 0:
            R = np.eye(3)
        else:
            R = np.diag([1.0, -1.0, -1.0])
    else:
        axis /= s
        angle = math.atan2(s, c)
        R = axis_angle_to_matrix(axis, angle)

    # The floor satisfies n . X_cam + d = 0 in camera frame. After R, the
    # floor satisfies +Y . (R @ X_cam) + d = 0  ->  Y_world = -d. We then
    # translate by +d along +Y_world so the floor sits at Y=0.
    T = np.eye(4)
    T[:3, :3] = R
    T[1, 3] = d_floor_cam  # because Y_world = R*X . [0,1,0] = R[1,:] @ X
    # After R, plane equation in world is +Y . X_world + d = 0. Translating
    # by [0, +d, 0] converts that to +Y . X_world = 0  i.e. Y=0.
    return T


# ---------------------------------------------------------------------------
# Pairwise yaw + ICP registration in floor-up frame
# ---------------------------------------------------------------------------
def _rotation_about_y(theta: float, centre: np.ndarray) -> np.ndarray:
    """4x4 rotation by `theta` (radians) around the vertical axis through `centre`."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre - R @ centre
    return T


def _xz_translate(delta_xz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = float(delta_xz[0])
    T[2, 3] = float(delta_xz[1])
    return T


def _bottom_aabb_center(pcd: o3d.geometry.PointCloud, slab_mm: float = 5.0) -> np.ndarray:
    """Center of the AABB of points within `slab_mm` of the lowest Y in `pcd`.

    The bottom slab traces a near-complete silhouette of the stone footprint
    regardless of which side the camera sees, so its bbox center is far less
    biased than ``pcd.get_center()`` for partial-stone views.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3)
    y = pts[:, 1]
    bottom = pts[(y - y.min()) <= slab_mm * 1e-3]
    if bottom.shape[0] < 20:
        bottom = pts
    mn = bottom.min(axis=0); mx = bottom.max(axis=0)
    return 0.5 * (mn + mx)


def _yaw_from_R(R: np.ndarray) -> float:
    """Return the rotation angle (radians) about +Y for a rotation matrix.

    Decomposes R as a yaw around +Y; small pitch/roll components are ignored.
    """
    return math.atan2(float(R[0, 2]), float(R[0, 0]))


def _yaw_only(T: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Snap a 4x4 rigid transform to a yaw-only rotation around the +Y axis
    through ``pivot``, plus an XZ translation. Pitch/roll/Y translation are
    dropped (they are physically zero in the floor-up frame).
    """
    yaw = _yaw_from_R(T[:3, :3])
    R_yaw = _rotation_about_y(yaw, pivot)
    # Recompute translation so the original transform's image of pivot is
    # preserved in XZ.
    p_image = (T @ np.array([pivot[0], pivot[1], pivot[2], 1.0]))[:3]
    R_yaw_p = (R_yaw @ np.array([pivot[0], pivot[1], pivot[2], 1.0]))[:3]
    delta = p_image - R_yaw_p
    R_yaw[0, 3] += delta[0]
    R_yaw[2, 3] += delta[2]
    return R_yaw


def _ensure_normals(pcd: o3d.geometry.PointCloud, voxel: float) -> None:
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30)
        )


def _multi_stage_icp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    voxel: float,
    T_init: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse-to-fine point-to-plane ICP."""
    T = T_init
    last = None
    for corr in (5.0 * voxel, 2.5 * voxel, 1.2 * voxel):
        last = o3d.pipelines.registration.registration_icp(
            src, tgt, corr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
        )
        T = last.transformation
    return last


@dataclass
class PairResult:
    T: np.ndarray              # T_target<-source (Open3D convention)
    fitness: float
    rmse: float
    yaw_margin: float          # gap to second-best in coarse sweep (rmse units)
    yaw_ambiguous: bool        # True when second-best is within 3% of best
    used_fpfh: bool            # True if FPFH fallback was used


def yaw_align_pair(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    voxel: float,
    yaw_step_deg: float = 3.0,
    coarse_corr_factor: float = 5.0,
) -> PairResult:
    """Align ``src`` to ``tgt`` in the floor-up frame using yaw sweep + ICP.

    Both clouds must already be in the floor-up frame (Y is vertical and
    floor is at Y=0). The translation reference is the bottom-slab AABB
    center, not the full centroid, so opposite-view bias is suppressed.
    """
    src_d = src.voxel_down_sample(voxel)
    tgt_d = tgt.voxel_down_sample(voxel)
    _ensure_normals(src_d, voxel)
    _ensure_normals(tgt_d, voxel)

    centre_src = _bottom_aabb_center(src_d)
    centre_tgt = _bottom_aabb_center(tgt_d)
    delta_xz = np.array([centre_tgt[0] - centre_src[0], centre_tgt[2] - centre_src[2]])
    T_translate = _xz_translate(delta_xz)
    pivot = np.array([centre_tgt[0], 0.5 * (centre_src[1] + centre_tgt[1]), centre_tgt[2]])

    coarse_corr = max(coarse_corr_factor * voxel, 1.5e-3)
    scores: List[Tuple[float, float, float]] = []
    angles = np.arange(0.0, 360.0, max(1.0, float(yaw_step_deg)))
    for ang in angles:
        T_yaw = _rotation_about_y(math.radians(ang), pivot)
        T_total = T_yaw @ T_translate
        ev = o3d.pipelines.registration.evaluate_registration(
            src_d, tgt_d, max_correspondence_distance=coarse_corr,
            transformation=T_total,
        )
        scores.append((float(ang), float(ev.inlier_rmse), float(ev.fitness)))

    valid = [s for s in scores if s[2] > 0.05]
    if not valid:
        valid = scores
    valid.sort(key=lambda s: s[1])
    best_ang, best_rmse, _ = valid[0]
    second_rmse = valid[1][1] if len(valid) > 1 else best_rmse * 2
    yaw_margin = max(0.0, second_rmse - best_rmse)
    rel_gap = yaw_margin / best_rmse if best_rmse > 1e-9 else float("inf")
    yaw_ambiguous = rel_gap < 0.03

    T_init = _rotation_about_y(math.radians(best_ang), pivot) @ T_translate
    res = _multi_stage_icp(src, tgt, voxel, T_init)
    return PairResult(
        T=np.asarray(res.transformation),
        fitness=float(res.fitness),
        rmse=float(res.inlier_rmse),
        yaw_margin=yaw_margin,
        yaw_ambiguous=yaw_ambiguous,
        used_fpfh=False,
    )


def fpfh_yaw_align_pair(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    voxel: float,
) -> PairResult:
    """FPFH+RANSAC global registration fallback, snapped to yaw-only + ICP.

    Used when the standard yaw sweep + ICP returned a poor fit (e.g. when
    coarse correspondence basins are misaligned by symmetry).
    """
    src_d = src.voxel_down_sample(voxel)
    tgt_d = tgt.voxel_down_sample(voxel)
    _ensure_normals(src_d, voxel)
    _ensure_normals(tgt_d, voxel)

    fp_radius = 8.0 * voxel
    src_fp = o3d.pipelines.registration.compute_fpfh_feature(
        src_d, o3d.geometry.KDTreeSearchParamHybrid(radius=fp_radius, max_nn=80)
    )
    tgt_fp = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=fp_radius, max_nn=80)
    )

    distance_threshold = 4.0 * voxel
    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_d, tgt_d, src_fp, tgt_fp, mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
        )
        T_global = np.asarray(ransac.transformation)
    except Exception as e:
        LOG.warning("FPFH RANSAC failed: %s", e)
        T_global = np.eye(4)

    pivot = _bottom_aabb_center(tgt_d)
    T_yaw_only = _yaw_only(T_global, pivot)
    res = _multi_stage_icp(src, tgt, voxel, T_yaw_only)
    return PairResult(
        T=np.asarray(res.transformation),
        fitness=float(res.fitness),
        rmse=float(res.inlier_rmse),
        yaw_margin=0.0,
        yaw_ambiguous=False,
        used_fpfh=True,
    )


def _refine_pair_from_init(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    voxel: float,
    T_init: np.ndarray,
) -> PairResult:
    """Lightweight pair refinement using a known initial transform.

    Used for the iterative pose-graph refinement step where the world poses
    from a previous global optimisation provide a known initial.
    """
    res = _multi_stage_icp(src, tgt, voxel, T_init)
    return PairResult(
        T=np.asarray(res.transformation),
        fitness=float(res.fitness),
        rmse=float(res.inlier_rmse),
        yaw_margin=0.0,
        yaw_ambiguous=False,
        used_fpfh=False,
    )


# ---------------------------------------------------------------------------
# All-pairs pose-graph
# ---------------------------------------------------------------------------
def _yaw_penalty_score(fit: float, T_rel: np.ndarray) -> float:
    """Demote pair-wise edges whose recovered yaw is large.

    Pairs with yaw > 60 deg correspond to views with reduced overlap, so
    high fitness in that regime is suspicious (likely symmetry-induced).
    The score is `fit * (1 - 0.6 * penalty)` where `penalty` ramps from 0
    at 60 deg to 1 at 180 deg.
    """
    yaw_deg = abs(math.degrees(_yaw_from_R(T_rel[:3, :3])))
    yaw_deg = min(yaw_deg, 360.0 - yaw_deg)
    penalty = max(0.0, yaw_deg - 60.0) / 120.0
    penalty = min(penalty, 1.0)
    return fit * (1.0 - 0.6 * penalty)


def _build_pose_graph(
    pair_results: dict,
    poses_world_init: List[np.ndarray],
    pcds_floor_up: List[o3d.geometry.PointCloud],
    min_edge_fitness: float,
    max_corr_dist: float,
) -> Tuple[o3d.pipelines.registration.PoseGraph, int]:
    """Construct an Open3D PoseGraph from per-pair results."""
    pg = o3d.pipelines.registration.PoseGraph()
    for T in poses_world_init:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))
    edges = 0
    for (s, t), pr in pair_results.items():
        if pr.fitness < min_edge_fitness:
            continue
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcds_floor_up[s], pcds_floor_up[t], max_corr_dist, pr.T,
        )
        is_certain = (pr.fitness >= 0.7) and (not pr.yaw_ambiguous) and (not pr.used_fpfh)
        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                s, t, pr.T, info, uncertain=not is_certain,
            )
        )
        edges += 1
    return pg, edges


def _spanning_tree_initial_poses(
    n: int,
    pair_results: dict,
    min_edge_fitness: float,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, float]]]:
    """Prim's algorithm using a yaw-penalised score so symmetric long-yaw
    edges with high fitness do not anchor the spanning tree."""
    poses: List[Optional[np.ndarray]] = [None] * n
    poses[0] = np.eye(4)
    visited = {0}
    tree_edges: List[Tuple[int, int, float]] = []
    while len(visited) < n:
        best = None  # (score, in_node, out_node, T_rel, s_is_in, fit)
        for (s, t), pr in pair_results.items():
            if (s in visited) == (t in visited):
                continue
            if pr.fitness < min_edge_fitness:
                continue
            score = _yaw_penalty_score(pr.fitness, pr.T)
            in_node, out_node = (s, t) if s in visited else (t, s)
            if best is None or score > best[0]:
                best = (score, in_node, out_node, pr.T, s == in_node, pr.fitness)
        if best is None:
            cand = None
            for (s, t), pr in pair_results.items():
                if (s in visited) == (t in visited):
                    continue
                score = _yaw_penalty_score(max(pr.fitness, 1e-6), pr.T)
                if cand is None or score > cand[0]:
                    cand = (score, s, t, pr.T, True, pr.fitness)
            if cand is None:
                LOG.error("Spanning tree disconnected at %d/%d nodes", len(visited), n)
                break
            LOG.warning(
                "Spanning tree fallback: weak edge (%d,%d) fit=%.3f score=%.3f",
                cand[1], cand[2], cand[5], cand[0],
            )
            best = cand
        score, in_node, out_node, T_rel, s_is_in, fit = best
        T_in_from_out = np.linalg.inv(T_rel) if s_is_in else T_rel
        poses[out_node] = poses[in_node] @ T_in_from_out
        visited.add(out_node)
        tree_edges.append((in_node, out_node, fit))
        LOG.info("Spanning tree: %d <- %d (fit=%.3f score=%.3f)",
                 in_node, out_node, fit, score)
    for i in range(n):
        if poses[i] is None:
            LOG.warning("Frame %d isolated; pose left at identity.", i)
            poses[i] = np.eye(4)
    return [p for p in poses], tree_edges


def run_sparse_pose_graph(
    pcds_floor_up: List[o3d.geometry.PointCloud],
    voxel: float,
    yaw_step_deg: float = 3.0,
    min_edge_fitness: float = 0.40,
    max_corr_dist: float = 4.0e-3,
    fpfh_fallback: bool = True,
    refinement_iters: int = 2,
) -> Tuple[List[np.ndarray], dict, dict]:
    """All-pairs registration + iterative pose-graph optimisation.

    Pipeline:
      1. Dense pairwise yaw-sweep + multi-stage ICP.
      2. FPFH+RANSAC fallback for any pair below ``min_edge_fitness``.
      3. Yaw-penalised Prim's spanning tree -> initial world poses.
      4. Build pose graph, run global optimisation.
      5. Refine: re-register every pair from refined poses, rebuild graph,
         re-optimise. Repeat ``refinement_iters`` times.

    Returns ``(refined_world_from_floor_up, pair_results, summary)``.
    ``summary`` contains diagnostic counts (edges_kept_per_iter, fpfh_used,
    ambiguous, isolated, ...).
    """
    n = len(pcds_floor_up)
    if n < 2:
        raise ValueError("Need at least 2 frames for pose-graph optimisation.")

    pair_results: dict = {}
    summary: dict = {
        "n_pairs_total": n * (n - 1) // 2,
        "fpfh_used": 0,
        "ambiguous": 0,
        "edges_kept_per_iter": [],
        "tree_edges": [],
        "iso_frames": [],
    }

    # Step 1: pairwise yaw + ICP.
    LOG.info("Pairwise registration: %d pairs", summary["n_pairs_total"])
    for s in range(n):
        for t in range(s + 1, n):
            pair_results[(s, t)] = yaw_align_pair(
                pcds_floor_up[s], pcds_floor_up[t],
                voxel=voxel, yaw_step_deg=yaw_step_deg,
            )

    # Step 2: FPFH fallback for weak pairs.
    if fpfh_fallback:
        weak_pairs = [(s, t) for (s, t), pr in pair_results.items()
                      if pr.fitness < min_edge_fitness]
        if weak_pairs:
            LOG.info("FPFH fallback: %d weak pairs (fit < %.2f)",
                     len(weak_pairs), min_edge_fitness)
        for (s, t) in weak_pairs:
            pr_new = fpfh_yaw_align_pair(pcds_floor_up[s], pcds_floor_up[t], voxel)
            if pr_new.fitness > pair_results[(s, t)].fitness:
                pair_results[(s, t)] = pr_new
                summary["fpfh_used"] += 1
        LOG.info("FPFH fallback: %d/%d weak pairs improved",
                 summary["fpfh_used"], len(weak_pairs))

    summary["ambiguous"] = sum(1 for pr in pair_results.values() if pr.yaw_ambiguous)

    # Step 3: yaw-penalised spanning tree -> initial poses.
    poses_world, tree_edges = _spanning_tree_initial_poses(
        n, pair_results, min_edge_fitness,
    )
    summary["tree_edges"] = tree_edges

    # Track which frames had no pose-graph edge to anchor them.
    edge_node_set = set()
    for (s, t), pr in pair_results.items():
        if pr.fitness >= min_edge_fitness:
            edge_node_set.add(s); edge_node_set.add(t)
    summary["iso_frames"] = [i for i in range(n) if i not in edge_node_set]
    if summary["iso_frames"]:
        LOG.warning("Isolated frames (no pose-graph edges): %s",
                    summary["iso_frames"])

    # Step 4: initial pose graph optimisation.
    pose_graph, edges_added = _build_pose_graph(
        pair_results, poses_world, pcds_floor_up,
        min_edge_fitness=min_edge_fitness, max_corr_dist=max_corr_dist,
    )
    LOG.info(
        "Pose graph (init): %d nodes, %d edges (%d kept of %d candidates)",
        n, len(pose_graph.edges), edges_added, len(pair_results),
    )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_dist,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    summary["edges_kept_per_iter"].append(edges_added)

    # Step 5: iterative refinement. Re-register each pair using refined poses
    # as initial transforms, rebuild the graph, re-optimise.
    for it in range(refinement_iters):
        refined_world = [pose_graph.nodes[i].pose.copy() for i in range(n)]
        for (s, t) in list(pair_results.keys()):
            T_init = np.linalg.inv(refined_world[t]) @ refined_world[s]
            pr_new = _refine_pair_from_init(
                pcds_floor_up[s], pcds_floor_up[t], voxel, T_init,
            )
            # Keep whichever (old, new) has higher fitness so we never lose
            # a previously-good edge to a regression in this iteration.
            if pr_new.fitness >= pair_results[(s, t)].fitness:
                pair_results[(s, t)] = pr_new
        pose_graph, edges_added = _build_pose_graph(
            pair_results, refined_world, pcds_floor_up,
            min_edge_fitness=min_edge_fitness, max_corr_dist=max_corr_dist,
        )
        LOG.info(
            "Pose graph (refine %d/%d): %d edges kept", it + 1, refinement_iters,
            edges_added,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        summary["edges_kept_per_iter"].append(edges_added)

    refined = [pose_graph.nodes[i].pose.copy() for i in range(n)]
    return refined, pair_results, summary


# ---------------------------------------------------------------------------
# Mesh post-processing: keep all reasonably-sized components, not just one.
# ---------------------------------------------------------------------------
def keep_components_above(
    mesh: o3d.geometry.TriangleMesh, fraction: float = 0.05, min_tris: int = 50,
) -> Tuple[o3d.geometry.TriangleMesh, List[int]]:
    """Drop only the smallest connected components.

    Returns ``(mesh, kept_sizes)``. ``kept_sizes`` is the list of triangle
    counts of every component that survived, sorted descending.
    """
    if len(mesh.triangles) == 0:
        return mesh, []
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    sizes = np.asarray(sizes)
    if sizes.size == 0:
        return mesh, []
    largest = int(sizes.max())
    threshold = max(min_tris, int(fraction * largest))
    keep = sizes >= threshold
    bad = ~keep[clusters]
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(bad)
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    kept_sizes = sorted(sizes[keep].tolist(), reverse=True)
    LOG.info(
        "Components kept: %d/%d (sizes=%s, threshold=%d tris)",
        int(keep.sum()), int(sizes.size),
        kept_sizes[:6] + (["..."] if len(kept_sizes) > 6 else []),
        threshold,
    )
    return out, kept_sizes


# ---------------------------------------------------------------------------
# Auto-segmentation preview (for sanity checking the depth-only mask)
# ---------------------------------------------------------------------------
def write_segmentation_preview(
    frames: List[Frame],
    floors: List[FloorFit],
    out_path: str,
) -> None:
    """Save a grid of depth heatmaps with auto-segmented stone outlined."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        LOG.warning("Matplotlib unavailable, skipping segmentation preview: %s", e)
        return

    n = len(frames)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6))
    axes = np.atleast_2d(axes)

    for k in range(rows * cols):
        ax = axes.flat[k]
        ax.axis("off")
        if k >= n:
            continue
        f = frames[k]
        d = f.depth.copy()
        d_finite = d[np.isfinite(d) & (d > 0)]
        if d_finite.size:
            vmin, vmax = float(np.percentile(d_finite, 2)), float(np.percentile(d_finite, 98))
        else:
            vmin, vmax = 0.0, 1.0
        ax.imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
        if f.mask.any():
            # Draw mask outline (red) by overlaying a contour.
            ax.contour(f.mask.astype(float), levels=[0.5], colors="red", linewidths=0.8)
        floor = floors[k]
        ax.set_title(
            f"#{f.index} mask={int(f.mask.sum())}px floor_inl={floor.inlier_ratio:.2f}",
            fontsize=8,
        )

    fig.suptitle("Auto-segmentation: depth (viridis) with stone mask outline (red)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    LOG.info("Wrote segmentation preview: %s", out_path)


# ---------------------------------------------------------------------------
# Sparse-tailored report
# ---------------------------------------------------------------------------
def write_sparse_report(
    out_path: str,
    intrinsics: Intrinsics,
    floors: List[FloorFit],
    poses_world_from_cam: List[np.ndarray],
    voxel_mm: float,
    sdf_trunc_mm: float,
    yaw_step_deg: float,
    edge_diagnostics: List[Tuple[int, int, float, float, float]],
    mesh_top: o3d.geometry.TriangleMesh,
    mesh_water: o3d.geometry.TriangleMesh,
    pointcloud: o3d.geometry.PointCloud,
    elapsed_s: float,
    n_frames: int,
    pair_results: Optional[dict] = None,
    pg_summary: Optional[dict] = None,
    kept_component_sizes: Optional[List[int]] = None,
    min_edge_fitness: float = 0.40,
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

    # Per-frame camera tilt from horizontal (using the per-frame floor normal).
    tilts = []
    cam_axis = np.array([0.0, 0.0, 1.0])
    for fl in floors:
        n_unit = fl.normal / np.linalg.norm(fl.normal)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(n_unit @ cam_axis)))))
        tilts.append(90.0 - ang)
    tilts = np.array(tilts)

    with open(out_path, "w") as f:
        f.write("Stone 3D reconstruction (sparse depth-only) report\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"Total runtime: {elapsed_s:.1f} s\n")
        f.write(f"Input frames: {n_frames}\n\n")
        f.write("Camera intrinsics (pixels):\n")
        f.write(f"  fx={intrinsics.fx:.3f} fy={intrinsics.fy:.3f}\n")
        f.write(f"  cx={intrinsics.cx:.3f} cy={intrinsics.cy:.3f}\n")
        f.write(f"  W={intrinsics.width} H={intrinsics.height}\n\n")
        f.write("Per-frame floor / camera tilt (deg from horizontal):\n")
        f.write(f"  tilts: min={tilts.min():.2f} max={tilts.max():.2f} "
                f"mean={tilts.mean():.2f}\n")
        f.write(f"  floor_inlier_ratio: min={min(fl.inlier_ratio for fl in floors):.3f} "
                f"max={max(fl.inlier_ratio for fl in floors):.3f}\n\n")
        f.write("Registration:\n")
        f.write(f"  yaw search step: {yaw_step_deg:.2f} deg\n")
        f.write(f"  min edge fitness: {min_edge_fitness:.2f}\n")
        f.write(f"  pose-graph edges (s,t,fit,rmse): {len(edge_diagnostics)} candidates\n")
        if edge_diagnostics:
            fits = np.array([d[2] for d in edge_diagnostics])
            f.write(f"    fitness  : min={fits.min():.3f} median={np.median(fits):.3f} max={fits.max():.3f}\n")
            rmses = np.array([d[3] for d in edge_diagnostics])
            f.write(f"    inlier_rmse (m): min={rmses.min():.5f} median={np.median(rmses):.5f} max={rmses.max():.5f}\n")
            kept_count = int((fits >= min_edge_fitness).sum())
            f.write(f"    edges with fitness >= min_edge_fitness: {kept_count}/{len(edge_diagnostics)}\n")
            # Per-frame surviving edge count.
            per_frame: List[int] = [0] * n_frames
            for (s, t, fit_, _r, _m) in edge_diagnostics:
                if fit_ >= min_edge_fitness:
                    per_frame[s] += 1
                    per_frame[t] += 1
            f.write(
                f"    per-frame surviving edges: min={min(per_frame)} median={int(np.median(per_frame))} max={max(per_frame)}\n"
            )
        # Yaw vs fitness histogram (30 deg bins).
        if pair_results is not None and pair_results:
            bins = list(range(0, 181, 30))
            counts = [0] * (len(bins) - 1)
            fits_in_bin = [[] for _ in range(len(bins) - 1)]
            for pr in pair_results.values():
                yaw_deg = abs(math.degrees(_yaw_from_R(pr.T[:3, :3])))
                yaw_deg = min(yaw_deg, 360.0 - yaw_deg)
                for k in range(len(bins) - 1):
                    if bins[k] <= yaw_deg < bins[k + 1] + (1e-6 if k == len(bins) - 2 else 0):
                        counts[k] += 1
                        fits_in_bin[k].append(pr.fitness)
                        break
            f.write("    yaw histogram (count | mean fitness):\n")
            for k in range(len(bins) - 1):
                mean_fit = float(np.mean(fits_in_bin[k])) if fits_in_bin[k] else 0.0
                f.write(
                    f"      [{bins[k]:>3}-{bins[k+1]:>3}] deg: {counts[k]:>4} pairs | fit={mean_fit:.3f}\n"
                )
        if pg_summary is not None:
            f.write(f"    yaw-ambiguous pairs (rmse gap < 3%): {pg_summary.get('ambiguous', 0)}\n")
            f.write(f"    FPFH fallback used: {pg_summary.get('fpfh_used', 0)}\n")
            f.write(f"    isolated frames: {pg_summary.get('iso_frames', [])}\n")
            kept = pg_summary.get("edges_kept_per_iter", [])
            if kept:
                f.write(f"    edges kept per pose-graph iter: {kept}\n")
            tree_edges = pg_summary.get("tree_edges", [])
            if tree_edges:
                weak = [(a, b, ff) for (a, b, ff) in tree_edges if ff < min_edge_fitness]
                f.write(f"    spanning-tree weak edges (<{min_edge_fitness:.2f}): {len(weak)} of {len(tree_edges)}\n")
        f.write("\nFusion:\n")
        f.write(f"  voxel size={voxel_mm:.3f} mm\n")
        f.write(f"  sdf trunc={sdf_trunc_mm:.3f} mm\n")
        f.write("\nMesh stats:\n")
        f.write(f"  TSDF top   : verts={len(mesh_top.vertices)} tris={len(mesh_top.triangles)}\n")
        f.write(f"  watertight : verts={len(mesh_water.vertices)} tris={len(mesh_water.triangles)}\n")
        if kept_component_sizes:
            largest = kept_component_sizes[0]
            total = sum(kept_component_sizes)
            f.write(
                f"  TSDF components kept: {len(kept_component_sizes)} "
                f"(largest covers {100 * largest / max(total, 1):.1f}% of kept tris)\n"
            )
            f.write("    sizes (top 8): " + ", ".join(str(s) for s in kept_component_sizes[:8]) + "\n")
        if bbox is not None:
            f.write(
                f"  bbox extent (mm): x={extent[0]*1000:.3f} y={extent[1]*1000:.3f} z={extent[2]*1000:.3f}\n"
            )
        f.write(f"  surface area (mm^2): {surface_area_mm2:.2f}\n")
        if not math.isnan(volume_mm3):
            f.write(f"  volume (mm^3): {volume_mm3:.2f}\n")
        f.write(f"  edge-manifold : {is_em}\n")
        f.write(f"  vertex-manifold: {is_vm}\n")
        f.write(f"  watertight    : {is_wt}\n")
        f.write(f"  merged point cloud: {len(pointcloud.points)} points\n")
        f.write("\nOutputs:\n")
        f.write("  stone_mesh_top.ply             - TSDF top-surface mesh (open at floor)\n")
        f.write("  stone_mesh_watertight.ply      - Watertight mesh with floor cap\n")
        f.write("  stone_mesh_watertight.obj      - Same as above, OBJ format\n")
        f.write("  stone_pointcloud.ply           - Merged point cloud (post pose-graph)\n")
        f.write("  stone_3d_views_composite.png   - 6-viewpoint preview render\n")
        f.write("  auto_segmentation_preview.png  - Per-frame depth + auto-mask overlay\n")
    LOG.info("Wrote report: %s", out_path)


# ---------------------------------------------------------------------------
# CLI / main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--depth_dir", required=True,
                   help="Folder with .npy depth files (one per view).")
    p.add_argument("--intrinsics", default="splits/stone/intrinsics.txt")
    p.add_argument("--sequence", default="stone_01",
                   help="Key in the intrinsics file used for fx/fy/cx/cy.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--voxel_mm", type=float, default=0.5,
                   help="TSDF voxel size in mm (sparse default 0.5).")
    p.add_argument("--sdf_trunc_mm", type=float, default=3.0,
                   help="TSDF SDF truncation distance, in mm.")
    p.add_argument("--icp_voxel_mm", type=float, default=0.6,
                   help="Voxel size for ICP downsampling, in mm.")
    p.add_argument("--icp_max_corr_mm", type=float, default=4.0,
                   help="ICP correspondence distance cutoff, in mm.")
    p.add_argument("--stone_height_thresh_mm", type=float, default=1.0,
                   help="Height above floor (mm) for stone-vs-floor classification.")
    p.add_argument("--cluster_eps_mm", type=float, default=2.0,
                   help="DBSCAN epsilon for stone clustering, in mm.")
    p.add_argument("--cluster_min_points", type=int, default=30)
    p.add_argument("--yaw_step_deg", type=float, default=3.0,
                   help="Coarse yaw sweep step size, in degrees.")
    p.add_argument("--min_edge_fitness", type=float, default=0.40,
                   help="Drop pose-graph edges with ICP fitness below this.")
    p.add_argument("--no_fpfh_fallback", action="store_true",
                   help="Disable FPFH+RANSAC fallback for low-fitness pairs.")
    p.add_argument("--refinement_iters", type=int, default=2,
                   help="Number of pose-graph refinement passes after the initial fit.")
    p.add_argument("--component_keep_fraction", type=float, default=0.05,
                   help="Keep TSDF components whose triangle count is at least this "
                        "fraction of the largest component's.")
    p.add_argument("--output_dir", default="reconstruction_output_sparse")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _stone_pcd_in_camera(
    frame: Frame, K: Intrinsics, voxel: float
) -> o3d.geometry.PointCloud:
    """Build per-frame downsampled stone point cloud in camera coords."""
    ys, xs = np.where(frame.mask)
    zs = frame.depth[ys, xs].astype(np.float64)
    valid = np.isfinite(zs) & (zs > 0)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]
    if zs.size == 0:
        return o3d.geometry.PointCloud()
    X = (xs - K.cx) * zs / K.fx
    Y = (ys - K.cy) * zs / K.fy
    pts = np.stack([X, Y, zs], axis=1)
    pcd = make_pcd(pts)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30)
    )
    return pcd


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    K = load_intrinsics(args.intrinsics, args.sequence, args.width, args.height)

    # 1) Load depth-only frames.
    frames = load_depth_only_frames(args.depth_dir, expected_size=(args.height, args.width))
    LOG.info("Loaded %d depth-only frames", len(frames))

    # 2) Auto-segment stone + fit floor in each frame.
    floors: List[FloorFit] = []
    for f in frames:
        mask, floor = auto_segment_stone(
            f.depth, K,
            stone_height_thresh_m=args.stone_height_thresh_mm * 1e-3,
            cluster_eps_m=args.cluster_eps_mm * 1e-3,
            cluster_min_points=args.cluster_min_points,
        )
        f.mask = mask
        floors.append(floor)
    nonzero = [f for f in frames if f.mask.any()]
    if len(nonzero) < 2:
        raise RuntimeError("Auto-segmentation produced fewer than 2 non-empty frames.")
    if len(nonzero) != len(frames):
        LOG.warning("%d/%d frames had empty stone mask after auto-segmentation",
                    len(frames) - len(nonzero), len(frames))
        # Drop empty-mask frames to avoid downstream errors.
        keep = [(f, fl) for f, fl in zip(frames, floors) if f.mask.any()]
        frames = [k[0] for k in keep]
        floors = [k[1] for k in keep]

    # 3) Build per-frame floor-up point clouds.
    icp_voxel = args.icp_voxel_mm * 1e-3
    pcds_cam = [_stone_pcd_in_camera(f, K, icp_voxel) for f in frames]
    floor_up_T = [floor_up_transform(fl.normal, fl.d) for fl in floors]
    pcds_floor_up: List[o3d.geometry.PointCloud] = []
    for pcd, T in zip(pcds_cam, floor_up_T):
        cp = o3d.geometry.PointCloud(pcd)
        cp.transform(T)
        # Re-estimate normals in the new frame (needed for point-to-plane ICP).
        cp.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * icp_voxel, max_nn=30)
        )
        pcds_floor_up.append(cp)

    LOG.info(
        "Floor-up point counts: min=%d median=%d max=%d",
        min(len(p.points) for p in pcds_floor_up),
        int(np.median([len(p.points) for p in pcds_floor_up])),
        max(len(p.points) for p in pcds_floor_up),
    )

    # 4-5) All-pairs registration + iterative pose-graph optimisation.
    max_corr = args.icp_max_corr_mm * 1e-3
    refined_world_from_floor_up, pair_results, pg_summary = run_sparse_pose_graph(
        pcds_floor_up,
        voxel=icp_voxel,
        yaw_step_deg=args.yaw_step_deg,
        min_edge_fitness=args.min_edge_fitness,
        max_corr_dist=max_corr,
        fpfh_fallback=not args.no_fpfh_fallback,
        refinement_iters=args.refinement_iters,
    )
    # Flatten per-pair results into a (s, t, fitness, rmse, yaw_margin) list
    # for the report writer.
    edge_diag: List[Tuple[int, int, float, float, float]] = [
        (s, t, pr.fitness, pr.rmse, pr.yaw_margin)
        for (s, t), pr in pair_results.items()
    ]

    # Final per-frame T_world<-cam_i = T_world<-floor_up_i @ floor_up_T[i].
    poses_world_from_cam: List[np.ndarray] = [
        Tw @ Tf for Tw, Tf in zip(refined_world_from_floor_up, floor_up_T)
    ]

    # 6) TSDF integration with the auto mask zeroing non-stone depth.
    mesh_top = integrate_tsdf(
        frames, poses_world_from_cam, K,
        voxel_length_m=args.voxel_mm * 1e-3,
        sdf_trunc_m=args.sdf_trunc_mm * 1e-3,
        depth_trunc_m=1.0,
    )
    mesh_top, kept_component_sizes = keep_components_above(
        mesh_top, fraction=args.component_keep_fraction,
    )

    # Merged post-registration point cloud (for QA).
    merged = merge_pointclouds(pcds_cam, poses_world_from_cam, voxel=icp_voxel)

    # 7) Watertight closure on the world-frame floor.
    # World (= floor_up frame of frame 0) has the floor at Y=0 with normal +Y.
    n_floor_world = np.array([0.0, 1.0, 0.0])
    d_floor_world = 0.0
    mesh_watertight = make_watertight_mesh(
        mesh_top, n_floor_world, d_floor_world,
        voxel_m=args.voxel_mm * 1e-3,
    )

    # 8) Save outputs.
    out = args.output_dir
    top_path = os.path.join(out, "stone_mesh_top.ply")
    water_ply = os.path.join(out, "stone_mesh_watertight.ply")
    water_obj = os.path.join(out, "stone_mesh_watertight.obj")
    pcd_path = os.path.join(out, "stone_pointcloud.ply")
    preview_path = os.path.join(out, "stone_3d_views_composite.png")
    seg_path = os.path.join(out, "auto_segmentation_preview.png")
    report_path = os.path.join(out, "reconstruction_report.txt")

    o3d.io.write_triangle_mesh(top_path, mesh_top, write_ascii=False)
    o3d.io.write_triangle_mesh(water_ply, mesh_watertight, write_ascii=False)
    o3d.io.write_triangle_mesh(water_obj, mesh_watertight, write_ascii=True)
    o3d.io.write_point_cloud(pcd_path, merged, write_ascii=False)
    LOG.info("Wrote: %s", top_path)
    LOG.info("Wrote: %s", water_ply)
    LOG.info("Wrote: %s", water_obj)
    LOG.info("Wrote: %s", pcd_path)

    write_segmentation_preview(frames, floors, seg_path)

    try:
        render_preview(mesh_watertight, preview_path, up_axis_world=n_floor_world)
    except Exception as e:
        LOG.warning("Preview render failed: %s", e)

    write_sparse_report(
        report_path, K, floors, poses_world_from_cam,
        voxel_mm=args.voxel_mm, sdf_trunc_mm=args.sdf_trunc_mm,
        yaw_step_deg=args.yaw_step_deg,
        edge_diagnostics=edge_diag,
        mesh_top=mesh_top, mesh_water=mesh_watertight,
        pointcloud=merged,
        elapsed_s=time.time() - t0,
        n_frames=len(frames),
        pair_results=pair_results,
        pg_summary=pg_summary,
        kept_component_sizes=kept_component_sizes,
        min_edge_fitness=args.min_edge_fitness,
    )
    LOG.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
