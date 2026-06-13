#!/usr/bin/env python3
"""Align Blender GT meshes to the depth-merge registration frame via ICP.

For each stone:
  1. Load the original Blender PLY mesh and sample surface points.
  2. Load the depth-merge reference cloud (from prepare_gt.py).
  3. RANSAC global registration + ICP refinement to find the transform.
  4. Apply the transform, FPS to 16384 points, center.
  5. Save the aligned GT cloud, update registration.npz and volumes JSON.

Usage:
    python align_blender_gt.py \
        --gt_cloud_dir stone_syn_dataset/gt_clouds \
        --blender_dir stone_syn_dataset/blender_gt_backup
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

LOG = logging.getLogger("align_blender_gt")


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Greedy FPS to downsample to n points."""
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
        selected.append(int(np.argmax(dists)))
    return pts[np.array(selected)]


def _compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30,
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100,
        ),
    )
    return pcd_down, fpfh


def _ransac_global(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
) -> np.ndarray:
    """RANSAC-based global registration using FPFH features."""
    src_down, src_fpfh = _compute_fpfh(source, voxel_size)
    tgt_down, tgt_fpfh = _compute_fpfh(target, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                voxel_size * 2.0
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=4_000_000, confidence=0.999,
        ),
    )
    LOG.info("  RANSAC fitness=%.4f, RMSE=%.6f", result.fitness, result.inlier_rmse)
    return result.transformation


def _icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Point-to-plane ICP refinement."""
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist * 4, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist * 4, max_nn=30)
    )
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_dist,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,
        ),
    )
    LOG.info("  ICP fitness=%.4f, RMSE=%.6f", result.fitness, result.inlier_rmse)
    return result.transformation


def align_one_stone(
    stone_id: str,
    blender_ply: str,
    depthmerge_ply: str,
    output_dir: str,
    n_surface: int = 100_000,
    n_final: int = 16_384,
    voxel_size: float = 0.002,
    icp_max_dist: float = 0.005,
) -> bool:
    """Align one Blender GT mesh to the depth-merge reference frame."""
    LOG.info("Aligning %s ...", stone_id)
    LOG.info("  Blender PLY:   %s", blender_ply)
    LOG.info("  Depth-merge:   %s", depthmerge_ply)

    mesh = o3d.io.read_triangle_mesh(blender_ply)
    has_faces = len(mesh.triangles) > 0
    if has_faces:
        mesh.compute_vertex_normals()
        blender_pts = np.asarray(
            mesh.sample_points_uniformly(n_surface).points, dtype=np.float64
        )
        LOG.info("  Blender mesh: %d verts, %d tris",
                 len(mesh.vertices), len(mesh.triangles))
    else:
        blender_pts = np.asarray(mesh.vertices, dtype=np.float64)
        LOG.info("  Blender point cloud: %d pts", blender_pts.shape[0])

    target_pcd = o3d.io.read_point_cloud(depthmerge_ply)
    target_pts = np.asarray(target_pcd.points, dtype=np.float64)
    LOG.info("  Depth-merge target: %d points", target_pts.shape[0])

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(blender_pts)

    src_centroid = blender_pts.mean(axis=0)
    tgt_centroid = target_pts.mean(axis=0)
    T_init = np.eye(4, dtype=np.float64)
    T_init[:3, 3] = tgt_centroid - src_centroid
    src_pcd_shifted = o3d.geometry.PointCloud(src_pcd)
    src_pcd_shifted.transform(T_init)

    LOG.info("  Running RANSAC global registration ...")
    T_ransac = _ransac_global(src_pcd_shifted, target_pcd, voxel_size)

    LOG.info("  Running ICP refinement ...")
    T_icp = _icp_refine(src_pcd_shifted, target_pcd, T_ransac, icp_max_dist)

    T_full = T_icp @ T_init
    aligned_pts = (T_full[:3, :3] @ blender_pts.T).T + T_full[:3, 3]

    aligned_fps = _farthest_point_sample(
        aligned_pts.astype(np.float32), n_final
    ).astype(np.float64)
    gt_centroid = aligned_fps.mean(axis=0)
    aligned_centered = aligned_fps - gt_centroid

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(aligned_centered)
    out_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30)
    )

    out_path = os.path.join(output_dir, f"{stone_id}_gt_aligned.ply")
    o3d.io.write_point_cloud(out_path, out_pcd)
    LOG.info("  Saved aligned GT: %s (%d pts)", out_path, n_final)

    reg_path = os.path.join(output_dir, f"{stone_id}_registration.npz")
    if os.path.isfile(reg_path):
        reg_data = dict(np.load(reg_path))
    else:
        reg_data = {}
    reg_data["gt_centroid"] = gt_centroid.astype(np.float64)
    reg_data["blender_to_registered"] = T_full.astype(np.float64)
    np.savez(reg_path, **reg_data)
    LOG.info("  Updated registration params: %s", reg_path)

    span = aligned_centered.max(axis=0) - aligned_centered.min(axis=0)
    LOG.info("  Aligned span: X=%.2f cm  Y=%.2f cm  Z=%.2f cm",
             span[0] * 100, span[1] * 100, span[2] * 100)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Align Blender GT meshes to the depth-merge registration frame"
    )
    parser.add_argument(
        "--blender_dir", required=True,
        help="Directory with original Blender PLY files "
             "(stone_XX_gt_pointcloud.ply or stone_XX_gt.ply or stone_XX.ply)",
    )
    parser.add_argument(
        "--gt_cloud_dir", required=True,
        help="Directory with depth-merge reference clouds and registration.npz "
             "(output of prepare_gt.py). Aligned GT is written here.",
    )
    parser.add_argument(
        "--stones", nargs="*", default=None,
        help="Stone IDs to process (default: auto-detect from blender_dir)",
    )
    parser.add_argument("--n_surface", type=int, default=100_000)
    parser.add_argument("--n_final", type=int, default=16_384)
    parser.add_argument("--voxel_size", type=float, default=0.002,
                        help="Voxel size for FPFH feature computation")
    parser.add_argument("--icp_max_dist", type=float, default=0.005,
                        help="Max correspondence distance for ICP")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    import re
    blender_dir = args.blender_dir
    gt_cloud_dir = args.gt_cloud_dir

    if args.stones:
        stone_ids = args.stones
    else:
        stone_ids = []
        pattern = re.compile(r"(stone_\d+)")
        for fname in sorted(os.listdir(blender_dir)):
            if fname.lower().endswith(".ply"):
                m = pattern.search(fname)
                if m and m.group(1) not in stone_ids:
                    stone_ids.append(m.group(1))
        LOG.info("Auto-detected stones from %s: %s", blender_dir, stone_ids)

    if not stone_ids:
        LOG.error("No stone PLY files found in %s", blender_dir)
        sys.exit(1)

    n_ok, n_fail = 0, 0

    for sid in stone_ids:
        blender_ply = None
        for name_pattern in [
            f"{sid}_gt_pointcloud.ply", f"{sid}_gt.ply", f"{sid}.ply",
        ]:
            candidate = os.path.join(blender_dir, name_pattern)
            if os.path.isfile(candidate):
                blender_ply = candidate
                break
        if blender_ply is None:
            LOG.warning("No Blender PLY found for %s in %s — skipping", sid, blender_dir)
            n_fail += 1
            continue

        depthmerge_ply = os.path.join(gt_cloud_dir, f"{sid}_depthmerge_ref.ply")
        if not os.path.isfile(depthmerge_ply):
            LOG.warning("No depth-merge reference for %s at %s — skipping",
                        sid, depthmerge_ply)
            n_fail += 1
            continue

        try:
            ok = align_one_stone(
                sid, blender_ply, depthmerge_ply, gt_cloud_dir,
                n_surface=args.n_surface, n_final=args.n_final,
                voxel_size=args.voxel_size, icp_max_dist=args.icp_max_dist,
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            LOG.error("Failed to align %s: %s", sid, e, exc_info=True)
            n_fail += 1

    LOG.info("Done — %d aligned, %d failed.", n_ok, n_fail)
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
