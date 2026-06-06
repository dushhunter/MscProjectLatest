"""Shared geometry utilities for the neural stone reconstruction pipeline.

Self-contained module: no imports from the classical reconstruction scripts
(``reconstruct_stone_3d.py`` or ``reconstruct_stone_3d_sparse.py``).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")

import open3d as o3d  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

LOG = logging.getLogger("stone3d_neural.geometry")


# =========================================================================
# Dataclasses
# =========================================================================
@dataclass
class Intrinsics:
    """Pinhole camera intrinsics in pixel units."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        K = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def to_o3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


@dataclass
class Frame:
    """One captured rotational frame."""

    index: int
    depth: np.ndarray   # float32, (H, W), metres
    mask: np.ndarray    # bool, (H, W)
    color: np.ndarray   # uint8, (H, W, 3)


@dataclass
class FloorFit:
    normal: np.ndarray   # unit, points toward camera (negative z)
    d: float             # plane offset: n . X + d = 0
    inlier_ratio: float  # |inliers| / |full cloud|


@dataclass
class PairResult:
    T: np.ndarray              # T_target<-source (Open3D convention)
    fitness: float
    rmse: float
    yaw_margin: float          # gap to second-best in coarse sweep (rmse units)
    yaw_ambiguous: bool        # True when second-best is within 3% of best
    used_fpfh: bool            # True if FPFH fallback was used


# =========================================================================
# Loaders
# =========================================================================
def load_intrinsics(path: str, sequence: str, width: int, height: int) -> Intrinsics:
    """Parse the normalized KV intrinsics file (matches datasets/stone_dataset.py)."""
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == sequence:
                fx_n, fy_n, cx_n, cy_n = (float(x) for x in parts[1:5])
                fx, fy = fx_n * width, fy_n * height
                cx, cy = cx_n * width, cy_n * height
                LOG.info(
                    "Intrinsics for %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f (W=%d H=%d)",
                    sequence, fx, fy, cx, cy, width, height,
                )
                return Intrinsics(width, height, fx, fy, cx, cy)
    raise KeyError(f"Sequence '{sequence}' not found in {path}")


def list_depth_files(depth_dir: str) -> List[str]:
    """Return sorted list of .npy depth files in *depth_dir* (non-recursive)."""
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
    """Load one depth-only frame as a :class:`Frame`.

    A uniform grey colour image is synthesised (Open3D's TSDF integrate
    requires an RGBDImage with both colour and depth). The mask is
    initialised to all-False and replaced by the auto-segmentation step.
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


# =========================================================================
# Geometry helpers
# =========================================================================
def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis by *angle* (right-hand rule)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    return R


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


def floor_up_transform(n_floor_cam: np.ndarray, d_floor_cam: float) -> np.ndarray:
    """Build the camera-frame -> "floor-up" world transform.

    After applying this transform:
      - The floor plane lies on Y=0.
      - The floor normal points to +Y (so "up" is +Y in this frame).
      - The X and Z axes lie in the floor plane (yaw is the only free
        rotational DoF).
    """
    n = n_floor_cam / np.linalg.norm(n_floor_cam)
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(n, target)
    s = np.linalg.norm(axis)
    c = float(n @ target)
    if s < 1e-9:
        if c > 0:
            R = np.eye(3)
        else:
            R = np.diag([1.0, -1.0, -1.0])
    else:
        axis /= s
        angle = math.atan2(s, c)
        R = axis_angle_to_matrix(axis, angle)

    T = np.eye(4)
    T[:3, :3] = R
    T[1, 3] = d_floor_cam
    return T


def _rotation_about_y(theta: float, centre: np.ndarray) -> np.ndarray:
    """4x4 rotation by *theta* (radians) around the vertical axis through *centre*."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre - R @ centre
    return T


def _yaw_from_R(R: np.ndarray) -> float:
    """Return the rotation angle (radians) about +Y for a rotation matrix."""
    return math.atan2(float(R[0, 2]), float(R[0, 0]))


def _yaw_only(T: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Snap a 4x4 rigid transform to a yaw-only rotation around +Y through *pivot*."""
    yaw = _yaw_from_R(T[:3, :3])
    R_yaw = _rotation_about_y(yaw, pivot)
    p_image = (T @ np.array([pivot[0], pivot[1], pivot[2], 1.0]))[:3]
    R_yaw_p = (R_yaw @ np.array([pivot[0], pivot[1], pivot[2], 1.0]))[:3]
    delta = p_image - R_yaw_p
    R_yaw[0, 3] += delta[0]
    R_yaw[2, 3] += delta[2]
    return R_yaw


def _bottom_aabb_center(pcd: o3d.geometry.PointCloud, slab_mm: float = 5.0) -> np.ndarray:
    """Center of the AABB of points within *slab_mm* of the lowest Y in *pcd*."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3)
    y = pts[:, 1]
    bottom = pts[(y - y.min()) <= slab_mm * 1e-3]
    if bottom.shape[0] < 20:
        bottom = pts
    mn = bottom.min(axis=0); mx = bottom.max(axis=0)
    return 0.5 * (mn + mx)


# =========================================================================
# Point cloud / mesh utilities
# =========================================================================
def make_pcd(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    estimate_normals: bool = False,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and colors.size:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if estimate_normals and points.shape[0] >= 10:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=20)
    return pcd


def merge_pointclouds(
    pcds: List[o3d.geometry.PointCloud],
    poses: List[np.ndarray],
    voxel: float,
) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for pcd, T in zip(pcds, poses):
        cp = o3d.geometry.PointCloud(pcd)
        cp.transform(T)
        merged += cp
    merged = merged.voxel_down_sample(voxel)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=6 * voxel, max_nn=40)
    )
    return merged


def _fit_floor_plane(
    pts: np.ndarray,
    distance_threshold: float = 3e-4,
    num_iterations: int = 3000,
) -> Tuple[FloorFit, np.ndarray]:
    """RANSAC plane fit. Returns plane params plus boolean inlier mask of *pts*."""
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
    if n[2] > 0:
        n = -n
        d_off = -d_off
    inlier_mask = np.zeros(len(pts), dtype=bool)
    inlier_mask[np.asarray(inliers, dtype=np.int64)] = True
    return FloorFit(n, d_off, float(inlier_mask.mean())), inlier_mask


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


def _refine_pair_from_init(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    voxel: float,
    T_init: np.ndarray,
) -> PairResult:
    """Lightweight pair refinement using a known initial transform."""
    res = _multi_stage_icp(src, tgt, voxel, T_init)
    return PairResult(
        T=np.asarray(res.transformation),
        fitness=float(res.fitness),
        rmse=float(res.inlier_rmse),
        yaw_margin=0.0,
        yaw_ambiguous=False,
        used_fpfh=False,
    )


def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh
    largest = int(np.argmax(cluster_n_triangles))
    triangles_to_remove = triangle_clusters != largest
    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(triangles_to_remove)
    cleaned.remove_unreferenced_vertices()
    cleaned.compute_vertex_normals()
    LOG.info(
        "Largest component kept: %d/%d triangles",
        cluster_n_triangles[largest], int(cluster_n_triangles.sum()),
    )
    return cleaned


def keep_components_above(
    mesh: o3d.geometry.TriangleMesh, fraction: float = 0.05, min_tris: int = 50,
) -> Tuple[o3d.geometry.TriangleMesh, List[int]]:
    """Drop only the smallest connected components.

    Returns ``(mesh, kept_sizes)``.
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


def _boundary_loops(mesh: o3d.geometry.TriangleMesh) -> List[List[int]]:
    """Return ordered vertex-index loops along the open boundary of the mesh."""
    tris = np.asarray(mesh.triangles)
    edge_count: dict = {}
    edge_dir: dict = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_dir.setdefault(key, (a, b))
    boundary = [edge_dir[k] for k, c in edge_count.items() if c == 1]
    if not boundary:
        return []

    nxt: dict = {}
    for a, b in boundary:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)

    used: set = set()
    loops: List[List[int]] = []
    for a, b in boundary:
        edge_key = (min(a, b), max(a, b))
        if edge_key in used:
            continue
        loop = [a]
        prev = a
        cur = b
        used.add(edge_key)
        while cur != a:
            loop.append(cur)
            options = nxt.get(cur, [])
            chosen = None
            for nb in options:
                k = (min(cur, nb), max(cur, nb))
                if k in used:
                    continue
                if nb == prev:
                    continue
                chosen = nb
                used.add(k)
                break
            if chosen is None:
                break
            prev, cur = cur, chosen
        loops.append(loop)
    return loops


def make_watertight_mesh(
    top_mesh: o3d.geometry.TriangleMesh,
    floor_normal: np.ndarray,
    floor_d: float,
    voxel_m: float,
    smoothing_iters: int = 4,
    **_unused,
) -> o3d.geometry.TriangleMesh:
    """Produce a watertight mesh by cropping above the floor, capping with a
    polygon-respecting Delaunay triangulation, and merging into a closed mesh.
    """
    if len(top_mesh.triangles) == 0:
        raise RuntimeError("Top mesh is empty; cannot create watertight closure.")

    floor_normal = floor_normal / np.linalg.norm(floor_normal)

    verts = np.asarray(top_mesh.vertices, dtype=np.float64)
    side = verts @ floor_normal + floor_d
    keep_mask = side >= 0.5 * voxel_m
    if keep_mask.sum() < 100:
        keep_mask = side >= -voxel_m
    cropped = o3d.geometry.TriangleMesh(top_mesh)
    cropped.remove_vertices_by_mask(~keep_mask)
    cropped = keep_largest_component(cropped)
    cropped.remove_duplicated_vertices()
    cropped.remove_duplicated_triangles()
    cropped.remove_degenerate_triangles()
    cropped.remove_unreferenced_vertices()

    cropped_verts = np.asarray(cropped.vertices, dtype=np.float64)
    cropped_tris = np.asarray(cropped.triangles, dtype=np.int64)
    LOG.info(
        "Cropped top mesh: %d verts %d tris (above floor)",
        len(cropped_verts), len(cropped_tris),
    )

    loops = _boundary_loops(cropped)
    if not loops:
        LOG.warning("No open boundary detected; mesh is already closed.")
        cropped.compute_vertex_normals()
        return cropped

    main_loop = max(loops, key=len)
    LOG.info("Boundary loops: %d (main length=%d)", len(loops), len(main_loop))

    boundary_pts3 = cropped_verts[main_loop]
    side_b = boundary_pts3 @ floor_normal + floor_d
    boundary_proj = boundary_pts3 - np.outer(side_b, floor_normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(floor_normal @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - (tmp @ floor_normal) * floor_normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(floor_normal, e1)
    p0 = boundary_proj.mean(axis=0)
    boundary2d = np.stack([(boundary_proj - p0) @ e1, (boundary_proj - p0) @ e2], axis=1)

    poly_path = MplPath(boundary2d)
    step = max(2.5 * voxel_m, 0.8e-3)
    pad = -step
    minxy = boundary2d.min(axis=0) - pad
    maxxy = boundary2d.max(axis=0) + pad
    if np.all(maxxy > minxy):
        gx = np.arange(minxy[0], maxxy[0], step)
        gy = np.arange(minxy[1], maxxy[1], step)
        GX, GY = np.meshgrid(gx, gy, indexing="xy")
        grid2d = np.stack([GX.ravel(), GY.ravel()], axis=1)
        inside = poly_path.contains_points(grid2d, radius=-1e-6)
        interior2d = grid2d[inside]
    else:
        interior2d = np.empty((0, 2), dtype=np.float64)

    cap_pts2d = np.concatenate([boundary2d, interior2d], axis=0)
    cap_pts3d = p0 + np.outer(cap_pts2d[:, 0], e1) + np.outer(cap_pts2d[:, 1], e2)
    LOG.info(
        "Cap (polygon-respecting): boundary=%d interior=%d (grid step=%.3f mm)",
        len(boundary2d), len(interior2d), step * 1000.0,
    )

    tri_all = Delaunay(cap_pts2d).simplices
    centroids2d = cap_pts2d[tri_all].mean(axis=1)
    inside_mask = poly_path.contains_points(centroids2d, radius=-1e-9)
    cap_tri_local = tri_all[inside_mask]

    n_top_v = len(cropped_verts)
    n_boundary = len(main_loop)
    boundary_cap_indices = np.array(main_loop, dtype=np.int64)
    interior_cap_indices = n_top_v + np.arange(len(interior2d), dtype=np.int64)
    cap_index_remap = np.concatenate([boundary_cap_indices, interior_cap_indices], axis=0)

    cap_tri_global = cap_index_remap[cap_tri_local]

    merged_verts = np.concatenate([cropped_verts, cap_pts3d[n_boundary:]], axis=0)
    a = merged_verts[cap_tri_global[:, 0]]
    b = merged_verts[cap_tri_global[:, 1]]
    c = merged_verts[cap_tri_global[:, 2]]
    cap_normals = np.cross(b - a, c - a)
    flip = (cap_normals @ floor_normal) > 0
    cap_tri_global[flip] = cap_tri_global[flip][:, [0, 2, 1]]

    merged_tris = np.concatenate([cropped_tris, cap_tri_global], axis=0)

    extra_loops = [lp for lp in loops if lp is not main_loop and len(lp) >= 3]
    if extra_loops:
        added_fan_tris = []
        for lp in extra_loops:
            pts = cropped_verts[lp]
            centre = pts.mean(axis=0)
            new_idx = len(merged_verts)
            merged_verts = np.concatenate([merged_verts, centre[None, :]], axis=0)
            for k in range(len(lp)):
                v0 = lp[k]
                v1 = lp[(k + 1) % len(lp)]
                added_fan_tris.append([new_idx, v0, v1])
        added_fan_tris = np.asarray(added_fan_tris, dtype=np.int64)
        merged_tris = np.concatenate([merged_tris, added_fan_tris], axis=0)
        LOG.info("Closed %d interior holes via fan triangulation", len(extra_loops))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(merged_verts)
    mesh.triangles = o3d.utility.Vector3iVector(merged_tris)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = keep_largest_component(mesh)
    if smoothing_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iters)
    mesh.compute_vertex_normals()

    is_em = mesh.is_edge_manifold()
    is_vm = mesh.is_vertex_manifold()
    is_wt = mesh.is_watertight()
    LOG.info(
        "Watertight mesh: %d vertices, %d triangles (top+cap)  "
        "edge-manifold=%s vertex-manifold=%s watertight=%s",
        len(mesh.vertices), len(mesh.triangles), is_em, is_vm, is_wt,
    )
    return mesh


# =========================================================================
# Visualization / preview
# =========================================================================
def _matplotlib_preview(
    mesh: o3d.geometry.TriangleMesh,
    out_png: str,
    up_axis_world: np.ndarray,
    n_views: int = 6,
) -> None:
    """Pure-matplotlib fallback preview using triangle-normal shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        LOG.warning("Empty mesh; skipping preview.")
        return

    up = up_axis_world / np.linalg.norm(up_axis_world)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - (tmp @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    centre = verts.mean(axis=0)
    bbox = verts.max(axis=0) - verts.min(axis=0)
    extent = float(np.linalg.norm(bbox))
    cam_dist = max(0.05, 2.0 * extent)
    elevation = math.radians(25.0)

    cols = 3 if n_views >= 3 else n_views
    rows = math.ceil(n_views / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
    axes = np.atleast_2d(axes)

    key_world = (0.4 * up + 0.7 * np.array([0.6, 0.0, 0.6]))
    fill_world = (0.2 * up - 0.6 * np.array([0.6, 0.0, 0.6]))
    key_world /= np.linalg.norm(key_world)
    fill_world /= np.linalg.norm(fill_world)

    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    n_world = np.cross(b - a, c - a)
    n_norms = np.linalg.norm(n_world, axis=1, keepdims=True)
    n_norms[n_norms == 0] = 1.0
    n_world /= n_norms

    for k in range(rows * cols):
        ax = axes.flat[k]
        ax.set_aspect("equal")
        ax.axis("off")
        if k >= n_views:
            continue

        theta = 2 * math.pi * k / n_views
        view_dir = (math.cos(theta) * e1 + math.sin(theta) * e2) * math.cos(elevation) \
                   + math.sin(elevation) * up
        right = np.cross(view_dir, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, view_dir)
        cam_up /= np.linalg.norm(cam_up)

        rel = verts - centre
        u = rel @ right
        v = rel @ cam_up
        depth = rel @ view_dir
        tri_v = np.stack([u[tris], v[tris]], axis=-1)
        tri_d = depth[tris].mean(axis=1)

        n = n_world.copy()
        flip = (n @ -view_dir) < 0
        n[flip] = -n[flip]

        shade_key = np.clip(n @ key_world, 0.0, 1.0)
        shade_fill = np.clip(n @ fill_world, 0.0, 1.0)
        shade_head = np.clip(n @ -view_dir, 0.0, 1.0) ** 1.2
        ambient = 0.35
        intensity = np.clip(
            ambient + 0.45 * shade_key + 0.20 * shade_fill + 0.25 * shade_head,
            0.0, 1.0,
        )
        base_color = np.array([0.86, 0.74, 0.62])
        face_rgb = intensity[:, None] * base_color[None, :]

        order = np.argsort(tri_d)[::-1]
        polys = tri_v[order]
        colors = np.clip(face_rgb[order], 0.0, 1.0)
        coll = PolyCollection(polys, facecolors=colors, edgecolors="none",
                              linewidths=0, antialiased=False)
        ax.add_collection(coll)
        m = max(np.ptp(u), np.ptp(v)) * 0.6 + 1e-6
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_title(f"view {k * 360 // n_views} deg", fontsize=10)

    fig.suptitle("Stone 3D reconstruction \u2014 multi-view preview", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)
    LOG.info("Wrote preview: %s", out_png)


def render_preview(
    mesh: o3d.geometry.TriangleMesh,
    out_png: str,
    up_axis_world: Optional[np.ndarray] = None,
    width: int = 900,
    height: int = 900,
    n_views: int = 6,
    use_gl: bool = False,
) -> None:
    """Render multiple viewpoints of the mesh to a single composite PNG."""
    if len(mesh.triangles) == 0:
        LOG.warning("Empty mesh; skipping preview render.")
        return

    up = up_axis_world if up_axis_world is not None else np.array([0.0, 1.0, 0.0])
    up = up / np.linalg.norm(up)

    if not use_gl:
        _matplotlib_preview(mesh, out_png, up_axis_world=up, n_views=n_views)
        return

    try:
        mesh_norm = o3d.geometry.TriangleMesh(mesh)
        mesh_norm.compute_vertex_normals()
        bbox = mesh_norm.get_axis_aligned_bounding_box()
        centre = np.asarray(bbox.get_center())
        extent = float(np.linalg.norm(np.asarray(bbox.get_extent())))
        cam_dist = max(0.05, 2.0 * extent)

        tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ up) * up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.86, 0.74, 0.62, 1.0]
        mat.base_roughness = 0.85
        mat.base_metallic = 0.0
        renderer.scene.add_geometry("mesh", mesh_norm, mat)
        sun_dir = -(0.4 * up + 0.7 * e1 + 0.6 * e2)
        sun_dir /= np.linalg.norm(sun_dir)
        renderer.scene.scene.set_sun_light(sun_dir.tolist(), [1.0, 1.0, 1.0], 90000)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS, sun_dir
        )

        elevation = math.radians(25.0)
        images: List[np.ndarray] = []
        for k in range(n_views):
            theta = 2 * math.pi * k / n_views
            view_dir = (math.cos(theta) * e1 + math.sin(theta) * e2) * math.cos(elevation) \
                       + math.sin(elevation) * up
            eye = centre + cam_dist * view_dir
            renderer.setup_camera(50.0, centre, eye, up)
            img = np.asarray(renderer.render_to_image())
            images.append(img)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = 3 if n_views >= 3 else n_views
        rows = math.ceil(n_views / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.atleast_2d(axes)
        for k in range(rows * cols):
            ax = axes.flat[k]
            ax.axis("off")
            if k < n_views:
                ax.imshow(images[k])
                ax.set_title(f"view {k * 360 // n_views} deg", fontsize=10)
        fig.suptitle("Stone 3D reconstruction \u2014 multi-view preview", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, facecolor="white")
        plt.close(fig)
        LOG.info("Wrote preview (GL): %s", out_png)
        return
    except Exception as e:
        LOG.warning("GL preview failed (%s); falling back to matplotlib renderer.", e)

    _matplotlib_preview(mesh, out_png, up_axis_world=up, n_views=n_views)


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
