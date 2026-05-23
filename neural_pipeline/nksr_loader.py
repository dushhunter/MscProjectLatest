"""NKSR (Neural Kernel Surface Reconstruction) loader."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

LOG = logging.getLogger("stone3d_neural.nksr")


def load_nksr(weights: str, device: str = "cuda") -> "NKSRCallable":
    try:
        import torch
        import nksr  # type: ignore
    except Exception as e:
        raise RuntimeError(f"NKSR import failed: {e}") from e

    try:
        reconstructor = nksr.Reconstructor(device=device)
        if weights:
            try:
                state = torch.load(weights, map_location=device)
                reconstructor.network.load_state_dict(state, strict=False)
            except FileNotFoundError:
                LOG.info("NKSR weights file %s not found; using package default", weights)
        LOG.info("Loaded NKSR on %s (weights=%s)", device, weights)
        return NKSRCallable(reconstructor, device)
    except Exception as e:
        raise RuntimeError(f"NKSR construction failed: {e}") from e


class NKSRCallable:
    def __init__(self, reconstructor, device: str) -> None:
        self.reconstructor = reconstructor
        self.device = device

    def reconstruct(self, pcd: o3d.geometry.PointCloud, voxel_m: float) -> o3d.geometry.TriangleMesh:
        import torch
        pts = np.asarray(pcd.points, dtype=np.float32)
        normals = (
            np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        )
        if normals is None or normals.shape[0] != pts.shape[0]:
            cp = o3d.geometry.PointCloud(pcd)
            cp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel_m, max_nn=30)
            )
            normals = np.asarray(cp.normals, dtype=np.float32)

        x = torch.from_numpy(pts).to(self.device)
        n = torch.from_numpy(normals).to(self.device)
        with torch.inference_mode():
            field = self.reconstructor.reconstruct(x, normal=n, voxel_size=voxel_m)
            mesh = field.extract_dual_mesh()
        verts = mesh.v.detach().cpu().numpy().astype(np.float64)
        tris = mesh.f.detach().cpu().numpy().astype(np.int32)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(tris)
        m.compute_vertex_normals()
        return m
