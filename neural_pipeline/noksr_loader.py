"""NoKSR loader (kernel-free neural surface reconstruction)."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

LOG = logging.getLogger("stone3d_neural.noksr")


def load_noksr(weights: str, device: str = "cuda") -> "NoKSRCallable":
    try:
        import torch
        import noksr  # type: ignore
    except Exception as e:
        raise RuntimeError(f"NoKSR import failed: {e}") from e

    try:
        net = noksr.NoKSRReconstructor.from_pretrained(weights, device=device)
        net.eval()
        LOG.info("Loaded NoKSR on %s (weights=%s)", device, weights)
        return NoKSRCallable(net, device)
    except Exception as e:
        raise RuntimeError(f"NoKSR construction failed ({weights}): {e}") from e


class NoKSRCallable:
    def __init__(self, net, device: str) -> None:
        self.net = net
        self.device = device

    def reconstruct(self, pcd: o3d.geometry.PointCloud, voxel_m: float) -> o3d.geometry.TriangleMesh:
        import torch
        pts = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        if normals is None or normals.shape[0] != pts.shape[0]:
            cp = o3d.geometry.PointCloud(pcd)
            cp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel_m, max_nn=30)
            )
            normals = np.asarray(cp.normals, dtype=np.float32)
        x = torch.from_numpy(pts).to(self.device)
        n = torch.from_numpy(normals).to(self.device)
        with torch.inference_mode():
            verts, tris = self.net.reconstruct(x, n, voxel_size=voxel_m)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy().astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(tris.detach().cpu().numpy().astype(np.int32))
        m.compute_vertex_normals()
        return m
