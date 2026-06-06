"""Surface reconstruction stage (NKSR / NoKSR only)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d

from .config import (
    NeuralConfig,
    StageStatus,
    require_cuda,
    require_modules,
    require_weights,
)

from .geometry import Frame, Intrinsics, make_watertight_mesh, keep_components_above


LOG = logging.getLogger("stone3d_neural.surface")

NKSR_WEIGHTS = "nksr_shapenet_scannet.pt"
NOKSR_WEIGHTS = "noksr_ptv3.pt"


@dataclass
class SurfaceResult:
    mesh: o3d.geometry.TriangleMesh
    mesh_pre_closure: o3d.geometry.TriangleMesh
    kept_component_sizes: List[int]
    status: StageStatus


class SurfaceReconstructor:
    def __init__(self, cfg: NeuralConfig) -> None:
        self.cfg = cfg
        self._nksr = None
        self._noksr = None

    def reconstruct(
        self,
        frames: List[Frame],
        poses_world_from_cam: List[np.ndarray],
        merged_pcd: o3d.geometry.PointCloud,
        K: Intrinsics,
        voxel_mm: float,
        sdf_trunc_mm: float,
        depth_trunc_m: float,
        component_keep_fraction: float,
    ) -> SurfaceResult:
        model = self.cfg.surface_model
        pkg = "nksr" if model == "nksr" else "noksr"
        wfile = NKSR_WEIGHTS if model == "nksr" else NOKSR_WEIGHTS

        require_cuda("surface")
        require_modules("surface", ["torch", pkg])
        require_weights("surface", wfile, self.cfg.models_dir)

        t0 = time.time()
        mesh_pre, kept_sizes = self._reconstruct_neural(merged_pcd, voxel_mm)
        mesh_water = self._ensure_watertight(mesh_pre, voxel_mm)

        status = StageStatus(
            stage="surface",
            latency_s=time.time() - t0,
            extra={
                "model": model,
                "kept_component_sizes": kept_sizes,
                "n_vertices": int(np.asarray(mesh_water.vertices).shape[0]),
                "n_triangles": int(np.asarray(mesh_water.triangles).shape[0]),
                "is_watertight": bool(mesh_water.is_watertight()) if len(mesh_water.triangles) else False,
            },
        )
        if self.cfg.log_backend_decisions:
            LOG.info(str(status))
        return SurfaceResult(
            mesh=mesh_water,
            mesh_pre_closure=mesh_pre,
            kept_component_sizes=kept_sizes,
            status=status,
        )

    def _reconstruct_neural(
        self,
        merged_pcd: o3d.geometry.PointCloud,
        voxel_mm: float,
    ) -> Tuple[o3d.geometry.TriangleMesh, List[int]]:
        if self.cfg.surface_model == "nksr":
            mesh = self._get_nksr().reconstruct(merged_pcd, voxel_m=voxel_mm * 1e-3)
        else:
            mesh = self._get_noksr().reconstruct(merged_pcd, voxel_m=voxel_mm * 1e-3)
        mesh.compute_vertex_normals()
        mesh, kept_sizes = keep_components_above(mesh, fraction=0.05)
        return mesh, kept_sizes

    def _ensure_watertight(
        self, mesh: o3d.geometry.TriangleMesh, voxel_mm: float
    ) -> o3d.geometry.TriangleMesh:
        if len(mesh.triangles) == 0:
            raise RuntimeError("Neural surface reconstruction returned an empty mesh")
        if mesh.is_watertight():
            return mesh
        LOG.info("Surface mesh is not watertight; running polygon-cap closure")
        n_world = np.array([0.0, 1.0, 0.0])
        return make_watertight_mesh(mesh, n_world, 0.0, voxel_m=voxel_mm * 1e-3)

    def _get_nksr(self):
        if self._nksr is not None:
            return self._nksr
        from .nksr_loader import load_nksr
        self._nksr = load_nksr(
            weights=f"{self.cfg.models_dir}/{NKSR_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._nksr

    def _get_noksr(self):
        if self._noksr is not None:
            return self._noksr
        from .noksr_loader import load_noksr
        self._noksr = load_noksr(
            weights=f"{self.cfg.models_dir}/{NOKSR_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._noksr
