"""Pairwise point cloud registration (PARE-Net + GeoTransformer fallback)."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    NeuralConfig,
    StageStatus,
    require_cuda,
    require_modules,
    require_weights,
    weights_exist,
)

import os
os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "1")
import open3d as o3d  # noqa: E402

from .geometry import (  # noqa: E402
    PairResult,
    _refine_pair_from_init,
    _multi_stage_icp,
    _yaw_only,
    _bottom_aabb_center,
)


LOG = logging.getLogger("stone3d_neural.pair")

PARE_WEIGHTS = "parenet_3dmatch.pth"
GEOTR_WEIGHTS = "geotransformer_3dmatch.pth"


class PairwiseRegistrar:
    def __init__(self, cfg: NeuralConfig) -> None:
        self.cfg = cfg
        self._pare = None
        self._geotr = None

    def register_all_pairs(
        self,
        pcds_floor_up: List[o3d.geometry.PointCloud],
        voxel_m: float,
        min_edge_fitness: float,
    ) -> Tuple[Dict[Tuple[int, int], PairResult], StageStatus]:
        require_cuda("pairwise_registration")
        require_modules("pairwise_registration", ["torch"])
        require_weights("pairwise_registration", PARE_WEIGHTS, self.cfg.models_dir)

        t0 = time.time()
        pair_results, per_pair_backend = self._register_neural(
            pcds_floor_up, voxel_m, min_edge_fitness,
        )
        status = StageStatus(
            stage="pairwise_registration",
            latency_s=time.time() - t0,
            extra={
                "n_pairs": len(pair_results),
                "n_pare_pairs": sum(1 for v in per_pair_backend.values() if v == "pare"),
                "n_geotr_pairs": sum(1 for v in per_pair_backend.values() if v == "geotr"),
                "mean_fitness": float(np.mean([pr.fitness for pr in pair_results.values()])),
                "mean_rmse": float(np.mean([pr.rmse for pr in pair_results.values()])),
            },
        )
        if self.cfg.log_backend_decisions:
            LOG.info(str(status))
        return pair_results, status

    def _register_neural(
        self,
        pcds_floor_up: List[o3d.geometry.PointCloud],
        voxel_m: float,
        min_edge_fitness: float,
    ) -> Tuple[Dict[Tuple[int, int], PairResult], Dict[Tuple[int, int], str]]:
        pare = self._get_pare()
        geotr = self._get_geotr() if weights_exist(GEOTR_WEIGHTS, self.cfg.models_dir) else None

        n = len(pcds_floor_up)
        pair_results: Dict[Tuple[int, int], PairResult] = {}
        per_pair_backend: Dict[Tuple[int, int], str] = {}

        for s in range(n):
            for t in range(s + 1, n):
                pr, used = self._register_neural_pair(
                    pare, geotr, pcds_floor_up[s], pcds_floor_up[t], voxel_m, min_edge_fitness,
                )
                pair_results[(s, t)] = pr
                per_pair_backend[(s, t)] = used
        return pair_results, per_pair_backend

    def _register_neural_pair(
        self,
        pare,
        geotr,
        src: o3d.geometry.PointCloud,
        tgt: o3d.geometry.PointCloud,
        voxel_m: float,
        min_edge_fitness: float,
    ) -> Tuple[PairResult, str]:
        used = "pare"
        T_global = pare.register(src, tgt, voxel_m=self.cfg.pare_voxel_size_mm * 1e-3)
        pivot = _bottom_aabb_center(tgt.voxel_down_sample(voxel_m))
        T_yaw = _yaw_only(np.asarray(T_global), pivot)
        res = _multi_stage_icp(src, tgt, voxel_m, T_yaw)
        pr = PairResult(
            T=np.asarray(res.transformation),
            fitness=float(res.fitness),
            rmse=float(res.inlier_rmse),
            yaw_margin=0.0,
            yaw_ambiguous=False,
            used_fpfh=False,
        )
        if pr.fitness >= min_edge_fitness:
            return pr, used
        if geotr is None:
            raise RuntimeError(
                f"Pair fitness {pr.fitness:.3f} below threshold {min_edge_fitness:.3f} "
                f"and GeoTransformer weights ({GEOTR_WEIGHTS}) are unavailable"
            )

        T_geo = geotr.register(src, tgt, voxel_m=voxel_m)
        T_yaw_g = _yaw_only(np.asarray(T_geo), pivot)
        res_g = _multi_stage_icp(src, tgt, voxel_m, T_yaw_g)
        pr_g = PairResult(
            T=np.asarray(res_g.transformation),
            fitness=float(res_g.fitness),
            rmse=float(res_g.inlier_rmse),
            yaw_margin=0.0,
            yaw_ambiguous=False,
            used_fpfh=False,
        )
        if pr_g.fitness > pr.fitness:
            return pr_g, "geotr"
        if pr_g.fitness < min_edge_fitness:
            raise RuntimeError(
                f"Pair registration failed: best fitness {max(pr.fitness, pr_g.fitness):.3f} "
                f"< {min_edge_fitness:.3f}"
            )
        return pr, used

    def _get_pare(self):
        if self._pare is not None:
            return self._pare
        from .pare_loader import load_parenet
        self._pare = load_parenet(
            weights=f"{self.cfg.models_dir}/{PARE_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._pare

    def _get_geotr(self):
        if self._geotr is not None:
            return self._geotr
        from .geotr_loader import load_geotransformer
        self._geotr = load_geotransformer(
            weights=f"{self.cfg.models_dir}/{GEOTR_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._geotr

    def refine_pair(
        self,
        src: o3d.geometry.PointCloud,
        tgt: o3d.geometry.PointCloud,
        voxel_m: float,
        T_init: np.ndarray,
    ) -> PairResult:
        """Refine a pair from a known pose using multi-stage point-to-plane ICP."""
        return _refine_pair_from_init(src, tgt, voxel_m, T_init)
