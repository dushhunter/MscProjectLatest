"""Multi-view alignment stage (SGHR only)."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from .config import (
    NeuralConfig,
    StageStatus,
    require_cuda,
    require_modules,
    require_weights,
)
from .registration_pair import PairResult, PairwiseRegistrar


LOG = logging.getLogger("stone3d_neural.multi")

SGHR_WEIGHTS = "sghr_3dmatch.pth"


class MultiViewRegistrar:
    def __init__(self, cfg: NeuralConfig, pair_registrar: PairwiseRegistrar) -> None:
        self.cfg = cfg
        self.pair_registrar = pair_registrar
        self._sghr = None

    def solve(
        self,
        pcds_floor_up: List[o3d.geometry.PointCloud],
        pair_results: Dict[Tuple[int, int], PairResult],
        voxel_m: float,
        max_corr_m: float,
        min_edge_fitness: float,
        refinement_iters: int = 2,
    ) -> Tuple[List[np.ndarray], dict, StageStatus]:
        require_cuda("multiview_registration")
        require_modules("multiview_registration", ["torch"])
        require_weights("multiview_registration", SGHR_WEIGHTS, self.cfg.models_dir)

        t0 = time.time()
        poses, summary = self._solve_neural(
            pcds_floor_up, pair_results, voxel_m,
            min_edge_fitness, refinement_iters,
        )
        status = StageStatus(
            stage="multiview_registration",
            latency_s=time.time() - t0,
            extra={"n_frames": len(poses), **summary},
        )
        if self.cfg.log_backend_decisions:
            LOG.info(str(status))
        return poses, summary, status

    def _solve_neural(
        self,
        pcds_floor_up: List[o3d.geometry.PointCloud],
        pair_results: Dict[Tuple[int, int], PairResult],
        voxel_m: float,
        min_edge_fitness: float,
        refinement_iters: int,
    ) -> Tuple[List[np.ndarray], dict]:
        sghr = self._get_sghr()
        n = len(pcds_floor_up)

        pair_list = []
        for (s, t), pr in pair_results.items():
            if pr.fitness < min_edge_fitness:
                continue
            pair_list.append((s, t, pr.T, pr.fitness))

        if not pair_list:
            raise RuntimeError("No pair edges above min_edge_fitness for SGHR")

        T_world = sghr.solve(
            pcds_floor_up,
            edges=pair_list,
            voxel_m=self.cfg.sghr_voxel_size_mm * 1e-3,
        )

        summary = {
            "method": "SGHR neural overlap + history-IRLS",
            "edges_kept_per_iter": [len(pair_list)],
            "tree_edges": [],
            "iso_frames": [],
            "irls_residual": float(getattr(sghr, "last_residual", float("nan"))),
            "n_pairs_total": n * (n - 1) // 2,
            "ambiguous": sum(1 for pr in pair_results.values() if pr.yaw_ambiguous),
        }

        for _ in range(refinement_iters):
            for (s, t) in list(pair_results.keys()):
                T_init = np.linalg.inv(T_world[t]) @ T_world[s]
                pr_new = self.pair_registrar.refine_pair(
                    pcds_floor_up[s], pcds_floor_up[t], voxel_m, T_init,
                )
                if pr_new.fitness >= pair_results[(s, t)].fitness:
                    pair_results[(s, t)] = pr_new

            pair_list = [
                (s, t, pr.T, pr.fitness)
                for (s, t), pr in pair_results.items()
                if pr.fitness >= min_edge_fitness
            ]
            if not pair_list:
                raise RuntimeError("All pair edges dropped below min_edge_fitness during refinement")
            T_world = sghr.solve(
                pcds_floor_up,
                edges=pair_list,
                voxel_m=self.cfg.sghr_voxel_size_mm * 1e-3,
                warm_start=T_world,
            )
            summary["edges_kept_per_iter"].append(len(pair_list))

        return T_world, summary

    def _get_sghr(self):
        if self._sghr is not None:
            return self._sghr
        from .sghr_loader import load_sghr
        self._sghr = load_sghr(
            weights=f"{self.cfg.models_dir}/{SGHR_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._sghr
