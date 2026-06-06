"""Multi-view alignment stage (RAP flow-matching registration).

Replaces the previous SGHR + MinkowskiEngine backend with RAP
(Register Any Point), a single-stage flow-matching transformer that
directly generates registered point clouds without pairwise pose
graph synchronisation or sparse convolution dependencies.
"""

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

RAP_WEIGHTS = "rap_model_10.ckpt"


class MultiViewRegistrar:
    def __init__(self, cfg: NeuralConfig, pair_registrar: PairwiseRegistrar) -> None:
        self.cfg = cfg
        self.pair_registrar = pair_registrar
        self._rap = None

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
        require_weights("multiview_registration", RAP_WEIGHTS, self.cfg.models_dir)

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
        rap = self._get_rap()
        n = len(pcds_floor_up)

        T_world = rap.solve(
            pcds_floor_up,
            voxel_m=self.cfg.rap_voxel_size_mm * 1e-3,
            max_points_per_part=self.cfg.rap_max_points_per_part,
        )

        summary = {
            "method": "RAP flow-matching + Procrustes",
            "edges_kept_per_iter": [len(pair_results)],
            "tree_edges": [],
            "iso_frames": [],
            "rap_chamfer_d": float(getattr(rap, "last_chamfer_d", float("nan"))),
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

            T_world = rap.solve(
                pcds_floor_up,
                voxel_m=self.cfg.rap_voxel_size_mm * 1e-3,
                max_points_per_part=self.cfg.rap_max_points_per_part,
            )
            summary["edges_kept_per_iter"].append(len(pair_list))

        return T_world, summary

    def _get_rap(self):
        if self._rap is not None:
            return self._rap
        from .rap_loader import load_rap
        self._rap = load_rap(
            rap_dir=self.cfg.rap_dir,
            weights=f"{self.cfg.models_dir}/{RAP_WEIGHTS}",
            device=self.cfg.torch_device(),
            sampling_steps=self.cfg.rap_sampling_steps,
            rigidity_forcing=self.cfg.rap_rigidity_forcing,
        )
        return self._rap
