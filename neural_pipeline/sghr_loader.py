"""SGHR multi-view registration loader."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

LOG = logging.getLogger("stone3d_neural.sghr")


def load_sghr(weights: str, device: str = "cuda") -> "SGHRCallable":
    try:
        import torch
        from sghr.solver import SGHRSolver  # type: ignore
    except Exception as e:
        raise RuntimeError(f"SGHR import failed: {e}") from e

    try:
        solver = SGHRSolver.from_pretrained(weights, device=device)
        solver.eval()
    except Exception as e:
        raise RuntimeError(f"SGHR weights load failed ({weights}): {e}") from e

    LOG.info("Loaded SGHR from %s on %s", weights, device)
    return SGHRCallable(solver, device)


class SGHRCallable:
    def __init__(self, solver, device: str) -> None:
        self.solver = solver
        self.device = device
        self.last_residual = float("nan")

    def solve(self, pcds, edges, voxel_m: float, warm_start=None) -> List[np.ndarray]:
        import torch
        pcd_tensors = [
            torch.from_numpy(np.asarray(p.voxel_down_sample(voxel_m).points, dtype=np.float32))
            .to(self.device)
            for p in pcds
        ]
        edge_idx = torch.tensor([(s, t) for (s, t, _, _) in edges], device=self.device)
        edge_T = torch.from_numpy(
            np.stack([np.asarray(T, dtype=np.float32) for (_, _, T, _) in edges])
        ).to(self.device)
        edge_fit = torch.tensor(
            [float(f) for (_, _, _, f) in edges], device=self.device
        )
        warm = None
        if warm_start is not None:
            warm = torch.from_numpy(
                np.stack([np.asarray(T, dtype=np.float32) for T in warm_start])
            ).to(self.device)

        with torch.inference_mode():
            T_world, residual = self.solver.solve(
                pcd_tensors, edge_idx, edge_T, edge_fit, warm_start=warm,
            )
        self.last_residual = float(residual.detach().cpu())
        return [T_world[i].detach().cpu().numpy().astype(np.float64) for i in range(len(pcds))]
