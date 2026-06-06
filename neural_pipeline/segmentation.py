"""Stone segmentation stage (PointTransformerV3 only)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import (
    NeuralConfig,
    StageStatus,
    require_cuda,
    require_modules,
    require_weights,
)

from .geometry import Frame, Intrinsics, FloorFit, _backproject_full, _fit_floor_plane

LOG = logging.getLogger("stone3d_neural.seg")


PTV3_WEIGHTS = "ptv3_stone_binary.pth"


@dataclass
class SegResult:
    masks: List[np.ndarray]
    floors: List[FloorFit]
    status: StageStatus


class StoneSegmenter:
    def __init__(self, cfg: NeuralConfig) -> None:
        self.cfg = cfg
        self._ptv3 = None

    def segment(self, frames: List[Frame], K: Intrinsics) -> SegResult:
        require_cuda("segmentation")
        require_modules("segmentation", ["torch", "spconv"])
        require_weights("segmentation", PTV3_WEIGHTS, self.cfg.models_dir)

        t0 = time.time()
        masks, floors = self._segment_neural(frames, K)
        status = StageStatus(
            stage="segmentation",
            latency_s=time.time() - t0,
            extra={
                "n_frames": len(frames),
                "mask_pixel_counts": [int(m.sum()) for m in masks],
            },
        )
        if self.cfg.log_backend_decisions:
            LOG.info(str(status))
        return SegResult(masks=masks, floors=floors, status=status)

    def _segment_neural(
        self, frames: List[Frame], K: Intrinsics
    ) -> Tuple[List[np.ndarray], List[FloorFit]]:
        import torch

        ptv3 = self._get_ptv3()
        device = torch.device(self.cfg.torch_device())

        masks: List[np.ndarray] = []
        floors: List[FloorFit] = []
        for i, f in enumerate(frames):
            pts, flat_idx = _backproject_full(f.depth, K, stride=1)
            if pts.size == 0:
                raise RuntimeError(f"Frame {i}: depth back-projection produced no points")

            with torch.inference_mode():
                logits = ptv3(
                    torch.from_numpy(pts.astype(np.float32)).to(device),
                    grid_size_m=self.cfg.ptv3_grid_size_mm * 1e-3,
                )
                stone_prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            stone = stone_prob > 0.5

            H, W = f.depth.shape
            mask = np.zeros(H * W, dtype=bool)
            mask[flat_idx[stone]] = True
            mask = mask.reshape(H, W)
            if not mask.any():
                raise RuntimeError(f"Frame {i}: PTv3 segmentation produced an empty stone mask")
            masks.append(mask)

            floor_pts = pts[~stone]
            if floor_pts.shape[0] < 100:
                raise RuntimeError(
                    f"Frame {i}: PTv3 returned fewer than 100 floor points; "
                    "cannot fit a stable floor plane"
                )
            fit, _ = _fit_floor_plane(floor_pts)
            floors.append(fit)
        return masks, floors

    def _get_ptv3(self):
        if self._ptv3 is not None:
            return self._ptv3
        from .ptv3_loader import load_ptv3_binary_segmenter
        self._ptv3 = load_ptv3_binary_segmenter(
            weights=f"{self.cfg.models_dir}/{PTV3_WEIGHTS}",
            device=self.cfg.torch_device(),
        )
        return self._ptv3
