"""PARE-Net pairwise registration loader."""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("stone3d_neural.pare")


def load_parenet(weights: str, device: str = "cuda") -> "ParenetCallable":
    try:
        import torch
        from parenet.pipeline import PARENetPipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PARE-Net import failed: {e}") from e

    try:
        pipeline = PARENetPipeline.from_pretrained(weights, device=device)
        pipeline.eval()
    except Exception as e:
        raise RuntimeError(f"PARE-Net weights load failed ({weights}): {e}") from e

    LOG.info("Loaded PARE-Net from %s on %s", weights, device)
    return ParenetCallable(pipeline, device)


class ParenetCallable:
    def __init__(self, pipeline, device: str) -> None:
        self.pipeline = pipeline
        self.device = device

    def register(self, src_pcd, tgt_pcd, voxel_m: float) -> np.ndarray:
        import torch

        src = src_pcd.voxel_down_sample(voxel_m)
        tgt = tgt_pcd.voxel_down_sample(voxel_m)
        src_pts = torch.from_numpy(np.asarray(src.points, dtype=np.float32)).to(self.device)
        tgt_pts = torch.from_numpy(np.asarray(tgt.points, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            T = self.pipeline.register(src_pts, tgt_pts, voxel_size=voxel_m)
        T = T.detach().cpu().numpy().astype(np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"PARE-Net returned unexpected shape: {T.shape}")
        return T
