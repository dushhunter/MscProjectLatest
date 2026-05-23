"""GeoTransformer pairwise registration loader (ML fallback when PARE-Net fails a pair)."""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("stone3d_neural.geotr")


def load_geotransformer(weights: str, device: str = "cuda") -> "GeotrCallable":
    try:
        import torch
        from geotransformer.experiments.geotransformer_3dmatch import (  # type: ignore
            create_model,
        )
        from geotransformer.utils.data import registration_collate_fn_stack_mode  # type: ignore
    except Exception as e:
        raise RuntimeError(f"GeoTransformer import failed: {e}") from e

    try:
        model = create_model().to(device)
        state = torch.load(weights, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"GeoTransformer weights load failed ({weights}): {e}") from e

    LOG.info("Loaded GeoTransformer from %s on %s", weights, device)
    return GeotrCallable(model, registration_collate_fn_stack_mode, device)


class GeotrCallable:
    def __init__(self, model, collate_fn, device: str) -> None:
        self.model = model
        self.collate = collate_fn
        self.device = device

    def register(self, src_pcd, tgt_pcd, voxel_m: float) -> np.ndarray:
        import torch
        src = src_pcd.voxel_down_sample(voxel_m)
        tgt = tgt_pcd.voxel_down_sample(voxel_m)
        data_dict = {
            "ref_points": torch.from_numpy(np.asarray(tgt.points, dtype=np.float32)),
            "src_points": torch.from_numpy(np.asarray(src.points, dtype=np.float32)),
            "ref_feats": torch.ones((len(tgt.points), 1)),
            "src_feats": torch.ones((len(src.points), 1)),
        }
        batch = self.collate([data_dict], 4, voxel_m, voxel_m * 2.5)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        with torch.inference_mode():
            out = self.model(batch)
        T = out["estimated_transform"].detach().cpu().numpy().astype(np.float64)
        return T
