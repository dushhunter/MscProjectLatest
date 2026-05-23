"""PointTransformerV3 binary stone-vs-floor segmenter loader."""

from __future__ import annotations

import logging
from typing import Callable

LOG = logging.getLogger("stone3d_neural.ptv3")


def load_ptv3_binary_segmenter(weights: str, device: str = "cuda") -> Callable:
    try:
        import torch
        from ptv3.model import PointTransformerV3  # type: ignore
        from ptv3.serialization import encode as ptv3_encode  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PointTransformerV3 import failed: {e}") from e

    model = PointTransformerV3(
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        num_classes=1,
    ).to(device)
    state = torch.load(weights, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    LOG.info("Loaded PTv3 binary segmenter from %s on %s", weights, device)
    return _PTv3Callable(model, ptv3_encode, device)


class _PTv3Callable:
    def __init__(self, model, encode_fn, device: str) -> None:
        self.model = model
        self.encode = encode_fn
        self.device = device

    def __call__(self, points, grid_size_m: float):
        import torch
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"PTv3 expects (N, 3) point tensor; got {tuple(points.shape)}")
        data = self.encode(points, grid_size=grid_size_m)
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.model(data)
        if hasattr(logits, "feat"):
            logits = logits.feat.squeeze(-1)
        return logits
