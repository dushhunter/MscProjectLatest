"""Configuration for the neural-only stone reconstruction pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


LOG = logging.getLogger("stone3d_neural.config")


@dataclass
class NeuralConfig:
    """Configuration for the pure-ML reconstruction pipeline.

    Every stage requires CUDA, the upstream Python packages, and pretrained
    weights under ``models_dir``. There is no classical fallback path.
    """

    surface_model: str = "nksr"  # "nksr" or "noksr"

    models_dir: str = "models"
    device: str = "cuda"  # "cuda" or explicit torch device string
    log_backend_decisions: bool = True

    # Stage-specific neural knobs.
    ptv3_grid_size_mm: float = 1.0
    nksr_voxel_size_mm: float = 0.5
    pare_voxel_size_mm: float = 0.6

    # RAP multi-view registration (replaces SGHR + MinkowskiEngine).
    rap_dir: str = "/tmp/RAP"
    rap_voxel_size_mm: float = 0.6
    rap_sampling_steps: int = 10
    rap_rigidity_forcing: bool = True
    rap_max_points_per_part: int = 500

    def __post_init__(self) -> None:
        if self.surface_model not in ("nksr", "noksr"):
            raise ValueError(
                f"surface_model must be 'nksr' or 'noksr'; got {self.surface_model!r}"
            )

    def torch_device(self) -> str:
        if self.device != "cuda":
            return self.device
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required but torch.cuda.is_available() is False"
                )
            return "cuda"
        except ImportError as e:
            raise RuntimeError("torch is required for the neural pipeline") from e


@dataclass
class StageStatus:
    """What ran for a given pipeline stage."""

    stage: str
    backend_used: str = "neural"
    latency_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        msg = f"[{self.stage}] backend={self.backend_used} took={self.latency_s:.2f}s"
        return msg


def require_cuda(stage: str) -> None:
    if not have_torch_cuda():
        raise RuntimeError(f"{stage}: CUDA GPU is required")


def require_modules(stage: str, deps: List[str]) -> None:
    missing = [d for d in deps if not have_module(d)]
    if missing:
        raise RuntimeError(f"{stage}: missing Python packages: {missing}")


def require_weights(stage: str, weight_names: Union[str, List[str]], models_dir: str) -> None:
    names = [weight_names] if isinstance(weight_names, str) else list(weight_names)
    missing = [w for w in names if not weights_exist(w, models_dir)]
    if missing:
        raise RuntimeError(f"{stage}: weights not found under {models_dir}: {missing}")


def have_torch_cuda() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def have_module(name: str) -> bool:
    """Cheap probe for an optional dependency without importing it."""
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def weights_exist(path: str, models_dir: str) -> bool:
    if os.path.isabs(path):
        return os.path.exists(path)
    return os.path.exists(os.path.join(models_dir, path))
