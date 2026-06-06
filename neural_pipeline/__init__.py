"""Neural-only pipeline for depth-only stone reconstruction.

Every stage runs a learned model (PointTransformerV3, PARE-Net, RAP,
NKSR / NoKSR). CUDA, upstream packages, and weights under ``models/`` are
required. Use :mod:`reconstruct_stone_3d_sparse` for the classical baseline.
"""

from .config import NeuralConfig
from .segmentation import StoneSegmenter
from .registration_pair import PairwiseRegistrar, PairResult
from .registration_multi import MultiViewRegistrar
from .surface import SurfaceReconstructor
from .report import write_neural_report

__all__ = [
    "NeuralConfig",
    "StoneSegmenter",
    "PairwiseRegistrar",
    "PairResult",
    "MultiViewRegistrar",
    "SurfaceReconstructor",
    "write_neural_report",
]
