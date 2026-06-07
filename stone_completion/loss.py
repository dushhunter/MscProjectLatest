"""Loss functions for stone completion training.

Combines:
  1. BCE segmentation loss (stone vs floor)
  2. Multi-scale Chamfer Distance (coarse + fine1 + fine2 vs GT)

Chamfer Distance is implemented in pure PyTorch (no custom CUDA ops).
If pytorch3d is available, its optimized version is used instead.

Reference: PointSea (IJCV 2025) uses coarse-to-fine CD with FPS-matched GT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _farthest_point_sample, _index_points


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def _chamfer_distance_pure(
    pred: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    """Chamfer Distance (L2, mean) in pure PyTorch.

    Args:
        pred: (B, N, 3)
        gt: (B, M, 3)
    Returns:
        scalar loss
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist_matrix = (diff ** 2).sum(dim=-1)

    pred_to_gt = dist_matrix.min(dim=2)[0].mean(dim=1)
    gt_to_pred = dist_matrix.min(dim=1)[0].mean(dim=1)

    return (pred_to_gt + gt_to_pred).mean()


try:
    from pytorch3d.loss import chamfer_distance as _p3d_chamfer

    def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss, _ = _p3d_chamfer(pred, gt)
        return loss

except ImportError:
    chamfer_distance = _chamfer_distance_pure


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def _seg_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute segmentation quality metrics."""
    with torch.no_grad():
        preds = (logits > 0).float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    seg: float = 1.0
    coarse: float = 1.0
    fine1: float = 1.0
    fine2: float = 1.0


class StoneCompletionLoss(nn.Module):
    """Combined segmentation + multi-scale Chamfer Distance loss.

    For each completion stage, the GT cloud is FPS-downsampled to match
    the prediction resolution before computing CD.
    """

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        seg_labels: torch.Tensor,
        gt_complete: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: Model output dict with seg_logits, coarse, fine1, fine2.
            seg_labels: (B, N) ground truth segmentation.
            gt_complete: (B, M, 3) complete stone GT cloud.
        Returns:
            Dict with individual and total loss values + metrics.
        """
        losses: Dict[str, torch.Tensor] = {}

        seg_logits = output["seg_logits"]
        losses["seg_bce"] = F.binary_cross_entropy_with_logits(seg_logits, seg_labels)

        metrics = _seg_metrics(seg_logits, seg_labels)
        for k, v in metrics.items():
            losses[f"seg_{k}"] = v

        for stage_name in ("coarse", "fine1", "fine2"):
            pred = output[stage_name]
            n_pred = pred.shape[1]

            if gt_complete.shape[1] > n_pred:
                fps_idx = _farthest_point_sample(gt_complete, n_pred)
                gt_stage = _index_points(gt_complete, fps_idx)
            else:
                gt_stage = gt_complete

            losses[f"cd_{stage_name}"] = chamfer_distance(pred, gt_stage)

        total = (
            self.w.seg * losses["seg_bce"]
            + self.w.coarse * losses["cd_coarse"]
            + self.w.fine1 * losses["cd_fine1"]
            + self.w.fine2 * losses["cd_fine2"]
        )
        losses["loss"] = total

        return losses
