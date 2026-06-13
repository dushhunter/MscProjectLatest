"""Loss functions for StoneReconNet training.

Three loss terms:
  - BCE segmentation loss (stone vs floor/background)
  - MSE flow velocity loss (RPF rectified flow)
  - Chamfer distance loss (upsampled flow output vs GT2 cloud)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Relative weights for each loss term."""
    seg: float = 1.0
    flow: float = 1.0
    chamfer: float = 0.5


def _chamfer_distance(
    pred: torch.Tensor, gt: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Chamfer distance between two point clouds.

    Args:
        pred: (B, M, 3) predicted points.
        gt: (B, K, 3) ground-truth points.

    Returns:
        Scalar mean Chamfer distance across the batch.
    """
    dists = torch.cdist(pred, gt)
    min_pred_to_gt = dists.min(dim=2).values.mean(dim=1)
    min_gt_to_pred = dists.min(dim=1).values.mean(dim=1)
    return (min_pred_to_gt + min_gt_to_pred).mean()


class StoneReconLoss(nn.Module):
    """Segmentation + flow + Chamfer loss for StoneReconNet.

    Components:
      1. BCE loss for per-point stone vs floor segmentation.
      2. MSE loss for flow velocity field (RPF rectified flow).
      3. Chamfer distance between upsampled flow output and GT2 cloud.
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        device = batch["points"].device
        total = torch.tensor(0.0, device=device, requires_grad=True)

        if "seg_logits" in output and self.w.seg > 0:
            seg_loss = self._segmentation_loss(
                output["seg_logits"], batch["seg_labels"],
                batch["pad_mask"], batch["n_points"],
            )
            losses["seg_loss"] = seg_loss
            total = total + self.w.seg * seg_loss

        if "v_pred" in output and "v_t" in output:
            flow_loss = self._flow_velocity_loss(output["v_pred"], output["v_t"])
            losses["flow_loss"] = flow_loss
            total = total + self.w.flow * flow_loss

        if "upsampled_points" in output and "gt_cloud" in output:
            chamfer_loss = _chamfer_distance(
                output["upsampled_points"], output["gt_cloud"],
            )
            losses["chamfer_loss"] = chamfer_loss
            total = total + self.w.chamfer * chamfer_loss

        losses["loss"] = total

        return losses

    @staticmethod
    def _flow_velocity_loss(
        v_pred: torch.Tensor, v_target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on predicted vs target velocity field (RAP Eq.6).

        Raw MSE without center-subtraction: the model must learn the full
        velocity including the mean translation component.  Our inputs and
        GT are already zero-centered, so the mean velocity is near zero and
        the gradient signal is well-conditioned.
        """
        return F.mse_loss(v_pred, v_target)

    def _segmentation_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pad_mask: torch.Tensor,
        n_points: torch.Tensor,
    ) -> torch.Tensor:
        """Masked BCE loss for segmentation."""
        valid = ~pad_mask
        valid_logits = logits[valid]
        valid_labels = labels[valid]

        if valid_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pos_weight = self._compute_pos_weight(valid_labels)
        return F.binary_cross_entropy_with_logits(
            valid_logits, valid_labels, pos_weight=pos_weight,
        )

    @staticmethod
    def _compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
        """Compute class weight to handle stone/background imbalance."""
        n_pos = labels.sum().clamp(min=1.0)
        n_neg = (labels.numel() - n_pos).clamp(min=1.0)
        return (n_neg / n_pos).clamp(0.5, 10.0)
