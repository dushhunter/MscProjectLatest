"""Loss functions for StoneReconNet training.

Two loss terms following RPF's training pattern:
  - BCE segmentation loss (stone vs floor/background)
  - MSE flow velocity loss (RPF rectified flow registration)
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


class StoneReconLoss(nn.Module):
    """Segmentation + flow registration loss for StoneReconNet.

    Components:
      1. BCE loss for per-point stone vs floor/background segmentation.
      2. MSE loss for flow velocity field (RPF rectified flow).

    The flow velocity MSE follows RPF's loss() pattern: it supervises the
    predicted velocity v_pred against the rectified flow target v_t = x_1 - x_0.
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: model output dict with seg_logits, and optionally v_pred, v_t.
            batch: data dict with seg_labels, pad_mask, n_points.

        Returns:
            dict with 'loss' (total) and individual loss terms.
        """
        losses = {}

        seg_loss = self._segmentation_loss(
            output["seg_logits"], batch["seg_labels"],
            batch["pad_mask"], batch["n_points"],
        )
        losses["seg_loss"] = seg_loss

        total = self.w.seg * seg_loss

        if "v_pred" in output and "v_t" in output:
            flow_loss = self._flow_velocity_loss(output["v_pred"], output["v_t"])
            losses["flow_loss"] = flow_loss
            total = total + self.w.flow * flow_loss

        losses["loss"] = total

        with torch.no_grad():
            seg_probs = torch.sigmoid(output["seg_logits"])
            valid = ~batch["pad_mask"]
            pred_seg = (seg_probs > 0.5).float()
            gt_seg = batch["seg_labels"]

            pred_valid = pred_seg[valid]
            gt_valid = gt_seg[valid]
            n_valid = valid.sum().clamp(min=1)

            correct = (pred_valid == gt_valid).sum()
            losses["seg_acc"] = (correct.float() / n_valid.float()).detach()

            tp = ((pred_valid == 1) & (gt_valid == 1)).sum().float()
            fp = ((pred_valid == 1) & (gt_valid == 0)).sum().float()
            fn = ((pred_valid == 0) & (gt_valid == 1)).sum().float()
            tn = ((pred_valid == 0) & (gt_valid == 0)).sum().float()

            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
            iou = tp / (tp + fp + fn).clamp(min=1)

            losses["seg_precision"] = precision.detach()
            losses["seg_recall"] = recall.detach()
            losses["seg_f1"] = f1.detach()
            losses["seg_iou"] = iou.detach()

            stone_ratio = gt_valid.sum().float() / n_valid.float()
            losses["seg_stone_ratio"] = stone_ratio.detach()

        return losses

    @staticmethod
    def _flow_velocity_loss(
        v_pred: torch.Tensor, v_target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on predicted vs target velocity field (RPF-style)."""
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
