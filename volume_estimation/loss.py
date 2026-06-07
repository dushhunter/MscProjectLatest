"""Multi-task loss for StoneVolumeNet training with RPF flow loss.

Combines:
  - BCE segmentation loss (stone vs background)
  - Chamfer distance loss (registration quality)
  - L1 + MAPE volume loss (volume accuracy)
  - MSE flow velocity loss (RPF rectified flow registration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_distance_l2(
    pred: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """One-directional L2 Chamfer distance from pred to target.

    Args:
        pred: (M, 3) predicted point positions.
        target: (K, 3) ground-truth point positions.

    Returns:
        Scalar mean squared nearest-neighbor distance.
    """
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    dists = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    min_pred_to_tgt = dists.min(dim=1)[0].mean()
    min_tgt_to_pred = dists.min(dim=0)[0].mean()
    return (min_pred_to_tgt + min_tgt_to_pred) / 2.0


@dataclass
class LossWeights:
    """Relative weights for each loss term."""
    seg: float = 1.0
    volume: float = 0.1
    mape: float = 0.05
    flow: float = 1.0


class StoneVolumeLoss(nn.Module):
    """Multi-task loss for stone volume estimation with RPF flow loss.

    Components:
      1. BCE loss for per-point segmentation.
      2. L1 loss for absolute volume error.
      3. MAPE loss for relative volume error.
      4. MSE loss for flow velocity field (RPF rectified flow).

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
            output: model output dict with seg_logits, pred_volume,
                    and optionally v_pred, v_t (flow outputs).
            batch: data dict with seg_labels, gt_volume, pad_mask, n_points.

        Returns:
            dict with 'loss' (total) and individual loss terms.
        """
        losses = {}

        seg_loss = self._segmentation_loss(
            output["seg_logits"], batch["seg_labels"],
            batch["pad_mask"], batch["n_points"],
        )
        losses["seg_loss"] = seg_loss

        vol_l1 = F.l1_loss(output["pred_volume"], batch["gt_volume"])
        losses["vol_l1"] = vol_l1

        mape = self._mape_loss(output["pred_volume"], batch["gt_volume"])
        losses["vol_mape"] = mape

        total = (
            self.w.seg * seg_loss
            + self.w.volume * vol_l1
            + self.w.mape * mape
        )

        if "v_pred" in output and "v_t" in output:
            flow_loss = self._flow_velocity_loss(output["v_pred"], output["v_t"])
            losses["flow_loss"] = flow_loss
            total = total + self.w.flow * flow_loss

        losses["loss"] = total

        with torch.no_grad():
            losses["vol_mae"] = vol_l1.detach()
            losses["vol_mape_pct"] = (mape * 100.0).detach()

        return losses

    @staticmethod
    def _flow_velocity_loss(
        v_pred: torch.Tensor, v_target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on predicted vs target velocity field (RPF-style).

        Args:
            v_pred: (B, M, 3) predicted velocity from flow head.
            v_target: (B, M, 3) target velocity = x_1 - x_0.

        Returns:
            Scalar MSE loss.
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
        B, N = logits.shape
        valid = ~pad_mask
        valid_logits = logits[valid]
        valid_labels = labels[valid]

        if valid_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pos_weight = self._compute_pos_weight(valid_labels)
        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_labels,
            pos_weight=pos_weight,
        )
        return loss

    @staticmethod
    def _compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
        """Compute class weight to handle stone/background imbalance."""
        n_pos = labels.sum().clamp(min=1.0)
        n_neg = (labels.numel() - n_pos).clamp(min=1.0)
        weight = (n_neg / n_pos).clamp(0.5, 10.0)
        return weight

    @staticmethod
    def _mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Percentage Error, safe for near-zero targets."""
        denom = target.abs().clamp(min=1e-4)
        return ((pred - target).abs() / denom).mean()
