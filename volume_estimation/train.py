#!/usr/bin/env python3
"""PyTorch Lightning training for StoneReconNet.

Follows RPF's (Rectified Point Flow) training pattern:
  forward() -> loss() -> training_step() -> validation_step()

Training objectives (no direct volume regression):
  - BCE segmentation loss  (stone vs floor/background)
  - MSE flow velocity loss (RPF rectified flow registration)

Volume is NOT predicted by the network. At inference time, the registered
point cloud is turned into a watertight mesh via Poisson reconstruction,
and volume is computed geometrically from the mesh.

Supports:
  - Rectified flow velocity field learning alongside segmentation
  - Train/val split (stones 1-10 train, 11-12 val) or leave-one-out CV
  - OneCycleLR scheduler with AdamW
  - Mixed precision (fp16) for RTX 4080 16GB
  - tf32 + cudnn.benchmark CUDA optimizations (from RPF)
  - Frozen encoder support (from RPF)
  - wandb and/or TensorBoard logging
  - Automatic checkpointing on best validation loss
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
except ImportError:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

from volume_estimation.dataset import (
    StoneReconDataset,
    collate_variable_points,
)
from volume_estimation.loss import LossWeights, StoneReconLoss
from volume_estimation.model import StoneReconNet, StoneReconNetConfig

LOG = logging.getLogger("train_stone_recon")


def _enable_cuda_optimizations():
    """Enable tf32 and cudnn.benchmark for faster training (from RPF)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


class StoneReconLightning(pl.LightningModule):
    """Lightning wrapper following RPF's forward/loss/training_step pattern.

    Training objectives: segmentation BCE + flow velocity MSE.
    No volume regression -- volume is computed from the mesh at inference.
    """

    def __init__(
        self,
        model_cfg: StoneReconNetConfig,
        loss_weights: LossWeights,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        freeze_encoder_after: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = StoneReconNet(model_cfg)
        self.criterion = StoneReconLoss(loss_weights)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.freeze_encoder_after = freeze_encoder_after

    # ------------------------------------------------------------------
    # RPF-style forward / loss / step pattern
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def loss(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.criterion(output, batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self.forward(batch)
        losses = self.loss(output, batch)

        B = batch["points"].shape[0]
        prog_keys = {"loss", "seg_iou", "seg_f1"}
        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=(k in prog_keys), batch_size=B)

        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        output = self.forward(batch)
        losses = self.loss(output, batch)

        B = batch["points"].shape[0]
        prog_keys = {"loss", "seg_iou", "seg_f1"}
        for k, v in losses.items():
            self.log(f"val/{k}", v, prog_bar=(k in prog_keys),
                     batch_size=B, sync_dist=True)

    # ------------------------------------------------------------------
    # RPF-style hooks for frozen encoder
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_encoder_after >= 0
            and self.current_epoch >= self.freeze_encoder_after
        ):
            self.model.freeze_encoder()

    def on_validation_epoch_start(self) -> None:
        if (
            self.freeze_encoder_after >= 0
            and self.current_epoch >= self.freeze_encoder_after
        ):
            self.model.freeze_encoder()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_datasets(
    dataset_dir: str,
    volumes_json: Optional[str],
    intrinsics_path: str,
    train_stones: List[str],
    val_stones: List[str],
    width: int = 1024,
    height: int = 576,
    max_points_per_view: int = 4096,
    train_samples_per_epoch: int = 500,
    val_samples_per_epoch: int = 100,
    random_views_suffix: str = "_random_npy",
) -> tuple:
    train_ds = StoneReconDataset(
        dataset_dir=dataset_dir,
        volumes_json=volumes_json,
        intrinsics_path=intrinsics_path,
        stone_ids=train_stones,
        width=width,
        height=height,
        max_points_per_view=max_points_per_view,
        augment=True,
        samples_per_epoch=train_samples_per_epoch,
        random_views_suffix=random_views_suffix,
    )
    val_ds = StoneReconDataset(
        dataset_dir=dataset_dir,
        volumes_json=volumes_json,
        intrinsics_path=intrinsics_path,
        stone_ids=val_stones,
        width=width,
        height=height,
        max_points_per_view=max_points_per_view,
        augment=False,
        samples_per_epoch=val_samples_per_epoch,
        random_views_suffix=random_views_suffix,
    )
    return train_ds, val_ds


def get_stone_split(
    volumes_json: str,
    val_stones: Optional[List[str]] = None,
) -> tuple:
    """Determine train/val split from available stones."""
    with open(volumes_json) as f:
        data = json.load(f)
    all_stones = sorted(data.keys())

    if val_stones is None:
        val_stones = [s for s in all_stones if s in ("stone_11", "stone_12")]
        if not val_stones:
            val_stones = all_stones[-2:] if len(all_stones) > 2 else all_stones[-1:]

    train_stones = [s for s in all_stones if s not in val_stones]
    return train_stones, val_stones


def train(
    dataset_dir: str,
    volumes_json: str,
    intrinsics_path: str,
    output_dir: str,
    val_stones: Optional[List[str]] = None,
    max_epochs: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_points_per_view: int = 4096,
    train_samples_per_epoch: int = 500,
    val_samples_per_epoch: int = 100,
    num_workers: int = 4,
    precision: str = "16-mixed",
    use_wandb: bool = False,
    wandb_project: str = "stone-recon",
    loss_w_seg: float = 1.0,
    loss_w_flow: float = 1.0,
    patience: int = 30,
    width: int = 1024,
    height: int = 576,
    freeze_encoder_after: int = -1,
    random_views_suffix: str = "_random_npy",
):
    """Run the full training pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    _enable_cuda_optimizations()

    train_stones, val_stones_final = get_stone_split(volumes_json, val_stones)
    LOG.info("Train stones: %s", train_stones)
    LOG.info("Val stones:   %s", val_stones_final)

    train_ds, val_ds = build_datasets(
        dataset_dir=dataset_dir,
        volumes_json=volumes_json,
        intrinsics_path=intrinsics_path,
        train_stones=train_stones,
        val_stones=val_stones_final,
        width=width,
        height=height,
        max_points_per_view=max_points_per_view,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=val_samples_per_epoch,
        random_views_suffix=random_views_suffix,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_variable_points, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variable_points, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )

    model_cfg = StoneReconNetConfig()
    loss_weights = LossWeights(seg=loss_w_seg, flow=loss_w_flow)

    lit_model = StoneReconLightning(
        model_cfg=model_cfg,
        loss_weights=loss_weights,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        freeze_encoder_after=freeze_encoder_after,
    )

    param_counts = lit_model.model.count_parameters()
    LOG.info("Model parameters: %s", param_counts)
    LOG.info("Loss weights -- seg: %.3f, flow: %.3f", loss_w_seg, loss_w_flow)
    if freeze_encoder_after >= 0:
        LOG.info("Encoder freezes after epoch %d", freeze_encoder_after)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    loggers = []
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            from lightning.pytorch.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=wandb_project,
            name=f"stone_recon_{len(train_stones)}train",
            save_dir=output_dir,
        ))
    if not loggers:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError:
            from lightning.pytorch.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger(
            save_dir=output_dir, name="tb_logs",
        ))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=output_dir,
        gradient_clip_val=0.5,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = callbacks[0].best_model_path
    LOG.info("Best checkpoint: %s", best_ckpt)

    final_path = os.path.join(output_dir, "stone_recon_net.pt")
    if best_ckpt and os.path.exists(best_ckpt):
        best = StoneReconLightning.load_from_checkpoint(best_ckpt)
        torch.save(best.model.state_dict(), final_path)
        LOG.info("Saved final model weights: %s", final_path)

    summary = {
        "train_stones": train_stones,
        "val_stones": val_stones_final,
        "best_checkpoint": best_ckpt,
        "model_params": param_counts,
        "max_epochs": max_epochs,
        "lr": lr,
        "loss_w_seg": loss_w_seg,
        "loss_w_flow": loss_w_flow,
        "freeze_encoder_after": freeze_encoder_after,
        "random_views_suffix": random_views_suffix,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train StoneReconNet")
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of stone_syn_dataset/")
    parser.add_argument("--volumes_json", required=True,
                        help="Path to stone_volumes_gt.json")
    parser.add_argument("--intrinsics", required=True,
                        help="Path to intrinsics.txt")
    parser.add_argument("--output_dir", default="volume_training_output",
                        help="Output directory for checkpoints and logs")

    parser.add_argument("--val_stones", nargs="*", default=None,
                        help="Stones for validation (default: stone_11 stone_12)")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_points_per_view", type=int, default=4096)
    parser.add_argument("--train_samples_per_epoch", type=int, default=500)
    parser.add_argument("--val_samples_per_epoch", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", default="16-mixed",
                        choices=["16-mixed", "32", "bf16-mixed"])
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)

    parser.add_argument("--loss_w_seg", type=float, default=1.0,
                        help="Weight for segmentation BCE loss")
    parser.add_argument("--loss_w_flow", type=float, default=1.0,
                        help="Weight for RPF flow velocity MSE loss")

    parser.add_argument("--freeze_encoder_after", type=int, default=-1,
                        help="Freeze PointNet++ encoder after this epoch (-1 = never)")
    parser.add_argument("--random_views_suffix", default="_random_npy",
                        help="Suffix for random-views directories (default: _random_npy). "
                             "Set to empty string to disable random views.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="stone-recon")

    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        volumes_json=args.volumes_json,
        intrinsics_path=args.intrinsics,
        output_dir=args.output_dir,
        val_stones=args.val_stones,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_points_per_view=args.max_points_per_view,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch,
        num_workers=args.num_workers,
        precision=args.precision,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        loss_w_seg=args.loss_w_seg,
        loss_w_flow=args.loss_w_flow,
        patience=args.patience,
        width=args.width,
        height=args.height,
        freeze_encoder_after=args.freeze_encoder_after,
        random_views_suffix=args.random_views_suffix,
    )


if __name__ == "__main__":
    main()
