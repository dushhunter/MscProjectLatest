"""PyTorch Lightning training for StoneCompletionNet.

Follows PointSea's coarse-to-fine training with Chamfer Distance loss,
combined with stone/floor segmentation.

Training objectives:
  - BCE segmentation loss (stone vs floor)
  - Multi-scale Chamfer Distance (coarse, fine1, fine2 vs GT complete cloud)

CUDA optimizations adopted from RPF: tf32, cudnn.benchmark.

Prerequisite -- generate GT complete clouds from depth .npy:
    python -m stone_completion.prepare_gt \
        --dataset_dir stone_syn_dataset \
        --intrinsics splits/stone/intrinsics.txt \
        --output_dir stone_syn_dataset/gt_complete

Usage:
    python -m stone_completion.train \
        --dataset_dir stone_syn_dataset \
        --gt_complete_dir stone_syn_dataset/gt_complete \
        --intrinsics splits/stone/intrinsics.txt \
        --train_stones stone_01 stone_02 stone_03 stone_04 stone_05 \
                       stone_06 stone_07 stone_08 stone_09 stone_10 \
        --val_stones stone_11 stone_12 \
        --max_epochs 200 --batch_size 4 --precision 16-mixed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
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

from stone_completion.dataset import StoneCompletionDataset
from stone_completion.loss import LossWeights, StoneCompletionLoss
from stone_completion.model import CompletionConfig, StoneCompletionNet

LOG = logging.getLogger("train_stone_completion")


def _enable_cuda_optimizations():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


class StoneCompletionLightning(pl.LightningModule):
    """Lightning wrapper for StoneCompletionNet training."""

    def __init__(
        self,
        cfg: CompletionConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        seg_weight: float = 1.0,
        coarse_weight: float = 1.0,
        fine1_weight: float = 1.0,
        fine2_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = StoneCompletionNet(cfg)
        self.loss_fn = StoneCompletionLoss(
            LossWeights(seg_weight, coarse_weight, fine1_weight, fine2_weight)
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch["scene_points"], batch["partial_stone"])

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        output = self.forward(batch)
        losses = self.loss_fn(output, batch["seg_labels"], batch["gt_complete"])

        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, prog_bar=(k == "loss"), sync_dist=True)

        return losses["loss"]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train StoneCompletionNet")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--gt_complete_dir", required=True,
                    help="Dir with stone_XX_gt_complete.ply files")
    p.add_argument("--intrinsics", required=True,
                    help="Path to intrinsics.txt")
    p.add_argument("--train_stones", nargs="+", required=True)
    p.add_argument("--val_stones", nargs="+", required=True)
    p.add_argument("--output_dir", default="stone_completion_output")

    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)

    p.add_argument("--input_points", type=int, default=4096,
                    help="Number of scene points per sample (seg input)")
    p.add_argument("--completion_points", type=int, default=2048,
                    help="Number of partial stone points (completion input)")
    p.add_argument("--gt_points", type=int, default=8192,
                    help="Number of GT complete cloud points")
    p.add_argument("--coarse_points", type=int, default=512)

    p.add_argument("--min_views", type=int, default=4)
    p.add_argument("--max_views", type=int, default=18)
    p.add_argument("--max_points_per_view", type=int, default=4096)

    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--precision", default="16-mixed",
                    help="Training precision: 32, 16-mixed, bf16-mixed")

    p.add_argument("--seg_weight", type=float, default=1.0)
    p.add_argument("--coarse_weight", type=float, default=1.0)
    p.add_argument("--fine1_weight", type=float, default=1.0)
    p.add_argument("--fine2_weight", type=float, default=1.0)

    p.add_argument("--samples_per_epoch", type=int, default=500)
    p.add_argument("--val_samples", type=int, default=100)

    p.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience (epochs)")

    p.add_argument("--resume_ckpt", type=str, default=None,
                    help="Path to checkpoint to resume from")

    return p


def main():
    _enable_cuda_optimizations()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(42, workers=True)

    cfg = CompletionConfig(
        input_points=args.input_points,
        coarse_points=args.coarse_points,
        fine1_points=args.completion_points,
        fine2_points=args.gt_points,
    )

    train_ds = StoneCompletionDataset(
        dataset_dir=args.dataset_dir,
        intrinsics_file=args.intrinsics,
        stone_ids=args.train_stones,
        gt_complete_dir=args.gt_complete_dir,
        width=args.width,
        height=args.height,
        input_points=args.input_points,
        completion_points=args.completion_points,
        gt_points=args.gt_points,
        min_views=args.min_views,
        max_views=args.max_views,
        max_points_per_view=args.max_points_per_view,
        augment=True,
        samples_per_epoch=args.samples_per_epoch,
    )

    val_ds = StoneCompletionDataset(
        dataset_dir=args.dataset_dir,
        intrinsics_file=args.intrinsics,
        stone_ids=args.val_stones,
        gt_complete_dir=args.gt_complete_dir,
        width=args.width,
        height=args.height,
        input_points=args.input_points,
        completion_points=args.completion_points,
        gt_points=args.gt_points,
        min_views=args.min_views,
        max_views=args.max_views,
        max_points_per_view=args.max_points_per_view,
        augment=False,
        samples_per_epoch=args.val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    LOG.info(
        "Train: %d stones, %d samples/epoch | Val: %d stones, %d samples",
        len(train_ds.stone_ids), len(train_ds),
        len(val_ds.stone_ids), len(val_ds),
    )

    model = StoneCompletionLightning(
        cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seg_weight=args.seg_weight,
        coarse_weight=args.coarse_weight,
        fine1_weight=args.fine1_weight,
        fine2_weight=args.fine2_weight,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.output_dir, name="tb_logs"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        default_root_dir=args.output_dir,
    )

    LOG.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_ckpt,
    )

    best_path = callbacks[0].best_model_path
    if best_path:
        LOG.info("Best checkpoint: %s", best_path)
        state = torch.load(best_path, map_location="cpu")
        model_state = {
            k.replace("model.", "", 1): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
        save_path = os.path.join(args.output_dir, "stone_completion_net.pt")
        torch.save(model_state, save_path)
        LOG.info("Saved standalone model weights: %s", save_path)


if __name__ == "__main__":
    main()
