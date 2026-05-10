# StoneVol_main vs SPIdepth-main: this file differs (kept for comparison).
# mlx worker launch --gpu 1 --memory 32 --type v100-32g python3 ./finetune/train_ft_SQLdepth.py ./conf/b5_in_conf.txt ./finetune/txt_args/train/inc_nyu.txt
import argparse
import os
import sys
import uuid
from datetime import datetime as dt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
try:
    import wandb
except ImportError:
    print("wandb not found, logging disabled. To enable logging, install wandb.")
    wandb = None
from tqdm import tqdm

import model_io
import utils
from dataloader import DepthDataLoader
# from loss import SILogLoss, BinsChamferLoss
from loss import SILogLoss, L2Loss
from utils import RunningAverage, colorize
from SQLdepth import SQLdepth, MonodepthOptions
from mlflow_tracking import MLflowTracker, parse_mlflow_tags
# from options import MonodepthOptions

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "ft-SQLdepth"
logging = True


def is_rank_zero(args):
    return args.rank == 0


import matplotlib

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def _compute_mesh_quality_metrics(gt_depth_map, pred_depth_map, valid_mask, edge_percentile=90.0):
    """Compute mesh-oriented quality metrics on a full depth map.

    Metrics are reported in units that are easy to interpret for reconstruction:
    - mae_mm: mean absolute depth error in millimetres
    - p95_mm: 95th percentile absolute depth error in millimetres
    - normal_mae_deg: mean angular normal error in degrees
    - edge_mae_mm: mean absolute depth error on GT edge regions in millimetres
    """
    gt = np.asarray(gt_depth_map, dtype=np.float64)
    pred = np.asarray(pred_depth_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)

    if gt.shape != pred.shape or gt.shape != valid.shape:
        return None

    valid = valid & np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if not np.any(valid):
        return None

    abs_err = np.abs(pred - gt)
    mae_mm = float(np.mean(abs_err[valid]) * 1000.0)
    p95_mm = float(np.percentile(abs_err[valid], 95.0) * 1000.0)

    gt_dy, gt_dx = np.gradient(gt)
    pred_dy, pred_dx = np.gradient(pred)

    gt_normals = np.stack((-gt_dx, -gt_dy, np.ones_like(gt_dx)), axis=-1)
    pred_normals = np.stack((-pred_dx, -pred_dy, np.ones_like(pred_dx)), axis=-1)
    gt_normals /= np.linalg.norm(gt_normals, axis=-1, keepdims=True) + 1e-8
    pred_normals /= np.linalg.norm(pred_normals, axis=-1, keepdims=True) + 1e-8

    normal_mask = np.zeros_like(valid, dtype=bool)
    normal_mask[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    if not np.any(normal_mask):
        normal_mask = valid

    cos_sim = np.sum(gt_normals * pred_normals, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    normal_mae_deg = float(np.mean(np.degrees(np.arccos(cos_sim[normal_mask]))))

    edge_pct = float(np.clip(edge_percentile, 0.0, 100.0))
    gt_grad_mag = np.sqrt(gt_dx ** 2 + gt_dy ** 2)
    edge_threshold = np.percentile(gt_grad_mag[valid], edge_pct)
    edge_mask = valid & (gt_grad_mag >= edge_threshold)
    if not np.any(edge_mask):
        edge_mask = valid
    edge_mae_mm = float(np.mean(abs_err[edge_mask]) * 1000.0)

    return {
        "mae_mm": mae_mm,
        "p95_mm": p95_mm,
        "normal_mae_deg": normal_mae_deg,
        "edge_mae_mm": edge_mae_mm,
    }


def _get_checkpoint_selection_score(metrics, args):
    """Return (score, mode_used, component_dict) where lower score is better."""
    mode = str(getattr(args, "best_model_selection", "abs_rel")).lower()

    if mode == "composite":
        required = ["mae_mm", "p95_mm", "normal_mae_deg", "edge_mae_mm"]
        components = {}
        for key in required:
            value = metrics.get(key)
            if value is None or not np.isfinite(value):
                abs_rel = metrics.get("abs_rel")
                if abs_rel is None or not np.isfinite(abs_rel):
                    return np.inf, "composite(fallback_failed)", {}
                return float(abs_rel), "abs_rel(fallback)", {"abs_rel": float(abs_rel)}
            components[key] = float(value)

        w_abs_rel = float(getattr(args, "composite_w_abs_rel", 0.0))
        if w_abs_rel != 0.0:
            abs_rel = metrics.get("abs_rel")
            if abs_rel is None or not np.isfinite(abs_rel):
                return np.inf, "composite(fallback_failed)", {}
            components["abs_rel"] = float(abs_rel)

        score = (
            float(getattr(args, "composite_w_mae_mm", 1.0)) * components["mae_mm"]
            + float(getattr(args, "composite_w_p95_mm", 1.0)) * components["p95_mm"]
            + float(getattr(args, "composite_w_normal_deg", 1.0)) * components["normal_mae_deg"]
            + float(getattr(args, "composite_w_edge_mae_mm", 1.0)) * components["edge_mae_mm"]
            + w_abs_rel * components.get("abs_rel", 0.0)
        )
        return float(score), "composite", components

    abs_rel = metrics.get("abs_rel")
    if abs_rel is None or not np.isfinite(abs_rel):
        return np.inf, "abs_rel", {}
    return float(abs_rel), "abs_rel", {"abs_rel": float(abs_rel)}


def _compute_gradient_and_normal_losses(pred, depth, valid_mask):
    """Compute geometric supervision terms on valid depth pixels."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    gt_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

    valid_dx = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
    valid_dy = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]

    grad_loss = pred.new_tensor(0.0)
    if valid_dx.any():
        grad_loss = grad_loss + nn.functional.l1_loss(pred_dx[valid_dx], gt_dx[valid_dx])
    if valid_dy.any():
        grad_loss = grad_loss + nn.functional.l1_loss(pred_dy[valid_dy], gt_dy[valid_dy])

    def _depth_to_normals(depth_map):
        dx = depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1]
        dy = depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :]
        dx = nn.functional.pad(dx, (0, 1, 0, 0))
        dy = nn.functional.pad(dy, (0, 0, 0, 1))
        ones = torch.ones_like(dx)
        normals = torch.cat([-dx, -dy, ones], dim=1)
        return nn.functional.normalize(normals, dim=1)

    pred_normals = _depth_to_normals(pred)
    gt_normals = _depth_to_normals(depth)
    cos_sim = torch.clamp((pred_normals * gt_normals).sum(dim=1, keepdim=True), -1.0, 1.0)

    valid_n = valid_mask.float()
    normal_loss = pred.new_tensor(0.0)
    if valid_n.sum() > 0:
        normal_loss = ((1.0 - cos_sim) * valid_n).sum() / valid_n.sum().clamp(min=1.0)

    return grad_loss, normal_loss


def _compute_background_smoothness_loss(pred_depth, gt_depth, image, valid_mask, edge_percentile=85.0):
    """Edge-aware smoothness on low-gradient GT regions to suppress background artifacts."""
    pred = pred_depth.float()
    depth = gt_depth.float()
    img = image.float()

    if img.shape[-2:] != pred.shape[-2:]:
        img = nn.functional.interpolate(img, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    # Use disparity-like representation so smoothness remains scale-aware.
    disp = 1.0 / torch.clamp(pred, min=1e-6)

    grad_disp_x = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    grad_disp_y = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)

    weights_x = torch.exp(-grad_img_x)
    weights_y = torch.exp(-grad_img_y)

    gt_dx = nn.functional.pad(
        torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1]),
        (0, 1, 0, 0),
    )
    gt_dy = nn.functional.pad(
        torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :]),
        (0, 0, 0, 1),
    )
    gt_grad_mag = torch.sqrt(gt_dx * gt_dx + gt_dy * gt_dy + 1e-12)

    valid = valid_mask.to(torch.bool)
    valid_vals = gt_grad_mag[valid]
    if valid_vals.numel() == 0:
        return pred_depth.new_tensor(0.0)

    q = float(np.clip(edge_percentile, 0.0, 100.0)) / 100.0
    edge_threshold = torch.quantile(valid_vals, q)
    bg_mask = valid & (gt_grad_mag <= edge_threshold)
    if not bg_mask.any():
        bg_mask = valid

    bg_mask_x = bg_mask[:, :, :, 1:] & bg_mask[:, :, :, :-1]
    bg_mask_y = bg_mask[:, :, 1:, :] & bg_mask[:, :, :-1, :]

    loss_x = pred_depth.new_tensor(0.0)
    loss_y = pred_depth.new_tensor(0.0)

    if bg_mask_x.any():
        loss_x = (grad_disp_x * weights_x)[bg_mask_x].mean()
    if bg_mask_y.any():
        loss_y = (grad_disp_y * weights_y)[bg_mask_y].mean()

    return loss_x + loss_y


def main_process(gpu, args, opt):
    args.gpu = gpu

    ###################################### Load model ##############################################

    model = SQLdepth(opt)
    # model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
    #                                       norm=args.norm)

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    print("using DataParallel")
    model = torch.nn.DataParallel(model)
    args.multigpu = False
    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
            experiment_name=args.name, optimizer_state_dict=None, model_opt=opt)

def main_worker(gpu, ngpus_per_node, args, opt):
    args.gpu = gpu

    ###################################### Load model ##############################################

    model = SQLdepth(opt)
    # model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
    #                                       norm=args.norm)

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        print("using DataParallel")
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
            experiment_name=args.name, optimizer_state_dict=None, model_opt=opt)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None, model_opt=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = not args.distributed
    ###CURSOR >>>
    #should_log = should_write and logging
    should_log = should_write and logging and (wandb is not None)
    ###<<<CURSOR
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        # wandb.watch(model)

    mlflow_tags = parse_mlflow_tags(getattr(args, "mlflow_tags", ""))
    mlflow_tags.setdefault("pipeline", "finetune/train_ft_SQLdepth.py")
    mlflow_tags.setdefault("dataset", getattr(args, "dataset", "unknown"))
    mlflow_tags.setdefault("experiment_name", experiment_name)
    mlflow_tracker = MLflowTracker(
        enabled=getattr(args, "mlflow", False),
        tracking_uri=(getattr(args, "mlflow_tracking_uri", "") or None),
        experiment_name=(getattr(args, "mlflow_experiment_name", "StoneVolMain-finetune") or "StoneVolMain-finetune"),
        run_name=(getattr(args, "mlflow_run_name", "") or name),
        tags=mlflow_tags,
    )
    mlflow_tracker.start_run()
    mlflow_tracker.log_params(vars(args), prefix="args")
    if model_opt is not None:
        mlflow_tracker.log_params(vars(model_opt), prefix="model_opt")
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    # criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and torch.cuda.is_available()))

    model.train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_score = np.inf

    ###################################### Scheduler ###############################################
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # not good
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1) # bad
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    ################################################################################################

    run_status = "FINISHED"
    checkpoint_dir = os.path.join(root, "checkpoints")

    try:
        # max_iter = len(train_loader) * epochs
        for epoch in range(args.epoch, epochs):
            ################################# Train loop ##########################################################
            # model.eval()
            # init_metrics, init_val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)
            # wandb.log({f"Metrics/{k}": v for k, v in init_metrics.items()}, step=step)
            # print(f"Metrics: {init_metrics}")
            # model.train()
            if should_log:
                wandb.log({"Epoch": epoch}, step=step)
            mlflow_tracker.log_metrics({"epoch": float(epoch)}, step=step, prefix="train")

            for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                                 total=len(train_loader)):
            # if is_rank_zero( args) else enumerate(train_loader):

                optimizer.zero_grad()

                img = batch['image'].to(device)
                depth = batch['depth'].to(device)
                if 'has_valid_depth' in batch:
                    if not batch['has_valid_depth']:
                        continue

                with torch.amp.autocast('cuda', enabled=(args.amp and torch.cuda.is_available())):
                    # bin_edges, pred = model(img)
                    output = model(img)
                    pred = output
                    # pred = output["disp", 0]
                    # pred = disp_to_depth(pred, 0.1, 100)
                    pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
                # Stone/metric dataset: no per-sample median scaling — model is trained with
                # metric GT depth supervision, so we want absolute scale to be preserved.

                mask = depth > args.min_depth
                with torch.amp.autocast('cuda', enabled=(args.amp and torch.cuda.is_available())):
                    # l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
                    # l_dense = criterion_ueff(pred, depth, mask=valid_mask, interpolate=False)
                    l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=False)

                # if args.w_chamfer > 0:
                #     l_chamfer = criterion_bins(bin_edges, depth)
                # else:
                #     l_chamfer = torch.Tensor([0]).to(img.device)
                ###CURSOR >>>
                    l1_weight = getattr(args, "l1_weight", 0.5)
                    if l1_weight > 0 and mask.any():
                        l_l1 = nn.L1Loss()(pred[mask], depth[mask])
                    else:
                        l_l1 = pred.new_tensor(0.0)

                    grad_weight = float(getattr(args, "gt_grad_weight", 0.0))
                    normal_weight = float(getattr(args, "gt_normal_weight", 0.0))
                    if (grad_weight > 0 or normal_weight > 0) and mask.any():
                        grad_loss, normal_loss = _compute_gradient_and_normal_losses(pred, depth, mask.to(torch.bool))
                    else:
                        grad_loss = pred.new_tensor(0.0)
                        normal_loss = pred.new_tensor(0.0)

                    smooth_weight = float(getattr(args, "background_smoothness_weight", 0.001))
                    use_bg_smooth = bool(getattr(args, "use_background_smoothness_loss", False))
                    smooth_edge_pct = float(getattr(args, "background_edge_percentile", 85.0))
                    if use_bg_smooth and smooth_weight > 0 and mask.any():
                        bg_smooth_loss = _compute_background_smoothness_loss(
                            pred,
                            depth,
                            img,
                            mask,
                            edge_percentile=smooth_edge_pct,
                        )
                    else:
                        bg_smooth_loss = pred.new_tensor(0.0)

                # loss = l_dense + args.w_chamfer * l_chamfer
                #loss = l_dense
                loss = (
                    l_dense
                    + l1_weight * l_l1
                    + grad_weight * grad_loss
                    + normal_weight * normal_loss
                    + smooth_weight * bg_smooth_loss
                )
                ###<<<CURSOR
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if step % 5 == 0:
                    mlflow_tracker.log_metrics(
                        {
                            criterion_ueff.name: l_dense.item(),
                            "total_loss": loss.item(),
                            "l1_loss": l_l1.item(),
                            "grad_loss": grad_loss.item(),
                            "normal_loss": normal_loss.item(),
                            "background_smooth_loss": bg_smooth_loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                        prefix="train",
                    )
                if should_log and step % 5 == 0:
                    wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                    # wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

                step += 1
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()

                ########################################################################################################

                if should_write and step % args.validate_every == 0:

                    ################################# Validation loop ##################################################
                    model.eval()
                    metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                    # print("Validated: {}".format(metrics))
                    if should_log:
                        wandb.log({
                            f"Test/{criterion_ueff.name}": val_si.get_value(),
                            # f"Test/{criterion_bins.name}": val_bins.get_value()
                        }, step=step)

                        if metrics:
                            wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                        print(f"Metrics: {metrics}") # log
                        model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                                 root=checkpoint_dir)

                    if metrics:
                        mlflow_tracker.log_metrics(metrics, step=step, prefix="val")
                    mlflow_tracker.log_metrics(
                        {criterion_ueff.name: val_si.get_value()},
                        step=step,
                        prefix="val",
                    )

                    selection_score, selection_mode_used, selection_components = _get_checkpoint_selection_score(metrics, args)
                    print(
                        f"Checkpoint selection mode={selection_mode_used}, "
                        f"score={selection_score:.6f}"
                    )
                    mlflow_tracker.log_metrics(
                        {"selection_score": selection_score},
                        step=step,
                        prefix="val",
                    )
                    if selection_components:
                        mlflow_tracker.log_metrics(selection_components, step=step, prefix="val_selection")

                    if should_log and np.isfinite(selection_score):
                        wandb.log({"Metrics/selection_score": selection_score}, step=step)
                        if selection_components:
                            wandb.log(
                                {f"Metrics/selection_component/{k}": v for k, v in selection_components.items()},
                                step=step,
                            )

                    if should_write and np.isfinite(selection_score) and selection_score < best_score:
                        best_ckpt_name = f"{experiment_name}_{run_id}_best.pt"
                        model_io.save_checkpoint(model, optimizer, epoch, best_ckpt_name,
                                                 root=checkpoint_dir)
                        best_score = selection_score
                        mlflow_tracker.log_metrics(
                            {"best_selection_score": best_score, "best_epoch": float(epoch)},
                            step=step,
                            prefix="best",
                        )
                        if getattr(args, "mlflow_log_models", False):
                            best_ckpt_path = os.path.join(checkpoint_dir, best_ckpt_name)
                            if os.path.isfile(best_ckpt_path):
                                mlflow_tracker.log_artifact(best_ckpt_path, artifact_path="checkpoints")
                    model.train()
                    #################################################################################################

        if getattr(args, "mlflow_log_models", False):
            latest_ckpt_path = os.path.join(checkpoint_dir, f"{experiment_name}_{run_id}_latest.pt")
            if os.path.isfile(latest_ckpt_path):
                mlflow_tracker.log_artifact(latest_ckpt_path, artifact_path="checkpoints")
    except Exception:
        run_status = "FAILED"
        raise
    finally:
        mlflow_tracker.end_run(status=run_status)

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"):
        # if is_rank_zero( args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            # print(depth.shape, " ==")
            with torch.amp.autocast('cuda', enabled=(args.amp and torch.cuda.is_available())):
                # bins, pred = model(img)
                output = model(img)
                pred = output
                # pred = output["disp", 0]
                # pred = disp_to_depth(pred, 0.1, 100)

                mask = depth > args.min_depth
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
                l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            val_si.append(l_dense.item())

            pred_map = pred.squeeze().cpu().numpy()
            gt_depth_map = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth_map > args.min_depth_eval, gt_depth_map < args.max_depth_eval)
            eval_mask = np.ones(valid_mask.shape, dtype=bool)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth_map.shape
                eval_mask = np.zeros(valid_mask.shape, dtype=bool)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            if not np.any(valid_mask):
                continue

            pred_map[pred_map < args.min_depth_eval] = args.min_depth_eval
            pred_map[pred_map > args.max_depth_eval] = args.max_depth_eval
            pred_map[np.isinf(pred_map)] = args.max_depth_eval
            pred_map[np.isnan(pred_map)] = args.min_depth_eval

            pred_valid = pred_map[valid_mask]
            gt_depth_valid = gt_depth_map[valid_mask]

            if pred_valid.size == 0 or gt_depth_valid.size == 0:
                continue

            sample_metrics = utils.compute_errors(gt_depth_valid, pred_valid)
            if sample_metrics is None:
                continue

            mesh_metrics = _compute_mesh_quality_metrics(
                gt_depth_map,
                pred_map,
                valid_mask,
                edge_percentile=getattr(args, "composite_edge_percentile", 90.0),
            )
            if mesh_metrics is not None:
                sample_metrics.update(mesh_metrics)

            sample_metrics = {k: v for k, v in sample_metrics.items() if np.isfinite(v)}
            if not sample_metrics:
                continue

            metrics.update(sample_metrics)
            # metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='enable mixed precision training to reduce GPU memory usage')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--l1_weight", default=0.5, type=float,
                        help="weight for L1 loss component in total loss")
    parser.add_argument("--gt_grad_weight", default=0.0, type=float,
                        help="weight for depth-gradient supervision in finetuning")
    parser.add_argument("--gt_normal_weight", default=0.0, type=float,
                        help="weight for surface-normal supervision in finetuning")
    parser.add_argument("--use_background_smoothness_loss", default=False, action="store_true",
                        help="if set, applies edge-aware smoothness regularization on low-gradient regions")
    parser.add_argument("--background_smoothness_weight", default=0.001, type=float,
                        help="weight for background smoothness regularization")
    parser.add_argument("--background_edge_percentile", default=85.0, type=float,
                        help="GT gradient percentile used to define low-gradient regions")
    parser.add_argument("--mlflow", default=False, action="store_true",
                        help="enable MLflow tracking for fine-tuning")
    parser.add_argument("--mlflow_tracking_uri", default="", type=str,
                        help="MLflow tracking URI (e.g. file:./mlruns or http://host:5000)")
    parser.add_argument("--mlflow_experiment_name", default="StoneVolMain-finetune", type=str,
                        help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", default="", type=str,
                        help="optional MLflow run name override")
    parser.add_argument("--mlflow_tags", default="", type=str,
                        help="comma-separated tags, e.g. project=stone,phase=finetune")
    parser.add_argument("--mlflow_log_models", default=False, action="store_true",
                        help="if set, logs best/latest checkpoints to MLflow artifacts")
    parser.add_argument(
        "--best_model_selection",
        default="abs_rel",
        choices=["abs_rel", "composite"],
        help=(
            "Metric used for selecting and saving the best checkpoint: "
            "abs_rel (legacy behavior) or composite (mesh-oriented score)."
        ),
    )
    parser.add_argument(
        "--composite_w_mae_mm",
        default=1.0,
        type=float,
        help="Weight for MAE (mm) term in composite checkpoint score.",
    )
    parser.add_argument(
        "--composite_w_p95_mm",
        default=1.0,
        type=float,
        help="Weight for P95 absolute depth error (mm) term in composite checkpoint score.",
    )
    parser.add_argument(
        "--composite_w_normal_deg",
        default=1.0,
        type=float,
        help="Weight for normal angular error (deg) term in composite checkpoint score.",
    )
    parser.add_argument(
        "--composite_w_edge_mae_mm",
        default=1.0,
        type=float,
        help="Weight for edge-region MAE (mm) term in composite checkpoint score.",
    )
    parser.add_argument(
        "--composite_w_abs_rel",
        default=0.0,
        type=float,
        help="Weight for abs_rel term in composite checkpoint score (0 disables this term).",
    )
    parser.add_argument(
        "--composite_edge_percentile",
        default=90.0,
        type=float,
        help="GT depth-gradient percentile used to define edge regions for edge MAE.",
    )
    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--depth_scale', type=float, default=256.0,
                        help='GT depth PNG encoding scale: depth_m = uint16 / depth_scale (default: 256)')
    parser.add_argument('--depth_encoding', type=str, default='uint16',
                        choices=['uint16', 'float32_rgba', 'auto'],
                        help='GT depth PNG encoding used by finetune dataloader')

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=0.0)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="/mnt/bn/hy01/data/nyu",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="/mnt/bn/hy01/data/nyu",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument("--load_weights_folder", type=str, help="path of pth model to load")

    # opt = argparse.ArgumentParser(description="test_simple options", fromfile_prefix_chars='@')
    SQLdepth_options = MonodepthOptions()
    SQLdepth_options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    # SQLdepth_options.convert_arg_line_to_args = convert_arg_line_to_args

    if sys.argv.__len__() >= 3 and not sys.argv[1].startswith("-") and not sys.argv[2].startswith("-"):
        arg_filename_with_prefix = '@' + sys.argv[2]
        cli_overrides = sys.argv[3:]
        args = parser.parse_args([arg_filename_with_prefix] + cli_overrides)
        SQLdepth_opt_filename = '@' + sys.argv[1]
        opt = SQLdepth_options.parser.parse_args([SQLdepth_opt_filename])
    else:
        print("Need options for SQLdepth, error")
        args = parser.parse_args()
        opt = SQLdepth_options.parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    ###CURSOR >>>
    if not hasattr(args, 'rank') or args.rank is None:
        args.rank = 0
    ###<<<CURSOR
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    # try:
    #     node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
    #     nodes = node_str.split(',')

    #     args.world_size = len(nodes)
    #     args.rank = int(os.environ['SLURM_PROCID'])

    # except KeyError as e:
    #     # We are NOT using SLURM
    #     args.world_size = 1
    #     args.rank = 0
    #     nodes = ["127.0.0.1"]

    # if args.distributed:
    #     mp.set_start_method('forkserver')

    #     print(args.rank)
    #     port = np.random.randint(15000, 15025)
    #     args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
    #     print(args.dist_url)
    #     args.dist_backend = 'nccl'
    #     args.gpu = None

    # ngpus_per_node = torch.cuda.device_count()
    # args.num_workers = args.workers
    # args.ngpus_per_node = ngpus_per_node

    # if args.distributed:
    #     args.world_size = ngpus_per_node * args.world_size
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     if ngpus_per_node == 1:
    #         args.gpu = 0
    # SQLdepth loading uses `opt.load_pt_folder`; ensure finetune train args can override
    # the checkpoint path independently of the inference config file.
    if args.load_weights_folder:
        opt.load_pt_folder = args.load_weights_folder
    opt.load_weights_folder = args.load_weights_folder
    main_process(args.gpu, args, opt)

