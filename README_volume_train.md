# StoneVolumeNet: End-to-End Multi-View Stone Volume Estimation

An end-to-end deep learning model that takes multi-view depth maps of a stone as input and directly predicts its volume. The model incorporates stone segmentation, multi-view feature fusion, and **RPF-style rectified flow registration** -- all in a single network.

Adapted from **Rectified Point Flow (RPF)** ([GradientSpaces/Rectified-Point-Flow](https://github.com/GradientSpaces/Rectified-Point-Flow), NeurIPS 2025 Spotlight) and **RAP** ([PRBonn/RAP](https://github.com/PRBonn/RAP), NeurIPS 2025).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration Reference](#configuration-reference)
- [Key Concepts](#key-concepts)
- [References](#references)

---

## Overview

**Problem**: Given multi-view depth maps of a stone captured on a turntable, estimate the stone's volume.

**Approach**: StoneVolumeNet is a multi-task model that simultaneously:

1. **Segments** stone vs background points (per-point binary classification)
2. **Registers** point clouds from different views using a learned velocity field (RPF rectified flow)
3. **Predicts** the scalar volume of the stone from fused multi-view features

**Training data**: 12 synthetic stones rendered in Blender, 120 turntable views each (3 degrees/frame), with ground-truth segmentation masks and known volumes.

**Inference**: Works with sparse views (e.g., 24 out of 120) and produces a volume prediction plus an optional registered 3D point cloud.

**Hardware**: Designed for NVIDIA RTX 4080 16GB with fp16 mixed precision.

---

## Architecture

```
Input: Multi-view depth maps (.npy)
         |
         v
[Back-projection]  depth -> 3D points per view
         |
         v
[PointNet++ Encoder]  (shared weights across views)
  |              |
  v              v
[Seg Head]    [Multi-View Attention]  (RAP-inspired DiTLayer)
  |              |            |
  |              v            v
  |         [Flow Head]  [Volume Head]
  |              |            |
  v              v            v
BCE loss    MSE loss      L1 + MAPE loss
(stone/bg)  (velocity)    (volume)
```

### Module Breakdown

| Module | Description | Parameters |
|--------|-------------|------------|
| **PointNet++ Encoder** | 3 Set Abstraction layers (FPS + ball query + shared MLP). Encodes each view's point cloud with shared weights. | ~280K |
| **Segmentation Head** | Per-point binary classifier. Propagates SA3-level features back to full resolution via NN interpolation. | ~42K |
| **Multi-View Attention** | 4-layer DiTLayer stack with part-wise (within-view) and global (cross-view) attention, sinusoidal 3D position encoding, and learnable view embeddings. | ~5.3M |
| **Flow Head** | Small MLP that predicts per-point 3D velocity from attention features. Trained with RPF's rectified flow objective. | ~67K |
| **Volume Head** | MLP that regresses scalar volume from max-pooled global features. Softplus output ensures positive values. | ~99K |
| **Total** | | **~5.8M** |

### RPF Rectified Flow Branch

During training, the flow branch learns to transport random noise to GT registered positions:

1. **Sample timestep** `t` from a U-shaped distribution (more samples near t=0 and t=1)
2. **Interpolate**: `x_t = (1 - t) * x_0 + t * x_1` where `x_0` = GT positions, `x_1` = Gaussian noise
3. **Predict velocity**: `v_pred = flow_head(attention_features)`
4. **Supervise**: MSE between `v_pred` and target `v_t = x_1 - x_0`

During inference, **Euler ODE integration** from t=1 (noise) to t=0 produces the registered point cloud:

```
x_t = random_noise
for step in range(num_steps):
    v = flow_head(features)
    x_t = x_t - v * dt
# x_t is now the registered point cloud
```

---

## Project Structure

```
volume_estimation/
    __init__.py
    encoder.py        # PointNet++ encoder (FPS, ball query, SA layers)
    attention.py       # Multi-view attention (DiTLayer, position encoding)
    model.py           # StoneVolumeNet (combines all modules + RPF flow)
    loss.py            # Multi-task loss (BCE + L1 + MAPE + flow MSE)
    dataset.py         # StoneVolumeDataset with augmentation + GT flow targets
    train.py           # PyTorch Lightning training script
    prepare_gt.py      # Generate GT volumes via voxelization/convex hull

predict_stone_volume.py  # CLI inference tool with optional flow registration
```

---

## Prerequisites

**Python 3.10** (required for `from __future__ import annotations`):

```bash
# The project venv already has Python 3.10
./venv/bin/python --version  # Should print Python 3.10.x
```

**Dependencies** (install into venv if not already present):

```bash
./venv/bin/pip install torch pytorch-lightning open3d numpy pillow
```

**GPU**: NVIDIA RTX 4080 16GB (or equivalent). Training uses fp16 mixed precision.

---

## Data Preparation

### 1. Convert EXR depth maps to NumPy

If your Blender depth maps are in EXR format, convert them first:

```bash
./venv/bin/python convert_exr_to_npy.py \
  --input_dir stone_syn_dataset/data_depth_annotated/train/groundtruth/stone_01_depth/ \
  --output_dir stone_syn_dataset/stone_01_depth_npy \
  --recursive
```

Repeat for each stone (stone_01 through stone_12).

### 2. Create the ground-truth volumes JSON

Create `stone_volumes_gt.json` in the project root with volumes obtained from Blender (in cm3):

```json
{
  "stone_01": { "volume_cm3": 1.23 },
  "stone_02": { "volume_cm3": 2.45 },
  "stone_03": { "volume_cm3": 0.87 },
  "stone_04": { "volume_cm3": 3.10 },
  "stone_05": { "volume_cm3": 1.95 },
  "stone_06": { "volume_cm3": 2.78 },
  "stone_07": { "volume_cm3": 0.56 },
  "stone_08": { "volume_cm3": 4.12 },
  "stone_09": { "volume_cm3": 1.67 },
  "stone_10": { "volume_cm3": 3.34 },
  "stone_11": { "volume_cm3": 2.01 },
  "stone_12": { "volume_cm3": 1.45 }
}
```

**How to get volume from Blender**: Select the stone object, then in the Python console:

```python
import bpy, bmesh
obj = bpy.context.active_object
bm = bmesh.new()
bm.from_mesh(obj.data)
volume = bm.calc_volume()
print(f"Volume: {volume * 1e6:.4f} cm3")
bm.free()
```

### 3. Expected dataset layout

```
stone_syn_dataset/
    stone_01_depth_npy/       # 120 depth files: depth_0001.npy ... depth_0120.npy
    stone_01/
        masks/                # 120 mask files: mask_0001.png ... mask_0120.png
    stone_02_depth_npy/
    stone_02/
        masks/
    ...
    stone_01_sparse_npy_n24/  # (for inference) 24 sparse views
```

### 4. (Optional) Generate GT point clouds via prepare_gt.py

This merges all 120 views into a dense registered point cloud and computes volume via voxelization. Useful for validation but not required if you already have Blender volumes:

```bash
./venv/bin/python -m volume_estimation.prepare_gt \
  --dataset_dir stone_syn_dataset \
  --intrinsics splits/stone/intrinsics.txt \
  --stone_id stone_01 \
  --width 1024 --height 576
```

---

## Training

### Basic training command

```bash
./venv/bin/python -m volume_estimation.train \
  --dataset_dir stone_syn_dataset \
  --volumes_json stone_volumes_gt.json \
  --intrinsics splits/stone/intrinsics.txt \
  --output_dir volume_training_output \
  --max_epochs 200 \
  --batch_size 4 \
  --lr 1e-3 \
  --precision 16-mixed \
  --loss_w_seg 1.0 \
  --loss_w_volume 0.1 \
  --loss_w_mape 0.05 \
  --loss_w_flow 1.0 \
  --patience 30
```

### Two-stage training (freeze encoder after warmup)

Following RPF's approach, freeze the PointNet++ encoder after the initial epochs to let the flow and volume heads learn more independently:

```bash
./venv/bin/python -m volume_estimation.train \
  --dataset_dir stone_syn_dataset \
  --volumes_json stone_volumes_gt.json \
  --intrinsics splits/stone/intrinsics.txt \
  --output_dir volume_training_output_2stage \
  --max_epochs 200 \
  --batch_size 4 \
  --lr 1e-3 \
  --precision 16-mixed \
  --loss_w_flow 1.0 \
  --freeze_encoder_after 50
```

### With experiment tracking

```bash
# With Weights & Biases
./venv/bin/python -m volume_estimation.train \
  --dataset_dir stone_syn_dataset \
  --volumes_json stone_volumes_gt.json \
  --intrinsics splits/stone/intrinsics.txt \
  --output_dir volume_training_output \
  --wandb --wandb_project stone-volume

# With MLflow
./venv/bin/python -m volume_estimation.train \
  --dataset_dir stone_syn_dataset \
  --volumes_json stone_volumes_gt.json \
  --intrinsics splits/stone/intrinsics.txt \
  --output_dir volume_training_output \
  --mlflow --mlflow_experiment stone-volume
```

### Training output

```
volume_training_output/
    checkpoints/
        best-epoch=042-val_vol_mae=0.1234.ckpt   # Top-3 checkpoints by val MAE
        last.ckpt                                  # Latest checkpoint
    stone_volume_net.pt            # Final model weights (state_dict only)
    training_summary.json          # Training config and results
    tb_logs/                       # TensorBoard logs (default logger)
```

### What the training loop does (RPF pattern)

Each training step follows RPF's `forward() -> loss() -> training_step()` pattern:

1. **forward()**: Encode points with PointNet++, run segmentation head, run multi-view attention, run flow branch (sample timestep, interpolate, predict velocity), run volume head
2. **loss()**: Compute `w_seg * BCE + w_vol * L1 + w_mape * MAPE + w_flow * MSE(v_pred, v_target)`
3. **training_step()**: Call forward(), call loss(), log all metrics
4. **validation_step()**: Same as training + compute segmentation accuracy and R2 score

### Logged metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total weighted loss |
| `train/seg_loss` | Segmentation BCE |
| `train/flow_loss` | RPF velocity MSE |
| `train/vol_l1` | Volume L1 error |
| `train/vol_mape` | Volume MAPE |
| `val/vol_mae` | Validation volume MAE (checkpoint monitor) |
| `val/vol_r2` | Validation R2 score |
| `val/seg_acc` | Validation segmentation accuracy |

### Train/val split

- **Default**: stones 1-10 for training, stones 11-12 for validation
- **Custom**: use `--val_stones stone_03 stone_07` to specify validation stones

### CUDA optimizations (from RPF)

The training script automatically enables:

- `tf32` matmul and cuDNN for faster computation on Ampere+ GPUs
- `cudnn.benchmark` for optimized convolution algorithms
- Gradient clipping at 0.5

---

## Inference

### Direct mode (volume prediction only)

Fast single forward pass. Produces segmented point cloud and predicted volume:

```bash
./venv/bin/python predict_stone_volume.py \
  --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
  --intrinsics splits/stone/intrinsics.txt \
  --sequence stone_01 \
  --checkpoint volume_training_output/stone_volume_net.pt \
  --output_dir volume_output/stone_01
```

### Flow registration mode (volume + registered point cloud)

Adds RPF Euler ODE integration to produce a flow-registered 3D point cloud. Use this when you want to visualize or inspect the merged 3D shape:

```bash
./venv/bin/python predict_stone_volume.py \
  --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
  --intrinsics splits/stone/intrinsics.txt \
  --sequence stone_01 \
  --checkpoint volume_training_output/stone_volume_net.pt \
  --output_dir volume_output/stone_01_flow \
  --use_flow --flow_steps 10
```

### Inference output

```
volume_output/stone_01/
    stone_pointcloud.ply       # Segmented stone points only
    full_pointcloud.ply        # All input points (stone + background)
    flow_registered.ply        # RPF flow-registered cloud (only with --use_flow)
    volume_report.txt          # Human-readable report
    prediction_result.json     # Machine-readable results
```

### Sample report

```
============================================================
StoneVolumeNet - Prediction Report
============================================================

Input:            stone_syn_dataset/stone_01_sparse_npy_n24
Views:            24
Total points:     85432
Stone points:     62105 (72.7%)

Predicted Volume: 1.234567 cm3
                  1234.57 mm3

Inference time:   0.342 s
Flow ODE steps:   10
Flow points:      128

Output files:
  Point cloud:    stone_pointcloud.ply
  Full cloud:     full_pointcloud.ply
  Flow cloud:     flow_registered.ply
  Report:         volume_report.txt
============================================================
```

---

## Configuration Reference

### Training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | (required) | Root directory containing `stone_XX_depth_npy/` and `stone_XX/masks/` |
| `--volumes_json` | (required) | Path to `stone_volumes_gt.json` |
| `--intrinsics` | (required) | Path to `intrinsics.txt` |
| `--output_dir` | `volume_training_output` | Output directory for checkpoints and logs |
| `--max_epochs` | 200 | Maximum training epochs |
| `--batch_size` | 4 | Batch size (4 fits in 16GB VRAM with fp16) |
| `--lr` | 1e-3 | Peak learning rate (OneCycleLR) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--precision` | `16-mixed` | Training precision (`16-mixed`, `32`, `bf16-mixed`) |
| `--max_points_per_view` | 4096 | Max points sampled per depth view |
| `--train_samples_per_epoch` | 500 | Synthetic samples per training epoch |
| `--val_samples_per_epoch` | 100 | Validation samples per epoch |
| `--num_workers` | 4 | DataLoader workers |
| `--patience` | 30 | Early stopping patience (epochs without improvement) |
| `--loss_w_seg` | 1.0 | Segmentation BCE loss weight |
| `--loss_w_volume` | 0.1 | Volume L1 loss weight |
| `--loss_w_mape` | 0.05 | Volume MAPE loss weight |
| `--loss_w_flow` | 1.0 | RPF flow velocity MSE loss weight |
| `--freeze_encoder_after` | -1 | Freeze PointNet++ after this epoch (-1 = never) |
| `--val_stones` | `stone_11 stone_12` | Stones held out for validation |
| `--wandb` | off | Enable Weights & Biases logging |
| `--mlflow` | off | Enable MLflow logging |

### Inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth_dir` | (required) | Directory with sparse `.npy` depth files |
| `--intrinsics` | (required) | Path to `intrinsics.txt` |
| `--sequence` | (required) | Stone ID (e.g., `stone_01`) |
| `--checkpoint` | (required) | Path to trained `.pt` weights |
| `--output_dir` | `volume_output` | Output directory |
| `--use_flow` | off | Enable RPF Euler ODE flow registration |
| `--flow_steps` | 10 | Number of Euler ODE integration steps |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Model configuration (StoneVolumeNetConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sa1_npoint` | 2048 | Points after Set Abstraction layer 1 |
| `sa2_npoint` | 512 | Points after Set Abstraction layer 2 |
| `sa3_npoint` | 128 | Points after Set Abstraction layer 3 |
| `feature_dim` | 256 | Encoder output feature dimension |
| `attn_embed_dim` | 256 | Attention embedding dimension |
| `attn_n_layers` | 4 | Number of DiTLayer attention blocks |
| `attn_n_heads` | 8 | Number of attention heads |
| `timestep_sampling` | `u_shaped` | Timestep distribution (`u_shaped` or `uniform`) |
| `inference_sampling_steps` | 10 | Default Euler ODE steps at inference |

### Loss function

The total loss is a weighted sum of four terms:

```
L = w_seg  * BCE(seg_logits, seg_labels)
  + w_flow * MSE(v_pred, v_target)
  + w_vol  * L1(pred_volume, gt_volume)
  + w_mape * MAPE(pred_volume, gt_volume)
```

| Term | What it supervises | Default weight |
|------|-------------------|----------------|
| `seg_loss` | Per-point stone vs background classification | 1.0 |
| `flow_loss` | Velocity field for point registration (RPF) | 1.0 |
| `vol_l1` | Absolute volume error in cm3 | 0.1 |
| `vol_mape` | Relative volume error (%) | 0.05 |

### Data augmentation (training only)

| Augmentation | Default | Description |
|-------------|---------|-------------|
| Depth noise | 2mm Gaussian | Added to valid depth pixels before back-projection |
| Point dropout | 10% | Random point removal per view |
| Rotation perturbation | 5 degrees | Small random 3D rotation of entire point cloud |
| Scale jitter | 5% | Random uniform scaling of entire point cloud |
| Random view count | 4-24 views | Different number of views sampled per training example |

---

## Key Concepts

### Turntable camera model

The Blender data uses a fixed camera with the stone rotating on a turntable at 3 degrees per frame. This gives analytically known poses: frame `i` has rotation `R_y(i * 3 deg)` about the Y axis. These known poses provide the GT registered positions used as the flow target `x_0`.

### Rectified flow (from RPF)

Rectified flow learns a straight-line transport map between two distributions. In our case:

- **x_0** = GT registered point positions (from turntable poses, pre-augmentation)
- **x_1** = Gaussian noise (sampled randomly)
- **x_t** = (1-t) * x_0 + t * x_1 (linear interpolation at timestep t)
- **v_target** = x_1 - x_0 (constant velocity along the straight line)

The model learns to predict this velocity field. At inference, integrating the learned velocity from t=1 to t=0 transports noise into registered positions.

### U-shaped timestep sampling

RPF samples timesteps `t` with higher density near 0 and 1 (U-shaped distribution via `asinh` transform). This gives the model more training signal at the boundaries where the flow direction changes most rapidly.

### Frozen encoder (two-stage training)

RPF's approach: train everything jointly for N epochs, then freeze the PointNet++ encoder and train only the attention, flow head, and volume head. This prevents the encoder from overfitting on the small dataset while the downstream heads continue to improve. Enable with `--freeze_encoder_after N`.

### GT flow target (gt_points_registered)

The dataset provides `gt_points_registered` -- the clean turntable-aligned point positions **before** data augmentation (rotation perturbation, scale jitter). These serve as the flow learning target `x_0`. The augmented `points` are what the model sees as input, creating a natural mismatch that the flow branch learns to correct.

---

## References

1. **Rectified Point Flow (RPF)**: Gu, Li, Gao, Porikli, "Rectified Diffusion Guidance for Conditional Generation", NeurIPS 2025 Spotlight. [GitHub](https://github.com/GradientSpaces/Rectified-Point-Flow)

2. **RAP**: Pan, Lim, Vizzo, Stachniss, "RAP: Retrieval-Augmented Point Cloud Registration", NeurIPS 2025. [GitHub](https://github.com/PRBonn/RAP)

3. **PointNet++**: Qi, Yi, Su, Guibas, "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017.

4. **Flow Matching**: Lipman, Chen, Ben-Hamu, Nickel, Le, "Flow Matching for Generative Modeling", ICLR 2023.
