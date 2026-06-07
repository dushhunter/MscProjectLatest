# StoneReconNet: Neural Stone Segmentation, Registration, and Volume Measurement

A deep learning model that segments stone points from multi-view depth maps, registers them into a unified 3D coordinate frame via **RPF-style rectified flow**, and reconstructs a watertight mesh for **geometric volume measurement**.

The neural network handles segmentation and registration. Volume is computed geometrically from the Poisson-reconstructed mesh -- not predicted by the network.

Adapted from **Rectified Point Flow (RPF)** ([GradientSpaces/Rectified-Point-Flow](https://github.com/GradientSpaces/Rectified-Point-Flow), NeurIPS 2025 Spotlight) and **RAP** ([PRBonn/RAP](https://github.com/PRBonn/RAP), NeurIPS 2025).

## Table of Contents

- [Overview](#overview)
- [Pipeline Visualization](#pipeline-visualization)
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

**Problem**: Given multi-view depth maps of a stone captured on a turntable, measure the stone's volume accurately.

**Approach**: StoneReconNet replaces the classical pipeline (RANSAC + ICP + pose graph + TSDF) with a trained neural model for segmentation and registration, followed by classical Poisson mesh reconstruction for volume:

```
Multi-view depth maps
        |
        v
[Neural] Segmentation: stone vs floor/background (PointNet++ + BCE)
[Neural] Registration:  align views via RPF flow  (Euler ODE integration)
        |
        v
    Registered stone point cloud
        |
        v
[Classical] Poisson surface reconstruction (Open3D)
        |
        v
    Watertight mesh --> mesh.get_volume()
```

**Training data**: 12 synthetic stones rendered in Blender, 120 turntable views each (3 degrees/frame) + optional 30-40 random views with varied camera angles (for full surface coverage including the bottom). GT volumes from Blender are used for validation only.

**Inference**: Works with sparse views (e.g., 18 turntable + 6-10 random-angle views). Produces a 3D mesh and geometric volume.

**Hardware**: Designed for NVIDIA RTX 4080 16GB with fp16 mixed precision.

---

## Pipeline Visualization

### Animated Pipeline Overview

**Conceptual pipeline:**

![StoneReconNet Pipeline](docs/pipeline_flow.gif)

**Real data pipeline (stone_01, 18 sparse views):**

![Real Data Pipeline](docs/pipeline_real_stone01.gif)

### End-to-End Inference Pipeline (step by step)

| Step | Visualization |
|------|---------------|
| **1. Input depth maps** | ![Step 1](docs/step_1.png) |
| **2. Back-project to 3D** | ![Step 2](docs/step_2.png) |
| **3. Neural segmentation** | ![Step 3](docs/step_3.png) |
| **4. Multi-view fusion** | ![Step 4](docs/step_4.png) |
| **5. RPF flow registration** | ![Step 5](docs/step_5.png) |
| **6. Poisson mesh** | ![Step 6](docs/step_6.png) |
| **7. Geometric volume** | ![Step 7](docs/step_7.png) |
| **Full pipeline summary** | ![Summary](docs/step_8.png) |

### Detailed Text Diagrams

The complete flow from raw depth maps to measured volume:

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 1: INPUT — Multiple Depth Views (.npy)                       │
 │                                                                     │
 │   View 1          View 2          View 3    ...    View 24          │
 │  ┌─────────┐    ┌─────────┐    ┌─────────┐      ┌─────────┐       │
 │  │ ░░▓▓▓░░ │    │ ░░▓▓░░░ │    │ ░▓▓▓▓░░ │      │ ░░▓▓▓░░ │       │
 │  │ ░▓▓▓▓▓░ │    │ ░▓▓▓▓░░ │    │ ▓▓▓▓▓▓░ │      │ ░▓▓▓▓▓░ │       │
 │  │ ▓▓▓▓▓▓▓ │    │ ▓▓▓▓▓▓░ │    │ ▓▓▓▓▓▓▓ │      │ ▓▓▓▓▓▓▓ │       │
 │  │ ▓▓▓▓▓▓▓ │    │ ▓▓▓▓▓▓▓ │    │ ▓▓▓▓▓▓▓ │      │ ▓▓▓▓▓▓▓ │       │
 │  │▄▄▄▄▄▄▄▄▄│    │▄▄▄▄▄▄▄▄▄│    │▄▄▄▄▄▄▄▄▄│      │▄▄▄▄▄▄▄▄▄│       │
 │   depth.npy       depth.npy      depth.npy         depth.npy        │
 │                                                                     │
 │   (Each .npy is a depth map from a different turntable angle)       │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 2: BACK-PROJECTION — Depth pixels to 3D points              │
 │                                                                     │
 │   depth(u,v) ──► X = (u - cx) * Z / fx                            │
 │                  Y = (v - cy) * Z / fy     Each view becomes       │
 │                  Z = depth(u,v)             a 3D point cloud       │
 │                                                                     │
 │   View 1 cloud    View 2 cloud    View 3 cloud   ...               │
 │      ·  ··            · ·            ··  ·                          │
 │     · ·· ·          ·· ··          · ··· ·                          │
 │    ······ ·        ·····          ········                           │
 │   ·········       ·······        ·········                          │
 │   ▄▄▄▄▄▄▄▄▄      ▄▄▄▄▄▄▄▄      ▄▄▄▄▄▄▄▄▄                        │
 │   (stone+floor)   (stone+floor)  (stone+floor)                     │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 3: NEURAL SEGMENTATION — Separate stone from floor           │
 │                                                                     │
 │   PointNet++ encodes each view (shared weights)                     │
 │   Segmentation Head classifies each point: stone or floor?          │
 │                                                                     │
 │   Before segmentation:        After segmentation:                   │
 │      · ·· ·                      · ·· ·  ← stone points (keep)     │
 │     · ·· · ·                    · ·· · ·                            │
 │    ········ ·                  ········ ·                            │
 │   ···········                 ···········                            │
 │   ▄▄▄▄▄▄▄▄▄▄▄  ← floor      ░░░░░░░░░░░  ← floor removed        │
 │                                                                     │
 │   Loss: BCE(predicted_label, GT_mask)                               │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 4: MULTI-VIEW ATTENTION — Fuse features across all views     │
 │                                                                     │
 │   View 1 feats    View 2 feats    View 3 feats                     │
 │   ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
 │   │ f1,f2,f3 │    │ f1,f2,f3 │    │ f1,f2,f3 │                     │
 │   └────┬─────┘    └────┬─────┘    └────┬─────┘                     │
 │        │               │               │                            │
 │        └───────────────┼───────────────┘                            │
 │                        ▼                                            │
 │              ┌─────────────────┐                                    │
 │              │  4x DiTLayer    │  RAP-inspired attention            │
 │              │  (part-wise +   │  with sinusoidal 3D               │
 │              │   global attn)  │  position encoding                │
 │              └────────┬────────┘                                    │
 │                       ▼                                             │
 │              Fused features (all views merged)                      │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 5: RPF FLOW REGISTRATION — Align all views together          │
 │                                                                     │
 │   Euler ODE integration (10 steps, from noise to registered):       │
 │                                                                     │
 │   t=1.0 (noise)   t=0.7          t=0.3          t=0.0 (registered) │
 │     · ·  ·          ·· ·           ··· ·          ·····             │
 │    ·   · ·         · ·· ·         ·····          ······             │
 │   ·  ·  · ·       ·· ··· ·      ······ ·       ········            │
 │    · ·  ·          ·· ···        ········       ·········           │
 │                                                                     │
 │   Random noise     Partially      Mostly         All views          │
 │                    aligned        aligned         registered!        │
 │                                                                     │
 │   Loss: MSE(v_predicted, v_target)                                  │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 6: POISSON SURFACE RECONSTRUCTION — Point cloud to mesh      │
 │                                                                     │
 │   Registered stone           Estimate normals     Poisson mesh      │
 │   point cloud                per point            (watertight)      │
 │                                                                     │
 │      ·····                   ↗·····↗               ┌─────────┐     │
 │     ······                  ↗······↗              ╱           ╲     │
 │    ········      ──►       ↗········↗    ──►     │             │    │
 │   ·········               ↗·········↗             ╲           ╱     │
 │                                                     └─────────┘     │
 │                                                    (closed surface  │
 │                                                     including the   │
 │   Open3D: create_from_point_cloud_poisson()         unseen bottom!) │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 7: GEOMETRIC VOLUME — Exact volume from mesh                 │
 │                                                                     │
 │   volume_cm3 = abs(mesh.get_volume())                               │
 │                                                                     │
 │    ┌─────────┐                                                      │
 │   ╱           ╲     Watertight mesh defines a closed                │
 │  │   V = ?     │    3D region. Volume is computed                   │
 │   ╲           ╱     by summing signed tetrahedra                    │
 │    └─────────┘      volumes (divergence theorem).                   │
 │                                                                     │
 │   Output: 1.234 cm3                                                 │
 └─────────────────────────────────────────────────────────────────────┘
```

### Training vs Inference

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        TRAINING                                     │
 │                                                                     │
 │  120 views/stone ──► PointNet++ ──► Seg Head ──────► BCE loss       │
 │  (with GT masks)      Encoder    ╲                                  │
 │  (with GT poses)                  ╲                                 │
 │                                    Multi-View ──► Flow ──► MSE loss │
 │                                    Attention      Head              │
 │                                                                     │
 │  The model learns:                                                  │
 │    1. Which points belong to the stone (segmentation)               │
 │    2. How to align scattered views into one cloud (registration)    │
 │                                                                     │
 │  NOTE: Volume is NOT part of the loss! No volume regression.        │
 └─────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                       INFERENCE                                     │
 │                                                                     │
 │  24 sparse     Trained        Registered         Poisson       Vol  │
 │  views ──────► Model ───────► Stone Point ──────► Mesh ──────► cm3  │
 │  (.npy)        (seg+flow)     Cloud (.ply)        (.ply)            │
 │                                                                     │
 │  The model does: segment + register (neural)                        │
 │  Then we do:     mesh + volume             (classical, exact)       │
 └─────────────────────────────────────────────────────────────────────┘
```

### Classical Pipeline vs StoneReconNet

```
 CLASSICAL (reconstruct_stone_3d_sparse.py):

  depth ──► RANSAC ──► ICP ──► Pose Graph ──► TSDF ──► Mesh ──► Volume
            (floor    (pair    (global        (voxel   (marching
            removal)   align)   optimize)     fusion)   cubes)

  Pros: No training needed
  Cons: Fragile to noise, slow, requires good overlap


 NEURAL (StoneReconNet):

  depth ──► PointNet++ ──► Seg Head  ──► RPF Flow ──► Poisson ──► Volume
            (learned       (learned       (learned      (exact
            features)      separation)    alignment)    geometry)

  Pros: Robust, fast, learns from data, handles sparse views
  Cons: Requires training data (12 stones, 120 views each)
```

### What Happens to Each Point

```
 Single depth pixel at position (u, v):

  ┌──────────┐     ┌──────────────┐     ┌───────────────┐
  │ depth=   │     │ 3D point in  │     │ Is it stone   │
  │ 0.532 m  │ ──► │ camera frame │ ──► │ or floor?     │
  │ at (u,v) │     │ (X, Y, Z)   │     │               │
  └──────────┘     └──────────────┘     └───────┬───────┘
                                                │
                            ┌───────────────────┼────────────────────┐
                            │                   │                    │
                            ▼                   ▼                    │
                     ┌────────────┐      ┌────────────┐              │
                     │ STONE      │      │ FLOOR      │              │
                     │ seg > 0.5  │      │ seg < 0.5  │              │
                     │            │      │            │              │
                     │ Keep for   │      │ Discard    │              │
                     │ registration│      │            │              │
                     └─────┬──────┘      └────────────┘              │
                           │                                         │
                           ▼                                         │
                    ┌─────────────┐                                  │
                    │ RPF Flow    │                                  │
                    │ registers   │  All stone points from           │
                    │ this point  │  all views get aligned           │
                    │ with points │  into one coordinate frame       │
                    │ from other  │                                  │
                    │ views       │                                  │
                    └─────┬───────┘                                  │
                          │                                          │
                          ▼                                          │
                   Part of the final                                 │
                   registered stone                                  │
                   point cloud                                       │
```

### RPF Rectified Flow: How Registration Works

```
 TRAINING: Learn the velocity field

   GT registered         Gaussian noise          Interpolated
   positions (x_0)       (x_1)                   position (x_t)
                                                  at time t
       ·····               · ·  ·
      ······     ◄─────   ·   · ·        x_t = (1-t)·x_0 + t·x_1
     ········    v_target  ·  ·  · ·
    ·········              · ·  ·         v_target = x_1 - x_0

   The Flow Head learns to predict v_target from the features.

 INFERENCE: Integrate the learned velocity

   Step 0        Step 3        Step 7        Step 10
   t=1.0         t=0.7         t=0.3         t=0.0
   (noise)                                   (registered)

    · ·  ·         ·· ·          ····          ·····
   ·   · ·        · ·· ·        ·····         ······
   ·  ·  · ·     ·· ··· ·     ······ ·      ········
    · ·  ·        ·· ···       ·······       ·········

   x_t ◄── x_t - v_pred * dt   (Euler step, repeated 10 times)
```

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
  |              |
  |              v
  |         [Flow Head]  --> velocity field (RPF rectified flow)
  |              |
  v              v
BCE loss    MSE loss
(stone/bg)  (velocity)
```

### Module Breakdown

| Module | Description | Parameters |
|--------|-------------|------------|
| **PointNet++ Encoder** | 3 Set Abstraction layers (FPS + ball query + shared MLP). Shared weights across views. | ~280K |
| **Segmentation Head** | Per-point binary classifier. NN interpolation from SA3 to full resolution. | ~42K |
| **Multi-View Attention** | 4-layer DiTLayer with part-wise and global attention, sinusoidal 3D position encoding. | ~5.3M |
| **Flow Head** | MLP predicting per-point 3D velocity from attention features. | ~67K |
| **Total** | | **~5.7M** |

### RPF Rectified Flow Branch

During training, the flow branch learns to transport random noise to GT registered positions:

1. **Sample timestep** `t` from a U-shaped distribution
2. **Interpolate**: `x_t = (1 - t) * x_0 + t * x_1` where `x_0` = GT positions, `x_1` = Gaussian noise
3. **Predict velocity**: `v_pred = flow_head(attention_features)`
4. **Supervise**: MSE between `v_pred` and target `v_t = x_1 - x_0`

During inference, **Euler ODE integration** produces the registered point cloud:

```python
x_t = random_noise       # start from t=1
for step in range(num_steps):
    v = flow_head(features)
    x_t = x_t - v * dt   # step toward t=0
# x_t is now the registered stone point cloud
```

### Poisson Mesh & Volume (inference only)

After the model produces a registered point cloud:

```python
pcd.estimate_normals(...)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
volume_cm3 = abs(mesh.get_volume())
```

Poisson reconstruction creates a watertight surface (including the unseen bottom of the stone) from which volume is computed geometrically.

---

## Project Structure

```
volume_estimation/
    __init__.py
    encoder.py          # PointNet++ encoder (FPS, ball query, SA layers)
    attention.py         # Multi-view attention (DiTLayer, position encoding)
    model.py             # StoneReconNet (PointNet++ + SegHead + Attention + FlowHead)
    loss.py              # Segmentation BCE + flow velocity MSE
    dataset.py           # StoneReconDataset (turntable + random views, augmentation)
    train.py             # PyTorch Lightning training script
    prepare_gt.py        # Generate GT volumes via voxelization/convex hull

predict_stone_volume.py          # CLI: model -> Poisson mesh -> volume
blender_export_random_views.py   # Blender script: render depth + poses from random angles
```

---

## Prerequisites

**Python 3.10** (required for `from __future__ import annotations`):

```bash
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

Create `stone_volumes_gt.json` with volumes from Blender (in cm3). These are used for validation metrics only -- the network does **not** predict volume directly.

```json
{
  "stone_01": { "volume_cm3": 1.23 },
  "stone_02": { "volume_cm3": 2.45 },
  ...
  "stone_12": { "volume_cm3": 1.45 }
}
```

**How to get volume from Blender**: Select the stone object, then:

```python
import bpy, bmesh
obj = bpy.context.active_object
bm = bmesh.new()
bm.from_mesh(obj.data)
volume = bm.calc_volume()
print(f"Volume: {volume * 1e6:.4f} cm3")
bm.free()
```

### 3. Generate random-view depth maps (for bottom coverage)

Turntable views only see the sides and top of the stone -- the bottom (where it sits on the flat surface) is never visible. To capture the full surface, render 30-40 depth maps from random camera positions with varied elevation angles (including views from below).

Run the Blender export script:

```bash
blender --background scene.blend --python blender_export_random_views.py -- \
    --stone_name stone_01 \
    --output_dir stone_syn_dataset/stone_01_random_npy \
    --n_views 30 \
    --elev_min -30 \
    --elev_max 80 \
    --seed 42
```

This creates:
- `stone_01_random_npy/depth_0001.npy` ... `depth_0030.npy` (depth maps)
- `stone_01_random_npy/poses.json` (per-view 4x4 camera-to-world extrinsics)

Repeat for each stone.

**Why both turntable + random?**

| | Turntable (120 views) | Random (30-40 views) |
|---|---|---|
| Side coverage | Excellent (3° spacing) | Good |
| Bottom coverage | None | Excellent |
| Pose accuracy | Analytical (exact) | From Blender (exact) |
| View overlap | Very high (redundant) | Variable |
| Result | Good for segmentation training | Critical for accurate volume |

Combined training gives the model both dense side coverage AND bottom coverage.

### 4. Expected dataset layout

```
stone_syn_dataset/
    stone_01_depth_npy/       # 120 turntable depth files
    stone_01_random_npy/      # 30-40 random-view depth files (optional)
        poses.json            # per-view 4x4 camera-to-world matrices
        masks/                # segmentation masks (optional)
    stone_01/
        masks/                # 120 turntable mask PNGs
    stone_02_depth_npy/
    stone_02_random_npy/
        poses.json
    stone_02/
        masks/
    ...
    stone_01_sparse_npy_n24/  # (for inference) 24 sparse views
```

---

## Training

Training teaches the model two tasks:
1. **Segmentation**: which points are stone vs floor/background (BCE loss)
2. **Registration**: how to align multi-view point clouds (flow velocity MSE loss)

### Basic training command (turntable views only)

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
  --loss_w_flow 1.0 \
  --patience 30
```

### Training with combined turntable + random views (recommended)

If you have generated random-view data (see Data Preparation step 3), it is automatically detected and combined with turntable views:

```bash
./venv/bin/python -m volume_estimation.train \
  --dataset_dir stone_syn_dataset \
  --volumes_json stone_volumes_gt.json \
  --intrinsics splits/stone/intrinsics.txt \
  --output_dir volume_training_output_combined \
  --max_epochs 200 \
  --batch_size 4 \
  --lr 1e-3 \
  --precision 16-mixed \
  --loss_w_seg 1.0 \
  --loss_w_flow 1.0 \
  --patience 30 \
  --random_views_suffix _random_npy
```

The dataset automatically detects `stone_XX_random_npy/` directories next to `stone_XX_depth_npy/`. Each batch randomly samples from the combined pool of turntable + random views, so the model learns to handle both view types.

### Two-stage training (freeze encoder after warmup)

Freeze the PointNet++ encoder after initial epochs. The attention and flow head continue learning:

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
  --loss_w_seg 1.0 \
  --loss_w_flow 1.0 \
  --freeze_encoder_after 50
```

### Training output

```
volume_training_output/
    checkpoints/
        best-epoch=042-val_loss=0.1234.ckpt  # Top-3 by val/loss
        last.ckpt
    stone_recon_net.pt           # Final model weights (state_dict)
    training_summary.json        # Training config and results
    tb_logs/                     # TensorBoard logs
```

### What the training loop does (RPF pattern)

Each step follows RPF's `forward() -> loss() -> training_step()`:

1. **forward()**: Encode points, segment, run multi-view attention, compute flow velocity
2. **loss()**: `w_seg * BCE(seg_logits, seg_labels) + w_flow * MSE(v_pred, v_target)`
3. **training_step()**: forward -> loss -> log

### Logged metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total weighted loss |
| `train/seg_loss` | Segmentation BCE |
| `train/flow_loss` | RPF velocity MSE |
| `train/seg_acc` | Segmentation accuracy |
| `val/loss` | Validation total loss (checkpoint monitor) |
| `val/seg_loss` | Validation segmentation loss |
| `val/flow_loss` | Validation flow loss |
| `val/seg_acc` | Validation segmentation accuracy |

### Monitor with TensorBoard

```bash
./venv/bin/python -m tensorboard.main --logdir volume_training_output/tb_logs --port 6006
```

### Train/val split

- **Default**: stones 1-10 for training, stones 11-12 for validation
- **Custom**: use `--val_stones stone_03 stone_07`

---

## Inference

The inference pipeline is fully automatic:

```
Model checkpoint + sparse depth maps
        |
        v
  [1] Encode + segment + RPF flow registration (Euler ODE)
        |
        v
  [2] Poisson surface reconstruction -> watertight mesh
        |
        v
  [3] mesh.get_volume() -> geometric volume in cm3
```

### Inference command (turntable views only)

```bash
./venv/bin/python predict_stone_volume.py \
  --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
  --intrinsics splits/stone/intrinsics.txt \
  --sequence stone_01 \
  --checkpoint volume_training_output/stone_recon_net.pt \
  --output_dir volume_output/stone_01 \
  --flow_steps 10 \
  --poisson_depth 9
```

### Inference with turntable + random views (recommended for better bottom coverage)

```bash
./venv/bin/python predict_stone_volume.py \
  --depth_dir stone_syn_dataset/stone_01_sparse_npy_n24 \
  --random_depth_dir stone_syn_dataset/stone_01_random_npy \
  --intrinsics splits/stone/intrinsics.txt \
  --sequence stone_01 \
  --checkpoint volume_training_output/stone_recon_net.pt \
  --output_dir volume_output/stone_01 \
  --flow_steps 10
```

### Inference output

```
volume_output/stone_01/
    stone_registered.ply     # RPF flow-registered point cloud
    stone_segmented.ply      # Segmented stone points (input space)
    stone_mesh.ply           # Watertight Poisson mesh
    volume_report.txt        # Human-readable report
    prediction_result.json   # Machine-readable results
```

### Sample report

```
============================================================
StoneReconNet -- Volume Prediction Report
============================================================

Input:            stone_syn_dataset/stone_01_sparse_npy_n24
Views:            24
Total points:     85432
Stone points:     62105 (72.7%)
Flow points:      128
Flow ODE steps:   10

--- Mesh Reconstruction ---
Mesh vertices:    8532
Mesh triangles:   17060
Watertight:       True

Volume:           1.234567 cm3
                  1234.57 mm3

Inference time:   0.542 s

Output files:
  Registered PC:  stone_registered.ply
  Segmented PC:   stone_segmented.ply
  Mesh:           stone_mesh.ply
  Report:         volume_report.txt
============================================================
```

---

## Configuration Reference

### Training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | (required) | Root directory with depth data and masks |
| `--volumes_json` | (required) | Path to `stone_volumes_gt.json` |
| `--intrinsics` | (required) | Path to `intrinsics.txt` |
| `--output_dir` | `volume_training_output` | Output directory |
| `--max_epochs` | 200 | Maximum training epochs |
| `--batch_size` | 4 | Batch size (4 fits in 16GB VRAM with fp16) |
| `--lr` | 1e-3 | Peak learning rate (OneCycleLR) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--precision` | `16-mixed` | Training precision |
| `--loss_w_seg` | 1.0 | Segmentation BCE loss weight |
| `--loss_w_flow` | 1.0 | RPF flow velocity MSE loss weight |
| `--freeze_encoder_after` | -1 | Freeze encoder after this epoch (-1 = never) |
| `--patience` | 30 | Early stopping patience |
| `--val_stones` | `stone_11 stone_12` | Validation stones |
| `--random_views_suffix` | `_random_npy` | Suffix for random-view directories |

### Inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth_dir` | (required) | Directory with turntable `.npy` depth files |
| `--random_depth_dir` | (none) | Optional directory with random-view `.npy` files (must contain `poses.json`) |
| `--intrinsics` | (required) | Path to `intrinsics.txt` |
| `--sequence` | (required) | Stone ID (e.g., `stone_01`) |
| `--checkpoint` | (required) | Path to trained `.pt` weights |
| `--output_dir` | `volume_output` | Output directory |
| `--flow_steps` | 10 | Euler ODE integration steps |
| `--poisson_depth` | 9 | Octree depth for Poisson reconstruction |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Model configuration (StoneReconNetConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sa1_npoint` | 2048 | Points after SA layer 1 |
| `sa2_npoint` | 512 | Points after SA layer 2 |
| `sa3_npoint` | 128 | Points after SA layer 3 |
| `feature_dim` | 256 | Encoder feature dimension |
| `attn_embed_dim` | 256 | Attention embedding dimension |
| `attn_n_layers` | 4 | Number of DiTLayer attention blocks |
| `attn_n_heads` | 8 | Number of attention heads |
| `timestep_sampling` | `u_shaped` | Timestep distribution |
| `inference_sampling_steps` | 10 | Default Euler ODE steps |

### Loss function

```
L = w_seg  * BCE(seg_logits, seg_labels)
  + w_flow * MSE(v_pred, v_target)
```

| Term | What it supervises | Default weight |
|------|-------------------|----------------|
| `seg_loss` | Per-point stone vs background classification | 1.0 |
| `flow_loss` | Velocity field for point cloud registration (RPF) | 1.0 |

---

## Key Concepts

### Why geometric volume instead of neural regression?

- A neural network predicting a scalar volume must learn the concept of 3D shape and enclosed space implicitly -- very hard with limited training data
- Poisson reconstruction creates a watertight surface (including the unseen bottom), from which volume is mathematically exact
- The network only needs to learn segmentation and registration -- much simpler tasks

### Turntable camera model

The Blender data uses a fixed camera with the stone rotating at 3 degrees/frame. Frame `i` has rotation `R_y(i * 3 deg)` about the Y axis. These known poses provide the GT registered positions used as the flow target.

### Random views (for bottom coverage)

Turntable views never see the bottom of the stone. Random views place the camera at arbitrary azimuth + elevation angles (including below the equator), providing depth data for the bottom surface. Each random view stores its 4x4 camera-to-world matrix in `poses.json`. Combined with turntable views, this gives complete surface coverage for accurate volume measurement.

```
Turntable views:             Random views:
  fixed elevation              varied elevation

     camera                    camera positions:
       |                         ·   ·   ·
  ─────┼─────  equator        ·  [stone]  ·
       |                         ·   ·   ·
   [stone]                    ← includes below
                                 equator!
```

### Rectified flow (from RPF)

Learns a straight-line transport map between noise and registered positions:

- **x_0** = GT registered positions (from turntable poses, pre-augmentation)
- **x_1** = Gaussian noise
- **x_t** = (1-t) * x_0 + t * x_1
- **v_target** = x_1 - x_0

At inference, integrating the learned velocity from t=1 to t=0 produces the registered cloud.

### U-shaped timestep sampling

RPF samples timesteps `t` with higher density near 0 and 1, giving more training signal at the boundaries where the flow direction changes most rapidly.

### Frozen encoder (two-stage training)

Train everything jointly for N epochs, then freeze PointNet++ and train only the attention + flow head. Prevents encoder overfitting on small datasets.

---

## References

1. **Rectified Point Flow (RPF)**: NeurIPS 2025 Spotlight. [GitHub](https://github.com/GradientSpaces/Rectified-Point-Flow)

2. **RAP**: Pan et al., NeurIPS 2025. [GitHub](https://github.com/PRBonn/RAP)

3. **PointNet++**: Qi et al., NeurIPS 2017.

4. **Flow Matching**: Lipman et al., ICLR 2023.
