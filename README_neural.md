# Neural sparse stone reconstruction

Pure deep-learning pipeline in `reconstruct_stone_3d_neural.py`. It replaces
the classical sparse depth-only script `reconstruct_stone_3d_sparse.py` with
2024–2026 learned models at every major stage.

**CUDA is required.** There is no classical fallback inside this script. If
torch, CUDA, upstream packages, or weights under `models/` are missing, the
run fails immediately with a clear error. For dissertation baselines, run
`reconstruct_stone_3d_sparse.py` separately on the same inputs.

## Pipeline

| Stage | Method | Module |
| ----- | ------ | ------ |
| Load depth + intrinsics | Pinhole back-projection | `reconstruct_stone_3d_neural.py` |
| Stone segmentation | PointTransformerV3 binary head | `neural_pipeline/segmentation.py` |
| Floor plane + floor-up frame | RANSAC on PTv3 floor points + Rodrigues | `reconstruct_stone_3d_sparse.py` (geometry helper) |
| Pairwise registration | PARE-Net + ICP; GeoTransformer if a pair is weak | `neural_pipeline/registration_pair.py` |
| Multi-view alignment | SGHR overlap + history-IRLS | `neural_pipeline/registration_multi.py` |
| Surface reconstruction | NKSR (default) or NoKSR ablation | `neural_pipeline/surface.py` |
| Watertight closure | Native to NKSR/NoKSR; polygon cap if a floor hole remains | `neural_pipeline/surface.py` |

Post-neural geometry helpers (not alternative backends):

- **ICP refinement** after PARE-Net / between SGHR iterations
- **Polygon flat cap** only when the neural mesh is not watertight

## Install (CUDA 12.1, 15 GB VRAM minimum)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements_neural.txt

# Replace CPU torch with CUDA torch.
pip install --upgrade torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# GPU backbones (see profile_gpu block in requirements_neural.txt):
pip install spconv-cu121==2.3.6
pip install git+https://github.com/Pointcept/PointTransformerV3.git@main#egg=ptv3
pip install git+https://github.com/yaorz97/PARENet.git@main#egg=parenet
pip install git+https://github.com/qinzheng93/GeoTransformer.git@main#egg=geotransformer
pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@v0.5.4#egg=MinkowskiEngine
pip install git+https://github.com/WHU-USI3DV/SGHR.git@main#egg=sghr
pip install nksr==1.0.4 \
    --extra-index-url https://nksr.huangjh.tech/whl/torch-2.2.0+cu121.html
# NoKSR ablation: git+https://github.com/yi-li/NoKSR.git@main#egg=noksr

# Fetch pretrained weights.
bash download_models.sh
```

Required weights in `models/`:

| File | Stage |
| ---- | ----- |
| `ptv3_stone_binary.pth` | Segmentation |
| `parenet_3dmatch.pth` | Pairwise registration |
| `geotransformer_3dmatch.pth` | Per-pair fallback when PARE-Net fitness is low |
| `sghr_3dmatch.pth` | Multi-view alignment |
| `nksr_shapenet_scannet.pt` | Surface (NKSR) |
| `noksr_ptv3.pt` | Surface (NoKSR ablation) |

## Run

```bash
# Standard neural run (CUDA required).
python reconstruct_stone_3d_neural.py \
    --depth_dir stone_syn_dataset/stone_01_sparse_npy_n18 \
    --intrinsics splits/stone/intrinsics.txt \
    --sequence stone_01 \
    --device cuda \
    --output_dir reconstruction_output_neural_n18

# With chamfer / F-score vs a dense reference mesh.
python reconstruct_stone_3d_neural.py \
    --depth_dir stone_syn_dataset/stone_01_sparse_npy_n18 \
    --intrinsics splits/stone/intrinsics.txt \
    --sequence stone_01 \
    --device cuda \
    --reference_mesh reconstruction_output/stone_mesh_watertight.ply \
    --output_dir reconstruction_output_neural_n18

# NKSR vs NoKSR surface ablation.
for model in nksr noksr; do
    python reconstruct_stone_3d_neural.py \
        --surface_model $model \
        --depth_dir stone_syn_dataset/stone_01_sparse_npy_n18 \
        --intrinsics splits/stone/intrinsics.txt \
        --sequence stone_01 \
        --device cuda \
        --reference_mesh reconstruction_output/stone_mesh_watertight.ply \
        --output_dir reconstruction_output_neural_n18_${model}
done
```

### Classical baseline (separate script)

```bash
python reconstruct_stone_3d_sparse.py \
    --depth_dir stone_syn_dataset/stone_01_sparse_npy_n18 \
    --intrinsics splits/stone/intrinsics.txt \
    --sequence stone_01 \
    --output_dir reconstruction_output_sparse_n18
```

Compare neural vs classical outputs with `neural_pipeline/compare.py`.

## Outputs

Every run writes under `--output_dir`:

- `stone_mesh_pre_closure.ply`   – mesh from NKSR/NoKSR before optional cap
- `stone_mesh_watertight.ply`    – watertight mesh (cap applied if needed)
- `stone_mesh_watertight.obj`    – same, OBJ format
- `stone_pointcloud.ply`         – merged post–pose-graph point cloud
- `stone_3d_views_composite.png` – 6-viewpoint preview render
- `auto_segmentation_preview.png` – per-frame depth + PTv3 mask overlay
- `reconstruction_report.txt`    – per-stage latency, geometry stats,
   pair-fitness distribution, IRLS residual, and (optional)
   chamfer + F-score against `--reference_mesh`.

The report lists each neural stage, its latency, and (for surface) the model
name (`nksr` or `noksr`).
