#!/usr/bin/env bash
set -euo pipefail

# ── Generate GT clouds + registration params for all 12 stones ──
#
# Prerequisites (on your GPU machine):
#   stone_syn_dataset/
#     stone_01/masks/          (mask_0001.png … mask_0120.png)
#     stone_01_depth_npy/      (depth_0001.npy … depth_0120.npy)
#     stone_02/masks/
#     stone_02_depth_npy/
#     …
#     stone_12/masks/
#     stone_12_depth_npy/
#   splits/stone/intrinsics.txt
#
# Usage:
#   chmod +x generate_all_gt.sh
#   ./generate_all_gt.sh
#
# Output (in stone_syn_dataset/gt_clouds/):
#   stone_XX_gt_pointcloud.ply    – registered GT cloud (16384 pts, centered)
#   stone_XX_registration.npz     – R_floor_up, turntable_center, gt_centroid
#   stone_volumes_gt.json         – volumes for all stones
#
# After this, you can train with:
#   ./venv/bin/python -m volume_estimation.train \
#       --dataset_dir stone_syn_dataset \
#       --volumes_json stone_syn_dataset/gt_clouds/stone_volumes_gt.json \
#       --intrinsics splits/stone/intrinsics.txt \
#       --output_dir volume_training_output \
#       --gt_cloud_dir stone_syn_dataset/gt_clouds \
#       --max_epochs 100 --batch_size 4 --lr 1e-3 \
#       --loss_w_seg 0.0 --loss_w_flow 1.0 --loss_w_chamfer 0.5 \
#       --patience 30

PYTHON="${PYTHON:-./venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-stone_syn_dataset}"
INTRINSICS="${INTRINSICS:-splits/stone/intrinsics.txt}"
GT_DIR="${GT_DIR:-${DATASET_DIR}/gt_clouds}"

STONES="stone_01 stone_02 stone_03 stone_04 stone_05 stone_06 \
        stone_07 stone_08 stone_09 stone_10 stone_11 stone_12"

echo "============================================================"
echo " Generate GT clouds + registration params for all stones"
echo "============================================================"
echo "Python:      ${PYTHON}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Intrinsics:  ${INTRINSICS}"
echo "Output dir:  ${GT_DIR}"
echo ""

# ── Step 1: Verify prerequisites ──
echo "--- Checking prerequisites ---"
missing=0
for sid in $STONES; do
    depth_dir="${DATASET_DIR}/${sid}_depth_npy"
    mask_dir="${DATASET_DIR}/${sid}/masks"
    if [ ! -d "$depth_dir" ]; then
        echo "  MISSING depth dir: $depth_dir"
        missing=$((missing + 1))
    fi
    if [ ! -d "$mask_dir" ]; then
        echo "  MISSING mask dir:  $mask_dir"
        missing=$((missing + 1))
    fi
done

if [ ! -f "$INTRINSICS" ]; then
    echo "  MISSING intrinsics: $INTRINSICS"
    missing=$((missing + 1))
fi

if [ $missing -gt 0 ]; then
    echo ""
    echo "ERROR: $missing prerequisite(s) missing. Fix them and re-run."
    exit 1
fi
echo "  All prerequisites OK."
echo ""

# ── Step 2: Delete old cached GT .npy files ──
echo "--- Cleaning old cached GT files ---"
rm -fv "${GT_DIR}"/*_cached_*.npy 2>/dev/null || true
echo ""

# ── Step 3: Run prepare_gt.py for all stones ──
echo "--- Running prepare_gt.py (depth-merge mode) ---"
echo ""

$PYTHON -m volume_estimation.prepare_gt \
    --dataset_dir "$DATASET_DIR" \
    --intrinsics "$INTRINSICS" \
    --output_dir "$GT_DIR" \
    --stones $STONES

echo ""

# ── Step 4: Verify outputs ──
echo "--- Verifying outputs ---"
ok=0
fail=0
for sid in $STONES; do
    ply="${GT_DIR}/${sid}_gt_pointcloud.ply"
    reg="${GT_DIR}/${sid}_registration.npz"
    if [ -f "$ply" ] && [ -f "$reg" ]; then
        echo "  OK: $sid"
        ok=$((ok + 1))
    else
        echo "  FAIL: $sid (missing ply or registration.npz)"
        fail=$((fail + 1))
    fi
done

vol_json="${GT_DIR}/stone_volumes_gt.json"
if [ -f "$vol_json" ]; then
    echo "  OK: stone_volumes_gt.json"
else
    echo "  FAIL: stone_volumes_gt.json missing"
    fail=$((fail + 1))
fi

echo ""
echo "============================================================"
echo " DONE: $ok stones OK, $fail failed"
echo " GT dir:      $GT_DIR"
echo " Volumes JSON: $vol_json"
echo "============================================================"

if [ $fail -gt 0 ]; then
    exit 1
fi
