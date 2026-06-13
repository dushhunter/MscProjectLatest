#!/usr/bin/env bash
set -euo pipefail

# ── Generate GT clouds + registration params for all 12 stones ──
#
# This script:
#   1. Backs up existing Blender GT PLYs
#   2. Runs prepare_gt.py to produce depth-merge reference clouds + registration params
#   3. Runs align_blender_gt.py to ICP-align the Blender GT into the registered frame
#   4. Cleans old cached files
#
# Prerequisites (on your GPU machine):
#   stone_syn_dataset/
#     stone_01/masks/          (mask_0001.png … mask_0120.png)
#     stone_01_depth_npy/      (depth_0001.npy … depth_0120.npy)
#     …
#     stone_12/masks/
#     stone_12_depth_npy/
#     gt_clouds/
#       stone_01_gt_pointcloud.ply   (original Blender GT)
#       …
#       stone_12_gt_pointcloud.ply
#   splits/stone/intrinsics.txt
#
# Usage:
#   chmod +x generate_all_gt.sh
#   ./generate_all_gt.sh
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
BLENDER_BACKUP="${BLENDER_BACKUP:-${DATASET_DIR}/blender_gt_backup}"

STONES="stone_01 stone_02 stone_03 stone_04 stone_05 stone_06 \
        stone_07 stone_08 stone_09 stone_10 stone_11 stone_12"

echo "============================================================"
echo " Generate GT clouds + registration params for all stones"
echo "============================================================"
echo "Python:          ${PYTHON}"
echo "Dataset dir:     ${DATASET_DIR}"
echo "Intrinsics:      ${INTRINSICS}"
echo "GT output dir:   ${GT_DIR}"
echo "Blender backup:  ${BLENDER_BACKUP}"
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

blender_found=0
for sid in $STONES; do
    ply="${GT_DIR}/${sid}_gt_pointcloud.ply"
    if [ -f "$ply" ]; then
        blender_found=$((blender_found + 1))
    else
        echo "  MISSING Blender GT: $ply"
        missing=$((missing + 1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "ERROR: $missing prerequisite(s) missing. Fix them and re-run."
    exit 1
fi
echo "  All prerequisites OK ($blender_found Blender PLYs found)."
echo ""

# ── Step 2: Backup Blender GT PLYs ──
echo "--- Backing up Blender GT PLYs to ${BLENDER_BACKUP}/ ---"
mkdir -p "$BLENDER_BACKUP"
for sid in $STONES; do
    src="${GT_DIR}/${sid}_gt_pointcloud.ply"
    dst="${BLENDER_BACKUP}/${sid}_gt_pointcloud.ply"
    if [ -f "$src" ]; then
        cp -v "$src" "$dst"
    fi
done
echo "  Backup complete."
echo ""

# ── Step 3: Clean old cached GT files ──
echo "--- Cleaning old cached GT files ---"
rm -fv "${GT_DIR}"/*_cached_*.npy 2>/dev/null || true
echo ""

# ── Step 4: Run prepare_gt.py (depth-merge reference clouds + registration params) ──
echo "--- Running prepare_gt.py (depth-merge mode) ---"
echo "  This generates _depthmerge_ref.ply + _registration.npz per stone."
echo ""

$PYTHON -m volume_estimation.prepare_gt \
    --dataset_dir "$DATASET_DIR" \
    --intrinsics "$INTRINSICS" \
    --output_dir "$GT_DIR" \
    --stones $STONES

echo ""

# ── Step 5: Verify depth-merge outputs ──
echo "--- Verifying depth-merge outputs ---"
dm_ok=0
dm_fail=0
for sid in $STONES; do
    dm="${GT_DIR}/${sid}_depthmerge_ref.ply"
    reg="${GT_DIR}/${sid}_registration.npz"
    if [ -f "$dm" ] && [ -f "$reg" ]; then
        echo "  OK: $sid (_depthmerge_ref.ply + _registration.npz)"
        dm_ok=$((dm_ok + 1))
    else
        echo "  FAIL: $sid (missing depthmerge_ref.ply or registration.npz)"
        dm_fail=$((dm_fail + 1))
    fi
done

if [ $dm_fail -gt 0 ]; then
    echo ""
    echo "ERROR: $dm_fail depth-merge outputs missing. Cannot proceed to alignment."
    exit 1
fi
echo "  All $dm_ok depth-merge references ready."
echo ""

# ── Step 6: ICP-align Blender GT to the registered frame ──
echo "--- Running align_blender_gt.py (ICP alignment) ---"
echo "  Aligning Blender GT meshes from backup to the depth-merge frame."
echo ""

$PYTHON align_blender_gt.py \
    --blender_dir "$BLENDER_BACKUP" \
    --gt_cloud_dir "$GT_DIR"

echo ""

# ── Step 7: Final verification ──
echo "--- Final verification ---"
ok=0
fail=0
for sid in $STONES; do
    ply="${GT_DIR}/${sid}_gt_pointcloud.ply"
    reg="${GT_DIR}/${sid}_registration.npz"
    dm="${GT_DIR}/${sid}_depthmerge_ref.ply"
    if [ -f "$ply" ] && [ -f "$reg" ] && [ -f "$dm" ]; then
        echo "  OK: $sid (aligned GT + depthmerge ref + registration)"
        ok=$((ok + 1))
    else
        echo "  FAIL: $sid"
        fail=$((fail + 1))
    fi
done

vol_json="${GT_DIR}/stone_volumes_gt.json"
if [ -f "$vol_json" ]; then
    echo "  OK: stone_volumes_gt.json (user-provided, not overwritten)"
else
    echo "  NOTE: stone_volumes_gt.json not found — provide your Blender-measured volumes file"
fi

echo ""
echo "============================================================"
echo " DONE: $ok stones OK, $fail failed"
echo " GT dir:        $GT_DIR"
echo " Blender backup: $BLENDER_BACKUP"
echo " Volumes JSON:  $vol_json"
echo "============================================================"

if [ $fail -gt 0 ]; then
    exit 1
fi
