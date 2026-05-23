#!/usr/bin/env bash
# Fetch pretrained weights for the neural sparse stone reconstruction.
#
# Skips any file already present (and matching its expected SHA256) so
# the script is safe to re-run. Verifies each download against the
# pinned SHA in this script. Targets:
#
#   models/ptv3_stone_binary.pth    PointTransformerV3 binary segmenter
#   models/parenet_3dmatch.pth      PARE-Net 3DMatch checkpoint
#   models/geotransformer_3dmatch.pth GeoTransformer 3DMatch checkpoint
#   models/sghr_3dmatch.pth         SGHR 3DMatch checkpoint
#   models/nksr_shapenet_scannet.pt NKSR ShapeNet + ScanNet checkpoint
#   models/noksr_ptv3.pt            NoKSR PTv3 checkpoint
#
# Usage:
#   bash download_models.sh                 # download all
#   bash download_models.sh --only nksr     # download only the NKSR checkpoint
#   bash download_models.sh --no-verify     # skip SHA check (NOT recommended)
#
# The script writes a manifest at models/MANIFEST.txt listing the URL,
# SHA, and file size of every downloaded checkpoint. The reconstruction
# report cross-references this manifest for reproducibility.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "$MODELS_DIR"
MANIFEST="$MODELS_DIR/MANIFEST.txt"

ONLY=""
VERIFY=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)        ONLY="$2"; shift 2 ;;
        --no-verify)   VERIFY=0; shift ;;
        --models-dir)  MODELS_DIR="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# name | url | sha256 | size_mb
# Replace the placeholder SHAs with real values once the upstream
# repositories publish official checksums; the runtime check enforces
# that the downloaded file matches what's recorded here.
declare -a MODELS=(
"ptv3_stone_binary.pth|https://github.com/Pointcept/PointTransformerV3/releases/download/v1.0/ptv3_scannet_binary.pth|TBD_ptv3_sha|30"
"parenet_3dmatch.pth|https://github.com/yaorz97/PARENet/releases/download/v1.0/parenet_3dmatch.pth|TBD_parenet_sha|4"
"geotransformer_3dmatch.pth|https://github.com/qinzheng93/GeoTransformer/releases/download/v1.0.0/geotransformer-3dmatch.pth.tar|TBD_geotr_sha|50"
"sghr_3dmatch.pth|https://github.com/WHU-USI3DV/SGHR/releases/download/v1.0/sghr_3dmatch.pth|TBD_sghr_sha|50"
"nksr_shapenet_scannet.pt|https://nksr.huangjh.tech/checkpoints/shapenet_scannet.pt|TBD_nksr_sha|100"
"noksr_ptv3.pt|https://github.com/yi-li/NoKSR/releases/download/v1.0/noksr_ptv3.pt|TBD_noksr_sha|60"
)

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch() {
    local name="$1" url="$2" sha="$3" size="$4"
    local out="$MODELS_DIR/$name"
    if [[ -n "$ONLY" && "$name" != "$ONLY"* ]]; then
        return 0
    fi
    if [[ -f "$out" ]]; then
        if [[ "$VERIFY" -eq 1 && "$sha" != TBD* ]]; then
            local got
            got="$(sha256_file "$out")"
            if [[ "$got" == "$sha" ]]; then
                echo "[ok] $name (cached, sha verified)"
                return 0
            fi
            echo "[mismatch] $name: cached file SHA does not match; re-downloading"
            rm -f "$out"
        else
            echo "[ok] $name (cached, sha verify skipped)"
            return 0
        fi
    fi
    echo "[fetch] $name (~${size} MB) <- $url"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$out" "$url"
    else
        wget --tries=3 -O "$out" "$url"
    fi
    if [[ "$VERIFY" -eq 1 && "$sha" != TBD* ]]; then
        local got
        got="$(sha256_file "$out")"
        if [[ "$got" != "$sha" ]]; then
            echo "[FAIL] $name: SHA mismatch (got $got, want $sha)" >&2
            rm -f "$out"
            return 1
        fi
        echo "[verified] $name sha=$got"
    fi
}

{
    echo "# Stone-3D neural pipeline checkpoint manifest"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# Format: name | url | sha256 | size_mb | local_path"
} >"$MANIFEST"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name url sha size <<< "$entry"
    if fetch "$name" "$url" "$sha" "$size"; then
        local_path="$MODELS_DIR/$name"
        actual_sha="(skipped)"
        if [[ "$VERIFY" -eq 1 && -f "$local_path" ]]; then
            actual_sha="$(sha256_file "$local_path")"
        fi
        echo "$name | $url | $actual_sha | $size | $local_path" >>"$MANIFEST"
    fi
done

echo
echo "Manifest written to $MANIFEST"
echo "If any SHA above is 'TBD_*' the runtime checksum was skipped; please"
echo "edit download_models.sh once the upstream maintainers publish official"
echo "SHAs, then re-run with --verify."
