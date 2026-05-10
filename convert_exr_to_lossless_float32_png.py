#!/usr/bin/env python3
"""Convert EXR depth to lossless float32 RGBA PNG.

This format preserves 100% of float32 depth information by packing each float32
pixel into 4 uint8 channels (RGBA) without quantization.

Round-trip (EXR -> PNG -> float32) is bit-exact for finite float32 values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: OpenEXR/Imath. Install with: pip install OpenEXR"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow. Install with: pip install Pillow"
    ) from exc


CHANNEL_CANDIDATES = [
    "Depth",
    "depth",
    "Z",
    "z",
    "V",
    "v",
    "Depth.V",
    "ViewLayer.Depth",
    "RenderLayer.Depth",
    "Combined.Z",
]


def find_depth_channel(channels: Iterable[str], preferred: Optional[str]) -> str:
    channels = list(channels)

    if preferred:
        if preferred in channels:
            return preferred
        for name in channels:
            if name.split(".")[-1] == preferred:
                return name
        raise ValueError(
            f"Requested channel '{preferred}' was not found. Available: {channels}"
        )

    for name in CHANNEL_CANDIDATES:
        if name in channels:
            return name

    for name in channels:
        lower = name.lower()
        if "depth" in lower or lower.endswith(".z"):
            return name

    if len(channels) == 1:
        return channels[0]

    raise ValueError(f"No depth channel found. Available channels: {channels}")


def read_exr_depth(path: Path, channel: Optional[str]) -> tuple[np.ndarray, str]:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    data_window = header["dataWindow"]

    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    channels = list(header["channels"].keys())
    chosen = find_depth_channel(channels, channel)

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(chosen, pixel_type)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)

    return depth, chosen


def encode_float32_to_rgba(depth_f32: np.ndarray) -> np.ndarray:
    """Pack float32 array into RGBA uint8 image (H, W, 4), losslessly."""
    depth = np.asarray(depth_f32, dtype="<f4")  # explicit little-endian float32
    rgba = depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)
    return rgba


def decode_rgba_to_float32(rgba_u8: np.ndarray) -> np.ndarray:
    """Unpack RGBA uint8 image (H, W, 4) back to float32 depth."""
    rgba = np.asarray(rgba_u8, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with shape (H, W, 4), got {rgba.shape}")
    depth = rgba.reshape(-1, 4).view("<f4").reshape(rgba.shape[0], rgba.shape[1]).copy()
    return depth


def save_lossless_depth_png(depth_f32: np.ndarray, out_png: Path, compress_level: int) -> None:
    rgba = encode_float32_to_rgba(depth_f32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(str(out_png), compress_level=compress_level)


def verify_roundtrip(depth_f32: np.ndarray, png_path: Path) -> tuple[bool, float, float]:
    rgba = np.array(Image.open(str(png_path)).convert("RGBA"), dtype=np.uint8)
    decoded = decode_rgba_to_float32(rgba)

    # Bit-exact check on raw float32 bits.
    a = np.asarray(depth_f32, dtype="<f4").view(np.uint32)
    b = np.asarray(decoded, dtype="<f4").view(np.uint32)
    exact = np.array_equal(a, b)

    abs_err = np.abs(decoded.astype(np.float64) - depth_f32.astype(np.float64))
    max_abs_err = float(np.nanmax(abs_err)) if abs_err.size else 0.0
    mean_abs_err = float(np.nanmean(abs_err)) if abs_err.size else 0.0

    return exact, max_abs_err, mean_abs_err


def gather_exr_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.exr" if recursive else "*.exr"
    return sorted(input_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EXR depth files to lossless float32 RGBA PNG"
    )
    parser.add_argument("--input_dir", required=True, help="Folder with EXR files")
    parser.add_argument("--output_dir", required=True, help="Folder for output PNG files")
    parser.add_argument(
        "--channel",
        default=None,
        help="Depth channel name in EXR (optional). Example: Depth, Depth.V, Z",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for EXR files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read the written PNG and verify bit-exact round-trip",
    )
    parser.add_argument(
        "--compress_level",
        type=int,
        default=6,
        choices=range(0, 10),
        metavar="0-9",
        help="PNG compression level (0=fast/larger, 9=slow/smaller). Default: 6",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    exr_files = gather_exr_files(input_dir, args.recursive)
    if not exr_files:
        raise SystemExit(f"No EXR files found in: {input_dir}")

    print(f"Found {len(exr_files)} EXR files")

    chosen_channel: Optional[str] = args.channel

    for idx, exr_path in enumerate(exr_files, start=1):
        depth_f32, used_channel = read_exr_depth(exr_path, chosen_channel)
        if chosen_channel is None:
            chosen_channel = used_channel
            print(f"Using detected channel: {chosen_channel}")

        out_png = output_dir / (exr_path.stem + ".png")
        save_lossless_depth_png(depth_f32, out_png, compress_level=args.compress_level)

        finite = np.isfinite(depth_f32)
        if finite.any():
            vals = depth_f32[finite]
            stats = f"min={float(vals.min()):.6f}m max={float(vals.max()):.6f}m"
        else:
            stats = "min=n/a max=n/a"

        if args.verify:
            exact, max_abs_err, mean_abs_err = verify_roundtrip(depth_f32, out_png)
            verify_txt = (
                f" verify_exact={exact} max_abs_err={max_abs_err:.9g} "
                f"mean_abs_err={mean_abs_err:.9g}"
            )
        else:
            verify_txt = ""

        print(f"[{idx}/{len(exr_files)}] Wrote {out_png.name} {stats}{verify_txt}")

    print("Done.")


if __name__ == "__main__":
    main()
