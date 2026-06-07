#!/usr/bin/env python3
"""Convert EXR depth maps to NumPy .npy files (float32)."""

import argparse
import os
import sys

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    sys.exit("Missing dependency: OpenEXR/Imath. Install with: pip install OpenEXR")


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


def find_depth_channel(channels, preferred):
    """Pick a depth channel, honoring a preferred name when provided."""
    channels = list(channels)

    if preferred:
        if preferred in channels:
            return preferred
        for name in channels:
            if name.split(".")[-1] == preferred:
                return name
        raise ValueError(
            "Requested channel '{}' not found. Available: {}".format(preferred, channels)
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

    raise ValueError("No depth channel found. Available: {}".format(channels))


def read_exr_depth(path, channel):
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header["dataWindow"]

    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    chosen = find_depth_channel(channels, channel)

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(chosen, pixel_type)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)

    return depth, chosen


def gather_exr_files(input_dir, recursive):
    results = []
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for f in sorted(files):
                if f.lower().endswith(".exr"):
                    results.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith(".exr"):
                results.append(os.path.join(input_dir, f))
    return results


def save_depth_npy(depth_f32, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(out_path, np.asarray(depth_f32, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Convert EXR depth files to NumPy .npy float32 arrays"
    )
    parser.add_argument("--input_dir", required=True, help="Folder with EXR files")
    parser.add_argument("--output_dir", required=True, help="Folder for output .npy files")
    parser.add_argument(
        "--channel", default=None,
        help="Depth channel name in EXR (optional). Example: Depth, Depth.V, Z",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search for EXR files",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        sys.exit("Input directory does not exist: {}".format(input_dir))

    exr_files = gather_exr_files(input_dir, args.recursive)
    if not exr_files:
        sys.exit("No EXR files found in: {}".format(input_dir))

    total = len(exr_files)
    print("Found {} EXR files".format(total))

    chosen_channel = args.channel

    for idx, exr_path in enumerate(exr_files, start=1):
        depth_f32, used_channel = read_exr_depth(exr_path, chosen_channel)
        if chosen_channel is None:
            chosen_channel = used_channel
            print("Using detected channel: {}".format(chosen_channel))

        rel = os.path.relpath(exr_path, input_dir)
        stem, _ = os.path.splitext(rel)
        out_path = os.path.join(output_dir, stem + ".npy")
        save_depth_npy(depth_f32, out_path)

        finite = np.isfinite(depth_f32)
        if finite.any():
            vals = depth_f32[finite]
            stats = "min={:.6f} max={:.6f}".format(float(vals.min()), float(vals.max()))
        else:
            stats = "min=n/a max=n/a"

        print("[{}/{}] {} {}".format(idx, total, out_path, stats))

    print("Done.")


if __name__ == "__main__":
    main()
