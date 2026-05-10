from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import re
import shutil
import sys

from SQLdepth import MonodepthOptions
from test_simple_SQL_config import test_simple


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def parse_weight_index(path):
    base = os.path.basename(path)
    m = re.match(r"weights_(\d+)$", base)
    if m is None:
        return None
    return int(m.group(1))


def list_weight_dirs(models_dir):
    candidates = glob.glob(os.path.join(models_dir, "weights_*"))
    dirs = [p for p in candidates if os.path.isdir(p)]

    weighted = []
    for p in dirs:
        idx = parse_weight_index(p)
        if idx is not None:
            weighted.append((idx, p))

    weighted.sort(key=lambda x: x[0])
    return weighted


def stage_inputs_as_symlinks(input_dir, output_dir, ext):
    pattern = os.path.join(input_dir, "*.{}".format(ext))
    inputs = sorted(glob.glob(pattern))
    staged_links = []

    for src in inputs:
        dst = os.path.join(output_dir, os.path.basename(src))
        if os.path.lexists(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.isfile(dst):
                os.remove(dst)
            else:
                raise RuntimeError("Output path collides with a directory: {}".format(dst))

        os.symlink(os.path.abspath(src), dst)
        staged_links.append(dst)

    return staged_links


def cleanup_links(paths):
    for p in paths:
        if os.path.islink(p):
            os.unlink(p)


def build_base_opts(config_path):
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    return options.parser.parse_args(["@" + config_path])


def main():
    parser = argparse.ArgumentParser(
        description="Run test_simple_SQL_config over all weights_* checkpoints and save separate outputs"
    )
    parser.add_argument("config", type=str, help="Path to inference args file (e.g. ./configs/infer_args.txt)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input images. Defaults to --image_path from config.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for per-weight outputs. Defaults to --image_path from config.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing weights_* folders. Defaults to parent directory of --load_pt_folder from config.",
    )
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="If set, clears each output folder before running inference for that weight.",
    )
    args = parser.parse_args()

    base_opts = build_base_opts(args.config)

    input_dir = os.path.abspath(args.input_dir if args.input_dir else base_opts.image_path)
    output_root = os.path.abspath(args.output_root if args.output_root else base_opts.image_path)

    if args.models_dir:
        models_dir = os.path.abspath(args.models_dir)
    else:
        if not getattr(base_opts, "load_pt_folder", None):
            raise ValueError("Config must define --load_pt_folder, or pass --models_dir")
        models_dir = os.path.dirname(os.path.abspath(base_opts.load_pt_folder))

    if not os.path.isdir(input_dir):
        raise ValueError("Input directory not found: {}".format(input_dir))
    if not os.path.isdir(models_dir):
        raise ValueError("Models directory not found: {}".format(models_dir))

    weight_dirs = list_weight_dirs(models_dir)
    if not weight_dirs:
        raise RuntimeError("No weights_* directories found in {}".format(models_dir))

    os.makedirs(output_root, exist_ok=True)

    print("Input dir   : {}".format(input_dir))
    print("Output root : {}".format(output_root))
    print("Models dir  : {}".format(models_dir))
    print("Checkpoints : {}".format(len(weight_dirs)))

    for weight_idx, weight_dir in weight_dirs:
        out_dir = os.path.join(output_root, "weight_{}".format(weight_idx))

        if args.clear_output_dir and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        print("\n=== Running checkpoint weights_{} ===".format(weight_idx))
        print("weights: {}".format(weight_dir))
        print("output : {}".format(out_dir))

        staged_links = stage_inputs_as_symlinks(input_dir, out_dir, base_opts.ext)
        if not staged_links:
            raise RuntimeError(
                "No input images with extension .{} found in {}".format(base_opts.ext, input_dir)
            )

        try:
            run_opts = argparse.Namespace(**vars(base_opts))
            run_opts.load_pt_folder = weight_dir
            run_opts.image_path = out_dir
            test_simple(run_opts)
        finally:
            cleanup_links(staged_links)

    print("\nAll checkpoints processed.")


if __name__ == "__main__":
    main()
