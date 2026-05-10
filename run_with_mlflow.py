#!/usr/bin/env python3
"""Launch StoneVolMain training with MLflow flags injected.

This script does not replace existing training code. It simply calls:
- train.py
- finetune/train_ft_SQLdepth.py
with MLflow-related CLI options.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def _append_mlflow_flags(cmd: List[str], args: argparse.Namespace) -> List[str]:
    cmd = list(cmd)
    cmd.append("--mlflow")

    if args.tracking_uri:
        cmd.extend(["--mlflow_tracking_uri", args.tracking_uri])
    if args.experiment_name:
        cmd.extend(["--mlflow_experiment_name", args.experiment_name])
    if args.run_name:
        cmd.extend(["--mlflow_run_name", args.run_name])
    if args.tags:
        cmd.extend(["--mlflow_tags", args.tags])
    if args.log_models:
        cmd.append("--mlflow_log_models")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run StoneVolMain training with MLflow enabled")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tracking_uri", default="", type=str,
                        help="MLflow tracking URI (e.g. file:./mlruns or http://host:5000)")
    common.add_argument("--experiment_name", default="", type=str,
                        help="MLflow experiment name")
    common.add_argument("--run_name", default="", type=str,
                        help="MLflow run name")
    common.add_argument("--tags", default="", type=str,
                        help="comma-separated tags, e.g. project=stone,phase=train")
    common.add_argument("--log_models", action="store_true",
                        help="also log model checkpoints as MLflow artifacts")
    common.add_argument("--dry_run", action="store_true",
                        help="print the command without executing")

    p_train = subparsers.add_parser("train", parents=[common], help="Run train.py with MLflow")
    p_train.add_argument("config", help="Path to train args config file")

    p_ft = subparsers.add_parser("finetune", parents=[common], help="Run finetune script with MLflow")
    p_ft.add_argument("model_config", help="Path to model config file (e.g. configs/model_cvnXt.txt)")
    p_ft.add_argument("finetune_config", help="Path to finetune args config file")

    args, extra_args = parser.parse_known_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable

    if args.mode == "train":
        cmd = [python_exec, os.path.join(project_root, "train.py"), args.config]
        cmd = _append_mlflow_flags(cmd, args)
        if extra_args:
            cmd.extend(extra_args)
    else:
        cmd = [
            python_exec,
            os.path.join(project_root, "finetune", "train_ft_SQLdepth.py"),
            args.model_config,
            args.finetune_config,
        ]
        cmd = _append_mlflow_flags(cmd, args)
        if extra_args:
            cmd.extend(extra_args)

    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=project_root)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
