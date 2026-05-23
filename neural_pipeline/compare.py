"""Side-by-side neural vs classical comparison table.

Reads the watertight meshes from a list of (label, output_dir) pairs,
compares each to a dense reference mesh with shape-aligned chamfer /
F-score, and writes a Markdown table and a CSV file. Used by the
dissertation chapter to show how the neural pipeline compares to the
classical baseline at N = 12, 18, 24.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .report import compute_chamfer_fscore


LOG = logging.getLogger("stone3d_neural.compare")


def read_report(out_dir: str) -> Dict[str, str]:
    report_path = os.path.join(out_dir, "reconstruction_report.txt")
    metrics: Dict[str, str] = {}
    if not os.path.exists(report_path):
        return metrics
    text = open(report_path).read()

    patterns = {
        "total_time_s":    r"Total time:\s*([\d.]+)s",
        "n_frames":        r"Frames input:\s*(\d+)",
        "bbox_x_mm":       r"Bounding box \(mm\):\s*x=([\d.]+)",
        "bbox_y_mm":       r"y=([\d.]+)\s+z=",
        "bbox_z_mm":       r"z=([\d.]+)",
        "surface_area_mm2": r"Surface area \(mm\^2\):\s*([\d.]+)",
        "volume_mm3":      r"Volume \(mm\^3\):\s*([\d.]+)",
        "watertight":      r"Watertight:\s*(\w+)",
        "n_triangles":     r"Triangles:\s*(\d+)",
        "edges_kept":      r"edges kept per iter:\s*\[([0-9, ]+)\]",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = m.group(1)

    # Parse per-stage backend lines from the neural-only report.
    backend_pat = re.compile(
        r"^\s+(segmentation|pairwise_registration|multiview_registration|surface)\s+"
        r"(\w+)\s+[\d.]+s",
        re.MULTILINE,
    )
    stages: Dict[str, Dict[str, str]] = {}
    for m in backend_pat.finditer(text):
        stages[m.group(1)] = {"used": m.group(2)}
    metrics["stages"] = stages
    # Surface model from "model: nksr|noksr" or "surface_model config:" line.
    m_model = re.search(r"^\s+model:\s*(\w+)", text, re.MULTILINE)
    if m_model:
        metrics["surface_model"] = m_model.group(1)
    else:
        m_cfg = re.search(r"surface_model config:\s*(\w+)", text)
        if m_cfg:
            metrics["surface_model"] = m_cfg.group(1)
    return metrics


def mesh_metrics(mesh_path: str, reference_mesh_path: Optional[str]) -> Dict[str, float]:
    if not os.path.exists(mesh_path):
        return {}
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.triangles) == 0:
        return {}

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent()) * 1000.0
    out: Dict[str, float] = {
        "n_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "n_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "bbox_x_mm": float(extent[0]),
        "bbox_y_mm": float(extent[1]),
        "bbox_z_mm": float(extent[2]),
        "surface_area_mm2": float(mesh.get_surface_area() * 1e6),
        "is_watertight": bool(mesh.is_watertight()),
    }

    if reference_mesh_path and os.path.exists(reference_mesh_path):
        ref = o3d.io.read_triangle_mesh(reference_mesh_path)
        if len(ref.triangles):
            ch = compute_chamfer_fscore(mesh, ref)
            out.update({
                "chamfer_mm": ch["chamfer_mm"],
                "f_score_1mm": ch["f_score_1mm"],
                "precision_1mm": ch["precision_1mm"],
                "recall_1mm": ch["recall_1mm"],
                "align_rmse_mm": ch.get("alignment_rmse_mm", float("nan")),
            })
    return out


def make_table(
    rows: List[Tuple[str, str]],
    reference_mesh: Optional[str],
) -> Tuple[List[Dict[str, str]], str]:
    """Returns (csv_rows, markdown_table)."""
    csv_rows = []
    md_lines = [
        "| run | N | seg | pair | multi | surf | model | tris | "
        "bbox xyz (mm) | surf (mm^2) | wt | chamfer (mm) | F@1mm | "
        "align RMSE (mm) | time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, out_dir in rows:
        rep = read_report(out_dir)
        mesh_path = os.path.join(out_dir, "stone_mesh_watertight.ply")
        m = mesh_metrics(mesh_path, reference_mesh)
        if not m:
            md_lines.append(
                f"| {label} | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        stages = rep.get("stages", {}) or {}
        seg = stages.get("segmentation", {}).get("used", "?")
        pair = stages.get("pairwise_registration", {}).get("used", "?")
        multi = stages.get("multiview_registration", {}).get("used", "?")
        surf_used = stages.get("surface", {}).get("used", "?")
        model = rep.get("surface_model", "?")
        n = rep.get("n_frames", "?")
        tris = int(m["n_triangles"])
        bbox = f"{m['bbox_x_mm']:.1f}x{m['bbox_y_mm']:.1f}x{m['bbox_z_mm']:.1f}"
        surf_area = f"{m['surface_area_mm2']:.0f}"
        wt = "yes" if m["is_watertight"] else "no"
        ch = m.get("chamfer_mm", float("nan"))
        f1 = m.get("f_score_1mm", float("nan"))
        align = m.get("align_rmse_mm", float("nan"))
        t = rep.get("total_time_s", "?")
        md_lines.append(
            f"| {label} | {n} | {seg} | {pair} | {multi} | {surf_used} | {model} | "
            f"{tris} | {bbox} | {surf_area} | {wt} | "
            f"{ch:.3f} | {f1:.3f} | {align:.3f} | {t} |"
        )
        csv_rows.append({
            "label": label, "out_dir": out_dir, "n_frames": n,
            "seg_backend": seg, "pair_backend": pair, "multi_backend": multi,
            "surface_backend": surf_used, "surface_model": model,
            "n_triangles": tris, "bbox_x_mm": m['bbox_x_mm'],
            "bbox_y_mm": m['bbox_y_mm'], "bbox_z_mm": m['bbox_z_mm'],
            "surface_area_mm2": m['surface_area_mm2'],
            "is_watertight": m['is_watertight'],
            "chamfer_mm": ch, "f_score_1mm": f1, "align_rmse_mm": align,
            "total_time_s": t,
        })
    return csv_rows, "\n".join(md_lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--row", action="append", required=True,
        help="Label=output_dir entry. Pass multiple times. "
             "Example: --row classical_n12=reconstruction_output_sparse_n12 "
             "--row neural_n12=reconstruction_output_neural_n12",
    )
    p.add_argument("--reference_mesh", default=None,
                   help="Dense reference mesh for chamfer/F-score comparison.")
    p.add_argument("--out_md", default="reconstruction_output_neural_compare.md")
    p.add_argument("--out_csv", default="reconstruction_output_neural_compare.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[Tuple[str, str]] = []
    for entry in args.row:
        if "=" not in entry:
            print(f"--row entries must be label=dir; got {entry!r}", file=sys.stderr)
            return 2
        label, d = entry.split("=", 1)
        rows.append((label, d))

    csv_rows, md = make_table(rows, args.reference_mesh)

    with open(args.out_md, "w") as f:
        f.write("# Neural vs classical comparison\n\n")
        if args.reference_mesh:
            f.write(f"Reference mesh: `{args.reference_mesh}`\n\n")
        f.write(md + "\n")
    print(f"Wrote markdown table: {args.out_md}")

    if csv_rows:
        with open(args.out_csv, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"Wrote csv table: {args.out_csv}")

    print(); print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
