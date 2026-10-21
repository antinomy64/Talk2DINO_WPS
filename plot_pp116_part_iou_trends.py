#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize per-part IoU across PP-116 evaluation checkpoints.

Only reads the existing per_part_iou.csv. No training or inference.
Usage:
  python plot_pp116_part_iou_trends.py \
    --csv output/pp116_noabs_iou_curve/per_part_iou.csv \
    --out output/pp116_noabs_iou_curve/trend_figures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/pp116_noabs_iou_curve/per_part_iou.csv")
    parser.add_argument("--out", default="output/pp116_noabs_iou_curve/trend_figures")
    parser.add_argument("--threshold", type=float, default=20.0, help="Reference IoU in percent")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        parser.error(f"Input CSV not found: {csv_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    epochs = list(range(0, 101, 10))
    cols = [f"E{e:03d}" for e in epochs]
    needed = {"part", "GT_valid", *cols}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.part.duplicated().any():
        raise ValueError("Repeated part names in CSV")
    if len(df) != 116:
        print(f"[WARNING] Expected 116 parts, got {len(df)}")

    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # Retain taxonomy order if original CSV lists it, but group by the object prefix.
    df["object"] = df.part.astype(str).str.split("'s ", n=1).str[0]
    df["short_part"] = df.part.astype(str).str.split("'s ", n=1).str[-1]
    df["valid"] = df.GT_valid.astype(str).str.lower().isin(["true", "1", "yes"])
    df = df.sort_values(["object", "part"], kind="stable").reset_index(drop=True)

    valid = df[df.valid].copy()
    missing_gt = df[~df.valid].part.to_list()
    if valid[cols].isna().any().any():
        bad = valid.loc[valid[cols].isna().any(axis=1), "part"].to_list()
        raise ValueError(f"GT-valid parts contain missing IoU: {bad}")
    if ((valid[cols] < 0) | (valid[cols] > 100)).any().any():
        raise ValueError("IoU must be percentages between 0 and 100")

    # 1) Global heatmap: every part (including no-GT rows, colored gray).
    data = df[cols].to_numpy(dtype=float)
    data[~df.valid.to_numpy(), :] = np.nan
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D6D6D6")
    fig_height = max(18, 0.26 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    fig.subplots_adjust(left=0.28, right=0.90, top=0.975, bottom=0.04)
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", vmin=0, vmax=100, cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(epochs)), epochs, fontsize=9)
    ax.set_yticks(np.arange(len(df)), df.part, fontsize=7)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Epoch", fontsize=11, labelpad=10)
    ax.set_title("PP-116 | Per-part IoU by training epoch (0–100)\nGray = no GT pixels in validation", fontsize=13, pad=16)
    boundaries = df.groupby("object", sort=False).size().cumsum().iloc[:-1]
    for b in boundaries:
        ax.axhline(b - 0.5, linewidth=1.0, color="white", alpha=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Part IoU (%)")
    heatmap = out / "all_116_parts_heatmap.png"
    fig.savefig(heatmap, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", heatmap)

    # 2) A grid of line charts for each object, with consistent [0, 100] y-axis.
    per_object = out / "by_object"
    per_object.mkdir(exist_ok=True)
    for object_name, g in df.groupby("object", sort=False):
        n = len(g)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.25 * nrows), squeeze=False)
        for ax, (_, r) in zip(axes.flat, g.iterrows()):
            if not r.valid:
                ax.text(0.5, 0.52, "No GT in validation", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(r.short_part, fontsize=12)
            else:
                values = r[cols].to_numpy(dtype=float)
                ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8)
                ax.axhline(args.threshold, linestyle="--", linewidth=1, alpha=0.45)
                peak_idx = int(np.argmax(values))
                ax.set_title(
                    f"{r.short_part}  |  E0 {values[0]:.1f} → E100 {values[-1]:.1f}\n"
                    f"Peak {values[peak_idx]:.1f}% @ E{epochs[peak_idx]}",
                    fontsize=10,
                )
                if np.any(values < args.threshold):
                    ax.spines["left"].set_linewidth(2)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("IoU (%)", fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"{object_name.title()} | {n} Parts | Reference: IoU = {args.threshold:g}%", fontsize=15, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        name = re.sub(r"[^a-z0-9_-]+", "_", object_name.lower())
        path = per_object / f"{name}.png"
        fig.savefig(path, dpi=145)
        plt.close(fig)
        print("[SAVED]", path)

    # 3) Machine-readable summary, with "ANY epoch < threshold", not ALL.
    summary = df[["object", "part", "valid"]].copy()
    summary["min_IoU"] = df[cols].min(axis=1)
    summary["max_IoU"] = df[cols].max(axis=1)
    summary["IoU_E0"] = df["E000"]
    summary["IoU_E100"] = df["E100"]
    summary["any_epoch_below_threshold"] = df.valid & df[cols].lt(args.threshold).any(axis=1)
    summary.to_csv(out / "all_part_trend_summary.csv", index=False)

    print(f"\nValid GT parts: {len(valid)} / {len(df)}")
    print(f"Parts with ANY epoch IoU < {args.threshold:g}%: {int(summary.any_epoch_below_threshold.sum())}")
    print(f"No GT parts: {missing_gt}")
    print(f"Results directory: {out}")

if __name__ == "__main__":
    main()
