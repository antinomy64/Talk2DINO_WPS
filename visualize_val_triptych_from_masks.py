#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["pp116", "ade234"], required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--ids", nargs="*", default=None,
                   help="optional image stems to visualize")
    p.add_argument("--pred-suffix", default=".png",
                   help="prediction file suffix: .png / .npy / .npz")
    return p.parse_args()


def make_palette(n=256):
    # standard VOC-style palette
    palette = np.zeros((n, 3), dtype=np.uint8)
    for j in range(n):
        lab = j
        for i in range(8):
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


PALETTE = make_palette(65536 if False else 256)


def dataset_ignore_value(dataset: str) -> int:
    if dataset == "pp116":
        return 255
    if dataset == "ade234":
        return 65535
    raise ValueError(dataset)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        x = np.load(path)
    elif path.suffix.lower() == ".npz":
        obj = np.load(path)
        if len(obj.files) != 1:
            raise ValueError(f"NPZ should contain exactly one array: {path}")
        x = obj[obj.files[0]]
    else:
        x = np.array(Image.open(path))
    if x.ndim != 2:
        raise ValueError(f"mask must be HxW, got {x.shape} from {path}")
    return x


def find_image(image_root: Path, stem: str) -> Path | None:
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in exts:
        p = image_root / f"{stem}{ext}"
        if p.is_file():
            return p
    hits = list(image_root.glob(f"{stem}.*"))
    for p in hits:
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return p
    return None


def find_pred(pred_root: Path, stem: str, pred_suffix: str) -> Path | None:
    p = pred_root / f"{stem}{pred_suffix}"
    if p.is_file():
        return p
    hits = list(pred_root.glob(f"{stem}.*"))
    for q in hits:
        if q.suffix.lower() in {".png", ".npy", ".npz"}:
            return q
    return None


def colorize_mask(mask: np.ndarray, ignore_val: int) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = mask != ignore_val
    m = mask.copy()
    m[~valid] = 0
    m8 = (m % 256).astype(np.uint8)
    out[valid] = PALETTE[m8[valid]]
    return out


def overlay_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    ignore_val: int,
    alpha: float,
) -> np.ndarray:
    color = colorize_mask(mask, ignore_val)
    out = image_rgb.copy().astype(np.float32)
    valid = (mask != ignore_val)
    out[valid] = (1.0 - alpha) * out[valid] + alpha * color[valid].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def add_title(img: Image.Image, title: str, bar_h: int = 34) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, bar_h], fill=(245, 245, 245))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (bar_h - th) / 2 - 1), title, fill=(0, 0, 0), font=font)
    return canvas


def hstack(images: list[Image.Image], pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    out_w = sum(widths) + pad * (len(images) - 1)
    out_h = max(heights)
    canvas = Image.new("RGB", (out_w, out_h), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + pad
    return canvas


def main():
    args = parse_args()
    ignore_val = dataset_ignore_value(args.dataset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_trip = args.out_dir / "triptychs"
    out_trip.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(args.gt_root.glob("*.png"))
    if args.ids:
        wanted = set(args.ids)
        gt_files = [p for p in gt_files if p.stem in wanted]

    report_lines = []
    done = 0

    for gt_path in gt_files:
        stem = gt_path.stem
        img_path = find_image(args.image_root, stem)
        pred_path = find_pred(args.pred_root, stem, args.pred_suffix)

        if img_path is None or pred_path is None:
            report_lines.append(f"MISSING\t{stem}\timg={img_path}\tpred={pred_path}")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        img = Image.open(img_path).convert("RGB")
        if img.size != (gt.shape[1], gt.shape[0]):
            img = img.resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR)

        if pred.shape != gt.shape:
            raise ValueError(
                f"shape mismatch for {stem}: pred {pred.shape}, gt {gt.shape}"
            )

        img_np = np.array(img)
        gt_overlay = overlay_on_image(img_np, gt, ignore_val=ignore_val, alpha=args.alpha)

        pred_ignore = ignore_val
        pred_vis_mask = pred.astype(np.int64).copy()
        # prediction usually has no ignore; if it has negative values, hide them
        pred_vis_mask[pred_vis_mask < 0] = pred_ignore
        pred_overlay = overlay_on_image(
            img_np,
            pred_vis_mask.astype(gt.dtype, copy=False),
            ignore_val=pred_ignore,
            alpha=args.alpha,
        )

        panel_img = add_title(Image.fromarray(img_np), f"{stem} | Input")
        panel_gt = add_title(Image.fromarray(gt_overlay), "GT")
        panel_pred = add_title(Image.fromarray(pred_overlay), "Ours")

        trip = hstack([panel_img, panel_gt, panel_pred], pad=10)
        trip.save(out_trip / f"{stem}.jpg", quality=args.jpeg_quality)

        report_lines.append(f"OK\t{stem}\t{img_path.name}\t{pred_path.name}")
        done += 1

        if args.limit > 0 and done >= args.limit:
            break

    (args.out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[DONE] saved {done} triptychs to {out_trip}")
    print(f"[REPORT] {(args.out_dir / 'report.txt')}")


if __name__ == "__main__":
    main()
