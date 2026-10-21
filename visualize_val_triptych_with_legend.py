#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import List

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
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--pred-suffix", default=".npy", help=".npy / .npz / .png")
    return p.parse_args()


def make_palette(n=256):
    palette = np.zeros((n, 3), dtype=np.uint8)
    for j in range(n):
        lab = j
        for i in range(8):
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


PALETTE = make_palette(256)


def dataset_ignore_value(dataset: str) -> int:
    return 255 if dataset == "pp116" else 65535


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
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
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


def overlay_on_image(image_rgb: np.ndarray, mask: np.ndarray, ignore_val: int, alpha: float) -> np.ndarray:
    color = colorize_mask(mask, ignore_val)
    out = image_rgb.copy().astype(np.float32)
    valid = (mask != ignore_val)
    out[valid] = (1.0 - alpha) * out[valid] + alpha * color[valid].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def hstack(images: List[Image.Image], pad: int = 10, bg=(255, 255, 255)) -> Image.Image:
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


def vstack(images: List[Image.Image], pad: int = 6, bg=(255, 255, 255)) -> Image.Image:
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    out_w = max(widths)
    out_h = sum(heights) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (out_w, out_h), bg)
    y = 0
    for im in images:
        x = (out_w - im.size[0]) // 2
        canvas.paste(im, (x, y))
        y += im.size[1] + pad
    return canvas


def load_class_names(dataset: str) -> list[str]:
    if dataset == "pp116":
        path = Path("src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py")
        var_name = "PART_CLASSES"
    else:
        candidates = [
            Path("ade20kpart234_taxonomy.py"),
            Path("src/open_vocabulary_segmentation/segmentation/datasets/ade20kpart234.py"),
        ]
        path = None
        for c in candidates:
            if c.is_file():
                path = c
                break
        if path is None:
            raise FileNotFoundError("Cannot find ADE234 taxonomy file")
        var_name = "PART_CLASSES"

    spec = importlib.util.spec_from_file_location(f"tmp_{dataset}_classes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    names = getattr(module, var_name)
    return [str(x) for x in names]


def unique_present_labels(mask: np.ndarray, ignore_val: int, valid_region: np.ndarray | None = None) -> list[int]:
    x = mask
    if valid_region is not None:
        x = x[valid_region]
    vals = np.unique(x)
    vals = [int(v) for v in vals if int(v) != int(ignore_val) and int(v) >= 0]
    return vals


def render_legend(panel_w: int, ids: list[int], class_names: list[str], title: str) -> Image.Image:
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    if len(ids) == 0:
        h = 42
        canvas = Image.new("RGB", (panel_w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{title}: none", fill=(0, 0, 0), font=font_title)
        return canvas

    item_h = 20
    sw = 14
    left_pad = 8
    top_pad = 8
    gap_x = 16
    text_offset = 6

    entries = []
    for idx in ids:
        name = class_names[idx] if 0 <= idx < len(class_names) else f"class {idx}"
        entries.append((idx, name))

    max_text_len = max(len(name) for _, name in entries)
    est_char_w = 7
    col_w = sw + text_offset + max(120, min(260, max_text_len * est_char_w))
    ncol = max(1, panel_w // max(col_w + gap_x, 1))
    nrow = math.ceil(len(entries) / ncol)

    legend_h = top_pad + 24 + nrow * item_h + 8
    canvas = Image.new("RGB", (panel_w, legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((left_pad, 6), title, fill=(0, 0, 0), font=font_title)

    for i, (idx, name) in enumerate(entries):
        col = i // nrow
        row = i % nrow
        x = left_pad + col * (col_w + gap_x)
        y = 28 + row * item_h
        color = tuple(int(v) for v in PALETTE[idx % 256])
        draw.rectangle([x, y + 2, x + sw, y + 2 + sw], fill=color, outline=(0, 0, 0))
        draw.text((x + sw + text_offset, y), name, fill=(0, 0, 0), font=font_text)

    return canvas


def main():
    args = parse_args()
    ignore_val = dataset_ignore_value(args.dataset)
    class_names = load_class_names(args.dataset)

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
            raise ValueError(f"shape mismatch for {stem}: pred {pred.shape}, gt {gt.shape}")

        img_np = np.array(img)

        gt_overlay = overlay_on_image(img_np, gt, ignore_val=ignore_val, alpha=args.alpha)

        # Ours 可视化也只在 GT valid 区域上着色
        pred_mask_for_vis = np.full_like(pred, fill_value=ignore_val)
        valid_gt = (gt != ignore_val)
        pred_mask_for_vis[valid_gt] = pred[valid_gt]

        pred_overlay = overlay_on_image(
            img_np,
            pred_mask_for_vis.astype(gt.dtype, copy=False),
            ignore_val=ignore_val,
            alpha=args.alpha,
        )

        gt_ids = unique_present_labels(gt, ignore_val=ignore_val)
        pred_ids = unique_present_labels(pred, ignore_val=-1, valid_region=valid_gt)

        panel_img = add_title(Image.fromarray(img_np), f"{stem} | Input")

        panel_gt_img = add_title(Image.fromarray(gt_overlay), "GT")
        panel_gt_leg = render_legend(panel_gt_img.size[0], gt_ids, class_names, "GT legend")
        panel_gt = vstack([panel_gt_img, panel_gt_leg], pad=4)

        panel_pred_img = add_title(Image.fromarray(pred_overlay), "Ours")
        panel_pred_leg = render_legend(panel_pred_img.size[0], pred_ids, class_names, "Ours legend")
        panel_pred = vstack([panel_pred_img, panel_pred_leg], pad=4)

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
