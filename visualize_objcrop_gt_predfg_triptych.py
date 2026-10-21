#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["pp116", "ade234"], required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--part-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None, help="optional image_id list")
    return p.parse_args()


def torch_load(path: Path):
    return torch.load(path, map_location="cpu")


def load_cache_as_list(path: Path) -> list[dict[str, Any]]:
    payload = torch_load(path)
    if isinstance(payload, list):
        data = payload
    elif isinstance(payload, dict):
        for key in ["annotations", "data", "items", "samples"]:
            if key in payload and isinstance(payload[key], list):
                data = payload[key]
                break
        else:
            raise ValueError(f"Cannot find list-like annotations in {path}")
    else:
        raise ValueError(f"Unsupported cache type: {type(payload)}")

    out = []
    for x in data:
        if isinstance(x, dict):
            out.append(x)
    if not out:
        raise ValueError(f"No valid dict annotations found in {path}")
    return out


def dataset_ignore_value(dataset: str) -> int:
    return 255 if dataset == "pp116" else 65535


def get_image_id(ann: dict) -> str:
    for k in ["image_id", "img_id", "id"]:
        if k in ann:
            return str(ann[k])
    if "file_name" in ann:
        return Path(str(ann["file_name"])).stem
    raise KeyError("Cannot find image_id/file_name in annotation")


def get_crop_box_xyxy(ann: dict):
    for k in ["cropaug_box_xyxy", "crop_box_xyxy", "box_xyxy", "bbox_xyxy"]:
        if k in ann:
            box = ann[k]
            if torch.is_tensor(box):
                box = box.detach().cpu().tolist()
            box = [int(round(float(v))) for v in box]
            if len(box) == 4:
                return box
    raise KeyError("Cannot find crop box xyxy in annotation")


def get_present_part_ids(ann: dict) -> list[int]:
    for k in ["part_category_id", "part_ids", "present_part_ids"]:
        if k in ann:
            ids = ann[k]
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().tolist()
            if isinstance(ids, (list, tuple, np.ndarray)):
                ids = [int(x) for x in ids]
                return sorted(set(ids))
            if isinstance(ids, (int, np.integer)):
                return [int(ids)]
    return []


def get_pred_fg_patch_mask32(ann: dict) -> np.ndarray:
    if "pred_obj_mask_patch" not in ann:
        raise KeyError("Annotation missing pred_obj_mask_patch")
    x = ann["pred_obj_mask_patch"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    if x.ndim == 1:
        if x.size != 32 * 32:
            raise ValueError(f"pred_obj_mask_patch flat size must be 1024, got {x.size}")
        x = x.reshape(32, 32)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported pred_obj_mask_patch shape: {x.shape}")

    # robust binarization
    if x.dtype == np.bool_:
        m = x
    else:
        # many caches already store bool or 0/1
        # if logits/probabilities appear, threshold at >0.5
        uniq = np.unique(x)
        if np.all(np.isin(uniq, [0, 1])):
            m = x.astype(bool)
        else:
            m = x > 0.5
    return m.astype(bool)


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


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def crop_and_resize_rgb(img_path: Path, box_xyxy, side: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = box_xyxy
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((side, side), resample=Image.BILINEAR)
    return np.asarray(crop)


def crop_part_mask_to_32(mask_path: Path, box_xyxy) -> np.ndarray:
    m = Image.open(mask_path)
    x1, y1, x2, y2 = box_xyxy
    crop = m.crop((x1, y1, x2, y2))
    crop = crop.resize((32, 32), resample=Image.NEAREST)
    return np.asarray(crop)


def compute_gt_fg_from_part_union(part_crop_32: np.ndarray, present_part_ids: list[int], ignore_val: int) -> np.ndarray:
    if present_part_ids:
        gt_fg = np.isin(part_crop_32, np.array(present_part_ids, dtype=part_crop_32.dtype))
    else:
        gt_fg = part_crop_32 != ignore_val
    gt_fg = gt_fg & (part_crop_32 != ignore_val)
    return gt_fg.astype(bool)


def resize_mask_bool(mask32: np.ndarray, side: int) -> np.ndarray:
    img = Image.fromarray((mask32.astype(np.uint8) * 255))
    img = img.resize((side, side), resample=Image.NEAREST)
    return (np.asarray(img) > 127)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    idx = mask.astype(bool)
    out[idx] = (1 - alpha) * out[idx] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def add_title(img: Image.Image, title: str, bar_h: int = 34) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, bar_h], fill=(245, 245, 245))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (bar_h - th) / 2 - 1), title, fill=(0, 0, 0), font=font)
    return canvas


def add_footer(img: Image.Image, lines: list[str], pad=8) -> Image.Image:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    draw0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_h = 0
    widths = []
    for s in lines:
        bb = draw0.textbbox((0, 0), s, font=font)
        widths.append(bb[2] - bb[0])
        line_h = max(line_h, bb[3] - bb[1])

    w, h = img.size
    footer_h = pad * 2 + len(lines) * (line_h + 2)
    canvas = Image.new("RGB", (w, h + footer_h), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = h + pad
    for s in lines:
        draw.text((8, y), s, fill=(0, 0, 0), font=font)
        y += line_h + 2
    return canvas


def hstack(images, pad=10, bg=(255, 255, 255)):
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


def iou_and_dice(a: np.ndarray, b: np.ndarray):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    sa = int(a.sum())
    sb = int(b.sum())
    iou = inter / union if union > 0 else float("nan")
    dice = (2 * inter) / (sa + sb) if (sa + sb) > 0 else float("nan")
    return inter, union, sa, sb, iou, dice



def get_class_names_for_crop_vis(dataset: str) -> list[str]:
    """Read exact dataset taxonomy, preserving the numeric part ID ordering."""
    if dataset == "pp116":
        taxonomy = Path("src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py")
        source = ast.parse(taxonomy.read_text(encoding="utf-8"), filename=str(taxonomy))
        classes = None
        for node in ast.walk(source):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PART_CLASSES" for t in node.targets):
                classes = list(ast.literal_eval(node.value))
                break
        if classes is None:
            raise RuntimeError("PART_CLASSES not found in PP116 taxonomy")
        n_expected = 116
    elif dataset == "ade234":
        from ade20kpart234_taxonomy import PART_CLASSES
        classes = list(PART_CLASSES)
        n_expected = 234
    else:
        raise ValueError(dataset)
    if len(classes) != n_expected or len(set(classes)) != n_expected:
        raise ValueError(f"{dataset}: expected {n_expected} unique canonical part names, found {len(classes)}")
    if any("'s " not in str(x) for x in classes):
        raise ValueError(f"{dataset}: unrecognized object-part label format")
    return [str(x) for x in classes]


def get_target_object_for_crop(part_ids: list[int], class_names: list[str], crop_idx: int) -> str:
    """Derive object from part IDs; fail on ambiguous cross-object crops."""
    if not part_ids:
        raise ValueError(f"Crop {crop_idx}: empty part_category_id, cannot determine object")
    if any(x < 0 or x >= len(class_names) for x in part_ids):
        raise ValueError(f"Crop {crop_idx}: invalid part_category_id: {part_ids}")
    objects = {class_names[p].split("'s ", 1)[0] for p in part_ids}
    if len(objects) != 1:
        raise ValueError(f"Crop {crop_idx}: multiple object categories {sorted(objects)} from part IDs {part_ids}")
    return next(iter(objects))

def main():
    args = parse_args()
    ignore_val = dataset_ignore_value(args.dataset)

    anns = load_cache_as_list(args.cache)
    class_names = get_class_names_for_crop_vis(args.dataset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trip_dir = args.out_dir / "triptychs"
    trip_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"

    wanted_ids = set(args.ids) if args.ids else None

    rows = []
    saved = 0

    for idx, ann in enumerate(anns):
        if idx < args.start:
            continue
        if args.limit > 0 and saved >= args.limit:
            break

        try:
            image_id = get_image_id(ann)
        except Exception:
            continue

        if wanted_ids is not None and image_id not in wanted_ids:
            continue

        img_path = find_image(args.image_root, image_id)
        if img_path is None:
            continue

        part_path = args.part_root / f"{image_id}.png"
        if not part_path.is_file():
            continue

        try:
            box = get_crop_box_xyxy(ann)
            present_parts = get_present_part_ids(ann)
            pred_fg_32 = get_pred_fg_patch_mask32(ann)
        except Exception:
            continue

        object_name = get_target_object_for_crop(present_parts, class_names, idx)
        rgb_crop = crop_and_resize_rgb(img_path, box, args.side)
        part_crop_32 = crop_part_mask_to_32(part_path, box)
        gt_fg_32 = compute_gt_fg_from_part_union(part_crop_32, present_parts, ignore_val)

        gt_fg = resize_mask_bool(gt_fg_32, args.side)
        pred_fg = resize_mask_bool(pred_fg_32, args.side)

        inter, union, gt_sum, pred_sum, iou, dice = iou_and_dice(gt_fg_32, pred_fg_32)

        left = add_title(Image.fromarray(rgb_crop), f"Obj Crop | {object_name}")
        mid = add_title(Image.fromarray(overlay_mask(rgb_crop, gt_fg, color=(0, 255, 0), alpha=args.alpha)), f"GT FG | {object_name}")
        right = add_title(Image.fromarray(overlay_mask(rgb_crop, pred_fg, color=(255, 0, 0), alpha=args.alpha)), f"Pred FG | {object_name}")

        footer_lines = [
            f"OBJECT={object_name}  |  image_id={image_id}  |  crop={idx}",
            f"idx={idx}",
            f"box={box}",
            f"n_present_parts={len(present_parts)}",
            f"GT_fg_px32={gt_sum} | Pred_fg_px32={pred_sum}",
            f"inter={inter} | union={union}",
            f"IoU={iou:.4f} | Dice={dice:.4f}",
        ]

        panel = hstack([left, mid, right], pad=10)
        panel = add_footer(panel, footer_lines)

        out_name = f"{idx:05d}_{image_id}.jpg"
        panel.save(trip_dir / out_name, quality=args.jpeg_quality)

        rows.append({
            "idx": idx,
            "object": object_name,
            "image_id": image_id,
            "out_name": out_name,
            "box_x1": box[0],
            "box_y1": box[1],
            "box_x2": box[2],
            "box_y2": box[3],
            "n_present_parts": len(present_parts),
            "gt_fg_px32": gt_sum,
            "pred_fg_px32": pred_sum,
            "inter_px32": inter,
            "union_px32": union,
            "iou": iou,
            "dice": dice,
        })
        saved += 1

        if saved <= 10 or saved % 500 == 0:
            print(f"[SAVE] {saved} {out_name} object={object_name} IoU={iou:.4f} Dice={dice:.4f}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "object", "image_id", "out_name",
            "box_x1", "box_y1", "box_x2", "box_y2",
            "n_present_parts",
            "gt_fg_px32", "pred_fg_px32",
            "inter_px32", "union_px32",
            "iou", "dice",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved {saved} triptychs to {trip_dir}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()
