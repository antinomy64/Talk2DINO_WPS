#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Ours on the complete Pascal-Part-116 split and save:

  pred_mask/         raw 0..115 prediction, invalid GT pixels = 255
  pred_color/        colored segmentation, invalid pixels = black
  pred_labeled/      colored segmentation + compact part labels
  overlay/           original image + color ONLY on valid pixels
  overlay_labeled/   overlay + compact part labels

IMPORTANT:
GT is used ONLY after inference to determine which pixels are visualized.
GT does NOT affect model input, text embeddings, logits, or argmax prediction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.io import read_image


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------
sys.path.insert(0, "src/open_vocabulary_segmentation")

from models import build_model
from segmentation.datasets.pascalpart116_part import PART_CLASSES

# Reuse the exact visualization/inference implementation already in repo.
from visualize_pp116_final import (
    infer_sliding,
    make_palette,
    colorize_label,
    _get_region_labels,
    _save_labeled_visualization,
    _short_part_name,
)


IGNORE_INDEX = 255
NUM_CLASSES = 116


def parse_args():
    p = argparse.ArgumentParser(
        "Whole-dataset PP-116 visualization for final Ours"
    )

    p.add_argument(
        "--data_root",
        default="data/PascalPart116",
    )

    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "all"],
        help="val = formal PP116 evaluation set; all = train + val",
    )

    p.add_argument(
        "--output_root",
        default="output/pp116_ours_segmentation",
    )

    p.add_argument(
        "--config",
        default=(
            "src/open_vocabulary_segmentation/configs/voc116_part/"
            "dinotext_voc116_part_vitb_mlp_infonce.yml"
        ),
    )

    p.add_argument(
        "--proj_name",
        required=True,
        help="Projector filename under weights/, WITHOUT .pth",
    )

    p.add_argument("--device", default="cuda:0")

    # Exact formal PP116 sliding inference setting.
    p.add_argument("--crop_size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)

    p.add_argument(
        "--template",
        default="sub_imagenet_template",
    )

    # Current paper protocol: PAMR off during final part segmentation.
    p.add_argument(
        "--pamr",
        action="store_true",
        help="Enable PAMR. Default OFF.",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.58,
        help="Color overlay opacity",
    )

    p.add_argument(
        "--label_min_area",
        type=int,
        default=40,
        help="Do not draw labels for very tiny predicted regions",
    )

    p.add_argument(
        "--font_size",
        type=float,
        default=7,
    )

    p.add_argument(
        "--full_labels",
        action="store_true",
        help="Use \"cat's head\" instead of compact \"head\"",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run images whose labeled output already exists",
    )

    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 = all images; useful for testing with e.g. --max_images 5",
    )

    return p.parse_args()


def read_split_ids(data_root: Path, split: str):
    split_file = data_root / f"{split}.txt"

    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    ids = []

    for line in split_file.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        # Robust to:
        #   2008_000008
        #   2008_000008.jpg
        #   images/val/2008_000008.jpg
        token = line.split()[0]
        ids.append(Path(token).stem)

    return ids


def prepare_output_dirs(root: Path, split: str):
    base = root / split

    dirs = {
        "mask_raw": base / "pred_mask_raw",
        "mask": base / "pred_mask",
        "color": base / "pred_color",
        "overlay": base / "overlay",
        "overlay_labeled": base / "overlay_labeled",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def compact_region_labels(region_labels, full_labels=False):
    if full_labels:
        return region_labels

    out = []

    for x, y, text, cid, area in region_labels:
        out.append(
            (
                x,
                y,
                _short_part_name(text),
                cid,
                area,
            )
        )

    return out


@torch.no_grad()
def process_one(
    model,
    text_emb,
    classnames,
    palette,
    image_path,
    gt_path,
    dirs,
    image_id,
    args,
):
    # ---------------------------------------------------------------
    # RGB image
    # ---------------------------------------------------------------
    image = read_image(str(image_path)).float()

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.shape[0] == 4:
        image = image[:3]

    H, W = image.shape[-2:]

    image_batch = image.unsqueeze(0).to(args.device)

    # ---------------------------------------------------------------
    # MODEL INFERENCE FIRST.
    #
    # Absolutely no GT information is used here.
    # ---------------------------------------------------------------
    score = infer_sliding(
        model=model,
        image=image_batch,
        text_emb=text_emb,
        classnames=classnames,
        crop_size=args.crop_size,
        stride=args.stride,
        apply_pamr=args.pamr,
    )

    if score.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} output channels, "
            f"got {tuple(score.shape)}"
        )

    pred = (
        score.argmax(dim=1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # ---------------------------------------------------------------
    # Load GT ONLY AFTER prediction.
    #
    # It is used only to identify valid/ignore pixels for visualization.
    # ---------------------------------------------------------------
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3:
        gt = gt[..., 0]

    gt = gt.astype(np.uint8)

    if gt.shape != (H, W):
        raise RuntimeError(
            f"{image_id}: image/GT size mismatch: "
            f"image={(H, W)}, gt={gt.shape}"
        )

    valid = gt != IGNORE_INDEX

    # ---------------------------------------------------------------
    # Mask prediction ONLY FOR SAVING/VISUALIZATION.
    # ---------------------------------------------------------------
    pred_valid = pred.copy()
    pred_valid[~valid] = IGNORE_INDEX

    pred_rgb, _ = colorize_label(
        pred_valid,
        palette,
        ignore_index=IGNORE_INDEX,
    )

    image_np = (
        image.permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    # ---------------------------------------------------------------
    # Overlay:
    #
    # valid pixel   -> segmentation color
    # invalid pixel -> untouched original RGB
    # ---------------------------------------------------------------
    overlay = image_np.astype(np.float32).copy()

    overlay[valid] = (
        (1.0 - args.alpha)
        * image_np[valid].astype(np.float32)
        +
        args.alpha
        * pred_rgb[valid].astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    # ---------------------------------------------------------------
    # Compact part labels
    #
    # Existing repo behavior:
    # one label on the largest connected component of each predicted
    # semantic part, positioned at an interior point.
    # ---------------------------------------------------------------
    region_labels = _get_region_labels(
        pred_valid,
        classnames,
        ignore_index=IGNORE_INDEX,
        min_area=args.label_min_area,
    )

    # Keep the complete PP116 semantic class name on pred_mask.
    # Example: "cat's head", "cat's leg", rather than only "head"/"leg".
    # region_labels returned by _get_region_labels already uses classnames.

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    raw_mask_path = dirs["mask_raw"] / f"{image_id}.png"
    mask_path = dirs["mask"] / f"{image_id}.png"
    color_path = dirs["color"] / f"{image_id}.png"
    overlay_path = dirs["overlay"] / f"{image_id}.png"
    overlay_labeled_path = (
        dirs["overlay_labeled"] / f"{image_id}.png"
    )

    # Raw class-index prediction for machine use.
    Image.fromarray(pred_valid).save(raw_mask_path)

    # Pure color mask without text.
    Image.fromarray(pred_rgb).save(color_path)

    # IMPORTANT:
    # pred_mask is the final human-readable segmentation result:
    # colored valid pixels + corresponding semantic part labels.
    _save_labeled_visualization(
        pred_rgb,
        region_labels,
        mask_path,
        fontsize=args.font_size,
    )

    Image.fromarray(overlay).save(overlay_path)

    _save_labeled_visualization(
        overlay,
        region_labels,
        overlay_labeled_path,
        fontsize=args.font_size,
    )

    valid_count = int(valid.sum())
    total_count = int(valid.size)

    pred_ids = np.unique(pred_valid[valid]).tolist()

    return {
        "valid": valid_count,
        "total": total_count,
        "num_pred_classes": len(pred_ids),
        "labels": len(region_labels),
    }


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if NUM_CLASSES != len(PART_CLASSES):
        raise RuntimeError(
            f"Expected 116 PART_CLASSES, got {len(PART_CLASSES)}"
        )

    # ---------------------------------------------------------------
    # Build model exactly as existing repo visualization.
    # ---------------------------------------------------------------
    cfg = OmegaConf.load(args.config)
    cfg.model.proj_name = args.proj_name

    print("=" * 88)
    print("PP-116 WHOLE-DATASET SEGMENTATION")
    print("=" * 88)
    print("projector :", args.proj_name)
    print("data root :", data_root)
    print("split     :", args.split)
    print("output    :", output_root)
    print("crop      :", args.crop_size)
    print("stride    :", args.stride)
    print("PAMR      :", args.pamr)
    print("classes   :", NUM_CLASSES)
    print("ignore    :", IGNORE_INDEX)
    print("=" * 88)

    device = torch.device(args.device)

    model = build_model(cfg.model)
    model = model.to(device).eval()

    # PP116 has NO extra predicted background class.
    if hasattr(model, "with_bg_clean"):
        model.with_bg_clean = False

    classnames = list(PART_CLASSES)

    # Exact PP116 prompt/text representation construction.
    with torch.no_grad():
        class_tokens = model.build_dataset_class_tokens(
            args.template,
            classnames,
        )

        text_emb = model.build_text_embedding(
            class_tokens
        )

    palette = make_palette(NUM_CLASSES)

    if args.split == "all":
        splits = ["train", "val"]
    else:
        splits = [args.split]

    global_done = 0
    global_total = 0

    for split in splits:
        ids = read_split_ids(
            data_root,
            split,
        )

        if args.max_images > 0:
            ids = ids[: args.max_images]

        global_total += len(ids)

        dirs = prepare_output_dirs(
            output_root,
            split,
        )

        image_root = data_root / "images" / split
        gt_root = (
            data_root
            / "annotations_detectron2_part"
            / split
        )

        print()
        print("=" * 88)
        print(f"SPLIT: {split}")
        print(f"images: {len(ids)}")
        print("=" * 88)

        for local_idx, image_id in enumerate(ids, 1):
            image_path = image_root / f"{image_id}.jpg"
            gt_path = gt_root / f"{image_id}.png"

            final_path = (
                dirs["overlay_labeled"]
                / f"{image_id}.png"
            )

            if (
                final_path.exists()
                and not args.overwrite
            ):
                global_done += 1

                print(
                    f"[{local_idx:04d}/{len(ids):04d}] "
                    f"SKIP {image_id}"
                )

                continue

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)

            print()
            print(
                f"[{local_idx:04d}/{len(ids):04d}] "
                f"{split}/{image_id}"
            )

            stat = process_one(
                model=model,
                text_emb=text_emb,
                classnames=classnames,
                palette=palette,
                image_path=image_path,
                gt_path=gt_path,
                dirs=dirs,
                image_id=image_id,
                args=args,
            )

            global_done += 1

            print(
                f"  valid pixels : "
                f"{stat['valid']}/{stat['total']} "
                f"({100.0 * stat['valid'] / stat['total']:.2f}%)"
            )

            print(
                f"  pred classes : "
                f"{stat['num_pred_classes']}"
            )

            print(
                f"  labels drawn : "
                f"{stat['labels']}"
            )

            print(
                f"  saved        : "
                f"{dirs['overlay_labeled'] / (image_id + '.png')}"
            )

            # Release per-image tensors.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print("processed :", global_done)
    print("requested :", global_total)
    print("output    :", output_root)


if __name__ == "__main__":
    main()
