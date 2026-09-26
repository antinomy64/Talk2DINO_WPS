#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.io import read_image

sys.path.insert(0, "src/open_vocabulary_segmentation")

from models import build_model
from segmentation.datasets.pascalpart116_part import PART_CLASSES

from visualize_pp116_final import (
    infer_sliding,
    make_palette,
    colorize_label,
    _get_region_labels,
    _save_labeled_visualization,
)

IGNORE_INDEX = 255
NUM_CLASSES = 116


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        default="data/PascalPart116",
    )

    p.add_argument(
        "--output_root",
        default="output/qualitative_pp116_850",
    )

    p.add_argument(
        "--config",
        default=(
            "src/open_vocabulary_segmentation/configs/voc116_part/"
            "dinotext_voc116_part_vitb_mlp_infonce.yml"
        ),
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop_size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)

    p.add_argument(
        "--template",
        default="sub_imagenet_template",
    )

    p.add_argument(
        "--min_label_area",
        type=int,
        default=40,
    )

    p.add_argument(
        "--font_size",
        type=float,
        default=7,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 means all 850; useful for smoke test.",
    )

    return p.parse_args()


def collect_images(data_root):
    image_root = data_root / "images" / "val"
    gt_root = data_root / "annotations_detectron2_part" / "val"

    files = {}

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in image_root.glob(ext):
            files[p.stem] = p

    ids = sorted(files)

    if len(ids) != 850:
        raise RuntimeError(
            f"Expected exactly 850 PP116 val images, found {len(ids)} "
            f"in {image_root}"
        )

    samples = []

    for image_id in ids:
        img = files[image_id]
        gt = gt_root / f"{image_id}.png"

        if not gt.is_file():
            raise FileNotFoundError(gt)

        samples.append((image_id, img, gt))

    return samples


def build_pp116_model(config_path, proj_name, device, template):
    cfg = OmegaConf.load(config_path)

    cfg.model.proj_name = proj_name

    print()
    print("=" * 90)
    print("Loading projector:", proj_name)
    print("=" * 90)

    model = build_model(cfg.model)
    model = model.to(device).eval()

    # PP116 = exactly 116 semantic part channels.
    # No additional background prediction class.
    if hasattr(model, "with_bg_clean"):
        model.with_bg_clean = False

    classnames = list(PART_CLASSES)

    if len(classnames) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} PP116 classes, "
            f"got {len(classnames)}"
        )

    with torch.no_grad():
        class_tokens = model.build_dataset_class_tokens(
            template,
            classnames,
        )

        text_emb = model.build_text_embedding(
            class_tokens
        )

    return model, text_emb, classnames


def save_gt_once(
    image_id,
    gt,
    palette,
    classnames,
    out_root,
    min_area,
    fontsize,
):
    gt_dir = out_root / "gt"
    gt_raw_dir = out_root / "gt_raw"

    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_raw_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = gt_dir / f"{image_id}.png"
    raw_path = gt_raw_dir / f"{image_id}.png"

    if labeled_path.exists() and raw_path.exists():
        return

    Image.fromarray(gt).save(raw_path)

    gt_rgb, _ = colorize_label(
        gt,
        palette,
        ignore_index=IGNORE_INDEX,
    )

    gt_labels = _get_region_labels(
        gt,
        classnames,
        ignore_index=IGNORE_INDEX,
        min_area=min_area,
    )

    _save_labeled_visualization(
        gt_rgb,
        gt_labels,
        labeled_path,
        fontsize=fontsize,
    )


@torch.no_grad()
def run_one_model(
    method_name,
    proj_name,
    samples,
    args,
    palette,
):
    device = torch.device(args.device)
    root = Path(args.output_root)

    model, text_emb, classnames = build_pp116_model(
        args.config,
        proj_name,
        device,
        args.template,
    )

    method_root = root / method_name

    raw_dir = method_root / "pred_mask_raw"
    pred_dir = method_root / "pred_mask"
    color_dir = method_root / "pred_color"

    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    if args.max_images > 0:
        current_samples = samples[:args.max_images]
    else:
        current_samples = samples

    total = len(current_samples)

    for idx, (image_id, image_path, gt_path) in enumerate(
        current_samples,
        1,
    ):
        final_path = pred_dir / f"{image_id}.png"

        print()
        print("=" * 90)
        print(
            f"[{method_name}] "
            f"{idx:03d}/{total:03d}  {image_id}"
        )
        print("=" * 90)

        if final_path.exists() and not args.overwrite:
            print("SKIP:", final_path)
            continue

        # ----------------------------------------------------------
        # Image
        # ----------------------------------------------------------
        image = read_image(
            str(image_path)
        ).float()

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] == 4:
            image = image[:3]

        H, W = image.shape[-2:]

        image_batch = (
            image.unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------------
        # FULL IMAGE MODEL PREDICTION
        #
        # No GT information enters this step.
        # Same 448 / 224 sliding inference as final visualization.
        # Final PAMR OFF.
        # ----------------------------------------------------------
        score = infer_sliding(
            model=model,
            image=image_batch,
            text_emb=text_emb,
            classnames=classnames,
            crop_size=args.crop_size,
            stride=args.stride,
            apply_pamr=False,
        )

        if score.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"{image_id}: expected 116 channels, "
                f"got {tuple(score.shape)}"
            )

        pred = (
            score.argmax(dim=1)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # ----------------------------------------------------------
        # GT loaded only AFTER prediction.
        # It controls valid-pixel visualization only.
        # ----------------------------------------------------------
        gt = np.array(
            Image.open(gt_path)
        )

        if gt.ndim == 3:
            gt = gt[..., 0]

        gt = gt.astype(np.uint8)

        if gt.shape != (H, W):
            raise RuntimeError(
                f"{image_id}: image size {(H, W)} "
                f"!= GT size {gt.shape}"
            )

        valid = gt != IGNORE_INDEX

        # Keep model prediction unchanged on every valid pixel.
        # Ignore/non-valid area -> 255 only for saved visualization.
        pred_valid = pred.copy()
        pred_valid[~valid] = IGNORE_INDEX

        # ----------------------------------------------------------
        # Save GT once.
        # ----------------------------------------------------------
        save_gt_once(
            image_id=image_id,
            gt=gt,
            palette=palette,
            classnames=classnames,
            out_root=root,
            min_area=args.min_label_area,
            fontsize=args.font_size,
        )

        # ----------------------------------------------------------
        # RAW MASK
        # 0..115 = semantic part IDs
        # 255    = invalid/ignore
        # ----------------------------------------------------------
        Image.fromarray(
            pred_valid
        ).save(
            raw_dir / f"{image_id}.png"
        )

        # ----------------------------------------------------------
        # COLOR MASK
        # ----------------------------------------------------------
        pred_rgb, _ = colorize_label(
            pred_valid,
            palette,
            ignore_index=IGNORE_INDEX,
        )

        Image.fromarray(
            pred_rgb
        ).save(
            color_dir / f"{image_id}.png"
        )

        # ----------------------------------------------------------
        # FINAL QUALITATIVE PRED_MASK
        #
        # Base image: original RGB image.
        #
        # GT != 255:
        #     replace RGB completely with predicted semantic color.
        #
        # GT == 255:
        #     keep the original RGB image unchanged.
        #
        # Then draw the complete PP116 semantic label on each
        # sufficiently large predicted semantic region.
        # ----------------------------------------------------------

        image_np = (
            image.permute(1, 2, 0)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Start from ORIGINAL IMAGE.
        overlay = image_np.copy()

        # 100% opaque prediction color on valid pixels.
        overlay[valid] = pred_rgb[valid]

        # Semantic labels from predicted segmentation.
        region_labels = _get_region_labels(
            pred_valid,
            classnames,
            ignore_index=IGNORE_INDEX,
            min_area=args.min_label_area,
        )

        # IMPORTANT:
        # pred_mask/*.png = ORIGINAL IMAGE
        #                 + opaque prediction on valid pixels
        #                 + semantic labels
        _save_labeled_visualization(
            overlay,
            region_labels,
            final_path,
            fontsize=args.font_size,
        )

        pred_classes = sorted(
            int(x)
            for x in np.unique(pred_valid[valid])
        )

        print(
            "valid pixels:",
            int(valid.sum()),
            "/",
            int(valid.size),
        )

        print(
            "pred semantic classes:",
            len(pred_classes),
        )

        print(
            "labels drawn:",
            len(region_labels),
        )

        print(
            "saved:",
            final_path,
        )

        del score
        del image_batch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free first model before loading second.
    del text_emb
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = collect_images(
        data_root
    )

    print("=" * 90)
    print("PP116 QUALITATIVE GENERATION")
    print("=" * 90)
    print("val images :", len(samples))
    print("crop       :", args.crop_size)
    print("stride     :", args.stride)
    print("final PAMR : False")
    print("valid      : GT != 255")
    print("labels     : full PP116 semantic names")
    print("output     :", out_root)
    print("=" * 90)

    palette = make_palette(
        NUM_CLASSES
    )

    # --------------------------------------------------------------
    # 1. Main-comparison Talk2DINO-Part
    #    NOT the 21.70 unadapted ablation baseline.
    # --------------------------------------------------------------
    run_one_model(
        method_name="talk2dino_part",
        proj_name=(
            "vitb_mlp_infonce_pp116_continue_"
            "pp116_official_continue_seed123"
        ),
        samples=samples,
        args=args,
        palette=palette,
    )

    # --------------------------------------------------------------
    # 2. Ours: Relative + Orthogonal, L=8, epoch 30
    # --------------------------------------------------------------
    run_one_model(
        method_name="ours",
        proj_name="ours_pp116_relative_orthogonal_L8_e030",
        samples=samples,
        args=args,
        palette=palette,
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

    expected = (
        args.max_images
        if args.max_images > 0
        else 850
    )

    for name in (
        "talk2dino_part",
        "ours",
    ):
        n = len(
            list(
                (
                    out_root
                    / name
                    / "pred_mask"
                ).glob("*.png")
            )
        )

        print(
            f"{name:18s}: "
            f"{n}/{expected}"
        )


if __name__ == "__main__":
    main()
