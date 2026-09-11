#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize ALL ready PredObj crops for one image, using the exact same
prediction / crop construction flow as extract_predobj_cropaug.py.

For a given image_stem, this script automatically:
  1) derives image-present object labels and per-object part-presence labels
     from GT semantic masks (labels only, no GT spatial supervision afterward),
  2) runs the frozen Talk2DINO object predictor on the image-present object set,
  3) applies background competition,
  4) finds every object class that would produce a ready W-training annotation,
  5) visualizes ALL such objects without manual --target_class selection.

Output:
  - one summary panel showing the original image and every ready object crop,
  - optionally one separate panel per ready object.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--repo_root", default=".")
    p.add_argument("--image_root", default="data/PascalPart116/images/train")
    p.add_argument("--obj_mask_root", default="data/PascalPart116/annotations_detectron2_obj/train")
    p.add_argument("--part_mask_root", default="data/PascalPart116/annotations_detectron2_part/train")
    p.add_argument("--image_stem", required=True)

    p.add_argument("--obj_mask_mode", choices=("zero_based", "voc21"), default="zero_based")
    p.add_argument("--obj_ignore", type=int, default=255)
    p.add_argument("--part_id_offset", type=int, choices=(0, 1), default=0)
    p.add_argument("--part_ignore", type=int, default=255)

    p.add_argument("--projector_weight", default="weights/vitb_mlp_infonce_coco2014_clean_ft10_partstruct_w1e4_lr1e5.pth")
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--clip_model_name", default="ViT-B/16")
    p.add_argument("--proj_class", default="vitb_mlp_infonce")
    p.add_argument("--proj_model", default="ProjectionLayer")
    p.add_argument("--template", default="sub_imagenet_template")

    p.add_argument("--bg_thresh", type=float, default=0.54)
    p.add_argument("--lambda_bg", type=float, default=0.2)
    p.add_argument("--pamr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_max_long", type=int, default=2048)
    p.add_argument("--eval_max_short", type=int, default=448)
    p.add_argument("--slide_crop", type=int, default=448)
    p.add_argument("--slide_stride", type=int, default=224)
    p.add_argument("--crop_expand_ratio", type=float, default=1.2)
    p.add_argument("--device", default="cuda")

    p.add_argument("--overlay_alpha", type=float, default=0.45)
    p.add_argument("--summary_png", default=None)
    p.add_argument(
        "--save_individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save one PNG per ready object.",
    )
    p.add_argument("--individual_dir", default=None)
    return p.parse_args()


def resolve_path(repo: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color = np.zeros_like(out)
    color[..., 0] = 255.0
    m = mask.astype(bool)[..., None]
    out = np.where(m, (1.0 - alpha) * out + alpha * color, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_title_lines(parts: list[str], max_items: int = 6) -> str:
    if len(parts) <= max_items:
        return ", ".join(parts)
    return ", ".join(parts[:max_items]) + f", ... (+{len(parts)-max_items})"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from extract_predobj_cropaug import (
        VOC20_CLASSES,
        VISION_DIM,
        PATCH_GRID,
        suffix_list,
        find_by_stem,
        load_rgb,
        load_label_mask,
        decode_present_objects,
        gt_object_region,
        derive_part_presence,
        build_model,
        talk2dino_slide_scores,
        square_crop_box,
        mask_to_patch_grid,
    )

    image_root = resolve_path(repo, args.image_root)
    obj_root = resolve_path(repo, args.obj_mask_root)
    part_root = resolve_path(repo, args.part_mask_root)

    image_suffixes = suffix_list(".jpg,.jpeg,.png")
    mask_suffixes = suffix_list(".png")

    img_path = find_by_stem(image_root, args.image_stem, image_suffixes)
    obj_path = find_by_stem(obj_root, args.image_stem, mask_suffixes)
    part_path = find_by_stem(part_root, args.image_stem, mask_suffixes)
    if img_path is None:
        raise FileNotFoundError(f"image not found: {args.image_stem}")
    if obj_path is None:
        raise FileNotFoundError(f"object mask not found: {args.image_stem}")
    if part_path is None:
        raise FileNotFoundError(f"part mask not found: {args.image_stem}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    pil = load_rgb(img_path)
    obj_mask = load_label_mask(obj_path)
    part_mask = load_label_mask(part_path)
    if obj_mask.shape != (pil.height, pil.width):
        raise ValueError(f"object mask shape {obj_mask.shape} != RGB {(pil.height, pil.width)}")
    if part_mask.shape != (pil.height, pil.width):
        raise ValueError(f"part mask shape {part_mask.shape} != RGB {(pil.height, pil.width)}")

    object_ids = decode_present_objects(obj_mask, args.obj_mask_mode, args.obj_ignore)
    if not object_ids:
        raise ValueError("no image-present object labels")
    object_names = [VOC20_CLASSES[c] for c in object_ids]

    weak_parts = {}
    for cid in object_ids:
        name = VOC20_CLASSES[cid]
        region = gt_object_region(obj_mask, cid, args.obj_mask_mode)
        pids, pnames, foreign = derive_part_presence(
            part_mask,
            region,
            name,
            part_id_offset=args.part_id_offset,
            part_ignore=args.part_ignore,
        )
        weak_parts[cid] = (pids, pnames, foreign)

    model, weight = build_model(args, repo, device)
    object_tokens = model.build_dataset_class_tokens(args.template, VOC20_CLASSES)
    object_text = model.build_text_embedding(object_tokens).to(device=device, dtype=torch.float32)
    if tuple(object_text.shape) != (20, VISION_DIM):
        raise ValueError(f"bad object text bank shape: {tuple(object_text.shape)}")

    present_text = object_text[torch.tensor(object_ids, dtype=torch.long, device=device)]
    fg_scores = talk2dino_slide_scores(
        model,
        pil,
        present_text,
        object_names,
        device=device,
        pamr=args.pamr,
        lambda_bg=args.lambda_bg,
        eval_max_long=args.eval_max_long,
        eval_max_short=args.eval_max_short,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
    )
    background = torch.full(
        (1, pil.height, pil.width),
        float(args.bg_thresh),
        device=fg_scores.device,
        dtype=fg_scores.dtype,
    )
    hard = torch.cat([background, fg_scores], dim=0).argmax(dim=0)

    ready = []
    for local_channel, cid in enumerate(object_ids, start=1):
        pids, pnames, foreign = weak_parts[cid]
        if not pids:
            continue
        pred = (hard == local_channel).detach().cpu().numpy().astype(np.uint8)
        if int(pred.sum()) == 0:
            continue
        box = square_crop_box(
            pred,
            width=pil.width,
            height=pil.height,
            expand_ratio=args.crop_expand_ratio,
        )
        patch_mask = mask_to_patch_grid(pred, box).reshape(PATCH_GRID, PATCH_GRID).cpu().numpy().astype(bool)
        x1, y1, x2, y2 = box
        crop_pil = pil.crop((x1, y1, x2, y2))
        crop_448 = crop_pil.resize((448, 448), resample=getattr(Image, "Resampling", Image).BICUBIC)
        crop_mask_pixel = pred[y1:y2, x1:x2]
        crop_mask_vis = np.asarray(
            Image.fromarray((crop_mask_pixel * 255).astype(np.uint8), mode="L").resize(
                (448, 448),
                resample=getattr(Image, "Resampling", Image).NEAREST,
            )
        ) > 0
        ready.append({
            "cid": int(cid),
            "class_name": VOC20_CLASSES[cid],
            "pids": list(pids),
            "pnames": list(pnames),
            "pred": pred,
            "box": (x1, y1, x2, y2),
            "patch_mask": patch_mask,
            "crop_pil": crop_pil,
            "crop_448": crop_448,
            "crop_mask_vis": crop_mask_vis,
        })

    if not ready:
        raise RuntimeError("this image creates no ready PredObj W-training crop")

    rgb = np.asarray(pil, dtype=np.uint8)

    # Save one summary figure.
    n = len(ready)
    cols = 3
    rows = math.ceil((n + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.6 * rows))
    axes = np.array(axes).reshape(rows, cols)

    ax0 = axes.flat[0]
    ax0.imshow(rgb)
    for item in ready:
        x1, y1, x2, y2 = item["box"]
        ax0.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        ax0.text(x1, max(0, y1 - 3), item["class_name"], fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
    ax0.set_title(
        f"Original image\nimage-present objects={object_names}\nready W-training crops={[x['class_name'] for x in ready]}"
    )
    ax0.axis('off')

    for idx, item in enumerate(ready, start=1):
        ax = axes.flat[idx]
        overlay = overlay_mask(rgb, item["pred"], args.overlay_alpha)
        x1, y1, x2, y2 = item["box"]
        ax.imshow(overlay)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        ax.set_title(
            f"{item['class_name']}\n"
            f"pixel={int(item['pred'].sum())}, patch={int(item['patch_mask'].sum())}\n"
            f"parts: {to_title_lines(item['pnames'])}"
        )
        ax.axis('off')

    for j in range(n + 1, rows * cols):
        axes.flat[j].axis('off')

    fig.suptitle(
        f"{args.image_stem} | projector={Path(weight).name} | bg_thresh={args.bg_thresh} | PAMR={args.pamr}",
        fontsize=13,
    )
    fig.tight_layout()

    if args.summary_png is None:
        summary_path = repo / 'final_exp' / 'relproto_alignment' / 'visualize_predobj_all' / f'{args.image_stem}__all_ready_objects.png'
    else:
        summary_path = resolve_path(repo, args.summary_png)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Optionally save one figure per object with more detail.
    individual_paths = []
    if args.save_individual:
        if args.individual_dir is None:
            individual_dir = repo / 'final_exp' / 'relproto_alignment' / 'visualize_predobj_all' / args.image_stem
        else:
            individual_dir = resolve_path(repo, args.individual_dir)
        individual_dir.mkdir(parents=True, exist_ok=True)

        for item in ready:
            fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.8))
            x1, y1, x2, y2 = item['box']
            overlay = overlay_mask(rgb, item['pred'], args.overlay_alpha)
            crop_rgb = np.asarray(item['crop_pil'])
            crop448_rgb = np.asarray(item['crop_448'])
            crop_overlay = overlay_mask(crop448_rgb, item['crop_mask_vis'].astype(np.uint8), args.overlay_alpha)

            axes2[0].imshow(rgb)
            axes2[0].add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.5))
            axes2[0].set_title('Original RGB + crop box')
            axes2[0].axis('off')

            axes2[1].imshow(overlay)
            axes2[1].add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.5))
            axes2[1].set_title(f"Predicted mask overlay\npixel area={int(item['pred'].sum())}")
            axes2[1].axis('off')

            axes2[2].imshow(crop_overlay)
            axes2[2].set_title(f"Exact crop -> resize 448×448\nbox={item['box']}")
            axes2[2].axis('off')

            axes2[3].imshow(item['patch_mask'], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axes2[3].set_title(f"pred_obj_mask_patch 32×32\npatch area={int(item['patch_mask'].sum())}")
            axes2[3].axis('off')

            fig2.suptitle(
                f"{args.image_stem} | {item['class_name']} | visible parts={item['pnames']}",
                fontsize=12,
            )
            fig2.tight_layout()
            out_path = individual_dir / f"{args.image_stem}__{item['class_name']}.png"
            fig2.savefig(out_path, dpi=180, bbox_inches='tight')
            plt.close(fig2)
            individual_paths.append(str(out_path))

    print("============================================================")
    print("ALL-OBJECT PredObj visualization")
    print("============================================================")
    print("image_stem         :", args.image_stem)
    print("objects_present    :", object_names)
    print("ready_object_count :", len(ready))
    print("ready_objects      :", [x['class_name'] for x in ready])
    print("bg_thresh          :", args.bg_thresh)
    print("PAMR               :", args.pamr)
    print("summary_png        :", summary_path)
    if individual_paths:
        print("individual_pngs    :")
        for p in individual_paths:
            print("  -", p)
    print("============================================================")


if __name__ == "__main__":
    main()
