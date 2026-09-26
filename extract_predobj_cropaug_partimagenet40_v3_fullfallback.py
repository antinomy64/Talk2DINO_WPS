#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PartImageNet-40 predicted-object cache.

This file deliberately reuses the final Pascal-Part-116
object-prediction/crop implementation from:

    extract_predobj_cropaug.py

Dataset-specific changes ONLY:
    - weak labels come from prebuilt image-level manifest
    - actual ImageNet object class is used for Talk2DINO foreground
    - PartImageNet superclass is kept separately for part grouping
    - 40 semantic part classes

No target object/part mask, point, or box is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from tqdm import tqdm

from partimagenet40_taxonomy import (
    PART_CLASSES,
    OBJECT_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)

# ------------------------------------------------------------
# Reuse the PP116 pipeline implementation EXACTLY.
# ------------------------------------------------------------

from extract_predobj_cropaug import (
    VISION_DIM,
    PATCH_COUNT,
    build_model,
    talk2dino_slide_scores,
    square_crop_box,
    mask_to_patch_grid,
    crop_transform,
    load_rgb,
    atomic_torch_save,
    sha256,
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "PartImageNet-40 predicted-object crop cache "
            "based on the PP116 Ours pipeline."
        ),
    )

    p.add_argument(
        "--repo_root",
        default=".",
    )

    p.add_argument(
        "--image_root",
        default="",
    )

    p.add_argument(
        "--presence_manifest",
        default="",
        help=(
            "partimagenet_train_weak_v2.json "
            "containing actual object class + part presence."
        ),
    )

    p.add_argument(
        "--output_pth",
        default="",
    )

    # Same models / geometry as PP116.
    p.add_argument(
        "--projector_weight",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_reproduce_clean.pth"
        ),
    )

    p.add_argument(
        "--model_name",
        default="dinov2_vitb14_reg",
    )

    p.add_argument(
        "--clip_model_name",
        default="ViT-B/16",
    )

    p.add_argument(
        "--proj_class",
        default="vitb_mlp_infonce",
    )

    p.add_argument(
        "--proj_model",
        default="ProjectionLayer",
    )

    p.add_argument(
        "--template",
        default="sub_imagenet_template",
    )

    p.add_argument(
        "--bg_thresh",
        type=float,
        default=0.55,
    )

    p.add_argument(
        "--lambda_bg",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--pamr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--eval_max_long",
        type=int,
        default=2048,
    )

    p.add_argument(
        "--eval_max_short",
        type=int,
        default=448,
    )

    p.add_argument(
        "--slide_crop",
        type=int,
        default=448,
    )

    p.add_argument(
        "--slide_stride",
        type=int,
        default=224,
    )

    p.add_argument(
        "--crop_expand_ratio",
        type=float,
        default=1.2,
    )

    p.add_argument(
        "--crop_batch_size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--max_images",
        type=int,
        default=0,
    )

    p.add_argument(
        "--self_test",
        action="store_true",
    )

    return p.parse_args()


def self_test():
    assert len(PART_CLASSES) == 40
    assert len(OBJECT_CLASSES) == 11
    assert set(PART_TO_OBJECT) == set(range(40))

    for group, pids in OBJECT_GROUPS.items():
        assert group in OBJECT_CLASSES

        for pid in pids:
            obj_id = PART_TO_OBJECT[pid]
            assert OBJECT_CLASSES[obj_id] == group

    print("SELF_TEST_PASS")


def load_manifest(path: Path):
    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(obj, list):
        raise TypeError(
            "PartImageNet v2 manifest must be a JSON list"
        )

    rows = []

    for i, rec in enumerate(obj):
        image = str(rec["image"])
        wnid = str(rec["object_wnid"])
        object_name = str(rec["object_name"]).strip()
        group = str(rec["part_parent_group"]).strip()

        pids = [
            int(x)
            for x in rec["part_ids"]
        ]

        if not object_name:
            raise ValueError(
                f"row {i}: empty object_name"
            )

        if group not in OBJECT_GROUPS:
            raise ValueError(
                f"row {i}: unknown part group {group}"
            )

        if not pids:
            raise ValueError(
                f"row {i}: empty part IDs"
            )

        if len(pids) != len(set(pids)):
            raise ValueError(
                f"row {i}: duplicate part IDs"
            )

        allowed = set(
            OBJECT_GROUPS[group]
        )

        if not set(pids).issubset(allowed):
            raise ValueError(
                f"row {i}: {pids} not in group "
                f"{group}: {sorted(allowed)}"
            )

        expected_group_ids = {
            int(PART_TO_OBJECT[x])
            for x in pids
        }

        if len(expected_group_ids) != 1:
            raise ValueError(
                f"row {i}: cross-group parts"
            )

        expected_group = OBJECT_CLASSES[
            next(iter(expected_group_ids))
        ]

        if expected_group != group:
            raise ValueError(
                f"row {i}: manifest group={group}, "
                f"taxonomy={expected_group}"
            )

        rows.append({
            "image": image,
            "object_wnid": wnid,
            "object_name": object_name,
            "part_parent_group": group,
            "part_ids": pids,
            "part_names": [
                PART_CLASSES[x]
                for x in pids
            ],
        })

    return rows


@torch.inference_mode()
def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return

    for name in (
        "image_root",
        "presence_manifest",
        "output_pth",
    ):
        if not getattr(args, name):
            raise ValueError(
                f"--{name} is required unless --self_test is used"
            )

    if args.model_name != "dinov2_vitb14_reg":
        raise ValueError(
            "final pipeline requires dinov2_vitb14_reg"
        )

    if args.clip_model_name != "ViT-B/16":
        raise ValueError(
            "final pipeline requires CLIP ViT-B/16"
        )

    if (
        args.eval_max_long,
        args.eval_max_short,
    ) != (2048, 448):
        raise ValueError(
            "final object path requires img_scale=(2048,448)"
        )

    if (
        args.slide_crop,
        args.slide_stride,
    ) != (448, 224):
        raise ValueError(
            "final object path requires "
            "slide crop=448 stride=224"
        )

    if (
        not math.isfinite(args.crop_expand_ratio)
        or args.crop_expand_ratio < 1.0
    ):
        raise ValueError(
            "crop_expand_ratio must be >= 1"
        )

    repo = Path(
        args.repo_root
    ).expanduser().resolve()

    image_root = Path(
        args.image_root
    ).expanduser().resolve()

    manifest_path = Path(
        args.presence_manifest
    ).expanduser().resolve()

    output = Path(
        args.output_pth
    ).expanduser()

    if not output.is_absolute():
        output = (repo / output).resolve()

    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    rows = load_manifest(
        manifest_path
    )

    if args.max_images > 0:
        rows = rows[:args.max_images]

    # --------------------------------------------------------
    # Build actual ImageNet object vocabulary.
    #
    # Important:
    #   object class != PartImageNet parent group.
    # --------------------------------------------------------

    wnid_to_name = {}

    for rec in rows:
        wnid = rec["object_wnid"]
        name = rec["object_name"]

        if (
            wnid in wnid_to_name
            and wnid_to_name[wnid] != name
        ):
            raise ValueError(
                f"{wnid}: inconsistent object names "
                f"{wnid_to_name[wnid]!r} vs {name!r}"
            )

        wnid_to_name[wnid] = name

    object_wnids = sorted(
        wnid_to_name
    )

    object_names = [
        wnid_to_name[x]
        for x in object_wnids
    ]

    wnid_to_row = {
        wnid: i
        for i, wnid in enumerate(object_wnids)
    }

    print(
        "[manifest]",
        "images=", len(rows),
        "actual_object_classes=", len(object_wnids),
    )

    for rec in rows[:10]:
        print(
            " ",
            rec["image"],
            "| query =", rec["object_name"],
            "| group =", rec["part_parent_group"],
            "| parts =", rec["part_ids"],
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    # --------------------------------------------------------
    # Exact PP116 Talk2DINO model construction.
    # --------------------------------------------------------

    model, weight = build_model(
        args,
        repo,
        device,
    )

    # Fixed full actual-object text bank, same principle as
    # PP116's fixed VOC object text bank.
    object_tokens = (
        model.build_dataset_class_tokens(
            args.template,
            object_names,
        )
    )

    object_text = (
        model.build_text_embedding(
            object_tokens
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    expected = (
        len(object_names),
        VISION_DIM,
    )

    if tuple(object_text.shape) != expected:
        raise ValueError(
            f"bad object text bank "
            f"{tuple(object_text.shape)} != {expected}"
        )

    transform = crop_transform()

    annotations = []
    images_meta = []

    pending_crops = []
    pending_indices = []

    stats = defaultdict(int)

    start_time = time.perf_counter()

    def flush():
        nonlocal pending_crops
        nonlocal pending_indices

        if not pending_crops:
            return

        batch = torch.stack([
            transform(im)
            for im in pending_crops
        ]).to(
            device=device,
            dtype=torch.float32,
        )

        dino_out = model.model(
            batch,
            is_training=True,
        )

        if (
            not isinstance(dino_out, Mapping)
            or "x_norm_patchtokens"
            not in dino_out
        ):
            raise RuntimeError(
                "DINOv2 output missing "
                "x_norm_patchtokens"
            )

        tokens = dino_out[
            "x_norm_patchtokens"
        ]

        expected_shape = (
            len(pending_crops),
            PATCH_COUNT,
            VISION_DIM,
        )

        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"DINO crop tokens "
                f"{tuple(tokens.shape)} "
                f"!= {expected_shape}"
            )

        if not torch.isfinite(tokens).all():
            raise ValueError(
                "NaN/Inf in DINO crop tokens"
            )

        for j, ann_idx in enumerate(
            pending_indices
        ):
            annotations[
                ann_idx
            ]["cropaug_patch_tokens"] = (
                tokens[j]
                .to(
                    device="cpu",
                    dtype=torch.float16,
                )
                .contiguous()
            )

        stats["crop_batches"] += 1

        pending_crops = []
        pending_indices = []

    # --------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------

    for rec in tqdm(
        rows,
        desc=(
            "actual object -> pred obj -> "
            "crop -> DINO cache"
        ),
    ):
        stats["requested_images"] += 1

        rel = Path(
            rec["image"]
        )

        img_path = (
            rel
            if rel.is_absolute()
            else image_root / rel.name
        )

        if not img_path.is_file():
            stats["missing_image"] += 1
            continue

        pil = load_rgb(
            img_path
        )

        object_wnid = rec[
            "object_wnid"
        ]

        object_name = rec[
            "object_name"
        ]

        parent_group = rec[
            "part_parent_group"
        ]

        pids = list(
            rec["part_ids"]
        )

        pnames = list(
            rec["part_names"]
        )

        object_row = wnid_to_row[
            object_wnid
        ]

        # PartImageNet image has one known actual ImageNet object
        # category.  Compete that object against background.
        present_text = object_text[
            object_row:object_row + 1
        ]

        fg_scores = talk2dino_slide_scores(
            model,
            pil,
            present_text,
            [object_name],
            device=device,
            pamr=args.pamr,
            lambda_bg=args.lambda_bg,
            eval_max_long=args.eval_max_long,
            eval_max_short=args.eval_max_short,
            slide_crop=args.slide_crop,
            slide_stride=args.slide_stride,
        )

        if tuple(fg_scores.shape) != (
            1,
            pil.height,
            pil.width,
        ):
            raise ValueError(
                f"{rec['image']}: bad fg score "
                f"shape {tuple(fg_scores.shape)}"
            )

        background = torch.full(
            (1, pil.height, pil.width),
            float(args.bg_thresh),
            device=fg_scores.device,
            dtype=fg_scores.dtype,
        )

        # local 0 = background
        # local 1 = actual ImageNet object
        hard = torch.cat(
            [background, fg_scores],
            dim=0,
        ).argmax(dim=0)

        pred = (
            (hard == 1)
            .detach()
            .cpu()
            .numpy()
            .astype("uint8")
        )

        full_image_fallback = False

        if int(pred.sum()) <= 0:
            stats[
                "empty_predicted_object_mask"
            ] = (
                stats.get(
                    "empty_predicted_object_mask",
                    0,
                )
                + 1
            )

            stats[
                "full_image_fallback"
            ] = (
                stats.get(
                    "full_image_fallback",
                    0,
                )
                + 1
            )

            full_image_fallback = True

            # Frozen Talk2DINO failed to return a valid
            # object foreground. Do NOT discard this
            # image. Fall back to the full image.
            #
            # Preserve the original mask type/shape.
            pred[...] = True

        if full_image_fallback:
            # Full-image fallback: use every image region.
            box = (
                0,
                0,
                pil.width,
                pil.height,
            )
        else:
            box = square_crop_box(
                pred,
                width=pil.width,
                height=pil.height,
                expand_ratio=args.crop_expand_ratio,
            )

        patch_mask = mask_to_patch_grid(
            pred,
            box,
        )

        x1, y1, x2, y2 = box

        crop = pil.crop(
            (x1, y1, x2, y2)
        )

        record = {
            "id": len(annotations),

            "image_id": Path(
                rec["image"]
            ).stem,

            # Object category row in the current full
            # actual-object vocabulary.
            "category_id": int(
                object_row
            ),

            # Keep class_name semantically equal to the
            # actual foreground object, just like PP116.
            "class_name": object_name,

            "foreground_object_wnid":
                object_wnid,

            "foreground_object_name":
                object_name,

            # Separate relational parent taxonomy.
            "part_parent_group":
                parent_group,

            # Image-level present parts.
            "part_category_id":
                pids,

            "part_class_name":
                pnames,

            # Everything below comes only from the
            # predicted Talk2DINO object mask.
            "cropaug_box_xyxy":
                torch.tensor(
                    box,
                    dtype=torch.long,
                ),

            "pred_obj_mask_patch":
                patch_mask.contiguous(),

            "pred_obj_mask_pixel_area":
                int(pred.sum()),

            "pred_obj_mask_patch_area":
                int(
                    patch_mask.sum().item()
                ),

            "pred_obj_mask_source":
                (
                    "talk2dino_"
                    "image_level_actual_object_label"
                ),
        }

        record[
            "full_image_fallback"
        ] = bool(
            full_image_fallback
        )

        if full_image_fallback:
            record[
                "pred_obj_mask_source"
            ] = (
                "full_image_all_patches_fallback"
            )

        annotations.append(
            record
        )

        pending_crops.append(
            crop
        )

        pending_indices.append(
            len(annotations) - 1
        )

        stats[
            "ready_object_crops"
        ] += 1

        if (
            len(pending_crops)
            >= args.crop_batch_size
        ):
            flush()

        images_meta.append({
            "id": Path(
                rec["image"]
            ).stem,
            "file_name": str(img_path),
            "height": pil.height,
            "width": pil.width,

            "foreground_object_wnid":
                object_wnid,

            "foreground_object_name":
                object_name,

            "part_parent_group":
                parent_group,

            "part_category_id":
                pids,

            "ready": True,
        })

        stats["processed_images"] += 1

    flush()

    # --------------------------------------------------------
    # Final cache contract.
    # --------------------------------------------------------

    for i, ann in enumerate(
        annotations
    ):
        tok = ann.get(
            "cropaug_patch_tokens"
        )

        pm = ann.get(
            "pred_obj_mask_patch"
        )

        box = ann.get(
            "cropaug_box_xyxy"
        )

        pids = ann.get(
            "part_category_id",
            [],
        )

        if (
            not torch.is_tensor(tok)
            or tuple(tok.shape)
            != (1024, 768)
            or tok.dtype
            != torch.float16
        ):
            raise ValueError(
                f"annotation {i}: "
                "invalid cropaug_patch_tokens"
            )

        if (
            not torch.is_tensor(pm)
            or tuple(pm.shape)
            != (1024,)
            or pm.dtype
            != torch.bool
            or int(pm.sum()) <= 0
        ):
            raise ValueError(
                f"annotation {i}: "
                "invalid pred_obj_mask_patch"
            )

        if (
            not torch.is_tensor(box)
            or tuple(box.shape)
            != (4,)
            or box.dtype
            != torch.long
        ):
            raise ValueError(
                f"annotation {i}: "
                "invalid cropaug_box_xyxy"
            )

        if (
            not pids
            or len(pids)
            != len(set(pids))
        ):
            raise ValueError(
                f"annotation {i}: "
                "invalid part IDs"
            )

        group = ann[
            "part_parent_group"
        ]

        allowed = set(
            OBJECT_GROUPS[group]
        )

        if any(
            pid not in allowed
            for pid in pids
        ):
            raise ValueError(
                f"annotation {i}: "
                f"parts {pids} do not belong "
                f"to {group}"
            )

        # Critical semantic invariant.
        if not ann[
            "foreground_object_name"
        ]:
            raise ValueError(
                f"annotation {i}: "
                "missing actual object name"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    stats = dict(stats)

    stats[
        "output_annotations"
    ] = len(annotations)

    payload = {
        "images": images_meta,
        "annotations": annotations,

        "pred_obj_cropaug_meta": {
            "format":
                (
                    "partimagenet40_"
                    "actualobj_predobj_cropaug_v2"
                ),

            "protocol": (
                "Image-level actual ImageNet object "
                "category is used only to query the "
                "frozen Talk2DINO foreground predictor. "
                "Image-level PartImageNet part presence "
                "defines the present semantic parts. "
                "No target object/part mask, point, or "
                "bounding box is opened during cache "
                "generation."
            ),

            "presence_manifest":
                str(manifest_path),

            "presence_manifest_sha256":
                sha256(manifest_path),

            "target_spatial_supervision":
                "none",

            "foreground_object_semantics":
                "actual_imagenet_object_class",

            "part_group_semantics":
                "partimagenet_supercategory",

            "num_actual_object_classes":
                len(object_names),

            "actual_object_wnids":
                object_wnids,

            "actual_object_names":
                object_names,

            "model_name":
                args.model_name,

            "clip_model_name":
                args.clip_model_name,

            "template":
                args.template,

            "projector_weight":
                str(weight),

            "projector_weight_sha256":
                sha256(weight),

            "with_bg_clean":
                True,

            "pamr":
                bool(args.pamr),

            "bg_thresh":
                float(args.bg_thresh),

            "lambda_bg":
                float(args.lambda_bg),

            "outer_resize": [
                int(args.eval_max_long),
                int(args.eval_max_short),
            ],

            "slide_crop":
                int(args.slide_crop),

            "slide_stride":
                int(args.slide_stride),

            "crop_expand_ratio":
                float(args.crop_expand_ratio),

            "crop_resize":
                [448, 448],

            "crop_interpolation":
                "BICUBIC",

            "patch_grid":
                [32, 32],

            "patch_token_shape":
                [1024, 768],

            "patch_token_dtype":
                "float16",

            "foreground_key":
                "pred_obj_mask_patch",

            "part_presence_key":
                "part_category_id",

            "elapsed_seconds":
                elapsed,

            "stats":
                stats,
        },
    }

    print()
    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"[save] {len(annotations)} "
        f"object crops -> {output}"
    )

    atomic_torch_save(
        payload,
        output,
    )

    print("[done]")
    print(
        "[W loader] "
        "patch_key=cropaug_patch_tokens"
    )
    print(
        "[W loader] "
        "foreground_key=pred_obj_mask_patch"
    )
    print(
        "[W loader] "
        "presence_key=part_category_id"
    )


if __name__ == "__main__":
    main()
