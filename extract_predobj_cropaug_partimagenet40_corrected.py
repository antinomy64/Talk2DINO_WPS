#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct PartImageNet-40 predicted-object crop/cache builder.

Critical distinction
--------------------
1) The PartImageNet superclass/group:
       quadruped / biped / fish / ...
   is used ONLY to define which part labels belong together.

2) Talk2DINO foreground prediction uses the REAL ImageNet object semantic
   name recovered from the image filename WNID:
       n02099601_xxx.JPEG -> golden retriever
   NOT:
       quadruped

Supervision used for target adaptation:
- RGB image
- image-level part-presence labels
- image-level object identity from ImageNet WNID / filename

NO target part/object spatial masks, points or boxes are read.

Output cache contract remains compatible with
train_relproto_alignment_partimagenet40.py:

annotation:
    cropaug_patch_tokens : Float16Tensor[1024,768]
    pred_obj_mask_patch  : BoolTensor[1024]
    part_category_id     : list[int]
    part_class_name      : list[str]

Additional provenance:
    foreground_object_wnid
    foreground_object_name
    part_parent_group
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------
# Repository-local imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partimagenet40_taxonomy import (
    PART_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)

# IMPORTANT:
# Reuse the already validated PP116/Talk2DINO spatial inference utilities.
# We do NOT duplicate/reimplement foreground inference here.
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


GROUP_NAMES = list(OBJECT_GROUPS.keys())

PART_NAME_TO_ID = {
    name: i
    for i, name in enumerate(PART_CLASSES)
}

GROUP_NAME_TO_ID = {
    name: i
    for i, name in enumerate(GROUP_NAMES)
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Correct PartImageNet-40 predicted-object cropaug cache: "
            "real ImageNet object query + PartImageNet relational groups."
        ),
    )

    p.add_argument(
        "--repo_root",
        default=".",
        help="Talk2DINO_official_bg repository root",
    )

    p.add_argument(
        "--image_root",
        default="",
        help="PartImageNet images/train",
    )

    p.add_argument(
        "--presence_manifest",
        default="",
        help=(
            "JSON list with image-level part labels, e.g. "
            "{'image':'n01440764_10029.JPEG','labels':[9,10,11,12]}"
        ),
    )

    p.add_argument(
        "--imagenet_labels",
        default="",
        help="ImageNet LOC_synset_mapping.txt",
    )

    p.add_argument(
        "--output_pth",
        default="",
    )

    # Frozen Talk2DINO COCO object-level projector.
    p.add_argument(
        "--projector_weight",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_reproduce_clean.pth"
        ),
    )

    # Must match formal Ours.
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

    # Formal PP116-compatible object foreground settings.
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
        help="Debug/smoke only. 0 means all.",
    )

    p.add_argument(
        "--self_test",
        action="store_true",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# ImageNet WNID -> semantic object name
# ---------------------------------------------------------------------

def load_imagenet_synset_mapping(path: Path) -> dict[str, str]:
    """
    Expected LOC_synset_mapping.txt format:

        n01440764 tench, Tinca tinca
        n01443537 goldfish, Carassius auratus
        ...

    We follow the common PartImageNet preprocessing convention and use
    the first semantic synonym before the first comma.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"ImageNet synset mapping not found: {path}"
        )

    mapping: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line:
                continue

            fields = line.split(maxsplit=1)

            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{lineno}: malformed line: {line!r}"
                )

            wnid, names = fields

            if not wnid.startswith("n"):
                raise ValueError(
                    f"{path}:{lineno}: invalid WNID {wnid!r}"
                )

            # Same convention used by common PartImageNet preprocessing:
            # use first synonym.
            object_name = names.split(",", maxsplit=1)[0].strip()

            if not object_name:
                raise ValueError(
                    f"{path}:{lineno}: empty object name"
                )

            if wnid in mapping:
                raise ValueError(
                    f"duplicate WNID in {path}: {wnid}"
                )

            mapping[wnid] = object_name

    if not mapping:
        raise ValueError(
            f"no ImageNet classes loaded from {path}"
        )

    return mapping


def wnid_from_image_name(name: str) -> str:
    """
    PartImageNet naming:
        n01440764_10029.JPEG
        -> n01440764
    """

    stem = Path(name).stem

    if "_" not in stem:
        raise ValueError(
            f"PartImageNet image filename has no WNID prefix: {name}"
        )

    wnid = stem.split("_", maxsplit=1)[0]

    if not (
        len(wnid) == 9
        and wnid.startswith("n")
        and wnid[1:].isdigit()
    ):
        raise ValueError(
            f"bad ImageNet WNID parsed from {name!r}: {wnid!r}"
        )

    return wnid


# ---------------------------------------------------------------------
# Presence manifest
# ---------------------------------------------------------------------

def load_presence_manifest(
    path: Path,
    image_root: Path,
    imagenet_id2name: dict[str, str],
) -> list[dict[str, Any]]:

    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(
            "PartImageNet presence manifest must be a JSON list"
        )

    records: list[dict[str, Any]] = []

    seen_images: set[str] = set()

    for idx, rec in enumerate(data):

        if not isinstance(rec, dict):
            raise ValueError(
                f"manifest row {idx} must be dict"
            )

        if "image" not in rec or "labels" not in rec:
            raise ValueError(
                f"manifest row {idx} missing image/labels"
            )

        rel_name = str(rec["image"])

        if rel_name in seen_images:
            raise ValueError(
                f"duplicate image in manifest: {rel_name}"
            )

        seen_images.add(rel_name)

        raw_labels = rec["labels"]

        if not isinstance(raw_labels, list) or not raw_labels:
            raise ValueError(
                f"manifest row {idx}: invalid labels {raw_labels}"
            )

        pids = [int(x) for x in raw_labels]

        if len(pids) != len(set(pids)):
            raise ValueError(
                f"manifest row {idx}: duplicate part labels {pids}"
            )

        if any(
            pid < 0 or pid >= len(PART_CLASSES)
            for pid in pids
        ):
            raise ValueError(
                f"manifest row {idx}: part ID outside [0,39]: {pids}"
            )

        # -------------------------------------------------------------
        # RELATIONAL parent group.
        # This is NOT the foreground object semantic query.
        # -------------------------------------------------------------
        group_ids = sorted({
            int(PART_TO_OBJECT[pid])
            for pid in pids
        })

        if len(group_ids) != 1:
            raise ValueError(
                f"manifest row {idx} spans multiple relational groups: "
                f"image={rel_name} labels={pids} groups={group_ids}"
            )

        group_id = int(group_ids[0])
        group_name = GROUP_NAMES[group_id]

        # -------------------------------------------------------------
        # TRUE foreground semantic object identity.
        # -------------------------------------------------------------
        wnid = wnid_from_image_name(rel_name)

        if wnid not in imagenet_id2name:
            raise KeyError(
                f"manifest row {idx}: WNID {wnid} from {rel_name} "
                f"is absent from LOC_synset_mapping.txt"
            )

        foreground_object_name = imagenet_id2name[wnid]

        img_path = image_root / rel_name

        if not img_path.is_file():
            raise FileNotFoundError(
                f"manifest row {idx}: image missing: {img_path}"
            )

        pnames = [
            PART_CLASSES[pid]
            for pid in pids
        ]

        # Strong taxonomy check.
        allowed = set(OBJECT_GROUPS[group_name])

        if any(pid not in allowed for pid in pids):
            raise AssertionError(
                f"{rel_name}: part IDs {pids} do not belong to "
                f"group {group_name}"
            )

        records.append({
            "image": rel_name,
            "image_path": img_path.resolve(),
            "part_ids": sorted(pids),
            "part_names": [
                PART_CLASSES[pid]
                for pid in sorted(pids)
            ],

            # Relational structure identity:
            "part_parent_group_id": group_id,
            "part_parent_group": group_name,

            # Foreground semantic identity:
            "foreground_object_wnid": wnid,
            "foreground_object_name": foreground_object_name,
        })

    return records


# ---------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------

def self_test() -> None:

    assert len(PART_CLASSES) == 40
    assert len(OBJECT_GROUPS) == 11
    assert set(PART_TO_OBJECT.keys()) == set(range(40))

    assert PART_CLASSES[0] == "Quadruped Head"
    assert PART_CLASSES[25] == "Car Tire"
    assert PART_CLASSES[30] == "Bicycle Tire"
    assert PART_CLASSES[39] == "Bottle Body"

    assert wnid_from_image_name(
        "n01440764_10029.JPEG"
    ) == "n01440764"

    for group_id, (group_name, pids) in enumerate(
        OBJECT_GROUPS.items()
    ):
        assert GROUP_NAMES[group_id] == group_name
        assert pids

        for pid in pids:
            assert PART_TO_OBJECT[pid] == group_id

    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:

    args = parse_args()

    if args.self_test:
        self_test()
        return

    required = (
        "image_root",
        "presence_manifest",
        "imagenet_labels",
        "output_pth",
    )
    missing = [
        name
        for name in required
        if not str(getattr(args, name, "")).strip()
    ]
    if missing:
        raise SystemExit(
            "Missing required arguments for cache generation: "
            + ", ".join("--" + x for x in missing)
        )

    repo = Path(args.repo_root).expanduser().resolve()

    image_root = Path(
        args.image_root
    ).expanduser().resolve()

    manifest_path = Path(
        args.presence_manifest
    ).expanduser().resolve()

    imagenet_labels_path = Path(
        args.imagenet_labels
    ).expanduser().resolve()

    output = Path(
        args.output_pth
    ).expanduser()

    if not output.is_absolute():
        output = (repo / output).resolve()

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    if not repo.is_dir():
        raise NotADirectoryError(repo)

    if not image_root.is_dir():
        raise NotADirectoryError(image_root)

    # -------------------------------------------------------------
    # Load object semantic mapping and weak supervision.
    # -------------------------------------------------------------

    imagenet_id2name = load_imagenet_synset_mapping(
        imagenet_labels_path
    )

    records = load_presence_manifest(
        manifest_path,
        image_root,
        imagenet_id2name,
    )

    if args.max_images > 0:
        records = records[:args.max_images]

    if not records:
        raise ValueError("no training records")

    print("=" * 72)
    print("Correct PartImageNet predicted-object cache")
    print("=" * 72)
    print("images              :", len(records))
    print("part classes        :", len(PART_CLASSES))
    print("relational groups   :", len(OBJECT_GROUPS))
    print("ImageNet mapping    :", imagenet_labels_path)
    print("projector           :", args.projector_weight)
    print("bg_thresh           :", args.bg_thresh)
    print("pamr                :", args.pamr)
    print("output              :", output)

    # -------------------------------------------------------------
    # Audit semantic-object vs relational-group identities.
    # -------------------------------------------------------------

    group_counts = Counter(
        rec["part_parent_group"]
        for rec in records
    )

    wnid_counts = Counter(
        rec["foreground_object_wnid"]
        for rec in records
    )

    object_name_counts = Counter(
        rec["foreground_object_name"]
        for rec in records
    )

    print()
    print("relational-group counts:")
    for name in GROUP_NAMES:
        print(
            f"  {name:12s}: "
            f"{group_counts.get(name, 0)}"
        )

    print()
    print(
        "unique foreground WNIDs :",
        len(wnid_counts),
    )
    print(
        "unique foreground names :",
        len(object_name_counts),
    )

    print()
    print("examples:")
    for rec in records[:10]:
        print(
            f"  {rec['image']} | "
            f"{rec['foreground_object_wnid']} | "
            f"{rec['foreground_object_name']} | "
            f"group={rec['part_parent_group']} | "
            f"parts={rec['part_ids']}"
        )

    # -------------------------------------------------------------
    # Frozen Talk2DINO.
    # -------------------------------------------------------------

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    model, projector_weight = build_model(
        args,
        repo,
        device,
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # Build text bank from REAL ImageNet object names.
    #
    # Do NOT use:
    #   quadruped / bird / fish / ...
    # -------------------------------------------------------------

    unique_object_names = sorted({
        rec["foreground_object_name"]
        for rec in records
    })

    object_name_to_row = {
        name: i
        for i, name in enumerate(unique_object_names)
    }

    object_tokens = model.build_dataset_class_tokens(
        args.template,
        unique_object_names,
    )

    object_text = model.build_text_embedding(
        object_tokens
    ).to(
        device=device,
        dtype=torch.float32,
    )

    expected_shape = (
        len(unique_object_names),
        VISION_DIM,
    )

    if tuple(object_text.shape) != expected_shape:
        raise ValueError(
            f"bad real-object text bank shape: "
            f"{tuple(object_text.shape)} "
            f"!= {expected_shape}"
        )

    if not torch.isfinite(object_text).all():
        raise ValueError(
            "nonfinite real-object text embeddings"
        )

    print()
    print(
        "[foreground] semantic object vocabulary:",
        len(unique_object_names),
    )

    # -------------------------------------------------------------
    # Crop transform / output containers.
    # -------------------------------------------------------------

    transform = crop_transform()

    annotations: list[dict[str, Any]] = []
    images_meta: list[dict[str, Any]] = []

    pending_crops = []
    pending_indices: list[int] = []

    stats: defaultdict[str, int] = defaultdict(int)

    start = time.perf_counter()

    def flush() -> None:
        nonlocal pending_crops, pending_indices

        if not pending_crops:
            return

        batch = torch.stack([
            transform(im)
            for im in pending_crops
        ], dim=0).to(
            device=device,
            dtype=torch.float32,
        )

        dino_out = model.model(
            batch,
            is_training=True,
        )

        if (
            not isinstance(dino_out, dict)
            or "x_norm_patchtokens" not in dino_out
        ):
            # Some DINO output objects implement Mapping
            # but are not plain dicts.
            try:
                tokens = dino_out[
                    "x_norm_patchtokens"
                ]
            except Exception as exc:
                raise RuntimeError(
                    "DINOv2 output missing "
                    "x_norm_patchtokens"
                ) from exc
        else:
            tokens = dino_out[
                "x_norm_patchtokens"
            ]

        expected = (
            len(pending_crops),
            PATCH_COUNT,
            VISION_DIM,
        )

        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"DINO crop tokens "
                f"{tuple(tokens.shape)} != {expected}"
            )

        if not torch.isfinite(tokens).all():
            raise ValueError(
                "NaN/Inf in DINO crop patch tokens"
            )

        for j, ann_idx in enumerate(
            pending_indices
        ):
            annotations[ann_idx][
                "cropaug_patch_tokens"
            ] = (
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

    # -------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------

    for rec in tqdm(
        records,
        desc=(
            "WNID object -> pred obj "
            "-> crop -> DINO cache"
        ),
    ):

        stats["requested_images"] += 1

        img_path = Path(rec["image_path"])

        pil = load_rgb(img_path)

        pids = list(rec["part_ids"])
        pnames = list(rec["part_names"])

        group_id = int(
            rec["part_parent_group_id"]
        )
        group_name = str(
            rec["part_parent_group"]
        )

        object_wnid = str(
            rec["foreground_object_wnid"]
        )
        object_name = str(
            rec["foreground_object_name"]
        )

        # ---------------------------------------------------------
        # Foreground query = exact ImageNet object semantic name.
        # ONE object channel for this PartImageNet image.
        # ---------------------------------------------------------

        text_row = object_name_to_row[
            object_name
        ]

        present_text = object_text[
            text_row : text_row + 1
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
                f"{rec['image']}: bad foreground "
                f"score shape {tuple(fg_scores.shape)}"
            )

        # Same background competition as PP116.
        background = torch.full(
            (
                1,
                pil.height,
                pil.width,
            ),
            float(args.bg_thresh),
            device=fg_scores.device,
            dtype=fg_scores.dtype,
        )

        # channel 0 = background
        # channel 1 = this image's real ImageNet object
        hard = torch.cat(
            [
                background,
                fg_scores,
            ],
            dim=0,
        ).argmax(dim=0)

        pred = (
            (hard == 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        if int(pred.sum()) == 0:

            stats[
                "empty_predicted_object_mask"
            ] += 1

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

                "part_parent_group_id":
                    group_id,
                "part_parent_group":
                    group_name,

                "part_category_id_present":
                    pids,

                "ready":
                    False,
            })

            stats["processed_images"] += 1
            continue

        # ---------------------------------------------------------
        # Predicted object mask -> crop / patch foreground.
        # ---------------------------------------------------------

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

        if (
            tuple(patch_mask.shape)
            != (PATCH_COUNT,)
        ):
            raise ValueError(
                f"bad patch foreground shape: "
                f"{tuple(patch_mask.shape)}"
            )

        if int(
            patch_mask.sum().item()
        ) <= 0:
            raise ValueError(
                "non-empty pixel foreground became "
                "empty patch foreground"
            )

        x1, y1, x2, y2 = box

        crop = pil.crop(
            (x1, y1, x2, y2)
        )

        # ---------------------------------------------------------
        # IMPORTANT CACHE SEMANTICS
        #
        # class_name is retained as the RELATIONAL parent group
        # for backward compatibility.
        #
        # foreground_object_name contains the actual object query.
        # ---------------------------------------------------------

        record = {
            "id": len(annotations),
            "image_id": Path(
                rec["image"]
            ).stem,

            # Backward-compatible relational parent identity:
            "category_id": group_id,
            "class_name": group_name,

            # Explicitly separate actual foreground identity:
            "foreground_object_wnid":
                object_wnid,
            "foreground_object_name":
                object_name,

            "part_parent_group_id":
                group_id,
            "part_parent_group":
                group_name,

            # Image-level weak part supervision:
            "part_category_id":
                pids,
            "part_class_name":
                pnames,

            # Spatial fields are predicted only:
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
                    "talk2dino_exact_imagenet_"
                    "object_name_from_wnid"
                ),
        }

        annotations.append(record)

        pending_crops.append(crop)
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

            "part_parent_group_id":
                group_id,
            "part_parent_group":
                group_name,

            "part_category_id_present":
                pids,

            "ready":
                True,
        })

        stats["processed_images"] += 1

    flush()

    # -----------------------------------------------------------------
    # Strong final cache validation
    # -----------------------------------------------------------------

    for i, ann in enumerate(annotations):

        tok = ann.get(
            "cropaug_patch_tokens"
        )

        pm = ann.get(
            "pred_obj_mask_patch"
        )

        box = ann.get(
            "cropaug_box_xyxy"
        )

        pids = list(
            ann.get(
                "part_category_id",
                [],
            )
        )

        pnames = list(
            ann.get(
                "part_class_name",
                [],
            )
        )

        if (
            not torch.is_tensor(tok)
            or tuple(tok.shape)
            != (PATCH_COUNT, VISION_DIM)
            or tok.dtype != torch.float16
        ):
            raise ValueError(
                f"annotation {i}: invalid "
                f"cropaug_patch_tokens"
            )

        if (
            not torch.is_tensor(pm)
            or tuple(pm.shape)
            != (PATCH_COUNT,)
            or pm.dtype != torch.bool
            or int(pm.sum()) <= 0
        ):
            raise ValueError(
                f"annotation {i}: invalid "
                f"pred_obj_mask_patch"
            )

        if (
            not torch.is_tensor(box)
            or tuple(box.shape) != (4,)
            or box.dtype != torch.long
        ):
            raise ValueError(
                f"annotation {i}: invalid "
                f"cropaug_box_xyxy"
            )

        if not pids:
            raise ValueError(
                f"annotation {i}: empty part presence"
            )

        if len(pids) != len(set(pids)):
            raise ValueError(
                f"annotation {i}: duplicate part IDs"
            )

        if len(pids) != len(pnames):
            raise ValueError(
                f"annotation {i}: part id/name count mismatch"
            )

        expected_names = [
            PART_CLASSES[pid]
            for pid in pids
        ]

        if pnames != expected_names:
            raise ValueError(
                f"annotation {i}: part taxonomy mismatch: "
                f"{pnames} != {expected_names}"
            )

        group_name = ann[
            "part_parent_group"
        ]

        allowed = set(
            OBJECT_GROUPS[group_name]
        )

        if any(
            pid not in allowed
            for pid in pids
        ):
            raise ValueError(
                f"annotation {i}: parts {pids} "
                f"do not belong to {group_name}"
            )

        # Critical anti-regression check:
        # foreground query must NOT silently be the superclass.
        foreground_name = str(
            ann["foreground_object_name"]
        )

        if foreground_name == group_name:
            # A few ImageNet class names could theoretically equal
            # a group word, but for this dataset this is almost
            # certainly a regression. Fail hard.
            raise ValueError(
                f"annotation {i}: foreground object name "
                f"{foreground_name!r} equals relational group "
                f"{group_name!r}; superclass/query separation "
                f"has likely regressed"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    stats = dict(stats)

    stats[
        "output_annotations"
    ] = len(annotations)

    stats[
        "unique_foreground_wnids"
    ] = len(wnid_counts)

    stats[
        "unique_foreground_object_names"
    ] = len(object_name_counts)

    total = max(
        1,
        stats.get(
            "requested_images",
            0,
        ),
    )

    stats[
        "empty_predicted_object_mask_rate"
    ] = (
        stats.get(
            "empty_predicted_object_mask",
            0,
        )
        / total
    )

    payload = {
        "images":
            images_meta,

        "annotations":
            annotations,

        "pred_obj_cropaug_meta": {
            "format":
                (
                    "partimagenet40_weaklabels_"
                    "predobj_cropaug_realobj_v2"
                ),

            "protocol": (
                "Image-level PartImageNet part-presence labels "
                "define the visible semantic parts and their "
                "11-way relational parent group. "
                "The image filename WNID is mapped through "
                "LOC_synset_mapping.txt to the real ImageNet "
                "object semantic name, which is used as the "
                "frozen Talk2DINO foreground query. "
                "No target object/part spatial masks, points, "
                "or boxes are accessed. cropaug_box_xyxy, "
                "pred_obj_mask_patch and cropaug_patch_tokens "
                "are derived only from the frozen Talk2DINO "
                "predicted object foreground."
            ),

            "target_spatial_supervision":
                "none",

            "weak_part_supervision":
                "image_level_part_presence",

            "object_identity_source":
                "image_filename_wnid",

            "foreground_object_name_source":
                str(
                    imagenet_labels_path
                ),

            "foreground_object_name_source_sha256":
                sha256(
                    imagenet_labels_path
                ),

            "presence_manifest":
                str(manifest_path),

            "presence_manifest_sha256":
                sha256(
                    manifest_path
                ),

            # This is the critical corrected definition:
            "object_prediction_candidate_scope":
                (
                    "background + exact ImageNet semantic "
                    "object name derived from image WNID"
                ),

            "relational_group_scope":
                (
                    "PartImageNet 11-way superclass; "
                    "used only to group semantic parts"
                ),

            "superclass_used_as_foreground_query":
                False,

            "model_name":
                args.model_name,

            "clip_model_name":
                args.clip_model_name,

            "template":
                args.template,

            "projector_weight":
                str(projector_weight),

            "projector_weight_sha256":
                sha256(
                    projector_weight
                ),

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
                float(
                    args.crop_expand_ratio
                ),

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
    print("=" * 72)
    print("FINAL CACHE STATISTICS")
    print("=" * 72)
    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    empty_rate = stats[
        "empty_predicted_object_mask_rate"
    ]

    print()
    print(
        "empty foreground rate:",
        f"{100.0 * empty_rate:.2f}%",
    )

    print(
        "[save]",
        len(annotations),
        "object crops ->",
        output,
    )

    atomic_torch_save(
        payload,
        output,
    )

    print("[done]")
    print(
        "[W loader] patch_key="
        "cropaug_patch_tokens"
    )
    print(
        "[W loader] foreground_source="
        "precomputed"
    )
    print(
        "[W loader] foreground_key="
        "pred_obj_mask_patch"
    )
    print(
        "[W loader] presence_source="
        "annotation"
    )
    print(
        "[W loader] presence_key="
        "part_category_id"
    )


if __name__ == "__main__":
    main()
