#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Talk2DINO-Part target-adaptation datasets.

Supervision:
  image + image-level present-part labels only.

Output follows Talk2DINO DinoClipDataset format:
{
    "images": [
        {
            "id": ...,
            "file_name": ...,
            "disentangled_self_attn": Tensor[num_heads, 768],
        },
        ...
    ],
    "annotations": [
        {
            "id": ...,
            "image_id": ...,
            "caption": part_name,
            "part_id": int,
            "ann_feats": Tensor[512],
        },
        ...
    ],
    "meta": {...}
}

No target masks / boxes / points are read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import clip
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hooks import (
    get_self_attention,
    process_self_attention,
    feats,
)


DINO_MODEL = "dinov2_vitb14_reg"
DINO_DIM = 768
CLIP_MODEL = "ViT-B/16"

RESIZE_DIM = 448
CROP_DIM = 448
PATCH_SIZE = 14
PATCH_COUNT = (CROP_DIM // PATCH_SIZE) ** 2

# DINOv2 ViT-B/14-Reg:
# 1 CLS + 4 register tokens + 1024 patch tokens
NUM_GLOBAL_TOKENS = 5
NUM_TOKENS = NUM_GLOBAL_TOKENS + PATCH_COUNT

# Official Talk2DINO extraction uses this scale.
ATTN_SCALE = 0.125


def parse_args():
    p = argparse.ArgumentParser(
        "Extract target features for Talk2DINO-Part."
    )

    p.add_argument(
        "--dataset",
        required=True,
        choices=["pp116", "partimagenet40", "ade234"],
    )
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--image_root",
        default="",
        help="Needed for manifests containing relative image names.",
    )
    p.add_argument("--out", required=True)

    p.add_argument("--device", default="cuda")
    p.add_argument("--image_batch_size", type=int, default=32)
    p.add_argument("--text_batch_size", type=int, default=256)

    p.add_argument(
        "--dinov2_repo",
        default="",
        help=(
            "Optional local facebookresearch/dinov2 checkout. "
            "If empty, torch.hub cached/default repo is used."
        ),
    )

    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Smoke only; 0 means all.",
    )

    return p.parse_args()


def load_manifest(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))

    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else None
    )

    samples = []

    if args.dataset == "partimagenet40":
        from partimagenet40_taxonomy import PART_CLASSES

        if not isinstance(obj, list):
            raise TypeError(
                "PartImageNet manifest must be a JSON list."
            )

        class_names = list(PART_CLASSES)

        for i, rec in enumerate(obj):
            image_name = str(rec["image"])
            rel = Path(image_name)

            image_path = (
                rel if rel.is_absolute()
                else image_root / rel
            )

            pids = sorted({int(x) for x in rec["labels"]})

            samples.append({
                "id": rel.stem,
                "image_path": str(image_path),
                "part_ids": pids,
            })

    else:
        if not isinstance(obj, Mapping):
            raise TypeError(
                "PP116 manifest must be a JSON mapping."
            )

        if "samples" not in obj or "part_classes" not in obj:
            raise ValueError(
                "PP116 manifest requires samples + part_classes."
            )

        class_names = list(obj["part_classes"])

        expected_classes = (
            234
            if args.dataset == "ade234"
            else 116
        )

        if len(class_names) != expected_classes:
            raise ValueError(
                f"Expected {expected_classes} classes for "
                f"{args.dataset}, got {len(class_names)}"
            )

        for i, rec in enumerate(obj["samples"]):
            path = Path(str(rec["image"]))

            if not path.is_absolute():
                if image_root is None:
                    raise ValueError(
                        "Relative PP116 image path requires --image_root."
                    )
                path = image_root / path

            pids = sorted({int(x) for x in rec["part_ids"]})

            samples.append({
                "id": str(rec.get("stem", path.stem)),
                "image_path": str(path),
                "part_ids": pids,
            })

    if args.max_images > 0:
        samples = samples[:args.max_images]

    ncls = len(class_names)

    seen_ids = set()

    for rec in samples:
        if not rec["part_ids"]:
            raise ValueError(
                f"{rec['id']}: empty part presence"
            )

        for pid in rec["part_ids"]:
            if not (0 <= pid < ncls):
                raise ValueError(
                    f"{rec['id']}: invalid part id {pid}"
                )
            seen_ids.add(pid)

        if not Path(rec["image_path"]).is_file():
            raise FileNotFoundError(rec["image_path"])

    print("dataset       :", args.dataset)
    print("samples       :", len(samples))
    print("classes       :", ncls)
    print("seen classes  :", len(seen_ids))

    return samples, class_names, manifest_path


def load_dino(args, device):
    if args.dinov2_repo:
        model = torch.hub.load(
            str(Path(args.dinov2_repo).expanduser().resolve()),
            DINO_MODEL,
            source="local",
        )
    else:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            DINO_MODEL,
        )

    model.eval()
    model.requires_grad_(False)
    model.to(device)

    # Same hook used by official Talk2DINO extractor.
    model.blocks[-1].attn.qkv.register_forward_hook(
        get_self_attention
    )

    return model


def image_transform():
    return T.Compose([
        T.Resize(
            RESIZE_DIM,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.CenterCrop(CROP_DIM),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@torch.inference_mode()
def extract_visual(samples, args, device):
    model = load_dino(args, device)
    transform = image_transform()

    num_heads = int(model.num_heads)

    results = []

    for start in tqdm(
        range(0, len(samples), args.image_batch_size),
        desc="DINOv2 disentangled self-attn",
    ):
        batch_rec = samples[
            start:start + args.image_batch_size
        ]

        images = []

        for rec in batch_rec:
            with Image.open(rec["image_path"]) as im:
                images.append(
                    transform(im.convert("RGB"))
                )

        batch = torch.stack(images).to(device)

        out = model(batch, is_training=True)

        patches = out["x_norm_patchtokens"]

        expected = (
            len(batch_rec),
            PATCH_COUNT,
            DINO_DIM,
        )
        if tuple(patches.shape) != expected:
            raise ValueError(
                f"DINO patches {tuple(patches.shape)} "
                f"!= {expected}"
            )

        _, attn_maps = process_self_attention(
            feats["self_attn"],
            len(batch_rec),
            NUM_TOKENS,
            num_heads,
            DINO_DIM,
            ATTN_SCALE,
            NUM_GLOBAL_TOKENS,
            ret_self_attn_maps=True,
        )

        # This exactly follows Talk2DINO's extraction:
        # softmax per head -> weighted mean of DINO patches.
        attn_maps = attn_maps.softmax(dim=-1)

        disentangled = (
            patches.unsqueeze(1)
            * attn_maps.unsqueeze(-1)
        ).mean(dim=2)

        expected_dis = (
            len(batch_rec),
            num_heads,
            DINO_DIM,
        )

        if tuple(disentangled.shape) != expected_dis:
            raise ValueError(
                f"disentangled feature "
                f"{tuple(disentangled.shape)} "
                f"!= {expected_dis}"
            )

        if not torch.isfinite(disentangled).all():
            raise RuntimeError(
                "NaN/Inf in visual features."
            )

        disentangled = (
            disentangled.float().cpu().contiguous()
        )

        for j, rec in enumerate(batch_rec):
            results.append({
                "id": rec["id"],
                "file_name": rec["image_path"],
                "disentangled_self_attn":
                    disentangled[j].clone(),
            })

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


@torch.inference_mode()
def extract_text_bank(
    class_names,
    args,
    device,
):
    model, _ = clip.load(
        CLIP_MODEL,
        device=device,
        jit=False,
    )

    model.eval()
    model.requires_grad_(False)

    all_features = []

    for start in tqdm(
        range(0, len(class_names), args.text_batch_size),
        desc="CLIP part text",
    ):
        names = class_names[
            start:start + args.text_batch_size
        ]

        tokens = clip.tokenize(
            names,
            truncate=True,
        ).to(device)

        feats_txt = model.encode_text(tokens)

        all_features.append(
            feats_txt.float().cpu()
        )

    text_bank = torch.cat(all_features, dim=0)

    expected = (len(class_names), 512)
    if tuple(text_bank.shape) != expected:
        raise ValueError(
            f"text bank {tuple(text_bank.shape)} "
            f"!= {expected}"
        )

    if not torch.isfinite(text_bank).all():
        raise RuntimeError(
            "NaN/Inf in CLIP text features."
        )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return text_bank.contiguous()


def main():
    args = parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    samples, class_names, manifest_path = load_manifest(
        args
    )

    images = extract_visual(
        samples,
        args,
        device,
    )

    text_bank = extract_text_bank(
        class_names,
        args,
        device,
    )

    annotations = []

    ann_id = 0

    for rec in samples:
        image_id = rec["id"]

        for pid in rec["part_ids"]:
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "caption": class_names[pid],
                "part_id": int(pid),
                "ann_feats": text_bank[pid].clone(),
            })
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "meta": {
            "format": "talk2dino_part_target_v1",
            "dataset": args.dataset,
            "manifest": str(manifest_path),
            "supervision":
                "image_level_part_presence_only",
            "target_spatial_supervision": "none",
            "dino_model": DINO_MODEL,
            "visual_feature":
                "disentangled_self_attn",
            "visual_shape": [
                int(images[0][
                    "disentangled_self_attn"
                ].shape[0]),
                768,
            ],
            "clip_model": CLIP_MODEL,
            "text_feature": "ann_feats",
            "text_prompt":
                "raw semantic part class name",
            "num_images": len(images),
            "num_annotations": len(annotations),
            "num_classes": len(class_names),
            "class_names": class_names,
        },
    }

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()

    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {out}"
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(payload, out)

    print("=" * 60)
    print("Talk2DINO-Part feature dataset")
    print("=" * 60)
    print("images       :", len(images))
    print("annotations  :", len(annotations))
    print("classes      :", len(class_names))
    print(
        "visual shape :",
        tuple(
            images[0][
                "disentangled_self_attn"
            ].shape
        ),
    )
    print("text shape   :", tuple(text_bank.shape))
    print("output       :", out)
    print("spatial GT   : NONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
