#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_predobj_cropaug.py

Offline cache builder for the final ADE20K-Part-234 W-training stage.

Input supervision source:
  RGB image + GT object semantic mask + GT part semantic mask.

The GT masks are used ONLY to derive weak semantic labels:
  1) which object classes are present in the image;
  2) for each present object class, which part classes are present.

After those labels are derived, GT spatial information is never used for
object cropping, foreground masking, DINO patch selection, prototype creation,
or W training.

Spatial training cache is produced only from the frozen Talk2DINO prediction:
  present object labels -> predicted semantic object masks
  -> predicted-mask crop -> RGB crop -> DINOv2 crop patch tokens
  -> same predicted mask mapped to the 32x32 crop patch grid.

One crop is created per (image, semantic object class). Multiple instances of
the same class are therefore represented by one semantic-class union mask/crop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# ADE20K-Part-234 taxonomy
# -----------------------------------------------------------------------------

from ade20kpart234_taxonomy import (
    NUM_PARTS,
    NUM_OBJECTS,
    IGNORE_LABEL,
    PART_CLASSES,
    OBJECT_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)

PART_CLASSES = list(PART_CLASSES)
OBJECT_CLASSES = list(OBJECT_CLASSES)

OBJ_NAME_TO_ID = {
    name: i for i, name in enumerate(OBJECT_CLASSES)
}

PART_NAME_TO_ID = {
    name: i for i, name in enumerate(PART_CLASSES)
}

PART_IDS_BY_OBJECT = {
    name: list(OBJECT_GROUPS[name])
    for name in OBJECT_CLASSES
}

assert NUM_PARTS == 234
assert NUM_OBJECTS == 44
assert len(PART_CLASSES) == 234
assert len(OBJECT_CLASSES) == 44
assert sorted(PART_TO_OBJECT) == list(range(234))

VISION_DIM = 768
PATCH_SIZE = 14
CROP_DIM = 448
PATCH_GRID = CROP_DIM // PATCH_SIZE
PATCH_COUNT = PATCH_GRID * PATCH_GRID


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Offline predicted-object cropaug cache for ADE20K-Part-234",
    )

    p.add_argument("--repo_root", default=".", help="Talk2DINO_official_bg repository root")
    p.add_argument("--image_root", default=None)
    p.add_argument("--part_mask_root", default=None)
    p.add_argument("--split_file", default="", help="Optional txt file containing image stems")
    p.add_argument("--image_suffixes", default=".jpg,.jpeg,.png")
    p.add_argument("--mask_suffixes", default=".png")

    # ADE20K-Part-234 released masks:
    # semantic part IDs 0..233; background/unlabeled = 65535.
    # Only np.unique(mask) is used to obtain image-level presence.
    p.add_argument("--part_ignore", type=int, default=65535)

    p.add_argument(
        "--projector_weight",
        default="weights/vitb_mlp_infonce_coco2014_clean_ft10_partstruct_w1e4_lr1e5.pth",
    )
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--clip_model_name", default="ViT-B/16")
    p.add_argument("--proj_class", default="vitb_mlp_infonce")
    p.add_argument("--proj_model", default="ProjectionLayer")
    p.add_argument("--template", default="sub_imagenet_template")

    # Match the current Pascal/VOC bg-aware PAMR evaluation geometry.
    p.add_argument("--bg_thresh", type=float, default=0.55)
    p.add_argument("--lambda_bg", type=float, default=0.2)
    p.add_argument("--pamr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_max_long", type=int, default=2048)
    p.add_argument("--eval_max_short", type=int, default=448)
    p.add_argument("--slide_crop", type=int, default=448)
    p.add_argument("--slide_stride", type=int, default=224)

    # Match the historical cropaug branch.
    p.add_argument("--crop_expand_ratio", type=float, default=1.2)
    p.add_argument("--crop_batch_size", type=int, default=16)

    p.add_argument("--device", default="cuda")
    p.add_argument("--output_pth", default=None)
    p.add_argument("--max_images", type=int, default=0, help="Debug only; 0 means all")
    p.add_argument("--self_test", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(value, str(tmp))
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_split(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def suffix_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def find_by_stem(root: Path, stem: str, suffixes: list[str]) -> Path | None:
    for suffix in suffixes:
        p = root / f"{stem}{suffix}"
        if p.is_file():
            return p.resolve()
    return None


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def load_label_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im).copy()
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(f"{path}: expected integer-ID mask, got color image {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D mask, got {arr.shape}")
    return arr.astype(np.int32)


# -----------------------------------------------------------------------------
# GT masks -> weak labels ONLY
# -----------------------------------------------------------------------------

def decode_present_objects(mask: np.ndarray, mode: str, ignore: int) -> list[int]:
    values = [int(x) for x in np.unique(mask).tolist()]

    if mode == "zero_based":
        ids = [x for x in values if x != ignore and 0 <= x < 20]
    elif mode == "voc21":
        ids = [x - 1 for x in values if x not in (0, ignore) and 1 <= x <= 20]
    else:
        raise ValueError(mode)

    return sorted(set(ids))


def gt_object_region(mask: np.ndarray, object_id: int, mode: str) -> np.ndarray:
    if mode == "zero_based":
        return mask == object_id
    if mode == "voc21":
        return mask == (object_id + 1)
    raise ValueError(mode)


def derive_part_presence(
    part_mask: np.ndarray,
    object_region: np.ndarray,
    object_name: str,
    *,
    part_id_offset: int,
    part_ignore: int,
) -> tuple[list[int], list[str], list[int]]:
    """Return valid part IDs/names for one semantic object and foreign raw IDs for audit."""
    allowed = set(PART_IDS_BY_OBJECT.get(object_name, []))
    raw_values = [
        int(x) for x in np.unique(part_mask[object_region]).tolist()
        if int(x) != int(part_ignore)
    ]

    valid: list[int] = []
    foreign: list[int] = []
    for raw in raw_values:
        pid = raw - int(part_id_offset)
        if pid in allowed:
            valid.append(pid)
        elif 0 <= pid < len(PART_CLASSES):
            foreign.append(pid)

    valid = sorted(set(valid))
    foreign = sorted(set(foreign))
    return valid, [PART_CLASSES[pid] for pid in valid], foreign


# -----------------------------------------------------------------------------
# Frozen Talk2DINO predictor
# -----------------------------------------------------------------------------

def unwrap_projector_state(ckpt: Any, expected: set[str]) -> dict[str, torch.Tensor]:
    obj = ckpt
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "projector_state_dict"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise ValueError("projector checkpoint is not a state_dict mapping")

    state = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    for prefix in ("proj.", "model.proj."):
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub and set(sub) == expected:
            state = sub
            break

    if set(state) != expected:
        raise ValueError(
            "projector state_dict keys mismatch\n"
            f"missing={sorted(expected - set(state))[:20]}\n"
            f"extra={sorted(set(state) - expected)[:20]}"
        )
    return state


def build_model(args: argparse.Namespace, repo: Path, device: torch.device):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src" / "open_vocabulary_segmentation"))

    from models.dinotext.dinotext import DINOText

    model = DINOText(
        model_name=args.model_name,
        resize_dim=CROP_DIM,
        clip_model_name=args.clip_model_name,
        proj_class=args.proj_class,
        proj_name="__manual_load__",
        proj_model=args.proj_model,
        pre_trained=False,
        is_eval=True,
        use_avg_text_token=False,
        keep_cls=False,
        keep_end_seq=False,
        with_bg_clean=True,
    ).to(device)

    weight = Path(args.projector_weight).expanduser()
    if not weight.is_absolute():
        weight = (repo / weight).resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)

    state = unwrap_projector_state(
        torch_load(weight),
        set(model.proj.state_dict().keys()),
    )
    model.proj.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, weight


def bgr_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.ascontiguousarray(bgr)
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .contiguous()
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
    )


@torch.inference_mode()
def talk2dino_slide_scores(
    model,
    pil: Image.Image,
    text_emb: torch.Tensor,
    classnames: list[str],
    *,
    device: torch.device,
    pamr: bool,
    lambda_bg: float,
    eval_max_long: int,
    eval_max_short: int,
    slide_crop: int,
    slide_stride: int,
) -> torch.Tensor:
    """Dynamic-class version of the Pascal/VOC 448/224 slide inference."""
    import mmcv

    rgb = np.asarray(pil, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    resized = mmcv.imrescale(
        bgr,
        (int(eval_max_long), int(eval_max_short)),
        return_scale=False,
        interpolation="bilinear",
    )
    rh, rw = resized.shape[:2]
    k = len(classnames)
    if text_emb.shape[0] != k or k == 0:
        raise ValueError("text/class count mismatch")

    crop = int(slide_crop)
    stride = int(slide_stride)
    h_grids = max(rh - crop + stride - 1, 0) // stride + 1
    w_grids = max(rw - crop + stride - 1, 0) // stride + 1

    preds = torch.zeros((1, k, rh, rw), dtype=torch.float32, device=device)
    count = torch.zeros((1, 1, rh, rw), dtype=torch.float32, device=device)

    for hi in range(h_grids):
        for wi in range(w_grids):
            y1, x1 = hi * stride, wi * stride
            y2, x2 = min(y1 + crop, rh), min(x1 + crop, rw)
            y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)

            crop_bgr = resized[y1:y2, x1:x2]
            scores, _ = model.generate_masks(
                bgr_tensor(crop_bgr, device),
                img_metas=[{
                    "ori_filename": "",
                    "img_shape": (crop_bgr.shape[0], crop_bgr.shape[1], 3),
                }],
                text_emb=text_emb,
                classnames=classnames,
                apply_pamr=pamr,
                lambda_bg=lambda_bg,
            )
            scores = scores.float()
            expected = (1, k, y2 - y1, x2 - x1)
            if tuple(scores.shape) != expected:
                raise ValueError(f"Talk2DINO crop score shape {tuple(scores.shape)} != {expected}")
            if not torch.isfinite(scores).all():
                raise ValueError("NaN/Inf in Talk2DINO scores")

            preds[:, :, y1:y2, x1:x2] += scores
            count[:, :, y1:y2, x1:x2] += 1.0

    if (count == 0).any():
        raise RuntimeError("slide inference left uncovered pixels")
    preds = preds / count

    oh, ow = pil.height, pil.width
    if (rh, rw) != (oh, ow):
        preds = F.interpolate(preds, size=(oh, ow), mode="bilinear", align_corners=False)
    return preds[0]


# -----------------------------------------------------------------------------
# Predicted-mask crop -> offline DINO crop cache
# -----------------------------------------------------------------------------

def square_crop_box(mask: np.ndarray, width: int, height: int, expand_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty predicted object mask")

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    side = max(x2 - x1, y2 - y1) * float(expand_ratio)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    left = max(0, int(round(cx - side / 2.0)))
    top = max(0, int(round(cy - side / 2.0)))
    right = min(width, int(round(cx + side / 2.0)))
    bottom = min(height, int(round(cy + side / 2.0)))
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"invalid crop box {(left, top, right, bottom)}")
    return left, top, right, bottom


def mask_to_patch_grid(mask: np.ndarray, box: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Map a predicted pixel mask inside ``box`` to the 32x32 DINO crop grid.

    Use ANY-OVERLAP / occupancy rasterization rather than nearest-neighbor
    downsampling.  Direct NEAREST resize can completely erase a thin or tiny
    but non-empty predicted region.  Here every positive source pixel votes
    for the corresponding output patch bin, so a non-empty predicted mask
    cannot disappear solely because of patch-grid discretization.

    This changes only mask discretization.  The predicted mask itself, bbox,
    RGB crop, 448x448 DINO re-encoding, and all downstream W-training logic
    remain unchanged.
    """
    x1, y1, x2, y2 = box
    crop = np.asarray(mask[y1:y2, x1:x2], dtype=np.uint8)
    if crop.ndim != 2 or crop.size == 0 or not crop.any():
        raise ValueError("predicted foreground vanished before patch rasterization")

    h, w = crop.shape
    ys, xs = np.nonzero(crop)

    # Assign each foreground source pixel to its geometrically corresponding
    # cell in the 32x32 crop grid.  floor(coord * grid / size) implements
    # half-open bins and guarantees indices in [0, PATCH_GRID-1].
    gy = np.minimum((ys.astype(np.int64) * PATCH_GRID) // h, PATCH_GRID - 1)
    gx = np.minimum((xs.astype(np.int64) * PATCH_GRID) // w, PATCH_GRID - 1)

    patch = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.bool_)
    patch[gy, gx] = True

    out = torch.from_numpy(patch.reshape(-1).copy()).to(dtype=torch.bool)
    if int(out.sum().item()) == 0:
        raise AssertionError(
            "non-empty predicted foreground disappeared during occupancy rasterization"
        )
    return out


def crop_transform() -> T.Compose:
    return T.Compose([
        T.Resize((CROP_DIM, CROP_DIM), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def self_test() -> None:
    assert len(PART_CLASSES) == 234
    assert len(OBJECT_CLASSES) == 44
    assert sorted(PART_TO_OBJECT) == list(range(234))

    assert PART_CLASSES[0] == "person's head"
    assert PART_CLASSES[11] == "door's handle"
    assert PART_CLASSES[193] == "van's license plate"
    assert PART_CLASSES[233] == "light's diffusor"

    part = np.full((6, 8), 65535, dtype=np.uint16)
    part[1:3, 1:4] = 0
    part[3:5, 2:6] = 1

    present = sorted(
        int(x) for x in np.unique(part)
        if int(x) != 65535
    )
    assert present == [0, 1]
    object_ids = sorted({int(PART_TO_OBJECT[x]) for x in present})
    assert object_ids == [0]

    pm = np.zeros((10, 20), dtype=np.uint8)
    pm[2:8, 5:15] = 1
    box = square_crop_box(pm, 20, 10, 1.2)
    patch = mask_to_patch_grid(pm, box)

    assert tuple(patch.shape) == (1024,)
    assert patch.dtype == torch.bool
    assert int(patch.sum().item()) > 0

    print("SELF_TEST_PASS")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    for required_name in ("image_root", "part_mask_root", "output_pth"):
        if not getattr(args, required_name):
            raise ValueError(f"--{required_name} is required unless --self_test is used")

    if args.model_name != "dinov2_vitb14_reg":
        raise ValueError("final pipeline requires dinov2_vitb14_reg")
    if args.clip_model_name != "ViT-B/16":
        raise ValueError("final pipeline requires CLIP ViT-B/16")
    if (args.eval_max_long, args.eval_max_short) != (2048, 448):
        raise ValueError("final object path requires img_scale=(2048,448)")
    if (args.slide_crop, args.slide_stride) != (448, 224):
        raise ValueError("final object path requires slide crop=448 stride=224")
    if not math.isfinite(args.crop_expand_ratio) or args.crop_expand_ratio < 1.0:
        raise ValueError("crop_expand_ratio must be finite and >=1")
    if args.crop_batch_size < 1:
        raise ValueError("crop_batch_size must be >=1")

    repo = Path(args.repo_root).expanduser().resolve()
    if not (repo / "src" / "open_vocabulary_segmentation").is_dir():
        raise FileNotFoundError(f"not a Talk2DINO_official_bg repo: {repo}")

    image_root = Path(args.image_root).expanduser().resolve()
    part_root = Path(args.part_mask_root).expanduser().resolve()
    for root in (image_root, part_root):
        if not root.is_dir():
            raise FileNotFoundError(root)

    output = Path(args.output_pth).expanduser()
    if not output.is_absolute():
        output = (repo / output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    image_suffixes = suffix_list(args.image_suffixes)
    mask_suffixes = suffix_list(args.mask_suffixes)
    if args.split_file:
        split_path = Path(args.split_file).expanduser()
        if not split_path.is_absolute():
            split_path = (repo / split_path).resolve()
        stems = read_split(split_path)
    else:
        image_stems = {p.stem for p in image_root.iterdir() if p.is_file()}
        part_stems = {p.stem for p in part_root.iterdir() if p.is_file()}
        stems = sorted(image_stems & part_stems)

    if args.max_images > 0:
        stems = stems[: args.max_images]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model, weight = build_model(args, repo, device)

    # Fixed full object text bank. During each image only present rows are used.
    object_tokens = model.build_dataset_class_tokens(args.template, OBJECT_CLASSES)
    object_text = model.build_text_embedding(object_tokens).to(device=device, dtype=torch.float32)
    expected_object_text_shape = (len(OBJECT_CLASSES), VISION_DIM)
    if tuple(object_text.shape) != expected_object_text_shape:
        raise ValueError(
            f"bad object text bank shape: {tuple(object_text.shape)} "
            f"!= {expected_object_text_shape}"
        )

    transform = crop_transform()
    annotations: list[dict] = []
    images_meta: list[dict] = []
    pending_crops: list[Image.Image] = []
    pending_indices: list[int] = []
    stats: defaultdict[str, int] = defaultdict(int)
    start = time.perf_counter()

    def flush() -> None:
        nonlocal pending_crops, pending_indices
        if not pending_crops:
            return
        batch = torch.stack([transform(im) for im in pending_crops], dim=0).to(
            device=device, dtype=torch.float32
        )
        dino_out = model.model(batch, is_training=True)
        if not isinstance(dino_out, Mapping) or "x_norm_patchtokens" not in dino_out:
            raise RuntimeError("DINOv2 output missing x_norm_patchtokens")
        tokens = dino_out["x_norm_patchtokens"]
        expected = (len(pending_crops), PATCH_COUNT, VISION_DIM)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"DINO crop tokens {tuple(tokens.shape)} != {expected}")
        if not torch.isfinite(tokens).all():
            raise ValueError("NaN/Inf in crop patch tokens")
        for j, ann_idx in enumerate(pending_indices):
            annotations[ann_idx]["cropaug_patch_tokens"] = (
                tokens[j].to(device="cpu", dtype=torch.float16).contiguous()
            )
        stats["crop_batches"] += 1
        pending_crops = []
        pending_indices = []

    for stem in tqdm(
        stems,
        desc="ADE weak labels -> pred obj -> crop -> DINO cache",
    ):
        stats["requested_images"] += 1

        img_path = find_by_stem(image_root, stem, image_suffixes)
        part_path = find_by_stem(part_root, stem, mask_suffixes)

        if img_path is None or part_path is None:
            stats["missing_pair"] += 1
            continue

        pil = load_rgb(img_path)
        part_mask = load_label_mask(part_path)

        if part_mask.shape != (pil.height, pil.width):
            raise ValueError(
                f"{stem}: part mask shape {part_mask.shape} "
                f"!= RGB {(pil.height, pil.width)}"
            )

        # Weak-label extraction ONLY: keep the set of IDs, never positions.
        present_part_ids = sorted(
            int(x)
            for x in np.unique(part_mask)
            if int(x) != int(args.part_ignore)
        )

        invalid = [
            pid for pid in present_part_ids
            if pid < 0 or pid >= len(PART_CLASSES)
        ]
        if invalid:
            raise ValueError(
                f"{stem}: invalid ADE234 part IDs {invalid}"
            )

        if not present_part_ids:
            stats["no_part_label"] += 1
            continue

        object_ids = sorted({
            int(PART_TO_OBJECT[pid])
            for pid in present_part_ids
        })

        if not object_ids:
            stats["no_object_label"] += 1
            continue

        object_names = [
            OBJECT_CLASSES[cid]
            for cid in object_ids
        ]

        weak_parts: dict[int, tuple[list[int], list[str]]] = {}

        for cid in object_ids:
            pids = [
                pid
                for pid in present_part_ids
                if int(PART_TO_OBJECT[pid]) == cid
            ]
            pnames = [
                PART_CLASSES[pid]
                for pid in pids
            ]
            if not pids:
                raise AssertionError(
                    f"{stem}: empty part set for {OBJECT_CLASSES[cid]}"
                )
            weak_parts[cid] = (pids, pnames)

        # Destroy the GT spatial array before any Talk2DINO/crop/DINO operation.
        del part_mask

        foreign_part_audit: dict[str, list[int]] = {}

        # From here onward no GT spatial array is used in any computation.
        present_text = object_text[
            torch.tensor(object_ids, dtype=torch.long, device=device)
        ]
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
        # local 0=background, local 1..K correspond to object_ids in order.
        hard = torch.cat([background, fg_scores], dim=0).argmax(dim=0)

        ready_classes: list[str] = []
        for local_channel, cid in enumerate(object_ids, start=1):
            pids, pnames = weak_parts[cid]
            # Objects with no part presence remain valid competitors for object
            # segmentation, but they have nothing to contribute to W training.
            if not pids:
                stats["object_without_part_presence"] += 1
                continue

            pred = (hard == local_channel).detach().cpu().numpy().astype(np.uint8)
            if int(pred.sum()) == 0:
                stats["empty_predicted_object_mask"] += 1
                continue

            box = square_crop_box(
                pred,
                width=pil.width,
                height=pil.height,
                expand_ratio=args.crop_expand_ratio,
            )
            patch_mask = mask_to_patch_grid(pred, box)
            x1, y1, x2, y2 = box
            crop = pil.crop((x1, y1, x2, y2))

            record = {
                "id": len(annotations),
                "image_id": stem,
                "category_id": int(cid),
                "class_name": OBJECT_CLASSES[cid],
                # ADE image-level part-presence labels only; no GT positions retained.
                "part_category_id": pids,
                "part_class_name": pnames,
                # All spatial/cache fields below come from the PREDICTED object mask.
                "cropaug_box_xyxy": torch.tensor(box, dtype=torch.long),
                "pred_obj_mask_patch": patch_mask.contiguous(),
                "pred_obj_mask_pixel_area": int(pred.sum()),
                "pred_obj_mask_patch_area": int(patch_mask.sum().item()),
                "pred_obj_mask_source": "talk2dino_image_present_object_labels",
            }
            annotations.append(record)
            pending_crops.append(crop)
            pending_indices.append(len(annotations) - 1)
            ready_classes.append(OBJECT_CLASSES[cid])
            stats["ready_object_crops"] += 1

            if len(pending_crops) >= args.crop_batch_size:
                flush()

        images_meta.append({
            "id": stem,
            "file_name": str(img_path),
            "height": pil.height,
            "width": pil.width,
            "object_category_id_present": object_ids,
            "object_class_name_present": object_names,
            "ready_object_class_name": ready_classes,
            "foreign_part_id_audit": foreign_part_audit,
        })
        stats["processed_images"] += 1

    flush()

    # Final offline-W cache contract.
    for i, ann in enumerate(annotations):
        tok = ann.get("cropaug_patch_tokens")
        pm = ann.get("pred_obj_mask_patch")
        box = ann.get("cropaug_box_xyxy")
        pids = ann.get("part_category_id", [])
        if not torch.is_tensor(tok) or tuple(tok.shape) != (1024, 768) or tok.dtype != torch.float16:
            raise ValueError(f"annotation {i}: invalid cropaug_patch_tokens")
        if not torch.is_tensor(pm) or tuple(pm.shape) != (1024,) or pm.dtype != torch.bool or int(pm.sum()) <= 0:
            raise ValueError(f"annotation {i}: invalid pred_obj_mask_patch")
        if not torch.is_tensor(box) or tuple(box.shape) != (4,) or box.dtype != torch.long:
            raise ValueError(f"annotation {i}: invalid cropaug_box_xyxy")
        if not pids or len(set(pids)) != len(pids):
            raise ValueError(f"annotation {i}: invalid part presence IDs")
        parent = ann["class_name"]
        allowed = set(PART_IDS_BY_OBJECT[parent])
        if any(pid not in allowed for pid in pids):
            raise ValueError(f"annotation {i}: part IDs do not belong to {parent}")

    elapsed = time.perf_counter() - start
    stats = dict(stats)
    stats["output_annotations"] = len(annotations)

    payload = {
        "images": images_meta,
        "annotations": annotations,
        "pred_obj_cropaug_meta": {
            "format": "ade20kpart234_weaklabels_predobj_cropaug_v1",
            "protocol": (
                "ADE20K-Part-234 GT part masks are read only through np.unique "
                "to derive image-level semantic part presence and parent-object "
                "presence. No GT pixel location, object mask, point, or box is "
                "used by Talk2DINO foreground prediction, cropping, DINO patch "
                "selection, RelProto construction, or W training. "
                "cropaug_box_xyxy, pred_obj_mask_patch, and cropaug_patch_tokens "
                "are all derived from the frozen Talk2DINO predicted object mask."
            ),
            "part_ignore": int(args.part_ignore),
            "target_spatial_supervision": "none",
            "object_prediction_candidate_scope": "background + image-present semantic object labels",
            "same_class_instances": "one semantic-class union mask/crop",
            "model_name": args.model_name,
            "clip_model_name": args.clip_model_name,
            "template": args.template,
            "projector_weight": str(weight),
            "projector_weight_sha256": sha256(weight),
            "with_bg_clean": True,
            "pamr": bool(args.pamr),
            "bg_thresh": float(args.bg_thresh),
            "lambda_bg": float(args.lambda_bg),
            "outer_resize": [int(args.eval_max_long), int(args.eval_max_short)],
            "slide_crop": int(args.slide_crop),
            "slide_stride": int(args.slide_stride),
            "crop_expand_ratio": float(args.crop_expand_ratio),
            "crop_resize": [448, 448],
            "crop_interpolation": "BICUBIC",
            "patch_grid": [32, 32],
            "patch_token_shape": [1024, 768],
            "patch_token_dtype": "float16",
            "foreground_key": "pred_obj_mask_patch",
            "part_presence_key": "part_category_id",
            "elapsed_seconds": elapsed,
            "stats": stats,
        },
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[save] {len(annotations)} object crops -> {output}")
    atomic_torch_save(payload, output)
    print("[done]")
    print("[W loader] patch_key=cropaug_patch_tokens")
    print("[W loader] foreground_source=precomputed")
    print("[W loader] foreground_key=pred_obj_mask_patch")
    print("[W loader] presence_source=annotation")
    print("[W loader] presence_key=part_category_id")


if __name__ == "__main__":
    main()
