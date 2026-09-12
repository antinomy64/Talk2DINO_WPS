#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the core hypothesis directly:

  within-object TEXT part-relation structure
                vs
  within-object GT DINO visual-part structure.

Three text representations are compared against the SAME GT visual prototypes:
  1) raw CLIP prompt-mean features
  2) initial COCO projector: normalize(P_initial(raw))
  3) corrected PartStruct projector: normalize(P_partstruct(raw))

GT visual prototype protocol (TRAIN split by default):
  GT object semantic region -> square crop (expand=1.2) -> RGB resize 448x448
  -> same DINOv2 ViT-B/14-reg x_norm_patchtokens path used by
     extract_predobj_cropaug.py -> 32x32 patch grid.
  GT part masks are used ONLY for this scientific audit to compute, for each
  patch, the fraction of pixels belonging to each part. One weighted DINO
  prototype is computed per (GT object crop, part), then those prototypes are
  averaged equally across crops for the same semantic part.

For each object with >=3 available parts, pairwise cosine similarities are
vectorized from the upper triangle and exact scipy Spearman is computed against
GT visual structure. The final metric is an equal-weight macro mean over objects.

This script is read-only with respect to source RGB/masks and checkpoints. It
writes only a GT-visual prototype cache and a JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

NUM_PARTS = 116
VISION_DIM = 768
PATCH_GRID = 32
PATCH_COUNT = PATCH_GRID * PATCH_GRID
PROTOCOL = "text_vs_gt_visual_structure_v1"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo_root", default=".")
    p.add_argument("--data_root", default="data/PascalPart116")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--split_file", default="")
    p.add_argument(
        "--text_bank",
        default="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt",
    )
    p.add_argument("--model_config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument(
        "--initial_weight",
        default="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
    )
    p.add_argument(
        "--partstruct_weight",
        default=(
            "weights/vitb_mlp_infonce_coco2014_clean_ft10_"
            "partstruct_rawpath_w1e4_lr1e5.pth"
        ),
    )
    # Only needed to construct current DINOText image backbone; projector output
    # is never used when building GT visual prototypes.
    p.add_argument(
        "--dino_loader_projector_weight",
        default="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
    )
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--clip_model_name", default="ViT-B/16")
    p.add_argument("--proj_class", default="vitb_mlp_infonce")
    p.add_argument("--proj_model", default="ProjectionLayer")
    p.add_argument("--obj_mask_mode", choices=("zero_based", "voc21"), default="zero_based")
    p.add_argument("--obj_ignore", type=int, default=255)
    p.add_argument("--part_id_offset", type=int, choices=(0, 1), default=0)
    p.add_argument("--part_ignore", type=int, default=255)
    p.add_argument("--crop_expand_ratio", type=float, default=1.2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_images", type=int, default=0, help="debug only; 0=all")
    p.add_argument(
        "--visual_cache",
        default=(
            "feature/pascalpart116_gt_visual_structure/"
            "gt_dinov2_vitb14reg_train_objcrop_x1p2.pt"
        ),
    )
    p.add_argument(
        "--json_out",
        default="output/text_vs_gt_visual_structure_audit.json",
    )
    p.add_argument("--rebuild_visual_cache", action="store_true")
    p.add_argument("--overwrite_json", action="store_true")
    return p.parse_args()


def resolve(repo: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def torch_load(path: Path) -> Any:
    # User's PyTorch requires a string path when mmap=True.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im).copy()
    if a.ndim == 3:
        if a.shape[2] >= 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]):
            a = a[..., 0]
        else:
            raise ValueError(f"{path}: expected ID mask, got {a.shape}")
    if a.ndim != 2:
        raise ValueError(f"{path}: expected 2D mask, got {a.shape}")
    return a.astype(np.int32)


def find_by_stem(root: Path, stem: str, suffixes=(".jpg", ".jpeg", ".png")):
    for s in suffixes:
        p = root / f"{stem}{s}"
        if p.is_file():
            return p
    return None


def get_stems(data_root: Path, split: str, split_file: str, repo: Path):
    images = data_root / "images" / split
    objs = data_root / "annotations_detectron2_obj" / split
    parts = data_root / "annotations_detectron2_part" / split
    sf = resolve(repo, split_file) if split_file else data_root / f"{split}.txt"
    if sf.is_file():
        stems = [Path(x.strip()).stem for x in sf.read_text().splitlines() if x.strip()]
        if len(stems) != len(set(stems)):
            raise RuntimeError(f"duplicate entries in {sf}")
        source = str(sf)
    else:
        stems = sorted(
            {p.stem for p in images.iterdir() if p.is_file()}
            & {p.stem for p in objs.iterdir() if p.is_file()}
            & {p.stem for p in parts.iterdir() if p.is_file()}
        )
        source = "intersection(images,obj_masks,part_masks)"
    if not stems:
        raise RuntimeError("no samples resolved")
    return stems, source


def gt_patch_fractions(part_crop: np.ndarray, pid: int) -> np.ndarray:
    """GT part-pixel fraction in each geometrically corresponding 32x32 bin."""
    h, w = part_crop.shape
    yy = np.arange(h, dtype=np.int64)[:, None]
    xx = np.arange(w, dtype=np.int64)[None, :]
    gy = np.minimum((yy * PATCH_GRID) // h, PATCH_GRID - 1)
    gx = np.minimum((xx * PATCH_GRID) // w, PATCH_GRID - 1)
    cell = np.broadcast_to(gy * PATCH_GRID + gx, (h, w)).reshape(-1)
    flat = part_crop.reshape(-1)
    total = np.bincount(cell, minlength=PATCH_COUNT).astype(np.float64)
    hit = np.bincount(cell[flat == int(pid)], minlength=PATCH_COUNT).astype(np.float64)
    frac = np.divide(hit, total, out=np.zeros_like(hit), where=total > 0)
    return frac.astype(np.float32)


def build_visual_cache(args, repo, data_root, stems, split_source, cache_path):
    # Reuse the current W-cache code for taxonomy, GT decoding, crop geometry,
    # DINO model loading, and 448x448 transform. This prevents implementation drift.
    import extract_predobj_cropaug as pred

    if len(pred.PART_CLASSES) != NUM_PARTS:
        raise RuntimeError("PascalPart116 taxonomy mismatch")
    if args.model_name != "dinov2_vitb14_reg":
        raise ValueError("audit is hard-locked to dinov2_vitb14_reg")
    if not math.isfinite(args.crop_expand_ratio) or args.crop_expand_ratio < 1:
        raise ValueError("crop_expand_ratio must be >=1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    image_root = data_root / "images" / args.split
    obj_root = data_root / "annotations_detectron2_obj" / args.split
    part_root = data_root / "annotations_detectron2_part" / args.split
    for p in (image_root, obj_root, part_root):
        if not p.is_dir():
            raise FileNotFoundError(p)

    loader_args = SimpleNamespace(
        model_name=args.model_name,
        clip_model_name=args.clip_model_name,
        proj_class=args.proj_class,
        proj_model=args.proj_model,
        projector_weight=str(resolve(repo, args.dino_loader_projector_weight)),
    )
    model, loader_weight = pred.build_model(loader_args, repo, device)
    transform = pred.crop_transform()

    # Primary aggregation: equal mean across (GT object crop, part) prototypes.
    crop_sum = torch.zeros(NUM_PARTS, VISION_DIM, dtype=torch.float64)
    crop_count = torch.zeros(NUM_PARTS, dtype=torch.long)

    # Secondary robustness aggregation: pool all GT-weighted patch evidence.
    pooled_sum = torch.zeros(NUM_PARTS, VISION_DIM, dtype=torch.float64)
    pooled_weight = torch.zeros(NUM_PARTS, dtype=torch.float64)

    pixel_count = torch.zeros(NUM_PARTS, dtype=torch.long)
    object_crop_count = defaultdict(int)
    stats = defaultdict(int)
    pending_images = []
    pending_meta = []  # (object_name, pids, part_crop)

    @torch.inference_mode()
    def flush():
        nonlocal pending_images, pending_meta
        if not pending_images:
            return
        batch = torch.stack([transform(im) for im in pending_images], 0).to(device=device, dtype=torch.float32)
        out = model.model(batch, is_training=True)
        if "x_norm_patchtokens" not in out:
            raise RuntimeError("DINO output missing x_norm_patchtokens")
        tokens = out["x_norm_patchtokens"].float()
        expected = (len(pending_images), PATCH_COUNT, VISION_DIM)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"DINO tokens {tuple(tokens.shape)} != {expected}")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("NaN/Inf in DINO tokens")

        for b, (obj_name, pids, part_crop) in enumerate(pending_meta):
            tok = tokens[b]
            for pid in pids:
                w = torch.from_numpy(gt_patch_fractions(part_crop, pid)).to(device=device, dtype=torch.float32)
                ws = float(w.sum())
                if ws <= 0:
                    stats["part_vanished_on_grid"] += 1
                    continue
                proto = (tok * w[:, None]).sum(0) / w.sum()
                if not torch.isfinite(proto).all():
                    raise RuntimeError("non-finite GT visual prototype")
                crop_sum[pid] += proto.cpu().double()
                crop_count[pid] += 1
                pooled_sum[pid] += (tok * w[:, None]).sum(0).cpu().double()
                pooled_weight[pid] += ws
                stats["part_crop_prototypes"] += 1
            object_crop_count[obj_name] += 1
            stats["gt_object_crops"] += 1

        pending_images, pending_meta = [], []
        del batch, out, tokens

    for stem in tqdm(stems, desc="GT object crop -> GT DINO part prototypes"):
        stats["requested_images"] += 1
        ip = find_by_stem(image_root, stem)
        op = find_by_stem(obj_root, stem, (".png",))
        pp = find_by_stem(part_root, stem, (".png",))
        if ip is None or op is None or pp is None:
            stats["missing_triplet"] += 1
            continue
        pil = load_rgb(ip)
        obj = load_mask(op)
        part = load_mask(pp)
        if obj.shape != (pil.height, pil.width) or part.shape != (pil.height, pil.width):
            raise ValueError(f"{stem}: RGB/mask shape mismatch")

        for cid in pred.decode_present_objects(obj, args.obj_mask_mode, args.obj_ignore):
            obj_name = pred.VOC20_CLASSES[cid]
            region = pred.gt_object_region(obj, cid, args.obj_mask_mode)
            pids, _, foreign = pred.derive_part_presence(
                part, region, obj_name,
                part_id_offset=args.part_id_offset,
                part_ignore=args.part_ignore,
            )
            if foreign:
                raise RuntimeError(f"{stem}/{obj_name}: foreign part IDs={foreign}")
            if not pids:
                continue
            for pid in pids:
                pixel_count[pid] += int(np.count_nonzero(part[region] == pid))

            box = pred.square_crop_box(
                region.astype(np.uint8), pil.width, pil.height,
                args.crop_expand_ratio,
            )
            x1, y1, x2, y2 = box
            pending_images.append(pil.crop((x1, y1, x2, y2)))
            pending_meta.append((obj_name, pids, part[y1:y2, x1:x2].copy()))
            if len(pending_images) >= args.batch_size:
                flush()
        stats["processed_images"] += 1
    flush()

    crop_balanced = torch.zeros(NUM_PARTS, VISION_DIM, dtype=torch.float32)
    pixel_weighted = torch.zeros(NUM_PARTS, VISION_DIM, dtype=torch.float32)
    available = torch.zeros(NUM_PARTS, dtype=torch.bool)
    for pid in range(NUM_PARTS):
        if int(crop_count[pid]) > 0 and float(pooled_weight[pid]) > 0:
            crop_balanced[pid] = (crop_sum[pid] / float(crop_count[pid])).float()
            pixel_weighted[pid] = (pooled_sum[pid] / float(pooled_weight[pid])).float()
            available[pid] = True

    payload = {
        "protocol_version": PROTOCOL,
        "protocol": {
            "split": args.split,
            "split_source": split_source,
            "requested_images": len(stems),
            "gt_object_crop": True,
            "crop_expand_ratio": float(args.crop_expand_ratio),
            "crop_resize": [448, 448],
            "dino_model": args.model_name,
            "dino_output": "x_norm_patchtokens",
            "patch_grid": [32, 32],
            "part_patch_weight": "GT part-pixel fraction per patch bin",
            "primary_aggregation": "mean of per-GT-object-crop part prototypes",
            "secondary_aggregation": "globally pooled GT-weighted patch evidence",
            "gt_use": "analysis only; never used to train W or segmentation model",
        },
        "part_classes": list(pred.PART_CLASSES),
        "crop_balanced_prototypes": crop_balanced,
        "pixel_weighted_prototypes": pixel_weighted,
        "available": available,
        "crop_prototype_count": crop_count,
        "part_pixel_count": pixel_count,
        "pooled_weight": pooled_weight,
        "object_crop_count": dict(object_crop_count),
        "stats": dict(stats),
        "dino_loader_projector": str(loader_weight),
        "dino_loader_projector_sha256": sha256(loader_weight),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(cache_path))
    return payload


def validate_visual_cache(payload, canonical_names, args):
    if payload.get("protocol_version") != PROTOCOL:
        raise ValueError("visual-cache protocol mismatch")
    if list(payload.get("part_classes", [])) != canonical_names:
        raise ValueError("visual-cache taxonomy/order mismatch")
    if payload.get("protocol", {}).get("split") != args.split:
        raise ValueError("visual-cache split mismatch")
    if abs(float(payload.get("protocol", {}).get("crop_expand_ratio", -1)) - args.crop_expand_ratio) > 1e-12:
        raise ValueError("visual-cache crop_expand_ratio mismatch")
    for k in ("crop_balanced_prototypes", "pixel_weighted_prototypes"):
        x = payload.get(k)
        if not torch.is_tensor(x) or tuple(x.shape) != (NUM_PARTS, VISION_DIM) or not torch.isfinite(x).all():
            raise ValueError(f"bad visual-cache tensor: {k}")


def text_variants(repo, text_bank, model_config, initial_weight, partstruct_weight, device):
    import train_relproto_alignemt as tr
    raw, names, occurrences = tr.load_raw_clip_bank(
        text_bank,
        expected_clip_model="ViT-B/16",
        expected_template="sub_imagenet_template",
    )
    raw = raw.float().cpu()
    if tuple(raw.shape) != (116, 512):
        raise ValueError(raw.shape)
    out = {"raw_clip": raw}
    provenance = {
        "raw_text_bank": str(text_bank),
        "raw_text_bank_sha256": sha256(text_bank),
    }
    for label, weight in (
        ("initial_projector", initial_weight),
        ("partstruct_projector", partstruct_weight),
    ):
        projector, info = tr.load_frozen_projector(
            project_root=repo,
            config_path=model_config,
            weight_path=weight,
            device=device,
        )
        with torch.inference_mode():
            # EXACT RelProto/eval text path: raw prompt mean -> projector -> L2.
            z = tr.project_raw_clip_bank(raw, projector, device=device, batch_size=128)
        if tuple(z.shape) != (116, 768) or not torch.isfinite(z).all():
            raise RuntimeError(f"bad projected text bank: {label}")
        out[label] = z.cpu().float()
        provenance[label] = {
            "weight": str(weight),
            "weight_sha256": info["weights_sha256"],
        }
        del projector
    return names, out, provenance


def relation_vector(feat: torch.Tensor, ids: list[int]) -> np.ndarray:
    idx = torch.tensor(ids, dtype=torch.long)
    x = feat.index_select(0, idx).double()
    x = F.normalize(x, dim=-1, eps=1e-12)
    sim = x @ x.T
    tri = torch.triu_indices(len(ids), len(ids), offset=1)
    return sim[tri[0], tri[1]].numpy()


def rho(a, b):
    if a.shape != b.shape or a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def compare(texts, visual, available, names):
    groups = defaultdict(list)
    for pid, n in enumerate(names):
        groups[n.split("'s ", 1)[0]].append(pid)
    per_obj = {}
    scores = {k: [] for k in texts}
    for obj, full_ids in groups.items():
        ids = [i for i in full_ids if bool(available[i])]
        row = {
            "defined_parts": len(full_ids),
            "available_parts": len(ids),
        }
        if len(ids) < 3:
            row["status"] = "SKIP_<3_parts"
            per_obj[obj] = row
            continue
        v = relation_vector(visual, ids)
        if np.ptp(v) == 0:
            row["status"] = "SKIP_constant_visual"
            per_obj[obj] = row
            continue
        row.update({
            "status": "OK",
            "part_ids": ids,
            "part_names": [names[i] for i in ids],
            "pair_count": len(v),
        })
        ok = True
        for label, feat in texts.items():
            r = rho(relation_vector(feat, ids), v)
            row[label] = r
            ok = ok and np.isfinite(r)
        if ok:
            for label in texts:
                scores[label].append(row[label])
        else:
            row["status"] = "SKIP_nonfinite"
        per_obj[obj] = row
    lengths = {k: len(v) for k, v in scores.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values()), 0) == 0:
        raise RuntimeError(f"inconsistent valid-object counts: {lengths}")
    return {
        "valid_objects": next(iter(lengths.values())),
        "object_macro_spearman": {k: float(np.mean(v)) for k, v in scores.items()},
        "per_object": per_obj,
    }


def main():
    args = parse_args()
    started = time.time()
    repo = Path(args.repo_root).expanduser().resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if not (repo / "train_relproto_alignemt.py").is_file():
        raise FileNotFoundError("run from Talk2DINO_WPS repository root")

    data_root = resolve(repo, args.data_root)
    text_bank = resolve(repo, args.text_bank)
    model_config = resolve(repo, args.model_config)
    initial_weight = resolve(repo, args.initial_weight)
    partstruct_weight = resolve(repo, args.partstruct_weight)
    visual_cache = resolve(repo, args.visual_cache)
    json_out = resolve(repo, args.json_out)
    for p in (data_root, text_bank, model_config, initial_weight, partstruct_weight):
        if not p.exists():
            raise FileNotFoundError(p)

    stems, split_source = get_stems(data_root, args.split, args.split_file, repo)
    if args.max_images > 0:
        stems = stems[:args.max_images]

    import extract_predobj_cropaug as pred
    canonical_names = list(pred.PART_CLASSES)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("=" * 78)
    print("TEXT vs GT VISUAL PART-STRUCTURE AUDIT")
    print("=" * 78)
    print("split/images       :", args.split, len(stems))
    print("split source       :", split_source)
    print("raw text bank      :", text_bank)
    print("initial projector  :", initial_weight)
    print("PartStruct proj.   :", partstruct_weight)
    print("GT visual cache    :", visual_cache)
    print("GT crop expand     :", args.crop_expand_ratio)
    print()

    if visual_cache.is_file() and not args.rebuild_visual_cache:
        print("[visual] loading existing cache")
        visual_payload = torch_load(visual_cache)
    else:
        if visual_cache.exists() and args.rebuild_visual_cache:
            visual_cache.unlink()
        print("[visual] extracting GT-DINO prototypes")
        visual_payload = build_visual_cache(
            args, repo, data_root, stems, split_source, visual_cache
        )
    validate_visual_cache(visual_payload, canonical_names, args)
    available = visual_payload["available"].bool().cpu()

    names, texts, text_prov = text_variants(
        repo, text_bank, model_config, initial_weight, partstruct_weight, device
    )
    if names != canonical_names:
        bad = [(i, names[i], canonical_names[i]) for i in range(116) if names[i] != canonical_names[i]]
        raise RuntimeError(f"text/GT taxonomy mismatch: {bad[:10]}")

    primary = compare(
        texts,
        visual_payload["crop_balanced_prototypes"].float(),
        available,
        names,
    )
    secondary = compare(
        texts,
        visual_payload["pixel_weighted_prototypes"].float(),
        available,
        names,
    )

    missing = torch.nonzero(~available, as_tuple=False).view(-1).tolist()
    report = {
        "protocol_version": PROTOCOL,
        "question": "Do text part relations resemble GT DINO visual-part relations?",
        "primary_visual_aggregation": "crop_balanced",
        "primary": primary,
        "secondary_visual_aggregation": "pixel_weighted",
        "secondary": secondary,
        "visual_cache": str(visual_cache),
        "visual_protocol": visual_payload["protocol"],
        "available_parts": int(available.sum()),
        "missing_part_ids": missing,
        "missing_part_names": [names[i] for i in missing],
        "crop_prototype_count": visual_payload["crop_prototype_count"].tolist(),
        "text_provenance": text_prov,
        "elapsed_seconds": time.time() - started,
    }

    if json_out.exists() and not args.overwrite_json:
        raise FileExistsError(f"{json_out} exists; add --overwrite_json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print()
    print("=" * 78)
    print("PRIMARY: crop-balanced GT DINO prototypes")
    print("=" * 78)
    print(f"{'Object':16s} {'Parts':>5s} {'RawCLIP':>12s} {'InitialProj':>12s} {'PartStruct':>12s}")
    for obj, row in primary["per_object"].items():
        if row["status"] != "OK":
            print(f"{obj:16s} {row['available_parts']:5d}  {row['status']}")
        else:
            print(
                f"{obj:16s} {row['available_parts']:5d} "
                f"{row['raw_clip']:12.8f} "
                f"{row['initial_projector']:12.8f} "
                f"{row['partstruct_projector']:12.8f}"
            )
    print()
    m = primary["object_macro_spearman"]
    print("Valid objects:", primary["valid_objects"])
    print(f"RAW CLIP      vs GT visual macro Spearman = {m['raw_clip']:.8f}")
    print(f"Initial proj. vs GT visual macro Spearman = {m['initial_projector']:.8f}")
    print(f"PartStruct    vs GT visual macro Spearman = {m['partstruct_projector']:.8f}")

    print()
    print("ROBUSTNESS: pixel-weighted GT visual prototypes")
    m2 = secondary["object_macro_spearman"]
    print(f"RAW CLIP      vs GT visual macro Spearman = {m2['raw_clip']:.8f}")
    print(f"Initial proj. vs GT visual macro Spearman = {m2['initial_projector']:.8f}")
    print(f"PartStruct    vs GT visual macro Spearman = {m2['partstruct_projector']:.8f}")
    print()
    print("JSON:", json_out)
    print("GT visual cache:", visual_cache)
    print("TEXT_VS_GT_VISUAL_STRUCTURE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
