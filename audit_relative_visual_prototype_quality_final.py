#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP-116 Relative Visual Prototype Quality audit.

Formal controlled comparison:
  Absolute + Orthogonal, L=8
  Relative + Orthogonal, L=8

The selector/prototype construction intentionally mirrors the nightly
train_relproto_variant.py implementation:

  X = L2-normalized DINO patch tokens inside predicted object foreground.
  T = L2Norm(T0[pids] @ W).

Absolute:
  S_jp = T_j dot X_p
  anchor = argmax S_jp
  support = anchor + top-(L-1) remaining finite absolute-score patches
          = top-L absolute-similarity foreground patches.

Relative:
  R_jp = S_jp - max_{k != j} S_kp, with R=S when K=1
  anchor = argmax R_jp
  support = anchor + top-(L-1) non-anchor patches with R_jp > 0
  no force filling.

For every epoch checkpoint and every image/object-part instance:
  1) Support Purity: fraction of selected support patches whose directly
     NN-resized GT patch-grid label equals the corresponding part ID.
  2) Proto-GT Cosine: cosine between the induced image-specific pseudo
     prototype and an image-specific oracle prototype formed by the mean of
     normalized DINO crop tokens whose NN GT patch-grid label equals that part.

Main reported metrics are class-macro:
  image-part instance metric -> mean within each part class -> mean over classes.
GT masks are used only for this post-hoc audit and never for support selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

NUM_PARTS = 116
DINO_DIM = 768
PATCH_COUNT = 1024
GT_CACHE_PROTOCOL = "pp116_crop_nn32_direct_patchlabel_v2"


def tload(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit PP-116 Absolute/Relative visual prototype quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project_root", default=".")
    p.add_argument(
        "--cache",
        default=(
            "feature/pascalpart116_predobj_clip_struct/"
            "train_predobj_cropaug_bg055_reproduce.pth"
        ),
    )
    p.add_argument("--data_root", default="data/PascalPart116")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--absolute_weight_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "runs/core/absolute_orthogonal_L8/train"
        ),
    )
    p.add_argument(
        "--relative_weight_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "runs/core/relative_orthogonal_L8/train"
        ),
    )
    p.add_argument(
        "--base_projector",
        default="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
    )
    p.add_argument(
        "--text_bank",
        default=(
            "feature/pascalpart116_clip_text/"
            "pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"
        ),
    )
    p.add_argument("--model_config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument(
        "--classes_source",
        default=(
            "src/open_vocabulary_segmentation/segmentation/datasets/"
            "pascalpart116_part.py:PART_CLASSES"
        ),
    )
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--end_epoch", type=int, default=30)
    p.add_argument("--expected_bg_thresh", type=float, default=0.55)
    p.add_argument("--no_require_pamr", dest="require_pamr", action="store_false")
    p.add_argument(
        "--no_require_cache_projector_match",
        dest="require_cache_projector_match",
        action="store_false",
    )
    p.set_defaults(require_pamr=True, require_cache_projector_match=True)
    p.add_argument("--gt_label_offset", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "analysis/relative_visual_prototype_quality_L8_controlled"
        ),
    )
    p.add_argument("--gt_audit_cache", default=None)
    p.add_argument("--rebuild_gt_audit_cache", action="store_true")
    return p.parse_args()


def resolve(root: Path, raw) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def recursive_pairs(obj: Any, frag: str, path: str = "root"):
    ans = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if frag.lower() in str(k).lower():
                ans.append((p, v))
            if isinstance(v, (dict, list, tuple)):
                ans.extend(recursive_pairs(v, frag, p))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:20]):
            if isinstance(v, (dict, list, tuple)):
                ans.extend(recursive_pairs(v, frag, f"{path}[{i}]"))
    return ans


# -----------------------------------------------------------------------------
# Exact nightly-trainer text/cache semantics
# -----------------------------------------------------------------------------

def get_text_features(bank):
    if torch.is_tensor(bank):
        return bank
    if isinstance(bank, dict):
        for k in ("features", "mean_feats", "text_features", "feats"):
            if k in bank and torch.is_tensor(bank[k]):
                return bank[k]
    raise RuntimeError("Could not find text features in text bank")


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "projector", "proj"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Projector checkpoint is not a state_dict-like object")
    if obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj


def parts_to_list(x):
    """Exact order-preserving de-duplication used by train_relproto_variant.py."""
    if torch.is_tensor(x):
        x = x.detach().cpu().reshape(-1).tolist()
    elif isinstance(x, np.ndarray):
        x = x.reshape(-1).tolist()
    elif isinstance(x, (list, tuple, set)):
        x = list(x)
    else:
        x = [x]
    out = []
    for z in x:
        if torch.is_tensor(z):
            z = z.item()
        try:
            z = int(z)
        except Exception:
            continue
        if 0 <= z < NUM_PARTS and z not in out:
            out.append(z)
    return out


def canonicalize_annotation(a):
    """Match the nightly trainer's accepted annotation subset."""
    if not isinstance(a, dict):
        return None
    required = ("cropaug_patch_tokens", "pred_obj_mask_patch", "part_category_id")
    if not all(k in a for k in required):
        return None
    x = torch.as_tensor(a["cropaug_patch_tokens"]).detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if tuple(x.shape) != (PATCH_COUNT, DINO_DIM):
        return None
    if not x.dtype.is_floating_point or not torch.isfinite(x.float()).all():
        return None
    m = torch.as_tensor(a["pred_obj_mask_patch"]).detach().cpu().reshape(-1) > 0
    if tuple(m.shape) != (PATCH_COUNT,) or not bool(m.any()):
        return None
    c = parts_to_list(a["part_category_id"])
    if not c:
        return None
    return x.contiguous(), m.contiguous(), c


def build_image_index(images):
    index = {}
    if isinstance(images, dict):
        iterable = []
        for k, v in images.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("id", k)
                iterable.append(vv)
    elif isinstance(images, (list, tuple)):
        iterable = [v for v in images if isinstance(v, dict)]
    else:
        iterable = []
    for im in iterable:
        for key in ("id", "image_id", "img_id"):
            if key in im:
                index[im[key]] = im
                index[str(im[key])] = im
    return index


def prepare_records(cache):
    if not isinstance(cache, dict) or "annotations" not in cache:
        raise RuntimeError("cache must contain top-level 'annotations'")
    annotations = cache["annotations"]
    if not isinstance(annotations, (list, tuple)):
        raise RuntimeError("cache['annotations'] must be a list/tuple")

    image_index = build_image_index(cache.get("images", []))
    records = []
    skipped = 0
    for ann_idx, ann in enumerate(annotations):
        z = canonicalize_annotation(ann)
        if z is None:
            skipped += 1
            continue
        x, m, pids = z
        record = dict(ann)
        record["_tokens"] = x
        record["_foreground"] = m
        record["_part_ids"] = pids
        record["_annotation_index"] = ann_idx

        image_id = record.get("image_id", record.get("img_id"))
        im = image_index.get(image_id, image_index.get(str(image_id)))
        if isinstance(im, dict):
            for key in (
                "file_name", "filename", "image_name", "img_name",
                "image_path", "img_path", "height", "width"
            ):
                if key in im:
                    record.setdefault(key, im[key])
        records.append(record)

    if not records:
        raise RuntimeError("no valid cache annotations")
    print(f"[INFO] cache annotations={len(annotations)} valid={len(records)} skipped={skipped}")
    return records


def validate_cache_meta(meta, args, projector_path: Path):
    bg = []
    for p, v in recursive_pairs(meta, "bg_thresh"):
        if torch.is_tensor(v) and v.numel() == 1:
            v = v.item()
        if isinstance(v, (int, float)):
            bg.append((p, float(v)))
    if bg:
        if not any(abs(v - args.expected_bg_thresh) < 1e-8 for _, v in bg):
            raise RuntimeError(
                f"cache bg threshold mismatch: {bg} vs expected {args.expected_bg_thresh}"
            )
        print(f"[INFO] cache bg_thresh matches {args.expected_bg_thresh}: {bg[:4]}")
    else:
        print("[WARN] no bg_thresh found in pred_obj_cropaug_meta")

    if args.require_pamr:
        ph = []
        for p, v in recursive_pairs(meta, "pamr"):
            if torch.is_tensor(v) and v.numel() == 1:
                v = v.item()
            if isinstance(v, bool):
                ph.append((p, v))
        if ph and not any(v for _, v in ph):
            raise RuntimeError(f"cache metadata says PAMR disabled: {ph}")
        if ph:
            print(f"[INFO] cache PAMR check: {ph[:4]}")
        else:
            print("[WARN] PAMR flag not found in pred_obj_cropaug_meta")

    if args.require_cache_projector_match:
        wanted_sha = sha256(projector_path)
        sha_hits = []
        for p, v in recursive_pairs(meta, "sha256"):
            if isinstance(v, str) and "projector" in p.lower():
                sha_hits.append((p, v))
        if sha_hits:
            if not any(v == wanted_sha for _, v in sha_hits):
                raise RuntimeError(
                    "cache projector SHA mismatch: "
                    f"wanted={wanted_sha}, metadata={sha_hits[:8]}"
                )
            print("[INFO] cache projector SHA matches current projector")
        else:
            path_hits = []
            for p, v in recursive_pairs(meta, "projector"):
                if isinstance(v, (str, Path)):
                    path_hits.append((p, str(v)))
            pth_hits = [(p, v) for p, v in path_hits if v.endswith(".pth")]
            if pth_hits:
                want = projector_path.name
                if not any(Path(v).name == want for _, v in pth_hits):
                    print(
                        f"[WARN] projector metadata basename does not match {want}: "
                        f"{pth_hits[:8]}"
                    )
                else:
                    print("[INFO] cache projector basename matches current projector")
            else:
                print("[WARN] no cache projector SHA/path metadata found")


def load_base_text(args, root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.model import ProjectionLayer

    device = torch.device(args.device)
    model_config = resolve(root, args.model_config)
    projector_path = resolve(root, args.base_projector)
    text_bank_path = resolve(root, args.text_bank)

    projector = ProjectionLayer.from_config(str(model_config))
    projector.load_state_dict(unwrap_state_dict(tload(projector_path)), strict=True)
    projector = projector.to(device).eval().requires_grad_(False)

    raw = get_text_features(tload(text_bank_path)).float()
    if raw.ndim != 2 or tuple(raw.shape) != (NUM_PARTS, 512):
        raise RuntimeError(f"text bank shape {tuple(raw.shape)} != (116,512)")
    with torch.no_grad():
        text = projector.project_clip_txt(raw.to(device=device, dtype=torch.float32))
        text = F.normalize(text.float(), dim=-1)
    if tuple(text.shape) != (NUM_PARTS, DINO_DIM):
        raise RuntimeError(f"projected text shape {tuple(text.shape)} != (116,768)")
    print(f"[INFO] base text={tuple(text.shape)} strict projector load=True")
    return text.cpu(), projector_path


# -----------------------------------------------------------------------------
# GT crop -> direct NN patch-grid labels
# -----------------------------------------------------------------------------

def build_mask_index(mask_root: Path):
    paths = list(mask_root.rglob("*.png"))
    if not paths:
        raise RuntimeError(f"No GT masks under {mask_root}")
    index = defaultdict(list)
    for p in paths:
        index[p.stem].append(p)
    return index


def get_image_ref(record):
    for key in (
        "image_name", "img_name", "file_name", "filename",
        "image_path", "img_path", "image_id", "img_id"
    ):
        if key in record:
            x = record[key]
            if torch.is_tensor(x) and x.numel() == 1:
                x = x.item()
            return str(x)
    return str(record.get("_annotation_index"))


def resolve_mask(image_ref: str, mask_root: Path, index):
    stem = Path(str(image_ref)).stem
    direct = mask_root / f"{stem}.png"
    if direct.exists():
        return direct
    hits = index.get(stem, [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous GT mask for {image_ref}: {hits}")
    raise FileNotFoundError(f"GT mask not found for {image_ref}")


def get_crop_box(record):
    for key in ("cropaug_box_xyxy", "crop_box_xyxy", "box_xyxy", "crop_box"):
        if key in record:
            x = torch.as_tensor(record[key]).detach().cpu().reshape(-1)
            if x.numel() != 4:
                raise RuntimeError(f"Invalid {key}: {x}")
            return tuple(float(v) for v in x.tolist())
    raise RuntimeError(
        f"annotation {record.get('_annotation_index')} is missing cropaug_box_xyxy"
    )


def crop_mask(mask: np.ndarray, box, fill=255):
    vals = [float(v) for v in box]
    rounded = [int(round(v)) for v in vals]
    if any(abs(v - r) > 1e-4 for v, r in zip(vals, rounded)):
        print(f"[WARN] non-integer crop box rounded for GT audit: {vals} -> {rounded}")
    x1, y1, x2, y2 = rounded
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Bad crop box {box}")

    out = np.full((y2 - y1, x2 - x1), fill, dtype=mask.dtype)
    H, W = mask.shape
    sx1, sy1 = max(x1, 0), max(y1, 0)
    sx2, sy2 = min(x2, W), min(y2, H)
    if sx2 <= sx1 or sy2 <= sy1:
        return out
    dx1, dy1 = sx1 - x1, sy1 - y1
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
    out[dy1:dy2, dx1:dx2] = mask[sy1:sy2, sx1:sx2]
    return out


def gt_crop_to_patch_labels(gt_crop: np.ndarray, grid: int):
    """Direct NN mapping of the GT crop to the DINO patch grid."""
    patch_labels = np.array(
        Image.fromarray(gt_crop).resize(
            (grid, grid), resample=Image.Resampling.NEAREST
        )
    )
    return torch.from_numpy(patch_labels.reshape(-1).astype(np.int64))


def build_gt_audit_cache(records, args, root: Path, output: Path):
    mask_root = resolve(root, args.data_root) / "annotations_detectron2_part" / args.split
    index = build_mask_index(mask_root)
    audit = []
    print(f"[INFO] building GT patch-label cache from {mask_root}")

    for i, record in enumerate(records):
        n = record["_tokens"].shape[0]
        grid = int(round(math.sqrt(n)))
        if grid * grid != n:
            raise RuntimeError(f"Patch count {n} is not square")

        image_ref = get_image_ref(record)
        mask_path = resolve_mask(image_ref, mask_root, index)
        gt = np.array(Image.open(mask_path))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_crop = crop_mask(gt, get_crop_box(record), fill=255)
        patch_labels = gt_crop_to_patch_labels(gt_crop, grid)
        if patch_labels.numel() != n:
            raise AssertionError("GT patch-grid size mismatch")

        audit.append(
            {
                "part_ids": torch.tensor(record["_part_ids"], dtype=torch.long),
                "patch_labels": patch_labels,
                "image_ref": image_ref,
                "mask_path": str(mask_path),
            }
        )
        if (i + 1) % 250 == 0 or i + 1 == len(records):
            print(f"[GT] {i + 1}/{len(records)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": GT_CACHE_PROTOCOL,
            "source_cache": str(resolve(root, args.cache)),
            "gt_label_offset": args.gt_label_offset,
            "records": audit,
        },
        output,
    )
    print(f"[OK] saved GT audit cache: {output}")
    return audit


def load_gt_audit(records, args, root: Path):
    if args.gt_audit_cache:
        path = resolve(root, args.gt_audit_cache)
    else:
        path = resolve(root, args.out_dir) / "gt_crop_patchlabel_cache_v2.pt"

    if path.exists() and not args.rebuild_gt_audit_cache:
        obj = tload(path)
        if obj.get("protocol") != GT_CACHE_PROTOCOL:
            raise RuntimeError(
                f"GT audit cache protocol={obj.get('protocol')!r}, "
                f"expected={GT_CACHE_PROTOCOL!r}; use --rebuild_gt_audit_cache"
            )
        audit = obj["records"]
        if len(audit) != len(records):
            raise RuntimeError(
                "GT audit cache length mismatch; use --rebuild_gt_audit_cache"
            )
        print(f"[INFO] GT audit cache: {path}")
        return audit
    return build_gt_audit_cache(records, args, root, path)


# -----------------------------------------------------------------------------
# Exact selector/prototype semantics from nightly train_relproto_variant.py
# -----------------------------------------------------------------------------

def relative_scores(S):
    # S: [C,P] where P already contains foreground patches only.
    C, _ = S.shape
    if C == 1:
        return S
    out = []
    for j in range(C):
        other = torch.cat([S[:j], S[j + 1:]], dim=0)
        out.append(S[j] - other.max(dim=0).values)
    return torch.stack(out, dim=0)


def build_proto(X, score_row, L, selector):
    """Copied semantically from the nightly variant trainer.

    X: [P_fg,D], normalized foreground patch tokens.
    Returns normalized pseudo prototype and LOCAL foreground support indices.
    """
    vals = score_row.detach()
    anchor = int(vals.argmax().item())
    idx = [anchor]
    if L > 1:
        keep = torch.ones_like(vals, dtype=torch.bool)
        keep[anchor] = False
        if selector == "relative":
            keep &= vals > 0
        elif selector == "absolute":
            # Exact nightly baseline behavior for explicitly requested L>1:
            # every finite non-anchor absolute score is eligible.
            keep &= torch.isfinite(vals)
        else:
            raise ValueError(selector)
        cand = torch.where(keep)[0]
        if cand.numel() > 0:
            k = min(L - 1, int(cand.numel()))
            top_local = torch.topk(vals[cand], k=k, largest=True).indices
            idx.extend(cand[top_local].tolist())
    proto = F.normalize(X[idx].mean(dim=0, keepdim=True), dim=-1)[0]
    return proto, torch.tensor(idx, dtype=torch.long, device=X.device)


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------

def ckpt_meta_value(obj, key):
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    if isinstance(obj, dict) and isinstance(obj.get("args"), dict) and key in obj["args"]:
        return obj["args"][key]
    return None


def load_W(weight_dir: Path, epoch: int, expected_selector: str, L: int):
    if epoch == 0:
        return torch.eye(DINO_DIM)
    path = weight_dir / f"W_epoch_{epoch:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = tload(path)
    if not isinstance(obj, dict) or "W" not in obj:
        raise RuntimeError(f"{path} has no W")

    checks = {
        "selector": expected_selector,
        "mapping": "orthogonal",
        "prototype_max_patches": L,
    }
    for key, expected in checks.items():
        got = ckpt_meta_value(obj, key)
        if got is None:
            raise RuntimeError(f"{path}: cannot verify checkpoint metadata field {key!r}")
        if key == "prototype_max_patches":
            got = int(got)
        else:
            got = str(got)
        if got != expected:
            raise RuntimeError(f"{path}: {key}={got!r}, expected={expected!r}")

    W = obj["W"].float()
    if tuple(W.shape) != (DINO_DIM, DINO_DIM):
        raise RuntimeError(f"{path}: bad W shape {tuple(W.shape)}")
    return W


def validate_weight_dir(weight_dir: Path, expected_selector: str, L: int, start: int, end: int):
    probe = next((ep for ep in range(max(1, start), end + 1) if (weight_dir / f"W_epoch_{ep:03d}.pt").exists()), None)
    if probe is None:
        raise RuntimeError(f"No W_epoch_*.pt found in {weight_dir} for requested range")
    _ = load_W(weight_dir, probe, expected_selector, L)
    print(
        f"[INFO] checkpoint dir verified: {weight_dir} "
        f"selector={expected_selector} mapping=orthogonal L={L}"
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def mean_dict_values(d):
    return {int(k): float(np.mean(v)) for k, v in d.items() if v}


@torch.no_grad()
def evaluate(records, gt_audit, base_text, W, selector, args, class_names):
    device = torch.device(args.device)
    base_text = base_text.to(device=device, dtype=torch.float32)
    W = W.to(device=device, dtype=torch.float32)

    class_purity = defaultdict(list)
    class_proto = defaultdict(list)
    purity_instances = []
    proto_instances = []
    support_sizes = []
    total_support = 0
    total_patch_hits = 0
    valid_instances = 0
    skipped_no_gt_patch = 0

    for record, gt_info in zip(records, gt_audit):
        tokens_full = F.normalize(
            record["_tokens"].to(device=device, dtype=torch.float32), dim=-1
        )
        foreground = record["_foreground"].to(device=device, dtype=torch.bool)
        part_ids = record["_part_ids"]
        gt_ids = torch.as_tensor(gt_info["part_ids"]).long().tolist()
        if part_ids != gt_ids:
            raise RuntimeError("part ID mismatch between cache record and GT audit cache")

        # IMPORTANT: exactly like nightly trainer: compact to foreground first.
        fg_full_idx = torch.where(foreground)[0]
        X = tokens_full[foreground]
        ids = torch.tensor(part_ids, device=device, dtype=torch.long)
        T = F.normalize(base_text.index_select(0, ids) @ W, dim=-1)
        S = T @ X.T
        score = S if selector == "absolute" else relative_scores(S)

        patch_labels = torch.as_tensor(gt_info["patch_labels"]).long().to(device)
        if patch_labels.numel() != tokens_full.shape[0]:
            raise RuntimeError("GT patch-label length mismatch")

        for j, pid in enumerate(part_ids):
            induced, local_support = build_proto(X, score[j], args.L, selector)
            full_support = fg_full_idx.index_select(0, local_support)

            label = int(pid) + int(args.gt_label_offset)
            gt_part_mask = patch_labels == label
            if not bool(gt_part_mask.any()):
                skipped_no_gt_patch += 1
                continue

            purity = float((patch_labels[full_support] == label).float().mean().item())
            oracle = F.normalize(
                tokens_full[gt_part_mask].mean(dim=0, keepdim=True), dim=-1
            )[0]
            cosine = float(torch.dot(induced, oracle).item())

            n_support = int(full_support.numel())
            hits = int((patch_labels[full_support] == label).sum().item())
            total_support += n_support
            total_patch_hits += hits
            support_sizes.append(n_support)
            purity_instances.append(purity)
            proto_instances.append(cosine)
            class_purity[pid].append(purity)
            class_proto[pid].append(cosine)
            valid_instances += 1

    if valid_instances == 0:
        raise RuntimeError("No valid image-part instances evaluated")

    purity_by_class = mean_dict_values(class_purity)
    proto_by_class = mean_dict_values(class_proto)
    valid_class_ids = sorted(set(purity_by_class) & set(proto_by_class))
    if not valid_class_ids:
        raise RuntimeError("No valid classes for class-macro metrics")

    # Main metrics: instance -> class mean -> class macro.
    support_purity_class_macro = float(np.mean([purity_by_class[i] for i in valid_class_ids]))
    proto_gt_class_macro = float(np.mean([proto_by_class[i] for i in valid_class_ids]))

    per_class_rows = []
    for pid in valid_class_ids:
        per_class_rows.append(
            {
                "class_id": pid,
                "class_name": class_names[pid] if pid < len(class_names) else str(pid),
                "instances": len(class_purity[pid]),
                "support_purity": purity_by_class[pid],
                "proto_gt_cosine": proto_by_class[pid],
            }
        )

    return {
        # Formal main metrics.
        "support_purity": support_purity_class_macro,
        "proto_gt_cosine": proto_gt_class_macro,
        # Diagnostics.
        "support_purity_instance_macro": float(np.mean(purity_instances)),
        "support_purity_patch_micro": total_patch_hits / total_support,
        "proto_gt_cosine_instance_macro": float(np.mean(proto_instances)),
        "proto_gt_cosine_median": float(np.median(proto_instances)),
        "mean_support_size": float(np.mean(support_sizes)),
        "selected_support_patches": int(total_support),
        "valid_part_instances": int(valid_instances),
        "valid_classes": len(valid_class_ids),
        "skipped_no_gt_patch": int(skipped_no_gt_patch),
        "per_class": per_class_rows,
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, out_dir: Path):
    epochs = [r["epoch"] for r in rows]
    abs_p = [r["absolute_support_purity"] for r in rows]
    rel_p = [r["relative_support_purity"] for r in rows]
    abs_c = [r["absolute_proto_gt_cosine"] for r in rows]
    rel_c = [r["relative_proto_gt_cosine"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    ax = axes[0]
    ax.plot(epochs, abs_p, linewidth=2.4, marker="o", markersize=4, markevery=2, label="Absolute + Orthogonal")
    ax.plot(epochs, rel_p, linewidth=2.4, marker="s", markersize=4, markevery=2, label="Relative + Orthogonal")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Support Purity", fontsize=13)
    ax.set_title("(a) Support Purity", fontsize=13)
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_xticks(range(0, 31, 5))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, abs_c, linewidth=2.4, marker="o", markersize=4, markevery=2, label="Absolute + Orthogonal")
    ax.plot(epochs, rel_c, linewidth=2.4, marker="s", markersize=4, markevery=2, label="Relative + Orthogonal")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Proto-GT Cosine", fontsize=13)
    ax.set_title("(b) Proto-GT Cosine", fontsize=13)
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_xticks(range(0, 31, 5))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "figure4_relative_visual_prototype_quality.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure4_relative_visual_prototype_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_class_names(root: Path, spec: str):
    try:
        raw_path, attr = spec.rsplit(":", 1)
        path = resolve(root, raw_path)
        module_name = "_pp116_classes_for_audit"
        mspec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(mspec)
        assert mspec.loader is not None
        mspec.loader.exec_module(module)
        names = list(getattr(module, attr))
        if len(names) != NUM_PARTS:
            raise RuntimeError(f"expected {NUM_PARTS} class names, got {len(names)}")
        return [str(x) for x in names]
    except Exception as exc:
        print(f"[WARN] could not load class names ({exc}); using numeric IDs")
        return [str(i) for i in range(NUM_PARTS)]


def main():
    args = parse_args()
    if args.L < 1:
        raise SystemExit("--L must be >= 1")
    if not (0 <= args.start_epoch <= args.end_epoch):
        raise SystemExit("require 0 <= start_epoch <= end_epoch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path(args.project_root).expanduser().resolve()
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = resolve(root, args.cache)
    abs_dir = resolve(root, args.absolute_weight_dir)
    rel_dir = resolve(root, args.relative_weight_dir)
    projector_path = resolve(root, args.base_projector)

    print(f"[INFO] loading cache: {cache_path}")
    cache = tload(cache_path)
    if not isinstance(cache, dict):
        raise RuntimeError("pred-object cache must be a dict")
    meta = cache.get("pred_obj_cropaug_meta", {})
    records = prepare_records(cache)

    # Base text is generated exactly like the nightly variant trainer.
    base_text, loaded_projector_path = load_base_text(args, root)
    if loaded_projector_path != projector_path:
        raise AssertionError("internal projector path mismatch")
    validate_cache_meta(meta, args, projector_path)

    # Controlled comparison must be L8-vs-L8 and both orthogonal.
    validate_weight_dir(abs_dir, "absolute", args.L, args.start_epoch, args.end_epoch)
    validate_weight_dir(rel_dir, "relative", args.L, args.start_epoch, args.end_epoch)

    gt_audit = load_gt_audit(records, args, root)
    class_names = load_class_names(root, args.classes_source)

    rows = []
    per_class_rows = []
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch:03d}")
        print("=" * 80)

        W_abs = load_W(abs_dir, epoch, "absolute", args.L)
        W_rel = load_W(rel_dir, epoch, "relative", args.L)

        abs_metrics = evaluate(records, gt_audit, base_text, W_abs, "absolute", args, class_names)
        rel_metrics = evaluate(records, gt_audit, base_text, W_rel, "relative", args, class_names)

        row = {
            "epoch": epoch,
            # Formal class-macro metrics.
            "absolute_support_purity": abs_metrics["support_purity"],
            "relative_support_purity": rel_metrics["support_purity"],
            "absolute_proto_gt_cosine": abs_metrics["proto_gt_cosine"],
            "relative_proto_gt_cosine": rel_metrics["proto_gt_cosine"],
            # Diagnostics.
            "absolute_support_purity_instance_macro": abs_metrics["support_purity_instance_macro"],
            "relative_support_purity_instance_macro": rel_metrics["support_purity_instance_macro"],
            "absolute_support_purity_patch_micro": abs_metrics["support_purity_patch_micro"],
            "relative_support_purity_patch_micro": rel_metrics["support_purity_patch_micro"],
            "absolute_proto_gt_cosine_instance_macro": abs_metrics["proto_gt_cosine_instance_macro"],
            "relative_proto_gt_cosine_instance_macro": rel_metrics["proto_gt_cosine_instance_macro"],
            "absolute_mean_support_size": abs_metrics["mean_support_size"],
            "relative_mean_support_size": rel_metrics["mean_support_size"],
            "absolute_valid_classes": abs_metrics["valid_classes"],
            "relative_valid_classes": rel_metrics["valid_classes"],
            "absolute_valid_part_instances": abs_metrics["valid_part_instances"],
            "relative_valid_part_instances": rel_metrics["valid_part_instances"],
            "absolute_skipped_no_gt_patch": abs_metrics["skipped_no_gt_patch"],
            "relative_skipped_no_gt_patch": rel_metrics["skipped_no_gt_patch"],
        }
        rows.append(row)

        for method, metrics in (("absolute", abs_metrics), ("relative", rel_metrics)):
            for r in metrics["per_class"]:
                per_class_rows.append({"epoch": epoch, "method": method, **r})

        print(
            f"[ABS] class-macro Purity={abs_metrics['support_purity']:.6f}  "
            f"Proto-GT={abs_metrics['proto_gt_cosine']:.6f}  "
            f"mean|P|={abs_metrics['mean_support_size']:.3f}  "
            f"classes={abs_metrics['valid_classes']}"
        )
        print(
            f"[REL] class-macro Purity={rel_metrics['support_purity']:.6f}  "
            f"Proto-GT={rel_metrics['proto_gt_cosine']:.6f}  "
            f"mean|P|={rel_metrics['mean_support_size']:.3f}  "
            f"classes={rel_metrics['valid_classes']}"
        )

        # Incremental saves protect a long 31-checkpoint audit.
        save_csv(rows, out_dir / "relative_visual_prototype_quality.csv")
        save_csv(per_class_rows, out_dir / "relative_visual_prototype_quality_per_class.csv")

    plot_results(rows, out_dir)
    print("\nDONE")
    print("CSV:", out_dir / "relative_visual_prototype_quality.csv")
    print("Per-class CSV:", out_dir / "relative_visual_prototype_quality_per_class.csv")
    print("PDF:", out_dir / "figure4_relative_visual_prototype_quality.pdf")


if __name__ == "__main__":
    main()
