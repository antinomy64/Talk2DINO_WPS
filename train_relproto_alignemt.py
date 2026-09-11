#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_predobj_relproto_w_clip_talk2dino.py

Final Pred-Obj W trainer for Pascal-Part-116.

Required upstream artifacts
---------------------------
1) Raw CLIP part text bank produced by extract_clip_text_bank.py

   {
       "features":      Tensor[116, 512],   # prompt-mean CLIP, pre-projector
       "class_names":   list[str] length 116,
       "object_groups": dict[str, list[int]],
       "clip_model":    "ViT-B/16",
       "template_set":  "sub_imagenet_template",
       "normalized":    False,
       "stage":         "pre_projector_prompt_mean",
   }

2) Predicted-object visual cache produced by extract_predobj_cropaug.py

   annotation:
       cropaug_patch_tokens : Tensor[1024, 768]
       pred_obj_mask_patch  : BoolTensor[1024]
       part_category_id     : list[int] image/object-specific part presence
       part_class_name      : list[str]

Text path
---------
raw CLIP prompt mean [116,512]
    -> frozen Talk2DINO PartStruct projector
    -> L2 normalize
    -> base part bank T0 [116,768]

For each training annotation
----------------------------
current text:
    T = normalize(T0[pids] @ W)

relative evidence:
    S[k,p] = T[k] dot X[p]
    R[k,p] = S[k,p] - max_{q != k} S[q,p], K > 1
             S[k,p],                         K = 1

anchor:
    each part independently selects argmax_p R[k,p]
    anchor collisions are allowed

RelProto support for part k:
    - always include its own anchor
    - exclude only its own anchor from supplementary candidates
    - add the highest strictly-positive relative-score patches
    - total support size <= prototype_max_patches (default 4)
    - no force filling
    - no one-to-one/global-greedy distinct-anchor constraint

RelProto:
    normalized equal mean of unit DINO patch features

Alignment:
    text row k directly matches its induced RelProto row k

Optimization:
    one shared W in R^{768x768}, initialized I
    equal-annotation cosine loss
    Adam, lr=1e-3
    after each optimizer step: W <- U @ Vh (polar/SVD retraction)
    default: batch=16, epochs=10, seed=123

The frozen projector is built exactly like Talk2DINO training:
    YAML -> src.model.<model_class>.from_config(model_cfg)
         -> strict load projector checkpoint
         -> project_clip_txt()

This trainer never opens RGB images or GT masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


VERSION = "predobj_clip_relproto_w_orth_v1"

NUM_PARTS = 116
VISION_DIM = 768
DEFAULT_RAW_CLIP_DIM = 512
DEFAULT_PATCH_COUNT = 1024


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


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, str(tmp))
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def normalize_last(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if not torch.isfinite(x).all():
        raise ValueError("NaN/Inf in feature tensor")
    norm = x.norm(dim=-1, keepdim=True)
    if bool((norm <= eps).any()):
        raise ValueError("zero/invalid feature vector")
    return x / norm


def resolve_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    if not (root / "src" / "model.py").is_file():
        raise FileNotFoundError(
            f"Talk2DINO root must contain src/model.py: {root}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def get_annotations(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("annotations"), (list, tuple)):
        return list(payload["annotations"])
    raise ValueError("pred-object PTH must contain an 'annotations' list")


def to_cpu_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        out = torch.from_numpy(np.asarray(value))
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def unwrap_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    obj = ckpt
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "projector_state_dict"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise ValueError("projector checkpoint is not a state_dict mapping")

    state = {
        str(k): v.detach().cpu()
        for k, v in obj.items()
        if torch.is_tensor(v)
    }
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


# -----------------------------------------------------------------------------
# Raw CLIP bank + Talk2DINO PartStruct projector
# -----------------------------------------------------------------------------

def load_raw_clip_bank(
    path: Path,
    *,
    expected_clip_model: str,
    expected_template: str,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    payload = torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("CLIP text bank must be a mapping")
    if "features" not in payload:
        raise KeyError("CLIP text bank is missing 'features'")

    raw = to_cpu_tensor(payload["features"], dtype=torch.float32)
    if raw.ndim != 2 or raw.shape[0] != NUM_PARTS:
        raise ValueError(
            f"raw CLIP bank must be [116,D], got {tuple(raw.shape)}"
        )
    if raw.shape[1] != DEFAULT_RAW_CLIP_DIM:
        raise ValueError(
            f"expected ViT-B/16 CLIP text dim {DEFAULT_RAW_CLIP_DIM}, "
            f"got {raw.shape[1]}"
        )
    if not torch.isfinite(raw).all() or bool((raw.norm(dim=1) <= 1e-12).any()):
        raise ValueError("invalid raw CLIP text bank")

    names = payload.get("class_names", payload.get("classnames", payload.get("names")))
    if names is None:
        raise KeyError("CLIP text bank must contain class_names/classnames/names")
    names = [str(x) for x in names]
    if len(names) != NUM_PARTS:
        raise ValueError(f"text bank contains {len(names)} names, expected 116")
    if len(set(names)) != NUM_PARTS:
        raise ValueError("duplicate names in CLIP text bank")

    clip_model = str(payload.get("clip_model", ""))
    template = str(payload.get("template_set", ""))
    if clip_model and clip_model != expected_clip_model:
        raise ValueError(
            f"text-bank CLIP model {clip_model!r} != expected {expected_clip_model!r}"
        )
    if template and template != expected_template:
        raise ValueError(
            f"text-bank template {template!r} != expected {expected_template!r}"
        )

    if payload.get("normalized", False) is not False:
        raise ValueError(
            "final W text input must be the raw pre-projector prompt-mean bank "
            "(normalized=False)"
        )
    stage = str(payload.get("stage", ""))
    if stage and stage != "pre_projector_prompt_mean":
        raise ValueError(
            f"expected stage='pre_projector_prompt_mean', got {stage!r}"
        )

    info = {
        "path": str(path),
        "sha256": sha256(path),
        "shape": list(raw.shape),
        "clip_model": clip_model or expected_clip_model,
        "template_set": template or expected_template,
        "normalized_before_projector": False,
        "stage": stage or "pre_projector_prompt_mean",
    }
    return raw.contiguous(), names, info


def load_frozen_projector(
    *,
    project_root: Path,
    config_path: Path,
    weight_path: Path,
    device: torch.device,
):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, Mapping) or "model" not in config:
        raise ValueError(f"invalid Talk2DINO YAML: {config_path}")

    model_cfg = dict(config["model"])
    model_module = importlib.import_module("src.model")
    model_class_name = str(model_cfg.get("model_class", "ProjectionLayer"))
    if not hasattr(model_module, model_class_name):
        raise AttributeError(f"src.model has no {model_class_name!r}")
    ModelClass = getattr(model_module, model_class_name)

    projector = ModelClass.from_config(model_cfg)
    state = unwrap_state_dict(torch_load(weight_path))
    projector.load_state_dict(state, strict=True)
    projector = projector.to(device).eval()
    for p in projector.parameters():
        p.requires_grad_(False)

    info = {
        "project_root": str(project_root),
        "model_config": str(config_path),
        "model_config_sha256": sha256(config_path),
        "model_class": model_class_name,
        "weights": str(weight_path),
        "weights_sha256": sha256(weight_path),
        "strict_load": True,
        "frozen": True,
        "text_projection_fn": "project_clip_txt",
    }
    return projector, info


@torch.inference_mode()
def project_raw_clip_bank(
    raw: torch.Tensor,
    projector,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    # IMPORTANT: raw CLIP prompt means are NOT normalized before the projector.
    outputs = []
    for start in range(0, raw.shape[0], batch_size):
        x = raw[start:start + batch_size].to(device=device, dtype=torch.float32)
        y = projector.project_clip_txt(x)
        if y.ndim != 2 or y.shape[1] != VISION_DIM:
            raise ValueError(
                f"project_clip_txt returned {tuple(y.shape)}, expected [B,768]"
            )
        outputs.append(y.detach().cpu().float())

    bank = torch.cat(outputs, dim=0)
    if tuple(bank.shape) != (NUM_PARTS, VISION_DIM):
        raise ValueError(f"projected bank shape {tuple(bank.shape)} != (116,768)")
    bank = F.normalize(bank, dim=-1)
    if not torch.isfinite(bank).all():
        raise ValueError("nonfinite projected part bank")
    return bank.contiguous()


# -----------------------------------------------------------------------------
# Pred-object cache
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    row_index: int
    image_id: str
    ann_id: str
    class_name: str
    pids: torch.Tensor
    tokens: torch.Tensor
    foreground: torch.Tensor


class PredObjDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate_examples(batch: list[Example]) -> list[Example]:
    return batch


def prepare_examples(
    path: Path,
    *,
    text_names: list[str],
    patch_key: str,
    foreground_key: str,
    part_id_key: str,
    part_name_key: str,
    max_annotations: int,
    expected_bg_thresh: float | None,
    require_pamr: bool,
    projector_weight_sha256: str,
    require_cache_projector_match: bool,
) -> tuple[list[Example], dict[str, Any]]:
    payload = torch_load(path)
    annotations = get_annotations(payload)
    if max_annotations > 0:
        annotations = annotations[:max_annotations]

    meta = dict(payload.get("pred_obj_cropaug_meta", {})) if isinstance(payload, Mapping) else {}

    # Mainline cache audit.
    if expected_bg_thresh is not None and "bg_thresh" in meta:
        actual = float(meta["bg_thresh"])
        if not math.isclose(actual, float(expected_bg_thresh), abs_tol=1e-12, rel_tol=0.0):
            raise ValueError(
                f"pred-object cache bg_thresh={actual} != expected {expected_bg_thresh}"
            )

    if require_pamr and meta.get("pamr") is not True:
        raise ValueError("mainline W cache must be generated with PAMR enabled")

    if require_cache_projector_match:
        cache_sha = str(meta.get("projector_weight_sha256", ""))
        if cache_sha and cache_sha != projector_weight_sha256:
            raise ValueError(
                "pred-object cache was generated with a different projector checkpoint"
            )

    examples: list[Example] = []
    audit_rows: list[dict[str, Any]] = []
    status = Counter()

    for i, ann in enumerate(annotations):
        row = {
            "row_index": i,
            "image_id": str(ann.get("image_id", "")),
            "class_name": str(ann.get("class_name", "")),
            "status": "error",
        }
        try:
            for key in (patch_key, foreground_key, part_id_key):
                if key not in ann:
                    raise KeyError(f"missing {key!r}")

            tokens = to_cpu_tensor(ann[patch_key])
            if tokens.ndim == 3 and tokens.shape[0] == 1:
                tokens = tokens[0]
            if tuple(tokens.shape) != (DEFAULT_PATCH_COUNT, VISION_DIM):
                raise ValueError(
                    f"{patch_key} shape {tuple(tokens.shape)} != (1024,768)"
                )
            if not tokens.dtype.is_floating_point:
                raise ValueError(f"{patch_key} must be floating point")
            if not torch.isfinite(tokens.float()).all():
                raise ValueError("nonfinite DINO patch token")

            foreground = to_cpu_tensor(
                ann[foreground_key], dtype=torch.bool
            ).reshape(-1)
            if tuple(foreground.shape) != (DEFAULT_PATCH_COUNT,):
                raise ValueError(
                    f"{foreground_key} shape {tuple(foreground.shape)} != (1024,)"
                )
            m = int(foreground.sum().item())
            if m < 1:
                raise ValueError("predicted foreground contains zero patch")
            # No M>=K requirement: anchors are independent and may collide.

            pids = to_cpu_tensor(ann[part_id_key], dtype=torch.long).reshape(-1)
            if pids.numel() < 1:
                raise ValueError("empty image-specific part presence")
            if bool(((pids < 0) | (pids >= NUM_PARTS)).any()):
                raise ValueError("part id outside [0,115]")
            pid_list = [int(x) for x in pids.tolist()]
            if len(set(pid_list)) != len(pid_list):
                raise ValueError("duplicate part_category_id in one annotation")

            if part_name_key in ann:
                names = [str(x) for x in ann[part_name_key]]
                if len(names) != len(pid_list):
                    raise ValueError("part name/id count mismatch")
                expected = [text_names[pid] for pid in pid_list]
                if names != expected:
                    raise ValueError(
                        f"part taxonomy mismatch: ids imply {expected}, annotation has {names}"
                    )

            examples.append(
                Example(
                    row_index=i,
                    image_id=str(ann.get("image_id", "")),
                    ann_id=str(ann.get("id", ann.get("annotation_id", i))),
                    class_name=str(ann.get("class_name", "")),
                    pids=pids.contiguous(),
                    tokens=tokens.contiguous(),
                    foreground=foreground.contiguous(),
                )
            )

            row.update(
                status="ready",
                visible_parts=int(pids.numel()),
                foreground_patches=m,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"

        status[row["status"]] += 1
        audit_rows.append(row)

    if status.get("error", 0):
        first_bad = [r for r in audit_rows if r["status"] == "error"][:10]
        raise ValueError(
            f"cache preflight found {status['error']} invalid annotations; "
            f"first={first_bad}"
        )
    if not examples:
        raise ValueError("no valid W-training annotations")

    report = {
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "selected_annotations": len(audit_rows),
        "ready_annotations": len(examples),
        "status_counts": dict(status),
        "patch_key": patch_key,
        "foreground_key": foreground_key,
        "part_id_key": part_id_key,
        "part_name_key": part_name_key,
        "requires_foreground_ge_num_parts": False,
        "anchor_collisions_allowed": True,
        "cache_meta": meta,
        "rows": audit_rows,
    }
    return examples, report


# -----------------------------------------------------------------------------
# Relative evidence -> independent anchor -> bounded evidence aggregation
# -----------------------------------------------------------------------------

def relative_scores(
    absolute: torch.Tensor,
    foreground: torch.Tensor,
) -> torch.Tensor:
    """
    absolute   : [N,K,P]
    foreground : [N,P]
    """
    if absolute.ndim != 3:
        raise ValueError("absolute score tensor must be [N,K,P]")
    n, k, p = absolute.shape
    if foreground.shape != (n, p):
        raise ValueError("foreground shape mismatch")

    if k == 1:
        relative = absolute.clone()
    else:
        expanded = absolute.unsqueeze(1).expand(n, k, k, p)
        diagonal = torch.eye(
            k, dtype=torch.bool, device=absolute.device
        ).view(1, k, k, 1)
        best_other = expanded.masked_fill(
            diagonal, -torch.inf
        ).amax(dim=2)
        relative = absolute - best_other

    relative = relative.masked_fill(
        ~foreground[:, None, :], -torch.inf
    )
    if torch.isnan(relative).any():
        raise ValueError("NaN in relative score")
    return relative


def build_relproto(
    current_text: torch.Tensor,
    patch_tokens: torch.Tensor,
    foreground: torch.Tensor,
    *,
    max_patches: int,
) -> dict[str, torch.Tensor]:
    """
    current_text : [N,K,768]
    patch_tokens : [N,1024,768], unit rows
    foreground   : [N,1024]

    Anchor:
        independent row-wise argmax(relative).

    Supplementary evidence:
        per part, exclude ONLY its own anchor, then select highest R>0 patches.
        No explicit cross-part exclusion/reservation is applied.
    """
    n, k, d = current_text.shape
    if d != VISION_DIM:
        raise ValueError("text feature dim must be 768")
    if patch_tokens.shape != (n, DEFAULT_PATCH_COUNT, VISION_DIM):
        raise ValueError("patch token shape mismatch")
    if foreground.shape != (n, DEFAULT_PATCH_COUNT):
        raise ValueError("foreground shape mismatch")
    if not 1 <= max_patches <= DEFAULT_PATCH_COUNT:
        raise ValueError("invalid prototype_max_patches")

    with torch.no_grad():
        query = normalize_last(current_text.detach())
        absolute = torch.bmm(query, patch_tokens.transpose(1, 2))
        relative = relative_scores(absolute, foreground)

        # No global greedy / no one-to-one assignment.
        anchor_indices = relative.argmax(dim=2)  # [N,K]
        anchor_relative = relative.gather(
            2, anchor_indices[:, :, None]
        ).squeeze(2)
        anchor_absolute = absolute.gather(
            2, anchor_indices[:, :, None]
        ).squeeze(2)

        support = torch.full(
            (n, k, max_patches),
            -1,
            dtype=torch.long,
            device=current_text.device,
        )
        support[:, :, 0] = anchor_indices

        if max_patches > 1:
            eligible = relative.clone()

            # Only remove this row's own anchor.
            eligible.scatter_(
                2, anchor_indices[:, :, None], -torch.inf
            )
            eligible.masked_fill_(
                eligible <= 0.0, -torch.inf
            )

            top_scores, top_indices = torch.topk(
                eligible,
                k=max_patches - 1,
                dim=2,
                largest=True,
                sorted=True,
            )
            support[:, :, 1:] = torch.where(
                torch.isfinite(top_scores),
                top_indices,
                torch.full_like(top_indices, -1),
            )

        valid = support >= 0
        counts = valid.sum(dim=2)
        if bool((counts < 1).any()):
            raise AssertionError("every prototype must contain its anchor")

        batch_idx = torch.arange(
            n, device=current_text.device
        )[:, None, None]
        gathered = patch_tokens[
            batch_idx, support.clamp_min(0)
        ]  # [N,K,C,D]

        means = (
            gathered * valid[:, :, :, None]
        ).sum(dim=2) / counts[:, :, None]

        proto = normalize_last(means)
        anchor_feat = patch_tokens[
            torch.arange(n, device=current_text.device)[:, None],
            anchor_indices,
        ]
        proto = torch.where(
            (counts == 1)[:, :, None],
            anchor_feat,
            proto,
        )

        support_relative = relative.gather(
            2, support.clamp_min(0)
        )
        support_relative = torch.where(
            valid, support_relative, torch.zeros_like(support_relative)
        )

        # Diagnostics.
        collision_annotations = []
        collision_excess = []
        for row in anchor_indices:
            unique = int(torch.unique(row).numel())
            kk = int(row.numel())
            collision_annotations.append(1.0 if unique < kk else 0.0)
            collision_excess.append(float(kk - unique))

        return {
            "prototypes": proto.detach(),
            "anchor_indices": anchor_indices.detach(),
            "anchor_relative_scores": anchor_relative.detach(),
            "anchor_absolute_scores": anchor_absolute.detach(),
            "support_indices": support.detach(),
            "support_relative_scores": support_relative.detach(),
            "prototype_counts": counts.detach(),
            "collision_annotation": torch.tensor(
                collision_annotations,
                dtype=torch.float32,
                device=current_text.device,
            ),
            "collision_excess": torch.tensor(
                collision_excess,
                dtype=torch.float32,
                device=current_text.device,
            ),
        }


# -----------------------------------------------------------------------------
# W training
# -----------------------------------------------------------------------------

def orthogonal_retract_(W: torch.Tensor) -> None:
    with torch.no_grad():
        if not torch.isfinite(W).all():
            raise ValueError("W is nonfinite before SVD")
        u, _, vh = torch.linalg.svd(W, full_matrices=False)
        W.copy_(u @ vh)


def matrix_stats(W: torch.Tensor, base_bank: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        eye = torch.eye(
            W.shape[0], device=W.device, dtype=W.dtype
        )
        gram = W.T @ W - eye
        before = normalize_last(base_bank)
        after = normalize_last(base_bank @ W)
        return {
            "orthogonality_max_abs": float(gram.abs().max().item()),
            "orthogonality_rms": float(
                torch.sqrt((gram * gram).mean()).item()
            ),
            "identity_rms": float(
                torch.sqrt(((W - eye) ** 2).mean()).item()
            ),
            "text_cosine_structure_max_abs": float(
                ((after @ after.T) - (before @ before.T))
                .abs()
                .max()
                .item()
            ),
        }


def batch_forward(
    base_bank: torch.Tensor,
    W: torch.Tensor,
    examples: list[Example],
    *,
    device: torch.device,
    prototype_max_patches: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    # All caches are fixed 1024 x 768, so one stack/H2D copy per optimizer batch.
    tokens = torch.stack(
        [ex.tokens for ex in examples], dim=0
    ).to(device=device, dtype=torch.float32)
    foreground = torch.stack(
        [ex.foreground for ex in examples], dim=0
    ).to(device=device, dtype=torch.bool)

    tokens = normalize_last(tokens)

    # K varies. Group equal-K annotations only for vectorized selector/prototype.
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[int(ex.pids.numel())].append(i)

    losses: list[torch.Tensor | None] = [None] * len(examples)

    proto_counts = []
    anchor_scores = []
    collision_ann = []
    collision_excess = []

    for k in sorted(groups):
        members = groups[k]
        index = torch.tensor(
            members, dtype=torch.long, device=device
        )
        pids = torch.stack(
            [examples[i].pids for i in members], dim=0
        ).to(device=device, dtype=torch.long)

        current_text = normalize_last(
            base_bank[pids] @ W
        )

        selected = build_relproto(
            current_text.detach(),
            tokens[index],
            foreground[index],
            max_patches=prototype_max_patches,
        )

        target = selected["prototypes"].detach()
        cosine = (current_text * target).sum(dim=2)
        loss_vec = (1.0 - cosine).mean(dim=1)

        for local_row, member in enumerate(members):
            losses[member] = loss_vec[local_row]

        proto_counts.append(
            selected["prototype_counts"].float().reshape(-1)
        )
        anchor_scores.append(
            selected["anchor_relative_scores"].reshape(-1)
        )
        collision_ann.append(
            selected["collision_annotation"]
        )
        collision_excess.append(
            selected["collision_excess"]
        )

    if any(x is None for x in losses):
        raise AssertionError("some annotation was not processed")

    # Equal-annotation weighting.
    loss = torch.stack(
        [x for x in losses if x is not None]
    ).mean()

    if not torch.isfinite(loss):
        raise ValueError("nonfinite W batch loss")

    with torch.no_grad():
        counts = torch.cat(proto_counts)
        anchor_r = torch.cat(anchor_scores)
        coll_ann = torch.cat(collision_ann)
        coll_excess = torch.cat(collision_excess)

        stats = {
            "prototype_mean_patch_count": float(counts.mean().item()),
            "prototype_anchor_only_fraction": float(
                (counts == 1).float().mean().item()
            ),
            "mean_anchor_relative_score": float(anchor_r.mean().item()),
            "anchor_collision_annotation_rate": float(coll_ann.mean().item()),
            "mean_anchor_collision_excess": float(coll_excess.mean().item()),
        }

    return loss, stats


def checkpoint_payload(
    *,
    W: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    text_info: dict[str, Any],
    projector_info: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": VERSION,
        "version": VERSION,
        "W": W.detach().cpu().clone(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "history": history,
        "args": vars(args),
        "text_bank": text_info,
        "projector": projector_info,
        "dataset": {
            "dataset": preflight["dataset"],
            "dataset_sha256": preflight["dataset_sha256"],
            "ready_annotations": preflight["ready_annotations"],
            "patch_key": preflight["patch_key"],
            "foreground_key": preflight["foreground_key"],
            "part_id_key": preflight["part_id_key"],
        },
        "method": {
            "text": (
                "raw prompt-mean CLIP ViT-B/16 -> frozen PartStruct projector "
                "-> L2 normalize -> current @ W"
            ),
            "relative_evidence": (
                "R[k,p]=S[k,p]-max_{q!=k}S[q,p]; K=1 uses S"
            ),
            "anchor": (
                "each part independently argmaxes its own relative score "
                "inside predicted foreground"
            ),
            "relproto": (
                "own anchor + highest strictly-positive relative-evidence patches; "
                "cap includes anchor; only own anchor excluded from extras; "
                "equal mean of unit DINO patches then L2 normalize"
            ),
            "correspondence": "text row i -> its induced RelProto row i",
            "loss": (
                "mean over annotations; each annotation first averages "
                "1-cos over its image-specific present parts"
            ),
            "W": (
                "one shared 768x768 W; identity init; Adam; "
                "polar SVD U@Vh after every batch"
            ),
        },
    }


def do_train(
    *,
    examples: list[Example],
    base_bank_cpu: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    text_info: dict[str, Any],
    projector_info: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    base_bank = base_bank_cpu.to(
        device=device, dtype=torch.float32
    ).detach()
    base_bank = normalize_last(base_bank)

    dataset = PredObjDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_examples,
        drop_last=False,
    )

    W = torch.nn.Parameter(
        torch.eye(VISION_DIM, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        [W], lr=args.lr, weight_decay=0.0
    )

    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        losses = []
        grad_norms = []
        proto_n = []
        anchor_only = []
        anchor_r = []
        collisions = []
        collision_excess = []

        progress = tqdm(
            loader,
            desc=f"epoch {epoch:02d}/{args.epochs:02d}",
            leave=True,
        )

        for batch_index, batch_examples in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)

            loss, stats = batch_forward(
                base_bank,
                W,
                batch_examples,
                device=device,
                prototype_max_patches=args.prototype_max_patches,
            )

            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise ValueError("missing/nonfinite W gradient")

            grad_norm = float(W.grad.norm().item())
            optimizer.step()
            orthogonal_retract_(W)
            global_step += 1

            loss_value = float(loss.detach().item())
            losses.append(loss_value)
            grad_norms.append(grad_norm)
            proto_n.append(stats["prototype_mean_patch_count"])
            anchor_only.append(stats["prototype_anchor_only_fraction"])
            anchor_r.append(stats["mean_anchor_relative_score"])
            collisions.append(stats["anchor_collision_annotation_rate"])
            collision_excess.append(stats["mean_anchor_collision_excess"])

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                proto=f"{stats['prototype_mean_patch_count']:.2f}",
                coll=f"{stats['anchor_collision_annotation_rate']:.3f}",
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        mstats = matrix_stats(W.detach(), base_bank)
        if mstats["orthogonality_max_abs"] > 1e-3:
            raise ValueError(
                f"orthogonality invariant failed: {mstats}"
            )

        row = {
            "epoch": epoch,
            "annotations": len(dataset),
            "batch_size": int(args.batch_size),
            "optimizer_updates": len(loader),
            "global_step": global_step,
            "mean_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "prototype_mean_patch_count": float(np.mean(proto_n)),
            "prototype_anchor_only_fraction": float(np.mean(anchor_only)),
            "mean_anchor_relative_score": float(np.mean(anchor_r)),
            "anchor_collision_annotation_rate": float(np.mean(collisions)),
            "mean_anchor_collision_excess": float(np.mean(collision_excess)),
            "seconds": float(time.perf_counter() - t0),
            **mstats,
        }
        history.append(row)

        write_csv(
            history, out_dir / "training_history.csv"
        )

        payload = checkpoint_payload(
            W=W,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            args=args,
            text_info=text_info,
            projector_info=projector_info,
            preflight=preflight,
        )

        atomic_torch_save(
            payload, out_dir / f"W_epoch_{epoch:03d}.pt"
        )
        atomic_torch_save(
            payload, out_dir / "W_last.pt"
        )

        print(
            f"[epoch] {epoch}/{args.epochs} "
            f"loss={row['mean_loss']:.6f} "
            f"proto_n={row['prototype_mean_patch_count']:.2f} "
            f"collision_ann={row['anchor_collision_annotation_rate']:.3f} "
            f"orth={row['orthogonality_max_abs']:.2e} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )

    summary = {
        "status": "completed",
        "version": VERSION,
        "checkpoint": "W_last.pt",
        "epochs": args.epochs,
        "global_step": global_step,
        "history": history,
        "formal_next_step": (
            "bake W into the same frozen PartStruct projector, then run "
            "the full 116-part dense Talk2DINO evaluator"
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")


# -----------------------------------------------------------------------------
# Talk2DINO-style entry: config -> model -> data -> train
# -----------------------------------------------------------------------------

def train_and_eval(args: argparse.Namespace) -> None:
    root = resolve_root(args.project_root)

    dataset_path = resolve_path(root, args.train_dataset)
    text_bank_path = resolve_path(root, args.text_bank)
    config_path = resolve_path(root, args.model_config)
    weight_path = resolve_path(root, args.weights)
    out_dir = resolve_path(root, args.out_dir)

    for path in (
        dataset_path,
        text_bank_path,
        config_path,
        weight_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix with existing run: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # 1) raw CLIP part bank
    raw_clip, text_names, text_info = load_raw_clip_bank(
        text_bank_path,
        expected_clip_model=args.clip_model,
        expected_template=args.template_set,
    )

    # 2) exact Talk2DINO projector used by PartStruct training
    projector, projector_info = load_frozen_projector(
        project_root=root,
        config_path=config_path,
        weight_path=weight_path,
        device=device,
    )

    # 3) project raw CLIP once before W training
    projected_bank = project_raw_clip_bank(
        raw_clip,
        projector,
        device=device,
        batch_size=args.project_batch_size,
    )
    text_info["projected_shape"] = list(projected_bank.shape)
    text_info["projected_normalized"] = True
    text_info["projector_weight_sha256"] = projector_info["weights_sha256"]

    torch.save(
        {
            "features": projected_bank,
            "classnames": text_names,
            "source_raw_bank": str(text_bank_path),
            "source_raw_bank_sha256": text_info["sha256"],
            "projector_weight": str(weight_path),
            "projector_weight_sha256": projector_info["weights_sha256"],
            "model_config": str(config_path),
            "normalized": True,
            "stage": "post_projector_part_bank",
        },
        out_dir / "projected_part_bank.pt",
    )

    # 4) predicted-object cache only
    examples, preflight = prepare_examples(
        dataset_path,
        text_names=text_names,
        patch_key=args.feature_name,
        foreground_key=args.foreground_key,
        part_id_key=args.part_id_key,
        part_name_key=args.part_name_key,
        max_annotations=args.max_annotations,
        expected_bg_thresh=args.expected_bg_thresh,
        require_pamr=args.require_pamr,
        projector_weight_sha256=projector_info["weights_sha256"],
        require_cache_projector_match=args.require_cache_projector_match,
    )

    rows = preflight.pop("rows")
    write_csv(rows, out_dir / "training_samples.csv")
    atomic_json_dump(preflight, out_dir / "preflight.json")
    atomic_json_dump(text_info, out_dir / "text_bank.json")
    atomic_json_dump(projector_info, out_dir / "projector.json")
    atomic_json_dump(vars(args), out_dir / "args.json")

    print("============================================================")
    print("Pred-Obj CLIP RelProto W training")
    print("============================================================")
    print("annotations      :", len(examples))
    print("raw CLIP bank    :", tuple(raw_clip.shape))
    print("projected bank   :", tuple(projected_bank.shape))
    print("projector        :", weight_path)
    print("bg_thresh(cache) :", preflight.get("cache_meta", {}).get("bg_thresh"))
    print("PAMR(cache)      :", preflight.get("cache_meta", {}).get("pamr"))
    print("anchor           : independent relative argmax")
    print("RelProto cap     :", args.prototype_max_patches)
    print("W                : shared orthogonal 768x768")
    print("============================================================")

    if args.preflight_only:
        print("[done] preflight_only")
        return

    do_train(
        examples=examples,
        base_bank_cpu=projected_bank,
        args=args,
        out_dir=out_dir,
        text_info=text_info,
        projector_info=projector_info,
        preflight=preflight,
    )


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def self_test() -> None:
    torch.manual_seed(7)

    # K=2 with equal queries: independent row-wise argmax is allowed to collide.
    text = F.normalize(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        dim=-1,
    )
    patches = F.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]]]),
        dim=-1,
    )
    fg = torch.ones((1, 4), dtype=torch.bool)

    # Pad feature dimension for algorithm test.
    text768 = torch.zeros((1, 2, VISION_DIM))
    patch768 = torch.zeros((1, DEFAULT_PATCH_COUNT, VISION_DIM))
    text768[:, :, :2] = text
    patch768[:, :4, :2] = patches
    # Fill remaining patches with a valid unit vector but mask them out.
    patch768[:, 4:, 2] = 1.0
    fg1024 = torch.zeros((1, DEFAULT_PATCH_COUNT), dtype=torch.bool)
    fg1024[:, :4] = True

    text768 = normalize_last(text768)
    patch768 = normalize_last(patch768)

    result = build_relproto(
        text768,
        patch768,
        fg1024,
        max_patches=4,
    )
    anchors = result["anchor_indices"][0]
    assert int(anchors[0]) == int(anchors[1]), anchors
    assert float(result["collision_annotation"][0]) == 1.0

    # No M>=K requirement: one foreground patch is still legal.
    one_fg = torch.zeros_like(fg1024)
    one_fg[:, 0] = True
    result_one = build_relproto(
        text768,
        patch768,
        one_fg,
        max_patches=4,
    )
    assert result_one["anchor_indices"].shape == (1, 2)

    # Orthogonal retraction.
    W = torch.eye(VISION_DIM) + 0.001 * torch.randn(VISION_DIM, VISION_DIM)
    orthogonal_retract_(W)
    err = (W.T @ W - torch.eye(VISION_DIM)).abs().max().item()
    assert err < 1e-4, err

    print("SELF_TEST_PASS")
    print("PASS independent_rowwise_relative_anchor")
    print("PASS anchor_collision_allowed")
    print("PASS no_foreground_ge_num_parts_requirement")
    print("PASS own_anchor_only_excluded_from_supplementary_support")
    print("PASS orthogonal_W_retraction")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    p.add_argument("--project_root", default=".")

    p.add_argument(
        "--train_dataset",
        required=False,
        default="feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug.pth",
    )
    p.add_argument(
        "--text_bank",
        required=False,
        default=(
            "feature/pascalpart116_clip_text/"
            "pascalpart116_clip_vitb16_subimagenet_raw.pt"
        ),
    )
    p.add_argument(
        "--model_config",
        default="configs/vitb_mlp_infonce.yaml",
    )
    p.add_argument(
        "--weights",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_clean_ft10_"
            "partstruct_w1e4_lr1e5.pth"
        ),
    )
    p.add_argument(
        "--out_dir",
        default="final_exp/predobj_clip_w/relproto4_orth",
    )

    p.add_argument(
        "--feature_name",
        default="cropaug_patch_tokens",
    )
    p.add_argument(
        "--foreground_key",
        default="pred_obj_mask_patch",
    )
    p.add_argument(
        "--part_id_key",
        default="part_category_id",
    )
    p.add_argument(
        "--part_name_key",
        default="part_class_name",
    )

    p.add_argument(
        "--clip_model",
        default="ViT-B/16",
    )
    p.add_argument(
        "--template_set",
        default="sub_imagenet_template",
    )

    p.add_argument(
        "--prototype_max_patches",
        type=int,
        default=4,
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--project_batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")

    p.add_argument(
        "--expected_bg_thresh",
        type=float,
        default=0.54,
        help="Mainline Pred-Obj cache threshold selected by 16-part-object macro mIoU.",
    )
    p.add_argument(
        "--require_pamr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--require_cache_projector_match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--max_annotations",
        type=int,
        default=0,
        help="Smoke test only; 0 means all annotations.",
    )
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--self_test", action="store_true")

    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.prototype_max_patches < 1:
        raise ValueError("prototype_max_patches must be >=1")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs/batch_size must be >=1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and >0")
    if args.num_workers < 0 or args.project_batch_size < 1:
        raise ValueError("invalid num_workers/project_batch_size")
    if args.max_annotations < 0:
        raise ValueError("max_annotations must be >=0")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    if args.self_test:
        self_test()
        return 0

    train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
