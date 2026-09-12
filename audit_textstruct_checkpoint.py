#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Spearman audit for a projector checkpoint.

For the SAME TRAIN-GT crop-balanced DINO part prototype bank, report:

  1) raw CLIP text      vs GT visual structure
  2) initial projector  vs GT visual structure
  3) current checkpoint vs GT visual structure

All are exact object-macro Spearman over object groups with >=3 available parts.

GT is audit-only here. This file never trains a model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr


def parse():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--repo_root", default=".")
    p.add_argument("--model_config", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--initial_weight", required=True)
    p.add_argument("--text_bank", required=True)
    p.add_argument("--gt_bank", required=True)
    p.add_argument("--gt_key", default="crop_balanced_prototypes")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve(root, x):
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def tload(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def unwrap_state(x):
    if isinstance(x, dict):
        for k in ("state_dict", "model_state_dict", "projector_state_dict"):
            if k in x and isinstance(x[k], dict):
                x = x[k]
                break
    if not isinstance(x, dict):
        raise ValueError("checkpoint is not a state_dict mapping")
    if x and all(str(k).startswith("module.") for k in x):
        x = {str(k)[7:]: v for k, v in x.items()}
    return x


def load_projector(cfg_path, weight_path, device):
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cls = getattr(
        importlib.import_module("src.model"),
        cfg["model"].get("model_class", "ProjectionLayer"),
    )
    model = cls.from_config(cfg["model"])
    model.load_state_dict(unwrap_state(tload(weight_path)), strict=True)
    return model.to(device).eval()


def pairvec(feat, ids):
    idx = torch.tensor(ids, dtype=torch.long, device=feat.device)
    x = F.normalize(feat.index_select(0, idx).float(), dim=-1, eps=1e-12)
    g = x @ x.T
    tri = torch.triu_indices(len(ids), len(ids), offset=1, device=x.device)
    return g[tri[0], tri[1]].detach().cpu().double().numpy()


def macro_spearman(a, b, groups, available):
    vals = {}
    macro = []
    for obj, ids0 in groups.items():
        ids = [int(i) for i in ids0 if bool(available[int(i)])]
        if len(ids) < 3:
            continue
        x, y = pairvec(a, ids), pairvec(b, ids)
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rho = float(spearmanr(x, y).correlation)
        if not np.isfinite(rho):
            continue
        vals[str(obj)] = rho
        macro.append(rho)
    if not macro:
        raise RuntimeError("no valid objects for Spearman")
    return float(np.mean(macro)), vals


def main():
    args = parse()
    root = Path(args.repo_root).resolve()
    os.chdir(root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = resolve(root, args.model_config)
    weight = resolve(root, args.weight)
    init_weight = resolve(root, args.initial_weight)
    text_bank = resolve(root, args.text_bank)
    gt_bank = resolve(root, args.gt_bank)
    output = resolve(root, args.output)

    for p in (cfg, weight, init_weight, text_bank, gt_bank):
        if not p.is_file():
            raise FileNotFoundError(p)

    tb = tload(text_bank)
    if tb.get("normalized", False) is not False:
        raise ValueError("expected raw CLIP bank normalized=False")
    raw = torch.as_tensor(
        tb.get("features", tb.get("raw_features")),
        dtype=torch.float32,
    )
    if tuple(raw.shape) != (116, 512):
        raise ValueError(f"raw text bank shape={tuple(raw.shape)}")

    names = tb.get("class_names", tb.get("classnames", tb.get("names")))
    if names is None or len(names) != 116:
        raise ValueError("text bank must contain 116 class names")
    names = [str(x) for x in names]

    groups = tb.get("object_groups")
    if groups is None:
        groups = defaultdict(list)
        for i, name in enumerate(names):
            if "'s " not in name:
                raise ValueError(f"cannot infer object from {name!r}")
            groups[name.split("'s ", 1)[0]].append(i)
        groups = dict(groups)
    else:
        groups = {str(k): [int(i) for i in v] for k, v in groups.items()}

    gt = tload(gt_bank)
    if [str(x) for x in gt.get("part_classes", [])] != names:
        raise ValueError("GT/text taxonomy mismatch")
    protocol = gt.get("protocol", {})
    if protocol.get("split") not in (None, "", "train"):
        raise ValueError("GT audit bank must be TRAIN split")
    if args.gt_key not in gt:
        raise KeyError(args.gt_key)

    visual = torch.as_tensor(gt[args.gt_key], dtype=torch.float32)
    if tuple(visual.shape) != (116, 768):
        raise ValueError(f"GT prototype shape={tuple(visual.shape)}")
    available = torch.as_tensor(
        gt.get("available", torch.ones(116, dtype=torch.bool)),
        dtype=torch.bool,
    ).reshape(-1)
    if tuple(available.shape) != (116,):
        raise ValueError("GT available must be [116]")

    raw = raw.to(device)
    visual = visual.to(device)
    available_cpu = available.cpu()

    initial = load_projector(cfg, init_weight, device)
    current = load_projector(cfg, weight, device)

    with torch.no_grad():
        raw_u = F.normalize(raw.float(), dim=-1, eps=1e-12)
        init_z = F.normalize(initial.project_clip_txt(raw).float(), dim=-1, eps=1e-12)
        curr_z = F.normalize(current.project_clip_txt(raw).float(), dim=-1, eps=1e-12)
        vis_u = F.normalize(visual.float(), dim=-1, eps=1e-12)

    raw_rho, raw_per = macro_spearman(raw_u, vis_u, groups, available_cpu)
    init_rho, init_per = macro_spearman(init_z, vis_u, groups, available_cpu)
    curr_rho, curr_per = macro_spearman(curr_z, vis_u, groups, available_cpu)

    report = {
        "gt_key": args.gt_key,
        "valid_objects": len(curr_per),
        "raw_clip_vs_gt_visual_macro_spearman": raw_rho,
        "initial_vs_gt_visual_macro_spearman": init_rho,
        "current_vs_gt_visual_macro_spearman": curr_rho,
        "raw_clip_vs_gt_visual_per_object": raw_per,
        "initial_vs_gt_visual_per_object": init_per,
        "current_vs_gt_visual_per_object": curr_per,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Valid objects: {report['valid_objects']}")
    print(
        "RAW CLIP     vs GT visual macro Spearman = "
        f"{raw_rho:.8f}"
    )
    print(
        "INITIAL      vs GT visual macro Spearman = "
        f"{init_rho:.8f}"
    )
    print(
        "CURRENT      vs GT visual macro Spearman = "
        f"{curr_rho:.8f}"
    )
    print("THREE_SPEARMAN_AUDIT_PASS")


if __name__ == "__main__":
    main()
