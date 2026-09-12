#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment-2 Spearman audit.

Reports the GT-visual structural correlation that is currently well-defined:

  1) raw CLIP part feature          vs GT part prototypes
  2) initial projected CLIP feature vs GT part prototypes
  3) raw LLaMA3 part feature        vs GT part prototypes

Optionally, if a compatible LLaMA projector config/checkpoint is supplied:
  4) projected LLaMA3 feature       vs GT part prototypes

All metrics are exact object-macro Spearman on the SAME GT-valid subset.
"""

from __future__ import annotations
import argparse, importlib, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr


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
        raise TypeError("checkpoint is not a state_dict mapping")
    if x and all(str(k).startswith("module.") for k in x):
        x = {str(k)[7:]: v for k, v in x.items()}
    return x


def load_projector(cfg_path, weight_path, device):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cls = getattr(
        importlib.import_module("src.model"),
        cfg["model"].get("model_class", "ProjectionLayer"),
    )
    model = cls.from_config(cfg["model"])
    model.load_state_dict(unwrap_state(tload(weight_path)), strict=True)
    return model.to(device).eval()


def relvec(feat, ids):
    idx = torch.as_tensor(ids, dtype=torch.long, device=feat.device)
    x = F.normalize(feat.index_select(0, idx).float(), dim=-1, eps=1e-12)
    sim = x @ x.T
    r, c = torch.triu_indices(len(ids), len(ids), offset=1, device=x.device)
    return sim[r, c].detach().cpu().double().numpy()


def macro(a, b, groups, available, min_parts):
    vals, per = [], {}
    for obj, ids0 in groups.items():
        ids = [int(i) for i in ids0 if bool(available[int(i)])]
        if len(ids) < min_parts:
            continue
        x, y = relvec(a, ids), relvec(b, ids)
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rho = float(spearmanr(x, y).correlation)
        if np.isfinite(rho):
            per[str(obj)] = rho
            vals.append(rho)
    if not vals:
        raise RuntimeError("No valid objects for Spearman.")
    return float(np.mean(vals)), per


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--clip_bank", required=True)
    p.add_argument("--llama_bank", required=True)
    p.add_argument("--gt_bank", required=True)
    p.add_argument("--clip_model_config", required=True)
    p.add_argument("--clip_weight", required=True)
    p.add_argument("--llama_model_config", default=None)
    p.add_argument("--llama_weight", default=None)
    p.add_argument("--gt_key", default="crop_balanced_prototypes")
    p.add_argument("--min_parts", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if (args.llama_model_config is None) != (args.llama_weight is None):
        raise ValueError(
            "--llama_model_config and --llama_weight must be supplied together."
        )

    device = torch.device(args.device)
    cb, lb, gt = map(tload, [args.clip_bank, args.llama_bank, args.gt_bank])

    clip_names = [str(x) for x in cb["class_names"]]
    llama_names = [str(x) for x in lb["class_names"]]
    gt_names = [str(x) for x in gt["part_classes"]]
    if not (clip_names == llama_names == gt_names):
        raise RuntimeError("CLIP/LLaMA/GT taxonomy or row order mismatch.")

    cg = {str(k): [int(i) for i in v] for k, v in cb["object_groups"].items()}
    lg = {str(k): [int(i) for i in v] for k, v in lb["object_groups"].items()}
    if cg != lg:
        raise RuntimeError("CLIP/LLaMA object_groups mismatch.")

    raw_clip = torch.as_tensor(cb["features"], dtype=torch.float32, device=device)
    raw_llama = torch.as_tensor(lb["features"], dtype=torch.float32, device=device)
    visual = torch.as_tensor(gt[args.gt_key], dtype=torch.float32, device=device)
    available = torch.as_tensor(
        gt.get("available", torch.ones(116, dtype=torch.bool)),
        dtype=torch.bool,
    ).reshape(-1)

    if tuple(raw_clip.shape) != (116, 512):
        raise RuntimeError(f"CLIP bank shape {tuple(raw_clip.shape)} != (116,512)")
    if tuple(raw_llama.shape) != (116, 4096):
        raise RuntimeError(f"LLaMA bank shape {tuple(raw_llama.shape)} != (116,4096)")
    if tuple(visual.shape) != (116, 768):
        raise RuntimeError(f"GT bank shape {tuple(visual.shape)} != (116,768)")

    clip_proj = load_projector(
        args.clip_model_config, args.clip_weight, device
    )
    with torch.no_grad():
        projected_clip = clip_proj.project_clip_txt(raw_clip).float()

    result = {"gt_key": args.gt_key, "metrics": {}, "per_object": {}}

    def add(name, a):
        rho, per = macro(a, visual, cg, available, args.min_parts)
        result["metrics"][name] = rho
        result["per_object"][name] = per
        return rho, len(per)

    raw_clip_rho, n = add("raw_clip_vs_gt_visual", raw_clip)
    projected_clip_rho, _ = add(
        "projected_clip_vs_gt_visual", projected_clip
    )
    raw_llama_rho, _ = add("raw_llama3_vs_gt_visual", raw_llama)

    print("=" * 72)
    print("GT PART-PROTOTYPE STRUCTURE SPEARMAN")
    print("=" * 72)
    print(f"Valid objects: {n}")
    print(f"RAW CLIP       vs GT visual = {raw_clip_rho:.8f}")
    print(f"PROJECTED CLIP vs GT visual = {projected_clip_rho:.8f}")
    print(f"RAW LLAMA3     vs GT visual = {raw_llama_rho:.8f}")

    if args.llama_weight is not None:
        llama_proj = load_projector(
            args.llama_model_config, args.llama_weight, device
        )
        expected = int(llama_proj.linear_layer.in_features)
        if expected != 4096:
            raise RuntimeError(
                f"LLaMA projector expects {expected} dims, not 4096."
            )
        with torch.no_grad():
            projected_llama = llama_proj.project_clip_txt(raw_llama).float()
        projected_llama_rho, _ = add(
            "projected_llama3_vs_gt_visual", projected_llama
        )
        print(
            f"PROJECTED LLAMA3 vs GT visual = {projected_llama_rho:.8f}"
        )
    else:
        print(
            "PROJECTED LLAMA3 vs GT visual = N/A "
            "(no compatible 4096->768 LLaMA projector supplied)"
        )

    print("=" * 72)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Saved:", out)


if __name__ == "__main__":
    main()
