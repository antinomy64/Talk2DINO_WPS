#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-part-prototype oracle: direct cosine supervision, no COCO/PartStruct/W."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def parse():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--repo_root", default=".")
    p.add_argument("--model_config", required=True)
    p.add_argument("--init_weight", required=True)
    p.add_argument("--text_bank", required=True)
    p.add_argument("--gt_bank", required=True)
    p.add_argument("--gt_key", default="crop_balanced_prototypes")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics_json", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve(root, x):
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    os.chdir(root)
    device = torch.device(args.device)

    with open(resolve(root, args.model_config)) as f:
        cfg = yaml.safe_load(f)
    cls = getattr(
        importlib.import_module("src.model"),
        cfg["model"].get("model_class", "ProjectionLayer"),
    )
    model = cls.from_config(cfg["model"])
    init = load(resolve(root, args.init_weight))
    if isinstance(init, dict) and "state_dict" in init:
        init = init["state_dict"]
    model.load_state_dict(init, strict=True)
    model.to(device)

    tb = load(resolve(root, args.text_bank))
    if tb.get("normalized", False) is not False:
        raise ValueError("oracle requires raw CLIP bank normalized=False")
    raw = torch.as_tensor(
        tb.get("features", tb.get("raw_features")), dtype=torch.float32
    ).to(device)
    if tuple(raw.shape) != (116, 512):
        raise ValueError(f"bad raw text bank shape: {tuple(raw.shape)}")
    text_names = tb.get("class_names", tb.get("classnames", tb.get("names")))
    if text_names is None or len(text_names) != 116:
        raise ValueError("text bank must contain 116 class names")
    text_names = [str(x) for x in text_names]

    gt = load(resolve(root, args.gt_bank))
    gt_names = [str(x) for x in gt.get("part_classes", [])]
    if gt_names != text_names:
        raise ValueError("GT prototype bank taxonomy/order != raw text bank")
    protocol = gt.get("protocol", {})
    if protocol.get("split") not in (None, "", "train"):
        raise ValueError("GT oracle bank must be TRAIN split only")
    target = torch.as_tensor(gt[args.gt_key], dtype=torch.float32).to(device)
    if tuple(target.shape) != (116, 768):
        raise ValueError(f"bad GT prototype shape: {tuple(target.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError("GT prototype bank contains NaN/Inf")
    avail = torch.as_tensor(
        gt.get("available", torch.ones(target.shape[0], dtype=torch.bool)),
        dtype=torch.bool,
        device=device,
    )
    if int(avail.sum()) != raw.shape[0]:
        raise ValueError("oracle requires all GT part prototypes")

    target = F.normalize(target, dim=-1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = []

    with torch.no_grad():
        z0 = F.normalize(model.project_clip_txt(raw).float(), dim=-1)
        init_cos = float((z0 * target).sum(dim=1)[avail].mean().item())

    best = float("inf")
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        z = F.normalize(model.project_clip_txt(raw).float(), dim=-1)
        loss = (1.0 - (z * target).sum(dim=1)[avail]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            zz = F.normalize(model.project_clip_txt(raw).float(), dim=-1)
            post_loss = (1.0 - (zz * target).sum(dim=1)[avail]).mean()
            lv = float(post_loss.item())
            mc = float((zz * target).sum(dim=1)[avail].mean().item())
        if lv < best:
            best = lv
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        if step == 1 or step % 50 == 0 or step == args.steps:
            row = {"step": step, "loss": lv, "mean_matching_cosine": mc}
            hist.append(row)
            print("[oracle]", json.dumps(row))

    if best_state is None:
        raise RuntimeError("no best oracle state")
    model.load_state_dict(best_state, strict=True)

    output = resolve(root, args.output)
    metrics = resolve(root, args.metrics_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output))
    metrics.write_text(json.dumps({
        "protocol": "TRAIN GT part prototype direct cosine ORACLE",
        "initial_mean_matching_cosine": init_cos,
        "best_loss": best,
        "history": hist,
        "output": str(output),
    }, indent=2) + "\n")
    print("GTPROTO_ORACLE_PASS")


if __name__ == "__main__":
    main()
