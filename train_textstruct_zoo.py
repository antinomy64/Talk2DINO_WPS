#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one text-structure candidate with the current Talk2DINO COCO protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import yaml

from src.dataset import DinoClipDataset
from src.loss_textstruct_zoo import METHODS
from src.train_util_textstruct_zoo import do_train_textstruct


def parse():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--repo_root", default=".")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--train_dataset", required=True)
    p.add_argument("--val_dataset", required=True)
    p.add_argument("--feature_name", default="disentangled_self_attn")
    p.add_argument("--text_features", default="ann_feats")
    p.add_argument("--init_weight", required=True)
    p.add_argument("--structure_bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metrics_json", required=True)

    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--optimizer", choices=("Adam", "AdamW"), default="Adam")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--base_structure_weight", type=float, default=1e-4)
    p.add_argument("--no_calibrate_grad", action="store_true")

    p.add_argument("--rank_temperature", type=float, default=0.05)
    p.add_argument("--neighbor_temperature", type=float, default=0.07)
    p.add_argument("--confidence_delta", type=float, default=0.03)
    p.add_argument("--confidence_alpha", type=float, default=0.5)
    p.add_argument("--confidence_margin_max", type=float, default=0.10)
    p.add_argument("--confidence_max_triplets_per_anchor", type=int, default=256)
    p.add_argument("--rkd_angle_triplets", type=int, default=16384)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--topk_margin", type=float, default=0.03)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve(root, raw):
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def load_state(path):
    try:
        x = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        x = torch.load(str(path), map_location="cpu")
    if isinstance(x, dict) and "state_dict" in x:
        x = x["state_dict"]
    return x


def main():
    args = parse()
    root = Path(args.repo_root).resolve()
    os.chdir(root)

    cfg_path = resolve(root, args.model_config)
    init = resolve(root, args.init_weight)
    train_path = resolve(root, args.train_dataset)
    val_path = resolve(root, args.val_dataset)
    bank = resolve(root, args.structure_bank)
    output = resolve(root, args.output)
    metrics_json = resolve(root, args.metrics_json)

    for p in (cfg_path, init, train_path, val_path, bank):
        if not p.is_file():
            raise FileNotFoundError(p)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    ModelClass = getattr(
        importlib.import_module("src.model"),
        cfg["model"].get("model_class", "ProjectionLayer"),
    )
    model = ModelClass.from_config(cfg["model"])
    model.load_state_dict(load_state(init), strict=True)
    model.to(device)

    train_cfg = dict(cfg["train"])
    train_cfg["num_epochs"] = int(args.num_epochs)
    train_cfg["lr"] = float(args.lr)

    train_dataset = DinoClipDataset(
        str(train_path),
        features_name=args.feature_name,
        text_features=args.text_features,
        load_attn_maps=args.feature_name == "patch_tokens",
        is_wds=".tar" in str(train_path),
    )
    val_dataset = DinoClipDataset(
        str(val_path),
        features_name=(
            "avg_self_attn_out"
            if args.feature_name == "disentangled_self_attn"
            else args.feature_name
        ),
        text_features=args.text_features,
        load_attn_maps=args.feature_name == "patch_tokens",
        is_wds=".tar" in str(val_path),
    )

    model, metrics = do_train_textstruct(
        model,
        train_dataset,
        val_dataset,
        train_cfg,
        method=args.method,
        structure_bank=str(bank),
        base_structure_weight=args.base_structure_weight,
        calibrate_grad=not args.no_calibrate_grad,
        rank_temperature=args.rank_temperature,
        neighbor_temperature=args.neighbor_temperature,
        confidence_delta=args.confidence_delta,
        confidence_alpha=args.confidence_alpha,
        confidence_margin_max=args.confidence_margin_max,
        confidence_max_triplets_per_anchor=args.confidence_max_triplets_per_anchor,
        rkd_angle_triplets=args.rkd_angle_triplets,
        topk=args.topk,
        topk_margin=args.topk_margin,
        seed=args.seed,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        warmup=args.warmup,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output))

    metrics.update({
        "output": str(output),
        "init_weight": str(init),
        "model_config": str(cfg_path),
        "structure_bank": str(bank),
        "protocol": "COCO InfoNCE + text-internal structure loss",
        "structure_supervision": "raw CLIP text internal relations only",
    })
    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )

    print("[saved]", output)
    print("[metrics]", metrics_json)
    print("TEXTSTRUCT_TRAIN_PASS")


if __name__ == "__main__":
    main()
