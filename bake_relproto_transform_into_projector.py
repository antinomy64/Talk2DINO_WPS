#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bake a learned right-side 768x768 RelProto transform into a Talk2DINO projector.

For projector output y = P(x), Stage-3 uses:
    y_new = y @ A

For the final nn.Linear y = z @ W_final.T + b_final, this is exactly:
    W_final_new = A.T @ W_final
    b_final_new = b_final @ A

Supports both the orthogonal W and the unconstrained-linear ablation because
both checkpoints store the learned matrix under key "W".
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Mapping, Any

import torch
import yaml


def tload(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def unwrap_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "projector_state_dict"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise TypeError("projector checkpoint is not a state_dict mapping")
    state = {
        str(k): v.detach().cpu().clone()
        for k, v in obj.items()
        if torch.is_tensor(v)
    }
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


def final_linear_keys(state: Mapping[str, torch.Tensor]):
    # Current checkpoints.
    hidden = []
    for key in state:
        if key.startswith("hidden_layers.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) == 3 and parts[1].isdigit():
                hidden.append(int(parts[1]))
    if hidden:
        i = max(hidden)
        return f"hidden_layers.{i}.weight", f"hidden_layers.{i}.bias"

    # Legacy Talk2DINO checkpoints are remapped by ProjectionLayer.load_state_dict
    # from linear_layer2.* -> hidden_layers.0.*.  Bake the actual final layer
    # before that compatibility remap.
    if "linear_layer2.weight" in state:
        return "linear_layer2.weight", "linear_layer2.bias"

    # No hidden layer: the first linear is also the output linear.
    return "linear_layer.weight", "linear_layer.bias"


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--projector", required=True)
    p.add_argument("--transform_checkpoint", required=True)
    p.add_argument("--model_config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--verify_device", default="cpu")
    args = p.parse_args()

    projector_path = Path(args.projector).resolve()
    transform_path = Path(args.transform_checkpoint).resolve()
    config_path = Path(args.model_config).resolve()
    output_path = Path(args.output).resolve()

    for x in (projector_path, transform_path, config_path):
        if not x.is_file():
            raise FileNotFoundError(x)
    if output_path.exists():
        raise FileExistsError(output_path)

    state = unwrap_state(tload(projector_path))
    trans_ckpt = tload(transform_path)
    if not isinstance(trans_ckpt, Mapping) or "W" not in trans_ckpt:
        raise KeyError("transform checkpoint must contain W")

    A = torch.as_tensor(trans_ckpt["W"], dtype=torch.float32).cpu()
    if tuple(A.shape) != (768, 768):
        raise ValueError(f"expected transform [768,768], got {tuple(A.shape)}")
    if not torch.isfinite(A).all():
        raise ValueError("nonfinite transform")

    wk, bk = final_linear_keys(state)
    if wk not in state or bk not in state:
        raise KeyError(f"final projector linear keys missing: {wk}, {bk}")

    old_w = state[wk].float()
    old_b = state[bk].float()
    if old_w.shape[0] != 768 or old_b.shape != (768,):
        raise ValueError(
            f"final output layer must produce 768 dims; "
            f"weight={tuple(old_w.shape)} bias={tuple(old_b.shape)}"
        )

    state[wk] = (A.T @ old_w).to(dtype=state[wk].dtype)
    state[bk] = (old_b @ A).to(dtype=state[bk].dtype)

    # Numerical identity check against real ProjectionLayer implementation.
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]
    cls_name = cfg.get("model_class", "ProjectionLayer")
    cls = getattr(importlib.import_module("src.model"), cls_name)

    old_model = cls.from_config(cfg)
    new_model = cls.from_config(cfg)
    old_model.load_state_dict(unwrap_state(tload(projector_path)), strict=True)
    new_model.load_state_dict(state, strict=True)
    old_model.eval()
    new_model.eval()

    in_dim = int(old_model.linear_layer.in_features)
    torch.manual_seed(3407)
    x = torch.randn(17, in_dim)

    with torch.no_grad():
        expected = old_model.project_clip_txt(x).float() @ A
        actual = new_model.project_clip_txt(x).float()

    max_err = float((expected - actual).abs().max().item())
    if max_err > 5e-5:
        raise RuntimeError(f"bake verification failed: max_abs_error={max_err}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)

    print("BAKE_PASS")
    print("projector :", projector_path)
    print("transform :", transform_path)
    print("format    :", trans_ckpt.get("format", trans_ckpt.get("version", "")))
    print("layer     :", wk)
    print("output    :", output_path)
    print("max error :", max_err)


if __name__ == "__main__":
    main()
