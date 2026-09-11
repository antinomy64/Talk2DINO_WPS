#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final exact bake for the current CLIP PartStruct projector.

Verified local architecture:
    raw CLIP [B,512]
      -> linear_layer: 512 -> 768
      -> tanh
      -> hidden_layers.0: 768 -> 768
      -> projected text [B,768]
      -> evaluator L2 normalization

Train-W uses:
    T0 = normalize(project_clip_txt(raw_clip))
    T  = normalize(T0 @ W)

Therefore bake W into the LAST affine layer:
    A' = W.T @ A
    b' = W.T @ b

Then:
    normalize(project_clip_txt_baked(x))
      == normalize(normalize(project_clip_txt_original(x)) @ W)

The script:
  - strict-loads the current ProjectionLayer from configs/vitb_mlp_infonce.yaml
  - verifies exact expected parameter shapes
  - verifies source-projector SHA256 against W checkpoint metadata
  - bakes only hidden_layers.0.{weight,bias}
  - certifies equivalence on all 116 real raw CLIP part features
  - reloads the saved checkpoint and certifies again
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml


TARGET_WEIGHT = "hidden_layers.0.weight"
TARGET_BIAS = "hidden_layers.0.bias"
EXPECTED_SHAPES = {
    "linear_layer.weight": (768, 512),
    "linear_layer.bias": (768,),
    "hidden_layers.0.weight": (768, 768),
    "hidden_layers.0.bias": (768,),
}


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


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(obj, str(tmp))
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def resolve(root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def unwrap_state_dict(ckpt: Any) -> tuple[dict[str, torch.Tensor], str | None]:
    if not isinstance(ckpt, Mapping):
        raise ValueError("projector checkpoint must be a mapping")

    for key in ("state_dict", "model_state_dict", "projector_state_dict"):
        if key in ckpt and isinstance(ckpt[key], Mapping):
            state = dict(ckpt[key])
            return state, key

    # Current PartStruct projector is a plain state_dict.
    if not all(torch.is_tensor(v) for v in ckpt.values()):
        raise ValueError("top-level projector checkpoint is not a plain tensor state_dict")
    return dict(ckpt), None


def build_projector(root: Path, config_path: Path, state: Mapping[str, torch.Tensor], device: torch.device):
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, Mapping) or "model" not in cfg:
        raise ValueError("invalid model YAML")

    model_cfg = dict(cfg["model"])
    if model_cfg.get("hidden_layer") is not True:
        raise ValueError(f"expected hidden_layer=True, got {model_cfg.get('hidden_layer')!r}")
    if int(model_cfg.get("dino_embed_dim", -1)) != 768:
        raise ValueError("expected dino_embed_dim=768")

    module = importlib.import_module("src.model")
    class_name = str(model_cfg.get("model_class", "ProjectionLayer"))
    if class_name != "ProjectionLayer":
        raise ValueError(f"expected ProjectionLayer, got {class_name!r}")
    ModelClass = getattr(module, class_name)

    model = ModelClass.from_config(model_cfg)
    model.load_state_dict(dict(state), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def check_exact_architecture(state: Mapping[str, torch.Tensor]) -> None:
    keys = set(state.keys())
    expected = set(EXPECTED_SHAPES.keys())
    if keys != expected:
        raise ValueError(
            "unexpected projector state_dict keys\n"
            f"expected={sorted(expected)}\n"
            f"actual={sorted(keys)}"
        )
    for key, shape in EXPECTED_SHAPES.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(f"{key}: shape {tuple(state[key].shape)} != {shape}")


def normalized_project(model, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        y = model.project_clip_txt(x.float())
        if tuple(y.shape) != (x.shape[0], 768):
            raise ValueError(f"bad projector output shape: {tuple(y.shape)}")
        if not torch.isfinite(y).all():
            raise ValueError("NaN/Inf projector output")
        return F.normalize(y.float(), dim=-1)


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project_root", default=".")
    p.add_argument(
        "--projector",
        default="weights/vitb_mlp_infonce_coco2014_clean_ft10_partstruct_w1e4_lr1e5.pth",
    )
    p.add_argument(
        "--w_checkpoint",
        default="final_exp/relproto_alignment/predobj_bg054_cap4_orth/W_last.pt",
    )
    p.add_argument(
        "--text_bank",
        default="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt",
    )
    p.add_argument("--model_config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument(
        "--output",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_clean_ft10_partstruct_"
            "w1e4_lr1e5_predobj_bg054_relproto4_w_ep10_baked.pth"
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--atol", type=float, default=2e-5)
    p.add_argument("--rtol", type=float, default=2e-5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    projector_path = resolve(root, args.projector)
    w_path = resolve(root, args.w_checkpoint)
    text_path = resolve(root, args.text_bank)
    config_path = resolve(root, args.model_config)
    output_path = resolve(root, args.output)

    for x in (projector_path, w_path, text_path, config_path):
        if not x.is_file():
            raise FileNotFoundError(x)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    if output_path == projector_path:
        raise ValueError("output must not overwrite source projector")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # ------------------------------------------------------------------
    # Source projector
    # ------------------------------------------------------------------
    source_ckpt = torch_load(projector_path)
    source_state, wrapper = unwrap_state_dict(source_ckpt)
    check_exact_architecture(source_state)

    source_sha = sha256(projector_path)
    source_model = build_projector(root, config_path, source_state, device)

    # ------------------------------------------------------------------
    # W checkpoint
    # ------------------------------------------------------------------
    w_ckpt = torch_load(w_path)
    if not isinstance(w_ckpt, Mapping) or not torch.is_tensor(w_ckpt.get("W")):
        raise ValueError("W checkpoint must contain Tensor 'W'")

    W = w_ckpt["W"].detach().cpu().float().contiguous()
    if tuple(W.shape) != (768, 768):
        raise ValueError(f"W shape {tuple(W.shape)} != (768,768)")
    if not torch.isfinite(W).all():
        raise ValueError("W contains NaN/Inf")

    projector_meta = w_ckpt.get("projector", {})
    if not isinstance(projector_meta, Mapping):
        raise ValueError("W checkpoint has invalid projector metadata")
    trained_sha = str(projector_meta.get("weights_sha256", ""))
    if not trained_sha:
        raise ValueError("W checkpoint does not record projector weights_sha256")
    if trained_sha != source_sha:
        raise ValueError(
            "source projector does not match W-training projector\n"
            f"W recorded: {trained_sha}\n"
            f"source    : {source_sha}"
        )

    method = w_ckpt.get("method", {})
    if isinstance(method, Mapping):
        text_contract = str(method.get("text", ""))
        if text_contract and "current @ W" not in text_contract:
            raise ValueError(f"unexpected W text contract: {text_contract}")

    orth_err = float(
        (W.double().T @ W.double() - torch.eye(768, dtype=torch.float64))
        .abs().max().item()
    )
    if orth_err > 1e-3:
        raise ValueError(f"W orthogonality error too large: {orth_err}")

    # ------------------------------------------------------------------
    # Full 116-part raw CLIP bank used in Train-W
    # ------------------------------------------------------------------
    text_payload = torch_load(text_path)
    if not isinstance(text_payload, Mapping) or "features" not in text_payload:
        raise ValueError("text bank must contain 'features'")
    raw_clip = torch.as_tensor(text_payload["features"]).detach().cpu().float()
    if tuple(raw_clip.shape) != (116, 512):
        raise ValueError(f"raw CLIP shape {tuple(raw_clip.shape)} != (116,512)")
    if text_payload.get("normalized", False) is not False:
        raise ValueError("expected raw CLIP bank with normalized=False")

    x = raw_clip.to(device)

    # Exact Train-W target:
    # normalize(normalize(projector(x)) @ W)
    with torch.inference_mode():
        y0 = source_model.project_clip_txt(x.float())
        t0 = F.normalize(y0.float(), dim=-1)
        target = F.normalize(t0 @ W.to(device), dim=-1)

    # ------------------------------------------------------------------
    # Bake only hidden_layers.0, the verified final affine layer.
    # ------------------------------------------------------------------
    baked_state = {k: v.detach().clone() for k, v in source_state.items()}

    A = source_state[TARGET_WEIGHT].detach().cpu()
    b = source_state[TARGET_BIAS].detach().cpu()

    # nn.Linear output uses row-vector y = h @ A.T + b.
    # To obtain y @ W:
    # A' = W.T @ A ; b' = W.T @ b.
    W64 = W.double()
    A_baked = (W64.T @ A.double()).to(dtype=A.dtype)
    b_baked = (W64.T @ b.double()).to(dtype=b.dtype)

    baked_state[TARGET_WEIGHT] = A_baked
    baked_state[TARGET_BIAS] = b_baked

    baked_model = build_projector(root, config_path, baked_state, device)
    actual = normalized_project(baked_model, x)

    max_abs = float((actual - target).abs().max().item())
    mean_abs = float((actual - target).abs().mean().item())
    if not torch.allclose(actual, target, atol=args.atol, rtol=args.rtol):
        raise AssertionError(
            f"pre-save bake equivalence FAILED: max_abs={max_abs:.6e}, "
            f"mean_abs={mean_abs:.6e}"
        )

    # Preserve source checkpoint container format.
    if wrapper is None:
        out_ckpt = baked_state
    else:
        out_ckpt = dict(source_ckpt)
        out_ckpt[wrapper] = baked_state

    atomic_torch_save(out_ckpt, output_path)

    # ------------------------------------------------------------------
    # Reload the exact saved bytes and certify again.
    # ------------------------------------------------------------------
    reload_ckpt = torch_load(output_path)
    reload_state, _ = unwrap_state_dict(reload_ckpt)
    check_exact_architecture(reload_state)
    reload_model = build_projector(root, config_path, reload_state, device)
    reload_actual = normalized_project(reload_model, x)

    reload_max_abs = float((reload_actual - target).abs().max().item())
    reload_mean_abs = float((reload_actual - target).abs().mean().item())
    if not torch.allclose(reload_actual, target, atol=args.atol, rtol=args.rtol):
        raise AssertionError(
            f"reloaded bake equivalence FAILED: max_abs={reload_max_abs:.6e}, "
            f"mean_abs={reload_mean_abs:.6e}"
        )

    report = {
        "status": "PASS",
        "source_projector": str(projector_path),
        "source_projector_sha256": source_sha,
        "W_checkpoint": str(w_path),
        "W_checkpoint_sha256": sha256(w_path),
        "W_epoch": int(w_ckpt.get("epoch", -1)),
        "W_global_step": int(w_ckpt.get("global_step", -1)),
        "W_orthogonality_max_abs": orth_err,
        "text_bank": str(text_path),
        "text_bank_sha256": sha256(text_path),
        "target_weight": TARGET_WEIGHT,
        "target_bias": TARGET_BIAS,
        "bake_rule_weight": "A_baked = W.T @ A",
        "bake_rule_bias": "b_baked = W.T @ b",
        "full_real_text_rows_checked": 116,
        "pre_save_max_abs": max_abs,
        "pre_save_mean_abs": mean_abs,
        "reload_max_abs": reload_max_abs,
        "reload_mean_abs": reload_mean_abs,
        "atol": args.atol,
        "rtol": args.rtol,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }

    report_path = Path(str(output_path) + ".bake_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("============================================================")
    print("Final CLIP PartStruct W bake")
    print("============================================================")
    print("source projector :", projector_path)
    print("W checkpoint     :", w_path)
    print("W epoch          :", report["W_epoch"])
    print("target layer     :", TARGET_WEIGHT)
    print("source shapes    :", EXPECTED_SHAPES)
    print("W orth max       :", f"{orth_err:.3e}")
    print("pre-save max_abs :", f"{max_abs:.3e}")
    print("reload max_abs   :", f"{reload_max_abs:.3e}")
    print("output           :", output_path)
    print("report           :", report_path)
    print("============================================================")
    print("BAKE_PREDOBJ_RELPROTO_W_INTO_PROJECTOR_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
