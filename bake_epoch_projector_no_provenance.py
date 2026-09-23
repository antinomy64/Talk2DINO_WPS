#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path

import torch
from src.model import ProjectionLayer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_projector", required=True)
    p.add_argument("--w_checkpoint", required=True)
    p.add_argument("--text_bank", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ["state_dict", "model_state_dict", "model", "projector", "proj"]:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def clean_state_dict(state):
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "proj.", "projection_layer."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def find_tensor(obj, n=116, d=512):
    found = []

    def visit(x, name="root"):
        if torch.is_tensor(x):
            if x.ndim == 2 and tuple(x.shape) == (n, d):
                found.append((name, x))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, f"{name}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                visit(v, f"{name}[{i}]")

    visit(obj)
    if not found:
        raise RuntimeError(f"Cannot find [{n},{d}] text tensor in {type(obj)}")
    if len(found) > 1:
        print("[INFO] multiple text tensors found; using", found[0][0])
    return found[0][1].float()


def load_model(config_path, weight_path, device):
    model = ProjectionLayer.from_config(config_path)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = clean_state_dict(unwrap_state_dict(ckpt))
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def get_module(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(name)
    return modules[name]


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    base = load_model(args.model_config, args.base_projector, device)

    text_obj = torch.load(args.text_bank, map_location="cpu")
    raw_text = find_tensor(text_obj).to(device)

    w_obj = torch.load(args.w_checkpoint, map_location="cpu")
    if not isinstance(w_obj, dict) or "W" not in w_obj:
        raise RuntimeError(f"{args.w_checkpoint} does not contain key 'W'")
    W = w_obj["W"].float().to(device)
    if tuple(W.shape) != (768, 768):
        raise RuntimeError(f"Expected W [768,768], got {tuple(W.shape)}")

    base_out = base.project_clip_txt(raw_text.float()).float()
    target = base_out @ W

    executed = []
    handles = []

    def hook(name):
        def fn(module, inputs, output):
            if torch.is_tensor(output):
                executed.append((name, tuple(output.shape)))
        return fn

    for name, module in base.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 768:
            handles.append(module.register_forward_hook(hook(name)))

    _ = base.project_clip_txt(raw_text.float())
    for h in handles:
        h.remove()

    candidates = []
    seen = set()
    for name, shape in reversed(executed):
        if name not in seen and len(shape) >= 2 and shape[-1] == 768:
            candidates.append(name)
            seen.add(name)

    if not candidates:
        candidates = [
            name for name, module in reversed(list(base.named_modules()))
            if isinstance(module, torch.nn.Linear) and module.out_features == 768
        ]

    if not candidates:
        raise RuntimeError("Cannot find a 768-d output Linear layer in ProjectionLayer")

    best = None
    for name in candidates:
        model = copy.deepcopy(base)
        layer = get_module(model, name)
        if not isinstance(layer, torch.nn.Linear):
            continue
        if layer.weight.ndim != 2 or layer.weight.shape[0] != 768:
            continue

        layer.weight.copy_(W.T @ layer.weight)
        if layer.bias is not None:
            layer.bias.copy_(W.T @ layer.bias)

        out = model.project_clip_txt(raw_text.float()).float()
        err = (out - target).abs()
        max_abs = err.max().item()
        mean_abs = err.mean().item()
        print(f"[CHECK] candidate={name} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e}")

        if best is None or max_abs < best[0]:
            best = (max_abs, mean_abs, name, model)

    if best is None:
        raise RuntimeError("No valid output Linear candidate could be transformed")

    max_abs, mean_abs, name, model = best
    if max_abs > 5e-4:
        raise RuntimeError(
            f"Bake verification failed: best layer={name}, max_abs={max_abs:.8e}. "
            "Expected baked projector output to equal base_output @ W."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_path)

    print(f"[OK] selected output layer: {name}")
    print(f"[OK] selector={w_obj.get('selector')} mapping={w_obj.get('mapping')} epoch={w_obj.get('epoch')}")
    print(f"[OK] verification max_abs={max_abs:.8e}, mean_abs={mean_abs:.8e}")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
