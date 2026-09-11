#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


def load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def main() -> int:
    p = argparse.ArgumentParser(description="Apply learned W to a fixed [116,768] part text bank")
    p.add_argument("--bank", required=True)
    p.add_argument("--w_checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bank_key", default="features")
    args = p.parse_args()

    bank_payload = load(Path(args.bank).expanduser().resolve())
    if torch.is_tensor(bank_payload):
        bank = bank_payload
        names = None
    elif isinstance(bank_payload, Mapping):
        if args.bank_key not in bank_payload:
            raise KeyError(f"missing bank key {args.bank_key!r}")
        bank = bank_payload[args.bank_key]
        names = bank_payload.get("classnames", bank_payload.get("names"))
    else:
        raise ValueError("bank must be a Tensor or mapping")
    if not torch.is_tensor(bank):
        bank = torch.tensor(bank, dtype=torch.float32)

    w_payload = load(Path(args.w_checkpoint).expanduser().resolve())
    if not isinstance(w_payload, Mapping) or "W" not in w_payload:
        raise ValueError("W checkpoint must contain key 'W'")
    W = w_payload["W"]
    if not torch.is_tensor(W):
        W = torch.tensor(W, dtype=torch.float32)

    bank = bank.float()
    W = W.float()
    if bank.shape != (116, 768) or W.shape != (768, 768):
        raise ValueError(f"expected bank [116,768] and W [768,768], got {tuple(bank.shape)}, {tuple(W.shape)}")
    aligned = F.normalize(bank @ W, dim=-1)

    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": aligned, "source_bank": str(Path(args.bank).expanduser()), "w_checkpoint": str(Path(args.w_checkpoint).expanduser())}
    if names is not None:
        payload["classnames"] = list(names)
    torch.save(payload, str(out))
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
