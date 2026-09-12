#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract LLaMA3 caption features for Talk2DINO/WPS COCO projector training.

This is the LLaMA3 counterpart of the repository's text_features_extraction.py.

Talk2DINO CLIP path:
    annotation["caption"]
      -> CLIP encode_text
      -> annotation["ann_feats"] [512]
      -> DinoClipDataset(..., text_features="ann_feats")
      -> ProjectionLayer(512 -> 768)

This LLaMA3 path:
    annotation["caption"]
      -> fixed visual-description system/user instruction
      -> LLaMA3 generated response
      -> response-token corr_feats
      -> mean pool response tokens
      -> annotation["llama3_ann_feats"] [4096]
      -> DinoClipDataset(..., text_features="llama3_ann_feats")
      -> ProjectionLayer(4096 -> 768)

Feature convention intentionally matches extract_llama3_text_bank.py:
- one generated response feature = arithmetic mean of response corr_feats;
- NO L2 normalization before saving;
- raw pre-projector feature;
- default storage is float32 (use --save_half only to reduce disk usage).

The original CLIP ann_feats field is preserved.

For long COCO extraction, features are checkpointed into small shard files.
`--resume` safely reuses only shards whose extraction signature matches the
current prompt/model/extraction settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from llama.generation_feat import Llama


# ---------------------------------------------------------------------------
# Built-in COCO-caption prompts.
# The system prompt constrains the LLM to visual information supported by the
# original caption; the user prompt supplies exactly one COCO caption.
# ---------------------------------------------------------------------------

COCO_SYSTEM_TEMPLATE = (
    "You are a visual image description assistant. "
    "Represent only visual information that is explicitly supported by the "
    "given image caption. Focus on visible objects, appearance, attributes, "
    "spatial relations, and scene layout. Do not add hidden facts, intentions, "
    "functions, actions not stated by the caption, or unsupported details."
)

COCO_USER_TEMPLATE = (
    "Describe the visible content expressed by this image caption using concise "
    "visual information useful for image-text alignment. Preserve the objects "
    "and relations stated in the caption and do not invent extra details.\n\n"
    "Caption: {caption}"
)


def tload(path: str | Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_env(master_port: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")


def chunked_indices(start: int, end: int, batch_size: int):
    for s in range(start, end, batch_size):
        yield s, min(s + batch_size, end)


def build_dialog(caption: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": COCO_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": COCO_USER_TEMPLATE.format(caption=str(caption)),
        },
    ]


def pool_corr_feats(corr_feats: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Same response-feature convention as extract_llama3_text_bank.py:
      response token features -> arithmetic mean -> raw vector

    No L2 normalization is applied.
    """
    if len(corr_feats) == 0:
        raise RuntimeError(
            "LLaMA generation returned empty corr_feats. "
            "Check max_gen_len / stop-token behavior."
        )
    x = torch.stack(
        [torch.as_tensor(v).detach().float().cpu() for v in corr_feats],
        dim=0,
    )
    feat = x.mean(dim=0)
    if feat.ndim != 1:
        raise RuntimeError(
            f"Expected pooled response feature [D], got {tuple(feat.shape)}"
        )
    if not torch.isfinite(feat).all():
        raise RuntimeError("Non-finite LLaMA caption feature.")
    return feat


def extraction_signature(args: argparse.Namespace) -> str:
    payload = {
        "system": COCO_SYSTEM_TEMPLATE,
        "user": COCO_USER_TEMPLATE,
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "max_seq_len": int(args.max_seq_len),
        "max_gen_len": int(args.max_gen_len),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "expected_dim": int(args.expected_dim),
        "feature_logic": "generated_response_corr_feats_mean_raw_v1",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def shard_path(cache_dir: Path, start: int, end: int) -> Path:
    return cache_dir / f"caption_feats_{start:07d}_{end:07d}.pt"


@torch.inference_mode()
def extract_shard(
    generator: Llama,
    annotations: Sequence[Dict[str, Any]],
    start: int,
    end: int,
    batch_size: int,
    expected_dim: int,
    signature: str,
    save_half: bool,
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    rows: List[torch.Tensor] = []
    annotation_ids: List[int] = []

    for bs, be in tqdm(
        list(chunked_indices(start, end, batch_size)),
        desc=f"extract {start}:{end}",
        leave=False,
    ):
        batch_annotations = annotations[bs:be]
        dialogs = []

        for ann in batch_annotations:
            if "caption" not in ann:
                raise KeyError(
                    f"COCO annotation has no 'caption'. keys={list(ann.keys())}"
                )
            dialogs.append(build_dialog(ann["caption"]))

        outputs = generator.chat_completion_response_feat(
            dialogs,  # type: ignore[arg-type]
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        if len(outputs) != len(batch_annotations):
            raise RuntimeError(
                f"LLaMA returned {len(outputs)} outputs for "
                f"{len(batch_annotations)} captions."
            )

        for ann, out in zip(batch_annotations, outputs):
            generation = out["generation"]
            feat = pool_corr_feats(generation["corr_feats"])

            if expected_dim > 0 and int(feat.numel()) != expected_dim:
                raise RuntimeError(
                    f"Expected LLaMA feature dim={expected_dim}, "
                    f"got {feat.numel()}."
                )

            rows.append(feat)
            annotation_ids.append(int(ann.get("id", -1)))

    features = torch.stack(rows, dim=0)
    if save_half:
        features = features.half()
    else:
        features = features.float()

    return {
        "start": int(start),
        "end": int(end),
        "annotation_ids": annotation_ids,
        "features": features.contiguous(),
        "signature": signature,
        "normalized": False,
        "stage": "pre_projector_response_mean",
    }


def validate_existing_shard(
    obj: Dict[str, Any],
    start: int,
    end: int,
    signature: str,
    expected_dim: int,
) -> bool:
    try:
        if obj["signature"] != signature:
            return False
        if int(obj["start"]) != int(start) or int(obj["end"]) != int(end):
            return False
        x = torch.as_tensor(obj["features"])
        if tuple(x.shape[:1]) != (end - start,):
            return False
        if expected_dim > 0 and int(x.shape[-1]) != expected_dim:
            return False
        if not torch.isfinite(x.float()).all():
            return False
        return True
    except Exception:
        return False


def inject_shards(
    data: Dict[str, Any],
    cache_dir: Path,
    n_annotations: int,
    shard_size: int,
    output_key: str,
    signature: str,
    expected_dim: int,
) -> None:
    annotations = data["annotations"]
    written = 0

    for start in range(0, n_annotations, shard_size):
        end = min(start + shard_size, n_annotations)
        p = shard_path(cache_dir, start, end)
        if not p.is_file():
            raise FileNotFoundError(f"Missing feature shard: {p}")

        shard = tload(p)
        if not validate_existing_shard(
            shard, start, end, signature, expected_dim
        ):
            raise RuntimeError(f"Invalid/mismatched shard: {p}")

        feats = torch.as_tensor(shard["features"]).cpu()
        ids = shard["annotation_ids"]

        for local_i, row_idx in enumerate(range(start, end)):
            ann = annotations[row_idx]
            ann_id = int(ann.get("id", -1))
            if ann_id != int(ids[local_i]):
                raise RuntimeError(
                    f"Annotation-order mismatch at row {row_idx}: "
                    f"PTH id={ann_id}, shard id={ids[local_i]}"
                )
            ann[output_key] = feats[local_i].contiguous()
            written += 1

    if written != n_annotations:
        raise RuntimeError(
            f"Injected {written} features but expected {n_annotations}."
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract LLaMA3 generated-response features for COCO captions "
            "and inject them into a Talk2DINO ready PTH."
        )
    )
    p.add_argument("--ann_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument(
        "--output_key",
        default="llama3_ann_feats",
        help=(
            "Annotation field consumed by DinoClipDataset via "
            "--text_features. Original ann_feats is preserved."
        ),
    )

    p.add_argument("--ckpt_dir", default="Meta-Llama-3-8B-Instruct")
    p.add_argument(
        "--tokenizer_path",
        default="Meta-Llama-3-8B-Instruct/tokenizer.model",
    )

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--max_gen_len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--expected_dim", type=int, default=4096)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--master_port", default="5688")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--shard_size",
        type=int,
        default=2048,
        help="Number of caption features per resumable cache shard.",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Resumable feature-shard directory. Default: "
            "<out_path>.llama3_cache"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing feature shards.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing final out_path.",
    )
    p.add_argument(
        "--save_half",
        action="store_true",
        help=(
            "Store injected features as fp16 to reduce disk size. "
            "Training converts annotations to fp32."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")
    if args.max_gen_len <= 0:
        raise ValueError("--max_gen_len must be > 0")

    ann_path = Path(args.ann_path).resolve()
    out_path = Path(args.out_path).resolve()

    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    if ann_path == out_path:
        raise ValueError(
            "Refusing in-place modification. Use a different --out_path."
        )
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_path}. Use --overwrite if intentional."
        )

    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else Path(str(out_path) + ".llama3_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading Talk2DINO ready PTH...")
    data = tload(ann_path)
    if not isinstance(data, dict):
        raise TypeError("Expected top-level dict in PTH.")
    if "annotations" not in data or "images" not in data:
        raise KeyError("PTH must contain top-level 'images' and 'annotations'.")

    annotations = data["annotations"]
    n = len(annotations)
    if n == 0:
        raise RuntimeError("No annotations found.")

    for i, ann in enumerate(annotations[:10]):
        if "caption" not in ann:
            raise KeyError(f"Annotation {i} has no caption.")

    signature = extraction_signature(args)

    print(f"annotations : {n}")
    print(f"output key  : {args.output_key}")
    print(f"signature   : {signature}")
    print(f"cache dir   : {cache_dir}")
    print("normalized  : False")
    print("feature     : generated response corr_feats -> token mean")
    print()

    init_distributed_env(args.master_port)
    set_seed(args.seed)

    print("[2/4] Building LLaMA3...")
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        seed=args.seed,
        local_rank=args.local_rank,
        MASTER_PORT=str(args.master_port),
    )

    print("[3/4] Extracting resumable caption-feature shards...")
    starts = list(range(0, n, args.shard_size))
    for shard_idx, start in enumerate(starts):
        end = min(start + args.shard_size, n)
        p = shard_path(cache_dir, start, end)

        if p.is_file() and args.resume:
            old = tload(p)
            if validate_existing_shard(
                old, start, end, signature, args.expected_dim
            ):
                print(
                    f"[resume {shard_idx + 1}/{len(starts)}] "
                    f"{start}:{end}"
                )
                continue
            raise RuntimeError(
                f"Existing shard has different settings or is invalid: {p}\n"
                "Delete the cache directory or use a new --cache_dir."
            )

        shard = extract_shard(
            generator=generator,
            annotations=annotations,
            start=start,
            end=end,
            batch_size=args.batch_size,
            expected_dim=args.expected_dim,
            signature=signature,
            save_half=args.save_half,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        atomic_torch_save(shard, p)
        print(
            f"[saved {shard_idx + 1}/{len(starts)}] "
            f"{p.name} {tuple(shard['features'].shape)} "
            f"{shard['features'].dtype}"
        )

    print("[4/4] Injecting features and writing final Talk2DINO PTH...")
    inject_shards(
        data=data,
        cache_dir=cache_dir,
        n_annotations=n,
        shard_size=args.shard_size,
        output_key=args.output_key,
        signature=signature,
        expected_dim=args.expected_dim,
    )

    data.setdefault("text_feature_banks", {})["llama3_caption"] = {
        "field": args.output_key,
        "dim": int(args.expected_dim),
        "normalized": False,
        "stage": "pre_projector_response_mean",
        "feature_logic": "generated_response_corr_feats_mean_raw_v1",
        "system_template": COCO_SYSTEM_TEMPLATE,
        "user_template": COCO_USER_TEMPLATE,
        "model": str(Path(args.ckpt_dir).name),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_gen_len": int(args.max_gen_len),
        "seed": int(args.seed),
        "signature": signature,
        "stored_dtype": "float16" if args.save_half else "float32",
    }

    atomic_torch_save(data, out_path)

    # Final interface smoke checks.
    check = tload(out_path)
    first = torch.as_tensor(check["annotations"][0][args.output_key])
    last = torch.as_tensor(check["annotations"][-1][args.output_key])

    if tuple(first.shape) != (args.expected_dim,):
        raise RuntimeError(f"First feature shape mismatch: {tuple(first.shape)}")
    if tuple(last.shape) != (args.expected_dim,):
        raise RuntimeError(f"Last feature shape mismatch: {tuple(last.shape)}")
    if not torch.isfinite(first.float()).all():
        raise RuntimeError("First feature contains non-finite values.")
    if not torch.isfinite(last.float()).all():
        raise RuntimeError("Last feature contains non-finite values.")

    print()
    print("============================================================")
    print("LLaMA3 COCO caption extraction COMPLETE")
    print("============================================================")
    print("input       :", ann_path)
    print("output      :", out_path)
    print("annotations :", n)
    print("field       :", args.output_key)
    print("shape       :", tuple(first.shape))
    print("dtype       :", first.dtype)
    print("normalized  : False")
    print("projector in: 4096")
    print("DinoClipDataset training arg:")
    print(f"  --text_features {args.output_key}")
    print("============================================================")


if __name__ == "__main__":
    main()
