#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic Talk2DINO-style LLaMA3 text-bank extractor.

This file intentionally follows the SAME bank convention as
`extract_clip_text_bank.py`.

CLIP version:
    class names
      -> Talk2DINO prompt/template set
      -> CLIP text encoder
      -> mean across prompts
      -> raw text bank [N, 512]

This LLaMA3 version:
    class names
      -> built-in visual-description prompt set
      -> LLaMA3 response generation
      -> response-token corr_feats
      -> pool response tokens
      -> mean across prompts / repeats
      -> raw text bank [N, 4096]

IMPORTANT:
- The saved `features` are BEFORE the Talk2DINO projector.
- The final class feature is the direct arithmetic mean of the per-prompt /
  per-repeat LLaMA3 features.
- NO L2 normalization is applied before saving.
- This matches the current CLIP extractor's feature-processing convention:
      prompt features -> arithmetic mean -> save raw mean
- No images, masks, DINO features, or dataset annotations are required.

Supported class-name inputs (choose exactly one):
1) --classes "cat" "dog" "horse"
2) --classes_file classes.txt
3) --classes_file classes.json
4) --classes_source module.path:CLASSES
5) --classes_source module.path:SomeDataset.CLASSES
6) --classes_source /path/to/file.py:CLASSES
7) --classes_source /path/to/file.py:SomeDataset.CLASSES

If the first class is exactly "background" (case-insensitive), it is dropped
by default to match Talk2DINO foreground text encoding. Use
--keep_background to keep it.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from llama.generation_feat import Llama


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CKPT_DIR = "Meta-Llama-3-8B-Instruct"
DEFAULT_TOKENIZER_PATH = "Meta-Llama-3-8B-Instruct/tokenizer.model"

OBJ_SYSTEM_TEMPLATE = (
    "You are a visual object description assistant. "
    "Answer only with concise visual appearance information useful for image segmentation. "
    "Focus on foreground shape, boundary, surface appearance, and visible components. "
    "Avoid function, usage, behavior, scene context, and non-visual commonsense."
)

PART_SYSTEM_TEMPLATE = (
    "You are a visual part description assistant. "
    "Answer only with concise visual appearance information useful for image segmentation. "
    "Focus on local shape, boundary, surface appearance, and visual cues. "
    "Avoid function, usage, behavior, scene context, and non-visual commonsense."
)


# ---------------------------------------------------------------------------
# Built-in prompt templates.
# These are the exact contents previously stored in:
#   llama3_obj_prompts.txt
#   llama3_part_prompts.txt
#
# Keeping them in this file makes feature extraction self-contained.
# ---------------------------------------------------------------------------

OBJ_PROMPTS = [
    "Describe the visible appearance of [cls] for image segmentation. Focus on its overall shape, boundary, color, texture, and distinctive visual components.",
    "What does [cls] usually look like in an image? Describe only visual cues that help recognize and segment the object.",
    "List the key visual characteristics of [cls], including its global shape, surface appearance, common colors, and visible structural parts.",
    "Describe [cls] as an object category using concise visual appearance information, focusing on what can be seen from the image.",
]

PART_PROMPTS = [
    "For the object part named {name}, describe only stable visual attributes useful for part segmentation. Use short phrases for: shape, boundary, local texture, typical color if visually stable, relative position on the object, and adjacent parts. Do not describe function, usage, or object-level context.",
    "Describe the visible cues that distinguish {name} from other parts of the same object. Focus on local shape, contour, size, texture, relative location, and neighboring parts. Avoid function, action, purpose, and general object knowledge.",
    "Describe what image pixels belonging to {name} usually look like in a segmentation mask. Mention visible shape, boundary sharpness, local texture or color, position relative to the whole object, and parts that usually touch it. Do not mention what the part does.",
    "Give compact visual attribute phrases for {name}: shape; boundary; texture; color if stable; relative location; adjacent parts; visual differences from nearby parts. Use no full sentences about function or usage.",
]


def _resolve_dotted_attr(obj, dotted: str):
    cur = obj
    for name in dotted.split("."):
        if not hasattr(cur, name):
            raise AttributeError(
                f"Cannot resolve attribute {dotted!r}; "
                f"missing component {name!r}"
            )
        cur = getattr(cur, name)
    return cur


def _coerce_names(values: Iterable) -> List[str]:
    names = [str(x) for x in values]
    if not names:
        raise ValueError("Class-name list is empty.")
    if any(not x.strip() for x in names):
        raise ValueError("Class-name list contains an empty name.")
    return names


def load_names_from_file(path: str) -> List[str]:
    """
    Same class-file behavior as extract_clip_text_bank.py.

    Supported:
      .json : JSON list, or {"CLASSES": [...]}, {"classes": [...]}
      .txt  : one class name per non-empty line
      .py   : literal-style CLASSES assignment when simple; for arbitrary
              Python use --classes_source path.py:CLASSES instead.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()

    if suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if "CLASSES" in obj:
                obj = obj["CLASSES"]
            elif "classes" in obj:
                obj = obj["classes"]
            else:
                raise ValueError(
                    "JSON dict must contain 'CLASSES' or 'classes'."
                )
        if not isinstance(obj, (list, tuple)):
            raise TypeError("JSON class file must resolve to a list/tuple.")
        return _coerce_names(obj)

    if suffix == ".txt":
        return _coerce_names(
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    if suffix == ".py":
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "CLASSES":
                        value = ast.literal_eval(node.value)
                        return _coerce_names(value)
        raise ValueError(
            f"No literal top-level CLASSES assignment found in {p}. "
            "Use --classes_source file.py:Attr for arbitrary Python."
        )

    raise ValueError(
        f"Unsupported classes_file suffix: {suffix}. "
        "Use .txt, .json, or .py."
    )


def load_names_from_source(spec: str) -> List[str]:
    """
    Same class-source behavior as extract_clip_text_bank.py.

    Examples:
      pkg.dataset:CLASSES
      pkg.dataset:PascalPart116.CLASSES
      /path/to/dataset.py:CLASSES
      /path/to/dataset.py:PascalPart116.CLASSES
    """
    if ":" not in spec:
        raise ValueError(
            "--classes_source must have format SOURCE:ATTRIBUTE, "
            "e.g. module.path:CLASSES"
        )

    source, attr = spec.rsplit(":", 1)

    if source.endswith(".py") or Path(source).exists():
        py_path = Path(source).resolve()
        if not py_path.exists():
            raise FileNotFoundError(py_path)

        module_name = f"_classes_source_{py_path.stem}"
        module_spec = importlib.util.spec_from_file_location(
            module_name, str(py_path)
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot import {py_path}")

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)

    values = _resolve_dotted_attr(module, attr)
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"{spec} must resolve to list/tuple, got {type(values).__name__}"
        )
    return _coerce_names(values)


def resolve_class_names(args) -> List[str]:
    """
    Identical class-list preprocessing convention to the CLIP extractor.
    """
    provided = sum(
        x is not None
        for x in (args.classes, args.classes_file, args.classes_source)
    )
    if provided != 1:
        raise ValueError(
            "Choose exactly one of --classes, --classes_file, "
            "or --classes_source."
        )

    if args.classes is not None:
        names = _coerce_names(args.classes)
    elif args.classes_file is not None:
        names = load_names_from_file(args.classes_file)
    else:
        names = load_names_from_source(args.classes_source)

    if (
        not args.keep_background
        and names
        and names[0].strip().lower() == "background"
    ):
        print("[classes] dropping leading 'background' to match Talk2DINO")
        names = names[1:]

    if not names:
        raise ValueError("No class names remain after preprocessing.")

    if len(set(names)) != len(names):
        duplicates = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"Duplicate class names found: {duplicates}")

    return names


def build_object_groups(class_names: Sequence[str]):
    """
    Same object-group construction as extract_clip_text_bank.py.

    Build object -> row-index groups for part taxonomies written as:
        "horse's head", "horse's leg", ...

    The stored integers are ALWAYS row indices into features/class_names,
    not external dataset category IDs.

    For a generic non-part class list (e.g. VOC21 object names), no grouping
    is forced; an empty dict is returned.
    """
    delimiter = "'s"

    has_delimiter = [delimiter in name for name in class_names]

    if not any(has_delimiter):
        return {}

    if not all(has_delimiter):
        bad = [
            (i, name)
            for i, (name, ok) in enumerate(zip(class_names, has_delimiter))
            if not ok
        ]
        raise ValueError(
            "Mixed grouped/non-grouped class names. When possessive part names "
            "are present, every class must contain \"'s\". "
            f"Examples without it: {bad[:5]}"
        )

    groups = {}
    for row_idx, name in enumerate(class_names):
        object_name = name.split(delimiter, 1)[0].strip()
        if not object_name:
            raise ValueError(
                f"Cannot parse object name from row {row_idx}: {name!r}"
            )
        groups.setdefault(object_name, []).append(row_idx)

    return groups


def get_builtin_prompts(is_part_taxonomy: bool):
    """
    Return the built-in prompt set.

    Part taxonomies are detected by the same possessive-name convention used
    by build_object_groups(). Generic class lists use object prompts.
    """
    if is_part_taxonomy:
        return list(PART_PROMPTS)
    return list(OBJ_PROMPTS)

def fill_prompt(template: str, name: str, bracket_cls: bool) -> str:
    """
    Support both the old object prompt token [cls] and part prompt token {name}.
    """
    cls_text = f"[{name}]" if bracket_cls else name

    out = template.replace("[cls]", cls_text)
    out = out.replace("{name}", name)
    out = out.replace("{cls}", cls_text)
    return out


def build_dialog(
    system_template: str,
    prompt_template: str,
    name: str,
    bracket_cls: bool,
):
    return [
        {"role": "system", "content": system_template},
        {
            "role": "user",
            "content": fill_prompt(
                prompt_template,
                name,
                bracket_cls=bracket_cls,
            ),
        },
    ]


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


def chunked(xs: Sequence[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def pool_corr_feats(corr_feats: List[torch.Tensor], pool: str) -> torch.Tensor:
    """
    Collapse the variable-length LLaMA response-token feature sequence into
    one feature vector for one generated response.

    IMPORTANT:
    No L2 normalization is applied here.

    This gives the closest LLaMA analogue of one CLIP prompt embedding.
    """
    if len(corr_feats) == 0:
        raise RuntimeError(
            "chat_completion_response_feat returned empty corr_feats; "
            "increase max_gen_len or check stop tokens."
        )

    feats = torch.stack(
        [x.detach().float().cpu() for x in corr_feats],
        dim=0,
    )

    if pool == "mean":
        return feats.mean(dim=0)

    if pool == "last":
        return feats[-1]

    raise ValueError(f"Unknown response_pool: {pool}")


@torch.inference_mode()
def encode_talk2dino_llama3_text_bank(
    class_names: Sequence[str],
    prompts: Sequence[str],
    ckpt_dir: str,
    tokenizer_path: str,
    system_template: str,
    bracket_cls: bool,
    num_epochs: int,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    response_pool: str,
    max_seq_len: int,
    max_batch_size: int,
    local_rank: int,
    master_port: str,
    seed: int,
):
    """
    LLaMA3 counterpart of encode_talk2dino_text_bank() in the CLIP extractor.

    Processing convention:
      prompts -> LLaMA3 -> one feature per generated response
              -> arithmetic mean across all prompt/repeat features

    There is deliberately NO per-response normalization and NO final
    normalization before returning the bank.
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("Built-in prompt list is empty.")

    init_distributed_env(master_port)
    set_seed(seed)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
        local_rank=local_rank,
        MASTER_PORT=str(master_port),
    )

    feature_blocks: List[torch.Tensor] = []

    for epoch in range(num_epochs):
        for prompt_idx, prompt_template in enumerate(prompts):
            dialogs = [
                build_dialog(
                    system_template,
                    prompt_template,
                    name,
                    bracket_cls=bracket_cls,
                )
                for name in class_names
            ]

            class_features: List[torch.Tensor | None] = [
                None for _ in class_names
            ]

            indexed_dialogs = list(enumerate(dialogs))
            batches = list(chunked(indexed_dialogs, max_batch_size))

            pbar = tqdm(
                batches,
                desc=(
                    f"epoch {epoch + 1}/{num_epochs}, "
                    f"prompt {prompt_idx + 1}/{len(prompts)}"
                ),
            )

            for batch in pbar:
                batch_indices = [idx for idx, _ in batch]
                batch_dialogs = [dialog for _, dialog in batch]

                outputs = generator.chat_completion_response_feat(
                    batch_dialogs,  # type: ignore[arg-type]
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                if len(outputs) != len(batch_indices):
                    raise RuntimeError(
                        f"LLaMA returned {len(outputs)} outputs "
                        f"for batch size {len(batch_indices)}"
                    )

                for local_i, out in enumerate(outputs):
                    class_idx = batch_indices[local_i]
                    generation = out["generation"]

                    feature = pool_corr_feats(
                        generation["corr_feats"],
                        pool=response_pool,
                    ).float()

                    if feature.ndim != 1:
                        raise RuntimeError(
                            "Expected pooled LLaMA3 response feature to be 1-D, "
                            f"got {tuple(feature.shape)}"
                        )

                    if not torch.isfinite(feature).all():
                        raise RuntimeError(
                            f"Non-finite LLaMA3 feature for class row {class_idx}"
                        )

                    class_features[class_idx] = feature.cpu()

            missing = [
                idx
                for idx, feature in enumerate(class_features)
                if feature is None
            ]
            if missing:
                raise RuntimeError(
                    f"Missing LLaMA3 features for class rows: {missing[:30]}"
                )

            # [N, D] -- one LLaMA feature per class for this prompt/repeat.
            feature_blocks.append(
                torch.stack(
                    [x for x in class_features if x is not None],
                    dim=0,
                )
            )

    # Exact analogue of CLIP:
    #
    # CLIP:
    #   prompt_features [N, num_templates, D]
    #   mean_features = prompt_features.mean(dim=1)
    #
    # LLaMA3:
    #   prompt_features [N, num_epochs*num_prompts, D]
    #   mean_features = prompt_features.mean(dim=1)
    prompt_features = torch.stack(feature_blocks, dim=1)
    mean_features = prompt_features.mean(dim=1).float().cpu()

    return mean_features, prompts


def main():
    parser = argparse.ArgumentParser(
        "Generic Talk2DINO-style LLaMA3 text-bank extractor."
    )

    # ------------------------------------------------------------------
    # Keep the same class-name input interface as extract_clip_text_bank.py.
    # ------------------------------------------------------------------
    input_group = parser.add_argument_group("class-name input")
    input_group.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help='Direct class list, e.g. --classes "cat" "dog" "horse"',
    )
    input_group.add_argument(
        "--classes_file",
        default=None,
        help="Class list from .txt, .json, or simple .py file.",
    )
    input_group.add_argument(
        "--classes_source",
        default=None,
        help=(
            "Load Python CLASSES-like attribute: "
            "module.path:CLASSES, module.path:Dataset.CLASSES, "
            "/path/file.py:CLASSES, or /path/file.py:Dataset.CLASSES"
        ),
    )

    parser.add_argument("--out_path", required=True)

    # LLaMA3-specific encoder inputs.
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--tokenizer_path",
        default=DEFAULT_TOKENIZER_PATH,
    )

    parser.add_argument(
        "--system_template",
        default=None,
        help=(
            "Optional explicit system prompt. If omitted, choose object/part "
            "default from the class taxonomy."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["auto", "obj", "part"],
        default="auto",
        help=(
            "Choose the built-in prompt set. 'auto' uses part prompts when "
            "class names form a Pascal-Part-style taxonomy, otherwise object prompts."
        ),
    )
    parser.add_argument(
        "--bracket_cls",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_bracket_cls",
        dest="bracket_cls",
        action="store_false",
    )

    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--response_pool",
        choices=["mean", "last"],
        default="mean",
    )

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", type=str, default="5678")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--keep_background",
        action="store_true",
        help=(
            "Keep a leading 'background' class. By default it is dropped "
            "to match Talk2DINO VOC-style text encoding."
        ),
    )

    args = parser.parse_args()

    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be > 0")
    if args.max_batch_size <= 0:
        raise ValueError("--max_batch_size must be > 0")
    if args.max_gen_len <= 0:
        raise ValueError("--max_gen_len must be > 0")

    class_names = resolve_class_names(args)

    # Same grouping rule as the CLIP bank extractor.
    object_groups = build_object_groups(class_names)

    # Choose built-in prompt set and system prompt.
    if args.prompt_mode == "auto":
        is_part = bool(object_groups)
    else:
        is_part = args.prompt_mode == "part"

    prompts = get_builtin_prompts(is_part_taxonomy=is_part)

    if args.system_template is None:
        system_template = PART_SYSTEM_TEMPLATE if is_part else OBJ_SYSTEM_TEMPLATE
    else:
        system_template = args.system_template

    features, prompts = encode_talk2dino_llama3_text_bank(
        class_names=class_names,
        prompts=prompts,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        system_template=system_template,
        bracket_cls=args.bracket_cls,
        num_epochs=args.num_epochs,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        response_pool=args.response_pool,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        local_rank=args.local_rank,
        master_port=args.master_port,
        seed=args.seed,
    )

    if features.ndim != 2 or features.shape[0] != len(class_names):
        raise RuntimeError(
            f"Unexpected feature shape {tuple(features.shape)} "
            f"for {len(class_names)} classes."
        )

    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite values found in output features.")

    # ------------------------------------------------------------------
    # SAME core bank schema as extract_clip_text_bank.py.
    # ------------------------------------------------------------------
    output = {
        "features": features.contiguous(),       # [N,D], raw prompt mean
        "class_names": list(class_names),
        "object_groups": object_groups,          # object -> feature row indices
        "llama_model": Path(args.ckpt_dir).name,
        "prompt_source": "builtin",
        "prompt_mode": "part" if is_part else "obj",
        "num_prompts": len(prompts),
        "num_epochs": args.num_epochs,
        "num_features_per_class": args.num_epochs * len(prompts),
        "response_pool": args.response_pool,
        "normalized": False,
        "stage": "pre_projector_prompt_mean",
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    norms = features.norm(dim=-1)

    print("============================================================")
    print("Talk2DINO-style LLaMA3 text bank")
    print("============================================================")
    print("output        :", out_path)
    print("classes       :", len(class_names))
    print("object groups :", len(object_groups))
    print("feature shape :", tuple(features.shape))
    print("dtype         :", features.dtype)
    print("LLaMA         :", Path(args.ckpt_dir).name)
    print("prompt source :", "builtin")
    print("prompt mode   :", "part" if is_part else "obj")
    print("prompts       :", len(prompts))
    print("epochs        :", args.num_epochs)
    print("features/class:", args.num_epochs * len(prompts))
    print("response pool :", args.response_pool)
    print("normalized    : False")
    print("stage         : pre_projector_prompt_mean")
    print("finite        :", bool(torch.isfinite(features).all()))
    print(
        "raw norm      : "
        f"min={norms.min().item():.6f}, "
        f"mean={norms.mean().item():.6f}, "
        f"max={norms.max().item():.6f}"
    )

    if object_groups:
        print()
        print("Object groups (row indices):")
        total_grouped = 0
        for object_name in sorted(object_groups):
            rows = object_groups[object_name]
            total_grouped += len(rows)
            print(
                f"  {object_name:<16} "
                f"{len(rows):>2} parts  rows={rows}"
            )
        print("  grouped total   :", total_grouped)

        if total_grouped != len(class_names):
            raise RuntimeError(
                f"Grouped rows={total_grouped}, classes={len(class_names)}"
            )

    print()
    print("First classes:")
    for idx, name in enumerate(class_names[:10]):
        print(f"  {idx:03d}: {name}")
    print("============================================================")


if __name__ == "__main__":
    main()
