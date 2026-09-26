#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic Talk2DINO-style CLIP text-bank extractor.

It reproduces the text side used by Talk2DINO evaluation up to (but NOT
including) the learned projector:

    class names
      -> Talk2DINO template set (default: sub_imagenet_template)
      -> clip.tokenize
      -> OpenAI CLIP text encoder
      -> mean across prompts
      -> raw text bank [N, D]

Important:
- The saved features are the prompt-mean CLIP embeddings BEFORE the Talk2DINO
  projector.
- They are intentionally NOT L2-normalized before saving, matching the
  Talk2DINO evaluation order.
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
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import clip
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_get_template():
    """Resolve Talk2DINO's get_template from common repository layouts."""
    candidates = [
        "src.open_vocabulary_segmentation.datasets",
        "src.open_vocabulary_segmentation.datasets.templates",
    ]
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "get_template"):
                return getattr(module, "get_template")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Could not import Talk2DINO get_template.\n"
        + "\n".join(errors)
    )


get_template = _import_get_template()


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
    Load an attribute from either an importable module or a Python file.

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


@torch.no_grad()
def encode_talk2dino_text_bank(
    class_names: Sequence[str],
    clip_model_name: str,
    template_set: str,
    device: str,
    chunk_size: int,
):
    """
    Match Talk2DINO's evaluation text path BEFORE the projector:
      get_template -> template.format(name) -> clip.tokenize
      -> CLIP encode_text -> prompt mean.
    """
    templates = list(get_template(template_set))
    if not templates:
        raise RuntimeError(f"Template set {template_set!r} is empty.")

    token_blocks = []
    for class_name in class_names:
        prompts = [template.format(class_name) for template in templates]
        token_blocks.append(clip.tokenize(prompts))

    tokens = torch.stack(token_blocks)  # [N,T,77]
    n_classes, n_templates = tokens.shape[:2]

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()
    model.requires_grad_(False)

    flat_tokens = tokens.reshape(n_classes * n_templates, -1).to(device)

    encoded = []
    for start in range(0, flat_tokens.shape[0], chunk_size):
        encoded.append(
            model.encode_text(flat_tokens[start:start + chunk_size])
        )

    prompt_features = torch.cat(encoded, dim=0)
    prompt_features = prompt_features.reshape(
        n_classes, n_templates, -1
    )

    # Talk2DINO averages prompts before the learned projector.
    mean_features = prompt_features.mean(dim=1).float().cpu()

    return mean_features, templates


def main():
    parser = argparse.ArgumentParser(
        "Generic Talk2DINO-style CLIP text-bank extractor."
    )

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
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--template_set", default="sub_imagenet_template")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument(
        "--keep_background",
        action="store_true",
        help=(
            "Keep a leading 'background' class. By default it is dropped "
            "to match Talk2DINO VOC-style text encoding."
        ),
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")

    device = (
        args.device
        if torch.cuda.is_available()
        and str(args.device).startswith("cuda")
        else "cpu"
    )

    class_names = resolve_class_names(args)

    features, templates = encode_talk2dino_text_bank(
        class_names=class_names,
        clip_model_name=args.clip_model,
        template_set=args.template_set,
        device=device,
        chunk_size=args.chunk_size,
    )

    if features.ndim != 2 or features.shape[0] != len(class_names):
        raise RuntimeError(
            f"Unexpected feature shape {tuple(features.shape)} "
            f"for {len(class_names)} classes."
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite values found in output features.")

    # Pascal-Part-style names such as "horse's head" are automatically
    # grouped by the possessive prefix. For generic object-only class lists
    # this remains an empty dictionary.
    object_groups = build_object_groups(class_names)

    output = {
        "features": features.contiguous(),       # [N,D], raw prompt mean
        "class_names": list(class_names),
        "object_groups": object_groups,          # object -> feature row indices
        "clip_model": args.clip_model,
        "template_set": args.template_set,
        "num_templates": len(templates),
        "normalized": False,
        "stage": "pre_projector_prompt_mean",
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    norms = features.norm(dim=-1)

    print("============================================================")
    print("Talk2DINO-style CLIP text bank")
    print("============================================================")
    print("output        :", out_path)
    print("classes       :", len(class_names))
    print("object groups :", len(object_groups))
    print("feature shape :", tuple(features.shape))
    print("dtype         :", features.dtype)
    print("CLIP          :", args.clip_model)
    print("template set  :", args.template_set)
    print("templates     :", len(templates))
    print("normalized    : False")
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
