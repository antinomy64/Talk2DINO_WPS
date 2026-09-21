#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict RelProto transform ablation: unconstrained shared Linear transform.

This script deliberately reuses the current repository implementation in
train_relproto_alignemt.py for:
  - raw CLIP bank loading
  - frozen Talk2DINO projector loading
  - predicted-object cache validation
  - relative evidence
  - independent anchors
  - RelProto construction
  - cosine loss
  - optimizer, batching, seed, logging conventions

The ONLY optimization change versus train_relproto_alignemt.py is:

    Orthogonal-W:
        optimizer.step()
        W <- U @ Vh             # SVD / polar retraction

    This ablation:
        optimizer.step()
        # NO SVD retraction

The transform is still one shared 768x768 matrix, initialized to identity,
without bias, and applied on the same side:

    current_text = normalize(T0[pids] @ W)

Thus parameter count, initialization, data, loss and optimization are identical.
Only the structure-preserving orthogonality constraint is removed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_relproto_alignemt as base


VERSION = "predobj_clip_relproto_linear_unconstrained_v1"


def checkpoint_payload(
    *,
    W: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    text_info: dict[str, Any],
    projector_info: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    payload = base.checkpoint_payload(
        W=W,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        history=history,
        args=args,
        text_info=text_info,
        projector_info=projector_info,
        preflight=preflight,
    )
    payload["format"] = VERSION
    payload["version"] = VERSION
    payload["method"]["W"] = (
        "one shared unconstrained 768x768 linear matrix; identity init; "
        "Adam; NO polar/SVD retraction; no bias"
    )
    payload["method"]["ablation"] = (
        "strict orthogonality ablation: all training logic matches "
        "train_relproto_alignemt.py except SVD/polar retraction is removed"
    )
    return payload


def do_train_linear(
    *,
    examples: list[base.Example],
    base_bank_cpu: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    text_info: dict[str, Any],
    projector_info: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    base_bank = base_bank_cpu.to(
        device=device, dtype=torch.float32
    ).detach()
    base_bank = base.normalize_last(base_bank)

    dataset = base.PredObjDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=base.collate_examples,
        drop_last=False,
    )

    # EXACT same parameter count/init as Orthogonal W.
    W = torch.nn.Parameter(
        torch.eye(base.VISION_DIM, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        [W], lr=args.lr, weight_decay=0.0
    )

    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        losses = []
        grad_norms = []
        proto_n = []
        anchor_only = []
        anchor_r = []
        collisions = []
        collision_excess = []

        progress = tqdm(
            loader,
            desc=f"linear epoch {epoch:02d}/{args.epochs:02d}",
            leave=True,
        )

        for batch_examples in progress:
            optimizer.zero_grad(set_to_none=True)

            # EXACT repository forward/loss.
            loss, stats = base.batch_forward(
                base_bank,
                W,
                batch_examples,
                device=device,
                prototype_max_patches=args.prototype_max_patches,
            )

            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise ValueError("missing/nonfinite linear transform gradient")

            grad_norm = float(W.grad.norm().item())
            optimizer.step()

            # ---------------- ONLY ABLATED LINE ----------------
            # Orthogonal baseline calls:
            #     base.orthogonal_retract_(W)
            # Here it is intentionally omitted.
            # ---------------------------------------------------

            if not torch.isfinite(W).all():
                raise ValueError("unconstrained linear transform became nonfinite")

            global_step += 1
            loss_value = float(loss.detach().item())

            losses.append(loss_value)
            grad_norms.append(grad_norm)
            proto_n.append(stats["prototype_mean_patch_count"])
            anchor_only.append(stats["prototype_anchor_only_fraction"])
            anchor_r.append(stats["mean_anchor_relative_score"])
            collisions.append(stats["anchor_collision_annotation_rate"])
            collision_excess.append(stats["mean_anchor_collision_excess"])

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                proto=f"{stats['prototype_mean_patch_count']:.2f}",
                coll=f"{stats['anchor_collision_annotation_rate']:.3f}",
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Same diagnostics as orthogonal run; for Linear these are allowed to move.
        mstats = base.matrix_stats(W.detach(), base_bank)

        # Useful extra conditioning diagnostics for the unconstrained matrix.
        with torch.no_grad():
            sv = torch.linalg.svdvals(W.detach())
            cond = float((sv.max() / sv.min().clamp_min(1e-12)).item())
            sv_min = float(sv.min().item())
            sv_max = float(sv.max().item())

        row = {
            "epoch": epoch,
            "transform": "linear_unconstrained",
            "annotations": len(dataset),
            "batch_size": int(args.batch_size),
            "optimizer_updates": len(loader),
            "global_step": global_step,
            "mean_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "prototype_mean_patch_count": float(np.mean(proto_n)),
            "prototype_anchor_only_fraction": float(np.mean(anchor_only)),
            "mean_anchor_relative_score": float(np.mean(anchor_r)),
            "anchor_collision_annotation_rate": float(np.mean(collisions)),
            "mean_anchor_collision_excess": float(np.mean(collision_excess)),
            "singular_value_min": sv_min,
            "singular_value_max": sv_max,
            "condition_number": cond,
            "seconds": float(time.perf_counter() - t0),
            **mstats,
        }
        history.append(row)
        base.write_csv(history, out_dir / "training_history.csv")

        payload = checkpoint_payload(
            W=W,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            args=args,
            text_info=text_info,
            projector_info=projector_info,
            preflight=preflight,
        )
        base.atomic_torch_save(
            payload, out_dir / f"W_epoch_{epoch:03d}.pt"
        )
        base.atomic_torch_save(
            payload, out_dir / "W_last.pt"
        )

        print(
            f"[linear epoch] {epoch}/{args.epochs} "
            f"loss={row['mean_loss']:.6f} "
            f"struct_delta={row['text_cosine_structure_max_abs']:.4e} "
            f"orth={row['orthogonality_max_abs']:.4e} "
            f"cond={row['condition_number']:.3f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )

    summary = {
        "status": "completed",
        "version": VERSION,
        "transform": "linear_unconstrained",
        "checkpoint": "W_last.pt",
        "epochs": args.epochs,
        "global_step": global_step,
        "history": history,
        "controlled_difference_vs_orthogonal": (
            "SVD/polar retraction removed; all other RelProto training logic reused"
        ),
    }
    base.atomic_json_dump(summary, out_dir / "summary.json")


def train_and_eval(args: argparse.Namespace) -> None:
    root = base.resolve_root(args.project_root)
    dataset_path = base.resolve_path(root, args.train_dataset)
    text_bank_path = base.resolve_path(root, args.text_bank)
    config_path = base.resolve_path(root, args.model_config)
    weight_path = base.resolve_path(root, args.weights)
    out_dir = base.resolve_path(root, args.out_dir)

    for path in (dataset_path, text_bank_path, config_path, weight_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix with existing run: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # EXACT current repository preprocessing path.
    raw_clip, text_names, text_info = base.load_raw_clip_bank(
        text_bank_path,
        expected_clip_model=args.clip_model,
        expected_template=args.template_set,
    )
    projector, projector_info = base.load_frozen_projector(
        project_root=root,
        config_path=config_path,
        weight_path=weight_path,
        device=device,
    )
    projected_bank = base.project_raw_clip_bank(
        raw_clip,
        projector,
        device=device,
        batch_size=args.project_batch_size,
    )
    text_info["projected_shape"] = list(projected_bank.shape)
    text_info["projected_normalized"] = True
    text_info["projector_weight_sha256"] = projector_info["weights_sha256"]

    torch.save(
        {
            "features": projected_bank,
            "classnames": text_names,
            "source_raw_bank": str(text_bank_path),
            "source_raw_bank_sha256": text_info["sha256"],
            "projector_weight": str(weight_path),
            "projector_weight_sha256": projector_info["weights_sha256"],
            "model_config": str(config_path),
            "normalized": True,
            "stage": "post_projector_part_bank",
        },
        out_dir / "projected_part_bank.pt",
    )

    examples, preflight = base.prepare_examples(
        dataset_path,
        text_names=text_names,
        patch_key=args.feature_name,
        foreground_key=args.foreground_key,
        part_id_key=args.part_id_key,
        part_name_key=args.part_name_key,
        max_annotations=args.max_annotations,
        expected_bg_thresh=args.expected_bg_thresh,
        require_pamr=args.require_pamr,
        projector_weight_sha256=projector_info["weights_sha256"],
        require_cache_projector_match=args.require_cache_projector_match,
    )

    rows = preflight.pop("rows")
    base.write_csv(rows, out_dir / "training_samples.csv")
    base.atomic_json_dump(preflight, out_dir / "preflight.json")
    base.atomic_json_dump(text_info, out_dir / "text_bank.json")
    base.atomic_json_dump(projector_info, out_dir / "projector.json")
    base.atomic_json_dump(vars(args), out_dir / "args.json")

    print("=" * 68)
    print("Pred-Obj CLIP RelProto TRANSFORM ABLATION")
    print("=" * 68)
    print("transform        : unconstrained 768x768 linear, no bias")
    print("initialization   : identity")
    print("difference       : NO SVD/polar retraction")
    print("annotations      :", len(examples))
    print("projector        :", weight_path)
    print("projected bank   :", tuple(projected_bank.shape))
    print("RelProto cap     :", args.prototype_max_patches)
    print("=" * 68)

    if args.preflight_only:
        print("[done] preflight_only")
        return

    do_train_linear(
        examples=examples,
        base_bank_cpu=projected_bank,
        args=args,
        out_dir=out_dir,
        text_info=text_info,
        projector_info=projector_info,
        preflight=preflight,
    )


def main(argv=None) -> int:
    args = base.build_parser().parse_args(argv)
    base.validate_args(args)
    if args.self_test:
        base.self_test()
        print("LINEAR_ABLATION_IMPORT_SELF_TEST_PASS")
        return 0
    train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
