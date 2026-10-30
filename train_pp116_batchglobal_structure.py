#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 Ours + batch-global visual Relation Loss (strict Ours derivative).

RUN FROM REPOSITORY ROOT:
  python -u train_pp116_batchglobal_structure.py --part_setting presence ...

Untouched/reused from train_pp116_softsingle_dual.py:
 - GT-free predicted object Crop and presence example loading/preflight
 - frozen COCO Talk2DINO projector / CLIP bank / DINOv2 token features
 - original K>=2 positive relative margin / argmax fallback Softmax
 - original K=1 absolute foreground Softmax pooling
 - original *global 116-way* visual-stopgradient InfoNCE
 - original within-Crop Relation Loss and Adam+SVD on W only

Only additional learning objective: cross-Crop visual relationship ordering
against globally frozen initial text relationships. Both losses retain the
original per-annotation reduction. Pure alpha=0 dispatches EXACTLY to the
official Ours batch_forward to enable numerical regression checks.

No GT part/pixel/box masks are used in training.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_relproto_alignemt as original
import train_pp116_softsingle_dual as ours
from pp116_batchglobal_relation import cross_crop_rank_loss

VERSION = "pp116_ours_batchglobal_visualrelation_20261030_v1"


EXTRA_KEYS = (
    "loss_local_relation", "loss_global_relation", "global_structure_active_fraction",
    "mean_crosscrop_triplets_per_anchor", "global_crossobject_triplet_fraction",
)
BASE_KEYS = (
    "loss_total", "loss_nce", "loss_relation", "structure_active_fraction",
    "mean_triplets_per_annotation", "mean_positive_count_multi",
    "fallback_fraction_multi", "mean_effective_patch_count", "mean_weighted_margin",
    "single_part_annotation_fraction",
)


def batch_forward(
    base_bank: torch.Tensor,
    W: torch.Tensor,
    examples: list[Any],
    *,
    device: torch.device,
    frozen_relations: torch.Tensor,
    args,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    alpha = float(args.global_structure_alpha)
    # Zero-alpha MUST be identical to the official Ours, including its
    # precise arithmetic, losses and active-sample metrics.
    if alpha == 0 or args.lambda_struct == 0:
        result, info, components = ours.batch_forward(
            base_bank, W, examples, device=device,
            frozen_relations=frozen_relations, args=args,
        )
        info = dict(info)
        info.update({
            "loss_local_relation": info["loss_relation"],
            "loss_global_relation": 0.0,
            "global_structure_active_fraction": 0.0,
            "mean_crosscrop_triplets_per_anchor": 0.0,
            "global_crossobject_triplet_fraction": 0.0,
        })
        return result, info, components

    tokens = torch.stack([e.tokens for e in examples], dim=0).to(device, dtype=torch.float32)
    foreground = torch.stack([e.foreground for e in examples], dim=0).to(device, dtype=torch.bool)
    tokens = original.normalize_last(tokens)
    all_text = original.normalize_last(base_bank @ W)  # differentiable W
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[int(ex.pids.numel())].append(i)

    B_total = len(examples)
    nce_parts: list[torch.Tensor | None] = [None] * B_total
    local_parts: list[torch.Tensor | None] = [None] * B_total
    local_active_parts: list[torch.Tensor | None] = [None] * B_total
    local_triplets_parts: list[torch.Tensor | None] = [None] * B_total

    visual_collected: list[torch.Tensor] = []
    pids_collected: list[torch.Tensor] = []
    crops_collected: list[torch.Tensor] = []
    objects_collected: list[torch.Tensor] = []
    batch_object_names = {x: i for i, x in enumerate(sorted({e.class_name for e in examples}))}
    stat_sum: defaultdict[str, float] = defaultdict(float)
    part_count = 0

    for K, member_indices in sorted(groups.items()):
        indices = torch.tensor(member_indices, device=device, dtype=torch.long)
        pids = torch.stack([examples[i].pids for i in member_indices]).to(device, dtype=torch.long)
        B = len(member_indices)
        current_text = all_text[pids]
        patch = tokens[indices]
        fg = foreground[indices]
        if K == 1:
            # EXACTLY Ours' Soft Absolute single-candidate path.
            absolute = torch.bmm(current_text, patch.transpose(1, 2))
            scores = absolute.masked_fill(~fg[:, None, :], -torch.inf)
            weights = torch.softmax(scores / args.soft_temperature, dim=-1)
            visual = F.normalize(torch.bmm(weights, patch), dim=-1, eps=1e-12)
            positives = fg.sum(dim=-1, keepdim=True).float()
            fallback = torch.zeros_like(positives)
            effective = weights.square().sum(-1).clamp_min(1e-12).reciprocal()
            weighted_margin = (weights * absolute).sum(-1)
        else:
            visual, diag = ours.soft_relative_prototypes(
                current_text, patch, fg, temperature=args.soft_temperature,
            )
            positives = diag["positive_count"]
            fallback = diag["fallback"]
            effective = diag["effective_count"]
            weighted_margin = diag["weighted_margin"]

        # ORIGINAL GLOBAL 116-class InfoNCE; completely detached visual target.
        logits = torch.matmul(visual.detach(), all_text.T) / args.nce_temperature
        part_losses = F.cross_entropy(
            logits.reshape(-1, original.NUM_PARTS),
            pids.reshape(-1), reduction="none",
        ).reshape(B, K)
        nce_per_annotation = part_losses.mean(dim=1)

        if K >= 3 and args.lambda_struct > 0:
            local_g = frozen_relations[pids.unsqueeze(-1), pids.unsqueeze(-2)]
            local_rank, local_active, local_ntrip = ours.relational_rank_loss(
                visual, local_g, delta=args.rank_delta, margin=args.rank_margin,
            )
        else:
            local_rank = nce_per_annotation.new_zeros((B,))
            local_active = nce_per_annotation.new_zeros((B,))
            local_ntrip = nce_per_annotation.new_zeros((B,))

        for j, idx in enumerate(member_indices):
            nce_parts[idx] = nce_per_annotation[j]
            local_parts[idx] = local_rank[j]
            local_active_parts[idx] = local_active[j]
            local_triplets_parts[idx] = local_ntrip[j]

        # Preserve all visual prototypes WITH gradient graph for cross-crop ranking.
        visual_collected.append(visual.reshape(B * K, visual.size(-1)))
        pids_collected.append(pids.reshape(-1))
        crops_collected.append(indices.repeat_interleave(K))
        object_codes = torch.tensor(
            [batch_object_names[examples[i].class_name] for i in member_indices],
            device=device, dtype=torch.long,
        )
        objects_collected.append(object_codes.repeat_interleave(K))
        part_count += B * K
        with torch.no_grad():
            stat_sum["mean_positive_count"] += float(positives.sum().item()) if K >= 2 else 0.0
            stat_sum["fallback_count"] += float(fallback.sum().item()) if K >= 2 else 0.0
            stat_sum["effective_patch_count"] += float(effective.sum().item())
            stat_sum["mean_weighted_margin"] += float(weighted_margin.sum().item())
            stat_sum["single_part_count"] += float(B if K == 1 else 0)
            stat_sum["multi_part_occurrences"] += float(B * K if K >= 2 else 0)

    if any(v is None for v in nce_parts + local_parts + local_active_parts + local_triplets_parts):
        raise RuntimeError("missing per-annotation losses")

    visual_global = torch.cat(visual_collected, dim=0)
    ids_global = torch.cat(pids_collected, dim=0)
    crops_global = torch.cat(crops_collected, dim=0)
    objs_global = torch.cat(objects_collected, dim=0)
    g = cross_crop_rank_loss(
        visual_global, ids_global, crops_global, objs_global,
        frozen_relations, n_crops=B_total,
        delta=args.rank_delta, margin=args.rank_margin,
        class_balance=args.global_class_balance,
    )
    nce_ann = torch.stack([v for v in nce_parts if v is not None])
    local_ann = torch.stack([v for v in local_parts if v is not None])
    local_active = torch.stack([v for v in local_active_parts if v is not None])
    local_triplets = torch.stack([v for v in local_triplets_parts if v is not None])
    global_ann = g.loss_per_crop
    relation_ann = (1.0 - alpha) * local_ann + alpha * global_ann
    total = (nce_ann + args.lambda_struct * relation_ann).mean()
    nce = nce_ann.mean()
    relation = relation_ann.mean()
    local = local_ann.mean()
    global_r = global_ann.mean()
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite training loss")

    num_struct_active = (
        ((local_active > 0) & ((1.0 - alpha) > 0)) |
        ((g.active_fraction_per_crop > 0) & (alpha > 0))
    ).float().mean()
    nparts = max(1, int(stat_sum["multi_part_occurrences"]))
    info = {
        "loss_total": float(total.detach().item()),
        "loss_nce": float(nce.detach().item()),
        "loss_relation": float(relation.detach().item()),
        "loss_local_relation": float(local.detach().item()),
        "loss_global_relation": float(global_r.detach().item()),
        "structure_active_fraction": float(num_struct_active.item()),
        "global_structure_active_fraction": float((g.active_fraction_per_crop > 0).float().mean().item()),
        "mean_triplets_per_annotation": float(local_triplets.sum().item()) / B_total,
        "mean_crosscrop_triplets_per_anchor": float(g.triplets_per_crop.mean().item()),
        "global_crossobject_triplet_fraction": float(g.cross_object_fraction.item()),
        "mean_positive_count_multi": stat_sum["mean_positive_count"] / nparts,
        "fallback_fraction_multi": stat_sum["fallback_count"] / nparts,
        "mean_effective_patch_count": stat_sum["effective_patch_count"] / part_count,
        "mean_weighted_margin": stat_sum["mean_weighted_margin"] / part_count,
        "single_part_annotation_fraction": stat_sum["single_part_count"] / B_total,
    }
    return total, info, {
        "nce": nce, "relation": relation,
        "local_relation": local, "global_relation": global_r,
    }


def do_train(*, examples, base_bank_cpu, args, out_dir, text_info, projector_info, preflight):
    """Same official optimizer/DataLoader/checkpoint flow; extra loss logging."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    original.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    base_bank = original.normalize_last(base_bank_cpu.to(device=device, dtype=torch.float32).detach())
    frozen_relations = (base_bank @ base_bank.T).detach()
    loader = DataLoader(
        original.PredObjDataset(examples), batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=original.collate_examples,
    )
    W = torch.nn.Parameter(torch.eye(original.VISION_DIM, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    global_step = 0
    gradient_path_checked = False
    extra_gradient_checked = False
    diagnostics_keys = BASE_KEYS + EXTRA_KEYS
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        logs = defaultdict(list)
        progress = tqdm(loader, desc=f"epoch {epoch:02d}/{args.epochs:02d}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss, info, separate = batch_forward(
                base_bank, W, batch, device=device,
                frozen_relations=frozen_relations, args=args,
            )
            if (not gradient_path_checked and args.lambda_struct > 0 and
                    info["structure_active_fraction"] > 0):
                if not separate["relation"].requires_grad:
                    raise RuntimeError("relation loss disconnected from W")
                g_struct = torch.autograd.grad(
                    separate["relation"], W, retain_graph=True, allow_unused=True,
                )[0]
                if g_struct is None or not torch.isfinite(g_struct).all():
                    raise RuntimeError("invalid relation gradient")
                print(f"[check] relation_to_W_grad_norm={g_struct.norm().item():.6g} "
                      f"active={info['structure_active_fraction']:.3f}", flush=True)
                gradient_path_checked = True
            if (not extra_gradient_checked and args.lambda_struct > 0 and
                    args.global_structure_alpha > 0 and
                    info["global_structure_active_fraction"] > 0):
                g_global = torch.autograd.grad(
                    separate["global_relation"], W, retain_graph=True, allow_unused=True,
                )[0]
                if g_global is None or not torch.isfinite(g_global).all():
                    raise RuntimeError("global relation gradient disconnected")
                print(f"[check] global_relation_to_W_grad_norm={g_global.norm().item():.6g}", flush=True)
                extra_gradient_checked = True
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError("nonfinite or missing W gradient")
            logs["grad_norm"].append(float(W.grad.norm().item()))
            optimizer.step()
            original.orthogonal_retract_(W)
            global_step += 1
            for key in diagnostics_keys:
                logs[key].append(info[key])
            progress.set_postfix(nce=f"{info['loss_nce']:.3f}",
                                 local=f"{info['loss_local_relation']:.3f}",
                                 global_rel=f"{info['loss_global_relation']:.3f}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        matrix_info = original.matrix_stats(W.detach(), base_bank)
        if matrix_info["orthogonality_max_abs"] > 1e-3:
            raise AssertionError(f"W orthogonality failed: {matrix_info}")
        row = {
            "epoch": epoch,
            "annotations": len(examples), "optimizer_updates": len(loader),
            "global_step": global_step,
            "mean_grad_norm": float(np.mean(logs["grad_norm"])),
            **{k: float(np.mean(logs[k])) for k in diagnostics_keys},
            "seconds": float(time.perf_counter() - tick),
            **matrix_info,
            "global_structure_alpha": args.global_structure_alpha,
        }
        row["mean_loss"] = row["loss_total"]
        history.append(row)
        original.write_csv(history, out_dir / "training_history.csv")
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight,
        )
        ckpt["format"] = VERSION
        ckpt["version"] = VERSION
        ckpt["method"].update({
            "relproto": "identical to Ours train_pp116_softsingle_dual.py (positive-margin soft; fallback; K=1 absolute soft)",
            "loss": "116-way InfoNCE stopgrad visual target + 4*((1-alpha)*LocalRelation + alpha*CrossCropRelation)",
            "structure_selection": "local Ours ranking + cross-crop visual ranking; frozen global text relation; gradient through V(W)",
            "crosscrop_relation": "distinct part-class triplets, >=1 cross-crop comparison; per-anchor then per-annotation reduction",
            "class_balanced_global_pairs": bool(args.global_class_balance),
            "global_structure_alpha": float(args.global_structure_alpha),
            "part_supervision": args.part_setting,
            "candidate_source": "image-specific part presence" if args.part_setting == "presence" else "full object taxonomy",
            "W": "one 768x768 shared orthogonal map, Adam + per-step SVD retraction",
        })
        ckpt["experiment_arm"] = args.part_setting
        original.atomic_torch_save(ckpt, out_dir / f"W_epoch_{epoch:03d}.pt")
        original.atomic_torch_save(ckpt, out_dir / "W_last.pt")
        print(f"[epoch] {epoch:02d}/{args.epochs} "
              f"loss={row['loss_total']:.5f} nce={row['loss_nce']:.5f} "
              f"relation={row['loss_relation']:.5f} "
              f"local={row['loss_local_relation']:.5f} "
              f"global={row['loss_global_relation']:.5f} "
              f"active={row['structure_active_fraction']:.3f} "
              f"global_active={row['global_structure_active_fraction']:.3f} "
              f"eff={row['mean_effective_patch_count']:.1f} "
              f"orth={row['orthogonality_max_abs']:.2e} "
              f"seconds={row['seconds']:.1f}", flush=True)
    original.atomic_json_dump({
        "status": "completed", "version": VERSION,
        "checkpoint": "W_last.pt", "epochs": args.epochs,
        "global_step": global_step, "history": history,
    }, out_dir / "summary.json")


def build_parser():
    p = ours.build_parser()
    p.description = __doc__
    p.set_defaults(out_dir="final_exp/relproto_alignment/pp116_batchglobal_alpha025_l4_100ep")
    p.add_argument("--global_structure_alpha", type=float, default=0.25,
                   help="Structure mixture: (1-alpha)*Ours Local + alpha*CrossCrop")
    p.add_argument("--global_class_balance", dest="global_class_balance",
                   action="store_true", default=True,
                   help="Balance repeated part classes in q/r comparisons (default: true)")
    p.add_argument("--no_global_class_balance", dest="global_class_balance",
                   action="store_false")
    return p


def validate_args(args):
    ours.validate_args(args)
    if not math.isfinite(args.global_structure_alpha) or not 0 <= args.global_structure_alpha <= 1:
        raise ValueError("--global_structure_alpha must be in [0,1]")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.self_test:
        # This includes all original Soft Relative checks.
        ours.self_test()
        from verify_batchglobal_structure import check_relation_unit_tests
        check_relation_unit_tests()
        return 0
    ours._SETTING = args.part_setting
    original.prepare_examples = ours.prepare_examples_checked
    original.do_train = do_train
    original.train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
