#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 Soft Relative + Structure + within-part patch coherence.

Based on the user's train_pp116_softsingle_dual(2).py. The ONLY added training
objective is lambda_coherence * mean_crop(mean_candidate_part(L_coherence)).
For the existing soft assignment a and frozen, normalized patch tokens f:
  L_coherence = sum_{i != j} a_i a_j (1 - f_i dot f_j)
                / sum_{i != j} a_i a_j.
One-patch supports (including fallback) contribute zero. The implementation
excludes self pairs and avoids an N x N patch matrix and near-one-hot 0/0.
Gradients flow through the soft assignment to W; frozen tokens do NOT change.
This is a consistency regularizer, not a guarantee of semantic correctness.
--lambda_coherence 0 uses the original loss path exactly.

New file; does not modify the official train_relproto_alignemt.py.

For K>=2 candidate parts (image-present OR full-object taxonomy):
  r[b,k,i]  = cos(t_k W, f_i) - max_{q != k} cos(t_q W, f_i), foreground only
  C[b,k]    = {i: r[b,k,i] > 0}, or the row-wise argmax anchor if empty
  A[b,k,:]  = masked_softmax(r[b,k,:] / soft_temperature over C[b,k])
  v[b,k]    = L2_normalize(sum_i A[b,k,i] * f_i)
  loss      = equal-annotation 116-class visual-to-text InfoNCE(v.detach(),tW)
              + lambda_struct * equal-annotation relation-ranking loss(v, G_text)
              + lambda_coherence * equal-annotation within-part patch coherence

The InfoNCE branch STOPs gradients through v, whereas relation ranking allows
  dL_rel/dW through v <- A <- r <- normalize(T0 @ W).

K=1 uses SOFT absolute-score pooling over ALL foreground patches (no Top-L).
--part_setting presence: original weak-part-presence cache, with label-selected parts.
--part_setting objectfull: independently extracted object-only cache, requires strict
provenance marker and ALL part classes of the known object; NEVER part GT masks.
Both modes share identical loss, optimizer, W, hyperparameters, and pooling. Frozen CLIP, DINOv2, COCO2014 projector; only W
is optimized by Adam followed by SVD (polar) retraction.

This is a research *experiment*, not an assertion that soft prototypes improve
accuracy. Use --lambda_struct 0 for a matched Soft Relative control.
"""
from __future__ import annotations

import argparse
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

VERSION = "pp116_softsingle_patchcoherence_v1"
# Same checkpoint schema as the supplied baseline; retain the format identifier.
CHECKPOINT_FORMAT = "pp116_nightly_softsingle_presence_vs_objectfull_v1"


def patch_coherence_loss(
    weights: torch.Tensor,       # [B,K,N], the existing softmax weights (NO detach)
    masked_logits: torch.Tensor, # [B,K,N], same scores / temperature, -inf outside support
    tokens: torch.Tensor,        # [B,N,D], frozen unit tokens
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Conditional weighted cosine distance of two DISTINCT patch tokens.

    Directly evaluating (1 - ||sum a_i f_i||^2) / (1 - sum a_i^2) suffers
    cancellation near a one-hot assignment. Instead split off a maximum-weight
    anchor m. Let r=sum_{i!=m} a_i and u_i=a_i/r for the remaining tokens:

      numerator/r = 2*a_m*(1-f_m dot sum u_i f_i)
                    + r*(1-||sum u_i f_i||^2)
      denominator/r = 2*a_m + r*(1-sum u_i^2).

    Compute u by softmax of NON-anchor logits, so it remains well-defined even
    when float32 rounds a_m to 1 or underflows the other original weights.
    This implements the continuous limit at numerical saturation. No threshold
    discards a highly concentrated but multi-patch support. The discrete anchor
    choice is an algebraic decomposition; the value is anchor-independent.
    Complexity: O(B*K*N*D), no pairwise N x N allocation.
    """
    if weights.shape != masked_logits.shape or weights.ndim != 3:
        raise ValueError("coherence expects matching B x K x N weights/logits")
    if tokens.ndim != 3 or tokens.size(0) != weights.size(0) or tokens.size(1) != weights.size(2):
        raise ValueError("coherence tokens must be B x N x D")
    support = torch.isfinite(masked_logits)
    counts = support.sum(-1)
    if not bool((counts >= 1).all()):
        raise ValueError("coherence received an empty support")
    active = counts >= 2
    anchor_idx = masked_logits.argmax(-1)
    anchor_hot = F.one_hot(anchor_idx, weights.size(-1)).bool()
    other_logits = masked_logits.masked_fill(anchor_hot, -torch.inf)
    # Inactive one-token rows need a safe dummy softmax; their loss is exactly 0.
    other_logits = torch.where(active[..., None], other_logits, torch.zeros_like(other_logits))
    other_weights = torch.softmax(other_logits, dim=-1)
    other_mean = torch.bmm(other_weights, tokens)
    anchor = tokens.gather(1, anchor_idx[..., None].expand(-1, -1, tokens.size(-1)))
    anchor_weight = weights.gather(-1, anchor_idx[..., None]).squeeze(-1)
    rest_mass = weights.masked_fill(anchor_hot, 0.0).sum(-1)
    anchor_distance = (1.0 - (anchor * other_mean).sum(-1)).clamp(0.0, 2.0)
    other_distance = (1.0 - other_mean.square().sum(-1)).clamp(0.0, 2.0)
    other_pair_mass = (1.0 - other_weights.square().sum(-1)).clamp_min(0.0)
    numerator = 2.0 * anchor_weight * anchor_distance + rest_mass * other_distance
    denominator = 2.0 * anchor_weight + rest_mass * other_pair_mass
    # denominator >= 2/N because anchor_weight is a maximum softmax weight.
    per_part = torch.where(active, numerator / denominator, torch.zeros_like(numerator))
    if not bool(torch.isfinite(per_part).all()):
        raise RuntimeError("nonfinite patch coherence loss")
    with torch.no_grad():
        info = {
            "active": active.float(),
            "pair_cosine": torch.where(active, 1.0 - per_part, torch.zeros_like(per_part)),
            "pair_mass": (rest_mass * denominator).detach(),
        }
    return per_part, info


def soft_relative_prototypes(
    current_text: torch.Tensor,  # [B,K,D], differentiable, normalized
    tokens: torch.Tensor,        # [B,1024,D], frozen normalized
    foreground: torch.Tensor,    # [B,1024], bool
    *,
    temperature: float,
    return_assignment: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """K>=2 only; all positive-margin patches, no Top-K.

    The >0 eligibility and argmax fallback are discrete/non-differentiable;
    conditional on that mask, weights, pooling and relation loss are differentiable.
    """
    if current_text.ndim != 3 or current_text.size(1) < 2:
        raise ValueError("soft_relative_prototypes requires B x K>=2 x D")
    B, K, D = current_text.shape
    if tokens.shape != (B, original.DEFAULT_PATCH_COUNT, D):
        raise ValueError("unexpected tokens shape")
    if foreground.shape != (B, original.DEFAULT_PATCH_COUNT):
        raise ValueError("unexpected foreground shape")
    if not bool(foreground.any(dim=-1).all()):
        raise ValueError("empty predicted foreground")

    absolute = torch.bmm(current_text, tokens.transpose(1, 2))
    # Reuse the exact original Relative Competition definition and foreground mask.
    relative = original.relative_scores(absolute, foreground)
    positives = relative > 0.0
    pos_count = positives.sum(dim=-1)  # [B,K]
    anchor_idx = relative.argmax(dim=-1)  # [B,K], same as original
    empty = pos_count == 0

    # Each part with zero strictly-positive margin gets ONLY its original argmax.
    # This is an eligibility fallback, not a learned extra candidate.
    anchor_hot = F.one_hot(anchor_idx, relative.size(-1)).to(torch.bool)
    candidates = positives | (anchor_hot & empty[..., None])
    if not bool(candidates.any(dim=-1).all()):
        raise AssertionError("empty candidate row after fallback")

    logits = (relative / temperature).masked_fill(~candidates, -torch.inf)
    weights = torch.softmax(logits, dim=-1)
    # Softmax is well-defined: each row has >=1 finite candidate.
    if not torch.isfinite(weights).all():
        raise RuntimeError("invalid soft assignment weights")

    # Normalize unit patch features AFTER weighted pooling, just like original.
    unnormalized = torch.bmm(weights, tokens)
    prototypes = F.normalize(unnormalized, dim=-1, eps=1e-12)
    if not torch.isfinite(prototypes).all():
        raise RuntimeError("nonfinite soft visual prototype")

    # Diagnostics should not add graph connections to the training loss.
    with torch.no_grad():
        effective = weights.square().sum(dim=-1).clamp_min(1e-12).reciprocal()
        expected_margin = (weights * relative.masked_fill(~candidates, 0)).sum(-1)
        info = {
            "positive_count": pos_count.detach(),
            "fallback": empty.float().detach(),
            "effective_count": effective.detach(),
            "weighted_margin": expected_margin.detach(),
            "anchor_margin": relative.gather(-1, anchor_idx[..., None]).squeeze(-1).detach(),
        }
    if return_assignment:
        # These two tensors intentionally retain their graph for coherence.
        info["assignment"] = weights
        info["assignment_logits"] = logits
    return prototypes, info


def relational_rank_loss(
    prototypes: torch.Tensor,    # [B,K,D] differentiable
    local_text_rel: torch.Tensor, # [B,K,K] frozen
    *,
    delta: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-annotation relational ranking loss and active triplet indicators.

    Eligible triples have distinct p, q, r and Gt[p,q]-Gt[p,r]>delta.
    Only the VISUAL relation matrix is differentiable. Each annotation is
    normalized by its own valid triplet count (or zero when none exist).
    """
    B, K, _ = prototypes.shape
    if local_text_rel.shape != (B, K, K):
        raise ValueError("local text relationship shape mismatch")
    if K < 3:
        return (
            prototypes.sum(dim=(1, 2)) * 0.0,
            torch.zeros(B, device=prototypes.device, dtype=prototypes.dtype),
            torch.zeros(B, device=prototypes.device, dtype=prototypes.dtype),
        )
    with torch.no_grad():
        ids = torch.arange(K, device=prototypes.device)
        p = ids[:, None, None]
        q = ids[None, :, None]
        r = ids[None, None, :]
        distinct = (p != q) & (p != r) & (q != r)
        target_differences = (
            local_text_rel[:, :, :, None] - local_text_rel[:, :, None, :]
        )
        valid = distinct[None] & (target_differences > delta)
        ntrips = valid.sum(dim=(1, 2, 3))

    visual_rel = torch.bmm(prototypes, prototypes.transpose(1, 2))
    visual_differences = visual_rel[:, :, :, None] - visual_rel[:, :, None, :]
    hinges = F.relu(margin - visual_differences)
    per_annotation = (hinges * valid.to(hinges.dtype)).sum((1, 2, 3)) / ntrips.clamp_min(1)
    return per_annotation, (ntrips > 0).float(), ntrips.float()


def batch_forward(
    base_bank: torch.Tensor,
    W: torch.Tensor,
    examples: list[Any],
    *,
    device: torch.device,
    frozen_relations: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Same equal-annotation treatment as the original/SGRC trainer."""
    tokens = torch.stack([e.tokens for e in examples], dim=0).to(device, dtype=torch.float32)
    foreground = torch.stack([e.foreground for e in examples], dim=0).to(device, dtype=torch.bool)
    tokens = original.normalize_last(tokens)

    all_text = original.normalize_last(base_bank @ W)  # NO detach
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[int(ex.pids.numel())].append(i)

    loss_parts: list[torch.Tensor | None] = [None] * len(examples)
    loss_nce_parts: list[torch.Tensor | None] = [None] * len(examples)
    loss_rel_parts: list[torch.Tensor | None] = [None] * len(examples)
    loss_coh_parts: list[torch.Tensor | None] = [None] * len(examples)
    stat_sum = defaultdict(float)
    part_count = 0
    num_struct_samples = 0
    num_samples = len(examples)
    for K, member_indices in sorted(groups.items()):
        indices = torch.tensor(member_indices, device=device, dtype=torch.long)
        pids = torch.stack([examples[i].pids for i in member_indices]).to(device, dtype=torch.long)
        B = len(member_indices)
        current_text = all_text[pids]
        patch = tokens[indices]
        fg = foreground[indices]

        if K == 1:
            # K=1 in either supervision mode: no competing part in this sample.
            # Keep SOFT pooling, and use absolute cosine similarity as scores.
            absolute = torch.bmm(current_text, patch.transpose(1, 2))
            scores = absolute.masked_fill(~fg[:, None, :], -torch.inf)
            assignment_logits = scores / args.soft_temperature
            weights = torch.softmax(assignment_logits, dim=-1)
            visual = F.normalize(torch.bmm(weights, patch), dim=-1, eps=1e-12)
            positives = fg.sum(dim=-1, keepdim=True).float()
            fallback = torch.zeros_like(positives)
            effective = weights.square().sum(-1).clamp_min(1e-12).reciprocal()
            weighted_margin = (weights * absolute).sum(-1)
        else:
            visual, diag = soft_relative_prototypes(
                current_text, patch, fg, temperature=args.soft_temperature,
                return_assignment=args.lambda_coherence > 0,
            )
            positives = diag["positive_count"]
            fallback = diag["fallback"]
            effective = diag["effective_count"]
            weighted_margin = diag["weighted_margin"]
            if args.lambda_coherence > 0:
                weights = diag["assignment"]
                assignment_logits = diag["assignment_logits"]

        # The only InfoNCE target: completely STOP the pseudo visual target.
        logits = torch.matmul(visual.detach(), all_text.T) / args.nce_temperature
        part_losses = F.cross_entropy(
            logits.reshape(-1, original.NUM_PARTS),
            pids.reshape(-1), reduction="none",
        ).reshape(B, K)
        nce_per_annotation = part_losses.mean(dim=1)

        if K >= 3 and args.lambda_struct > 0:
            local_g = frozen_relations[pids.unsqueeze(-1), pids.unsqueeze(-2)]
            rank_per_annotation, struct_active, triplet_count = relational_rank_loss(
                visual, local_g, delta=args.rank_delta, margin=args.rank_margin,
            )
        else:
            rank_per_annotation = nce_per_annotation.new_zeros((B,))
            struct_active = nce_per_annotation.new_zeros((B,))
            triplet_count = nce_per_annotation.new_zeros((B,))

        total_per_annotation = nce_per_annotation + args.lambda_struct * rank_per_annotation
        if args.lambda_coherence > 0:
            coh_per_part, coh_info = patch_coherence_loss(weights, assignment_logits, patch)
            # Same denominator as NCE: average ALL candidate parts in each crop,
            # counting one-token supports as zero; then average crops equally.
            coh_per_annotation = coh_per_part.mean(dim=1)
            total_per_annotation = total_per_annotation + args.lambda_coherence * coh_per_annotation
            with torch.no_grad():
                stat_sum["coherence_active_parts"] += float(coh_info["active"].sum().item())
                stat_sum["coherence_pair_cosine"] += float(coh_info["pair_cosine"].sum().item())
                stat_sum["coherence_pair_mass"] += float(coh_info["pair_mass"].sum().item())
        else:
            # No zero-times-extra-graph contribution: preserve the baseline path.
            coh_per_annotation = nce_per_annotation.new_zeros((B,))
        for j, i in enumerate(member_indices):
            loss_parts[i] = total_per_annotation[j]
            loss_nce_parts[i] = nce_per_annotation[j]
            loss_rel_parts[i] = rank_per_annotation[j]
            loss_coh_parts[i] = coh_per_annotation[j]
        num_struct_samples += int((struct_active > 0).sum().item())
        part_count += B * K
        with torch.no_grad():
            stat_sum["mean_positive_count"] += float(positives.sum().item()) if K>=2 else 0.0
            stat_sum["fallback_count"] += float(fallback.sum().item()) if K>=2 else 0.0
            stat_sum["effective_patch_count"] += float(effective.sum().item())
            stat_sum["near_single_patch_count"] += float((effective < 1.1).sum().item())
            stat_sum["mean_weighted_margin"] += float(weighted_margin.sum().item())
            stat_sum["mean_triplets_per_annotation"] += float(triplet_count.sum().item())
            stat_sum["single_part_count"] += float(B if K == 1 else 0)
            stat_sum["multi_part_occurrences"] += float(B*K if K >= 2 else 0)

    if any(v is None for v in loss_parts):
        raise RuntimeError("missing per-annotation losses")
    assert all(v is not None for v in loss_nce_parts)
    assert all(v is not None for v in loss_rel_parts)
    assert all(v is not None for v in loss_coh_parts)
    total_loss = torch.stack([v for v in loss_parts if v is not None]).mean()
    nce_loss = torch.stack([v for v in loss_nce_parts if v is not None]).mean()
    rel_loss = torch.stack([v for v in loss_rel_parts if v is not None]).mean()
    coh_loss = torch.stack([v for v in loss_coh_parts if v is not None]).mean()
    if not torch.isfinite(total_loss):
        raise RuntimeError("nonfinite train loss")

    # Diagnostics: only K>=2 samples have genuine relative margins.
    nparts = max(1, int(stat_sum["multi_part_occurrences"]))
    diag_out = {
        "loss_total": float(total_loss.detach().item()),
        "loss_nce": float(nce_loss.detach().item()),
        "loss_relation": float(rel_loss.detach().item()),
        "loss_coherence": float(coh_loss.detach().item()),
        "loss_coherence_weighted": float(args.lambda_coherence * coh_loss.detach().item()),
        "coherence_enabled": float(args.lambda_coherence > 0),
        "coherence_active_part_fraction": stat_sum["coherence_active_parts"] / part_count,
        "coherence_mean_pair_cosine": stat_sum["coherence_pair_cosine"] / max(1, stat_sum["coherence_active_parts"]),
        "coherence_mean_pair_mass": stat_sum["coherence_pair_mass"] / part_count,
        "near_single_patch_fraction": stat_sum["near_single_patch_count"] / part_count,
        "structure_active_fraction": float(num_struct_samples / num_samples),
        "mean_triplets_per_annotation": stat_sum["mean_triplets_per_annotation"] / num_samples,
        "mean_positive_count_multi": stat_sum["mean_positive_count"] / nparts,
        "fallback_fraction_multi": stat_sum["fallback_count"] / nparts,
        "mean_effective_patch_count": stat_sum["effective_patch_count"] / part_count,
        "mean_weighted_margin": stat_sum["mean_weighted_margin"] / part_count,
        "single_part_annotation_fraction": stat_sum["single_part_count"] / num_samples,
    }
    # Return separated scalars for a first-batch structural gradient sanity check.
    return total_loss, diag_out, {"nce": nce_loss, "relation": rel_loss, "coherence": coh_loss}


def do_train(*, examples, base_bank_cpu, args, out_dir, text_info, projector_info, preflight):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    original.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    base_bank = original.normalize_last(base_bank_cpu.to(device=device, dtype=torch.float32).detach())
    frozen_relations = (base_bank @ base_bank.T).detach()  # NOT current adapted text
    loader = DataLoader(
        original.PredObjDataset(examples),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=original.collate_examples,
    )
    W = torch.nn.Parameter(torch.eye(original.VISION_DIM, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    global_step = 0
    gradient_path_checked = False
    coherence_path_checked = False
    diagnostics_keys = (
        "loss_total", "loss_nce", "loss_relation", "structure_active_fraction",
        "mean_triplets_per_annotation", "mean_positive_count_multi",
        "fallback_fraction_multi", "mean_effective_patch_count", "mean_weighted_margin",
        "single_part_annotation_fraction",
        "loss_coherence", "loss_coherence_weighted", "coherence_enabled",
        "coherence_active_part_fraction", "coherence_mean_pair_cosine",
        "coherence_mean_pair_mass", "near_single_patch_fraction",
    )
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        logs = defaultdict(list)
        progress = tqdm(loader, desc=f"epoch {epoch:02d}/{args.epochs:02d}")
        for examples_batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss, info, separate = batch_forward(
                base_bank, W, examples_batch, device=device,
                frozen_relations=frozen_relations, args=args,
            )
            if (not gradient_path_checked and args.lambda_struct > 0
                    and info["structure_active_fraction"] > 0):
                # Once per run: confirm relation loss actually connects to W.
                if not separate["relation"].requires_grad:
                    raise RuntimeError("relation loss has no autograd path to W")
                g_struct = torch.autograd.grad(
                    separate["relation"], W, retain_graph=True, allow_unused=True,
                )[0]
                if g_struct is None or not torch.isfinite(g_struct).all():
                    raise RuntimeError("relation loss disconnected from W or nonfinite")
                print(f"[check] relation_to_W_grad_norm={g_struct.norm().item():.6g} "
                      f"active={info['structure_active_fraction']:.3f}", flush=True)
                gradient_path_checked = True
            if (not coherence_path_checked and args.lambda_coherence > 0
                    and info["coherence_active_part_fraction"] > 0):
                if not separate["coherence"].requires_grad:
                    raise RuntimeError("coherence loss has no autograd path to W")
                g_coh = torch.autograd.grad(
                    separate["coherence"], W, retain_graph=True, allow_unused=True,
                )[0]
                if g_coh is None or not torch.isfinite(g_coh).all():
                    raise RuntimeError("coherence loss disconnected from W or nonfinite")
                # Exactly two supported frozen tokens have a constant pair distance;
                # a zero gradient is mathematically valid, so report rather than fail.
                print(f"[check] coherence_to_W_grad_norm={g_coh.norm().item():.6g} "
                      f"weighted_grad_norm={args.lambda_coherence * g_coh.norm().item():.6g} "
                      f"active={info['coherence_active_part_fraction']:.3f}", flush=True)
                coherence_path_checked = True
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError("nonfinite or missing gradient of W")
            logs["grad_norm"].append(float(W.grad.norm().item()))
            optimizer.step()
            original.orthogonal_retract_(W)
            global_step += 1
            for k in diagnostics_keys:
                logs[k].append(info[k])
            progress.set_postfix(nce=f"{info['loss_nce']:.3f}", rank=f"{info['loss_relation']:.3f}",
                                 coh=f"{info['loss_coherence']:.3f}")

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
        }
        row["mean_loss"] = row["loss_total"]
        history.append(row)
        original.write_csv(history, out_dir / "training_history.csv")
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight,
        )
        ckpt["format"] = CHECKPOINT_FORMAT
        ckpt["version"] = VERSION
        ckpt["method"].update({
            "relproto": "K>=2: all positive-margin foreground patches, fallback argmax if empty; "
                        "K=1: absolute-score foreground softmax; normalized soft pooling; "
                        "candidate parts determined by --part_setting (presence or objectfull)",
            "structure_selection": "no SGRC discrete search; differentiable visual relation ranking "
                                   "through soft assignment to shared orthogonal W",
            "loss": "116-way visual-to-text InfoNCE with stopgrad target PLUS relational ranking "
                    "with gradient through V(W) PLUS weighted off-diagonal patch coherence; "
                    "equal-annotation mean; same for both arms",
            "patch_coherence": "weighted mean cosine distance between distinct frozen tokens "
                               "in each existing part support; gradients through soft assignments; "
                               "singleton/fallback=0; lambda_coherence=" + str(args.lambda_coherence),
            "part_supervision": args.part_setting,
            "candidate_source": ("image-specific present part classes" if args.part_setting == "presence"
                                 else "ALL taxonomy part classes under known object"),
            "W": "one 768x768 shared orthogonal map, Adam + per-step SVD retraction",
        })
        ckpt["experiment_arm"] = args.part_setting
        original.atomic_torch_save(ckpt, out_dir / f"W_epoch_{epoch:03d}.pt")
        original.atomic_torch_save(ckpt, out_dir / "W_last.pt")
        print(f"[epoch] {epoch:02d}/{args.epochs} "
              f"loss={row['loss_total']:.5f} nce={row['loss_nce']:.5f} "
              f"relation={row['loss_relation']:.5f} "
              f"coherence={row['loss_coherence']:.5f} "
              f"paircos={row['coherence_mean_pair_cosine']:.4f} "
              f"singleton={row['near_single_patch_fraction']:.3f} "
              f"active={row['structure_active_fraction']:.3f} "
              f"eff={row['mean_effective_patch_count']:.1f} "
              f"orth={row['orthogonality_max_abs']:.2e} "
              f"seconds={row['seconds']:.1f}", flush=True)
    original.atomic_json_dump({
        "status": "completed", "version": VERSION, "checkpoint": "W_last.pt",
        "epochs": args.epochs, "global_step": global_step, "history": history,
    }, out_dir / "summary.json")


def build_parser() -> argparse.ArgumentParser:
    p = original.build_parser()
    p.description = __doc__
    # Overrides only defaults: all entry points and validation remain official.
    p.set_defaults(
        train_dataset="feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth",
        text_bank="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt",
        weights="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
        model_config="configs/vitb_mlp_infonce.yaml",
        out_dir="final_exp/relproto_alignment/pp116_softrelative_structure_patchcoherence_l01_100ep",
        prototype_max_patches=8,  # retained CLI compatibility; unused (all-soft)
        expected_bg_thresh=0.55,
        epochs=100, batch_size=16, lr=1e-5, seed=123,
    )
    p.add_argument("--part_setting", choices=("presence", "objectfull"), required=True)
    p.add_argument("--soft_temperature", type=float, default=0.05)
    p.add_argument("--nce_temperature", type=float, default=0.07)
    p.add_argument("--lambda_struct", type=float, default=4.0)
    p.add_argument("--lambda_coherence", type=float, default=0.1,
                   help="Weight of within-part distinct-patch cosine consistency; 0 = original Ours")
    p.add_argument("--rank_delta", type=float, default=0.05)
    p.add_argument("--rank_margin", type=float, default=0.05)
    return p


def validate_args(args) -> None:
    original.validate_args(args)
    for name in ("soft_temperature", "nce_temperature"):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and >0")
    for name in ("lambda_struct", "lambda_coherence", "rank_delta", "rank_margin"):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and >=0")
    if not 1 <= args.prototype_max_patches <= original.DEFAULT_PATCH_COUNT:
        raise ValueError("--prototype_max_patches must be within 1..1024 (single-part control only)")


def self_test() -> None:
    """Runs without data or GPU, but requires original train_relproto_alignemt.py."""
    original.self_test()
    torch.manual_seed(123)
    D, N = original.VISION_DIM, original.DEFAULT_PATCH_COUNT
    # Simple 3-part geometry with explicit nonzero relational gradient.
    base = F.normalize(torch.randn(original.NUM_PARTS, D), dim=-1)
    W = torch.nn.Parameter(torch.eye(D))
    pids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    tokens = F.normalize(torch.randn(1, N, D), dim=-1)
    foreground = torch.zeros(1, N, dtype=torch.bool)
    foreground[:, :64] = True
    text = F.normalize(base @ W, dim=-1)
    visual, stats = soft_relative_prototypes(
        text[pids], tokens, foreground, temperature=0.05,
    )
    assert visual.shape == (1, 3, D)
    assert bool((stats["positive_count"] >= 0).all())
    local_text_rel = (base @ base.T)[pids.unsqueeze(-1), pids.unsqueeze(-2)].detach()
    # delta < 0? no: pick delta=0 and high hinge margin => active hinges.
    loss_rel, active, _ = relational_rank_loss(
        visual, local_text_rel, delta=0.0, margin=1.0,
    )
    if not bool(active.any()):
        raise AssertionError("self-test expected valid part triplets")
    g = torch.autograd.grad(loss_rel.mean(), W, retain_graph=True, allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all()
    assert float(g.norm()) > 1e-9, "structure loss must backprop to W"
    logits = (visual.detach() @ text.T / 0.07).reshape(-1, original.NUM_PARTS)
    loss_nce = F.cross_entropy(logits, pids.view(-1))
    ng = torch.autograd.grad(loss_nce, W, allow_unused=True)[0]
    assert ng is not None and torch.isfinite(ng).all()
    # Verify no gradient can leak through detached pseudo targets.
    v_only = visual.detach().requires_grad_(True)
    logits_detached = (v_only.detach() @ text.detach().T / 0.07).reshape(-1, original.NUM_PARTS)
    loss = F.cross_entropy(logits_detached, pids.view(-1))
    assert not loss.requires_grad, "detach must block pseudo-target grad"

    # Deterministic tied scores. A random normalized vector repeated three
    # times can yield tiny positive margins on some BLAS/GPU backends;
    # since eligibility is strictly R>0, such a test is not reproducible.
    # Construct exact one-hot unit vectors so every foreground dot product
    # equals 1, every competing score ties exactly, and every margin is 0.
    fallback_tokens = torch.zeros(1, N, D)
    fallback_tokens[:, :, 0] = 1.0
    same = torch.zeros(1, 3, D)
    same[:, :, 0] = 1.0
    visual_zero, diag_zero = soft_relative_prototypes(
        same, fallback_tokens, foreground, temperature=0.05,
    )
    assert bool((diag_zero["positive_count"] == 0).all()), (
        f"exact tie produced positive candidates: {diag_zero['positive_count']}"
    )
    assert bool((diag_zero["fallback"] == 1).all()), (
        f"exact tie failed anchor fallback: {diag_zero['fallback']}"
    )
    assert torch.allclose(diag_zero["effective_count"], torch.ones(1, 3))
    assert torch.allclose(visual_zero, same, atol=0, rtol=0)

    # Also check the non-tie case: one part is the unique winner, the
    # other two parts have NEGATIVE margins on every foreground token.
    unequal = same.clone()
    unequal[:, 1, 0] = 0.5
    unequal[:, 1, 1] = math.sqrt(0.75)  # unit vector, score=0.5
    unequal[:, 2, 0] = -1.0  # unit vector, score=-1
    _, diag_negative = soft_relative_prototypes(
        unequal, fallback_tokens, foreground, temperature=0.05,
    )
    assert bool((diag_negative["positive_count"][:, 0] == 64).all())
    assert bool((diag_negative["positive_count"][:, 1:] == 0).all())
    assert bool((diag_negative["fallback"][:, 0] == 0).all())
    assert bool((diag_negative["fallback"][:, 1:] == 1).all())
    assert torch.isfinite(visual_zero).all()
    # No spatially defined part GT is used in the test or training.
    # Explicit single-part soft pooling sanity test: no hard selection.
    single = F.normalize(base[0][None, None, :], dim=-1)
    one_abs = torch.bmm(single, tokens.transpose(1, 2))
    one_alpha = torch.softmax(
        one_abs.masked_fill(~foreground[:, None], -torch.inf) / 0.05,
        dim=-1,
    )
    assert torch.isfinite(one_alpha).all()
    assert torch.allclose(one_alpha.sum(-1), torch.ones(1, 1), atol=1e-6)
    assert bool((one_alpha[..., 64:] == 0).all())
    print("PASS all-positive softmax + exact-tie/negative-margin anchor fallback")
    print("PASS nonzero relational gradient from visual prototype to W")
    print("PASS stop-grad InfoNCE target")
    print("SELF_TEST_PASS_SOFT_STRUCTURE")
    coh_tokens = F.normalize(torch.randn(1, 7, 5), dim=-1)
    coh_logits = torch.randn(1, 2, 7, requires_grad=True)
    coh, _ = patch_coherence_loss(coh_logits.softmax(-1), coh_logits, coh_tokens)
    coh_grad = torch.autograd.grad(coh.mean(), coh_logits)[0]
    assert torch.isfinite(coh_grad).all() and coh_grad.norm().item() > 1e-9
    print("SELF_TEST_PASS_PATCH_COHERENCE")


def prepare_examples_checked(*args, **kwargs):
    """Hard provenance and taxonomy checks for both arms (no data rewriting)."""
    examples, report = _ORIGINAL_PREPARE_EXAMPLES(*args, **kwargs)
    meta = report.get("cache_meta", {})
    names = kwargs.get("text_names")
    if not names or len(names) != original.NUM_PARTS:
        raise RuntimeError("Missing or incomplete 116-way text taxonomy")
    taxonomy = defaultdict(list)
    for pid, name in enumerate(names):
        if "'s " not in name:
            raise ValueError(f"Invalid part class name: {name!r}")
        taxonomy[name.split("'s ", 1)[0]].append(pid)

    if _SETTING == "objectfull":
        # A presence-derived cache, even with labels overwritten, would be contaminated
        # because the upstream extraction discards objects with no part presence.
        if (meta.get("part_supervision") != "none"
                or meta.get("format") != "pascalpart116_objectlabels_only_full_object_taxonomy_v1"
                or "fixed 116-way taxonomy" not in str(meta.get("part_candidate_source", ""))):
            raise RuntimeError(
                "OBJECTFULL cannot use a cache built with part GT/presence. "
                "Regenerate with this package's object-only extractor."
            )
    else:
        if (meta.get("part_supervision") == "none"
                or meta.get("format") != "pascalpart116_weaklabels_predobj_cropaug_v1"):
            raise RuntimeError("PRESENCE requires the original weak-part-presence cache")

    counts = defaultdict(int)
    for ex in examples:
        expected = taxonomy.get(ex.class_name, [])
        actual = [int(v) for v in ex.pids.tolist()]
        if not expected or not actual or len(actual) != len(set(actual)):
            raise RuntimeError(f"Invalid candidate list: row={ex.row_index}, class={ex.class_name}")
        if _SETTING == "objectfull":
            if actual != expected:
                raise RuntimeError(
                    f"OBJECTFULL row {ex.row_index} {ex.class_name!r}: "
                    f"pids={actual}, full taxonomy={expected}"
                )
        elif not set(actual).issubset(set(expected)):
            raise RuntimeError(f"PRESENCE has foreign part IDs: {ex.class_name}: {actual}")
        counts["single_part" if len(actual) == 1 else "multi_part"] += 1
    report["experiment_arm"] = _SETTING
    report["part_candidate_source"] = (
        "image-level present parts from prior weak label cache" if _SETTING == "presence"
        else "object taxonomy only; NO image-specific part presence read"
    )
    report["candidate_counts"] = dict(counts)
    print(f"[AUDIT] arm={_SETTING} examples={len(examples)} "
          f"single={counts['single_part']} multi={counts['multi_part']} "
          f"cache_format={meta.get('format')}", flush=True)
    return examples, report


_ORIGINAL_PREPARE_EXAMPLES = original.prepare_examples
_SETTING = "unset"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.self_test:
        self_test()
        return 0
    global _SETTING
    _SETTING = args.part_setting
    original.prepare_examples = prepare_examples_checked  # strict provenance hook
    original.do_train = do_train  # runtime-only hook, does not alter official source
    original.train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
