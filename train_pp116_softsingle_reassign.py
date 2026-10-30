#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 Ours + fixed-W explicit patch-weight reassignment (experimental).

Training: original Ours A0 -> fixed-W projected descent on A over all predicted
foreground -> detached refined NCE target. The OUTER relational loss still uses
the ORIGINAL differentiable Ours prototype. Encoders/projector stay frozen;
only shared orthogonal W is trained, with the original per-step SVD retraction.

Inner objective per crop:
  inner_nce_weight * mean_part(116-way NCE)
  + inner_rank_weight * mean_valid_triplet(relation hinge)
  + inner_anchor_weight * mean_part(sum_patch((A-A0)^2)).
Only K>=3 crops with valid text-relation triplets are refined. K=1/2 and crops
without valid triplets preserve the exact original target. Original zero-weight
foreground patches can enter by simplex projection; foreground is not expanded.
Three inner steps and coefficients 1/1/1 are starting settings, not validated
optimal hyperparameters. Backtracking checks objective descent, not GT purity.

--inner_steps 0 is the exact Ours loss/gradient control.
--inner_rank_weight 0 isolates NCE-based reassignment on the same eligible crops.
--self_test runs tensor tests with PyTorch only, without official repo or data.
Normal training requires the existing train_relproto_alignemt.py in this folder.
No GT masks, GT prototypes, additional encoders, Hard12, or coherence are used.
Fixed presence-covering probes save BEFORE/AFTER weights for later offline audit.
This script never edits the original trainer/helper/projector. Place it in the
repository root; train from a NEW output directory. No resume is implemented.
"""
from __future__ import annotations

import argparse
import math
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import train_relproto_alignemt as original
except ModuleNotFoundError as exc:
    if exc.name != "train_relproto_alignemt":
        raise
    original = None  # --self_test is standalone; training still requires it.

VERSION = "pp116_softsingle_reassign_v1_1"
CHECKPOINT_FORMAT = "pp116_nightly_softsingle_presence_vs_objectfull_v1"
BASELINE_SOURCE_SHA256 = "62b56d689138a66f397f9b6a2bd4fd84713207e4e154c1461f77781c878fac5c"


def soft_relative_prototypes(
    current_text: torch.Tensor,  # [B,K,D], differentiable, normalized
    tokens: torch.Tensor,        # [B,1024,D], frozen normalized
    foreground: torch.Tensor,    # [B,1024], bool
    *,
    temperature: float,
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
            "weights": weights,  # original differentiable weights; no new computation
            "positive_count": pos_count.detach(),
            "fallback": empty.float().detach(),
            "effective_count": effective.detach(),
            "weighted_margin": expected_margin.detach(),
            "anchor_margin": relative.gather(-1, anchor_idx[..., None]).squeeze(-1).detach(),
        }
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


def project_foreground_simplex(z: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of every part row onto the foreground simplex.

    A zero initial weight does not freeze a coordinate. Non-foreground entries
    are always zero. Subtracting the row maximum improves numeric stability.
    This projection is used without differentiation in the inner optimizer.
    """
    if z.ndim != 3 or foreground.shape != (z.size(0), z.size(2)):
        raise ValueError("expected B x K x N weights and B x N foreground")
    if not bool(foreground.any(-1).all()):
        raise ValueError("empty foreground is not a valid simplex")
    mask = foreground[:, None, :].expand_as(z)
    masked = z.masked_fill(~mask, -torch.inf)
    shifted = masked - masked.amax(-1, keepdim=True)
    ordered = shifted.sort(dim=-1, descending=True).values
    ranks = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    cumulative = ordered.cumsum(-1) - 1.0
    active = torch.isfinite(ordered) & (ordered - cumulative / ranks > 0)
    rho = active.sum(-1, keepdim=True).clamp_min(1)
    theta = cumulative.gather(-1, rho - 1) / rho.to(z.dtype)
    result = (shifted - theta).clamp_min(0).masked_fill(~mask, 0)
    # Remove floating point summation drift; this is not an extra regularizer.
    return result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)


def valid_relation_triplets(local_g, delta):
    K = local_g.size(-1)
    ids = torch.arange(K, device=local_g.device)
    p, q, r = ids[:, None, None], ids[None, :, None], ids[None, None, :]
    distinct = (p != q) & (p != r) & (q != r)
    return distinct[None] & ((local_g[:, :, :, None] - local_g[:, :, None, :]) > delta)


def inner_objective(a, a0, patches, texts, pids, valid, args):
    """One scalar per crop; all text/features are detached by the caller.

    NCE: mean over present parts, with all text classes as competitors.
    Rank: mean over valid local triplets, exactly the outer rank definition.
    Anchor: mean_part(sum_patch((a-a0)^2)); no division by patch count.
    """
    B, K, _ = a.shape
    visual = F.normalize(torch.bmm(a, patches), dim=-1, eps=1e-12)
    logits = visual @ texts.T / args.nce_temperature
    nce = F.cross_entropy(logits.reshape(-1, texts.size(0)), pids.reshape(-1),
                          reduction="none").reshape(B, K).mean(-1)
    relations = visual @ visual.transpose(1, 2)
    differences = relations[:, :, :, None] - relations[:, :, None, :]
    rank = (F.relu(args.rank_margin - differences) * valid.to(a.dtype)).sum((1, 2, 3))
    rank = rank / valid.sum((1, 2, 3)).clamp_min(1)
    anchor = (a - a0).square().sum(-1).mean(-1)
    total = (args.inner_nce_weight * nce + args.inner_rank_weight * rank
             + args.inner_anchor_weight * anchor)
    return total, nce, rank, anchor, visual


def refine_assignment(weights, patches, foreground, all_text, pids, local_g, args):
    """Fixed-W projected gradient descent on A, with per-crop backtracking.

    Only crops with K>=3 and at least one valid relation triplet are eligible,
    including the inner_rank_weight=0 ablation (identical eligible population).
    No optimizer state, GT, persistent A, or higher-order gradient is used.
    A trial is accepted only when its full inner objective does not increase.
    This guarantees an objective check, NOT semantic correctness.
    """
    a0 = weights.detach()
    patches, texts, local_g = patches.detach(), all_text.detach(), local_g.detach()
    B, K, _ = a0.shape
    valid = valid_relation_triplets(local_g, args.rank_delta)
    eligible = valid.flatten(1).any(-1) & (K >= 3)
    if args.inner_steps == 0:
        eligible = torch.zeros_like(eligible)
    accepted_steps = torch.zeros(B, device=a0.device, dtype=a0.dtype)
    with torch.no_grad():
        initial = inner_objective(a0, a0, patches, texts, pids, valid, args)
    a = a0.clone()
    if bool(eligible.any()):
        for _ in range(args.inner_steps):
            with torch.enable_grad():
                variable = a.detach().requires_grad_(True)
                objective = inner_objective(variable, a0, patches, texts, pids, valid, args)[0]
                # Sum, not batch mean: a crop's solution is independent of the
                # number/order of other crops packed into the same K-group.
                gradient = torch.autograd.grad((objective * eligible.to(a.dtype)).sum(),
                                               variable, create_graph=False)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("non-finite inner assignment gradient")
            with torch.no_grad():
                old_value = objective.detach()
                pending = eligible.clone()
                accepted = torch.zeros_like(eligible)
                next_a = a.clone()
                step_size = a.new_full((B, 1, 1), args.inner_lr)
                for _attempt in range(args.inner_backtracks + 1):
                    candidate = project_foreground_simplex(a - step_size * gradient, foreground)
                    value = inner_objective(candidate, a0, patches, texts, pids, valid, args)[0]
                    feasible = (torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
                                & ((candidate.sum(-1) - 1).abs().amax(-1) < 1e-5))
                    take = pending & feasible & torch.isfinite(value) & (value <= old_value)
                    next_a = torch.where(take[:, None, None], candidate, next_a)
                    accepted |= take
                    pending &= ~take
                    if not bool(pending.any()):
                        break
                    step_size = step_size * 0.5
                a = next_a.detach()
                accepted_steps += accepted.to(a.dtype)
                if not bool(accepted.any()):
                    break
    with torch.no_grad():
        final = inner_objective(a, a0, patches, texts, pids, valid, args)
        if not bool(torch.isfinite(final[0]).all()):
            raise FloatingPointError("non-finite refined objective")
        if bool((final[0][eligible] > initial[0][eligible] + 1e-6).any()):
            raise AssertionError("inner objective increased")
        diagnostics = {
            "eligible": eligible,
            "objective_before": initial[0], "objective_after": final[0],
            "nce_before": initial[1], "nce_after": final[1],
            "rank_before": initial[2], "rank_after": final[2],
            "anchor_after": final[3],
            "accepted_steps": accepted_steps,
            "new_mass": (a * (a0 == 0).to(a.dtype)).sum(-1).mean(-1),
            "moved_mass": (0.5 * (a - a0).abs().sum(-1)).mean(-1),
            "prototype_cosine": (initial[4] * final[4]).sum(-1).mean(-1),
            "effective_support": a.square().sum(-1).clamp_min(1e-12).reciprocal().mean(-1),
        }
    return a.detach(), final[4].detach(), diagnostics


REFINE_SCALARS = (
    "objective_before", "objective_after", "nce_before", "nce_after",
    "rank_before", "rank_after", "anchor_after", "accepted_steps",
    "new_mass", "moved_mass", "prototype_cosine", "effective_support",
)


def select_fixed_probe(examples, max_crops):
    """Deterministic part-presence coverage, then cache order; consumes no RNG."""
    selected, covered = [], set()
    for i, example in enumerate(examples):
        present = set(int(x) for x in example.pids.tolist())
        if present - covered:
            selected.append(i)
            covered.update(present)
        if len(selected) >= max_crops:
            break
    selected_set = set(selected)
    for i in range(len(examples)):
        if len(selected) >= max_crops:
            break
        if i not in selected_set:
            selected.append(i)
            selected_set.add(i)
    return sorted(selected)


def save_assignment_probe(examples, indices, bank, W, args, out_dir, epoch):
    """Save paired weights for later GT-only auditing, without reading GT."""
    if not indices:
        return
    records = []
    device = W.device
    with torch.no_grad():
        text = original.normalize_last(bank @ W).detach()
        relations = (bank @ bank.T).detach()
    # Groups/batches keep memory bounded and leave the training RNG untouched.
    groups = defaultdict(list)
    for i in indices:
        groups[int(examples[i].pids.numel())].append(i)
    for K, ids in sorted(groups.items()):
        for start in range(0, len(ids), args.batch_size):
            subset = ids[start:start + args.batch_size]
            with torch.no_grad():
                patches = original.normalize_last(torch.stack([examples[i].tokens for i in subset]).to(device, dtype=torch.float32))
                fg = torch.stack([examples[i].foreground for i in subset]).to(device, dtype=torch.bool)
                pids = torch.stack([examples[i].pids for i in subset]).to(device, dtype=torch.long)
                if K == 1:
                    scores = torch.bmm(text[pids], patches.transpose(1, 2))
                    a0 = torch.softmax(scores.masked_fill(~fg[:, None], -torch.inf) / args.soft_temperature, -1)
                else:
                    _, old_diag = soft_relative_prototypes(text[pids], patches, fg, temperature=args.soft_temperature)
                    a0 = old_diag["weights"]
                local_g = relations[pids.unsqueeze(-1), pids.unsqueeze(-2)]
            refined, _, info = refine_assignment(a0, patches, fg, text, pids, local_g, args)
            for j, index in enumerate(subset):
                example = examples[index]
                records.append({
                    "cache_example_index": index, "row_index": int(example.row_index),
                    "class_name": str(example.class_name), "pids": pids[j].cpu(),
                    "foreground": fg[j].cpu(), "original_weights": a0[j].cpu(),
                    "refined_weights": refined[j].cpu(),
                    "eligible": bool(info["eligible"][j].item()),
                    **{key: float(info[key][j].item()) for key in REFINE_SCALARS},
                })
    destination = out_dir / "assignment_probes"
    destination.mkdir(parents=True, exist_ok=True)
    original.atomic_torch_save({
        "version": VERSION, "epoch": epoch,
        "purpose": "fixed cache rows, weights BEFORE/AFTER inner refinement; NO GT read",
        "selection": "deterministic part-presence coverage then cache order",
        "train_dataset": str(args.train_dataset), "records": records,
    }, destination / f"assignments_epoch_{epoch:03d}.pt")
    print(f"[probe] epoch={epoch} crops={len(records)} saved paired assignments", flush=True)


def refinement_self_test():
    """CPU tensor tests; no cache, projector, GPU, or part-GT required."""
    from types import SimpleNamespace
    print(f"[self_test] version={VERSION} torch={torch.__version__}", flush=True)
    torch.manual_seed(123)
    args = SimpleNamespace(inner_steps=5, inner_lr=0.1, inner_backtracks=8,
                           inner_nce_weight=1.0, inner_rank_weight=1.0,
                           inner_anchor_weight=1.0, rank_delta=0.05,
                           rank_margin=0.05, nce_temperature=0.07)
    torch.set_num_threads(1)
    # Independent exact small-simplex cases, including newly admitted zeros.
    z = torch.tensor([[[0.2, -0.1, 0.4, 100.0], [2.0, -1.0, 0.0, 100.0]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    p = project_foreground_simplex(z, mask)
    expected = torch.tensor([[[11/30, 1/15, 17/30, 0], [1, 0, 0, 0]]], dtype=torch.float64)
    torch.testing.assert_close(p, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(project_foreground_simplex(z + 10000, mask), p, rtol=1e-10, atol=1e-10)
    print("PASS foreground-only simplex and zero-to-positive admission")

    B, K, N, D, C = 2, 3, 9, 12, 8
    text = F.normalize(torch.randn(C, D), dim=-1).requires_grad_(True)
    pids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    patches = F.normalize(torch.randn(B, N, D), dim=-1)
    patches[:, :3] = text.detach()[:3]
    patches.requires_grad_(True)
    fg = torch.ones(B, N, dtype=torch.bool); fg[:, -2:] = False
    a0 = torch.zeros(B, K, N)
    a0[:, 0, 1] = 1; a0[:, 1, 2] = 1; a0[:, 2, 0] = 1
    a0.requires_grad_(True)
    g = (text.detach() @ text.detach().T)[pids.unsqueeze(-1), pids.unsqueeze(-2)]
    refined, visual, info = refine_assignment(a0, patches, fg, text, pids, g, args)
    assert bool(info["eligible"].all())
    assert not refined.requires_grad and not visual.requires_grad
    assert text.grad is None and patches.grad is None and a0.grad is None
    assert bool((refined >= 0).all()) and bool((refined[..., -2:] == 0).all())
    torch.testing.assert_close(refined.sum(-1), torch.ones(B, K), atol=1e-6, rtol=1e-6)
    assert bool((info["objective_after"] < info["objective_before"]).all())
    assert bool((info["new_mass"] > 0).all())
    print("PASS fixed-text refinement, monotonic objective, masks, and stop-gradient")

    # Packing a crop with other crops must not change its refinement.
    first, _, _ = refine_assignment(a0[:1], patches[:1], fg[:1], text, pids[:1], g[:1], args)
    torch.testing.assert_close(first[0], refined[0], atol=2e-6, rtol=2e-6)
    args0 = SimpleNamespace(**vars(args)); args0.inner_steps = 0
    disabled, _, _ = refine_assignment(a0, patches, fg, text, pids, g, args0)
    torch.testing.assert_close(disabled, a0.detach(), rtol=0, atol=0)
    no_rank = torch.eye(K).repeat(B, 1, 1)
    unchanged, _, no_info = refine_assignment(a0, patches, fg, text, pids, no_rank, args)
    torch.testing.assert_close(unchanged, a0.detach(), rtol=0, atol=0)
    assert not bool(no_info["eligible"].any())
    for k in (1, 2):
        unchanged, _, _ = refine_assignment(a0[:, :k], patches, fg, text, pids[:, :k], g[:, :k, :k], args)
        torch.testing.assert_close(unchanged, a0[:, :k].detach(), rtol=0, atol=0)
    print("PASS batch independence, disabled control, K=1/2 and no-triplet fallbacks")

    # Separate finite-difference check of d(inner objective)/dA.
    valid = valid_relation_triplets(g.double(), args.rank_delta)
    av = project_foreground_simplex(torch.rand(B, K, N, dtype=torch.float64), fg).requires_grad_(True)
    fn = lambda x: inner_objective(x, a0.detach().double(), patches.detach().double(),
                                   text.detach().double(), pids, valid, args)[0]
    assert torch.autograd.gradcheck(fn, (av,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print("PASS finite-difference inner-gradient check")
    print("SELF_TEST_PASS_REASSIGN (tensor tests only; real cache/GPU not exercised)")


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
    stat_sum = defaultdict(float)
    part_count = 0
    num_struct_samples = 0
    num_samples = len(examples)
    refine_sums = defaultdict(float)
    refine_count = 0
    refine_changed = 0
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
            weights = torch.softmax(scores / args.soft_temperature, dim=-1)
            visual = F.normalize(torch.bmm(weights, patch), dim=-1, eps=1e-12)
            positives = fg.sum(dim=-1, keepdim=True).float()
            fallback = torch.zeros_like(positives)
            effective = weights.square().sum(-1).clamp_min(1e-12).reciprocal()
            weighted_margin = (weights * absolute).sum(-1)
        else:
            visual, diag = soft_relative_prototypes(
                current_text, patch, fg, temperature=args.soft_temperature,
            )
            positives = diag["positive_count"]
            fallback = diag["fallback"]
            effective = diag["effective_count"]
            weighted_margin = diag["weighted_margin"]

        # Change ONLY the detached NCE target. The relational branch below
        # must continue using the original visual, with its W gradient path.
        nce_target = visual.detach()
        if args.inner_steps > 0 and K >= 3:
            local_g_inner = frozen_relations[pids.unsqueeze(-1), pids.unsqueeze(-2)]
            _, refined_visual, refine_info = refine_assignment(
                diag["weights"], patch, fg, all_text, pids, local_g_inner, args,
            )
            eligible = refine_info["eligible"]
            nce_target = torch.where(eligible[:, None, None], refined_visual, nce_target)
            refine_count += int(eligible.sum().item())
            refine_changed += int((eligible & (refine_info["moved_mass"] > 1e-6)).sum().item())
            for key in REFINE_SCALARS:
                refine_sums[key] += float(refine_info[key][eligible].sum().item())
        logits = torch.matmul(nce_target, all_text.T) / args.nce_temperature
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
        for j, i in enumerate(member_indices):
            loss_parts[i] = total_per_annotation[j]
            loss_nce_parts[i] = nce_per_annotation[j]
            loss_rel_parts[i] = rank_per_annotation[j]
        num_struct_samples += int((struct_active > 0).sum().item())
        part_count += B * K
        with torch.no_grad():
            stat_sum["mean_positive_count"] += float(positives.sum().item()) if K>=2 else 0.0
            stat_sum["fallback_count"] += float(fallback.sum().item()) if K>=2 else 0.0
            stat_sum["effective_patch_count"] += float(effective.sum().item())
            stat_sum["mean_weighted_margin"] += float(weighted_margin.sum().item())
            stat_sum["mean_triplets_per_annotation"] += float(triplet_count.sum().item())
            stat_sum["single_part_count"] += float(B if K == 1 else 0)
            stat_sum["multi_part_occurrences"] += float(B*K if K >= 2 else 0)

    if any(v is None for v in loss_parts):
        raise RuntimeError("missing per-annotation losses")
    assert all(v is not None for v in loss_nce_parts)
    assert all(v is not None for v in loss_rel_parts)
    total_loss = torch.stack([v for v in loss_parts if v is not None]).mean()
    nce_loss = torch.stack([v for v in loss_nce_parts if v is not None]).mean()
    rel_loss = torch.stack([v for v in loss_rel_parts if v is not None]).mean()
    if not torch.isfinite(total_loss):
        raise RuntimeError("nonfinite train loss")

    # Diagnostics: only K>=2 samples have genuine relative margins.
    nparts = max(1, int(stat_sum["multi_part_occurrences"]))
    diag_out = {
        "loss_total": float(total_loss.detach().item()),
        "loss_nce": float(nce_loss.detach().item()),
        "loss_relation": float(rel_loss.detach().item()),
        "structure_active_fraction": float(num_struct_samples / num_samples),
        "mean_triplets_per_annotation": stat_sum["mean_triplets_per_annotation"] / num_samples,
        "mean_positive_count_multi": stat_sum["mean_positive_count"] / nparts,
        "fallback_fraction_multi": stat_sum["fallback_count"] / nparts,
        "mean_effective_patch_count": stat_sum["effective_patch_count"] / part_count,
        "mean_weighted_margin": stat_sum["mean_weighted_margin"] / part_count,
        "single_part_annotation_fraction": stat_sum["single_part_count"] / num_samples,
    }
    diag_out.update({
        "refine_active_fraction": refine_count / num_samples,
        "refine_changed_fraction": refine_changed / num_samples,
        **{"refine_" + key: refine_sums[key] / max(1, refine_count) for key in REFINE_SCALARS},
        "_batch_count": num_samples, "_refine_count": refine_count,
    })
    # Return separated scalars for a first-batch structural gradient sanity check.
    return total_loss, diag_out, {"nce": nce_loss, "relation": rel_loss}


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
    diagnostics_keys = (
        "loss_total", "loss_nce", "loss_relation", "structure_active_fraction",
        "mean_triplets_per_annotation", "mean_positive_count_multi",
        "fallback_fraction_multi", "mean_effective_patch_count", "mean_weighted_margin",
        "single_part_annotation_fraction", "refine_active_fraction", "refine_changed_fraction",
        *("refine_" + key for key in REFINE_SCALARS),
    )
    probe_indices = select_fixed_probe(examples, args.probe_crops) if args.probe_crops > 0 else []
    save_assignment_probe(examples, probe_indices, base_bank, W, args, out_dir, 0)
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
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError("nonfinite or missing gradient of W")
            logs["grad_norm"].append(float(W.grad.norm().item()))
            optimizer.step()
            original.orthogonal_retract_(W)
            global_step += 1
            for k in diagnostics_keys:
                logs[k].append(info[k])
            logs["_batch_count"].append(info["_batch_count"])
            logs["_refine_count"].append(info["_refine_count"])
            progress.set_postfix(nce=f"{info['loss_nce']:.3f}", rank=f"{info['loss_relation']:.3f}",
                                 new=f"{info['refine_new_mass']:.3f}")

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
        # Preserve original loss logging, but give new diagnostics explicit
        # crop-weighted denominators (not a mean of conditional batch means).
        n_annotations = sum(logs["_batch_count"])
        n_refined = sum(logs["_refine_count"])
        for key in ("refine_active_fraction", "refine_changed_fraction"):
            row[key] = sum(v*n for v,n in zip(logs[key], logs["_batch_count"])) / max(1, n_annotations)
        for name in REFINE_SCALARS:
            key = "refine_" + name
            row[key] = sum(v*n for v,n in zip(logs[key], logs["_refine_count"])) / max(1, n_refined)
        row["refine_eligible_crops"] = n_refined
        row["mean_loss"] = row["loss_total"]
        history.append(row)
        original.write_csv(history, out_dir / "training_history.csv")
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight,
        )
        ckpt["format"] = CHECKPOINT_FORMAT  # keep the existing W baker schema
        ckpt["version"] = VERSION
        ckpt["reassignment"] = {
            key: getattr(args, key) for key in (
                "inner_steps", "inner_lr", "inner_backtracks", "inner_nce_weight",
                "inner_rank_weight", "inner_anchor_weight", "probe_crops", "probe_every",
            )
        }
        ckpt["reassignment"].update({"gt_used": False, "outer_relation_target": "original Ours V(W)"})
        ckpt["method"].update({
            "relproto": "A0: original positive-margin soft relative pooling, single fallback; "
                        "K=1 absolute soft pooling. NCE target: A refined over all predicted "
                        "foreground for eligible crops. Outer relation: ORIGINAL A0 prototype.",
            "structure_selection": "outer relation: original Ours V(W); inner refinement: fixed-W "
                                   "projected gradient on A, all predicted foreground candidates",
            "loss": "116-way InfoNCE with detached refined target PLUS original differentiable "
                    "Ours relational loss; equal-annotation mean",
            "inner_objective": "w_nce*NCE + w_rank*rank + w_anchor*mean_part(sum_patch((A-A0)^2))",
            "inner_scope": "K>=3 and valid local text-relation triplets; no GT read",
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
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
              f"active={row['structure_active_fraction']:.3f} "
              f"eff={row['mean_effective_patch_count']:.1f} "
              f"new_mass={row['refine_new_mass']:.4f} "
              f"inner={row['refine_objective_before']:.4f}->{row['refine_objective_after']:.4f} "
              f"orth={row['orthogonality_max_abs']:.2e} "
              f"seconds={row['seconds']:.1f}", flush=True)
        if probe_indices and (epoch % args.probe_every == 0 or epoch == args.epochs):
            save_assignment_probe(examples, probe_indices, base_bank, W, args, out_dir, epoch)
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
        out_dir="final_exp/relproto_alignment/pp116_reassign_v1_s123_30ep",
        prototype_max_patches=8,  # retained CLI compatibility; unused (all-soft)
        expected_bg_thresh=0.55,
        epochs=30, batch_size=16, lr=1e-5, seed=123,
    )
    p.add_argument("--part_setting", choices=("presence", "objectfull"), required=True)
    p.add_argument("--soft_temperature", type=float, default=0.05)
    p.add_argument("--nce_temperature", type=float, default=0.07)
    p.add_argument("--lambda_struct", type=float, default=4.0)
    p.add_argument("--rank_delta", type=float, default=0.05)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--inner_steps", type=int, default=3, help="0 gives exact Ours control")
    p.add_argument("--inner_lr", type=float, default=0.1, help="initial projected-descent step size")
    p.add_argument("--inner_backtracks", type=int, default=5, help="maximum step halvings per inner iteration")
    p.add_argument("--inner_nce_weight", type=float, default=1.0)
    p.add_argument("--inner_rank_weight", type=float, default=1.0, help="0: matched no-inner-relation ablation")
    p.add_argument("--inner_anchor_weight", type=float, default=1.0)
    p.add_argument("--probe_crops", type=int, default=128, help="0 disables fixed paired-assignment exports")
    p.add_argument("--probe_every", type=int, default=10, help="save probes at E0, every N epochs and final")
    return p


def validate_args(args) -> None:
    original.validate_args(args)
    for name in ("soft_temperature", "nce_temperature"):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and >0")
    for name in ("lambda_struct", "rank_delta", "rank_margin"):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and >=0")
    if args.part_setting != "presence":
        raise ValueError("This controlled experiment supports only --part_setting presence")
    if not 0 <= args.inner_steps <= 100 or not 0 <= args.inner_backtracks <= 20:
        raise ValueError("inner_steps must be in 0..100; inner_backtracks in 0..20")
    if not math.isfinite(args.inner_lr) or args.inner_lr <= 0:
        raise ValueError("inner_lr must be finite and >0")
    for key in ("inner_nce_weight", "inner_rank_weight", "inner_anchor_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(key + " must be finite and >=0")
    if args.inner_steps > 0 and args.inner_nce_weight + args.inner_rank_weight <= 0:
        raise ValueError("At least one inner semantic/relational objective must be active")
    if args.probe_crops < 0 or args.probe_every <= 0:
        raise ValueError("probe_crops must be >=0 and probe_every >0")
    if not 1 <= args.prototype_max_patches <= original.DEFAULT_PATCH_COUNT:
        raise ValueError("--prototype_max_patches must be within 1..1024 (single-part control only)")


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


_ORIGINAL_PREPARE_EXAMPLES = original.prepare_examples if original is not None else None
_SETTING = "unset"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self_test" in argv:
        refinement_self_test()
        return 0
    if original is None:
        raise RuntimeError("Place this file beside your existing train_relproto_alignemt.py in the repository root")
    args = build_parser().parse_args(argv)
    validate_args(args)
    run_dir = Path(args.out_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path(args.project_root).expanduser().resolve() / run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Choose a NEW --out_dir; refusing to overwrite {run_dir}")
    global _SETTING
    _SETTING = args.part_setting
    original.prepare_examples = prepare_examples_checked  # strict provenance hook
    original.do_train = do_train  # runtime-only hook, does not alter official source
    original.train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())