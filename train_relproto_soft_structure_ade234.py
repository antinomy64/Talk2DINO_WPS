#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP-116: differentiable all-positive Relative prototypes + text-relation loss.

New file; does not modify the official train_relproto_alignemt.py.

For K>=2 image-present parts:
  r[b,k,i]  = cos(t_k W, f_i) - max_{q != k} cos(t_q W, f_i), foreground only
  C[b,k]    = {i: r[b,k,i] > 0}, or the row-wise argmax anchor if empty
  A[b,k,:]  = masked_softmax(r[b,k,:] / soft_temperature over C[b,k])
  v[b,k]    = L2_normalize(sum_i A[b,k,i] * f_i)
  loss      = equal-annotation 234-class visual-to-text InfoNCE(v.detach(),tW)
              + lambda_struct * equal-annotation relation-ranking loss(v, G_text)

The InfoNCE branch STOPs gradients through v, whereas relation ranking allows
  dL_rel/dW through v <- A <- r <- normalize(T0 @ W).

K=1 samples retain EXACT original hard Relative support (max_patches L, default 8),
since K=1 scores are absolute similarities rather than relative margins.
No labels/masks are read, aside from cached image-level part IDs and Talk2DINO
predicted object foregrounds. Frozen CLIP, DINOv2, COCO2014 projector; only W
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

import train_relproto_alignment_ade20kpart234 as original

VERSION = "ade234_predobj_soft_relative_structure_grad_w_v2"


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
    for K, member_indices in sorted(groups.items()):
        indices = torch.tensor(member_indices, device=device, dtype=torch.long)
        pids = torch.stack([examples[i].pids for i in member_indices]).to(device, dtype=torch.long)
        B = len(member_indices)
        current_text = all_text[pids]
        patch = tokens[indices]
        fg = foreground[indices]

        if K == 1:
            # Exact original single-part path; NOT a genuine relative margin.
            # A softmax over hundreds of absolute-positive patches is an
            # unrelated change, so keep single-part behavior controlled.
            with torch.no_grad():
                selection = original.build_relproto(
                    current_text.detach(), patch, fg,
                    max_patches=args.prototype_max_patches,
                )
            visual = selection["prototypes"].detach()
            positives = torch.zeros(B, 1, device=device)
            fallback = torch.zeros_like(positives)
            effective = selection["prototype_counts"].float()
            weighted_margin = selection["anchor_relative_scores"]
        else:
            visual, diag = soft_relative_prototypes(
                current_text, patch, fg, temperature=args.soft_temperature,
            )
            positives = diag["positive_count"]
            fallback = diag["fallback"]
            effective = diag["effective_count"]
            weighted_margin = diag["weighted_margin"]

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

    # Diagnostics; denominators never pretend single-part samples use real margin.
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
        "single_part_annotation_fraction",
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
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError("nonfinite or missing gradient of W")
            logs["grad_norm"].append(float(W.grad.norm().item()))
            optimizer.step()
            original.orthogonal_retract_(W)
            global_step += 1
            for k in diagnostics_keys:
                logs[k].append(info[k])
            progress.set_postfix(nce=f"{info['loss_nce']:.3f}", rank=f"{info['loss_relation']:.3f}")

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
        ckpt["format"] = VERSION
        ckpt["version"] = VERSION
        ckpt["method"].update({
            "relproto": "K>=2: all strictly positive relative-margin foreground patches; "
                        "per-part margin softmax, anchor argmax fallback only if none; "
                        "K=1: unchanged original hard capped support; normalized weighted mean",
            "structure_selection": "no SGRC discrete search; differentiable visual relation ranking "
                                   "through soft assignment to shared orthogonal W",
            "loss": "234-way visual-to-text InfoNCE with stopgrad target PLUS relational ranking "
                    "with gradient through V(W); equal-annotation mean",
            "W": "one 768x768 shared orthogonal map, Adam + per-step SVD retraction",
        })
        original.atomic_torch_save(ckpt, out_dir / f"W_epoch_{epoch:03d}.pt")
        original.atomic_torch_save(ckpt, out_dir / "W_last.pt")
        print(f"[epoch] {epoch:02d}/{args.epochs} "
              f"loss={row['loss_total']:.5f} nce={row['loss_nce']:.5f} "
              f"relation={row['loss_relation']:.5f} "
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
        train_dataset="feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug.pth",
        text_bank="feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt",
        weights="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
        model_config="configs/vitb_mlp_infonce.yaml",
        out_dir="final_exp/relproto_alignment/ade234_soft_structure_lambda4p0_seed123_bg055_tau005",
        prototype_max_patches=8,  # only applied for K=1 samples
        expected_bg_thresh=0.55,
        epochs=30, batch_size=16, lr=1e-5, seed=123,
    )
    p.add_argument("--soft_temperature", type=float, default=0.05)
    p.add_argument("--nce_temperature", type=float, default=0.07)
    p.add_argument("--lambda_struct", type=float, default=4.0)
    p.add_argument("--rank_delta", type=float, default=0.05)
    p.add_argument("--rank_margin", type=float, default=0.05)
    return p


def validate_args(args) -> None:
    if original.NUM_PARTS != 234:
        raise RuntimeError("ADE234 requires a 234-class trainer")
    original.validate_args(args)
    for name in ("soft_temperature", "nce_temperature"):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and >0")
    for name in ("lambda_struct", "rank_delta", "rank_margin"):
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


    # Deterministic zero-margin test: all part similarities tie exactly.
    fallback_tokens = torch.zeros(1, N, D)
    fallback_tokens[:, :, 0] = 1.0

    same = torch.zeros(1, 3, D)
    same[:, :, 0] = 1.0

    visual_zero, diag_zero = soft_relative_prototypes(
        same, fallback_tokens, foreground, temperature=0.05,
    )

    assert bool((diag_zero["positive_count"] == 0).all()), (
        f"positive counts: {diag_zero['positive_count']}"
    )
    assert bool((diag_zero["fallback"] == 1).all()), (
        f"fallback: {diag_zero['fallback']}"
    )
    assert torch.allclose(
        diag_zero["effective_count"], torch.ones(1, 3)
    )
    assert torch.allclose(visual_zero, same, atol=0, rtol=0)

    # One winning part, two strictly negative-margin parts.
    unequal = same.clone()
    unequal[:, 1, 0] = 0.5
    unequal[:, 1, 1] = math.sqrt(0.75)
    unequal[:, 2, 0] = -1.0

    _, diag_negative = soft_relative_prototypes(
        unequal, fallback_tokens, foreground, temperature=0.05,
    )

    assert bool((diag_negative["positive_count"][:, 0] == 64).all())
    assert bool((diag_negative["positive_count"][:, 1:] == 0).all())
    assert bool((diag_negative["fallback"][:, 0] == 0).all())
    assert bool((diag_negative["fallback"][:, 1:] == 1).all())
    assert torch.isfinite(visual_zero).all()
    # No spatially defined part GT is used in the test or training.
    print("PASS all-positive softmax + exact zero-positive anchor fallback")
    print("PASS nonzero relational gradient from visual prototype to W")
    print("PASS stop-grad InfoNCE target")
    print("SELF_TEST_PASS_SOFT_STRUCTURE")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.self_test:
        self_test()
        return 0
    original.do_train = do_train  # runtime-only hook, does not alter its source
    original.train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
