#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PascalPart-116 SGRC + orthogonal W + visual-to-text InfoNCE.

Designed to live alongside the *unchanged* train_relproto_alignemt.py.
Reuses its exact cache reader, frozen text projector, original Relative selector,
Adam + SVD retraction, output directory guards, and checkpoint conventions.

SGRC selection (no gradient):
  - independently anchored Relative prototypes (anchor never swapped);
  - preserve the original count of strictly-positive supplementary supports;
  - choose new positive candidates among top `candidate_k` evidence patches;
  - coordinate greedy swaps maximize mean Relative score minus lambda *
    cross-part prototype relation ranking hinge loss;
  - rank triplets are derived from the frozen projected text bank.
Training (gradient):
  - stop-gradient visual prototypes;
  - 116-way visual-to-text InfoNCE (or original 1-cos control);
  - train only the shared 768x768 orthogonal W.
No GT part masks, image reading, or pretrained model weights are changed.
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

VERSION = "predobj_sgrc_orth_infonce_v1"


def relation_triplets(text_relation: torch.Tensor, delta: float):
    """Triples (p,q,r) for R_text[p,q] > R_text[p,r] + delta."""
    num = text_relation.size(0)
    if num < 3:
        return None
    ids = torch.arange(num, device=text_relation.device)
    p = ids[:, None, None]
    q = ids[None, :, None]
    r = ids[None, None, :]
    diffs = text_relation[:, :, None] - text_relation[:, None, :]
    mask = (p != q) & (p != r) & (q != r) & (diffs > delta)
    triples = torch.where(mask)
    return triples if triples[0].numel() else None


def ranking_loss_for_relations(relations: torch.Tensor, triplets, margin: float):
    """Relations [..., P, P] -> vector of hinge losses [...]."""
    if triplets is None:
        return relations.new_zeros(relations.shape[:-2])
    p, q, r = triplets
    return F.relu(margin - relations[..., p, q] + relations[..., p, r]).mean(dim=-1)


@torch.no_grad()
def refine_one_annotation(
    features: torch.Tensor,     # [1024, 768], normalized
    relative: torch.Tensor,     # [P, 1024], foreground masked to -inf
    support: torch.Tensor,      # [P, L], -1 padded, index 0 is own anchor
    text_relation: torch.Tensor,  # [P,P] frozen relations for this annotation
    *,
    candidate_k: int,
    lam: float,
    delta: float,
    margin: float,
    rounds: int,
    improvement_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Vectorized exhaustive *single* swaps for each part, greedy coordinate steps.

    Never swaps/removes an anchor. Does not force-fill supports or impose
    uniqueness between distinct parts. Uses no backprop through selection.
    """
    device = features.device
    p_count, max_l = support.shape
    supports = support.clone()
    valid = supports >= 0
    counts = valid.sum(dim=1)
    if not bool((counts >= 1).all()):
        raise RuntimeError("Every part must include its anchor")

    idx = supports.clamp_min(0)
    sums = (features[idx] * valid[..., None]).sum(dim=1)
    protos = F.normalize(sums, dim=-1)
    selected_rel = relative.gather(1, idx).masked_fill(~valid, 0)
    per_part_mean = selected_rel.sum(dim=1) / counts
    evidence = per_part_mean.mean()
    triplets = relation_triplets(text_relation, delta)
    initial_relations = protos @ protos.T
    initial_rank = ranking_loss_for_relations(initial_relations, triplets, margin)
    start_rank = float(initial_rank.item())
    start_evidence = float(evidence.item())
    swaps = 0

    if lam > 0 and triplets is not None and rounds > 0:
        # Precompute bounded strictly-positive supplementary candidates.
        candidates: list[torch.Tensor] = []
        for part in range(p_count):
            eligible = relative[part].clone()
            eligible[supports[part, 0]] = -torch.inf
            eligible = eligible.masked_fill(eligible <= 0, -torch.inf)
            kk = min(max(candidate_k - 1, 0), eligible.numel())
            if kk:
                values, locations = torch.topk(eligible, kk, sorted=True)
                candidates.append(locations[torch.isfinite(values)])
            else:
                candidates.append(torch.empty(0, dtype=torch.long, device=device))

        for _ in range(rounds):
            any_swap = False
            for part in range(p_count):
                n_sel = int(counts[part].item())
                # No valid supplementary support -> cannot swap anything.
                if n_sel < 2:
                    continue
                old = supports[part, 1:n_sel]  # anchor is immutable
                candidate = candidates[part]
                if candidate.numel() == 0:
                    continue
                # Exclude currently selected positions (for this part only).
                candidate = candidate[~torch.isin(candidate, supports[part, :n_sel])]
                if candidate.numel() == 0:
                    continue

                old_rep = old[:, None].expand(-1, candidate.numel()).reshape(-1)
                new_rep = candidate[None, :].expand(old.numel(), -1).reshape(-1)
                if not old_rep.numel():
                    continue

                # Test all old->new replacements together on GPU.
                candidate_sums = sums[part][None] - features[old_rep] + features[new_rep]
                candidate_protos = F.normalize(candidate_sums, dim=-1)
                new_similarity = candidate_protos @ protos.T  # [trials, P]
                candidate_relations = (
                    protos @ protos.T
                ).unsqueeze(0).expand(old_rep.numel(), -1, -1).clone()
                candidate_relations[:, part, :] = new_similarity
                candidate_relations[:, :, part] = new_similarity
                candidate_relations[:, part, part] = 1.0
                new_rank = ranking_loss_for_relations(
                    candidate_relations, triplets, margin
                )

                change_evidence = (
                    relative[part, new_rep] - relative[part, old_rep]
                ) / (n_sel * p_count)
                gain = change_evidence - lam * (new_rank - initial_rank)
                best_gain, best_index = gain.max(dim=0)
                if float(best_gain.item()) <= improvement_eps:
                    continue
                j = int(best_index.item())
                old_patch = int(old_rep[j].item())
                new_patch = int(new_rep[j].item())
                slot_matches = (supports[part, :n_sel] == old_patch).nonzero()
                if slot_matches.numel() != 1:
                    raise AssertionError("Ambiguous selected supplementary patch")
                slot = int(slot_matches[0].item())
                assert slot > 0, "Anchor must never change"
                supports[part, slot] = new_patch
                sums[part] = candidate_sums[j]
                protos[part] = candidate_protos[j]
                evidence += change_evidence[j]
                initial_rank = new_rank[j]
                swaps += 1
                any_swap = True
            if not any_swap:
                break

    return supports, protos, {
        "swaps": float(swaps),
        "rank_before": start_rank,
        "rank_after": float(initial_rank.item()),
        "evidence_before": start_evidence,
        "evidence_after": float(evidence.item()),
        "structure_active": float(triplets is not None),
        "triplets": float(0 if triplets is None else triplets[0].numel()),
    }


@torch.no_grad()
def select_prototypes(
    current_text: torch.Tensor,
    tokens: torch.Tensor,
    foreground: torch.Tensor,
    pids: torch.Tensor,
    text_relation_bank: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """Start with the EXACT original Relative selector, optionally refine."""
    selected = original.build_relproto(
        current_text.detach(), tokens, foreground,
        max_patches=args.prototype_max_patches,
    )
    n = tokens.size(0)
    diagnostics: list[dict[str, float]] = []
    if args.selector == "relative":
        return selected, diagnostics

    absolute = torch.bmm(current_text.detach(), tokens.transpose(1, 2))
    relative = original.relative_scores(absolute, foreground)
    new_support = selected["support_indices"].clone()
    new_protos = selected["prototypes"].clone()

    for row in range(n):
        local_relation = text_relation_bank[pids[row]][:, pids[row]]
        support, proto, info = refine_one_annotation(
            tokens[row], relative[row], new_support[row], local_relation,
            candidate_k=args.candidate_k,
            lam=args.lambda_struct,
            delta=args.rank_delta,
            margin=args.rank_margin,
            rounds=args.swap_rounds,
            improvement_eps=args.improvement_eps,
        )
        new_support[row] = support
        new_protos[row] = proto
        diagnostics.append(info)
    if not torch.equal(new_support[:, :, 0], selected["anchor_indices"]):
        raise AssertionError("Structure selection altered an anchor")
    if not torch.equal(new_support.ge(0), selected["support_indices"].ge(0)):
        raise AssertionError("Structure selection changed support cardinality")
    selected["prototypes"] = new_protos.detach()
    selected["support_indices"] = new_support.detach()
    selected["support_relative_scores"] = (
        relative.gather(2, new_support.clamp_min(0))
        .masked_fill(new_support < 0, 0)
        .detach()
    )
    return selected, diagnostics


def batch_forward(
    base_bank: torch.Tensor,
    W: torch.Tensor,
    examples: list[Any],
    *,
    device: torch.device,
    text_relation_bank: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = torch.stack([e.tokens for e in examples], dim=0).to(device, dtype=torch.float32)
    foreground = torch.stack([e.foreground for e in examples], dim=0).to(device, dtype=torch.bool)
    tokens = original.normalize_last(tokens)
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[int(ex.pids.numel())].append(i)

    # Keep the differentiable all-class bank for InfoNCE; do not detach W.
    all_text = original.normalize_last(base_bank @ W)
    losses: list[torch.Tensor | None] = [None] * len(examples)
    proto_counts = []
    anchor_scores = []
    coll_ann = []
    coll_excess = []
    extra: list[dict[str, float]] = []

    for k, members in sorted(groups.items()):
        index = torch.tensor(members, dtype=torch.long, device=device)
        pids = torch.stack([examples[i].pids for i in members]).to(device=device, dtype=torch.long)
        current_text = all_text[pids]
        selected, diagnostics = select_prototypes(
            current_text, tokens[index], foreground[index], pids,
            text_relation_bank, args,
        )
        target = selected["prototypes"].detach()
        if args.loss_type == "infonce":
            logits = torch.matmul(target, all_text.T) / args.nce_temperature
            n, present_k, classes = logits.shape
            per_part_loss = F.cross_entropy(
                logits.reshape(n * present_k, classes),
                pids.reshape(-1), reduction="none",
            ).reshape(n, present_k)
        else:
            per_part_loss = 1.0 - (target * current_text).sum(dim=-1)
        loss_vec = per_part_loss.mean(dim=1)
        for row, member in enumerate(members):
            losses[member] = loss_vec[row]

        proto_counts.append(selected["prototype_counts"].float().reshape(-1))
        anchor_scores.append(selected["anchor_relative_scores"].reshape(-1))
        coll_ann.append(selected["collision_annotation"])
        coll_excess.append(selected["collision_excess"])
        extra.extend(diagnostics)

    if any(v is None for v in losses):
        raise AssertionError("Missing annotation loss")
    loss = torch.stack([v for v in losses if v is not None]).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")

    with torch.no_grad():
        counts = torch.cat(proto_counts)
        score = torch.cat(anchor_scores)
        ca = torch.cat(coll_ann)
        ce = torch.cat(coll_excess)
        stats = {
            "prototype_mean_patch_count": float(counts.mean().item()),
            "prototype_anchor_only_fraction": float((counts == 1).float().mean().item()),
            "mean_anchor_relative_score": float(score.mean().item()),
            "anchor_collision_annotation_rate": float(ca.mean().item()),
            "mean_anchor_collision_excess": float(ce.mean().item()),
            "mean_swaps_per_annotation": float(np.mean([v["swaps"] for v in extra])) if extra else 0.0,
            "structure_active_fraction": float(np.mean([v["structure_active"] for v in extra])) if extra else 0.0,
            "mean_rank_before": float(np.mean([v["rank_before"] for v in extra])) if extra else 0.0,
            "mean_rank_after": float(np.mean([v["rank_after"] for v in extra])) if extra else 0.0,
            "mean_evidence_change": float(np.mean([
                v["evidence_after"] - v["evidence_before"] for v in extra
            ])) if extra else 0.0,
            "mean_active_triplets": float(np.mean([v["triplets"] for v in extra])) if extra else 0.0,
        }
    return loss, stats


def do_train(*, examples, base_bank_cpu, args, out_dir, text_info, projector_info, preflight):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    original.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    base_bank = original.normalize_last(base_bank_cpu.to(device=device, dtype=torch.float32).detach())
    text_relation_bank = (base_bank @ base_bank.T).detach()
    loader = DataLoader(
        original.PredObjDataset(examples), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=original.collate_examples, drop_last=False,
    )
    W = torch.nn.Parameter(torch.eye(original.VISION_DIM, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history = []
    global_step = 0
    keys = (
        "prototype_mean_patch_count", "prototype_anchor_only_fraction",
        "mean_anchor_relative_score", "anchor_collision_annotation_rate",
        "mean_anchor_collision_excess", "mean_swaps_per_annotation",
        "structure_active_fraction", "mean_rank_before", "mean_rank_after",
        "mean_evidence_change", "mean_active_triplets",
    )
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss_log, grad_log = [], []
        stats_log = {name: [] for name in keys}
        progress = tqdm(loader, desc=f"epoch {epoch:02d}/{args.epochs:02d}")
        for ex_batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = batch_forward(
                base_bank, W, ex_batch, device=device,
                text_relation_bank=text_relation_bank, args=args,
            )
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError("Missing/nonfinite gradient of W")
            grad_log.append(float(W.grad.norm().item()))
            optimizer.step()
            original.orthogonal_retract_(W)
            global_step += 1
            loss_value = float(loss.detach().item())
            loss_log.append(loss_value)
            for name in keys:
                stats_log[name].append(stats[name])
            progress.set_postfix(loss=f"{loss_value:.4f}", swap=f"{stats['mean_swaps_per_annotation']:.2f}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        mstats = original.matrix_stats(W.detach(), base_bank)
        if mstats["orthogonality_max_abs"] > 1e-3:
            raise AssertionError(f"Orthogonality invariant failed: {mstats}")
        row = {
            "epoch": epoch, "annotations": len(examples), "batch_size": args.batch_size,
            "optimizer_updates": len(loader), "global_step": global_step,
            "mean_loss": float(np.mean(loss_log)),
            "mean_grad_norm": float(np.mean(grad_log)),
            **{name: float(np.mean(vals)) for name, vals in stats_log.items()},
            "seconds": float(time.perf_counter() - start),
            **mstats,
        }
        history.append(row)
        original.write_csv(history, out_dir / "training_history.csv")
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight,
        )
        # Keep the existing bake contract (method.text contains 'current @ W').
        ckpt["format"] = VERSION
        ckpt["version"] = VERSION
        ckpt["method"].update({
            "relproto": "original independent anchor + positive evidence, fixed-count SGRC greedy swaps",
            "structure_selection": "maximize mean Relative evidence minus lambda * text-guided prototype ranking hinge; no backprop",
            "loss": ("116-class visual-to-text InfoNCE" if args.loss_type == "infonce"
                     else "original equal-annotation 1-cosine"),
            "W": "shared 768x768, Adam, SVD orthogonal retraction after every optimizer step",
        })
        original.atomic_torch_save(ckpt, out_dir / f"W_epoch_{epoch:03d}.pt")
        original.atomic_torch_save(ckpt, out_dir / "W_last.pt")
        print(
            f"[epoch] {epoch}/{args.epochs} loss={row['mean_loss']:.6f} "
            f"swaps={row['mean_swaps_per_annotation']:.3f} "
            f"active={row['structure_active_fraction']:.3f} "
            f"rank={row['mean_rank_before']:.4f}->{row['mean_rank_after']:.4f} "
            f"orth={row['orthogonality_max_abs']:.2e} "
            f"seconds={row['seconds']:.1f}", flush=True,
        )
    original.atomic_json_dump({
        "status": "completed", "version": VERSION, "checkpoint": "W_last.pt",
        "epochs": args.epochs, "global_step": global_step, "history": history,
        "formal_next_step": "bake W into the frozen projector, then run full-image PP116 eval",
    }, out_dir / "summary.json")


def build_parser():
    parser = original.build_parser()
    parser.description = __doc__
    parser.add_argument("--selector", choices=["relative", "sgrc"], default="sgrc")
    parser.add_argument("--loss_type", choices=["cosine", "infonce"], default="infonce")
    parser.add_argument("--candidate_k", type=int, default=32)
    parser.add_argument("--lambda_struct", type=float, default=0.1)
    parser.add_argument("--rank_delta", type=float, default=0.05)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--swap_rounds", type=int, default=2)
    parser.add_argument("--improvement_eps", type=float, default=1e-8)
    parser.add_argument("--nce_temperature", type=float, default=0.07)
    return parser


def validate_args(args):
    original.validate_args(args)
    if not (args.prototype_max_patches <= args.candidate_k <= original.DEFAULT_PATCH_COUNT):
        raise ValueError("require prototype_max_patches <= candidate_k <= 1024")
    for name in ("lambda_struct", "rank_delta", "rank_margin", "improvement_eps"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.swap_rounds < 0:
        raise ValueError("swap_rounds must be >=0")
    if not math.isfinite(args.nce_temperature) or args.nce_temperature <= 0:
        raise ValueError("nce_temperature must be finite and >0")


def self_test():
    """CPU synthetic checks (no datasets / checkpoint required)."""
    original.self_test()
    torch.manual_seed(10)
    d = original.VISION_DIM
    features = F.normalize(torch.randn(1024, d), dim=-1)
    text_bank = F.normalize(torch.randn(original.NUM_PARTS, d), dim=-1)
    text_relation = text_bank @ text_bank.T
    pids = torch.tensor([0, 1, 2, 3])
    x = F.normalize(text_bank[pids], dim=-1)
    foreground = torch.zeros(1, 1024, dtype=torch.bool)
    foreground[:, :80] = True
    init = original.build_relproto(
        x[None], features[None], foreground, max_patches=8,
    )
    absolute = x @ features.T
    rel = original.relative_scores(absolute[None], foreground)[0]
    support, proto, diag = refine_one_annotation(
        features, rel, init["support_indices"][0],
        text_relation[pids][:, pids], candidate_k=32,
        lam=0.1, delta=0.0, margin=0.1, rounds=2,
        improvement_eps=1e-8,
    )
    assert torch.equal(support[:, 0], init["anchor_indices"][0])
    assert torch.equal(support.ge(0), init["support_indices"][0].ge(0))
    assert diag["rank_after"] <= diag["rank_before"] + 1e-5 or diag["swaps"] > 0
    assert proto.shape == (4, d)
    W = torch.nn.Parameter(torch.eye(d))
    all_text = F.normalize(text_bank @ W, dim=-1)
    logits = proto.detach() @ all_text.T / 0.07
    loss = F.cross_entropy(logits, pids)
    loss.backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    print("PASS sgrc_selection_anchor_and_cardinality")
    print("PASS 116way_infonce_gradients")
    print("SELF_TEST_PASS_SGRC_INFONCE")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.self_test:
        self_test()
        return 0
    # Reuse the exact original preflight/projector/bank entry point, without
    # editing its source file. Its train hook is swapped only in this process.
    original.do_train = do_train
    original.train_and_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())