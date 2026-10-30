#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 Object-only InfoNCE + Soft Relative + Structure trainer.

New file; does not modify the official train_relproto_alignemt.py.

For K>=2 candidate parts (image-present OR full-object taxonomy):
  r[b,k,i]  = cos(t_k W, f_i) - max_{q != k} cos(t_q W, f_i), foreground only
  C[b,k]    = {i: r[b,k,i] > 0}, or the row-wise argmax anchor if empty
  A[b,k,:]  = masked_softmax(r[b,k,:] / soft_temperature over C[b,k])
  v[b,k]    = L2_normalize(sum_i A[b,k,i] * f_i)
  loss      = equal-annotation Object-only InfoNCE(v.detach(),tW), whose
              negatives are ALL other parts of the same object, not only
              present parts; plus lambda_struct * relation ranking

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
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# The training entry point may be executed as a file (python opn_nce/...)
# or as a module (python -m opn_nce....). Resolve the official trainer
# from this package's actual repository root in either case.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import train_relproto_alignemt as original

VERSION = "pp116_object_only_nce_20261029_v1"

# Used only for globally retained NCE with downweighted cross-object negatives.
# Fixed taxonomy derived from the frozen text bank. It never contains GT pixel masks.
_OBJECT_TAXONOMY_MASKS: dict[str, torch.Tensor] = {}
# Computed exclusively from TRAINING image-level Presence, not GT pixels.
_PART_FREQ_WEIGHTS = torch.ones(original.NUM_PARTS,dtype=torch.float32)


_SUFFIX_CODES = torch.zeros(original.NUM_PARTS,dtype=torch.long)
_PART_NAMES: list[str] = []
_NEG_GENERATOR = None


# [116,116] strict taxonomy-only negative mask. Filled after the official text
# bank has been loaded and validated; it never uses image masks or GT pixels.
_TAXONOMY_NEG_MASK = torch.zeros((original.NUM_PARTS, original.NUM_PARTS), dtype=torch.bool)
_TAXONOMY_CLASS_NAMES: list[str] = []

# Strict object-local InfoNCE negatives, excluding self.
# [116,116], constructed from TEXT CLASS NAMES only. No part-mask information.
_OBJECT_ONLY_NEG_MASK = torch.zeros((original.NUM_PARTS, original.NUM_PARTS), dtype=torch.bool)


def build_object_only_negative_mask(names: list[str]) -> torch.Tensor:
    """Fixed taxonomy negatives: ALL other parts from the same object.

    This is deliberately independent of current crop Presence labels.
    Singleton-object labels (PP116: tvmonitor's screen) have zero negatives.
    """
    if len(names) != original.NUM_PARTS or len(set(names)) != len(names):
        raise ValueError(f"Expected {original.NUM_PARTS} distinct class names")
    objects = []
    for name in names:
        if "'s " not in name:
            raise ValueError(f"Invalid taxonomy name: {name!r}")
        obj, part = name.split("'s ", 1)
        if not obj.strip() or not part.strip():
            raise ValueError(f"Invalid taxonomy name: {name!r}")
        objects.append(obj.strip().lower())
    out = torch.tensor([
        [i != j and objects[i] == objects[j] for j in range(len(names))]
        for i in range(len(names))
    ], dtype=torch.bool)
    if not torch.equal(out, out.T) or bool(out.diagonal().any()):
        raise AssertionError("Object-only mask must be symmetric without diagonal")
    return out


def write_object_only_negative_csv(out_dir: Path) -> None:
    if not _TAXONOMY_CLASS_NAMES:
        raise RuntimeError("Text taxonomy is not initialized")
    rows = []
    for idx, name in enumerate(_TAXONOMY_CLASS_NAMES):
        ids = _OBJECT_ONLY_NEG_MASK[idx].nonzero(as_tuple=True)[0].tolist()
        rows.append({
            "part_id": idx, "positive_part": name,
            "same_object_negative_count": len(ids),
            "same_object_negatives": "; ".join(_TAXONOMY_CLASS_NAMES[j] for j in ids),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "object_only_negative_sets.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = [r["same_object_negative_count"] for r in rows]
    print(f"[OBJ-NCE] within-object only, mean negatives={sum(counts)/len(counts):.3f}, "
          f"min={min(counts)}, max={max(counts)}, zero={sum(k==0 for k in counts)}", flush=True)
    print(f"[OBJ-NCE] Saved {path}", flush=True)



def build_taxonomy_negative_mask(names: list[str]) -> torch.Tensor:
    """Negatives: other parts of same object UNION same-name parts of other objects.

    Matching is exact after lowercase/outer whitespace normalization. It does
    not infer synonyms (paw vs foot), nor use current crop presence for negatives.
    """
    if len(names) != original.NUM_PARTS or len(set(names)) != len(names):
        raise ValueError(f"Expected {original.NUM_PARTS} distinct class names")
    object_ids, part_ids = [], []
    for name in names:
        if "'s " not in name:
            raise ValueError(f"Invalid taxonomy name: {name!r}")
        obj, part = name.split("'s ", 1)
        obj, part = obj.strip().lower(), part.strip().lower()
        if not obj or not part:
            raise ValueError(f"Empty object/part component: {name!r}")
        object_ids.append(obj)
        part_ids.append(part)
    mask = torch.tensor([
        [(object_ids[i] == object_ids[j] or part_ids[i] == part_ids[j])
         and i != j for j in range(len(names))]
        for i in range(len(names))
    ], dtype=torch.bool)
    if mask.shape != (original.NUM_PARTS, original.NUM_PARTS):
        raise AssertionError('Invalid 116x116 taxonomy mask')
    if not bool((mask == mask.T).all()) or bool(mask.diagonal().any()):
        raise AssertionError('Taxonomy mask must be symmetric and omit self')
    return mask


def write_taxonomy_negative_csv(out_dir: Path) -> None:
    if not _TAXONOMY_CLASS_NAMES or _TAXONOMY_NEG_MASK.shape != (original.NUM_PARTS, original.NUM_PARTS):
        raise RuntimeError('Taxonomy not initialized before training')
    rows = []
    for pid, name in enumerate(_TAXONOMY_CLASS_NAMES):
        obj, part = (v.strip().lower() for v in name.split("'s ", 1))
        internal, same_name = [], []
        for j in _TAXONOMY_NEG_MASK[pid].nonzero(as_tuple=True)[0].tolist():
            other_obj, other_part = (v.strip().lower() for v in _TAXONOMY_CLASS_NAMES[j].split("'s ", 1))
            if obj == other_obj:
                internal.append(_TAXONOMY_CLASS_NAMES[j])
            elif part == other_part:
                same_name.append(_TAXONOMY_CLASS_NAMES[j])
            else:
                raise AssertionError('Foreign negative present in taxonomy mask')
        rows.append({
            'part_id':pid, 'positive_part':name,
            'same_object_negative_count':len(internal),
            'cross_object_same_name_negative_count':len(same_name),
            'total_negative_count':len(internal)+len(same_name),
            'same_object_negatives':'; '.join(internal),
            'cross_object_same_name_negatives':'; '.join(same_name),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/'taxonomy_negative_sets.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    counts=[r['total_negative_count'] for r in rows]
    print(f"[OPN] strict object+part-name union; neg_count min={min(counts)} "
          f"mean={sum(counts)/len(counts):.3f} max={max(counts)} "
          f"zero={sum(c==0 for c in counts)}", flush=True)
    print(f"[OPN] Saved {out_dir/'taxonomy_negative_sets.csv'}", flush=True)


def effective_hardneg_k(args):
    """K for this epoch. Fixed / global-to-hard / hard-to-global / delayed.

    Epoch 1...epochs. Schedule depends only on current epoch; no GT/validation.
    """
    maximum=original.NUM_PARTS-1
    end=int(args.hardneg_k)
    start=int(getattr(args,"hardneg_start",115))
    ep=int(getattr(args,'_current_epoch',1))
    total=int(getattr(args,"epochs",30))
    mode=getattr(args,"hardneg_schedule","fixed")
    if mode=='fixed': return max(1,min(maximum,end))
    if mode=='delayed':
        return maximum if ep<=args.hardneg_warmup_epochs else end
    progress = (ep-1)/max(1,total-1)
    if mode=='linear':
        return max(1,min(maximum,int(round(start+(end-start)*progress))))
    if mode=='cosine':
        return max(1,min(maximum,int(round(end+(start-end)*(1+math.cos(math.pi*progress))/2))))
    raise ValueError('unknown schedule '+mode)


def experimental_nce_losses(visual, all_text, pids, examples, member_indices, args):
    """Always uses the global 116-part vocabulary except explicit negative tests.

    visual.detach() is immutable: no NCE gradients through patch assignment.
    Output per positive Part [B,K]. All variants preserve original K averaging.
    """
    B, K, _ = visual.shape
    C = original.NUM_PARTS
    logits = torch.matmul(visual.detach(), all_text.T) / args.nce_temperature
    targets = pids.reshape(-1)
    flat = logits.reshape(-1, C)
    method = args.nce_variant
    if method == "object_only":
        if not _TAXONOMY_CLASS_NAMES:
            raise RuntimeError("116-class taxonomy not initialized")
        negmask = _OBJECT_ONLY_NEG_MASK.to(device=flat.device).index_select(0, targets)
        keep = negmask.clone()
        keep.scatter_(1, targets[:, None], True)  # positive always included
        selected = flat.masked_fill(~keep, -torch.inf)
        return F.cross_entropy(selected, targets, reduction="none").reshape(B, K)
    if method == "taxonomy_union":
        # Index by each POSITIVE category. There is no Hardest Top-K, no current
        # Crop-Presence restriction, and no unlisted global negatives.
        if not _TAXONOMY_CLASS_NAMES:
            raise RuntimeError('The 116-class text taxonomy was not initialized')
        negmask = _TAXONOMY_NEG_MASK.to(device=flat.device).index_select(0, targets)
        keep = negmask.clone()
        keep.scatter_(1, targets[:, None], True)
        # For a class with zero eligible negatives, e.g. tvmonitor's screen,
        # CE contains only the positive and equals 0, with zero NCE gradient.
        selected = flat.masked_fill(~keep, -torch.inf)
        return F.cross_entropy(selected, targets, reduction='none').reshape(B, K)
    if method == "global":
        return F.cross_entropy(flat, targets, reduction="none").reshape(B, K)
    if method == "smooth":
        return F.cross_entropy(flat, targets, reduction="none",
                               label_smoothing=args.label_smoothing).reshape(B, K)
    if method == "clip":
        raw = F.cross_entropy(flat, targets, reduction="none").reshape(B, K)
        # Hard-loss truncation is an explicit noise-robust *ablation*, not an asserted fix.
        return raw.clamp(max=args.nce_clip)
    if method in ("hardneg", "hardneg_quota", "hardneg_samename", "hardneg_random", "hardneg_easiest", "hardneg_textstatic", "hardneg_mixglobal", "hardneg_softgate"):
        n = effective_hardneg_k(args)
        neg = flat.detach().clone()
        neg.scatter_(1, targets[:, None], -torch.inf)
        if n == C - 1 and method == "hardneg" and getattr(args,"hardneg_schedule","fixed") == "fixed":
            # Exactly the global 116-class CE, avoids gratuitous scatter rounding.
            return F.cross_entropy(flat, targets, reduction="none").reshape(B, K)
        if method == "hardneg_softgate":
            # Continuous, GT-free weighting of all negatives around Kth-hardest.
            # Gate is detached: no gradient through negative selection threshold.
            cutoff=neg.topk(n,dim=1).values[:,-1:]
            loggate=F.logsigmoid((neg-cutoff)/args.hardneg_gate_tau)
            loggate.scatter_(1,targets[:,None],0.0)
            return (torch.logsumexp(flat+loggate,dim=1)
                    - flat.gather(1,targets[:,None]).squeeze(1)).reshape(B,K)
        if method == "hardneg_random":
            generator=_NEG_GENERATOR
            score=torch.rand(neg.shape,device=neg.device,generator=generator,dtype=neg.dtype)
            score.scatter_(1,targets[:,None],-torch.inf)
            idx=torch.topk(score,k=n,dim=1).indices
            keep=torch.zeros_like(flat,dtype=torch.bool)
            keep.scatter_(1,idx,True)
        elif method == "hardneg_easiest":
            score=-neg
            score.scatter_(1,targets[:,None],-torch.inf)
            idx=torch.topk(score,k=n,dim=1).indices
            keep=torch.zeros_like(flat,dtype=torch.bool)
            keep.scatter_(1,idx,True)
        elif method == "hardneg_textstatic":
            # Similarity of part-text to other part-text; orthogonal W keeps its
            # cosine relations, so selection is sample-independent within a class.
            sim=all_text.detach()@all_text.detach().T
            score=sim[targets].clone()
            score.scatter_(1,targets[:,None],-torch.inf)
            idx=torch.topk(score,k=n,dim=1).indices
            keep=torch.zeros_like(flat,dtype=torch.bool)
            keep.scatter_(1,idx,True)
        elif method == "hardneg_quota":
            local = torch.stack([_OBJECT_TAXONOMY_MASKS[examples[i].class_name]
                                 for i in member_indices],dim=0).to(device=flat.device)
            local = local[:,None,:].expand(B,K,C).reshape(-1,C).clone()
            local.scatter_(1,targets[:,None],False)
            # Require up to q same-object negatives; remaining slots are globally hardest.
            q = min(int(args.hardneg_local_quota), n)
            keep = torch.zeros_like(flat,dtype=torch.bool)
            if q > 0:
                pool = neg.masked_fill(~local,-torch.inf)
                idx_local = torch.topk(pool,k=q,dim=1).indices
                capacity = local.sum(dim=1).clamp(max=q)
                chosen = torch.arange(q,device=flat.device)[None,:] < capacity[:,None]
                keep.scatter_(1,idx_local,chosen)
                keep &= local
            remaining = n - keep.sum(dim=1)
            spare = neg.masked_fill(keep,-torch.inf)
            extra = torch.topk(spare,k=n,dim=1).indices
            selected = torch.arange(n,device=flat.device)[None,:] < remaining[:,None]
            # scatter_add handles duplicate indices safely, unlike scattering False.
            keep |= (torch.zeros_like(flat,dtype=torch.int32)
                     .scatter_add(1,extra,selected.to(torch.int32)) > 0)
        else:
            idx = torch.topk(neg,k=n,dim=1).indices
            keep = torch.zeros_like(flat,dtype=torch.bool)
            keep.scatter_(1,idx,True)
        keep.scatter_(1,targets[:,None],True)
        if not bool((keep.sum(dim=1)==n+1).all()):
            raise RuntimeError('Expected exactly K negatives plus positive')
        selected_logits = flat.masked_fill(~keep,-torch.inf)
        if method == "hardneg_samename":
            # Same suffix (e.g. cat's ear vs dog's ear) is a semantic proxy,
            # not GT pixel annotation. Retain as negative but downweight in denominator.
            suffix = _SUFFIX_CODES.to(device=flat.device)
            same = suffix[targets[:,None]] == suffix[None,:]
            same.scatter_(1,targets[:,None],False)
            selected_logits = selected_logits + same.float() * math.log(args.same_name_weight)
        hard_losses=F.cross_entropy(selected_logits,targets,reduction="none")
        if method == "hardneg_mixglobal":
            full_losses=F.cross_entropy(flat,targets,reduction="none")
            hard_losses=(1-args.hardneg_mix_global)*hard_losses + args.hardneg_mix_global*full_losses
        return hard_losses.reshape(B,K)
    if method == "crossweight":
        # Keep ALL 116 classes and the positive; cross-object negatives are
        # still present but weighted by gamma (0 < gamma <= 1) in denominator.
        local = torch.stack([_OBJECT_TAXONOMY_MASKS[examples[i].class_name]
                            for i in member_indices], dim=0).to(device=flat.device)
        local = local[:, None, :].expand(B, K, C).reshape(-1, C)
        if not bool(local.gather(1, targets[:, None]).all()):
            raise RuntimeError("Cross-weight loss positive ID not in own taxonomy")
        weighted_logits = flat + (~local).to(flat.dtype) * math.log(args.cross_weight)
        return (torch.logsumexp(weighted_logits, dim=-1)
                - flat.gather(1, targets[:, None]).squeeze(1)).reshape(B, K)
    raise ValueError("Unexpected NCE variant: " + method)


def pseudo_reliability(weights, patch, positive_count, fallback, weighted_margin,
                       *, mode, floor):
    """GT-free reliability proxy for per-positive loss weighting; intentionally detached.

    Does not treat max softmax weight as calibrated correctness confidence.
    floor>0 prevents rare/failed Part from having no learning signal.
    """
    B, K, N = weights.shape
    if mode == "off":
        return torch.ones((B,K), device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        # Margin reliability: combines positive support and weighted advantage.
        support = (torch.log1p(positive_count.float()) / math.log(13.0)).clamp(0,1)
        margin = torch.sigmoid((weighted_margin.float() - 0.02)/0.02)
        score = support * margin * (1-fallback.float())
        if mode == "coherence":
            # Weighted within-prototype visual consistency; no labels are used.
            pooled = torch.bmm(weights, patch).norm(dim=-1)
            coherence = ((pooled-0.55)/0.40).clamp(0,1)
            score = score * coherence
        elif mode != "margin":
            raise ValueError(mode)
        return (floor + (1-floor)*score).detach()



def soft_relative_prototypes(
    current_text: torch.Tensor,  # [B,K,D], differentiable, normalized
    tokens: torch.Tensor,        # [B,1024,D], frozen normalized
    foreground: torch.Tensor,    # [B,1024], bool
    *,
    temperature: float,
    mode: str = "positive",
    top_k: int = 8,
    gate_tau: float = 0.03,
    margin_power: float = 1.0,
    fallback_k: int = 4,
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
    if mode == "topk":
        # Only change candidate SUPPORT. Never introduce non-positive patches
        # unless the original fallback was required.
        k = min(top_k, relative.size(-1))
        top = relative.masked_fill(~candidates, -torch.inf).topk(k, dim=-1).indices
        keep = torch.zeros_like(candidates)
        keep.scatter_(-1, top, True)
        candidates = candidates & keep
    elif mode == "fallback_k":
        # For no-positive rows, aggregate top-k uncertain foreground tokens
        # instead of a single automatically overconfident anchor.
        k = min(fallback_k, relative.size(-1))
        top = relative.topk(k,dim=-1).indices
        hot = torch.zeros_like(candidates)
        hot.scatter_(-1, top, True)
        candidates = positives | (hot & empty[..., None])
    elif mode == "softgate":
        # Continuous reliability gating on *all* foreground patches. Critically,
        # this does NOT require any spatial GT labels or modify Presence.
        candidates = foreground[:,None,:].expand_as(relative)
    elif mode not in ("positive", "marginpower"):
        raise ValueError("Unexpected prototype variant: " + mode)
    if not bool(candidates.any(dim=-1).all()):
        raise AssertionError("empty candidate row after fallback")

    logits = (relative / temperature)
    if mode == "softgate":
        logits = logits + F.logsigmoid(relative / gate_tau)
    elif mode == "marginpower":
        # Within positive support, emphasize high relative margins; fallback
        # retains its original single-anchor weight of one.
        logits = logits + margin_power * torch.log(
            (relative.clamp_min(1e-6) / 0.03).clamp_min(1e-6)) * positives.float()
    logits = logits.masked_fill(~candidates, -torch.inf)
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
            "weights": weights.detach(),
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
                mode=args.proto_variant, top_k=args.proto_topk,
                gate_tau=args.gate_tau, margin_power=args.margin_power,
                fallback_k=args.fallback_k,
            )
            positives = diag["positive_count"]
            fallback = diag["fallback"]
            effective = diag["effective_count"]
            weighted_margin = diag["weighted_margin"]

        # OPN uses strict taxonomy union when nce_variant=taxonomy_union.
        # The original global and hardest variants remain optional controls.
        part_losses = experimental_nce_losses(
            visual, all_text, pids, examples, member_indices, args)
        if args.reliability_variant == "off":
            conf = torch.ones_like(part_losses)
        elif K == 1:
            # No relative competition exists: do not penalize singleton Presence.
            conf = torch.ones_like(part_losses)
        else:
            # Recompute weights from normalized prototypes is wrong; use the
            # exact weights returned by the original support construction below.
            conf = pseudo_reliability(
                weights=diag["weights"], patch=patch,
                positive_count=positives, fallback=fallback,
                weighted_margin=weighted_margin,
                mode=args.reliability_variant, floor=args.reliability_floor)
        if args.balance_presence:
            class_weights = _PART_FREQ_WEIGHTS.to(device=device)[pids]
            part_losses = part_losses * class_weights
        nce_per_annotation = (part_losses * conf).mean(dim=1)

        if K >= 3 and args.lambda_struct > 0:
            local_g = frozen_relations[pids.unsqueeze(-1), pids.unsqueeze(-2)]
            rank_per_annotation, struct_active, triplet_count = relational_rank_loss(
                visual, local_g, delta=args.rank_delta, margin=args.rank_margin,
            )
        else:
            rank_per_annotation = nce_per_annotation.new_zeros((B,))
            struct_active = nce_per_annotation.new_zeros((B,))
            triplet_count = nce_per_annotation.new_zeros((B,))

        if args.reliability_on_structure and K >= 3:
            # Detached confidence prevents W manipulating its own scalar weight.
            rank_per_annotation = rank_per_annotation * conf.mean(dim=1)
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
            if args.reliability_variant != "off":
                stat_sum["pseudo_confidence_sum"] += float(conf.sum().item())

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
        "mean_pseudo_confidence": (stat_sum["pseudo_confidence_sum"] / part_count
                                   if args.reliability_variant != "off" else 1.0),
    }
    # Return separated scalars for a first-batch structural gradient sanity check.
    return total_loss, diag_out, {"nce": nce_loss, "relation": rel_loss}


def build_fixed_probe(examples, count_per_part):
    """Deterministic GT-free greedy coverage: target 3 crop instances per part.

    Uses only original training presence IDs, no validation and no pixel labels.
    """
    seen=defaultdict(int)
    chosen=[]
    # Stable cache ordering; same list across variants, seeds and epochs.
    for ex in examples:
        part_ids=[int(p) for p in ex.pids.tolist()]
        if any(seen[p]<count_per_part for p in part_ids):
            chosen.append(ex)
            for pid in part_ids:seen[pid]+=1
    return chosen


def audit_hardest_probe(probe, base_bank, W, device, args, dest):
    """Read-only fixed-probe snapshots, NEVER fed back into optimization.

    Score all 116 negatives independently of the training K. Records the
    strongest rival, exact top16 composition and positive/rival gap.
    """
    result={}
    instance_rows=[]
    with torch.no_grad():
        text=original.normalize_last(base_bank @ W)
        for offset in range(0,len(probe),args.batch_size):
            subset=probe[offset:offset+args.batch_size]
            by_k=defaultdict(list)
            for j,ex in enumerate(subset):by_k[int(ex.pids.numel())].append(j)
            tokens=original.normalize_last(torch.stack([ex.tokens for ex in subset]).to(device,dtype=torch.float32))
            fg=torch.stack([ex.foreground for ex in subset]).to(device,dtype=torch.bool)
            for k,idx in sorted(by_k.items()):
                ids=torch.tensor(idx,device=device)
                pids=torch.stack([subset[j].pids for j in idx]).to(device,dtype=torch.long)
                curr=text[pids]; pat=tokens[ids];mask=fg[ids]
                if k==1:
                    absolute=torch.bmm(curr,pat.transpose(1,2))
                    weights=torch.softmax(absolute.masked_fill(~mask[:,None,:],-torch.inf)/args.soft_temperature,dim=-1)
                    v=F.normalize(torch.bmm(weights,pat),dim=-1)
                    diag=None
                else:
                    v,diag=soft_relative_prototypes(curr,pat,mask,temperature=args.soft_temperature,
                        mode=args.proto_variant,top_k=args.proto_topk,gate_tau=args.gate_tau,
                        margin_power=args.margin_power,fallback_k=args.fallback_k)
                logits=(v @ text.T / args.nce_temperature).reshape(-1,original.NUM_PARTS)
                positive=pids.reshape(-1)
                competitors=logits.clone()
                competitors.scatter_(1,positive[:,None],-torch.inf)
                top=competitors.topk(16,dim=1)
                pos=logits.gather(1,positive[:,None]).squeeze(1)
                probabilities=torch.softmax(logits,dim=1).gather(1,positive[:,None]).squeeze(1)
                suffix_codes=_SUFFIX_CODES.to(device)
                for row,(pid,nidx) in enumerate(zip(positive.tolist(),top.indices.cpu().tolist())):
                    sid=int(pid)
                    obj=_PART_NAMES[sid].split("'s ",1)[0]
                    same_obj=sum(_PART_NAMES[q].split("'s ",1)[0]==obj for q in nidx)
                    same_suffix=sum(_SUFFIX_CODES[q].item()==_SUFFIX_CODES[sid].item() for q in nidx)
                    e=subset[idx[row//k]]
                    instance_rows.append({
                        'epoch':args._current_epoch,'crop_index':int(e.row_index),
                        'pid':sid,'part':_PART_NAMES[sid],
                        'global_top1_pid':nidx[0],
                        'global_top1_name':_PART_NAMES[nidx[0]],
                        'global_top16_ids':json.dumps(nidx,separators=(',',':')),
                        'top16_same_object_fraction':same_obj/16,
                        'top16_same_suffix_fraction':same_suffix/16,
                        'full116_p_positive':float(probabilities[row].item()),
                        'positive_minus_top1_logit':float((pos[row]-top.values[row,0]).item()),
                        'positive_count':int(diag['positive_count'][row//k,row%k].item()) if diag else '',
                        'fallback':int(diag['fallback'][row//k,row%k].item()) if diag else 0,
                        'effective_patch_count':float(diag['effective_count'][row//k,row%k].item()) if diag else ''})
                    key=sid
                    if key not in result:
                        result[key]={'n':0,'p_full_sum':0.,'gap_sum':0.,'same_obj_sum':0.,
                                     'same_suffix_sum':0.,'top1':defaultdict(int),'negative_hist':defaultdict(int)}
                    z=result[key];z['n']+=1
                    z['p_full_sum']+=float(probabilities[row].item())
                    z['gap_sum']+=float((pos[row]-top.values[row,0]).item())
                    z['same_obj_sum']+=same_obj/16
                    z['same_suffix_sum']+=same_suffix/16
                    z['top1'][nidx[0]]+=1
                    for q in nidx:z['negative_hist'][q]+=1
    rows=[]
    for pid,z in sorted(result.items()):
        n=z['n'];top1,count=max(z['top1'].items(),key=lambda x:(x[1],-x[0]))
        rows.append({'epoch':args._current_epoch,'pid':pid,'part':_PART_NAMES[pid],
                     'n_probe':n,'hardneg_effective_k':effective_hardneg_k(args) if args.nce_variant.startswith('hardneg') else 115,
                     'mean_full116_p_positive':z['p_full_sum']/n,
                     'mean_positive_minus_top1_logit':z['gap_sum']/n,
                     'top16_same_object_fraction':z['same_obj_sum']/n,
                     'top16_same_suffix_fraction':z['same_suffix_sum']/n,
                     'modal_top1_pid':top1,'modal_top1_name':_PART_NAMES[top1],
                     'modal_top1_fraction':count/n,
                     'top16_id_counts_json':json.dumps(dict(z['negative_hist']),separators=(',',':'))})
    if rows:
        with dest.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    if instance_rows:
        ipath=dest.with_name(dest.name.replace('probe_epoch_','probe_instances_epoch_'))
        with ipath.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(instance_rows[0]));writer.writeheader();writer.writerows(instance_rows)
    print(f'[probe] epoch={args._current_epoch} fixed_crops={len(probe)} parts={len(rows)} instances={len(instance_rows)} -> {dest}',flush=True)


def do_train(*, examples, base_bank_cpu, args, out_dir, text_info, projector_info, preflight):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.nce_variant == "object_only":
        write_object_only_negative_csv(out_dir)
    else:
        write_taxonomy_negative_csv(out_dir)
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
    global _NEG_GENERATOR
    _NEG_GENERATOR=torch.Generator(device=device).manual_seed(args.seed+9173)
    history: list[dict[str, Any]] = []
    global_step = 0
    gradient_path_checked = False
    diagnostics_keys = (
        "loss_total", "loss_nce", "loss_relation", "structure_active_fraction",
        "mean_triplets_per_annotation", "mean_positive_count_multi",
        "fallback_fraction_multi", "mean_effective_patch_count", "mean_weighted_margin",
        "single_part_annotation_fraction", "mean_pseudo_confidence",
    )
    probe = build_fixed_probe(examples, args.probe_per_part) if args.probe else []
    probe_root = out_dir / "hardest_probe"
    probe_root.mkdir(parents=True,exist_ok=True)
    if probe and 0 in args.probe_epochs:
        args._current_epoch = 0
        audit_hardest_probe(probe,base_bank,W,device,args,probe_root/'probe_epoch_000.csv')
    for epoch in range(1, args.epochs + 1):
        args._current_epoch = epoch
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
        row["hardneg_effective_k"] = (float(_OBJECT_ONLY_NEG_MASK.sum(dim=1).float().mean().item()) if args.nce_variant == "object_only" else float(_TAXONOMY_NEG_MASK.sum(dim=1).float().mean().item()) if args.nce_variant == "taxonomy_union" else effective_hardneg_k(args) if args.nce_variant.startswith("hardneg") else original.NUM_PARTS-1)
        history.append(row)
        original.write_csv(history, out_dir / "training_history.csv")
        if probe and epoch in set(args.probe_epochs):
            audit_hardest_probe(probe, base_bank, W, device, args,
                                probe_root / f"probe_epoch_{epoch:03d}.csv")
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight,
        )
        ckpt["format"] = VERSION
        ckpt["version"] = VERSION
        ckpt["method"].update({
            "relproto": "K>=2: all positive-margin foreground patches, fallback argmax if empty; "
                        "K=1: absolute-score foreground softmax; normalized soft pooling; "
                        "candidate parts determined by --part_setting (presence or objectfull)",
            "structure_selection": "no SGRC discrete search; differentiable visual relation ranking "
                                   "through soft assignment to shared orthogonal W",
            "loss": ("strict object-only InfoNCE + relation ranking" if args.nce_variant == "object_only"
                     else "strict same-object plus cross-object exact same-name InfoNCE + relation ranking"),
            "part_supervision": args.part_setting,
            "candidate_source": ("image-specific present part classes" if args.part_setting == "presence"
                                 else "ALL taxonomy part classes under known object"),
            "W": "one 768x768 shared orthogonal map, Adam + per-step SVD retraction",
        })
        ckpt["taxonomy_nce"] = {
            "scope": ("same_object_only" if args.nce_variant == "object_only" else "same_object_plus_cross_object_same_name"),
            "part_names": _TAXONOMY_CLASS_NAMES,
            "negative_counts": (_OBJECT_ONLY_NEG_MASK if args.nce_variant == "object_only" else _TAXONOMY_NEG_MASK).sum(1).tolist(),
            "exact_name_matching": True,
            "scope_uses_all_object_parts_not_only_presence": args.nce_variant == "object_only",
        }
        ckpt["experiment_arm"] = args.part_setting
        ckpt["search_variant"] = {k: getattr(args,k) for k in (
            "hardneg_schedule", "hardneg_start", "hardneg_warmup_epochs",
            "hardneg_local_quota", "same_name_weight", "hardneg_mix_global",
            "hardneg_gate_tau", "probe_per_part",
            "proto_variant", "proto_topk", "gate_tau", "margin_power", "fallback_k",
            "nce_variant", "label_smoothing", "cross_weight", "hardneg_k", "nce_clip",
            "reliability_variant", "reliability_floor", "reliability_on_structure", "balance_presence")}
        ckpt["method"]["nce_scope"] = ("all other taxonomy parts of same object ONLY" if args.nce_variant == "object_only" else "all other object parts + other objects same-name parts ONLY" if args.nce_variant == "taxonomy_union" else "explicit baseline variant")
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
    # Overrides only defaults: original data loading and checks remain official.
    p.set_defaults(
        train_dataset="feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth",
        text_bank="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt",
        weights="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
        model_config="configs/vitb_mlp_infonce.yaml",
        out_dir="final_exp/relproto_alignment/pp116_nightly_softsingle_default",
        prototype_max_patches=8,  # retained CLI compatibility; unused (all-soft)
        expected_bg_thresh=0.55,
        epochs=100, batch_size=16, lr=1e-5, seed=123,
    )
    p.add_argument("--part_setting", choices=("presence", "objectfull"), required=True)
    p.add_argument("--soft_temperature", type=float, default=0.05)
    p.add_argument("--nce_temperature", type=float, default=0.07)
    p.add_argument("--lambda_struct", type=float, default=4.0)
    p.add_argument("--rank_delta", type=float, default=0.05)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--proto_variant", choices=("positive","topk","fallback_k","softgate","marginpower"), default="positive")
    p.add_argument("--proto_topk", type=int, default=8)
    p.add_argument("--fallback_k", type=int, default=4)
    p.add_argument("--gate_tau", type=float, default=0.03)
    p.add_argument("--margin_power", type=float, default=1.0)
    p.add_argument("--nce_variant", choices=("object_only","taxonomy_union","global","smooth","crossweight","hardneg","hardneg_quota","hardneg_samename","hardneg_random","hardneg_easiest","hardneg_textstatic","hardneg_mixglobal","hardneg_softgate","clip"), default="object_only")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--cross_weight", type=float, default=0.25)
    p.add_argument("--hardneg_k", type=int, default=16)
    p.add_argument("--hardneg_schedule", choices=("fixed","linear","cosine","delayed"), default="fixed")
    p.add_argument("--hardneg_start", type=int, default=115)
    p.add_argument("--hardneg_warmup_epochs", type=int, default=5)
    p.add_argument("--hardneg_local_quota", type=int, default=4)
    p.add_argument("--same_name_weight", type=float, default=0.5)
    p.add_argument("--hardneg_mix_global", type=float, default=0.15)
    p.add_argument("--hardneg_gate_tau", type=float, default=0.5)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--probe_per_part", type=int, default=3)
    p.add_argument("--probe_epochs", type=str, default="0,1,3,5,10,20,30")
    p.add_argument("--nce_clip", type=float, default=4.0)
    p.add_argument("--reliability_variant", choices=("off","margin","coherence"), default="off")
    p.add_argument("--reliability_floor", type=float, default=0.25)
    p.add_argument("--reliability_on_structure", action="store_true")
    p.add_argument("--balance_presence", action="store_true",
                   help="inverse-sqrt training Presence-frequency positive weights; clipped and mean-normalized")
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
    if not 1 <= args.prototype_max_patches <= original.DEFAULT_PATCH_COUNT:
        raise ValueError("--prototype_max_patches must be within 1..1024 (single-part control only)")
    if not (1 <= args.proto_topk <= original.DEFAULT_PATCH_COUNT and
            1 <= args.fallback_k <= original.DEFAULT_PATCH_COUNT):
        raise ValueError("topk/fallback_k must be within 1..1024")
    if not (args.gate_tau>0 and math.isfinite(args.gate_tau) and
            math.isfinite(args.margin_power) and args.margin_power >= 0):
        raise ValueError("invalid gating parameters")
    if not (0 <= args.label_smoothing < 1 and 0 < args.cross_weight <= 1 and
            1 <= args.hardneg_k < original.NUM_PARTS and args.nce_clip>0):
        raise ValueError("invalid NCE ablation parameters")
    if not 1<=args.hardneg_start<original.NUM_PARTS:
        raise ValueError('hardneg_start must be in 1..115')
    if not 0<=args.hardneg_local_quota<=115:
        raise ValueError('hardneg_local_quota must be in 0..115')
    if not 0<args.same_name_weight<=1 or not math.isfinite(args.same_name_weight):
        raise ValueError('same_name_weight must be in (0,1]')
    if not 0<=args.hardneg_mix_global<=1 or not math.isfinite(args.hardneg_mix_global):
        raise ValueError('hardneg_mix_global must be in [0,1]')
    if args.hardneg_gate_tau<=0 or not math.isfinite(args.hardneg_gate_tau):
        raise ValueError('hardneg_gate_tau must be >0')
    if args.hardneg_warmup_epochs<0 or args.probe_per_part<1:
        raise ValueError('invalid warmup or probe_per_part')
    if not 0 < args.reliability_floor <= 1:
        raise ValueError("reliability_floor must be in (0,1]")
    if args.reliability_on_structure and args.reliability_variant == "off":
        raise ValueError("reliability_on_structure requires a reliability_variant")
    if args.nce_variant == "taxonomy_union" and args.probe:
        raise ValueError("The legacy --probe audits global hardest negatives, not OPN negatives; "
                         "inspect taxonomy_negative_sets.csv instead")


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
    # Run the exact global NCE branch and ensure it agrees with the old formula.
    from types import SimpleNamespace
    sim_v = torch.nn.functional.normalize(torch.randn(2,3,D), dim=-1)
    sim_t = torch.nn.functional.normalize(torch.randn(original.NUM_PARTS,D),dim=-1).requires_grad_(True)
    sim_p = torch.tensor([[1,4,6],[7,11,18]],dtype=torch.long)
    common = dict(nce_temperature=0.07,nce_variant="global",label_smoothing=0.05,
                  cross_weight=0.25,hardneg_k=16,nce_clip=4.0)
    a = SimpleNamespace(**common)
    got = experimental_nce_losses(sim_v,sim_t,sim_p,None,None,a)
    ref = F.cross_entropy((sim_v.detach()@sim_t.T/0.07).reshape(-1,original.NUM_PARTS),
                          sim_p.reshape(-1),reduction="none").reshape(2,3)
    assert torch.equal(got,ref), "baseline NCE must match original bitwise"
    # Object-only NCE: same-object, all taxonomy parts, no cross-object negatives.
    # Test on the real taxonomy when available via the standalone verification script.
    for variant in ("smooth","clip","hardneg"):
        a.nce_variant=variant
        x=experimental_nce_losses(sim_v,sim_t,sim_p,None,None,a)
        assert torch.isfinite(x).all() and x.shape==(2,3)
    a.nce_variant="crossweight"
    _OBJECT_TAXONOMY_MASKS.clear()
    _OBJECT_TAXONOMY_MASKS["test"] = torch.ones(original.NUM_PARTS,dtype=torch.bool)
    class Ex:
        class_name="test"
    obj=experimental_nce_losses(sim_v,sim_t,sim_p,[Ex(),Ex()],[0,1],a)
    assert torch.allclose(obj,ref,atol=1e-6),"crossweight with all local=global"
    _OBJECT_TAXONOMY_MASKS.clear()
    for mode in ("positive","topk","fallback_k","softgate","marginpower"):
        x,_=soft_relative_prototypes(text[pids],tokens,foreground,temperature=0.05,mode=mode)
        assert torch.isfinite(x).all() and x.shape==(1,3,D)
    print("SELF_TEST_PASS_PIPELINE_SEARCH")


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
    global _PART_NAMES, _SUFFIX_CODES
    _PART_NAMES = list(names)
    global _TAXONOMY_NEG_MASK, _TAXONOMY_CLASS_NAMES, _OBJECT_ONLY_NEG_MASK
    _TAXONOMY_CLASS_NAMES = list(names)
    _TAXONOMY_NEG_MASK = build_taxonomy_negative_mask(list(names))
    _OBJECT_ONLY_NEG_MASK = build_object_only_negative_mask(list(names))
    suffixes = {name.split("'s ",1)[1] for name in names}
    suffix_id={name:i for i,name in enumerate(sorted(suffixes))}
    _SUFFIX_CODES=torch.tensor([suffix_id[name.split("'s ",1)[1]] for name in names],dtype=torch.long)
    _OBJECT_TAXONOMY_MASKS.clear()
    for obj, ids in taxonomy.items():
        mask = torch.zeros(original.NUM_PARTS, dtype=torch.bool)
        mask[ids] = True
        _OBJECT_TAXONOMY_MASKS[obj] = mask
    # Frequency balancing is derived only from image-level Presence IDs in the
    # original training cache. No spatial GT is used or loaded here.
    counts = torch.zeros(original.NUM_PARTS,dtype=torch.float64)
    for ex in examples:
        counts[ex.pids.long()] += 1
    present = counts>0
    inv = torch.zeros_like(counts)
    if present.any():
        inv[present] = counts[present].clamp_min(1).rsqrt()
        inv[present] /= ((inv[present]*counts[present]).sum()/counts[present].sum()).clamp_min(1e-12)
        inv[present] = inv[present].clamp(0.33,3.0)
    inv[~present] = 1.0
    global _PART_FREQ_WEIGHTS
    _PART_FREQ_WEIGHTS=inv.float()

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
    args.probe_epochs = sorted(set(int(v) for v in args.probe_epochs.split(',') if v.strip()))
    # Disabled probes must not reject short --epochs speed tests.
    if args.probe and any(v<0 or v>args.epochs for v in args.probe_epochs):
        raise ValueError('probe_epochs must fit training length, with 0 allowed')
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
