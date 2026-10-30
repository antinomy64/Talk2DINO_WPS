#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable cross-crop visual relation-ranking for PP116.

All text relations come from the frozen initial text bank. No pixel-level GT.
This module is standalone: can be unit-tested without the project repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class GlobalRelationResult:
    loss_per_crop: torch.Tensor               # [B] differentiable
    active_fraction_per_crop: torch.Tensor    # [B] no grad; proportion of active anchors
    triplets_per_crop: torch.Tensor           # [B] no grad; average number per anchor
    valid_triplet_total: torch.Tensor         # [] no grad
    cross_object_fraction: torch.Tensor       # [] no grad


def cross_crop_rank_loss(
    prototypes: torch.Tensor,       # [M,D], normalized, differentiable
    part_ids: torch.Tensor,         # [M], global PP116 part IDs
    crop_ids: torch.Tensor,         # [M], values in 0..B-1
    object_ids: torch.Tensor,       # [M], object taxonomy IDs (used only for reporting)
    frozen_relations: torch.Tensor, # [116,116], frozen text cosine
    *,
    n_crops: int,
    delta: float,
    margin: float,
    class_balance: bool = True,
) -> GlobalRelationResult:
    """Compare text vs visual pairwise *order* across Crops.

    Valid (anchor p, q, r) requires three DISTINCT PP116 class IDs,
    frozen_text[p,q] - frozen_text[p,r] > delta, and at least one of
    q/r comes from a different crop than p. This defines cross-crop-only
    triples; local triples are handled independently by the official Ours.

    Per anchor: average positive hinge over its valid (q,r) pairs.
    Per crop: average over its anchors. Final training: mean over crops.
    This retains Ours' equal-annotation weighting and includes K=1 crops.

    Optional inverse class-occurrence weighting for q,r prevents recurring
    part types in a batch from dominating the cross-crop relation target.

    The masks/text relationship and class-balance weights have no gradient.
    Gradients flow through prototypes -> soft assignment -> adapted text W.
    """
    if prototypes.ndim != 2:
        raise ValueError("prototypes must be [M,D]")
    M = prototypes.size(0)
    device, dtype = prototypes.device, prototypes.dtype
    if n_crops < 1 or M < 1:
        raise ValueError("n_crops and number of prototypes must be positive")
    if any(tuple(x.shape) != (M,) for x in (part_ids, crop_ids, object_ids)):
        raise ValueError("part_ids/crop_ids/object_ids must all have shape [M]")
    if part_ids.dtype != torch.long or crop_ids.dtype != torch.long:
        raise TypeError("part_ids and crop_ids must be torch.long")
    if frozen_relations.ndim != 2 or frozen_relations.size(0) != frozen_relations.size(1):
        raise ValueError("frozen_relations must be square")
    if part_ids.min().item() < 0 or part_ids.max().item() >= frozen_relations.size(0):
        raise ValueError("part ID outside text relation bank")
    if crop_ids.min().item() < 0 or crop_ids.max().item() >= n_crops:
        raise ValueError("crop ID outside [0,B)")

    # N >= 3 is needed for a triplet; zero loss must remain connected to W.
    if M < 3:
        zero = prototypes.sum() * 0.0
        z_b = zero.expand(n_crops)
        return GlobalRelationResult(
            z_b, z_b.detach(), z_b.detach(),
            zero.detach(), zero.detach(),
        )

    with torch.no_grad():
        same_part = part_ids[:, None] == part_ids[None, :]
        # Distinct semantic classes, not merely different crop instances.
        distinct = (~same_part[:, :, None]) & (~same_part[:, None, :]) & (~same_part[None, :, :])
        other_crop = (crop_ids[:, None] != crop_ids[None, :])
        cross_crop = other_crop[:, :, None] | other_crop[:, None, :]
        text_similarity = frozen_relations[part_ids[:, None], part_ids[None, :]]
        text_gap = text_similarity[:, :, None] - text_similarity[:, None, :]
        valid = distinct & cross_crop & (text_gap > delta)

        if class_balance:
            counts = torch.bincount(part_ids, minlength=frozen_relations.size(0))
            inverse = counts.clamp_min(1).to(dtype).reciprocal()[part_ids]
            triplet_weights = inverse[None, :, None] * inverse[None, None, :]
        else:
            triplet_weights = None

        count_by_anchor = valid.sum(dim=(1, 2)).to(dtype)
        triplet_total = valid.sum()
        cross_object = (object_ids[:, None] != object_ids[None, :])
        cross_obj_mask = cross_object[:, :, None] | cross_object[:, None, :]
        cross_object_fraction = (valid & cross_obj_mask).sum().to(dtype) / triplet_total.clamp_min(1)

    # The only differentiable relation path.
    visual_similarity = prototypes @ prototypes.T
    visual_gap = visual_similarity[:, :, None] - visual_similarity[:, None, :]
    hinge = F.relu(margin - visual_gap)
    if class_balance:
        weight = valid.to(dtype) * triplet_weights
        denom = weight.sum(dim=(1, 2))
        numer = (hinge * weight).sum(dim=(1, 2))
    else:
        denom = count_by_anchor
        numer = (hinge * valid.to(dtype)).sum(dim=(1, 2))
    per_anchor_loss = numer / denom.clamp_min(1e-12)

    # Reorder/normalize by original annotation index, even after K grouping.
    anchors_per_crop = torch.bincount(crop_ids, minlength=n_crops).clamp_min(1).to(dtype)
    per_crop_loss = torch.zeros(n_crops, device=device, dtype=dtype).index_add(
        0, crop_ids, per_anchor_loss
    ) / anchors_per_crop
    per_crop_active = torch.zeros(n_crops, device=device, dtype=dtype).index_add(
        0, crop_ids, (count_by_anchor > 0).to(dtype)
    ) / anchors_per_crop
    per_crop_triplets = torch.zeros(n_crops, device=device, dtype=dtype).index_add(
        0, crop_ids, count_by_anchor
    ) / anchors_per_crop
    return GlobalRelationResult(
        loss_per_crop=per_crop_loss,
        active_fraction_per_crop=per_crop_active.detach(),
        triplets_per_crop=per_crop_triplets.detach(),
        valid_triplet_total=triplet_total.detach(),
        cross_object_fraction=cross_object_fraction.detach(),
    )
