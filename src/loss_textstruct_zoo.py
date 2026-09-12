# -*- coding: utf-8 -*-
"""
Text-internal structural losses for Talk2DINO projector fine-tuning.

All supervision targets in this file are computed ONLY from the fixed raw CLIP
part-text bank. No image features, GT masks, GT prototypes, RelProto, or W are
used by these losses.

Student path always matches Talk2DINO/RelProto:
    raw CLIP prompt mean -> projector.project_clip_txt(raw) -> L2 normalize
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.loss import PartStructureRankLoss


def _load_bank(path):
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(str(path), map_location="cpu")

    if data.get("normalized", False) is not False:
        raise ValueError("Expected raw CLIP text bank with normalized=False")

    raw = data.get("features", data.get("raw_features"))
    if raw is None:
        raise KeyError("text bank must contain features or raw_features")
    raw = torch.as_tensor(raw, dtype=torch.float32)
    if tuple(raw.shape) != (116, 512):
        raise ValueError(
            f"expected current ViT-B/16 raw PascalPart116 bank [116,512], "
            f"got {tuple(raw.shape)}"
        )
    if not torch.isfinite(raw).all() or (raw.norm(dim=-1) <= 0).any():
        raise ValueError("raw text bank contains non-finite/zero rows")

    clip_model = str(data.get("clip_model", ""))
    if clip_model and clip_model != "ViT-B/16":
        raise ValueError(f"expected clip_model='ViT-B/16', got {clip_model!r}")
    template = str(data.get("template_set", ""))
    if template and template != "sub_imagenet_template":
        raise ValueError(
            f"expected template_set='sub_imagenet_template', got {template!r}"
        )
    stage = str(data.get("stage", ""))
    if stage and stage != "pre_projector_prompt_mean":
        raise ValueError(
            f"expected stage='pre_projector_prompt_mean', got {stage!r}"
        )

    groups = data.get("object_groups")
    if groups is None:
        names = data.get("class_names", data.get("classnames", data.get("names")))
        if names is None:
            raise KeyError("text bank missing object_groups and class names")
        groups = defaultdict(list)
        for i, name in enumerate(names):
            name = str(name)
            if "'s " not in name:
                raise ValueError(f"cannot infer object from {name!r}")
            groups[name.split("'s ", 1)[0]].append(i)
        groups = dict(groups)
    else:
        groups = {str(k): [int(x) for x in v] for k, v in groups.items()}

    return raw.contiguous(), groups, data


def _soft_rank(values, temperature):
    if values.ndim != 1:
        raise ValueError("soft-rank values must be 1-D")
    diff = (values[:, None] - values[None, :]) / float(temperature)
    return 1.0 + torch.sigmoid(diff).sum(dim=1) - 0.5


def _corr_loss(source, target, eps=1e-6):
    source = source - source.mean()
    target = target - target.mean()
    source = source / source.norm().clamp_min(eps)
    target = target / target.norm().clamp_min(eps)
    return 1.0 - (source * target).sum()


def _student_features(projector, raw):
    # IMPORTANT: no normalization before projector.
    return F.normalize(
        projector.project_clip_txt(raw).float(), dim=-1, eps=1e-12
    )


class _TextStructureBase(nn.Module):
    def __init__(self, raw_features, object_groups, eps=1e-6):
        super().__init__()
        raw = torch.as_tensor(raw_features, dtype=torch.float32).detach().clone()
        self.register_buffer("raw_part_features", raw.contiguous())
        raw_u = F.normalize(raw, dim=-1, eps=1e-12)
        self.register_buffer("raw_unit", raw_u.contiguous())
        self.register_buffer("raw_gram", (raw_u @ raw_u.T).contiguous())
        self.object_groups = {
            str(k): [int(x) for x in v] for k, v in object_groups.items()
        }
        self.eps = float(eps)

    def student(self, projector):
        return _student_features(projector, self.raw_part_features)

    @classmethod
    def bank(cls, path):
        return _load_bank(path)


class GlobalRowRankLoss(_TextStructureBase):
    """For each text row, preserve its rank over all other text rows."""

    def __init__(self, raw_features, object_groups, rank_temperature=0.05, eps=1e-6):
        super().__init__(raw_features, object_groups, eps)
        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be >0")
        self.rank_temperature = float(rank_temperature)
        n = self.raw_gram.shape[0]
        self.n = int(n)
        for i in range(n):
            mask = torch.arange(n) != i
            self.register_buffer(f"mask_{i}", mask)
            target = _soft_rank(self.raw_gram[i, mask], self.rank_temperature)
            self.register_buffer(f"target_{i}", target.detach().contiguous())

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        losses = []
        for i in range(self.n):
            mask = getattr(self, f"mask_{i}")
            target = getattr(self, f"target_{i}")
            losses.append(
                _corr_loss(
                    _soft_rank(g[i, mask], self.rank_temperature),
                    target,
                    self.eps,
                )
            )
        return torch.stack(losses).mean()


class NeighborKLLoss(_TextStructureBase):
    """
    Preserve text semantic-neighborhood distributions.

    scope='global': every row distributes over all other 115 rows.
    scope='object': every row distributes only over parts of the same object.
    """

    def __init__(
        self,
        raw_features,
        object_groups,
        temperature=0.07,
        scope="global",
        min_parts=2,
        eps=1e-6,
    ):
        super().__init__(raw_features, object_groups, eps)
        if temperature <= 0:
            raise ValueError("temperature must be >0")
        if scope not in {"global", "object"}:
            raise ValueError(scope)
        self.temperature = float(temperature)
        self.scope = scope
        self.min_parts = int(min_parts)

        if scope == "global":
            n = self.raw_gram.shape[0]
            eye = torch.eye(n, dtype=torch.bool)
            self.register_buffer("global_eye", eye)
            logits = self.raw_gram / self.temperature
            logits = logits.masked_fill(eye, -torch.inf)
            q = F.softmax(logits, dim=1)
            self.register_buffer("teacher_q", q.detach().contiguous())
            self.valid_group_names = []
        else:
            self.valid_group_names = []
            gi = 0
            for name, ids0 in self.object_groups.items():
                ids = torch.as_tensor(ids0, dtype=torch.long)
                if ids.numel() < self.min_parts:
                    continue
                sub = self.raw_gram.index_select(0, ids).index_select(1, ids)
                eye = torch.eye(ids.numel(), dtype=torch.bool)
                q = F.softmax(
                    (sub / self.temperature).masked_fill(eye, -torch.inf),
                    dim=1,
                )
                self.register_buffer(f"ids_{gi}", ids.contiguous())
                self.register_buffer(f"eye_{gi}", eye)
                self.register_buffer(f"q_{gi}", q.detach().contiguous())
                self.valid_group_names.append(str(name))
                gi += 1
            if not self.valid_group_names:
                raise ValueError("no valid object groups for NeighborKLLoss")

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        if self.scope == "global":
            logits = (g / self.temperature).masked_fill(
                self.global_eye, -torch.inf
            )
            logp = F.log_softmax(logits, dim=1)
            # F.kl_div(-inf, 0) can produce NaN on the masked diagonal.
            # The diagonal has zero teacher mass, so replace only those
            # masked log-probabilities by a finite dummy value before KL.
            logp_safe = logp.masked_fill(self.global_eye, 0.0)
            return F.kl_div(logp_safe, self.teacher_q, reduction="batchmean")

        losses = []
        for gi, _ in enumerate(self.valid_group_names):
            ids = getattr(self, f"ids_{gi}")
            eye = getattr(self, f"eye_{gi}")
            q = getattr(self, f"q_{gi}")
            sub = g.index_select(0, ids).index_select(1, ids)
            logp = F.log_softmax(
                (sub / self.temperature).masked_fill(eye, -torch.inf),
                dim=1,
            )
            logp_safe = logp.masked_fill(eye, 0.0)
            losses.append(F.kl_div(logp_safe, q, reduction="batchmean"))
        return torch.stack(losses).mean()


class ConfidenceRankLoss(_TextStructureBase):
    """
    Preserve only high-confidence teacher orderings.

    Teacher triplet (i,j,k) is kept only if:
        S_raw[i,j] - S_raw[i,k] > delta

    Student must satisfy:
        S_proj[i,j] - S_proj[i,k] >= min(margin_max, alpha * teacher_gap)

    Once the margin is satisfied, that triplet contributes zero loss.
    """

    def __init__(
        self,
        raw_features,
        object_groups,
        scope="object",
        delta=0.03,
        alpha=0.5,
        margin_max=0.10,
        max_triplets_per_anchor=256,
        min_parts=3,
        eps=1e-6,
    ):
        super().__init__(raw_features, object_groups, eps)
        if scope not in {"object", "global"}:
            raise ValueError(scope)
        if delta < 0 or alpha <= 0 or margin_max <= 0:
            raise ValueError("invalid confidence-rank hyperparameters")
        self.scope = scope
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.margin_max = float(margin_max)
        self.max_triplets_per_anchor = int(max_triplets_per_anchor)

        trip_i, trip_j, trip_k, trip_margin = [], [], [], []

        def add_anchor(anchor, candidates):
            cand = [int(x) for x in candidates if int(x) != int(anchor)]
            if len(cand) < 2:
                return
            s = self.raw_gram[int(anchor), cand]
            local = []
            for a in range(len(cand)):
                for b in range(len(cand)):
                    if a == b:
                        continue
                    gap = float((s[a] - s[b]).item())
                    if gap > self.delta:
                        local.append(
                            (
                                gap,
                                int(anchor),
                                cand[a],
                                cand[b],
                                min(self.margin_max, self.alpha * gap),
                            )
                        )
            # Strongest teacher orderings first; deterministic.
            local.sort(key=lambda x: (-x[0], x[2], x[3]))
            if self.max_triplets_per_anchor > 0:
                local = local[: self.max_triplets_per_anchor]
            for _, i, j, k, m in local:
                trip_i.append(i); trip_j.append(j); trip_k.append(k)
                trip_margin.append(m)

        if scope == "global":
            all_ids = list(range(self.raw_gram.shape[0]))
            for i in all_ids:
                add_anchor(i, all_ids)
        else:
            for _, ids in self.object_groups.items():
                if len(ids) < min_parts:
                    continue
                for i in ids:
                    add_anchor(i, ids)

        if not trip_i:
            raise ValueError("confidence-rank generated zero triplets")
        self.register_buffer("trip_i", torch.tensor(trip_i, dtype=torch.long))
        self.register_buffer("trip_j", torch.tensor(trip_j, dtype=torch.long))
        self.register_buffer("trip_k", torch.tensor(trip_k, dtype=torch.long))
        self.register_buffer(
            "trip_margin", torch.tensor(trip_margin, dtype=torch.float32)
        )
        self.triplet_count = len(trip_i)

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        student_gap = g[self.trip_i, self.trip_j] - g[self.trip_i, self.trip_k]
        return F.relu(self.trip_margin - student_gap).mean()


class RKDDistanceLoss(_TextStructureBase):
    """RKD-style normalized pairwise distance preservation."""

    def __init__(self, raw_features, object_groups, eps=1e-6):
        super().__init__(raw_features, object_groups, eps)
        n = self.raw_unit.shape[0]
        tri = torch.triu_indices(n, n, offset=1)
        self.register_buffer("tri_r", tri[0])
        self.register_buffer("tri_c", tri[1])
        d = torch.linalg.vector_norm(
            self.raw_unit[tri[0]] - self.raw_unit[tri[1]], dim=-1
        )
        d = d / d.mean().clamp_min(self.eps)
        self.register_buffer("teacher_distance", d.detach().contiguous())

    def forward(self, projector):
        z = self.student(projector)
        d = torch.linalg.vector_norm(
            z[self.tri_r] - z[self.tri_c], dim=-1
        )
        d = d / d.mean().clamp_min(self.eps)
        return F.smooth_l1_loss(d, self.teacher_distance)


class RKDAngleLoss(_TextStructureBase):
    """
    Sampled RKD-style triplet-angle preservation.

    Angles are computed independently in teacher/student spaces, so feature
    dimensions may differ (512 vs 768).
    """

    def __init__(
        self,
        raw_features,
        object_groups,
        num_triplets=16384,
        seed=123,
        eps=1e-6,
    ):
        super().__init__(raw_features, object_groups, eps)
        n = self.raw_unit.shape[0]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        triplets = []
        seen = set()
        target_n = min(int(num_triplets), n * (n - 1) * (n - 2))
        while len(triplets) < target_n:
            x = torch.randperm(n, generator=gen)[:3].tolist()
            t = (int(x[0]), int(x[1]), int(x[2]))
            if len(set(t)) < 3 or t in seen:
                continue
            seen.add(t)
            triplets.append(t)

        t = torch.tensor(triplets, dtype=torch.long)
        self.register_buffer("ti", t[:, 0])
        self.register_buffer("tj", t[:, 1])
        self.register_buffer("tk", t[:, 2])

        a = F.normalize(
            self.raw_unit[self.ti] - self.raw_unit[self.tj],
            dim=-1,
            eps=self.eps,
        )
        b = F.normalize(
            self.raw_unit[self.tk] - self.raw_unit[self.tj],
            dim=-1,
            eps=self.eps,
        )
        target = (a * b).sum(dim=-1)
        self.register_buffer("teacher_angle", target.detach().contiguous())
        self.triplet_count = len(triplets)

    def forward(self, projector):
        z = self.student(projector)
        a = F.normalize(z[self.ti] - z[self.tj], dim=-1, eps=self.eps)
        b = F.normalize(z[self.tk] - z[self.tj], dim=-1, eps=self.eps)
        angle = (a * b).sum(dim=-1)
        return F.smooth_l1_loss(angle, self.teacher_angle)


class GramMSELoss(_TextStructureBase):
    """Strong control: preserve every global off-diagonal cosine value."""

    def __init__(self, raw_features, object_groups, eps=1e-6):
        super().__init__(raw_features, object_groups, eps)
        n = self.raw_gram.shape[0]
        tri = torch.triu_indices(n, n, offset=1)
        self.register_buffer("tri_r", tri[0])
        self.register_buffer("tri_c", tri[1])
        self.register_buffer(
            "teacher_values",
            self.raw_gram[tri[0], tri[1]].detach().contiguous(),
        )

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        return F.mse_loss(g[self.tri_r, self.tri_c], self.teacher_values)


class GramCorrelationLoss(GramMSELoss):
    """Preserve global pairwise-cosine shape, not exact magnitudes."""

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        return _corr_loss(
            g[self.tri_r, self.tri_c], self.teacher_values, self.eps
        )


class TopKNeighborMarginLoss(_TextStructureBase):
    """Preserve only raw-text top-k neighborhood topology."""

    def __init__(
        self,
        raw_features,
        object_groups,
        topk=5,
        margin=0.03,
        eps=1e-6,
    ):
        super().__init__(raw_features, object_groups, eps)
        n = self.raw_gram.shape[0]
        if not (1 <= topk < n - 1):
            raise ValueError("topk out of range")
        self.topk = int(topk)
        self.margin = float(margin)
        eye = torch.eye(n, dtype=torch.bool)
        masked = self.raw_gram.masked_fill(eye, -torch.inf)
        top_idx = masked.topk(k=self.topk, dim=1).indices
        negmask = ~eye
        negmask.scatter_(1, top_idx, False)
        self.register_buffer("top_idx", top_idx.contiguous())
        self.register_buffer("negmask", negmask.contiguous())

    def forward(self, projector):
        z = self.student(projector)
        g = z @ z.T
        pos = g.gather(1, self.top_idx).mean(dim=1)
        hardest_neg = g.masked_fill(~self.negmask, -torch.inf).amax(dim=1)
        return F.relu(self.margin - pos + hardest_neg).mean()


METHODS = (
    "object_rank",
    "global_row_rank",
    "object_neighbor_kl",
    "global_neighbor_kl",
    "object_confidence_rank",
    "global_confidence_rank",
    "rkd_distance",
    "rkd_angle",
    "gram_mse",
    "gram_corr",
    "topk_margin",
)


def build_text_structure_loss(
    method,
    structure_bank,
    *,
    rank_temperature=0.05,
    neighbor_temperature=0.07,
    confidence_delta=0.03,
    confidence_alpha=0.5,
    confidence_margin_max=0.10,
    confidence_max_triplets_per_anchor=256,
    rkd_angle_triplets=16384,
    topk=5,
    topk_margin=0.03,
    seed=123,
):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; valid={METHODS}")

    raw, groups, _ = _load_bank(structure_bank)

    if method == "object_rank":
        # Reuse the repository's corrected implementation exactly.
        return PartStructureRankLoss(
            raw_part_features=raw,
            object_groups=groups,
            rank_temperature=rank_temperature,
            min_parts=3,
        )

    if method == "global_row_rank":
        return GlobalRowRankLoss(
            raw, groups, rank_temperature=rank_temperature
        )
    if method == "object_neighbor_kl":
        return NeighborKLLoss(
            raw, groups, temperature=neighbor_temperature, scope="object"
        )
    if method == "global_neighbor_kl":
        return NeighborKLLoss(
            raw, groups, temperature=neighbor_temperature, scope="global"
        )
    if method == "object_confidence_rank":
        return ConfidenceRankLoss(
            raw, groups,
            scope="object",
            delta=confidence_delta,
            alpha=confidence_alpha,
            margin_max=confidence_margin_max,
            max_triplets_per_anchor=confidence_max_triplets_per_anchor,
        )
    if method == "global_confidence_rank":
        return ConfidenceRankLoss(
            raw, groups,
            scope="global",
            delta=confidence_delta,
            alpha=confidence_alpha,
            margin_max=confidence_margin_max,
            max_triplets_per_anchor=confidence_max_triplets_per_anchor,
        )
    if method == "rkd_distance":
        return RKDDistanceLoss(raw, groups)
    if method == "rkd_angle":
        return RKDAngleLoss(
            raw, groups, num_triplets=rkd_angle_triplets, seed=seed
        )
    if method == "gram_mse":
        return GramMSELoss(raw, groups)
    if method == "gram_corr":
        return GramCorrelationLoss(raw, groups)
    if method == "topk_margin":
        return TopKNeighborMarginLoss(
            raw, groups, topk=topk, margin=topk_margin
        )
    raise AssertionError(method)


def gradient_norm(loss, model):
    """Unweighted parameter-gradient norm of one scalar structure loss."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
    model.zero_grad(set_to_none=True)
    return math.sqrt(sq)
