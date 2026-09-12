# -*- coding: utf-8 -*-
"""
Oracle-only GT part-prototype supervision.

This file is intentionally independent from src/loss.py so the existing
Talk2DINO / PartStruct experiments remain untouched.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GTPartPrototypeCosineLoss(nn.Module):
    """
    Directly align each projected semantic part-text vector with the matching
    GT-DINO semantic part prototype.

    The projector input is NOT normalized here. The caller must pass the raw
    CLIP prompt-mean features into projector.project_clip_txt(...), matching
    Talk2DINO eval and the RelProto T0 path.

    Both projected text and GT visual prototypes are L2-normalized only at the
    comparison boundary.

    Loss:
        mean_i [1 - cos(projected_text_i, gt_visual_proto_i)]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        projected_text: torch.Tensor,
        gt_visual_proto: torch.Tensor,
        available: torch.Tensor | None = None,
        *,
        return_per_part: bool = False,
    ):
        if projected_text.ndim != 2 or gt_visual_proto.ndim != 2:
            raise ValueError(
                "projected_text and gt_visual_proto must both be [N,D]"
            )
        if projected_text.shape != gt_visual_proto.shape:
            raise ValueError(
                f"shape mismatch: projected={tuple(projected_text.shape)} "
                f"GT={tuple(gt_visual_proto.shape)}"
            )
        if not projected_text.dtype.is_floating_point:
            raise TypeError("projected_text must be floating point")
        if not gt_visual_proto.dtype.is_floating_point:
            raise TypeError("gt_visual_proto must be floating point")
        if not torch.isfinite(projected_text).all():
            raise ValueError("projected_text contains NaN/Inf")
        if not torch.isfinite(gt_visual_proto).all():
            raise ValueError("gt_visual_proto contains NaN/Inf")

        text = F.normalize(
            projected_text.float(), p=2, dim=-1, eps=self.eps
        )
        visual = F.normalize(
            gt_visual_proto.float(), p=2, dim=-1, eps=self.eps
        )

        cosine = (text * visual).sum(dim=-1)

        if available is None:
            mask = torch.ones(
                cosine.shape[0], dtype=torch.bool, device=cosine.device
            )
        else:
            mask = available.to(
                device=cosine.device, dtype=torch.bool
            ).reshape(-1)
            if mask.shape != cosine.shape:
                raise ValueError(
                    f"available mask {tuple(mask.shape)} != "
                    f"{tuple(cosine.shape)}"
                )

        if not bool(mask.any()):
            raise ValueError("no available GT part prototypes")

        per_part_loss = 1.0 - cosine
        loss = per_part_loss[mask].mean()

        if return_per_part:
            return loss, cosine, per_part_loss
        return loss
