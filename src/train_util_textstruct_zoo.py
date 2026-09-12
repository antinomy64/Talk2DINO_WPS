# -*- coding: utf-8 -*-
"""
Generic Talk2DINO COCO InfoNCE + text-internal structure fine-tuning.

This mirrors src/train_util_partstruct_ft.py, but swaps the structure criterion.
The original repository files are not modified.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.loss import ContrastiveLoss
from src.loss_textstruct_zoo import (
    build_text_structure_loss,
    gradient_norm,
)
from src.train_util_partstruct_ft import (
    set_seed,
    const_lr,
    cosine_lr,
)


def _batch_object_loss(criterion, batch, device):
    annotations = batch["annotation"].to(device, dtype=torch.float32)
    images = batch["image"].to(device)
    text_argmax = batch.get("text_argmax")
    if text_argmax is not None:
        text_argmax = text_argmax.to(device)
    self_attn_maps = batch.get("self_attn_maps")
    cls = None
    if self_attn_maps is not None:
        self_attn_maps = self_attn_maps.to(device)
        cls = batch["dino_features"].to(device)
    text_input_mask = batch.get("text_input_mask")
    if text_input_mask is not None:
        text_input_mask = text_input_mask.to(device)

    return criterion(
        images,
        annotations,
        return_similarity_mat=False,
        self_attn_maps=self_attn_maps,
        cls=cls,
        text_input_mask=text_input_mask,
        text_argmax=text_argmax,
    )


def _make_optimizer(model, name, lr, weight_decay):
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "AdamW":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    raise ValueError(name)


def _make_scheduler(optimizer, name, lr, warmup, total_steps):
    if name == "linear" and warmup == 0:
        return None
    if name == "linear":
        return const_lr(optimizer, lr, warmup, total_steps)
    if name == "cosine":
        return cosine_lr(optimizer, lr, warmup, total_steps)
    raise ValueError(name)


def calibrate_structure_weight(
    model,
    method_criterion,
    reference_criterion,
    base_weight,
):
    """
    Match the initial weighted structure-gradient norm to the repository's
    current object_rank loss at base_weight.

    effective_lambda * ||grad L_method||
      = base_weight * ||grad L_object_rank||
    """
    model.eval()

    ref_loss = reference_criterion(model)
    ref_grad = gradient_norm(ref_loss, model)

    method_loss = method_criterion(model)
    method_grad = gradient_norm(method_loss, model)

    if not math.isfinite(ref_grad) or ref_grad <= 0:
        raise RuntimeError(f"invalid reference grad norm: {ref_grad}")
    if not math.isfinite(method_grad) or method_grad <= 0:
        raise RuntimeError(f"invalid method grad norm: {method_grad}")

    factor = ref_grad / method_grad
    effective = float(base_weight) * factor

    return {
        "reference_loss": float(ref_loss.detach().item()),
        "reference_grad_norm": float(ref_grad),
        "method_loss": float(method_loss.detach().item()),
        "method_grad_norm": float(method_grad),
        "scale_factor": float(factor),
        "base_structure_weight": float(base_weight),
        "effective_structure_weight": float(effective),
    }


def do_train_textstruct(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    *,
    method,
    structure_bank,
    base_structure_weight=1e-4,
    calibrate_grad=True,
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
    optimizer_name="Adam",
    weight_decay=0.05,
    scheduler_name="linear",
    warmup=0,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = float(train_cfg["lr"])
    ltype = train_cfg["ltype"]
    num_epochs = int(train_cfg["num_epochs"])
    batch_size = int(train_cfg["batch_size"])
    margin = train_cfg.get("margin", 0.2)
    max_violation = train_cfg.get("max_violation", True)
    shuffle = train_cfg.get("shuffle", True)
    save_best_model = train_cfg.get("save_best_model", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    contrastive = ContrastiveLoss(
        model,
        margin=margin,
        max_violation=max_violation,
        ltype=ltype,
    )

    kwargs = dict(
        rank_temperature=rank_temperature,
        neighbor_temperature=neighbor_temperature,
        confidence_delta=confidence_delta,
        confidence_alpha=confidence_alpha,
        confidence_margin_max=confidence_margin_max,
        confidence_max_triplets_per_anchor=confidence_max_triplets_per_anchor,
        rkd_angle_triplets=rkd_angle_triplets,
        topk=topk,
        topk_margin=topk_margin,
        seed=seed,
    )
    structure = build_text_structure_loss(
        method, structure_bank, **kwargs
    ).to(device)

    # Reference is ALWAYS the repository's corrected current object-rank.
    reference = build_text_structure_loss(
        "object_rank", structure_bank, **kwargs
    ).to(device)

    if calibrate_grad:
        calibration = calibrate_structure_weight(
            model, structure, reference, base_structure_weight
        )
        structure_weight = calibration["effective_structure_weight"]
    else:
        structure_weight = float(base_structure_weight)
        with torch.no_grad():
            calibration = {
                "base_structure_weight": float(base_structure_weight),
                "effective_structure_weight": float(structure_weight),
                "calibration_disabled": True,
                "initial_method_loss": float(structure(model).item()),
                "initial_reference_loss": float(reference(model).item()),
            }

    print("[TextStructZoo calibration] " + json.dumps(
        calibration, sort_keys=True
    ))
    print(
        "[TextStructZoo] "
        f"method={method} lr={lr} epochs={num_epochs} "
        f"base_lambda={base_structure_weight:.8g} "
        f"effective_lambda={structure_weight:.8g}"
    )

    optimizer = _make_optimizer(
        model, optimizer_name, lr, weight_decay
    )
    scheduler = _make_scheduler(
        optimizer,
        scheduler_name,
        lr,
        warmup,
        len(train_loader) * num_epochs,
    )

    history = []
    best_model = None
    best_val_total = None

    for epoch in range(num_epochs):
        model.train()
        sum_total = sum_obj = sum_struct = 0.0
        n_train = 0
        prev_iter = epoch * len(train_loader)

        for n_batch, batch in enumerate(train_loader):
            if scheduler is not None:
                scheduler(n_batch + prev_iter)

            object_loss = _batch_object_loss(
                contrastive, batch, device
            )
            structure_loss = structure(model)
            total = object_loss + structure_weight * structure_loss

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            sum_total += float(total.detach().item())
            sum_obj += float(object_loss.detach().item())
            sum_struct += float(structure_loss.detach().item())
            n_train += 1

        model.eval()
        with torch.no_grad():
            val_struct = float(structure(model).item())
            sum_vobj = 0.0
            n_val = 0
            for batch in val_loader:
                obj = _batch_object_loss(
                    contrastive, batch, device
                )
                sum_vobj += float(obj.item())
                n_val += 1

        train_total = sum_total / max(n_train, 1)
        train_obj = sum_obj / max(n_train, 1)
        train_struct = sum_struct / max(n_train, 1)
        val_obj = sum_vobj / max(n_val, 1)
        val_total = val_obj + structure_weight * val_struct

        row = {
            "epoch": epoch,
            "train_total": train_total,
            "train_object": train_obj,
            "train_structure": train_struct,
            "val_total": val_total,
            "val_object": val_obj,
            "val_structure": val_struct,
            "effective_structure_weight": structure_weight,
        }
        history.append(row)
        print("[TextStructZoo epoch] " + json.dumps(
            row, sort_keys=True
        ))

        if (
            not save_best_model
            or best_val_total is None
            or val_total < best_val_total
        ):
            best_val_total = val_total
            best_model = deepcopy(model)

    returned = (
        model
        if not save_best_model
        else best_model
    )
    if returned is None:
        raise RuntimeError("no model produced")

    return returned, {
        "method": method,
        "calibration": calibration,
        "effective_structure_weight": structure_weight,
        "history": history,
        "best_val_total": best_val_total,
    }
