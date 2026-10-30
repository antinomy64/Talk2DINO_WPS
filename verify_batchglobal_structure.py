#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests (no data required) and optional official Ours exact-parity test."""
from __future__ import annotations

import argparse
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from pp116_batchglobal_relation import cross_crop_rank_loss


def check_relation_unit_tests() -> None:
    torch.manual_seed(20261030)
    D, N = 8, 5
    # Three K=1 crops. The only possible valid ranking triple must be cross-Crop.
    ids = torch.tensor([0, 1, 2], dtype=torch.long)
    crops = torch.tensor([0, 1, 2], dtype=torch.long)
    objects = torch.tensor([0, 1, 2], dtype=torch.long)
    # Text says: for anchor 0, label 1 is substantially closer than 2.
    G = torch.eye(N)
    G[0,1] = G[1,0] = 0.8
    G[0,2] = G[2,0] = 0.1
    G[1,2] = G[2,1] = 0.2
    # Deliberately violate this visual ordering.
    V = torch.tensor([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0.],
    ], requires_grad=True)
    r = cross_crop_rank_loss(V, ids, crops, objects, G, n_crops=3,
                             delta=.05, margin=.05)
    assert r.loss_per_crop.shape == (3,), r.loss_per_crop
    assert r.active_fraction_per_crop[0] > 0
    assert r.loss_per_crop[0].item() > 0.1
    assert r.valid_triplet_total.item() > 0
    g = torch.autograd.grad(r.loss_per_crop.mean(), V)[0]
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 1e-8
    print("PASS: K=1 crops contribute cross-crop triplets and gradients")

    # All anchors share same crop: local constraint handled separately, global=0.
    V2 = V.detach().clone().requires_grad_(True)
    r2 = cross_crop_rank_loss(V2, ids, torch.zeros_like(crops), objects,
                              G, n_crops=1, delta=.05, margin=.05)
    assert r2.valid_triplet_total.item() == 0
    assert r2.loss_per_crop.sum().item() == 0
    assert torch.autograd.grad(r2.loss_per_crop.sum(), V2)[0] is not None
    print("PASS: cross-crop-only mask excludes purely local triples")

    # Duplicates from separate crops cannot be treated as distinct semantic IDs.
    ids_duplicate = torch.tensor([0, 1, 1], dtype=torch.long)
    r3 = cross_crop_rank_loss(V2, ids_duplicate, crops, objects, G,
                              n_crops=3, delta=0.0, margin=.05)
    assert r3.valid_triplet_total.item() == 0
    print("PASS: repeated part classes excluded from distinct-class triplets")

    # 2 nodes can have no triplet, but remain connected to the prototype graph.
    V_short = V[:2].detach().clone().requires_grad_(True)
    r4 = cross_crop_rank_loss(V_short, ids[:2], crops[:2], objects[:2], G,
                              n_crops=2, delta=.05, margin=.05)
    assert r4.loss_per_crop.sum().item() == 0
    assert torch.autograd.grad(r4.loss_per_crop.sum(), V_short)[0] is not None
    print("PASS: B<3 triplets safely return zero differentiable loss")

    # Nonzero alpha mixture changes gradients, and pair frequency balancing finite.
    ids_rep = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    crops_rep = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    objs_rep = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    V_rep = F.normalize(torch.randn(4, D), dim=-1).requires_grad_()
    rb = cross_crop_rank_loss(V_rep, ids_rep, crops_rep, objs_rep, G,
                              n_crops=4, delta=0, margin=.5, class_balance=True)
    ru = cross_crop_rank_loss(V_rep, ids_rep, crops_rep, objs_rep, G,
                              n_crops=4, delta=0, margin=.5, class_balance=False)
    assert torch.isfinite(rb.loss_per_crop).all()
    assert torch.isfinite(ru.loss_per_crop).all()
    print("PASS: repeated part class pair-frequency balance / no-balance both valid")


def check_full_parity() -> None:
    """Needs the real repo; cross-checks the trainer's official alpha=0 path."""
    import train_pp116_softsingle_dual as ours
    import train_pp116_batchglobal_structure as modified
    import train_relproto_alignemt as original

    torch.manual_seed(1234)
    D, P = original.VISION_DIM, original.DEFAULT_PATCH_COUNT
    bank = F.normalize(torch.randn(original.NUM_PARTS, D), dim=-1)
    W = torch.nn.Parameter(torch.eye(D))
    examples = []
    for c, ps in [("cow", [49, 50, 51]), ("cat", [40, 41, 42]), ("dog", [59])]:
        tk = F.normalize(torch.randn(P, D), dim=-1)
        fg = torch.zeros(P, dtype=torch.bool)
        fg[:96] = True
        examples.append(SimpleNamespace(tokens=tk, foreground=fg,
                                        pids=torch.tensor(ps, dtype=torch.long), class_name=c))
    frozen_rel = (bank @ bank.T).detach()
    args = SimpleNamespace(
        global_structure_alpha=0.0, global_class_balance=True,
        soft_temperature=0.05, nce_temperature=0.07,
        lambda_struct=4.0, rank_delta=0.05, rank_margin=0.05,
    )
    a, ia, sa = ours.batch_forward(bank, W, examples, device=torch.device("cpu"),
                                    frozen_relations=frozen_rel, args=args)
    b, ib, sb = modified.batch_forward(bank, W, examples, device=torch.device("cpu"),
                                        frozen_relations=frozen_rel, args=args)
    assert torch.equal(a, b), (a, b)
    for k in ia:
        assert ia[k] == ib[k], (k, ia[k], ib[k])
    ga = torch.autograd.grad(a, W, retain_graph=True)[0]
    gb = torch.autograd.grad(b, W, retain_graph=True)[0]
    assert torch.equal(ga, gb), (ga-gb).abs().max().item()
    print("PASS: alpha=0 EXACT Ours total loss, diagnostics and W gradients")

    args.global_structure_alpha = 0.25
    c, ic, sc = modified.batch_forward(bank, W, examples, device=torch.device("cpu"),
                                        frozen_relations=frozen_rel, args=args)
    assert abs(ic["loss_nce"] - ia["loss_nce"]) <= 1e-6
    assert abs(ic["loss_local_relation"] - ia["loss_relation"]) <= 1e-6
    assert abs(ic["loss_relation"] - (0.75*ic["loss_local_relation"] +
                                      0.25*ic["loss_global_relation"])) <= 1e-6
    assert ic["global_structure_active_fraction"] > 0, ic
    grad = torch.autograd.grad(sc["global_relation"], W, retain_graph=True)[0]
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.norm().item() > 1e-9
    print("PASS: alpha=.25 unchanged NCE/local + nonzero global Relation gradient to W")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-parity", action="store_true",
                   help="additional official Ours check; run from actual repository")
    args = p.parse_args()
    check_relation_unit_tests()
    if args.full_parity:
        check_full_parity()
    print("BATCHGLOBAL_STRUCTURE_TESTS_PASS")


if __name__ == "__main__":
    main()
