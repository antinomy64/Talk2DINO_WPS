#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only PP-116 gradient-interaction audit for Soft Relative + Structure + Absent.

Uses the EXACT batch_forward() / stop-gradient conventions from
train_pp116_softsingle_relative_mil.py, frozen feature bank + crop cache, and
saved orthogonal W. No optimizer steps, no pseudo-label spatial GT, no training.

Output: per_batch.csv, summary.csv, summary.json.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import train_relproto_alignemt as original
import train_pp116_softsingle_relative_mil as model


def load_pt(p):
    try:
        return torch.load(str(p), map_location='cpu', weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError):
        return torch.load(str(p), map_location='cpu', weights_only=False)


def cosine(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    den = a.norm() * b.norm()
    return float((a @ b / den).item()) if den.item() > 1e-15 else float('nan')


def tangent(W, G):
    # Orthogonal group tangent space, Frobenius metric.
    WT_G = W.T @ G
    sym = 0.5 * (WT_G + WT_G.T)
    return G - W @ sym


def compact_means(rows, field):
    v = [float(r[field]) for r in rows if math.isfinite(float(r[field]))]
    return (sum(v)/len(v), len(v)) if v else (float('nan'), 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--out-dir', default='output/pp116_absrelative_gradient_audit')
    p.add_argument('--arms', default='noabs,l0p5,l1,l8')
    p.add_argument('--epochs', default='1,30,100')
    p.add_argument('--samples', type=int, default=128, help='Fixed seeded crops, 0=all')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cache', default='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth')
    p.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    args = p.parse_args()
    root = Path(args.root).resolve()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.batch_size < 1 or args.samples < 0:
        raise ValueError('Invalid sample count or batch size')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False

    specs = {
        'noabs': ('pp116_softrelative_structure_reproduce_100ep', 0.0),
        'l0p5': ('pp116_absrelative_l0p5_sg_seed123_100ep', 0.5),
        'l1': ('pp116_absrelative_l1_sg_seed123_100ep', 1.0),
        'l8': ('pp116_absrelative_l8_sg_seed123_100ep', 8.0),
    }
    arms = [x.strip() for x in args.arms.split(',') if x.strip()]
    epochs = [int(x.strip()) for x in args.epochs.split(',') if x.strip()]
    if any(a not in specs for a in arms) or not epochs:
        raise ValueError(f'Unknown arms or empty epochs: {arms} {epochs}')
    for arm in arms:
        for ep in epochs:
            c = root / 'final_exp/relproto_alignment' / specs[arm][0] / f'W_epoch_{ep:03d}.pt'
            if not c.is_file():
                raise FileNotFoundError(c)

    print('[1/3] Loading frozen base text bank...', flush=True)
    raw, names, _ = original.load_raw_clip_bank(root/args.text_bank,
        expected_clip_model='ViT-B/16', expected_template='sub_imagenet_template')
    projector, _ = original.load_frozen_projector(
        project_root=root, config_path=root/args.model_config,
        weight_path=root/args.projector, device=device)
    with torch.no_grad():
        base = original.project_raw_clip_bank(raw, projector, device=device, batch_size=128)
        base = original.normalize_last(base.to(device=device, dtype=torch.float32)).detach()
    del projector
    if base.shape != (116, 768):
        raise RuntimeError(f'Unexpected base-bank dimensions: {tuple(base.shape)}')
    relations = (base @ base.T).detach()
    taxonomy = defaultdict(list)
    for pid, name in enumerate(names):
        taxonomy[name.split("'s ", 1)[0]].append(pid)
    model._ABS_TAXONOMY = dict(taxonomy)

    print('[2/3] Sampling identical cached crops for every checkpoint...', flush=True)
    cached = load_pt(root/args.cache)
    annotations = cached['annotations']
    order = list(range(len(annotations)))
    if args.samples:
        order = sorted(random.Random(args.seed).sample(order, min(args.samples, len(order))))
    examples = []
    for j in order:
        ann = annotations[j]
        label = str(ann['class_name'])
        pids = torch.tensor(ann['part_category_id'], dtype=torch.long)
        if not set(pids.tolist()).issubset(taxonomy[label]):
            raise RuntimeError(f'Taxonomy conflict at crop {j}')
        ex = SimpleNamespace(
            pids=pids,
            tokens=torch.as_tensor(ann['cropaug_patch_tokens'], dtype=torch.float32),
            foreground=torch.as_tensor(ann['pred_obj_mask_patch'], dtype=torch.bool),
            class_name=label,
        )
        examples.append(ex)
    cfg = SimpleNamespace(soft_temperature=.05, nce_temperature=.07,
                          lambda_struct=4.0, rank_delta=.05, rank_margin=.05,
                          lambda_abs=1.0, abs_reference_grad=False)
    batches = [examples[i:i+args.batch_size] for i in range(0,len(examples),args.batch_size)]
    print(f'  samples={len(examples)} batches={len(batches)}', flush=True)

    print('[3/3] Computing separate W gradients, with no optimizer update...', flush=True)
    records=[]
    summaries=[]
    for arm in arms:
        exp, weight = specs[arm]
        for ep in epochs:
            ckpt = load_pt(root/'final_exp/relproto_alignment'/exp/f'W_epoch_{ep:03d}.pt')
            W = torch.as_tensor(ckpt['W'], dtype=torch.float32, device=device).detach().clone().requires_grad_(True)
            orth_err = float((W.detach().T@W.detach()-torch.eye(768,device=device)).abs().max())
            one=[]
            for ib, batch in enumerate(batches):
                _, diag, separated = model.batch_forward(
                    base, W, batch, device=device, frozen_relations=relations, args=cfg)
                grads=[]
                for name in ('nce','relation','abs'):
                    if separated[name].requires_grad:
                        g = torch.autograd.grad(separated[name], W, allow_unused=True,
                                                retain_graph=name != 'abs')[0]
                    else:
                        g = None  # e.g. structure ranking has no eligible K>=3 crop.
                    if g is None:
                        g = torch.zeros_like(W)
                    grads.append(g.detach())
                gn, gs, ga = grads
                gbase = gn + 4.0*gs
                # Use the true relative coefficients from each training run.
                gweighted = weight * ga
                t_n, t_s, t_a, t_b = (tangent(W.detach(),g) for g in (gn, 4*gs, ga, gbase))
                nbase = float(t_b.norm())
                na = float(t_a.norm())
                r = dict(
                    arm=arm, lambda_abs=weight, epoch=ep, batch=ib+1, batch_n=len(batch),
                    orthogonality_max_abs=orth_err,
                    nce=float(separated['nce'].detach()),
                    structure=float(separated['relation'].detach()),
                    abs=float(separated['abs'].detach()),
                    abs_part_viol=float(diag['abs_violating_part_fraction']),
                    raw_cos_nce_abs=cosine(gn,ga),
                    raw_cos_base_abs=cosine(gbase,ga),
                    tangent_cos_nce_abs=cosine(t_n,t_a),
                    tangent_cos_structure_abs=cosine(t_s,t_a),
                    tangent_cos_base_abs=cosine(t_b,t_a),
                    tangent_nce_norm=float(t_n.norm()),
                    tangent_structure_weighted_norm=float(t_s.norm()),
                    tangent_abs_norm=na,
                    tangent_abs_weighted_over_base=(weight*na/nbase if nbase>1e-15 else float('nan')),
                )
                records.append(r)
                one.append(r)
            row = {'arm':arm,'lambda_abs':weight,'epoch':ep,'batches':len(one),'samples':len(examples),
                   'orthogonality_max_abs':orth_err}
            for field in ('nce','structure','abs','abs_part_viol',
                          'raw_cos_nce_abs','raw_cos_base_abs',
                          'tangent_cos_nce_abs','tangent_cos_structure_abs','tangent_cos_base_abs',
                          'tangent_nce_norm','tangent_structure_weighted_norm','tangent_abs_norm',
                          'tangent_abs_weighted_over_base'):
                row[field], n=compact_means(one,field)
                if field.startswith('tangent_cos'):
                    row[field+'_negative_fraction'] = (
                        sum(float(r[field])<0 for r in one if math.isfinite(float(r[field]))) / n
                        if n else float('nan'))
            summaries.append(row)
            print(f'[{arm} ep{ep:03d}] NCE={row["nce"]:.4f} '
                  f'abs={row["abs"]:.4f} cos_tangent(base,abs)={row["tangent_cos_base_abs"]:.4f} '
                  f'negative_batches={row["tangent_cos_base_abs_negative_fraction"]:.1%} '
                  f'weighted_abs/base_grad={row["tangent_abs_weighted_over_base"]:.3f}',flush=True)

    def dump_csv(name, items):
        path=out/name
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]))
            w.writeheader();w.writerows(items)
        print(f'[SAVED] {path}',flush=True)
    dump_csv('per_batch.csv',records)
    dump_csv('summary.csv',summaries)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8')
    print('[NOTE] These are instantaneous Euclidean/tangent gradients, NOT actual Adam+SVD update vectors.')
    print('[NOTE] Absent gradient uses stopgrad on the best-Present reference, exactly as trained.')

if __name__=='__main__':
    main()
