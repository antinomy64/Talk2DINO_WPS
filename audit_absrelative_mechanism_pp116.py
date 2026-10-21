#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, apples-to-apples offline audit of PP-116 absent-relative mechanism.

Compares saved W checkpoints on the EXACT SAME cached predicted-object crops.
No training; no GT in these metrics; no edits/downloads to official repo.

Metrics match training semantics: same-object P+=part_category_id,
P-=object full taxonomy minus P+, Talk2DINO predicted foreground,
cosine margin, per-absent worst positive margin, equal-crop averaging.
Also reports per-patch absent winner and Soft Relative effective supports.

Place as a NEW file in Talk2DINO_official_bg repo root alongside
train_relproto_alignemt.py and run using the provided command.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import train_relproto_alignemt as original


def load_pt(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError):
        return torch.load(str(path), map_location='cpu', weights_only=False)


def resolve(root, name):
    p = Path(name).expanduser()
    return p if p.is_absolute() else root / p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--cache', default='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth')
    parser.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    parser.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    parser.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    parser.add_argument('--out-dir', default='output/absrelative_mechanism_audit_e100')
    parser.add_argument('--max-crops', type=int, default=0, help='0=all 8379; >0=seeded subsample')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--soft-temperature', type=float, default=0.05)
    parser.add_argument('--checkpoint', action='append', metavar='TAG=PATH', help='Override default W files; repeat for variants')
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = resolve(root, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu')
    if args.max_crops < 0 or args.soft_temperature <= 0:
        raise ValueError('invalid max-crops/soft-temperature')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False

    specs = {
        'noabs': 'final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep/W_epoch_100.pt',
        'abs0p5': 'final_exp/relproto_alignment/pp116_absrelative_l0p5_sg_seed123_100ep/W_epoch_100.pt',
        'abs1': 'final_exp/relproto_alignment/pp116_absrelative_l1_sg_seed123_100ep/W_epoch_100.pt',
        'abs8': 'final_exp/relproto_alignment/pp116_absrelative_l8_sg_seed123_100ep/W_epoch_100.pt',
    }
    if args.checkpoint:
        specs = {}
        for entry in args.checkpoint:
            if '=' not in entry:
                raise ValueError('--checkpoint must be TAG=PATH')
            tag, path = entry.split('=', 1)
            if not tag or not path or tag in specs:
                raise ValueError('empty/duplicate checkpoint tag')
            specs[tag] = path

    print('[1/3] Loading exact original frozen projector and 116 CLIP features...', flush=True)
    raw, names, _ = original.load_raw_clip_bank(
        resolve(root, args.text_bank), expected_clip_model='ViT-B/16',
        expected_template='sub_imagenet_template')
    if len(names) != 116:
        raise RuntimeError('expected 116 classes')
    projector, _ = original.load_frozen_projector(
        project_root=root, config_path=resolve(root, args.model_config),
        weight_path=resolve(root, args.projector), device=device)
    with torch.no_grad():
        base = original.project_raw_clip_bank(raw, projector, device=device, batch_size=128)
        base = F.normalize(base.float().to(device), dim=-1)
    del projector
    taxonomy = defaultdict(list)
    for i, name in enumerate(names):
        if "'s " not in name:
            raise RuntimeError(f'bad taxonomy name: {name!r}')
        taxonomy[name.split("'s ", 1)[0]].append(i)

    texts = {}
    for tag, filename in specs.items():
        path = resolve(root, filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        ckpt = load_pt(path)
        if not isinstance(ckpt, dict) or 'W' not in ckpt:
            raise RuntimeError(f'{path}: expected W in checkpoint')
        W = torch.as_tensor(ckpt['W'], dtype=torch.float32, device=device)
        if W.shape != (768, 768):
            raise RuntimeError(f'{tag}: invalid W shape {tuple(W.shape)}')
        with torch.no_grad():
            texts[tag] = F.normalize(base @ W, dim=-1)
        print(f'  checkpoint {tag}: {path}', flush=True)

    print('[2/3] Loading frozen cached crops...', flush=True)
    cache = load_pt(resolve(root, args.cache))
    if not isinstance(cache, dict) or not isinstance(cache.get('annotations'), list):
        raise RuntimeError('unexpected crop-cache format')
    annotations = cache['annotations']
    if args.max_crops and args.max_crops < len(annotations):
        annotations = [annotations[i] for i in sorted(random.Random(args.seed).sample(range(len(annotations)), args.max_crops))]
    print(f'  selected crops: {len(annotations)}', flush=True)

    totals = {tag: defaultdict(float) for tag in texts}
    per_object = {tag: defaultdict(lambda: defaultdict(float)) for tag in texts}
    per_absent = {tag: defaultdict(lambda: defaultdict(float)) for tag in texts}
    with torch.no_grad():
        for cidx, crop in enumerate(annotations, start=1):
            name = str(crop['class_name'])
            full_ids = taxonomy[name]
            present_ids = [int(z) for z in crop['part_category_id']]
            if not present_ids or not set(present_ids).issubset(set(full_ids)):
                raise RuntimeError(f'bad P+ for crop {cidx}: {name} {present_ids}')
            absent_ids = [z for z in full_ids if z not in set(present_ids)]
            patches = torch.as_tensor(crop['cropaug_patch_tokens'], device=device, dtype=torch.float32)
            foreground = torch.as_tensor(crop['pred_obj_mask_patch'], device=device, dtype=torch.bool)
            patches = F.normalize(patches, dim=-1)
            if patches.shape != (1024, 768) or foreground.shape != (1024,) or not foreground.any():
                raise RuntimeError(f'invalid patch geometry crop {cidx}')
            X = patches[foreground].T.contiguous()  # [768,n_fg]
            n_fg = int(X.shape[1])
            for tag, text in texts.items():
                d = totals[tag]
                o = per_object[tag][name]
                d['crops'] += 1
                o['crops'] += 1
                sp = text[present_ids] @ X  # [K,n_fg]
                # Identical soft relative prototype weighting as trainer.
                if len(present_ids) == 1:
                    weights = F.softmax(sp / args.soft_temperature, dim=1)
                else:
                    top2 = sp.topk(2, dim=0).values
                    winner = sp.argmax(dim=0)
                    competitor = top2[0][None, :].expand_as(sp).clone()
                    competitor.scatter_(0, winner[None], top2[1][None])
                    relative = sp - competitor
                    eligible = relative > 0
                    missing = ~eligible.any(dim=1)
                    if missing.any():
                        anchors = relative.argmax(dim=1)
                        eligible[missing, anchors[missing]] = True
                    weights = F.softmax((relative / args.soft_temperature).masked_fill(~eligible, -torch.inf), dim=1)
                    # Unlabeled: pseudo prototypes are independent of absent ids by design.
                    d['present_winner_margin_sum'] += float((top2[0] - top2[1]).sum().item())
                    d['present_winner_margin_count'] += n_fg
                effective = weights.square().sum(dim=1).clamp_min(1e-12).reciprocal()
                d['eff_sum'] += float(effective.sum().item())
                d['eff_parts'] += len(present_ids)
                if not absent_ids:
                    continue
                sa = text[absent_ids] @ X  # [n_abs,n_fg]
                margin = sa - sp.max(dim=0).values[None]
                violation = margin.clamp_min(0)
                worst = violation.max(dim=1).values
                n_abs = len(absent_ids)
                num_viol_parts = int((worst > 0).sum().item())
                num_viol_patch_pairs = int((margin > 0).sum().item())
                num_viol_patches = int((margin.max(dim=0).values > 0).sum().item())
                avg_loss = float(worst.mean().item())
                worst_crop = float(worst.max().item())
                for accum in (d, o):
                    accum['active_crops'] += 1
                    accum['abs_loss_sum'] += avg_loss
                    accum['abs_parts'] += n_abs
                    accum['violating_parts'] += num_viol_parts
                    accum['patch_pairs'] += n_abs * n_fg
                    accum['violating_patch_pairs'] += num_viol_patch_pairs
                    accum['foreground_patches'] += n_fg
                    accum['patches_with_any_absent_win'] += num_viol_patches
                    accum['worst_sum'] += worst_crop
                for j, pid in enumerate(absent_ids):
                    row = per_absent[tag][pid]
                    row['crops'] += 1
                    row['part_viol'] += int(worst[j].item() > 0)
                    row['part_worst_sum'] += float(worst[j].item())
                    row['patches'] += n_fg
                    row['patch_viol'] += int((margin[j] > 0).sum().item())
            if cidx % 1000 == 0:
                print(f'  processed {cidx}/{len(annotations)} crops', flush=True)

    def summarize(x, *, include_effective=False):
        outrow = {
            'crops': int(x['crops']),
            'active_crops': int(x['active_crops']),
            'abs_active_fraction': x['active_crops'] / max(1, x['crops']),
            # Training's equal-crop mean: crops with no absent classes count as zero.
            'loss_abs': x['abs_loss_sum'] / max(1, x['crops']),
            'abs_violating_part_fraction': x['violating_parts'] / max(1, x['abs_parts']),
            'abs_violating_patch_fraction': x['violating_patch_pairs'] / max(1, x['patch_pairs']),
            'abs_worst_positive_margin': x['worst_sum'] / max(1, x['active_crops']),
            # A complementary metric: fraction of foreground patches where *any*
            # absent part beats all Present parts.
            'any_absent_wins_patch_fraction': x['patches_with_any_absent_win'] / max(1, x['foreground_patches']),
        }
        if include_effective:
            outrow['mean_effective_patch_count'] = x['eff_sum'] / max(1, x['eff_parts'])
            outrow['mean_present_top2_gap'] = x['present_winner_margin_sum'] / max(1, x['present_winner_margin_count'])
        return outrow

    print('[3/3] Writing read-only audit comparisons...', flush=True)
    results = {tag: summarize(x, include_effective=True) for tag, x in totals.items()}
    summary_path = out / 'comparison.json'
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'per_object.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'object', 'crops', 'active_crops', 'abs_active_fraction', 'loss_abs', 'abs_violating_part_fraction', 'abs_violating_patch_fraction', 'abs_worst_positive_margin', 'any_absent_wins_patch_fraction'])
        writer.writeheader()
        for tag, objects in per_object.items():
            for name, d in sorted(objects.items()):
                writer.writerow({'tag': tag, 'object': name, **summarize(d)})
    with (out / 'per_absent_part.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'part_id', 'part_name', 'crops', 'violating_part_fraction', 'mean_worst_margin', 'violating_patch_fraction'])
        writer.writeheader()
        for tag, parts in per_absent.items():
            for pid, d in sorted(parts.items()):
                writer.writerow({'tag': tag, 'part_id': pid, 'part_name': names[pid], 'crops': int(d['crops']),
                                 'violating_part_fraction': d['part_viol'] / max(1, d['crops']),
                                 'mean_worst_margin': d['part_worst_sum'] / max(1, d['crops']),
                                 'violating_patch_fraction': d['patch_viol'] / max(1, d['patches'])})
    for tag, r in results.items():
        print(f'{tag:>8}  abs={r["loss_abs"]:.5f}  part_viol={r["abs_violating_part_fraction"]:.4f}  '
              f'patch_viol={r["abs_violating_patch_fraction"]:.4f}  '
              f'abs_wins_patch={r["any_absent_wins_patch_fraction"]:.4f}  '
              f'worst={r["abs_worst_positive_margin"]:.5f}  eff={r["mean_effective_patch_count"]:.2f}', flush=True)
    print(f'Saved: {summary_path}', flush=True)
    print('Interpretation: these are TRAIN-CROP diagnostics, not validation mIoU, GT localization, or causal proof.')


if __name__ == '__main__':
    main()
