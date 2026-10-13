#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only PP116 PredObj cache audit: all positive relative-margin patches.

Run beside the UNMODIFIED train_relproto_alignemt.py. Reuses its exact:
  raw CLIP -> frozen Talk2DINO text projector -> normalized 116x768 bank;
  row-vector text @ W; normalized DINO patch features;
  relative_scores() and independent argmax anchor.
No training, GT masks, image decoding, network or checkpoint modification.

An annotation is ONE cached object/crop. One image_id can occur multiple times.
Reports per annotation-part and per unique (image_id, part_id) separately.
For K=1: original implementation defines relative = absolute similarity.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import train_relproto_alignemt as original


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project_root', default='.')
    p.add_argument('--train_dataset', default='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth')
    p.add_argument('--text_bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--model_config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--weights', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    p.add_argument('--w_checkpoint', default=None, help='Optional W_last.pt / W_epoch_NNN.pt; omit for identity W (epoch 0).')
    p.add_argument('--out_dir', default='final_exp/relproto_alignment/margin_patch_statistics_pp116')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--project_batch_size', type=int, default=128)
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected_bg_thresh', type=float, default=0.55)
    p.add_argument('--max_annotations', type=int, default=0, help='0=all; positive number for quick smoke test.')
    p.add_argument('--feature_name', default='cropaug_patch_tokens')
    p.add_argument('--foreground_key', default='pred_obj_mask_patch')
    p.add_argument('--part_id_key', default='part_category_id')
    p.add_argument('--part_name_key', default='part_class_name')
    p.add_argument('--clip_model', default='ViT-B/16')
    p.add_argument('--template_set', default='sub_imagenet_template')
    p.add_argument('--prototype_max_patches', type=int, default=8, help='Only for reference counts, no truncation of positive candidates.')
    p.add_argument('--require_pamr', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--require_cache_projector_match', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.batch_size <= 0 or args.project_batch_size <= 0 or args.prototype_max_patches < 1:
        p.error('batch sizes and prototype_max_patches must be >=1')
    return args


def stat(vals):
    x = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if not x.size:
        return dict(n=0, mean=None, median=None, std=None, min=None, p10=None, p25=None,
                    p75=None, p90=None, p95=None, max=None)
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return dict(n=int(x.size), mean=float(x.mean()), median=float(q[2]), std=float(x.std()),
                min=float(x.min()), p10=float(q[0]), p25=float(q[1]), p75=float(q[3]),
                p90=float(q[4]), p95=float(q[5]), max=float(x.max()))


def write_csv(path: Path, rows: list[dict], fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def scan_batch(examples, *, base_bank, W, device, class_names, max_l):
    """Same arithmetic as original.batch_forward/build_relproto, without selection cap."""
    tokens = torch.stack([e.tokens for e in examples], dim=0).to(device=device, dtype=torch.float32)
    fg = torch.stack([e.foreground for e in examples], dim=0).to(device=device, dtype=torch.bool)
    tokens = original.normalize_last(tokens)

    all_text = original.normalize_last(base_bank @ W)
    groups = defaultdict(list)
    for index, ex in enumerate(examples):
        groups[int(ex.pids.numel())].append(index)
    rows = []
    for k, member_ids in sorted(groups.items()):
        indices = torch.tensor(member_ids, dtype=torch.long, device=device)
        local_ids = torch.stack([examples[i].pids for i in member_ids]).to(device=device, dtype=torch.long)
        query = all_text[local_ids]
        absolute = torch.bmm(query, tokens[indices].transpose(1, 2))
        relative = original.relative_scores(absolute, fg[indices])
        anchor_idx = relative.argmax(dim=-1)
        anchor_margin = relative.gather(-1, anchor_idx[..., None]).squeeze(-1)
        anchor_abs = absolute.gather(-1, anchor_idx[..., None]).squeeze(-1)
        positive_mask = relative > 0  # masked background == -inf, never positive
        positive_count = positive_mask.sum(-1)
        fg_count = fg[indices].sum(-1)
        # This is the exact original support cardinality if L=max_l, with anchor retained.
        original_count = 1 + torch.clamp(
            positive_count - (anchor_margin > 0).long(),
            min=0, max=max_l - 1,
        )
        relative_positive_sum = relative.masked_fill(~positive_mask, 0.0).sum(dim=-1)
        positive_mean = relative_positive_sum / positive_count.clamp_min(1)
        # Anchor collisions remain allowed in the real trainer.
        collision = [(torch.unique(row).numel() < row.numel()) for row in anchor_idx]
        for a, original_index in enumerate(member_ids):
            ex = examples[original_index]
            for j, pid in enumerate(ex.pids.tolist()):
                npos = int(positive_count[a, j].item())
                nfg = int(fg_count[a].item())
                am = float(anchor_margin[a, j].item())
                selected_count = int(original_count[a, j].item())
                rows.append({
                    'row_index': int(ex.row_index),
                    'image_id': str(ex.image_id),
                    'ann_id': str(ex.ann_id),
                    'object_class_name': str(ex.class_name),
                    'part_id': int(pid),
                    'part_name': str(class_names[pid]),
                    'present_part_count': k,
                    'single_part_no_competitor': int(k == 1),
                    'foreground_patch_count': nfg,
                    'positive_margin_count': npos,
                    'positive_margin_fraction_of_foreground': float(npos / nfg),
                    'positive_margin_mean': float(positive_mean[a, j].item()) if npos else None,
                    'positive_margin_sum': float(relative_positive_sum[a, j].item()),
                    'anchor_patch_index': int(anchor_idx[a, j].item()),
                    'anchor_margin': am,
                    'anchor_absolute_similarity': float(anchor_abs[a, j].item()),
                    'anchor_margin_gt_zero': int(am > 0),
                    'anchor_collided_with_other_part': int(bool(collision[a])),
                    'original_max_L_count': selected_count,
                    'unbounded_anchor_plus_positive_count': npos + int(am <= 0),
                    'positive_candidates_excluding_anchor': npos - int(am > 0),
                    'gt_8': int(npos > 8),
                    'gt_32': int(npos > 32),
                })
    return rows


def aggregate_per_image(rows):
    """One (image_id, part_id) row, averaging object-crop annotations for that image."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['image_id'], r['part_id'])].append(r)
    result = []
    for (image_id, pid), group in groups.items():
        result.append({
            'image_id': image_id,
            'part_id': pid,
            'part_name': group[0]['part_name'],
            'object_crop_count': len(group),
            'mean_positive_margin_count': float(np.mean([r['positive_margin_count'] for r in group])),
            'mean_anchor_margin': float(np.mean([r['anchor_margin'] for r in group])),
            'mean_foreground_patches': float(np.mean([r['foreground_patch_count'] for r in group])),
            'mean_positive_margin_fraction': float(np.mean([r['positive_margin_fraction_of_foreground'] for r in group])),
            'mean_original_L_count': float(np.mean([r['original_max_L_count'] for r in group])),
            'single_part_crop_fraction': float(np.mean([r['single_part_no_competitor'] for r in group])),
        })
    return sorted(result, key=lambda x: (x['image_id'], x['part_id']))


def part_report(rows, image_rows, class_names):
    by_class = defaultdict(list)
    by_class_image = defaultdict(list)
    for r in rows: by_class[r['part_id']].append(r)
    for r in image_rows: by_class_image[r['part_id']].append(r)
    report = []
    for pid, name in enumerate(class_names):
        group = by_class[pid]
        igrp = by_class_image[pid]
        if not group:
            report.append({'part_id':pid, 'part_name':name, 'occurrences':0, 'unique_images':0})
            continue
        count_stats = stat([r['positive_margin_count'] for r in group])
        anchor_stats = stat([r['anchor_margin'] for r in group])
        result = {
            'part_id': pid,
            'part_name': name,
            'occurrences': len(group),
            'unique_images': len(igrp),
            'positive_mean': count_stats['mean'],
            'positive_median': count_stats['median'],
            'positive_p10': count_stats['p10'],
            'positive_p25': count_stats['p25'],
            'positive_p75': count_stats['p75'],
            'positive_p90': count_stats['p90'],
            'positive_p95': count_stats['p95'],
            'positive_min': count_stats['min'],
            'positive_max': count_stats['max'],
            'positive_mean_per_unique_image': stat([r['mean_positive_margin_count'] for r in igrp])['mean'],
            'anchor_margin_mean': anchor_stats['mean'],
            'anchor_margin_median': anchor_stats['median'],
            'anchor_margin_p10': anchor_stats['p10'],
            'anchor_margin_p90': anchor_stats['p90'],
            'anchor_margin_min': anchor_stats['min'],
            'anchor_margin_max': anchor_stats['max'],
            'anchor_abs_mean': stat([r['anchor_absolute_similarity'] for r in group])['mean'],
            'foreground_count_mean': stat([r['foreground_patch_count'] for r in group])['mean'],
            'positive_foreground_fraction_mean': stat([r['positive_margin_fraction_of_foreground'] for r in group])['mean'],
            'original_max_L_count_mean': stat([r['original_max_L_count'] for r in group])['mean'],
            'fraction_zero_positive': float(np.mean([r['positive_margin_count'] == 0 for r in group])),
            'fraction_more_than_8': float(np.mean([r['positive_margin_count'] > 8 for r in group])),
            'fraction_more_than_32': float(np.mean([r['positive_margin_count'] > 32 for r in group])),
            'fraction_anchor_margin_positive': float(np.mean([r['anchor_margin_gt_zero'] for r in group])),
            'fraction_single_part': float(np.mean([r['single_part_no_competitor'] for r in group])),
        }
        report.append(result)
    return report


def main():
    args = cli()
    root = original.resolve_root(args.project_root)
    def path(s): return original.resolve_path(root, s)
    dataset_path, bank_path, config_path, weights_path = map(
        path, (args.train_dataset, args.text_bank, args.model_config, args.weights)
    )
    out_dir = path(args.out_dir)
    for x in (dataset_path, bank_path, config_path, weights_path):
        if not x.is_file(): raise FileNotFoundError(x)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False  # mirror original training
    if args.w_checkpoint:
        w_path = path(args.w_checkpoint)
        if not w_path.is_file(): raise FileNotFoundError(w_path)
    else:
        w_path = None

    print('[1/4] Loading frozen CLIP features and frozen Talk2DINO projector...', flush=True)
    raw, names, bank_info = original.load_raw_clip_bank(
        bank_path, expected_clip_model=args.clip_model, expected_template=args.template_set
    )
    projector, proj_info = original.load_frozen_projector(
        project_root=root, config_path=config_path, weight_path=weights_path, device=device,
    )
    base_cpu = original.project_raw_clip_bank(
        raw, projector, device=device, batch_size=args.project_batch_size,
    )
    del projector
    base_bank = original.normalize_last(base_cpu.to(device=device, dtype=torch.float32))
    if w_path is None:
        W = torch.eye(original.VISION_DIM, dtype=torch.float32, device=device)
        w_desc = 'identity'
    else:
        ckpt = original.torch_load(w_path)
        if not isinstance(ckpt, dict) or 'W' not in ckpt:
            raise ValueError('W checkpoint missing tensor key "W"')
        W = ckpt['W'].detach().to(device=device, dtype=torch.float32)
        if W.shape != (original.VISION_DIM, original.VISION_DIM):
            raise ValueError(f'Unexpected W shape: {tuple(W.shape)}')
        w_desc = str(w_path)
    W = W.detach()
    print('[2/4] Loading/validating existing training cache...', flush=True)
    examples, preflight = original.prepare_examples(
        dataset_path,
        text_names=names,
        patch_key=args.feature_name,
        foreground_key=args.foreground_key,
        part_id_key=args.part_id_key,
        part_name_key=args.part_name_key,
        max_annotations=args.max_annotations,
        expected_bg_thresh=args.expected_bg_thresh,
        require_pamr=args.require_pamr,
        projector_weight_sha256=proj_info['weights_sha256'],
        require_cache_projector_match=args.require_cache_projector_match,
    )
    print(f'    {len(examples)} cached object/crop annotations ready', flush=True)
    print('[3/4] Scanning all foreground patches and their relative margins...', flush=True)
    rows = []
    for start in tqdm(range(0, len(examples), args.batch_size), desc='Margin audit', unit='batch'):
        rows.extend(scan_batch(
            examples[start:start+args.batch_size],
            base_bank=base_bank, W=W, device=device,
            class_names=names, max_l=args.prototype_max_patches,
        ))
    image_rows = aggregate_per_image(rows)
    classes = part_report(rows, image_rows, names)
    n_anno = len(examples)
    n_images = len({e.image_id for e in examples})
    stats = {
        'dataset': str(dataset_path), 'text_bank':str(bank_path),
        'projector': str(weights_path), 'W':w_desc,
        'definitions': {
            'relative': 'S[p,i] - max_(q != p) S[q,i] for K>1; S[p,i] for K=1',
            'foreground': 'pred_obj_mask_patch True',
            'positive': 'strict margin >0 inside predicted foreground, no candidate cap',
            'anchor': 'independent argmax over predicted foreground; included even when <=0',
            'sample_unit': 'cached object/crop annotation; repeated image IDs possible',
            'original_max_L_count': f'anchor + up to {args.prototype_max_patches-1} other strictly positive patches',
            'unbounded_count': 'all >0 patches plus anchor if anchor is <=0',
        },
        'annotations': n_anno,
        'unique_image_ids': n_images,
        'annotation_part_occurrences': len(rows),
        'unique_image_part_pairs': len(image_rows),
        'parts_observed': sum(c['occurrences'] > 0 for c in classes),
        'positive_count_by_annotation_part': stat([r['positive_margin_count'] for r in rows]),
        'positive_count_by_unique_image_part': stat([r['mean_positive_margin_count'] for r in image_rows]),
        'anchor_margin_by_annotation_part': stat([r['anchor_margin'] for r in rows]),
        'anchor_abs_by_annotation_part': stat([r['anchor_absolute_similarity'] for r in rows]),
        'positive_fraction_of_foreground': stat([r['positive_margin_fraction_of_foreground'] for r in rows]),
        'foreground_patches': stat([r['foreground_patch_count'] for r in rows]),
        'original_max_L_count': stat([r['original_max_L_count'] for r in rows]),
        'anchor_nonpositive_fraction': float(np.mean([r['anchor_margin'] <= 0 for r in rows])),
        'zero_positive_fraction': float(np.mean([r['positive_margin_count'] == 0 for r in rows])),
        'fraction_positive_gt_8': float(np.mean([r['positive_margin_count'] > 8 for r in rows])),
        'fraction_positive_gt_32': float(np.mean([r['positive_margin_count'] > 32 for r in rows])),
        'fraction_single_part_no_competitor': float(np.mean([r['single_part_no_competitor'] for r in rows])),
    }
    # O(n) calculation of independent anchor collisions.
    anchors_by_annotation = defaultdict(list)
    for r in rows: anchors_by_annotation[r['row_index']].append(r['anchor_patch_index'])
    stats['annotation_anchor_collision_fraction'] = float(np.mean([
        len(set(v)) < len(v) for v in anchors_by_annotation.values()
    ]))
    multi_rows = [r for r in rows if r['present_part_count'] >= 2]
    single_rows = [r for r in rows if r['present_part_count'] == 1]
    stats['competitive_K_ge_2'] = {
        'occurrences': len(multi_rows),
        'positive_patch_count': stat([r['positive_margin_count'] for r in multi_rows]),
        'anchor_margin': stat([r['anchor_margin'] for r in multi_rows]),
    }
    stats['single_part_K_eq_1_absolute_similarity'] = {
        'occurrences': len(single_rows),
        'positive_patch_count': stat([r['positive_margin_count'] for r in single_rows]),
        'anchor_score': stat([r['anchor_margin'] for r in single_rows]),
    }
    buckets = [('0',lambda n:n==0),('1',lambda n:n==1),('2-4',lambda n:2<=n<=4),
               ('5-8',lambda n:5<=n<=8),('9-16',lambda n:9<=n<=16),
               ('17-32',lambda n:17<=n<=32),('33-64',lambda n:33<=n<=64),
               ('65-128',lambda n:65<=n<=128),('129+',lambda n:n>=129)]
    stats['positive_count_histogram'] = {name: sum(fn(r['positive_margin_count']) for r in rows)
                                         for name,fn in buckets}

    print('[4/4] Writing read-only diagnostics...', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir/'per_annotation_part.csv', rows)
    write_csv(out_dir/'per_image_part.csv', image_rows)
    write_csv(out_dir/'per_part_summary.csv', classes)
    with (out_dir/'overall_summary.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, allow_nan=False)

    pc = stats['positive_count_by_annotation_part']
    ac = stats['anchor_margin_by_annotation_part']
    print('\n================ MARGIN AUDIT ================')
    print(f'Annotations (object crops): {n_anno}; unique image_ids: {n_images}')
    print(f'Annotation-part pairs: {len(rows)}; unique image-part pairs: {len(image_rows)}')
    print(f'W: {w_desc}')
    print('Positive patches / annotation-part: mean={:.2f}, median={:.1f}, p90={:.1f}, p95={:.1f}, max={:.0f}'.format(
        pc['mean'],pc['median'],pc['p90'],pc['p95'],pc['max']))
    print('Anchor relative margin: mean={:.4f}, median={:.4f}, p10={:.4f}, p90={:.4f}, min={:.4f}, max={:.4f}'.format(
        ac['mean'],ac['median'],ac['p10'],ac['p90'],ac['min'],ac['max']))
    print('Zero positive patches: {:.1%}; >8 positive: {:.1%}; >32 positive: {:.1%}'.format(
        stats['zero_positive_fraction'], stats['fraction_positive_gt_8'], stats['fraction_positive_gt_32']))
    print('Anchor margin <=0: {:.1%}; single-part (no competitors): {:.1%}'.format(
        stats['anchor_nonpositive_fraction'],stats['fraction_single_part_no_competitor']))
    if multi_rows:
        print('K>=2 true competitive margin: mean positive patches={:.2f}; anchor mean={:.4f}'.format(
            stats['competitive_K_ge_2']['positive_patch_count']['mean'],
            stats['competitive_K_ge_2']['anchor_margin']['mean']))
    if single_rows:
        print('K=1 (absolute similarity, NOT competition): mean positive patches={:.2f}'.format(
            stats['single_part_K_eq_1_absolute_similarity']['positive_patch_count']['mean']))
    print('Positive patch count histogram:', stats['positive_count_histogram'])
    print('Original capped support count mean: {:.2f}; Anchor collision annotations: {:.1%}'.format(
        stats['original_max_L_count']['mean'], stats['annotation_anchor_collision_fraction']))
    print('\nPer-part results (conditional on the part being present in that crop):')
    print(f"{'ID':>3}  {'Part':<29} {'n':>6} {'Mean +':>8} {'P50 +':>7} {'P90 +':>7} {'Anchor mean':>12} {'>32':>7}")
    for c in classes:
        if c['occurrences']:
            print(f"{c['part_id']:3d}  {c['part_name'][:29]:<29} {c['occurrences']:6d} "
                  f"{c['positive_mean']:8.2f} {c['positive_median']:7.1f} {c['positive_p90']:7.1f} "
                  f"{c['anchor_margin_mean']:12.4f} {c['fraction_more_than_32']*100:6.1f}%")
    print('\nSaved:')
    for name in ['overall_summary.json','per_part_summary.csv','per_annotation_part.csv','per_image_part.csv']:
        print(' ', out_dir / name)
    print('MARGIN_AUDIT_PASS')


if __name__ == '__main__':
    main()
