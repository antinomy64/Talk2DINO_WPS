#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP-116: compare frozen COCO object-projector text geometry with TRAIN GT DINOv2 geometry.

No training, no external downloads. Part GT prototypes are used only for post-hoc audit.
Run from Talk2DINO_official_bg; script should be placed in that repository root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def pathof(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def rank_correlation(a, b, min_points=3):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < min_points:
        return float('nan')
    ar = pd.Series(a).rank(method='average').to_numpy(dtype=np.float64)
    br = pd.Series(b).rank(method='average').to_numpy(dtype=np.float64)
    if ar.std() < 1e-12 or br.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a, b, min_points=3):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < min_points or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def ranking_agreement(a, b):
    """Fraction of peer-pair orderings (q,r) concordant between text and GT.
    Excludes exact/small ties. This is a descriptive triplet statistic, not accuracy.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float('nan'), 0
    i, j = np.triu_indices(len(a), 1)
    da, db = a[i] - a[j], b[i] - b[j]
    good = np.isfinite(da) & np.isfinite(db) & (np.abs(da) > 1e-6) & (np.abs(db) > 1e-6)
    n = int(good.sum())
    if not n:
        return float('nan'), 0
    return float(np.mean(np.sign(da[good]) == np.sign(db[good]))), n


def overlap_topk(a, b, peers, k):
    peers = np.asarray(peers, dtype=int)
    n = len(peers)
    # If n<=k, all choices are returned so the metric is trivially 1: mark unavailable.
    if n <= k or k < 1:
        return float('nan')
    tx = set(peers[np.argsort(-np.asarray(a))[:k]])
    vis = set(peers[np.argsort(-np.asarray(b))[:k]])
    return len(tx & vis) / float(k)


def arg_str(a, peers, names, k=4):
    if len(peers) == 0:
        return ''
    order = np.argsort(-np.asarray(a))[:min(k, len(peers))]
    return ' | '.join(str(names[int(peers[j])]) for j in order)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth')
    p.add_argument('--gt-bank', default='feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt')
    p.add_argument('--iou-csv', default='output/pp116_noabs_iou_curve/per_part_iou.csv',
                   help='Optional E000..E100 IoU CSV for side-by-side correlation; empty string disables.')
    p.add_argument('--out-dir', default='output/pp116_text_gt_structure_init')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()

    import train_relproto_alignemt as original  # exact repo implementation; not reimplemented
    inputs = {k: pathof(v) for k, v in [
        ('text', args.text_bank), ('projector', args.projector),
        ('config', args.model_config), ('gt', args.gt_bank),
    ]}
    for key, fp in inputs.items():
        if not fp.is_file():
            raise FileNotFoundError(f'{key}: {fp}')
    out = pathof(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False

    print('[1/5] Read raw CLIP part text; use the repository object-projector forward', flush=True)
    raw, names, _ = original.load_raw_clip_bank(
        inputs['text'], expected_clip_model='ViT-B/16',
        expected_template='sub_imagenet_template')
    names = [str(x) for x in names]
    if len(names) != 116 or len(set(names)) != 116 or not all("'s " in n for n in names):
        raise RuntimeError('Expected 116 unique full names of the form "object\'s part"')
    projector, proj_meta = original.load_frozen_projector(
        project_root=ROOT, config_path=inputs['config'],
        weight_path=inputs['projector'], device=device)
    with torch.no_grad():
        projected = original.project_raw_clip_bank(
            raw, projector, device=device, batch_size=args.batch_size)
        text = F.normalize(projected.detach().to(device=device, dtype=torch.float32), dim=-1, eps=1e-12)
    del projector
    if tuple(text.shape) != (116, 768) or not bool(torch.isfinite(text).all()):
        raise RuntimeError(f'Bad projected text shape/values: {tuple(text.shape)}')

    print('[2/5] Read the fixed training-set global GT prototype bank', flush=True)
    # trusted local research files only
    gt_obj = torch.load(inputs['gt'], map_location='cpu', weights_only=False)
    if not isinstance(gt_obj, dict) or 'prototypes' not in gt_obj:
        raise RuntimeError('GT bank must be a dict with key "prototypes"')
    g_raw = torch.as_tensor(gt_obj['prototypes'], dtype=torch.float32)
    if tuple(g_raw.shape) != (116, 768):
        raise RuntimeError(f'Expected GT [116,768], got {tuple(g_raw.shape)}')
    counts = gt_obj.get('patch_counts')
    if counts is not None:
        counts = torch.as_tensor(counts).reshape(-1)
        if tuple(counts.shape) != (116,):
            raise RuntimeError('GT patch_counts should be [116]')
        valid = counts > 0
    else:
        valid = torch.ones(116, dtype=torch.bool)
    valid = valid & torch.isfinite(g_raw).all(-1) & (g_raw.norm(dim=-1) > 1e-9)
    for key in ('class_names', 'classes', 'part_classes'):
        if key in gt_obj:
            ref_names = gt_obj[key]
            if isinstance(ref_names, (list, tuple)) and len(ref_names) == 116:
                if [str(x) for x in ref_names] != names:
                    raise RuntimeError(f'GT {key} order differs from CLIP text bank: fix taxonomy before analysis')
                print(f'[check] GT name alignment verified by {key}', flush=True)
                break
    else:
        print('[note] GT bank has no comparable class name list; assumes its 116 rows use identical class IDs.', flush=True)
    gt = F.normalize(g_raw.to(device), dim=-1, eps=1e-12)
    valid_np = valid.cpu().numpy().astype(bool)
    if valid_np.sum() < 5:
        raise RuntimeError('Too few valid GT prototypes')
    print(f'    valid GT prototypes = {valid_np.sum()}/116', flush=True)

    print('[3/5] Compute text/GT pairwise cosine similarity matrices', flush=True)
    s_txt = (text @ text.T).cpu().numpy().astype(np.float64)
    s_gt = (gt @ gt.T).cpu().numpy().astype(np.float64)
    cross = (text @ gt.T).cpu().numpy().astype(np.float64)
    s_gt[~valid_np, :] = np.nan
    s_gt[:, ~valid_np] = np.nan
    np.fill_diagonal(s_txt, 1.0)
    for i in np.flatnonzero(valid_np):
        s_gt[i, i] = 1.0
    np.save(out / 'text_pairwise_cosine.npy', s_txt.astype('float32'))
    np.save(out / 'gt_pairwise_cosine.npy', s_gt.astype('float32'))
    write_csv(out / 'text_pairwise_cosine.csv',
              [dict(part=names[i], **{f'c{j:03d}': round(float(s_txt[i,j]), 7) for j in range(116)}) for i in range(116)])
    write_csv(out / 'gt_pairwise_cosine.csv',
              [dict(part=names[i], **{f'c{j:03d}': (round(float(s_gt[i,j]), 7) if np.isfinite(s_gt[i,j]) else np.nan) for j in range(116)}) for i in range(116)])

    objects = [n.split("'s ", 1)[0] for n in names]
    suffixes = [n.split("'s ", 1)[1] for n in names]
    mask_valid_pairs = np.outer(valid_np, valid_np) & ~np.eye(116, dtype=bool)
    iu = np.triu_indices(116, k=1)
    intra = np.equal.outer(objects, objects)
    pair_rows = []
    for i, j in zip(*iu):
        if not (valid_np[i] and valid_np[j]):
            continue
        st, sg = float(s_txt[i,j]), float(s_gt[i,j])
        pair_rows.append({
            'part_i': names[i], 'part_j': names[j],
            'same_object': bool(intra[i,j]),
            'same_part_type': bool(suffixes[i] == suffixes[j]),
            'text_cosine': st, 'gt_cosine': sg,
            'signed_diff_text_minus_gt': st-sg,
            'abs_error': abs(st-sg),
        })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / 'all_valid_pairs.csv', index=False)
    pairs[pairs.same_object].sort_values('abs_error', ascending=False).to_csv(
        out / 'worst_intra_object_pairs.csv', index=False)

    global_rows = []
    for scope_name, dd in [('all', pairs), ('same_object', pairs[pairs.same_object]),
                            ('cross_object', pairs[~pairs.same_object]),
                            ('same_part_type', pairs[pairs.same_part_type])]:
        global_rows.append({
            'scope': scope_name, 'n_pairs': len(dd),
            'Spearman': rank_correlation(dd.text_cosine, dd.gt_cosine),
            'Pearson': pearson(dd.text_cosine, dd.gt_cosine),
            'mean_abs_cosine_error': float(dd.abs_error.mean()) if len(dd) else np.nan,
        })
    write_csv(out / 'overall_structure_summary.csv', global_rows)
    object_rows = []
    for obj_name in sorted(set(objects)):
        members = [i for i, n in enumerate(objects) if n == obj_name and valid_np[i]]
        ii, jj = np.triu_indices(len(members), 1)
        x = [s_txt[members[a], members[b]] for a,b in zip(ii,jj)]
        y = [s_gt[members[a], members[b]] for a,b in zip(ii,jj)]
        object_rows.append({
            'object': obj_name, 'n_valid_parts': len(members), 'n_pairs': len(x),
            'object_pair_spearman': rank_correlation(x,y),
            'object_pair_pearson': pearson(x,y),
            'object_pair_mae': float(np.mean(np.abs(np.asarray(x)-np.asarray(y)))) if len(x) else np.nan,
        })
    write_csv(out / 'by_object_structure_summary.csv', object_rows)

    print('[4/5] Compute part-centered structure correspondence', flush=True)
    class_rows = []
    valid_ids = np.flatnonzero(valid_np)
    for i in range(116):
        others = valid_ids[valid_ids != i] if valid_np[i] else np.array([], dtype=int)
        same = np.array([j for j in others if objects[j] == objects[i]], dtype=int)
        different = np.array([j for j in others if objects[j] != objects[i]], dtype=int)
        same_type = np.array([j for j in others if suffixes[j] == suffixes[i]], dtype=int)
        row = {
            'part_id': i, 'part': names[i], 'object': objects[i], 'part_type': suffixes[i],
            'GT_valid': bool(valid_np[i]),
            'GT_patch_count': int(counts[i]) if counts is not None else np.nan,
            'n_global_peers': len(others), 'n_intra_peers': len(same),
            'n_cross_peers': len(different), 'n_same_type_peers': len(same_type),
            'text_gt_direct_cosine': float(cross[i,i]) if valid_np[i] else np.nan,
        }
        if valid_np[i]:
            ordered_cross = np.argsort(-cross[i, valid_ids])
            ranked_ids = valid_ids[ordered_cross]
            row['crossmodal_gt_rank_116'] = int(np.where(ranked_ids == i)[0][0] + 1)
            max_comp = float(np.max(cross[i, others])) if len(others) else float('nan')
            row['crossmodal_gt_margin'] = float(cross[i,i] - max_comp)
            row['crossmodal_top1_gt'] = names[int(ranked_ids[0])]
        else:
            row['crossmodal_gt_rank_116'] = np.nan
            row['crossmodal_gt_margin'] = np.nan
            row['crossmodal_top1_gt'] = ''

        for scope, ids in [('global', others), ('intra', same), ('cross', different),
                            ('same_type', same_type)]:
            a, b = s_txt[i, ids], s_gt[i, ids]
            row[f'{scope}_rho'] = rank_correlation(a, b)
            row[f'{scope}_pearson'] = pearson(a, b)
            row[f'{scope}_mae'] = float(np.mean(np.abs(a-b))) if len(ids) else np.nan
            agree, total = ranking_agreement(a, b)
            row[f'{scope}_rank_agreement'] = agree
            row[f'{scope}_n_peer_pairs'] = total
            k = 5 if scope != 'intra' else 2
            row[f'{scope}_top{k}_overlap'] = overlap_topk(a, b, ids, k)
            row[f'{scope}_text_nearest'] = arg_str(a, ids, names)
            row[f'{scope}_gt_nearest'] = arg_str(b, ids, names)
        # 4 intra peers -> 6 pair-order comparisons (still noisy: inspect n)
        if not valid_np[i]:
            row['intra_structure_group'] = 'NO_GT'
        elif len(same) < 4 or not np.isfinite(row['intra_rho']):
            row['intra_structure_group'] = 'TOO_FEW_PEERS'
        elif row['intra_rho'] >= .5:
            row['intra_structure_group'] = 'CONSISTENT'
        elif row['intra_rho'] <= 0.0:
            row['intra_structure_group'] = 'MISMATCH'
        else:
            row['intra_structure_group'] = 'MIXED'
        class_rows.append(row)

    per_part = pd.DataFrame(class_rows)
    iou = pathof(args.iou_csv) if args.iou_csv else None
    if iou is not None and iou.is_file():
        history = pd.read_csv(iou)
        needed = {'part', 'E000', 'E100'}
        if not needed.issubset(history.columns):
            raise RuntimeError(f'IoU CSV missing columns {sorted(needed - set(history.columns))}: {iou}')
        if history.part.duplicated().any():
            raise RuntimeError('IoU CSV has duplicate class names')
        ecols = [c for c in history.columns if c.startswith('E') and c[1:].isdigit()]
        keep = ['part'] + ecols
        per_part = per_part.merge(history[keep], on='part', how='left', validate='one_to_one')
        if per_part.E100.isna().sum() > int((~valid_np).sum()):
            raise RuntimeError('IoU CSV does not fully match the 116 class names')
        per_part['IoU_delta_100_minus_0'] = per_part.E100 - per_part.E000
        per_part['IoU_min_observed'] = per_part[ecols].min(axis=1)
        per_part['IoU_max_observed'] = per_part[ecols].max(axis=1)
        per_part['IoU_ever_below_20'] = per_part['IoU_min_observed'] < 20
        print(f'[check] Joined IoU timeline: {iou}', flush=True)
    else:
        print('[note] No IoU timeline merged (optional).', flush=True)
    per_part.sort_values(['intra_structure_group','intra_rho','global_rho'], na_position='last').to_csv(
        out / 'per_part_structure.csv', index=False, encoding='utf-8-sig')

    type_rows = []
    for label, df in per_part.groupby('part_type', sort=True):
        v = df[df.GT_valid]
        analyzable = v[v.intra_structure_group.isin(['CONSISTENT','MISMATCH','MIXED'])]
        type_rows.append({
            'part_type': label, 'n_in_vocab': len(df), 'n_valid_gt': len(v),
            'n_intra_evaluable': len(analyzable),
            'n_intra_mismatch': int((analyzable.intra_structure_group == 'MISMATCH').sum()),
            'mean_global_rho': float(v.global_rho.mean()),
            'mean_intra_rho': float(analyzable.intra_rho.mean()) if len(analyzable) else np.nan,
            'mean_same_type_rho': float(v.same_type_rho.mean()),
            'mean_same_type_mae': float(v.same_type_mae.mean()),
            'parts': ' | '.join(df.part.tolist()),
        })
    pd.DataFrame(type_rows).sort_values(['n_intra_mismatch','n_intra_evaluable'], ascending=False).to_csv(
        out / 'part_type_structure_summary.csv', index=False, encoding='utf-8-sig')

    print('[5/5] Create overview plots', flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)
    images = [s_txt, s_gt, s_txt - s_gt]
    titles = ['Projected text: pairwise cosine', 'GT visual: pairwise cosine', 'Text - GT cosine relations']
    for ax, m, title in zip(axes, images, titles):
        kw = dict(vmin=-1, vmax=1, cmap='coolwarm') if 'Text - GT' not in title else dict(vmin=-1, vmax=1, cmap='coolwarm')
        im = ax.imshow(m, **kw)
        ax.set_title(title)
        ax.set_xlabel('Part ID')
        ax.set_ylabel('Part ID')
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / 'pairwise_structure_matrices.png', dpi=180)
    plt.close(fig)

    plot_df = per_part[per_part.GT_valid & per_part.intra_rho.notna()].copy().sort_values('intra_rho')
    if len(plot_df):
        fig, ax = plt.subplots(figsize=(12, max(7, min(26, .26 * len(plot_df)))))
        sub = plot_df.head(50)
        ax.barh(sub.part, sub.intra_rho)
        ax.invert_yaxis()
        ax.set_xlim(-1.05, 1.05)
        ax.axvline(0, linestyle='--', linewidth=1)
        ax.set_xlabel('Spearman of each part\'s within-object cosine relations')
        ax.set_title('Lowest within-object structure agreement (up to 50 parts)')
        fig.tight_layout()
        fig.savefig(out / 'worst_part_structure_rho.png', dpi=200)
        plt.close(fig)

    if 'E100' in per_part.columns:
        good = per_part[per_part.GT_valid & per_part.global_rho.notna() & per_part.E100.notna()]
        if len(good):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(good.global_rho, good.E100, alpha=.75)
            ax.set_xlabel('Part-centered global relation Spearman')
            ax.set_ylabel('Part IoU at Epoch 100 (%)')
            ax.set_title('Structure correspondence vs PP-116 IoU (exploratory)')
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(out / 'structure_vs_IoU_E100.png', dpi=180)
            plt.close(fig)
            print('[stats] global_rho vs IoU E100 Pearson =', pearson(good.global_rho, good.E100))
            print('[stats] global_rho vs IoU E100 Spearman =', rank_correlation(good.global_rho, good.E100))
        intra_good = per_part[per_part.GT_valid & per_part.intra_rho.notna() & per_part.E100.notna() & (per_part.n_intra_peers >= 4)]
        if len(intra_good):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(intra_good.intra_rho, intra_good.E100, alpha=.75)
            ax.set_xlabel('Within-object part-relation Spearman')
            ax.set_ylabel('Part IoU at Epoch 100 (%)')
            ax.set_title('Within-object text/GT structure vs final IoU')
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(out / 'intra_structure_vs_IoU_E100.png', dpi=180)
            plt.close(fig)
            print('[stats] intra_rho vs IoU E100 Spearman =', rank_correlation(intra_good.intra_rho, intra_good.E100))

    with open(out / 'metadata.json', 'w') as f:
        json.dump({
            'text_bank': str(inputs['text']), 'projector': str(inputs['projector']),
            'gt_global_prototypes': str(inputs['gt']),
            'valid_gt_parts': int(valid_np.sum()),
            'text_shape': list(text.shape), 'gt_shape': list(gt.shape),
            'semantics': 'Post-hoc comparison of cosine Gram matrices; no part GT in optimization',
            'intra_group_thresholds': {'min_peers': 4, 'consistent_rho_ge': .5, 'mismatch_rho_le': 0.0},
        }, f, indent=2, ensure_ascii=False)

    print('\n========= ALL STRUCTURE =========', flush=True)
    print(pd.read_csv(out / 'overall_structure_summary.csv').to_string(index=False), flush=True)
    print('\n========= LOWEST WITHIN-OBJECT RHO =========', flush=True)
    print(per_part[per_part.intra_rho.notna()].sort_values('intra_rho')[
        ['part','n_intra_peers','intra_rho','intra_rank_agreement','global_rho',
         'text_gt_direct_cosine','crossmodal_gt_rank_116']].head(25).to_string(index=False), flush=True)
    print('\n========= EYE PARTS =========', flush=True)
    print(per_part[per_part.part_type == 'eye'][
        ['part','n_intra_peers','intra_rho','global_rho','intra_structure_group',
         'text_gt_direct_cosine','crossmodal_gt_rank_116'] + (['E100'] if 'E100' in per_part else [])
    ].to_string(index=False), flush=True)
    print('\nSaved to:', out, flush=True)


if __name__ == '__main__':
    main()
