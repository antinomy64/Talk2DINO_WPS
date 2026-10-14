#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 read-only Soft Relative pseudo-label reliability audit.

A) Global confidence vs correctness, using ONLY strictly-positive r>0 patches
   in frozen-Talk2DINO predicted object foreground, when K>=2.
   - margin-accuracy deciles, absolute-threshold quality/coverage, top-rank
     quality/coverage, AUROC and Spearman where both labels occur.
   - report correctness both with GT ignore=255 as wrong (user's definition),
     and conditional on valid GT, as separate statistics.
B) SAME-CROP predicted Soft Relative visual prototype vs SAME-CROP GT DINO
   prototype (mean of normalized DINO tokens with GT p), cosine and
   discriminative margin against other GT prototypes of candidate parts.
   For K>=2 supports exactly reproduce positive r>0 plus 1-patch fallback;
   K==1 uses current Soft Absolute over predicted foreground.

GT masks are loaded only after support/prototype generation. No gradients,
no training, no model modifications, no download. Uses identical cached
predicted crops and 32x32 nearest-neighbor GT alignment as previous audit.
Note: geometric alignment is conditional on cached crop box correctly recording
all augmentations (verify with geometry audit before causal conclusions).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import audit_softrelative_support_pp116 as previous

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cache', default='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth')
    p.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean_pp116_night_softsingle_presence_e100.pth')
    p.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--gt-part-root', default='data/PascalPart116/annotations_detectron2_part/train')
    p.add_argument('--out-dir', default='output/pp116_softrelative_confidence_prototypes_e100')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--max-examples', type=int, default=0)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--absolute-thresholds', default='0,0.005,0.01,0.02,0.03,0.05,0.08,0.10')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    assert a.batch_size > 0 and a.temperature > 0 and a.max_examples >= 0
    return a


def pathof(x):
    x = Path(x).expanduser()
    return x if x.is_absolute() else ROOT / x


def loadpt(path):
    path = str(path)  # torch 2.1 mmap=True needs a *string* filename
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError):
        return torch.load(path, map_location='cpu', weights_only=False)


def div(a, b):
    return float(a) / float(b) if b else None


def mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def savecsv(p, rows, columns):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def counts(correct, valid):
    # correctness is 0 or 1, invalid GT counted as 0 for all-token metric
    n = len(correct)
    nc = int(correct.sum())
    nv = int(valid.sum())
    return {
        'tokens': n, 'correct_tokens': nc, 'valid_gt_tokens': nv,
        'gt_ignore_tokens': n-nv,
        'purity_all_selected': div(nc, n),
        'purity_valid_gt_only': div(nc, nv),
        'ignore_fraction': div(n-nv, n),
    }


def rank_auc(scores, labels):
    """Tie-aware AUROC using Mann-Whitney; labels boolean; no sklearn required."""
    pos = int(labels.sum()); neg = len(labels)-pos
    if not pos or not neg: return None
    order = np.argsort(scores, kind='stable')
    s = scores[order]; y = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    ends = np.r_[starts[1:], len(s)]
    positive_per_tie = np.add.reduceat(y.astype(np.float64), starts)
    # Average 1-based rank for the tied block [start, end).
    midranks = (starts.astype(np.float64) + ends + 1.) / 2.
    sum_ranks = float(np.dot(positive_per_tie, midranks))
    return float((sum_ranks - pos*(pos+1)/2)/(pos*neg))


def confidence_tables(margins, correct, valid, abs_thresholds):
    if not len(margins): raise RuntimeError('No positive-margin tokens, cannot compute quality curves')
    order = np.argsort(margins, kind='stable')
    n = len(order)
    deciles = []
    n_bins = min(10, n)
    for j in range(n_bins):
        idx = order[(j*n)//n_bins:((j+1)*n)//n_bins]
        row = {'decile_low_to_high': j+1,
               'margin_min':float(margins[idx].min()),
               'margin_mean':float(margins[idx].mean()),
               'margin_max':float(margins[idx].max())}
        row.update(counts(correct[idx], valid[idx]))
        deciles.append(row)
    abs_rows = []
    for th in abs_thresholds:
        sel = margins > th
        row = {'margin_threshold_strict_gt': float(th), 'coverage': div(int(sel.sum()),n)}
        row.update(counts(correct[sel], valid[sel]))
        abs_rows.append(row)
    fractions = (1., .9, .8, .7, .6, .5, .4, .3, .2, .1, .05, .01)
    top_rows = []
    for fraction in fractions:
        m = max(1, int(round(n*fraction)))
        sel = order[-m:]
        row = {'top_margin_fraction':fraction,'coverage':div(m,n),
               'threshold_observed_at_rank':float(margins[sel].min())}
        row.update(counts(correct[sel],valid[sel]))
        top_rows.append(row)
    return deciles, abs_rows, top_rows


def gt_prototypes_from_crop(tokens, gtlabels, pids):
    """One GT prototype per candidate part with >=1 valid GT patch, all crop tokens."""
    out = {}
    for pid in pids:
        indexes = np.flatnonzero(gtlabels == int(pid))
        if len(indexes) == 0: continue
        avg = tokens[torch.as_tensor(indexes, dtype=torch.long, device=tokens.device)].mean(0)
        out[int(pid)] = (F.normalize(avg,dim=0,eps=1e-12),len(indexes))
    return out


def proto_metrics(pred, gtproto, part, alternative_ids):
    if part not in gtproto: return None, None, None, 0
    target = gtproto[part][0]
    c = float((pred * target).sum().item())
    others = [float((pred * gtproto[q][0]).sum().item()) for q in alternative_ids if q!=part and q in gtproto]
    if not others: return c, None, None, 0
    dm = c - max(others)
    return c, dm, float(dm > 0), len(others)


def self_test():
    previous.audit_unit_tests()
    # Simulate 2-part competition on 4 predicted-FG patches; 5th is background.
    sc = torch.tensor([[.7,.8,.1,.3,.9],[.2,.4,.8,.9,.1]])
    fg = torch.tensor([1,1,1,1,0],dtype=torch.bool)
    selected, positive, fb, weights = previous.support_from_scores(sc,fg,.05)
    assert positive[0].tolist() == [True,True,False,False,False]
    assert positive[1].tolist() == [False,False,True,True,False]
    assert not fb.any()
    feats = F.normalize(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[1.,1.]]),dim=-1)
    pred = F.normalize(weights @ feats,dim=-1)
    gtlabels = np.array([0,0,1,1,255])
    refs = gt_prototypes_from_crop(feats,gtlabels,[0,1])
    a,b,c,n=proto_metrics(pred[0],refs,0,[0,1])
    assert abs(a-1) < 1e-6 and b>0 and c==1 and n==1
    margins = np.array([.01,.02,.2,.3],dtype=np.float32)
    correct = np.array([0,0,1,1],dtype=np.bool_)
    valid = np.array([1,1,1,1],dtype=np.bool_)
    dec,abs_rows,top = confidence_tables(margins,correct,valid,[0.,.1])
    assert len(dec)==4  # <=10 deciles if a small self-test batch
    assert abs_rows[-1]['coverage']==.5 and abs_rows[-1]['purity_all_selected']==1.
    assert top[-1]['tokens']==1
    assert abs(rank_auc(margins,correct)-1.)<1e-8
    print('SELF_TEST_PASS: positive support, GT prototype, discriminative margin, confidence/coverage, tie-aware AUROC')


def main():
    a = arguments()
    if a.self_test: self_test(); return
    import train_relproto_alignemt as original
    inputs = {
      'cache':pathof(a.cache), 'projector':pathof(a.projector),
      'text_bank':pathof(a.text_bank), 'model_config':pathof(a.model_config),
      'gt_part_root':pathof(a.gt_part_root), 'out_dir':pathof(a.out_dir),
    }
    for key in ('cache','projector','text_bank','model_config'):
        if not inputs[key].is_file():raise FileNotFoundError(inputs[key])
    if not inputs['gt_part_root'].is_dir():raise NotADirectoryError(inputs['gt_part_root'])
    output = inputs['out_dir']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already exists, refusing to overwrite: {output}')
    device = torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu')
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False

    print('[1/4] Load baked projector / 116-class text bank',flush=True)
    raw,names,_ = original.load_raw_clip_bank(inputs['text_bank'],
        expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if len(names)!=116: raise ValueError('Expected 116 part text classes')
    projector,_=original.load_frozen_projector(project_root=ROOT,
         config_path=inputs['model_config'],weight_path=inputs['projector'],device=device)
    with torch.no_grad():
        feats=original.project_raw_clip_bank(raw,projector,device=device,batch_size=128)
        text=F.normalize(feats.float().to(device),dim=-1,eps=1e-12)
    del projector
    if tuple(text.shape)!=(116,768):raise RuntimeError(f'Bad projected text shape: {text.shape}')

    print('[2/4] Load original predicted-object training cache',flush=True)
    cache = loadpt(inputs['cache'])
    if not isinstance(cache,dict) or not isinstance(cache.get('annotations'),list):
        raise RuntimeError('Expected train predobj cache with annotations list')
    meta=cache.get('pred_obj_cropaug_meta',{})
    if meta.get('format')!='pascalpart116_weaklabels_predobj_cropaug_v1':
        raise RuntimeError('Wrong cache format: '+str(meta.get('format')))
    if meta.get('patch_grid')!=[32,32] or meta.get('crop_resize')!=[448,448]:
        raise RuntimeError('Expected 32x32 DINO grid and 448x448 crop')
    if float(meta.get('bg_thresh',-1))!=.55 or not meta.get('pamr',False):
        raise RuntimeError('Cache protocol mismatch: expected bg=0.55 and PAMR')
    image_sizes={str(im['id']):(int(im['height']),int(im['width'])) for im in cache['images']}
    annotations=cache['annotations'][:a.max_examples] if a.max_examples else cache['annotations']
    print(f'   cached crops={len(annotations)}; device={device}',flush=True)
    gt_root=inputs['gt_part_root']
    @lru_cache(maxsize=12)
    def gt(stem):
        p=gt_root/f'{stem}.png'
        if not p.is_file():raise FileNotFoundError(p)
        with Image.open(p) as img: arr=np.asarray(img).copy()
        if arr.ndim==3:
            if arr.shape[-1]>=3 and np.array_equal(arr[...,0],arr[...,1]) and np.array_equal(arr[...,0],arr[...,2]):
                arr=arr[...,0]
            else:raise RuntimeError(f'Unexpected color-coded GT: {p}')
        assert arr.ndim==2
        if str(stem) in image_sizes and tuple(arr.shape)!=image_sizes[str(stem)]:
            raise RuntimeError(f'GT/image geometry mismatch: {stem}')
        ignore=int(meta.get('part_ignore',255)); offset=int(meta.get('part_id_offset',0))
        out=arr.astype(np.int32)
        ok=(out==ignore) | ((out-offset>=0)&(out-offset<116))
        if not ok.all():raise RuntimeError(f'Bad GT labels in {p}')
        return np.where(out==ignore,255,out-offset).astype(np.int32)

    grouped=defaultdict(list)
    for item in annotations:
        pids=[int(x) for x in item['part_category_id']]
        if not pids or len(set(pids))!=len(pids) or any(x<0 or x>=116 for x in pids):
            raise RuntimeError(f'Bad part candidates in crop {item.get("id")}')
        if tuple(item['cropaug_patch_tokens'].shape)!=(1024,768):
            raise RuntimeError('Unexpected 1024x768 DINO cached features')
        if tuple(item['pred_obj_mask_patch'].shape)!=(1024,):
            raise RuntimeError('Unexpected 1024 foreground mask')
        grouped[len(pids)].append(item)

    proto_rows=[]
    margin_chunks=[]; correct_chunks=[]; valid_chunks=[]
    print('[3/4] Replay support and construct same-crop predicted / GT visual prototypes',flush=True)
    with torch.no_grad():
        for k,group in tqdm(sorted(grouped.items()),desc='Part candidate groups'):
            for start in range(0,len(group),a.batch_size):
                batch=group[start:start+a.batch_size]
                x=original.normalize_last(torch.stack([z['cropaug_patch_tokens'] for z in batch]).to(device=device,dtype=torch.float32))
                fg=torch.stack([z['pred_obj_mask_patch'] for z in batch]).to(device=device,dtype=torch.bool)
                pid_tensor=torch.as_tensor([[int(p) for p in z['part_category_id']] for z in batch],device=device,dtype=torch.long)
                scores=torch.bmm(text[pid_tensor],x.transpose(1,2))
                rel=original.relative_scores(scores,fg) if k>=2 else None
                for j,rec in enumerate(batch):
                    selected,positive,fallback,weights=previous.support_from_scores(
                        scores[j],fg[j],a.temperature,relative_scores_fn=original.relative_scores)
                    if k>=2 and not torch.equal(positive,(rel[j]>0)):
                        raise RuntimeError('Relative margin support differs from original trainer')
                    protos=F.normalize(weights@x[j],dim=-1,eps=1e-12)
                    stem=str(rec['image_id'])
                    box=rec['cropaug_box_xyxy'].tolist() if torch.is_tensor(rec['cropaug_box_xyxy']) else rec['cropaug_box_xyxy']
                    grid=previous.resize_gt_mask(gt(stem),box)
                    pids=pid_tensor[j].tolist()
                    gtref=gt_prototypes_from_crop(x[j],grid,pids)
                    for ii,p in enumerate(pids):
                        label = 'single_absolute' if k==1 else ('fallback' if fallback[ii] else 'positive_margin')
                        chosen=selected[ii].cpu().numpy().astype(bool)
                        observed=grid[chosen]
                        valid=(observed!=255)
                        correct=(observed==p)
                        c,disc,top1,nalt=proto_metrics(protos[ii],gtref,p,pids)
                        n_gt=gtref[p][1] if p in gtref else 0
                        row={
                            'image_id':stem,'crop_id':int(rec['id']),
                            'object_name':str(rec['class_name']),'part_id':p,'part_name':str(names[p]),
                            'candidate_count':k,'mode':label,'has_gt_proto':int(p in gtref),
                            'gt_part_patches_in_crop':n_gt,
                            'selected_patch_count':int(chosen.sum()),
                            'selected_correct_patches':int(correct.sum()),
                            'selected_gt_ignore_patches':int((~valid).sum()),
                            'selected_purity':div(int(correct.sum()),int(chosen.sum())),
                            'proto_gt_cosine':c,
                            'proto_gt_discriminative_margin':disc,
                            'proto_gt_top1':top1,
                            'gt_competitors_available':nalt,
                        }
                        proto_rows.append(row)
                        if k>=2 and label=='positive_margin':
                            assert int(chosen.sum())==int(positive[ii].sum())
                            # All positive r values and per-patch GT match for the primary confidence analysis.
                            m=rel[j,ii][positive[ii]].float().cpu().numpy()
                            if not np.isfinite(m).all() or not (m>0).all():
                                raise RuntimeError('Non-positive selected margin')
                            labs=grid[positive[ii].cpu().numpy().astype(bool)]
                            margin_chunks.append(m.astype(np.float32,copy=False))
                            correct_chunks.append((labs==p).astype(np.bool_))
                            valid_chunks.append((labs!=255).astype(np.bool_))
    if not proto_rows:raise RuntimeError('No audit rows')
    if not margin_chunks:raise RuntimeError('No positive-margin tokens')
    margins=np.concatenate(margin_chunks)
    correct=np.concatenate(correct_chunks)
    valid=np.concatenate(valid_chunks)
    assert len(margins)==len(correct)==len(valid)

    thresholds = [float(s) for s in a.absolute_thresholds.split(',') if s.strip()]
    if any(t<0 for t in thresholds):raise ValueError('Margin thresholds must be >= 0')
    deciles,absolute,ranked=confidence_tables(margins,correct,valid,thresholds)
    # Optional Spearman over binary correctness on valid GT, avoiding a scipy hard dependency.
    auc_valid=rank_auc(margins[valid],correct[valid]) if valid.any() else None
    auc_all=rank_auc(margins,correct)
    try:
        from scipy.stats import spearmanr
        spear_valid=float(spearmanr(margins[valid], correct[valid].astype(float)).statistic) if len(np.unique(correct[valid]))==2 else None
        spear_all=float(spearmanr(margins,correct.astype(float)).statistic) if len(np.unique(correct))==2 else None
    except ImportError:
        spear_valid=None; spear_all=None
    
    pairs=defaultdict(list)
    for r in proto_rows:pairs[r['mode']].append(r)
    mode_stats={}
    for mode,rs in sorted(pairs.items()):
        valid_r=[r for r in rs if r['proto_gt_cosine'] is not None]
        ranked_r=[r for r in valid_r if r['proto_gt_discriminative_margin'] is not None]
        mode_stats[mode]={
            'crop_part_pairs':len(rs),'valid_gt_proto_pairs':len(valid_r),
            'with_gt_competitors':len(ranked_r),
            'proto_gt_cosine_pair_macro':mean([r['proto_gt_cosine'] for r in valid_r]),
            'proto_discriminative_margin_pair_macro':mean([r['proto_gt_discriminative_margin'] for r in ranked_r]),
            'proto_gt_top1_accuracy':mean([r['proto_gt_top1'] for r in ranked_r]),
        }
    by_class=defaultdict(list)
    for r in proto_rows:
        if r['proto_gt_cosine'] is not None: by_class[r['part_id']].append(r)
    class_rows=[]
    for pid in range(116):
        rr=by_class[pid]
        class_rows.append({'part_id':pid,'part_name':str(names[pid]),
            'valid_pairs':len(rr), 'proto_gt_cosine_macro':mean([r['proto_gt_cosine'] for r in rr]),
            'proto_gt_discriminative_margin_macro':mean([r['proto_gt_discriminative_margin'] for r in rr]),
            'proto_gt_top1_accuracy':mean([r['proto_gt_top1'] for r in rr if r['proto_gt_top1'] is not None]),
        })
    stats={
        'projector':str(inputs['projector']),'cache':str(inputs['cache']),
        'crop_count':len(annotations),'all_crop_part_pairs':len(proto_rows),
        'margin_positive_tokens':len(margins),
        'margin_positive_all_gt':counts(correct,valid),
        'margin_auc_correct_vs_wrong_all_gt':auc_all,
        'margin_auc_correct_vs_wrong_valid_gt':auc_valid,
        'margin_spearman_all_gt':spear_all,
        'margin_spearman_valid_gt':spear_valid,
        'per_mode_prototype_quality':mode_stats,
        'prototype_cosine_all_valid_pairs_macro':mean([r['proto_gt_cosine'] for r in proto_rows]),
        'prototype_cosine_class_macro':mean([r['proto_gt_cosine_macro'] for r in class_rows]),
        'discriminative_margin_all_valid_pairs_macro':mean([r['proto_gt_discriminative_margin'] for r in proto_rows]),
        'discriminative_margin_class_macro':mean([r['proto_gt_discriminative_margin_macro'] for r in class_rows]),
        'gt_proto_construction':'mean normalized frozen DINOv2 tokens with corresponding GT part ID in SAME crop, normalize result; all crop tokens',
        'confidence_selection':'K>=2 strictly r>0 within Talk2DINO predicted FG; all positive patches; no K=1 or fallback in confidence curves',
        'gt_geometry':'cached predicted cropaug_box_xyxy + nearest resize 32x32 (same as earlier audit)',
        'gt_ignore_handling':'two versions output: ignore=255 counts wrong in all-token purity; valid-only purity excludes it',
        'training':'none; GT only posthoc verification',
    }
    print('[4/4] Save curves, prototype CSVs and summary',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    savecsv(output/'margin_accuracy_deciles.csv',deciles,list(deciles[0]))
    savecsv(output/'quality_coverage_absolute_threshold.csv',absolute,list(absolute[0]))
    savecsv(output/'quality_coverage_top_fraction.csv',ranked,list(ranked[0]))
    savecsv(output/'prototype_per_crop_part.csv',proto_rows,list(proto_rows[0]))
    savecsv(output/'prototype_per_class.csv',class_rows,list(class_rows[0]))
    (output/'summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7.6,4.5))
        dx=[r['decile_low_to_high'] for r in deciles]
        ax.plot(dx,[100*r['purity_all_selected'] if r['purity_all_selected'] is not None else np.nan for r in deciles], 'o-',label='GT ignore counted as incorrect')
        ax.plot(dx,[100*r['purity_valid_gt_only'] if r['purity_valid_gt_only'] is not None else np.nan for r in deciles], 's--',label='Valid GT only')
        ax.set(xlabel='Margin rank decile (1=lowest, 10=highest)',ylabel='Correct selected patches (%)',title='Positive relative margin vs GT correctness')
        ax.set_xticks(range(1,11));ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(output/'margin_accuracy.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.6,4.5))
        xcov=[100*r['coverage'] for r in ranked]
        ax.plot(xcov,[100*r['purity_all_selected'] for r in ranked], 'o-',label='GT ignore counted as incorrect')
        ax.plot(xcov,[100*r['purity_valid_gt_only'] if r['purity_valid_gt_only'] is not None else np.nan for r in ranked], 's--',label='Valid GT only')
        ax.set(xlabel='Retained positive-margin patches (%)',ylabel='Correct retained patches (%)',title='Quality–Coverage (highest margin first)')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(output/'quality_coverage.png',dpi=180);plt.close(fig)
    except ImportError as e:
        print(f'WARNING: plotting unavailable: {e}',flush=True)
    print('\n========== Soft Relative confidence & prototype audit ==========',flush=True)
    print(f'Positive margin patches: {len(margins):,}',flush=True)
    print('Margin AUROC (all GT / valid GT):', auc_all,auc_valid,flush=True)
    print('Margin Spearman (all GT / valid GT):',spear_all,spear_valid,flush=True)
    for mode,ms in mode_stats.items():
        print(f'[{mode}] pairs={ms["crop_part_pairs"]}, validGT={ms["valid_gt_proto_pairs"]}, cos={ms["proto_gt_cosine_pair_macro"]}, '
              f'discriminative-margin={ms["proto_discriminative_margin_pair_macro"]}, top1={ms["proto_gt_top1_accuracy"]}',flush=True)
    print('Saved:',output,flush=True)


if __name__=='__main__':
    main()
