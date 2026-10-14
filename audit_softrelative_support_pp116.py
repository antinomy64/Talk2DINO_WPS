#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 Soft Relative + Structure: read-only two-metric analysis.

Metric 1 (primary): Within each (cached predicted-object crop, candidate part),
  replay the *training* support selection using the final baked projector:
    K >= 2: r[p,i] = cos(t_p,x_i) - max_{q!=p} cos(t_q,x_i), on predicted FG;
            C_p={i:r[p,i]>0}; if C_p empty, add argmax fallback only.
    K == 1: C_p = all predicted foreground patch tokens (Soft Absolute).
  Apply the cached PREDICTED crop box to TRAIN GT part mask, then nearest-resize
  to the same 32x32 patch grid, solely for post-hoc correctness checking.
  Support Purity = number of selected tokens with matching GT part ID / total
  selected tokens. GT ignore (255) is counted as *not correct* in this primary
  denominator exactly as requested; a valid-GT-only value is also exported.
  Strict r>0 purity (K>=2, excluding fallback) is separately reported.

Metric 2: Cosine between baked-projector 116x768 text embedding and previously
  computed GLOBAL GT DINOv2 part prototype bank, macro over valid GT classes.

Important: Cache and model used to generate the supports are NEVER changed.
No GT mask is read before the support is selected. No training, no re-extraction,
no downloads, no edits to official repository files.
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

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cache', default='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth')
    p.add_argument('--projector', default='weights/vitb_mlp_infonce_coco2014_reproduce_clean_pp116_night_softsingle_presence_e100.pth',
                   help='BAKED final-epoch projector; do not also apply W again')
    p.add_argument('--text-bank', default='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--gt-part-root', default='data/PascalPart116/annotations_detectron2_part/train')
    p.add_argument('--gt-prototypes', default='feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt')
    p.add_argument('--out-dir', default='output/pp116_softrelative_support_e100_audit')
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-examples', type=int, default=0, help='0=all cached crops; >0=quick smoke test')
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--self-test', action='store_true', help='Pure synthetic support/mask audit; does not load repo')
    a = p.parse_args()
    if a.batch_size <= 0 or a.temperature <= 0 or a.max_examples < 0:
        p.error('batch size, temperature and max-examples invalid')
    return a


def absolute(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_pt(path):
    path = str(path)
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location='cpu', weights_only=False)


def finite_mean(xs):
    xs = [float(v) for v in xs if v is not None and np.isfinite(v)]
    return float(np.mean(xs)) if xs else None


def frac(a, b):
    return float(a/b) if b else None


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def support_from_scores(scores, foreground, temperature, *, relative_scores_fn=None):
    """Return support, strictly positive selection, fallback, normalized weights.

    scores: [K,N], foreground [N]. Relative scorer from original trainer used
    to guarantee exact foreground + max-competitor arithmetic.
    """
    k, n = scores.shape
    if foreground.shape != (n,) or not bool(foreground.any()):
        raise ValueError('missing/invalid predicted foreground')
    if k == 1:
        # Actual soft-single training path: ALL foreground patches, not s>0.
        selected = foreground[None].clone()
        positive = torch.zeros_like(selected)
        fallback = torch.zeros((1,), dtype=torch.bool, device=scores.device)
        masked = scores.masked_fill(~selected, float('-inf'))
    else:
        if relative_scores_fn is None:
            competitor = scores[None, :, :].expand(k, k, n).clone()
            diag = torch.arange(k, device=scores.device)
            competitor[diag, diag] = -torch.inf
            rel = scores - competitor.max(dim=1).values
            rel = rel.masked_fill(~foreground[None], -torch.inf)
        else:
            rel = relative_scores_fn(scores[None], foreground[None])[0]
        positive = rel > 0
        fallback = ~positive.any(dim=1)
        anchor = rel.argmax(dim=-1)
        selected = positive | (F.one_hot(anchor, n).to(torch.bool) & fallback[:, None])
        masked = rel.masked_fill(~selected, float('-inf'))
        # Same training invariant: strict positive disjoint for K>=2.
        assert bool((positive.sum(dim=0) <= 1).all())
    assert bool(selected.any(dim=-1).all())
    weights = F.softmax(masked / temperature, dim=-1)
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError('non-finite soft support weight')
    return selected, positive, fallback, weights


def resize_gt_mask(gt_array, box, grid=32):
    if gt_array.ndim != 2:
        raise ValueError('GT part mask must be a 2D ID map')
    left, top, right, bottom = (int(v) for v in box)
    h, w = gt_array.shape
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise RuntimeError(f'invalid predicted crop box {box} for GT {(h, w)}')
    # Exactly the box from the frozen Talk2DINO predicted mask; NOT GT bbox.
    crop = np.asarray(gt_array[top:bottom, left:right], dtype=np.int32)
    # Match the GT patch-grid analysis protocol (NEAREST, integer class IDs).
    grid_mask = Image.fromarray(crop, mode='I').resize((grid, grid), resample=Image.Resampling.NEAREST)
    return np.asarray(grid_mask, dtype=np.int32).reshape(-1).copy()


def audit_unit_tests():
    device='cpu'
    x = torch.tensor([[.8,.6,.4,.3],[.7,.7,.1,.5],[.2,.1,.8,.2]], device=device)
    fg = torch.tensor([True, True, True, True])
    sel,pos,fb,weights = support_from_scores(x,fg,.05)
    assert pos[0].tolist()==[True,False,False,False]
    assert pos[1].tolist()==[False,True,False,True]
    assert pos[2].tolist()==[False,False,True,False]
    assert not fb.any() and sel.sum().item()==4
    assert torch.allclose(weights.sum(-1),torch.ones(3))
    single = x[:1]
    s,_,f,_ = support_from_scores(single,fg,.05)
    assert s.sum().item()==4 and not f.any()
    ties = torch.tensor([[.1,.3],[.4,.6],[.7,.9]])
    s,p,f,_ = support_from_scores(ties,torch.ones(2,dtype=torch.bool),.05)
    assert f[:2].all() and s[:2].sum().item()==2 and p[:2].sum().item()==0
    label = np.full((64,64),255,dtype=np.int32)
    label[16:48,16:48]=41
    out=resize_gt_mask(label,[16,16,48,48])
    assert out.size==1024 and np.all(out==41)
    corr=int(np.count_nonzero(out[np.array([0,1,2])]==41))
    assert corr==3
    print('SELF_TEST_PASS: positive r>0, fallback, soft absolute, crop+nearest GT, denominator')


def main():
    args=parse_args()
    if args.self_test:
        audit_unit_tests(); return
    paths={
        'cache':absolute(args.cache),'projector':absolute(args.projector),
        'text_bank':absolute(args.text_bank),'config':absolute(args.model_config),
        'gt_parts':absolute(args.gt_part_root), 'gt_protos':absolute(args.gt_prototypes),
        'out':absolute(args.out_dir),
    }
    for key in ('cache','projector','text_bank','config','gt_protos'):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    if not paths['gt_parts'].is_dir():
        raise NotADirectoryError(paths['gt_parts'])
    if paths['out'].exists() and any(paths['out'].iterdir()):
        raise FileExistsError(f'Output directory must be empty, refusing to overwrite: {paths["out"]}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False

    # Use exactly the same CLIP bank, projector builder, and normalization as trainer.
    import train_relproto_alignemt as original
    print('[1/5] Loading 116 CLIP part features through baked final-epoch projector...',flush=True)
    raw,names,_=original.load_raw_clip_bank(
        paths['text_bank'], expected_clip_model='ViT-B/16',
        expected_template='sub_imagenet_template')
    assert len(names)==116, f'bad text labels: {len(names)}'
    projector,_=original.load_frozen_projector(
        project_root=ROOT, config_path=paths['config'],
        weight_path=paths['projector'], device=device)
    with torch.no_grad():
        proj=original.project_raw_clip_bank(raw,projector,device=device,batch_size=128)
        text=F.normalize(proj.float().to(device),dim=-1,eps=1e-12)
    del projector
    if tuple(text.shape)!=(116,768):
        raise RuntimeError(f'wrong projected text shape: {tuple(text.shape)}')

    print('[2/5] Text–GT Prototype cosine (precomputed global GT bank)...',flush=True)
    gtobj=load_pt(paths['gt_protos'])
    if not isinstance(gtobj,dict) or 'prototypes' not in gtobj:
        raise RuntimeError('GT prototype file must have dict["prototypes"]')
    gtproto=torch.as_tensor(gtobj['prototypes'],dtype=torch.float32)
    if tuple(gtproto.shape)!=(116,768):
        raise RuntimeError(f'GT bank shape mismatch: {gtproto.shape}')
    for name_key in ('class_names','classes','part_classes'):
        if name_key in gtobj and len(gtobj[name_key])==116:
            if [str(n) for n in gtobj[name_key]]!=[str(n) for n in names]:
                raise RuntimeError(f'GT prototype taxonomy mismatch: {name_key}')
    available=torch.isfinite(gtproto).all(-1) & (gtproto.norm(dim=-1)>1e-9)
    if 'patch_counts' in gtobj:
        available &= torch.as_tensor(gtobj['patch_counts']).reshape(-1)>0
    gtn=F.normalize(gtproto.to(device),dim=-1,eps=1e-12)
    cs=(text*gtn).sum(dim=-1).detach().cpu().numpy()
    validity=available.cpu().numpy()
    proto_rows=[{
        'part_id':i,'part_name':str(names[i]),
        'gt_prototype_valid':bool(validity[i]),
        'text_gt_cosine':float(cs[i]) if validity[i] else None,
    } for i in range(116)]
    gt_cosine_macro=finite_mean([r['text_gt_cosine'] for r in proto_rows])

    print('[3/5] Loading original PP116 predicted-object CROP cache with mmap...',flush=True)
    cache=load_pt(paths['cache'])
    if not isinstance(cache,dict) or not isinstance(cache.get('annotations'),list):
        raise RuntimeError('Expected dict with list field "annotations"')
    meta=cache.get('pred_obj_cropaug_meta',{})
    if meta.get('format')!='pascalpart116_weaklabels_predobj_cropaug_v1':
        raise RuntimeError('This audit is for Presence Soft Relative, expected original part-presence cache format; got '+str(meta.get('format')))
    if meta.get('patch_grid')!=[32,32] or meta.get('crop_resize')!=[448,448]:
        raise RuntimeError('Cache grid / geometry differs from expected 32x32/448x448')
    if float(meta.get('bg_thresh',-1))!=.55 or not meta.get('pamr',False):
        raise RuntimeError('Cache foreground provenance differs from intended bg=.55/PAMR')
    part_ignore=int(meta.get('part_ignore',255)); part_offset=int(meta.get('part_id_offset',0))
    image_sizes={str(im['id']):(int(im['height']),int(im['width'])) for im in cache['images']}
    annotations=cache['annotations']
    if args.max_examples:
        annotations=annotations[:args.max_examples]
    print(f'  object crops={len(annotations)}; unique cached images={len(image_sizes)}',flush=True)

    # Per-image mask LRU avoids repetitive image reads for several object crops.
    @lru_cache(maxsize=12)
    def get_gt(stem):
        p=paths['gt_parts']/f'{stem}.png'
        if not p.is_file():
            raise FileNotFoundError(f'GT part mask missing: {p}')
        with Image.open(p) as im:
            arr=np.asarray(im).copy()
        if arr.ndim==3:
            if (arr.shape[2]>=3 and np.array_equal(arr[...,0],arr[...,1])
                    and np.array_equal(arr[...,0],arr[...,2])):
                arr=arr[...,0]
            else:
                raise RuntimeError(f'color GT mask cannot be used as class IDs: {p}')
        if arr.ndim!=2:
            raise RuntimeError(f'GT part mask is not 2D: {p} {arr.shape}')
        if stem in image_sizes and tuple(arr.shape)!=image_sizes[stem]:
            raise RuntimeError(f'GT mask and cached RGB dimensions differ for {stem}: {arr.shape} != {image_sizes[stem]}')
        raw_labels=arr.astype(np.int32)
        good=(raw_labels==part_ignore) | ((raw_labels-part_offset>=0)&(raw_labels-part_offset<116))
        if not bool(good.all()):
            raise RuntimeError(f'GT part ID outside 0..115/ignore in {stem}: {np.unique(raw_labels[~good])[:10]}')
        return np.where(raw_labels==part_ignore,255,raw_labels-part_offset).astype(np.int32)

    print('[4/5] Replaying training support (without GT), then auditing each selected token...',flush=True)
    rows=[]
    # GPU mini-batches grouped by number of current candidate parts, as in trainer.
    grouped=defaultdict(list)
    for a in annotations:
        pids=[int(v) for v in a['part_category_id']]
        if not pids or len(set(pids))!=len(pids):
            raise RuntimeError(f'Invalid part IDs for crop {a.get("id")}')
        if any(v<0 or v>=116 for v in pids):
            raise RuntimeError(f'Part ID out of range in crop {a.get("id")}')
        if tuple(a['cropaug_patch_tokens'].shape)!=(1024,768):
            raise RuntimeError(f'Token shape mismatch crop {a.get("id")}')
        if tuple(a['pred_obj_mask_patch'].shape)!=(1024,):
            raise RuntimeError(f'FG shape mismatch crop {a.get("id")}')
        if a.get('cropaug_box_xyxy') is None:
            raise RuntimeError(f'Crop box missing crop {a.get("id")}')
        grouped[len(pids)].append(a)
    with torch.no_grad():
        for k, group in tqdm(sorted(grouped.items()),desc='Candidate-size groups'):
            for start in range(0,len(group),args.batch_size):
                batch=group[start:start+args.batch_size]
                tokens=torch.stack([a['cropaug_patch_tokens'] for a in batch]).to(device=device,dtype=torch.float32)
                tokens=original.normalize_last(tokens)
                foreground=torch.stack([a['pred_obj_mask_patch'] for a in batch]).to(device=device,dtype=torch.bool)
                part_ids=torch.tensor([[int(i) for i in a['part_category_id']] for a in batch],device=device,dtype=torch.long)
                scores=torch.bmm(text[part_ids],tokens.transpose(1,2))
                for j,a in enumerate(batch):
                    # Stage 1: compute supports using ONLY frozen features, cached predicted mask and known part IDs.
                    selected,positive,fallback,weights=support_from_scores(
                        scores[j],foreground[j],args.temperature,relative_scores_fn=original.relative_scores)
                    selected=selected.cpu().numpy()
                    positive=positive.cpu().numpy()
                    fallback=fallback.cpu().numpy()
                    weights=weights.cpu().numpy()
                    # Stage 2: read GT mask for OFFLINE evaluation *after* supports were selected.
                    stem=str(a['image_id'])
                    box=a['cropaug_box_xyxy'].tolist() if torch.is_tensor(a['cropaug_box_xyxy']) else a['cropaug_box_xyxy']
                    gtgrid=resize_gt_mask(get_gt(stem),box)
                    for pp,pid in enumerate(a['part_category_id']):
                        pid=int(pid)
                        chosen=selected[pp]
                        nsel=int(chosen.sum())
                        correct=int((gtgrid[chosen]==pid).sum())
                        ignored=int((gtgrid[chosen]==255).sum())
                        valid=nsel-ignored
                        npos=int(positive[pp].sum())
                        pos_correct=int((gtgrid[positive[pp]]==pid).sum()) if npos else 0
                        wt_correct=float(weights[pp][gtgrid==pid].sum())
                        wt_ignore=float(weights[pp][gtgrid==255].sum())
                        rows.append({
                            'crop_id':int(a['id']),'image_id':stem,
                            'object_name':str(a['class_name']),'part_id':pid,
                            'part_name':str(names[pid]),'candidate_part_count':k,
                            'selection_type':('soft_absolute_K1' if k==1 else
                                              'positive_margin' if not fallback[pp] else 'fallback_argmax'),
                            'positive_margin_tokens':npos,'fallback':int(fallback[pp]),
                            'selected_tokens':nsel,'correct_tokens':correct,
                            'wrong_valid_gt_tokens':valid-correct,'ignore_gt_tokens':ignored,
                            'purity_all_selected':frac(correct,nsel),
                            'purity_valid_gt_only':frac(correct,valid),
                            'strict_positive_purity':(frac(pos_correct,npos) if k>=2 else None),
                            'weighted_purity_aux':wt_correct,
                            'weighted_ignore_aux':wt_ignore,
                            'crop_box_xyxy':','.join(map(str,box)),
                        })

    print('[5/5] Aggregating CSV/JSON...',flush=True)
    classgroups=defaultdict(list); imagegroups=defaultdict(list)
    for r in rows:
        classgroups[r['part_id']].append(r)
        imagegroups[r['image_id']].append(r)
    per_class=[]
    for pid in range(116):
        g=classgroups[pid]
        per_class.append({
            'part_id':pid,'part_name':str(names[pid]),'crop_part_occurrences':len(g),
            'macro_purity':finite_mean([r['purity_all_selected'] for r in g]),
            'pooled_selected_purity':frac(sum(r['correct_tokens'] for r in g),sum(r['selected_tokens'] for r in g)),
            'strict_positive_macro':finite_mean([r['strict_positive_purity'] for r in g]),
            'text_gt_cosine':proto_rows[pid]['text_gt_cosine'],
        })
    per_image=[]
    for stem,group in sorted(imagegroups.items()):
        per_image.append({
            'image_id':stem,'crop_part_occurrences':len(group),
            'mean_purity':finite_mean([r['purity_all_selected'] for r in group]),
            'selected_tokens':sum(r['selected_tokens'] for r in group),
            'correct_tokens':sum(r['correct_tokens'] for r in group),
            'pooled_purity':frac(sum(r['correct_tokens'] for r in group),sum(r['selected_tokens'] for r in group)),
        })
    competitive=[r for r in rows if r['candidate_part_count']>=2]
    pos=[r for r in competitive if r['positive_margin_tokens']>0]
    fallback=[r for r in competitive if r['fallback']]
    single=[r for r in rows if r['candidate_part_count']==1]
    summary={
        'projector':str(paths['projector']),'cache':str(paths['cache']),
        'gt_part_root':str(paths['gt_parts']),'gt_prototype_bank':str(paths['gt_protos']),
        'evaluation_units':'each cached predicted-object crop x candidate part (train split)',
        'gt_used_only_for_offline_check':True,'all_116_text_classes_for_gt_cosine':True,
        'definition_primary':'matching GT tokens / all selected tokens, GT ignore counts in denominator as incorrect',
        'support_definition':'K>=2 strictly relative margin >0, argmax fallback when empty; K==1 all predicted FG Soft Absolute',
        'mask_geometry':'cached predicted cropaug_box_xyxy, crop nearest resize 32x32 (one GT label per DINO patch)',
        'n_crops':len(annotations),'n_unique_images':len(imagegroups),'n_crop_part_pairs':len(rows),
        'support_purity_all_candidates_image_macro':finite_mean([r['mean_purity'] for r in per_image]),
        'support_purity_all_candidates_pair_macro':finite_mean([r['purity_all_selected'] for r in rows]),
        'support_purity_all_candidates_class_macro':finite_mean([r['macro_purity'] for r in per_class]),
        'support_purity_all_candidates_token_micro':frac(sum(r['correct_tokens'] for r in rows),sum(r['selected_tokens'] for r in rows)),
        'competitive_K_ge_2':{
            'crop_part_pairs':len(competitive),
            'strict_positive_pairs':len(pos),
            'fallback_pairs':len(fallback),
            'strict_positive_pair_macro':finite_mean([r['strict_positive_purity'] for r in pos]),
            'strict_positive_class_macro':finite_mean([finite_mean([r['strict_positive_purity'] for r in pos if r['part_id']==p]) for p in range(116)]),
            'strict_positive_token_micro':frac(sum(r['correct_tokens'] for r in pos),sum(r['selected_tokens'] for r in pos)),
            'fallback_pair_macro':finite_mean([r['purity_all_selected'] for r in fallback]),
        },
        'single_K_eq_1':{'crop_part_pairs':len(single),'pair_macro':finite_mean([r['purity_all_selected'] for r in single])},
        'gt_text_cosine_macro':gt_cosine_macro,
        'gt_text_cosine_valid_classes':int(validity.sum()),
    }
    out=paths['out']; out.mkdir(parents=True,exist_ok=True)
    if len(rows)==0:
        raise RuntimeError('zero annotation-part rows')
    write_csv(out/'support_purity_per_crop_part.csv',rows,list(rows[0].keys()))
    write_csv(out/'support_purity_per_image.csv',per_image,list(per_image[0].keys()))
    write_csv(out/'support_purity_per_class.csv',per_class,list(per_class[0].keys()))
    write_csv(out/'text_gt_prototype_similarity.csv',proto_rows,list(proto_rows[0].keys()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('\n===== Soft Relative support audit (final baked projector) =====')
    print('Crop-part pairs:',len(rows),'unique images:',len(imagegroups))
    for label,key in [
        ('All actual constructed supports, image macro','support_purity_all_candidates_image_macro'),
        ('All actual supports, crop-part macro','support_purity_all_candidates_pair_macro'),
        ('All actual supports, class macro','support_purity_all_candidates_class_macro'),
        ('All actual supports, patch-token micro','support_purity_all_candidates_token_micro'),
    ]:
        v=summary[key];print(f'{label}: {v:.4%}' if v is not None else f'{label}: N/A')
    c=summary['competitive_K_ge_2']
    print('K>=2 true r>0 support crop-part macro:',c['strict_positive_pair_macro'])
    print('K>=2 true r>0 support class macro:',c['strict_positive_class_macro'])
    print('K>=2 fallback crop-part pairs:',c['fallback_pairs'],'/',c['crop_part_pairs'])
    print('K=1 soft absolute crop-part pairs:',len(single))
    print(f'Text–GT global prototype cosine class macro: {gt_cosine_macro:.6f} ({int(validity.sum())}/116)' if gt_cosine_macro is not None else 'GT prototype cosine: N/A')
    print('Saved to:',out)


if __name__=='__main__':
    main()
