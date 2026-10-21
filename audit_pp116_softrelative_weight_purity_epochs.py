#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""READ-ONLY PP116 audit: Soft Relative/Absolute SOFTMAX mass on GT part masks.

This is an offline diagnostic; GT masks NEVER affect patch selection or training.
Load original train_pp116_softsingle_dual.py and cross-check its prototype output
against independently reconstructed per-token weights (K>=2).
Run from Talk2DINO_official_bg root; creates output CSV/JSON only.
"""
from __future__ import annotations
import argparse, csv, json, math, os, functools
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import train_relproto_alignemt as original
import train_pp116_softsingle_dual as trainer

DEFAULT_RUN='final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'
DEFAULT_CACHE='feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'
DEFAULT_TEXT='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
DEFAULT_BASE='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'

def load_pt(p):
    # Only load locally generated, trusted model/cache files.
    return torch.load(str(p),map_location='cpu',weights_only=False)

def write_csv(p,rows):
    if not rows: raise RuntimeError('No rows for '+str(p))
    p=Path(p)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def class_names():
    # Reuse the training taxonomy, rather than creating an independent ID map.
    from analyze_pp116_within_object_structure import canonical_classes
    return canonical_classes(Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))

def get_annotations(cache):
    if isinstance(cache,(list,tuple)): return cache
    if isinstance(cache,dict):
        for k in ('annotations','records','samples','items'):
            if isinstance(cache.get(k),(list,tuple)): return cache[k]
    raise TypeError('Unexpected cache structure: inspect cache keys, no guessing allowed: '+str(list(cache)[:30] if isinstance(cache,dict) else type(cache)))

def values(x):
    if torch.is_tensor(x): return x.detach().cpu().flatten().tolist()
    if isinstance(x,np.ndarray): return x.reshape(-1).tolist()
    if isinstance(x,(list,tuple)): return list(x)
    return [x]

def get_pids(ann):
    if 'part_category_id' not in ann: raise KeyError('Cache annotation missing part_category_id')
    pids=[int(q) for q in values(ann['part_category_id'])]
    if not pids or len(pids)!=len(set(pids)) or any(p<0 or p>=116 for p in pids):
        raise ValueError('Invalid 0..115 part IDs: '+str(pids))
    return pids

def find_mask(root,image_id):
    stem=Path(str(image_id)).stem
    for ext in ('.png','.tif','.tiff','.bmp'):
        q=root/(stem+ext)
        if q.exists(): return q
    matches=[p for p in root.glob(stem+'.*') if p.suffix.lower() in ('.png','.tif','.tiff','.bmp')]
    if len(matches)==1:return matches[0]
    raise FileNotFoundError(f'Cannot uniquely find part GT for image_id={image_id!r} in {root}; matches={matches}')

def nearest_crop_gt(path,box,grid=32):
    with Image.open(path) as im:
        gt=np.array(im)
    if gt.ndim!=2:raise ValueError(f'GT mask must be indexed semantic labels, got {gt.shape}: {path}')
    H,W=gt.shape
    coords=np.asarray(values(box),dtype=np.float64)
    if coords.shape!=(4,) or not np.isfinite(coords).all():
        raise ValueError('invalid cropaug_box_xyxy: '+str(coords))
    x0,y0,x1,y1=(int(round(float(v))) for v in coords)
    x0=max(0,min(W,x0));x1=max(0,min(W,x1));y0=max(0,min(H,y0));y1=max(0,min(H,y1))
    if x1<=x0 or y1<=y0: raise ValueError(f'Invalid crop box {coords} for GT shape {(H,W)}')
    cropped=gt[y0:y1,x0:x1]
    # Preserve integer IDs and ignore=255; never bilinear-interpolate labels.
    out=np.asarray(Image.fromarray(cropped.astype(np.int32),mode='I').resize((grid,grid),resample=Image.Resampling.NEAREST),dtype=np.int64).reshape(-1)
    invalid=np.unique(out[(out!=255)&((out<0)|(out>=116))])
    if len(invalid):raise ValueError(f'Unexpected GT labels {invalid.tolist()}: {path}')
    return out

def selection(current_text, tokens, fg, tau):
    """Replicates train_pp116_softsingle_dual.py's K=1 and K>=2 branches.

    Returns [K,1024] float weights and candidate bool; computes visual and
    checks against the original trainer's soft_relative_prototypes function.
    """
    K=current_text.size(0);n=tokens.size(0)
    if K>=2:
        abs_scores=current_text@tokens.T
        rel=original.relative_scores(abs_scores[None],fg[None])[0]
        positive=rel>0
        empty=~positive.any(-1)
        anchor=rel.argmax(dim=-1)
        cand=positive | (F.one_hot(anchor,n).bool() & empty[:,None])
        logits=(rel/tau).masked_fill(~cand,-torch.inf)
        weight=logits.softmax(dim=-1)
        visual=F.normalize(weight@tokens,dim=-1,eps=1e-12)
        # REAL source function comparison. Fail fast if training is different.
        expected,_=trainer.soft_relative_prototypes(current_text[None],tokens[None],fg[None],temperature=tau)
        err=float((visual-expected[0]).abs().max().item())
        if err>2e-5: raise RuntimeError(f'Local support selection diverges from trainer, max err={err}')
        return weight,cand,positive,empty,visual
    if K==1:
        abs_scores=current_text@tokens.T
        cand=fg[None].expand(1,n).clone()
        score=(abs_scores/tau).masked_fill(~cand,-torch.inf)
        weight=score.softmax(-1)
        visual=F.normalize(weight@tokens,dim=-1,eps=1e-12)
        return weight,cand,cand,torch.tensor([False],device=fg.device),visual
    raise ValueError('Zero part candidates')

def self_test():
    D=768;N=1024
    device=torch.device('cpu')
    tok=F.normalize(torch.randn(N,D,device=device),dim=-1)
    fg=torch.zeros(N,dtype=torch.bool);fg[:63]=True
    t=F.normalize(torch.randn(3,D),dim=-1)
    w,c,p,e,v=selection(t,tok,fg,0.05)
    assert w.shape==(3,N) and torch.allclose(w.sum(-1),torch.ones(3),atol=1e-5)
    assert (w[:,~fg]==0).all() and (c&~fg).sum()==0
    w1,c1,p1,e1,v1=selection(t[:1],tok,fg,0.05)
    assert w1.shape==(1,N) and bool((w1[:,~fg]==0).all())
    print('SELF_TEST_PASS: original multi-part prototype agrees, single-part branch and foreground checked')

def run(args):
    if args.self_test: return self_test()
    if not args.gt_part_root.is_dir(): raise FileNotFoundError(args.gt_part_root)
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError('Refusing overwrite: '+str(args.out_dir))
    if not 0 in args.epochs or max(args.epochs)>100: raise ValueError('Include E0; supported checkpoints up to E100')
    device=torch.device(args.device)
    names=class_names()
    raw,names_raw,_=original.load_raw_clip_bank(args.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if names!=list(names_raw):raise ValueError('Class order mismatch between CLIP text bank and taxonomy')
    projector,_=original.load_frozen_projector(project_root=Path.cwd(),config_path=args.model_config,weight_path=args.projector,device=device)
    base=original.project_raw_clip_bank(raw,projector,device=device,batch_size=128).to(device=device,dtype=torch.float32)
    base=original.normalize_last(base)
    del projector
    texts={}
    for ep in args.epochs:
        if ep==0: W=torch.eye(768,device=device)
        else:
            ck=args.weights_dir/f'W_epoch_{ep:03d}.pt'
            if not ck.exists():raise FileNotFoundError(ck)
            data=load_pt(ck)
            if not isinstance(data,dict) or 'W' not in data:raise ValueError(f'Bad checkpoint {ck}')
            W=torch.as_tensor(data['W'],dtype=torch.float32,device=device)
            if W.shape!=(768,768):raise ValueError(f'Bad W shape {W.shape}')
            err=float((W.T@W-torch.eye(768,device=device)).abs().max())
            if err>1e-2:raise ValueError(f'Nonorthogonal W at E{ep}: {err}')
        texts[ep]=original.normalize_last(base@W)
    cache=load_pt(args.cache)
    ann=get_annotations(cache)
    if args.max_crops>0:ann=ann[:args.max_crops]
    if not ann:raise RuntimeError('No annotations')
    args.out_dir.mkdir(parents=True)
    per_crop=[]
    aggr=defaultdict(lambda:defaultdict(float))
    conf_mass=defaultdict(float)  # (epoch, predicted pseudo-part, GT part) -> softmax mass
    @functools.lru_cache(maxsize=256)
    def gt_cached(stem):
        path=find_mask(args.gt_part_root,stem)
        with Image.open(path) as im: return np.asarray(im).astype(np.int32)
    for ii,a in enumerate(ann):
        if not isinstance(a,dict):raise TypeError(f'annotation {ii} not dict')
        req=('image_id','cropaug_box_xyxy','cropaug_patch_tokens','pred_obj_mask_patch','part_category_id')
        miss=[k for k in req if k not in a]
        if miss:raise KeyError(f'annotation {ii} missing {miss}')
        ids=get_pids(a)
        token=torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32,device=device)
        fg=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool,device=device).flatten()
        if token.shape!=(1024,768) or fg.shape!=(1024,) or not bool(fg.any()):raise ValueError(f'Bad tokens/foreground annotation={ii} shape={token.shape}/{fg.shape}')
        token=original.normalize_last(token)
        rawgt=gt_cached(Path(str(a['image_id'])).stem)
        box=np.asarray(values(a['cropaug_box_xyxy']),dtype=np.float64)
        if box.shape!=(4,):raise ValueError(f'Bad crop box {box}')
        HH,WW=rawgt.shape
        x0,y0,x1,y1=(int(round(float(x))) for x in box)
        x0=max(0,min(WW,x0));x1=max(0,min(WW,x1));y0=max(0,min(HH,y0));y1=max(0,min(HH,y1))
        if x1<=x0 or y1<=y0:raise RuntimeError(f'Empty crop for image {a["image_id"]}')
        gt=np.asarray(Image.fromarray(rawgt[y0:y1,x0:x1].astype(np.int32),mode='I').resize((32,32),resample=Image.Resampling.NEAREST),dtype=np.int32).reshape(-1)
        if ((gt!=255)&((gt<0)|(gt>=116))).any(): raise ValueError('GT labels incompatible with [0,115]/255')
        gt_t=torch.from_numpy(gt.astype(np.int64)).to(device)
        with torch.no_grad():
            for ep in args.epochs:
                t=texts[ep][ids]
                weights,cands,pos,fallback,proto=selection(t,token,fg,args.soft_temperature)
                for local,pid in enumerate(ids):
                    w=weights[local]
                    candidate=cands[local]
                    true=(gt_t==pid)
                    goodmass=float(w[true].sum().item())
                    ignmass=float(w[gt_t==255].sum().item())
                    wrongmass=max(0.,1-goodmass-ignmass)
                    ngt=int(true.sum().item());nfg=int((true&fg).sum().item())
                    nc=int(candidate.sum().item());correct=int((candidate&true).sum().item())
                    entropy=float((-(w[w>0]*w[w>0].log())).sum().item())
                    wrong_gt=(gt_t!=pid)&(gt_t!=255)
                    wrong_labels=gt_t[wrong_gt]
                    masses=w[wrong_gt]
                    agg_mass=torch.zeros(116,device=device).scatter_add_(0,wrong_labels,masses)
                    j=int(agg_mass.argmax().item())
                    for gt_id in np.unique(gt[(gt>=0)&(gt<116)]):
                        conf_mass[(ep,pid,int(gt_id))] += float(w[gt_t==int(gt_id)].sum().item())
                    rec={'epoch':ep,'crop_index':ii,'image_id':str(a['image_id']),
                         'class_id':pid,'part':names[pid],'object':names[pid].split("'s ",1)[0],
                         'candidate_parts':len(ids),'gt_patches_in_crop':ngt,'gt_patches_in_foreground':nfg,
                         'foreground_recall':nfg/ngt if ngt else math.nan,
                         'candidate_patch_count':nc,'candidate_gt_correct_count':correct,
                         'candidate_purity':correct/nc if nc else math.nan,
                         'candidate_weighted_gt_purity':goodmass,
                         'candidate_weighted_ignore_mass':ignmass,
                         'candidate_weighted_wrong_mass':wrongmass,
                         'weighted_valid_purity':goodmass/(1-ignmass) if ignmass<1-1e-9 else math.nan,
                         'fallback':int(bool(fallback[local])) if len(ids)>1 else 0,
                         'k1_soft_absolute':int(len(ids)==1),'effective_patch_count':float(1/w.square().sum().item()),
                         'weight_entropy':entropy,
                         'dominant_wrong_gt_part':names[j] if float(agg_mass[j])>1e-8 else '',
                         'dominant_wrong_gt_mass':float(agg_mass[j]),
                         'pseudo_gt_same_crop_cos':float((proto[local]@F.normalize(token[true].mean(0),dim=-1)).item()) if ngt>0 else math.nan}
                    per_crop.append(rec)
                    agg=aggr[(ep,pid)]
                    agg['n']+=1;agg['weight_correct_sum']+=goodmass;agg['weight_ignore_sum']+=ignmass
                    agg['candidate_correct']+=correct;agg['candidate_count']+=nc
                    agg['fg_recall_num']+=nfg;agg['fg_recall_den']+=ngt
                    agg['fallback']+=rec['fallback'];agg['k1']+=rec['k1_soft_absolute']
                    agg['proto_cos_sum']+=rec['pseudo_gt_same_crop_cos'] if math.isfinite(rec['pseudo_gt_same_crop_cos']) else 0
                    agg['proto_cos_n']+=int(math.isfinite(rec['pseudo_gt_same_crop_cos']))
                    agg['wrong_mass_sum']+=wrongmass
        if (ii+1)%500==0:print(f'[audit] crops {ii+1}/{len(ann)}; crop-part records={len(per_crop)}',flush=True)
    summary=[]
    for (ep,pid),v in sorted(aggr.items()):
        n=v['n'];summary.append({'epoch':ep,'class_id':pid,'part':names[pid],
          'n_crop_part':int(n),'mean_weighted_gt_purity':v['weight_correct_sum']/n,
          'mean_weighted_ignore_mass':v['weight_ignore_sum']/n,
          'mean_weighted_wrong_mass':v['wrong_mass_sum']/n,
          'candidate_patch_purity_micro':v['candidate_correct']/max(1,v['candidate_count']),
          'foreground_recall_micro':v['fg_recall_num']/v['fg_recall_den'] if v['fg_recall_den'] else math.nan,
          'mean_pseudo_gt_same_crop_cos':v['proto_cos_sum']/v['proto_cos_n'] if v['proto_cos_n'] else math.nan,
          'fallback_fraction':v['fallback']/n,'k1_fraction':v['k1']/n})
    write_csv(args.out_dir/'per_crop_part.csv',per_crop)
    write_csv(args.out_dir/'per_part_epoch.csv',summary)
    confusion=[]
    for (ep,pid,gt_pid),mass in sorted(conf_mass.items()):
        num=aggr[(ep,pid)]['n']
        confusion.append({'epoch':ep,'pseudo_part':names[pid],'gt_part':names[gt_pid],
                          'sum_softmax_mass':mass,'mean_softmax_mass_per_crop_part':mass/num,
                          'is_correct_part':int(pid==gt_pid),'n_crop_part':int(num)})
    write_csv(args.out_dir/'soft_assignment_gt_confusion.csv',confusion)
    with (args.out_dir/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump({'experiment':'GT-only OFFLINE audit; no training with GT',
          'source_trainer':'train_pp116_softsingle_dual.py; K>=2 prototype verified numerically',
          'epochs':args.epochs,'cache':str(args.cache),'weights_dir':str(args.weights_dir),
          'part_gt_dir':str(args.gt_part_root),'num_crops':len(ann),
          'gt_geometry':'round(cropaug_box_xyxy), PIL nearest to 32x32; verify cache augment provenance',
          'gt_ignore':255,'denominator':'full softmax weight including ignore; valid purity additionally reported',
          'soft_temperature':args.soft_temperature,'max_crops':args.max_crops},f,ensure_ascii=False,indent=2)
    print('[DONE]',args.out_dir,'crops=',len(ann),'crop-part-epochs=',len(per_crop))

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--text-bank',type=Path,default=DEFAULT_TEXT)
    p.add_argument('--projector',type=Path,default=DEFAULT_BASE)
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--gt-part-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--weights-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--epochs',nargs='+',type=int,default=[0,10,30,100])
    p.add_argument('--soft-temperature',type=float,default=0.05)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--max-crops',type=int,default=0)
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_softrelative_weight_purity_epochs'))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    a.epochs=sorted(set(a.epochs))
    return a

if __name__=='__main__':run(parse())
