#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""READ-ONLY offline GT-weighted Soft Relative prototype purity audit, PP116.

GT is read ONLY AFTER predicted foreground / per-part Soft Relative scores and
Softmax weights have been computed. There is no optimizer, gradient or GT in training.
Uses frozen CLIP + COCO projector and original.relative_scores; checks the
resulting pseudo prototype numerically against trainer.soft_relative_prototypes.

GT projection: indexed semantic PNG mask -> original-pixel cropaug_box_xyxy ->
(optional recorded H-flip) -> nearest resize to 32x32. Inspect geometry metadata
before treating results as trustworthy; a box alone cannot reconstruct any
unrecorded spatial augmentation.
"""
from __future__ import annotations
import argparse, ast, csv, json, math, re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

D=768; N=1024; C=116; IGN=255
DEFAULT_FOCUS=["person's upper arm", "cow's horn", "sheep's horn", "bicycle's saddle", "cat's leg", "aeroplane's body", "dog's tail", "train's front"]


def load_pt(p):
    # Only use local files you trust; checkpoints/cache may contain pickle objects.
    return torch.load(str(p),map_location='cpu',weights_only=False)


def canonical_names(p):
    tree=ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PART_CLASSES' for t in n.targets):
            names=list(ast.literal_eval(n.value))
            if len(names)==C and len(set(names))==C:return names
    raise RuntimeError(f'Cannot parse 116 PART_CLASSES from {p}')


def annotations(payload):
    if isinstance(payload,(list,tuple)):return payload
    if isinstance(payload,dict):
        for k in ('annotations','records','samples','items'):
            if isinstance(payload.get(k),(list,tuple)):return payload[k]
    raise TypeError(f'Unsupported cached record structure: {type(payload)}')


def flat(x):
    if torch.is_tensor(x):return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x,np.ndarray):return x.reshape(-1).tolist()
    if isinstance(x,(list,tuple)):return list(x)
    return [x]


def part_ids(a):
    if 'part_category_id' not in a:raise KeyError('Missing part_category_id')
    ids=[int(x) for x in flat(a['part_category_id'])]
    if not ids or len(ids)!=len(set(ids)) or any(not 0<=p<C for p in ids):
        raise ValueError(f'Invalid part IDs {ids}')
    return ids


def get_box(a):
    if 'cropaug_box_xyxy' not in a:raise KeyError('Missing cropaug_box_xyxy')
    b=np.asarray(flat(a['cropaug_box_xyxy']),dtype=np.float64)
    if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError(f'Bad crop box {b}')
    return b


def get_flip(a, mode):
    if mode=='none':return False
    if mode=='all':return True
    # Explicit recorded crop flip metadata. Other spatial transformations cannot
    # be reconstructed from xyxy and trigger a fail-fast when recognizable.
    for key in ('cropaug_hflip','cropaug_flipped','cropaug_flip','hflip','is_flipped','flipped','flip'):
        if key not in a:continue
        v=a[key]
        if torch.is_tensor(v):
            if v.numel()!=1:raise ValueError(f'Unrecognized {key}={v}')
            v=v.item()
        if isinstance(v,(bool,int,np.integer)):return bool(v)
        if isinstance(v,str):
            v=v.lower().strip()
            if v in ('horizontal','h','hflip','true','1','yes'):return True
            if v in ('none','false','0','no',''):return False
            raise ValueError(f'Unsupported recorded geometric flip: {key}={v}')
        raise ValueError(f'Unsupported flip metadata: {key}={v}')
    if a.get('flip_direction') not in (None,'',False):
        if str(a['flip_direction']).lower()=='horizontal':return True
        raise ValueError(f'Unsupported flip_direction={a["flip_direction"]!r}')
    return False


class GTLookup:
    def __init__(self,root,suffix=''):
        self.root=Path(root);self.suffix=suffix
        if not self.root.is_dir():raise FileNotFoundError(f'GT mask directory absent: {root}. Locate actual train GT indexed PNG folder first.')
        self.index=None

    def path(self,image_id):
        stem=Path(str(image_id)).stem+self.suffix
        for ext in ('.png','.tif','.tiff','.bmp'):
            p=self.root/(stem+ext)
            if p.is_file():return p
        if self.index is None:
            self.index=defaultdict(list)
            for p in self.root.rglob('*'):
                if p.suffix.lower() in ('.png','.tif','.tiff','.bmp'):
                    self.index[p.stem].append(p)
        hits=self.index.get(stem,[])
        if len(hits)!=1:raise FileNotFoundError(f'Expected exactly one indexed GT mask for image_id={image_id!r} (stem {stem!r}) inside {self.root}; found {hits[:8]}')
        return hits[0]

    @lru_cache(maxsize=256)
    def full(self,image_id):
        path=self.path(image_id)
        with Image.open(path) as img:
            mode=img.mode
            gt=np.asarray(img)
        if gt.ndim!=2 or mode in ('RGB','RGBA'):
            raise ValueError(f'{path}: GT needs *indexed semantic class IDs*, not RGB colors. Got shape={gt.shape}, mode={mode}')
        gt=gt.astype(np.int32)
        invalid=np.unique(gt[(gt!=IGN)&((gt<0)|(gt>=C))])
        if invalid.size:raise ValueError(f'{path}: invalid part GT labels {invalid[:20].tolist()}. Verify this is 0..115, 255-ignore part mask, not object-mask.')
        return gt

    def crop32(self,a,flip_mode):
        image_id=str(a['image_id'])
        gt=self.full(image_id)
        h,w=gt.shape
        box=get_box(a)
        if 0<=box.max()<=1.01 and max(h,w)>8:
            raise ValueError(f'{image_id}: crop box appears normalized [0,1], expected pixel XYXY: {box}')
        x0,y0,x1,y1=[int(round(float(q))) for q in box]
        x0=max(0,min(w,x0));x1=max(0,min(w,x1));y0=max(0,min(h,y0));y1=max(0,min(h,y1))
        if x1<=x0 or y1<=y0:raise ValueError(f'{image_id}: invalid crop box {box} for GT {gt.shape}')
        crop=gt[y0:y1,x0:x1]
        flipped=get_flip(a,flip_mode)
        if flipped:crop=np.ascontiguousarray(crop[:,::-1])
        grid=np.asarray(Image.fromarray(crop.astype(np.int32),mode='I').resize((32,32),resample=Image.Resampling.NEAREST),dtype=np.int32)
        if grid.shape!=(32,32):raise AssertionError('GT resize failed')
        return grid.reshape(-1),flipped,(w,h),tuple(box)


def checkpoint_W(run,ep,lam):
    if ep==0:return torch.eye(D)
    p=run/f'W_epoch_{ep:03d}.pt'
    if not p.is_file():raise FileNotFoundError(p)
    obj=load_pt(p)
    if not isinstance(obj,dict) or 'W' not in obj:raise ValueError(f'Invalid checkpoint {p}')
    opts=obj.get('args',{})
    lamb=opts.get('lambda_struct') if isinstance(opts,dict) else getattr(opts,'lambda_struct',None)
    if lamb is None or not math.isclose(float(lamb),float(lam),abs_tol=1e-6):raise ValueError(f'Lambda mismatch in {p}: expected {lam}, got {lamb}')
    W=torch.as_tensor(obj['W'],dtype=torch.float32)
    if W.shape!=(D,D):raise ValueError(f'Bad W shape {W.shape}')
    if (W.T@W-torch.eye(D)).abs().max().item()>1e-2:raise ValueError(f'Nonorthogonal W: {p}')
    return W


def assignment(local_txt,tokens,fg,tau,rel_scores):
    K=local_txt.shape[0]
    if K==0:raise ValueError('Empty candidate list')
    if K==1:
        sim=local_txt@tokens.T
        valid=fg[None,:]
        w=torch.softmax((sim/tau).masked_fill(~valid,-torch.inf),dim=-1)
        cand=valid
        fallback=torch.zeros(1,dtype=torch.bool,device=tokens.device)
    else:
        sim=local_txt@tokens.T
        r=rel_scores(sim[None],fg[None])[0]
        positives=(r>0)&fg[None,:]
        fallback=~positives.any(-1)
        anchor=r.masked_fill(~fg[None,:],-torch.inf).argmax(-1)
        cand=positives | (F.one_hot(anchor,tokens.size(0)).bool() & fallback[:,None])
        w=torch.softmax((r/tau).masked_fill(~cand,-torch.inf),dim=-1)
    if not torch.isfinite(w).all() or not torch.allclose(w.sum(-1),torch.ones(K,device=w.device),atol=2e-5):
        raise RuntimeError('Softmax weights malformed')
    if bool((w[:,~fg]!=0).any()):raise RuntimeError('Nonforeground patch has positive weight')
    return w,cand,fallback


def writer(path,header):
    f=path.open('w',encoding='utf-8-sig',newline='')
    w=csv.DictWriter(f,fieldnames=header,extrasaction='ignore');w.writeheader()
    return f,w


def spearman_pair(x,y):
    from scipy.stats import spearmanr
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    keep=np.isfinite(x)&np.isfinite(y)
    a=x[keep];b=y[keep]
    if len(a)<5 or np.ptp(a)<1e-12 or np.ptp(b)<1e-12:return math.nan,int(len(a))
    return float(spearmanr(a,b).statistic),int(len(a))


def read_iou(path,names):
    if not path.is_file():raise FileNotFoundError(f'Need prior IoU CSV: {path}')
    out={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            ep=int(r['epoch']);pid=int(r['class_id'])
            if names[pid]!=r['part']:raise ValueError(f'IoU taxonomy differs at {pid}: {r["part"]}')
            key=(ep,pid)
            if key in out:raise ValueError(f'Duplicate IoU {key}')
            good=int(r['valid_val_gt'])==1
            out[key]=(good,{'lam0':float(r['lam0_iou']),'lam4':float(r['lam4_iou'])})
    return out


def preflight(args):
    cache=load_pt(args.cache);anns=annotations(cache)
    if not anns:raise ValueError('Empty cache')
    print(f'[CACHE] {len(anns)} records, first annotation keys:')
    a=anns[0]
    for k in sorted(a):
        v=a[k]
        if torch.is_tensor(v):desc=f'tensor {tuple(v.shape)}, dtype={v.dtype}'
        elif isinstance(v,np.ndarray):desc=f'ndarray {v.shape}, dtype={v.dtype}'
        else:desc=repr(v)[:140]
        print('  ',k,':',desc)
    print('[GUIDANCE] Verify image_id, cropaug_box_xyxy, recorded spatial augmentations, and actual GT mask directory.')
    print('[GUIDANCE] A bounding box alone CANNOT undo a spatial transformation that was not recorded in the cache.')
    if args.gt_root.is_dir():
        loc=GTLookup(args.gt_root,args.gt_suffix)
        p=loc.path(str(a['image_id']))
        grid,flip,shape,box=loc.crop32(a,args.flip_mode)
        print('[GT FOUND]',p,'image_wh=',shape,'box=',box,'hflip=',flip,'grid_labels=',np.unique(grid).tolist())
    else:print('[GT NOT FOUND] gt-root missing:',args.gt_root)


def self_test():
    dev='cpu'
    t=torch.zeros(3,D);t[0,0]=1;t[1,1]=1;t[2,2]=1
    tok=torch.zeros(N,D);tok[:,0]=1;tok[1]=t[1];tok[2]=t[2]
    fg=torch.zeros(N,dtype=torch.bool);fg[:4]=True
    def relative(sim,fg):
        B,K,P=sim.shape
        x=sim[:,None,:,:].expand(B,K,K,P)
        diag=torch.eye(K,dtype=torch.bool).reshape(1,K,K,1)
        mx=x.masked_fill(diag,-torch.inf).max(2).values
        return (sim-mx).masked_fill(~fg[:,None,:],-torch.inf)
    w,c,fb=assignment(t,tok,fg,0.05,relative)
    assert w.shape==(3,N) and int(c.sum())>=3
    assert torch.allclose(w.sum(-1),torch.ones(3))
    assert bool((w[:,~fg]==0).all())
    # K=1 must use all foreground, without fallback.
    w1,c1,fb1=assignment(t[:1],tok,fg,0.05,relative)
    assert c1.sum().item()==4 and not bool(fb1[0])
    gt=torch.full((N,),116,dtype=torch.long);gt[0]=0;gt[1]=1;gt[2]=2
    mass=torch.zeros((3,117)).scatter_add_(1,gt[None,:].expand(3,-1),w)
    assert torch.allclose(mass.sum(-1),torch.ones(3))
    print('SELF_TEST_PASS: relative/absolute weighting; fallback; ignore mass; indexed GT accumulation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--inspect-cache',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--run0',type=Path,default=Path('final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep'))
    p.add_argument('--run4',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--iou-csv',type=Path,default=Path('output/pp116_DIAGREL_comparison/per_part_epoch_lambda0_vs_lambda4.csv'))
    p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--gt-suffix',default='',help='Suffix in indexed GT mask stem after image_id')
    p.add_argument('--flip-mode',choices=('auto','none','all'),default='auto',help='auto trusts recorded horizontal flip key if found. Inspect cache for unsupported transforms!')
    p.add_argument('--epochs',nargs='+',type=int,default=[0,10,30,100])
    p.add_argument('--soft-temperature',type=float,default=0.05)
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_gt_weighted_purity'))
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--max-crops',type=int,default=0)
    p.add_argument('--min-support',type=int,default=10)
    p.add_argument('--focus',nargs='*',default=DEFAULT_FOCUS)
    args=p.parse_args()
    if args.self_test:return self_test()
    if args.inspect_cache:return preflight(args)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
    if args.soft_temperature<=0 or not math.isfinite(args.soft_temperature):raise ValueError('Invalid temperature')
    if not args.epochs or any(ep<0 or ep>100 for ep in args.epochs):raise ValueError('Invalid epochs')
    args.epochs=sorted(set(args.epochs))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):raise FileExistsError(f'Refusing to mix prior output: {args.out_dir}')
    for f in (args.cache,args.text_bank,args.projector,args.model_config,args.classes_file,args.iou_csv):
        if not f.is_file():raise FileNotFoundError(f)
    # Load all necessary models/weights before writing any result.
    import train_relproto_alignemt as src
    import train_pp116_softsingle_dual as train
    names=canonical_names(args.classes_file)
    focus={names.index(x) for x in args.focus if x in names}
    if len(focus)!=len(args.focus):raise ValueError(f'Focus part does not match taxonomy: {[x for x in args.focus if x not in names]}')
    raw,raw_names,_=src.load_raw_clip_bank(args.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if list(raw_names)!=names:raise ValueError('CLIP text bank vs canonical names order mismatch')
    device=torch.device(args.device)
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    project,_=src.load_frozen_projector(project_root=Path.cwd(),config_path=args.model_config,weight_path=args.projector,device=device)
    base=src.normalize_last(src.project_raw_clip_bank(raw,project,device=device,batch_size=128).detach().to(device=device,dtype=torch.float32))
    if base.shape!=(C,D):raise ValueError(f'Wrong projected text dims: {base.shape}')
    del project,raw
    text={}
    for arm,run,lam in [('lam0',args.run0,0),('lam4',args.run4,4)]:
        for ep in args.epochs:
            W=checkpoint_W(run,ep,lam).to(device)
            text[arm,ep]=src.normalize_last(base@W).detach()
    if 0 in args.epochs and not torch.equal(text['lam0',0],text['lam4',0]):raise AssertionError('E0 two arms differ')
    ious=read_iou(args.iou_csv,names)
    for ep in args.epochs:
        for pid in range(C):
            if (ep,pid) not in ious:raise ValueError(f'IoU missing {ep,pid}')
    lookup=GTLookup(args.gt_root,args.gt_suffix)
    anns=annotations(load_pt(args.cache))
    total=len(anns)
    if args.max_crops>0:anns=anns[:args.max_crops]
    if not anns:raise ValueError('Empty cache')
    # Fail-fast on mask access before committing to long audit.
    x=anns[0]
    if not isinstance(x,dict):raise TypeError('Cache records should be dicts')
    preview=lookup.crop32(x,args.flip_mode)
    print('[GT PREFLIGHT] first image',x['image_id'],'GT image_wh',preview[2], 'GT grid unique',np.unique(preview[0]).tolist(), 'flip',preview[1],flush=True)
    print('[DATA] crops',len(anns),'/',total,'arms=2 epochs=',args.epochs,flush=True)
    args.out_dir.mkdir(parents=True)
    s_header=['arm','epoch','class_id','part','object','n_candidate_crops','n_gt_present_crops','n_gt_present_and_fg_crops','n_single_part_crops','n_multi_part_crops','fallback_rate_multi','mean_purity_all_crops','mean_purity_gt_present_crops','mean_purity_gt_present_and_fg','mean_valid_conditioned_purity','mean_ignore_mass','mean_wrong_mass','mean_fg_recall_gt_present','micro_candidate_patch_gt_purity','mean_effective_patch_count','val_gt_valid','val_part_iou_pct']
    conf_header=['arm','epoch','class_id','pseudo_part','gt_id','gt_part','sum_weight','mean_weight_per_crop','n_candidate_crops']
    focus_header=['arm','epoch','crop_index','image_id','class_id','part','K','gt_patch_count','gt_fg_patch_count','fg_recall','fallback','selected_patch_count','weighted_gt_purity','weighted_ignore_mass','weighted_wrong_mass','dominant_wrong_gt_part','dominant_wrong_mass','hflip','crop_box_xyxy']
    agg=defaultdict(lambda:defaultdict(float))
    confusion=defaultdict(lambda:np.zeros(C+1,dtype=np.float64))
    ff, fw=writer(args.out_dir/'per_crop_focus.csv',focus_header)
    checked=False
    flip_crops=0;masked_out=0
    with torch.no_grad():
      for ii,a in enumerate(anns):
        if not isinstance(a,dict):raise TypeError(f'Non-dict crop #{ii}')
        ids=part_ids(a)
        if any(k not in a for k in ('image_id','cropaug_box_xyxy','cropaug_patch_tokens','pred_obj_mask_patch')):raise KeyError(f'Crop #{ii} missing required keys')
        token=torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32,device=device)
        fg=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool,device=device).reshape(-1)
        if token.shape!=(N,D) or fg.shape!=(N,) or not bool(fg.any()):raise ValueError(f'Bad tokens/fg at crop#{ii}: {token.shape}/{fg.shape}')
        token=src.normalize_last(token)
        labels,flipped,shape,box=lookup.crop32(a,args.flip_mode)
        flip_crops+=int(flipped)
        gt=torch.as_tensor(np.where(labels==IGN,C,labels),device=device,dtype=torch.long)
        gtcount=torch.bincount(gt,minlength=C+1)
        masked_out+=int(gtcount[:C].sum()==0)
        ones=torch.ones((len(ids),),device=device)
        for (arm,ep),textbank in text.items():
            local=textbank[ids]
            weights,cand,fb=assignment(local,token,fg,args.soft_temperature,src.relative_scores)
            if not checked and len(ids)>1:
                proto,info=train.soft_relative_prototypes(local[None],token[None],fg[None],temperature=args.soft_temperature)
                own=F.normalize(weights@token,dim=-1,eps=1e-12)
                err=(proto[0]-own).abs().max().item()
                if err>2e-5 or not torch.equal(fb,info['fallback'][0].bool()):
                    raise RuntimeError(f'Pseudo-prototype construction DIFFERENT from actual trainer! err={err}')
                checked=True
                print('[CHECK] Soft Relative weights reproduce original trainer proto and fallback',flush=True)
            mass=torch.zeros((len(ids),C+1),device=device)
            mass.scatter_add_(1,gt[None,:].expand(len(ids),-1),weights)
            if not torch.allclose(mass.sum(-1),ones,atol=2e-5):raise AssertionError('Lost softmax mass')
            mass_np=mass.cpu().numpy()
            gt_counts=gtcount[ids].cpu().numpy()
            gt_fg=[int(((gt==p)&fg).sum().item()) for p in ids]
            correct_count=[int((cand[j]&(gt==p)).sum().item()) for j,p in enumerate(ids)]
            candcounts=cand.sum(-1).cpu().numpy()
            eff=(1/weights.square().sum(-1)).cpu().numpy()
            for j,pid in enumerate(ids):
                key=(arm,ep,pid)
                part=agg[key]
                good=float(mass_np[j,pid]);ignore=float(mass_np[j,C]);bad=max(0.,1.-good-ignore)
                ngt=int(gt_counts[j]);nfg=int(gt_fg[j]);eligible=int(ngt>0);fg_el=int(nfg>0)
                part['n']+=1;part['gt_present']+=eligible;part['gt_fg_present']+=fg_el
                part['single']+=int(len(ids)==1);part['multi']+=int(len(ids)>1)
                part['fallback']+=int(fb[j]) if len(ids)>1 else 0
                part['good']+=good;part['good_present']+=good*eligible;part['good_fg_present']+=good*fg_el
                part['valid_purity_sum']+=good/(1-ignore) if ignore<1-1e-9 else 0
                part['valid_purity_n']+=int(ignore<1-1e-9)
                part['ignore']+=ignore;part['wrong']+=bad
                part['gt_fg_num']+=nfg;part['gt_den']+=ngt
                part['candidate_correct']+=correct_count[j];part['candidate_count']+=int(candcounts[j]);part['effective']+=float(eff[j])
                confusion[key]+=mass_np[j,:]
                if pid in focus:
                    alt=mass_np[j,:C].copy();alt[pid]=-1
                    winner=int(np.argmax(alt));maxwrong=float(alt[winner])
                    fw.writerow({'arm':arm,'epoch':ep,'crop_index':ii,'image_id':str(a['image_id']),'class_id':pid,'part':names[pid],
                       'K':len(ids),'gt_patch_count':ngt,'gt_fg_patch_count':nfg,'fg_recall':nfg/ngt if ngt else math.nan,
                       'fallback':int(fb[j]) if len(ids)>1 else 0,'selected_patch_count':int(candcounts[j]),
                       'weighted_gt_purity':good,'weighted_ignore_mass':ignore,'weighted_wrong_mass':bad,
                       'dominant_wrong_gt_part':names[winner] if maxwrong>1e-7 else '', 'dominant_wrong_mass':max(0,maxwrong),
                       'hflip':int(flipped),'crop_box_xyxy':json.dumps(box)})
        if (ii+1)%300==0:print(f'[PROGRESS] {ii+1}/{len(anns)} crops',flush=True)
    ff.close()
    if not checked:raise ValueError('No multi-part crop available to validate source Soft Relative')
    summary=[]
    for arm,ep in text:
        for pid in range(C):
            v=agg[arm,ep,pid];n=int(v['n'])
            if n==0:continue
            valid,iou=ious[ep,pid]
            summary.append({'arm':arm,'epoch':ep,'class_id':pid,'part':names[pid],'object':names[pid].split("'s ",1)[0],
               'n_candidate_crops':n,'n_gt_present_crops':int(v['gt_present']),'n_gt_present_and_fg_crops':int(v['gt_fg_present']),
               'n_single_part_crops':int(v['single']),'n_multi_part_crops':int(v['multi']),
               'fallback_rate_multi':v['fallback']/v['multi'] if v['multi'] else math.nan,
               'mean_purity_all_crops':v['good']/n,
               'mean_purity_gt_present_crops':v['good_present']/v['gt_present'] if v['gt_present'] else math.nan,
               'mean_purity_gt_present_and_fg':v['good_fg_present']/v['gt_fg_present'] if v['gt_fg_present'] else math.nan,
               'mean_valid_conditioned_purity':v['valid_purity_sum']/v['valid_purity_n'] if v['valid_purity_n'] else math.nan,
               'mean_ignore_mass':v['ignore']/n,'mean_wrong_mass':v['wrong']/n,
               'mean_fg_recall_gt_present':v['gt_fg_num']/v['gt_den'] if v['gt_den'] else math.nan,
               'micro_candidate_patch_gt_purity':v['candidate_correct']/v['candidate_count'] if v['candidate_count'] else math.nan,
               'mean_effective_patch_count':v['effective']/n,
               'val_gt_valid':int(valid),'val_part_iou_pct':iou[arm] if valid else math.nan})
    sf,sw=writer(args.out_dir/'per_part_epoch.csv',s_header)
    sw.writerows(summary);sf.close()
    cf,cw=writer(args.out_dir/'soft_assignment_gt_confusion.csv',conf_header)
    for (arm,ep,pid),vals in sorted(confusion.items()):
        n=int(agg[arm,ep,pid]['n'])
        for gt_pid,mass in enumerate(vals):
            if mass<=1e-9:continue
            cw.writerow({'arm':arm,'epoch':ep,'class_id':pid,'pseudo_part':names[pid],
              'gt_id':gt_pid if gt_pid<C else IGN,'gt_part':names[gt_pid] if gt_pid<C else 'IGNORE_255',
              'sum_weight':float(mass),'mean_weight_per_crop':float(mass/n),'n_candidate_crops':n})
    cf.close()
    correlations=[]
    for arm,ep in text:
        rows=[r for r in summary if r['arm']==arm and r['epoch']==ep and r['val_gt_valid']
              and r['n_gt_present_crops']>=args.min_support]
        for k in ['mean_purity_all_crops','mean_purity_gt_present_crops','mean_purity_gt_present_and_fg','micro_candidate_patch_gt_purity','fallback_rate_multi']:
            rho,num=spearman_pair([r[k] for r in rows],[r['val_part_iou_pct'] for r in rows])
            correlations.append({'arm':arm,'epoch':ep,'metric':k,'n_valid_parts':num,'spearman_vs_part_IoU':rho})
    csf,csw=writer(args.out_dir/'purity_vs_IoU_spearman.csv',['arm','epoch','metric','n_valid_parts','spearman_vs_part_IoU']);csw.writerows(correlations);csf.close()
    paired=[]
    index={(r['arm'],r['epoch'],r['class_id']):r for r in summary}
    for ep in args.epochs:
        pairs=[]
        for pid in range(C):
            a=index.get(('lam0',ep,pid));b=index.get(('lam4',ep,pid))
            if a is None or b is None or not (a['val_gt_valid'] and a['n_gt_present_crops']>=args.min_support):continue
            pairs.append((a,b))
        for key in ['mean_purity_all_crops','mean_purity_gt_present_crops','mean_purity_gt_present_and_fg']:
            x=[b[key]-a[key] for a,b in pairs];y=[b['val_part_iou_pct']-a['val_part_iou_pct'] for a,b in pairs]
            rho,num=spearman_pair(x,y)
            paired.append({'epoch':ep,'metric':key,'n_valid_parts':num,'spearman_delta_purity_vs_delta_IoU':rho})
    pf,pw=writer(args.out_dir/'paired_lambda_spearman.csv',['epoch','metric','n_valid_parts','spearman_delta_purity_vs_delta_IoU']);pw.writerows(paired);pf.close()
    protocol={'purpose':'GT-used-only-after-Soft-Relative-assignment; read-only diagnosis, NOT WSS training',
      'cache':str(args.cache),'crops_used':len(anns),'crops_total':total,'arms':['lam0','lam4'],
      'epochs':args.epochs,'gt_root':str(args.gt_root),'gt_suffix':args.gt_suffix,
      'geometry':'crop original pixel XYXY; horizontal flip only if recorded or explicit; PIL NEAREST to 32x32; verify original extraction pipeline before interpretation',
      'flip_mode':args.flip_mode,'flipped_crops':flip_crops,'no_valid_gt_patch_crops':masked_out,
      'soft_temperature':args.soft_temperature,'ignore_id':IGN,'fallback':'K>=2 only; K1 uses absolute softmax',
      'text':'Raw CLIP ViT-B/16 -> frozen COCO Projector -> normalized T0 W',
      'mean_purity_all_crops':'mean sum of pseudo-part softmax mass on GT=part; includes candidate crops without part spatial GT',
      'mean_purity_gt_present_crops':'conditional on target part GT patch existing in cropped GT',
      'mean_purity_gt_present_and_fg':'conditional on at least one GT target part patch inside predicted foreground',
      'sampling_note':'nearest 32x32 indexed GT is a diagnostic approximation to patch footprint; small/boundary parts sensitive; should visually check alignment',
      'no_overwrite':True}
    (args.out_dir/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE]',args.out_dir,'sample rows=',len(summary),'flipped=',flip_crops,'no_gt=',masked_out)
    print('[E100 focus, mean conditional GT-present purity and IoU]')
    for arm in ('lam0','lam4'):
        for r in summary:
            if r['arm']==arm and r['epoch']==100 and r['class_id'] in focus:
                print(f'{arm:4s} {r["part"]:26s} support={r["n_gt_present_crops"]:4d} '
                      f'purity={r["mean_purity_gt_present_crops"]:.4f} IoU={r["val_part_iou_pct"]:.2f}')

if __name__=='__main__':main()
