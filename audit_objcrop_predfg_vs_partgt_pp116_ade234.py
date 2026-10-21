#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit cached predicted-object foreground versus the corresponding parent-object GT Part union.

READ-ONLY; uses GT annotations solely for offline analysis. Grid is the cached 32x32
(1024 patches), not full-resolution pixels. No model inference or training.

Target parent object is derived from cached image-level part_category_id + fixed
canonical taxonomy, NOT from GT. GT part union uses *all* labeled Part classes
of that object, not only candidate ids. Each part's coverage is also reported.

Geometry: cropaug_box_xyxy, optionally RECORDED horizontal flip, nearest 32x32.
Absent flip metadata is NOT proof of no flip. Never choose a flip by maximizing IoU.
"""
from __future__ import annotations
import argparse, ast, csv, json, math, re, sys
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

GRID=32
N=GRID*GRID


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('pp116','ade234'),required=False)
    p.add_argument('--cache',type=Path)
    p.add_argument('--gt-root',type=Path)
    p.add_argument('--image-root',type=Path,default=None,help='Optional: verify GT mask vs original RGB dimensions')
    p.add_argument('--taxonomy',type=Path,default=None,help='Optional canonical PART_CLASSES Python source')
    p.add_argument('--part-id-offset',type=int,choices=(0,1),default=0)
    p.add_argument('--flip-mode',choices=('auto','none','all'),default='auto')
    p.add_argument('--pred-threshold',type=float,default=None,help='Only for nonbinary cached masks; prefer original bool mask')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=0,help='0 processes the entire cache')
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--self-test',action='store_true')
    return p.parse_args()


def load_pt(path):
    # Use only the locally generated cache you trust; torch.load may unpickle Python objects.
    return torch.load(str(path),map_location='cpu',weights_only=False)


def records(payload):
    if isinstance(payload,(list,tuple)):return payload
    if isinstance(payload,dict):
        for k in ('annotations','records','samples','items','data'):
            if isinstance(payload.get(k),(list,tuple)):return payload[k]
    raise ValueError('Unsupported cached dataset: expected list or dict with annotation records')


def flatten(x):
    if torch.is_tensor(x):return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x,np.ndarray):return x.reshape(-1).tolist()
    if isinstance(x,(tuple,list)):
        ans=[]
        for el in x:
            ans.extend(flatten(el))
        return ans
    return [x]


def names_from_file(path,nclasses):
    if not path.is_file():raise FileNotFoundError(f'Canonical taxonomy file not found: {path}')
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    names=None
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PART_CLASSES' for t in node.targets):
            try:names=ast.literal_eval(node.value)
            except (ValueError,TypeError,KeyError):pass
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='PART_CLASSES':
            try:names=ast.literal_eval(node.value)
            except (ValueError,TypeError,KeyError):pass
    if path.name=='ade20kpart234_taxonomy.py' and (
        names is None or len(names)!=nclasses
    ):
        # ADE taxonomy may build PART_CLASSES dynamically.
        # Execute the existing local taxonomy file and read its final value.
        import runpy
        namespace=runpy.run_path(str(path))
        names=namespace.get('PART_CLASSES')
        if names is None:
            raise ValueError(
                f'{path}: PART_CLASSES is missing after execution'
            )
        print(f'[TAXONOMY] Loaded {len(names)} ADE Part classes dynamically')
    if names is None:raise ValueError(f'Unable to load PART_CLASSES from {path}')
    names=[str(s) for s in names]
    if len(names)!=nclasses or len(set(names))!=nclasses:
        raise ValueError(f'{path}: expected {nclasses} unique categories, got {len(names)}')
    for n in names:
        if "'s " not in n and '’s ' not in n:
            raise ValueError(f'Unexpected taxonomy name {n!r}; cannot infer parent object')
    return names


def split_parent(n):
    return re.split(r"['’]s\s+",n,maxsplit=1)[0]


def candidate(a,names):
    ids=[int(q) for q in flatten(a['part_category_id'])]
    if not ids or len(ids)!=len(set(ids)) or any(k<0 or k>=len(names) for k in ids):
        raise ValueError(f'Invalid candidate part ids {ids}')
    parents={split_parent(names[i]) for i in ids}
    if len(parents)!=1:raise ValueError(f'Crop has multiple target parent objects: {parents}; refuse to guess')
    return ids,next(iter(parents))


def pred32(a,threshold):
    if 'pred_obj_mask_patch' not in a:raise KeyError('Missing pred_obj_mask_patch')
    x=np.asarray(torch.as_tensor(a['pred_obj_mask_patch']).cpu())
    if x.size!=N:raise ValueError(f'Pred mask expected {N} values, got {x.shape}')
    if x.dtype==np.bool_:return x.reshape(GRID,GRID).copy()
    if not np.isfinite(x).all():raise ValueError('NaN or inf in predicted foreground mask')
    uni=np.unique(x)
    if np.isin(uni,[0,1]).all():return x.reshape(GRID,GRID).astype(bool)
    if threshold is None:raise ValueError(f'Nonbinary predicted foreground values {uni[:10]}; set --pred-threshold explicitly if verified')
    return (x.reshape(GRID,GRID)>threshold)


def flip_info(a,mode):
    if mode=='none':return False,False
    if mode=='all':return True,False
    for k in ('cropaug_hflip','cropaug_flipped','cropaug_flip','hflip','is_flipped','flipped','flip'):
        if k not in a:continue
        v=a[k]
        if torch.is_tensor(v):
            if v.numel()!=1:raise ValueError(f'Unsupported flip metadata {k}={v!r}')
            v=v.item()
        if isinstance(v,(int,np.integer,bool)):return bool(v),True
        if isinstance(v,str):
            v=v.lower().strip()
            if v in ('horizontal','h','hflip','true','1','yes'):return True,True
            if v in ('none','false','0','no',''):return False,True
        raise ValueError(f'Unsupported flip metadata {k}={v!r}')
    v=a.get('flip_direction')
    if v not in (None,'',False):
        if v=='horizontal':return True,True
        raise ValueError(f'Unsupported flip direction {v!r}')
    return False,False


def cropped_gt32(gt,a,flip_mode):
    h,w=gt.shape
    if 'cropaug_box_xyxy' not in a:raise KeyError('Missing cropaug_box_xyxy')
    box=np.array(flatten(a['cropaug_box_xyxy']),dtype=np.float64)
    if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError(f'Bad crop box {box}')
    if np.all((0<=box)&(box<=1.01)) and max(h,w)>8:
        raise ValueError(f'Crop box looks normalized rather than original-pixel XYXY: {box}')
    x0,y0,x1,y1=[int(round(t)) for t in box]
    x0=max(0,min(w,x0));x1=max(0,min(w,x1));y0=max(0,min(h,y0));y1=max(0,min(h,y1))
    if x0>=x1 or y0>=y1:raise ValueError(f'Invalid crop coordinates {box} for GT shape {gt.shape}')
    out=gt[y0:y1,x0:x1]
    flip,recorded=flip_info(a,flip_mode)
    if flip:out=np.ascontiguousarray(out[:,::-1])
    # Preserve integer 16-bit mask class IDs exactly (never bilinear interpolation).
    resized=np.asarray(Image.fromarray(out.astype(np.int32),mode='I').resize((GRID,GRID),resample=Image.Resampling.NEAREST),dtype=np.int32)
    return resized,flip,recorded,(x0,y0,x1,y1)


def locate(root,image_id):
    stem=Path(str(image_id)).stem
    for ext in ('.png','.tif','.tiff'):
        p=root/(stem+ext)
        if p.is_file():return p
    # ADE may organize images/annotations by subfolders; build indexed lookup lazily.
    if not hasattr(locate,'index') or locate.index[0]!=str(root.resolve()):
        idx=defaultdict(list)
        for p in root.rglob('*'):
            if p.suffix.lower() in ('.png','.tif','.tiff'):
                idx[p.stem].append(p)
        locate.index=(str(root.resolve()),idx)
    match=locate.index[1].get(stem,[])
    if len(match)!=1:raise FileNotFoundError(f'Expected 1 GT mask for {stem} in {root}; got {match[:5]}')
    return match[0]


def rgb_check(root,image_id,wh):
    stem=Path(str(image_id)).stem
    choices=[root/(stem+e) for e in ('.jpg','.jpeg','.png','.JPG','.JPEG','.PNG')]
    found=[p for p in choices if p.is_file()]
    if len(found)!=1:raise FileNotFoundError(f'Expected 1 original image for {stem} in {root}, got {found}')
    with Image.open(found[0]) as im:
        if im.size!=wh:raise ValueError(f'Image dimensions {im.size} != GT mask dimensions {wh} at {stem}')


class GTLoader:
    def __init__(self,root,nclasses,offset,images):
        self.root=root;self.nclasses=nclasses;self.offset=offset;self.images=images
        self.checked=set()
    @lru_cache(maxsize=128)
    def load(self,image_id):
        p=locate(self.root,image_id)
        with Image.open(p) as im:
            if im.mode in ('RGB','RGBA'):
                raise ValueError(f'Indexed part mask required, found RGB at {p}')
            a=np.asarray(im)
        if a.ndim!=2 or not np.issubdtype(a.dtype,np.integer):
            raise ValueError(f'GT mask is not a single-channel integer class mask: {p} {a.shape} {a.dtype}')
        a=a.astype(np.int32)
        bad=~((a>=self.offset)&(a<self.offset+self.nclasses)) & (a!=255) & (a!=65535)
        if bool(bad.any()):raise ValueError(f'Invalid GT labels {np.unique(a[bad])[:12]} at {p}; check --part-id-offset')
        if self.images is not None:rgb_check(self.images,image_id,(a.shape[1],a.shape[0]))
        return a


def safe_ratio(num,den):return float(num/den) if den else math.nan


def measures(gt,pred,object_labels,offset,names,parent):
    annotated=(gt>=offset)&(gt<offset+len(names))
    objectgt=np.isin(gt,np.asarray(object_labels,dtype=np.int32))
    p=pred.astype(bool)
    tp=int((p&objectgt).sum());pg=int(p.sum());gg=int(objectgt.sum())
    pv=int((p&annotated).sum());ag=int(annotated.sum())
    others=int((p&annotated&~objectgt).sum());ign=int((p&~annotated).sum())
    u=pg+gg-tp
    validunion=int(((p&annotated)|objectgt).sum())
    return {
        'gt_part_union_patches':gg,
        'pred_fg_patches':pg,
        'overlap_patches':tp,
        'gt_valid_all_parts_patches':ag,
        'pred_fg_gt_valid_patches':pv,
        'pred_fg_other_obj_part_patches':others,
        'pred_fg_unannotated_patches':ign,
        'gt_parent_part_coverage':safe_ratio(tp,gg),
        'pred_fg_parent_part_fraction_all':safe_ratio(tp,pg),
        'pred_fg_parent_part_fraction_gt_valid':safe_ratio(tp,pv),
        'pred_fg_annotated_fraction':safe_ratio(pv,pg),
        'pred_gt_part_union_iou':safe_ratio(tp,u),
        'pred_gt_part_union_iou_gt_valid':safe_ratio(tp,validunion),
        'all_GT_valid_part_coverage':safe_ratio(pv,ag),
    }


class Acc:
    def __init__(self):
        self.n=0;self.cnt=Counter();self.sums=defaultdict(float);self.macro=defaultdict(float);self.macro_n=Counter()
    def update(self,r):
        self.n+=1
        for k,v in r.items():
            if k.endswith('_patches') and isinstance(v,(int,np.integer)):
                self.cnt[k]+=int(v)
        for k in ('gt_parent_part_coverage','pred_fg_parent_part_fraction_all','pred_fg_parent_part_fraction_gt_valid','pred_fg_annotated_fraction','pred_gt_part_union_iou','pred_gt_part_union_iou_gt_valid','all_GT_valid_part_coverage'):
            v=r.get(k)
            if v is not None and math.isfinite(float(v)):
                self.macro[k]+=float(v);self.macro_n[k]+=1
        if r['gt_part_union_patches']==0:self.cnt['empty_gt_crops']+=1
        if r['pred_fg_patches']==0:self.cnt['empty_pred_crops']+=1
    def summary(self,kind,name):
        d=dict(group_type=kind,name=name,n_crops=self.n,**dict(self.cnt))
        c=self.cnt
        variants={
            'gt_parent_part_coverage':('overlap_patches','gt_part_union_patches'),
            'pred_fg_parent_part_fraction_all':('overlap_patches','pred_fg_patches'),
            'pred_fg_parent_part_fraction_gt_valid':('overlap_patches','pred_fg_gt_valid_patches'),
            'pred_fg_annotated_fraction':('pred_fg_gt_valid_patches','pred_fg_patches'),
            'all_GT_valid_part_coverage':('pred_fg_gt_valid_patches','gt_valid_all_parts_patches'),
        }
        for k,(num,den) in variants.items():
            d[k+'_micro']=safe_ratio(c[num],c[den]);d[k+'_macro']=safe_ratio(self.macro[k],self.macro_n[k])
        u=c['pred_fg_patches']+c['gt_part_union_patches']-c['overlap_patches']
        d['pred_gt_part_union_iou_micro']=safe_ratio(c['overlap_patches'],u)
        v=c['pred_fg_gt_valid_patches']+c['gt_part_union_patches']-c['overlap_patches']
        d['pred_gt_part_union_iou_gt_valid_micro']=safe_ratio(c['overlap_patches'],v)
        for k in ('pred_gt_part_union_iou','pred_gt_part_union_iou_gt_valid'):
            d[k+'_macro']=safe_ratio(self.macro[k],self.macro_n[k])
        return d


def save_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def test():
    names=["cat's nose","cat's ear","dog's nose"]
    gt=np.array([[0,1,255,2],[0,1,255,2],[0,1,255,2],[0,1,255,2]],dtype=np.int32)
    p=np.array([[1,1,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]],dtype=bool)
    m=measures(gt,p,[0,1],0,names,'cat')
    assert m['gt_part_union_patches']==8 and m['overlap_patches']==6 and m['pred_fg_patches']==6,m
    assert m['gt_parent_part_coverage']==.75 and m['pred_fg_parent_part_fraction_gt_valid']==1
    a=Acc();a.update(m); s=a.summary('dataset','test')
    assert s['gt_parent_part_coverage_micro']==.75 and s['pred_fg_parent_part_fraction_gt_valid_micro']==1
    # Check 1-based taxonomy offset support.
    assert measures(gt+1,p,[1,2],1,names,'cat')['gt_part_union_patches']==8
    a={'cropaug_box_xyxy':[0,0,32,32], 'cropaug_hflip':True}
    out,flipped,recorded,_=cropped_gt32(np.tile(np.arange(32,dtype=np.int32),(32,1)),a,'auto')
    assert flipped and recorded and out[0,0]==31 and out[0,-1]==0
    print('SELF_TEST_PASS: parent object coverage; precision valid; offset; 32x32 flip; aggregation')


def main():
    a=cli()
    if a.self_test:return test()
    if a.dataset is None:raise ValueError('Pass --dataset pp116 or ade234')
    if a.cache is None or a.gt_root is None or a.out_dir is None:
        raise ValueError('Specify --cache, --gt-root and --out-dir')
    if not a.cache.is_file():raise FileNotFoundError(f'No cache: {a.cache}')
    if not a.gt_root.is_dir():raise FileNotFoundError(f'No GT dir: {a.gt_root}')
    if a.image_root is not None and not a.image_root.is_dir():raise FileNotFoundError(f'No image dir: {a.image_root}')
    c=116 if a.dataset=='pp116' else 234
    taxonomy=a.taxonomy or (Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py') if a.dataset=='pp116' else Path('ade20kpart234_taxonomy.py'))
    names=names_from_file(taxonomy,c)
    by_parent=defaultdict(list)
    for pid,name in enumerate(names):by_parent[split_parent(name)].append(pid)
    loader=GTLoader(a.gt_root,c,a.part_id_offset,a.image_root)
    payload=load_pt(a.cache)
    data=records(payload)
    if a.start<0 or a.start>=len(data):raise ValueError(f'Bad start={a.start}, cache={len(data)}')
    end=min(len(data),a.start+a.limit) if a.limit>0 else len(data)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise FileExistsError(f'Output directory not empty: {a.out_dir}')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    crops=[];per_part_raw=defaultdict(lambda:dict(n_crops=0,n_gt_present_crops=0,gt_part_patches=0,gt_part_in_pred_fg=0,macro_sum=0.0))
    allacc=Acc();byobj=defaultdict(Acc)
    unknown_flip=0;flipped_count=0
    for idx in range(a.start,end):
        r=data[idx]
        if not isinstance(r,dict):raise ValueError(f'Cache row {idx} not dict')
        if 'image_id' not in r:raise KeyError(f'Cache row {idx}: missing image_id')
        image=Path(str(r['image_id'])).stem
        ids,parent=candidate(r,names)
        gt_full=loader.load(image)
        gt,flipped,has_recorded,box=cropped_gt32(gt_full,r,a.flip_mode)
        pred=pred32(r,a.pred_threshold)
        if not has_recorded:unknown_flip+=1
        if flipped:flipped_count+=1
        object_labels=[pid+a.part_id_offset for pid in by_parent[parent]]
        stats=measures(gt,pred,object_labels,a.part_id_offset,names,parent)
        row={'row_index':idx,'image_id':image,'object':parent,'crop_box_xyxy':json.dumps(box),
             'flip':int(flipped),'flip_metadata_recorded':int(has_recorded),'n_candidate_part_ids':len(ids),**stats}
        crops.append(row);allacc.update(stats);byobj[parent].update(stats)
        for pid in by_parent[parent]:
            target=(gt==(pid+a.part_id_offset))
            den=int(target.sum());num=int((target&pred).sum())
            h=per_part_raw[(parent,pid)]
            h['n_crops']+=1
            h['gt_part_patches']+=den
            h['gt_part_in_pred_fg']+=num
            if den:
                h['n_gt_present_crops']+=1
                h['macro_sum']+=num/den
        if (idx+1)%1000==0 or idx==a.start:
            print(f'[PROGRESS] {idx+1}/{end} obj={parent} GT={stats["gt_part_union_patches"]} '
                  f'pred={stats["pred_fg_patches"]} intersection={stats["overlap_patches"]} '
                  f'GTcov={stats["gt_parent_part_coverage"]:.3f}',flush=True)
    save_csv(a.out_dir/'per_crop.csv',crops)
    save_csv(a.out_dir/'per_object.csv',[byobj[name].summary('object',name) for name in sorted(byobj)])
    partrows=[]
    for (obj,pid),h in sorted(per_part_raw.items(),key=lambda t:t[0][1]):
        partrows.append({'object':obj,'part_id':pid,'part':names[pid],
                         'n_crops':h['n_crops'],'n_gt_present_crops':h['n_gt_present_crops'],
                         'gt_part_patches':h['gt_part_patches'],'gt_part_in_pred_fg':h['gt_part_in_pred_fg'],
                         'gt_part_coverage_micro':safe_ratio(h['gt_part_in_pred_fg'],h['gt_part_patches']),
                         'gt_part_coverage_macro':safe_ratio(h['macro_sum'],h['n_gt_present_crops'])})
    save_csv(a.out_dir/'per_part.csv',partrows)
    s=allacc.summary('dataset',a.dataset)
    s['total_cache_records']=len(data);s['audited_records']=end-a.start
    s['unknown_flip_metadata_records']=unknown_flip;s['recorded_flipped_records']=flipped_count
    s['note']='Audits nearest 32x32 crop-aligned GT Part labels; not full-resolution pixel coverage'
    (a.out_dir/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False if all(math.isfinite(v) for v in s.values() if isinstance(v,float)) else True),encoding='utf-8')
    protocol={'dataset':a.dataset,'cache':str(a.cache.resolve()),'gt_root':str(a.gt_root.resolve()),
              'taxonomy':str(taxonomy.resolve()),'part_id_offset':a.part_id_offset,
              'grid':[32,32],'pred_field':'pred_obj_mask_patch','GT_semantics':'all annotated GT Parts from the cached crop parent object; not an independent GT instance mask',
              'primary_GT_coverage':'|PredFG intersect GT_parent_PartUnion| / |GT_parent_PartUnion|',
              'pred_valid_parent_fraction':'|PredFG intersect GT_parent_PartUnion| / |PredFG intersect ANY annotated GT Part|',
              'pred_all_parent_fraction':'|PredFG intersect GT_parent_PartUnion| / |PredFG|',
              'gt_ignored_labels':[255,65535],'flip_mode':a.flip_mode,
              'unrecorded_flips':'Unknown flip metadata is assumed no flip for this reconstruction. Inference from GT not performed.',
              'output_dir':str(a.out_dir.resolve()),'offline_GT_only':'GT accessed only for auditing; not model training'}
    (a.out_dir/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n[RESULT]',a.dataset,'n_crops=',s['n_crops'])
    for key in ('gt_parent_part_coverage_micro','gt_parent_part_coverage_macro',
                'pred_fg_parent_part_fraction_gt_valid_micro','pred_fg_parent_part_fraction_all_micro',
                'pred_fg_annotated_fraction_micro','pred_gt_part_union_iou_micro'):
        print(f'{key:52s} {100*s[key]:8.3f}%' if math.isfinite(s[key]) else f'{key:52s} NaN')
    print('[GEOMETRY] missing explicit flip metadata:',unknown_flip,'/',end-a.start)
    print('[OUTPUT]',a.out_dir,flush=True)

if __name__=='__main__':main()
