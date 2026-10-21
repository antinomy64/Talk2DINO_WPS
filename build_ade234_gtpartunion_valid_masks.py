#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADE234 PRIVILEGED Oracle: union all labeled GT PARTS of the crop's parent object.

NOT an independently annotated GT object foreground. ALL spatial GT part mask
labels are merged to form ONE boolean valid-patch mask, and the model never sees
which part each valid patch belongs to. Using GT part pixels in training makes
this a PRIVILEGED control, not a weakly supervised method and not a matched
replication of PP116's independent GT-object-mask Oracle.

Keeps original predicted-object crop geometry, DINO tokens, image-level
candidate part list, all loss functions and optimizer unchanged. Nearest32 GT.
"""
from __future__ import annotations
import argparse, csv, json, hashlib, math, tempfile
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
import torch
from PIL import Image

N=1024

def load(p):
    return torch.load(str(p),map_location='cpu',weights_only=False)

def rows(payload):
    if isinstance(payload,(list,tuple)):return payload
    if isinstance(payload,dict):
        for k in ('annotations','records','samples','items'):
            if isinstance(payload.get(k),(list,tuple)):return payload[k]
    raise ValueError('Unexpected ADE crop cache container')

def flat(x):
    if torch.is_tensor(x):return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x,np.ndarray):return x.reshape(-1).tolist()
    if isinstance(x,(list,tuple)):return list(x)
    return [x]

def names234():
    from ade20kpart234_taxonomy import PART_CLASSES
    names=list(PART_CLASSES)
    if len(names)!=234 or len(set(names))!=234:
        raise RuntimeError(f'ADE taxonomy must contain 234 unique classes, got {len(names)}')
    if any("'s " not in x for x in names):
        raise ValueError('ADE234 taxonomy does not use expected "object\'s part" names')
    return names

def candidate(r,names):
    ids=[int(z) for z in flat(r['part_category_id'])]
    if not ids or len(ids)!=len(set(ids)) or any(not 0<=z<234 for z in ids):
        raise ValueError(f'Invalid candidate part indices: {ids}')
    objs={names[z].split("'s ",1)[0] for z in ids}
    if len(objs)!=1:raise RuntimeError(f'Non-single-object crop: {objs}')
    return ids,next(iter(objs))

def flip(r):
    for k in ('cropaug_hflip','cropaug_flipped','cropaug_flip','hflip','is_flipped','flipped','flip'):
        if k in r:
            v=r[k]
            if torch.is_tensor(v):v=v.item()
            if isinstance(v,(bool,int,np.integer)):return bool(v)
            if isinstance(v,str):
                if v.lower() in ('horizontal','h','hflip','true','yes','1'):return True
                if v.lower() in ('none','false','no','0',''):return False
            raise ValueError(f'Unrecognized flip value {k}={v!r}')
    if r.get('flip_direction') not in (None,'',False):
        if r['flip_direction']=='horizontal':return True
        raise ValueError(f'Unsupported flip direction {r["flip_direction"]}')
    return False

@lru_cache(maxsize=128)
def read_png(path):
    with Image.open(path) as im:
        a=np.asarray(im)
        mode=im.mode
    if a.ndim!=2 or mode in ('RGB','RGBA'):
        raise ValueError(f'Expected indexed/single-channel OBJECT mask {path}; got {mode} {a.shape}')
    if not np.issubdtype(a.dtype,np.integer):
        raise ValueError(f'Mask must contain integer IDs, got {a.dtype} in {path}')
    return a.astype(np.int32,copy=False)

def mask_path(root,stem,obj,mode):
    if mode=='part_union':
        p=root/(stem+'.png')
    elif mode=='indexed':
        p=root/(stem+'.png')
    else:
        p=root/obj/(stem+'.png')
    if not p.is_file():raise FileNotFoundError(f'Missing real GT object mask: {p}')
    return p

def crop_fg(r,gt,obj,target,mode,positive_value,images_root=None):
    h,w=gt.shape
    stem=Path(str(r['image_id'])).stem
    if images_root is not None:
        finds=[images_root/(stem+ex) for ex in ('.jpg','.jpeg','.png','.JPG')]
        imgs=[p for p in finds if p.exists()]
        if len(imgs)!=1:raise FileNotFoundError(f'Expected exactly one image for {stem}: {imgs}')
        with Image.open(imgs[0]) as im:
            if im.size!=(w,h):raise ValueError(f'GT mask {gt.shape} vs RGB image {im.size} size mismatch {stem}')
    box=np.asarray(flat(r['cropaug_box_xyxy']),dtype=np.float64)
    if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError(f'Bad crop box {stem}: {box}')
    if box.min()>=0 and box.max()<=1.01 and max(h,w)>8:
        raise ValueError('Crop coordinates look normalized, expected absolute pixels')
    x0,y0,x1,y1=[int(round(float(z))) for z in box]
    x0,x1=max(0,min(w,x0)),max(0,min(w,x1))
    y0,y1=max(0,min(h,y0)),max(0,min(h,y1))
    if x1<=x0 or y1<=y0:raise ValueError(f'Invalid crop box {stem}: {box}')
    isflip=flip(r)
    cut=gt[y0:y1,x0:x1]
    if isflip:cut=np.ascontiguousarray(cut[:,::-1])
    resized=np.asarray(Image.fromarray(cut.astype(np.int32),mode='I').resize((32,32),Image.Resampling.NEAREST),dtype=np.int32)
    if mode=='part_union':valid=np.isin(resized,np.asarray(target,dtype=np.int32))
    elif mode=='indexed':valid=resized==target
    else:valid=resized==positive_value
    return np.ascontiguousarray(valid.reshape(N)),dict(image_id=str(r['image_id']),object=obj,object_id=target,
        crop_box_xyxy=[x0,y0,x1,y1],flip=int(isflip),gt_valid_patches=int(valid.sum()),
        gt_unique_values=np.unique(gt).tolist()[:50],crop_wh=[x1-x0,y1-y0])

def self_test():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        gt=np.full((40,40),255,dtype=np.uint8);gt[4:30,4:25]=13
        r={'image_id':'x.jpg','cropaug_box_xyxy':[0,0,40,40],'part_category_id':[0]}
        fg,info=crop_fg(r,gt,'example',13,'indexed',1)
        assert fg.shape==(1024,) and fg.sum()>0 and fg.sum()<1024
        r['cropaug_hflip']=1
        flipfg,_=crop_fg(r,gt,'example',13,'indexed',1)
        assert np.array_equal(fg.reshape(32,32)[:,::-1],flipfg.reshape(32,32))
        binary=np.zeros((40,40),dtype=np.uint8);binary[4:30,4:25]=1
        r.pop('cropaug_hflip')
        bfg,_=crop_fg(r,binary,'example',None,'binary',1)
        assert np.array_equal(fg,bfg)
        parts=np.full((40,40),255,dtype=np.int32)
        parts[4:30,4:15]=5
        parts[4:30,15:25]=7
        uparts,_=crop_fg(r,parts,'example',[5,7],'part_union',1)
        assert np.array_equal(uparts,fg)
        assert not np.array_equal(crop_fg(r,parts,'example',[5],'part_union',1)[0],fg)
    print('SELF_TEST_PASS_ADE_GT_PART_UNION_MASK')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug.pth'))
    p.add_argument('--gt-part-root',type=Path,required=False)
    p.add_argument('--mask-mode',choices=['part_union'],default='part_union')
    p.add_argument('--part-id-offset',type=int,choices=[0,1],default=0,help='GT pixel label = ADE234 part index + offset; verify against mask conventions')
    p.add_argument('--images-root',type=Path,default=None)
    p.add_argument('--preflight',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('output/ade234_GTPartUnionFG_oracle/gt_partunion_valid_masks.pt'))
    p.add_argument('--empty-policy',choices=['drop','error'],default='error')
    args=p.parse_args()
    if args.self_test:return self_test()
    if args.gt_part_root is None:
        p.error('Specify --gt-part-root explicitly')
    if not args.cache.is_file():raise FileNotFoundError(args.cache)
    if not args.gt_part_root.is_dir():raise FileNotFoundError(args.gt_part_root)
    if args.images_root is not None and not args.images_root.is_dir():raise FileNotFoundError(args.images_root)
    names=names234()
    part_ids_by_object=defaultdict(list)
    for pid,pname in enumerate(names):
        part_ids_by_object[pname.split("'s ",1)[0]].append(pid+args.part_id_offset)
    records=rows(load(args.cache))
    n=len(records);limit=min(n,args.preflight) if args.preflight else n
    masks=torch.zeros((limit,N),dtype=torch.bool)
    manifest=[];tot=Counter();byobj=defaultdict(Counter)
    for i,r in enumerate(records[:limit]):
        ids,obj=candidate(r,names)
        if args.mask_mode=='part_union':
            label=part_ids_by_object[obj]
            if not label:raise RuntimeError(f'No ADE part IDs for object {obj}')
            if 65535 in label:raise RuntimeError('ADE part label collides with ignore index 65535')
        stem=Path(str(r['image_id'])).stem
        path=mask_path(args.gt_part_root,stem,obj,args.mask_mode)
        gt=read_png(str(path))
        if args.mask_mode=='part_union':
            vals=np.unique(gt)
            wrong=vals[(vals!=65535) & ((vals<args.part_id_offset) | (vals>=234+args.part_id_offset))]
            if len(wrong):
                raise ValueError(f'Unexpected GT part label values {wrong.tolist()} in {path}; verify ID offset / file format')
        valid,info=crop_fg(r,gt,obj,label,args.mask_mode,1,args.images_root)
        pred=np.asarray(torch.as_tensor(r['pred_obj_mask_patch']).bool().reshape(-1),dtype=np.bool_)
        if pred.shape!=(N,):raise ValueError(f'Wrong predicted foreground size at row {i}')
        masks[i]=torch.from_numpy(valid.copy())
        info.update(row_index=i,source_pids=ids,mask_path=str(path),pred_fg_patches=int(pred.sum()),overlap_patches=int((valid & pred).sum()))
        manifest.append(info)
        tot['n']+=1;tot['gt_valid_sum']+=info['gt_valid_patches'];tot['pred_valid_sum']+=info['pred_fg_patches'];tot['empty']+=int(not valid.any())
        byobj[obj]['n']+=1;byobj[obj]['empty']+=int(not valid.any())
        if i<12 or (i+1)%1000==0:
            print(f'[MASK] {i+1}/{limit} {stem} obj={obj} n_parent_part_ids={len(label)} GT={info["gt_valid_patches"]} '
                  f'pred={info["pred_fg_patches"]} overlap={info["overlap_patches"]} wh={info["crop_wh"]}',flush=True)
    print('[STATS]',dict(tot),'[BY_OBJECT]',{k:dict(v) for k,v in byobj.items()},flush=True)
    if args.preflight:
        print('[PREFLIGHT ONLY] NO FILES WRITTEN')
        return
    if args.empty_policy=='error' and tot['empty']:
        raise RuntimeError(f'{tot["empty"]} crops have empty GT part-union foreground. Refusing to train. '
                           'If acceptable after inspection, explicitly use --empty-policy drop.')
    bad=[(o,v['n'],v['empty']) for o,v in byobj.items() if v['n']>=20 and v['empty']/v['n']>0.75]
    if bad:raise RuntimeError(f'>75% empty part union for {bad}: part mask ID offset, crop geometry or dataset may be wrong')
    if args.out.exists():raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    st=args.cache.stat()
    protocol=dict(format='ade234_GT_part_union_foreground_oracle_v1',GT_part_privileged=True,GT_object_privileged=False,gt_part_id_offset=args.part_id_offset,source_cache=str(args.cache.resolve()),
        source_cache_size=st.st_size,source_cache_mtime_ns=st.st_mtime_ns,gt_part_root=str(args.gt_part_root.resolve()),
        mask_mode=args.mask_mode,empty_policy=args.empty_policy,n_records=n,stats=dict(tot),
        by_object={k:dict(v) for k,v in byobj.items()},only_change='valid_patch bool 1024 from union of spatial GT part pixels of candidate parent object; original crops/DINO tokens/part presence unchanged',
        warnings=['GT PART spatial annotation used during training: NOT a weakly supervised result, NOT independent GT object masks',
                  'Unlabeled object pixels excluded; mask union may miss parts and merge same-class instances; does not reproduce independent object GT',
                  'Any empty-GT crop dropped changes training sample count',
                  'GT mask must match actual cached image and crop geometry, not only same image_id'])
    torch.save({'metadata':protocol,'masks':masks,'manifest':manifest},args.out)
    args.out.with_suffix('.protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    with args.out.with_suffix('.manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['row_index','image_id','object','object_id','crop_box_xyxy','flip','gt_valid_patches','pred_fg_patches','overlap_patches','crop_wh','mask_path']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for x in manifest:w.writerow(x)
    print('[SAVED]',args.out,'SHAPE',tuple(masks.shape),'EMPTY_GT',tot['empty'],flush=True)

if __name__=='__main__':main()
