#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an indexed GT-OBJECT-valid PATCH mask sidecar for PP116 predicted-object crops.

ORACLE-ONLY. Never use this file to claim weakly supervised training.
Keep predicted object crops, cached DINO patch tokens and part-presence labels fixed.
Replace ONLY the validity mask: GT object-ID pixels -> same crop box -> same recorded
horizontal flip -> nearest 32x32 -> (label == cached crop object category).
The class-ID map below corresponds to Pascal VOC20 ZERO-BASED object masks;
pass --object-map-json to override if your data uses another mapping.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from audit_pp116_gt_weighted_purity import load_pt, annotations, part_ids, get_box, get_flip, canonical_names

VOC20 = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
         'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')
DEFAULT_MAP={name:i for i,name in enumerate(VOC20)}
IGNORE=255
N=1024

@lru_cache(maxsize=128)
def read_mask(pathstr):
    path=Path(pathstr)
    with Image.open(path) as f:
        mode=f.mode
        x=np.asarray(f)
    if x.ndim!=2 or mode in ('RGB','RGBA'):
        raise ValueError(f'GT object mask must be indexed single channel, not RGB: {path}, mode={mode}, shape={x.shape}')
    x=x.astype(np.int32,copy=False)
    vals=np.unique(x)
    invalid=vals[(vals!=IGNORE)&((vals<0)|(vals>=20))]
    if invalid.size:
        raise ValueError(f'Unexpected GT object IDs in {path}: {invalid.tolist()}. Check object_map and mask convention.')
    return x


def oracle_fg_grid(annotation, root, object_id, flip_mode='auto'):
    image_id=str(annotation['image_id'])
    path=root/(Path(image_id).stem+'.png')
    if not path.is_file():raise FileNotFoundError(path)
    full=read_mask(str(path))
    h,w=full.shape
    b=get_box(annotation)
    if np.all(b>=0) and float(b.max())<=1.01 and max(h,w)>8:
        raise ValueError(f'Expected pixel XYXY crop, got normalized box {image_id}: {b}')
    x0,y0,x1,y1=[int(round(float(v))) for v in b]
    x0,x1=max(0,min(w,x0)),max(0,min(w,x1))
    y0,y1=max(0,min(h,y0)),max(0,min(h,y1))
    if x1<=x0 or y1<=y0:raise ValueError(f'Invalid crop {b} for {image_id} {w}x{h}')
    cut=np.ascontiguousarray(full[y0:y1,x0:x1])
    flip=get_flip(annotation,flip_mode)
    if flip:cut=np.ascontiguousarray(cut[:,::-1])
    grid=np.asarray(Image.fromarray(cut,mode='I').resize((32,32),resample=Image.Resampling.NEAREST),dtype=np.int32)
    valid=np.ascontiguousarray((grid==object_id).reshape(N))
    return valid,dict(image_id=image_id,object_id=object_id,full_obj_pixel_count=int((full==object_id).sum()),
                      crop_obj_grid_count=int(valid.sum()),pred_fg_grid_count=int(torch.as_tensor(annotation['pred_obj_mask_patch']).bool().sum()),
                      overlap_grid_count=int((valid&torch.as_tensor(annotation['pred_obj_mask_patch']).bool().cpu().numpy().reshape(N)).sum()),
                      applied_hflip=int(flip),box=[int(x0),int(y0),int(x1),int(y1)],gt_ids_full=np.unique(full).tolist())


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        a=np.ones((16,16),dtype=np.uint8)*IGNORE
        a[:,2:9]=14
        a[3:5,11:13]=7
        Image.fromarray(a).save(root/'t.png')
        record=dict(image_id='t',cropaug_box_xyxy=[0,0,16,16],pred_obj_mask_patch=torch.ones(N,dtype=torch.bool))
        fg,_=oracle_fg_grid(record,root,14)
        assert fg.shape==(N,) and fg.dtype==np.bool_ and 0<fg.sum()<N
        fg7,_=oracle_fg_grid(record,root,7)
        assert 0<fg7.sum()<fg.sum()
        fg255,_=oracle_fg_grid(record,root,19)
        assert not fg255.any()
        record['cropaug_hflip']=True
        fgflip,_=oracle_fg_grid(record,root,14)
        assert np.array_equal(fg.reshape(32,32)[:,::-1],fgflip.reshape(32,32))
    print('SELF_TEST_PASS_GT_OBJECT_VALID_MASK')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--gt-object-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_obj/train'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--object-map-json',type=Path,default=None,help='Optional JSON dictionary object-name -> zero-based GT object ID')
    p.add_argument('--flip-mode',choices=('auto','none','all'),default='auto')
    p.add_argument('--preflight',type=int,default=0,help='Check first N records only, do not save mask bank')
    p.add_argument('--out',type=Path,default=Path('output/pp116_gtobj_fg_oracle/gt_object_valid_masks.pt'))
    p.add_argument('--empty-policy',choices=('error','drop'),default='drop',help='GT empty cannot be used for Softmax. drop skips these records, never reuses predicted FG')
    a=p.parse_args()
    if a.self_test:return self_test()
    for f in (a.cache,a.classes_file):
        if not f.is_file():raise FileNotFoundError(f)
    if not a.gt_object_root.is_dir():raise FileNotFoundError(a.gt_object_root)
    names=canonical_names(a.classes_file)
    objmap=json.loads(a.object_map_json.read_text()) if a.object_map_json else DEFAULT_MAP
    if any(int(x)<0 or int(x)>19 for x in objmap.values()):raise ValueError('Object mask label IDs must be 0..19 for zero_based')
    cache=load_pt(a.cache)
    records=annotations(cache)
    n=len(records)
    if not records:raise RuntimeError('No original records')
    limit=min(n,a.preflight) if a.preflight else n
    masks=torch.zeros((limit,N),dtype=torch.bool)
    manifest=[]
    stats=Counter()
    per_obj=defaultdict(lambda:Counter())
    for i in range(limit):
        r=records[i]; ids=part_ids(r)
        objects={names[pid].split("'s ",1)[0] for pid in ids}
        if len(objects)!=1:raise RuntimeError(f'Mixed object candidate parts at row {i}: {objects}')
        obj=next(iter(objects))
        if obj not in objmap:raise KeyError(f'Missing GT object class id for {obj}')
        oid=int(objmap[obj])
        valid,d=oracle_fg_grid(r,a.gt_object_root,oid,a.flip_mode)
        masks[i]=torch.from_numpy(valid.copy())
        d.update(row_index=i,object=obj,source_pids=ids)
        manifest.append(d)
        stats['records']+=1;stats['original_pred_fg_patches']+=d['pred_fg_grid_count']
        stats['gt_obj_fg_patches']+=d['crop_obj_grid_count'];stats['shared_fg_patches']+=d['overlap_grid_count']
        stats['zero_gt_mask']+=int(not valid.any());stats['zero_full_obj_mask']+=int(d['full_obj_pixel_count']==0)
        per_obj[obj]['records']+=1;per_obj[obj]['empty_gt_crop']+=int(not valid.any())
        if i<12 or (i+1)%1000==0:
            print(f'[MASK] {i+1}/{limit} img={d["image_id"]} obj={obj} id={oid} '
                  f'GT full={d["full_obj_pixel_count"]} GT crop32={d["crop_obj_grid_count"]} '
                  f'pred={d["pred_fg_grid_count"]} overlap={d["overlap_grid_count"]}',flush=True)
    print('[STATS]',dict(stats),'per object',dict(per_obj),flush=True)
    if a.preflight:
        print('[PREFLIGHT ONLY] No files written')
        return
    # Guard against accidentally applying VOC object IDs to an incompatible annotation scheme.
    too_many_missing=[(o,z['records'],z['empty_gt_crop']) for o,z in per_obj.items()
                      if z['records']>=20 and z['empty_gt_crop']/z['records']>.75]
    if too_many_missing:
        raise RuntimeError(f'>75% GT object crop masks are empty for objects {too_many_missing}; verify zero-based object ID mapping, masks and crop geometry')
    if a.empty_policy=='error' and stats['zero_gt_mask']:
        raise RuntimeError(f'{stats["zero_gt_mask"]} empty GT object foreground crops. Choose --empty-policy drop explicitly or inspect masks.')
    if a.out.exists():raise FileExistsError(f'Refusing to overwrite: {a.out}')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    st=a.cache.stat()
    report={
        'format':'pp116_gt_object_foreground_oracle_sidecar_v1',
        'oracle_privileged_gt_used':True,
        'source_cache':str(a.cache.resolve()),'source_cache_size':st.st_size,'source_cache_mtime_ns':st.st_mtime_ns,
        'gt_object_root':str(a.gt_object_root.resolve()),'class_id_map':objmap,
        'object_mask_mode':'zero_based_target_object_id; 255 ignored; nonmatching IDs excluded',
        'crop_rule':'round xyxy, clip to GT dimensions, optional recorded horizontal flip, 32x32 nearest',
        'candidate_parts':'UNCHANGED image-level presence from original cache',
        'crop_features':'UNCHANGED cached DINOv2 patch tokens',
        'empty_gt_policy':a.empty_policy,'n_records':n,'stats':dict(stats),
        'by_object':{key:dict(val) for key,val in per_obj.items()},
    }
    torch.save({'metadata':report,'masks':masks,'manifest':manifest},a.out)
    csvpath=a.out.with_suffix('.manifest.csv')
    with csvpath.open('w',encoding='utf-8-sig',newline='') as fp:
        fields=['row_index','image_id','object','object_id','box','full_obj_pixel_count','crop_obj_grid_count',
                'pred_fg_grid_count','overlap_grid_count','applied_hflip']
        w=csv.DictWriter(fp,fields);w.writeheader()
        for d in manifest:w.writerow({k:d[k] for k in fields})
    a.out.with_suffix('.protocol.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[SAVED]',a.out,'MASK_SHAPE',tuple(masks.shape),'EMPTY_GT',stats['zero_gt_mask'],flush=True)

if __name__=='__main__':main()
