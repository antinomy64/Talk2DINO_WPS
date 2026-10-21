#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export class-wise IoU, Acc (class pixel recall), and contribution to aAcc.

Only existing held-out evaluation JSON files are read. No predictions are changed.
Class contribution to aAcc = GT_count[class] / GT_count[all valid] * Acc[class].
This is NOT a standard independent per-class aAcc and is labelled accordingly.
"""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

METHODS={'PP116':('ours','gtobj','gtproto'),'ADE234':('ours','gtfg','gtproto')}
DEFAULTS={
 'pp_ours':'segmentation_results/vitb_mlp_infonce_coco2014_reproduce_clean_softrelative_structure_e100_reproduce.json',
 'pp_gtobj':'segmentation_results/vitb_mlp_infonce_coco2014_GTObjFG_ORACLE_lam4_e100.json',
 'pp_gtproto':'',  # Choose the actual PP116 GT CE ORACLE evaluation JSON.
 'ade_ours':'segmentation_results/vitb_mlp_infonce_coco2014_reproduce_clean_ade234_soft_structure_lambda4p0_100ep.json',
 'ade_gtfg':'segmentation_results/vitb_mlp_infonce_coco2014_ade234_GTPartUnionFG_ORACLE_lam4_e100.json',
 'ade_gtproto':'segmentation_results/vitb_mlp_infonce_coco2014_ade234_GTProto_OrthCE_150step.json',
}

def read_eval(path):
    if not path:return None
    p=Path(path)
    if not p.is_file():print('[PENDING] Not found:',p);return None
    d=json.loads(p.read_text())
    if not isinstance(d,dict) or not d.get('metrics'):raise ValueError(f'Bad eval JSON {p}')
    m=d['metrics'][0]
    cl={k[4:] for k in m if k.startswith('IoU.')}
    return m,cl,str(p)

def take(m,key):
    try:
        z=float(m[key]);return z if math.isfinite(z) else None
    except (ValueError,TypeError,KeyError):return None

def histogram(gt_root,num_classes,ignore,expected):
    root=Path(gt_root)
    if not root.is_dir():
        print('[NO GT PIXEL HISTOGRAM]',root);return None
    masks=sorted(root.glob('*.png'))
    if len(masks)!=expected:
        print(f'[NO GT PIXEL HISTOGRAM] Got {len(masks)} masks under {root}, expected {expected} evaluated images.\n'
              'Class-aAcc contributions require the exact SAME evaluation subset; not approximating.')
        return None
    counts=np.zeros(num_classes,dtype=np.int64)
    for p in masks:
        with Image.open(p) as im:
            a=np.asarray(im,dtype=np.int64)
        valid=a!=ignore
        if valid.any():
            arr=a[valid]
            if arr.min()<0 or arr.max()>=num_classes:raise RuntimeError(f'Unexpected val GT labels in {p}')
            counts+=np.bincount(arr,minlength=num_classes)
    if not counts.any():raise RuntimeError('Empty eval GT masks')
    return counts

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        wr.writeheader();wr.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key,default in DEFAULTS.items():ap.add_argument('--'+key.replace('_','-'),dest=key,default=default)
    ap.add_argument('--pp-gt-root',default='data/PascalPart116/annotations_detectron2_part/val')
    ap.add_argument('--ade-gt-root',default='../../../dataset/lyx/ADE20KPart234/annotations_detectron2_part/val')
    ap.add_argument('--out-dir',default='output/partwise_six_methods')
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(exist_ok=True,parents=True)
    summary=[]
    for dataset,C,maskroot,ignore,expected in [('PP116',116,args.pp_gt_root,255,850),('ADE234',234,args.ade_gt_root,65535,1016)]:
        key=dataset.lower() if dataset=='PP116' else 'ade'
        stem='pp' if dataset=='PP116' else 'ade'
        vals={}
        names=None
        for method in METHODS[dataset]:
            r=read_eval(getattr(args,f'{stem}_{method}'))
            if r is not None:
                m,ns,path=r
                if len(ns)!=C:raise ValueError(f'{dataset} {method}: expected {C} parts, got {len(ns)}')
                if names is not None and names!=ns:raise ValueError(f'{dataset} class taxonomy differs: {method}')
                names=ns
                vals[method]=(m,path)
                summary.append({'Dataset':dataset,'Method':method,'mIoU_pct':100*take(m,'mIoU'),
                                'mAcc_pct':100*take(m,'mAcc'),'aAcc_pct':100*take(m,'aAcc'),
                                'Eval_JSON':path})
        if not vals:raise RuntimeError(f'No {dataset} evaluation JSON found')
        # Preserve dataset's canonical class order from its package taxonomy.
        if dataset=='ADE234':
            from ade20kpart234_taxonomy import PART_CLASSES
            parts=list(PART_CLASSES)
        else:
            try:
                from src.open_vocabulary_segmentation.segmentation.datasets.pascalpart116_part import PART_CLASSES
                parts=list(PART_CLASSES)
            except Exception:
                m=list(vals.values())[0][0]
                parts=[k[4:] for k in m if k.startswith('IoU.')]
        if len(parts)!=C or set(parts)!=names:raise RuntimeError(f'{dataset} class order/taxonomy invalid')
        counts=histogram(maskroot,C,ignore,expected)
        if counts is not None:
            for method,(m,path) in vals.items():
                reconstructed=100*sum(counts[i]*take(m,'Acc.'+p) for i,p in enumerate(parts) if take(m,'Acc.'+p) is not None)/counts.sum()
                log_acc=100*take(m,'aAcc')
                print(f'[CROSSCHECK] {dataset} {method}: GT reconstruction aAcc={reconstructed:.4f} logged={log_acc:.4f}')
                if abs(reconstructed-log_acc)>0.2:raise RuntimeError(
                    f'{dataset} GT mask coverage differs from evaluation; refusing per-part aAcc contribution')
        rows=[]
        for i,p in enumerate(parts):
            row={'class_id':i,'Object':p.split("'s ",1)[0],'Part':p,
                 'GT_pixels_val':int(counts[i]) if counts is not None else '',
                 'GT_in_validation':int(counts[i]>0) if counts is not None else (int(take(next(iter(vals.values()))[0],'Acc.'+p) is not None))}
            for method in METHODS[dataset]:
                m=vals.get(method,(None,None))[0]
                iou=take(m,'IoU.'+p) if m is not None else None
                acc=take(m,'Acc.'+p) if m is not None else None
                row[method+'_IoU_pct']=round(100*iou,6) if iou is not None else ''
                row[method+'_Acc_pct']=round(100*acc,6) if acc is not None else ''
                row[method+'_aAcc_contribution_pct']=round(100*acc*counts[i]/counts.sum(),8) if (acc is not None and counts is not None) else ''
            rows.append(row)
        path=out/(dataset+'_all_parts.csv');write_csv(path,rows)
        print('[SAVED]',path,'rows=',len(rows))
    write_csv(out/'summary.csv',summary)
    print('[SAVED]',out/'summary.csv')

if __name__=='__main__':main()
