#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bake W and run unchanged official 116-way evaluator, plus post-hoc class means.

No seen/unseen filtering is performed at inference. Missing GT classes
with NaN IoU/Acc are excluded from the relevant macro mean, as in mmseg.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path

UNSEEN_OBJECTS = frozenset({'bird','car','dog','motorbike','sheep'})
PROJECTOR = 'weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
TEXT = 'feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
MODEL_CONFIG = 'configs/vitb_mlp_infonce.yaml'
EVAL_CFG = 'src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml'
EVAL_BASE_CFG = 'src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml'

def mean_finite(vals):
    good=[v for v in vals if math.isfinite(v)]
    return (sum(good)/len(good) if good else float('nan')),len(good)

def parse_official_output(txt):
    m = re.search(r'mIoU of\s+850\s+test images:\s*([\d.]+)%', txt)
    if not m:
        raise ValueError('Official 850-image mIoU line missing; evaluation may be incomplete')
    found, in_table = [], False
    for ln in txt.splitlines():
        if 'per class results:' in ln.lower():
            in_table=True
            continue
        if in_table and 'Summary:' in ln:
            break
        if in_table and ln.strip().startswith('|'):
            fields=[s.strip() for s in ln.strip().split('|')[1:-1]]
            if len(fields)==3 and "'s " in fields[0]:
                found.append((fields[0],float(fields[1]),float(fields[2])))
    if len(found)!=116 or len({x[0] for x in found})!=116:
        raise ValueError(f'Expected 116 distinct part rows, found {len(found)}')
    seen=[x[1] for x in found if x[0].split("'s ",1)[0] not in UNSEEN_OBJECTS]
    unseen=[x[1] for x in found if x[0].split("'s ",1)[0] in UNSEEN_OBJECTS]
    if len(seen)!=74 or len(unseen)!=42:
        raise ValueError(f'74/42 split mismatch: {len(seen)}/{len(unseen)}')
    s,n_s=mean_finite(seen)
    u,n_u=mean_finite(unseen)
    acc,n_acc=mean_finite([x[2] for x in found])
    allv,n_all=mean_finite([x[1] for x in found])
    h=2*s*u/(s+u) if math.isfinite(s+u) and (s+u)>0 else float('nan')
    return {
       'all116_official_mIoU_pct':float(m.group(1)),
       'all116_recomputed_mIoU_pct':allv,
       'all116_recomputed_mACC_pct':acc,
       'seen74_mIoU_pct':s, 'unseen42_mIoU_pct':u,'hIoU_pct':h,
       'seen_valid':n_s,'unseen_valid':n_u,
       'all_valid_IoU':n_all,'all_valid_Acc':n_acc,
    }, found

def run(cmd, output, env):
    output.parent.mkdir(parents=True,exist_ok=True)
    print('[EXEC]',' '.join(map(str,cmd)),flush=True)
    with output.open('w',encoding='utf8') as f:
        result=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,
                              env=env,check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed (exit={result.returncode}): {output}\n'+
                           '\n'.join(output.read_text(errors='replace').splitlines()[-40:]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='3')
    p.add_argument('--run_dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',default='3,10,20,30')
    p.add_argument('--master_port',type=int,default=30191)
    args=p.parse_args()
    root=Path.cwd()
    for path in (PROJECTOR,TEXT,MODEL_CONFIG,EVAL_CFG,EVAL_BASE_CFG,
                 'bake_predobj_relproto_w_into_projector_final.py'):
        if not (root/path).is_file():
            raise FileNotFoundError(f'Run in original repository root; missing {path}')
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    env['PYTHONUNBUFFERED']='1'
    out=Path(args.output)
    runpath=Path(args.run_dir).resolve()
    slug=hashlib.sha256(str(runpath).encode()).hexdigest()[:10]
    rows=[]
    each=[]
    for epoch in [int(i) for i in args.epochs.split(',') if i.strip()]:
        ck=Path(args.run_dir)/f'W_epoch_{epoch:03d}.pt'
        if not ck.is_file():
            raise FileNotFoundError(f'Checkpoint missing: {ck}')
        proj_name=f'pp116_weighted_nce_{slug}_e{epoch:03d}'
        baked=Path('weights')/(proj_name+'.pth')
        ep_dir=out/f'epoch_{epoch:03d}'
        bake_log=ep_dir/'bake_stdout.log'
        eval_log=ep_dir/'official_eval_stdout.log'
        if not baked.exists() or baked.stat().st_mtime < ck.stat().st_mtime:
            run([sys.executable,'bake_predobj_relproto_w_into_projector_final.py',
                 '--project_root','.','--projector',PROJECTOR,
                 '--w_checkpoint',ck,'--text_bank',TEXT,
                 '--model_config',MODEL_CONFIG,'--output',baked,'--device','cuda'],
                 bake_log,env)
        cached=False
        if eval_log.exists() and eval_log.stat().st_mtime >= baked.stat().st_mtime:
            try:
                parse_official_output(eval_log.read_text(errors='replace'))
                cached=True
            except ValueError:
                pass
        if not cached:
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1',
                 '--master_port',str(args.master_port),
                 'src/open_vocabulary_segmentation/main.py','--eval',
                 '--output',ep_dir/'official_output',
                 '--eval_cfg',EVAL_CFG,'--eval_base_cfg',EVAL_BASE_CFG,
                 '--opts',f'model.proj_name={proj_name}','evaluate.pamr=false'],
                 eval_log,env)
        scores, parts=parse_official_output(eval_log.read_text(errors='replace'))
        rows.append({'epoch':epoch,**scores})
        each += [dict(epoch=epoch,part=name,IoU_pct=iou,Acc_pct=acc,
                       split=('unseen' if name.split("'s ",1)[0] in UNSEEN_OBJECTS else 'seen'))
                 for name,iou,acc in parts]
        print(f"[E{epoch:03d}] official mIoU={scores['all116_official_mIoU_pct']:.3f}% "
              f"mACC={scores['all116_recomputed_mACC_pct']:.3f}% "
              f"Seen={scores['seen74_mIoU_pct']:.3f}% "
              f"Unseen={scores['unseen42_mIoU_pct']:.3f}% "
              f"hIoU={scores['hIoU_pct']:.3f}%",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    for path,data in [(out/'official_macro_metrics.csv',rows),
                      (out/'all116_per_part.csv',each)]:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]))
            w.writeheader();w.writerows(data)
    print('[DONE]',out/'official_macro_metrics.csv',flush=True)

if __name__=='__main__':
    main()
