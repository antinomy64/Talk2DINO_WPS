#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ORIGINAL PP116 official 116-way evaluation for a trained Talk2DINO projector.
Report Seen/Unseen mIoU via finite per-class IoU means; IoU=0 is retained.
"""
from __future__ import annotations
import argparse,csv,math,os,re,subprocess,sys
from pathlib import Path
UNSEEN={'bird','car','dog','motorbike','sheep'}

def parse(log):
    rows=[]; in_table=False
    for line in log.splitlines():
        if 'per class results:' in line.lower(): in_table=True;rows=[];continue
        if in_table and 'Summary:' in line:break
        if in_table:
            m=re.match(r'^\s*\|\s*(.*?)\s*\|\s*(nan|[0-9.]+)\s*\|\s*(nan|[0-9.]+)\s*\|\s*$',line,re.I)
            if m and "'s " in m.group(1):rows.append((m.group(1).strip(),float(m.group(2))))
    if len(rows)!=116 or len(set(name for name,iou in rows))!=116:
        raise ValueError(f'Expected 116 unique official class IoUs; got {len(rows)}')
    all_values=[x for n,x in rows if math.isfinite(x)]
    seen=[x for n,x in rows if n.split("'s ",1)[0] not in UNSEEN and math.isfinite(x)]
    unseen=[x for n,x in rows if n.split("'s ",1)[0] in UNSEEN and math.isfinite(x)]
    if len(seen) not in (73,74) or len(unseen)!=42:
        raise ValueError(f'Unexpected valid class counts Seen={len(seen)}, Unseen={len(unseen)}')
    s=sum(seen)/len(seen); u=sum(unseen)/len(unseen);h=2*s*u/(s+u) if s+u else 0
    return rows,{'seen_mIoU_pct':s,'unseen_mIoU_pct':u,'hIoU_pct':h,'all_mIoU_pct':sum(all_values)/len(all_values),'seen_valid_classes':len(seen),'unseen_valid_classes':len(unseen)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--projector',default='weights/vitb_mlp_infonce_pp116_continue_pp116_seen74_30ep_seed123.pth')
    p.add_argument('--output',default='output/pp116_talk2dino_part_seen74_30ep_official')
    p.add_argument('--log',default='pp116_talk2dino_part_seen74_30ep_eval.log')
    p.add_argument('--master_port',type=int,default=30182)
    p.add_argument('--parse_only',action='store_true')
    a=p.parse_args()
    project=Path(a.projector)
    if not a.parse_only:
        if not project.is_file(): raise FileNotFoundError(f'Trained projector not found: {project}')
        if project.parent.resolve()!=Path('weights').resolve():
            raise ValueError('Official evaluator model.proj_name loads from weights/. Put trained checkpoint there.')
        if Path(a.log).exists():
            raise FileExistsError(f'Log exists: {a.log}. Use --parse_only to reuse it, or choose a new --log.')
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(a.master_port),
             'src/open_vocabulary_segmentation/main.py','--eval','--output',a.output,
             '--eval_cfg','src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
             '--eval_base_cfg','src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
             '--opts',f'model.proj_name={project.stem}','evaluate.pamr=false']
        print('[EXEC]',' '.join(cmd),flush=True)
        with open(a.log,'w',encoding='utf-8') as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Official evaluation failed ({r.returncode}). Inspect {a.log}')
    log=Path(a.log).read_text(errors='replace')
    rows,summary=parse(log)
    print('===== Official 116-way PP116 evaluation =====')
    for k,v in summary.items(): print(f'{k}: {v:.5f}' if isinstance(v,float) else f'{k}: {v}')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'seen_unseen_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary));w.writeheader();w.writerow(summary)
    with (out/'all116_per_part_iou.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['part','split','IoU_percent']);w.writerows((n,'unseen' if n.split("'s ",1)[0] in UNSEEN else 'seen',iou) for n,iou in rows)
    print('[SAVED]',out,flush=True)
if __name__=='__main__':main()
