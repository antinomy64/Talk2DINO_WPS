#!/usr/bin/env python3
"""Parse authoritative 850-image official val results and ABCD consequences."""
from __future__ import annotations
import argparse,csv,json,math,re,zipfile
from collections import defaultdict
from pathlib import Path

def official(path):
    if not path.is_file():return None
    txt=path.read_text(errors='replace')
    m=re.search(r'mIoU of (\d+) test images:\s*([\d.]+)%',txt)
    if not m:return None
    n=int(m.group(1))
    if n!=850:raise RuntimeError(f'Not expected 850 val images: {path}, n={n}')
    values={}
    for line in txt.splitlines():
        chunks=[v.strip() for v in line.strip().strip('|').split('|')]
        if len(chunks)!=3 or "'s " not in chunks[0]:continue
        try:
            iou=float(chunks[1]);acc=float(chunks[2])
            if math.isfinite(iou):values[chunks[0]]=(iou,acc)
        except ValueError:pass
    if len(values)!=116:raise RuntimeError(f'Expected 116 per-class IoU rows, got {len(values)} at {path}')
    return float(m.group(2)),values

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    report=args.report
    reg=json.loads((report/'registry.json').read_text())
    classes={}
    with (Path(__file__).resolve().parent/'pp116_four_groups_per_part.csv').open(encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f):classes[x['part']]=x['类别编码']
    rows=[];metrics=[];baseline={};missing=[]
    for v in reg:
        name=v['variant']
        ck=Path(v['train_dir'])/'training_history.csv'
        if ck.exists():
            with ck.open(newline='') as f:
                for x in csv.DictReader(f):
                    metrics.append({'variant':name,**x})
        paths=sorted((report/'val'/name).glob('e*/evaluation_stdout.log'))
        for path in paths:
            result=official(path)
            if not result:
                missing.append(str(path));continue
            miou,parts=result
            ep=int(path.parent.name[1:])
            for label,(iou,acc) in parts.items():
                # Official Acc is per-class recall, NOT precision.
                I=iou/100.0; R=acc/100.0
                den=R*(1+I)-I
                precision=100*(I*R/den) if math.isfinite(R) and den>1e-12 else ''
                rows.append({'variant':name,'epoch':ep,'part':label,'object':label.split("'s ",1)[0],
                             'group':classes.get(label,'UNLABELED'),'IoU_pct':iou,'Acc_pct':acc,
                             'Precision_inferred_pct':precision,
                             'mIoU_pct':miou,'num_val_images':850})
    summary=report/'summary'
    write(summary/'official_116_per_part.csv',rows)
    write(summary/'all_training_history.csv',metrics)
    if missing:(summary/'missing_evaluations.txt').write_text('\n'.join(missing))
    means=[]
    grouped=defaultdict(list)
    for r in rows:grouped[(r['variant'],r['epoch'],r['group'])].append(r)
    for (name,ep,grp),partrows in sorted(grouped.items()):
        means.append({'variant':name,'epoch':ep,'group':grp,'part_count':len(partrows),
                      'group_mean_IoU_pct':round(sum(r['IoU_pct'] for r in partrows)/len(partrows),6),
                      'all116_mIoU_pct':partrows[0]['mIoU_pct']})
    write(summary/'ABCD_official_summary.csv',means)
    # Same-epoch matched comparison, *not* selecting winners by training pseudo-label GT.
    by={(r['epoch'],r['part']):r['IoU_pct'] for r in rows if r['variant']=='baseline_retrain'}
    changes=[]
    for r in rows:
        if r['variant']=='baseline_retrain':continue
        key=(r['epoch'],r['part'])
        if key in by:
            changes.append({**r,'baseline_IoU_pct':by[key],
                            'delta_IoU_pp':round(r['IoU_pct']-by[key],6)})
    write(summary/'delta_vs_retrained_baseline.csv',changes)
    outcomes=[]
    final_by=defaultdict(list)
    for r in rows:final_by[(r['variant'],r['epoch'])].append(r)
    for (variant,ep),pr in final_by.items():
        bygrp=defaultdict(list)
        for r in pr:bygrp[r['group']].append(r['IoU_pct'])
        baserows=final_by.get(('baseline_retrain',ep),[])
        basemiou=baserows[0]['mIoU_pct'] if baserows else float('nan')
        outcomes.append({'variant':variant,'epoch':ep,'all116_mIoU_pct':pr[0]['mIoU_pct'],
                         'delta_mIoU_pp':round(pr[0]['mIoU_pct']-basemiou,6) if baserows else '',
                         **{f'group_{g}_mean_IoU_pct':round(sum(bygrp[g])/len(bygrp[g]),6) if g in bygrp else ''
                            for g in ('A','B','C','D')},'mean_116_perclass_IoU_pct':round(sum(x['IoU_pct'] for x in pr)/len(pr),6)})
    write(summary/'variant_scorecard.csv',sorted(outcomes,key=lambda x:(x['epoch'],x['variant'])))
    # Bundle only lightweight logs/CSVs/manifests; large checkpoints never included.
    dest=report/'review_upload.zip'
    allowed={'.csv','.json','.log','.txt','.md'}
    tozip=[p for p in report.rglob('*') if p.is_file() and p!=dest and p.suffix.lower() in allowed
           and p.stat().st_size<25*1024*1024]
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in tozip:
            z.write(path,arcname=str(path.relative_to(report)))
        z.write(Path(__file__).resolve().parent/'train_pp116_pipeline_search.py',arcname='scripts/train_pp116_pipeline_search.py')
        z.write(Path(__file__).resolve().parent/'run_search.py',arcname='scripts/run_search.py')
        z.write(Path(__file__).resolve().parent/'summarize_search.py',arcname='scripts/summarize_search.py')
    print('OFFICIAL EVALUATIONS:',len(final_by))
    print('ROWS:',len(rows),'OUTPUT:',dest,'SIZE_MiB:',round(dest.stat().st_size/(1024**2),2))

if __name__=='__main__':main()
