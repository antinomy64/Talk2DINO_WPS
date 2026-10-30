#!/usr/bin/env python3
"""Parse ONLY actual official validation logs; make per-Ear contrasts & fixed GT probe analysis."""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
import pandas as pd

EAR=('cat','cow','dog','horse','person','sheep')

def parse_eval(path):
    path=Path(path)
    if not path.is_file():return None
    text=path.read_text(errors='replace')
    mi=re.search(r'mIoU of (\d+) test images:\s*([\d.]+)%',text)
    if not mi:return None
    out={'num_val_images':int(mi.group(1)),'mIoU_pct':float(mi.group(2))}
    if out['num_val_images']!=850:
        raise RuntimeError(f'Evaluation validation image count mismatch {path}: {out["num_val_images"]} != 850')
    for part in EAR:
        for noun in ('ear','head'):
            key=f"{part}'s {noun}"
            pattern=rf'\|\s*{re.escape(key)}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
            mats=re.findall(pattern,text)
            if not mats:return None
            iou,acc=mats[-1]
            out[f'{part}_{noun}_IoU_pct']=float(iou)
            out[f'{part}_{noun}_Acc_pct']=float(acc)
    return out

def save(df,path):
    if df is not None and not df.empty:
        df.to_csv(path,index=False,encoding='utf-8-sig')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    report=args.report
    summary=report/'summary';summary.mkdir(parents=True,exist_ok=True)
    registry=report/'registry.json'
    if not registry.exists():raise FileNotFoundError(registry)
    records=json.loads(registry.read_text())
    meta={r['variant']:r for r in records}
    valrows=[];missing=[]
    for r in records:
        for log in sorted((report/'val'/r['variant']).glob('e*/evaluation_stdout.log')):
            ep=int(log.parent.name[1:]);parsed=parse_eval(log)
            if parsed:
                valrows.append({'variant':r['variant'],'epoch':ep,**parsed})
            else:missing.append(str(log))
    val=pd.DataFrame(valrows)
    save(val,summary/'official_ear_head_iou_acc.csv')
    if missing:(summary/'unparsed_evaluations.txt').write_text('\n'.join(missing))
    fixed=[];grad=[]
    for r in records:
        p=report/'probes'/r['variant']/'fixed_crop_ear_epoch.csv'
        if p.is_file():
            df=pd.read_csv(p)
            df.insert(0,'variant',r['variant'])
            fixed.append(df)
        q=report/'probes'/r['variant']/'gradient_by_target.csv'
        if q.is_file():
            g=pd.read_csv(q)
            g.insert(0,'variant',r['variant']);grad.append(g)
    if fixed:
        fixed=pd.concat(fixed,ignore_index=True)
        save(fixed,summary/'fixed_six_ear_crop_epoch.csv')
        latest=fixed[fixed.epoch==10].copy()
        save(latest,summary/'fixed_six_ear_e10.csv')
    if grad:save(pd.concat(grad,ignore_index=True),summary/'baseline_gradient_target_other5_nonear.csv')
    if val.empty:
        print('[NOTE] No official validation results: only fixed Crop probe reports available');return
    val10=val[val.epoch==10].set_index('variant')
    metrics=[]
    for var in val10.index:
        r=meta[var]
        for name in EAR:
            metrics.append({'variant':var,'target_for_run':r['target'],
                'positive_mode':r['mode'],'lambda_struct':r['lambda_struct'],
                'part':name,"ear_IoU_pct":val10.loc[var,f'{name}_ear_IoU_pct'],
                'ear_Acc_pct':val10.loc[var,f'{name}_ear_Acc_pct'],
                'head_IoU_pct':val10.loc[var,f'{name}_head_IoU_pct'],
                'all116_mIoU_pct':val10.loc[var,'mIoU_pct']})
    save(pd.DataFrame(metrics),summary/'e10_six_ear_official_long.csv')
    contrast=[]
    for target in EAR:
        row={'target':target}
        prefix={
          'full':'full_l4','drop_target':f'loo_{target}_l4',
          'target_plus_non_ear':f'target_nonear_{target}_l4',
          'non_ear':'non_ear_l4','other_five_ears':f'other5_{target}_l4',
          'only_target':f'only_{target}_l4','all_ears':'all_ears_l4',
        }
        for mode,var in prefix.items():
            row[mode+'_IoU_pct']=float(val10.loc[var,f'{target}_ear_IoU_pct']) if var in val10.index else float('nan')
            row[mode+'_Acc_pct']=float(val10.loc[var,f'{target}_ear_Acc_pct']) if var in val10.index else float('nan')
            row[mode+'_Head_IoU_pct']=float(val10.loc[var,f'{target}_head_IoU_pct']) if var in val10.index else float('nan')
        f=row['full_IoU_pct']; d=row['drop_target_IoU_pct']; t=row['target_plus_non_ear_IoU_pct']; n=row['non_ear_IoU_pct']
        row['own_positive_marginal_pp']=f-d
        row['other_five_marginal_when_target_on_pp']=f-t
        row['other_five_marginal_when_target_off_pp']=d-n
        row['own_positive_marginal_other_five_off_pp']=t-n
        row['interaction_difference_in_differences_pp']=f-d-t+n
        contrast.append(row)
    df=pd.DataFrame(contrast)
    save(df,summary/'six_target_factorial_l4.csv')
    cow=[]
    for lam in (0,4):
        for mode,prefix in [('full','full'),('drop_target','loo_cow'),('target_plus_non_ear','target_nonear_cow'),
            ('other_five_ears','other5_cow'),('only_target','only_cow'),('non_ear','non_ear'),('all_ears','all_ears')]:
            key=f'{prefix}_l{lam}'
            if key not in val10.index:continue
            cow.append({'lambda_struct':lam,'mode':mode,'variant':key,
                'cow_ear_IoU_pct':val10.loc[key,'cow_ear_IoU_pct'],
                'cow_ear_Acc_pct':val10.loc[key,'cow_ear_Acc_pct'],
                'cow_head_IoU_pct':val10.loc[key,'cow_head_IoU_pct'],
                'all116_mIoU_pct':val10.loc[key,'mIoU_pct']})
    save(pd.DataFrame(cow),summary/'cow_full_factorial_lambda0_lambda4.csv')
    # Pair official IoU and fixed probe metrics for each same Ear target.
    if isinstance(fixed,pd.DataFrame):
        f0=fixed[fixed.epoch==10][['variant','part','gt_positive_fraction','mean_gt_margin','weighted_gt_purity','weighted_ignore_mass','other_gt_fg_positive_fraction']].copy()
        f0['part']=f0.part.str.replace("'s ear",'',regex=False)
        v0=pd.DataFrame(metrics)
        j=v0.merge(f0,on=['variant','part'],how='left')
        save(j,summary/'e10_fixed_probe_vs_val_IoU.csv')
    readme=['# Ear transfer mechanism report (actual measurements only)','',
        'The authoritative measure of successful segmentation is **official validation Ear IoU**, not fixed-probe GT Margin coverage.','',
        'All training uses frozen visual cache, Presence candidates, entire 116-way InfoNCE denominator; GT only offline.','',
        '## Six-Ear lambda4 leave-one-out contrasts (percentage points)','',
        df.round(3).to_string(index=False),'',
        'Interaction = Full - DropTarget - TargetPlusNonEar + NonEarOnly. This is a **behavioral/nonlinear interaction**, NOT an additive decomposition of W updates.','',
        '## Cow lambda0 and lambda4','',pd.DataFrame(cow).round(3).to_string(index=False),
        '',f'Valid official evaluation rows: {len(valrows)}',
        '',f'Missing/unparsed official logs: {len(missing)}','',
        'Missing entries are NaN, never estimated. Single seed=123 and six fixed audit crops limit generalization.']
    (summary/'results_explained.md').write_text('\n'.join(readme),encoding='utf8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5))
        names=df.target.tolist();x=range(6)
        for col,label in [('full_IoU_pct','Full'),('drop_target_IoU_pct','Leave target out'),('target_plus_non_ear_IoU_pct','Target + nonEar'),('non_ear_IoU_pct','NonEar only')]:
            ax.plot(x,df[col].values,marker='o',label=label)
        ax.set_xticks(list(x),names);ax.set_ylabel('Official validation Ear IoU (%)');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(summary/'six_ear_official_loo_compare.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print('Wrote:',summary)
    print('Official validation rows:',len(valrows),' | missing/unparsed:',len(missing))

if __name__=='__main__':main()
