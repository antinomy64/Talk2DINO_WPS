#!/usr/bin/env python3
"""Summarize actual night results; NEVER fill in missing metrics."""
import argparse
import csv
import re
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    base=a.report/'gradient'
    output=a.report/'summary'
    output.mkdir(parents=True,exist_ok=True)
    files=sorted(base.glob('*/six_ear_margin_direction.csv'))
    if not files:
        print('No fixed-probe results yet:',base)
        return
    data=[];grad=[]
    for f in files:
        name=f.parent.name
        d=pd.read_csv(f);d.insert(0,'arm',name);data.append(d)
        g=f.parent/'gradient_components.csv'
        if g.exists():
            x=pd.read_csv(g);x.insert(0,'arm',name);grad.append(x)
    all_m=pd.concat(data,ignore_index=True)
    all_m.to_csv(output/'all_arms_six_ear_margin.csv',index=False)
    if grad:
        all_g=pd.concat(grad,ignore_index=True)
        all_g.to_csv(output/'all_arms_gradient_components.csv',index=False)
    ep0=all_m[all_m.epoch==0]
    endpoints=all_m[all_m.epoch==10].copy()
    if len(endpoints):
        init=ep0[['arm','part','mean_gt_margin','gt_positive_fraction']].rename(columns={
            'mean_gt_margin':'e0_margin','gt_positive_fraction':'e0_coverage'})
        e=endpoints[['arm','part','mean_gt_margin','gt_positive_fraction']].rename(columns={
            'mean_gt_margin':'e10_margin','gt_positive_fraction':'e10_coverage'})
        summary=e.merge(init,on=['arm','part'],how='left',validate='one_to_one')
        summary['gain_margin']=summary.e10_margin-summary.e0_margin
        summary['gain_coverage_pp']=100*(summary.e10_coverage-summary.e0_coverage)
        summary.to_csv(output/'six_ear_e0_e10_comparison.csv',index=False)
        pivot=summary.pivot(index='part',columns='arm',values='e10_coverage')
        pivot.to_csv(output/'e10_coverage_matrix.csv')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(11,5.5))
            for part,d in all_m.groupby('part'):
                dd=d[d.arm=='baseline_struct'].sort_values('epoch')
                if dd.empty:continue
                ax.plot(dd.epoch,100*dd.gt_positive_fraction,marker='o',label=part)
            ax.set(xlabel='Epoch',ylabel='GT Ear foreground patches with positive margin (%)',title='Baseline six fixed Ear crops, E0–E10')
            ax.legend(fontsize=8,loc='best');ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(output/'baseline_ear_coverage.png',dpi=160);plt.close(fig)
            # Variant effects per ear; no hidden smoothing.
            fig,ax=plt.subplots(figsize=(12,5.5))
            pivot.plot(kind='bar',ax=ax)
            ax.set(xlabel='Ear part',ylabel='E10 GT Ear positive-margin fraction',title='Controlled ablations, fixed six Crop probes')
            ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(output/'e10_ablation_coverage.png',dpi=160);plt.close(fig)
        except ImportError:
            print('matplotlib unavailable; tables still written')
    # Keep original logs as authoritative rather than inventing parser entries.
    log_lines=[]
    for logfile in sorted(a.report.glob('val/*/e*/evaluation_stdout.log')):
        lines=logfile.read_text(errors='replace').splitlines()
        for i,l in enumerate(lines):
            if any(x in l.lower() for x in ('mIoU'.lower(), "cat's ear", "cow's ear", "dog's ear", "horse's ear", "person's ear", "sheep's ear")):
                log_lines.append({'arm':logfile.parts[-3], 'epoch':logfile.parts[-2],
                                  'source':str(logfile),'line_num':i+1,'raw_line':l.strip()})
    if log_lines:
        with (output/'eval_lines_unparsed.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(log_lines[0]));writer.writeheader();writer.writerows(log_lines)
    overview=[]
    for mode,df in all_m.groupby('arm'):
        for ep,g in df.groupby('epoch'):
            overview.append({'arm':mode,'epoch':ep,
                             'mean_gt_margin_over_six_crops':g.mean_gt_margin.mean(),
                             'mean_gt_coverage_over_six_crops':g.gt_positive_fraction.mean(),
                             'num_ear_crops':len(g)})
    pd.DataFrame(overview).to_csv(output/'mean_six_crops_by_arm_epoch.csv',index=False)
    print('Wrote:',output)
    print('NOTE: all numbers from fixed six GT-audit crops, NOT all validation Ear mIoU.')
    print('Read eval_lines_unparsed.csv/evaluation_stdout.log for official pixel IoU.')

if __name__=='__main__':main()
