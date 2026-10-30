#!/usr/bin/env python3
"""Optional final charts from offline official metrics/probe results only."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path

def read_csv(p):
    if not p.is_file():return []
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    a=parser.parse_args();summary=a.report/'summary';out=summary/'plots';out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ranked=read_csv(summary/'ranked_final.csv')
    if ranked:
        s=sorted(ranked,key=lambda r:float(r['final_mIoU_pct']))
        fig,ax=plt.subplots(figsize=(11,max(5,len(s)*0.42)))
        ax.barh([r['variant'] for r in s],[float(r['final_mIoU_pct']) for r in s])
        ax.set_xlabel('PP116 official mIoU (%)');ax.set_title('Hard Negative variants: final epoch')
        ax.grid(axis='x',alpha=.25);fig.tight_layout();fig.savefig(out/'final_miou_all_variants.png',dpi=180);plt.close(fig)
    all_ep=read_csv(summary/'variant_scorecard.csv')
    if all_ep:
        by=defaultdict(list)
        for r in all_ep:by[r['variant']].append((int(r['epoch']),float(r['all116_mIoU_pct'])))
        fig,ax=plt.subplots(figsize=(10,6))
        focus=['baseline_116','hard_k04','hard_k08','hard_k12','hard_k16','hard_k24','hard_k32','hard_k48','hard_k64','hard_k96','hard_k115']
        for label in focus:
            if label in by:
                xy=sorted(by[label]);ax.plot([a for a,b in xy],[b for a,b in xy],marker='o',label=label)
        ax.set_xlabel('Epoch');ax.set_ylabel('PP116 mIoU (%)');ax.set_title('Negative-count sweep')
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'negative_count_epoch_curves.png',dpi=180);plt.close(fig)
    stable=read_csv(summary/'fixed_probe_group_competition_stability.csv')
    if stable:
        group=defaultdict(list)
        for r in stable:
            if r['variant'] in ('baseline_116','hard_k16','static_text_k16','random_k16'):
                group[(r['variant'],r['group'])].append((int(r['epoch_to']),float(r['top1_switch_fraction'])))
        fig,ax=plt.subplots(figsize=(10,6))
        for (variant,g),xy in sorted(group.items()):
            if g in ('B','C'):
                z=sorted(xy);ax.plot([x for x,y in z],[y for x,y in z],marker='o',label=f'{variant}/{g}')
        ax.set_xlabel('Epoch');ax.set_ylabel('Fraction of fixed Crop–Part probes whose Top1 rival changed')
        ax.set_title('B/C: negative-competition stability');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/'BC_top1_rival_switch.png',dpi=180);plt.close(fig)
    print('Optional charts:',len(list(out.glob('*.png'))),'at',out)

if __name__=='__main__':main()
