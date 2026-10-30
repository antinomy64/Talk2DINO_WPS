#!/usr/bin/env python3
"""Parse authoritative 850-image official val results and ABCD consequences."""
from __future__ import annotations
import argparse,csv,json,math,re,zipfile,subprocess,sys
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
    by={(r['epoch'],r['part']):r['IoU_pct'] for r in rows if r['variant']=='baseline_116'}
    changes=[]
    for r in rows:
        if r['variant']=='baseline_116':continue
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
        baserows=final_by.get(('baseline_116',ep),[])
        basemiou=baserows[0]['mIoU_pct'] if baserows else float('nan')
        outcomes.append({'variant':variant,'epoch':ep,'all116_mIoU_pct':pr[0]['mIoU_pct'],
                         'delta_mIoU_pp':round(pr[0]['mIoU_pct']-basemiou,6) if baserows else '',
                         **{f'group_{g}_mean_IoU_pct':round(sum(bygrp[g])/len(bygrp[g]),6) if g in bygrp else ''
                            for g in ('A','B','C','D')},'mean_116_perclass_IoU_pct':round(sum(x['IoU_pct'] for x in pr)/len(pr),6)})
    write(summary/'variant_scorecard.csv',sorted(outcomes,key=lambda x:(x['epoch'],x['variant'])))
    # Comprehensive, FULLY GT-free negative-competition probe analysis.
    # Probe snapshots use a fixed training-Presence sample, NEVER pixel masks.
    allprobes=[]
    for v in reg:
        var=v['variant']
        for pf in sorted((Path(v['train_dir'])/'hardest_probe').glob('probe_epoch_*.csv')):
            with pf.open(newline='',encoding='utf-8') as f:
                for rec in csv.DictReader(f):
                    allprobes.append({'variant':var,**rec,
                                      'group':classes.get(rec['part'],'UNLABELED')})
    write(summary/'fixed_probe_per_part_epochs.csv',allprobes)
    switches=[]
    by_var_pid=defaultdict(list)
    for r in allprobes:
        by_var_pid[(r['variant'],r['pid'])].append(r)
    for (variant,pid), series in sorted(by_var_pid.items()):
        series=sorted(series,key=lambda x:int(x['epoch']))
        for previous,current in zip(series,series[1:]):
            a=json.loads(previous['top16_id_counts_json'])
            b=json.loads(current['top16_id_counts_json'])
            keys=set(a)|set(b)
            overlap=sum(min(int(a.get(k,0)),int(b.get(k,0))) for k in keys)
            denom=sum(max(int(a.get(k,0)),int(b.get(k,0))) for k in keys)
            switches.append({'variant':variant,'pid':pid,'part':current['part'],
                'group':current['group'],'epoch_from':previous['epoch'],'epoch_to':current['epoch'],
                'aggregate_top16_histogram_overlap':overlap/denom if denom else '',
                'modal_top1_changed':int(previous['modal_top1_pid']!=current['modal_top1_pid']),
                'delta_mean_full116_p_positive':float(current['mean_full116_p_positive'])-float(previous['mean_full116_p_positive']),
                'delta_positive_minus_top1_logit':float(current['mean_positive_minus_top1_logit'])-float(previous['mean_positive_minus_top1_logit']),
                'delta_top16_same_object_fraction':float(current['top16_same_object_fraction'])-float(previous['top16_same_object_fraction'])})
    write(summary/'fixed_probe_negative_changes.csv',switches)

    # Same-instance Top16 Jaccard across checkpoints, not merely changes in a
    # per-class modal negative. This supports exact competition-switch tracing.
    instance_rows=[]
    for v in reg:
        for ipath in sorted((Path(v['train_dir'])/'hardest_probe').glob('probe_instances_epoch_*.csv')):
            with ipath.open(newline='',encoding='utf8') as f:
                for r in csv.DictReader(f):
                    instance_rows.append({'variant':v['variant'],**r,
                                          'group':classes.get(r['part'],'UNLABELED')})
    write(summary/'fixed_probe_crop_part_epochs.csv',instance_rows)
    inst_by_key=defaultdict(list)
    for r in instance_rows:inst_by_key[(r['variant'],r['crop_index'],r['pid'])].append(r)
    fixed_changes=[]
    for key,seq in inst_by_key.items():
        seq=sorted(seq,key=lambda x:int(x['epoch']))
        for a,b in zip(seq,seq[1:]):
            old=set(json.loads(a['global_top16_ids']))
            new=set(json.loads(b['global_top16_ids']))
            union=old|new
            fixed_changes.append({'variant':b['variant'],
                'crop_index':b['crop_index'],'pid':b['pid'],'part':b['part'],'group':b['group'],
                'epoch_from':a['epoch'],'epoch_to':b['epoch'],
                'top16_jaccard':len(old&new)/len(union) if union else '',
                'top1_switched':int(a['global_top1_pid']!=b['global_top1_pid']),
                'top1_before':a['global_top1_name'],'top1_after':b['global_top1_name'],
                'delta_p_full116':float(b['full116_p_positive'])-float(a['full116_p_positive']),
                'delta_positive_top1_logit_gap':float(b['positive_minus_top1_logit'])-float(a['positive_minus_top1_logit']),
                'delta_effective_count':(float(b['effective_patch_count'])-float(a['effective_patch_count'])
                                         if a['effective_patch_count'] and b['effective_patch_count'] else ''),
                'fallback_from':a['fallback'],'fallback_to':b['fallback']})
    write(summary/'fixed_probe_same_crop_competition_changes.csv',fixed_changes)
    transition_summary=[]
    grouped_changes=defaultdict(list)
    for r in fixed_changes:grouped_changes[(r['variant'],r['epoch_from'],r['epoch_to'],r['group'])].append(r)
    for (variant,ep0,ep1,grp),rlist in sorted(grouped_changes.items()):
        transition_summary.append({'variant':variant,'epoch_from':ep0,'epoch_to':ep1,
            'group':grp,'n_probe_part_instances':len(rlist),
            'mean_top16_jaccard':sum(r['top16_jaccard'] for r in rlist)/len(rlist),
            'top1_switch_fraction':sum(r['top1_switched'] for r in rlist)/len(rlist),
            'mean_delta_positive_vs_top1_gap':sum(r['delta_positive_top1_logit_gap'] for r in rlist)/len(rlist),
            'mean_delta_full116_p_positive':sum(r['delta_p_full116'] for r in rlist)/len(rlist)})
    write(summary/'fixed_probe_group_competition_stability.csv',transition_summary)

    # Final-epoch ranked table, full range and area under *measured* mIoU curve.
    finalists=[]; curves=defaultdict(list)
    for o in outcomes:curves[o['variant']].append(o)
    for var,series in curves.items():
        seq=sorted(series,key=lambda x:int(x['epoch']))
        final=seq[-1]
        aum=0.
        for left,right in zip(seq,seq[1:]):
            aum+=(int(right['epoch'])-int(left['epoch']))*(float(left['all116_mIoU_pct'])+float(right['all116_mIoU_pct']))/2
        denom=int(seq[-1]['epoch'])-int(seq[0]['epoch'])
        area=aum/denom if denom>0 else float(seq[-1]['all116_mIoU_pct'])
        finalists.append({'variant':var,'epoch':final['epoch'],
            'final_mIoU_pct':final['all116_mIoU_pct'],
            'delta_mIoU_pp':final['delta_mIoU_pp'],
            'measured_epoch_trapezoid_avg_mIoU_pct':round(area,5),
            **{f'group_{g}_mean_IoU_pct':final[f'group_{g}_mean_IoU_pct'] for g in ('A','B','C','D')},
            'num_official_eval_epochs':len(seq)})
    finalists.sort(key=lambda x:-float(x['final_mIoU_pct']))
    write(summary/'ranked_final.csv',finalists)

    # FP/GT and pred/GT are derived only when official per-class recall is valid.
    # This does not provide pairwise pixel-level confusion.
    p_rec=[]
    for r in rows:
        I=r['IoU_pct']/100.;R=r['Acc_pct']/100.
        p=r['Precision_inferred_pct']
        pred_to_gt=(R/(float(p)/100.)) if p!='' and float(p)>0 else ''
        p_rec.append({'variant':r['variant'],'epoch':r['epoch'],'part':r['part'],
                     'group':r['group'],'IoU_pct':r['IoU_pct'],'recall_pct':r['Acc_pct'],
                     'precision_derived_pct':p,'predicted_over_GT_area_ratio':pred_to_gt})
    write(summary/'perpart_precision_recall_area.csv',p_rec)

    # Negative trace + outcome linkage; validation never touches optimization.
    perf={(r['variant'],int(r['epoch']),r['part']):r for r in rows}
    joined=[]
    for r in allprobes:
        corresponding=perf.get((r['variant'],int(r['epoch']),r['part']))
        if corresponding:
            joined.append({'variant':r['variant'],'epoch':r['epoch'],'part':r['part'],
                'group':r['group'],'n_probe':r['n_probe'],
                'mean_positive_minus_top1_logit':r['mean_positive_minus_top1_logit'],
                'mean_full116_p_positive':r['mean_full116_p_positive'],
                'top16_same_object_fraction':r['top16_same_object_fraction'],
                'top16_same_suffix_fraction':r['top16_same_suffix_fraction'],
                'modal_top1_name':r['modal_top1_name'],
                'official_IoU_pct':corresponding['IoU_pct'],
                'official_Recall_pct':corresponding['Acc_pct'],
                'official_Precision_derived_pct':corresponding['Precision_inferred_pct']})
    write(summary/'probe_vs_official_perpart.csv',joined)

    # Auxiliary Markdown factual summary for upload: no causal claims.
    lines=['# Hardest Negative Search: official evaluation summary','',
        'All results are official 850-image 116-way PP116 metrics.',
        'Probe metrics are computed from a fixed, GT-free train-Crop subset; they are not GT purity.',
        'All experiments are same-seed, and are NOT a multi-seed significance test.','',
        '| Rank | Variant | Efinal mIoU | Delta vs matched baseline | A | B | C | D |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for i,f in enumerate(finalists,1):
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(i,f['variant'],f['final_mIoU_pct'],f['delta_mIoU_pp'],*(f[f'group_{g}_mean_IoU_pct'] for g in ('A','B','C','D'))))
    (summary/'auto_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    # Figures are diagnostic extras, never required for success of training/eval.
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent/'plot_hardest.py'),
                        '--report', str(report)],check=True)
    except Exception as e:
        (summary/'optional_plot_error.txt').write_text(str(e),encoding='utf8')
    # Bundle only lightweight logs/CSVs/manifests; large checkpoints never included.
    dest=report/'review_upload.zip'
    allowed={'.csv','.json','.log','.txt','.md','.png'}
    tozip=[p for p in report.rglob('*') if p.is_file() and p!=dest and p.suffix.lower() in allowed
           and p.stat().st_size<25*1024*1024]
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in tozip:
            z.write(path,arcname=str(path.relative_to(report)))
        z.write(Path(__file__).resolve().parent/'train_pp116_hardest.py',arcname='scripts/train_pp116_hardest.py')
        z.write(Path(__file__).resolve().parent/'run_hardest.py',arcname='scripts/run_hardest.py')
        z.write(Path(__file__).resolve().parent/'summarize_hardest.py',arcname='scripts/summarize_hardest.py')
        z.write(Path(__file__).resolve().parent/'plot_hardest.py',arcname='scripts/plot_hardest.py')
    print('OFFICIAL EVALUATIONS:',len(final_by),'FIXED PROBE PART EPOCHS:',len(allprobes))
    print('ROWS:',len(rows),'OUTPUT:',dest,'SIZE_MiB:',round(dest.stat().st_size/(1024**2),2))

if __name__=='__main__':main()
