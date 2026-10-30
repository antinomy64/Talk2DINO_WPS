#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate completed all-part audit SQLite into analysis-ready CSVs and figures.

Metrics explicitly separate count/patch/crop averages and GT-observed cases.
ABCD classifications are loaded from historical reference, NEVER recomputed.
"""
from __future__ import annotations
import argparse,csv,json,math,sqlite3,statistics
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

NUM_FIELDS=['fg_recall','selected_gt_fraction','selected_valid_precision','q_gt','q_other','q_ignore','q_gt_valid',
 'fallback','gt_margin_positive','other_margin_positive','mean_margin_gt','mean_margin_other','mean_own_cos_gt',
 'mean_rival_cos_gt','max_weight','effective_support','entropy','normalized_entropy','pseudo_oracle_cos',
 'pseudo_text_cos','global_nce_p_true','global_nce_rank','global_nce_loss','struct_loss_crop','struct_triplets','structure_rank_satisfaction']
RELIABILITY_FIELDS=['global_nce_p_true','max_weight','normalized_entropy']
TRANS_FIELDS=['gained_selected_gt','lost_selected_gt','gained_selected_wrong','lost_selected_wrong',
 'support_jaccard','weight_total_variation','delta_q_gt','delta_nce_loss','gained_positive_gt',
 'lost_positive_gt','competitor_changed_gt','delta_gt_margin','delta_gt_own','delta_gt_rival',
 'gain_own_up_rival_down','gain_own_rises_faster','gain_rival_falls_faster']

def write_csv(path,records,fields=None):
    rows=list(records);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=fields or list(rows[0].keys())
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def aggregate(rows,keys,fields,kind='normal'):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(x) for x in keys)].append(r)
    ret=[]
    for key,rs in sorted(groups.items(),key=lambda k:str(k[0])):
        row=dict(zip(keys,key));row['n_rows']=len(rs)
        if kind=='normal':
            row['n_gt_observed_crops']=sum(int(r.get('n_gt',0)>0) for r in rs)
            row['n_gt_in_fg_crops']=sum(int(r.get('n_gt_fg',0)>0) for r in rs)
            row['gt_fg_total']=sum(int(r.get('n_gt_fg',0)) for r in rs)
            row['gt_total']=sum(int(r.get('n_gt',0)) for r in rs)
            row['selected_true_total']=sum(int(r.get('selected_gt',0)) for r in rs)
            row['selected_other_total']=sum(int(r.get('selected_other',0)) for r in rs)
            row['selected_ignore_total']=sum(int(r.get('selected_ignore',0)) for r in rs)
            row['positive_gt_total']=sum(int(r['pos_on_gt_count'] or 0) for r in rs)
            row['positive_other_total']=sum(int(r['pos_on_other_count'] or 0) for r in rs)
            row['weighted_q_true_total']=sum(float(r['q_gt']) for r in rs)
            row['weighted_q_other_total']=sum(float(r['q_other']) for r in rs)
            row['weighted_q_ignore_total']=sum(float(r['q_ignore']) for r in rs)
            row['gt_purity_weighted_patch_exposures']=row['weighted_q_true_total']/max(1,row['n_rows'])
            row['gt_fg_recall_micro']=row['gt_fg_total']/row['gt_total'] if row['gt_total'] else None
            row['selected_precision_valid_micro']=row['selected_true_total']/(row['selected_true_total']+row['selected_other_total']) if row['selected_true_total']+row['selected_other_total'] else None
            row['high_conf_wrong_rate_gt_observed']=sum(int(r.get('high_conf_wrong',0)) for r in rs if r['n_gt']>0)/max(1,row['n_gt_observed_crops'])
        for field in fields:
            v=[float(r[field]) for r in rs if r.get(field) is not None and isinstance(r[field],(int,float)) and math.isfinite(float(r[field]))]
            row['mean_'+field]=float(np.mean(v)) if v else None
            row['median_'+field]=float(np.median(v)) if v else None
            row['n_'+field]=len(v)
        ret.append(row)
    return ret

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('output/pp116_allpart_audit_ours_v1'))
    p.add_argument('--groups',type=Path,default=Path(__file__).resolve().parent/'pp116_four_groups_per_part.csv')
    p.add_argument('--write-crop-rows',action='store_true',help='Optional huge per_crop_part_epoch.csv.gz; not for upload')
    p.add_argument('--plot',action='store_true')
    return p.parse_args()

def main():
    args=parse_args()
    con=sqlite3.connect(str(args.out/'audit.sqlite'))
    res=args.out/'summary';res.mkdir(parents=True,exist_ok=True)
    entry=con.execute('SELECT value FROM metadata WHERE key=?',('config',)).fetchone()
    if entry is None:raise RuntimeError('Missing audit config in SQLite')
    config=json.loads(entry[0]);epochs=config['epochs']
    n_done=dict(con.execute('SELECT epoch, COUNT(*) FROM epoch_done GROUP BY epoch'))
    for ep in epochs:
        if n_done.get(ep,0)!=config['n_crops']:
            raise RuntimeError(f'Incomplete E{ep}: {n_done.get(ep,0)}/{config["n_crops"]}. Resume the audit first.')
    from audit_all_116 import load_names
    names,classes=load_names(args.groups,Path(config['bank']))
    part_metrics=[];n_partrows=0
    import gzip
    bigfile=None;bigwriter=None
    if args.write_crop_rows:
        bigfile=gzip.open(res/'per_crop_part_epoch.csv.gz','wt',encoding='utf-8',newline='',compresslevel=2)
    try:
        for pid in range(116):
            for ep in epochs:
                rs=[]
                for (payload,) in con.execute('SELECT payload FROM part_epoch WHERE pid=? AND epoch=?',(pid,ep)):
                    row=json.loads(payload);rs.append(row);n_partrows+=1
                    if bigfile:
                        plain={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
                        if bigwriter is None:
                            bigwriter=csv.DictWriter(bigfile,fieldnames=list(plain));bigwriter.writeheader()
                        bigwriter.writerow(plain)
                if not rs:
                    # A part can have no image-level presence in this cache.
                    part_metrics.append(dict(pid=pid,part=names[pid],group=classes[pid]['类别编码'] or 'NO_VALID_GT',
                       object_name=names[pid].split("'s ",1)[0],epoch=ep,n_rows=0,
                       n_gt_observed_crops=0,n_gt_in_fg_crops=0,reference_val_IoU=(
                       float(classes[pid][f'E{ep:03d}']) if f'E{ep:03d}' in classes[pid]
                       and classes[pid].get(f'E{ep:03d}') and str(classes[pid]['GT_valid']).upper()=='TRUE' else None)))
                    continue
                got=aggregate(rs,('pid','part','group','object_name','epoch'),NUM_FIELDS)
                if len(got)!=1:raise RuntimeError('Bad aggregate uniqueness')
                record=got[0];cc=classes[pid];col=f'E{ep:03d}'
                record['reference_val_IoU']=float(cc[col]) if col in cc and cc[col] and str(cc['GT_valid']).upper()=='TRUE' else None
                record['reference_val_IoU_source']='pp116_four_groups_per_part.csv: independent reference validation; verify checkpoint identity'
                record['historical_A_B_C_D']=cc['类别编码'] or 'NO_VALID_GT'
                record['historical_reference_peak_IoU']=cc['peak_IoU']
                part_metrics.append(record)
            if pid%20==0:print(f'[summary] per-part processed {pid+1}/116',flush=True)
    finally:
        if bigfile:bigfile.close()
    write_csv(res/'per_part_epoch_metrics.csv',part_metrics)
    # Entire-object GT coverage, including annotated parts NOT present in image-level labels.
    presence_coverage=[]
    for pid in range(116):
        data=con.execute('SELECT COUNT(*),SUM(gt_patches),SUM(gt_fg_patches),SUM(in_presence) FROM gt_presence_coverage WHERE pid=?',(pid,)).fetchone()
        nc=int(data[0] or 0);tot=int(data[1] or 0);inside=int(data[2] or 0);present=int(data[3] or 0)
        presence_coverage.append(dict(pid=pid,part=names[pid],group=classes[pid]['类别编码'] or 'NO_VALID_GT',
           n_gt_observed_crop_instances=nc,n_gt_patches=tot,n_gt_fg_patches=inside,
           n_gt_crop_present_in_imagelevel_labels=present,
           n_gt_crop_missing_presence=nc-present,
           gt_crop_presence_rate=present/nc if nc else None,
           gt_patch_foreground_recall=inside/tot if tot else None))
    write_csv(res/'part_presence_and_foreground_coverage.csv',presence_coverage)
    bygroup=[]
    for grp in ('A','B','C','D','NO_VALID_GT'):
        for ep in epochs:
            prs=[r for r in part_metrics if ((r.get('historical_A_B_C_D') or r.get('group'))==grp and r['epoch']==ep)]
            row=dict(group=grp,epoch=ep,n_parts=len(prs),n_part_occurrences=sum(r.get('n_rows',0) for r in prs))
            for metric in ('q_gt','q_ignore','q_gt_valid','gt_margin_positive','mean_margin_gt','fallback',
                           'pseudo_oracle_cos','global_nce_p_true','effective_support','global_nce_loss','struct_loss_crop','structure_rank_satisfaction','reference_val_IoU'):
                key='mean_'+metric
                vals=[float(r[key]) for r in prs if r.get(key) is not None]
                row['part_macro_'+metric]=float(np.mean(vals)) if vals else None
            bygroup.append(row)
    write_csv(res/'ABCD_epoch_macro.csv',bygroup)
    # Scan all rows ONE more time for rival graph, label-free confidence proxy bins,
    # and deterministic best/worst exemplars. Memory O(116*epochs*G*features), not O(all rows).
    edges=defaultdict(Counter)
    bins=defaultdict(lambda:dict(n=0,proxy_sum=0.,gt_sum=0.,ignore_sum=0.,valid_sum=0.,valid_n=0))
    chosen_best={};chosen_worst={};maxep=max(epochs)
    for (payload,) in con.execute('SELECT payload FROM part_epoch ORDER BY epoch,pid,crop_index'):
        r=json.loads(payload);ep=r['epoch'];pid=r['pid'];group=classes[pid]['类别编码'] or 'NO_VALID_GT'
        for rival,count in r.get('rival_gt_hist',{}).items():
            edges[(ep,pid,'GT_target')][int(rival)]+=int(count)
        for rival,count in r.get('rival_other_hist',{}).items():
            edges[(ep,pid,'GT_other')][int(rival)]+=int(count)
        if r['n_gt']>0:
            for feature in RELIABILITY_FIELDS:
                val=r.get(feature)
                if val is None or not math.isfinite(float(val)):continue
                b=max(0,min(9,int(float(val)*10)))
                for gr in (group,'ALL'):
                    st=bins[(ep,gr,feature,b)];st['n']+=1;st['proxy_sum']+=float(val)
                    st['gt_sum']+=float(r['q_gt']);st['ignore_sum']+=float(r['q_ignore'])
                    if r['q_gt_valid'] is not None:
                        st['valid_sum']+=float(r['q_gt_valid']);st['valid_n']+=1
        if ep==maxep and r['n_gt_fg']>=3:
            b=chosen_best.get(pid);w=chosen_worst.get(pid)
            if b is None or (r['q_gt'],r['n_gt_fg'])>(b['q_gt'],b['n_gt_fg']):chosen_best[pid]=r
            if w is None or (r['q_gt'],-r['n_gt_fg'])<(w['q_gt'],-w['n_gt_fg']):chosen_worst[pid]=r
    edge_rows=[]
    for (ep,pid,region),hist in sorted(edges.items()):
        total=sum(hist.values())
        for rank,(rid,count) in enumerate(hist.most_common(),1):
            edge_rows.append(dict(epoch=ep,pid=pid,part=names[pid],group=classes[pid]['类别编码'],
                region=region,rival_pid=rid,rival_part=names[rid],rival_count=count,
                source_region_total=total,rival_share=count/total if total else None,rank=rank))
    write_csv(res/'part_competitor_edges.csv',edge_rows)
    reliability=[]
    for (ep,group,feature,b),st in sorted(bins.items()):
        n=st['n'];reliability.append(dict(epoch=ep,group=group,confidence_proxy=feature,
          bin_low=b/10,bin_high=(b+1)/10,n_observed_gt_crops=n,
          mean_proxy=st['proxy_sum']/n,mean_gt_weight_mass=st['gt_sum']/n,
          mean_ignore_mass=st['ignore_sum']/n,
          mean_valid_conditional_purity=st['valid_sum']/st['valid_n'] if st['valid_n'] else None))
    write_csv(res/'confidence_proxy_vs_GT_reliability.csv',reliability)
    # Transitions also streamed per part+epoch instead of retaining all rows.
    per_trans=[]
    for pid in range(116):
        for ep in epochs[1:]:
            rs=[json.loads(p) for (p,) in con.execute('SELECT payload FROM transition WHERE pid=? AND epoch=?',(pid,ep))]
            if rs:per_trans.extend(aggregate(rs,('pid','part','group','epoch'),TRANS_FIELDS,kind='transition'))
    write_csv(res/'per_part_epoch_transitions.csv',per_trans)
    endpoint_stats=[]
    endpoint_fields=['gained_selected_gt','lost_selected_gt','gained_selected_other','lost_selected_other',
        'delta_q_gt','delta_q_other','delta_q_ignore','support_jaccard','weight_total_variation',
        'gained_positive_gt','lost_positive_gt','positive_gt_start','positive_gt_end',
        'delta_gt_margin','delta_gt_own_cos','delta_gt_rival_cos','competitor_switched_on_gt',
        'gain_own_up_rival_down','gain_own_up_more','gain_rival_down_more',
        'gt_ever_positive','gt_never_positive','mean_first_positive_observed_epoch',
        'gt_lost_after_first_win','gt_positive_at_final']
    for pid in range(116):
        rs=[json.loads(p) for (p,) in con.execute('SELECT payload FROM endpoint_transition WHERE pid=?',(pid,))]
        if rs:
            a=aggregate(rs,('pid','part','group'),endpoint_fields,kind='transition')
            for row in a:
                row['n_gt_fg_patch_exposures']=sum(int(q['n_gt_fg']) for q in rs)
                for field in ('gained_selected_gt','lost_selected_gt','gained_positive_gt',
                              'lost_positive_gt','gain_own_up_rival_down','gain_own_up_more',
                              'gain_rival_down_more','gt_ever_positive','gt_never_positive',
                              'gt_lost_after_first_win','gt_positive_at_final'):
                    row['sum_'+field]=sum(int(q.get(field) or 0) for q in rs)
            endpoint_stats.extend(a)
    write_csv(res/'per_part_E0_Eend_endpoint_changes.csv',endpoint_stats)
    by={(r['pid'],r['epoch']):r for r in part_metrics}
    contrasts=[]
    for pid in range(116):
        a=by.get((pid,0));b=by.get((pid,maxep));cc=classes[pid]
        if a is None or b is None:continue
        dq=(b.get('mean_q_gt') or 0)-(a.get('mean_q_gt') or 0)
        dval=cc.get('delta_E0_E100')
        diou=float(dval) if dval and maxep==100 else None
        contrasts.append(dict(pid=pid,part=names[pid],group=cc['类别编码'] or 'NO_VALID_GT',
          n_crops=b.get('n_rows',0),GT_observed_end=b.get('n_gt_observed_crops',0),
          q0=a.get('mean_q_gt'),q_end=b.get('mean_q_gt'),delta_q=dq,q_ignore_end=b.get('mean_q_ignore'),
          gt_margin0=a.get('mean_mean_margin_gt'),gt_margin_end=b.get('mean_mean_margin_gt'),
          IoU_0=cc['E000'],IoU_end=cc.get(f'E{maxep:03d}',''),delta_IoU=diou,
          transition_notes='Q improves but IoU drops' if diou is not None and dq>0.05 and diou<0 else '',
          reference_group_criteria=cc['分类判据']))
    write_csv(res/'per_part_outcome_disagreement.csv',contrasts)
    ref=[]
    for grp in 'ABCD':
        plist=[r for r in contrasts if r['group']==grp and r['GT_observed_end']>0]
        plist.sort(key=lambda r:(-r['n_crops'],r['pid']))
        for part in plist[:6]:
            pid=part['pid']
            for role,rr in [('best_Q',chosen_best.get(pid)),('worst_Q',chosen_worst.get(pid))]:
                if rr is not None:ref.append(dict(group=grp,role=role,pid=pid,part=part['part'],
                     crop_index=rr['crop_index'],q_gt_end=rr['q_gt'],n_gt_fg=rr['n_gt_fg']))
    write_csv(res/'exemplar_crop_manifest.csv',ref)
    (res/'manifest.json').write_text(json.dumps(dict(audit_source=config,part_rows=n_partrows,
       group_sizes=dict(Counter(c['类别编码'] or 'NO_VALID_GT' for c in classes)),
       output_files=['per_part_epoch_metrics.csv','ABCD_epoch_macro.csv','per_part_epoch_transitions.csv',
       'part_competitor_edges.csv','part_presence_and_foreground_coverage.csv','confidence_proxy_vs_GT_reliability.csv',
       'per_part_outcome_disagreement.csv','per_part_E0_Eend_endpoint_changes.csv','exemplar_crop_manifest.csv'],
       restrictions=['GT used ONLY for offline audit','Ignore label 255 reported separately',
       'Crop conditional purity is not validation IoU','Partial cache coverage and absent GT are explicitly counted',
       'Reference validation IoU from user-provided historical file; verify checkpoint/run identity',
       'Structure and NCE gradients must not be inferred causally from these curves']),
       indent=2,ensure_ascii=False),encoding='utf-8')
    if args.plot:
        try:plot_summary(res,bygroup)
        except ImportError as ex:print('[WARN] plot skipped:',ex,flush=True)
    print('[DONE] summary',res.resolve(),'part_epoch_records',len(part_metrics),'patch_rows',n_partrows,flush=True)

def plot_summary(res,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field,stem,ylabel in [('part_macro_q_gt','pseudo_true_mass','Mean true patch weight mass'),
                              ('part_macro_reference_val_IoU','official_mIoU_per_part','Part IoU (%)'),
                              ('part_macro_fallback','fallback','Fallback fraction'),
                              ('part_macro_global_nce_p_true','global_nce_probability','NCE target probability')]:
        fig,ax=plt.subplots(figsize=(8,4.3))
        for grp in ('A','B','C','D'):
            sub=sorted((r for r in rows if r['group']==grp and r.get(field) is not None),key=lambda r:r['epoch'])
            if sub:ax.plot([r['epoch'] for r in sub],[r[field] for r in sub],marker='.',label=f'{grp} ({sub[0]["n_parts"]})')
        ax.set(xlabel='Epoch',ylabel=ylabel,title=stem.replace('_',' '));ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(res/f'{stem}.png',dpi=145);plt.close(fig)
if __name__=='__main__':main()
