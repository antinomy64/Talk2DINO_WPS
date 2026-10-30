#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only PP116 all-ear, all-crop, all-patch E0--E10 relative-margin audit.

Run from Talk2DINO_official_bg root. Requires repository scripts:
    train_relproto_alignemt.py, train_pp116_softsingle_dual.py,
    audit_pp116_gt_weighted_purity.py.

No labels enter score/selection. GT is opened only AFTER per-epoch weights are
computed for audit / visualization. Outputs dense, lossless float32 NPZ for each
Ear crop, compact CSV aggregates, all-FG E0/E10 patch deltas and plot examples.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import train_relproto_alignemt as src
import train_pp116_softsingle_dual as train
import audit_pp116_gt_weighted_purity as audit

N, D = 1024, 768


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--text-bank', type=Path, default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector', type=Path, default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config', type=Path, default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--classes-file', type=Path, default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--gt-root', type=Path, default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--image-root', type=Path, default=Path('data/PascalPart116/images/train'))
    p.add_argument('--run', type=Path, default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--lambda-struct', type=float, default=4.0)
    p.add_argument('--soft-temperature', type=float, default=0.05)
    p.add_argument('--start-epoch', type=int, default=0)
    p.add_argument('--end-epoch', type=int, default=10)
    p.add_argument('--max-ear-crops', type=int, default=0, help='0: all Ear crops. For smoke test e.g. 10 (per ear class).')
    p.add_argument('--plots-per-ear', type=int, default=2, help='0: disable plots. Chosen after processing all crops.')
    p.add_argument('--write-all-epoch-csv', action='store_true', help='Optional very large gzip CSV, all FG patches for every epoch; NPZ always contains these.')
    p.add_argument('--out-dir', type=Path, default=Path('output/pp116_ear_patch_margin_e0_e10'))
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--flip-mode', choices=('auto', 'none', 'all'), default='auto')
    p.add_argument('--progress-every', type=int, default=100)
    return p.parse_args()


def writer(path, fields, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = (gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=2)
         if compressed else path.open('w', newline='', encoding='utf-8'))
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    return f, w


def compute_epoch(tbank, ids, ear_pos, tokens, fg, tau, source_checked):
    local = tbank[ids]
    k = len(ids)
    # Actual trainer formula for all K part candidates in the SAME object crop.
    weights, support, fallback = audit.assignment(local, tokens, fg, tau, src.relative_scores)
    scores = local @ tokens.T
    abs_ear = scores[ear_pos]
    finite = fg
    if k >= 2:
        rel = src.relative_scores(scores[None, :, :], fg[None, :])[0][ear_pos]
        competitor_indices = [j for j in range(k) if j != ear_pos]
        rivals = scores[competitor_indices]
        rival_score, rival_loc = rivals.max(dim=0)
        top_rival = torch.as_tensor(competitor_indices, device=fg.device)[rival_loc]
        finite = fg
        # Check numerical equivalence of score definitions on valid FG patches.
        if not torch.allclose(rel[finite], (abs_ear-rival_score)[finite], atol=3e-5, rtol=1e-5):
            raise AssertionError('Relative margin differs from Ear minus strongest rival')
    else:
        rel = scores[ear_pos].masked_fill(~fg, -torch.inf)
        rival_score = torch.full_like(abs_ear, float('nan'))
        top_rival = torch.full((N,), -1, device=fg.device, dtype=torch.long)
    if not source_checked and k >= 2:
        ref, diag = train.soft_relative_prototypes(local[None], tokens[None], fg[None], temperature=tau)
        own = torch.nn.functional.normalize(weights @ tokens, dim=-1, eps=1e-12)
        if (ref[0]-own).abs().max().item() > 2e-5 or not torch.equal(fallback, diag['fallback'][0].bool()):
            raise AssertionError('Mismatch against actual trainer prototype and fallback')
    margin = rel[finite].cpu().numpy()
    if not np.isfinite(margin).all():
        raise AssertionError('Non-finite margin on FG')
    # Use nan (not -inf) outside FG for portable float32 arrays / plotting.
    return {
        'score_ear': abs_ear.cpu().numpy().astype(np.float32),
        'score_rival': rival_score.cpu().numpy().astype(np.float32),
        'rival_local': top_rival.cpu().numpy().astype(np.int16),
        'margin': rel.masked_fill(~fg, 0).cpu().numpy().astype(np.float32),
        'support': support[ear_pos].cpu().numpy().astype(np.uint8),
        'weight': weights[ear_pos].cpu().numpy().astype(np.float32),
        'fallback': int(fallback[ear_pos].item()),
    }


def save_image_grid(file, arr, image_id, crop_record, epochs, names, ear_name, pid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    e_count = len(epochs)
    fig, axes = plt.subplots(4, e_count+1, figsize=(2.15*(e_count+1), 9.3), dpi=115)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    imgroot = crop_record['image_root']
    p = next((imgroot/(image_id+s) for s in ('.jpg','.jpeg','.png','.JPG') if (imgroot/(image_id+s)).exists()), None)
    if p is not None:
        rgb = np.array(Image.open(p).convert('RGB'))
        x0,y0,x1,y1 = [int(round(float(v))) for v in audit.get_box(crop_record['ann'])]
        h,w = rgb.shape[:2]
        x0,x1 = max(0,min(w,x0)),max(0,min(w,x1))
        y0,y1 = max(0,min(h,y0)),max(0,min(h,y1))
        cropped=rgb[y0:y1,x0:x1]
        if audit.get_flip(crop_record['ann'],crop_record['flip_mode']):cropped=np.ascontiguousarray(cropped[:, ::-1])
        if cropped.size:
            axes[0,0].imshow(np.asarray(Image.fromarray(cropped).resize((32,32))))
    gt = arr['gt'].reshape(32,32)
    fg = arr['fg'].reshape(32,32).astype(bool)
    # Ref panels: RGB, GT category, PredFG
    label = np.zeros((32,32),dtype=np.uint8)
    label[gt==255]=1
    label[(gt!=255)&(gt!=pid)]=2
    label[gt==pid]=3
    axes[1,0].imshow(label,vmin=0,vmax=3,cmap=ListedColormap(['#161b25','#868686','#efa95b','#2de175']),interpolation='nearest')
    axes[2,0].imshow(fg,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    for a,lab in zip(axes[:3,0],('RGB crop','GT ear = green','Predicted FG')):a.set_title(lab, fontsize=9)
    indices=arr['candidate_part_ids'].tolist()
    legend='Competitors\n'+'\n'.join(f'{j}: {names[p].split(chr(39)+"s ")[-1]}' for j,p in enumerate(indices))
    axes[3,0].text(.02,.9,legend,transform=axes[3,0].transAxes,va='top',fontsize=7)
    gt_positive=((gt==pid)&fg)
    vmin,vmax = -.20,.20  # fixed scale across all epochs, signed raw cosine margin
    positive_weight = arr['weight'][:,arr['fg'].astype(bool)]
    mxw = max(.02, float(np.percentile(positive_weight[positive_weight>0], 98)) if (positive_weight>0).any() else .1)
    cats=ListedColormap(['#13151a','#4075b7','#35cf75','#9d7ac9','#f4ab42'])
    for t,ep in enumerate(epochs):
        ax = axes[0,t+1]
        m = arr['margin'][t].reshape(32,32).astype(float).copy()
        m[~fg] = np.nan
        ax.imshow(m,vmin=vmin,vmax=vmax,cmap='RdBu_r',interpolation='nearest')
        ax.set_title(f'E{ep:02d}\nmargin',fontsize=8)
        axes[1,t+1].imshow(arr['weight'][t].reshape(32,32),cmap='magma',vmin=0,vmax=mxw,interpolation='nearest')
        # 0 unused, 1 GT ear but missed, 2 selected GT ear, 3 selected ignore, 4 selected other annotated
        cat = np.zeros(1024,dtype=np.uint8)
        cat[(arr['gt']==pid)&arr['fg'].astype(bool)] = 1
        sel = arr['support'][t].astype(bool)
        cat[sel&(arr['gt']==pid)]=2
        cat[sel&(arr['gt']==255)]=3
        cat[sel&(arr['gt']!=255)&(arr['gt']!=pid)]=4
        axes[2,t+1].imshow(cat.reshape(32,32),cmap=cats,vmin=0,vmax=4,interpolation='nearest')
        competitor_map=arr['rival_local'][t].reshape(32,32).astype(float)
        competitor_map[~fg]=np.nan
        axes[3,t+1].imshow(competitor_map,cmap='tab20',vmin=-.5,vmax=max(1,len(indices))-.5,interpolation='nearest')
    fig.suptitle(f'{ear_name} | image={image_id} | crop={crop_record["index"]} | red margin=Ear wins; green=GT Ear selected', fontsize=11)
    axes[1,1].set_ylabel('Softmax weight', fontsize=9)
    axes[2,1].set_ylabel('Selected patch vs GT', fontsize=9)
    axes[3,1].set_ylabel('Strongest other part', fontsize=9)
    fig.tight_layout(rect=[0,0,1,.94])
    file.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(file)
    plt.close(fig)


def main():
    args=parse_args()
    if not (0<=args.start_epoch < args.end_epoch) or args.soft_temperature<=0:
        raise ValueError('Require 0 <= start-epoch < end-epoch and positive temperature')
    if args.start_epoch != 0:
        raise ValueError('This script computes E0-to-E10 style transitions; use --start-epoch 0')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    inputs=(args.cache,args.text_bank,args.projector,args.model_config,args.classes_file,args.gt_root,args.run)
    for x in inputs:
        if not x.exists():raise FileNotFoundError(x)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Will not overwrite existing audit: {args.out_dir}')
    names=audit.canonical_names(args.classes_file)
    ears=[(pid,name) for pid,name in enumerate(names) if name.endswith("'s ear")]
    if not ears:raise RuntimeError('No Ear parts in PART_CLASSES')
    print('Ear classes:', ears, flush=True)
    epochs=list(range(args.start_epoch,args.end_epoch+1))
    for ep in epochs:
        if ep>0 and not (args.run/f'W_epoch_{ep:03d}.pt').exists():
            raise FileNotFoundError(f'Missing epoch checkpoint E{ep}: {args.run/f"W_epoch_{ep:03d}.pt"}')
    device=torch.device(args.device)
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    raw,bank_names,_=src.load_raw_clip_bank(args.text_bank, expected_clip_model='ViT-B/16', expected_template='sub_imagenet_template')
    if names!=list(bank_names):raise AssertionError('Taxonomy and text bank order mismatch')
    projector,_=src.load_frozen_projector(project_root=Path.cwd(), config_path=args.model_config, weight_path=args.projector,device=device)
    with torch.no_grad():
        base=src.normalize_last(src.project_raw_clip_bank(raw,projector,device=device,batch_size=128).detach().float()).to(device)
    del raw,projector
    text={}
    for ep in epochs:
        W=audit.checkpoint_W(args.run,ep,args.lambda_struct).to(device)
        with torch.no_grad():text[ep]=src.normalize_last(base@W).detach()
    # mmap backed cache can be very large; avoid read-only copies per epoch.
    anns=audit.annotations(src.torch_load(args.cache))
    lookup=audit.GTLookup(args.gt_root)
    # identify every occurrence of each ear part from presence labels
    targets=defaultdict(list)
    for idx,a in enumerate(anns):
        ids=audit.part_ids(a)
        for pid,name in ears:
            if pid in ids and (not args.max_ear_crops or len(targets[pid])<args.max_ear_crops):
                targets[pid].append((idx,a,ids))
    print('Candidate crop counts:',{names[pid]:len(v) for pid,v in targets.items()},flush=True)
    if sum(len(v) for v in targets.values())==0:raise RuntimeError('No Ear crops found')
    args.out_dir.mkdir(parents=True)
    meta={'run':str(args.run),'epochs':epochs,'tau':args.soft_temperature,'ear_names':[n for _,n in ears],'checkpoint_source':'trained W; E0 identity','gt_role':'audit only, never score selection','patch_index':'row=patch//32,col=patch%32','margin_definition':'Ear cosine minus highest other present part cosine (K>=2)'}
    (args.out_dir/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    crop_fields=['part','part_id','image_id','crop_index','K','epoch','fg_patches','gt_ear_fg_patches',
                 'positive_margin_patches','selected_patches','selected_correct_patches','fallback',
                 'weighted_purity','weighted_ignore_mass','mean_margin_fg','mean_margin_gt_ear',
                 'max_margin_fg','gt_ear_positive_margin_frac','gt_ear_selected_frac',
                 'mean_ear_score_on_gt','mean_rival_score_on_gt','top_competitor_on_gt']
    trans_fields=['part','part_id','image_id','crop_index','K','patch_id','row','col','gt_id','gt_type','fg',
                  'margin_E0','margin_E10','delta_margin','ear_score_E0','ear_score_E10','delta_ear_score',
                  'rival_score_E0','rival_score_E10','delta_rival_score','rival_E0','rival_E10',
                  'positive_E0','positive_E10','selected_E0','selected_E10','weight_E0','weight_E10',
                  'positive_flip','selected_flip','gain_driver']
    all_fields=['part','image_id','crop_index','epoch','patch_id','row','col','gt_id','foreground',
                'score_ear','score_competitor','margin','strongest_competitor','selected','weight','fallback']
    ef,ew=writer(args.out_dir/'per_crop_epoch.csv',crop_fields)
    tf,tw=writer(args.out_dir/'per_patch_E0_E10.csv.gz',trans_fields,compressed=True)
    af=aw=None
    if args.write_all_epoch_csv:
        af,aw=writer(args.out_dir/'per_patch_all_epochs.csv.gz',all_fields,compressed=True)
    source_checked=False
    plot_candidates=defaultdict(list)
    driver_stats=defaultdict(Counter)
    done=0
    try:
        with torch.no_grad():
            for pid,name in ears:
                for idx,a,ids in targets[pid]:
                    ear_pos=ids.index(pid)
                    if len(ids)<2:
                        print(f'[K=1] {name} crop={idx}: absolute softmax, not Relative Margin',flush=True)
                    image_id=str(a['image_id'])
                    tokens=torch.as_tensor(a['cropaug_patch_tokens'],device=device,dtype=torch.float32)
                    fg=torch.as_tensor(a['pred_obj_mask_patch'],device=device,dtype=torch.bool).reshape(-1)
                    if tokens.shape!=(N,D) or fg.shape!=(N,) or not bool(fg.any()):
                        raise RuntimeError(f'Invalid patch cache for crop {idx}')
                    tokens=src.normalize_last(tokens)
                    fg_np=fg.cpu().numpy().astype(bool)
                    states=[]
                    for ep in epochs:
                        s=compute_epoch(text[ep], ids, ear_pos, tokens, fg, args.soft_temperature, source_checked)
                        if len(ids)>=2:source_checked=True
                        states.append(s)
                    # Only after all selection / weights have been computed do we read GT.
                    gt,flipped,wh,box=lookup.crop32(a,args.flip_mode)
                    if gt.shape!=(N,):raise ValueError('Bad GT projection')
                    gt_ear=(gt==pid)&fg_np
                    n_gt=int(gt_ear.sum())
                    mat={k:np.stack([s[k] for s in states]) for k in ('score_ear','score_rival','rival_local','margin','support','weight')}
                    mat['fallback']=np.asarray([s['fallback'] for s in states],dtype=np.uint8)
                    mat['fg']=fg_np.astype(np.uint8)
                    mat['gt']=gt.astype(np.int16)
                    mat['epoch']=np.asarray(epochs,dtype=np.int16)
                    mat['candidate_part_ids']=np.asarray(ids,dtype=np.int16)
                    mat['target_pid']=np.asarray(pid,dtype=np.int16)
                    mat['crop_index']=np.asarray(idx,dtype=np.int32)
                    fname=f'crop_{idx:05d}_{image_id}.npz'
                    npath=args.out_dir/'dense_npz'/name.replace("'s ",'_').replace(' ','_')/fname
                    npath.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(npath,**mat)
                    for t,ep in enumerate(epochs):
                        sel=mat['support'][t].astype(bool)
                        pos=(mat['margin'][t]>0)&fg_np if len(ids)>1 else np.zeros(N,dtype=bool)
                        w=mat['weight'][t]
                        rivals=mat['rival_local'][t]
                        r_on_gt=rivals[gt_ear]
                        r_on_gt=r_on_gt[r_on_gt>=0]
                        top_c=names[ids[Counter(map(int,r_on_gt)).most_common(1)[0][0]]] if len(r_on_gt) else ''
                        ew.writerow({'part':name,'part_id':pid,'image_id':image_id,'crop_index':idx,'K':len(ids),
                                    'epoch':ep,'fg_patches':int(fg_np.sum()),'gt_ear_fg_patches':n_gt,
                                    'positive_margin_patches':int(pos.sum()),'selected_patches':int(sel.sum()),
                                    'selected_correct_patches':int((sel&gt_ear).sum()),'fallback':int(mat['fallback'][t]),
                                    'weighted_purity':float(w[gt==pid].sum()),'weighted_ignore_mass':float(w[gt==255].sum()),
                                    'mean_margin_fg':float(mat['margin'][t][fg_np].mean()),
                                    'mean_margin_gt_ear':float(mat['margin'][t][gt_ear].mean()) if n_gt else '',
                                    'max_margin_fg':float(mat['margin'][t][fg_np].max()),
                                    'gt_ear_positive_margin_frac':float(pos[gt_ear].mean()) if n_gt else '',
                                    'gt_ear_selected_frac':float(sel[gt_ear].mean()) if n_gt else '',
                                    'mean_ear_score_on_gt':float(mat['score_ear'][t][gt_ear].mean()) if n_gt else '',
                                    'mean_rival_score_on_gt':float(mat['score_rival'][t][gt_ear].mean()) if n_gt and len(ids)>1 else '',
                                    'top_competitor_on_gt':top_c})
                        if aw:
                            for j in np.flatnonzero(fg_np):
                                rid=int(mat['rival_local'][t,j])
                                aw.writerow({'part':name,'image_id':image_id,'crop_index':idx,'epoch':ep,'patch_id':int(j),'row':int(j//32),'col':int(j%32),
                                            'gt_id':int(gt[j]),'foreground':1,'score_ear':float(mat['score_ear'][t,j]),
                                            'score_competitor':float(mat['score_rival'][t,j]),'margin':float(mat['margin'][t,j]),
                                            'strongest_competitor':names[ids[rid]] if rid>=0 else '',
                                            'selected':int(sel[j]),'weight':float(w[j]),'fallback':int(mat['fallback'][t])})
                    # All fixed foreground patches; EXACT finite difference attribution for E0/E10.
                    s0,s1=states[0],states[-1]
                    for j in np.flatnonzero(fg_np):
                        dm=float(s1['margin'][j]-s0['margin'][j]);de=float(s1['score_ear'][j]-s0['score_ear'][j])
                        dc=float(s1['score_rival'][j]-s0['score_rival'][j]) if len(ids)>1 else float('nan')
                        if len(ids)>1 and abs(dm-(de-dc))>1e-4:raise AssertionError('Margin delta decomposition failed')
                        if len(ids)==1:driver='K1_absolute_not_relative'
                        elif de>0 and dc<0:driver='ear_up_and_rival_down'
                        elif dm>0 and de>0 and dc>=0:driver='ear_up_more_than_rival'
                        elif dm>0 and de<=0 and dc<0:driver='rival_down_more_than_ear'
                        elif dm>0:driver='other_margin_gain'
                        elif dm<0:driver='margin_loss'
                        else:driver='no_change'
                        r0=int(s0['rival_local'][j]);r1=int(s1['rival_local'][j])
                        gtype='ear' if gt[j]==pid else ('ignore' if gt[j]==255 else 'other_annotated')
                        if gtype=='ear':
                            dcount=driver_stats[name]
                            dcount['ear_gt_fg_patches']+=1
                            if s0['margin'][j]>0:dcount['positive_E0']+=1
                            if s1['margin'][j]>0:dcount['positive_E10']+=1
                            if s0['support'][j]:dcount['selected_E0']+=1
                            if s1['support'][j]:dcount['selected_E10']+=1
                            if s0['margin'][j]<=0 and s1['margin'][j]>0:
                                dcount['margin_neg_to_pos']+=1
                                dcount['driver_'+driver]+=1
                                if r0 != r1: dcount['competing_part_switched']+=1
                            if s0['support'][j]==0 and s1['support'][j]==1:
                                dcount['newly_selected_gt_ear']+=1
                        tw.writerow({'part':name,'part_id':pid,'image_id':image_id,'crop_index':idx,'K':len(ids),
                                    'patch_id':int(j),'row':int(j//32),'col':int(j%32),'gt_id':int(gt[j]),'gt_type':gtype,'fg':1,
                                    'margin_E0':float(s0['margin'][j]),'margin_E10':float(s1['margin'][j]),'delta_margin':dm,
                                    'ear_score_E0':float(s0['score_ear'][j]),'ear_score_E10':float(s1['score_ear'][j]),'delta_ear_score':de,
                                    'rival_score_E0':float(s0['score_rival'][j]),'rival_score_E10':float(s1['score_rival'][j]),'delta_rival_score':dc,
                                    'rival_E0':names[ids[r0]] if r0>=0 else '', 'rival_E10':names[ids[r1]] if r1>=0 else '',
                                    'positive_E0':int(s0['margin'][j]>0) if len(ids)>1 else '',
                                    'positive_E10':int(s1['margin'][j]>0) if len(ids)>1 else '',
                                    'selected_E0':int(s0['support'][j]),'selected_E10':int(s1['support'][j]),
                                    'weight_E0':float(s0['weight'][j]),'weight_E10':float(s1['weight'][j]),
                                    'positive_flip':int(s0['margin'][j]<=0 and s1['margin'][j]>0) if len(ids)>1 else '',
                                    'selected_flip':int(s0['support'][j]==0 and s1['support'][j]==1),
                                    'gain_driver':driver})
                    sel0=s0['support'].astype(bool);sel1=s1['support'].astype(bool)
                    new_correct=int((~sel0&sel1&gt_ear).sum())
                    # Representative 1: very clear gain in true Ear selected; 2: high initial purity
                    gain=(new_correct, float(s1['weight'][gt_ear].sum()))
                    init=float(s0['weight'][gt_ear].sum())
                    plot_candidates[pid].append((gain,init,idx,image_id,npath,a))
                    done+=1
                    if args.progress_every>0 and done%args.progress_every==0:
                        print(f'[audit] {done} Ear crops finished; current {name} crop {idx}',flush=True)
    finally:
        ef.close();tf.close()
        if af:af.close()
    # Aggregate summaries across individual CROP rows (equal-crop average).
    import pandas as pd
    df=pd.read_csv(args.out_dir/'per_crop_epoch.csv')
    overview=(df.groupby(['part','epoch'],as_index=False).agg(
        n_crops=('crop_index','nunique'),
        mean_selected=('selected_patches','mean'),mean_selected_correct=('selected_correct_patches','mean'),
        any_correct_crop_rate=('selected_correct_patches',lambda x:float((x>0).mean())),
        positive_margin_crop_rate=('positive_margin_patches',lambda x:float((x>0).mean())),
        fallback_rate=('fallback','mean'),mean_purity=('weighted_purity','mean'),
        mean_gt_margin=('mean_margin_gt_ear','mean'),mean_gt_positive_fraction=('gt_ear_positive_margin_frac','mean'),
        mean_ear_score_on_gt=('mean_ear_score_on_gt','mean'),mean_rival_score_on_gt=('mean_rival_score_on_gt','mean')))
    overview.to_csv(args.out_dir/'ear_epoch_summary.csv',index=False)
    all_driver_columns=sorted({k for c in driver_stats.values() for k in c})
    with (args.out_dir/'ear_gt_patch_margin_transition_summary.csv').open('w',encoding='utf-8',newline='') as fh:
        out=csv.DictWriter(fh,fieldnames=['part']+all_driver_columns)
        out.writeheader()
        for _,name in ears:
            out.writerow({'part':name,**driver_stats[name]})
    if args.plots_per_ear>0:
        for pid,name in ears:
            candidates=plot_candidates[pid]
            # Highest new correct patch count first, plus highest E0 initial purity for contrasting case.
            chosen=sorted(candidates,key=lambda x:(x[0][0],x[0][1]),reverse=True)[:max(1,args.plots_per_ear-1)]
            if len(chosen)<args.plots_per_ear:
                for c in sorted(candidates,key=lambda x:x[1],reverse=True):
                    if c[2] not in {s[2] for s in chosen}:
                        chosen.append(c)
                    if len(chosen)>=args.plots_per_ear:break
            for gain,init,idx,image_id,npz_path,a in chosen:
                with np.load(npz_path) as z:mat={k:z[k] for k in z.files}
                safe=name.replace("'s ",'_').replace(' ','_')
                path=args.out_dir/'plots'/safe/f'crop_{idx:05d}_{image_id}_E0_E{epochs[-1]:02d}.png'
                save_image_grid(path,mat,image_id,{'image_root':args.image_root,'ann':a,'flip_mode':args.flip_mode,'index':idx},epochs,names,name,pid)
                print('[plot]',path,'gain=',gain,'E0_purity=',init,flush=True)
    print('[DONE] Ear crops:',done,'out:',args.out_dir.resolve(),flush=True)
    print('Main files: ear_epoch_summary.csv, ear_gt_patch_margin_transition_summary.csv, per_crop_epoch.csv, per_patch_E0_E10.csv.gz, dense_npz/, plots/',flush=True)
    if args.write_all_epoch_csv:print('Additional full CSV: per_patch_all_epochs.csv.gz',flush=True)

if __name__=='__main__':
    main()
