#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only PP116 GT Part patch relative-competition mechanism audit.

For each focus Part in each existing predicted-object crop, compare fixed cached
DINOv2 patch tokens and image-level Part Presence under:
  (1) original E0/E10/E30/E100 W + predicted foreground
  (2) original E100 W + GT-OBJECT foreground
  (3) GT-OBJECT-foreground-trained E100 W + both foregrounds.

No GT Part used to select patches / calculate Soft Relative. GT Part is read only
AFTER computing scores, relative candidates, and softmax weights; results are
PRIVILEGED AUDIT statistics, never weak-supervision training.

Requires sibling audit_pp116_gt_weighted_purity.py and local repo source:
  train_relproto_alignemt.py, train_pp116_softsingle_dual.py
Outputs:
  per_crop_competition.csv, per_part_setting_summary.csv,
  true_gt_patch_competitor_votes.csv, protocol.json

GT Part is resized with cropaug_box_xyxy + recorded hflip + nearest 32x32.
An unrecorded augment could invalidate geometry; see existing visual audit.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn.functional as F
import audit_pp116_gt_weighted_purity as au

C,N,D,IGN=116,1024,768,255
DEFAULT_PARTS=["cow's horn","sheep's horn","bicycle's saddle","car's mirror","dog's ear","cat's ear","person's upper arm"]

def csv_save(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def mean(rows,key,cond=lambda r:True):
    a=[r.get(key) for r in rows if cond(r)]
    vals=[float(x) for x in a if x is not None and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else math.nan

def metric_one(*,ids,pid,txt,tokens,fg,gt,tau,names,src):
    """Computes ALL assignment from weak-supervision inputs, then accesses GT."""
    local=txt[ids]
    weights,candidates,fallback=au.assignment(local,tokens,fg,tau,src.relative_scores)
    k=ids.index(pid)
    w=weights[k]
    selected=candidates[k]
    target=(gt==pid)
    avail=target&fg
    ntrue=int(target.sum().item());nfg=int(avail.sum().item())
    selgt=int((selected&avail).sum().item())
    nselected=int(selected.sum().item())
    wc=float(w[target].sum().item())
    wi=float(w[gt==IGN].sum().item())
    wother=max(0.0,1-wc-wi)
    if abs(w.sum().item()-1)>2e-5 or bool((w[~fg]!=0).any()):
        raise AssertionError('Soft Relative weights are invalid')
    result={
        'K':len(ids),'gt_target_patches':ntrue,'gt_target_in_fg':nfg,
        'target_fg_recall':nfg/ntrue if ntrue else math.nan,
        'valid_fg_patches':int(fg.sum().item()),
        'selected_patches':nselected,'selected_correct_patches':selgt,
        'true_gt_patch_support_win_rate':selgt/nfg if nfg else math.nan,
        'true_gt_patch_strict_win_count':0,
        'fallback':int(fallback[k].item()),
        'purity':wc,'ignore_mass':wi,'other_gt_mass':wother,
        'effective_weighted_patches':float(w.square().sum().clamp_min(1e-12).reciprocal().item()),
        'global116_top1_rate_on_true_gt':math.nan,
        'global116_mean_gt_rank':math.nan,
        'global116_mean_gt_margin':math.nan,
        'local_true_gt_strict_win_rate':math.nan,
        'local_true_gt_mean_margin':math.nan,
        'local_true_gt_median_margin':math.nan,
        'local_true_gt_max_margin':math.nan,
        'local_true_gt_mean_target_cos':math.nan,
        'local_true_gt_mean_strongest_rival_cos':math.nan,
        'local_most_common_GT_patch_competitor':'',
        'global116_most_common_GT_patch_competitor':'',
    }
    local_votes=Counter();global_votes=Counter()
    if nfg:
        subset=tokens[avail]
        similarities=local@subset.T         # K x n_gt_patches
        gt_sim=similarities[k]
        if len(ids)>1:
            other_ids=[j for j in range(len(ids)) if j!=k]
            other_sim=similarities[other_ids]
            best_val,best_loc=other_sim.max(0)
            margin=gt_sim-best_val
            other_names=[names[ids[j]] for j in other_ids]
            local_votes.update(other_names[x] for x in best_loc.detach().cpu().tolist())
            result.update({
                'local_true_gt_strict_win_rate':float((margin>0).float().mean().item()),
                'true_gt_patch_strict_win_count':int((margin>0).sum().item()),
                'local_true_gt_mean_margin':float(margin.mean().item()),
                'local_true_gt_median_margin':float(margin.median().item()),
                'local_true_gt_max_margin':float(margin.max().item()),
                'local_true_gt_mean_target_cos':float(gt_sim.mean().item()),
                'local_true_gt_mean_strongest_rival_cos':float(best_val.mean().item()),
                'local_most_common_GT_patch_competitor':local_votes.most_common(1)[0][0],
            })
            # Strict margin and selected mask must agree for non-fallback rows.
            if not bool(fallback[k]) and selgt!=int((margin>0).sum().item()):
                raise AssertionError('True GT patch candidate count != strict relative-margin winners')
        else:
            result['local_true_gt_mean_target_cos']=float(gt_sim.mean().item())
        full116=txt@subset.T                # 116 x n_gt_patches
        correct=full116[pid]
        global_other=full116.clone();global_other[pid]=-torch.inf
        global_best,global_idx=global_other.max(0)
        global_margin=correct-global_best
        true_rank=1+(global_other>correct[None,:]).sum(0)
        global_votes.update(names[j] for j in global_idx.detach().cpu().tolist())
        result.update({
            'global116_top1_rate_on_true_gt':float((global_margin>0).float().mean().item()),
            'global116_mean_gt_rank':float(true_rank.float().mean().item()),
            'global116_mean_gt_margin':float(global_margin.mean().item()),
            'global116_most_common_GT_patch_competitor':global_votes.most_common(1)[0][0],
        })
    return result,local_votes,global_votes

def self_test():
    au.self_test()
    class Source:
        @staticmethod
        def relative_scores(s,f):
            k=s.shape[1]
            rivals=s[:,None,:,:].expand(-1,k,-1,-1).masked_fill(
                torch.eye(k,dtype=torch.bool)[None,:,:,None],-torch.inf).max(2).values
            return (s-rivals).masked_fill(~f[:,None,:],-torch.inf)
    names=[f"part_{i}" for i in range(C)]
    txt=F.normalize(torch.eye(C,D),dim=1)
    tokens=torch.zeros((N,D));tokens[:,2]=1.0
    tokens[0,0]=1;tokens[0,2]=0
    tokens[1,1]=1;tokens[1,2]=0
    tokens=F.normalize(tokens,dim=1)
    fg=torch.zeros(N,dtype=torch.bool);fg[:3]=True
    gt=torch.full((N,),IGN,dtype=torch.long);gt[0]=0;gt[1]=1
    result,lv,gv=metric_one(ids=[0,1],pid=0,txt=txt,tokens=tokens,fg=fg,gt=gt,tau=.05,names=names,src=Source)
    assert result['gt_target_in_fg']==1 and result['local_true_gt_strict_win_rate']==1
    assert result['true_gt_patch_support_win_rate']==1 and result['global116_top1_rate_on_true_gt']==1
    assert abs(result['purity']-1)<.0001
    fg[0]=False
    out,_,_=metric_one(ids=[0,1],pid=0,txt=txt,tokens=tokens,fg=fg,gt=gt,tau=.05,names=names,src=Source)
    assert out['gt_target_in_fg']==0 and out['purity']==0
    print('SELF_TEST_PASS_HORN_COMPETITION: strict winners, global rankings, masking, purity')

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--gtpart-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--gtobj-bank',type=Path,default=Path('output/pp116_gtobj_fg_oracle/gt_object_valid_masks.pt'))
    p.add_argument('--baseline-run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--oracle-run',type=Path,default=Path('final_exp/relproto_alignment/pp116_GTObjFG_ORACLE_softrelative_structure_lam4_100ep'))
    p.add_argument('--baseline-epochs',type=int,nargs='+',default=[0,10,30,100])
    p.add_argument('--focus',nargs='+',default=DEFAULT_PARTS)
    p.add_argument('--flip-mode',choices=['auto','none','all'],default='auto')
    p.add_argument('--soft-temperature',type=float,default=.05)
    p.add_argument('--max-crops',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_horn_margin_competitors'))
    return p.parse_args()

def main():
    args=parse_args()
    if args.self_test:return self_test()
    import train_relproto_alignemt as src
    import train_pp116_softsingle_dual as trainer
    if args.soft_temperature<=0:raise ValueError('soft temperature must be >0')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output {args.out_dir}')
    for path in [args.cache,args.text_bank,args.projector,args.model_config,args.classes_file,args.gtobj_bank]:
        if not path.is_file():raise FileNotFoundError(path)
    names=au.canonical_names(args.classes_file)
    focus={names.index(name) for name in args.focus}
    if len(focus)!=len(args.focus):raise ValueError('duplicate focus parts or invalid taxonomy')
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
    raw,raw_names,_=src.load_raw_clip_bank(args.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if list(raw_names)!=names:raise ValueError('Text classes mismatch')
    proj,_=src.load_frozen_projector(project_root=Path.cwd(),config_path=args.model_config,weight_path=args.projector,device=device)
    base=src.normalize_last(src.project_raw_clip_bank(raw,proj,device=device,batch_size=128).float()).detach().to(device)
    if tuple(base.shape)!=(C,D):raise ValueError('Unexpected raw text projection')
    del proj,raw
    cases=[]
    for ep in sorted(set(args.baseline_epochs)):
        if ep not in (0,10,30,100) and not 0<=ep<=100:raise ValueError(f'Invalid E{ep}')
        W=au.checkpoint_W(args.baseline_run,ep,4.).to(device)
        txt=src.normalize_last(base@W).detach()
        cases.append((f'baseline_E{ep}',ep,'pred',txt))
        if ep==100:cases.append((f'baseline_E{ep}',ep,'gtobj',txt))
    Woracle=au.checkpoint_W(args.oracle_run,100,4.).to(device)
    oracletext=src.normalize_last(base@Woracle).detach()
    cases.extend([(f'gtobj_trained_W_E100',100,'pred',oracletext),
                  (f'gtobj_trained_W_E100',100,'gtobj',oracletext)])
    data=au.annotations(au.load_pt(args.cache))
    maskbank=au.load_pt(args.gtobj_bank)
    if not isinstance(maskbank,dict) or 'masks' not in maskbank or 'manifest' not in maskbank:
        raise ValueError('Unrecognized GT object bank structure')
    masks=torch.as_tensor(maskbank['masks'],dtype=torch.bool)
    if tuple(masks.shape)!=(len(data),N):raise ValueError(f'GT Object sidecar shape mismatch {masks.shape} vs {(len(data),N)}')
    mf=maskbank['manifest']
    if len(mf)!=len(data):raise ValueError('GT Object manifest length mismatch')
    for j in {0,len(data)//2,len(data)-1}:
        if str(mf[j]['image_id'])!=str(data[j]['image_id']) or int(mf[j]['row_index'])!=j:
            raise ValueError(f'GT object sidecar row order differs at {j}')
    md=maskbank.get('metadata',{})
    if md.get('source_cache_size') and int(md['source_cache_size'])!=args.cache.stat().st_size:
        raise ValueError('GT Object sidecar source-cache size differs')
    gtlookup=au.GTLookup(args.gtpart_root)
    rows=[];votes=Counter();check=False
    limit=min(args.max_crops,len(data)) if args.max_crops>0 else len(data)
    cases=[(label,ep,foreground,txt) for label,ep,foreground,txt in cases]
    print('[CASES]',[(a,e,f) for a,e,f,_ in cases],flush=True)
    print('[CACHE]',limit,'/',len(data),'focus',args.focus,flush=True)
    with torch.no_grad():
      for i in range(limit):
        record=data[i];ids=au.part_ids(record)
        targets=sorted(set(ids)&focus)
        if not targets:continue
        tokens=src.normalize_last(torch.as_tensor(record['cropaug_patch_tokens'],device=device,dtype=torch.float32))
        if tuple(tokens.shape)!=(N,D):raise ValueError(f'Token shape row {i} = {tokens.shape}')
        pred_fg=torch.as_tensor(record['pred_obj_mask_patch'],device=device,dtype=torch.bool).reshape(-1)
        gtobj_fg=masks[i].to(device=device)
        if not bool(pred_fg.any()):raise ValueError(f'Empty original predicted fg at row {i}')
        spatial,flip,imshape,box=gtlookup.crop32(record,args.flip_mode)
        gt=torch.as_tensor(np.array(spatial, copy=True),device=device,dtype=torch.long)
        for label,ep,fgsource,txt in cases:
            fg=pred_fg if fgsource=='pred' else gtobj_fg
            if not bool(fg.any()):continue
            if not check and len(ids)>1:
                local=txt[ids]
                w,_,_=au.assignment(local,tokens,fg,args.soft_temperature,src.relative_scores)
                vp,_=trainer.soft_relative_prototypes(local[None],tokens[None],fg[None],temperature=args.soft_temperature)
                if (vp[0]-F.normalize(w@tokens,dim=-1,eps=1e-12)).abs().max().item()>2e-5:
                    raise AssertionError('Mismatch with original soft-relative training code')
                check=True
                print('[CHECK] audit weights reproduce training visual prototypes',flush=True)
            for pid in targets:
                result,localvotes,globalvotes=metric_one(ids=ids,pid=pid,txt=txt,tokens=tokens,fg=fg,gt=gt,
                    tau=args.soft_temperature,names=names,src=src)
                rows.append({'crop_index':i,'image_id':str(record['image_id']),
                    'part_id':pid,'part':names[pid],'arm':label,'epoch':ep,'foreground':fgsource,
                    'flip':int(flip),'crop_box_xyxy':json.dumps([float(x) for x in box]),**result})
                for cname,n in localvotes.items():votes[(names[pid],label,fgsource,'local',cname)]+=n
                for cname,n in globalvotes.items():votes[(names[pid],label,fgsource,'full116',cname)]+=n
        if (i+1)%500==0:print('[PROGRESS]',i+1,'/',limit,flush=True)
    if not check:raise ValueError('No >=2 candidate crop matched focus targets; inspect --max-crops')
    if not rows:raise RuntimeError('No focus part crops matched')
    out=args.out_dir;out.mkdir(parents=True)
    csv_save(out/'per_crop_competition.csv',rows)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['part'],row['arm'],row['foreground'],row['epoch'])].append(row)
    summaries=[]
    for (part,arm,fg,ep),rr in sorted(grouped.items()):
        a=[r for r in rr if r['gt_target_patches']>0]
        b=[r for r in a if r['gt_target_in_fg']>0]
        n_gt=sum(r['gt_target_patches'] for r in a)
        n_fg=sum(r['gt_target_in_fg'] for r in a)
        winner_gt=sum(r['true_gt_patch_strict_win_count'] for r in b)
        selected_gt=sum(r['selected_correct_patches'] for r in b)
        summaries.append({
            'part':part,'arm':arm,'foreground':fg,'epoch':ep,
            'candidate_crops':len(rr),'gt_present_crops':len(a),'gt_in_foreground_crops':len(b),
            'total_target_gt_patches':n_gt,'target_gt_patches_in_foreground':n_fg,
            'target_gt_fg_recall_micro':n_fg/n_gt if n_gt else math.nan,
            'true_patch_strict_win_rate_micro':winner_gt/n_fg if n_fg else math.nan,
            'true_patch_strict_win_rate_macro':mean(b,'local_true_gt_strict_win_rate'),
            'correct_selected_patch_count_total':selected_gt,
            'mean_correct_selected_patch_count_per_crop':mean(rr,'selected_correct_patches'),
            'mean_selected_patch_count':mean(rr,'selected_patches'),
            'mean_purity_all_crops':mean(rr,'purity'),
            'mean_purity_gt_in_fg_crops':mean(b,'purity'),
            'mean_ignore_mass':mean(rr,'ignore_mass'),
            'mean_fallback_rate':mean(rr,'fallback'),
            'mean_local_true_gt_margin':mean(b,'local_true_gt_mean_margin'),
            'mean_local_true_gt_target_cos':mean(b,'local_true_gt_mean_target_cos'),
            'mean_local_true_gt_best_rival_cos':mean(b,'local_true_gt_mean_strongest_rival_cos'),
            'mean_global116_gt_rank':mean(b,'global116_mean_gt_rank'),
            'mean_global116_top1_rate_on_true_gt':mean(b,'global116_top1_rate_on_true_gt'),
            'mean_global116_gt_margin':mean(b,'global116_mean_gt_margin'),
        })
    csv_save(out/'per_part_setting_summary.csv',summaries)
    vote_rows=[]
    totals=Counter()
    for (p,a,f,scope,c),v in votes.items():totals[(p,a,f,scope)]+=v
    for (p,a,f,scope,c),v in sorted(votes.items(),key=lambda x:(x[0][0],x[0][1],x[0][2],x[0][3],-x[1])):
        vote_rows.append({'part':p,'arm':a,'foreground':f,'scope':scope,'strongest_competing_text':c,
            'n_true_GT_patches':v,'fraction':v/totals[p,a,f,scope]})
    csv_save(out/'true_gt_patch_competitor_votes.csv',vote_rows)
    proto={
        'tag':'PRIVILEGED_GT_PART_OFFLINE_AUDIT_ONLY, NO TRAINING',
        'foregrounds':'pred = cached Talk2DINO prediction, gtobj = privileged GT object sidecar',
        'text_arms':'baseline E0/E10/E30/E100 W; GT-object-trained W E100',
        'GT_part_use':'GT Part only to score correct patch after assignment; NEVER select candidate nor compute W',
        'class_condition':'Image-level Part Presence, same cached candidate pids in all cases',
        'W_cases':[{'arm':a,'epoch':e,'foreground':f} for a,e,f,_ in cases],
        'patch_geometry':'cached crop box, optional recorded flip, nearest resize32; cannot exclude unrecorded transforms',
        'soft_temperature':args.soft_temperature,'total_cache':len(data),'processed':limit,'focus_parts':args.focus,
    }
    (out/'protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE]',out,'rows',len(rows),'summary rows',len(summaries),flush=True)
    for r in summaries:
        if r['part'] in ["cow's horn","sheep's horn","bicycle's saddle"]:
            print('{part:18s} {arm:25s} fg={foreground:5s} support={gt_in_foreground_crops:3d} '
                  'win={true_patch_strict_win_rate_micro:.3%} purity={mean_purity_all_crops:.3%} '
                  'ignore={mean_ignore_mass:.3%} global_top1={mean_global116_top1_rate_on_true_gt:.3%}'.format(**r))

if __name__=='__main__':main()
