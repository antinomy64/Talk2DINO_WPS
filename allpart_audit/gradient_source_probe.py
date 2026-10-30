#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional sampled first-order gradient-by-source audit for ORIGINAL 116-way Ours.

GT is ONLY used for off-line target margin functional M(W), NEVER in any
NCE/structure gradient calculation. The same 116-wide NCE and loss annotation
normalization are used as original trainer. This is NOT Adam/SVD attribution.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def parse():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,default=Path('output/pp116_allpart_audit_ours_v1'))
 p.add_argument('--run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
 p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
 p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
 p.add_argument('--epochs',default='0,1,3,10,30,100')
 p.add_argument('--num-source-crops',type=int,default=64,help='Deterministic greedy coverage: most distinct source positive part labels')
 p.add_argument('--targets-per-group',type=int,default=4)
 p.add_argument('--device',default='cuda')
 p.add_argument('--soft-temperature',type=float,default=.05)
 p.add_argument('--nce-temperature',type=float,default=.07)
 p.add_argument('--lambda-struct',type=float,default=4.)
 p.add_argument('--rank-delta',type=float,default=.05)
 p.add_argument('--rank-margin',type=float,default=.05)
 p.add_argument('--flip-mode',default='auto')
 return p.parse_args()

def make_example(idx,a,src):
 return src.Example(row_index=idx,image_id=str(a.get('image_id','')),ann_id=str(a.get('id',idx)),
    class_name=str(a.get('class_name','')),
    pids=torch.as_tensor(a['part_category_id'],dtype=torch.long).reshape(-1),
    tokens=torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32).reshape(1024,src.VISION_DIM),
    foreground=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool).reshape(-1))

def source_crops(anns,count):
 # Greedy set-cover over presence positive labels to avoid bias to abundant objects.
 pool=[]
 for i,a in enumerate(anns):
    ids=set(int(x) for x in a['part_category_id'])
    if ids:pool.append((i,ids))
 chosen=[];covered=set();used=set()
 for _ in range(count):
    remaining=[(len(ids-covered),len(ids),-i,i,ids) for i,ids in pool if i not in used]
    if not remaining:break
    _,_,_,i,ids=max(remaining)
    chosen.append(i);used.add(i);covered|=ids
 return chosen,sorted(covered)

def tangent(W,g):
 x=W.T@g;return g-W@((x+x.T)*.5)

def dotcos(a,b):
 den=float(a.norm()*b.norm());return float((a*b).sum()/den) if den>1e-16 else None

def write(path,rows):
 if not rows:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def batch_loss_by_pid(base,W,examples,rel,args,trainer,device):
 # Keep original K-wise visual prototypes, stopgrad pseudo target, 116 class logits,
 # per-annotation 1/K weight and batch mean. GT is not referenced anywhere here.
 text=F.normalize(base@W,dim=-1);terms=defaultdict(list);rank_terms=[]
 for ex in examples:
    ids=ex.pids.to(device);tok=F.normalize(ex.tokens.to(device),dim=-1)
    fg=ex.foreground.to(device)
    K=len(ids)
    if K>=2:
       vis,_=trainer.soft_relative_prototypes(text[ids][None],tok[None],fg[None],temperature=args.soft_temperature)
       vis=vis[0]
    else:
       scores=text[ids]@tok.T
       weights=torch.softmax(scores.masked_fill(~fg[None,:],-torch.inf)/args.soft_temperature,-1)
       vis=F.normalize(weights@tok,dim=-1,eps=1e-12)
    nce_logits=vis.detach()@text.T/args.nce_temperature
    per=F.cross_entropy(nce_logits,ids,reduction='none')/K
    for j,pid in enumerate(ids.tolist()):terms[pid].append(per[j]/len(examples))
    if K>=3 and args.lambda_struct>0:
       r,_a,_n=trainer.relational_rank_loss(vis[None],rel[ids[:,None],ids[None,:]][None],
                                 delta=args.rank_delta,margin=args.rank_margin)
       rank_terms.append(r[0]/len(examples))
    else:rank_terms.append(text.sum()*0.)
 return {pid:torch.stack(x).sum() for pid,x in terms.items()},torch.stack(rank_terms).sum()

def main():
 a=parse();start=time.time()
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 import train_relproto_alignemt as src
 import train_pp116_softsingle_dual as trainer
 import audit_pp116_gt_weighted_purity as oracle
 from audit_all_116 import load_bank,load_ws,load_names
 names,classes=load_names(Path(__file__).resolve().parent/'pp116_four_groups_per_part.csv',a.run/'projected_part_bank.pt')
 device=torch.device(a.device);torch.backends.cuda.matmul.allow_tf32=False
 base,bankpath=load_bank(a.run,None,device,names);rel=(base@base.T).detach()
 anns=oracle.annotations(src.torch_load(a.cache));epochs=[int(x) for x in a.epochs.split(',')]
 if sorted(epochs)!=epochs:raise ValueError('Epochs must be sorted')
 with torch.no_grad():
    pass
 source_ids,covered=source_crops(anns,a.num_source_crops)
 src_examples=[make_example(i,anns[i],src) for i in source_ids]
 # Targets: manifest from same GT audit; uses representative high/low cases per class.
 mani=a.out/'summary'/'exemplar_crop_manifest.csv'
 if not mani.exists():raise FileNotFoundError('Run summarize_all_116.py before gradient probe')
 with mani.open(encoding='utf-8-sig',newline='') as f: candidates=list(csv.DictReader(f))
 targets=[]
 for grp in 'ABCD':
    ps=[]
    for r in candidates:
       if r['group']==grp and r['role']=='best_Q' and r['pid'] not in ps:
           ps.append(r['pid'])
       if len(ps)>=a.targets_per_group:break
    for pid in ps:
       sel=next(r for r in candidates if r['pid']==pid and r['role']=='best_Q')
       targets.append((int(pid),int(sel['crop_index']),grp))
 if a.targets_per_group>6:
    # Exhaustive target option is drawn from the WHOLE SQLite audit, not just 24 exemplars.
    # A part without any GT target patches in frozen FG is explicitly excluded.
    import sqlite3
    db=sqlite3.connect(str(a.out/'audit.sqlite'))
    best={}
    for (payload,) in db.execute('SELECT payload FROM part_epoch WHERE epoch=0'):
        r=json.loads(payload);pid=int(r['pid']);gr=classes[pid]['类别编码']
        if gr not in ('A','B','C','D') or int(r.get('n_gt_fg',0))<3:continue
        if pid not in best or r['n_gt_fg']>best[pid][0]:
            best[pid]=(int(r['n_gt_fg']),int(r['crop_index']),gr)
    db.close()
    targets=[]
    for gr in 'ABCD':
        gparts=sorted((pid,n,crop,g) for pid,(n,crop,g) in best.items() if g==gr)
        for pid,n,crop,g in gparts[:a.targets_per_group]:targets.append((pid,crop,g))
 if not targets:raise RuntimeError('No target crops with GT')
 lookup=oracle.GTLookup(a.gt_root)
 tdata=[]
 for pid,i,grp in targets:
    ex=make_example(i,anns[i],src)
    if pid not in ex.pids.tolist():raise AssertionError('Missing target in presence list')
    gt,flip,wh,box=lookup.crop32(anns[i],a.flip_mode)
    gt=np.asarray(gt).reshape(-1)
    if not (np.asarray(ex.foreground.numpy())&(gt==pid)).any():raise AssertionError('Missing GT target FG')
    tdata.append((pid,i,grp,ex,gt))
 print('[gradient] source_crops',len(src_examples),'covered_positive_classes',len(covered),
       'target_parts',len(tdata),'epochs',epochs,flush=True)
 all_by_source=[];grouped=[];components=[]
 for epoch in epochs:
    # load exact W, not normalized text bank (load_ws returns text not W).
    if epoch==0:wv=torch.eye(src.VISION_DIM,device=device)
    else:
       ck=src.torch_load(a.run/f'W_epoch_{epoch:03d}.pt')
       wv=ck['W'].to(device=device,dtype=torch.float32)
    W=torch.nn.Parameter(wv.detach().clone())
    # use identical frozen cache candidate rows and structure loss.
    source_losses,struct=batch_loss_by_pid(base,W,src_examples,rel,a,trainer,device)
    if epoch==epochs[0]:
        full,info,parts=trainer.batch_forward(base,W,src_examples,device=device,
                       frozen_relations=rel,args=a)
        recomputed=sum(source_losses.values())
        if not torch.allclose(recomputed,parts['nce'],atol=2e-5,rtol=1e-5):
            raise AssertionError(f'NCE decomposition mismatch with original: {float(recomputed)} vs {float(parts["nce"])}')
        if not torch.allclose(struct,parts['relation'],atol=2e-5,rtol=1e-5):
            raise AssertionError(f'Structure decomposition mismatch: {float(struct)} vs {float(parts["relation"])}')
        g_ref=torch.autograd.grad(full,W,retain_graph=True)[0]
        g_dec=torch.autograd.grad(recomputed+a.lambda_struct*struct,W,retain_graph=True)[0]
        relerr=float((g_ref-g_dec).norm()/(g_ref.norm()+1e-12))
        if relerr>2e-4:raise AssertionError(f'Train loss/gradient equivalence failed: {relerr}')
        print('[check] original 116-way loss AND W gradient equivalence PASS, relative error',relerr,flush=True)
    g_src={}
    for pid,loss in source_losses.items():
        g=torch.autograd.grad(loss,W,retain_graph=True)[0]
        g_src[pid]=tangent(W.detach(),g.detach())
    if struct.requires_grad:
        g_struct=tangent(W.detach(),torch.autograd.grad(struct,W,retain_graph=False)[0])*a.lambda_struct
    else:g_struct=torch.zeros_like(W)
    for pid,idx,grp,ex,gt in tdata:
        # Make a fresh autograd graph ONLY for retrospective GT margin, not for training.
        t=F.normalize(base@W,dim=-1)[ex.pids.to(device)]
        tokens=F.normalize(ex.tokens.to(device),dim=-1)
        score=t@tokens.T
        target_pos=ex.pids.tolist().index(pid)
        if len(ex.pids)<2:continue
        other=torch.cat([score[:target_pos],score[target_pos+1:]],0)
        r=score[target_pos]-other.max(0).values
        mask=(torch.from_numpy(gt==pid).to(device)&ex.foreground.to(device))
        gt_margin=r[mask].mean()
        gm=tangent(W.detach(),torch.autograd.grad(gt_margin,W)[0])
        vals=defaultdict(float)
        for srcpid,g in g_src.items():
            src_object=names[srcpid].split("'s ",1)[0]
            this_object=names[pid].split("'s ",1)[0]
            src_part=names[srcpid].split("'s ",1)[1]
            this_part=names[pid].split("'s ",1)[1]
            bucket=('SELF' if srcpid==pid else 'SAME_OBJECT_OTHER' if src_object==this_object
                     else 'SAME_PART_OTHER_OBJECT' if src_part==this_part else 'OTHER_PART_OTHER_OBJECT')
            val=-float((gm*g).sum());vals[bucket]+=val
            all_by_source.append(dict(epoch=epoch,target_part=names[pid],target_pid=pid,target_group=grp,
                 target_crop_index=idx,source_pid=srcpid,source_part=names[srcpid],source_group=bucket,
                 directional_gain=val,source_tangent_norm=float(g.norm()),
                 source_present_in_fixed_probe=1))
        struct_gain=-float((gm*g_struct).sum())
        row=dict(epoch=epoch,target_part=names[pid],target_pid=pid,group=grp,crop_index=idx,
            n_source_crops=len(src_examples),n_source_positive_classes=len(source_losses),
            gt_margin=float(gt_margin.detach()),
            SELF=vals['SELF'],SAME_OBJECT_OTHER=vals['SAME_OBJECT_OTHER'],
            SAME_PART_OTHER_OBJECT=vals['SAME_PART_OTHER_OBJECT'],
            OTHER_PART_OTHER_OBJECT=vals['OTHER_PART_OTHER_OBJECT'],
            STRUCTURE=struct_gain)
        row['ALL_NCE']=sum(vals.values());row['TOTAL']=row['ALL_NCE']+struct_gain
        grouped.append(row)
    components.append(dict(epoch=epoch,source_crops=len(src_examples),source_positive_classes=len(source_losses),
       loss_nce=float(sum(v.detach() for v in source_losses.values())),
       loss_structure_raw=float(struct.detach()),
       tangent_total_nce_norm=float(sum(g_src.values(),torch.zeros_like(W)).norm()),
       tangent_structure_weighted_norm=float(g_struct.norm())))
    print('[gradient] E',epoch,'targets',len(tdata),'sources',len(g_src),'seconds',round(time.time()-start),flush=True)
 dest=a.out/'gradient';write(dest/'by_source_part.csv',all_by_source)
 write(dest/'grouped_source_vs_target.csv',grouped);write(dest/'per_epoch_gradient_norms.csv',components)
 (dest/'metadata.json').write_text(json.dumps(dict(epochs=epochs,targets=[dict(pid=p,crop=i,group=g) for p,i,g,_,_ in tdata],
    source_indices=source_ids,covered_source_positive_classes=covered,
    gt_role='Target M(W) retrospective diagnostic ONLY. Loss from image-level PIDs and GT-free pseudo targets.',
    directional_gain='-<tangent grad M, tangent grad source loss>; sampled local SGD predictor, NOT exact Adam/SVD causal attribution',
    absent_source_part='No gradient from this part as positive in the sampled source batch; never interpret as global zero',
    caveat='Source NCE still has every one of 116 classes in its denominator. Every source-positive term updates ALL text embeddings via W.'),indent=2),encoding='utf-8')
 print('[DONE] gradient attribution',dest,flush=True)
if __name__=='__main__':main()
