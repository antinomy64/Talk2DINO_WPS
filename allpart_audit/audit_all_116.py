#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, resume-safe all-116 part pseudo-label and competitor audit.

The trainer and frozen cache NEVER receive GT. GT is loaded after predicted
scores/support/weights/prototypes have been computed; GT serves AUDIT ONLY.
Requires existing server helper audit_pp116_gt_weighted_purity.py, used in prior
Ear experiments to reconstruct the *same* 32x32 GT-on-crop projection.

Outputs SQLite row ledger, no model training, no checkpoint writes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
NUM_PATCHES = 1024

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--projected-bank',type=Path,default=None,help='Defaults to RUN/projected_part_bank.pt; original frozen projected text [116,768]')
    p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--groups',type=Path,default=Path(__file__).resolve().parent/'pp116_four_groups_per_part.csv')
    p.add_argument('--out',type=Path,default=Path('output/pp116_allpart_audit_ours_v1'))
    p.add_argument('--epochs',default='0,1,2,3,5,10,20,30,40,50,60,70,80,90,100')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--tau',type=float,default=0.05)
    p.add_argument('--nce-tau',type=float,default=0.07)
    p.add_argument('--rank-delta',type=float,default=0.05)
    p.add_argument('--rank-margin',type=float,default=0.05)
    p.add_argument('--lambda-struct',type=float,default=4.0)
    p.add_argument('--flip-mode',default='auto',choices=['auto','none','all'])
    p.add_argument('--max-crops',type=int,default=0,help='0=ALL. Positive integer for smoke test only.')
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-source',action='store_true',help='Compare a real-crop forward against original Ours trainer')
    return p.parse_args()


def load_names(groups, projected_bank):
    """Taxonomy ID order MUST come from bank, NOT the A/B/C/D CSV row order."""
    import train_relproto_alignemt as src
    with groups.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=116:raise RuntimeError(f'Expected 116 A/B/C/D rows, got {len(rows)}')
    byname={r['part']:r for r in rows}
    if len(byname)!=116:raise RuntimeError('Duplicate part name in reference classifications')
    bank=src.torch_load(projected_bank)
    canonical=bank.get('classnames',bank.get('class_names'))
    if canonical is None or len(canonical)!=116:
        raise RuntimeError('Frozen projected bank missing canonical 116 class names')
    names=list(map(str,canonical))
    if set(names)!=set(byname) or len(set(names))!=116:
        raise RuntimeError('Historical ABCD names are not a 1:1 match to ORIGINAL projected text bank names')
    classes=[byname[n] for n in names]
    return names,classes


def init_db(db):
    con=sqlite3.connect(str(db));con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=NORMAL')
    con.executescript('''
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS epoch_done(crop_index INTEGER NOT NULL,epoch INTEGER NOT NULL,PRIMARY KEY(crop_index,epoch));
      CREATE TABLE IF NOT EXISTS part_epoch(crop_index INTEGER NOT NULL,epoch INTEGER NOT NULL,pid INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(crop_index,epoch,pid));
      CREATE INDEX IF NOT EXISTS ix_part_epoch_pid ON part_epoch(pid,epoch);
      CREATE TABLE IF NOT EXISTS transition(crop_index INTEGER NOT NULL,epoch INTEGER NOT NULL,pid INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(crop_index,epoch,pid));
      CREATE INDEX IF NOT EXISTS ix_transition_pid ON transition(pid,epoch);
      CREATE TABLE IF NOT EXISTS gt_presence_coverage(crop_index INTEGER NOT NULL,pid INTEGER NOT NULL,gt_patches INTEGER NOT NULL,gt_fg_patches INTEGER NOT NULL,in_presence INTEGER NOT NULL,PRIMARY KEY(crop_index,pid));
      CREATE INDEX IF NOT EXISTS ix_gt_presence_pid ON gt_presence_coverage(pid);
      CREATE TABLE IF NOT EXISTS endpoint_transition(crop_index INTEGER NOT NULL,pid INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(crop_index,pid));
      CREATE INDEX IF NOT EXISTS ix_endpoint_pid ON endpoint_transition(pid);
    ''')
    return con


def check_configuration(con,config,resume):
    existing={k:json.loads(v) for k,v in con.execute('SELECT key,value FROM metadata')}
    if existing:
        if existing.get('config')!=config:
            raise RuntimeError('Existing SQLite was made with DIFFERENT config. Use a new --out path.\nOld: '+str(existing.get('config'))+'\nNew: '+str(config))
        if not resume:raise RuntimeError('Output already exists. Add --resume to continue safely.')
    else:
        con.execute('INSERT INTO metadata(key,value) VALUES (?,?)',('config',json.dumps(config,ensure_ascii=False,sort_keys=True)))
        con.commit()


def load_bank(run,bank_path,device,names):
    import train_relproto_alignemt as src
    f=bank_path or run/'projected_part_bank.pt'
    if not f.is_file():raise FileNotFoundError(f)
    data=src.torch_load(f)
    feats=data['features']
    bank_names=data.get('classnames',data.get('class_names'))
    if list(map(str,bank_names))!=names:
        mismatch=next(((i,names[i],str(bank_names[i])) for i in range(min(len(names),len(bank_names))) if names[i]!=str(bank_names[i])),None)
        raise AssertionError(f'Projected bank must match EXACT taxonomy order: {mismatch}')
    base=F.normalize(torch.as_tensor(feats,device=device,dtype=torch.float32),dim=-1)
    if base.shape!=(116,src.VISION_DIM):raise ValueError(f'Invalid projected bank: {base.shape}')
    return base,f


def load_ws(run,epochs,base,device):
    import train_relproto_alignemt as src
    mats=[]
    for ep in epochs:
        if ep==0: W=torch.eye(base.size(1),device=device)
        else:
            path=run/f'W_epoch_{ep:03d}.pt'
            if not path.is_file():raise FileNotFoundError(path)
            ckpt=src.torch_load(path)
            W=ckpt['W'].to(device=device,dtype=torch.float32)
            if W.shape!=(base.size(1),base.size(1)):raise ValueError(f'Unexpected W: {path}')
            orth=float((W.T@W-torch.eye(W.size(0),device=device)).abs().max())
            if orth>0.003:raise ValueError(f'Bad orthogonality {path}: {orth:.6g}')
            if isinstance(ckpt,dict) and 'epoch' in ckpt and int(ckpt['epoch'])!=ep:
                raise RuntimeError(f'Checkpoint epoch conflict: {path}')
        mats.append(F.normalize(base @ W,dim=-1).detach())
    return torch.stack(mats)


def scalar(x):
    return float(x) if np.isfinite(x) else None


def weighted_quality(mask,w):
    return float(w[mask].sum())


def score_all_epochs(text,ids,token,fg,tau,nce_tau,frozen_rel,rank_delta,rank_margin,check_source=False):
    """All-score math exactly matching train_pp116_softsingle_dual.py."""
    import train_relproto_alignemt as src
    E=text.shape[0]; K=len(ids)
    local=text[:,ids,:]
    scores=torch.matmul(local,token.T)   # E,K,N
    if K>=2:
        relative=src.relative_scores(scores,fg[None,:].expand(E,-1))
        pos=relative>0
        anchor=relative.argmax(-1)
        empty=~pos.any(-1)
        selector=pos | (F.one_hot(anchor,NUM_PATCHES).bool() & empty[...,None])
        logits=(relative/tau).masked_fill(~selector,-torch.inf)
        weight=torch.softmax(logits,dim=-1)
        other_score=scores[:,None,:,:].expand(E,K,K,NUM_PATCHES).clone()
        eye=torch.eye(K,device=scores.device,dtype=torch.bool)
        other_score=other_score.masked_fill(eye[None,:,:,None],-torch.inf)
        rival_score,rival=other_score.max(2)
        if not torch.allclose(relative[...,fg],(scores-rival_score)[...,fg],atol=4e-5,rtol=1e-5):
            raise AssertionError('Relative margin differs from original-minus-strongest-competitor')
    else:
        relative=scores.masked_fill(~fg[None,None,:],-torch.inf)
        selector=fg[None,None,:].expand(E,1,-1)
        weight=torch.softmax(relative/tau,dim=-1)
        empty=torch.zeros((E,1),device=scores.device,dtype=torch.bool)
        rival=torch.full_like(scores,-1,dtype=torch.long)
        rival_score=torch.full_like(scores,float('nan'))
    proto=F.normalize(torch.matmul(weight,token),dim=-1,eps=1e-12)
    # Training InfoNCE uses GLOBAL 116 classes. No GT used.
    nce_logits=(proto @ text.transpose(1,2))/nce_tau
    nce_logp=torch.log_softmax(nce_logits,dim=-1)
    # Follows original rank loss, with all proto related under original presence candidates.
    rel_losses=[];trips=[];rank_satisfaction=[]
    for ep in range(E):
        if K>=3:
            v=proto[ep];g=frozen_rel[ids][:,ids]
            ids_k=torch.arange(K,device=token.device)
            p=ids_k[:,None,None];q=ids_k[None,:,None];r=ids_k[None,None,:]
            valid=(p!=q)&(p!=r)&(q!=r)&((g[:,:,None]-g[:,None,:])>rank_delta)
            d=(v@v.T)
            hinges=torch.relu(rank_margin-(d[:,:,None]-d[:,None,:]))
            rel_losses.append(float((hinges*valid).sum().item()/max(1,int(valid.sum()))))
            trips.append(int(valid.sum()))
            rank_satisfaction.append(float(((d[:,:,None]-d[:,None,:]>=rank_margin)&valid).sum().item()/max(1,int(valid.sum()))))
        else:rel_losses.append(0.);trips.append(0);rank_satisfaction.append(None)
    if check_source:
        import train_pp116_softsingle_dual as tr
        with torch.no_grad():
            one=local[0:1];m=fg[None]
            if K>=2:
                orig,diag=tr.soft_relative_prototypes(one,token[None],m,temperature=tau)
                if (proto[0:1]-orig).abs().max()>4e-5 or not torch.equal(empty[0:1],diag['fallback'].bool()):
                    raise AssertionError('Actual trainer pseudo-prototype mismatch')
            else:
                sc=torch.bmm(one,token[None].transpose(1,2)); wt=torch.softmax(sc.masked_fill(~m[:,None,:],-torch.inf)/tau,dim=-1)
                if (wt-weight[0:1]).abs().max()>2e-5:raise AssertionError('K=1 softmax mismatch')
    # CPU transfer once each tensor rather than synchronizing per-part/epoch.
    outs={k:v.detach().cpu().numpy() for k,v in dict(
        scores=scores,relative=relative,selector=selector,weight=weight,empty=empty,
        rival=rival,rival_score=rival_score,proto=proto,nce_logp=nce_logp,nce_logits=nce_logits).items()}
    outs['struct_loss']=rel_losses;outs['triplets']=trips;outs['structure_rank_satisfaction']=rank_satisfaction
    return outs


def part_row(a,idx,ep,pid,name,category,ids,pos,s,ti,gt,fg_np,fgclass,nce_tau):
    t=ti
    w=s['weight'][t,pos]; sel=s['selector'][t,pos].astype(bool)
    score=s['scores'][t,pos]; rival=s['rival_score'][t,pos]
    r=s['relative'][t,pos]; rival_id=s['rival'][t,pos]
    in_gt=(gt==pid); gt_fg=in_gt&fg_np;other=fg_np&(gt!=255)&(gt!=pid);ignore=fg_np&(gt==255)
    n_gt=int(in_gt.sum());n_gt_fg=int(gt_fg.sum());n_fg=int(fg_np.sum()); k=len(ids)
    qgt=weighted_quality(in_gt,w);qo=weighted_quality(other,w);qi=weighted_quality(ignore,w)
    pseudo=s['proto'][t,pos];vis_oracle=None
    if n_gt_fg:
        oracle=fgclass[gt_fg].mean(0);oracle/=max(1e-12,float(np.linalg.norm(oracle)))
        vis_oracle=float(pseudo @ oracle)
    selected_gt=int((sel&gt_fg).sum());selected_other=int((sel&other).sum());selected_ignore=int((sel&ignore).sum())
    p_true=float(np.exp(s['nce_logp'][t,pos,pid]));pred_id=int(s['nce_logp'][t,pos].argmax())
    rank=1+int((s['nce_logp'][t,pos]>s['nce_logp'][t,pos,pid]).sum())
    if k>=2:
        rs=r[fg_np];gtm=r[gt_fg];rm=r[other]
        cand_pos=(r>0)&fg_np
        cnt_gt=int((cand_pos&gt_fg).sum()); cnt_other=int((cand_pos&other).sum());cnt_ignore=int((cand_pos&ignore).sum())
        rivals_gt=Counter(ids[int(i)] for i in rival_id[gt_fg].tolist())
        rivals_bad=Counter(ids[int(i)] for i in rival_id[other].tolist())
        rival_true=int(rivals_gt.most_common(1)[0][0]) if rivals_gt else None
        rival_wrong=int(rivals_bad.most_common(1)[0][0]) if rivals_bad else None
    else:
        gtm=[];rm=[];cnt_gt=cnt_other=cnt_ignore=None;rivals_gt={};rivals_bad={};rival_true=rival_wrong=None
    ent=float(-(w[w>0]*np.log(w[w>0])).sum())
    row=dict(crop_index=idx,image_id=str(a.get('image_id','')),epoch=ep,pid=pid,part=name,
       group=category,object_name=name.split("'s ",1)[0],K=k,n_fg=n_fg,n_gt=n_gt,n_gt_fg=n_gt_fg,
       fg_recall=n_gt_fg/n_gt if n_gt else None,gt_observed=int(n_gt>0),
       selected_count=int(sel.sum()),selected_gt=selected_gt,selected_other=selected_other,selected_ignore=selected_ignore,
       selected_gt_fraction=selected_gt/n_gt_fg if n_gt_fg else None,
       selected_valid_precision=selected_gt/(selected_gt+selected_other) if selected_gt+selected_other else None,
       q_gt=qgt,q_other=qo,q_ignore=qi,q_gt_valid=qgt/(qgt+qo) if qgt+qo>1e-12 else None,
       high_conf_wrong=int(qgt<.2 and (qgt+qo)>.5),
       fallback=int(s['empty'][t,pos]) if k>=2 else None,
       candidate_positive_count=int((r[fg_np]>0).sum()) if k>=2 else None,
       pos_on_gt_count=cnt_gt,pos_on_other_count=cnt_other,pos_on_ignore_count=cnt_ignore,
       gt_margin_positive=cnt_gt/n_gt_fg if k>=2 and n_gt_fg else None,
       other_margin_positive=cnt_other/int(other.sum()) if k>=2 and other.any() else None,
       mean_margin_gt=float(np.mean(gtm)) if k>=2 and n_gt_fg else None,
       mean_margin_other=float(np.mean(rm)) if k>=2 and other.any() else None,
       mean_margin_fg=float(np.mean(rs)) if k>=2 else None,
       mean_own_cos_gt=float(score[gt_fg].mean()) if n_gt_fg else None,
       mean_rival_cos_gt=float(rival[gt_fg].mean()) if k>=2 and n_gt_fg else None,
       rival_on_gt=rival_true,rival_on_other=rival_wrong,
       rival_gt_hist=dict(rivals_gt),rival_other_hist=dict(rivals_bad),
       max_weight=float(w.max()),effective_support=1.0/max(1e-12,float((w*w).sum())),
       entropy=ent,normalized_entropy=ent/math.log(max(2,int(sel.sum()))) if sel.sum()>1 else 0.,
       pseudo_oracle_cos=vis_oracle,pseudo_text_cos=float(s['nce_logits'][t,pos,pid]*nce_tau),
       global_nce_p_true=p_true,global_nce_rank=rank,global_nce_top1=pred_id,
       global_nce_loss=-float(s['nce_logp'][t,pos,pid]),
       struct_loss_crop=s['struct_loss'][t],struct_triplets=s['triplets'][t],
       structure_rank_satisfaction=s['structure_rank_satisfaction'][t])
    return row


def transition_row(a,idx,ep,pid,name,category,ids,pos,s,t,gt,fg):
    prev=t-1; k=len(ids); g=(gt==pid)&fg; ng=int(g.sum());
    cur_sel=s['selector'][t,pos]; old_sel=s['selector'][prev,pos]
    curw=s['weight'][t,pos];oldw=s['weight'][prev,pos]
    st=dict(crop_index=idx,epoch=ep,pid=pid,part=name,group=category,n_gt_fg=ng,
        gained_selected_gt=int(((~old_sel)&cur_sel&g).sum()),
        lost_selected_gt=int((old_sel&(~cur_sel)&g).sum()),
        gained_selected_wrong=int(((~old_sel)&cur_sel&fg&(gt!=255)&(gt!=pid)).sum()),
        lost_selected_wrong=int((old_sel&(~cur_sel)&fg&(gt!=255)&(gt!=pid)).sum()),
        support_jaccard=float((cur_sel&old_sel).sum()/max(1,(cur_sel|old_sel).sum())),
        weight_total_variation=float(0.5*np.abs(curw-oldw).sum()),
        delta_q_gt=float(curw[gt==pid].sum()-oldw[gt==pid].sum()),
        delta_nce_loss=float(-s['nce_logp'][t,pos,pid]+s['nce_logp'][prev,pos,pid]))
    if k>=2:
        mr=s['relative'][t,pos];mp=s['relative'][prev,pos]
        gained=g&(mp<=0)&(mr>0);lost=g&(mp>0)&(mr<=0)
        st.update(gained_positive_gt=int(gained.sum()),lost_positive_gt=int(lost.sum()),
            competitor_changed_gt=int((s['rival'][t,pos,g]!=s['rival'][prev,pos,g]).sum()),
            delta_gt_margin=float((mr[g]-mp[g]).mean()) if ng else None,
            delta_gt_own=float((s['scores'][t,pos,g]-s['scores'][prev,pos,g]).mean()) if ng else None,
            delta_gt_rival=float((s['rival_score'][t,pos,g]-s['rival_score'][prev,pos,g]).mean()) if ng else None,
            gain_own_up_rival_down=int((gained&(s['scores'][t,pos]>s['scores'][prev,pos])&(s['rival_score'][t,pos]<s['rival_score'][prev,pos])).sum()),
            gain_own_rises_faster=int((gained&(s['scores'][t,pos]>s['scores'][prev,pos])&(s['rival_score'][t,pos]>=s['rival_score'][prev,pos])).sum()),
            gain_rival_falls_faster=int((gained&(s['scores'][t,pos]<=s['scores'][prev,pos])&(s['rival_score'][t,pos]<s['rival_score'][prev,pos])).sum()))
    else:
        st.update(gained_positive_gt=None,lost_positive_gt=None,competitor_changed_gt=None,
            delta_gt_margin=None,delta_gt_own=None,delta_gt_rival=None,
            gain_own_up_rival_down=None,gain_own_rises_faster=None,gain_rival_falls_faster=None)
    return st


def main():
    args=parse_args();start=time.time()
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.tau<=0 or args.nce_tau<=0:raise ValueError('Temperature must be positive')
    epochs=list(dict.fromkeys(int(x) for x in args.epochs.split(',') if x.strip()))
    if not epochs or epochs[0]!=0 or epochs!=sorted(epochs):raise ValueError('Epoch list must be ascending and begin with 0')
    for req in (args.run,args.cache,args.gt_root,args.groups):
        if not req.exists():raise FileNotFoundError(req)
    import train_relproto_alignemt as src
    import audit_pp116_gt_weighted_purity as gt_audit
    if not hasattr(gt_audit,'GTLookup') or not hasattr(gt_audit,'annotations'):
        raise RuntimeError('Old GT audit helper incompatible. Requires GTLookup.crop32 and annotations as in previous successful Ear audit.')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    projected_path=args.projected_bank or args.run/'projected_part_bank.pt'
    names,classes=load_names(args.groups,projected_path)
    base,bankpath=load_bank(args.run,args.projected_bank,device,names)
    with torch.no_grad(): text=load_ws(args.run,epochs,base,device)
    cache=src.torch_load(args.cache)
    anns=gt_audit.annotations(cache)
    if args.max_crops>0:anns=anns[:args.max_crops]
    lookup=gt_audit.GTLookup(args.gt_root)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    config=dict(audit_version='allpart_116_readonly_v1',run=str(args.run.resolve()),cache=str(args.cache.resolve()),
       bank=str(bankpath.resolve()),gt_root=str(args.gt_root.resolve()),group_file_sha256=hashlib.sha256(args.groups.read_bytes()).hexdigest(),
       epochs=epochs,tau=args.tau,nce_tau=args.nce_tau,lambda_struct=args.lambda_struct,
       rank_delta=args.rank_delta,rank_margin=args.rank_margin,flip_mode=args.flip_mode,
       max_crops=args.max_crops,n_crops=len(anns),num_parts=116,
       gt_role='OFFLINE EVALUATION ONLY; never affects scores, prototype assignment or training',
       W_source='RUN checkpoints, E0 identity; GLOBAL 116-way InfoNCE only')
    con=init_db(out/'audit.sqlite');check_configuration(con,config,args.resume)
    (out/'audit_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    already=set((int(i),int(e)) for i,e in con.execute('SELECT crop_index,epoch FROM epoch_done'))
    frozen=(base@base.T).detach()
    print('[audit] ',len(anns),'crops;',len(epochs),'epochs;',len(already),'previously finished crop-epochs',flush=True)
    verified=False;skipped=0;done=0
    for idx,a in enumerate(anns):
        if all((idx,ep) in already for ep in epochs):skipped+=1;continue
        ids=[int(x) for x in gt_audit.part_ids(a)]
        if not ids or len(ids)!=len(set(ids)) or not all(0<=i<116 for i in ids):
            raise ValueError(f'Invalid image-level candidate part ids at crop {idx}: {ids}')
        obj=str(a.get('class_name',''))
        if any(names[pid].split("'s ",1)[0]!=obj for pid in ids):
            raise AssertionError(f'Candidate contamination at crop {idx}: {obj} vs {ids}')
        tok=torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32,device=device)
        fg=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool,device=device).reshape(-1)
        if tok.shape!=(NUM_PATCHES,base.shape[1]) or fg.shape!=(NUM_PATCHES,) or not bool(fg.any()):
            raise RuntimeError(f'Invalid cache tokens/fg at crop {idx}')
        tok=F.normalize(tok,dim=-1)
        # GT is loaded only AFTER pseudo assignments/prototypes for every epoch are computed.
        with torch.no_grad():
            s=score_all_epochs(text,ids,tok,fg,args.tau,args.nce_tau,frozen,
                args.rank_delta,args.rank_margin,check_source=args.verify_source and not verified)
        if args.verify_source and not verified:verified=True;print('[check] original trainer forward equivalence PASS',flush=True)
        gt,flipped,wh,box=lookup.crop32(a,args.flip_mode)
        gt=np.asarray(gt,dtype=np.int16).reshape(-1);fg_np=fg.detach().cpu().numpy().astype(bool)
        tok_np=tok.detach().cpu().numpy()
        if len(gt)!=NUM_PATCHES:raise AssertionError('GT patch grid mismatch')
        if np.any((gt!=255)&((gt<0)|(gt>=116))):raise RuntimeError(f'Bad GT labels at {idx}')
        # OFFLINE-only: report classes with GT pixels but OMITTED from the image-level presence candidate list.
        # Never pass this table back to the trainer or to pseudo-label scoring.
        obj_all=[j for j,n in enumerate(names) if n.split("'s ",1)[0]==obj]
        offline_gt_presence=[]
        for qid in obj_all:
            mask=(gt==qid)
            if mask.any():
                offline_gt_presence.append((idx,qid,int(mask.sum()),int((mask&fg_np).sum()),int(qid in ids)))
        # Fixed E0-to-E_last transitions: unlike consecutive flips, these do NOT double count rebounds.
        # GT only tells the audit which patches to measure, never determines pseudo-label support.
        endpoint=[]
        for pos,pid in enumerate(ids):
            g=(gt==pid)&fg_np
            k=len(ids)
            old=s['selector'][0,pos];new=s['selector'][-1,pos]
            oldw=s['weight'][0,pos];neww=s['weight'][-1,pos]
            rec=dict(crop_index=idx,pid=pid,part=names[pid],group=classes[pid]['类别编码'],
                from_epoch=epochs[0],to_epoch=epochs[-1],n_gt_fg=int(g.sum()),
                gained_selected_gt=int(((~old)&new&g).sum()),lost_selected_gt=int((old&(~new)&g).sum()),
                gained_selected_other=int(((~old)&new&fg_np&(gt!=255)&(gt!=pid)).sum()),
                lost_selected_other=int((old&(~new)&fg_np&(gt!=255)&(gt!=pid)).sum()),
                delta_q_gt=float(neww[gt==pid].sum()-oldw[gt==pid].sum()),
                delta_q_other=float(neww[fg_np&(gt!=255)&(gt!=pid)].sum()-oldw[fg_np&(gt!=255)&(gt!=pid)].sum()),
                delta_q_ignore=float(neww[gt==255].sum()-oldw[gt==255].sum()),
                support_jaccard=float((old&new).sum()/max(1,(old|new).sum())),
                weight_total_variation=float(.5*np.abs(oldw-neww).sum()))
            if k>=2:
                m0=s['relative'][0,pos];m1=s['relative'][-1,pos]
                d_own=s['scores'][-1,pos]-s['scores'][0,pos]
                d_rival=s['rival_score'][-1,pos]-s['rival_score'][0,pos]
                gained=g&(m0<=0)&(m1>0)
                lost=g&(m0>0)&(m1<=0)
                # First win/persistence are observed ONLY at provided sparse checkpoints.
                win=(s['relative'][:,pos,g]>0)  # [epochs, GT-FG target patches]
                ever=win.any(axis=0)
                first_index=np.argmax(win,axis=0) if win.shape[1] else np.zeros(0,dtype=int)
                first_epochs=np.asarray(epochs)[first_index[ever]]
                lost_after_first=int(sum(bool((~win[first_index[j]+1:,j]).any()) for j in range(win.shape[1]) if ever[j]))
                rec.update(gt_ever_positive=int(ever.sum()),gt_never_positive=int((~ever).sum()),
                    mean_first_positive_observed_epoch=float(first_epochs.mean()) if len(first_epochs) else None,
                    gt_lost_after_first_win=lost_after_first,
                    gt_positive_at_final=int(win[-1].sum()) if win.shape[1] else 0)
                rec.update(gained_positive_gt=int(gained.sum()),lost_positive_gt=int(lost.sum()),
                    positive_gt_start=int(((m0>0)&g).sum()),positive_gt_end=int(((m1>0)&g).sum()),
                    delta_gt_margin=float((m1[g]-m0[g]).mean()) if g.any() else None,
                    delta_gt_own_cos=float(d_own[g].mean()) if g.any() else None,
                    delta_gt_rival_cos=float(d_rival[g].mean()) if g.any() else None,
                    competitor_switched_on_gt=int((s['rival'][0,pos,g]!=s['rival'][-1,pos,g]).sum()),
                    gain_own_up_rival_down=int((gained&(d_own>0)&(d_rival<0)).sum()),
                    gain_own_up_more=int((gained&(d_own>0)&(d_rival>=0)).sum()),
                    gain_rival_down_more=int((gained&(d_own<=0)&(d_rival<0)).sum()))
            else:
                rec.update(gained_positive_gt=None,lost_positive_gt=None,
                    positive_gt_start=None,positive_gt_end=None,delta_gt_margin=None,
                    delta_gt_own_cos=None,delta_gt_rival_cos=None,
                    competitor_switched_on_gt=None,gain_own_up_rival_down=None,
                    gain_own_up_more=None,gain_rival_down_more=None,
                    gt_ever_positive=None,gt_never_positive=None,
                    mean_first_positive_observed_epoch=None,gt_lost_after_first_win=None,
                    gt_positive_at_final=None)
            endpoint.append((idx,pid,json.dumps(rec,separators=(',',':'),ensure_ascii=False,allow_nan=False)))
        inserts=[];trans=[];done_ep=[]
        for t,ep in enumerate(epochs):
            if (idx,ep) in already:continue
            for pos,pid in enumerate(ids):
                row=part_row(a,idx,ep,pid,names[pid],classes[pid]['类别编码'],ids,pos,s,t,gt,fg_np,tok_np,args.nce_tau)
                inserts.append((idx,ep,pid,json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False)))
                if t>0:
                    tr=transition_row(a,idx,ep,pid,names[pid],classes[pid]['类别编码'],ids,pos,s,t,gt,fg_np)
                    trans.append((idx,ep,pid,json.dumps(tr,separators=(',',':'),ensure_ascii=False,allow_nan=False)))
            done_ep.append((idx,ep))
        with con:
            con.executemany('INSERT INTO part_epoch VALUES (?,?,?,?)',inserts)
            con.executemany('INSERT INTO transition VALUES (?,?,?,?)',trans)
            con.executemany('INSERT INTO epoch_done VALUES (?,?)',done_ep)
            con.executemany('INSERT OR REPLACE INTO gt_presence_coverage VALUES (?,?,?,?,?)',offline_gt_presence)
            con.executemany('INSERT OR REPLACE INTO endpoint_transition VALUES (?,?,?)',endpoint)
        done+=1
        if args.progress_every and done%args.progress_every==0:
            rate=done/max(1,time.time()-start)
            print(f'[audit] processed={done} skipped={skipped} / {len(anns)} | {rate:.2f} crops/s | elapsed={time.time()-start:.0f}s',flush=True)
    counts=dict(con.execute('SELECT epoch,COUNT(*) FROM part_epoch GROUP BY epoch'))
    checks=dict(con.execute('SELECT epoch,COUNT(*) FROM epoch_done GROUP BY epoch'))
    (out/'audit_completion.json').write_text(json.dumps(dict(status='completed',epochs=epochs,
       n_crops=len(anns),part_rows_by_epoch=counts,processed_crop_epochs=checks,
       duration_seconds=time.time()-start,source_verified=verified),indent=2),encoding='utf-8')
    con.close()
    print('[DONE] all-part audit',out,'elapsed_s',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
