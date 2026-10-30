#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 each Part's pseudo-label source GT composition, every requested epoch.

READ ONLY: caches, W checkpoints and GT masks are never modified.
GT is loaded only after all pseudo labels of that crop have been determined.

Required original project helpers:
  train_relproto_alignemt.py
  audit_pp116_gt_weighted_purity.py  (GTLookup.crop32 / annotations / part_ids)
The predicted FG and image-level Part Presence are taken from frozen train cache.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

N_PATCHES = 1024
N_PARTS = 116
IGNORE_ID = 255


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--projected-bank',type=Path,default=None,help='Default: <run>/projected_part_bank.pt')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--gt-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--epochs',default='0,10,20,30,40,50,60,70,80,90,100')
    p.add_argument('--tau',type=float,default=0.05,help='Soft relative patch-weight temperature')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--flip-mode',choices=['auto','none','all'],default='auto')
    p.add_argument('--max-crops',type=int,default=0,help='0 = all (use e.g. 5 for smoke test)')
    p.add_argument('--out',type=Path,default=Path('output/pp116_pseudo_gt_origin_E0_E100_every10'))
    p.add_argument('--resume',action='store_true',help='Resume an interrupted audit with exactly identical inputs')
    p.add_argument('--verify-trainer',action='store_true',help='Verify first real K>=2 and K=1 sample against original proto trainer')
    p.add_argument('--progress-every',type=int,default=100)
    return p.parse_args()


def make_weights(texts, ids, tok, fg, tau, relative_fn):
    """Return selectors [E,K,1024], weights [E,K,1024], fallbacks [E,K], modes [K].

    Exact original Soft Relative: K>=2 select all positive margins;
    if no positives use argmax; K==1 select all foreground with absolute cosine.
    """
    E=texts.shape[0];K=len(ids)
    scores=torch.matmul(texts[:,ids,:],tok.T)
    if K>=2:
        margins=relative_fn(scores,fg.unsqueeze(0).expand(E,-1))
        positive=margins>0
        no_pos=~positive.any(-1)
        anchor=margins.argmax(-1)
        fallback=F.one_hot(anchor,num_classes=N_PATCHES).bool()
        selector=positive | (fallback & no_pos.unsqueeze(-1))
        w=torch.softmax((margins/tau).masked_fill(~selector,-torch.inf),dim=-1)
        pos_count=positive.sum(-1)
    else:
        selector=fg.view(1,1,-1).expand(E,1,-1)
        w=torch.softmax(scores.masked_fill(~fg.view(1,1,-1),-torch.inf)/tau,dim=-1)
        no_pos=torch.zeros((E,1),device=tok.device,dtype=torch.bool)
        pos_count=torch.full((E,1),-1,device=tok.device,dtype=torch.long)
    if not torch.isfinite(w).all():raise AssertionError('Nonfinite pseudo weights')
    sums=w.sum(-1)
    if not torch.allclose(sums,torch.ones_like(sums),atol=2e-5):
        raise AssertionError(f'Weights not normalized: maxerr={(sums-1).abs().max()}')
    return selector,w,no_pos,pos_count


def open_db(path,meta,resume):
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    if exists and not resume:
        raise FileExistsError(f'{path} already exists; choose fresh --out or add --resume')
    db=sqlite3.connect(str(path));db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(k TEXT PRIMARY KEY,v TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS done(crop_index INT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS crop_part(
      crop_index INT,epoch INT,pid INT,image_id TEXT,candidate_count INT,
      selected_patch_count INT,positive_patch_count INT,selection_mode TEXT,
      gt_target_patch_in_fg INT,
      PRIMARY KEY(crop_index,epoch,pid));
    CREATE TABLE IF NOT EXISTS origin(
      crop_index INT,epoch INT,pid INT,gt_id INT,
      selected_patch_count INT,weight_mass REAL,
      PRIMARY KEY(crop_index,epoch,pid,gt_id));
    CREATE INDEX IF NOT EXISTS origin_epoch_idx ON origin(epoch,pid,gt_id);
    ''')
    existing=db.execute("SELECT v FROM metadata WHERE k='config'").fetchone()
    if existing:
        if json.loads(existing[0])!=meta:
            raise ValueError('Existing database config differs. Choose different --out folder.')
    else:
        db.execute('INSERT INTO metadata(k,v) VALUES (?,?)',('config',json.dumps(meta,ensure_ascii=False,sort_keys=True)))
        db.commit()
    return db


def write_csv(path,headers,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)


def summarize(db,out,names,epochs):
    """Make complete 116 x epoch x (116+Ignore) outputs including zero origins."""
    # meta: counts of actual training occurrences, not eval GT samples
    parts={(int(e),int(pid)):(int(n),int(s),int(k1),int(fb),int(gt0))
           for e,pid,n,s,k1,fb,gt0 in db.execute('''SELECT epoch,pid,COUNT(*),SUM(selected_patch_count),
             SUM(CASE WHEN selection_mode='K1绝对相似度' THEN 1 ELSE 0 END),
             SUM(CASE WHEN selection_mode='Fallback单Patch' THEN 1 ELSE 0 END),
             SUM(CASE WHEN gt_target_patch_in_fg=0 THEN 1 ELSE 0 END)
             FROM crop_part GROUP BY epoch,pid''')}
    origins={(int(e),int(pid),int(gt)):(int(n),float(w))
             for e,pid,gt,n,w in db.execute('''SELECT epoch,pid,gt_id,
                SUM(selected_patch_count),SUM(weight_mass) FROM origin GROUP BY epoch,pid,gt_id''')}
    all_gt=list(range(N_PARTS))+[IGNORE_ID]
    long=[];weight_rows=[];patch_rows=[];tops=[];summ=[];mode_rows=[]
    meta_hdr=['Epoch','目标Part ID','目标Part','参与训练的Crop数','总选中Patch数','K=1 Crop数','Fallback Crop数','目标GT在预测前景中没有Patch的Crop数']
    wide_hdr=meta_hdr+[f'GT:{i:03d} {names[i]}' for i in range(N_PARTS)]+['GT:255 Ignore']
    long_hdr=['Epoch','目标Part ID','目标Part','来源GT ID','来源GT类别','来源GT类别所属Object','参与训练的Crop数',
              '这个来源GT的选中Patch数量','所有选中Patch数量','数量占比(%)',
              '这个来源GT获得的Softmax权重总和','每Crop平均Softmax权重占比(%)','K=1 Crop数','Fallback Crop数']
    for ep in epochs:
        for pid,name in enumerate(names):
            nc,nt,nk,nf,n_gt0=parts.get((ep,pid),(0,0,0,0,0))
            prefix=[ep,pid,name,nc,nt,nk,nf,n_gt0]
            w_cells=[];n_cells=[];list_top=[]
            w_total=0.;count_total=0
            for gt in all_gt:
                count,mass=origins.get((ep,pid,gt),(0,0.))
                gtname=names[gt] if gt!=IGNORE_ID else 'GT=255 Ignore（未标注）'
                gtobj=gtname.split("'s ")[0] if gt!=IGNORE_ID else 'Ignore'
                count_pct=count/nt*100 if nt else None
                mass_pct=mass/nc*100 if nc else None
                w_cells.append(round(mass_pct,6) if mass_pct is not None else '')
                n_cells.append(round(count_pct,6) if count_pct is not None else '')
                long.append([ep,pid,name,gt,gtname,gtobj,nc,count,nt,
                             round(count_pct,7) if count_pct is not None else '',
                             round(mass,9),round(mass_pct,7) if mass_pct is not None else '',nk,nf])
                if count or mass:
                    list_top.append((gt,gtname,count,count_pct,mass,mass_pct))
                w_total+=mass;count_total+=count
            if nt and nt!=count_total:raise AssertionError(f'Count conservation fail: E{ep} {name} ({nt} != {count_total})')
            if nc and abs(w_total-nc)>2e-2:raise AssertionError(f'Weight conservation fail: E{ep} {name} ({w_total} != {nc})')
            weight_rows.append(prefix+w_cells);patch_rows.append(prefix+n_cells)
            # Target, other, ignore and TOP5 ranked by weight. Don't silently drop Ignore.
            tgt_n,tgt_mass=origins.get((ep,pid,pid),(0,0.))
            ignore_n,ignore_mass=origins.get((ep,pid,IGNORE_ID),(0,0.))
            for rank,(gt,gtname,count,cp,mass,mp) in enumerate(sorted(list_top,key=lambda t:(-t[5],-t[2],t[0]))[:5],1):
                tops.append([ep,pid,name,rank,gt,gtname,count,round(cp,6),round(mp,6),nc,nf,nk])
            summ.append([ep,pid,name,nc,nt,nk,nf,n_gt0,tgt_n,
                         round(tgt_n/nt*100,6) if nt else '',
                         round(tgt_mass/nc*100,6) if nc else '',
                         round(ignore_n/nt*100,6) if nt else '',
                         round(ignore_mass/nc*100,6) if nc else '',
                         round((nt-tgt_n-ignore_n)/nt*100,6) if nt else '',
                         round((nc-tgt_mass-ignore_mass)/nc*100,6) if nc else ''])
    write_csv(out/'每个Part_每10Epoch_GT来源_数量和权重占比_完整长表.csv',long_hdr,long)
    write_csv(out/'每个Part_每10Epoch_GT来源_权重占比_宽表.csv',wide_hdr,weight_rows)
    write_csv(out/'每个Part_每10Epoch_GT来源_选中Patch数量占比_宽表.csv',wide_hdr,patch_rows)
    write_csv(out/'每个Part_每10Epoch_来源GT权重Top5.csv',
              ['Epoch','目标ID','目标Part','名次','来源GT ID','来源GT Part','选中Patch数量','数量占比(%)',
               '每Crop平均权重占比(%)','训练Crop数','Fallback Crop数','K1 Crop数'],tops)
    write_csv(out/'每个Part_每10Epoch_正确_错误_Ignore汇总.csv',
              ['Epoch','目标ID','目标Part','训练Crop数','所有选中Patch数量','K1 Crop数','Fallback Crop数','目标GT前景为空的Crop数',
               '目标GT选中Patch数','目标GT占选中Patch数量比例(%)','目标GT获得Softmax权重比例(%)',
               'Ignore占选中Patch数量比例(%)','Ignore获得Softmax权重比例(%)',
               '其他已标注Part占选中Patch数量比例(%)','其他已标注Part获得Softmax权重比例(%)'],summ)
    # breakdown for K1, fallback, and normal positive-margin, independent denominators
    for ep,pid,mode,ncrop,tot in db.execute('''SELECT epoch,pid,selection_mode,COUNT(*),SUM(selected_patch_count)
              FROM crop_part GROUP BY epoch,pid,selection_mode ORDER BY epoch,pid,selection_mode'''):
        for gt,n,mass in db.execute('''SELECT o.gt_id,SUM(o.selected_patch_count),SUM(o.weight_mass)
            FROM origin o JOIN crop_part p ON p.crop_index=o.crop_index AND p.epoch=o.epoch AND p.pid=o.pid
            WHERE p.epoch=? AND p.pid=? AND p.selection_mode=? GROUP BY o.gt_id''',(ep,pid,mode)).fetchall():
            mode_rows.append([ep,pid,names[pid],mode,ncrop,tot,gt,
                              names[gt] if gt!=IGNORE_ID else 'GT=255 Ignore',n,round(n/tot*100,6) if tot else '',
                              round(mass/ncrop*100,6) if ncrop else ''])
    write_csv(out/'按选择模式_每个Part_Epoch_GT来源.csv',
              ['Epoch','目标ID','目标Part','模式','该模式Crop数','该模式选中Patch总数','来源GT ID','来源GT Part',
               '该GT选中Patch数','该模式内数量占比(%)','该模式内每Crop平均权重占比(%)'],mode_rows)
    # per crop part origin, enabling audits to inspect examples; actual IDs, no target GT leaked to training.
    with gzip.open(out/'逐Crop_逐Part_逐Epoch_GT来源明细.csv.gz','wt',newline='',encoding='utf-8-sig',compresslevel=3) as f:
        writer=csv.writer(f)
        writer.writerow(['Crop序号','图像ID','Epoch','目标Part ID','目标Part','候选部件数',
                         '选择模式','总选中Patch数','来源GT ID','来源GT Part','该来源选中Patch数',
                         '该来源数量占比(%)','该来源Softmax权重占比(%)'])
        cur=db.execute('''SELECT p.crop_index,p.image_id,p.epoch,p.pid,p.candidate_count,p.selection_mode,
                 p.selected_patch_count,o.gt_id,o.selected_patch_count,o.weight_mass
                 FROM crop_part p JOIN origin o ON p.crop_index=o.crop_index AND p.epoch=o.epoch AND p.pid=o.pid
                 ORDER BY p.epoch,p.pid,p.crop_index,o.weight_mass DESC''')
        for idx,img,ep,pid,k,mode,total,gt,n,mass in cur:
            writer.writerow([idx,img,ep,pid,names[pid],k,mode,total,gt,
                             names[gt] if gt!=IGNORE_ID else 'GT=255 Ignore',n,
                             round(n/total*100,6) if total else '',round(mass*100,6)])
    return {'per_epoch_part':len(summ),'long_rows':len(long),'per_crop_rows':db.execute('SELECT COUNT(*) FROM origin').fetchone()[0]}


def main():
    args=options();sys.path.insert(0,str(Path.cwd()))
    import train_relproto_alignemt as src
    import audit_pp116_gt_weighted_purity as gt_helper
    epochs=sorted(set(int(s) for s in args.epochs.split(',') if s.strip()))
    if not epochs or any(ep<0 for ep in epochs):raise ValueError('Bad --epochs')
    if args.tau<=0:raise ValueError('tau must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    bankpath=args.projected_bank or args.run/'projected_part_bank.pt'
    for p in (bankpath,args.cache,args.gt_root,args.run):
        if not p.exists():raise FileNotFoundError(p)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    bank=src.torch_load(bankpath)
    raw_names=bank.get('classnames',bank.get('class_names'))
    if raw_names is None or len(raw_names)!=N_PARTS:
        raise ValueError('Projected part bank has no 116 canonical class names')
    names=[str(x) for x in raw_names]
    if len(set(names))!=N_PARTS:raise ValueError('Duplicate Part names')
    feats=torch.as_tensor(bank['features'],dtype=torch.float32,device=device)
    if feats.shape!=(116,768):raise ValueError(f'Invalid text bank shape {feats.shape}')
    base=F.normalize(feats,dim=-1)
    texts=[]
    for ep in epochs:
        if ep==0:W=torch.eye(768,device=device)
        else:
            ckpt=args.run/f'W_epoch_{ep:03d}.pt'
            if not ckpt.exists():raise FileNotFoundError(f'Missing checkpoint {ckpt}')
            data=src.torch_load(ckpt)
            W=data['W'].to(device=device,dtype=torch.float32)
            if W.shape!=(768,768):raise ValueError(f'Unexpected W shape E{ep}: {W.shape}')
            if 'epoch' in data and int(data['epoch'])!=ep:
                raise ValueError(f'Checkpoint epoch mismatch: wanted {ep}, file says {data["epoch"]}')
        texts.append(F.normalize(base@W,dim=-1))
    texts=torch.stack(texts).detach()
    anns=gt_helper.annotations(src.torch_load(args.cache))
    if args.max_crops:anns=anns[:args.max_crops]
    meta={'version':'20261028_v1','epochs':epochs,'run':str(args.run.resolve()),
          'bank':str(bankpath.resolve()),'cache':str(args.cache.resolve()),'gt_root':str(args.gt_root.resolve()),
          'tau':args.tau,'flip_mode':args.flip_mode,'max_crops':args.max_crops,
          'training_rule':'positive Relative Margin for K>=2, fallback top margin, K1 absolute softmax',
          'ground_truth_use':'OFFLINE AUDIT ONLY, after pseudo selection'}
    args.out.mkdir(parents=True,exist_ok=True)
    db=open_db(args.out/'pseudo_gt_origin.sqlite',meta,args.resume)
    (args.out/'audit_settings.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    done={int(x[0]) for x in db.execute('SELECT crop_index FROM done')}
    lookup=gt_helper.GTLookup(args.gt_root)
    start=time.time();checked_k2=False;checked_k1=False
    print(f'[start] {len(anns)} crops, epochs={epochs}, already_done={len(done)}, run={args.run}',flush=True)
    for idx,a in enumerate(anns):
        if idx in done:continue
        ids=[int(x) for x in gt_helper.part_ids(a)]
        if not ids or len(set(ids))!=len(ids) or any(pid<0 or pid>=N_PARTS for pid in ids):
            raise ValueError(f'Invalid presence Part IDs at crop={idx}: {ids}')
        if any(names[pid].split("'s ")[0]!=str(a.get('class_name','')) for pid in ids):
            raise ValueError(f'Object mismatches Part names at crop {idx}: {a.get("class_name")} vs {ids}')
        tokens=torch.as_tensor(a['cropaug_patch_tokens'],device=device,dtype=torch.float32)
        fg=torch.as_tensor(a['pred_obj_mask_patch'],device=device,dtype=torch.bool).reshape(-1)
        if tokens.shape!=(N_PATCHES,768) or fg.shape!=(N_PATCHES,) or not fg.any():
            raise ValueError(f'Invalid Crop token cache at {idx}')
        tokens=F.normalize(tokens,dim=-1)
        with torch.no_grad():
            sel,w,fb,pos=make_weights(texts,ids,tokens,fg,args.tau,src.relative_scores)
            if args.verify_trainer and (len(ids)>=2 and not checked_k2 or len(ids)==1 and not checked_k1):
                import train_pp116_softsingle_dual as tr
                if len(ids)>=2:
                    ref,diag=tr.soft_relative_prototypes(texts[:1,ids,:],tokens.unsqueeze(0),fg.unsqueeze(0),temperature=args.tau)
                    got=F.normalize(w[:1]@tokens,dim=-1)
                    if (ref-got).abs().max()>3e-5 or not torch.equal(fb[:1],diag['fallback'].bool()):
                        raise AssertionError('Real trainer disagrees with pseudo selection')
                    checked_k2=True
                else:
                    scores=(texts[:1,ids,:]@tokens.T).masked_fill(~fg.view(1,1,-1),-torch.inf)
                    if not torch.allclose(w[:1],torch.softmax(scores/args.tau,dim=-1),atol=3e-6):
                        raise AssertionError('K1 does not match absolute softmax')
                    checked_k1=True
            sel=sel.cpu().numpy().astype(bool)
            w=w.cpu().numpy()
            fb=fb.cpu().numpy().astype(bool)
            pos=pos.cpu().numpy()
        # Read GT strictly after pseudo labels are computed for ALL epochs of this crop.
        gt,flipped,wh,box=lookup.crop32(a,args.flip_mode)
        gt=np.asarray(gt,dtype=np.int16).reshape(-1)
        if len(gt)!=N_PATCHES or np.any((gt!=IGNORE_ID)&((gt<0)|(gt>=N_PARTS))):
            raise ValueError(f'Invalid GT Crop32 at {idx}; unique={np.unique(gt)}')
        gt_internal=np.where(gt==IGNORE_ID,N_PARTS,gt).astype(np.int16)
        fgnp=fg.cpu().numpy().astype(bool)
        image_id=str(a.get('image_id',''))
        info=[];orig=[]
        for t,ep in enumerate(epochs):
            for j,pid in enumerate(ids):
                support=sel[t,j];weight=w[t,j]
                count=int(support.sum())
                if count<1:raise AssertionError('No selected patch')
                is_fallback=bool(fb[t,j]) if len(ids)>=2 else False
                mode='K1绝对相似度' if len(ids)==1 else ('Fallback单Patch' if is_fallback else '正Margin全部Patch')
                info.append((idx,ep,pid,image_id,len(ids),count,int(pos[t,j]),mode,int(((gt==pid)&fgnp).sum())))
                hist=np.bincount(gt_internal[support],minlength=N_PARTS+1)
                mass=np.bincount(gt_internal,weights=weight.astype(np.float64),minlength=N_PARTS+1)
                if int(hist.sum())!=count or abs(float(mass.sum())-1.)>1e-4:
                    raise AssertionError(f'Histogram conservation failed: crop {idx} E{ep} {names[pid]}')
                for cat in np.flatnonzero((hist>0)|(mass>1e-12)):
                    gt_id=IGNORE_ID if int(cat)==N_PARTS else int(cat)
                    orig.append((idx,ep,pid,gt_id,int(hist[cat]),float(mass[cat])))
        with db:
            db.executemany('INSERT INTO crop_part VALUES (?,?,?,?,?,?,?,?,?)',info)
            db.executemany('INSERT INTO origin VALUES (?,?,?,?,?,?)',orig)
            db.execute('INSERT INTO done VALUES (?)',(idx,))
        if args.progress_every and (idx+1)%args.progress_every==0:
            print(f'[audit] crops={idx+1}/{len(anns)} | elapsed={time.time()-start:.1f}s | {round((idx+1)/max(1,time.time()-start),2)} crops/s',flush=True)
    stats=summarize(db,args.out,names,epochs)
    (args.out/'complete.json').write_text(json.dumps({'status':'complete','epochs':epochs,'done_crops':db.execute('SELECT COUNT(*) FROM done').fetchone()[0],**stats},ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE]',stats,'output=',args.out,flush=True)
    db.close()


if __name__=='__main__':
    main()
