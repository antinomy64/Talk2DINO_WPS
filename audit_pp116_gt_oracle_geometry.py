#!/usr/bin/env python3
"""PP-116 causal-capacity diagnostic: frozen-text/GT-prototype oracle.

WARNING: uses TRAIN part GT prototypes as PRIVILEGED diagnostic targets.
Never report its output as weakly-supervised training or a valid held-out score.
All outputs in a NEW --out-dir; never overwrites official files.

Compare four conditions at the *same* 116-way GT prototype classification task:
  E0 text / weakly trained E100 text / orthogonal Procrustes oracle /
  optionally projected-gradient orthogonal CE oracle / unconstrained ridge oracle.
The two GT oracle models are in-sample upper-bound diagnostics, not proofs of
held-out dense segmentation quality. Use --val-gt-bank for held-out prototype test.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

D_TEXT='feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
D_INIT='weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
D_E100='weights/vitb_mlp_infonce_coco2014_reproduce_clean_softrelative_structure_e100_reproduce.pth'
D_GT='feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt'
D_CFG='configs/vitb_mlp_infonce.yaml'


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-bank',default=D_TEXT)
    p.add_argument('--init-projector',default=D_INIT)
    p.add_argument('--e100-projector',default=D_E100)
    p.add_argument('--model-config',default=D_CFG)
    p.add_argument('--gt-bank',default=D_GT)
    p.add_argument('--val-gt-bank',default='',help='Optional separately generated validation GT bank; no training on this bank.')
    p.add_argument('--out-dir',default='output/pp116_gt_oracle_geometry_diagnostic')
    p.add_argument('--device',default='cpu')
    p.add_argument('--oracle-steps',type=int,default=0,help='Optional CE-oracle refinement with SVD after each step; use GPU if many steps.')
    p.add_argument('--oracle-lr',type=float,default=0.001)
    p.add_argument('--oracle-tau',type=float,default=0.07)
    p.add_argument('--ridge',type=float,default=0.01)
    p.add_argument('--seed',type=int,default=123)
    return p.parse_args()


def read_pt(path):
    try: return torch.load(path, map_location='cpu',weights_only=True)
    except TypeError: return torch.load(path,map_location='cpu')


def get_bank(path,names):
    data=read_pt(path)
    if not isinstance(data,dict):raise ValueError('GT bank should be a dict')
    found=data.get('classes')
    if found is None: raise KeyError(f'{path}: missing classes')
    if [str(x) for x in found] != names: raise ValueError(f'{path}: GT class order mismatch')
    for k in ('prototypes','patch_counts'):
        if k not in data:raise KeyError(f'{path}: missing {k}')
    V=torch.as_tensor(data['prototypes'],dtype=torch.float32)
    count=torch.as_tensor(data['patch_counts'],dtype=torch.int64).view(-1)
    if V.shape!=(116,768) or count.shape!=(116,):raise ValueError(f'{path}: bad shapes {V.shape}, {count.shape}')
    valid=(count>0)&torch.isfinite(V).all(1)&(V.norm(dim=1)>1e-12)
    V=V.clone();V[~valid]=0
    V[valid]=F.normalize(V[valid],dim=1)
    return V,valid,count


def top_stats(T,V,valid,names):
    # Rows of logits are GT visual parts; columns are text class predictions.
    logits=(V@T.T).cpu().numpy().astype(np.float64)
    valid_mask=valid.cpu().numpy().astype(bool)
    rows=[]
    for i,n in enumerate(names):
        if not valid_mask[i]:
            rows.append({'part':n,'valid':0,'correct':math.nan,'top1':'','gt_score':math.nan,'wrong_score':math.nan,'margin':math.nan})
            continue
        s=logits[i]
        scores_wrong=s.copy();scores_wrong[i]=-np.inf
        q=int(np.argmax(scores_wrong))
        top=i if s[i]>=scores_wrong[q] else q
        rows.append({'part':n,'valid':1,'correct':int(top==i),'top1':names[top],
                     'gt_score':float(s[i]),'wrong_score':float(scores_wrong[q]),
                     'margin':float(s[i]-scores_wrong[q])})
    good=[r for r in rows if r['valid']]
    summary={'n':len(good),'top1_correct':sum(r['correct'] for r in good),
             'top1_pct':100*sum(r['correct'] for r in good)/len(good),
             'mean_margin':float(np.mean([r['margin'] for r in good])),
             'mean_gt_cos':float(np.mean([r['gt_score'] for r in good]))}
    return rows,summary


def orthogonal_procrustes(T,V,valid):
    # Minimize ||T W - V||_F^2 for valid rows, W^T W=I.
    C=T[valid].T @ V[valid]
    U,_,Vh=torch.linalg.svd(C,full_matrices=True)
    return U@Vh


def linear_ridge(T,V,valid,lam):
    # A = I + T^T (T T^T + lam I)^(-1) (V-T) ; efficient dual ridge.
    X=T[valid];Y=V[valid]
    eye=torch.eye(X.shape[0],dtype=X.dtype,device=X.device)
    A=torch.eye(T.shape[1],dtype=T.dtype,device=T.device) + X.T @ torch.linalg.solve(X@X.T + lam*eye, Y-X)
    return A


def ce_refine(T,V,valid,W,steps,lr,tau):
    W=torch.nn.Parameter(W.clone().detach())
    opt=torch.optim.Adam([W],lr=lr)
    idx=torch.nonzero(valid).flatten()
    # CE uses all 116 text classes, same vocabulary at inference.
    for step in range(steps):
        Z=F.normalize(T@W,dim=-1)
        logits=V[idx]@Z.T/tau
        loss=F.cross_entropy(logits,idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            U,_,Vh=torch.linalg.svd(W.detach(),full_matrices=False)
            W.copy_(U@Vh)
        if step==0 or (step+1)%25==0 or step+1==steps:
            print(f'[oracle orthogonal CE] step={step+1} loss={loss.item():.6f}',flush=True)
    return W.detach().clone()


def main():
    a=cli();torch.manual_seed(a.seed)
    import train_relproto_alignemt as tr
    device=torch.device(a.device)
    root=Path.cwd()
    for p in [a.text_bank,a.init_projector,a.e100_projector,a.model_config,a.gt_bank]:
        if not Path(p).is_file(): raise FileNotFoundError(p)
    raw,names,_=tr.load_raw_clip_bank(Path(a.text_bank),expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    names=[str(x) for x in names]
    banks={}
    for tag,path in [('E0',a.init_projector),('E100',a.e100_projector)]:
        projector,_=tr.load_frozen_projector(project_root=root,config_path=Path(a.model_config),weight_path=Path(path),device=device)
        t=tr.project_raw_clip_bank(raw,projector,device=device,batch_size=128).to(device=device,dtype=torch.float32)
        t=F.normalize(t,dim=1)
        banks[tag]=t
        del projector
    T=banks['E0']
    G,valid,count=get_bank(a.gt_bank,names)
    G=G.to(device);valid=valid.to(device)
    W=orthogonal_procrustes(T,G,valid)
    banks['GT_Orthogonal_Procrustes']=F.normalize(T@W,dim=-1)
    W_ref=None
    if a.oracle_steps:
        if device.type=='cpu': print('[WARNING] Projected orthogonal CE refinement on CPU can be slow for 768x768 SVD.')
        W_ref=ce_refine(T,G,valid,W,a.oracle_steps,a.oracle_lr,a.oracle_tau)
        banks['GT_Orthogonal_CE']=F.normalize(T@W_ref,dim=-1)
    A=linear_ridge(T,G,valid,a.ridge)
    banks['GT_Linear_Ridge']=F.normalize(T@A,dim=-1)
    out=Path(a.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output directory is not empty: {out}; provide new --out-dir')
    out.mkdir(parents=True,exist_ok=True)
    summaries=[];parts=[]
    eval_banks={'train_GT_in_sample':(G,valid)}
    if a.val_gt_bank:
        vg,vvalid,_=get_bank(a.val_gt_bank,names)
        eval_banks['val_GT_heldout']=(vg.to(device),vvalid.to(device))
    for split,(ref,mask) in eval_banks.items():
        for k,t in banks.items():
            r,s=top_stats(t,ref,mask,names)
            summaries.append({'split':split,'method':k,**s})
            parts.extend([{'split':split,'method':k,**x} for x in r])
            print(f'{split:19s} {k:28s} {s["top1_correct"]:3d}/{s["n"]:3d} = {s["top1_pct"]:.2f}% meanMargin={s["mean_margin"]:+.4f}',flush=True)
    with (out/'method_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with (out/'per_part_oracle.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(parts[0]));w.writeheader();w.writerows(parts)
    torch.save({'W':W.cpu(),'source':'TRAIN GT ORACLE; DIAGNOSTIC ONLY'},out/'diagnostic_GT_Procrustes_W.pt')
    if W_ref is not None:
        torch.save({'W':W_ref.cpu(),'source':'TRAIN GT ORACLE; DIAGNOSTIC ONLY'},out/'diagnostic_GT_Orthogonal_CE_W.pt')
    protocol={
      'warning':'GT-used-for-oracle diagnostic; never include these privileged weights in weak supervision experiments',
      'train_gt_bank':a.gt_bank,'heldout_gt_bank':a.val_gt_bank or None,
      'orthogonal_procrustes':'min ||T0 W - V_GT||^2 with W orthogonal, no GT in standard training',
      'orthogonal_ce':f'optional projected SGD/Adam; SVD projection every step, CE over all 116 text classes, tau={a.oracle_tau}',
      'linear_ridge':f'unconstrained affine-free linear map, ridge regularization={a.ridge}',
      'caveat':'In-sample GT prototype matching alone does NOT establish dense segmentation performance or generalization.',
      'settings':vars(a)
    }
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__':main()
