#!/usr/bin/env python3
"""Read-only fixed-probe Ear-margin gradient decomposition.

GT masks inside audit_only_reference are read ONLY for the probe metric; never
used in training loss, prototype construction, or optimizer updates.
For each W checkpoint, compute the exact three gradient components using the
same frozen visual prototype selection as night_ear_train.py. Report tangent-
projected first-order directions on six fixed GT-ear patch margin probes.
These are local SGD-direction diagnostics, NOT Adam/SVD causal attributions.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import train_relproto_alignemt as src
import ear_transfer_train as trainer

REF_IDS = {"cat_ear":8022,"cow_ear":5740,"dog_ear":3807,"horse_ear":5302,"sheep_ear":7779,"person_ear":1161}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--projected-bank',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep/projected_part_bank.pt'))
    p.add_argument('--gradient-targets',type=str,default='', help='Optional comma-delimited targets for gradient attribution')
    p.add_argument('--gradient-epochs',type=str,default='0,1,3,10', help='Gradient calculations only at these checkpoints')
    p.add_argument('--reference-dir',type=Path,default=Path(__file__).resolve().parent/'audit_only_reference')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=str,default='0,1,2,3,4,5,6,7,8,9,10')
    p.add_argument('--probe-other',type=int,default=10)
    p.add_argument('--lambda-struct',type=float,default=4.0)
    p.add_argument('--soft-temperature',type=float,default=.05)
    p.add_argument('--nce-temperature',type=float,default=.07)
    p.add_argument('--rank-delta',type=float,default=.05)
    p.add_argument('--rank-margin',type=float,default=.05)
    p.add_argument('--device',type=str,default='cuda')
    return p.parse_args()

def orth_tangent(w,g):
    # P_W(G) = G - W sym(W^T G), since W^T W ~= I.
    wtg=w.T@g
    return g-w@((wtg+wtg.T)*.5)

def load_w(run,epoch,device):
    if epoch==0:
        return torch.eye(src.VISION_DIM,device=device)
    path=run/f'W_epoch_{epoch:03d}.pt'
    if not path.is_file():raise FileNotFoundError(path)
    payload=src.torch_load(path)
    w=payload['W'].to(device=device,dtype=torch.float32)
    if w.shape!=(src.VISION_DIM,src.VISION_DIM):raise ValueError('Bad checkpoint W shape')
    if (w.T@w-torch.eye(w.shape[0],device=device)).abs().max()>2e-3:
        raise RuntimeError('W not orthogonal')
    return w

def get_ref_masks(root):
    records={}
    for name,idx in REF_IDS.items():
        matches=list((root/name).glob(f'crop_{idx:05d}_*.npz'))
        if len(matches)!=1:raise FileNotFoundError(f'{name} crop {idx}: found {matches}')
        with np.load(matches[0]) as x:
            records[idx]={k:x[k] for k in ('gt','fg','candidate_part_ids','target_pid','crop_index')}
    return records

def load_data(args, names, device):
    masks=get_ref_masks(args.reference_dir)
    anns=src.get_annotations(src.torch_load(args.cache))
    def mk(i):
        a=anns[i]
        pids=torch.as_tensor(a['part_category_id'],dtype=torch.long).reshape(-1)
        tok=torch.as_tensor(a['cropaug_patch_tokens'],dtype=torch.float32).reshape(src.DEFAULT_PATCH_COUNT,src.VISION_DIM)
        fg=torch.as_tensor(a['pred_obj_mask_patch'],dtype=torch.bool).reshape(-1)
        assert fg.any() and all(0<=int(q)<len(names) for q in pids)
        return src.Example(row_index=i,image_id=str(a.get('image_id','')),
            ann_id=str(a.get('id',i)),class_name=str(a.get('class_name','')),
            pids=pids,tokens=tok,foreground=fg)
    refs=[mk(i) for i in REF_IDS.values()]
    for ex in refs:
        rec=masks[ex.row_index]
        if not np.array_equal(ex.pids.numpy(),rec['candidate_part_ids']):
            raise AssertionError(f'Candidate-part set mismatch at row {ex.row_index}')
        if not np.array_equal(ex.foreground.numpy(),rec['fg'].astype(bool)):
            raise AssertionError(f'Foreground mismatch at row {ex.row_index}')
        if not (np.asarray(rec['gt'])==int(rec['target_pid'])).any():
            raise AssertionError('GT ear mask missing')
    # Choose deterministic additional non-Ear crops without GT for a broader
    # batch gradient probe; all six Ear reference crops are always included.
    ear_ids=set(int(v['target_pid']) for v in masks.values())
    others=[]
    blocked=set(REF_IDS.values())
    for i,a in enumerate(anns):
        if i in blocked:continue
        ids=a.get('part_category_id',[])
        if not any(int(x) in ear_ids for x in ids):
            others.append(mk(i))
        if len(others)>=args.probe_other:break
    return refs, refs+others, masks

def margin_probe(ex,bank,W,rec,device):
    tok=F.normalize(ex.tokens.to(device),dim=-1)
    fg=ex.foreground.to(device)
    pids=ex.pids.to(device)
    pid=int(rec['target_pid'])
    if len(pids)<2:raise AssertionError('K=1 has no genuine relative margin')
    ix=(pids==pid).nonzero().view(-1)
    if len(ix)!=1:raise AssertionError('missing Ear candidate')
    t=F.normalize(bank@W,dim=-1)[pids]
    sc=t@tok.T
    other=torch.cat((sc[:int(ix[0])],sc[int(ix[0])+1:]),0)
    rival=other.max(0).values
    margin=sc[int(ix[0])]-rival
    mask=torch.from_numpy(np.asarray(rec['gt'])==pid).to(device)&fg
    if not mask.any():raise AssertionError('empty GT Ear in FG')
    m=margin[mask].mean()
    return m, float((margin[mask]>0).float().mean().detach()),float(sc[int(ix[0])][mask].mean().detach()),float(rival[mask].mean().detach())

def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float((a*b).sum()/den) if den>1e-18 else 0.0

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    args=parse_args()
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if dev.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    run=args.run
    bank_data=src.torch_load(args.projected_bank)
    base=F.normalize(bank_data['features'].float().to(dev),dim=-1)
    names=bank_data.get('classnames',bank_data.get('class_names'))
    if names is None or len(names)!=116:raise AssertionError('Missing 116-way bank')
    ear_ids=tuple(i for i,n in enumerate(names) if str(n).endswith("'s ear"))
    if len(ear_ids)!=6:raise AssertionError('Six Ear IDs required')
    refs, examples, masks=load_data(args,names,dev)
    args.output.mkdir(parents=True,exist_ok=True)
    epochs=[int(x.strip()) for x in args.epochs.split(',') if x.strip()]
    grad_epochs={int(x.strip()) for x in args.gradient_epochs.split(',') if x.strip()}
    rows=[];grad_rows=[]
    for ep in epochs:
        wm=load_w(run,ep,dev)
        W=torch.nn.Parameter(wm.detach().clone())
        # GT masks never touch the loss; they enter only after all visual selection.
        for ref in refs:
            with torch.no_grad():
                m,coverage,ear_cos,rival_cos=margin_probe(ref,base,W,masks[ref.row_index],dev)
                tok=F.normalize(ref.tokens.to(dev),dim=-1)
                fg=ref.foreground.to(dev)
                pids=ref.pids.to(dev)
                target_pid=int(masks[ref.row_index]['target_pid'])
                k=pids.numel()
                score=F.normalize(base@W,dim=-1)[pids]@tok.T
                rel=src.relative_scores(score[None],fg[None])[0]
                pos=(rel>0)
                empty=~pos.any(dim=1)
                anc=rel.argmax(dim=1)
                supp=pos.clone()
                supp[empty,anc[empty]]=True
                weight=torch.softmax((rel/args.soft_temperature).masked_fill(~supp,-torch.inf),dim=1)
                idx=int((pids==target_pid).nonzero().item())
                gt=torch.as_tensor(masks[ref.row_index]['gt'],device=dev).reshape(-1)
                valid_fg=fg & (gt!=255)
                gtmask=fg & (gt==target_pid)
                othermask=valid_fg & (gt!=target_pid)
                wgt=weight[idx]
                row={'run':run.name,'epoch':ep,'part':str(names[target_pid]),
                     'crop_index':ref.row_index,'image_id':ref.image_id,
                     'gt_ear_fg_patches':int(gtmask.sum()),
                     'gt_positive_count':int((pos[idx]&gtmask).sum()),
                     'gt_positive_fraction':coverage,
                     'mean_gt_margin':float(m.detach()),
                     'mean_gt_ear_cosine':ear_cos,'mean_gt_rival_cosine':rival_cos,
                     'weighted_gt_purity':float(wgt[gtmask].sum()),
                     'weighted_ignore_mass':float(wgt[fg & (gt==255)].sum()),
                     'other_gt_fg_positive_fraction':float(pos[idx][othermask].float().mean()) if bool(othermask.any()) else float('nan'),
                     'fallback':int(empty[idx]),
                     'selected_patches':int(supp[idx].sum())}
                rows.append(row)
        if args.gradient_targets and ep in grad_epochs:
            # local probe only: separate NCE by target, other five Ears, non-Ears.
            # Gradients never update W; GT is used solely for d(Margin)/dW.
            for target in args.gradient_targets.split(','):
                target=target.strip()
                if not target:continue
                tpid, eids, positive=train_params(names,target)
                ns=argparse.Namespace(soft_temperature=args.soft_temperature,
                    nce_temperature=args.nce_temperature,rank_delta=args.rank_delta,
                    rank_margin=args.rank_margin,lambda_struct=args.lambda_struct,
                    _positive_allow=torch.ones(116,dtype=torch.bool,device=dev),
                    _ear_ids=eids,_target_pid=tpid)
                full,diag,components=trainer.batch_forward(base,W,examples,device=dev,
                    frozen_relations=(base@base.T).detach(),args=ns)
                rec=components['target']+components['other_ears']+components['non_ear']
                if not torch.allclose(rec,components['nce'],atol=1e-5):
                    raise AssertionError('NCE group decomposition mismatch')
                gcomp={}
                for name in ('target','other_ears','non_ear','relation'):
                    g=torch.autograd.grad(components[name],W,retain_graph=True)[0]
                    gcomp[name]=orth_tangent(W.detach(),g)
                whole=torch.autograd.grad(full,W,retain_graph=True)[0]
                recon=sum(gcomp[n] for n in ('target','other_ears','non_ear'))+args.lambda_struct*gcomp['relation']
                error=float((orth_tangent(W.detach(),whole)-recon).norm()/(whole.norm()+1e-12))
                if error>5e-4:raise AssertionError(f'grad reconstruction failure {error}')
                target_ref=next(r for r in refs if int(masks[r.row_index]['target_pid'])==tpid)
                margin,coverage,_,_=margin_probe(target_ref,base,W,masks[target_ref.row_index],dev)
                gm=torch.autograd.grad(margin,W)[0]
                gmargin=orth_tangent(W.detach(),gm)
                row={'run':run.name,'epoch':ep,'target':target,
                    'margin_probe_crop':target_ref.row_index,
                    'margin_probe_gt_positive_fraction':coverage,
                    'gradient_reconstruction_relative_error':error}
                for name,g in gcomp.items():
                    factor=args.lambda_struct if name=='relation' else 1.
                    row[f'norm_{name}']=float((factor*g).norm())
                    row[f'directional_gain_{name}']=float(-(gmargin*factor*g).sum())
                row['directional_gain_full']=sum(row[f'directional_gain_{k}'] for k in gcomp)
                grad_rows.append(row)
        print(f'[audit] {run.name} epoch={ep} fixed_Ear={len(refs)}',flush=True)
    write_rows(args.output/'fixed_crop_ear_epoch.csv',rows)
    if grad_rows:write_rows(args.output/'gradient_by_target.csv',grad_rows)
    (args.output/'metadata.json').write_text(json.dumps({
        'run':str(run),'source_bank':str(args.projected_bank),'epochs':epochs,
        'reference_crop_indices':list(REF_IDS.values()),
        'training_uses_GT':False,
        'offline_GT_used_only_to_measure_fixed_probe_margin_and_purity':True,
        'gradient_probe_only':'tangent-projected SGD first-order; NOT Adam/SVD attribution',
        'lambda_struct':args.lambda_struct,
    },indent=2),encoding='utf-8')
    print('[DONE]',args.output,flush=True)


def train_params(names,target):
    tgt,ears,allow=trainer.build_positive_mask([str(x) for x in names],target,'full')
    return tgt,ears,allow

if __name__=='__main__':main()
