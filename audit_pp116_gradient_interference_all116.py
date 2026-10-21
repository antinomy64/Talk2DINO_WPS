#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""READ-ONLY, ALL-116 first-order training-gradient interference audit.

Uses the ORIGINAL PP116 presence Soft Relative + Structure loss and the SAME
frozen cached DINO tokens and projected text bank. At each requested checkpoint:

* Exact additive 116-way InfoNCE source-by-target-part decomposition:
  each crop's CE(p) has 1/K weight, each crop 1/M. Crucially, CE has the
  visual prototype DETACHED; negatives among all 116 text classes are retained.
* Exact RELATION gradient, separately grouped by crop's parent OBJECT. Relation
  includes pseudo-prototype-selection gradients and the fixed text relation
  target, exactly as train_pp116_softsingle_dual.relational_rank_loss.
* For each source gradient, measure the tangent-space first-order effect of a
  **hypothetical SGD descent** direction on each of 116 GT prototype margins
  and (when GT masks provided) mean true-GT-patch local relative margins.

GT is loaded ONLY for offline, privileged evaluation/probe derivatives, NEVER
for constructing pseudo targets or source training gradients. This is NOT an
actual Adam trajectory or a causal re-training intervention. Results are
conditional on sampled cached crops and current checkpoint W. Full cached-data
mode: --per-part 0 (slower). Neither code nor original checkpoints are altered.

No web/network required. Requires local repo train_relproto_alignemt.py,
train_pp116_softsingle_dual.py and audit_pp116_gt_weighted_purity.py.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

C, D, N = 116, 768, 1024


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def normalized_backward(gT, base, W):
    """Convert dL/dT -> dL/dW for T = normalize(base @ W), works with leading dims."""
    u = base @ W
    t = F.normalize(u, dim=-1, eps=1e-12)
    gn = gT - t * (gT * t).sum(-1, keepdim=True)
    gn = gn / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum('ci,...cj->...ij', base, gn)


def tangent(g, W):
    """Orthogonal-group tangent projection at W; equivalent to G-W sym(W^T G)."""
    a = torch.matmul(W.T, g)
    return g - torch.matmul(W, 0.5 * (a + a.transpose(-1, -2)))


def directional_text_change(g, base, W):
    """dT per unit of negative projected-gradient norm; no optimizer/GT updates."""
    gd = tangent(g, W)
    norm = float(torch.linalg.vector_norm(gd).item())
    if norm <= 1e-20:
        return torch.zeros_like(base), norm
    direction = -gd / norm
    u = base @ W
    t = F.normalize(u, dim=-1, eps=1e-12)
    du = base @ direction
    dt = (du - t * (t * du).sum(-1, keepdim=True)) / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return dt, norm


def projected_text_and_gt(base, W, V):
    t = F.normalize(base @ W, dim=-1)
    sims = t @ V.T
    diag = sims.diagonal().clone()
    rival_sims = sims.T.contiguous()
    rival_sims[torch.arange(C, device=W.device), torch.arange(C, device=W.device)] = -torch.inf
    best, rivals = rival_sims.max(dim=-1)
    margin = diag - best
    return t, rivals, margin


def select_balanced(records, per_part, seed):
    """Deterministic fixed subset shared by ALL checkpoint comparisons."""
    if per_part == 0:
        return list(range(len(records)))
    rng = np.random.default_rng(seed)
    per_cls = defaultdict(list)
    for i, r in enumerate(records):
        for p in r['part_category_id'] if isinstance(r['part_category_id'], (list, tuple)) else flatten(r['part_category_id']):
            per_cls[int(p)].append(i)
    indices = set()
    for pid in range(C):
        choices = per_cls.get(pid, [])
        if choices:
            keep = rng.choice(choices, size=min(per_part, len(choices)), replace=False)
            indices.update(int(i) for i in keep)
    return sorted(indices)


def flatten(x):
    if isinstance(x, torch.Tensor):return x.detach().cpu().view(-1).tolist()
    if isinstance(x, np.ndarray):return x.reshape(-1).tolist()
    if isinstance(x, (list, tuple)):return list(x)
    return [x]


def one_cache_record(r, device, normalizer):
    tok = torch.as_tensor(r['cropaug_patch_tokens'], device=device, dtype=torch.float32)
    fg = torch.as_tensor(r['pred_obj_mask_patch'], device=device, dtype=torch.bool).reshape(-1)
    ids = torch.tensor([int(p) for p in flatten(r['part_category_id'])], device=device, dtype=torch.long)
    if tok.shape != (N, D) or fg.shape != (N,) or ids.numel()==0 or not fg.any():
        raise RuntimeError(f'Bad cache crop {r.get("image_id")}')
    return normalizer(tok),fg,ids


def visual_for_one(T, tok, fg, ids, tau, trainer):
    local = T[ids]
    if len(ids) == 1:
        scores = local @ tok.T
        weights = torch.softmax((scores.masked_fill(~fg[None], -torch.inf)) / tau, dim=-1)
        return F.normalize(weights @ tok, dim=-1, eps=1e-12)
    v,_=trainer.soft_relative_prototypes(local[None], tok[None], fg[None], temperature=tau)
    return v[0]


def nce_source_t_gradient(source_gT, ids, visual, T, tau, M):
    K = ids.numel()
    logits = (visual.detach() @ T.T) / tau
    probs = logits.softmax(-1)
    err = probs.clone()
    err[torch.arange(K, device=ids.device), ids] -= 1.0
    for j,p in enumerate(ids.tolist()):
        source_gT[p] += (err[j][:, None] * visual[j][None, :]) / (tau * K * M)
    return float(F.cross_entropy(logits,ids,reduction='sum').item()/K/M)


def mask_local_probe(probe_gT, target_counts, local_margin_sum, T, tok, fg, ids, labels):
    """Exact differential of mean GT local true-patch margin, with rivals fixed at current W.
    Reports only GT patch tokens INSIDE predicted foreground and candidate part present.
    """
    if len(ids)<2:return
    labels=torch.as_tensor(np.asarray(labels).copy(),device=T.device,dtype=torch.long)
    local=T[ids]
    for k,p in enumerate(ids.tolist()):
        hit=(labels == p) & fg
        if not bool(hit.any()):continue
        f=tok[hit]
        sim=local@f.T
        other_idx=[j for j in range(len(ids)) if j!=k]
        other=sim[other_idx]
        best,best_pos=other.max(0)
        rival_ids=ids[torch.tensor(other_idx, device=ids.device)][best_pos]
        n=f.shape[0]
        target_counts[p] += n
        local_margin_sum[p] += float((sim[k]-best).sum().item())
        probe_gT[p,p] += f.sum(0)
        for rival in torch.unique(rival_ids):
            probe_gT[p,int(rival)] -= f[rival_ids==rival].sum(0)


def batched_relation_grad(indices, records, base, W, trainer, src, args, M, device):
    """Exact per-parent-object 1/M * lambda * rank gradient; batch by candidate count.
    Matching training objective: rank is equal-annotation, independent of number of parts.
    """
    objects = sorted({args.names[int(flatten(records[i]['part_category_id'])[0])].split("'s ",1)[0] for i in indices})
    by_objK=defaultdict(list)
    for idx in indices:
        r=records[idx]; ids=[int(x) for x in flatten(r['part_category_id'])]
        if len(ids)<3:continue
        obj=args.names[ids[0]].split("'s ",1)[0]
        if any(args.names[j].split("'s ",1)[0]!=obj for j in ids):raise ValueError(f'Cross-object candidate cache row {idx}')
        by_objK[(obj,len(ids))].append(idx)
    gObj={obj:torch.zeros_like(W) for obj in objects}
    stats=defaultdict(lambda:defaultdict(float))
    orig_relation=(base@base.T).detach()
    for (obj,K), all_ix in sorted(by_objK.items()):
        for start in range(0,len(all_ix),args.batch_size):
            ix=all_ix[start:start+args.batch_size]
            tt,ff,pp=[],[],[]
            for i in ix:
                t,f,p=one_cache_record(records[i],device,src.normalize_last)
                tt.append(t);ff.append(f);pp.append(p)
            tokens=torch.stack(tt);fg=torch.stack(ff);pids=torch.stack(pp)
            Wv=W.detach().clone().requires_grad_(True)
            T=F.normalize(base@Wv,dim=-1)
            v,_=trainer.soft_relative_prototypes(T[pids], tokens, fg, temperature=args.soft_temperature)
            local_gr=orig_relation[pids.unsqueeze(-1),pids.unsqueeze(-2)]
            per,active,ntrips=trainer.relational_rank_loss(v,local_gr,delta=args.rank_delta,margin=args.rank_margin)
            loss=(args.lambda_struct/M)*per.sum()
            g=torch.autograd.grad(loss,Wv,allow_unused=False)[0]
            gObj[obj]+=g.detach()
            stats[obj]['n_rank_crops']+=len(ix)
            stats[obj]['rank_loss_contribution']+=float(loss.detach().item())
            stats[obj]['n_active_rank_crops']+=int((active>0).sum().item())
            del Wv,T,v,loss,g
        print(f'  [rank] {obj:<18s} crops={int(stats[obj]["n_rank_crops"]):5d}',flush=True)
    return gObj,stats


def analytic_check(base, W, records, indices, src, trainer, args, device):
    """Check exact original gradient vs independently decomposed NCE and rank on a batch."""
    ix=indices[:min(4,len(indices))]
    ex=[]
    for i in ix:
        t,f,p=one_cache_record(records[i],device,src.normalize_last)
        ex.append(SimpleNamespace(tokens=t.detach().cpu(),foreground=f.detach().cpu(),pids=p.detach().cpu()))
    Wv=W.clone().detach().requires_grad_(True)
    T=F.normalize(base@Wv,dim=-1)
    fr=(base@base.T).detach()
    total,diag,components=trainer.batch_forward(base,Wv,ex,device=device,frozen_relations=fr,args=args)
    gnce=torch.autograd.grad(components['nce'],Wv,retain_graph=True)[0]
    gstruct=torch.autograd.grad(components['relation'],Wv)[0] if components['relation'].requires_grad else torch.zeros_like(Wv)
    gT=torch.zeros((C,C,D),device=device)
    with torch.no_grad():
        for e in ex:
            ids=e.pids.to(device)
            tok=src.normalize_last(e.tokens.to(device))
            fg=e.foreground.to(device)
            v=visual_for_one(T.detach(),tok,fg,ids,args.soft_temperature,trainer)
            nce_source_t_gradient(gT,ids,v,T.detach(),args.nce_temperature,len(ex))
        analytic=normalized_backward(gT.sum(0),base,Wv.detach())
    err=float((analytic-gnce).abs().max().item())
    if err>2e-5:
        raise RuntimeError(f'InfoNCE analytic-source gradient mismatch: max_abs={err:.6g}')
    print(f'[PASS] same-trainer InfoNCE analytic decomposition: max_abs={err:.3g}; rank_grad_norm={gstruct.norm().item():.5g}',flush=True)


def self_test():
    torch.manual_seed(321)
    k,d,n=6,8,3
    base=F.normalize(torch.randn(k,d),dim=-1)
    W=torch.eye(d,requires_grad=True)
    v=F.normalize(torch.randn(n,d),dim=-1)
    ids=torch.tensor([0,2,5]);tau=0.21
    t=F.normalize(base@W,dim=-1)
    loss=F.cross_entropy((v.detach()@t.T)/tau,ids)
    gold=torch.autograd.grad(loss,W)[0]
    e=((v@t.detach().T)/tau).softmax(-1)
    e[torch.arange(n),ids]-=1
    grad_t=torch.einsum('nc,nd->cd',e,v)/(n*tau)
    got=normalized_backward(grad_t,base,W.detach())
    assert torch.allclose(gold,got,atol=1e-5,rtol=1e-5),(gold-got).abs().max()
    random_g=torch.randn_like(W)
    dt,gn=directional_text_change(random_g,base,W.detach())
    direction=-tangent(random_g,W.detach())/gn
    eps=1e-3
    actual=F.normalize(base@(W.detach()+eps*direction),dim=-1)
    approx=(actual-t.detach())/eps
    assert torch.allclose(dt,approx,atol=5e-4,rtol=0.02),(dt-approx).abs().max()
    print('SELF_TEST_PASS: exact stop-grad CE gradient; normalized text Jacobian; projected first-order slope')


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth'))
    p.add_argument('--text-bank',type=Path,default=Path('feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'))
    p.add_argument('--projector',type=Path,default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model-config',type=Path,default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--gt-prototypes',type=Path,default=Path('feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt'))
    p.add_argument('--gt-part-root',type=Path,default=Path('data/PascalPart116/annotations_detectron2_part/train'))
    p.add_argument('--classes-file',type=Path,default=Path('src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py'))
    p.add_argument('--run',type=Path,default=Path('final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep'))
    p.add_argument('--out-dir',type=Path,default=Path('output/pp116_gradient_interference_all116'))
    p.add_argument('--epochs',nargs='+',type=int,default=[0,10,30,100])
    p.add_argument('--per-part',type=int,default=12,help='Balanced sample: up to N randomly selected crops for EACH of 116 parts; 0 = ALL crops')
    p.add_argument('--sample-seed',type=int,default=123)
    p.add_argument('--batch-size',type=int,default=8,help='Relation-gradient microbatch size per object+K')
    p.add_argument('--soft-temperature',type=float,default=0.05)
    p.add_argument('--nce-temperature',type=float,default=0.07)
    p.add_argument('--lambda-struct',type=float,default=4.0)
    p.add_argument('--rank-delta',type=float,default=0.05)
    p.add_argument('--rank-margin',type=float,default=0.05)
    p.add_argument('--no-local-probe',action='store_true',help='Skip GT part masks/true-patch local-margin analysis')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--spotcheck',action='store_true',help='Check exact NCE source sum against actual original trainer autograd')
    return p.parse_args()


def main():
    a=parse()
    if a.self_test:return self_test()
    if not a.epochs or any(e<0 or e>100 for e in a.epochs):raise ValueError('Epoch must be in 0..100')
    if a.per_part<0 or a.batch_size<1:raise ValueError('per-part>=0 and batch-size>=1')
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise FileExistsError('Output dir nonempty: '+str(a.out_dir))
    import audit_pp116_gt_weighted_purity as au
    import train_relproto_alignemt as src
    import train_pp116_softsingle_dual as trainer
    required=[a.cache,a.text_bank,a.projector,a.model_config,a.gt_prototypes,a.classes_file]
    if not a.no_local_probe:required.append(a.gt_part_root)
    for p in required:
        if not p.exists():raise FileNotFoundError(p)
    device=torch.device(a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
    names=au.canonical_names(a.classes_file)
    a.names=names
    raw, raw_names, _=src.load_raw_clip_bank(a.text_bank,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if list(raw_names)!=names:raise RuntimeError('Text bank taxonomy/order differs from 116 classes')
    projector,_=src.load_frozen_projector(project_root=Path.cwd(),config_path=a.model_config,weight_path=a.projector,device=device)
    base=src.normalize_last(src.project_raw_clip_bank(raw,projector,device=device,batch_size=128).float()).detach().to(device=device, dtype=torch.float32)
    if base.shape!=(C,D):raise RuntimeError('Projected text bank dimension mismatch')
    del projector,raw
    bank=au.load_pt(a.gt_prototypes)
    if list(bank['classes'])!=names:raise RuntimeError('GT Visual prototype classes order mismatch')
    V=torch.as_tensor(bank['prototypes'],device=device,dtype=torch.float32)
    patches=torch.as_tensor(bank['patch_counts']).flatten()
    imgs=torch.as_tensor(bank['image_counts']).flatten()
    valid=(patches>0)&(imgs>0)&torch.isfinite(V).all(1).cpu()&(V.norm(dim=-1).cpu()>1e-8)
    V=F.normalize(V,dim=-1)
    records=au.annotations(au.load_pt(a.cache))
    indices=select_balanced(records,a.per_part,a.sample_seed)
    if not indices:raise RuntimeError('No training records selected')
    M=len(indices)
    nsrc=Counter()
    objs=sorted({n.split("'s ",1)[0] for n in names})
    for i in indices:
        ids=[int(p) for p in flatten(records[i]['part_category_id'])]
        for j in ids:nsrc[j]+=1
        if any(names[j].split("'s ",1)[0]!=names[ids[0]].split("'s ",1)[0] for j in ids):
            raise RuntimeError(f'Cache row {i} mixes multiple object groups')
    print('[SELECTION]',M,'/',len(records),'crops, all116 sources present=',sum(nsrc[j]>0 for j in range(C)),flush=True)
    if a.per_part and any(nsrc[j]==0 for j in range(C)):
        print('[WARNING] Some classes have zero candidate samples in this cache. They remain GT probe targets but cannot be source groups.',flush=True)
    gt_lookup=au.GTLookup(a.gt_part_root) if not a.no_local_probe else None
    # Prevent writing partial outputs before all preflight checks.
    if gt_lookup is not None:
        gt_lookup.crop32(records[indices[0]],'auto')
    epochs=sorted(set(a.epochs))
    Ws={e:au.checkpoint_W(a.run,e,a.lambda_struct).to(device) for e in epochs}
    a.out_dir.mkdir(parents=True)
    (a.out_dir/'selected_crop_indices.json').write_text(json.dumps({'indices':indices,'total_cache':len(records),'per_part_cap':a.per_part,'sample_seed':a.sample_seed}),encoding='utf-8')
    effects=[]; margins=[]; norms=[]
    for epoch in epochs:
        print(f'========== EPOCH {epoch} ========== ',flush=True)
        W=Ws[epoch]
        T,rivals,margin=projected_text_and_gt(base,W,V)
        if a.spotcheck:analytic_check(base,W,records,indices,src,trainer,a,device)
        source_gT=torch.zeros((C,C,D),device=device,dtype=torch.float32)
        local_gT=torch.zeros((C,C,D),device=device,dtype=torch.float32)
        local_n=torch.zeros(C,device=device,dtype=torch.float32)
        local_sum=torch.zeros(C,device=device,dtype=torch.float32)
        nce_total=0.0
        with torch.no_grad():
            for z,i in enumerate(indices):
                r=records[i]
                tok,fg,ids=one_cache_record(r,device,src.normalize_last)
                v=visual_for_one(T,tok,fg,ids,a.soft_temperature,trainer)
                nce_total+=nce_source_t_gradient(source_gT,ids,v,T,a.nce_temperature,M)
                if gt_lookup is not None:
                    gt,_,_,_=gt_lookup.crop32(r,'auto')
                    mask_local_probe(local_gT,local_n,local_sum,T,tok,fg,ids,gt)
                if (z+1)%500==0:print(f'  [NCE/local] {z+1}/{M} crops',flush=True)
            gNCE=normalized_backward(source_gT,base,W)
            gNCEtotal=gNCE.sum(0)
            local_mean=torch.where(local_n>0,local_sum/local_n.clamp_min(1),torch.full_like(local_sum,float('nan')))
            if gt_lookup is not None:
                local_gT=local_gT/local_n.clamp_min(1)[:,None,None]
        # Exact relation gradients grouped by parent object; GT is never used.
        relation_by_obj,rel_stats=batched_relation_grad(indices,records,base,W,trainer,src,a,M,device)
        grads={('NCE_part',names[p]):gNCE[p] for p in range(C)}
        for obj,g in relation_by_obj.items():grads[('Relation_object',obj)]=g
        grads[('NCE_ALL','ALL_116')]=gNCEtotal
        grads[('Relation_ALL','ALL_objects')]=torch.stack(list(relation_by_obj.values())).sum(0)
        grads[('Loss_ALL','ALL')]=grads['NCE_ALL','ALL_116']+grads['Relation_ALL','ALL_objects']
        for pid in range(C):
            margins.append({'epoch':epoch,'part_id':pid,'part':names[pid],
                'valid_GT_proto':int(valid[pid]),'gt_proto_margin':float(margin[pid].item()) if valid[pid] else math.nan,
                'gt_proto_top1':names[pid] if float(margin[pid].item())>0 else names[int(rivals[pid])],
                'gt_proto_best_competitor':names[int(rivals[pid])],
                'gt_local_valid_tokens_in_predFG':int(local_n[pid].item()),
                'gt_local_mean_margin':float(local_mean[pid].item()) if local_n[pid] else math.nan,
                'n_candidate_crops_in_sample':nsrc[pid]})
        for (kind,srcname),g in grads.items():
            dT,norm=directional_text_change(g,base,W)
            delta_global=((dT-dT[rivals])*V).sum(-1)
            if gt_lookup is None:delta_local=torch.full((C,),float('nan'),device=device)
            else:delta_local=torch.einsum('qjd,jd->q',local_gT,dT)
            unprojected_norm=float(torch.linalg.vector_norm(g).item())
            norms.append({'epoch':epoch,'source_kind':kind,'source':srcname,'source_sample_count':nsrc[names.index(srcname)] if kind=='NCE_part' else int(rel_stats[srcname]['n_rank_crops']) if kind=='Relation_object' else M,
                'euclidean_gradient_norm':unprojected_norm,'tangent_gradient_norm':norm})
            for p in range(C):
                effects.append({'epoch':epoch,'source_kind':kind,'source':srcname,
                    'target_id':p,'target_part':names[p],
                    'gt_proto_margin':float(margin[p].item()) if valid[p] else math.nan,
                    'first_order_gt_proto_margin_gain_per_unit':float(delta_global[p].item()) if valid[p] and norm>1e-20 else math.nan,
                    'first_order_gt_proto_margin_gain_weighted':float(delta_global[p].item()*norm) if valid[p] else math.nan,
                    'gt_local_margin':float(local_mean[p].item()) if local_n[p] else math.nan,
                    'first_order_gt_local_margin_gain_per_unit':float(delta_local[p].item()) if local_n[p] and norm>1e-20 else math.nan,
                    'first_order_gt_local_margin_gain_weighted':float(delta_local[p].item()*norm) if local_n[p] else math.nan,
                    'target_local_GT_patch_count':int(local_n[p].item())})
        print(f'[E{epoch:03d}] avg_NCE={nce_total:.5f}, GT prototype correctly top1={int(((margin>0)&valid.to(device)).sum().item())}/116',flush=True)
        write_csv(a.out_dir/'influence_all116.csv',effects)
        write_csv(a.out_dir/'target_margins.csv',margins)
        write_csv(a.out_dir/'source_gradient_norms.csv',norms)
    # Human-sized focused table: all source groups for important targets.
    focus=["cow's horn","sheep's horn","bicycle's saddle","dog's ear","aeroplane's body","person's upper arm"]
    focus_rows=[r for r in effects if r['target_part'] in focus]
    write_csv(a.out_dir/'focus_all_source_interference.csv',focus_rows)
    # All-116 mechanism panel: disjoint within/outside-object source attribution.
    lookup={(r['epoch'],r['source_kind'],r['source'],r['target_id']):r for r in effects}
    margin_lookup={(r['epoch'],r['part_id']):r for r in margins}
    mechanism=[]; top_sources=[]
    def gain(ep, kind, source, target, metric):
        r=lookup.get((ep,kind,source,target))
        return float(r[metric]) if r and math.isfinite(float(r[metric])) else 0.0
    metric_names=['first_order_gt_proto_margin_gain_weighted','first_order_gt_local_margin_gain_weighted']
    for ep in epochs:
        for target in range(C):
            obj=names[target].split("'s ",1)[0]
            v=margin_lookup[ep,target]
            r={'epoch':ep,'target_id':target,'part':names[target], 'object':obj,
               'GT_proto_margin':v['gt_proto_margin'],'GT_local_margin':v['gt_local_mean_margin'],
               'GT_local_patch_count':v['gt_local_valid_tokens_in_predFG'],
               'source_candidate_crops':v['n_candidate_crops_in_sample']}
            for metric in metric_names:
                tag='proto' if 'proto' in metric else 'local'
                if tag=='proto' and not bool(valid[target]):
                    for key in ('self_NCE','peer_NCE','other_object_NCE','own_object_Relation','other_object_Relation','NCE_total','Relation_total','Loss_total'):
                        r[tag+'_'+key]=math.nan
                    continue
                if tag=='local' and not v['gt_local_valid_tokens_in_predFG']:
                    for key in ('self_NCE','peer_NCE','other_object_NCE','own_object_Relation','other_object_Relation','NCE_total','Relation_total','Loss_total'):
                        r[tag+'_'+key]=math.nan
                    continue
                own=gain(ep,'NCE_part',names[target],target,metric)
                peer=sum(gain(ep,'NCE_part',names[j],target,metric) for j in range(C) if j!=target and names[j].split("'s ",1)[0]==obj)
                cross=sum(gain(ep,'NCE_part',names[j],target,metric) for j in range(C) if names[j].split("'s ",1)[0]!=obj)
                rank_own=gain(ep,'Relation_object',obj,target,metric)
                rank_others=sum(gain(ep,'Relation_object',o,target,metric) for o in objs if o!=obj)
                nce_all=gain(ep,'NCE_ALL','ALL_116',target,metric)
                rank_all=gain(ep,'Relation_ALL','ALL_objects',target,metric)
                loss_all=gain(ep,'Loss_ALL','ALL',target,metric)
                if abs(own+peer+cross-nce_all)>1e-5 or abs(rank_own+rank_others-rank_all)>1e-5 or abs(nce_all+rank_all-loss_all)>1e-5:
                    raise RuntimeError(f'Gradient linear-additivity check FAIL epoch={ep} target={names[target]} metric={metric}')
                r.update({tag+'_self_NCE':own,tag+'_peer_NCE':peer,tag+'_other_object_NCE':cross,
                          tag+'_own_object_Relation':rank_own,tag+'_other_object_Relation':rank_others,
                          tag+'_NCE_total':nce_all,tag+'_Relation_total':rank_all,tag+'_Loss_total':loss_all})
                contenders=[]
                for j in range(C):
                    sn=names[j]
                    if nsrc[j]==0:continue
                    contenders.append(('NCE_part',sn,gain(ep,'NCE_part',sn,target,metric)))
                for o in objs:
                    contenders.append(('Relation_object',o,gain(ep,'Relation_object',o,target,metric)))
                for order,(kind,sn,g) in enumerate(sorted(contenders,key=lambda x:x[2])[:5],1):
                    top_sources.append({'epoch':ep,'target_part':names[target],'target_id':target,'probe':tag,
                                        'effect':'most_harmful','rank':order,'source_type':kind,'source':sn,'weighted_margin_gain':g})
                for order,(kind,sn,g) in enumerate(sorted(contenders,key=lambda x:-x[2])[:5],1):
                    top_sources.append({'epoch':ep,'target_part':names[target],'target_id':target,'probe':tag,
                                        'effect':'most_helpful','rank':order,'source_type':kind,'source':sn,'weighted_margin_gain':g})
            if 0 in epochs and 100 in epochs and bool(valid[target]):
                e0=margin_lookup[0,target]['gt_proto_margin'];e100=margin_lookup[100,target]['gt_proto_margin']
                r['E0_to_E100_gt_mapping_transition']=('right' if e0>0 else 'wrong')+'->'+('right' if e100>0 else 'wrong')
            mechanism.append(r)
    write_csv(a.out_dir/'all116_mechanism_panel.csv',mechanism)
    write_csv(a.out_dir/'all116_top_harmful_helpful_sources.csv',top_sources)
    meta={'protocol':'PRIVILEGED GT OFFLINE PROBES ONLY; no GT leakage into training gradients',
          'epochs':epochs,'total_train_crops':len(records),'selected_crops':M,
          'per_part_crop_cap':a.per_part,'selection_seed':a.sample_seed,
          'K>=2':'same positive Relative margin and fallback as trainer',
          'K==1':'same soft absolute foreground pooling as trainer',
          'NCE':'exact analytic per target-label decomposition with stopgrad visual pseudo prototypes, all116 negatives, weight 1/(M*K)',
          'relation':'exact backprop through prototypes and selection, grouped by object, rank mean weight lambda/M',
          'metric':'first-order GT margin gain per unit tangent gradient; weighted gain = per-unit * tangent-gradient Frobenius norm; weighted effects are linearly additive',
          'caveats':['Not Adam step or actual retraining: no optimizer moments in partial source update',
                     'Tangent projection is linearized SVD retraction',
                     'Sampled gradients are not guaranteed to match full-data training batch updates',
                     'GT full-image visual prototype and patchwise relative margin are only diagnostics',
                     'GT crop geometry uses recorded bbox and optional hflip; unrecorded transformations may affect local GT alignment'],
          'source_count':len({e['source'] for e in effects if e['source_kind']=='NCE_part'}),
          'run':str(a.run)}
    (a.out_dir/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE] results in',a.out_dir,flush=True)

if __name__=='__main__':main()
