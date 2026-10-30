#!/usr/bin/env python3
"""CPU-only semantic, gradient, scope and baseline-reproduction checks.

Run from project root: python hardest_search/selftest_hardest.py
No training dataset, validation labels, or model weights are needed.
"""
from __future__ import annotations
from pathlib import Path
import math, sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))
import train_pp116_hardest as tr
import train_pp116_softsingle_dual as orig


def A(**kwargs):
    args=dict(nce_temperature=0.07,nce_variant='global',hardneg_k=16,
        hardneg_start=115,hardneg_schedule='fixed',hardneg_warmup_epochs=5,
        hardneg_local_quota=4,same_name_weight=0.5,hardneg_mix_global=0.15,
        hardneg_gate_tau=0.5,epochs=30,_current_epoch=10,
        label_smoothing=0.05,cross_weight=0.25,nce_clip=4.,
        proto_variant='positive',proto_topk=8,gate_tau=0.03,margin_power=1.,
        fallback_k=4,soft_temperature=0.05,lambda_struct=4.,rank_delta=0.05,
        rank_margin=0.05,reliability_variant='off',reliability_floor=0.25,
        reliability_on_structure=False,balance_presence=False)
    args.update(kwargs)
    return SimpleNamespace(**args)


def nce_checks():
    torch.manual_seed(321)
    d=tr.original.VISION_DIM;C=tr.original.NUM_PARTS
    text=F.normalize(torch.randn(C,d),dim=-1)
    visual=F.normalize(torch.randn(2,3,d),dim=-1)
    pid=torch.tensor([[0,1,2],[10,11,12]],dtype=torch.long)
    logits=(visual.detach() @ text.T)/0.07
    expected=F.cross_entropy(logits.reshape(-1,C),pid.reshape(-1),reduction='none').reshape(2,3)
    got=tr.experimental_nce_losses(visual,text,pid,None,None,A())
    assert torch.equal(got,expected),'Global baseline NCE not exactly equal'
    got115=tr.experimental_nce_losses(visual,text,pid,None,None,A(nce_variant='hardneg',hardneg_k=115))
    assert torch.equal(got115,expected),'Hard115 must be exact global loss'
    # Reproduce previous Hard16 selection, numeric and gradient equivalence.
    t1=text.detach().clone().requires_grad_(True)
    logits1=(visual.detach()@t1.T)/0.07
    flat=logits1.reshape(-1,C)
    targets=pid.reshape(-1)
    neg=flat.detach().clone()
    neg.scatter_(1,targets[:,None],-torch.inf)
    selected=neg.topk(16,dim=1).indices
    keep=torch.zeros_like(flat,dtype=torch.bool)
    keep.scatter_(1,selected,True)
    keep.scatter_(1,targets[:,None],True)
    reference=F.cross_entropy(flat.masked_fill(~keep,-torch.inf),targets,reduction='none').mean()
    gr=torch.autograd.grad(reference,t1)[0]
    t2=text.detach().clone().requires_grad_(True)
    learned=tr.experimental_nce_losses(visual,t2,pid,None,None,A(nce_variant='hardneg')).mean()
    gn=torch.autograd.grad(learned,t2)[0]
    assert torch.equal(reference,learned),'Hard16 diverges from previous top16 CE'
    assert torch.equal(gr,gn),'Hard16 W-gradient proxy differs'
    print('PASS global CE bitwise; hard115 exact; hard16 CE and text gradient identical to previous search')
    # quota: 4 local mandatory, but still use global negatives for remaining 12.
    tr._OBJECT_TAXONOMY_MASKS.clear()
    for obj,start,stop in [('cow',0,9),('cat',9,17)]:
        m=torch.zeros(C,dtype=torch.bool);m[start:stop]=True
        tr._OBJECT_TAXONOMY_MASKS[obj]=m
    class Ex:
        def __init__(self,name):self.class_name=name
    batch=[Ex('cow'),Ex('cat')]
    q=tr.experimental_nce_losses(visual,text,pid,batch,[0,1],A(nce_variant='hardneg_quota'))
    assert q.shape==pid.shape and bool(torch.isfinite(q).all())
    # Explicit manually reconstructed quota on sample 0 positive pid 0
    ll=logits[0,0].detach().clone();ll[0]=-torch.inf
    loc=torch.zeros(C,dtype=torch.bool);loc[1:9]=True
    own=ll.masked_fill(~loc,-torch.inf).topk(4).indices
    mm=torch.zeros(C,dtype=torch.bool);mm[own]=True
    gl=ll.masked_fill(mm,-torch.inf).topk(12).indices
    mm[gl]=True;mm[0]=True
    ref=F.cross_entropy(logits[0,0].masked_fill(~mm,-torch.inf).unsqueeze(0),torch.tensor([0]))
    assert torch.allclose(ref,q[0,0],atol=1e-6)
    # A one-Part object (e.g. TV Monitor) has no internal negatives: still
    # construct 16 valid global negatives, never accidentally zero out NCE.
    tr._OBJECT_TAXONOMY_MASKS['solo'] = torch.zeros(C,dtype=torch.bool)
    tr._OBJECT_TAXONOMY_MASKS['solo'][0] = True
    singlev=visual[:1,:1]; singlepid=torch.tensor([[0]])
    localnone=tr.experimental_nce_losses(singlev,text,singlepid,[Ex('solo')],[0],
        A(nce_variant='hardneg_quota',hardneg_k=16,hardneg_local_quota=8))
    globhard=tr.experimental_nce_losses(singlev,text,singlepid,[Ex('solo')],[0],
        A(nce_variant='hardneg',hardneg_k=16))
    assert torch.equal(localnone,globhard),'singleton-object quota must fall back to global hard16'
    # Regression test: B=1 with K>1 had an expanded zero-stride view.
    # In PyTorch, scatter_ into that view fails unless the [B*K,C] mask
    # is materialized with clone(). The previous B=2 test missed this.
    one_visual=visual[:1]
    one_pid=torch.tensor([[0,1,2]],dtype=torch.long)
    one_text=text.detach().clone().requires_grad_(True)
    for quota in (2,4,8):
        local_one=tr.experimental_nce_losses(
            one_visual,one_text,one_pid,[Ex('cow')],[0],
            A(nce_variant='hardneg_quota',hardneg_local_quota=quota,hardneg_k=16))
        assert local_one.shape==(1,3) and bool(torch.isfinite(local_one).all())
        grad=torch.autograd.grad(local_one.mean(),one_text,retain_graph=True)[0]
        assert grad is not None and bool(torch.isfinite(grad).all())
        assert float(grad.norm())>0
    print('PASS quota regression: B=1,K=3, quota=2/4/8 and text gradients')
    print('PASS same-object quota 4 + global-hard 12; no duplicates; one true positive')
    tr._SUFFIX_CODES=torch.arange(C,dtype=torch.long)
    tr._SUFFIX_CODES[0]=1;tr._SUFFIX_CODES[15]=1
    for variant in ('hardneg_random','hardneg_easiest','hardneg_textstatic','hardneg_samename','hardneg_mixglobal','hardneg_softgate'):
        args=A(nce_variant=variant)
        before=torch.get_rng_state().clone()
        tr._NEG_GENERATOR=torch.Generator().manual_seed(123)
        out=tr.experimental_nce_losses(visual,text,pid,batch,[0,1],args)
        assert out.shape==(2,3) and bool(torch.isfinite(out).all()),variant
        if variant=='hardneg_random':assert torch.equal(before,torch.get_rng_state()),'RandomK altered DataLoader global RNG'
    mix=tr.experimental_nce_losses(visual,text,pid,batch,[0,1],A(nce_variant='hardneg_mixglobal',hardneg_mix_global=1.0))
    assert torch.allclose(mix,expected,atol=1e-6),'global mix=1 must equal global CE'
    mix0=tr.experimental_nce_losses(visual,text,pid,batch,[0,1],A(nce_variant='hardneg_mixglobal',hardneg_mix_global=0.0))
    base16=tr.experimental_nce_losses(visual,text,pid,batch,[0,1],A(nce_variant='hardneg'))
    assert torch.equal(mix0,base16),'global mix=0 must equal hard16'
    print('PASS random/easiest/static-text/same-name/mix/softgate losses finite; mix endpoints; RNG isolation')
    a=A(hardneg_schedule='linear',hardneg_start=115,hardneg_k=16,_current_epoch=1)
    assert tr.effective_hardneg_k(a)==115
    a._current_epoch=30
    assert tr.effective_hardneg_k(a)==16
    a.hardneg_schedule='cosine';a._current_epoch=1
    assert tr.effective_hardneg_k(a)==115
    a._current_epoch=30
    assert tr.effective_hardneg_k(a)==16
    a.hardneg_schedule='delayed';a._current_epoch=5
    assert tr.effective_hardneg_k(a)==115
    a._current_epoch=6
    assert tr.effective_hardneg_k(a)==16
    print('PASS dynamic K boundaries and delayed warmup')


def batch_equivalence():
    torch.manual_seed(123)
    d=tr.original.VISION_DIM;n=tr.original.DEFAULT_PATCH_COUNT;C=tr.original.NUM_PARTS
    base=F.normalize(torch.randn(C,d),dim=-1)
    W=torch.nn.Parameter(torch.eye(d))
    frozen=(base@base.T).detach()
    e=[]
    for j,k in enumerate((1,3,4)):
        tokens=F.normalize(torch.randn(n,d),dim=-1)
        mask=torch.zeros(n,dtype=torch.bool);mask[:63+j]=True
        e.append(SimpleNamespace(tokens=tokens,foreground=mask,pids=torch.tensor(list(range(j*7,j*7+k)),dtype=torch.long),class_name='cow'))
    oa=A()
    na=A()
    kwargs=dict(device=torch.device('cpu'),frozen_relations=frozen)
    actual,diaga,pa=tr.batch_forward(base,W,e,args=na,**kwargs)
    reference,diagr,pr=orig.batch_forward(base,W,e,args=oa,**kwargs)
    assert torch.equal(actual,reference),f'Original baseline total differs {actual} != {reference}'
    assert torch.equal(pa['nce'],pr['nce'])
    assert torch.equal(pa['relation'],pr['relation'])
    g1=torch.autograd.grad(actual,W,retain_graph=True)[0]
    g2=torch.autograd.grad(reference,W)[0]
    assert torch.equal(g1,g2),'Original full batch W-gradient differs'
    print('PASS full original trainer baseline: total, NCE, relation, W gradients bitwise')


def main():
    nce_checks()
    batch_equivalence()
    print('SELF_TEST_PASS_HARDEST_DEEP_SEARCH')

if __name__=='__main__':main()
