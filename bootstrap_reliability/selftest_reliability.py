#!/usr/bin/env python3
"""CPU numeric/gradient parity tests against the original Ours trainer.
Run from repository root; no pretrained weights or dataset needed.
"""
from __future__ import annotations
import copy, importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
import train_relproto_alignemt as orig
import train_pp116_bootstrap_reliability as new

spec=importlib.util.spec_from_file_location('ours_source_reference',HERE/'reference_softsingle_dual.py')
ref=importlib.util.module_from_spec(spec); sys.modules[spec.name]=ref
spec.loader.exec_module(ref)

def run():
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(20261027)
    C=orig.NUM_PARTS; D=orig.VISION_DIM; N=orig.DEFAULT_PATCH_COUNT
    bank=F.normalize(torch.randn(C,D),dim=-1)
    raw=torch.randn(D,D)
    w,_=torch.linalg.qr(raw)
    examples=[]
    for i,k in enumerate((1,2,3,4)):
        t=F.normalize(torch.randn(N,D),dim=-1)
        fg=torch.zeros(N,dtype=torch.bool);fg[:(48+i*8)]=True
        # text-bank ids all within a simulated single object taxonomy
        examples.append(SimpleNamespace(tokens=t,foreground=fg,pids=torch.arange(k),row_index=i,class_name='testobject'))
    new._OBJECT_PARTS={'testobject':list(range(6))}
    g=(bank@bank.T).detach()
    args=SimpleNamespace(soft_temperature=.05,nce_temperature=.07,lambda_struct=4.,
        rank_delta=.05,rank_margin=.05,nce_mode='global',_current_epoch=1,
        evidence='none',warmup_epochs=5,ramp_epochs=7,reliability_eta=.7,
        reliability_floor=.35,risk_step=.35,temporal_threshold=.85,cross_threshold=-.02,
        perturb_threshold=.95,perturb_drop=.15,perturb_min_fg=8,
        min_anchor_count=3,anchor_min_effective=1.5,recovery_threshold=.04)
    wa=w.clone().requires_grad_(True)
    wb=w.clone().requires_grad_(True)
    ref_total, ref_stats, ref_sep=ref.batch_forward(bank,wa,examples,device=torch.device('cpu'),frozen_relations=g,args=args)
    now_total, now_stats, now_sep=new.batch_forward(bank,wb,examples,device=torch.device('cpu'),frozen_relations=g,args=args,monitor=None)
    assert torch.equal(ref_total,now_total),(ref_total,now_total)
    assert torch.equal(ref_sep['nce'],now_sep['nce'])
    assert torch.equal(ref_sep['relation'],now_sep['relation'])
    for key in ref_stats:assert ref_stats[key]==now_stats[key],key
    ra=torch.autograd.grad(ref_total,wa)[0]
    rb=torch.autograd.grad(now_total,wb)[0]
    assert torch.equal(ra,rb),f'Baseline W gradients differ max={abs(ra-rb).max()}'
    print('PASS EXACT baseline loss, diagnostics, W gradient',flush=True)

    for mode in ('temporal','cross','temporal,cross','temporal,cross,perturb','temporal,cross,perturb,trend'):
        args.evidence=mode
        mon=new.EvidenceMonitor(examples,bank,args,torch.device('cpu'))
        mon.begin_epoch(1)
        w1=w.clone().requires_grad_(True)
        loss,stats,terms=new.batch_forward(bank,w1,examples,device=torch.device('cpu'),frozen_relations=g,args=args,monitor=mon)
        assert torch.isfinite(loss)
        grad=torch.autograd.grad(loss,w1)[0]
        assert grad is not None and torch.isfinite(grad).all(),mode
        mon.finish_epoch(w1)
        mon.begin_epoch(2)
        args._current_epoch=2
        w2=w.clone().requires_grad_(True)
        loss2,stats2,terms2=new.batch_forward(bank,w2,examples,device=torch.device('cpu'),frozen_relations=g,args=args,monitor=mon)
        assert torch.isfinite(loss2),mode
        assert torch.isfinite(torch.autograd.grad(loss2,w2)[0]).all(),mode
        detail=mon.finish_epoch(w2)
        assert all(isinstance(x,(float,int)) and (not isinstance(x,float) or torch.isfinite(torch.tensor(x))) for x in detail.values())
        print(f'PASS mode={mode} 2 epochs forward/backward/anchor/teacher',flush=True)
        args._current_epoch=1

    args.evidence='none';args.nce_mode='hard12'
    wc=w.clone().requires_grad_(True)
    lo,st,_=new.batch_forward(bank,wc,examples,device=torch.device('cpu'),frozen_relations=g,args=args)
    assert torch.isfinite(lo) and torch.isfinite(torch.autograd.grad(lo,wc)[0]).all()
    print('PASS hard12 train compatibility')
    # Analytical check: risk weights change relative Part influence while the
    # per-Crop normalization is preserved and all-one weights recover original.
    fake_ce=torch.tensor([[.2,1.5,3.0]],requires_grad=True)
    trust=torch.tensor([[1.,.35,.75]])
    eta=.7
    old=fake_ce.mean(-1)
    modified=(fake_ce*trust).sum(-1)/trust.sum(-1)
    final=(1-eta)*old+eta*modified
    assert abs(final.item()-((1-eta)*(4.7/3)+eta*(.2+.525+2.25)/2.10))<1e-6
    assert torch.isfinite(torch.autograd.grad(final.sum(),fake_ce)[0]).all()
    assert torch.allclose((fake_ce*torch.ones_like(fake_ce)).sum(-1)/3,old)
    print('PASS relative per-crop trust normalization, active weighting, finite gradient')
    print('SELF_TEST_PASS_BOOTSTRAP_RELIABILITY')

if __name__=='__main__':run()
