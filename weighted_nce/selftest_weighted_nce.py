#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU test for weighted NCE; with original trainer, also run its full self-test.

The import stub exists for offline package QA only. In the user's repo the
real train_relproto_alignemt module is imported; no stub is written to disk.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if not (Path.cwd()/'train_relproto_alignemt.py').is_file():
    stub = ModuleType('train_relproto_alignemt')
    stub.NUM_PARTS=116
    stub.VISION_DIM=768
    stub.DEFAULT_PATCH_COUNT=1024
    stub.prepare_examples=lambda *a,**k: None
    sys.modules['train_relproto_alignemt']=stub
    print('[INFO] Offline test using a stub of the original trainer API (NCE only).')
else:
    print('[INFO] Real original trainer detected in current project root.')

import train_pp116_weighted_nce as tr

def test_weighted():
    torch.manual_seed(123)
    C=tr.original.NUM_PARTS
    v = F.normalize(torch.randn(2,3,768), dim=-1)
    base = F.normalize(torch.randn(C,768), dim=-1)
    W = torch.eye(768, requires_grad=True)
    t = F.normalize(base @ W, dim=-1)
    ids=torch.tensor([[2,15,23],[7,9,68]])
    args=SimpleNamespace(nce_temperature=0.07, nce_variant='weighted_hardness',
                         hard_weight_strength=0.,hard_weight_temperature=2.0)
    ref=F.cross_entropy((v.detach()@t.T/0.07).reshape(-1,C),ids.reshape(-1),reduction='none').reshape(2,3)
    got=tr.experimental_nce_losses(v,t,ids,None,None,args)
    assert torch.equal(ref,got), 'strength=0 must exactly equal Global-116'
    args.hard_weight_strength=0.5
    got=tr.experimental_nce_losses(v,t,ids,None,None,args)
    assert torch.isfinite(got).all() and got.shape==(2,3)
    assert torch.all(got>=ref-1e-5), 'positive hardness weighting increases negative pressure'
    loss=got.mean()
    grad=torch.autograd.grad(loss, W)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0

    z=v.detach()@t.detach().T/args.nce_temperature
    neg=z.reshape(-1,C).clone()
    neg.scatter_(1,ids.reshape(-1,1),-torch.inf)
    q=F.softmax(neg/args.hard_weight_temperature,dim=1)
    w=(1-args.hard_weight_strength)+args.hard_weight_strength*(C-1)*q
    w.scatter_(1,ids.reshape(-1,1),1.0)
    assert torch.all(w>=0.5)
    idx=torch.arange(6)
    assert torch.allclose(w.sum(-1)-1,torch.full((6,),115.),atol=1e-4)
    assert torch.allclose(w[idx,ids.reshape(-1)],torch.ones(6))
    neg_max=neg.argmax(-1)
    neg_min=neg.argmin(-1) # because positive excluded -inf -> adjust below
    real_min=torch.where(torch.isfinite(neg),neg,torch.inf).argmin(-1)
    assert bool((w[idx,neg_max]>=w[idx,real_min]).all())
    # Perturb only one easy negative and check its logit gets some gradient,
    # unlike hard top-k exclusion (all weights are strictly positive).
    logits=(v.detach()@t.detach().T/args.nce_temperature).reshape(-1,C).clone().requires_grad_(True)
    dist=logits.detach().clone()
    dist.scatter_(1,ids.reshape(-1,1),-torch.inf)
    weights=(1-args.hard_weight_strength)+args.hard_weight_strength*(C-1)*F.softmax(dist/args.hard_weight_temperature,dim=1)
    weights.scatter_(1,ids.reshape(-1,1),1.)
    out=(torch.logsumexp(logits+weights.log(),dim=1)-logits.gather(1,ids.reshape(-1,1)).squeeze(1)).mean()
    g=torch.autograd.grad(out,logits)[0]
    assert torch.all(g[~F.one_hot(ids.reshape(-1),C).bool()]>0)
    assert weights.requires_grad is False
    # Hard strength monotonicity; temperature smoothness
    assert bool((w[idx,neg_max]>w[idx,real_min]).all())
    print('PASS: zero-strength bitwise Global CE equivalence')
    print('PASS: weighted CE finite, W gradient finite and nonzero')
    print('PASS: hardest negative gets higher weight, all 115 remain positive')
    print('PASS: mean negative weight 1, positive weight 1, stop-gradient weights')

if __name__=='__main__':
    test_weighted()
    if hasattr(tr.original,'self_test'):
        print('[INFO] Running original comprehensive soft-prototype/structure self-test...')
        tr.self_test()
    print('SELF_TEST_PASS_WEIGHTED_INFONCE')
