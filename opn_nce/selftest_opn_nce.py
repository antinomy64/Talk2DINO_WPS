#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU self-test: run from repository root: python opn_nce/selftest_opn_nce.py

Works without project data. Checks loss, eligibility, gradients, singleton.
The full training entry point still uses the official local repository module.
"""
from __future__ import annotations
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util, sys, tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd()))

# Use a minimal shim if the repository module is not installed on the test PC.
# The project's real module is required for actual training, not this test.
try:
    import train_relproto_alignemt
except ImportError:
    m=ModuleType('train_relproto_alignemt')
    m.NUM_PARTS=116
    m.VISION_DIM=768
    m.DEFAULT_PATCH_COUNT=1024
    m.prepare_examples=lambda *a, **k: None
    sys.modules['train_relproto_alignemt']=m

script=Path(__file__).with_name('train_pp116_opn_nce.py')
spec=importlib.util.spec_from_file_location('opn_train_for_test',str(script))
tr=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def prepare_synthetic_names():
    names=["cat's eye", "cat's nose", "cat's head", "cat's ear",
           "dog's eye", "dog's head", "cow's eye", "person's eye",
           "sheep's eye", "bird's eye", "horse's eye", "car's wheel",
           "tvmonitor's screen"]
    # Every remaining name is distinct and isolated to make scope test clear.
    names += [f"dummyobj{i}'s uniquepart{i}" for i in range(116-len(names))]
    assert len(names)==len(set(names))==116
    return names


def main():
    names=prepare_synthetic_names()
    tr._TAXONOMY_CLASS_NAMES=names
    tr._TAXONOMY_NEG_MASK=tr.build_taxonomy_negative_mask(names)
    mask=tr._TAXONOMY_NEG_MASK
    eye0={names[i] for i in mask[0].nonzero(as_tuple=True)[0].tolist()}
    assert eye0=={"cat's nose", "cat's head", "cat's ear",
                  "dog's eye", "cow's eye", "person's eye", "sheep's eye",
                  "bird's eye", "horse's eye"},eye0
    assert not bool(mask[0,11]),"car's wheel must NOT be a negative for cat's eye"
    assert not bool(mask[0,0]),"The positive must not be selected as a negative"
    assert int(mask[12].sum())==0,"tvmonitor's screen has no eligible negative"
    assert bool((mask==mask.T).all()),"Union relation must be symmetric"
    print('PASS exact same-object and cross-object same-name taxonomy union')

    torch.manual_seed(8)
    v=F.normalize(torch.randn(1,3,20),dim=-1)
    t=F.normalize(torch.randn(116,20),dim=-1).detach().requires_grad_(True)
    targets=torch.tensor([[0,11,12]])
    args=SimpleNamespace(nce_variant='taxonomy_union',nce_temperature=0.07)
    got=tr.experimental_nce_losses(v,t,targets,examples=None,member_indices=None,args=args)
    assert got.shape==(1,3)
    flat=(v.detach()@t.T/args.nce_temperature).reshape(3,116)
    ref=[]
    for row,pid in enumerate(targets.reshape(-1).tolist()):
        keep=mask[pid].clone()
        keep[pid]=True
        logits=flat[row].masked_fill(~keep,-torch.inf)
        ref.append(torch.logsumexp(logits,dim=0)-logits[pid])
    reference=torch.stack(ref).reshape(1,3)
    assert torch.allclose(got,reference,atol=1e-6), (got,reference)
    assert float(got[0,2].detach())==0.0,"isolated class loss must be zero"
    print('PASS masked InfoNCE matches manual logsumexp; isolated class gives zero')

    first=tr.experimental_nce_losses(v[:,:1],t,targets[:,:1],None,None,args).sum()
    grad=torch.autograd.grad(first,t)[0]
    allowed=mask[0].clone()
    allowed[0]=True
    assert float(grad[~allowed].abs().max())==0.0, 'Unlisted classes must have exactly zero direct gradient'
    assert float(grad[allowed].abs().sum())>0.0, 'Eligible positive/negative text must have gradient'
    print('PASS unlisted classes zero direct gradient; included classes nonzero')

    with tempfile.TemporaryDirectory() as d:
        tr.write_taxonomy_negative_csv(Path(d))
        txt=(Path(d)/'taxonomy_negative_sets.csv').read_text(encoding='utf-8-sig')
        assert len(txt.splitlines())==117
        assert 'cross_object_same_name_negative_count' in txt
        assert "tvmonitor's screen" in txt
    print('PASS 116-class exported negative-set CSV')
    print('SELF_TEST_PASS_OPN_INFO_NCE')


if __name__=='__main__':
    main()
