#!/usr/bin/env python3
"""CPU, synthetic, no dataset or GT. Tests exact equal-annotation loss and gradients."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
src=types.ModuleType('train_relproto_alignemt')
src.DEFAULT_PATCH_COUNT=64;src.VISION_DIM=16;src.NUM_PARTS=116
src.normalize_last=lambda x:F.normalize(x,dim=-1)
def rel(a,fg):
    b,k,n=a.shape
    off=torch.eye(k,dtype=torch.bool,device=a.device).view(1,k,k,1)
    m=a-a.unsqueeze(1).expand(b,k,k,n).masked_fill(off,-torch.inf).amax(dim=2)
    return m.masked_fill(~fg[:,None,:],-torch.inf)
src.relative_scores=rel
src.prepare_examples=lambda *a,**k:None
sys.modules['train_relproto_alignemt']=src

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod
orig=load('base',ROOT/'baseline_reference.py')
train=load('train',ROOT/'ear_transfer_train.py')
names=[f'class {i}' for i in range(116)]
for i,n in [(50,'cat'),(57,'cow'),(68,'dog'),(76,'horse'),(90,'person'),(105,'sheep')]:names[i]=n+"'s ear"
for animal in ('cat','cow','dog','horse','person','sheep'):
    for mode in ('full','drop_target','target_plus_non_ear','other_five_ears','only_target','all_ears','non_ear'):
        tgt,ears,allowed=train.build_positive_mask(names,animal,mode)
        assert len(ears)==6 and tgt in ears
        if mode=='full':assert len(allowed)==116
        if mode=='drop_target':assert len(allowed)==115 and tgt not in allowed
        if mode=='target_plus_non_ear':assert len(allowed)==111 and tgt in allowed
        if mode=='other_five_ears':assert len(allowed)==5 and tgt not in allowed
        if mode=='only_target':assert allowed=={tgt}
        if mode=='all_ears':assert len(allowed)==6
        if mode=='non_ear':assert len(allowed)==110 and not set(ears)&allowed
print('PASS six target vocab sets x seven positive masks')
torch.manual_seed(123)
base=F.normalize(torch.randn(116,16),dim=-1)
W=torch.nn.Parameter(torch.eye(16))
Example=lambda p,t,f:SimpleNamespace(pids=p,tokens=t,foreground=f)
examples=[]
for ps in [[50,48,49],[57,55,56,58],[68,51,52],[90],[15,16,20],[105,99,106],[76,71,73]]:
    tok=F.normalize(torch.randn(64,16),dim=-1)
    fg=torch.zeros(64,dtype=torch.bool);fg[:32]=True
    examples.append(Example(torch.tensor(ps),tok,fg))
rels=base@base.T
args0=SimpleNamespace(soft_temperature=.05,nce_temperature=.07,lambda_struct=4.,rank_delta=.05,rank_margin=.05)
l0,diag0,parts0=orig.batch_forward(base,W,examples,device=torch.device('cpu'),frozen_relations=rels,args=args0)
g0=torch.autograd.grad(l0,W,retain_graph=True)[0]

def compute(mode,target='cow',lam=4.):
    tgt,ears,allow=train.build_positive_mask(names,target,mode)
    a=SimpleNamespace(soft_temperature=.05,nce_temperature=.07,lambda_struct=lam,rank_delta=.05,rank_margin=.05,
      _target_pid=tgt,_ear_ids=ears,_positive_allow=torch.tensor([i in allow for i in range(116)],dtype=torch.bool))
    return train.batch_forward(base,W,examples,device=torch.device('cpu'),frozen_relations=rels,args=a)
full,_,p=compute('full')
assert torch.allclose(full,l0,atol=1e-6)
assert torch.allclose(torch.autograd.grad(full,W,retain_graph=True)[0],g0,atol=1e-5)
assert torch.allclose(p['target']+p['other_ears']+p['non_ear'],p['nce'],atol=1e-6)
print('PASS original source vs modified full mode: loss AND W-gradient')
for target in ('cow','person'):
    f,_,parts=compute('full',target)
    a,_,_=compute('drop_target',target)
    b,_,_=compute('target_plus_non_ear',target)
    c,_,_=compute('other_five_ears',target)
    d,_,_=compute('only_target',target)
    no,_,_=compute('non_ear',target)
    all_e,_,_=compute('all_ears',target)
    S=4*parts['relation']
    for x,y in [(f,a+parts['target']), (f,b+parts['other_ears']),
                (a,c+parts['non_ear']), (b,d+parts['non_ear']),
                (all_e,c+d-S), (f,all_e+no-S)]:
        assert torch.allclose(x,y,atol=2e-5),(x,y)
    gf=torch.autograd.grad(f,W,retain_graph=True)[0]
    gs=[torch.autograd.grad(v,W,retain_graph=True)[0] for v in
        (parts['target'],parts['other_ears'],parts['non_ear'],parts['relation'])]
    assert torch.allclose(gf,gs[0]+gs[1]+gs[2]+4*gs[3],atol=2e-5)
    for mode in ('drop_target','target_plus_non_ear','other_five_ears','only_target','non_ear'):
        loss,_,_=compute(mode,target,0)
        assert torch.isfinite(torch.autograd.grad(loss,W)[0]).all()
print('PASS matched loss and gradient decomposition, no K renormalization')
print('SELF_TEST_PASS_EAR_TRANSFER')
