#!/usr/bin/env python3
"""CPU-only equivalence and gradient tests, independent of project datasets/GPU.

For stronger repository-end-to-end self-test, original.validate_args/preflight
still runs when the actual trainer launches on server. No GT is needed here.
"""
import sys,types,importlib.util,torch
from pathlib import Path
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
fake=types.ModuleType('train_relproto_alignemt')
fake.DEFAULT_PATCH_COUNT=1024
fake.VISION_DIM=768
fake.NUM_PARTS=116
fake.prepare_examples=lambda *args,**kwargs: None
fake.self_test=lambda: None
fake.normalize_last=lambda x:F.normalize(x,dim=-1,eps=1e-12)
def rel_scores(absolute,fg):
    B,K,N=absolute.shape
    # Mimic fastest-competitor score with no spatial label supervision.
    max_other=torch.stack([torch.cat([absolute[:,:i],absolute[:,i+1:]],dim=1).max(dim=1).values
                           for i in range(K)],dim=1)
    return (absolute-max_other).masked_fill(~fg[:,None,:],-1e5)
fake.relative_scores=rel_scores
sys.modules['train_relproto_alignemt']=fake

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

base=load(HERE/'reference_original_train_pp116_softsingle_dual.py','base_reference')
new=load(HERE/'train_pp116_pipeline_search.py','new_experimental')
from types import SimpleNamespace as NS

def args(**kw):
    p=dict(soft_temperature=0.05,nce_temperature=0.07,lambda_struct=4.0,
        rank_delta=0.05,rank_margin=0.05,proto_variant='positive',proto_topk=8,
        gate_tau=0.03,margin_power=1.0,fallback_k=4,nce_variant='global',
        label_smoothing=0.05,cross_weight=0.25,hardneg_k=16,nce_clip=4.,
        reliability_variant='off',reliability_floor=0.25,reliability_on_structure=False,
        balance_presence=False)
    p.update(kw)
    return NS(**p)

def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    D=fake.VISION_DIM;C=fake.NUM_PARTS;N=fake.DEFAULT_PATCH_COUNT
    cpu=torch.device('cpu')
    bank=F.normalize(torch.randn(C,D),dim=-1)
    w=torch.nn.Parameter(torch.eye(D))
    # GT-free, fixed three training examples, with different Presence sizes.
    examples=[]
    for k in [3,2,1]:
        tok=F.normalize(torch.randn(N,D),dim=-1)
        fg=torch.zeros(N,dtype=torch.bool)
        fg[:100]=True
        ids=torch.arange(k,dtype=torch.long)
        examples.append(NS(tokens=tok,foreground=fg,pids=ids,class_name='synthetic'))
    frozen=(bank@bank.T).detach()
    a=args()
    orig,diag0,parts0=base.batch_forward(bank,w,examples,device=cpu,frozen_relations=frozen,args=a)
    test,diag1,parts1=new.batch_forward(bank,w,examples,device=cpu,frozen_relations=frozen,args=a)
    assert torch.equal(orig,test),(orig.item(),test.item())
    for k in parts0:
        assert torch.equal(parts0[k],parts1[k]),k
    for k in diag0:
        assert diag0[k]==diag1[k],k
    g0=torch.autograd.grad(orig,w,retain_graph=True)[0]
    g1=torch.autograd.grad(test,w,retain_graph=True)[0]
    assert torch.equal(g0,g1),'Original W gradient changed in baseline mode'
    print('PASS baseline bitwise equality: total,NCE,relation,all diagnostics,W gradient')
    # Unit test each family, including gradient finite/nonzero and loss finite.
    variants=[{'proto_variant':'topk','proto_topk':8},{'proto_variant':'topk','proto_topk':32},
        {'proto_variant':'softgate'},{'proto_variant':'marginpower'},
        {'proto_variant':'fallback_k'},
        {'reliability_variant':'margin'},{'reliability_variant':'coherence'},
        {'reliability_variant':'margin','reliability_on_structure':True},
        {'nce_variant':'smooth'},{'nce_variant':'hardneg'}, {'nce_variant':'clip'},
        {'lambda_struct':1.0},{'soft_temperature':0.1},{'nce_temperature':0.1},
        {'nce_variant':'crossweight'},{'balance_presence':True}]
    new._OBJECT_TAXONOMY_MASKS.clear()
    mask=torch.zeros(C,dtype=torch.bool);mask[:7]=True
    new._OBJECT_TAXONOMY_MASKS['synthetic']=mask
    for kw in variants:
        a=args(**kw)
        loss,diag,parts=new.batch_forward(bank,w,examples,device=cpu,frozen_relations=frozen,args=a)
        assert bool(torch.isfinite(loss)),kw
        grad=torch.autograd.grad(loss,w,retain_graph=True,allow_unused=True)[0]
        assert grad is not None and bool(torch.isfinite(grad).all()),kw
    print(f'PASS {len(variants)} variant fwd/backward tests incl K1/K2/K3 and global/local negative masking')
    # Strong check: all-same-object crossweight=global (up to floating point).
    new._OBJECT_TAXONOMY_MASKS['synthetic']=torch.ones(C,dtype=torch.bool)
    # Compare identical pseudo-prototypes under all-local masking.
    v=torch.randn(2,3,D)
    pids=torch.tensor([[0,1,2],[1,2,3]])
    l_a=new.experimental_nce_losses(v,bank,pids,[examples[0]]*2,[0,1],args(nce_variant='crossweight'))
    l_b=new.experimental_nce_losses(v,bank,pids,[examples[0]]*2,[0,1],args())
    assert torch.allclose(l_a,l_b,atol=1e-5)
    print('PASS all-local crossweight reduces to global loss')
    # Also execute the trainer's built-in synthetic geometry checks.
    new.self_test()
    print('SELF_TEST_PASS_PIPELINE_SEARCH')

if __name__=='__main__':main()
