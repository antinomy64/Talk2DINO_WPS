"""Independent synthetic CPU sanity test: requires torch; no dataset/GT/GPU."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
orig = types.ModuleType('train_relproto_alignemt')
orig.DEFAULT_PATCH_COUNT = 64  # Tiny synthetic cache; both functions use this constant
orig.VISION_DIM = 16
orig.NUM_PARTS = 116
orig.normalize_last = lambda x: F.normalize(x, dim=-1)
def rel(absolute, fg):
    B,K,N=absolute.shape
    if K==1:
        r=absolute
    else:
        mask=torch.eye(K,dtype=torch.bool,device=absolute.device).view(1,K,K,1)
        r=absolute-absolute.unsqueeze(1).expand(B,K,K,N).masked_fill(mask,-torch.inf).amax(dim=2)
    return r.masked_fill(~fg[:,None,:],-torch.inf)
orig.relative_scores=rel
orig.prepare_examples=lambda *a, **kw: None
sys.modules['train_relproto_alignemt']=orig

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod
new=load('night_ear_train',ROOT/'night_ear_train.py')
# For stand-alone package test, parent user's file may not be provided on server.
old_path=ROOT/'baseline_reference.py'
if old_path.exists():
    old=load('old_reference',old_path)
else:
    old=None

torch.manual_seed(123)
B,D,N=4,16,64
base=F.normalize(torch.randn(116,D),dim=-1)
W=torch.nn.Parameter(torch.eye(D))
Example=lambda p,t,f: SimpleNamespace(pids=p,tokens=t,foreground=f)
examples=[]
for part_ids in ([50,48,49],[57,55,56,58],[13,14],[90]):
    toks=F.normalize(torch.randn(N,D),dim=-1)
    fg=torch.zeros(N,dtype=torch.bool);fg[:32]=True
    examples.append(Example(torch.tensor(part_ids),toks,fg))
relmat=base@base.T
common=dict(device=torch.device('cpu'),frozen_relations=relmat)
def compute(mode,lambda_struct=4.0):
    args=SimpleNamespace(soft_temperature=.05,nce_temperature=.07,rank_delta=.05,
        rank_margin=.05,lambda_struct=lambda_struct,positive_mode=mode,
        _ear_ids=(50,57,68,76,90,105))
    return new.batch_forward(base,W,examples,args=args,**common)
full,info,parts=compute('full')
drop,_,_=compute('drop_ear')
only,_,_=compute('only_ear')
assert torch.allclose(full,drop+only-4*parts['relation'],atol=1e-5)
assert torch.allclose(parts['ear']+parts['other'],parts['nce'],atol=1e-5)
gfull=torch.autograd.grad(full,W,retain_graph=True)[0]
gdrop=torch.autograd.grad(drop,W,retain_graph=True)[0]
gonly=torch.autograd.grad(only,W,retain_graph=True)[0]
grel=torch.autograd.grad(parts['relation'],W,retain_graph=True)[0]
assert torch.allclose(gfull,gdrop+gonly-4*grel,atol=2e-5)
for mode in ['full','drop_ear','only_ear','none']:
    loss,_,_=compute(mode)
    assert torch.isfinite(torch.autograd.grad(loss,W)[0]).all()
# Structure-only equals lambda_struct * structure gradient exactly.
none,_,_=compute('none')
assert torch.allclose(none,4*parts['relation'],atol=1e-6)
assert torch.allclose(torch.autograd.grad(none,W)[0],4*grel,atol=1e-5)
if old is not None:
    args=SimpleNamespace(soft_temperature=.05,nce_temperature=.07,rank_delta=.05,
        rank_margin=.05,lambda_struct=4.0)
    prior,_,_=old.batch_forward(base,W,examples,args=args,**common)
    assert torch.allclose(prior,full,atol=1e-6), (prior,full)
    gprior=torch.autograd.grad(prior,W)[0]
    assert torch.allclose(gprior,gfull,atol=1e-5)
    print('PASS baseline loss and grad equal original trainer')
else:
    print('SKIP baseline cross-check: baseline_reference.py not bundled')
print('PASS decomposition of loss, gradient, and mask at fixed K denominators')
print('PASS multiple-K fallback and valid gradient paths')
