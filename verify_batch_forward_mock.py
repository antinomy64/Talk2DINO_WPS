import sys, types, importlib, torch
from pathlib import Path
import torch.nn.functional as F
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
mock=types.ModuleType('train_relproto_alignemt')
mock.DEFAULT_PATCH_COUNT=12
mock.NUM_PARTS=116
mock.VISION_DIM=8
mock.prepare_examples=lambda *a,**k: None
mock.normalize_last=lambda x: F.normalize(x,dim=-1)
def rel(absolute,foreground):
    B,K,N=absolute.shape
    other=absolute.unsqueeze(2).expand(B,K,K,N).masked_fill(torch.eye(K,dtype=torch.bool)[None,:,:,None],-torch.inf)
    margin=absolute-other.amax(dim=2)
    return margin.masked_fill(~foreground[:,None,:],-torch.inf)
mock.relative_scores=rel
sys.modules[mock.__name__]=mock
mod=importlib.import_module('train_pp116_localreassign_globalrelation')
# Normalized text bank, fake gradients to W
bank=F.normalize(torch.randn(116,8),dim=-1)
W=torch.nn.Parameter(torch.eye(8))
class Ex:
    def __init__(self,k,start,name):
        self.tokens=F.normalize(torch.randn(12,8),dim=-1)
        self.foreground=torch.ones(12,dtype=torch.bool)
        self.foreground[-2:]=False
        self.pids=torch.tensor([start+i for i in range(k)],dtype=torch.long)
        self.class_name=name
        self.row_index=0
examples=[Ex(1,0,'cow'),Ex(2,3,'cat'),Ex(3,6,'dog'),Ex(4,10,'horse')]
from types import SimpleNamespace
args=SimpleNamespace(inner_steps=3,inner_lr=0.1,inner_backtracks=5,inner_nce_weight=1.0,inner_rank_weight=1.0,inner_anchor_weight=1.0,rank_delta=0.0,rank_margin=0.5,nce_temperature=0.07,soft_temperature=0.05,lambda_struct=4.,global_class_balance=True)
loss,metrics,sep=mod.batch_forward(bank,W,examples,device=torch.device('cpu'),frozen_relations=bank@bank.T,args=args)
assert torch.isfinite(loss).item(),loss
print('forward loss:',loss.item())
print('metrics:',metrics)
assert metrics['refine_active_fraction']==.75,metrics['refine_active_fraction']
assert metrics['global_valid_triplets']>0
assert sep['relation'].requires_grad
ncegrad=torch.autograd.grad(sep['nce'],W,retain_graph=True)[0]
relgrad=torch.autograd.grad(sep['relation'],W,retain_graph=True)[0]
assert ncegrad.norm()>0 and relgrad.norm()>0
print('W gradient norms',ncegrad.norm().item(),relgrad.norm().item())
loss.backward()
assert W.grad is not None and W.grad.norm()>0
print('BATCH_INTEGRATION_PASS')
