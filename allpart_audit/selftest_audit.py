#!/usr/bin/env python3
"""Tiny CPU test of ALL-PART snapshot math against actual original trainer.

No cache, GT file, GPU, or training required. Run from repository root:
    python allpart_audit/selftest_audit.py
"""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
if str(root) not in sys.path:sys.path.insert(0,str(root))
import torch
import torch.nn.functional as F
import train_relproto_alignemt as original
import train_pp116_softsingle_dual as train
from audit_all_116 import score_all_epochs

def main():
 torch.manual_seed(123)
 D,N=original.VISION_DIM,original.DEFAULT_PATCH_COUNT
 base=F.normalize(torch.randn(116,D),dim=-1)
 W=torch.eye(D)
 bank=torch.stack([base,F.normalize(base@(W+torch.randn(D,D)*.00001),dim=-1)])
 tokens=F.normalize(torch.randn(N,D),dim=-1)
 fg=torch.zeros(N,dtype=torch.bool);fg[:70]=True
 frozen=base@base.T
 for ids in ([1],[2,3],[4,5,6],[7,8,9,10]):
  states=score_all_epochs(bank,ids,tokens,fg,.05,.07,frozen,.05,.05,check_source=True)
  for t in range(2):
   assert torch.allclose(torch.tensor(states['weight'][t].sum(-1)),torch.ones(len(ids)),atol=1e-5)
   assert (states['weight'][t,:,70:]==0).all()
   assert (states['nce_logp'][t].shape==(len(ids),116))
   v=torch.tensor(states['proto'][t])
   lg=(v@bank[t].T)/.07
   ref=F.cross_entropy(lg,torch.tensor(ids),reduction='none')
   real=torch.tensor(states['nce_logp'][t])[torch.arange(len(ids)),torch.tensor(ids)]
   assert torch.allclose(-real,ref,atol=2e-5)
   if len(ids)>=3:
    target=frozen[ids][:,ids]
    vl,active,trips=train.relational_rank_loss(v[None],target[None],delta=.05,margin=.05)
    assert abs(float(vl[0])-states['struct_loss'][t])<3e-5
 print('SELF_TEST_PASS_ALLPART: original prototype, global 116-way CE, rank loss, K=1, fallback, weights')
if __name__=='__main__':main()
