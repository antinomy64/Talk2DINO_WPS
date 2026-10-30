#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU tests of selection, per GT origin accounting and export, no repo or CUDA required."""
import importlib.util
import tempfile
import sqlite3
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).parent
spec=importlib.util.spec_from_file_location('audit',here/'audit_pseudo_gt_origin.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def rel(sc,fg):
    e,k,n=sc.shape
    other=sc[:,None,:,:].expand(e,k,k,n).clone()
    other.masked_fill_(torch.eye(k,dtype=torch.bool)[None,:,:,None],-torch.inf)
    m=sc-other.max(2).values
    return m.masked_fill(~fg[:,None,:],-torch.inf)

# Select all positive patches for both parts when they exist
texts=torch.zeros((2,2,768));texts[:,:,0]=torch.tensor([[1.,-1.],[1.,-1.]])
tokens=torch.zeros((1024,768));tokens[:3,0]=torch.tensor([1.,-1.,1.]); fg=torch.zeros(1024,dtype=torch.bool);fg[:3]=True
sel,w,fb,pc=audit.make_weights(texts,[0,1],tokens,fg,0.05,rel)
assert sel.shape==(2,2,1024) and sel[0,0,:3].tolist()==[True,False,True]
assert sel[0,1,:3].tolist()==[False,True,False]
assert torch.allclose(w.sum(-1),torch.ones((2,2))) and not fb.any()

# Fallback: both parts identical on all FG patches, no strictly positive margin
texts[:]=0;sel,w,fb,pc=audit.make_weights(texts,[0,1],tokens,fg,0.05,rel)
assert fb.all() and (sel.sum(-1)==1).all() and torch.allclose(w.sum(-1),torch.ones((2,2)))
# K=1 selects all FG irrespective of margin
sel,w,fb,pc=audit.make_weights(texts[:,:1],[0],tokens,fg,0.05,rel)
assert (sel.sum(-1)==3).all() and not fb.any() and torch.allclose(w.sum(-1),torch.ones((2,1)))

# Aggregate a fixture with part 0: two selected patches (GT 0, GT 1) weight (.8,.2)
with tempfile.TemporaryDirectory() as t:
    t=Path(t)
    db=audit.open_db(t/'db.sqlite',{'fixture':1},resume=False)
    db.execute('INSERT INTO crop_part VALUES (?,?,?,?,?,?,?,?,?)',(0,0,0,'fixture',2,2,2,'正Margin全部Patch',1))
    db.execute('INSERT INTO origin VALUES (?,?,?,?,?,?)',(0,0,0,0,1,.8))
    db.execute('INSERT INTO origin VALUES (?,?,?,?,?,?)',(0,0,0,1,1,.2))
    db.commit()
    names=[f"object's part_{i}" for i in range(116)]
    stats=audit.summarize(db,t,names,[0]); db.close()
    import csv
    with (t/'每个Part_每10Epoch_正确_错误_Ignore汇总.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))[0]
    assert float(rr['目标GT占选中Patch数量比例(%)'])==50.0
    assert abs(float(rr['目标GT获得Softmax权重比例(%)'])-80.0)<1e-6
    with (t/'每个Part_每10Epoch_GT来源_数量和权重占比_完整长表.csv').open(encoding='utf-8-sig',newline='') as f: allrows=list(csv.DictReader(f))
    assert len(allrows)==116*117
print('[PASS] selection K>=2, fallback, K=1, conservation, class-origin export')
