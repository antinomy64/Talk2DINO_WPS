#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADE234 GT PART UNION foreground ONLY: sidecar runtime adapter; no source modification.
Uses exact existing train_relproto_soft_structure_ade234 trainer loss, W optimizer,
SVD retraction and checkpoints. GT part masks are UNIONED into foreground; this is privileged part-spatial-GT information.
"""
from __future__ import annotations
import argparse,dataclasses,sys
from pathlib import Path
import torch

def replace_fg(e,fg):
    if dataclasses.is_dataclass(e):return dataclasses.replace(e,foreground=fg)
    if hasattr(e,'_replace'):return e._replace(foreground=fg)
    if not hasattr(e,'foreground'):raise ValueError(f'Unsupported example {type(e)}')
    e.foreground=fg
    return e

def self_test():
    @dataclasses.dataclass(frozen=True)
    class Ex:
        row_index:int
        pids:torch.Tensor
        foreground:torch.Tensor
    e=Ex(0,torch.tensor([3,4]),torch.tensor([True,False]))
    f=replace_fg(e,torch.tensor([False,True]))
    assert f.pids.equal(e.pids) and f.row_index==0 and f.foreground.tolist()==[False,True]
    print('SELF_TEST_PASS_ADE_GT_PART_UNION_ADAPTER')

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--self-test-oracle',action='store_true')
    p.add_argument('--gtmask-bank',type=Path)
    opt,rest=p.parse_known_args()
    if opt.self_test_oracle:return self_test()
    if opt.gtmask_bank is None:raise ValueError('Pass --gtmask-bank')
    side=torch.load(str(opt.gtmask_bank),map_location='cpu',weights_only=False)
    meta=side['metadata'];masks=side['masks'];manifest=side['manifest']
    if meta.get('format')!='ade234_GT_part_union_foreground_oracle_v1' or not meta.get('GT_part_privileged'):
        raise RuntimeError('GT Part Union Oracle sidecar provenance invalid')
    if masks.shape!=(meta['n_records'],1024) or masks.dtype!=torch.bool or len(manifest)!=meta['n_records']:
        raise ValueError('Invalid mask bank dimensions / manifest')
    source=None
    for i,x in enumerate(rest):
        if x=='--train_dataset' and i+1<len(rest):source=Path(rest[i+1]);break
    if source is None or str(source.resolve())!=meta['source_cache']:
        raise RuntimeError(f'Explicit --train_dataset must match mask bank source cache: {meta["source_cache"]}')
    st=source.stat()
    if st.st_size!=meta['source_cache_size'] or st.st_mtime_ns!=meta['source_cache_mtime_ns']:
        raise RuntimeError('Dataset cache has changed; rebuild mask bank')
    import train_relproto_alignment_ade20kpart234 as base
    import train_relproto_soft_structure_ade234 as trainer
    old_prepare=base.prepare_examples
    accepted={'all':0,'used':0,'dropped':0}
    def patched_prepare(*args,**kwargs):
        examples,report=old_prepare(*args,**kwargs)
        accepted['all']=len(examples)
        out=[];dropped=[]
        for e in examples:
            if not hasattr(e,'row_index'):raise ValueError('Prepared ADE example lacks row_index; row ordering cannot be verified')
            ix=int(e.row_index)
            if ix<0 or ix>=len(manifest):raise IndexError(f'Invalid cache row={ix}')
            m=manifest[ix]
            if m['row_index']!=ix or set(int(v) for v in e.pids.tolist())!=set(int(v) for v in m['source_pids']):
                raise RuntimeError(f'Original ADE candidate labels mismatch at row {ix}')
            fg=masks[ix].clone()
            if not bool(fg.any()):
                dropped.append(ix)
                continue
            if tuple(e.foreground.shape)!=(1024,):raise RuntimeError(f'Foreground has wrong shape {e.foreground.shape}')
            out.append(replace_fg(e,fg))
        if dropped and meta['empty_policy']!='drop':raise RuntimeError('Unexpected empty GT part-union mask')
        if not out:raise RuntimeError('All ADE examples dropped! GT map likely wrong')
        accepted.update(used=len(out),dropped=len(dropped))
        report['GT_PART_UNION_ORACLE_PRIVILEGED']=True
        report['GT_PART_UNION_ORACLE_FOREGROUND_ONLY']='GT union-of-parts foreground replaces cached valid_patch; crops/tokens/candidate part IDs unchanged'
        report['gt_obj_oracle_examples']=len(out)
        report['gt_obj_oracle_dropped_empty']=len(dropped)
        print(f'[ADE GT PART UNION ORACLE] prepared={len(examples)} kept={len(out)} dropped_empty={len(dropped)}',flush=True)
        return out,report
    base.prepare_examples=patched_prepare
    old_ckpt=base.checkpoint_payload
    def marked_ckpt(*args,**kwargs):
        ck=old_ckpt(*args,**kwargs)
        ck.update(GT_PART_UNION_ORACLE_PRIVILEGED=True,GT_partunion_oracle_sidecar=str(opt.gtmask_bank.resolve()),
                  GT_partunion_oracle_kept=accepted['used'],GT_partunion_oracle_dropped=accepted['dropped'])
        return ck
    base.checkpoint_payload=marked_ckpt
    print('[GT PART UNION ORACLE] Uses part GT annotations (union) in training, NOT a weakly-supervised result',flush=True)
    return trainer.main(rest)

if __name__=='__main__':sys.exit(main())
