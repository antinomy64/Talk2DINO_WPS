#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-OBJECT foreground ORACLE trainer runtime adapter for train_pp116_softsingle_dual.py.

Does not modify original trainer. Replaces ONLY example.foreground immediately after
original example preparation. All cached object crops / DINO tokens / candidate PIDs,
soft relative / InfoNCE / rank loss / optimizer / per-step SVD unchanged.
Empty GT object masks are never replaced by predicted masks; records are dropped if
and only if the sidecar was generated with explicit empty-policy='drop'.
DO NOT REPORT THIS AS WEAK SUPERVISION.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import torch


def change_foreground(example, mask):
    """Replace only foreground in either dataclass, namedtuple or mutable example."""
    if dataclasses.is_dataclass(example):
        return dataclasses.replace(example,foreground=mask)
    if hasattr(example,'_replace'):
        return example._replace(foreground=mask)
    # Fallback mutable instance: avoid mutating any other field.
    if not hasattr(example,'foreground'):
        raise TypeError(f'Unsupported prepared example class: {type(example)!r}')
    example.foreground=mask
    return example


def self_test():
    @dataclasses.dataclass(frozen=True)
    class Ex:
        foreground:torch.Tensor
        pids:torch.Tensor
        row_index:int
    e=Ex(torch.tensor([True,False]),torch.tensor([4]),7)
    f=change_foreground(e,torch.tensor([False,True]))
    assert f.pids.equal(e.pids) and f.row_index==7 and f.foreground.tolist()==[False,True]
    print('SELF_TEST_PASS_GT_OBJECT_TRAINER_ADAPTER')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gtmask-bank',type=Path,required=False)
    parser.add_argument('--self-test-oracle',action='store_true')
    oracle_args,rest=parser.parse_known_args()
    if oracle_args.self_test_oracle:return self_test()
    if oracle_args.gtmask_bank is None:
        raise ValueError('Missing --gtmask-bank')
    sidecar=torch.load(oracle_args.gtmask_bank,map_location='cpu',weights_only=False)
    meta=sidecar['metadata'];mask_bank=sidecar['masks'];manifest=sidecar['manifest']
    if meta.get('format')!='pp116_gt_object_foreground_oracle_sidecar_v1' or not meta.get('oracle_privileged_gt_used'):
        raise RuntimeError('Not a GT object mask sidecar')
    if mask_bank.shape != (meta['n_records'],1024) or mask_bank.dtype!=torch.bool:
        raise RuntimeError(f'Invalid Oracle foreground mask shape/dtype {mask_bank.shape} {mask_bank.dtype}')
    if len(manifest)!=meta['n_records']:
        raise RuntimeError('Mask bank manifest length mismatch')
    cli_dataset=None
    for i,x in enumerate(rest):
        if x=='--train_dataset' and i+1<len(rest):cli_dataset=Path(rest[i+1]);break
    if cli_dataset is None:raise RuntimeError('Pass explicit --train_dataset to prevent using a different cache')
    if str(cli_dataset.resolve())!=meta['source_cache']:
        raise RuntimeError(f'Training cache mismatch: {cli_dataset.resolve()} != {meta["source_cache"]}')
    st=cli_dataset.stat()
    if st.st_size!=meta['source_cache_size'] or st.st_mtime_ns!=meta['source_cache_mtime_ns']:
        raise RuntimeError('Source cache changed since oracle-mask extraction; rebuild GT mask bank')
    import train_relproto_alignemt as original
    import train_pp116_softsingle_dual as trainer
    original_prepare=trainer._ORIGINAL_PREPARE_EXAMPLES
    if original_prepare is None:raise RuntimeError('No original preparation hook')
    used={'n':None,'dropped':None,'verified':0}
    def gt_prepare(*args,**kwargs):
        examples,report=original_prepare(*args,**kwargs)
        patched=[];dropped=[]
        for e in examples:
            if not hasattr(e,'row_index'):
                raise RuntimeError('Prepared example lacks row_index; cannot establish GT mask correspondence')
            index=int(e.row_index)
            if not 0<=index<len(manifest):
                raise RuntimeError(f'Unknown cache row {index}')
            source=manifest[index]
            if source['row_index']!=index or set(int(x) for x in e.pids.tolist()) != set(source['source_pids']):
                raise RuntimeError(f'Candidate labels or cache row order changed for example {index}')
            # No GT part mask, no additional filtering by which part occurs in crop.
            fg=mask_bank[index].clone()
            if not bool(fg.any()):
                dropped.append(index)
                continue
            if tuple(e.foreground.shape)!=(1024,) or fg.shape!=(1024,):
                raise RuntimeError(f'Unexpected foreground shape at row {index}: {e.foreground.shape}')
            patched.append(change_foreground(e,fg))
        if dropped and meta['empty_gt_policy']!='drop':
            raise RuntimeError(f'Empty GT masks detected but drop not authorized: {len(dropped)}')
        if not patched:raise RuntimeError('All training examples have empty GT foreground!')
        used.update(n=len(patched),dropped=len(dropped),verified=len(examples))
        report['GT_ORACLE_PRIVILEGED']=True
        report['GT_ORACLE_VALID_PATCH_ONLY']='GT object (not GT part) pixel ID in saved crop; original tokens and labels unchanged'
        report['oracle_example_count']=len(patched)
        report['oracle_dropped_empty_gt_rows']=len(dropped)
        print(f'[GT-OBJECT ORACLE] original prepared={len(examples)} kept={len(patched)} '
              f'dropped_empty_GT={len(dropped)}; foreground ONLY replaced',flush=True)
        return patched,report
    trainer._ORIGINAL_PREPARE_EXAMPLES=gt_prepare
    # Add explicit provenance to all checkpoints without touching original training losses.
    orig_checkpoint=original.checkpoint_payload
    def marked_checkpoint(*args,**kwargs):
        ck=orig_checkpoint(*args,**kwargs)
        ck['GT_ORACLE_PRIVILEGED']=True
        ck['GT_oracle_foreground_bank']=str(oracle_args.gtmask_bank.resolve())
        ck['GT_oracle_foreground_definition']='correct object GT ID (NOT part), saved crop, 32x32 nearest'
        ck['GT_oracle_count']=used['n']
        ck['GT_oracle_dropped_empty_gt']=used['dropped']
        return ck
    original.checkpoint_payload=marked_checkpoint
    # Official trainer parses exactly the same hyperparameters as the baseline.
    print('[PRIVILEGED ORACLE] This run uses object GT masks during training; never label it weakly supervised.',flush=True)
    # Route parser flags untouched to the exact existing training entrypoint.
    return trainer.main(rest)

if __name__=='__main__':
    sys.exit(main())
