#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audited, narrowly-scoped recovery of the quota scatter_ bug.

Call once after replacing the trainer and creating the backup of the old trainer.
Leaves the original provenance in a separate immutable backup and archives the
incomplete, pre-first-epoch quota run; all successful runs are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

ORIGINAL_SHA = "5259aa5c1a763e2ace85ad81f54ac277412546b640995ad790f8d46f63d7927e"
FIXED_SHA = "3cc1471a24c6c59d2153fadb21e718818d3d60f97e68e1737dd5d7c75bff0ee6"
FAILED_VARIANT = 'quota_k16_local02'
BUG_TEXT = 'unsupported operation: more than one element of the written-to tensor refers to a single memory location'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default='pp116_hardest_deep_20261024_v1')
    p.add_argument('--profile',choices=['all'],default='all')
    p.add_argument('--epochs',type=int,default=30)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]{4,60}',args.tag):
        raise ValueError('Unsafe experiment tag')
    root=Path.cwd()
    script=root/'hardest_search/train_pp116_hardest.py'
    backup=root/'hardest_search/train_pp116_hardest.py.bak_before_quota_fix'
    if not backup.is_file() or sha(backup)!=ORIGINAL_SHA:
        raise RuntimeError('Original code backup missing or hash mismatched: '+str(backup))
    if not script.is_file() or sha(script)!=FIXED_SHA:
        raise RuntimeError('Patched trainer missing or hash mismatched: '+str(script))
    label=f'{args.tag}_{args.profile}_e{args.epochs}'
    report=root/'output'/label
    runs=root/'final_exp/relproto_alignment'/label
    prov=report/'provenance.json'
    if not prov.exists():
        raise RuntimeError('No original provenance: '+str(prov))
    before=json.loads(prov.read_text(encoding='utf8'))
    if before.get('label')!=label:
        raise RuntimeError('Provenance label mismatch: refuse to modify')
    prev_sha=before.get('source_sha256')
    if prev_sha not in (ORIGINAL_SHA,FIXED_SHA):
        raise RuntimeError('Unknown trainer source SHA in provenance; refuse to modify')
    if before.get('training_protocol',{}).get('epochs')!=args.epochs:
        raise RuntimeError('Epoch protocol mismatch')
    historical=report/'provenance_before_quota_fix.json'
    audit=report/'quota_fix_resume_audit.json'
    if prev_sha==ORIGINAL_SHA:
        if historical.exists() and json.loads(historical.read_text(encoding='utf8'))!=before:
            raise RuntimeError('History backup exists but contents differ')
        if not historical.exists():
            historical.write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        after=dict(before)
        after['source_sha256']=FIXED_SHA
        temp=prov.with_name('provenance.json.quota_fix_tmp')
        temp.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        os.replace(temp,prov)
        print('PROVENANCE: original saved; source_sha256 migrated with explicit audit')
    else:
        print('PROVENANCE: already migrated, skipping')
    partial=runs/FAILED_VARIANT
    log=report/'train'/f'{FAILED_VARIANT}.log'
    archived=None
    if partial.exists():
        final=partial/f'W_epoch_{args.epochs:03d}.pt'
        if final.exists() or (partial/'summary.json').exists() or (partial/'W_last.pt').exists():
            raise RuntimeError('Quota run contains completed-model artifacts: manual review needed, refusing archive')
        # No epoch-level W can be discarded without a human checking it.
        if list(partial.glob('W_epoch_*.pt')):
            raise RuntimeError('Partial quota checkpoint exists: refuse automatic restart')
        if not log.exists() or BUG_TEXT not in log.read_text(errors='replace'):
            raise RuntimeError('No matching quota scatter_ traceback: refuse to archive')
        for i in range(1,1000):
            candidate=partial.with_name(partial.name+f'.failed_pre_quota_fix_{i:03d}')
            if not candidate.exists():
                partial.rename(candidate)
                archived=str(candidate.relative_to(root))
                print('ARCHIVED incomplete quota run:',archived)
                break
        if not archived:
            raise RuntimeError('No free archive suffix')
    else:
        print('PARTIAL: no uncompleted quota dir; nothing to archive')
    log_summary={}
    for name in ('baseline_116','hard_k04','hard_k08','hard_k12','hard_k16',
                 'hard_k24','hard_k32','hard_k48','hard_k64','hard_k96','hard_k115',
                 'random_k16','static_text_k16','easy_k16'):
        d=runs/name
        if (d/f'W_epoch_{args.epochs:03d}.pt').exists() and (d/'summary.json').exists():
            log_summary[name]='completed_pre_fix'
        else:
            log_summary[name]='missing_or_incomplete'
    data={'patch_type':'single clone() after expand(B,K,C).reshape(-1,C) in hardneg_quota',
          'original_trainer_sha256':ORIGINAL_SHA,'patched_trainer_sha256':FIXED_SHA,
          'historical_provenance':str(historical.relative_to(root)) if historical.exists() else None,
          'archived_failed_quota_dir':archived,
          'existing_variant_status':log_summary,
          'note':'Only quota selection branch changed; prior 14 variants are reused without retraining. No GT supervision added.'}
    if audit.exists():
        # Preserve existing audit; only regenerate if the old failed directory needed archiving.
        print('AUDIT: already exists',audit)
    else:
        audit.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print('AUDIT:',audit)
    print('READY: run_hardest.py will reuse 14 completed variants and resume quota at 15/31')

if __name__=='__main__':
    main()
