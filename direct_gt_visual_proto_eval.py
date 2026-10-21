#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct train-GT visual-prototype segmentation, PP116 / ADE234 (PRIVILEGED ORACLE).

No text projector training and no learned W. Reuse the existing Talk2DINO
main.py evaluation, sliding/window preprocessing, label mapping and metric code.
At RUNTIME ONLY, replace the complete part-text projection with the corresponding
train-split DINOv2 GT visual prototype vectors, aligned to the original taxonomy.

Fail-closed: if the actual evaluator never requests exactly this full part bank
or the CLIP input class order cannot be verified, throw instead of reporting
an apparently successful but incorrect result. Does not edit official code.

CLI: self-test | prepare --dataset pp116/ade234 | eval --dataset ... | export ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BASE_PROJ = ROOT / 'weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
DATASETS = {
    'pp116': {
        'n': 116,
        'bank': 'feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt',
        'text': 'feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt',
        'eval_cfg': 'src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml',
        'eval_base': 'src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml',
        'ckpt': 'weights/vitb_mlp_infonce_coco2014_GTVisualDirect_pp116_DIAGNOSTIC.pth',
        'logdir': 'output/direct_gt_visual_pp116_logs',
        'eval_out': 'output/direct_gt_visual_pp116',
        'expected_images': 850,
    },
    'ade234': {
        'n': 234,
        'bank': 'feature/ade20kpart234_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool_ORACLE.pt',
        'text': 'feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt',
        'eval_cfg': 'src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml',
        'eval_base': 'src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml',
        'ckpt': 'weights/vitb_mlp_infonce_coco2014_GTVisualDirect_ade234_DIAGNOSTIC.pth',
        'logdir': 'output/direct_gt_visual_ade234_logs',
        'eval_out': 'output/direct_gt_visual_ade234',
        'expected_images': 1016,
    },
}


def full_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def require(path):
    p = full_path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing required local file (no downloads): {p}')
    return p


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_pt(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def get_names_and_raw(payload, count):
    if not isinstance(payload, dict) or 'features' not in payload:
        raise ValueError("CLIP bank must contain dict['features']")
    raw = torch.as_tensor(payload['features'], dtype=torch.float32)
    if raw.shape != (count, 512) or not bool(torch.isfinite(raw).all()):
        raise ValueError(f'Expected nonfinite-free {count}x512 raw CLIP bank, got {tuple(raw.shape)}')
    names = payload.get('class_names', payload.get('classnames', payload.get('names')))
    if names is None or len(names) != count or len(set(names)) != count:
        raise ValueError('Text class_names missing, invalid, or duplicated')
    names = [str(x) for x in names]
    if payload.get('normalized', False) is not False:
        raise ValueError('Use pre-projector, unnormalized CLIP prompt-mean text bank')
    if payload.get('clip_model', 'ViT-B/16') != 'ViT-B/16':
        raise ValueError('CLIP checkpoint incompatible with existing evaluation')
    if payload.get('template_set', 'sub_imagenet_template') != 'sub_imagenet_template':
        raise ValueError('Prompt-template convention incompatible with existing evaluation')
    return names, raw


def load_sources(dataset):
    cfg = DATASETS[dataset]
    c = cfg['n']
    bp = load_pt(require(cfg['bank']))
    tp = load_pt(require(cfg['text']))
    names, raw = get_names_and_raw(tp, c)
    if dataset == 'ade234':
        from ade20kpart234_taxonomy import PART_CLASSES
        official_names = list(PART_CLASSES)
        if names != official_names:
            raise ValueError('ADE234 class order differs from ade20kpart234_taxonomy.py')
    if not isinstance(bp, dict) or 'prototypes' not in bp:
        raise ValueError('GT bank needs a prototypes field')
    proto = torch.as_tensor(bp['prototypes'], dtype=torch.float32)
    if proto.shape != (c, 768) or not bool(torch.isfinite(proto).all()):
        raise ValueError(f'GT bank invalid tensor shape {tuple(proto.shape)}')
    bank_names = bp.get('classes', bp.get('class_names'))
    if bank_names is not None and [str(x) for x in bank_names] != names:
        raise ValueError('GT prototype bank row order differs from CLIP taxonomy')
    cnt_field = 'patch_counts' if 'patch_counts' in bp else 'counts'
    if cnt_field not in bp:
        raise ValueError('GT bank lacks patch counts: cannot determine missing prototypes')
    counts = torch.as_tensor(bp[cnt_field], dtype=torch.long)
    if tuple(counts.shape) != (c,) or bool((counts < 0).any()):
        raise ValueError(f'Invalid GT patch counts: {counts.shape}')
    norms = proto.norm(dim=1)
    valid = (counts > 0) & (norms > 1e-8)
    if bool(((counts > 0) != valid).any()):
        raise ValueError('Nonzero GT patch count with zero prototype norm')
    if not bool(valid[0]):
        raise ValueError('Taxonomy index 0 must have a prototype for exact absent-class masking')
    proto = F.normalize(proto, dim=1, eps=1e-12)
    missing = torch.nonzero(~valid).view(-1).tolist()
    # Missing prototypes have NO real vector. Use a finite placeholder to keep
    # the full C-channel score tensor valid; explicitly mask the missing channels
    # to -inf in the FINAL classwise argmax (see ReplaceProjections). Identical
    # vectors ALONE are not a safe exclusion: floating-point GEMM may round
    # equal columns differently. Do not rely on tie-breaking alone.
    proto[~valid] = proto[0].clone()
    assert all(torch.equal(proto[i], proto[0]) for i in missing)
    return proto.contiguous(), raw.contiguous(), names, counts, missing


def ckpt_path(dataset):
    return full_path(DATASETS[dataset]['ckpt'])


def meta_path(dataset):
    return Path(str(ckpt_path(dataset)) + '.direct_gtvisual.json')


def get_projector_name(dataset):
    return ckpt_path(dataset).stem


def prepare(dataset):
    cfg = DATASETS[dataset]
    proto, raw, names, counts, missing = load_sources(dataset)
    base = require(BASE_PROJ)
    for key in ('eval_cfg', 'eval_base'):
        require(cfg[key])
    require(ROOT/'src/open_vocabulary_segmentation/main.py')
    out = ckpt_path(dataset)
    side = meta_path(dataset)
    intended = {
        'format': 'DIRECT_TRAIN_GT_VISUAL_PROTOTYPES_ORACLE_v1',
        'warning': 'PRIVILEGED TRAIN GT PART SUPERVISION; NOT A WEAKLY SUPERVISED RESULT',
        'dataset': dataset, 'num_classes': cfg['n'],
        'prototype_bank': str(require(cfg['bank'])),
        'prototype_sha256': sha(require(cfg['bank'])),
        'raw_clip_bank': str(require(cfg['text'])),
        'raw_clip_sha256': sha(require(cfg['text'])),
        'base_projector': str(base), 'base_projector_sha256': sha(base),
        'source': 'train-only GT DINOv2 fullimage patchpool, no CLIP and no W for classification',
        'valid_train_prototypes': cfg['n']-len(missing),
        'missing_classes': [{'index': int(i),'name': names[i]} for i in missing],
        'missing_rule': 'finite placeholder plus strictly -inf missing channels at segmentation classwise argmax',
        'taxonomy_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        'evaluation': 'stock main.py, full all-part vocabulary, sliding 448/224, PAMR=False',
        'projector_copy_purpose': 'A unique, UNMODIFIED base projector copy to choose a distinct eval name. Runtime part embeddings are intercepted.',
    }
    if out.exists() or side.exists():
        if not (out.exists() and side.exists()):
            raise RuntimeError(f'Partially prepared diagnostic artifact: {out} / {side}')
        saved = json.loads(side.read_text(encoding='utf-8'))
        for k, v in intended.items():
            if saved.get(k) != v:
                raise RuntimeError(f'Prepared sidecar mismatch for {k}; refuse reuse')
        if sha(out) != intended['base_projector_sha256']:
            raise RuntimeError('Diagnostic projector copy does not match the frozen base')
        print('[RESUME] prepared and verified',out)
    else:
        out.parent.mkdir(exist_ok=True, parents=True)
        # No changes to the original checkpoint, no overwriting.
        shutil.copyfile(base, out)
        side.write_text(json.dumps(intended, indent=2, ensure_ascii=False), encoding='utf-8')
        print('[PREPARED] unique unchanged baseline checkpoint:',out)
    print(f'[BANK] {dataset}: {cfg["n"]-len(missing)}/{cfg["n"]} train-GT prototypes')
    if missing:print('[MISSING TRAIN GT]',intended['missing_classes'])
    print('[METHOD] During evaluation replace projected part-text embeddings with DIRECT GT visual prototypes (no fitted W).')
    return out


class ReplaceProjections:
    def __init__(self, dataset, min_order_top1=0.95, min_diag_cos=0.80):
        self.dataset = dataset
        self.n = DATASETS[dataset]['n']
        self.p, self.raw, self.names, self.counts, self.missing = load_sources(dataset)
        self.raw_norm = F.normalize(self.raw, dim=1)
        self.min_order_top1 = min_order_top1
        self.min_diag_cos = min_diag_cos
        self.n_calls = 0
        self.n_replaced = 0
        self.n_masked_argmax = 0
        self.seen_shapes = []

    def install(self):
        from src.model import ProjectionLayer
        original = getattr(ProjectionLayer, 'project_clip_txt', None)
        if original is None or not callable(original):
            raise RuntimeError('src.model.ProjectionLayer.project_clip_txt missing; need repo-specific hook')
        state = self

        def replacement(layer, raw, *args, **kwargs):
            state.n_calls += 1
            if len(state.seen_shapes)<6:
                state.seen_shapes.append(tuple(raw.shape) if torch.is_tensor(raw) else str(type(raw)))
            if not torch.is_tensor(raw) or raw.ndim != 2 or tuple(raw.shape) != (state.n,512):
                return original(layer, raw, *args, **kwargs)
            r = F.normalize(raw.detach().float().cpu(), dim=-1)
            sim = r @ state.raw_norm.T
            nearest = sim.argmax(1)
            ordered_fraction = float((nearest == torch.arange(state.n)).float().mean())
            diagonal = sim.diag()
            minimum_diag = float(diagonal.min())
            if ordered_fraction < state.min_order_top1 or minimum_diag < state.min_diag_cos:
                mismatches = [(int(i),state.names[int(i)],int(nearest[i])) for i in range(state.n) if nearest[i]!=i][:15]
                raise RuntimeError(
                    'Class order / input CLIP bank check failed: '
                    f'top1_order_fraction={ordered_fraction:.3f}, min_diag_cos={minimum_diag:.3f}, '
                    f'first_bad={mismatches}. Cannot silently substitute GT vectors.')
            if state.n_replaced == 0:
                print(f'[INJECTION PASS] {state.dataset}, {state.n} classes, '
                      f'CLIP order top1={ordered_fraction:.3f}, min diagonal similarity={minimum_diag:.3f}, '
                      f'missing_train_GT={len(state.missing)}',flush=True)
            state.n_replaced += 1
            return state.p.to(device=raw.device,dtype=raw.dtype)

        ProjectionLayer.project_clip_txt = replacement
        self.original = original
        self.model_class = ProjectionLayer
        if self.missing:
            self._install_absent_mask()

    def _mask_logits(self, tensor, axis):
        if not (self.n_replaced and self.missing and torch.is_tensor(tensor)):
            return tensor
        if axis is None:
            return tensor
        if tensor.ndim < 3 or tensor.numel() <= self.n * 128:
            return tensor
        axis = int(axis) % tensor.ndim
        if tensor.shape[axis] != self.n:
            return tensor
        idx=torch.tensor(self.missing, device=tensor.device,dtype=torch.long)
        x=tensor.clone()
        x.index_fill_(axis,idx, -float('inf'))
        self.n_masked_argmax+=1
        if self.n_masked_argmax <= 3:
            print(f'[MISSING CLASS MASK PASS] class-axis={axis}, shape={tuple(tensor.shape)}, missing={self.missing}',flush=True)
        return x

    def _install_absent_mask(self):
        self.original_tensor_argmax = torch.Tensor.argmax
        self.original_torch_argmax = torch.argmax
        state=self
        def tensor_argmax(x,dim=None,keepdim=False):
            return state.original_tensor_argmax(state._mask_logits(x,dim), dim=dim, keepdim=keepdim)
        def global_argmax(input,dim=None,keepdim=False):
            return state.original_torch_argmax(state._mask_logits(input,dim), dim=dim, keepdim=keepdim)
        torch.Tensor.argmax=tensor_argmax
        torch.argmax=global_argmax

    def restore(self):
        if hasattr(self,'model_class'):
            self.model_class.project_clip_txt = self.original
            if hasattr(self,'original_tensor_argmax'):
                torch.Tensor.argmax=self.original_tensor_argmax
                torch.argmax=self.original_torch_argmax


def evaluation(dataset, out_dir=None, min_order_top1=0.95, min_diag_cos=0.80):
    cfg = DATASETS[dataset]
    prepare(dataset)
    out = full_path(out_dir or cfg['eval_out'])
    result = ROOT/'segmentation_results'/(get_projector_name(dataset)+'.json')
    if out.exists():
        raise FileExistsError(f'Refusing to reuse an existing eval output dir: {out}')
    if result.exists():
        raise FileExistsError(f'Refusing to overwrite an existing official result: {result}')
    # Import order matters: first monkey-patch src.model, THEN execute main.py
    # in the same distributed rank process, with the same CLI/config protocol.
    hook = ReplaceProjections(dataset, min_order_top1, min_diag_cos)
    hook.install()
    main = require(ROOT/'src/open_vocabulary_segmentation/main.py')
    current_argv = sys.argv[:]
    current_path = sys.path[:]
    try:
        sys.path.insert(0, str(main.parent))
        sys.argv = [str(main), '--eval','--output',str(out),
                    '--eval_cfg',str(require(cfg['eval_cfg'])),
                    '--eval_base_cfg',str(require(cfg['eval_base'])),
                    '--opts','model.proj_name='+get_projector_name(dataset),'evaluate.pamr=false']
        print('[EVAL] original main.py with identical eval cfg, gt visual injected at runtime',flush=True)
        print('[EVAL ARGUMENTS]', ' '.join(sys.argv),flush=True)
        try:
            runpy.run_path(str(main),run_name='__main__')
        except SystemExit as e:
            if e.code not in (0,None):
                raise
    finally:
        sys.argv = current_argv
        sys.path[:] = current_path
        hook.restore()
        print(f'[HOOK COUNTS] project_clip_txt calls={hook.n_calls}; '
              f'complete GT replacements={hook.n_replaced}; missing_class_masks={hook.n_masked_argmax}; seen_shapes={hook.seen_shapes}',flush=True)
    if hook.n_replaced < 1:
        raise RuntimeError('No complete GT prototype replacement occurred! Do not trust the metrics. '
                           'Send [HOOK COUNTS] and the repo text-classifier implementation for a precise adapter.')
    if hook.missing and hook.n_masked_argmax < 1:
        raise RuntimeError('Missing-class score masking NEVER occurred; evaluator may use max()/numpy argmax instead. '
                           'Do not trust this result. Send [HOOK COUNTS] and inference argmax code for a precise hook.')
    print('[EVALUATION DONE] main.py evaluation successfully used direct GT visual prototypes',flush=True)
    print('[EVAL RESULT FILE]',result,flush=True)


NUM = r'(?:nan|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
ROW_PATTERN = re.compile(r'^\|\s*(.*?)\s*\|\s*('+NUM+r')\s*\|\s*('+NUM+r')\s*\|\s*$',re.IGNORECASE)


def parse_result_log(log_text, names):
    # Evaluation log table follows the original code: Class | IoU | Acc,
    # and Summary rows aAcc, mIoU, mAcc, all in percentages.
    found = {}
    for line in log_text.splitlines():
        m = ROW_PATTERN.match(line.strip())
        if m and m.group(1).strip() in names:
            key=m.group(1).strip()
            found[key] = [float(m.group(2)),float(m.group(3))]
    if set(found) != set(names):
        missing=set(names)-set(found)
        extra=set(found)-set(names)
        raise RuntimeError(f'Class table incomplete: {len(found)}/{len(names)}; '
                           f'missing={sorted(missing)[:15]}, unexpected={sorted(extra)[:15]}')
    # Find the FINAL global Summary table after class rows.
    lines=log_text.splitlines()
    summary=None
    for i,line in enumerate(lines):
        if 'Summary:' in line:
            for candidate in lines[i+1:i+10]:
                cm = re.match(r'^\s*\|\s*('+NUM+r')\s*\|\s*('+NUM+r')\s*\|\s*('+NUM+r')\s*\|',candidate,re.IGNORECASE)
                if cm:
                    summary={'aAcc':float(cm.group(1)),'mIoU':float(cm.group(2)),'mAcc':float(cm.group(3))}
    if summary is None:
        raise RuntimeError('Cannot find evaluator Summary aAcc/mIoU/mAcc')
    return found,summary


def export(dataset, eval_log, out_dir):
    proto,raw,names,counts,missing=load_sources(dataset)
    log=full_path(eval_log)
    if not log.is_file():raise FileNotFoundError(log)
    found,summary=parse_result_log(log.read_text(encoding='utf-8',errors='replace'),names)
    out=full_path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    csv_file=out/(dataset.upper()+'_GTVisualDirect_per_part.csv')
    summary_file=out/(dataset.upper()+'_GTVisualDirect_summary.json')
    if csv_file.exists() or summary_file.exists():
        raise FileExistsError('Refuse to overwrite existing report; choose a new --out-dir')
    with csv_file.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['part_id','part_name','GT_train_patch_count','GT_train_proto_valid','IoU_pct','Acc_pct','GT_val_present'])
        for i,n in enumerate(names):
            iou,acc=found[n]
            w.writerow([i,n,int(counts[i]),int(i not in missing),
                        round(iou,2) if math.isfinite(iou) else 'nan',
                        round(acc,2) if math.isfinite(acc) else 'nan',
                        int(math.isfinite(acc))])
    diagnostics={
        'source':'PRIVILEGED train-GT direct visual classification',
        'dataset':dataset, 'evaluated_part_count':len(found),
        'summary_pct':summary,
        'gt_train_available':DATASETS[dataset]['n']-len(missing),
        'gt_train_missing':[(i,names[i]) for i in missing],
        'note':'IoU_p and Acc_p are per-part. mIoU and mAcc exist only as dataset means.',
        'eval_log':str(log),'per_part_csv':str(csv_file),
    }
    summary_file.write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[EXPORT SUCCESS]',csv_file)
    print('[METRICS] '+' '.join(f'{k}={v:.2f}%' for k,v in summary.items()))
    return csv_file


def self_test():
    torch.manual_seed(123)
    # Check exact missing-class suppression via deterministic identical columns.
    p=F.normalize(torch.randn(5,768),dim=1)
    valid=torch.tensor([1,1,0,1,0],dtype=torch.bool)
    p[~valid]=p[0].clone()
    # Explicit class-axis argmax mask, independent of floating-point tie rounding.
    scores=(F.normalize(torch.randn(1,768,13,13),dim=1).permute(0,2,3,1) @ p.T)
    scores.index_fill_(-1,torch.tensor([2,4]),-float('inf'))
    ids=scores.argmax(-1)
    assert not bool((ids==2).any()) and not bool((ids==4).any())
    # Table parsing: real-world column widths and NaN values.
    log='\n'.join(['| Class | IoU | Acc |',
        "| person's head | 59.75 | 61.44 |",
        "| person's leg | 0.0 | 0.0 |",
        "| door's frame | 0.0 | nan |",
        '[X] INFO Summary:',
        '| aAcc | mIoU | mAcc |',
        '| 52.25 | 21.1 | 47.72 |'])
    rows,summary=parse_result_log(log,["person's head","person's leg","door's frame"])
    assert len(rows)==3 and math.isnan(rows["door's frame"][1])
    assert summary['mIoU']==21.1 and summary['mAcc']==47.72
    print('SELF_TEST_PASS: prototype cloning excludes missing classes; NaN per-part parsing; summary parsing')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='command',required=True)
    sp.add_parser('self-test')
    for cmd in ('prepare','eval'):
        p=sp.add_parser(cmd)
        p.add_argument('--dataset',choices=tuple(DATASETS),required=True)
        if cmd=='eval':
            p.add_argument('--output',default=None)
            p.add_argument('--min-order-top1',type=float,default=0.95)
            p.add_argument('--min-diag-cos',type=float,default=0.80)
    p=sp.add_parser('export')
    p.add_argument('--dataset',choices=tuple(DATASETS),required=True)
    p.add_argument('--eval-log',required=True)
    p.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='prepare':prepare(a.dataset)
    elif a.command=='eval':evaluation(a.dataset,a.output,a.min_order_top1,a.min_diag_cos)
    else:export(a.dataset,a.eval_log,a.out_dir)

if __name__=='__main__':
    main()
