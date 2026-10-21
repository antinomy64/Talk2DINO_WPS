#!/usr/bin/env python3
"""PP116 diagnosis only: safely bake train-GT orthogonal Oracle W; compare held-out evaluations.

The Oracle used TRAIN part GT prototypes. Never use this artifact as a weakly
supervised method or as a GT-free result. It must be evaluated on validation
images that played NO role in fitting W.

Usage:
  python pp116_oracle_heldout_eval.py bake --oracle-w ... --output-projector weights/<UNIQUE_NAME>.pth
  python pp116_oracle_heldout_eval.py compare --baseline-json ... --oracle-json ... --out-dir ...

No internet, no editing original source files, refuses overwrite.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import torch
import torch.nn.functional as F

ROOT = Path.cwd()
DEFAULT_BASE = 'weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'
DEFAULT_ORACLE = 'output/pp116_gt_oracle_diagnostic_v1/diagnostic_GT_Orthogonal_CE_W.pt'
DEFAULT_TEXT = 'feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt'
DEFAULT_CFG = 'configs/vitb_mlp_infonce.yaml'
WEIGHT_KEY = 'hidden_layers.0.weight'
BIAS_KEY = 'hidden_layers.0.bias'
EXPECTED = {
    'linear_layer.weight': (768, 512),
    'linear_layer.bias': (768,),
    WEIGHT_KEY: (768, 768),
    BIAS_KEY: (768,),
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()


def load_pt(path):
    try: return torch.load(str(path),map_location='cpu',weights_only=True)
    except TypeError: return torch.load(str(path),map_location='cpu')


def checked_state(ckpt):
    if not isinstance(ckpt,Mapping):raise ValueError('Base projector must be dict')
    for k in ('state_dict','model_state_dict','projector_state_dict'):
        if k in ckpt and isinstance(ckpt[k],Mapping):
            ckpt=ckpt[k];break
    ckpt=dict(ckpt)
    if set(ckpt)!=set(EXPECTED):
        raise ValueError('Unexpected base projector keys: '+str(sorted(ckpt)))
    for k,shape in EXPECTED.items():
        if not torch.is_tensor(ckpt[k]) or tuple(ckpt[k].shape)!=shape:
            raise ValueError(f'Unexpected {k} shape: {getattr(ckpt[k],"shape",None)} vs {shape}')
    return ckpt


def build_model(config, state):
    import yaml
    from src.model import ProjectionLayer
    with Path(config).open('r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    model_cfg=cfg['model']
    assert str(model_cfg.get('model_class','ProjectionLayer'))=='ProjectionLayer'
    model=ProjectionLayer.from_config(model_cfg)
    model.load_state_dict(state,strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def bake(args):
    import train_relproto_alignemt as tr
    base=Path(args.base_projector); oracle=Path(args.oracle_w)
    text=Path(args.text_bank); cfg=Path(args.model_config)
    out=Path(args.output_projector)
    for p in (base,oracle,text,cfg):
        if not p.is_file():raise FileNotFoundError(p)
    if out.exists() or Path(str(out)+'.oracle_bake.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing diagnostic output: {out}')
    if out.resolve() in (base.resolve(),oracle.resolve()):
        raise ValueError('Output must differ from all input files')
    raw,names,_=tr.load_raw_clip_bank(text,expected_clip_model='ViT-B/16',expected_template='sub_imagenet_template')
    if tuple(raw.shape)!=(116,512) or len(names)!=116:raise ValueError('Expected 116 real CLIP part rows')
    wpack=load_pt(oracle)
    if not isinstance(wpack,Mapping) or 'W' not in wpack:raise KeyError('Oracle checkpoint missing W')
    source_tag=str(wpack.get('source','')).upper()
    if 'GT ORACLE' not in source_tag:
        raise ValueError(f'Expected diagnostic TRAIN GT ORACLE checkpoint; got source={source_tag!r}')
    W=torch.as_tensor(wpack['W'],dtype=torch.float32).cpu().contiguous()
    if tuple(W.shape)!=(768,768) or not torch.isfinite(W).all():raise ValueError('Invalid Oracle W')
    eye=torch.eye(768,dtype=W.dtype)
    orth_max=float((W.T@W-eye).abs().max().item())
    print(f'[CHECK] Oracle W orthogonality max deviation = {orth_max:.3e}',flush=True)
    if orth_max>1e-3:raise ValueError('Oracle W is not sufficiently orthogonal')
    src_state=checked_state(load_pt(base))
    src_model=build_model(cfg,src_state)
    with torch.inference_mode():
        t0=F.normalize(src_model.project_clip_txt(raw.float()),dim=-1)
        expected=F.normalize(t0@W,dim=-1)
    st={k:v.detach().clone() for k,v in src_state.items()}
    # Pytorch nn.Linear: y = x @ A.T + b; target y@W => A'=W.T@A, b'=W.T@b
    A=src_state[WEIGHT_KEY].detach().double()
    b=src_state[BIAS_KEY].detach().double()
    W64=W.double()
    st[WEIGHT_KEY]=(W64.T@A).to(src_state[WEIGHT_KEY].dtype)
    st[BIAS_KEY]=(W64.T@b).to(src_state[BIAS_KEY].dtype)
    baked=build_model(cfg,st)
    with torch.inference_mode():
        got=F.normalize(baked.project_clip_txt(raw.float()),dim=-1)
    max_abs=float((got-expected).abs().max().item())
    print(f'[CHECK] 116 real part vectors, pre-save max difference = {max_abs:.3e}',flush=True)
    if not torch.allclose(got,expected,rtol=2e-5,atol=2e-5):
        raise AssertionError('Bake did not reproduce original projector + Oracle W')
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.oracle_bake_',suffix='.pth',dir=str(out.parent))
    os.close(fd)
    try:
        torch.save(st,tmp)
        reread=checked_state(load_pt(tmp))
        reread_model=build_model(cfg,reread)
        with torch.inference_mode():
            reloaded=F.normalize(reread_model.project_clip_txt(raw.float()),dim=-1)
        reload_err=float((reloaded-expected).abs().max().item())
        print(f'[CHECK] 116 real part vectors, reload max difference = {reload_err:.3e}',flush=True)
        if not torch.allclose(reloaded,expected,rtol=2e-5,atol=2e-5):
            raise AssertionError('Reloaded Oracle projector differs from exact target')
        if out.exists():raise FileExistsError(out)
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    info={
        'warning':'PRIVILEGED DIAGNOSTIC ONLY: Oracle W was fitted with TRAIN part ground-truth prototypes; NOT weakly supervised',
        'base_projector':str(base),'base_sha256':sha256(base),
        'oracle_checkpoint':str(oracle),'oracle_sha256':sha256(oracle),
        'output_projector':str(out),'output_sha256':sha256(out),
        'text_bank':str(text),'text_bank_sha256':sha256(text),
        'model_config':str(cfg),'W_orthogonality_max_abs':orth_max,
        'part_count':116,'target':'normalize(normalize(project_raw_CLIP(raw)) @ W)',
        'bake_last_affine':'A_baked = W.T @ A; b_baked = W.T @ b',
        'max_abs_pre_save':max_abs,'max_abs_after_reload':reload_err,
        'verification_status':'PASS'
    }
    Path(str(out)+'.oracle_bake.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ORACLE_BAKE_PASS:',out)
    print('IMPORTANT: Do not include this GT-privileged checkpoint in weakly supervised benchmark tables.')


def read_eval(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(f'Evaluation JSON not found: {p}')
    obj=json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(obj,dict) or not isinstance(obj.get('metrics'),list) or not obj['metrics']:
        raise ValueError(f'{p}: expected JSON containing metrics[0]')
    m=obj['metrics'][0]
    if not isinstance(m,dict) or 'mIoU' not in m:raise ValueError(f'{p}: metrics[0].mIoU missing')
    names={k[4:] for k in m if k.startswith('IoU.')}
    if len(names)!=116:raise ValueError(f'{p}: expected 116 IoU.X keys, got {len(names)}')
    return m,names


def num(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')


def compare(args):
    baseline,n1=read_eval(args.baseline_json)
    oracle,n2=read_eval(args.oracle_json)
    if n1!=n2:raise ValueError('Validation class names differ between evaluations')
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Report directory not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in sorted(n1):
        b=num(baseline['IoU.'+name]);o=num(oracle['IoU.'+name])
        bgt=num(baseline.get('Acc.'+name));ogt=num(oracle.get('Acc.'+name))
        if math.isfinite(bgt)!=math.isfinite(ogt):raise ValueError(f'GT valid status differs for {name}')
        valid=math.isfinite(bgt)
        if valid and (not math.isfinite(b) or not math.isfinite(o)):
            raise ValueError(f'Valid IoU is nonfinite for {name}')
        rows.append({'part':name,'GT_valid_in_val':int(valid),
                     'E100_IoU_pct':100*b if valid else '',
                     'GT_oracle_IoU_pct':100*o if valid else '',
                     'delta_IoU_points':100*(o-b) if valid else ''})
    with (out/'per_part_E100_vs_GT_oracle.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    valid=[r for r in rows if r['GT_valid_in_val']]
    improved=sorted(valid,key=lambda r:r['delta_IoU_points'],reverse=True)
    failed=sorted(valid,key=lambda r:r['delta_IoU_points'])
    summary={
        'warning':'Oracle fitted using TRAIN part GT prototypes. Held-out validation scores are diagnostic PRIVILEGED-supervision results, not weak-supervision comparisons.',
        'baseline_json':str(args.baseline_json),'oracle_json':str(args.oracle_json),
        'baseline_mIoU_pct':100*num(baseline['mIoU']),
        'oracle_mIoU_pct':100*num(oracle['mIoU']),
        'delta_mIoU_points':100*(num(oracle['mIoU'])-num(baseline['mIoU'])),
        'num_parts':len(rows),'num_val_GT_valid':len(valid),
        'num_part_IoU_increase':sum(r['delta_IoU_points']>1e-5 for r in valid),
        'num_part_IoU_decrease':sum(r['delta_IoU_points']<-1e-5 for r in valid),
        'num_part_IoU_unchanged':sum(abs(r['delta_IoU_points'])<=1e-5 for r in valid),
        'largest_improvements':improved[:15],
        'largest_declines':failed[:15],
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"E100 weakly-trained mIoU: {summary['baseline_mIoU_pct']:.2f}%")
    print(f"GT Oracle held-out mIoU:  {summary['oracle_mIoU_pct']:.2f}%")
    print(f"Delta: {summary['delta_mIoU_points']:+.2f} percentage points")
    print('------ Focus parts ------')
    focus=['person\'s upper arm','person\'s leg','dog\'s tail',"dog's ear", "cow's horn", "sheep's horn", "car's mirror", "bicycle's saddle", "bird's eye", "person's lower arm", "bus's mirror"]
    byname={r['part']:r for r in valid}
    for name in focus:
        if name in byname:
            r=byname[name]
            print(f"{name:25s} E100={r['E100_IoU_pct']:6.2f}%  Oracle={r['GT_oracle_IoU_pct']:6.2f}%  Δ={r['delta_IoU_points']:+7.2f}")
    print('------ Biggest 10 gains ------')
    for r in improved[:10]:
        print(f"{r['part']:25s} Δ={r['delta_IoU_points']:+7.2f}")
    print('Results:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='mode',required=True)
    b=sp.add_parser('bake',help='Make a new DIAGNOSTIC baked projector from Oracle CE W')
    b.add_argument('--base-projector',default=DEFAULT_BASE)
    b.add_argument('--oracle-w',default=DEFAULT_ORACLE)
    b.add_argument('--text-bank',default=DEFAULT_TEXT)
    b.add_argument('--model-config',default=DEFAULT_CFG)
    b.add_argument('--output-projector',required=True)
    c=sp.add_parser('compare',help='Compare two PP116 validation evaluation JSON files')
    c.add_argument('--baseline-json',required=True)
    c.add_argument('--oracle-json',required=True)
    c.add_argument('--out-dir',required=True)
    args=p.parse_args()
    if args.mode=='bake':bake(args)
    if args.mode=='compare':compare(args)

if __name__=='__main__':main()
