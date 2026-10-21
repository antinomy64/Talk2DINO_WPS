#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bake a jointly TRAINED text projector and a shared orthogonal W into one native
Talk2DINO Projector checkpoint, preserving the official inference code.

For row-vector CLIP features x, projector's final affine y=x@A.T+b is changed to
A' = W.T @ A, b' = W.T @ b so that y'=y@W. Normalization commutes with W
when W is orthogonal; numerical equivalence is checked before writing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def core_bake(projector, W, test_raw):
    if W.ndim != 2 or W.size(0) != W.size(1):
        raise ValueError('W must be square')
    W = W.to(device=next(projector.parameters()).device, dtype=torch.float32)
    err = (W.T @ W - torch.eye(W.size(0), device=W.device)).abs().max().item()
    if err > 1e-3:
        raise ValueError(f'W not orthogonal; max err = {err}')
    with torch.no_grad():
        before = F.normalize(F.normalize(projector.project_clip_txt(test_raw), dim=-1) @ W, dim=-1)
        if hasattr(projector, 'hidden_layers') and len(projector.hidden_layers):
            last = projector.hidden_layers[-1]
            last_name = f'hidden_layers.{len(projector.hidden_layers)-1}'
        else:
            last = projector.linear_layer
            last_name = 'linear_layer'
        if not isinstance(last, torch.nn.Linear):
            raise RuntimeError('Output projector layer is not nn.Linear')
        if last.weight.shape[0] != W.size(0):
            raise RuntimeError('Projection output dimension does not match W')
        last.weight.copy_(W.T @ last.weight.clone())
        if last.bias is not None:
            last.bias.copy_(W.T @ last.bias.clone())
        after = F.normalize(projector.project_clip_txt(test_raw), dim=-1)
        dif = float((before-after).abs().max())
        if dif > 3e-5:
            raise RuntimeError(f'Bake numerical equivalence FAIL: maxdiff={dif}')
    return dif, last_name, err


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['pp116','ade234'], required=True)
    p.add_argument('--joint_checkpoint', type=Path, required=True)
    p.add_argument('--initial_projector', type=Path,
                   default=Path('weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth'))
    p.add_argument('--model_config', type=Path, default=Path('configs/vitb_mlp_infonce.yaml'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--self_test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.output.exists():
        raise FileExistsError(f'Will not overwrite: {a.output}')
    for name in ('joint_checkpoint','initial_projector','model_config'):
        f=getattr(a,name)
        if not f.is_file():raise FileNotFoundError(f)
    if a.dataset=='pp116':
        import train_relproto_alignemt as original
    else:
        import train_relproto_alignment_ade20kpart234 as original
    ckpt=original.torch_load(a.joint_checkpoint)
    if not ckpt.get('projector_is_trainable') or 'projector_state_dict' not in ckpt:
        raise RuntimeError('Checkpoint is not joint projector+W training; cannot bake')
    if ckpt.get('dataset_name') != a.dataset:
        raise RuntimeError(f'Dataset mismatch: {ckpt.get("dataset_name")!r} vs {a.dataset!r}')
    device=torch.device(a.device)
    proj,info=original.load_frozen_projector(
        project_root=Path.cwd(), config_path=a.model_config,
        weight_path=a.initial_projector, device=device)
    sha=ckpt['projector'].get('weights_sha256')
    if sha and sha!=info['weights_sha256']:
        raise RuntimeError('Initial COCO2014 projector checksum mismatch with training run')
    proj.load_state_dict(ckpt['projector_state_dict'],strict=True)
    proj.eval()
    W=ckpt['W'].float().to(device)
    torch.manual_seed(123)
    raw=torch.randn(64,512,device=device)
    difference, layer_name, ortho = core_bake(proj,W,raw)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    original.atomic_torch_save({k:v.detach().cpu().clone() for k,v in proj.state_dict().items()},a.output)
    manifest={
        'dataset':a.dataset,'joint_checkpoint':str(a.joint_checkpoint),
        'initial_projector':str(a.initial_projector),'epoch':ckpt.get('epoch'),
        'final_affine_layer':layer_name,'bake_max_abs_difference':difference,
        'W_orthogonality_max_abs':ortho,
        'official_evaluation_note':'Use model.proj_name=<output.stem> and evaluate.pamr=false;'
                                   ' frozen DINOv2 and all original eval configs unchanged.'}
    a.output.with_suffix('.bake.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[PASS] output layer:',layer_name,'orth max error:',ortho,flush=True)
    print('[PASS] bake max abs numerical error:',difference,flush=True)
    print('[SAVED]',a.output,flush=True)


def self_test():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_layer=torch.nn.Linear(5,6)
            self.hidden_layers=torch.nn.ModuleList([torch.nn.Linear(6,6)])
        def project_clip_txt(self,x):
            return self.hidden_layers[0](torch.tanh(self.linear_layer(x)))
    toy=Toy()
    Q,_=torch.linalg.qr(torch.randn(6,6))
    err,layer,ortho=core_bake(toy,Q,torch.randn(20,5))
    assert err<1e-5 and layer=='hidden_layers.0' and ortho<1e-5
    print('[SELF_TEST_PASS] last affine layer W bake numerical equivalence')

if __name__=='__main__':main()
