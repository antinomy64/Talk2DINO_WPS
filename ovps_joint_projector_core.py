#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train an existing Soft Relative + Structure trainer with an unfrozen text projector.

Does not change candidate selection, stop-gradient targets, losses, or W retraction.
Every training step re-projects the *raw pre-projector* CLIP text bank in autograd.
Training uses the existing frozen DINO patch tokens and predicted-foreground cache.
GT masks are never opened. An initial projector snapshot fixes G_text for the whole run.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

VERSION = "joint_trainable_text_projector_shared_orthogonal_W_v1"


def add_options(parser: argparse.ArgumentParser):
    parser.add_argument('--projector_lr', type=float, default=1e-6,
                        help='Learning rate of trainable COCO2014 text projector')
    parser.add_argument('--projector_weight_decay', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Optional gradient clipping for both parameter groups; 0 disables')
    return parser


def check_options(args):
    for name in ('projector_lr',):
        val = float(getattr(args, name))
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f'{name} must be finite and >0')
    for name in ('projector_weight_decay', 'max_grad_norm'):
        val = float(getattr(args, name))
        if not math.isfinite(val) or val < 0:
            raise ValueError(f'{name} must be finite and >=0')


def _named_text_parameters(projector):
    """The exact output of ProjectionLayer.project_clip_txt is linear_layer + hidden_layers."""
    if not hasattr(projector, 'linear_layer'):
        raise RuntimeError('Expected ProjectionLayer with linear_layer')
    selected = []
    for name, param in projector.named_parameters():
        if name.startswith('linear_layer.') or name.startswith('hidden_layers.'):
            selected.append((name, param))
        else:
            # Non-text-path attention weights are intentionally NOT optimized.
            param.requires_grad_(False)
    if not selected:
        raise RuntimeError('Projector has no trainable text projection parameters')
    for _, param in selected:
        param.requires_grad_(True)
    return selected


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_joint(*, examples, base_bank_cpu, args, out_dir, text_info,
                projector_info, preflight, original, variant, dataset):
    check_options(args)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA but not available')
    original.set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    root = original.resolve_root(args.project_root)
    bank_path = original.resolve_path(root, args.text_bank)
    config_path = original.resolve_path(root, args.model_config)
    init_weight_path = original.resolve_path(root, args.weights)

    # Keep exactly the same raw CLIP bank and strict weight initialization as the frozen run.
    raw_clip, names, reloaded_text_info = original.load_raw_clip_bank(
        bank_path, expected_clip_model=args.clip_model, expected_template=args.template_set)
    if reloaded_text_info['sha256'] != text_info['sha256']:
        raise RuntimeError('Raw text-bank changed after original preflight')
    raw_clip = raw_clip.to(device=device, dtype=torch.float32).detach()
    projector, project_reloaded_info = original.load_frozen_projector(
        project_root=root, config_path=config_path, weight_path=init_weight_path, device=device)
    if project_reloaded_info['weights_sha256'] != projector_info['weights_sha256']:
        raise RuntimeError('Initial projector checksum changed after original preflight')
    # Preserve the checkpoint's initial projected bank as frozen structure target.
    frozen_base = original.normalize_last(base_bank_cpu.to(device=device, dtype=torch.float32).detach())
    fixed_G0 = (frozen_base @ frozen_base.T).detach()
    with torch.no_grad():
        ref = original.normalize_last(projector.project_clip_txt(raw_clip))
        mismatch = (ref - frozen_base).abs().max().item()
        if mismatch > 2e-5:
            raise RuntimeError(f'Initial projection mismatch: {mismatch:.4g}')
        if ref.shape != (original.NUM_PARTS, original.VISION_DIM):
            raise RuntimeError(f'Unexpected bank shape {tuple(ref.shape)}')

    trainable = _named_text_parameters(projector)
    projector.eval()   # no change to stochastic behavior relative to initial baseline
    train_params = [param for _, param in trainable]
    projector_init = [p.detach().clone() for p in train_params]
    W = torch.nn.Parameter(torch.eye(original.VISION_DIM, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([
        {'params': [W], 'lr': args.lr, 'weight_decay': 0.0},
        {'params': train_params, 'lr': args.projector_lr,
         'weight_decay': args.projector_weight_decay},
    ])
    loader = DataLoader(original.PredObjDataset(examples), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'),
                        collate_fn=original.collate_examples)
    history: list[dict[str, Any]] = []
    global_step = 0
    checked = False
    grad_relation_checked = False
    diagnostic_keys = (
        'loss_total', 'loss_nce', 'loss_relation', 'structure_active_fraction',
        'mean_triplets_per_annotation', 'mean_positive_count_multi',
        'fallback_fraction_multi', 'mean_effective_patch_count',
        'mean_weighted_margin', 'single_part_annotation_fraction')
    print('[JOINT] dataset', dataset, 'crops', len(examples), 'classes', len(names), flush=True)
    print('[JOINT] trainable text parameters', [n for n, _ in trainable], flush=True)
    print('[JOINT] W lr', args.lr, 'projector lr', args.projector_lr, flush=True)
    print('[JOINT] fixed relation matrix is INITIAL frozen projected CLIP text cosine', flush=True)
    for epoch in range(1, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        logs = defaultdict(list)
        progress = tqdm(loader, desc=f'joint {dataset} epoch {epoch:03d}/{args.epochs:03d}')
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            # This graph links all text classes -> projector -> loss, recomputed on EVERY batch.
            base_bank = original.normalize_last(projector.project_clip_txt(raw_clip))
            loss, info, separate = variant.batch_forward(
                base_bank, W, batch, device=device, frozen_relations=fixed_G0,
                args=args)
            if not grad_relation_checked and args.lambda_struct > 0 and info['structure_active_fraction'] > 0:
                probe = torch.autograd.grad(separate['relation'], [W, train_params[0]],
                                            retain_graph=True, allow_unused=True)
                if any(g is None or not torch.isfinite(g).all() for g in probe):
                    raise RuntimeError('Structure loss disconnected from W or projector')
                print('[CHECK] structure loss gradient to W/projector =',
                      [float(g.norm()) for g in probe], flush=True)
                grad_relation_checked = True
            loss.backward()
            if W.grad is None or not torch.isfinite(W.grad).all():
                raise RuntimeError('W missing/nonfinite gradient')
            missing = [name for name, p in trainable
                       if p.grad is None or not torch.isfinite(p.grad).all()]
            if missing:
                raise RuntimeError(f'Text projector parameters missing gradients: {missing}')
            proj_grad = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in train_params))
            logs['grad_norm_W'].append(float(W.grad.detach().norm().item()))
            logs['grad_norm_projector'].append(proj_grad)
            if not checked:
                if proj_grad <= 1e-15:
                    raise RuntimeError('Projector has ZERO gradient; refusing false unfreeze experiment')
                before = [p.detach().clone() for p in train_params]
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([W] + train_params, args.max_grad_norm)
            optimizer.step()
            original.orthogonal_retract_(W)
            if not checked:
                step_delta = max(float((p.detach() - b).abs().max().item())
                                 for p, b in zip(train_params, before))
                if not math.isfinite(step_delta) or step_delta <= 0:
                    raise RuntimeError('Projector parameters did not actually UPDATE on step 1')
                print(f'[PASS] projector gradients nonzero; first-step parameter max delta={step_delta:.8g}',
                      flush=True)
                checked = True
            global_step += 1
            for key in diagnostic_keys:
                logs[key].append(info[key])
            progress.set_postfix(nce=f"{info['loss_nce']:.3f}",
                                 rank=f"{info['loss_relation']:.3f}",
                                 pgrad=f'{proj_grad:.2g}')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        # W's exact orthogonal invariance must continue to hold.
        mat_stats = original.matrix_stats(W.detach(), frozen_base)
        if mat_stats['orthogonality_max_abs'] > 1e-3:
            raise RuntimeError(f'W lost orthogonality: {mat_stats}')
        with torch.no_grad():
            end_base = original.normalize_last(projector.project_clip_txt(raw_clip))
            shift = float((end_base @ end_base.T - fixed_G0).abs().mean().item())
            proj_param_delta = max(float((p.detach() - q).abs().max().item())
                                   for p, q in zip(train_params, projector_init))
            if not (math.isfinite(shift) and math.isfinite(proj_param_delta)):
                raise RuntimeError('Nonfinite projector metrics')
        row = {'epoch': epoch, 'annotations': len(examples), 'global_step': global_step,
               'optimizer_updates': len(loader),
               'mean_grad_norm_W': float(np.mean(logs['grad_norm_W'])),
               'mean_grad_norm_projector': float(np.mean(logs['grad_norm_projector'])),
               'projector_param_delta_from_init_max': proj_param_delta,
               'projected_text_relation_mean_abs_change': shift,
               **{k: float(np.mean(logs[k])) for k in diagnostic_keys},
               'seconds': float(time.perf_counter() - tick), **mat_stats}
        row['mean_loss'] = row['loss_total']
        history.append(row)
        original.write_csv(history, out_dir / 'training_history.csv')
        ckpt = original.checkpoint_payload(
            W=W, optimizer=optimizer, epoch=epoch, global_step=global_step,
            history=history, args=args, text_info=text_info,
            projector_info=projector_info, preflight=preflight)
        # The original preflight `projector_info` describes the *initially frozen*
        # COCO2014 load. Mark the train-time state accurately in the checkpoint.
        ckpt['projector'] = dict(ckpt['projector'])
        ckpt['projector'].update({
            'frozen': False,
            'initial_checkpoint_loaded_frozen': True,
            'training_mode': 'trainable_project_clip_txt_linear_layers',
        })
        ckpt.update({
            'format': VERSION, 'version': VERSION,
            'projector_state_dict': _cpu_state(projector),
            'projector_is_trainable': True,
            'projector_trainable_names': [name for name, _ in trainable],
            'projector_lr': args.projector_lr,
            'frozen_relation_reference': 'initial_COCO2014_projector_text_cosine',
            'baseline_trainer': variant.__name__,
            'dataset_name': dataset,
        })
        ckpt['method'].update({
            'text': 'Raw CLIP bank -> live trainable text projector -> normalization -> orthogonal W',
            'W': 'W separately optimized with Adam then exact SVD/polar retraction per batch',
            'loss': 'Original Soft Relative InfoNCE (visual target STOPGRAD) + structure relation',
            'foreground': 'Original cached predicted object foreground FIXED; no GT masks',
            'projector': 'Text linear_layer+hidden_layers jointly trained from COCO2014 weights',
            'structure_prior': 'Fixed cosine of PRETRAINED projector outputs, not the moving projector',
        })
        original.atomic_torch_save(ckpt, out_dir / f'W_epoch_{epoch:03d}.pt')
        original.atomic_torch_save(ckpt, out_dir / 'W_last.pt')
        print(f'[EPOCH] {epoch:03d}/{args.epochs} '
              f"loss={row['loss_total']:.5f} gradW={row['mean_grad_norm_W']:.4g} "
              f"gradP={row['mean_grad_norm_projector']:.4g} "
              f"projDelta={proj_param_delta:.4g} "
              f"textRelShift={shift:.4g} orth={mat_stats['orthogonality_max_abs']:.3g}",
              flush=True)
    if not checked:
        raise RuntimeError('Not a single projector training step passed checks')
    original.atomic_json_dump({
        'status': 'completed', 'version': VERSION, 'checkpoint': 'W_last.pt',
        'epochs': args.epochs, 'global_step': global_step,
        'trainable_projector': True, 'history': history,
    }, out_dir / 'summary.json')


def self_test_joint():
    # CPU-only algebra test: both projector and W receive gradients; no visual target grads.
    import torch.nn.functional as F
    torch.manual_seed(123)
    projector = torch.nn.Linear(6, 8)
    raw = torch.randn(4, 6)
    W = torch.nn.Parameter(torch.eye(8))
    text = F.normalize(F.normalize(projector(raw), dim=-1) @ W, dim=-1)
    visual = F.normalize(torch.randn(4, 8), dim=-1)
    visual.requires_grad_(True)
    loss = F.cross_entropy(visual.detach() @ text.T / 0.07, torch.arange(4))
    loss.backward()
    assert W.grad is not None and W.grad.norm().item() > 0
    assert projector.weight.grad is not None and projector.weight.grad.norm().item() > 0
    assert visual.grad is None
    print('[SELF_TEST_PASS] W and projector gradients; visual pseudo-target detached', flush=True)
