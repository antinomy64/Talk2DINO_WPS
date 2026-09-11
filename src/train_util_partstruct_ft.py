from copy import deepcopy
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.loss import ContrastiveLoss, PartStructureRankLoss
import os
import matplotlib.pyplot as plt
import wandb
import numpy as np
import random
import json

def set_seed(seed):
    print(f'Setting seed {seed}...')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def train(model, train_dataloader, contrastive_loss, optimizer, scheduler=None, wandb=False, save_head_attivations=None, n_epochs=0):
    """train the model for one epoch"""
    train_batch_losses = []
    device = next(model.parameters()).device
    prev_iter = n_epochs * len(train_dataloader)
    
    head_attivations = []
    ann_ids = []
    img_ids = []
    for n_batch, batch in enumerate(tqdm(train_dataloader)):
        annotations = batch['annotation'].to(device, dtype=torch.float32)
        images = batch['image'].to(device)
        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None
        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else: 
            self_attn_maps = None
            cls = None
            
        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None
            
        if scheduler is not None:
            scheduler(n_batch + prev_iter)
                    
        if not save_head_attivations:
            loss = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
        else:
            loss, batch_head_attivations = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax, return_index=True)
            head_attivations.append(batch_head_attivations)
            ann_ids.append(batch['metadata']['annotation_id'])
            img_ids.append(batch['metadata']['image_id'])
        train_batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2.0)
        optimizer.step()
        if wandb:
            wandb.log({'train_loss': loss.item()})
            
    if save_head_attivations is not None:
        head_attivations = torch.cat(head_attivations)
        ann_ids = torch.cat(ann_ids)
        img_ids = torch.cat(img_ids)
        act_dict = {}
        for act, ann, img in zip(head_attivations, ann_ids, img_ids):
            act_dict[ann.item()] = {
                'image_id': img.item(),
                'activation_head': act.item()
            }
        with open(save_head_attivations, 'w') as f:
            json.dump(act_dict, f)
            print(f"Saved activation heads summary at {save_head_attivations}")
        
    return torch.mean(torch.tensor(train_batch_losses)).item()

def validate(model, val_dataloader, contrastive_loss, verbose=False):
    # evaluate the model in the validation set
    device = next(model.parameters()).device
    val_batch_losses = []
    
    val_dataloader = tqdm(val_dataloader) if verbose else val_dataloader
    for n_batch, batch in enumerate(val_dataloader):
        annotations = batch['annotation'].to(device, dtype=torch.float32)
        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None

        images = batch['image'].to(device)
        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else: 
            self_attn_maps = None
            cls = None
            
        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None
        
        with torch.no_grad():
            loss = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
    
        val_batch_losses.append(loss.item())
    return torch.mean(torch.tensor(val_batch_losses)).item()
    

def do_train(model, train_dataset, val_dataset, train_cfg, seed=123, optimizer_name="Adam", weight_decay=0.05, scheduler_name='linear', warmup=0, save_head_attivations=None):
    device = next(model.parameters()).device
    # setting manual seed
    # torch.manual_seed(seed)
    set_seed(seed)
    
    # mandatory parameters
    lr, ltype, num_epochs, batch_size = train_cfg['lr'], train_cfg['ltype'], train_cfg['num_epochs'], train_cfg['batch_size']
    # optional parameters
    margin = train_cfg.get('margin', 0.2)
    max_violation = train_cfg.get('max_violation', True)
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)
    # early_stopping = train_cfg.get('early_stopping', 0) # 0 means no early-stopping
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    criterion = ContrastiveLoss(model, margin=margin, max_violation=max_violation, ltype=ltype)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")
    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == 'linear' and warmup == 0:
        scheduler = None
    elif scheduler_name == 'linear' and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == 'cosine':
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    
    # losses declaration
    train_losses = torch.zeros(num_epochs)
    val_losses = torch.zeros(num_epochs)
    for epoch in range(num_epochs):
        # train loss
        model.train()
        train_loss = train(model, train_dataloader, criterion, optimizer, scheduler, save_head_attivations=None if epoch < num_epochs - 1 else save_head_attivations, n_epochs=epoch)
        train_losses[epoch] = train_loss
        
        # eval loop
        model.eval()
        print("Performing Evaluation...")
        val_loss = validate(model, val_dataloader, criterion)
        val_losses[epoch] = val_loss
        
        print(f"Epoch {epoch}: train_loss={train_losses[epoch]} - val_loss={val_losses[epoch]}")
        # evaluating if best model and check early stopping
        if save_best_model and (epoch == 0 or val_losses[epoch] < min(val_losses[:epoch]).item()):
            print(f"Best validation loss, saving the model")
            best_model = deepcopy(model)
    
    model = model if not save_best_model else best_model

    return model, train_losses, val_losses

# ============================================================
# Part-structure-aware fine-tuning
# IMPORTANT:
# - The original train(), validate(), and do_train() above are
#   intentionally left unchanged.
# - When structure_weight == 0, train.py should call do_train()
#   directly so the legacy/clean code path is preserved.
# ============================================================

def train_partstruct(
    model,
    train_dataloader,
    contrastive_loss,
    structure_criterion,
    structure_weight,
    optimizer,
    scheduler=None,
    wandb=False,
    save_head_attivations=None,
    n_epochs=0,
):
    """Train one epoch with Talk2DINO InfoNCE + weighted structure loss."""
    train_total_losses = []
    train_object_losses = []
    train_structure_losses = []

    device = next(model.parameters()).device
    prev_iter = n_epochs * len(train_dataloader)

    head_attivations = []
    ann_ids = []
    img_ids = []

    for n_batch, batch in enumerate(tqdm(train_dataloader)):
        annotations = batch['annotation'].to(device, dtype=torch.float32)
        images = batch['image'].to(device)

        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None

        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else:
            self_attn_maps = None
            cls = None

        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        if not save_head_attivations:
            object_loss = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
            )
        else:
            object_loss, batch_head_attivations = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
                return_index=True,
            )
            head_attivations.append(batch_head_attivations)
            ann_ids.append(batch['metadata']['annotation_id'])
            img_ids.append(batch['metadata']['image_id'])

        # Fixed raw part bank -> current projector -> structure loss.
        structure_loss = structure_criterion(model)
        total_loss = object_loss + float(structure_weight) * structure_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_total_losses.append(total_loss.detach().item())
        train_object_losses.append(object_loss.detach().item())
        train_structure_losses.append(structure_loss.detach().item())

        if wandb:
            wandb.log({
                'train_loss': total_loss.detach().item(),
                'train_object_loss': object_loss.detach().item(),
                'train_structure_loss': structure_loss.detach().item(),
                'train_weighted_structure_loss':
                    float(structure_weight) * structure_loss.detach().item(),
            })

    if save_head_attivations is not None:
        head_attivations = torch.cat(head_attivations)
        ann_ids = torch.cat(ann_ids)
        img_ids = torch.cat(img_ids)
        act_dict = {}
        for act, ann, img in zip(head_attivations, ann_ids, img_ids):
            act_dict[ann.item()] = {
                'image_id': img.item(),
                'activation_head': act.item()
            }
        with open(save_head_attivations, 'w') as f:
            json.dump(act_dict, f)
            print(f"Saved activation heads summary at {save_head_attivations}")

    return (
        torch.mean(torch.tensor(train_total_losses)).item(),
        torch.mean(torch.tensor(train_object_losses)).item(),
        torch.mean(torch.tensor(train_structure_losses)).item(),
    )


def validate_partstruct(
    model,
    val_dataloader,
    contrastive_loss,
    structure_criterion,
    structure_weight,
    verbose=False,
):
    """Validation for InfoNCE + structure loss."""
    device = next(model.parameters()).device
    val_total_losses = []
    val_object_losses = []
    val_structure_losses = []

    val_dataloader = tqdm(val_dataloader) if verbose else val_dataloader

    # The structure term is independent of the COCO validation batch.
    # During validation the projector is fixed, so compute it once per epoch.
    with torch.no_grad():
        epoch_structure_loss = structure_criterion(model).detach()

    for n_batch, batch in enumerate(val_dataloader):
        annotations = batch['annotation'].to(device, dtype=torch.float32)

        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None

        images = batch['image'].to(device)

        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else:
            self_attn_maps = None
            cls = None

        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None

        with torch.no_grad():
            object_loss = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
            )
            structure_loss = epoch_structure_loss
            total_loss = object_loss + float(structure_weight) * structure_loss

        val_total_losses.append(total_loss.detach().item())
        val_object_losses.append(object_loss.detach().item())
        val_structure_losses.append(structure_loss.detach().item())

    return (
        torch.mean(torch.tensor(val_total_losses)).item(),
        torch.mean(torch.tensor(val_object_losses)).item(),
        torch.mean(torch.tensor(val_structure_losses)).item(),
    )


def do_train_partstruct(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    structure_bank,
    structure_weight,
    rank_temperature=0.05,
    structure_min_parts=3,
    seed=123,
    optimizer_name="Adam",
    weight_decay=0.05,
    scheduler_name='linear',
    warmup=0,
    save_head_attivations=None,
):
    """
    Fine-tune/projector training with:
        L_total = L_InfoNCE + structure_weight * L_structure

    This function is only intended for structure_weight > 0.
    For structure_weight == 0, use the unchanged do_train() above.
    """
    if float(structure_weight) <= 0:
        raise ValueError(
            "do_train_partstruct() expects structure_weight > 0. "
            "Use the unchanged do_train() path for structure_weight == 0."
        )
    if structure_bank is None:
        raise ValueError("structure_bank is required when structure_weight > 0")

    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg['lr']
    ltype = train_cfg['ltype']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']

    margin = train_cfg.get('margin', 0.2)
    max_violation = train_cfg.get('max_violation', True)
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    criterion = ContrastiveLoss(
        model,
        margin=margin,
        max_violation=max_violation,
        ltype=ltype,
    )

    structure_criterion = PartStructureRankLoss.from_file(
        structure_bank,
        rank_temperature=rank_temperature,
        min_parts=structure_min_parts,
    ).to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == 'linear' and warmup == 0:
        scheduler = None
    elif scheduler_name == 'linear' and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == 'cosine':
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not implemented")

    print(
        "[PartStructure] "
        f"bank={structure_bank} "
        f"weight={float(structure_weight):.8g} "
        f"rank_temperature={float(rank_temperature):.8g} "
        f"min_parts={int(structure_min_parts)}"
    )

    model.eval()
    print("[structure_input] raw CLIP -> projector -> normalize; matches RelProto T0")
    print("[exact_structure_initial] " + json.dumps(structure_criterion.exact_audit(model), sort_keys=True))

    train_losses = torch.zeros(num_epochs)
    val_losses = torch.zeros(num_epochs)

    train_object_losses = torch.zeros(num_epochs)
    val_object_losses = torch.zeros(num_epochs)
    train_structure_losses = torch.zeros(num_epochs)
    val_structure_losses = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        (
            train_total,
            train_object,
            train_structure,
        ) = train_partstruct(
            model,
            train_dataloader,
            criterion,
            structure_criterion,
            structure_weight,
            optimizer,
            scheduler,
            save_head_attivations=(
                None if epoch < num_epochs - 1 else save_head_attivations
            ),
            n_epochs=epoch,
        )

        train_losses[epoch] = train_total
        train_object_losses[epoch] = train_object
        train_structure_losses[epoch] = train_structure

        model.eval()
        print("Performing Evaluation...")
        (
            val_total,
            val_object,
            val_structure,
        ) = validate_partstruct(
            model,
            val_dataloader,
            criterion,
            structure_criterion,
            structure_weight,
        )

        print("[exact_structure_epoch] " + json.dumps({
            "epoch": epoch, **structure_criterion.exact_audit(model)
        }, sort_keys=True))

        val_losses[epoch] = val_total
        val_object_losses[epoch] = val_object
        val_structure_losses[epoch] = val_structure

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_losses[epoch]} - "
            f"val_loss={val_losses[epoch]} - "
            f"train_object={train_object_losses[epoch]} - "
            f"train_struct={train_structure_losses[epoch]} - "
            f"val_object={val_object_losses[epoch]} - "
            f"val_struct={val_structure_losses[epoch]}"
        )

        if save_best_model and (
            epoch == 0 or val_losses[epoch] < min(val_losses[:epoch]).item()
        ):
            print("Best validation loss, saving the model")
            best_model = deepcopy(model)

    model = model if not save_best_model else best_model

    model.eval()
    print("[exact_structure_returned_checkpoint] " + json.dumps(
        structure_criterion.exact_audit(model), sort_keys=True
    ))

    metrics = {
        'train_total': train_losses,
        'val_total': val_losses,
        'train_object': train_object_losses,
        'val_object': val_object_losses,
        'train_structure': train_structure_losses,
        'val_structure': val_structure_losses,
    }
    return model, metrics
