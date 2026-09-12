import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import yaml
import importlib
import torchvision.transforms as T
import clip

from src.dataset import DinoClipDataset, COCOCaptions
from src.metrics import get_image_and_text_tensor, i2t, t2i
from src.model import ProjectionLayer
from src.train_util_partstruct_ft import do_train, do_train_partstruct, set_seed
from tqdm import tqdm

device = 'cuda'

def train_and_eval(
    config_file,
    train_dataset,
    val_dataset,
    texts=None,
    images=None,
    model_type='cls',
    test_set=None,
    optimizer="adam",
    weight_decay=0.05,
    scheduler='linear',
    warmup=0,
    name_pedix='',
    save_head_activations=None,
    init_weight=None,
    num_epochs_override=None,
    lr_override=None,
    structure_bank=None,
    structure_teacher_bank=None,
    structure_weight=0.0,
    rank_temperature=0.05,
    structure_min_parts=3,
):
    set_seed(123)
    out_dir = 'weights'
    model_name = os.path.basename(config_file).split('.')[0]
    if name_pedix != '':
        model_name += f"_{name_pedix}"
    if model_type == '':
        out_path = os.path.join(out_dir, f"{model_name}")
    else:
        out_path = os.path.join(out_dir, f"{model_name}_{model_type}")

    config = {}
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    model_class_name = config['model'].get('model_class', 'ProjectionLayer')
    ModelClass = getattr(importlib.import_module('src.model'), model_class_name)
    
    model = ModelClass.from_config(config['model'])

    if init_weight is not None:
        state = torch.load(init_weight, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        print(f"Loaded initialization weights from {init_weight}")

    model.to(device)
    print(model)

    # Use a copy so CLI overrides do not edit the YAML on disk.
    train_cfg = dict(config['train'])
    if num_epochs_override is not None:
        train_cfg['num_epochs'] = int(num_epochs_override)
    if lr_override is not None:
        train_cfg['lr'] = float(lr_override)

    print(
        f"Training config: lr={train_cfg['lr']} "
        f"epochs={train_cfg['num_epochs']} "
        f"batch_size={train_cfg['batch_size']} "
        f"structure_weight={float(structure_weight)}"
    )

    # IMPORTANT: lambda=0 takes the original do_train() path unchanged.
    # This is the regression test against the clean COCO reproduction.
    if float(structure_weight) == 0.0:
        model, train_losses, val_losses = do_train(
            model,
            train_dataset,
            val_dataset,
            train_cfg,
            optimizer_name=optimizer,
            weight_decay=weight_decay,
            scheduler_name=scheduler,
            warmup=warmup,
            save_head_attivations=save_head_activations,
        )
    else:
        if structure_bank is None:
            raise ValueError(
                "--structure_bank is required when --structure_weight > 0"
            )

        model, structure_metrics = do_train_partstruct(
            model,
            train_dataset,
            val_dataset,
            train_cfg,
            structure_bank=structure_bank,
            structure_teacher_bank=structure_teacher_bank,
            structure_weight=structure_weight,
            rank_temperature=rank_temperature,
            structure_min_parts=structure_min_parts,
            optimizer_name=optimizer,
            weight_decay=weight_decay,
            scheduler_name=scheduler,
            warmup=warmup,
            save_head_attivations=save_head_activations,
        )
        train_losses = structure_metrics['train_total']
        val_losses = structure_metrics['val_total']

    # plot_losses(train_losses, val_losses)

    torch.save(model.state_dict(), f"{out_path}.pth")
    print(f"Saved model at {out_path}.pth\n")
    
    if model_type == 'patch_tokens':
        # if we are working with weighted attention head, images test tensors must be calculated after the model is trained
        images, texts = get_image_and_text_tensor(args.test_dataset, args.feature_name, model=model)
    
    if texts is not None:
        texts_proj = model.project_clip_txt(texts.to(device).float()).detach().cpu()
        print("Retrieval results (t2i, i2t):")
        t2i_rk = t2i(images.numpy(), texts_proj.numpy())
        i2t_rk = i2t(images.numpy(), texts_proj.numpy())

        data = [
            ['t2i'] + list(t2i_rk),
            ['i2t'] + list(i2t_rk)
        ]

        columns = ['type', 'r1', 'r5', 'r10', 'median_rank', 'mean_rank']

        df = pd.DataFrame(data, columns=columns)
        print(df)
        
def plot_losses(train_losses, val_losses, labels=["Training Loss", "Validation Loss"]):
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label=labels[0], color='blue', marker='o')
    plt.plot(val_losses, label=labels[1], color='red', marker='o')
    
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_dim', type=int, default=518, help="Crop dimension, irrelevant in case of pre-extracted features")
    parser.add_argument('--data_dir', type=str, default="../coco/", help="Directory of the images") 
    parser.add_argument('--feature_name', type=str, default="disentangled_self_attn", help="Name of the field of the features")
    parser.add_argument('--text_features', type=str, default='ann_feats', help="Name of the field of the text features")
    parser.add_argument('--model_config', type=str, default="dinov2_vitl14_reg", help="Model configuration")
    parser.add_argument('--resize_dim', type=int, default=518, help="Resize dimension, irrelevant in case of pre-extracted features")
    parser.add_argument('--test_dataset', type=str, default='../coco2014_features/test.pth', help="Directory of the test file") 
    parser.add_argument('--train_dataset', type=str, default='../coco2014_features/train.pth', help="Directory of the train file") 
    parser.add_argument('--val_dataset', type=str, default='../coco2014_features/val.pth', help="Directory of the validation file") 
    parser.add_argument('--use_wandb', default=False, action="store_true", help="If setted wandb will be used") 
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to be used")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay to be used")
    parser.add_argument('--scheduler', type=str, default='linear', help="Scheduler to be used")
    parser.add_argument('--name_pedix', type=str, default='', help="Model name to append to name of the configuration for weights name")
    parser.add_argument('--save_head_activations', type=str, default=None, help="If setted the occurences of the head activation of the last epoch will be saved at that path")
    parser.add_argument('--warmup', type=int, default=0, help="Number of warmup epochs")
    parser.add_argument('--init_weight', type=str, default=None,
                        help='Optional projector checkpoint used to initialize/fine-tune the model.')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Optional override of train.num_epochs from the YAML.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Optional override of train.lr from the YAML.')
    parser.add_argument('--structure_bank', type=str, default=None,
                        help='PTH containing raw part text features and object_groups.')
    parser.add_argument(
        '--structure_teacher_bank',
        type=str,
        default=None,
        help=(
            'Optional PTH providing the fixed PartStruct teacher relations. '
            'If omitted, teacher defaults to structure_bank (original behavior).'
        ),
    )
    parser.add_argument('--structure_weight', type=float, default=0.0,
                        help='Lambda for L_total = L_InfoNCE + lambda * L_structure.')
    parser.add_argument('--rank_temperature', type=float, default=0.05,
                        help='Soft-rank temperature for the structure loss.')
    parser.add_argument('--structure_min_parts', type=int, default=3,
                        help='Minimum number of parts required for one object group.')
    args = parser.parse_args()
    
    if args.use_wandb:
        import wandb
        wandb.init(project='dino-clip')
    
    # if the model config name contains 'dino', it means that we do not work with pre-extracted features
    if not ('dino' in args.model_config):
        val_dataset = DinoClipDataset(args.val_dataset, 
                                      features_name='avg_self_attn_out' if args.feature_name == 'disentangled_self_attn' else args.feature_name,
                                      text_features=args.text_features,
                                      load_attn_maps=args.feature_name == 'patch_tokens',
                                      is_wds='.tar' in args.val_dataset)
        train_dataset = DinoClipDataset(args.train_dataset,
                                        features_name=args.feature_name,
                                        text_features=args.text_features,
                                        load_attn_maps=args.feature_name == 'patch_tokens',
                                        is_wds='.tar' in args.train_dataset) 
    else:
        image_transforms = T.Compose([
            T.Resize(args.resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(args.crop_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        train_dataset = COCOCaptions(args.train_dataset, 'coco/train2014', "train", image_transforms, clip.tokenize)
        val_dataset = COCOCaptions(args.val_dataset, 'coco/val2014', "val", image_transforms, clip.tokenize)
    
    if args.feature_name == 'patch_tokens':
        if args.text_features == "clip_second_last_out":
            images, texts, text_argmax = get_image_and_text_tensor(args.test_dataset, args.feature_name, args.text_features)
        else:
            images, texts = get_image_and_text_tensor(args.test_dataset, args.feature_name, args.text_features)
    else:
        images = None
        texts = None
    
    train_and_eval(args.model_config,
                   train_dataset,
                   val_dataset,
                   texts,
                   images,
                   test_set=args.test_dataset,
                   model_type='',
                   optimizer=args.optimizer,
                   weight_decay=args.weight_decay,
                   scheduler=args.scheduler,
                   warmup=args.warmup,
                   name_pedix=args.name_pedix,
                   save_head_activations=args.save_head_activations,
                   init_weight=args.init_weight,
                   num_epochs_override=args.num_epochs,
                   lr_override=args.lr,
                   structure_bank=args.structure_bank,
                   structure_teacher_bank=args.structure_teacher_bank,
                   structure_weight=args.structure_weight,
                   rank_temperature=args.rank_temperature,
                   structure_min_parts=args.structure_min_parts)
