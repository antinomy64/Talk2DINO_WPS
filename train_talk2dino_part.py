#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Talk2DINO-Part target adaptation.

Initialize the original Talk2DINO projector from the
COCO-Captions-2014 checkpoint and continue optimizing the same
InfoNCE objective on target image / present-part pairs.

No target masks / boxes / points.
Final epoch is the formal checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DinoClipDataset
from src.loss import ContrastiveLoss


def parse_args():
    p = argparse.ArgumentParser(
        "Continue Talk2DINO projector training on image-level part labels."
    )

    p.add_argument("--train_pth", required=True)
    p.add_argument(
        "--model_config",
        default="configs/vitb_mlp_infonce.yaml",
    )
    p.add_argument(
        "--init_weights",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_reproduce_clean.pth"
        ),
    )
    p.add_argument("--out_dir", required=True)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")

    p.add_argument(
        "--save_every",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max_annotations",
        type=int,
        default=0,
        help="Smoke only; 0 means full dataset.",
    )

    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("seed:", seed)


def strip_module_prefix(sd):
    if sd and all(
        str(k).startswith("module.")
        for k in sd
    ):
        return {
            str(k)[7:]: v
            for k, v in sd.items()
        }

    return sd


def unwrap_state(obj):
    if isinstance(obj, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "projector_state_dict",
        ):
            if key in obj and isinstance(
                obj[key], dict
            ):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise TypeError(
            "Checkpoint is not a state dict."
        )

    sd = {
        str(k): v
        for k, v in obj.items()
        if torch.is_tensor(v)
    }

    return strip_module_prefix(sd)


def build_model(config_path, weight_path, device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]

    model_class_name = model_cfg.get(
        "model_class",
        "ProjectionLayer",
    )

    ModelClass = getattr(
        importlib.import_module("src.model"),
        model_class_name,
    )

    model = ModelClass.from_config(model_cfg)

    ckpt = torch.load(
        weight_path,
        map_location="cpu",
    )
    sd = unwrap_state(ckpt)

    # Accept checkpoints optionally wrapped with proj./model.proj.
    expected = set(model.state_dict().keys())

    if set(sd.keys()) != expected:
        for prefix in (
            "proj.",
            "model.proj.",
        ):
            sub = {
                k[len(prefix):]: v
                for k, v in sd.items()
                if k.startswith(prefix)
            }

            if set(sub.keys()) == expected:
                sd = sub
                break

    # Talk2DINO's projector class overrides load_state_dict()
    # and may return None rather than PyTorch's IncompatibleKeys.
    # We therefore validate keys explicitly before loading.
    if set(sd.keys()) != expected:
        missing = sorted(expected - set(sd.keys()))
        unexpected = sorted(set(sd.keys()) - expected)
        raise RuntimeError(
            "Projector checkpoint mismatch.\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model.load_state_dict(sd)

    model.to(device)

    return model, cfg


def audit_dataset(path):
    x = torch.load(
        path,
        map_location="cpu",
    )

    if "images" not in x or "annotations" not in x:
        raise ValueError(
            "Bad Talk2DINO-Part dataset."
        )

    images = x["images"]
    anns = x["annotations"]

    if not images or not anns:
        raise ValueError("Empty dataset.")

    v = images[0]["disentangled_self_attn"]
    t = anns[0]["ann_feats"]

    if v.ndim != 2 or v.shape[-1] != 768:
        raise ValueError(
            f"bad visual feature shape: {tuple(v.shape)}"
        )

    if tuple(t.shape) != (512,):
        raise ValueError(
            f"bad text feature shape: {tuple(t.shape)}"
        )

    meta = x.get("meta", {})

    if meta.get(
        "target_spatial_supervision",
        "none",
    ) != "none":
        raise RuntimeError(
            "Target spatial-supervision metadata is not none."
        )

    print("dataset images      :", len(images))
    print("dataset annotations :", len(anns))
    print("visual feature      :", tuple(v.shape))
    print("text feature        :", tuple(t.shape))
    print("dataset             :", meta.get("dataset"))
    print("spatial supervision :", meta.get(
        "target_spatial_supervision",
        "none",
    ))


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    train_path = Path(
        args.train_pth
    ).expanduser().resolve()

    config_path = Path(
        args.model_config
    ).expanduser()

    if not config_path.is_absolute():
        config_path = (
            REPO_ROOT / config_path
        ).resolve()

    init_path = Path(
        args.init_weights
    ).expanduser()

    if not init_path.is_absolute():
        init_path = (
            REPO_ROOT / init_path
        ).resolve()

    out_dir = Path(
        args.out_dir
    ).expanduser()

    if not out_dir.is_absolute():
        out_dir = (
            REPO_ROOT / out_dir
        ).resolve()

    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to mix with existing run: {out_dir}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    audit_dataset(train_path)

    dataset = DinoClipDataset(
        str(train_path),
        features_name="disentangled_self_attn",
        text_features="ann_feats",
        load_attn_maps=False,
        is_wds=False,
    )

    if args.max_annotations > 0:
        keep = min(
            args.max_annotations,
            len(dataset),
        )

        # DinoClipDataset stores entries in a dict.
        dataset.data = {
            i: dataset.data[i]
            for i in range(keep)
        }

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
        persistent_workers=args.workers > 0,
    )

    if len(loader) == 0:
        raise RuntimeError(
            "No training batches."
        )

    model, cfg = build_model(
        config_path,
        init_path,
        device,
    )

    # Same Talk2DINO contrastive criterion.
    criterion = ContrastiveLoss(
        model,
        margin=cfg["train"].get(
            "margin",
            0.2,
        ),
        max_violation=cfg["train"].get(
            "max_violation",
            True,
        ),
        ltype="infonce",
    ).to(device)

    # Talk2DINO trains the projector with Adam.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    history = []

    print("=" * 60)
    print("Talk2DINO-Part target adaptation")
    print("=" * 60)
    print("dataset       :", train_path)
    print("samples       :", len(dataset))
    print("batches       :", len(loader))
    print("init          :", init_path)
    print("epochs        :", args.epochs)
    print("batch size    :", args.batch_size)
    print("lr            :", args.lr)
    print("optimizer     : Adam")
    print("loss          : Talk2DINO InfoNCE")
    print("feature       : disentangled_self_attn")
    print("text feature  : ann_feats")
    print("seed          :", args.seed)
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n = 0

        bar = tqdm(
            loader,
            desc=f"epoch {epoch:03d}/{args.epochs:03d}",
        )

        for batch in bar:
            text = batch["annotation"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            image = batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss = criterion(
                image,
                text,
                return_similarity_mat=False,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n += 1

            bar.set_postfix(
                loss=f"{loss.item():.6f}",
            )

        mean_loss = total_loss / max(n, 1)

        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "lr": float(
                optimizer.param_groups[0]["lr"]
            ),
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={mean_loss:.8f}"
        )

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
        ):
            torch.save(
                model.state_dict(),
                out_dir
                / f"projector_epoch_{epoch:03d}.pth",
            )

        (
            out_dir / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    final_path = out_dir / "projector_final.pth"

    torch.save(
        model.state_dict(),
        final_path,
    )

    run_meta = {
        "method": "Talk2DINO-Part",
        "train_dataset": str(train_path),
        "init_weights": str(init_path),
        "model_config": str(config_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "loss": "InfoNCE",
        "feature_name":
            "disentangled_self_attn",
        "text_feature": "ann_feats",
        "seed": args.seed,
        "checkpoint_selection":
            "fixed_final_epoch",
        "target_spatial_supervision":
            "none",
    }

    (
        out_dir / "run_meta.json"
    ).write_text(
        json.dumps(
            run_meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DONE")
    print("final:", final_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
