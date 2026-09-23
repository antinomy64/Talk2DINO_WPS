#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.model import ProjectionLayer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weight_dir",
        default="nightly_ablation_pp116_seed123_20260921/runs/core/relative_orthogonal_L8/train",
    )
    p.add_argument(
        "--base_projector",
        default="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth",
    )
    p.add_argument(
        "--text_bank",
        default="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt",
    )
    p.add_argument(
        "--gt_prototypes",
        default="feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt",
    )
    p.add_argument("--model_config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument(
        "--classes_source",
        default="src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py:PART_CLASSES",
    )
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--end_epoch", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out_dir",
        default="nightly_ablation_pp116_seed123_20260921/analysis/relative_orthogonal_L8_gt_structure",
    )
    return p.parse_args()


def load_classes(source):
    py_path, var_name = source.split(":", 1)
    spec = importlib.util.spec_from_file_location("pp116_class_module", str(Path(py_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    classes = list(getattr(mod, var_name))
    if len(classes) != 116:
        raise RuntimeError(f"Expected 116 classes, got {len(classes)}")
    return classes


def find_tensor(obj, expected_n=116, expected_dim=None):
    candidates = []

    def visit(x, name="root"):
        if torch.is_tensor(x):
            if x.ndim == 2 and x.shape[0] == expected_n:
                if expected_dim is None or x.shape[1] == expected_dim:
                    candidates.append((name, x))
            return

        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, f"{name}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                visit(v, f"{name}[{i}]")

    visit(obj)

    if not candidates:
        raise RuntimeError(f"Cannot find tensor [{expected_n}, {expected_dim or '*'}]")

    if len(candidates) > 1:
        print("[INFO] candidate tensors:")
        for name, tensor in candidates:
            print(f"  {name}: {tuple(tensor.shape)}")

    name, tensor = candidates[0]
    print(f"[INFO] selected tensor: {name}, shape={tuple(tensor.shape)}")
    return tensor.float()


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj

    for key in ["state_dict", "model_state_dict", "model", "projector", "proj"]:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    return obj


def load_projector(config_path, weight_path, device):
    model = ProjectionLayer.from_config(config_path)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = unwrap_state_dict(ckpt)

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "proj.", "projection_layer."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def extract_projected_text(projector, raw_text):
    text = projector.project_clip_txt(raw_text.float()).float()
    return F.normalize(text, dim=-1)


def object_name_from_class(cls):
    if "'s " in cls:
        return cls.split("'s ", 1)[0]
    if "’s " in cls:
        return cls.split("’s ", 1)[0]
    return cls.split(" ", 1)[0]


def make_groups(classes):
    groups = defaultdict(list)

    for i, cls in enumerate(classes):
        groups[object_name_from_class(cls)].append(i)

    return dict(groups)


def upper_triangle_values(mat):
    n = mat.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=mat.device)
    return mat[idx[0], idx[1]]


def matched_cosine_metrics(text, gt, valid_mask, groups):
    cos = (text * gt).sum(dim=-1)
    values = cos[valid_mask]

    mean_cos = values.mean().item()
    median_cos = values.median().item()

    valid_cpu = valid_mask.detach().cpu()
    obj_means = []

    for _, ids in groups.items():
        ids = [i for i in ids if bool(valid_cpu[i].item())]
        if not ids:
            continue

        x = cos[ids].mean().item()
        if math.isfinite(x):
            obj_means.append(x)

    object_macro = float(np.mean(obj_means))
    return mean_cos, median_cos, object_macro


def relational_metrics(text, gt, valid_mask, groups):
    spearman_list = []
    structure_cos_list = []
    per_object = []

    valid_cpu = valid_mask.detach().cpu()

    for obj, ids in groups.items():
        ids = [i for i in ids if bool(valid_cpu[i].item())]

        if len(ids) < 3:
            continue

        t = text[ids]
        g = gt[ids]

        t_vec = upper_triangle_values(t @ t.T)
        g_vec = upper_triangle_values(g @ g.T)

        if t_vec.numel() < 3:
            continue

        t_np = t_vec.detach().cpu().numpy()
        g_np = g_vec.detach().cpu().numpy()

        if np.std(t_np) < 1e-12 or np.std(g_np) < 1e-12:
            continue

        rho = spearmanr(t_np, g_np).statistic
        if not np.isfinite(rho):
            continue

        structure_cos = F.cosine_similarity(
            t_vec.unsqueeze(0),
            g_vec.unsqueeze(0),
            dim=-1,
        ).item()

        spearman_list.append(float(rho))
        structure_cos_list.append(float(structure_cos))

        per_object.append(
            {
                "object": obj,
                "n_parts": len(ids),
                "n_pairs": len(t_np),
                "spearman": float(rho),
                "structure_cosine": float(structure_cos),
            }
        )

    if not spearman_list:
        raise RuntimeError("No valid object groups for Spearman.")

    return (
        float(np.mean(spearman_list)),
        float(np.mean(structure_cos_list)),
        per_object,
    )


def structure_retention_metrics(text, base_text, valid_mask, groups):
    spearman_list = []
    per_object = []

    valid_cpu = valid_mask.detach().cpu()

    for obj, ids in groups.items():
        ids = [i for i in ids if bool(valid_cpu[i].item())]

        if len(ids) < 3:
            continue

        t = text[ids]
        t0 = base_text[ids]

        t_vec = upper_triangle_values(t @ t.T)
        t0_vec = upper_triangle_values(t0 @ t0.T)

        if t_vec.numel() < 3:
            continue

        t_np = t_vec.detach().cpu().numpy()
        t0_np = t0_vec.detach().cpu().numpy()

        if np.std(t_np) < 1e-12 or np.std(t0_np) < 1e-12:
            continue

        rho = spearmanr(t_np, t0_np).statistic
        if not np.isfinite(rho):
            continue

        spearman_list.append(float(rho))

        per_object.append(
            {
                "object": obj,
                "n_parts": len(ids),
                "n_pairs": len(t_np),
                "retention_spearman": float(rho),
            }
        )

    if not spearman_list:
        raise RuntimeError("No valid object groups for structure retention.")

    return float(np.mean(spearman_list)), per_object


def global_structure_metrics(text, gt, valid_mask):
    ids = torch.where(valid_mask)[0]

    t = text[ids]
    g = gt[ids]

    t_vec = upper_triangle_values(t @ t.T)
    g_vec = upper_triangle_values(g @ g.T)

    t_np = t_vec.detach().cpu().numpy()
    g_np = g_vec.detach().cpu().numpy()

    rho = spearmanr(t_np, g_np).statistic

    structure_cos = F.cosine_similarity(
        t_vec.unsqueeze(0),
        g_vec.unsqueeze(0),
        dim=-1,
    ).item()

    return float(rho), float(structure_cos)


def load_epoch_W(path, device):
    ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected W checkpoint type: {type(ckpt)}")

    if "W" not in ckpt:
        raise KeyError(f"{path} does not contain key 'W'")

    W = ckpt["W"].float()

    if W.shape != (768, 768):
        raise RuntimeError(f"Expected W [768,768], got {tuple(W.shape)}")

    selector = ckpt.get("selector")
    mapping = ckpt.get("mapping")

    if selector is not None and selector != "relative":
        raise RuntimeError(f"Expected selector=relative, got {selector}")

    if mapping is not None and mapping not in ["orthogonal", "linear"]:
        raise RuntimeError(f"Unexpected mapping={mapping}")

    return W.to(device), ckpt


def orthogonality_metrics(W):
    I = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)

    err = W.T @ W - I
    max_abs = err.abs().max().item()
    fro = torch.linalg.norm(err, ord="fro").item()
    W_minus_I = torch.linalg.norm(W - I, ord="fro").item()

    return max_abs, fro, W_minus_I


def save_csv(path, rows, fieldnames=None):
    with open(path, "w", newline="") as f:
        if fieldnames is None:
            fieldnames = list(rows[0].keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device(args.device)

    weight_dir = Path(args.weight_dir)
    out_dir = Path(args.out_dir)
    per_obj_dir = out_dir / "per_object"

    out_dir.mkdir(parents=True, exist_ok=True)
    per_obj_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.classes_source)
    groups = make_groups(classes)

    print(f"[INFO] classes={len(classes)}, object_groups={len(groups)}")

    for obj, ids in groups.items():
        print(f"  {obj:15s}: {len(ids)} parts")

    text_obj = torch.load(args.text_bank, map_location="cpu")
    raw_text = find_tensor(
        text_obj,
        expected_n=116,
        expected_dim=512,
    ).to(device)

    gt_obj = torch.load(args.gt_prototypes, map_location="cpu")

    if "prototypes" not in gt_obj:
        raise KeyError("GT prototype file does not contain key 'prototypes'")

    gt = gt_obj["prototypes"].float()

    if gt.shape != (116, 768):
        raise RuntimeError(
            f"Expected GT prototypes [116,768], got {tuple(gt.shape)}"
        )

    if "patch_counts" in gt_obj:
        valid_mask = gt_obj["patch_counts"].long() > 0
    else:
        valid_mask = (
            torch.isfinite(gt).all(dim=-1)
            & (gt.norm(dim=-1) > 1e-8)
        )

    print("[INFO] GT prototype key: prototypes")
    print("[INFO] GT prototype shape:", tuple(gt.shape))
    print("[INFO] valid GT parts:", int(valid_mask.sum()), "/ 116")

    gt = F.normalize(gt, dim=-1).to(device)
    valid_mask = valid_mask.to(device)

    print()
    print("[INFO] loading base projector:")
    print("      ", args.base_projector)

    projector = load_projector(
        args.model_config,
        args.base_projector,
        device,
    )

    with torch.no_grad():
        base_text = extract_projected_text(
            projector,
            raw_text,
        )

    if base_text.shape != (116, 768):
        raise RuntimeError(
            f"Unexpected base text shape: {tuple(base_text.shape)}"
        )

    print("[INFO] base projected text shape:", tuple(base_text.shape))

    del projector

    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []

    for epoch in range(
        args.start_epoch,
        args.end_epoch + 1,
    ):
        print()
        print("=" * 80)
        print(f"EPOCH {epoch:03d}")
        print("=" * 80)

        if epoch == 0:
            text = base_text.clone()

            ortho_max_abs = 0.0
            ortho_fro = 0.0
            W_minus_I_fro = 0.0

            selector = "base"
            mapping = "identity"

            print("[INFO] epoch 0 = base Talk2DINO projected text")

        else:
            w_path = weight_dir / f"W_epoch_{epoch:03d}.pt"

            if not w_path.exists():
                print(f"[WARN] missing {w_path}; skip")
                continue

            W, w_ckpt = load_epoch_W(
                w_path,
                device,
            )

            selector = w_ckpt.get("selector", "unknown")
            mapping = w_ckpt.get("mapping", "unknown")

            print("[INFO] W:", w_path)
            print(
                f"[INFO] selector={selector}, "
                f"mapping={mapping}, "
                f"checkpoint_epoch={w_ckpt.get('epoch')}"
            )

            (
                ortho_max_abs,
                ortho_fro,
                W_minus_I_fro,
            ) = orthogonality_metrics(W)

            print(f"[INFO] ||W^T W-I||_max = {ortho_max_abs:.8f}")
            print(f"[INFO] ||W^T W-I||_F   = {ortho_fro:.8f}")
            print(f"[INFO] ||W-I||_F       = {W_minus_I_fro:.8f}")

            # Training / bake convention:
            # adapted_text = Norm(base_text @ W)
            text = F.normalize(
                base_text @ W,
                dim=-1,
            )

        if text.shape != gt.shape:
            raise RuntimeError(
                f"shape mismatch: "
                f"text={tuple(text.shape)}, "
                f"gt={tuple(gt.shape)}"
            )

        (
            matched_mean,
            matched_median,
            matched_obj_macro,
        ) = matched_cosine_metrics(
            text,
            gt,
            valid_mask,
            groups,
        )

        (
            spearman_macro,
            structure_cos_macro,
            per_object,
        ) = relational_metrics(
            text,
            gt,
            valid_mask,
            groups,
        )

        (
            retention_spearman_macro,
            retention_per_object,
        ) = structure_retention_metrics(
            text,
            base_text,
            valid_mask,
            groups,
        )

        (
            spearman_global,
            structure_cos_global,
        ) = global_structure_metrics(
            text,
            gt,
            valid_mask,
        )

        row = {
            "epoch": epoch,
            "spearman_macro": spearman_macro,
            "retention_spearman_macro": retention_spearman_macro,
            "matched_cosine_mean": matched_mean,
            "matched_cosine_median": matched_median,
            "matched_cosine_object_macro": matched_obj_macro,
            "structure_cosine_macro": structure_cos_macro,
            "spearman_global": spearman_global,
            "structure_cosine_global": structure_cos_global,
            "orthogonality_max_abs": ortho_max_abs,
            "orthogonality_fro": ortho_fro,
            "W_minus_I_fro": W_minus_I_fro,
            "selector": selector,
            "mapping": mapping,
            "valid_gt_parts": int(valid_mask.sum().item()),
            "valid_object_groups": len(per_object),
        }

        rows.append(row)

        print()
        print(f"[RESULT] epoch={epoch:03d}")
        print(f"  Spearman macro           = {spearman_macro:.6f}")
        print(f"  Structure retention rho  = {retention_spearman_macro:.6f}")
        print(f"  Matched cosine mean      = {matched_mean:.6f}")
        print(f"  Matched cosine median    = {matched_median:.6f}")
        print(f"  Matched cosine obj-macro = {matched_obj_macro:.6f}")
        print(f"  Structure cosine macro   = {structure_cos_macro:.6f}")
        print(f"  Spearman global          = {spearman_global:.6f}")
        print(f"  Valid object groups      = {len(per_object)}")

        save_csv(
            per_obj_dir / f"epoch_{epoch:03d}.csv",
            per_object,
            [
                "object",
                "n_parts",
                "n_pairs",
                "spearman",
                "structure_cosine",
            ],
        )

        save_csv(
            per_obj_dir / f"epoch_{epoch:03d}_retention.csv",
            retention_per_object,
            [
                "object",
                "n_parts",
                "n_pairs",
                "retention_spearman",
            ],
        )

        del text

        if epoch > 0:
            del W

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No epoch results generated.")

    output_csv = out_dir / "epoch_text_vs_gt_metrics.csv"
    save_csv(
        output_csv,
        rows,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()