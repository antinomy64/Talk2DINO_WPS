#!/usr/bin/env python3
"""PP116 offline within-object Text-vs-Full-image-GT relation audit.

Read-only use of existing project files. Does NOT rebuild GT prototypes, train, or
modify any official source/weights. Run from Talk2DINO_official_bg repository root.

Text: raw CLIP prompt means -> frozen COCO object projector -> L2 normalize.
Visual: existing FULL-IMAGE DINOv2 GT patchpool prototypes -> L2 normalize.
Only pairs/triplets belonging to the SAME object are evaluated.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TEXT = "feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"
DEFAULT_PROJECTOR = "weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
DEFAULT_CONFIG = "configs/vitb_mlp_infonce.yaml"
DEFAULT_GT = "feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool.pt"
DEFAULT_CLASSES = "src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py"
DEFAULT_OUTPUT = "output/pp116_within_object_text_gt_audit"


def parse_args():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--text-bank", default=DEFAULT_TEXT)
    a.add_argument("--projector", default=DEFAULT_PROJECTOR)
    a.add_argument("--model-config", default=DEFAULT_CONFIG)
    a.add_argument("--gt-bank", default=DEFAULT_GT)
    a.add_argument("--classes-file", default=DEFAULT_CLASSES)
    a.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    a.add_argument("--device", default="cpu")
    a.add_argument("--text-delta", type=float, default=0.05,
                   help="Only text triplets with positive text gap > this delta")
    a.add_argument("--visual-margin", type=float, default=0.05,
                   help="Visual ranking hinge margin, as in Structure Loss")
    a.add_argument("--iou-csv", default="", help="Optional existing per-part IoU CSV, no inference")
    a.add_argument("--baked-projector", default="",
                   help="Optional e100 baked projector .pth for structure-invariance check")
    a.add_argument("--worst-per-part", type=int, default=5)
    a.add_argument("--no-plots", action="store_true")
    a.add_argument("--self-test", action="store_true", help="Run independent triplet formula test")
    return a.parse_args()


def require_file(path):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_pt(path):
    # Only load the user's own local, trusted .pt files.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot safely load {path}: {exc}") from exc


def canonical_classes(path):
    """Read the official PART_CLASSES constant without importing mmseg/registration."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PART_CLASSES"
                                                  for t in node.targets):
            names = [str(v) for v in ast.literal_eval(node.value)]
            if len(names) != 116 or len(set(names)) != 116:
                raise ValueError("Official taxonomy must contain exactly 116 distinct parts")
            return names
    raise ValueError(f"PART_CLASSES not found in {path}")


def verify_names(observed, canonical, src):
    observed = [str(x) for x in observed]
    if len(observed) != len(canonical):
        raise ValueError(f"{src}: class count mismatch {len(observed)} != {len(canonical)}")
    mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(observed, canonical)) if a != b]
    if mismatch:
        raise ValueError(f"{src}: class ID ORDER/NAMES mismatch. First examples: {mismatch[:6]}")


def object_of(name):
    if "'s " not in name:
        raise ValueError(f"Unexpected PP116 class naming {name!r}")
    return name.split("'s ", 1)[0]


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 3:
        return math.nan

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        left = 0
        while left < len(values):
            right = left + 1
            while right < len(values) and sorted_values[right] == sorted_values[left]:
                right += 1
            ranks[order[left:right]] = 0.5 * (left + right - 1) + 1
            left = right
        return ranks

    xr, yr = ranks(x), ranks(y)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def nanmean(vals):
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).any():
        return math.nan
    return float(np.nanmean(a))


def triplet_rows(ids, txt, vis, names, delta, margin, patch_counts, image_counts):
    rows = []
    for p in ids:
        other = [q for q in ids if q != p]
        for a in range(len(other)):
            for b in range(a + 1, len(other)):
                q, r = other[a], other[b]
                gap = float(txt[p, q] - txt[p, r])
                if abs(gap) <= delta:  # ties / ambiguous text relations do not constrain GT
                    continue
                if gap < 0:
                    q, r = r, q
                    gap = -gap
                gt_gap = float(vis[p, q] - vis[p, r])
                hinge = max(0.0, margin - gt_gap)
                rows.append({
                    "object": object_of(names[p]), "anchor_id": p, "anchor": names[p],
                    "text_closer_id": q, "text_closer": names[q],
                    "text_farther_id": r, "text_farther": names[r],
                    "text_cos_closer": float(txt[p, q]), "text_cos_farther": float(txt[p, r]),
                    "text_gap": gap,
                    "gt_cos_closer": float(vis[p, q]), "gt_cos_farther": float(vis[p, r]),
                    "gt_gap": gt_gap, "rank_correct": int(gt_gap > 0),
                    "rank_inverted_or_tied": int(gt_gap <= 0),
                    "margin_satisfied": int(gt_gap >= margin), "gt_relation_hinge": hinge,
                    "anchor_gt_patches": int(patch_counts[p]),
                    "closer_gt_patches": int(patch_counts[q]),
                    "farther_gt_patches": int(patch_counts[r]),
                    "anchor_gt_images": int(image_counts[p]),
                })
    return rows


def summarize_trips(rows):
    n = len(rows)
    if not n:
        return {"valid_triplets": 0, "rank_accuracy": math.nan,
                "inversion_rate": math.nan, "margin_pass_rate": math.nan,
                "gt_relation_loss": math.nan}
    return {
        "valid_triplets": n,
        "rank_accuracy": sum(r["rank_correct"] for r in rows) / n,
        "inversion_rate": sum(r["rank_inverted_or_tied"] for r in rows) / n,
        "margin_pass_rate": sum(r["margin_satisfied"] for r in rows) / n,
        "gt_relation_loss": sum(r["gt_relation_hinge"] for r in rows) / n,
    }


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not columns:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_iou_csv(path, names):
    """Optional: accept long (epoch, part, iou) or wide (epoch, part-named columns).

    Refuse unknown formats instead of guessing rows/IDs or silently treating mIoU as part IoU.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        columns = reader.fieldnames or []
    norm = {c.lower().strip(): c for c in columns}
    epcol = next((norm[k] for k in ("epoch", "ep") if k in norm), None)
    namecol = next((norm[k] for k in ("class_name", "part_name", "part", "class") if k in norm), None)
    ioucol = next((norm[k] for k in ("iou", "part_iou", "class_iou") if k in norm), None)
    idcol = next((norm[k] for k in ("class_id", "part_id", "id") if k in norm), None)
    records = defaultdict(dict)
    epoch_columns = {int(m.group(1)): c for c in columns
                     if (m := re.fullmatch(r"E(\d{1,3})", c)) is not None}
    if epcol is None and epoch_columns and (namecol or idcol):
        # Actual historical PP116 file: one row per part, columns part,E000,E010,...,E100.
        for row in data:
            if namecol and row[namecol] in names:
                name = row[namecol]
            elif idcol and row[idcol].strip().isdigit() and 0 <= int(row[idcol]) < 116:
                name = names[int(row[idcol])]
            else:
                continue
            for ep, col in epoch_columns.items():
                value = row[col].strip()
                if value and value.lower() not in ("nan", "none", "-", "n/a"):
                    records[name][ep] = float(value)
    elif epcol is None:
        raise ValueError(f"IoU CSV needs epoch column or E000...E100 columns; available: {columns[:20]}")
    elif ioucol and (namecol or idcol):
        for row in data:
            ep = int(float(row[epcol]))
            if namecol and row[namecol] in names:
                name = row[namecol]
            elif idcol and row[idcol].strip().isdigit() and 0 <= int(row[idcol]) < 116:
                name = names[int(row[idcol])]
            else:
                continue
            value = row[ioucol].strip()
            if value and value.lower() not in ("nan", "none", "-", "n/a"):
                records[name][ep] = float(value)
    else:
        known = [c for c in columns if c in names]
        if not known:
            raise ValueError("IoU CSV not recognized: need (epoch,part,iou) or epoch + canonical class-name columns")
        for row in data:
            ep = int(float(row[epcol]))
            for name in known:
                value = row[name].strip()
                if value and value.lower() not in ("nan", "none", "-", "n/a"):
                    records[name][ep] = float(value)
    if not records:
        raise ValueError("IoU CSV contains no matched PP116 part records")
    all_vals = [v for d in records.values() for v in d.values()]
    # If points are fractions use percent; if already percentages leave them unchanged.
    # Explicitly record inferred scale in the protocol output.
    in_fraction = bool(all_vals and max(all_vals) <= 1.000001)
    factor = 100 if in_fraction else 1
    stats = {}
    for name, history in records.items():
        h = {ep: val * factor for ep, val in history.items()}
        end_ep = max(h)
        stats[name] = {
            "iou_ep0_pct": h.get(0, math.nan),
            "iou_ep100_pct": h.get(100, math.nan),
            "iou_last_epoch": end_ep,
            "iou_last_pct": h[end_ep],
            "iou_peak_pct": max(h.values()),
            "iou_peak_epoch": max(h, key=h.get),
            "iou_peak_to_last_drop_pct": max(h.values()) - h[end_ep],
            "iou_change_0_to_100_pct": h.get(100, math.nan) - h.get(0, math.nan),
            "iou_epoch_count": len(h),
        }
    return stats, ("fraction_to_percentage" if in_fraction else "already_percent")


def draw_plots(out, groups, txt, vis, names, per_parts, iou_present):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotroot = out / "object_heatmaps"
    plotroot.mkdir(exist_ok=True)
    for object_name, ids in groups.items():
        ids = [i for i in ids if any(r["class_id"] == i and r["valid_gt"] for r in per_parts)]
        if len(ids) < 2:
            continue
        arrt, arrv = txt[np.ix_(ids, ids)], vis[np.ix_(ids, ids)]
        labels = [names[i].split("'s ", 1)[-1] for i in ids]
        fig, axes = plt.subplots(1, 3, figsize=(max(11, len(ids) * 1.2), 4.6))
        for ax, arr, ttl, vmin, vmax, cmap in zip(
                axes, (arrt, arrv, arrv - arrt),
                ("Projected text cosine", "Full-image GT cosine", "GT - text"),
                (-1, -1, -1), (1, 1, 1), ("viridis", "viridis", "coolwarm")):
            im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(ttl)
            ax.set_xticks(range(len(ids)), labels, rotation=70, ha="right", fontsize=7)
            ax.set_yticks(range(len(ids)), labels, fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.74)
        fig.suptitle(f"Within-object relations: {object_name}")
        fig.tight_layout()
        fig.savefig(plotroot / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', object_name)}.png", dpi=150)
        plt.close(fig)
    xs = [r["within_spearman"] for r in per_parts if math.isfinite(r["within_spearman"])]
    ys = [r["rank_accuracy"] for r in per_parts if math.isfinite(r["within_spearman"])]
    if xs:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.scatter(xs, ys, s=25)
        ax.set(xlabel="Part within-object Spearman", ylabel="Part ranking accuracy",
               title="GT relation consistency (one point per part)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "spearman_vs_rank_accuracy.png", dpi=160)
        plt.close(fig)
    if iou_present:
        records = [(r["rank_accuracy"], r.get("iou_ep100_pct", math.nan), r["class_name"])
                   for r in per_parts]
        records = [(x,y,z) for x,y,z in records if math.isfinite(x) and math.isfinite(y)]
        if records:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter([r[0] for r in records], [r[1] for r in records], s=25)
            ax.set(xlabel="Within-object GT ranking accuracy", ylabel="Epoch 100 IoU (%)",
                   title="GT relation compatibility vs per-part IoU")
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(out / "rank_accuracy_vs_iou_e100.png", dpi=160)
            plt.close(fig)


def test_formulas():
    names = ["cat's head", "cat's ear", "cat's leg"]
    t = np.array([[1,.85,.45],[.85,1,.4],[.45,.4,1]])
    v = np.array([[1,.4,.7],[.4,1,.3],[.7,.3,1]])
    rs = triplet_rows([0,1,2], t, v, names, 0.05, 0.05, [10]*3, [3]*3)
    r = next(x for x in rs if x["anchor_id"] == 0)
    assert r["text_closer_id"] == 1 and r["text_farther_id"] == 2
    assert abs(r["gt_gap"] + .3) < 1e-9
    assert abs(r["gt_relation_hinge"] - .35) < 1e-9
    assert r["rank_inverted_or_tied"] == 1
    print("[PASS] Ranking triplet numerical self-test")


def main():
    args = parse_args()
    test_formulas()
    if args.self_test:
        return
    if args.text_delta < 0 or args.visual_margin < 0:
        raise ValueError("Margins cannot be negative")
    if args.worst_per_part < 1:
        raise ValueError("--worst-per-part must be >= 1")
    repo = Path.cwd()
    paths = {
        "text_bank": require_file(args.text_bank),
        "projector": require_file(args.projector),
        "model_config": require_file(args.model_config),
        "gt_bank": require_file(args.gt_bank),
        "classes_file": require_file(args.classes_file),
    }
    canonical = canonical_classes(paths["classes_file"])
    import train_relproto_alignemt as tr  # use training's actual projector path
    raw, raw_names, raw_info = tr.load_raw_clip_bank(
        paths["text_bank"], expected_clip_model="ViT-B/16",
        expected_template="sub_imagenet_template")
    verify_names(raw_names, canonical, "CLIP text bank")
    device = torch.device(args.device)
    projector, proj_info = tr.load_frozen_projector(
        project_root=repo, config_path=paths["model_config"],
        weight_path=paths["projector"], device=device)
    text_features = tr.project_raw_clip_bank(raw, projector, device=device, batch_size=128).float()
    text_features = tr.normalize_last(text_features).cpu()
    del projector
    gt = read_pt(paths["gt_bank"])
    if not isinstance(gt, dict):
        raise TypeError("GT bank must be a dict from build_pp116_gt_patchpool_prototypes.py")
    required = ("prototypes", "patch_counts", "image_counts", "classes")
    if any(k not in gt for k in required):
        raise KeyError(f"GT bank missing fields: {[k for k in required if k not in gt]}")
    verify_names(gt["classes"], canonical, "GT prototype bank")
    raw_visual = torch.as_tensor(gt["prototypes"], dtype=torch.float32)
    patch_counts = torch.as_tensor(gt["patch_counts"], dtype=torch.long).view(-1).numpy()
    image_counts = torch.as_tensor(gt["image_counts"], dtype=torch.long).view(-1).numpy()
    if raw_visual.shape != (116,768) or text_features.shape != (116,768):
        raise ValueError(f"Expected [116,768] both, got {tuple(raw_visual.shape)} and {tuple(text_features.shape)}")
    if patch_counts.shape != (116,) or image_counts.shape != (116,):
        raise ValueError("GT patch_counts/image_counts must both be [116]")
    if not torch.isfinite(raw_visual).all():
        raise ValueError("Nonfinite GT prototypes")
    norms = raw_visual.norm(dim=-1)
    valid = (patch_counts > 0) & (image_counts > 0) & (norms.numpy() > 1e-12)
    if np.any((patch_counts > 0) != (norms.numpy() > 1e-12)):
        raise ValueError("GT support and nonzero-prototype rows disagree")
    visual_features = torch.zeros_like(raw_visual)
    visual_features[valid] = F.normalize(raw_visual[valid], dim=-1)
    t = (text_features.double() @ text_features.double().T).numpy()
    v = (visual_features.double() @ visual_features.double().T).numpy()
    t = np.clip(t, -1., 1.)
    v = np.clip(v, -1., 1.)
    groups = defaultdict(list)
    for i, name in enumerate(canonical):
        groups[object_of(name)].append(i)
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty, refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    iou_stats, iou_scale = {}, None
    if args.iou_csv:
        iou_stats, iou_scale = read_iou_csv(require_file(args.iou_csv), canonical)
    per_parts, per_objects, all_trips, relation_pairs = [], [], [], []
    for obj, members in groups.items():
        ids = [i for i in members if valid[i]]
        all_obj_trips = triplet_rows(ids, t, v, canonical,
                                    args.text_delta, args.visual_margin,
                                    patch_counts, image_counts)
        all_trips.extend(all_obj_trips)
        top = np.triu_indices(len(ids), k=1)
        tt = t[np.ix_(ids, ids)][top]
        vv = v[np.ix_(ids, ids)][top]
        metric = summarize_trips(all_obj_trips)
        metric.update({
            "object": obj, "num_taxonomy_parts": len(members),
            "num_valid_gt_parts": len(ids), "num_missing_gt_parts": len(members)-len(ids),
            "num_pairs": len(tt), "within_spearman": spearman(tt, vv),
            "pair_cosine_mae": float(np.mean(np.abs(tt-vv))) if len(tt) else math.nan,
        })
        per_objects.append(metric)
        for aa, a in enumerate(ids):
            for b in ids[aa+1:]:
                relation_pairs.append({
                    "object": obj, "part_a_id": a, "part_a": canonical[a],
                    "part_b_id": b, "part_b": canonical[b],
                    "text_cos": float(t[a,b]), "gt_cos": float(v[a,b]),
                    "gt_minus_text": float(v[a,b]-t[a,b]),
                    "abs_cos_error": abs(float(v[a,b]-t[a,b])),
                    "a_gt_patches": int(patch_counts[a]), "b_gt_patches": int(patch_counts[b]),
                })
        for p in members:
            others = [q for q in ids if q != p] if valid[p] else []
            rtext = [t[p,q] for q in others]
            rgt = [v[p,q] for q in others]
            anchor_triplets = [r for r in all_obj_trips if r["anchor_id"] == p]
            row = {
                "class_id": p, "class_name": canonical[p], "object": obj,
                "part": canonical[p].split("'s ",1)[1],
                "valid_gt": int(valid[p]), "gt_patch_count": int(patch_counts[p]),
                "gt_image_count": int(image_counts[p]),
                "object_taxonomy_size": len(members), "valid_reference_parts": len(others),
                "within_spearman": spearman(rtext,rgt),
                "pair_cosine_mae": float(np.mean(np.abs(np.asarray(rtext)-np.asarray(rgt)))) if others else math.nan,
                **summarize_trips(anchor_triplets),
            }
            if args.iou_csv:
                for key in ("iou_ep0_pct", "iou_ep100_pct", "iou_last_epoch", "iou_last_pct",
                            "iou_peak_pct", "iou_peak_epoch", "iou_peak_to_last_drop_pct",
                            "iou_change_0_to_100_pct", "iou_epoch_count"):
                    row[key] = math.nan
                row.update(iou_stats.get(canonical[p], {}))
            per_parts.append(row)
    worst = []
    for p, name in enumerate(canonical):
        mine = [r for r in all_trips if r["anchor_id"] == p]
        mine.sort(key=lambda x: (x["gt_relation_hinge"], -x["gt_gap"]), reverse=True)
        worst.extend(mine[:args.worst_per_part])
    object_order = sorted(per_objects, key=lambda x: (math.isnan(x["gt_relation_loss"]),
                                                       -x["gt_relation_loss"] if math.isfinite(x["gt_relation_loss"]) else 0))
    part_order = sorted(per_parts, key=lambda x: (math.isnan(x["gt_relation_loss"]),
                                                   -x["gt_relation_loss"] if math.isfinite(x["gt_relation_loss"]) else 0))
    write_csv(out/"object_structure.csv", object_order)
    write_csv(out/"part_structure.csv", part_order)
    write_csv(out/"within_object_pairs.csv", relation_pairs)
    write_csv(out/"all_valid_triplets.csv", all_trips,
              columns=list(all_trips[0]) if all_trips else ["object", "anchor_id", "anchor", "text_closer", "text_farther", "text_gap", "gt_gap", "gt_relation_hinge"])
    write_csv(out/"worst_triplets_per_part.csv", worst,
              columns=list(all_trips[0]) if all_trips else ["object", "anchor_id", "anchor", "text_closer", "text_farther", "text_gap", "gt_gap", "gt_relation_hinge"])
    write_csv(out/"missing_gt_parts.csv", [
        {"class_id":i,"class_name":name,"object":object_of(name),
         "patch_counts":int(patch_counts[i]), "image_counts":int(image_counts[i])}
        for i,name in enumerate(canonical) if not valid[i]],
        ["class_id","class_name","object","patch_counts","image_counts"])
    baked = None
    if args.baked_projector:
        ckpt = require_file(args.baked_projector)
        baked_proj, _ = tr.load_frozen_projector(project_root=repo,
            config_path=paths["model_config"], weight_path=ckpt, device=device)
        baked_text = tr.project_raw_clip_bank(raw, baked_proj, device=device, batch_size=128).double()
        baked_sim = (baked_text @ baked_text.T).numpy()
        baked = {"checkpoint":str(ckpt), "sha256":file_sha256(ckpt),
            "within_object_max_abs_difference": max(
                (float(abs(baked_sim[i,j]-t[i,j])) for ids in groups.values()
                 for i in ids for j in ids if i<j), default=math.nan),
        }
        del baked_proj
        print("[E100 baked projector] max |text pairwise delta| within objects:",
              baked["within_object_max_abs_difference"])
        if baked["within_object_max_abs_difference"] > 1e-3:
            print("[WARN] e100 baked projector is not preserving pairwise cosine to 1e-3."
                  " Check source run, weights, and bake protocol; NOT evidence GT matches.")
    meta = {
        "protocol": "WITHIN_OBJECT_FULLIMAGE_GT_V1",
        "description": "Object and part internal relations, GT prototypes used for OFFLINE audit only",
        "baseline_text_path": "CLIP prompt means (UNNORMALIZED) -> frozen COCO object projector -> L2",
        "gt_path": "Full-image DINOv2 ViT-B/14-Reg pooled by GT part IDs, global patch mean -> L2",
        "reference_definition": "same-object valid GT classes in canonical PART_CLASSES taxonomy",
        "triplets_definition": "per anchor, unique unordered q/r comparison; text gap strictly > delta",
        "rank_accuracy": "fraction GT gap strictly > 0 (ties are incorrect)",
        "gt_relation_loss": "mean max(0, visual_margin - GT cosine gap)",
        "text_delta":args.text_delta,"visual_margin":args.visual_margin,
        "counts": {"num_objects":len(groups),"num_valid_parts":int(valid.sum()),
                   "num_triplets":len(all_trips),"num_taxonomy_parts":116},
        "iou_csv": args.iou_csv or None, "iou_scale": iou_scale,
        "input_files":{key:{"path":str(p), "sha256":file_sha256(p)} for key,p in paths.items()},
        "text_bank_info": raw_info,
        "gt_metadata": {str(k):v for k,v in gt.get("metadata",{}).items()
                        if isinstance(v,(str,int,float,bool,list))},
        "baked_projector_check":baked,
    }
    (out/"protocol.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),encoding="utf-8")
    if not args.no_plots:
        draw_plots(out, groups, t, v, canonical, per_parts, bool(iou_stats))
    print("\n" + "=" * 80)
    print("WITHIN-OBJECT TEXT vs FULL-IMAGE GT RELATION AUDIT")
    print("=" * 80)
    print("Objects:",len(groups),"valid GT parts:",valid.sum(),"valid triplets:",len(all_trips))
    print("\nHighest per-part GT relation loss (not automatically a wrong class):")
    for r in part_order[:15]:
        if not math.isfinite(r["gt_relation_loss"]):
            continue
        print(f"{r['class_name']:<30} spearman={r['within_spearman']:7.3f}  "
              f"rankAcc={r['rank_accuracy']:6.1%}  loss={r['gt_relation_loss']:7.4f}  "
              f"triplets={r['valid_triplets']:3}  patches={r['gt_patch_count']:7}")
    print("\nSaved:",out)
    for f in sorted(out.glob("*.csv")):
        print(" -",f.name)
    print(" - protocol.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}",file=sys.stderr)
        raise
