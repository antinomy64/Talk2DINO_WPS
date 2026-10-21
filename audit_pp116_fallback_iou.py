#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PP116 read-only fallback audit for Soft Relative Competition.

Counts the EXACT training fallback condition for K>=2 candidate parts:
    relative_scores(absolute_text_patch_scores, foreground) > 0
    fallback = (number of positive-margin foreground patches == 0)
K=1 is *absolute* softmax pooling, and is NEVER called a fallback.

Both lambda=0 and lambda=4 use the same cached, image-level part presence
candidate sets and the same original.relative_scores function. Uses NO GT masks.
Validation IoU is read from a finished comparison CSV for OFFLINE correlation.
Does NOT train, evaluate, or overwrite repository/official files.

Examples:
  python audit_pp116_fallback_iou.py --self-test
  CUDA_VISIBLE_DEVICES=0 python audit_pp116_fallback_iou.py --max-crops 64 \
    --epochs 0 10 100 --out-dir output/pp116_fallback_smoke
  CUDA_VISIBLE_DEVICES=0 python audit_pp116_fallback_iou.py \
    --out-dir output/pp116_fallback_full --device cuda
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NUM_PARTS = 116
VISION_DIM = 768
PATCH_COUNT = 1024


def load_trusted_pt(path):
    # Only user-created trusted checkpoints and local cache; may contain argparse Namespace.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def official_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PART_CLASSES" for target in node.targets
        ):
            names = [str(x) for x in ast.literal_eval(node.value)]
            if len(names) != NUM_PARTS or len(set(names)) != NUM_PARTS:
                raise ValueError("Invalid canonical PP116 PART_CLASSES")
            return names
    raise ValueError(f"Cannot find PART_CLASSES in {path}")


def annotations_from_cache(payload):
    if isinstance(payload, (list, tuple)):
        return payload
    if isinstance(payload, dict):
        for key in ("annotations", "records", "samples", "items"):
            if isinstance(payload.get(key), (list, tuple)):
                return payload[key]
    raise ValueError(f"Unknown annotation cache format: {type(payload)}")


def ids_of(annotation):
    if "part_category_id" not in annotation:
        raise KeyError("Expected image-level part_category_id in training cache")
    x = annotation["part_category_id"]
    if torch.is_tensor(x):
        x = x.detach().cpu().flatten().tolist()
    elif isinstance(x, np.ndarray):
        x = x.reshape(-1).tolist()
    elif not isinstance(x, (list, tuple)):
        x = [x]
    ids = [int(v) for v in x]
    if not ids or len(ids) != len(set(ids)) or any(x < 0 or x >= NUM_PARTS for x in ids):
        raise ValueError(f"Invalid candidate part IDs: {ids}")
    return ids


def state_W(run_dir, ep, expected_lambda):
    if ep == 0:
        return torch.eye(VISION_DIM, dtype=torch.float32)
    p = run_dir / f"W_epoch_{ep:03d}.pt"
    if not p.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    payload = load_trusted_pt(p)
    if not isinstance(payload, dict) or "W" not in payload:
        raise ValueError(f"Invalid W checkpoint: {p}")
    opts = payload.get("args", {})
    got_lambda = opts.get("lambda_struct") if isinstance(opts, dict) else getattr(opts, "lambda_struct", None)
    if got_lambda is None or not math.isclose(float(got_lambda), expected_lambda, abs_tol=1e-6):
        raise ValueError(f"Checkpoint lambda mismatch: {p} has {got_lambda}, expected {expected_lambda}")
    W = torch.as_tensor(payload["W"], dtype=torch.float32)
    if W.shape != (VISION_DIM, VISION_DIM):
        raise ValueError(f"Unexpected W dimension: {p}: {W.shape}")
    orth_err = float((W.T @ W - torch.eye(VISION_DIM)).abs().max())
    if orth_err > 1e-2:
        raise ValueError(f"Checkpoint is not orthogonal: {p}, error={orth_err}")
    return W


def positive_count_and_max(text, patch, fg, relative_scores):
    """text[B,K,D], patch[B,N,D], fg[B,N], all samples in group have K>=2.
    Uses source training relative_scores (not an approximation).
    """
    score = torch.bmm(text, patch.transpose(1, 2))
    rel = relative_scores(score, fg)
    if rel.shape != score.shape:
        raise RuntimeError(f"Unexpected original.relative_scores output: {rel.shape}")
    if fg.shape != score.shape[:1] + score.shape[2:]:
        raise RuntimeError("Foreground shape differs from source inputs")
    # Make the foreground-only requirement explicit, irrespective of the
    # sentinel value used internally by original.relative_scores.
    allowed = (rel > 0.0) & fg[:, None, :]
    count = allowed.sum(-1)  # B,K
    mx = rel.masked_fill(~fg[:, None, :], -torch.inf).amax(-1)
    return count, mx


def write_csv(path, rows, columns=None):
    if columns is None:
        if not rows:
            raise ValueError(f"No rows to write: {path}")
        columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_iou(path, names, selected_epochs):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior lambda comparison CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {"epoch", "class_id", "part", "valid_val_gt", "lam0_iou", "lam4_iou"}
        if not required.issubset(cols):
            raise ValueError(f"IoU CSV missing columns: {required-cols}; found {cols}")
        out = {}
        for r in reader:
            ep = int(r["epoch"])
            if ep not in selected_epochs:
                continue
            pid = int(r["class_id"])
            if not 0 <= pid < NUM_PARTS or names[pid] != r["part"]:
                raise ValueError(f"Class ID/name mismatch in IoU CSV: {r}")
            if (ep, pid) in out:
                raise ValueError(f"Duplicate IoU row: E{ep}, class {pid}")
            good = int(r["valid_val_gt"]) == 1
            values = {}
            for arm in ("lam0", "lam4"):
                v = float(r[f"{arm}_iou"])
                if good and not (math.isfinite(v) and 0.0 <= v <= 100.01):
                    raise ValueError(f"Invalid per-part IoU E{ep} {names[pid]} {arm}: {v}")
                values[arm] = v if good else math.nan
            out[ep, pid] = (good, values)
    missing = [(ep, pid) for ep in selected_epochs for pid in range(NUM_PARTS) if (ep, pid) not in out]
    if missing:
        raise ValueError(f"Missing {len(missing)} epoch/part rows in IoU CSV: {missing[:5]}")
    return out


def read_global_miou(path):
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (str(r["arm"]), int(r["epoch"]))
            if key in out:
                raise ValueError(f"Duplicate global mIoU row: {key}")
            out[key] = float(r["official_mIoU_pct"])
    return out


def calc_spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    xx = x[good]; yy = y[good]
    if len(xx) < 4 or np.ptp(xx) < 1e-12 or np.ptp(yy) < 1e-12:
        return math.nan, int(len(xx))
    return float(spearmanr(xx, yy).statistic), int(len(xx))


def percentile_interval(x, y, *, reps=300, seed=123):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]; y = y[good]
    if len(x) < 8 or reps <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        picks = rng.integers(0, len(x), size=len(x))
        rho, _ = calc_spearman(x[picks], y[picks])
        if math.isfinite(rho):
            estimates.append(rho)
    if len(estimates) < reps / 2:
        return math.nan, math.nan
    return tuple(float(v) for v in np.percentile(estimates, [2.5, 97.5]))


def self_test():
    # A deterministic example: each patch has a unique winner, while one
    # part never wins => this part must fallback even though it gets an argmax.
    N, D = PATCH_COUNT, VISION_DIM
    fg = torch.zeros((1, N), dtype=torch.bool)
    fg[0, :4] = True
    t = torch.zeros((1, 3, D))
    t[0, 0, 0] = 1.0
    t[0, 1, 1] = 1.0
    t[0, 2, 2] = 1.0
    patches = torch.zeros((1, N, D))
    patches[0, 0, 0] = 1.0
    patches[0, 1, 0] = 1.0
    patches[0, 2, 1] = 1.0
    patches[0, 3, 0] = 1.0
    def local_relative(absolute, foreground):
        # Only test the counterfactual margin and mask here; the REAL audit
        # calls the original repository's relative_scores function.
        b, k, n = absolute.shape
        other = absolute[:, None, :, :].expand(b, k, k, n)
        eye = torch.eye(k, dtype=torch.bool)[None, :, :, None]
        strongest = other.masked_fill(eye, -torch.inf).max(2).values
        return (absolute - strongest).masked_fill(~foreground[:, None, :], -torch.inf)
    cnt, mx = positive_count_and_max(t, patches, fg, local_relative)
    assert cnt.tolist() == [[3, 1, 0]], cnt
    assert bool((cnt == 0)[0, 2])
    assert float(mx[0, 2]) <= 0.0
    assert bool((cnt != 0)[0, 0])
    assert calc_spearman([0, 1, 2, 3], [3, 2, 1, 0])[0] == -1.0
    print("SELF_TEST_PASS: K>=2 fallback, fg mask, support count, correlation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth"))
    ap.add_argument("--text-bank", type=Path, default=Path("feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"))
    ap.add_argument("--projector", type=Path, default=Path("weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"))
    ap.add_argument("--model-config", type=Path, default=Path("configs/vitb_mlp_infonce.yaml"))
    ap.add_argument("--classes-file", type=Path, default=Path("src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py"))
    ap.add_argument("--run0", type=Path, default=Path("final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep"))
    ap.add_argument("--run4", type=Path, default=Path("final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep"))
    ap.add_argument("--iou-csv", type=Path, default=Path("output/pp116_DIAGREL_comparison/per_part_epoch_lambda0_vs_lambda4.csv"))
    ap.add_argument("--global-miou-csv", type=Path, default=Path("output/pp116_DIAGREL_comparison/epoch_summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/pp116_fallback_audit"))
    ap.add_argument("--epochs", nargs="+", type=int, default=list(range(0, 101, 10)))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-crops", type=int, default=0, help="0 means ALL crop records; positive is first N, only for smoke test")
    ap.add_argument("--min-support", type=int, default=10, help="Minimum K>=2 part occurrences for a part to enter correlations")
    ap.add_argument("--bootstrap", type=int, default=300, help="Number of resamples for per-part Spearman CI")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.batch_size < 1 or a.min_support < 1 or a.bootstrap < 0:
        raise ValueError("Invalid batch-size/min-support/bootstrap")
    epochs = sorted(set(a.epochs))
    if not epochs or any(ep < 0 or ep > 100 or ep % 10 for ep in epochs):
        raise ValueError("Use epochs from E000, E010, ..., E100")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError(f"Output already exists, refusing to mix results: {a.out_dir}")
    for p in [a.cache, a.text_bank, a.projector, a.model_config, a.classes_file, a.iou_csv]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")

    # Source functions ensure calculations are the SAME as current training.
    import train_relproto_alignemt as original
    import train_pp116_softsingle_dual as trainer
    names = official_names(a.classes_file)
    raw, raw_names, _ = original.load_raw_clip_bank(
        a.text_bank, expected_clip_model="ViT-B/16", expected_template="sub_imagenet_template"
    )
    if list(raw_names) != names:
        raise ValueError("Text bank and official taxonomy ID ORDER differ")
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    projector, _ = original.load_frozen_projector(
        project_root=Path.cwd(), config_path=a.model_config, weight_path=a.projector, device=device
    )
    base = original.project_raw_clip_bank(raw, projector, device=device, batch_size=128)
    base = original.normalize_last(base.detach().to(dtype=torch.float32, device=device))
    del projector, raw
    if base.shape != (NUM_PARTS, VISION_DIM):
        raise ValueError(f"Projected CLIP feature shape mismatch: {base.shape}")

    arms = {"lam0": (a.run0, 0.0), "lam4": (a.run4, 4.0)}
    text = {}
    with torch.no_grad():
        for arm, (directory, lamb) in arms.items():
            for ep in epochs:
                W = state_W(directory, ep, lamb).to(device)
                text[arm, ep] = original.normalize_last(base @ W).detach()
                del W
    if 0 in epochs:
        if not torch.allclose(text["lam0", 0], text["lam4", 0], atol=0.0, rtol=0.0):
            raise AssertionError("E0 text mismatch between two arms")

    ious = read_iou(a.iou_csv, names, epochs)
    global_miou = read_global_miou(a.global_miou_csv)
    cache = load_trusted_pt(a.cache)
    annotations = annotations_from_cache(cache)
    total_cache = len(annotations)
    if a.max_crops > 0:
        annotations = annotations[:a.max_crops]
    if not annotations:
        raise RuntimeError("Empty annotation cache")
    # Group crop records by K for memory-efficient batched scoring.
    by_k = defaultdict(list)
    part_multi = np.zeros(NUM_PARTS, dtype=np.int64)
    part_single = np.zeros(NUM_PARTS, dtype=np.int64)
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise TypeError(f"Cache item {i} is not a dict")
        ids = ids_of(ann)
        by_k[len(ids)].append(i)
        for p in ids:
            if len(ids) == 1:
                part_single[p] += 1
            else:
                part_multi[p] += 1
    print(f"[CACHE] records={len(annotations)}/{total_cache} "
          f"K=1 crops={len(by_k.get(1, []))} multi crops={len(annotations)-len(by_k.get(1, []))} "
          f"multi part occurrences={int(part_multi.sum())}", flush=True)
    if not np.any(part_multi):
        raise ValueError("There are no K>=2 crops; cannot measure relative fallback")

    # Each key contains a 116-dimensional sum over multi-part occurrences.
    fallback = {key: np.zeros(NUM_PARTS, dtype=np.int64) for key in text}
    positive_sum = {key: np.zeros(NUM_PARTS, dtype=np.float64) for key in text}
    margin_sum = {key: np.zeros(NUM_PARTS, dtype=np.float64) for key in text}
    fg_sum = np.zeros(NUM_PARTS, dtype=np.float64)
    checked_source = False
    processed_crops = 0

    with torch.no_grad():
        for K, indices in sorted(by_k.items()):
            if K == 1:
                continue  # K=1 uses absolute SOFTMAX; it CANNOT trigger fallback.
            for start in range(0, len(indices), a.batch_size):
                selected = indices[start:start + a.batch_size]
                ex = [annotations[i] for i in selected]
                pids = torch.tensor([ids_of(q) for q in ex], device=device, dtype=torch.long)
                tok = torch.stack([
                    torch.as_tensor(q["cropaug_patch_tokens"], dtype=torch.float32)
                    for q in ex
                ], dim=0).to(device=device)
                fg = torch.stack([
                    torch.as_tensor(q["pred_obj_mask_patch"], dtype=torch.bool).flatten()
                    for q in ex
                ], dim=0).to(device=device)
                if tok.shape != (len(ex), PATCH_COUNT, VISION_DIM):
                    raise ValueError(f"Unexpected token shape: {tok.shape}")
                if fg.shape != (len(ex), PATCH_COUNT) or not bool(fg.any(-1).all()):
                    raise ValueError(f"Unexpected or empty foreground mask: {fg.shape}")
                tok = original.normalize_last(tok)
                pids_np = pids.cpu().numpy().reshape(-1)
                fg_np = fg.sum(-1).cpu().numpy()
                np.add.at(fg_sum, pids_np, np.broadcast_to(fg_np[:, None], (len(ex), K)).reshape(-1))
                for (arm, ep), t in text.items():
                    cnt, max_margin = positive_count_and_max(
                        t[pids], tok, fg, original.relative_scores
                    )
                    if not checked_source:
                        proto, info = trainer.soft_relative_prototypes(
                            t[pids][:1], tok[:1], fg[:1], temperature=0.05
                        )
                        expected_fallback = info["fallback"].to(dtype=torch.bool).view(-1)
                        actual_fallback = (cnt[:1] == 0).view(-1)
                        if not torch.equal(expected_fallback, actual_fallback):
                            raise RuntimeError("Fallback definition differs from trainer.soft_relative_prototypes!")
                        if not torch.equal(info["positive_count"].view(-1).to(torch.long), cnt[:1].view(-1)):
                            raise RuntimeError("Positive patch count differs from trainer!")
                        if not torch.isfinite(proto).all():
                            raise RuntimeError("Source trainer produced invalid prototype")
                        print("[CHECK] Fallback and positive-count EXACTLY match source training", flush=True)
                        checked_source = True
                    c_np = cnt.cpu().numpy().reshape(-1)
                    margin_np = max_margin.cpu().numpy().reshape(-1)
                    np.add.at(fallback[arm, ep], pids_np, (c_np == 0).astype(np.int64))
                    np.add.at(positive_sum[arm, ep], pids_np, c_np)
                    np.add.at(margin_sum[arm, ep], pids_np, margin_np)
                processed_crops += len(ex)
            print(f"[PROGRESS] K={K}: {len(indices)} crops complete", flush=True)

    # Check that all row denominators are preserved exactly.
    for key, arr in fallback.items():
        if np.any(arr > part_multi) or np.any(arr < 0):
            raise AssertionError(f"Invalid fallback count {key}")
    out_rows = []
    for arm, ep in text:
        arr = fallback[arm, ep]
        for pid, name in enumerate(names):
            denom = int(part_multi[pid]); singles = int(part_single[pid])
            valid, iv = ious[ep, pid]
            out_rows.append({
                "arm": arm, "lambda_struct": 0 if arm == "lam0" else 4,
                "epoch": ep, "class_id": pid, "part": name,
                "object": name.split("'s ", 1)[0],
                "candidate_occurrences_all": denom + singles,
                "eligible_multi_occurrences": denom,
                "single_part_occurrences_NO_FALLBACK": singles,
                "fallback_count": int(arr[pid]),
                "fallback_rate_multi": (arr[pid] / denom) if denom else math.nan,
                "mean_positive_patch_count_multi": (positive_sum[arm, ep][pid] / denom) if denom else math.nan,
                "mean_best_relative_margin_multi": (margin_sum[arm, ep][pid] / denom) if denom else math.nan,
                "mean_pred_fg_patches_multi": (fg_sum[pid] / denom) if denom else math.nan,
                "valid_val_gt": int(valid),
                "val_part_iou_pct": iv[arm],
            })

    # Part-level Spearman at each (arm, epoch), and within-arm changes E0->E.
    correlations = []
    by_arm_epoch = {(arm, ep): [r for r in out_rows if r["arm"] == arm and r["epoch"] == ep]
                    for arm in arms for ep in epochs}
    rows_by_key = {(r["arm"], r["epoch"], r["class_id"]): r for r in out_rows}
    for arm in arms:
        for ep in epochs:
            rows = by_arm_epoch[arm, ep]
            keep = [r for r in rows if r["valid_val_gt"] and r["eligible_multi_occurrences"] >= a.min_support]
            x = np.array([r["fallback_rate_multi"] for r in keep], dtype=float)
            y = np.array([r["val_part_iou_pct"] for r in keep], dtype=float)
            rho, count = calc_spearman(x, y)
            low, high = percentile_interval(x, y, reps=a.bootstrap)
            correlations.append({
                "arm": arm, "epoch": ep, "metric": "fallback_rate_vs_same_epoch_part_IoU",
                "n_parts": count, "spearman_rho": rho, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "mean_fallback_rate_unweighted": float(np.mean(x)) if len(x) else math.nan,
                "global_fallback_rate_occurrence_weighted": float(fallback[arm, ep].sum() / part_multi.sum()),
                "official_mIoU_pct": global_miou.get((arm, ep), math.nan),
            })
            if ep and 0 in epochs:
                keep = [r for r in keep if (arm, 0, r["class_id"]) in rows_by_key]
                delta_x = np.asarray([r["fallback_rate_multi"] - rows_by_key[arm, 0, r["class_id"]]["fallback_rate_multi"] for r in keep])
                delta_y = np.asarray([r["val_part_iou_pct"] - rows_by_key[arm, 0, r["class_id"]]["val_part_iou_pct"] for r in keep])
                drho, dn = calc_spearman(delta_x, delta_y)
                dl, dh = percentile_interval(delta_x, delta_y, reps=a.bootstrap)
                correlations.append({
                    "arm": arm, "epoch": ep, "metric": "delta_fallback_E0_to_E_vs_delta_IoU_E0_to_E",
                    "n_parts": dn, "spearman_rho": drho, "bootstrap_ci_low": dl,
                    "bootstrap_ci_high": dh,
                    "mean_fallback_rate_unweighted": math.nan,
                    "global_fallback_rate_occurrence_weighted": float(fallback[arm, ep].sum() / part_multi.sum()),
                    "official_mIoU_pct": global_miou.get((arm, ep), math.nan),
                })

    # Does the *between-arm* change in fallback track the between-arm IoU change?
    paired = []
    paired_correlations = []
    for ep in epochs:
        for pid, name in enumerate(names):
            r0 = rows_by_key["lam0", ep, pid]
            r4 = rows_by_key["lam4", ep, pid]
            v0 = r0["val_part_iou_pct"]
            v4 = r4["val_part_iou_pct"]
            paired.append({
                "epoch": ep, "class_id": pid, "part": name,
                "eligible_multi_occurrences": r0["eligible_multi_occurrences"],
                "valid_val_gt": r0["valid_val_gt"],
                "lam0_fallback_rate": r0["fallback_rate_multi"],
                "lam4_fallback_rate": r4["fallback_rate_multi"],
                "delta_fallback_lam4_minus_lam0": r4["fallback_rate_multi"] - r0["fallback_rate_multi"],
                "lam0_iou": v0, "lam4_iou": v4,
                "delta_iou_lam4_minus_lam0": v4-v0,
            })
        keep = [r for r in paired if r["epoch"] == ep and r["valid_val_gt"] and r["eligible_multi_occurrences"] >= a.min_support]
        xx = [r["delta_fallback_lam4_minus_lam0"] for r in keep]
        yy = [r["delta_iou_lam4_minus_lam0"] for r in keep]
        rho, n = calc_spearman(xx, yy)
        low, high = percentile_interval(xx, yy, reps=a.bootstrap)
        paired_correlations.append({"epoch": ep, "n_parts": n, "rho_delta_fallback_vs_delta_IoU": rho,
                                    "bootstrap_ci_low": low, "bootstrap_ci_high": high})

    epoch_global = []
    for arm in arms:
        for ep in epochs:
            epoch_global.append({
                "arm": arm, "epoch": ep,
                "fallback_count": int(fallback[arm, ep].sum()),
                "eligible_multi_part_occurrences": int(part_multi.sum()),
                "global_fallback_rate": float(fallback[arm, ep].sum() / part_multi.sum()),
                "global_mIoU_pct": global_miou.get((arm, ep), math.nan),
                "multi_crop_count": processed_crops,
                "single_crop_count": int(len(by_k.get(1, []))),
            })

    # These time-series correlations are descriptive: 11 correlated Epoch points
    # cannot be treated as independent samples or causal evidence.
    global_curve_correlations = []
    for arm in arms:
        series = [r for r in epoch_global if r["arm"] == arm and math.isfinite(r["global_mIoU_pct"])]
        rho, count = calc_spearman(
            [r["global_fallback_rate"] for r in series],
            [r["global_mIoU_pct"] for r in series],
        )
        global_curve_correlations.append({
            "arm": arm, "n_epochs": count,
            "spearman_rho_global_fallback_vs_mIoU": rho,
            "warning": "Descriptive only: epoch autocorrelation/trend confounding",
        })

    e100_ranking = sorted(
        (r for r in paired if r["epoch"] == 100),
        key=lambda r: (-float(r["lam4_fallback_rate"]) if math.isfinite(float(r["lam4_fallback_rate"])) else math.inf,
                       -r["eligible_multi_occurrences"], r["part"])
    )
    a.out_dir.mkdir(parents=True)
    if 100 in epochs:
        write_csv(a.out_dir / "e100_parts_ranked_by_lambda4_fallback.csv", e100_ranking)
    write_csv(a.out_dir / "global_curve_spearman_DESCRIPTIVE.csv", global_curve_correlations)
    write_csv(a.out_dir / "per_part_epoch_fallback.csv", out_rows)
    write_csv(a.out_dir / "fallback_vs_IoU_spearman.csv", correlations)
    write_csv(a.out_dir / "lambda_paired_effect.csv", paired)
    write_csv(a.out_dir / "lambda_paired_spearman.csv", paired_correlations)
    write_csv(a.out_dir / "global_fallback_and_mIoU.csv", epoch_global)
    report = {
        "description": "Source-code-matched Soft Relative fallback audit, NO GT part masks",
        "formula": "fallback iff K>=2 and zero foreground patches with relative_score>0; K=1 ABS pooling excluded",
        "candidate_source": "image-level part_category_id from original predicted-object crop cache",
        "text": "Raw ViT-B/16 CLIP prompts -> frozen COCO projector -> T0 -> T0@W, normalize",
        "relative_scores_implementation": "import train_relproto_alignemt.relative_scores; compare first positive_count/fallback to train_pp116_softsingle_dual.soft_relative_prototypes",
        "epochs": epochs, "lambda_arms": list(arms), "cache": str(a.cache),
        "cache_records_total": total_cache, "cache_records_used": len(annotations),
        "multi_crop_records_used": processed_crops,
        "multi_part_occurrences_used": int(part_multi.sum()),
        "excluded_K1_crop_records": len(by_k.get(1, [])),
        "min_multi_part_occurrences_for_correlation": a.min_support,
        "per_part_IoU_source": str(a.iou_csv),
        "IoU": "Existing 850-validation-image part IoU, NOT mIoU; 2 val GT-invalid classes excluded",
        "statistics": "Between-part Spearman, same epoch; E0-to-E delta; paired lambda4-lambda0; iid-part bootstrap CI (descriptive only)",
        "limits": "Cross-sectional correlation is not causal; no GT crop alignment used; denominator is image-level candidate-part occurrences, not GT spatial presence",
    }
    (a.out_dir / "protocol.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[COMPLETE]", a.out_dir)
    print("[SUMMARY] Important per-class fallback frequencies at E100:")
    focus = ("person's upper arm", "cow's horn", "sheep's horn", "bicycle's saddle", "cat's leg", "train's front", "aeroplane's body", "dog's tail")
    for name in focus:
        if name not in names or 100 not in epochs:
            continue
        pid = names.index(name)
        if 100 in epochs:
            r0 = rows_by_key["lam0", 100, pid]
            r4 = rows_by_key["lam4", 100, pid]
            print(f"  {name:24s} n={r4['eligible_multi_occurrences']:5d} "
                  f"lam0={r0['fallback_rate_multi']:.3f} lam4={r4['fallback_rate_multi']:.3f} "
                  f"IoU0={r0['val_part_iou_pct']:.2f} IoU4={r4['val_part_iou_pct']:.2f}")
    for r in correlations:
        if r["epoch"] == 100 and r["metric"] == "fallback_rate_vs_same_epoch_part_IoU":
            print(f"[SPEARMAN E100 {r['arm']}] rho={r['spearman_rho']:+.4f} "
                  f"95%CI=[{r['bootstrap_ci_low']:+.3f},{r['bootstrap_ci_high']:+.3f}], n={r['n_parts']}")
    for r in paired_correlations:
        if r["epoch"] == 100:
            print(f"[PAIRED E100] rho(delta fallback, delta IoU)="
                  f"{r['rho_delta_fallback_vs_delta_IoU']:+.4f} n={r['n_parts']}")


if __name__ == "__main__":
    main()
