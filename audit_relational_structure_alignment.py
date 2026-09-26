#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PP-116 relational structure alignment audit.

Controlled comparison:
    Relative + Linear, L=8
    Relative + Orthogonal, L=8

For every epoch and every valid image/object instance, compute:

1) Text-Prototype Structure Correlation
   Spearman correlation between:
       pairwise cosine relations of the INITIAL projected part-text
       representations t_j
   and
       pairwise cosine relations of the image-specific relative
       visual prototypes induced at the current epoch.

2) Prototype-GT Structure Correlation
   Spearman correlation between:
       pairwise cosine relations of the image-specific relative
       visual prototypes
   and
       pairwise cosine relations of crop-matched oracle GT
       visual prototypes.

IMPORTANT:
- The text reference is always the INITIAL projected text feature T0.
- T0 @ W is used only to discover the current relative visual evidence,
  exactly following the training/evaluation implementation.
- Both structure metrics use exactly the same valid part subset.
- At least 3 valid parts are required for an object instance.
- GT masks are used only for this post-hoc audit.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import audit_structure_preservation_alignment as base


MIN_VALID_PARTS = 3


# ============================================================================
# Spearman helpers
# ============================================================================

def rankdata_average(x):
    """
    Tie-aware average ranks, equivalent to scipy.stats.rankdata(method="average").
    Implemented locally so this audit does not add a scipy dependency.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)

    i = 0
    while i < x.size:
        j = i + 1

        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1

        # ranks are 1-based; tied entries receive their average rank.
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank

        i = j

    return ranks


def spearman_1d(a, b):
    """
    Spearman correlation between two 1-D arrays.

    Returns NaN when correlation is undefined, e.g. one relation vector
    is constant.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"Spearman shape mismatch: {a.shape} vs {b.shape}"
        )

    if a.size < 2:
        return float("nan")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")

    ra = rankdata_average(a)
    rb = rankdata_average(b)

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denom = np.sqrt(
        np.dot(ra, ra) * np.dot(rb, rb)
    )

    if denom <= 1e-12:
        return float("nan")

    return float(
        np.dot(ra, rb) / denom
    )


def pairwise_upper_cosine(x):
    """
    Return the upper-triangular pairwise cosine relations.

    x: [K,D]
    output: [K*(K-1)/2]
    """
    if x.ndim != 2:
        raise ValueError(
            f"Expected [K,D], got {tuple(x.shape)}"
        )

    k = int(x.shape[0])

    if k < 2:
        return np.empty(0, dtype=np.float64)

    x = F.normalize(
        x.float(),
        dim=-1,
    )

    sim = x @ x.T

    idx = torch.triu_indices(
        k,
        k,
        offset=1,
        device=sim.device,
    )

    values = sim[
        idx[0],
        idx[1],
    ]

    return (
        values
        .detach()
        .cpu()
        .double()
        .numpy()
    )


# ============================================================================
# Object grouping
# ============================================================================

def object_name_from_part_ids(part_ids, class_names):
    """
    PP-116 names follow forms such as:
        aeroplane's wing
        cat's leg
        person's head

    Use the object prefix only for object-category macro aggregation.
    """
    objects = []

    for pid in part_ids:
        name = str(class_names[int(pid)])

        if "'s " in name:
            obj = name.split("'s ", 1)[0]
        elif "'s" in name:
            obj = name.split("'s", 1)[0]
        else:
            obj = name

        objects.append(obj)

    unique = sorted(set(objects))

    if len(unique) == 1:
        return unique[0]

    # This should not normally happen because one cache record
    # corresponds to one object.
    return "|".join(unique)


# ============================================================================
# Aggregation
# ============================================================================

def object_macro(values_by_object):
    """
    instance -> mean inside object category -> mean across object categories.
    """
    means = {
        obj: float(np.mean(values))
        for obj, values in values_by_object.items()
        if values
    }

    if not means:
        return float("nan"), {}

    return float(np.mean(list(means.values()))), means


# ============================================================================
# Main relational-structure evaluation
# ============================================================================

@torch.no_grad()
def evaluate_structure(
    records,
    gt_audit,
    base_text,
    W,
    args,
    class_names,
    method_name,
):
    device = torch.device(args.device)

    # IMPORTANT:
    # This is the INITIAL projected part-text representation.
    # It is the fixed structural reference in Text-Prototype Spearman.
    base_text = base_text.to(
        device=device,
        dtype=torch.float32,
    )

    W = W.to(
        device=device,
        dtype=torch.float32,
    )

    text_proto_instances = []
    proto_gt_instances = []

    text_proto_by_object = defaultdict(list)
    proto_gt_by_object = defaultdict(list)

    per_instance_rows = []

    skipped_lt3_parts = 0
    skipped_undefined_spearman = 0
    skipped_no_gt_parts = 0

    for record, gt_info in zip(
        records,
        gt_audit,
    ):
        # --------------------------------------------------------------
        # Exact crop visual features
        # --------------------------------------------------------------
        tokens_full = F.normalize(
            record["_tokens"].to(
                device=device,
                dtype=torch.float32,
            ),
            dim=-1,
        )

        foreground = record[
            "_foreground"
        ].to(
            device=device,
            dtype=torch.bool,
        )

        part_ids = record["_part_ids"]

        gt_ids = (
            torch.as_tensor(
                gt_info["part_ids"]
            )
            .long()
            .tolist()
        )

        if part_ids != gt_ids:
            raise RuntimeError(
                "part ID mismatch between cache record "
                "and GT audit cache"
            )

        # --------------------------------------------------------------
        # Exact nightly relative selector semantics
        # --------------------------------------------------------------
        X = tokens_full[foreground]

        ids = torch.tensor(
            part_ids,
            device=device,
            dtype=torch.long,
        )

        # IMPORTANT:
        # Adapted text is used ONLY to discover relative visual evidence.
        T_adapted = F.normalize(
            base_text.index_select(0, ids) @ W,
            dim=-1,
        )

        S = T_adapted @ X.T

        score = base.relative_scores(S)

        # --------------------------------------------------------------
        # GT patch labels
        # --------------------------------------------------------------
        patch_labels = (
            torch.as_tensor(
                gt_info["patch_labels"]
            )
            .long()
            .to(device)
        )

        if patch_labels.numel() != tokens_full.shape[0]:
            raise RuntimeError(
                "GT patch-label length mismatch"
            )

        # --------------------------------------------------------------
        # Collect induced / GT prototypes for the SAME valid parts
        # --------------------------------------------------------------
        valid_part_ids = []
        induced_prototypes = []
        gt_prototypes = []

        for j, pid in enumerate(part_ids):
            # Relative visual prototype induced exactly as training.
            induced, _ = base.build_proto(
                X,
                score[j],
                args.L,
                "relative",
            )

            label = (
                int(pid)
                + int(args.gt_label_offset)
            )

            gt_part_mask = (
                patch_labels == label
            )

            if not bool(
                gt_part_mask.any()
            ):
                skipped_no_gt_parts += 1
                continue

            oracle = F.normalize(
                tokens_full[
                    gt_part_mask
                ].mean(
                    dim=0,
                    keepdim=True,
                ),
                dim=-1,
            )[0]

            valid_part_ids.append(
                int(pid)
            )

            induced_prototypes.append(
                induced
            )

            gt_prototypes.append(
                oracle
            )

        # --------------------------------------------------------------
        # Need >=3 parts to obtain a meaningful relational ranking
        # --------------------------------------------------------------
        if len(valid_part_ids) < MIN_VALID_PARTS:
            skipped_lt3_parts += 1
            continue

        valid_ids_tensor = torch.tensor(
            valid_part_ids,
            device=device,
            dtype=torch.long,
        )

        # Fixed INITIAL projected text features.
        T0 = F.normalize(
            base_text.index_select(
                0,
                valid_ids_tensor,
            ),
            dim=-1,
        )

        R = F.normalize(
            torch.stack(
                induced_prototypes,
                dim=0,
            ),
            dim=-1,
        )

        G = F.normalize(
            torch.stack(
                gt_prototypes,
                dim=0,
            ),
            dim=-1,
        )

        # --------------------------------------------------------------
        # Pairwise relation vectors
        # --------------------------------------------------------------
        text_rel = pairwise_upper_cosine(T0)

        proto_rel = pairwise_upper_cosine(R)

        gt_rel = pairwise_upper_cosine(G)

        if not (
            len(text_rel)
            == len(proto_rel)
            == len(gt_rel)
        ):
            raise AssertionError(
                "relation-vector length mismatch"
            )

        # --------------------------------------------------------------
        # Structure Spearman
        # --------------------------------------------------------------
        rho_text_proto = spearman_1d(
            text_rel,
            proto_rel,
        )

        rho_proto_gt = spearman_1d(
            proto_rel,
            gt_rel,
        )

        # Use exactly the same valid object instances for both metrics.
        if not (
            np.isfinite(rho_text_proto)
            and np.isfinite(rho_proto_gt)
        ):
            skipped_undefined_spearman += 1
            continue

        object_name = object_name_from_part_ids(
            valid_part_ids,
            class_names,
        )

        text_proto_instances.append(
            rho_text_proto
        )

        proto_gt_instances.append(
            rho_proto_gt
        )

        text_proto_by_object[
            object_name
        ].append(
            rho_text_proto
        )

        proto_gt_by_object[
            object_name
        ].append(
            rho_proto_gt
        )

        image_ref = gt_info.get(
            "image_ref",
            record.get(
                "image_id",
                record.get(
                    "_annotation_index",
                    "",
                ),
            ),
        )

        per_instance_rows.append(
            {
                "method": method_name,
                "annotation_index":
                    record.get(
                        "_annotation_index"
                    ),
                "image_ref":
                    image_ref,
                "object":
                    object_name,
                "num_valid_parts":
                    len(valid_part_ids),
                "num_pairs":
                    len(text_rel),
                "text_proto_spearman":
                    rho_text_proto,
                "proto_gt_spearman":
                    rho_proto_gt,
            }
        )

    if not text_proto_instances:
        raise RuntimeError(
            "No valid relational-structure instances"
        )

    # ------------------------------------------------------------------
    # Instance macro
    # ------------------------------------------------------------------
    text_proto_instance_macro = float(
        np.mean(
            text_proto_instances
        )
    )

    proto_gt_instance_macro = float(
        np.mean(
            proto_gt_instances
        )
    )

    # ------------------------------------------------------------------
    # Object-category macro
    # ------------------------------------------------------------------
    (
        text_proto_object_macro,
        text_proto_object_values,
    ) = object_macro(
        text_proto_by_object
    )

    (
        proto_gt_object_macro,
        proto_gt_object_values,
    ) = object_macro(
        proto_gt_by_object
    )

    # Both should be based on the same set of object groups.
    text_objects = set(
        text_proto_object_values
    )

    gt_objects = set(
        proto_gt_object_values
    )

    if text_objects != gt_objects:
        raise AssertionError(
            "Text-Proto and Proto-GT object-group sets differ"
        )

    return {
        # --------------------------------------------------------------
        # Main paper metrics
        # --------------------------------------------------------------
        "text_proto_spearman":
            text_proto_object_macro,

        "proto_gt_spearman":
            proto_gt_object_macro,

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------
        "text_proto_spearman_instance_macro":
            text_proto_instance_macro,

        "proto_gt_spearman_instance_macro":
            proto_gt_instance_macro,

        "valid_structure_instances":
            len(text_proto_instances),

        "valid_object_groups":
            len(text_objects),

        "skipped_lt3_parts":
            skipped_lt3_parts,

        "skipped_undefined_spearman":
            skipped_undefined_spearman,

        "skipped_no_gt_parts":
            skipped_no_gt_parts,

        "per_instance":
            per_instance_rows,

        "per_object_text_proto":
            text_proto_object_values,

        "per_object_proto_gt":
            proto_gt_object_values,
    }


# ============================================================================
# Plot
# ============================================================================

def plot_results(rows, out_dir):
    epochs = [
        r["epoch"]
        for r in rows
    ]

    linear_text_proto = [
        r["linear_text_proto_spearman"]
        for r in rows
    ]

    ortho_text_proto = [
        r["orthogonal_text_proto_spearman"]
        for r in rows
    ]

    linear_proto_gt = [
        r["linear_proto_gt_spearman"]
        for r in rows
    ]

    ortho_proto_gt = [
        r["orthogonal_proto_gt_spearman"]
        for r in rows
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.3),
    )

    # ------------------------------------------------------------------
    # (a) Text-Prototype
    # ------------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        epochs,
        linear_text_proto,
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Relative + Linear",
    )

    ax.plot(
        epochs,
        ortho_text_proto,
        linewidth=2.4,
        marker="s",
        markersize=4,
        markevery=2,
        label="Relative + Orthogonal",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=13,
    )

    ax.set_ylabel(
        r"Spearman $\rho$",
        fontsize=13,
    )

    ax.set_title(
        "(a) Text--Prototype Structure Correlation",
        fontsize=12,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs),
    )

    ax.set_xticks(
        range(0, 31, 5)
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # (b) Prototype-GT
    # ------------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        epochs,
        linear_proto_gt,
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Relative + Linear",
    )

    ax.plot(
        epochs,
        ortho_proto_gt,
        linewidth=2.4,
        marker="s",
        markersize=4,
        markevery=2,
        label="Relative + Orthogonal",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=13,
    )

    ax.set_ylabel(
        r"Spearman $\rho$",
        fontsize=13,
    )

    ax.set_title(
        "(b) Prototype--GT Structure Correlation",
        fontsize=12,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs),
    )

    ax.set_xticks(
        range(0, 31, 5)
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    fig.savefig(
        out_dir
        / "figure_relational_structure_alignment.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir
        / "figure_relational_structure_alignment.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    # Reuse the exact CLI of the verified audit script.
    args = base.parse_args()

    if args.L < 1:
        raise SystemExit(
            "--L must be >= 1"
        )

    if not (
        0
        <= args.start_epoch
        <= args.end_epoch
    ):
        raise SystemExit(
            "require 0 <= start_epoch <= end_epoch"
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    root = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

    out_dir = base.resolve(
        root,
        args.out_dir,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = base.resolve(
        root,
        args.cache,
    )

    linear_dir = base.resolve(
        root,
        args.linear_weight_dir,
    )

    ortho_dir = base.resolve(
        root,
        args.orthogonal_weight_dir,
    )

    projector_path = base.resolve(
        root,
        args.base_projector,
    )

    print(
        f"[INFO] loading cache: {cache_path}"
    )

    cache = base.tload(
        cache_path
    )

    if not isinstance(
        cache,
        dict,
    ):
        raise RuntimeError(
            "pred-object cache must be a dict"
        )

    meta = cache.get(
        "pred_obj_cropaug_meta",
        {},
    )

    records = base.prepare_records(
        cache
    )

    # ------------------------------------------------------------------
    # Initial projected part-text representations
    # ------------------------------------------------------------------
    base_text, loaded_projector_path = (
        base.load_base_text(
            args,
            root,
        )
    )

    if (
        loaded_projector_path
        != projector_path
    ):
        raise AssertionError(
            "internal projector path mismatch"
        )

    base.validate_cache_meta(
        meta,
        args,
        projector_path,
    )

    # ------------------------------------------------------------------
    # Same selector / same L / only mapping differs
    # ------------------------------------------------------------------
    base.validate_weight_dir(
        linear_dir,
        "relative",
        "linear",
        args.L,
        args.start_epoch,
        args.end_epoch,
    )

    base.validate_weight_dir(
        ortho_dir,
        "relative",
        "orthogonal",
        args.L,
        args.start_epoch,
        args.end_epoch,
    )

    # ------------------------------------------------------------------
    # Reuse the exact GT cache from previous controlled audit
    # ------------------------------------------------------------------
    gt_audit = base.load_gt_audit(
        records,
        args,
        root,
    )

    class_names = base.load_class_names(
        root,
        args.classes_source,
    )

    rows = []
    per_instance_rows = []

    for epoch in range(
        args.start_epoch,
        args.end_epoch + 1,
    ):
        print(
            "\n"
            + "=" * 80
        )

        print(
            f"EPOCH {epoch:03d}"
        )

        print(
            "=" * 80
        )

        W_linear = base.load_W(
            linear_dir,
            epoch,
            "relative",
            "linear",
            args.L,
        )

        W_ortho = base.load_W(
            ortho_dir,
            epoch,
            "relative",
            "orthogonal",
            args.L,
        )

        linear_metrics = evaluate_structure(
            records,
            gt_audit,
            base_text,
            W_linear,
            args,
            class_names,
            "linear",
        )

        ortho_metrics = evaluate_structure(
            records,
            gt_audit,
            base_text,
            W_ortho,
            args,
            class_names,
            "orthogonal",
        )

        # --------------------------------------------------------------
        # Epoch-0 sanity check:
        # both branches use identity W and therefore MUST be identical.
        # --------------------------------------------------------------
        if epoch == 0:
            for key in (
                "text_proto_spearman",
                "proto_gt_spearman",
                "text_proto_spearman_instance_macro",
                "proto_gt_spearman_instance_macro",
            ):
                a = float(
                    linear_metrics[key]
                )

                b = float(
                    ortho_metrics[key]
                )

                if abs(a - b) > 1e-10:
                    raise RuntimeError(
                        "Epoch-0 sanity check failed for "
                        f"{key}: linear={a}, orthogonal={b}"
                    )

            print(
                "[OK] epoch-0 Linear/Orthogonal "
                "relational metrics are identical"
            )

        row = {
            "epoch":
                epoch,

            # Main object-category macro metrics.
            "linear_text_proto_spearman":
                linear_metrics[
                    "text_proto_spearman"
                ],

            "orthogonal_text_proto_spearman":
                ortho_metrics[
                    "text_proto_spearman"
                ],

            "linear_proto_gt_spearman":
                linear_metrics[
                    "proto_gt_spearman"
                ],

            "orthogonal_proto_gt_spearman":
                ortho_metrics[
                    "proto_gt_spearman"
                ],

            # Diagnostics.
            "linear_text_proto_spearman_instance_macro":
                linear_metrics[
                    "text_proto_spearman_instance_macro"
                ],

            "orthogonal_text_proto_spearman_instance_macro":
                ortho_metrics[
                    "text_proto_spearman_instance_macro"
                ],

            "linear_proto_gt_spearman_instance_macro":
                linear_metrics[
                    "proto_gt_spearman_instance_macro"
                ],

            "orthogonal_proto_gt_spearman_instance_macro":
                ortho_metrics[
                    "proto_gt_spearman_instance_macro"
                ],

            "linear_valid_structure_instances":
                linear_metrics[
                    "valid_structure_instances"
                ],

            "orthogonal_valid_structure_instances":
                ortho_metrics[
                    "valid_structure_instances"
                ],

            "linear_valid_object_groups":
                linear_metrics[
                    "valid_object_groups"
                ],

            "orthogonal_valid_object_groups":
                ortho_metrics[
                    "valid_object_groups"
                ],

            "linear_skipped_lt3_parts":
                linear_metrics[
                    "skipped_lt3_parts"
                ],

            "orthogonal_skipped_lt3_parts":
                ortho_metrics[
                    "skipped_lt3_parts"
                ],

            "linear_skipped_undefined_spearman":
                linear_metrics[
                    "skipped_undefined_spearman"
                ],

            "orthogonal_skipped_undefined_spearman":
                ortho_metrics[
                    "skipped_undefined_spearman"
                ],
        }

        rows.append(
            row
        )

        for r in linear_metrics[
            "per_instance"
        ]:
            per_instance_rows.append(
                {
                    "epoch":
                        epoch,
                    **r,
                }
            )

        for r in ortho_metrics[
            "per_instance"
        ]:
            per_instance_rows.append(
                {
                    "epoch":
                        epoch,
                    **r,
                }
            )

        print(
            "[LINEAR] "
            f"Text-Proto="
            f"{linear_metrics['text_proto_spearman']:.6f}  "
            f"Proto-GT="
            f"{linear_metrics['proto_gt_spearman']:.6f}  "
            f"instances="
            f"{linear_metrics['valid_structure_instances']}  "
            f"objects="
            f"{linear_metrics['valid_object_groups']}"
        )

        print(
            "[ORTHO]  "
            f"Text-Proto="
            f"{ortho_metrics['text_proto_spearman']:.6f}  "
            f"Proto-GT="
            f"{ortho_metrics['proto_gt_spearman']:.6f}  "
            f"instances="
            f"{ortho_metrics['valid_structure_instances']}  "
            f"objects="
            f"{ortho_metrics['valid_object_groups']}"
        )

        # Incremental save for long audits.
        base.save_csv(
            rows,
            out_dir
            / "relational_structure_alignment.csv",
        )

        base.save_csv(
            per_instance_rows,
            out_dir
            / "relational_structure_alignment_per_instance.csv",
        )

    plot_results(
        rows,
        out_dir,
    )

    print(
        "\nDONE"
    )

    print(
        "CSV:",
        out_dir
        / "relational_structure_alignment.csv",
    )

    print(
        "Per-instance CSV:",
        out_dir
        / "relational_structure_alignment_per_instance.csv",
    )

    print(
        "PDF:",
        out_dir
        / "figure_relational_structure_alignment.pdf",
    )


if __name__ == "__main__":
    main()
