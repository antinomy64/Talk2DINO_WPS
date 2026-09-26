#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4.4.1 Structure Preservation and Part-Level Alignment
=====================================================

Controlled comparison:
    Relative + Linear,      L=8
    Relative + Orthogonal,  L=8

For every epoch e, compute three relational-structure metrics:

    1) Adapted Text--Prototype Structure Correlation
       Spearman( relation(T0 @ W_e), relation(RelProto_e) )

    2) Prototype--GT Structure Correlation
       Spearman( relation(RelProto_e), relation(GTProto) )

    3) Adapted Text--GT Structure Correlation
       Spearman( relation(T0 @ W_e), relation(GTProto) )

Protocol
--------
A. T0:
   frozen CLIP text -> frozen Talk2DINO projector -> normalized projected
   part-text representations, shape [116, 768].

B. Adapted text at epoch e:
       T_e = Norm(T0 @ W_e)

C. Relative visual prototype:
   For every training image/object record, use T_e to compute relative
   discriminative evidence inside the predicted object foreground:

       S_jp = T_j dot X_p
       R_jp = S_jp - max_{k != j} S_kp

   The support for part j is:
       anchor = argmax R_jp
       + up to L-1 non-anchor foreground patches with R_jp > 0
       + no force filling

   The image-specific relative visual prototype is the normalized mean
   of selected normalized DINO patch tokens.

D. Global relative prototype bank:
   For every epoch and method, average all image-specific relative
   prototypes belonging to the same part class, then L2 normalize:

       RelProto_e[j] = Norm(mean_n r_{j,n}^{(e)})

   This gives up to 116 global relative visual prototypes.

E. GT visual prototype bank:
   Use the already-built global PP-116 GT visual prototype bank from
   the previous text-vs-GT structural audit.

F. Spearman:
   For each object category separately:
       - select all valid parts belonging to that object
       - require >= 3 valid parts
       - construct upper-triangular pairwise cosine relation vectors
       - compute the three Spearman correlations above

   Finally:
       macro Spearman = mean over valid object categories

Thus every epoch produces one macro value for each metric and method.

GT masks / GT prototypes are used only for post-hoc analysis.
They never participate in training or relative support selection.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Reuse the already-verified cache / text / checkpoint / selector semantics.
# The file must exist in the same directory:
#
#   audit_structure_preservation_alignment.py
# -------------------------------------------------------------------------

import audit_structure_preservation_alignment as base


NUM_PARTS = 116
DINO_DIM = 768


# =========================================================================
# Arguments
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "PP-116 4.4.1 relational structure audit: "
            "Relative+Linear vs Relative+Orthogonal."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--project_root",
        default=".",
    )

    p.add_argument(
        "--cache",
        default=(
            "feature/pascalpart116_predobj_clip_struct/"
            "train_predobj_cropaug_bg055_reproduce.pth"
        ),
    )

    p.add_argument(
        "--linear_weight_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "runs/core/relative_linear_L8/train"
        ),
    )

    p.add_argument(
        "--orthogonal_weight_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "runs/core/relative_orthogonal_L8/train"
        ),
    )

    p.add_argument(
        "--base_projector",
        default=(
            "weights/"
            "vitb_mlp_infonce_coco2014_reproduce_clean.pth"
        ),
    )

    p.add_argument(
        "--text_bank",
        default=(
            "feature/pascalpart116_clip_text/"
            "pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"
        ),
    )

    p.add_argument(
        "--model_config",
        default="configs/vitb_mlp_infonce.yaml",
    )

    p.add_argument(
        "--classes_source",
        default=(
            "src/open_vocabulary_segmentation/segmentation/datasets/"
            "pascalpart116_part.py:PART_CLASSES"
        ),
    )

    p.add_argument(
        "--gt_prototypes",
        default=(
            "feature/pascalpart116_gt_visual_structure/"
            "gt_dinov2_vitb14reg_train_fullimg_patchpool.pt"
        ),
        help=(
            "Global 116-class GT visual prototype bank used by the "
            "previous text-vs-GT structural audit."
        ),
    )

    p.add_argument(
        "--gt_prototype_key",
        default=None,
        help=(
            "Optional dot-separated key if the GT prototype checkpoint "
            "contains multiple [116,768] tensors."
        ),
    )

    p.add_argument(
        "--L",
        type=int,
        default=8,
    )

    p.add_argument(
        "--start_epoch",
        type=int,
        default=0,
    )

    p.add_argument(
        "--end_epoch",
        type=int,
        default=30,
    )

    p.add_argument(
        "--expected_bg_thresh",
        type=float,
        default=0.55,
    )

    p.add_argument(
        "--no_require_pamr",
        dest="require_pamr",
        action="store_false",
    )

    p.add_argument(
        "--no_require_cache_projector_match",
        dest="require_cache_projector_match",
        action="store_false",
    )

    p.set_defaults(
        require_pamr=True,
        require_cache_projector_match=True,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--out_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "analysis/4_4_1_relational_structure"
        ),
    )

    return p.parse_args()


# =========================================================================
# Small utilities
# =========================================================================

def resolve(root: Path, raw) -> Path:
    return base.resolve(root, raw)


def save_csv(rows, path: Path):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        dict.fromkeys(
            k
            for row in rows
            for k in row.keys()
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================================
# GT prototype loading
# =========================================================================

def get_by_dot_key(obj: Any, key: str):
    cur = obj

    for frag in key.split("."):
        if isinstance(cur, dict):
            cur = cur[frag]
        else:
            raise KeyError(
                f"Cannot descend through {frag!r} in {type(cur)}"
            )

    return cur


def collect_116x768_tensors(
    obj: Any,
    path: str = "root",
):
    """
    Recursively collect tensors / numpy arrays of shape [116,768].
    """

    hits = []

    if torch.is_tensor(obj):
        if tuple(obj.shape) == (
            NUM_PARTS,
            DINO_DIM,
        ):
            hits.append(
                (
                    path,
                    obj,
                )
            )

        return hits

    if isinstance(obj, np.ndarray):
        if tuple(obj.shape) == (
            NUM_PARTS,
            DINO_DIM,
        ):
            hits.append(
                (
                    path,
                    torch.from_numpy(obj),
                )
            )

        return hits

    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(
                collect_116x768_tensors(
                    v,
                    f"{path}.{k}",
                )
            )

        return hits

    if isinstance(
        obj,
        (list, tuple),
    ):
        for i, v in enumerate(obj):
            hits.extend(
                collect_116x768_tensors(
                    v,
                    f"{path}[{i}]",
                )
            )

    return hits


def load_gt_prototypes(
    path: Path,
    key: str | None,
):
    obj = base.tload(path)

    if key is not None:
        x = get_by_dot_key(
            obj,
            key,
        )

        x = torch.as_tensor(
            x
        ).float()

        if tuple(x.shape) != (
            NUM_PARTS,
            DINO_DIM,
        ):
            raise RuntimeError(
                f"--gt_prototype_key={key!r} gives "
                f"shape {tuple(x.shape)}, expected (116,768)"
            )

        print(
            f"[INFO] GT prototype tensor key: {key}"
        )

    elif torch.is_tensor(obj):
        x = obj.float()

        if tuple(x.shape) != (
            NUM_PARTS,
            DINO_DIM,
        ):
            raise RuntimeError(
                f"GT prototype tensor has shape {tuple(x.shape)}, "
                "expected (116,768)"
            )

        print(
            "[INFO] GT prototype checkpoint is directly a [116,768] tensor"
        )

    else:
        hits = collect_116x768_tensors(
            obj
        )

        if len(hits) == 0:
            raise RuntimeError(
                "Could not find a [116,768] tensor in GT prototype file"
            )

        if len(hits) > 1:
            print(
                "[INFO] Candidate [116,768] tensors in GT prototype file:"
            )

            for pth, _ in hits:
                print(
                    "   ",
                    pth,
                )

            # Prefer paths whose names suggest prototypes/features.
            preferred = []

            for pth, tensor in hits:
                s = pth.lower()

                if any(
                    frag in s
                    for frag in (
                        "prototype",
                        "proto",
                        "feature",
                        "feat",
                        "mean",
                    )
                ):
                    preferred.append(
                        (
                            pth,
                            tensor,
                        )
                    )

            if len(preferred) == 1:
                chosen_path, x = preferred[0]

            else:
                raise RuntimeError(
                    "GT file contains multiple [116,768] tensors. "
                    "Please pass --gt_prototype_key."
                )

        else:
            chosen_path, x = hits[0]

        print(
            f"[INFO] GT prototype tensor: {chosen_path}"
        )

        x = x.float()

    if not torch.isfinite(x).all():
        raise RuntimeError(
            "GT prototype bank contains NaN/Inf"
        )

    norms = x.norm(
        dim=-1
    )

    valid = (
        torch.isfinite(norms)
        & (norms > 1e-8)
    )

    x = F.normalize(
        x,
        dim=-1,
    )

    print(
        f"[INFO] GT prototypes shape={tuple(x.shape)} "
        f"valid={int(valid.sum())}/{NUM_PARTS}"
    )

    return x.cpu(), valid.cpu()


# =========================================================================
# Object groups
# =========================================================================

VOC_OBJECT_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def infer_object_name(
    class_name: str,
):
    """
    Infer the parent object from a PP-116 part-class name.

    Primary rule:
        startswith one of the Pascal VOC object names.

    This supports names such as:
        cat's head
        cat_head
        cat:head
        cat head
    """

    s = (
        str(class_name)
        .strip()
        .lower()
    )

    # Longer names first, e.g. pottedplant / diningtable / tvmonitor.
    for obj in sorted(
        VOC_OBJECT_NAMES,
        key=len,
        reverse=True,
    ):
        if s.startswith(obj):
            return obj

    # Fallbacks for unusual class naming.
    if "'s " in s:
        return s.split(
            "'s ",
            1,
        )[0].strip()

    if ":" in s:
        return s.split(
            ":",
            1,
        )[0].strip()

    if "/" in s:
        return s.split(
            "/",
            1,
        )[0].strip()

    raise RuntimeError(
        f"Cannot infer object category from part class name: {class_name!r}"
    )


def build_object_groups(
    class_names,
):
    groups = defaultdict(list)

    for pid, name in enumerate(
        class_names
    ):
        obj = infer_object_name(
            name
        )

        groups[obj].append(
            pid
        )

    groups = {
        obj: ids
        for obj, ids in groups.items()
    }

    print(
        "\n[INFO] PP-116 object groups inferred from PART_CLASSES:"
    )

    for obj, ids in groups.items():
        print(
            f"  {obj:14s}: "
            f"{len(ids):2d} parts  "
            f"{ids}"
        )

    print(
        f"[INFO] total object groups={len(groups)}"
    )

    if sum(
        len(v)
        for v in groups.values()
    ) != NUM_PARTS:
        raise RuntimeError(
            "Object grouping does not cover all 116 classes"
        )

    return groups


# =========================================================================
# Spearman
# =========================================================================

def rankdata_average(
    x: np.ndarray,
):
    """
    Average ranks for ties, equivalent to scipy.stats.rankdata(..., average).
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    n = x.size

    if n == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    i = 0

    while i < n:
        j = i + 1

        while (
            j < n
            and x[order[j]]
            == x[order[i]]
        ):
            j += 1

        # 1-based ranks.
        first_rank = i + 1
        last_rank = j

        avg = 0.5 * (
            first_rank
            + last_rank
        )

        ranks[
            order[i:j]
        ] = avg

        i = j

    return ranks


def spearman_1d(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(
            f"Spearman shape mismatch: {x.shape} vs {y.shape}"
        )

    if x.size < 2:
        return float(
            "nan"
        )

    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        return float(
            "nan"
        )

    rx = rankdata_average(
        x
    )

    ry = rankdata_average(
        y
    )

    rx = (
        rx
        - rx.mean()
    )

    ry = (
        ry
        - ry.mean()
    )

    denom = math.sqrt(
        float(
            np.dot(
                rx,
                rx,
            )
        )
        *
        float(
            np.dot(
                ry,
                ry,
            )
        )
    )

    if denom <= 1e-12:
        return float(
            "nan"
        )

    return float(
        np.dot(
            rx,
            ry,
        )
        / denom
    )


def relation_vector(
    features: torch.Tensor,
):
    """
    Pairwise cosine relation vector from upper triangle.

    Input:
        [K,D], assumed finite

    Output:
        [K*(K-1)/2]
    """

    if features.ndim != 2:
        raise ValueError(
            f"Expected [K,D], got {tuple(features.shape)}"
        )

    k = features.shape[0]

    if k < 2:
        return np.empty(
            0,
            dtype=np.float64,
        )

    x = F.normalize(
        features.float(),
        dim=-1,
    )

    sim = (
        x
        @ x.T
    )

    idx = torch.triu_indices(
        k,
        k,
        offset=1,
        device=sim.device,
    )

    return (
        sim[
            idx[0],
            idx[1],
        ]
        .detach()
        .cpu()
        .double()
        .numpy()
    )


# =========================================================================
# Relative visual prototype bank
# =========================================================================

@torch.no_grad()
def build_two_global_relative_prototype_banks(
    records,
    base_text,
    W_linear,
    W_orthogonal,
    L,
    device,
):
    """
    Generate one global 116-class relative visual prototype bank for
    Linear and one for Orthogonal.

    For every image/object record:
        1. compute current adapted text T0 @ W
        2. compute relative discriminative scores
        3. induce image-specific relative prototypes
        4. accumulate the normalized image-specific prototype into its class

    Final class prototype:
        normalize(mean of all image-specific normalized prototypes)
    """

    device = torch.device(
        device
    )

    base_text = base_text.to(
        device=device,
        dtype=torch.float32,
    )

    W_linear = W_linear.to(
        device=device,
        dtype=torch.float32,
    )

    W_orthogonal = W_orthogonal.to(
        device=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Current epoch adapted text.
    #
    # These are ALSO the text representations whose relational geometry
    # is evaluated in 4.4.1.
    # ------------------------------------------------------------------

    adapted_linear = F.normalize(
        base_text
        @ W_linear,
        dim=-1,
    )

    adapted_orthogonal = F.normalize(
        base_text
        @ W_orthogonal,
        dim=-1,
    )

    linear_sum = torch.zeros(
        NUM_PARTS,
        DINO_DIM,
        dtype=torch.float32,
        device=device,
    )

    ortho_sum = torch.zeros(
        NUM_PARTS,
        DINO_DIM,
        dtype=torch.float32,
        device=device,
    )

    linear_count = torch.zeros(
        NUM_PARTS,
        dtype=torch.long,
        device=device,
    )

    ortho_count = torch.zeros(
        NUM_PARTS,
        dtype=torch.long,
        device=device,
    )

    for rec_idx, record in enumerate(
        records
    ):
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

        X = tokens_full[
            foreground
        ]

        part_ids = record[
            "_part_ids"
        ]

        ids = torch.tensor(
            part_ids,
            device=device,
            dtype=torch.long,
        )

        # --------------------------------------------------------------
        # Relative + Linear
        # --------------------------------------------------------------

        T_linear = (
            adapted_linear
            .index_select(
                0,
                ids,
            )
        )

        S_linear = (
            T_linear
            @ X.T
        )

        R_linear = base.relative_scores(
            S_linear
        )

        # --------------------------------------------------------------
        # Relative + Orthogonal
        # --------------------------------------------------------------

        T_ortho = (
            adapted_orthogonal
            .index_select(
                0,
                ids,
            )
        )

        S_ortho = (
            T_ortho
            @ X.T
        )

        R_ortho = base.relative_scores(
            S_ortho
        )

        # --------------------------------------------------------------
        # Image-specific prototypes
        # --------------------------------------------------------------

        for j, pid in enumerate(
            part_ids
        ):
            proto_linear, _ = base.build_proto(
                X,
                R_linear[j],
                L,
                "relative",
            )

            proto_ortho, _ = base.build_proto(
                X,
                R_ortho[j],
                L,
                "relative",
            )

            pid = int(
                pid
            )

            linear_sum[
                pid
            ] += proto_linear

            ortho_sum[
                pid
            ] += proto_ortho

            linear_count[
                pid
            ] += 1

            ortho_count[
                pid
            ] += 1

        if (
            (rec_idx + 1) % 1000 == 0
            or rec_idx + 1 == len(records)
        ):
            print(
                f"[PROTO] "
                f"{rec_idx + 1}/{len(records)}"
            )

    # The two methods see exactly the same part instances.
    if not torch.equal(
        linear_count,
        ortho_count,
    ):
        raise AssertionError(
            "Linear and Orthogonal class counts differ"
        )

    valid = (
        linear_count
        > 0
    )

    linear_global = torch.zeros_like(
        linear_sum
    )

    ortho_global = torch.zeros_like(
        ortho_sum
    )

    if bool(
        valid.any()
    ):
        linear_mean = (
            linear_sum[
                valid
            ]
            /
            linear_count[
                valid
            ]
            .float()
            .unsqueeze(1)
        )

        ortho_mean = (
            ortho_sum[
                valid
            ]
            /
            ortho_count[
                valid
            ]
            .float()
            .unsqueeze(1)
        )

        linear_global[
            valid
        ] = F.normalize(
            linear_mean,
            dim=-1,
        )

        ortho_global[
            valid
        ] = F.normalize(
            ortho_mean,
            dim=-1,
        )

    return {
        "adapted_linear":
            adapted_linear.detach().cpu(),

        "adapted_orthogonal":
            adapted_orthogonal.detach().cpu(),

        "proto_linear":
            linear_global.detach().cpu(),

        "proto_orthogonal":
            ortho_global.detach().cpu(),

        "valid_proto":
            valid.detach().cpu(),

        "class_counts":
            linear_count.detach().cpu(),
    }


# =========================================================================
# Object-macro structure metrics
# =========================================================================

def compute_epoch_metrics(
    epoch,
    banks,
    gt_proto,
    gt_valid,
    object_groups,
):
    """
    For every object separately, compute:

        rho(TW, R)
        rho(R, GT)
        rho(TW, GT)

    for both Linear and Orthogonal.

    Only object categories for which ALL SIX correlations are defined
    are retained. Therefore both methods use exactly the same objects
    for the macro average.
    """

    adapted_linear = banks[
        "adapted_linear"
    ]

    adapted_ortho = banks[
        "adapted_orthogonal"
    ]

    proto_linear = banks[
        "proto_linear"
    ]

    proto_ortho = banks[
        "proto_orthogonal"
    ]

    proto_valid = banks[
        "valid_proto"
    ]

    common_part_valid = (
        proto_valid
        & gt_valid
    )

    object_rows = []

    macro_values = {
        "linear_text_proto": [],
        "linear_proto_gt": [],
        "linear_text_gt": [],
        "orthogonal_text_proto": [],
        "orthogonal_proto_gt": [],
        "orthogonal_text_gt": [],
    }

    skipped_lt3 = []
    skipped_undefined = []

    for object_name, raw_ids in (
        object_groups.items()
    ):
        ids = [
            int(pid)
            for pid in raw_ids
            if bool(
                common_part_valid[
                    int(pid)
                ]
            )
        ]

        if len(ids) < 3:
            skipped_lt3.append(
                object_name
            )
            continue

        idx = torch.tensor(
            ids,
            dtype=torch.long,
        )

        # --------------------------------------------------------------
        # Relation vectors
        # --------------------------------------------------------------

        text_linear_rel = relation_vector(
            adapted_linear.index_select(
                0,
                idx,
            )
        )

        text_ortho_rel = relation_vector(
            adapted_ortho.index_select(
                0,
                idx,
            )
        )

        proto_linear_rel = relation_vector(
            proto_linear.index_select(
                0,
                idx,
            )
        )

        proto_ortho_rel = relation_vector(
            proto_ortho.index_select(
                0,
                idx,
            )
        )

        gt_rel = relation_vector(
            gt_proto.index_select(
                0,
                idx,
            )
        )

        # --------------------------------------------------------------
        # Three metrics, Linear
        # --------------------------------------------------------------

        linear_text_proto = spearman_1d(
            text_linear_rel,
            proto_linear_rel,
        )

        linear_proto_gt = spearman_1d(
            proto_linear_rel,
            gt_rel,
        )

        linear_text_gt = spearman_1d(
            text_linear_rel,
            gt_rel,
        )

        # --------------------------------------------------------------
        # Three metrics, Orthogonal
        # --------------------------------------------------------------

        ortho_text_proto = spearman_1d(
            text_ortho_rel,
            proto_ortho_rel,
        )

        ortho_proto_gt = spearman_1d(
            proto_ortho_rel,
            gt_rel,
        )

        ortho_text_gt = spearman_1d(
            text_ortho_rel,
            gt_rel,
        )

        values = (
            linear_text_proto,
            linear_proto_gt,
            linear_text_gt,
            ortho_text_proto,
            ortho_proto_gt,
            ortho_text_gt,
        )

        # Keep exactly the same valid objects for every metric/method.
        if not all(
            np.isfinite(v)
            for v in values
        ):
            skipped_undefined.append(
                object_name
            )
            continue

        num_parts = len(
            ids
        )

        num_pairs = (
            num_parts
            * (num_parts - 1)
            // 2
        )

        object_rows.append(
            {
                "epoch":
                    epoch,

                "object":
                    object_name,

                "num_parts":
                    num_parts,

                "num_pairs":
                    num_pairs,

                "part_ids":
                    " ".join(
                        str(x)
                        for x in ids
                    ),

                "linear_text_proto_spearman":
                    linear_text_proto,

                "linear_proto_gt_spearman":
                    linear_proto_gt,

                "linear_text_gt_spearman":
                    linear_text_gt,

                "orthogonal_text_proto_spearman":
                    ortho_text_proto,

                "orthogonal_proto_gt_spearman":
                    ortho_proto_gt,

                "orthogonal_text_gt_spearman":
                    ortho_text_gt,
            }
        )

        macro_values[
            "linear_text_proto"
        ].append(
            linear_text_proto
        )

        macro_values[
            "linear_proto_gt"
        ].append(
            linear_proto_gt
        )

        macro_values[
            "linear_text_gt"
        ].append(
            linear_text_gt
        )

        macro_values[
            "orthogonal_text_proto"
        ].append(
            ortho_text_proto
        )

        macro_values[
            "orthogonal_proto_gt"
        ].append(
            ortho_proto_gt
        )

        macro_values[
            "orthogonal_text_gt"
        ].append(
            ortho_text_gt
        )

    valid_objects = len(
        object_rows
    )

    if valid_objects == 0:
        raise RuntimeError(
            "No valid object groups for Spearman macro"
        )

    row = {
        "epoch":
            epoch,

        "linear_text_proto_spearman":
            float(
                np.mean(
                    macro_values[
                        "linear_text_proto"
                    ]
                )
            ),

        "orthogonal_text_proto_spearman":
            float(
                np.mean(
                    macro_values[
                        "orthogonal_text_proto"
                    ]
                )
            ),

        "linear_proto_gt_spearman":
            float(
                np.mean(
                    macro_values[
                        "linear_proto_gt"
                    ]
                )
            ),

        "orthogonal_proto_gt_spearman":
            float(
                np.mean(
                    macro_values[
                        "orthogonal_proto_gt"
                    ]
                )
            ),

        "linear_text_gt_spearman":
            float(
                np.mean(
                    macro_values[
                        "linear_text_gt"
                    ]
                )
            ),

        "orthogonal_text_gt_spearman":
            float(
                np.mean(
                    macro_values[
                        "orthogonal_text_gt"
                    ]
                )
            ),

        "valid_object_groups":
            valid_objects,

        "skipped_object_groups_lt3":
            len(
                skipped_lt3
            ),

        "skipped_object_groups_undefined":
            len(
                skipped_undefined
            ),
    }

    return (
        row,
        object_rows,
        skipped_lt3,
        skipped_undefined,
    )


# =========================================================================
# Plot
# =========================================================================

def plot_results(
    rows,
    out_dir,
):
    epochs = [
        r["epoch"]
        for r in rows
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15.0,
            4.3,
        ),
    )

    # ------------------------------------------------------------------
    # (a) T0W <-> Relative Prototype
    # ------------------------------------------------------------------

    ax = axes[0]

    ax.plot(
        epochs,
        [
            r[
                "linear_text_proto_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Relative + Linear",
    )

    ax.plot(
        epochs,
        [
            r[
                "orthogonal_text_proto_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="s",
        markersize=4,
        markevery=2,
        label="Relative + Orthogonal",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Spearman $\rho$",
        fontsize=12,
    )

    ax.set_title(
        "(a) Adapted Text--Prototype",
        fontsize=12,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs),
    )

    ax.set_xticks(
        range(
            0,
            31,
            5,
        )
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # (b) Relative Prototype <-> GT Prototype
    # ------------------------------------------------------------------

    ax = axes[1]

    ax.plot(
        epochs,
        [
            r[
                "linear_proto_gt_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Relative + Linear",
    )

    ax.plot(
        epochs,
        [
            r[
                "orthogonal_proto_gt_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="s",
        markersize=4,
        markevery=2,
        label="Relative + Orthogonal",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Spearman $\rho$",
        fontsize=12,
    )

    ax.set_title(
        "(b) Prototype--GT",
        fontsize=12,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs),
    )

    ax.set_xticks(
        range(
            0,
            31,
            5,
        )
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # (c) T0W <-> GT Prototype
    # ------------------------------------------------------------------

    ax = axes[2]

    ax.plot(
        epochs,
        [
            r[
                "linear_text_gt_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Relative + Linear",
    )

    ax.plot(
        epochs,
        [
            r[
                "orthogonal_text_gt_spearman"
            ]
            for r in rows
        ],
        linewidth=2.4,
        marker="s",
        markersize=4,
        markevery=2,
        label="Relative + Orthogonal",
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Spearman $\rho$",
        fontsize=12,
    )

    ax.set_title(
        "(c) Adapted Text--GT",
        fontsize=12,
    )

    ax.set_xlim(
        min(epochs),
        max(epochs),
    )

    ax.set_xticks(
        range(
            0,
            31,
            5,
        )
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
            0.92,
        ]
    )

    pdf_path = (
        out_dir
        / "figure_4_4_1_relational_structure.pdf"
    )

    png_path = (
        out_dir
        / "figure_4_4_1_relational_structure.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

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
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    root = (
        Path(
            args.project_root
        )
        .expanduser()
        .resolve()
    )

    out_dir = resolve(
        root,
        args.out_dir,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = resolve(
        root,
        args.cache,
    )

    linear_dir = resolve(
        root,
        args.linear_weight_dir,
    )

    ortho_dir = resolve(
        root,
        args.orthogonal_weight_dir,
    )

    projector_path = resolve(
        root,
        args.base_projector,
    )

    gt_proto_path = resolve(
        root,
        args.gt_prototypes,
    )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

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
    # T0 = frozen projector output
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
    # Checkpoint metadata
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
    # Class / object structure
    # ------------------------------------------------------------------

    class_names = base.load_class_names(
        root,
        args.classes_source,
    )

    object_groups = build_object_groups(
        class_names
    )

    # ------------------------------------------------------------------
    # Fixed GT global visual prototypes
    # ------------------------------------------------------------------

    print(
        f"\n[INFO] loading GT prototype bank: {gt_proto_path}"
    )

    gt_proto, gt_valid = load_gt_prototypes(
        gt_proto_path,
        args.gt_prototype_key,
    )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    epoch_rows = []
    per_object_rows = []

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

        # --------------------------------------------------------------
        # Generate current epoch global relative prototypes.
        # --------------------------------------------------------------

        banks = build_two_global_relative_prototype_banks(
            records=records,
            base_text=base_text,
            W_linear=W_linear,
            W_orthogonal=W_ortho,
            L=args.L,
            device=args.device,
        )

        # --------------------------------------------------------------
        # Sanity checks on class counts.
        # --------------------------------------------------------------

        class_counts = banks[
            "class_counts"
        ]

        valid_proto_classes = int(
            (
                class_counts
                > 0
            ).sum()
        )

        print(
            f"[INFO] relative-prototype classes="
            f"{valid_proto_classes}/{NUM_PARTS}"
        )

        if valid_proto_classes < 3:
            raise RuntimeError(
                "Too few global relative prototypes"
            )

        # --------------------------------------------------------------
        # Object-macro Spearman
        # --------------------------------------------------------------

        (
            row,
            obj_rows,
            skipped_lt3,
            skipped_undefined,
        ) = compute_epoch_metrics(
            epoch=epoch,
            banks=banks,
            gt_proto=gt_proto,
            gt_valid=gt_valid,
            object_groups=object_groups,
        )

        # --------------------------------------------------------------
        # Epoch 0 must be identical:
        # W_linear = W_ortho = I.
        # --------------------------------------------------------------

        if epoch == 0:
            checks = (
                (
                    "text_proto",
                    row[
                        "linear_text_proto_spearman"
                    ],
                    row[
                        "orthogonal_text_proto_spearman"
                    ],
                ),
                (
                    "proto_gt",
                    row[
                        "linear_proto_gt_spearman"
                    ],
                    row[
                        "orthogonal_proto_gt_spearman"
                    ],
                ),
                (
                    "text_gt",
                    row[
                        "linear_text_gt_spearman"
                    ],
                    row[
                        "orthogonal_text_gt_spearman"
                    ],
                ),
            )

            for name, a, b in checks:
                if abs(
                    float(a)
                    - float(b)
                ) > 1e-10:
                    raise RuntimeError(
                        "Epoch-0 sanity check failed: "
                        f"{name}: "
                        f"linear={a}, "
                        f"orthogonal={b}"
                    )

            print(
                "[OK] epoch-0 Linear/Orthogonal "
                "metrics are exactly identical"
            )

        epoch_rows.append(
            row
        )

        per_object_rows.extend(
            obj_rows
        )

        # --------------------------------------------------------------
        # Console output
        # --------------------------------------------------------------

        print(
            "[LINEAR] "
            f"Text-Proto="
            f"{row['linear_text_proto_spearman']:.6f}  "
            f"Proto-GT="
            f"{row['linear_proto_gt_spearman']:.6f}  "
            f"Text-GT="
            f"{row['linear_text_gt_spearman']:.6f}  "
            f"objects="
            f"{row['valid_object_groups']}"
        )

        print(
            "[ORTHO]  "
            f"Text-Proto="
            f"{row['orthogonal_text_proto_spearman']:.6f}  "
            f"Proto-GT="
            f"{row['orthogonal_proto_gt_spearman']:.6f}  "
            f"Text-GT="
            f"{row['orthogonal_text_gt_spearman']:.6f}  "
            f"objects="
            f"{row['valid_object_groups']}"
        )

        if skipped_lt3:
            print(
                "[INFO] objects skipped (<3 valid parts):",
                skipped_lt3,
            )

        if skipped_undefined:
            print(
                "[WARN] objects skipped "
                "(undefined Spearman):",
                skipped_undefined,
            )

        # --------------------------------------------------------------
        # Incremental save
        # --------------------------------------------------------------

        save_csv(
            epoch_rows,
            out_dir
            / "4_4_1_relational_structure.csv",
        )

        save_csv(
            per_object_rows,
            out_dir
            / "4_4_1_relational_structure_per_object.csv",
        )

    # ------------------------------------------------------------------
    # Plot all three relational metrics.
    # mIoU is NOT recomputed here.
    # ------------------------------------------------------------------

    plot_results(
        epoch_rows,
        out_dir,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        "CSV:",
        out_dir
        / "4_4_1_relational_structure.csv",
    )

    print(
        "Per-object CSV:",
        out_dir
        / "4_4_1_relational_structure_per_object.csv",
    )

    print(
        "PDF:",
        out_dir
        / "figure_4_4_1_relational_structure.pdf",
    )

    print(
        "PNG:",
        out_dir
        / "figure_4_4_1_relational_structure.png",
    )


if __name__ == "__main__":
    main()