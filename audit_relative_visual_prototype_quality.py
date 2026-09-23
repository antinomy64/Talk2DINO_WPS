#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import ProjectionLayer


NUM_PARTS = 116
DINO_DIM = 768


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cache",
        default=(
            "feature/pascalpart116_predobj_clip_struct/"
            "train_predobj_cropaug_bg055.pth"
        ),
    )

    p.add_argument(
        "--data_root",
        default="data/PascalPart116",
    )

    p.add_argument(
        "--split",
        default="train",
    )

    p.add_argument(
        "--absolute_weight_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "runs/core/absolute_orthogonal_L8/train"
        ),
    )

    p.add_argument(
        "--relative_weight_dir",
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

    p.add_argument("--L", type=int, default=8)
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--end_epoch", type=int, default=30)

    p.add_argument(
        "--input_size",
        type=int,
        default=448,
    )

    p.add_argument(
        "--expected_bg_thresh",
        type=float,
        default=0.55,
    )

    p.add_argument(
        "--gt_label_offset",
        type=int,
        default=0,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--out_dir",
        default=(
            "nightly_ablation_pp116_seed123_20260921/"
            "analysis/relative_visual_prototype_quality"
        ),
    )

    p.add_argument(
        "--gt_audit_cache",
        default=None,
    )

    p.add_argument(
        "--rebuild_gt_audit_cache",
        action="store_true",
    )

    return p.parse_args()


# ============================================================
# Base text
# ============================================================

def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj

    for key in [
        "state_dict",
        "model_state_dict",
        "model",
        "projector",
        "proj",
    ]:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    return obj


def clean_state_dict(state):
    out = {}

    for k, v in state.items():
        nk = k

        for prefix in [
            "module.",
            "proj.",
            "projection_layer.",
        ]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        out[nk] = v

    return out


def find_text_tensor(obj):
    found = []

    def visit(x, name="root"):
        if torch.is_tensor(x):
            if x.ndim == 2 and tuple(x.shape) == (116, 512):
                found.append((name, x))
            return

        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, f"{name}.{k}")

        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                visit(v, f"{name}[{i}]")

    visit(obj)

    if not found:
        raise RuntimeError(
            "Cannot find [116,512] CLIP text tensor"
        )

    print(
        "[INFO] text tensor:",
        found[0][0],
        tuple(found[0][1].shape),
    )

    return found[0][1].float()


@torch.no_grad()
def load_base_text(args):

    device = torch.device(args.device)

    projector = ProjectionLayer.from_config(
        args.model_config
    )

    ckpt = torch.load(
        args.base_projector,
        map_location="cpu",
    )

    projector.load_state_dict(
        clean_state_dict(
            unwrap_state_dict(ckpt)
        ),
        strict=False,
    )

    projector = projector.to(device).eval()

    text_obj = torch.load(
        args.text_bank,
        map_location="cpu",
    )

    raw_text = find_text_tensor(
        text_obj
    ).to(device)

    text = projector.project_clip_txt(
        raw_text.float()
    )

    text = F.normalize(
        text.float(),
        dim=-1,
    )

    return text.cpu()


# ============================================================
# PredObj cache
# ============================================================

def get_any(d, names, default=None):

    for name in names:
        if isinstance(d, dict) and name in d:
            return d[name]

    return default


def is_record(x):

    if not isinstance(x, dict):
        return False

    return (
        "cropaug_patch_tokens" in x
        and "pred_obj_mask_patch" in x
    )


def cache_to_records(cache):

    if isinstance(cache, (list, tuple)):
        return list(cache), {}

    if not isinstance(cache, dict):
        raise RuntimeError(
            f"Unsupported cache type: {type(cache)}"
        )

    meta = get_any(
        cache,
        ["meta", "metadata", "__meta__"],
        {},
    )

    for key in [
        "records",
        "samples",
        "items",
        "data",
        "entries",
    ]:
        if key in cache and isinstance(
            cache[key],
            (list, tuple),
        ):
            return list(cache[key]), meta

    # dict: image_id -> record
    records = []

    for key, value in cache.items():

        if is_record(value):

            r = dict(value)
            r.setdefault(
                "_record_key",
                key,
            )

            records.append(r)

    if records:
        return records, meta

    # batched dict
    if "cropaug_patch_tokens" in cache:

        tokens = cache[
            "cropaug_patch_tokens"
        ]

        if (
            torch.is_tensor(tokens)
            and tokens.ndim >= 3
        ):
            n = tokens.shape[0]

            records = []

            for i in range(n):

                r = {}

                for k, v in cache.items():

                    if k in [
                        "meta",
                        "metadata",
                        "__meta__",
                    ]:
                        continue

                    if (
                        torch.is_tensor(v)
                        and v.ndim > 0
                        and v.shape[0] == n
                    ):
                        r[k] = v[i]

                    elif (
                        isinstance(v, (list, tuple))
                        and len(v) == n
                    ):
                        r[k] = v[i]

                    else:
                        r[k] = v

                r["_record_key"] = i

                records.append(r)

            return records, meta

    raise RuntimeError(
        "Cannot parse pred-object cache.\n"
        f"Top-level keys = {list(cache.keys())[:50]}"
    )


def canonical_tokens(x):

    x = torch.as_tensor(
        x
    ).float()

    if x.ndim == 3:

        if x.shape[-1] == 768:

            x = x.reshape(
                -1,
                768,
            )

        elif x.shape[0] == 768:

            x = x.permute(
                1,
                2,
                0,
            ).reshape(
                -1,
                768,
            )

        else:
            raise RuntimeError(
                f"Unknown token shape {tuple(x.shape)}"
            )

    if (
        x.ndim != 2
        or x.shape[1] != 768
    ):
        raise RuntimeError(
            f"Expected [N,768], got {tuple(x.shape)}"
        )

    return F.normalize(
        x,
        dim=-1,
    )


def canonical_fg(x, n):

    x = torch.as_tensor(
        x
    ).reshape(-1)

    if x.numel() != n:
        raise RuntimeError(
            f"foreground={x.numel()}, tokens={n}"
        )

    if x.dtype == torch.bool:
        return x

    return x.float() > 0.5


def decode_part_ids(x):

    if x is None:
        raise RuntimeError(
            "Missing part_category_id"
        )

    if torch.is_tensor(x):

        x = x.detach().cpu().reshape(-1)

        if (
            x.numel() == 116
            and bool(
                torch.all(
                    (x == 0) | (x == 1)
                )
            )
        ):

            ids = torch.nonzero(
                x > 0,
                as_tuple=False,
            ).reshape(-1).tolist()

        else:
            ids = [
                int(v)
                for v in x.tolist()
            ]

    elif isinstance(
        x,
        (list, tuple, set),
    ):

        x = list(x)

        if (
            len(x) == 116
            and all(
                v in [0, 1, False, True]
                for v in x
            )
        ):

            ids = [
                i
                for i, v in enumerate(x)
                if bool(v)
            ]

        else:
            ids = [
                int(v)
                for v in x
            ]

    else:

        ids = [int(x)]

    ids = sorted(
        set(
            i
            for i in ids
            if 0 <= i < 116
        )
    )

    if not ids:
        raise RuntimeError(
            "No valid part ids"
        )

    return ids


def get_part_ids(record):

    return decode_part_ids(
        get_any(
            record,
            [
                "part_category_id",
                "part_category_ids",
                "part_ids",
                "present_part_ids",
                "part_labels",
            ],
        )
    )


def get_image_ref(record):

    x = get_any(
        record,
        [
            "image_name",
            "img_name",
            "file_name",
            "filename",
            "image_path",
            "img_path",
            "image_id",
            "img_id",
        ],
        None,
    )

    if x is None:
        x = record.get(
            "_record_key"
        )

    if (
        torch.is_tensor(x)
        and x.numel() == 1
    ):
        x = x.item()

    return str(x)


def get_crop_box(record):

    x = get_any(
        record,
        [
            "cropaug_box_xyxy",
            "crop_box_xyxy",
            "box_xyxy",
            "crop_box",
        ],
    )

    if x is None:
        raise RuntimeError(
            "Missing cropaug_box_xyxy"
        )

    x = torch.as_tensor(
        x
    ).detach().cpu().reshape(-1)

    if x.numel() != 4:
        raise RuntimeError(
            f"Invalid crop box: {x}"
        )

    return tuple(
        float(v)
        for v in x.tolist()
    )


# ============================================================
# GT crop -> patch occupancy
# ============================================================

def build_mask_index(mask_root):

    mask_root = Path(
        mask_root
    )

    paths = list(
        mask_root.rglob("*.png")
    )

    if not paths:
        raise RuntimeError(
            f"No masks under {mask_root}"
        )

    index = defaultdict(list)

    for p in paths:
        index[p.stem].append(p)

    return index


def resolve_mask(
    image_ref,
    mask_root,
    index,
):

    stem = Path(
        str(image_ref)
    ).stem

    p = Path(
        mask_root
    ) / f"{stem}.png"

    if p.exists():
        return p

    hits = index.get(
        stem,
        [],
    )

    if len(hits) == 1:
        return hits[0]

    if len(hits) > 1:
        raise RuntimeError(
            f"Ambiguous mask: {hits}"
        )

    raise FileNotFoundError(
        f"GT mask not found for {image_ref}"
    )


def crop_mask(
    mask,
    box,
    fill=255,
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise RuntimeError(
            f"Bad crop box {box}"
        )

    out = np.full(
        (y2 - y1, x2 - x1),
        fill,
        dtype=mask.dtype,
    )

    H, W = mask.shape

    sx1 = max(
        x1,
        0,
    )

    sy1 = max(
        y1,
        0,
    )

    sx2 = min(
        x2,
        W,
    )

    sy2 = min(
        y2,
        H,
    )

    if (
        sx2 <= sx1
        or sy2 <= sy1
    ):
        return out

    dx1 = sx1 - x1
    dy1 = sy1 - y1

    dx2 = dx1 + (
        sx2 - sx1
    )

    dy2 = dy1 + (
        sy2 - sy1
    )

    out[
        dy1:dy2,
        dx1:dx2,
    ] = mask[
        sy1:sy2,
        sx1:sx2,
    ]

    return out


def make_gt_patch_stats(
    mask,
    part_ids,
    grid,
    input_size,
    label_offset,
):

    mask = np.array(
        Image.fromarray(
            mask
        ).resize(
            (
                input_size,
                input_size,
            ),
            resample=Image.Resampling.NEAREST,
        )
    )

    patch = (
        input_size
        // grid
    )

    if (
        grid * patch
        != input_size
    ):
        raise RuntimeError(
            f"{input_size=} not divisible by {grid=}"
        )

    # --------------------------------------------------------
    # Main Support Purity definition:
    # patch center belongs to corresponding GT part.
    # --------------------------------------------------------

    ys = (
        np.arange(grid)
        * patch
        + patch // 2
    )

    xs = (
        np.arange(grid)
        * patch
        + patch // 2
    )

    center_labels = mask[
        np.ix_(ys, xs)
    ].reshape(-1)

    # --------------------------------------------------------
    # Fractional occupancy for oracle GT prototype.
    # [Npatch, patch_pixels]
    # --------------------------------------------------------

    patch_pixels = mask.reshape(
        grid,
        patch,
        grid,
        patch,
    ).transpose(
        0,
        2,
        1,
        3,
    ).reshape(
        grid * grid,
        patch * patch,
    )

    center = []
    occupancy = []

    for pid in part_ids:

        label = (
            pid
            + label_offset
        )

        center.append(
            torch.from_numpy(
                center_labels
                == label
            )
        )

        occupancy.append(
            torch.from_numpy(
                (
                    patch_pixels
                    == label
                ).mean(
                    axis=1
                ).astype(
                    np.float32
                )
            )
        )

    return (
        torch.stack(
            center,
            dim=0,
        ),
        torch.stack(
            occupancy,
            dim=0,
        ),
    )


def build_gt_audit_cache(
    records,
    args,
    output,
):

    mask_root = (
        Path(args.data_root)
        / "annotations_detectron2_part"
        / args.split
    )

    index = build_mask_index(
        mask_root
    )

    audit = []

    print(
        "[INFO] building GT audit cache"
    )

    for i, record in enumerate(
        records
    ):

        tokens = canonical_tokens(
            record[
                "cropaug_patch_tokens"
            ]
        )

        n = tokens.shape[0]

        grid = int(
            round(
                math.sqrt(n)
            )
        )

        if grid * grid != n:
            raise RuntimeError(
                f"Patch count {n} is not square"
            )

        part_ids = get_part_ids(
            record
        )

        image_ref = get_image_ref(
            record
        )

        mask_path = resolve_mask(
            image_ref,
            mask_root,
            index,
        )

        gt = np.array(
            Image.open(
                mask_path
            )
        )

        if gt.ndim == 3:
            gt = gt[..., 0]

        gt = crop_mask(
            gt,
            get_crop_box(record),
            fill=255,
        )

        center, occupancy = (
            make_gt_patch_stats(
                gt,
                part_ids,
                grid,
                args.input_size,
                args.gt_label_offset,
            )
        )

        audit.append(
            {
                "part_ids": torch.tensor(
                    part_ids,
                    dtype=torch.long,
                ),
                "center": center.bool(),
                "occupancy": occupancy.half(),
            }
        )

        if (
            (i + 1) % 250 == 0
            or i + 1 == len(records)
        ):
            print(
                f"[GT] {i + 1}/{len(records)}"
            )

    Path(
        output
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "records": audit,
            "source_cache": args.cache,
            "input_size": args.input_size,
            "gt_label_offset": args.gt_label_offset,
        },
        output,
    )

    print(
        "[OK] saved GT audit cache:",
        output,
    )

    return audit


def load_gt_audit(
    records,
    args,
):

    path = (
        args.gt_audit_cache
        or str(
            Path(args.out_dir)
            / "gt_crop_audit_cache.pt"
        )
    )

    path = Path(path)

    if (
        path.exists()
        and not args.rebuild_gt_audit_cache
    ):

        obj = torch.load(
            path,
            map_location="cpu",
        )

        audit = obj[
            "records"
        ]

        if (
            len(audit)
            != len(records)
        ):
            raise RuntimeError(
                "GT audit cache length mismatch. "
                "Use --rebuild_gt_audit_cache"
            )

        print(
            "[INFO] GT audit cache:",
            path,
        )

        return audit

    return build_gt_audit_cache(
        records,
        args,
        path,
    )


# ============================================================
# W checkpoints
# ============================================================

def load_W(
    weight_dir,
    epoch,
):

    if epoch == 0:
        return torch.eye(
            768
        )

    path = (
        Path(weight_dir)
        / f"W_epoch_{epoch:03d}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    obj = torch.load(
        path,
        map_location="cpu",
    )

    if (
        not isinstance(obj, dict)
        or "W" not in obj
    ):
        raise RuntimeError(
            f"{path} has no W"
        )

    W = obj[
        "W"
    ].float()

    if tuple(
        W.shape
    ) != (
        768,
        768,
    ):
        raise RuntimeError(
            f"Bad W shape: {tuple(W.shape)}"
        )

    if (
        obj.get("mapping") is not None
        and str(
            obj.get("mapping")
        ) != "orthogonal"
    ):
        raise RuntimeError(
            f"{path}: mapping={obj.get('mapping')}"
        )

    return W


# ============================================================
# Absolute / Relative support
# ============================================================

def get_evidence(
    text,
    tokens,
    selector,
):

    S = text @ tokens.T

    if selector == "absolute":
        return S

    if selector != "relative":
        raise ValueError(
            selector
        )

    K = S.shape[0]

    if K == 1:
        return S

    R = torch.empty_like(
        S
    )

    for j in range(K):

        others = torch.cat(
            [
                S[:j],
                S[j + 1:],
            ],
            dim=0,
        )

        R[j] = (
            S[j]
            - others.max(
                dim=0
            ).values
        )

    return R


def select_supports(
    evidence,
    foreground,
    L,
):

    supports = []

    for j in range(
        evidence.shape[0]
    ):

        score = evidence[
            j
        ].clone()

        score[
            ~foreground
        ] = -torch.inf

        if not bool(
            torch.isfinite(
                score
            ).any()
        ):
            supports.append(
                None
            )
            continue

        # own anchor, foreground only
        anchor = int(
            torch.argmax(
                score
            ).item()
        )

        # ----------------------------------------------------
        # Both Absolute and Relative keep the same support rule:
        # anchor + up to L-1 non-anchor patches with evidence > 0.
        # No force filling.
        # ----------------------------------------------------

        positive = (
            foreground
            & (
                evidence[j]
                > 0
            )
        )

        positive[
            anchor
        ] = False

        ids = torch.nonzero(
            positive,
            as_tuple=False,
        ).reshape(-1)

        if (
            L > 1
            and ids.numel() > 0
        ):

            k = min(
                L - 1,
                ids.numel(),
            )

            order = torch.topk(
                evidence[
                    j,
                    ids,
                ],
                k=k,
            ).indices

            extra = ids[
                order
            ]

            support = torch.cat(
                [
                    torch.tensor(
                        [anchor],
                        device=evidence.device,
                    ),
                    extra,
                ],
                dim=0,
            )

        else:

            support = torch.tensor(
                [anchor],
                device=evidence.device,
            )

        supports.append(
            support
        )

    return supports


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate(
    records,
    gt_audit,
    base_text,
    W,
    selector,
    args,
):

    device = torch.device(
        args.device
    )

    base_text = base_text.to(
        device
    )

    W = W.to(
        device
    )

    total_support = 0
    center_hits = 0

    # robustness only
    majority_hits = 0
    fractional_hits = 0.0

    proto_cosines = []
    purity_instances = []
    support_sizes = []

    class_purity = defaultdict(
        list
    )

    class_proto = defaultdict(
        list
    )

    valid_instances = 0

    for record, gt_info in zip(
        records,
        gt_audit,
    ):

        tokens = canonical_tokens(
            record[
                "cropaug_patch_tokens"
            ]
        )

        fg = canonical_fg(
            record[
                "pred_obj_mask_patch"
            ],
            tokens.shape[0],
        )

        part_ids = get_part_ids(
            record
        )

        gt_ids = torch.as_tensor(
            gt_info[
                "part_ids"
            ]
        ).long().tolist()

        if part_ids != gt_ids:
            raise RuntimeError(
                "part id mismatch between "
                "PredObj and GT audit cache"
            )

        tokens = tokens.to(
            device
        )

        fg = fg.to(
            device
        )

        if not bool(
            fg.any()
        ):
            continue

        text = F.normalize(
            base_text[
                part_ids
            ] @ W,
            dim=-1,
        )

        evidence = get_evidence(
            text,
            tokens,
            selector,
        )

        supports = select_supports(
            evidence,
            fg,
            args.L,
        )

        center = torch.as_tensor(
            gt_info[
                "center"
            ]
        ).bool().to(
            device
        )

        occupancy = torch.as_tensor(
            gt_info[
                "occupancy"
            ]
        ).float().to(
            device
        )

        for j, pid in enumerate(
            part_ids
        ):

            support = supports[
                j
            ]

            if (
                support is None
                or support.numel() == 0
            ):
                continue

            occ = occupancy[
                j
            ]

            # GT part does not exist inside current crop
            if float(
                occ.sum().item()
            ) <= 1e-8:
                continue

            # ------------------------------------------------
            # Support Purity
            # main = patch-center correctness
            # ------------------------------------------------

            n = int(
                support.numel()
            )

            hits = int(
                center[
                    j,
                    support,
                ].sum().item()
            )

            total_support += n
            center_hits += hits

            selected_occ = occ[
                support
            ]

            majority_hits += int(
                (
                    selected_occ
                    > 0.5
                ).sum().item()
            )

            fractional_hits += float(
                selected_occ.sum().item()
            )

            purity = (
                hits
                / n
            )

            purity_instances.append(
                purity
            )

            class_purity[
                pid
            ].append(
                purity
            )

            support_sizes.append(
                n
            )

            # ------------------------------------------------
            # induced visual prototype
            # ------------------------------------------------

            induced = F.normalize(
                tokens[
                    support
                ].mean(
                    dim=0
                ),
                dim=0,
            )

            # ------------------------------------------------
            # crop-specific oracle GT prototype
            #
            # GT occupancy weights each SAME DINO patch token.
            # ------------------------------------------------

            oracle = F.normalize(
                (
                    occ[:, None]
                    * tokens
                ).sum(
                    dim=0
                ),
                dim=0,
            )

            cosine = float(
                torch.dot(
                    induced,
                    oracle,
                ).item()
            )

            proto_cosines.append(
                cosine
            )

            class_proto[
                pid
            ].append(
                cosine
            )

            valid_instances += 1

    if (
        total_support == 0
        or not proto_cosines
    ):
        raise RuntimeError(
            "No valid instances evaluated"
        )

    purity_class_macro = float(
        np.mean(
            [
                np.mean(v)
                for v in class_purity.values()
                if v
            ]
        )
    )

    proto_class_macro = float(
        np.mean(
            [
                np.mean(v)
                for v in class_proto.values()
                if v
            ]
        )
    )

    return {
        # main metrics
        "support_purity":
            center_hits
            / total_support,

        "proto_gt_cosine":
            float(
                np.mean(
                    proto_cosines
                )
            ),

        # diagnostics
        "support_purity_instance_macro":
            float(
                np.mean(
                    purity_instances
                )
            ),

        "support_purity_majority":
            majority_hits
            / total_support,

        "support_purity_fractional":
            fractional_hits
            / total_support,

        "support_purity_class_macro":
            purity_class_macro,

        "proto_gt_cosine_median":
            float(
                np.median(
                    proto_cosines
                )
            ),

        "proto_gt_cosine_class_macro":
            proto_class_macro,

        "mean_support_size":
            float(
                np.mean(
                    support_sizes
                )
            ),

        "selected_support_patches":
            total_support,

        "valid_part_instances":
            valid_instances,
    }


# ============================================================
# Output
# ============================================================

def save_csv(
    rows,
    path,
):

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def plot_results(
    rows,
    out_dir,
):

    epochs = [
        r["epoch"]
        for r in rows
    ]

    absolute_purity = [
        r[
            "absolute_support_purity"
        ]
        for r in rows
    ]

    relative_purity = [
        r[
            "relative_support_purity"
        ]
        for r in rows
    ]

    absolute_cos = [
        r[
            "absolute_proto_gt_cosine"
        ]
        for r in rows
    ]

    relative_cos = [
        r[
            "relative_proto_gt_cosine"
        ]
        for r in rows
    ]

    blue = "#1f77b4"
    orange = "#ff7f0e"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.2,
            4.3,
        ),
    )

    ax = axes[0]

    ax.plot(
        epochs,
        absolute_purity,
        color=blue,
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Absolute + Orthogonal",
    )

    ax.plot(
        epochs,
        relative_purity,
        color=orange,
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
        "Support Purity",
        fontsize=13,
    )

    ax.set_title(
        "(a) Support Purity",
        fontsize=13,
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

    ax = axes[1]

    ax.plot(
        epochs,
        absolute_cos,
        color=blue,
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=2,
        label="Absolute + Orthogonal",
    )

    ax.plot(
        epochs,
        relative_cos,
        color=orange,
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
        "Proto–GT Cosine",
        fontsize=13,
    )

    ax.set_title(
        "(b) Proto–GT Cosine",
        fontsize=13,
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
            0.93,
        ]
    )

    out_dir = Path(
        out_dir
    )

    plt.savefig(
        out_dir
        / "figure4_relative_visual_prototype_quality.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        out_dir
        / "figure4_relative_visual_prototype_quality.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[INFO] loading PredObj cache:",
        args.cache,
    )

    cache = torch.load(
        args.cache,
        map_location="cpu",
    )

    records, metadata = (
        cache_to_records(
            cache
        )
    )

    print(
        "[INFO] records:",
        len(records),
    )

    # cache metadata sanity check
    bg = None

    for source in [
        metadata,
        cache
        if isinstance(cache, dict)
        else {},
    ]:

        if not isinstance(
            source,
            dict,
        ):
            continue

        for key in [
            "bg_thresh",
            "background_threshold",
            "pred_bg_thresh",
        ]:

            if key in source:

                try:
                    bg = float(
                        source[key]
                    )
                except Exception:
                    pass

    if bg is not None:

        print(
            f"[INFO] cache bg_thresh={bg}"
        )

        if abs(
            bg
            - args.expected_bg_thresh
        ) > 1e-8:

            raise RuntimeError(
                f"cache bg_thresh={bg}, "
                f"expected={args.expected_bg_thresh}"
            )

    else:

        print(
            "[WARN] bg_thresh not exposed in cache metadata"
        )

    base_text = load_base_text(
        args
    )

    print(
        "[INFO] base text:",
        tuple(
            base_text.shape
        ),
    )

    gt_audit = load_gt_audit(
        records,
        args,
    )

    rows = []

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

        W_abs = load_W(
            args.absolute_weight_dir,
            epoch,
        )

        W_rel = load_W(
            args.relative_weight_dir,
            epoch,
        )

        abs_metrics = evaluate(
            records,
            gt_audit,
            base_text,
            W_abs,
            "absolute",
            args,
        )

        rel_metrics = evaluate(
            records,
            gt_audit,
            base_text,
            W_rel,
            "relative",
            args,
        )

        row = {
            "epoch":
                epoch,

            "absolute_support_purity":
                abs_metrics[
                    "support_purity"
                ],

            "relative_support_purity":
                rel_metrics[
                    "support_purity"
                ],

            "absolute_proto_gt_cosine":
                abs_metrics[
                    "proto_gt_cosine"
                ],

            "relative_proto_gt_cosine":
                rel_metrics[
                    "proto_gt_cosine"
                ],

            "absolute_support_purity_majority":
                abs_metrics[
                    "support_purity_majority"
                ],

            "relative_support_purity_majority":
                rel_metrics[
                    "support_purity_majority"
                ],

            "absolute_support_purity_fractional":
                abs_metrics[
                    "support_purity_fractional"
                ],

            "relative_support_purity_fractional":
                rel_metrics[
                    "support_purity_fractional"
                ],

            "absolute_proto_gt_cosine_class_macro":
                abs_metrics[
                    "proto_gt_cosine_class_macro"
                ],

            "relative_proto_gt_cosine_class_macro":
                rel_metrics[
                    "proto_gt_cosine_class_macro"
                ],

            "absolute_mean_support_size":
                abs_metrics[
                    "mean_support_size"
                ],

            "relative_mean_support_size":
                rel_metrics[
                    "mean_support_size"
                ],

            "absolute_valid_part_instances":
                abs_metrics[
                    "valid_part_instances"
                ],

            "relative_valid_part_instances":
                rel_metrics[
                    "valid_part_instances"
                ],
        }

        rows.append(
            row
        )

        print(
            "[ABS] "
            f"Purity={abs_metrics['support_purity']:.6f}  "
            f"Proto-GT={abs_metrics['proto_gt_cosine']:.6f}  "
            f"mean|P|={abs_metrics['mean_support_size']:.3f}"
        )

        print(
            "[REL] "
            f"Purity={rel_metrics['support_purity']:.6f}  "
            f"Proto-GT={rel_metrics['proto_gt_cosine']:.6f}  "
            f"mean|P|={rel_metrics['mean_support_size']:.3f}"
        )

        # incremental save
        save_csv(
            rows,
            out_dir
            / "relative_visual_prototype_quality.csv",
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
        / "relative_visual_prototype_quality.csv",
    )

    print(
        "PDF:",
        out_dir
        / "figure4_relative_visual_prototype_quality.pdf",
    )


if __name__ == "__main__":
    main()