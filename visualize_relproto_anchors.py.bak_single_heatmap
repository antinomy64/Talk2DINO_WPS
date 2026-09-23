import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from omegaconf import OmegaConf
from PIL import Image


# ---------------------------------------------------------------------
# Talk2DINO repo imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "open_vocabulary_segmentation"))

from models import build_model
from segmentation.datasets.pascalpart116_part import PART_CLASSES


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Visualize RelProto part response maps and anchors at "
            "epoch 0 (W=I) and a trained W checkpoint."
        )
    )

    p.add_argument(
        "--image",
        required=True,
        help="Original training image, used for identification/context."
    )

    p.add_argument(
        "--cache",
        default=(
            "feature/pascalpart116_predobj_clip_struct/"
            "train_predobj_cropaug_bg055_reproduce.pth"
        ),
    )

    p.add_argument(
        "--w_checkpoint",
        required=True,
        help="e.g. .../W_epoch_010.pt"
    )

    p.add_argument(
        "--eval_cfg",
        default=(
            "src/open_vocabulary_segmentation/configs/voc116_part/"
            "dinotext_voc116_part_vitb_mlp_infonce.yml"
        ),
    )

    p.add_argument(
        "--obj_proj_name",
        default="vitb_mlp_infonce_coco2014_reproduce_clean",
        help="Object-level projector name WITHOUT .pth",
    )

    p.add_argument(
        "--template",
        default="sub_imagenet_template",
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--heatmap_mode",
        choices=["relative", "absolute"],
        default="relative",
        help=(
            "relative = R_jp used by the method to select anchors; "
            "absolute = raw cosine S_jp"
        ),
    )

    p.add_argument(
        "--output_dir",
        default="visualization/relproto_anchors",
    )

    return p.parse_args()


def torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def normalize_image_id(x):
    if x is None:
        return ""

    s = str(x).replace("\\", "/")
    return Path(s).stem


def annotation_matches_image(ann, target_stem):
    """
    Cache versions may use slightly different field names.
    Check only identity/path-like fields, not arbitrary annotation contents.
    """
    keys = [
        "image_id",
        "file_name",
        "filename",
        "image_path",
        "img_path",
    ]

    for k in keys:
        if k in ann:
            if normalize_image_id(ann[k]) == target_stem:
                return True

    return False


def get_class_name(ann, idx):
    for k in ["class_name", "object_name", "obj_name", "category_name"]:
        if k in ann:
            return str(ann[k])

    return f"object_{idx}"


def get_tensor(ann, key, dtype=None):
    if key not in ann:
        raise KeyError(
            f"Annotation does not contain required key '{key}'.\n"
            f"Available keys:\n{sorted(ann.keys())}"
        )

    x = ann[key]

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.detach().cpu()

    if dtype is not None:
        x = x.to(dtype)

    return x


def extract_w(checkpoint, dim):
    """
    Robustly find the learned square W in the canonical checkpoint.

    Prefer an explicitly named W key. If unavailable, search for a
    dim x dim tensor and require the candidate to be unambiguous.
    """
    ckpt = torch_load(checkpoint)

    if torch.is_tensor(ckpt):
        if tuple(ckpt.shape) == (dim, dim):
            return ckpt.float()
        raise RuntimeError(
            f"Checkpoint is a tensor with shape {tuple(ckpt.shape)}, "
            f"expected {(dim, dim)}."
        )

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Unexpected checkpoint type: {type(ckpt)}"
        )

    preferred = [
        "W",
        "w",
        "transform",
        "orthogonal_W",
        "orthogonal_w",
    ]

    for key in preferred:
        if key in ckpt and torch.is_tensor(ckpt[key]):
            x = ckpt[key]
            if tuple(x.shape) == (dim, dim):
                print(f"[W] using checkpoint key: {key}")
                return x.detach().cpu().float()

    candidates = []

    def visit(obj, prefix=""):
        if torch.is_tensor(obj):
            if tuple(obj.shape) == (dim, dim):
                candidates.append((prefix, obj))
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{prefix}.{k}" if prefix else str(k))

    visit(ckpt)

    if len(candidates) == 1:
        print(f"[W] using checkpoint tensor: {candidates[0][0]}")
        return candidates[0][1].detach().cpu().float()

    if len(candidates) == 0:
        raise RuntimeError(
            f"Could not find a {dim}x{dim} W tensor in checkpoint.\n"
            f"Top-level keys: {list(ckpt.keys())}"
        )

    raise RuntimeError(
        "Multiple possible W tensors were found:\n"
        + "\n".join(
            f"  {name}: {tuple(x.shape)}"
            for name, x in candidates
        )
    )


def squeeze_text_embedding(x, num_classes):
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise RuntimeError(
                f"Unexpected text embedding list length: {len(x)}"
            )
        x = x[0]

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    # Remove singleton dimensions.
    while x.ndim > 2:
        singleton_dims = [
            i for i, s in enumerate(x.shape)
            if s == 1
        ]

        if not singleton_dims:
            break

        x = x.squeeze(singleton_dims[0])

    if x.ndim != 2:
        raise RuntimeError(
            f"Unexpected text embedding shape: {tuple(x.shape)}"
        )

    if x.shape[0] == num_classes:
        return x

    if x.shape[1] == num_classes:
        return x.T

    raise RuntimeError(
        f"Cannot identify class dimension in text embedding "
        f"{tuple(x.shape)}; expected {num_classes} classes."
    )


@torch.no_grad()
def build_base_text_bank(args):
    """
    Epoch 0:
        t_j = Norm(P_obj(e_j))

    Uses the same Talk2DINO model/text construction route as evaluation,
    with the frozen object projector.
    """
    cfg = OmegaConf.load(args.eval_cfg)
    cfg.model.proj_name = args.obj_proj_name

    model = build_model(cfg.model)
    model = model.to(args.device).eval()

    classnames = list(PART_CLASSES)

    if len(classnames) != 116:
        raise RuntimeError(
            f"Expected 116 PP-116 classes, got {len(classnames)}"
        )

    class_tokens = model.build_dataset_class_tokens(
        args.template,
        classnames,
    )

    text_emb = model.build_text_embedding(class_tokens)
    text_emb = squeeze_text_embedding(
        text_emb,
        len(classnames),
    )

    text_emb = F.normalize(
        text_emb.float(),
        dim=-1,
    )

    return text_emb.detach().cpu(), classnames


def compute_scores(
    base_text,
    W,
    part_ids,
    patch_tokens,
    foreground,
):
    """
    Canonical pipeline:

        current_text = Norm(T0[pids] @ W)
        x_p          = Norm(crop patch token)

        S_jp = current_text_j^T x_p

        if K > 1:
            R_jp = S_jp - max_{k != j} S_kp
        else:
            R_jp = S_jp

        foreground outside Omega is excluded from anchor search.
    """

    q = base_text[part_ids].float() @ W.float()
    q = F.normalize(q, dim=-1)

    x = F.normalize(
        patch_tokens.float(),
        dim=-1,
    )

    absolute = q @ x.T

    K, N = absolute.shape

    if K == 1:
        relative = absolute.clone()

    else:
        relative = torch.empty_like(absolute)

        for j in range(K):
            others = torch.cat(
                [
                    absolute[:j],
                    absolute[j + 1:],
                ],
                dim=0,
            )

            best_other = others.max(dim=0).values

            relative[j] = (
                absolute[j] - best_other
            )

    if foreground.numel() != N:
        raise RuntimeError(
            f"foreground has {foreground.numel()} patches, "
            f"but tokens have {N}"
        )

    if not foreground.any():
        raise RuntimeError(
            "Object has an empty predicted foreground."
        )

    rel_for_anchor = relative.clone()
    rel_for_anchor[:, ~foreground] = -torch.inf

    anchors = rel_for_anchor.argmax(dim=1)

    return {
        "absolute": absolute,
        "relative": relative,
        "anchors": anchors,
        "query": q,
    }


def infer_patch_grid(n):
    side = int(round(math.sqrt(n)))

    if side * side != n:
        raise RuntimeError(
            f"Expected square patch grid, got {n} patches."
        )

    return side, side


def safe_filename(s):
    out = []

    for c in str(s):
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("_")

    return "".join(out)


def value_range(a, b, foreground):
    """
    Shared color scale for epoch0/epoch10 for one semantic part.
    This makes visual change directly comparable.
    """
    values = torch.cat(
        [
            a[foreground],
            b[foreground],
        ]
    ).float()

    lo = float(values.min())
    hi = float(values.max())

    if abs(hi - lo) < 1e-8:
        hi = lo + 1e-8

    return lo, hi


def main():
    args = parse_args()

    image_path = Path(args.image)
    target_stem = image_path.stem

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("RELPROTO ANCHOR VISUALIZATION")
    print("=" * 90)
    print("image         :", image_path)
    print("cache         :", args.cache)
    print("W checkpoint  :", args.w_checkpoint)
    print("heatmap       :", args.heatmap_mode)
    print("object proj   :", args.obj_proj_name)
    print("=" * 90)

    # -----------------------------------------------------------------
    # 1. Epoch-0 projected part-text bank
    # -----------------------------------------------------------------
    print("\n[1/4] Building epoch-0 Talk2DINO part-text bank...")

    base_text, classnames = build_base_text_bank(args)

    D = base_text.shape[-1]

    print("base text:", tuple(base_text.shape))

    # -----------------------------------------------------------------
    # 2. W0 = I and W10 from canonical checkpoint
    # -----------------------------------------------------------------
    W0 = torch.eye(D, dtype=torch.float32)

    W10 = extract_w(
        args.w_checkpoint,
        D,
    )

    orth_err = (
        W10.T @ W10
        - torch.eye(D)
    ).abs().max().item()

    print(
        "epoch10 W orthogonality max abs:",
        f"{orth_err:.6e}",
    )

    # -----------------------------------------------------------------
    # 3. Read the exact PredObj/DINO cache used for training
    # -----------------------------------------------------------------
    print("\n[2/4] Loading training cache...")

    cache = torch_load(args.cache)

    if not isinstance(cache, dict):
        raise TypeError(
            f"Unexpected cache type: {type(cache)}"
        )

    if "annotations" not in cache:
        raise KeyError(
            f"Cache has no 'annotations' key. "
            f"Available: {list(cache.keys())}"
        )

    annotations = list(cache["annotations"])

    matches = [
        (idx, ann)
        for idx, ann in enumerate(annotations)
        if annotation_matches_image(
            ann,
            target_stem,
        )
    ]

    if not matches:
        print(
            f"\n[ERROR] No cached object annotation found for "
            f"{target_stem}."
        )

        print("\nFirst annotation keys:")
        if annotations:
            print(sorted(annotations[0].keys()))

        print(
            "\nThis can happen if bg=0.55 produced an empty "
            "object mask and the object was omitted from the cache."
        )

        raise SystemExit(2)

    print(
        f"Found {len(matches)} cached object(s) "
        f"for image {target_stem}."
    )

    # Context image, not used for computation.
    rgb = np.asarray(
        Image.open(image_path).convert("RGB")
    )

    summary_rows = []

    # -----------------------------------------------------------------
    # 4. Visualize every cached object independently
    # -----------------------------------------------------------------
    print("\n[3/4] Computing responses and anchors...")

    for object_no, (cache_idx, ann) in enumerate(
        matches,
        start=1,
    ):
        object_name = get_class_name(
            ann,
            object_no,
        )

        pids = get_tensor(
            ann,
            "part_category_id",
            torch.long,
        ).reshape(-1)

        tokens = get_tensor(
            ann,
            "cropaug_patch_tokens",
            torch.float32,
        )

        foreground = get_tensor(
            ann,
            "pred_obj_mask_patch",
            torch.bool,
        ).reshape(-1)

        if tokens.ndim != 2:
            raise RuntimeError(
                f"Expected [N,D] patch tokens, "
                f"got {tuple(tokens.shape)}"
            )

        if tokens.shape[1] != D:
            raise RuntimeError(
                f"Patch dim {tokens.shape[1]} != text dim {D}"
            )

        N = tokens.shape[0]
        gh, gw = infer_patch_grid(N)

        print()
        print("-" * 90)
        print(
            f"Object {object_no}/{len(matches)}"
        )
        print("cache annotation :", cache_idx)
        print("object            :", object_name)
        print("parts             :", pids.tolist())
        print("patch grid        :", f"{gh}x{gw}")
        print(
            "foreground patches:",
            f"{int(foreground.sum())}/{N}",
        )

        part_names = [
            classnames[int(pid)]
            for pid in pids
        ]

        print("part names:")
        for pid, name in zip(
            pids.tolist(),
            part_names,
        ):
            print(f"  {pid:3d}  {name}")

        out0 = compute_scores(
            base_text,
            W0,
            pids,
            tokens,
            foreground,
        )

        out10 = compute_scores(
            base_text,
            W10,
            pids,
            tokens,
            foreground,
        )

        score_key = args.heatmap_mode

        score0 = out0[score_key]
        score10 = out10[score_key]

        K = len(pids)

        # -------------------------------------------------------------
        # Figure:
        #
        # row = semantic part
        # col0 = epoch 0
        # col1 = epoch 10
        #
        # Heatmaps are on the exact 32x32 crop patch coordinate system
        # used by RelProto.
        #
        # Outside predicted foreground Omega is masked.
        # -------------------------------------------------------------
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(
            figsize=(11.5, max(3.8, 3.5 * K))
        )

        gs = GridSpec(
            K,
            3,
            figure=fig,
            width_ratios=[1.0, 1.0, 0.04],
            wspace=0.22,
            hspace=0.35,
        )

        axes = np.empty((K, 2), dtype=object)
        cbar_axes = []

        for j in range(K):
            axes[j, 0] = fig.add_subplot(gs[j, 0])
            axes[j, 1] = fig.add_subplot(gs[j, 1])
            cbar_axes.append(
                fig.add_subplot(gs[j, 2])
            )

        fig.suptitle(
            (
                f"{target_stem} | object: {object_name} | "
                f"{args.heatmap_mode.capitalize()} evidence\n"
                "white contour = PredObj foreground, "
                "★ = RelProto anchor"
            ),
            fontsize=13,
        )

        fg_grid = foreground.numpy().reshape(
            gh,
            gw,
        )

        for j in range(K):
            pid = int(pids[j])
            part_name = part_names[j]

            a0 = int(out0["anchors"][j])
            a10 = int(out10["anchors"][j])

            r0, c0 = divmod(a0, gw)
            r10, c10 = divmod(a10, gw)

            vmin, vmax = value_range(
                score0[j],
                score10[j],
                foreground,
            )

            row_images = []

            for col, (
                epoch_name,
                score,
                anchor,
                ar,
                ac,
            ) in enumerate(
                [
                    (
                        "Epoch 0 (W = I)",
                        score0[j],
                        a0,
                        r0,
                        c0,
                    ),
                    (
                        "Epoch 10",
                        score10[j],
                        a10,
                        r10,
                        c10,
                    ),
                ]
            ):
                ax = axes[j, col]

                grid = score.numpy().reshape(
                    gh,
                    gw,
                ).astype(np.float32)

                masked = np.ma.array(
                    grid,
                    mask=~fg_grid,
                )

                cmap = cm.get_cmap("magma").copy()
                cmap.set_bad((0.12, 0.12, 0.12, 1.0))

                im = ax.imshow(
                    masked,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )

                # Predicted object foreground boundary.
                ax.contour(
                    fg_grid.astype(np.float32),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.8,
                    alpha=0.8,
                )

                # Anchor star.
                ax.scatter(
                    [ac],
                    [ar],
                    marker="*",
                    s=230,
                    c="cyan",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=20,
                )

                anchor_score = float(
                    score[anchor]
                )

                relative_anchor = float(
                    (
                        out0["relative"]
                        if col == 0
                        else out10["relative"]
                    )[j, anchor]
                )

                ax.set_title(
                    (
                        f"{epoch_name}\n"
                        f"{part_name} [id={pid}] | "
                        f"anchor=({ar},{ac}) | "
                        f"value={anchor_score:.4f}"
                    ),
                    fontsize=9,
                )

                ax.set_xticks([])
                ax.set_yticks([])

                row_images.append(im)

                summary_rows.append({
                    "image": target_stem,
                    "cache_annotation_index": cache_idx,
                    "object_index": object_no,
                    "object_name": object_name,
                    "part_id": pid,
                    "part_name": part_name,
                    "epoch": 0 if col == 0 else 10,
                    "heatmap_mode": args.heatmap_mode,
                    "anchor_patch_index": anchor,
                    "anchor_row": ar,
                    "anchor_col": ac,
                    "display_value": anchor_score,
                    "relative_evidence_at_anchor": relative_anchor,
                })

            # One shared colorbar per semantic part.
            fig.colorbar(
                row_images[-1],
                cax=cbar_axes[j],
            )

        fig.subplots_adjust(
            top=0.93,
            bottom=0.05,
            left=0.05,
            right=0.95,
        )

        obj_tag = safe_filename(
            f"{object_no:02d}_{object_name}"
        )

        fig_path = (
            out_dir
            / f"{target_stem}_{obj_tag}_"
              f"{args.heatmap_mode}_anchors.png"
        )

        fig.savefig(
            fig_path,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("saved:", fig_path)

        # -------------------------------------------------------------
        # Save a simple context image as well.
        # This original image is NOT involved in anchor calculation.
        # -------------------------------------------------------------
        context_path = (
            out_dir
            / f"{target_stem}_context.png"
        )

        if not context_path.exists():
            Image.fromarray(rgb).save(
                context_path
            )

    # -----------------------------------------------------------------
    # CSV summary
    # -----------------------------------------------------------------
    print("\n[4/4] Saving anchor table...")

    csv_path = (
        out_dir
        / f"{target_stem}_{args.heatmap_mode}_anchors.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("Figures :", out_dir)
    print("CSV     :", csv_path)


if __name__ == "__main__":
    main()
