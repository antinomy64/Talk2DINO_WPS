import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

import visualize_relproto_anchors as base


def parse_args():
    p = argparse.ArgumentParser(
        "Visualize all RelProto parts and epochs in one figure"
    )

    p.add_argument("--image", required=True)

    p.add_argument(
        "--w_dir",
        required=True,
        help="Directory containing W_epoch_001.pt ...",
    )

    p.add_argument(
        "--max_epoch",
        type=int,
        default=10,
    )

    p.add_argument(
        "--cache",
        default=(
            "feature/pascalpart116_predobj_clip_struct/"
            "train_predobj_cropaug_bg055_reproduce.pth"
        ),
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
    )

    p.add_argument(
        "--output_dir",
        default="visualization/relproto_epoch_sequence",
    )

    return p.parse_args()


def row_range(outputs, j, mode, foreground, epochs):
    vals = []

    for e in epochs:
        s = outputs[e][mode][j]
        vals.append(
            s[foreground].float().reshape(-1)
        )

    vals = torch.cat(vals)

    lo = float(vals.min())
    hi = float(vals.max())

    if abs(hi - lo) < 1e-8:
        hi = lo + 1e-8

    return lo, hi


def main():
    args = parse_args()

    image_path = Path(args.image)
    stem = image_path.stem

    w_dir = Path(args.w_dir)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = list(
        range(0, args.max_epoch + 1)
    )

    print("=" * 90)
    print("RELPROTO ALL-PART / ALL-EPOCH VISUALIZATION")
    print("=" * 90)
    print("image       :", image_path)
    print("W dir       :", w_dir)
    print("epochs      :", epochs)
    print("heatmap     :", args.heatmap_mode)
    print("=" * 90)

    # ==============================================================
    # 1. Base text bank
    # ==============================================================
    print("\n[1/4] Building base Talk2DINO text bank...")

    base_text, classnames = (
        base.build_base_text_bank(args)
    )

    D = base_text.shape[-1]

    # ==============================================================
    # 2. Load W0 ... WN
    # ==============================================================
    print("\n[2/4] Loading W checkpoints...")

    Ws = {
        0: torch.eye(
            D,
            dtype=torch.float32,
        )
    }

    for e in range(
        1,
        args.max_epoch + 1,
    ):
        p = w_dir / f"W_epoch_{e:03d}.pt"

        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}"
            )

        Ws[e] = base.extract_w(
            p,
            D,
        )

        orth_err = (
            Ws[e].T @ Ws[e]
            - torch.eye(D)
        ).abs().max().item()

        print(
            f"epoch {e:02d}: "
            f"{p.name}, "
            f"orth_err={orth_err:.3e}"
        )

    # ==============================================================
    # 3. Load exact training cache
    # ==============================================================
    print("\n[3/4] Loading PredObj cache...")

    cache = base.torch_load(
        args.cache
    )

    anns = list(
        cache["annotations"]
    )

    matches = [
        (idx, ann)
        for idx, ann in enumerate(anns)
        if base.annotation_matches_image(
            ann,
            stem,
        )
    ]

    if not matches:
        raise RuntimeError(
            f"No cache annotation found for {stem}"
        )

    print(
        f"Found {len(matches)} object(s)."
    )

    csv_rows = []

    # ==============================================================
    # 4. One figure per OBJECT
    # ==============================================================
    for object_no, (
        cache_idx,
        ann,
    ) in enumerate(
        matches,
        start=1,
    ):
        obj_name = base.get_class_name(
            ann,
            object_no,
        )

        pids = base.get_tensor(
            ann,
            "part_category_id",
            torch.long,
        ).reshape(-1)

        tokens = base.get_tensor(
            ann,
            "cropaug_patch_tokens",
            torch.float32,
        )

        foreground = base.get_tensor(
            ann,
            "pred_obj_mask_patch",
            torch.bool,
        ).reshape(-1)

        gh, gw = base.infer_patch_grid(
            tokens.shape[0]
        )

        part_names = [
            classnames[int(pid)]
            for pid in pids
        ]

        K = len(pids)
        E = len(epochs)

        print()
        print("-" * 90)
        print("object     :", obj_name)
        print("parts      :", part_names)
        print("rows       :", K)
        print("epoch cols :", E)
        print(
            "foreground :",
            f"{int(foreground.sum())}/{foreground.numel()}",
        )

        # ----------------------------------------------------------
        # Compute all epochs
        # ----------------------------------------------------------
        outputs = {}

        for e in epochs:
            outputs[e] = base.compute_scores(
                base_text,
                Ws[e],
                pids,
                tokens,
                foreground,
            )

        fg_grid = (
            foreground.numpy()
            .reshape(gh, gw)
        )

        # ----------------------------------------------------------
        # Grid:
        #
        # rows = parts
        # cols = epochs
        # final narrow col = colorbar for each row
        # ----------------------------------------------------------
        fig_width = max(
            16,
            2.05 * E + 0.8,
        )

        fig_height = max(
            4,
            2.15 * K + 1.4,
        )

        fig = plt.figure(
            figsize=(
                fig_width,
                fig_height,
            )
        )

        gs = GridSpec(
            K,
            E + 1,
            figure=fig,
            width_ratios=(
                [1.0] * E
                + [0.035]
            ),
            wspace=0.10,
            hspace=0.18,
        )

        fig.suptitle(
            (
                f"{stem} | Object: {obj_name} | "
                f"{args.heatmap_mode.capitalize()} Evidence\n"
                "★ = RelProto Anchor"
            ),
            fontsize=16,
            y=0.985,
        )

        cmap = cm.get_cmap(
            "magma"
        ).copy()

        cmap.set_bad(
            (0.10, 0.10, 0.10, 1.0)
        )

        # ==========================================================
        # ROW = PART
        # ==========================================================
        for j, (
            pid,
            part_name,
        ) in enumerate(
            zip(
                pids.tolist(),
                part_names,
            )
        ):
            # Same scale across all epochs for THIS PART.
            vmin, vmax = row_range(
                outputs,
                j,
                args.heatmap_mode,
                foreground,
                epochs,
            )

            last_im = None

            # ======================================================
            # COLUMN = EPOCH
            # ======================================================
            for col, e in enumerate(
                epochs
            ):
                ax = fig.add_subplot(
                    gs[j, col]
                )

                out = outputs[e]

                score = out[
                    args.heatmap_mode
                ][j]

                anchor = int(
                    out["anchors"][j]
                )

                ar, ac = divmod(
                    anchor,
                    gw,
                )

                grid = (
                    score.numpy()
                    .reshape(gh, gw)
                    .astype(np.float32)
                )

                masked = np.ma.array(
                    grid,
                    mask=~fg_grid,
                )

                last_im = ax.imshow(
                    masked,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )

                # PredObj boundary
                ax.contour(
                    fg_grid.astype(
                        np.float32
                    ),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.55,
                    alpha=0.8,
                )

                # Anchor star
                ax.scatter(
                    [ac],
                    [ar],
                    marker="*",
                    s=110,
                    c="cyan",
                    edgecolors="black",
                    linewidths=0.65,
                    zorder=20,
                )

                # --------------------------------------------------
                # Top row: epoch names
                # --------------------------------------------------
                if j == 0:
                    ax.set_title(
                        f"Epoch {e}",
                        fontsize=10,
                        pad=5,
                    )

                # --------------------------------------------------
                # Left side: part name
                # --------------------------------------------------
                if col == 0:
                    ax.set_ylabel(
                        f"{part_name}\n[id={pid}]",
                        fontsize=10,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=42,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

                abs_anchor = float(
                    out["absolute"][
                        j,
                        anchor,
                    ]
                )

                rel_anchor = float(
                    out["relative"][
                        j,
                        anchor,
                    ]
                )

                csv_rows.append({
                    "image": stem,
                    "cache_annotation_index": cache_idx,
                    "object_index": object_no,
                    "object_name": obj_name,
                    "part_id": pid,
                    "part_name": part_name,
                    "epoch": e,
                    "heatmap_mode": args.heatmap_mode,
                    "anchor_patch_index": anchor,
                    "anchor_row": ar,
                    "anchor_col": ac,
                    "absolute_similarity_at_anchor": abs_anchor,
                    "relative_evidence_at_anchor": rel_anchor,
                })

            # ------------------------------------------------------
            # Independent colorbar for this part ROW
            # ------------------------------------------------------
            cax = fig.add_subplot(
                gs[j, E]
            )

            cb = fig.colorbar(
                last_im,
                cax=cax,
            )

            cb.ax.tick_params(
                labelsize=7
            )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.035,
            left=0.105,
            right=0.965,
        )

        obj_tag = base.safe_filename(
            f"{object_no:02d}_{obj_name}"
        )

        out_path = (
            out_dir
            / (
                f"{stem}_{obj_tag}_"
                f"{args.heatmap_mode}_"
                f"all_parts_epoch000_"
                f"{args.max_epoch:03d}.png"
            )
        )

        fig.savefig(
            out_path,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("saved:", out_path)

    # ==============================================================
    # CSV
    # ==============================================================
    csv_path = (
        out_dir
        / (
            f"{stem}_"
            f"{args.heatmap_mode}_"
            f"all_parts_epoch000_"
            f"{args.max_epoch:03d}.csv"
        )
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                csv_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print("figures:", out_dir)
    print("CSV    :", csv_path)


if __name__ == "__main__":
    main()
