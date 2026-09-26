import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = "nightly_ablation_pp116_seed123_20260921/analysis"

LINEAR_STRUCTURE_CSV = os.path.join(
    ROOT,
    "relative_linear_L8_gt_structure",
    "epoch_text_vs_gt_metrics.csv",
)

ORTH_STRUCTURE_CSV = os.path.join(
    ROOT,
    "relative_orthogonal_L8_gt_structure",
    "epoch_text_vs_gt_metrics.csv",
)

ALIGNMENT_CSV = os.path.join(
    ROOT,
    "structure_preservation_alignment",
    "structure_preservation_alignment.csv",
)

PROTO_QUALITY_CSV = os.path.join(
    ROOT,
    "relative_visual_prototype_quality_L8_controlled",
    "relative_visual_prototype_quality.csv",
)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# mIoU
# ============================================================

epochs = list(range(31))

linear_miou = [
    21.70, 25.17, 28.31, 30.33, 31.03, 31.35, 31.43, 31.38,
    31.23, 31.03, 30.86, 30.73, 30.60, 30.52, 30.47, 30.43,
    30.39, 30.34, 30.30, 30.25, 30.24, 30.18, 30.17, 30.14,
    30.10, 30.07, 30.04, 29.99, 29.94, 29.92, 29.88
]

orthogonal_miou = [
    21.70, 23.48, 25.20, 26.79, 28.34, 29.71, 30.59, 31.12,
    31.49, 31.73, 31.91, 32.06, 32.17, 32.27, 32.35, 32.42,
    32.48, 32.52, 32.57, 32.60, 32.62, 32.63, 32.64, 32.65,
    32.64, 32.64, 32.63, 32.63, 32.62, 32.61, 32.60
]


# ============================================================
# Load CSVs
# ============================================================

linear_struct = pd.read_csv(LINEAR_STRUCTURE_CSV)
orth_struct = pd.read_csv(ORTH_STRUCTURE_CSV)

alignment = pd.read_csv(ALIGNMENT_CSV)
proto_quality = pd.read_csv(PROTO_QUALITY_CSV)


# ============================================================
# Helper: tolerate slightly different column names
# ============================================================

def find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    lower = {c.lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]

    for name in candidates:
        for col in df.columns:
            if name.lower() in col.lower():
                return col

    raise KeyError(
        f"Cannot find any of {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# Resolve columns
# ============================================================

# Structure retention
linear_retention = linear_struct["retention_spearman_macro"].values
orth_retention = orth_struct["retention_spearman_macro"].values


# Relative + Linear vs Relative + Orthogonal
align_epoch = find_col(alignment, ["epoch"])

linear_gt_col = find_col(
    alignment,
    [
        "linear_proto_gt_cosine",
        "linear_proto_gt",
    ],
)

orth_gt_col = find_col(
    alignment,
    [
        "orthogonal_proto_gt_cosine",
        "ortho_proto_gt_cosine",
        "orthogonal_proto_gt",
    ],
)


# Absolute + Orthogonal vs Relative + Orthogonal
proto_epoch = find_col(proto_quality, ["epoch"])

abs_purity_col = find_col(
    proto_quality,
    [
        "absolute_support_purity",
        "abs_support_purity",
    ],
)

rel_purity_col = find_col(
    proto_quality,
    [
        "relative_support_purity",
        "rel_support_purity",
    ],
)

abs_gt_col = find_col(
    proto_quality,
    [
        "absolute_proto_gt_cosine",
        "abs_proto_gt_cosine",
    ],
)

rel_gt_col = find_col(
    proto_quality,
    [
        "relative_proto_gt_cosine",
        "rel_proto_gt_cosine",
    ],
)


# ============================================================
# Style
# ============================================================

blue = "#1f77b4"
orange = "#ff7f0e"

linear_label = "Relative + Linear"
orth_label = "Relative + Orthogonal"

abs_label = "Absolute + Orthogonal"
rel_label = "Relative + Orthogonal"


def style_axis(ax):
    ax.set_facecolor("white")
    ax.set_xlim(0, 30)
    ax.set_xticks(range(0, 31, 5))
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )
    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# ============================================================
# Figure 1:
# Structure Preservation and Part-Level Alignment
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.2, 3.8),
)

fig.patch.set_facecolor("white")


# ------------------------------------------------------------
# (a) mIoU
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    epochs,
    linear_miou,
    color=blue,
    linewidth=2.3,
    marker="o",
    markersize=4,
    markevery=2,
    label=linear_label,
)

ax.plot(
    epochs,
    orthogonal_miou,
    color=orange,
    linewidth=2.3,
    marker="s",
    markersize=4,
    markevery=2,
    label=orth_label,
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mIoU (%)", fontsize=12)
ax.set_ylim(20, 34)
ax.set_title("(a) mIoU", fontsize=12)

style_axis(ax)


# ------------------------------------------------------------
# (b) Structure Retention
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    linear_struct["epoch"],
    linear_retention,
    color=blue,
    linewidth=2.3,
    marker="o",
    markersize=4,
    markevery=2,
    label=linear_label,
)

ax.plot(
    orth_struct["epoch"],
    orth_retention,
    color=orange,
    linewidth=2.3,
    marker="s",
    markersize=4,
    markevery=2,
    label=orth_label,
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Structure Retention", fontsize=12)
ax.set_ylim(0.72, 1.02)
ax.set_title("(b) Structure Retention", fontsize=12)

style_axis(ax)


# ------------------------------------------------------------
# (c) GT Prototype Similarity
# ------------------------------------------------------------

ax = axes[2]

ax.plot(
    alignment[align_epoch],
    alignment[linear_gt_col],
    color=blue,
    linewidth=2.3,
    marker="o",
    markersize=4,
    markevery=2,
    label=linear_label,
)

ax.plot(
    alignment[align_epoch],
    alignment[orth_gt_col],
    color=orange,
    linewidth=2.3,
    marker="s",
    markersize=4,
    markevery=2,
    label=orth_label,
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("GT Prototype Similarity", fontsize=12)
ax.set_ylim(0.58, 0.75)
ax.set_title("(c) GT Prototype Similarity", fontsize=12)

style_axis(ax)


# Shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 1.02),
)

plt.tight_layout(
    rect=[0, 0, 1, 0.91]
)

out1_pdf = os.path.join(
    OUT_DIR,
    "structure_preservation_alignment.pdf",
)

out1_png = os.path.join(
    OUT_DIR,
    "structure_preservation_alignment.png",
)

plt.savefig(
    out1_pdf,
    bbox_inches="tight",
)

plt.savefig(
    out1_png,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2:
# Relative Visual Prototype Quality
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.8, 3.8),
)

fig.patch.set_facecolor("white")


# ------------------------------------------------------------
# (a) Support Purity
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    proto_quality[proto_epoch],
    proto_quality[abs_purity_col],
    color=blue,
    linewidth=2.3,
    marker="o",
    markersize=4,
    markevery=2,
    label=abs_label,
)

ax.plot(
    proto_quality[proto_epoch],
    proto_quality[rel_purity_col],
    color=orange,
    linewidth=2.3,
    marker="s",
    markersize=4,
    markevery=2,
    label=rel_label,
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Support Purity", fontsize=12)
ax.set_ylim(0.10, 0.40)
ax.set_title("(a) Support Purity", fontsize=12)

style_axis(ax)


# ------------------------------------------------------------
# (b) GT Prototype Similarity
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    proto_quality[proto_epoch],
    proto_quality[abs_gt_col],
    color=blue,
    linewidth=2.3,
    marker="o",
    markersize=4,
    markevery=2,
    label=abs_label,
)

ax.plot(
    proto_quality[proto_epoch],
    proto_quality[rel_gt_col],
    color=orange,
    linewidth=2.3,
    marker="s",
    markersize=4,
    markevery=2,
    label=rel_label,
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("GT Prototype Similarity", fontsize=12)
ax.set_ylim(0.57, 0.75)
ax.set_title("(b) GT Prototype Similarity", fontsize=12)

style_axis(ax)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 1.02),
)

plt.tight_layout(
    rect=[0, 0, 1, 0.91]
)

out2_pdf = os.path.join(
    OUT_DIR,
    "relative_visual_prototype_quality.pdf",
)

out2_png = os.path.join(
    OUT_DIR,
    "relative_visual_prototype_quality.png",
)

plt.savefig(
    out2_pdf,
    bbox_inches="tight",
)

plt.savefig(
    out2_png,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("Saved:")
print(out1_pdf)
print(out1_png)
print(out2_pdf)
print(out2_png)