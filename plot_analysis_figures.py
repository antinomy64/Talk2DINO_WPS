import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ANALYSIS_ROOT = (
    "nightly_ablation_pp116_seed123_20260921/analysis"
)

ALIGN_CSV = os.path.join(
    ANALYSIS_ROOT,
    "structure_preservation_alignment",
    "structure_preservation_alignment.csv",
)

PROTO_CSV = os.path.join(
    ANALYSIS_ROOT,
    "relative_visual_prototype_quality_L8_controlled",
    "relative_visual_prototype_quality.csv",
)

LINEAR_METRIC_CSV = "epoch_text_vs_gt_metrics_linear.csv"
ORTH_METRIC_CSV = "epoch_text_vs_gt_metrics_ortho.csv"

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def find_col(df, candidates):
    """Find a column by exact match first, then substring match."""
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        candidate = candidate.lower()
        for col in df.columns:
            if candidate in col.lower():
                return col

    raise KeyError(
        f"Cannot find any of {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find: {path}")
    df = pd.read_csv(path)
    print(f"\n===== {path} =====")
    print(df.columns.tolist())
    return df


# ============================================================
# Load data
# ============================================================

align = load_csv(ALIGN_CSV)
proto = load_csv(PROTO_CSV)
linear = load_csv(LINEAR_METRIC_CSV)
ortho = load_csv(ORTH_METRIC_CSV)


# ============================================================
# Resolve columns
# ============================================================

# ----- alignment CSV -----
align_epoch = find_col(align, ["epoch"])

linear_gt = find_col(
    align,
    [
        "linear_proto_gt_cosine",
        "linear_proto_gt",
        "linear_gt_prototype_similarity",
    ],
)

orth_gt = find_col(
    align,
    [
        "orthogonal_proto_gt_cosine",
        "orth_proto_gt_cosine",
        "orthogonal_proto_gt",
        "orth_gt_prototype_similarity",
    ],
)


# ----- epoch metric CSVs -----
linear_epoch = find_col(linear, ["epoch"])
ortho_epoch = find_col(ortho, ["epoch"])

linear_miou = find_col(
    linear,
    ["linear_miou", "miou"],
)

orth_miou = find_col(
    ortho,
    ["orthogonal_miou", "ortho_miou", "miou"],
)

linear_retention = find_col(
    linear,
    [
        "structure_retention",
        "text_structure_retention",
        "retention",
        "text_vs_initial_spearman",
    ],
)

orth_retention = find_col(
    ortho,
    [
        "structure_retention",
        "text_structure_retention",
        "retention",
        "text_vs_initial_spearman",
    ],
)


# ----- prototype-quality CSV -----
proto_epoch = find_col(proto, ["epoch"])

absolute_purity = find_col(
    proto,
    [
        "absolute_support_purity",
        "abs_support_purity",
    ],
)

relative_purity = find_col(
    proto,
    [
        "relative_support_purity",
        "rel_support_purity",
    ],
)

absolute_gt = find_col(
    proto,
    [
        "absolute_proto_gt_cosine",
        "abs_proto_gt_cosine",
        "absolute_proto_gt",
    ],
)

relative_gt = find_col(
    proto,
    [
        "relative_proto_gt_cosine",
        "rel_proto_gt_cosine",
        "relative_proto_gt",
    ],
)


# ============================================================
# Figure 1
# Structure Preservation and Part-Level Alignment
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(12.8, 3.5),
)

# (a) mIoU
ax = axes[0]

ax.plot(
    linear[linear_epoch],
    linear[linear_miou],
    linewidth=2,
    label="Relative + Linear",
)

ax.plot(
    ortho[ortho_epoch],
    ortho[orth_miou],
    linewidth=2,
    label="Relative + Orthogonal",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("mIoU")
ax.set_title("(a) mIoU")
ax.grid(alpha=0.25)


# (b) Structure Retention
ax = axes[1]

ax.plot(
    linear[linear_epoch],
    linear[linear_retention],
    linewidth=2,
    label="Relative + Linear",
)

ax.plot(
    ortho[ortho_epoch],
    ortho[orth_retention],
    linewidth=2,
    label="Relative + Orthogonal",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Structure Retention")
ax.set_title("(b) Structure Retention")
ax.grid(alpha=0.25)


# (c) GT Prototype Similarity
ax = axes[2]

ax.plot(
    align[align_epoch],
    align[linear_gt],
    linewidth=2,
    label="Relative + Linear",
)

ax.plot(
    align[align_epoch],
    align[orth_gt],
    linewidth=2,
    label="Relative + Orthogonal",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("GT Prototype Similarity")
ax.set_title("(c) GT Prototype Similarity")
ax.grid(alpha=0.25)


# One shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.tight_layout(rect=[0, 0, 1, 0.91])

fig.savefig(
    os.path.join(
        OUT_DIR,
        "structure_preservation_alignment.pdf",
    ),
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(
        OUT_DIR,
        "structure_preservation_alignment.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 2
# Relative Visual Prototype Quality
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(8.5, 3.5),
)

# (a) Support Purity
ax = axes[0]

ax.plot(
    proto[proto_epoch],
    proto[absolute_purity],
    linewidth=2,
    label="Absolute + Orthogonal",
)

ax.plot(
    proto[proto_epoch],
    proto[relative_purity],
    linewidth=2,
    label="Relative + Orthogonal",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Support Purity")
ax.set_title("(a) Support Purity")
ax.grid(alpha=0.25)


# (b) GT Prototype Similarity
ax = axes[1]

ax.plot(
    proto[proto_epoch],
    proto[absolute_gt],
    linewidth=2,
    label="Absolute + Orthogonal",
)

ax.plot(
    proto[proto_epoch],
    proto[relative_gt],
    linewidth=2,
    label="Relative + Orthogonal",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("GT Prototype Similarity")
ax.set_title("(b) GT Prototype Similarity")
ax.grid(alpha=0.25)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.tight_layout(rect=[0, 0, 1, 0.91])

fig.savefig(
    os.path.join(
        OUT_DIR,
        "relative_visual_prototype_quality.pdf",
    ),
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(
        OUT_DIR,
        "relative_visual_prototype_quality.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("\nDone.")
print(
    "Figure 1:",
    os.path.join(
        OUT_DIR,
        "structure_preservation_alignment.pdf",
    ),
)
print(
    "Figure 2:",
    os.path.join(
        OUT_DIR,
        "relative_visual_prototype_quality.pdf",
    ),
)