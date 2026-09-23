import matplotlib.pyplot as plt

epochs = list(range(31))

# ============================================================
# mIoU
# ============================================================
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
# Spearman
# ============================================================
linear_spearman = [
    0.531388, 0.556641, 0.562278, 0.592991, 0.595411,
    0.608701, 0.611525, 0.612057, 0.611832, 0.606410,
    0.599774, 0.593113, 0.586197, 0.589091, 0.585590,
    0.583366, 0.576238, 0.570928, 0.568207, 0.558968,
    0.561898, 0.554499, 0.549231, 0.542448, 0.542827,
    0.540358, 0.536440, 0.533164, 0.534386, 0.524712,
    0.524936
]

orthogonal_spearman = [
    0.531388, 0.531388, 0.531447, 0.531447, 0.531388,
    0.531388, 0.531447, 0.531388, 0.531447, 0.531447,
    0.531447, 0.531388, 0.531388, 0.531388, 0.531388,
    0.531388, 0.531388, 0.531388, 0.531388, 0.531388,
    0.531388, 0.531388, 0.531388, 0.531388, 0.531388,
    0.531388, 0.531388, 0.531388, 0.531388, 0.531388,
    0.531388
]

# ============================================================
# Style
# ============================================================
blue = "#1f77b4"
orange = "#ff7f0e"

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
fig.patch.set_facecolor("white")

# ------------------------------------------------------------
# (a) mIoU
# ------------------------------------------------------------
ax = axes[0]
ax.set_facecolor("white")

ax.plot(
    epochs, linear_miou,
    color=blue, linewidth=2.4,
    marker="o", markersize=4, markevery=2,
    label="Linear"
)
ax.plot(
    epochs, orthogonal_miou,
    color=orange, linewidth=2.4,
    marker="s", markersize=4, markevery=2,
    label="Orthogonal"
)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("mIoU (%)", fontsize=13)
ax.set_xlim(0, 30)
ax.set_xticks(range(0, 31, 5))
ax.set_ylim(20, 34)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
ax.tick_params(axis="both", labelsize=11)
ax.set_title("(a) mIoU", fontsize=13)

# ------------------------------------------------------------
# (b) Spearman
# ------------------------------------------------------------
ax = axes[1]
ax.set_facecolor("white")

ax.plot(
    epochs, linear_spearman,
    color=blue, linewidth=2.4, linestyle="--",
    marker="^", markersize=4, markevery=2,
    label="Linear"
)
ax.plot(
    epochs, orthogonal_spearman,
    color=orange, linewidth=2.4, linestyle="--",
    marker="D", markersize=3.8, markevery=2,
    label="Orthogonal"
)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel(r"Spearman $\rho$", fontsize=13)
ax.set_xlim(0, 30)
ax.set_xticks(range(0, 31, 5))
ax.set_ylim(0.52, 0.62)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
ax.tick_params(axis="both", labelsize=11)
ax.set_title(r"(b) Spearman $\rho$", fontsize=13)

# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.5, 1.02)
)

# Spine style
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig("figure4_text_visual_structure.pdf", bbox_inches="tight")
plt.savefig("figure4_text_visual_structure.png", dpi=300, bbox_inches="tight")
plt.show()