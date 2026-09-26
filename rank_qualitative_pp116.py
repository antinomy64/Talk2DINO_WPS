from pathlib import Path
from PIL import Image
import numpy as np
import csv

ROOT = Path("output/qualitative_pp116_850")

GT_DIR   = ROOT / "gt_raw"
TALK_DIR = ROOT / "talk2dino_part" / "pred_mask_raw"
OURS_DIR = ROOT / "ours" / "pred_mask_raw"

IGNORE = 255


def image_miou(pred, gt):
    valid = gt != IGNORE
    gt = gt[valid]
    pred = pred[valid]

    classes = np.unique(gt)

    ious = []

    for c in classes:
        g = gt == c
        p = pred == c

        inter = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()

        if union > 0:
            ious.append(inter / union)

    if not ious:
        return 0.0, 0

    return float(np.mean(ious)), len(classes)


rows = []

for gt_path in sorted(GT_DIR.glob("*.png")):
    name = gt_path.name

    talk_path = TALK_DIR / name
    ours_path = OURS_DIR / name

    if not talk_path.exists() or not ours_path.exists():
        continue

    gt = np.array(Image.open(gt_path))
    talk = np.array(Image.open(talk_path))
    ours = np.array(Image.open(ours_path))

    valid = gt != IGNORE

    talk_miou, n_cls = image_miou(talk, gt)
    ours_miou, _ = image_miou(ours, gt)

    # pixel accuracy just as an auxiliary signal
    talk_acc = float((talk[valid] == gt[valid]).mean())
    ours_acc = float((ours[valid] == gt[valid]).mean())

    rows.append({
        "image": gt_path.stem,
        "num_gt_parts": n_cls,
        "valid_pixels": int(valid.sum()),
        "talk_miou": talk_miou,
        "ours_miou": ours_miou,
        "delta_miou": ours_miou - talk_miou,
        "talk_acc": talk_acc,
        "ours_acc": ours_acc,
        "delta_acc": ours_acc - talk_acc,
    })


# Prefer:
# 1. Ours clearly better
# 2. >= 3 semantic parts
# 3. enough valid pixels for a readable qualitative figure
filtered = [
    r for r in rows
    if r["num_gt_parts"] >= 3
    and r["valid_pixels"] >= 3000
    and r["ours_miou"] > r["talk_miou"]
]

filtered.sort(
    key=lambda x: (
        x["delta_miou"],
        x["ours_miou"],
        x["delta_acc"],
    ),
    reverse=True
)

out_csv = ROOT / "qualitative_candidates.csv"

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
    writer.writeheader()
    writer.writerows(filtered)

print()
print("Top qualitative candidates")
print("=" * 100)

for i, r in enumerate(filtered[:30], 1):
    print(
        f"{i:02d}. {r['image']}  "
        f"parts={r['num_gt_parts']:2d}  "
        f"Talk={r['talk_miou']:.3f}  "
        f"Ours={r['ours_miou']:.3f}  "
        f"ΔIoU={r['delta_miou']:+.3f}  "
        f"ΔAcc={r['delta_acc']:+.3f}"
    )

print()
print("Saved:", out_csv)
