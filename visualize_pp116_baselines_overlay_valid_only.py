from pathlib import Path
import importlib.util
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visualize_pp116_final import (
    make_palette,
    colorize_label,
)

# ============================================================
# fixed paths
# ============================================================

ROOT = Path("/home/master/code/lyx/Talk2DINO_official_bg")

IMAGE_ROOT = ROOT / "data/PascalPart116/images/val"
GT_ROOT    = ROOT / "data/PascalPart116/annotations_detectron2_part/val"

RAW_ROOT = ROOT / "output/qualitative_pp116_baselines_raw"
OUT_ROOT = ROOT / "output/qualitative_pp116_baselines_overlay_valid_only"

STEMS = [
    "2010_001000",
    "2008_007548",
    "2009_000037",
    "2009_004497",
]

METHODS = {
    "simzss_part": RAW_ROOT / "simzss",
    "segclip_part": RAW_ROOT / "segclip",
    "tcl_part": RAW_ROOT / "tcl",
}



# ============================================================
# taxonomy + palette
# ============================================================

taxonomy_py = ROOT / "src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py"
spec = importlib.util.spec_from_file_location("pp116_taxonomy", taxonomy_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PART_CLASSES = list(mod.PART_CLASSES)
assert len(PART_CLASSES) == 116

PALETTE = make_palette(116)

def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMAGE_ROOT / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"image not found for {stem}")

def find_pred(pred_root, stem):
    direct = pred_root / f"{stem}.png"
    if direct.exists():
        return direct

    candidates = sorted(pred_root.rglob(f"{stem}.png"))
    if not candidates:
        raise FileNotFoundError(f"prediction not found for {stem} under {pred_root}")

    preferred = [
        p for p in candidates
        if ("e030" in str(p).lower()) or ("epoch30" in str(p).lower()) or ("epoch_030" in str(p).lower())
    ]
    if preferred:
        return preferred[0]

    return candidates[-1]

def overlay_valid_only(image_rgb, pred, valid):
    """
    EXACTLY the same visualization logic as visualize_pp116_final.py:

      valid pixel:
          completely replace RGB with predicted semantic color

      invalid pixel:
          keep original RGB unchanged

    No alpha blending.
    """

    pred_valid = pred.copy()
    pred_valid[~valid] = 255

    pred_rgb, _ = colorize_label(
        pred_valid,
        PALETTE,
        ignore_index=255,
    )

    overlay = image_rgb.copy()

    # transparency = 0 / 100% opaque semantic color
    overlay[valid] = pred_rgb[valid]

    return overlay


def fit_cell(img, width=480, height=360):
    src = Image.fromarray(img)
    src.thumbnail((width, height), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (width, height), "white")
    x = (width - src.width) // 2
    y = (height - src.height) // 2
    cell.paste(src, (x, y))
    return cell

def get_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

FONT = get_font(22)

# ============================================================
# output dirs
# ============================================================

OUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "input").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "gt_valid_mask").mkdir(parents=True, exist_ok=True)

for method in METHODS:
    (OUT_ROOT / method / "overlay").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / method / "pred_color").mkdir(parents=True, exist_ok=True)

# ============================================================
# generate
# ============================================================

rows = []

for stem in STEMS:
    image = np.asarray(Image.open(find_image(stem)).convert("RGB"))
    gt = np.asarray(Image.open(GT_ROOT / f"{stem}.png"))
    valid = (gt != 255) & (gt >= 0) & (gt < 116)

    assert image.shape[:2] == gt.shape, (stem, image.shape, gt.shape)

    # save input
    Image.fromarray(image).save(OUT_ROOT / "input" / f"{stem}.png")

    # save valid mask visualization
    valid_vis = image.copy()
    invalid = ~valid
    valid_vis[invalid] = (0.7 * valid_vis[invalid]).astype(np.uint8)
    Image.fromarray(valid_vis).save(OUT_ROOT / "gt_valid_mask" / f"{stem}.png")

    row = [image]

    for method, pred_root in METHODS.items():
        pred_path = find_pred(pred_root, stem)
        pred = np.asarray(Image.open(pred_path))
        pred = np.squeeze(pred)

        assert pred.ndim == 2, (method, stem, pred.shape)
        assert pred.shape == gt.shape, (method, stem, pred.shape, gt.shape)
        assert pred.min() >= 0 and pred.max() < 116, (method, stem, pred.min(), pred.max())

        pred = pred.astype(np.uint8)

        # pure color prediction
        pred_valid = pred.copy()
        pred_valid[~valid] = 255

        pred_color, _ = colorize_label(
            pred_valid,
            PALETTE,
            ignore_index=255,
        )

        Image.fromarray(pred_color).save(
            OUT_ROOT / method / "pred_color" / f"{stem}.png"
        )

        # overlay only on valid pixels
        overlay = overlay_valid_only(image, pred, valid)
        Image.fromarray(overlay).save(
            OUT_ROOT / method / "overlay" / f"{stem}.png"
        )

        row.append(overlay)
        print(method, stem, "<-", pred_path)

    rows.append(row)

# ============================================================
# build grid: 4 rows x 4 cols
# Input | SimZSS-Part | SegCLIP-Part | TCL-Part
# ============================================================

titles = ["Input", "SimZSS-Part", "SegCLIP-Part", "TCL-Part"]
cell_w, cell_h = 480, 360
title_h = 52

grid = Image.new(
    "RGB",
    (cell_w * len(titles), title_h + cell_h * len(rows)),
    "white"
)
draw = ImageDraw.Draw(grid)

for col, title in enumerate(titles):
    bbox = draw.textbbox((0, 0), title, font=FONT)
    tw = bbox[2] - bbox[0]
    draw.text(
        (col * cell_w + (cell_w - tw) // 2, 12),
        title,
        font=FONT,
        fill="black"
    )

for r, row in enumerate(rows):
    for c, img in enumerate(row):
        cell = fit_cell(img, cell_w, cell_h)
        grid.paste(cell, (c * cell_w, title_h + r * cell_h))

grid_path = OUT_ROOT / "qualitative_grid_4x4_overlay_valid_only.png"
grid.save(grid_path)

print("=" * 80)
print("DONE")
print("output dir:", OUT_ROOT)
print("grid      :", grid_path)
print("=" * 80)
