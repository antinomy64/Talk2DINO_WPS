import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision.io import read_image


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------
sys.path.insert(0, "src/open_vocabulary_segmentation")

from models import build_model
from segmentation.datasets.pascalpart116_part import PART_CLASSES


IGNORE_INDEX = 255
NUM_CLASSES = 116


def parse_args():
    p = argparse.ArgumentParser(
        "Visualize final PP-116 Talk2DINO/RelProto segmentation"
    )

    p.add_argument("--input", required=True, help="RGB input image")
    p.add_argument("--gt", required=True, help="PP-116 part GT PNG")
    p.add_argument("--output", required=True, help="Output visualization PNG")

    p.add_argument(
        "--config",
        default=(
            "src/open_vocabulary_segmentation/configs/voc116_part/"
            "dinotext_voc116_part_vitb_mlp_infonce.yml"
        ),
    )

    p.add_argument(
        "--proj_name",
        required=True,
        help="Projector filename without .pth, same as model.proj_name in eval",
    )

    p.add_argument("--device", default="cuda:0")

    # Current final eval protocol.
    p.add_argument("--crop_size", type=int, default=448)
    p.add_argument("--stride", type=int, default=224)

    p.add_argument(
        "--template",
        default="sub_imagenet_template",
    )

    p.add_argument(
        "--pamr",
        action="store_true",
        help="Enable final PAMR. Default OFF, matching current paper protocol.",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.58,
        help="Overlay transparency",
    )

    return p.parse_args()


def make_palette(n=116):
    """
    Deterministic high-contrast palette.
    Same label always gets same color.
    """
    rng = np.random.RandomState(12345)

    palette = rng.randint(
        25, 235,
        size=(n, 3),
        dtype=np.uint8,
    )

    return palette


def colorize_label(label, palette, ignore_index=255):
    """
    label: H x W, values 0..115 or 255.
    returns:
        rgb: H x W x 3
        valid: H x W bool
    """
    valid = label != ignore_index

    rgb = np.zeros(
        (label.shape[0], label.shape[1], 3),
        dtype=np.uint8,
    )

    ids = np.unique(label[valid])

    for cid in ids:
        cid = int(cid)

        if 0 <= cid < len(palette):
            rgb[label == cid] = palette[cid]

    return rgb, valid


def sliding_positions(length, crop, stride):
    if length <= crop:
        return [0]

    pos = list(range(0, length - crop + 1, stride))

    last = length - crop

    if pos[-1] != last:
        pos.append(last)

    return pos


@torch.no_grad()
def infer_sliding(
    model,
    image,
    text_emb,
    classnames,
    crop_size,
    stride,
    apply_pamr,
):
    """
    image:
        [1, 3, H, W], float, 0..255

    returns:
        score: [1, C, H, W]
    """

    assert image.ndim == 4 and image.shape[0] == 1

    _, _, H, W = image.shape

    ys = sliding_positions(H, crop_size, stride)
    xs = sliding_positions(W, crop_size, stride)

    score_sum = torch.zeros(
        (1, NUM_CLASSES, H, W),
        device=image.device,
        dtype=torch.float32,
    )

    count = torch.zeros(
        (1, 1, H, W),
        device=image.device,
        dtype=torch.float32,
    )

    total = len(ys) * len(xs)
    cur = 0

    for y in ys:
        for x in xs:
            cur += 1

            y2 = min(y + crop_size, H)
            x2 = min(x + crop_size, W)

            crop = image[:, :, y:y2, x:x2]

            real_h = crop.shape[-2]
            real_w = crop.shape[-1]

            # Pad small images / border crops to 448x448.
            pad_h = crop_size - real_h
            pad_w = crop_size - real_w

            if pad_h > 0 or pad_w > 0:
                crop_input = F.pad(
                    crop,
                    (0, pad_w, 0, pad_h),
                    mode="constant",
                    value=0,
                )
            else:
                crop_input = crop

            print(
                f"\rInference crop {cur}/{total}",
                end="",
                flush=True,
            )

            masks, _ = model.generate_masks(
                crop_input,
                img_metas=None,
                text_emb=text_emb,
                classnames=classnames,
                apply_pamr=apply_pamr,
            )

            # [1, 116, crop_size, crop_size]
            if masks.shape[1] != NUM_CLASSES:
                raise RuntimeError(
                    f"Expected {NUM_CLASSES} channels, "
                    f"got {tuple(masks.shape)}"
                )

            masks = masks[:, :, :real_h, :real_w].float()

            score_sum[:, :, y:y2, x:x2] += masks
            count[:, :, y:y2, x:x2] += 1.0

    print()

    if (count == 0).any():
        raise RuntimeError("Sliding inference left uncovered pixels.")

    score = score_sum / count

    return score


def main():
    args = parse_args()

    device = torch.device(args.device)

    input_path = Path(args.input)
    gt_path = Path(args.gt)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load config exactly like Talk2DINO demo/eval model construction.
    # -----------------------------------------------------------------
    cfg = OmegaConf.load(args.config)

    cfg.model.proj_name = args.proj_name

    print("=" * 80)
    print("PP-116 FINAL SEGMENTATION VISUALIZATION")
    print("=" * 80)
    print("input       :", input_path)
    print("gt          :", gt_path)
    print("config      :", args.config)
    print("proj_name   :", args.proj_name)
    print("crop        :", args.crop_size)
    print("stride      :", args.stride)
    print("final PAMR  :", args.pamr)
    print("ignore GT   :", IGNORE_INDEX)
    print("=" * 80)

    model = build_model(cfg.model)
    model = model.to(device).eval()

    # Important: PP-116 does NOT add an extra background prediction class.
    if hasattr(model, "with_bg_clean"):
        model.with_bg_clean = False

    classnames = list(PART_CLASSES)

    if len(classnames) != NUM_CLASSES:
        raise RuntimeError(
            f"PART_CLASSES should contain 116 classes, got {len(classnames)}"
        )

    # -----------------------------------------------------------------
    # Build the exact PP-116 text representations.
    # -----------------------------------------------------------------
    with torch.no_grad():
        class_tokens = model.build_dataset_class_tokens(
            args.template,
            classnames,
        )

        text_emb = model.build_text_embedding(class_tokens)

    # -----------------------------------------------------------------
    # Load image.
    # torchvision read_image gives RGB CxHxW uint8.
    # Talk2DINO demo feeds float 0..255 into generate_masks.
    # -----------------------------------------------------------------
    image = read_image(str(input_path)).float()

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.shape[0] == 4:
        image = image[:3]

    H, W = image.shape[-2:]

    image_batch = image.unsqueeze(0).to(device)

    # -----------------------------------------------------------------
    # Final dense prediction.
    # -----------------------------------------------------------------
    score = infer_sliding(
        model=model,
        image=image_batch,
        text_emb=text_emb,
        classnames=classnames,
        crop_size=args.crop_size,
        stride=args.stride,
        apply_pamr=args.pamr,
    )

    pred = score.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # -----------------------------------------------------------------
    # GT
    # -----------------------------------------------------------------
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3:
        gt = gt[..., 0]

    gt = gt.astype(np.uint8)

    if gt.shape != (H, W):
        raise RuntimeError(
            f"Image/GT size mismatch:\n"
            f"  image = {(H, W)}\n"
            f"  gt    = {gt.shape}"
        )

    valid = gt != IGNORE_INDEX

    print(
        "valid pixels:",
        int(valid.sum()),
        "/",
        int(valid.size),
        f"({100.0 * valid.mean():.2f}%)",
    )

    # -----------------------------------------------------------------
    # IMPORTANT:
    # Ignore pixels are masked OUT only for visualization.
    #
    # prediction at valid pixels = normal model prediction
    # prediction at GT==255     = 255
    # -----------------------------------------------------------------
    pred_valid = pred.copy()
    pred_valid[~valid] = IGNORE_INDEX

    palette = make_palette(NUM_CLASSES)

    gt_rgb, _ = colorize_label(gt, palette)
    pred_rgb, _ = colorize_label(pred_valid, palette)

    image_np = (
        image.permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    # -----------------------------------------------------------------
    # Overlay: only GT-valid pixels are colored.
    # Ignore pixels remain original RGB.
    # -----------------------------------------------------------------
    overlay = image_np.astype(np.float32).copy()

    overlay[valid] = (
        (1.0 - args.alpha) * image_np[valid].astype(np.float32)
        + args.alpha * pred_rgb[valid].astype(np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------------
    # Save raw masks.
    # -----------------------------------------------------------------
    stem = output_path.with_suffix("")

    pred_mask_path = Path(str(stem) + "_pred_mask.png")
    gt_vis_path = Path(str(stem) + "_gt.png")
    pred_vis_path = Path(str(stem) + "_pred.png")
    overlay_path = Path(str(stem) + "_overlay.png")

    Image.fromarray(pred_valid).save(pred_mask_path)
    Image.fromarray(gt_rgb).save(gt_vis_path)
    Image.fromarray(pred_rgb).save(pred_vis_path)
    Image.fromarray(overlay).save(overlay_path)

    # -----------------------------------------------------------------
    # Four-panel visualization.
    # Ignore region shown as black in GT/Pred panels,
    # and untouched original image in overlay.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Input")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Part GT\n(ignore=255 masked)")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction\n(on valid GT pixels only)")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()

    # -----------------------------------------------------------------
    # Print classes actually present.
    # -----------------------------------------------------------------
    gt_ids = sorted(int(x) for x in np.unique(gt[valid]))
    pred_ids = sorted(int(x) for x in np.unique(pred_valid[valid]))

    print("\nGT classes:")
    for cid in gt_ids:
        print(f"  {cid:3d}: {classnames[cid]}")

    print("\nPredicted classes:")
    for cid in pred_ids:
        print(f"  {cid:3d}: {classnames[cid]}")

    print("\nSaved:")
    print("  summary   :", output_path)
    print("  pred mask :", pred_mask_path)
    print("  GT color  :", gt_vis_path)
    print("  pred color:", pred_vis_path)
    print("  overlay   :", overlay_path)


if __name__ == "__main__":
    main()
