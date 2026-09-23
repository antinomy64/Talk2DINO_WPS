import argparse
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14_reg")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draw_grid", action="store_true")
    return parser.parse_args()


def find_local_dino_repo():
    candidates = sorted(glob.glob("/home/master/.cache/torch/hub/facebookresearch_dinov2_*"))
    if not candidates:
        raise FileNotFoundError("Cannot find local DINOv2 repo under /home/master/.cache/torch/hub/")
    return candidates[0]


def add_grid_pil(img, patch_size, color=(255, 255, 255), width=1):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w + 1, patch_size):
        draw.line((x, 0, x, h), fill=color, width=width)
    for y in range(0, h + 1, patch_size):
        draw.line((0, y, w, y), fill=color, width=width)
    return img


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    print("[1/5] Loading DINOv2 from local cache...", flush=True)
    dino_repo = find_local_dino_repo()
    dino_ckpt = "/home/master/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"

    model = torch.hub.load(
        dino_repo,
        "dinov2_vitb14_reg",
        source="local",
        pretrained=False,
    )

    state_dict = torch.load(dino_ckpt, map_location="cpu")
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("[1/5] DINOv2 loaded.", flush=True)

    print("[2/5] Loading image...", flush=True)
    img = Image.open(args.image).convert("RGB")
    img_vis = img.resize((args.input_size, args.input_size), Image.Resampling.BICUBIC)

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    x = transform(img).unsqueeze(0).to(device)

    print("[3/5] Extracting patch tokens...", flush=True)
    feat_dict = model.forward_features(x)
    patch_tokens = feat_dict["x_norm_patchtokens"][0].float()

    n = args.input_size // args.patch_size
    assert patch_tokens.shape[0] == n * n, f"unexpected token count: {patch_tokens.shape[0]} vs {n*n}"

    print("[4/5] Running PCA...", flush=True)
    feat_centered = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(feat_centered, q=3, center=False)
    pca = feat_centered @ V[:, :3]
    pca = pca.reshape(n, n, 3)

    pca_rgb = torch.zeros_like(pca)
    for c in range(3):
        ch = pca[..., c]
        pca_rgb[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    pca_tensor = pca_rgb.permute(2, 0, 1).unsqueeze(0)
    pca_up = F.interpolate(
        pca_tensor,
        size=(args.input_size, args.input_size),
        mode="nearest"
    )[0].permute(1, 2, 0).cpu().numpy()

    token_img = (pca_up * 255.0).clip(0, 255).astype(np.uint8)
    token_img = Image.fromarray(token_img)

    overlay_img = Image.blend(img_vis, token_img, alpha=0.55)

    if args.draw_grid:
        img_vis = add_grid_pil(img_vis.copy(), args.patch_size, color=(255, 255, 0), width=1)
        token_img = add_grid_pil(token_img.copy(), args.patch_size, color=(255, 255, 255), width=1)
        overlay_img = add_grid_pil(overlay_img.copy(), args.patch_size, color=(255, 255, 255), width=1)

    print("[5/5] Saving images...", flush=True)
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_path = args.output_prefix + "_input.png"
    tokens_path = args.output_prefix + "_tokens.png"
    overlay_path = args.output_prefix + "_overlay.png"

    img_vis.save(input_path)
    token_img.save(tokens_path)
    overlay_img.save(overlay_path)

    print(f"Saved: {input_path}", flush=True)
    print(f"Saved: {tokens_path}", flush=True)
    print(f"Saved: {overlay_path}", flush=True)


if __name__ == "__main__":
    main()
