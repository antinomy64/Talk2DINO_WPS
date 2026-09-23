#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


NUM_PARTS = 116
DINO_DIM = 768
DEFAULT_PATCH_SIZE = 14


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Pascal-Part-116 GT visual prototypes by globally "
            "pooling DINOv2 patch tokens according to GT part masks."
        )
    )

    parser.add_argument(
        "--image_root",
        default="data/PascalPart116/images/train",
    )

    parser.add_argument(
        "--part_mask_root",
        default="data/PascalPart116/annotations_detectron2_part/train",
    )

    parser.add_argument(
        "--classes_source",
        default=(
            "src/open_vocabulary_segmentation/segmentation/datasets/"
            "pascalpart116_part.py:PART_CLASSES"
        ),
    )

    parser.add_argument(
        "--model_name",
        default="dinov2_vitb14_reg",
    )

    parser.add_argument(
        "--dinov2_ckpt",
        default=(
            "/home/master/.cache/torch/hub/checkpoints/"
            "dinov2_vitb14_reg4_pretrain.pth"
        ),
    )

    parser.add_argument(
        "--dinov2_repo",
        default=None,
        help=(
            "Local DINOv2 torch-hub repo. "
            "If omitted, auto-detect under ~/.cache/torch/hub."
        ),
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--ignore_label",
        type=int,
        default=255,
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 = use all images. Useful for smoke testing.",
    )

    parser.add_argument(
        "--output",
        default=(
            "feature/pascalpart116_gt_visual_structure/"
            "gt_dinov2_vitb14reg_train_fullimg_patchpool.pt"
        ),
    )

    return parser.parse_args()


# ============================================================
# Class names
# ============================================================

def load_classes(source):
    path, variable = source.split(":", 1)

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "pp116_classes",
        str(path),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classes = list(getattr(module, variable))

    if len(classes) != NUM_PARTS:
        raise RuntimeError(
            f"Expected {NUM_PARTS} classes, got {len(classes)}"
        )

    return classes


# ============================================================
# DINOv2
# ============================================================

def find_local_dinov2_repo(user_path=None):

    if user_path is not None:
        repo = Path(user_path).expanduser().resolve()

        if not repo.exists():
            raise FileNotFoundError(repo)

        if not (repo / "hubconf.py").exists():
            raise RuntimeError(
                f"{repo} does not contain hubconf.py"
            )

        return repo

    hub_root = Path.home() / ".cache" / "torch" / "hub"

    candidates = sorted(
        p
        for p in hub_root.glob("facebookresearch_dinov2*")
        if p.is_dir() and (p / "hubconf.py").exists()
    )

    if not candidates:
        raise RuntimeError(
            "Cannot find local DINOv2 repo under "
            "~/.cache/torch/hub.\n"
            "Please pass --dinov2_repo explicitly."
        )

    return candidates[0]


def unwrap_checkpoint(obj):

    if not isinstance(obj, dict):
        return obj

    # Most official DINO checkpoints are already state_dict.
    tensor_count = sum(
        torch.is_tensor(v)
        for v in obj.values()
    )

    if tensor_count > 0 and tensor_count >= len(obj) * 0.5:
        return obj

    for key in [
        "model",
        "state_dict",
        "teacher",
        "student",
        "backbone",
    ]:
        if key in obj and isinstance(obj[key], dict):
            return unwrap_checkpoint(obj[key])

    return obj


def clean_state_dict(state):

    cleaned = {}

    for key, value in state.items():

        if not torch.is_tensor(value):
            continue

        new_key = key

        prefixes = [
            "module.",
            "model.",
            "backbone.",
            "teacher.",
            "student.",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_dinov2(args):

    repo = find_local_dinov2_repo(
        args.dinov2_repo
    )

    print("[INFO] DINOv2 repo:")
    print("      ", repo)

    print("[INFO] Build model:")
    print("      ", args.model_name)

    model = torch.hub.load(
        str(repo),
        args.model_name,
        source="local",
        pretrained=False,
    )

    ckpt_path = Path(
        args.dinov2_ckpt
    ).expanduser()

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print("[INFO] Load checkpoint:")
    print("      ", ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    state = clean_state_dict(
        unwrap_checkpoint(checkpoint)
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        print(
            f"[WARN] Missing model keys: {len(missing)}"
        )
        for key in missing[:20]:
            print("       ", key)

    if unexpected:
        print(
            f"[WARN] Unexpected checkpoint keys: "
            f"{len(unexpected)}"
        )
        for key in unexpected[:20]:
            print("       ", key)

    model.eval()
    model.to(args.device)

    return model


# ============================================================
# Dataset pairing
# ============================================================

def collect_files(root, extensions):

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(root)

    result = {}

    for path in root.rglob("*"):

        if path.suffix.lower() not in extensions:
            continue

        stem = path.stem

        if stem in result:
            raise RuntimeError(
                f"Duplicate stem '{stem}':\n"
                f"  {result[stem]}\n"
                f"  {path}"
            )

        result[stem] = path

    return result


def collect_pairs(image_root, mask_root):

    images = collect_files(
        image_root,
        {".jpg", ".jpeg", ".png", ".bmp"},
    )

    masks = collect_files(
        mask_root,
        {".png", ".tif", ".tiff"},
    )

    common = sorted(
        set(images.keys())
        & set(masks.keys())
    )

    if not common:
        raise RuntimeError(
            "No paired image/mask files found."
        )

    missing_masks = sorted(
        set(images.keys())
        - set(masks.keys())
    )

    print(
        f"[INFO] images       : {len(images)}"
    )
    print(
        f"[INFO] masks        : {len(masks)}"
    )
    print(
        f"[INFO] paired images: {len(common)}"
    )
    print(
        f"[INFO] no-mask imgs : {len(missing_masks)}"
    )

    return [
        (
            images[name],
            masks[name],
            name,
        )
        for name in common
    ]


# ============================================================
# Image / mask preprocessing
# ============================================================

def preprocess_image(image, input_size):
    """
    Full image only.

    No object crop.
    No center crop.

    The whole image is resized directly to input_size x input_size.
    """

    image = TF.resize(
        image,
        [input_size, input_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    image = TF.to_tensor(image)

    image = TF.normalize(
        image,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    return image


def mask_to_patch_grid(
    mask_image,
    grid_h,
    grid_w,
):
    """
    Direct GT-mask -> DINO patch grid.

    Example:
        original GT mask
              |
              | nearest-neighbor
              v
           32 x 32
              |
              v
       one GT part ID for
       each DINO patch token

    NO majority voting.
    NO >50% threshold.
    NO boundary filtering.
    """

    mask_np = np.asarray(mask_image)

    if mask_np.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D semantic mask, "
            f"got shape {mask_np.shape}"
        )

    # Important:
    # use raw integer mask IDs, not PIL RGB/grayscale conversion.
    mask_tensor = torch.from_numpy(
        mask_np.astype(
            np.int64,
            copy=True,
        )
    )

    patch_grid = F.interpolate(
        mask_tensor[
            None,
            None,
        ].float(),
        size=(grid_h, grid_w),
        mode="nearest",
    )[0, 0].long()

    return patch_grid


# ============================================================
# DINO patch tokens
# ============================================================

@torch.no_grad()
def extract_patch_tokens(
    model,
    image_batch,
):
    """
    ViT-B/14-Reg @ 448:
        [B, 1024, 768]
    """

    features = model.forward_features(
        image_batch
    )

    if not isinstance(features, dict):
        raise RuntimeError(
            "DINOv2 forward_features() "
            "did not return a dict."
        )

    if "x_norm_patchtokens" not in features:
        raise RuntimeError(
            "Missing x_norm_patchtokens.\n"
            f"Available keys: "
            f"{list(features.keys())}"
        )

    tokens = features[
        "x_norm_patchtokens"
    ]

    return tokens.float()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.input_size % args.patch_size != 0:
        raise ValueError(
            f"input_size={args.input_size} "
            f"is not divisible by "
            f"patch_size={args.patch_size}"
        )

    grid_h = (
        args.input_size
        // args.patch_size
    )

    grid_w = grid_h

    expected_tokens = (
        grid_h * grid_w
    )

    print("=" * 80)
    print(
        "PP-116 GT GLOBAL PATCH-TOKEN POOL"
    )
    print("=" * 80)

    print(
        f"input size   : "
        f"{args.input_size} x {args.input_size}"
    )
    print(
        f"patch size   : {args.patch_size}"
    )
    print(
        f"patch grid   : "
        f"{grid_h} x {grid_w}"
    )
    print(
        f"tokens/image : {expected_tokens}"
    )
    print(
        f"parts        : {NUM_PARTS}"
    )

    print()
    print("Protocol:")
    print("  full image")
    print("  GT part mask")
    print("  GT mask nearest -> patch grid")
    print("  one GT ID per patch token")
    print("  all selected patch tokens have weight 1")
    print("  no object crop")
    print("  no per-image mean")
    print("  no image balancing")
    print("  no top-k")
    print("  no majority voting")
    print("  no boundary filtering")
    print()

    classes = load_classes(
        args.classes_source
    )

    pairs = collect_pairs(
        args.image_root,
        args.part_mask_root,
    )

    if args.max_images > 0:
        pairs = pairs[
            :args.max_images
        ]

        print(
            f"[INFO] max_images active: "
            f"{len(pairs)}"
        )

    model = load_dinov2(args)

    # --------------------------------------------------------
    # Global accumulators
    #
    # sum_feat[j]:
    #     sum of ALL patch tokens assigned to part j
    #
    # patch_count[j]:
    #     actual number of patch tokens in pool j
    #
    # No per-image prototype is ever computed.
    # --------------------------------------------------------

    feature_sum = torch.zeros(
        NUM_PARTS,
        DINO_DIM,
        dtype=torch.float64,
    )

    patch_count = torch.zeros(
        NUM_PARTS,
        dtype=torch.long,
    )

    # This is diagnostic only.
    # It does NOT participate in prototype averaging.
    image_count = torch.zeros(
        NUM_PARTS,
        dtype=torch.long,
    )

    invalid_grid_count = 0
    total_grid_count = 0

    batch_images = []
    batch_patch_grids = []
    batch_names = []

    def process_batch():

        nonlocal batch_images
        nonlocal batch_patch_grids
        nonlocal batch_names

        nonlocal feature_sum
        nonlocal patch_count
        nonlocal image_count

        nonlocal invalid_grid_count
        nonlocal total_grid_count

        if len(batch_images) == 0:
            return

        images = torch.stack(
            batch_images,
            dim=0,
        ).to(
            args.device,
            non_blocking=True,
        )

        tokens = extract_patch_tokens(
            model,
            images,
        ).cpu()

        B, N, D = tokens.shape

        if N != expected_tokens:
            raise RuntimeError(
                f"Unexpected number of tokens: "
                f"{N}, expected {expected_tokens}"
            )

        if D != DINO_DIM:
            raise RuntimeError(
                f"Unexpected DINO dimension: "
                f"{D}, expected {DINO_DIM}"
            )

        for b in range(B):

            patch_ids = (
                batch_patch_grids[b]
                .reshape(-1)
            )

            if patch_ids.numel() != N:
                raise RuntimeError(
                    f"Patch-grid/token mismatch "
                    f"for {batch_names[b]}: "
                    f"{patch_ids.numel()} vs {N}"
                )

            valid = (
                (patch_ids >= 0)
                & (patch_ids < NUM_PARTS)
            )

            total_grid_count += N
            invalid_grid_count += int(
                (~valid).sum()
            )

            valid_part_ids = torch.unique(
                patch_ids[valid]
            )

            for part_id_tensor in valid_part_ids:

                part_id = int(
                    part_id_tensor
                )

                selected = (
                    patch_ids == part_id
                )

                part_tokens = tokens[
                    b,
                    selected,
                    :,
                ]

                n = part_tokens.shape[0]

                if n == 0:
                    continue

                # =================================================
                # THIS IS THE CORE DEFINITION.
                #
                # Do NOT average inside this image.
                #
                # Every patch token goes directly into
                # the global part pool.
                # =================================================

                feature_sum[
                    part_id
                ] += (
                    part_tokens
                    .double()
                    .sum(dim=0)
                )

                patch_count[
                    part_id
                ] += n

                # Diagnostic only
                image_count[
                    part_id
                ] += 1

        batch_images = []
        batch_patch_grids = []
        batch_names = []

    # ========================================================
    # Dataset loop
    # ========================================================

    for (
        image_path,
        mask_path,
        image_name,
    ) in tqdm(
        pairs,
        desc="Pooling GT patch tokens",
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        )

        image_tensor = preprocess_image(
            image,
            args.input_size,
        )

        patch_grid = mask_to_patch_grid(
            mask,
            grid_h,
            grid_w,
        )

        batch_images.append(
            image_tensor
        )

        batch_patch_grids.append(
            patch_grid
        )

        batch_names.append(
            image_name
        )

        if (
            len(batch_images)
            >= args.batch_size
        ):
            process_batch()

    process_batch()

    # ========================================================
    # Final global mean
    # ========================================================

    valid_parts = (
        patch_count > 0
    )

    prototypes_raw = torch.zeros(
        NUM_PARTS,
        DINO_DIM,
        dtype=torch.float32,
    )

    prototypes_raw[
        valid_parts
    ] = (
        feature_sum[
            valid_parts
        ]
        /
        patch_count[
            valid_parts
        ]
        .double()
        .unsqueeze(1)
    ).float()

    # For cosine/Spearman downstream.
    prototypes = torch.zeros_like(
        prototypes_raw
    )

    prototypes[
        valid_parts
    ] = F.normalize(
        prototypes_raw[
            valid_parts
        ],
        dim=-1,
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print()
    print("=" * 80)
    print("PART POOL STATISTICS")
    print("=" * 80)

    for part_id, class_name in enumerate(
        classes
    ):

        print(
            f"{part_id:3d} | "
            f"{class_name:30s} | "
            f"patches={int(patch_count[part_id]):8d} | "
            f"images={int(image_count[part_id]):5d}"
        )

    missing_parts = torch.where(
        ~valid_parts
    )[0].tolist()

    if missing_parts:
        print()
        print(
            "[WARN] Parts with zero GT patch tokens:"
        )

        for part_id in missing_parts:
            print(
                f"  {part_id:3d}: "
                f"{classes[part_id]}"
            )

    print()
    print(
        "Total valid part patch tokens:",
        int(patch_count.sum()),
    )

    print(
        "Total patch-grid positions:",
        total_grid_count,
    )

    print(
        "Ignored/background grid positions:",
        invalid_grid_count,
    )

    if valid_parts.any():

        counts = patch_count[
            valid_parts
        ].float()

        print(
            "Per-part patch count "
            "min / mean / max:",
            int(counts.min()),
            f"{counts.mean().item():.2f}",
            int(counts.max()),
        )

    # ========================================================
    # Save
    # ========================================================

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        # Main GT visual prototype used downstream
        "prototypes":
            prototypes.cpu(),

        # Before final L2 normalization
        "prototypes_raw":
            prototypes_raw.cpu(),

        # Actual N_j in each pool
        "patch_counts":
            patch_count.cpu(),

        # Diagnostic only; NOT used as weights
        "image_counts":
            image_count.cpu(),

        "classes":
            classes,

        "metadata": {
            "dataset":
                "PascalPart-116",

            "split":
                "train",

            "visual_encoder":
                args.model_name,

            "dinov2_checkpoint":
                str(args.dinov2_ckpt),

            "embedding_dim":
                DINO_DIM,

            "input_size":
                args.input_size,

            "patch_size":
                args.patch_size,

            "patch_grid":
                [grid_h, grid_w],

            "num_parts":
                NUM_PARTS,

            "protocol": (
                "For every training image, resize the full image "
                "to the DINO input size. Directly resize the GT "
                "part-ID mask to the DINO patch grid using nearest "
                "neighbor. Assign each DINO patch token to the part "
                "ID at the corresponding GT grid position. Pool all "
                "tokens from all training images globally for each "
                "part. Compute exactly one prototype per part as "
                "sum(tokens) / actual number of tokens in that pool."
            ),

            "object_crop":
                False,

            "per_image_mean":
                False,

            "image_balanced":
                False,

            "majority_voting":
                False,

            "boundary_filter":
                False,

            "topk":
                False,

            "token_weight":
                "1 for every valid patch token",

            "final_normalization":
                "L2",
        },
    }

    torch.save(
        result,
        output,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Saved:",
        output,
    )

    print(
        "GT prototype shape:",
        tuple(
            prototypes.shape
        ),
    )


if __name__ == "__main__":
    main()