#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


NUM_PARTS = 116
SRC_VALID = set(range(NUM_PARTS)) | {255}


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))

    if arr.ndim != 2:
        raise RuntimeError(
            f"{path}: expected single-channel label mask, "
            f"got shape={arr.shape}"
        )

    return arr


def load_split(path: Path):
    stems = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        stems.append(Path(line).stem)

    if len(stems) != len(set(stems)):
        raise RuntimeError("duplicate entries found in split file")

    return stems


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-ann",
        required=True,
        type=Path,
        help="Original PascalPart116 annotations_detectron2_part/val"
    )

    parser.add_argument(
        "--split",
        required=True,
        type=Path,
        help="PascalPart116 val.txt"
    )

    parser.add_argument(
        "--out-ann",
        required=True,
        type=Path,
        help="Output directory for 117-class masks"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    src_dir = args.src_ann.resolve()
    out_dir = args.out_ann.resolve()
    split_file = args.split.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)

    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    out_dir.mkdir(parents=True, exist_ok=True)

    stems = load_split(split_file)

    src_hist = np.zeros(256, dtype=np.int64)
    dst_hist = np.zeros(256, dtype=np.int64)

    print(f"source : {src_dir}")
    print(f"output : {out_dir}")
    print(f"split  : {split_file}")
    print(f"images : {len(stems)}")

    for idx, stem in enumerate(stems):
        src_path = src_dir / f"{stem}.png"
        dst_path = out_dir / f"{stem}.png"

        if not src_path.is_file():
            raise FileNotFoundError(src_path)

        if dst_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{dst_path} exists; use --overwrite"
            )

        src = load_mask(src_path)

        uniq = set(np.unique(src).tolist())
        invalid = uniq - SRC_VALID

        if invalid:
            raise RuntimeError(
                f"{src_path}: unexpected source labels "
                f"{sorted(invalid)}"
            )

        # --------------------------------------------------
        # Full-image PascalPart117 contract
        #
        # old:
        #   0..115 -> parts
        #   255    -> background / ignored region
        #
        # new:
        #   0      -> background
        #   1..116 -> parts
        #
        # No pixel is ignored.
        # --------------------------------------------------

        dst = np.zeros(src.shape, dtype=np.uint8)

        fg = src != 255

        dst[fg] = src[fg].astype(np.uint8) + 1

        # Hard safety check.
        dst_unique = set(np.unique(dst).tolist())

        invalid_dst = {
            x for x in dst_unique
            if not (0 <= x <= 116)
        }

        if invalid_dst:
            raise RuntimeError(
                f"{stem}: invalid target labels "
                f"{sorted(invalid_dst)}"
            )

        if np.any(dst == 255):
            raise RuntimeError(
                f"{stem}: target still contains 255"
            )

        src_hist += np.bincount(
            src.reshape(-1),
            minlength=256
        )

        dst_hist += np.bincount(
            dst.reshape(-1),
            minlength=256
        )

        Image.fromarray(dst).save(dst_path)

        if (idx + 1) % 100 == 0:
            print(
                f"[{idx + 1}/{len(stems)}] converted"
            )

    # --------------------------------------------------
    # Global exact-count invariants
    # --------------------------------------------------

    if src_hist[255] != dst_hist[0]:
        raise RuntimeError(
            "background pixel count mismatch: "
            f"source255={src_hist[255]} "
            f"target0={dst_hist[0]}"
        )

    for old_id in range(116):
        new_id = old_id + 1

        if src_hist[old_id] != dst_hist[new_id]:
            raise RuntimeError(
                f"class count mismatch: "
                f"src[{old_id}]={src_hist[old_id]} "
                f"dst[{new_id}]={dst_hist[new_id]}"
            )

    if dst_hist[255] != 0:
        raise RuntimeError(
            f"target contains {dst_hist[255]} ignored pixels"
        )

    print()
    print("=" * 70)
    print("PASCALPART117 CONVERSION PASS")
    print("=" * 70)

    print(
        f"background pixels : {dst_hist[0]:,}"
    )

    print(
        f"semantic pixels   : "
        f"{dst_hist[1:117].sum():,}"
    )

    print(
        f"ignored pixels    : {dst_hist[255]:,}"
    )

    print(
        f"total pixels      : "
        f"{dst_hist.sum():,}"
    )


if __name__ == "__main__":
    main()