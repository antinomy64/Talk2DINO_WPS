#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the clean Pascal-Part-116 evaluator before running Talk2DINO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=".")
    p.add_argument("--expected_val_images", type=int, default=850)
    args = p.parse_args()

    root = Path(args.repo_root).expanduser().resolve()
    ovseg = root / "src" / "open_vocabulary_segmentation"
    if not ovseg.is_dir():
        raise FileNotFoundError(ovseg)
    sys.path.insert(0, str(ovseg))

    import mmcv
    from mmseg.datasets import build_dataset

    cfg_path = (
        ovseg
        / "segmentation"
        / "configs"
        / "_base_"
        / "datasets"
        / "pascalpart116_part.py"
    )
    cfg = mmcv.Config.fromfile(str(cfg_path))
    dataset = build_dataset(cfg.data.test)

    classes = tuple(dataset.CLASSES)

    print("============================================================")
    print("Pascal-Part-116 clean evaluator audit")
    print("============================================================")
    print("dataset class        :", type(dataset).__name__)
    print("dataset length       :", len(dataset))
    print("num classes          :", len(classes))
    print("class[0]             :", classes[0])
    print("class[-1]            :", classes[-1])
    print("ignore_index         :", dataset.ignore_index)
    print("reduce_zero_label    :", dataset.reduce_zero_label)
    print("Talk2DINO with_bg    :", classes[0] == "background")
    print("test_cfg             :", cfg.test_cfg)

    assert len(dataset) == args.expected_val_images, (
        len(dataset), args.expected_val_images
    )
    assert len(classes) == 116
    assert classes[0] == "aeroplane's body"
    assert classes[-1] == "tvmonitor's screen"
    assert classes[0] != "background"
    assert dataset.ignore_index == 255
    assert dataset.reduce_zero_label is False

    ann_dir = root / "data" / "PascalPart116" / "annotations_detectron2_part" / "val"
    if not ann_dir.is_dir():
        raise FileNotFoundError(ann_dir)

    pngs = sorted(ann_dir.glob("*.png"))
    assert len(pngs) == args.expected_val_images, (
        f"val mask count={len(pngs)} expected={args.expected_val_images}"
    )

    all_ids = set()
    bad_files = []
    count_255 = 0
    count_zero = 0

    for path in pngs:
        with Image.open(path) as im:
            arr = np.asarray(im)
        if arr.ndim == 3:
            if (
                arr.shape[2] >= 3
                and np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 0], arr[..., 2])
            ):
                arr = arr[..., 0]
            else:
                bad_files.append((path.name, "color/non-ID mask"))
                continue

        ids = set(int(x) for x in np.unique(arr).tolist())
        all_ids.update(ids)
        invalid = sorted(x for x in ids if x != 255 and not (0 <= x <= 115))
        if invalid:
            bad_files.append((path.name, invalid))
        if 255 in ids:
            count_255 += 1
        if 0 in ids:
            count_zero += 1

    if bad_files:
        raise AssertionError(f"invalid val masks, first={bad_files[:10]}")

    assert 255 in all_ids, "No 255 ignore/background pixels found in val masks"
    assert all(x == 255 or 0 <= x <= 115 for x in all_ids)

    print("val mask files       :", len(pngs))
    print("global mask IDs      :", sorted(all_ids))
    print("masks containing 255 :", count_255)
    print("masks containing 0   :", count_zero)
    print()
    print("SEMANTIC CONTRACT:")
    print("  0..115 -> evaluated semantic part classes")
    print("  255    -> ignored/background pixels")
    print("  no background prediction channel")
    print("  class 0 is NOT reduced or shifted")
    print("============================================================")
    print("VOC116_PART_EVAL_AUDIT_PASS")


if __name__ == "__main__":
    main()
