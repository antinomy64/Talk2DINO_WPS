#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import mmcv

from mmseg.datasets import build_dataset
from mmseg.datasets.builder import PIPELINES


# ---------------------------------------------------------
# Talk2DINO official custom transform
# copied from main.py registration logic
# ---------------------------------------------------------
@PIPELINES.register_module()
class FloatImage:

    def __call__(self, results):
        results["img"] = results["img"].astype(np.float32)
        return results

    def __repr__(self):
        return "FloatImage()"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        default="."
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    sys.path.insert(
        0,
        str(repo_root / "src" / "open_vocabulary_segmentation")
    )

    cfg_path = (
        repo_root
        / "src/open_vocabulary_segmentation/"
        "segmentation/configs/_base_/datasets/"
        "pascalpart116_part.py"
    )

    cfg = mmcv.Config.fromfile(str(cfg_path))

    dataset = build_dataset(cfg.data.test)


    print("=" * 60)
    print("Pascal-Part-116 clean evaluator audit")
    print("=" * 60)

    print(
        "dataset class       :",
        type(dataset).__name__
    )

    print(
        "dataset length      :",
        len(dataset)
    )

    print(
        "num classes         :",
        len(dataset.CLASSES)
    )

    print(
        "class[0]            :",
        dataset.CLASSES[0]
    )

    print(
        "class[-1]           :",
        dataset.CLASSES[-1]
    )

    print(
        "ignore_index        :",
        dataset.ignore_index
    )

    print(
        "reduce_zero_label   :",
        dataset.reduce_zero_label
    )


    assert len(dataset) == 850
    assert len(dataset.CLASSES) == 116
    assert dataset.CLASSES[0] != "background"
    assert dataset.ignore_index == 255
    assert dataset.reduce_zero_label is False


    print()
    print("SEMANTIC CONTRACT:")
    print("  0..115 -> evaluated semantic part classes")
    print("  255    -> ignored background")
    print("  no background prediction channel")
    print("  class 0 is not shifted")

    print("=" * 60)
    print("VOC116_PART_EVAL_AUDIT_PASS")


if __name__ == "__main__":
    main()
