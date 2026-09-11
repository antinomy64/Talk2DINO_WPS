#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pascal-Part-116 fine part semantic segmentation dataset for MMSeg 0.30.x.

LABEL CONTRACT (hard-locked):
    valid semantic parts : 0..115  (116 classes)
    background / ignore  : 255
    reduce_zero_label    : False

Class 0 ("aeroplane's body") is a REAL semantic class. It must never be
converted to ignore and all other labels must never be shifted.

This dataset intentionally contains NO "background" entry in CLASSES.
Talk2DINO's dinotext_builder.py therefore sets with_bg=False and creates
exactly 116 text/segmentation channels. GT pixels with value 255 are excluded
by MMSeg's ignore_index=255 when mIoU is computed.
"""

import os

from mmseg.datasets import DATASETS, CustomDataset


PART_CLASSES = (
    "aeroplane's body",
    "aeroplane's stern",
    "aeroplane's wing",
    "aeroplane's tail",
    "aeroplane's engine",
    "aeroplane's wheel",
    "bicycle's wheel",
    "bicycle's saddle",
    "bicycle's handlebar",
    "bicycle's chainwheel",
    "bicycle's headlight",
    "bird's wing",
    "bird's tail",
    "bird's head",
    "bird's eye",
    "bird's beak",
    "bird's torso",
    "bird's neck",
    "bird's leg",
    "bird's foot",
    "bottle's body",
    "bottle's cap",
    "bus's wheel",
    "bus's headlight",
    "bus's front",
    "bus's side",
    "bus's back",
    "bus's roof",
    "bus's mirror",
    "bus's license plate",
    "bus's door",
    "bus's window",
    "car's wheel",
    "car's headlight",
    "car's front",
    "car's side",
    "car's back",
    "car's roof",
    "car's mirror",
    "car's license plate",
    "car's door",
    "car's window",
    "cat's tail",
    "cat's head",
    "cat's eye",
    "cat's torso",
    "cat's neck",
    "cat's leg",
    "cat's nose",
    "cat's paw",
    "cat's ear",
    "cow's tail",
    "cow's head",
    "cow's eye",
    "cow's torso",
    "cow's neck",
    "cow's leg",
    "cow's ear",
    "cow's muzzle",
    "cow's horn",
    "dog's tail",
    "dog's head",
    "dog's eye",
    "dog's torso",
    "dog's neck",
    "dog's leg",
    "dog's nose",
    "dog's paw",
    "dog's ear",
    "dog's muzzle",
    "horse's tail",
    "horse's head",
    "horse's eye",
    "horse's torso",
    "horse's neck",
    "horse's leg",
    "horse's ear",
    "horse's muzzle",
    "horse's hoof",
    "motorbike's wheel",
    "motorbike's saddle",
    "motorbike's handlebar",
    "motorbike's headlight",
    "person's head",
    "person's eye",
    "person's torso",
    "person's neck",
    "person's leg",
    "person's foot",
    "person's nose",
    "person's ear",
    "person's eyebrow",
    "person's mouth",
    "person's hair",
    "person's lower arm",
    "person's upper arm",
    "person's hand",
    "pottedplant's pot",
    "pottedplant's plant",
    "sheep's tail",
    "sheep's head",
    "sheep's eye",
    "sheep's torso",
    "sheep's neck",
    "sheep's leg",
    "sheep's ear",
    "sheep's muzzle",
    "sheep's horn",
    "train's headlight",
    "train's head",
    "train's front",
    "train's side",
    "train's back",
    "train's roof",
    "train's coach",
    "tvmonitor's screen",
)


def _voc_palette(n: int):
    """Deterministic Pascal-style palette, only for visualization."""
    palette = []
    for j in range(n):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
        palette.append([r, g, b])
    return palette


@DATASETS.register_module(force=True)
class PascalPart116PartDataset(CustomDataset):
    CLASSES = PART_CLASSES
    PALETTE = _voc_palette(len(PART_CLASSES))

    def __init__(self, **kwargs):
        requested_ignore = int(kwargs.pop("ignore_index", 255))
        requested_reduce = bool(kwargs.pop("reduce_zero_label", False))

        if requested_ignore != 255:
            raise ValueError(
                f"Pascal-Part-116 requires ignore_index=255, got {requested_ignore}"
            )
        if requested_reduce:
            raise ValueError(
                "Pascal-Part-116 class 0 is a valid part; reduce_zero_label must be False"
            )

        super().__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            ignore_index=255,
            reduce_zero_label=False,
            **kwargs,
        )

        if len(self.CLASSES) != 116:
            raise AssertionError(f"expected 116 part classes, got {len(self.CLASSES)}")
        if self.CLASSES[0] == "background":
            raise AssertionError("background must not be a semantic class")
        if self.CLASSES[0] != "aeroplane's body":
            raise AssertionError(f"unexpected class 0: {self.CLASSES[0]!r}")
        if self.ignore_index != 255:
            raise AssertionError(f"ignore_index changed to {self.ignore_index}")
        if self.reduce_zero_label is not False:
            raise AssertionError("reduce_zero_label changed from False")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        if self.ann_dir is None or not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(self.ann_dir)
