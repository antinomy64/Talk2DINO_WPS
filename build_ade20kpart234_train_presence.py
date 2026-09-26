#!/usr/bin/env python3

"""
Build image-level part-presence supervision for ADE20K-Part-234.

Input:
    ADE20K-Part-234 semantic part PNG masks.

Output:
    A JSON list containing only:
        image filename
        image stem
        parent object identity
        set of present semantic part IDs/names

No spatial information is retained:
    no masks
    no pixels
    no boxes
    no points
    no coordinates

The resulting JSON is intended to be the ONLY target semantic
supervision consumed by the subsequent RelProto cache extractor.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ade20kpart234_taxonomy import (
    IGNORE_LABEL,
    NUM_PARTS,
    NUM_OBJECTS,
    PART_CLASSES,
    OBJECT_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)


IMAGE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        default=(
            "/home/master/dataset/lyx/"
            "ADE20KPart234"
        ),
    )

    p.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
    )

    p.add_argument(
        "--output",
        default=None,
    )

    p.add_argument(
        "--meta_output",
        default=None,
    )

    return p.parse_args()


def collect_files_by_stem(
    root: Path,
    suffixes,
):
    out = {}

    suffixes = {
        x.lower()
        for x in suffixes
    }

    for p in sorted(root.iterdir()):

        if not p.is_file():
            continue

        if p.suffix.lower() not in suffixes:
            continue

        if p.stem in out:
            raise RuntimeError(
                f"duplicate stem under {root}: "
                f"{p.stem}"
            )

        out[p.stem] = p

    return out


def main():
    args = parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    image_root = (
        root /
        "images" /
        args.split
    )

    mask_root = (
        root /
        "annotations_detectron2_part" /
        args.split
    )

    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    if not mask_root.is_dir():
        raise FileNotFoundError(mask_root)

    if args.output is None:
        output = (
            root /
            f"ade20kpart234_{args.split}_presence.json"
        )
    else:
        output = Path(
            args.output
        ).expanduser()

        if not output.is_absolute():
            output = output.resolve()

    if args.meta_output is None:
        meta_output = output.with_suffix(
            ".meta.json"
        )
    else:
        meta_output = Path(
            args.meta_output
        ).expanduser()

        if not meta_output.is_absolute():
            meta_output = meta_output.resolve()

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    if meta_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {meta_output}"
        )

    images = collect_files_by_stem(
        image_root,
        IMAGE_SUFFIXES,
    )

    masks = collect_files_by_stem(
        mask_root,
        (".png",),
    )

    missing_mask = sorted(
        set(images) -
        set(masks)
    )

    missing_image = sorted(
        set(masks) -
        set(images)
    )

    if missing_mask:
        raise RuntimeError(
            f"{len(missing_mask)} images lack masks; "
            f"first={missing_mask[:10]}"
        )

    if missing_image:
        raise RuntimeError(
            f"{len(missing_image)} masks lack images; "
            f"first={missing_image[:10]}"
        )

    stems = sorted(images)

    records = []

    part_image_freq = Counter()
    object_image_freq = Counter()

    num_empty = 0
    num_multi_object = 0

    objects_per_image = Counter()
    parts_per_image = Counter()

    for stem in stems:

        image_path = images[stem]
        mask_path = masks[stem]

        mask = np.asarray(
            Image.open(mask_path)
        )

        if mask.ndim != 2:
            raise ValueError(
                f"expected 2D label mask: "
                f"{mask_path} shape={mask.shape}"
            )

        unique_ids = sorted(
            int(x)
            for x in np.unique(mask)
            if int(x) != IGNORE_LABEL
        )

        invalid = [
            x
            for x in unique_ids
            if not (0 <= x < NUM_PARTS)
        ]

        if invalid:
            raise ValueError(
                f"invalid labels in {mask_path}: "
                f"{invalid}"
            )

        # Image has no valid semantic part label.
        if not unique_ids:
            num_empty += 1
            continue

        # Group present parts by their official parent object.
        by_object = defaultdict(list)

        for pid in unique_ids:
            oid = int(
                PART_TO_OBJECT[pid]
            )

            by_object[oid].append(pid)

            part_image_freq[pid] += 1

        objects = []

        for oid in sorted(by_object):

            pids = sorted(
                by_object[oid]
            )

            object_name = (
                OBJECT_CLASSES[oid]
            )

            # Strong group contract.
            allowed = set(
                OBJECT_GROUPS[object_name]
            )

            if not set(pids).issubset(allowed):
                raise AssertionError(
                    f"group mismatch: "
                    f"image={image_path.name} "
                    f"object={object_name} "
                    f"pids={pids}"
                )

            object_image_freq[oid] += 1

            objects.append(
                {
                    "object_id": oid,
                    "object_name": object_name,
                    "labels": pids,
                    "part_names": [
                        PART_CLASSES[pid]
                        for pid in pids
                    ],
                }
            )

        if len(objects) > 1:
            num_multi_object += 1

        objects_per_image[
            len(objects)
        ] += 1

        parts_per_image[
            len(unique_ids)
        ] += 1

        records.append(
            {
                "image": image_path.name,
                "stem": stem,
                "objects": objects,
            }
        )

    # ---------------------------------------------------------
    # Final manifest contract.
    # ---------------------------------------------------------

    for rec in records:

        assert rec["objects"]

        seen_labels = []

        for obj in rec["objects"]:

            oid = int(
                obj["object_id"]
            )

            assert 0 <= oid < NUM_OBJECTS

            object_name = (
                OBJECT_CLASSES[oid]
            )

            assert (
                obj["object_name"]
                == object_name
            )

            pids = [
                int(x)
                for x in obj["labels"]
            ]

            assert pids
            assert pids == sorted(pids)
            assert (
                len(pids)
                == len(set(pids))
            )

            allowed = set(
                OBJECT_GROUPS[
                    object_name
                ]
            )

            assert set(pids).issubset(
                allowed
            )

            expected_names = [
                PART_CLASSES[pid]
                for pid in pids
            ]

            assert (
                obj["part_names"]
                == expected_names
            )

            seen_labels.extend(pids)

        assert (
            len(seen_labels)
            == len(set(seen_labels))
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    present_classes = sorted(
        part_image_freq
    )

    missing_classes = sorted(
        set(range(NUM_PARTS)) -
        set(present_classes)
    )

    meta = {
        "dataset": "ADE20K-Part-234",
        "split": args.split,

        "root": str(root),
        "image_root": str(image_root),
        "mask_root": str(mask_root),

        "num_raw_images": len(images),
        "num_raw_masks": len(masks),

        "num_manifest_images": len(records),
        "num_empty_part_images": num_empty,

        "num_multi_object_images": (
            num_multi_object
        ),

        "num_objects": NUM_OBJECTS,
        "num_parts": NUM_PARTS,

        "ignore_label": IGNORE_LABEL,

        "present_part_classes": (
            len(present_classes)
        ),

        "missing_part_ids": (
            missing_classes
        ),

        "missing_part_names": [
            PART_CLASSES[x]
            for x in missing_classes
        ],

        "objects_per_image": {
            str(k): int(v)
            for k, v in sorted(
                objects_per_image.items()
            )
        },

        "parts_per_image": {
            str(k): int(v)
            for k, v in sorted(
                parts_per_image.items()
            )
        },

        "object_image_frequency": {
            OBJECT_CLASSES[oid]:
                int(
                    object_image_freq.get(
                        oid,
                        0,
                    )
                )
            for oid in range(
                NUM_OBJECTS
            )
        },

        "part_image_frequency": {
            str(pid): {
                "name": PART_CLASSES[pid],
                "count": int(
                    part_image_freq.get(
                        pid,
                        0,
                    )
                ),
            }
            for pid in range(
                NUM_PARTS
            )
        },

        "supervision_written_to_manifest": (
            "image-level part presence only"
        ),

        "spatial_information_retained": False,
    }

    meta_output.write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("ADE20K-Part-234 presence manifest")
    print("=" * 72)

    print(
        "split                  :",
        args.split,
    )

    print(
        "raw images             :",
        len(images),
    )

    print(
        "raw masks              :",
        len(masks),
    )

    print(
        "manifest images        :",
        len(records),
    )

    print(
        "empty part images      :",
        num_empty,
    )

    print(
        "multi-object images    :",
        num_multi_object,
    )

    print(
        "present part classes   :",
        len(present_classes),
    )

    print(
        "missing part IDs       :",
        missing_classes,
    )

    print()

    print("object image frequency:")

    for oid, name in enumerate(
        OBJECT_CLASSES
    ):
        print(
            f"  {oid:02d} "
            f"{name:20s} "
            f"{object_image_freq.get(oid, 0)}"
        )

    print()

    print(
        "[save]",
        len(records),
        "images ->",
        output,
    )

    print(
        "[save] metadata ->",
        meta_output,
    )

    print()

    print(
        "ADE20K-PART-234 PRESENCE: PASS"
    )


if __name__ == "__main__":
    main()
