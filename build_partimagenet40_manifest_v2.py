#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build PartImageNet weak-supervision manifest v2.

Input:
    existing image-level part-presence manifest

Output per image:
    image
    object_wnid
    object_name          # actual ImageNet semantic object
    part_parent_group    # PartImageNet superclass
    part_ids
    part_names

No spatial annotation is stored in the output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from timm.data import ImageNetInfo

from partimagenet40_taxonomy import (
    PART_CLASSES,
    OBJECT_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)


def imagenet_object_name(info: ImageNetInfo, wnid: str) -> str:
    # timm >= 1.x
    if hasattr(info, "label_name_to_description"):
        desc = info.label_name_to_description(
            wnid,
            detailed=False,
        )
    else:
        names = list(info.label_names())
        if wnid not in names:
            raise KeyError(f"unknown ImageNet WNID: {wnid}")
        idx = names.index(wnid)
        desc = info.index_to_description(idx)

    name = str(desc).strip()

    # ImageNet descriptions may be:
    # "tench, Tinca tinca"
    # We want a normal semantic object query.
    name = name.split(",", 1)[0].strip()
    name = name.replace("_", " ")

    if not name:
        raise ValueError(
            f"empty ImageNet description for {wnid}"
        )

    return name


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default=(
            "/home/master/dataset/lyx/PartImageNet/"
            "PartImageNet/partimagenet_train_presence.json"
        ),
    )

    p.add_argument(
        "--output",
        default=(
            "/home/master/dataset/lyx/PartImageNet/"
            "PartImageNet/partimagenet_train_weak_v2.json"
        ),
    )

    args = p.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()

    data = json.loads(
        src.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise TypeError("input manifest must be a list")

    assert len(PART_CLASSES) == 40
    assert len(OBJECT_CLASSES) == 11
    assert set(PART_TO_OBJECT) == set(range(40))

    info = ImageNetInfo("imagenet-1k")

    output = []

    wnids = set()
    groups = Counter()
    class_count = Counter()

    for i, rec in enumerate(data):
        image = str(rec["image"])

        stem = Path(image).stem
        wnid = stem.split("_", 1)[0]

        if not (
            len(wnid) == 9
            and wnid.startswith("n")
            and wnid[1:].isdigit()
        ):
            raise ValueError(
                f"row {i}: cannot parse WNID from {image}"
            )

        raw_pids = rec.get(
            "labels",
            rec.get("part_ids"),
        )

        if raw_pids is None:
            raise KeyError(
                f"row {i}: missing labels/part_ids"
            )

        pids = sorted({
            int(x)
            for x in raw_pids
        })

        if not pids:
            raise ValueError(
                f"row {i}: empty part presence"
            )

        if any(x < 0 or x >= 40 for x in pids):
            raise ValueError(
                f"row {i}: invalid part IDs {pids}"
            )

        parent_ids = {
            int(PART_TO_OBJECT[x])
            for x in pids
        }

        if len(parent_ids) != 1:
            raise ValueError(
                f"row {i}: parts span multiple groups: "
                f"{image} {pids} {parent_ids}"
            )

        parent_id = next(iter(parent_ids))
        parent_group = OBJECT_CLASSES[parent_id]

        expected = set(
            OBJECT_GROUPS[parent_group]
        )

        if not set(pids).issubset(expected):
            raise ValueError(
                f"{image}: {pids} do not belong to "
                f"{parent_group}: {sorted(expected)}"
            )

        object_name = imagenet_object_name(
            info,
            wnid,
        )

        output.append({
            "image": image,

            # Actual object semantic identity.
            "object_wnid": wnid,
            "object_name": object_name,

            # PartImageNet relational taxonomy.
            "part_parent_group": parent_group,

            # Image-level part presence.
            "part_ids": pids,
            "part_names": [
                PART_CLASSES[x]
                for x in pids
            ],

            # Backward-compatible alias.
            "labels": pids,
        })

        wnids.add(wnid)
        groups[parent_group] += 1
        class_count.update(pids)

    if len(output) != len(data):
        raise AssertionError(
            (len(output), len(data))
        )

    if set(class_count) != set(range(40)):
        raise AssertionError(
            "40-part coverage failed: "
            f"{sorted(class_count)}"
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PartImageNet weak manifest v2")
    print("=" * 72)
    print("input images :", len(data))
    print("output images:", len(output))
    print("object WNIDs :", len(wnids))
    print("part classes :", len(class_count))
    print()

    print("groups:")
    for name in OBJECT_CLASSES:
        print(
            f"  {name:12s}: {groups[name]}"
        )

    print()
    print("first 10:")
    for x in output[:10]:
        print(
            x["image"],
            "| object =", x["object_name"],
            "| group =", x["part_parent_group"],
            "| parts =", x["part_ids"],
        )

    print()
    print("saved:", dst)
    print("PARTIMAGENET WEAK MANIFEST V2: PASS")


if __name__ == "__main__":
    main()
