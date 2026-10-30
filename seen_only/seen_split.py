#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official OV-PARTS / PP116 74-seen / 42-unseen split.

Unseen objects: bird, car, dog, motorbike, sheep (42 parts).
Other 11 objects: seen (74 parts).  Read only *text taxonomy* and weak
presence metadata, never any GT part mask.  Train/eval IDs stay in [0,115].
"""
from __future__ import annotations

from collections import Counter
from typing import Any

UNSEEN_OBJECTS = frozenset({"bird", "car", "dog", "motorbike", "sheep"})
EXPECTED_TOTAL_PARTS = 116
EXPECTED_SEEN_PARTS = 74
EXPECTED_UNSEEN_PARTS = 42
EXPECTED_SEEN_OBJECTS = 11
EXPECTED_UNSEEN_OBJECTS = 5


def split_from_text_names(text_names: list[str]) -> dict[str, Any]:
    if len(text_names) != EXPECTED_TOTAL_PARTS:
        raise ValueError(f"Expected 116 original part names; got {len(text_names)}")
    if len(set(text_names)) != len(text_names):
        raise ValueError("Duplicate full part names: check original taxonomy")
    ids_by_object: dict[str, list[int]] = {}
    for pid, name in enumerate(text_names):
        if not isinstance(name, str) or "'s " not in name:
            raise ValueError(f"Unexpected taxonomy entry at original ID {pid}: {name!r}")
        obj, part = name.split("'s ", 1)
        if not obj or not part:
            raise ValueError(f"Malformed part name {name!r}")
        ids_by_object.setdefault(obj, []).append(pid)
    seen_objects = sorted(set(ids_by_object) - UNSEEN_OBJECTS)
    unseen_objects = sorted(UNSEEN_OBJECTS)
    missing = UNSEEN_OBJECTS - ids_by_object.keys()
    seen_ids = tuple(pid for pid, name in enumerate(text_names)
                     if name.split("'s ", 1)[0] not in UNSEEN_OBJECTS)
    unseen_ids = tuple(pid for pid in range(len(text_names)) if pid not in seen_ids)
    if (missing or len(seen_objects) != EXPECTED_SEEN_OBJECTS
            or len(unseen_objects) != EXPECTED_UNSEEN_OBJECTS
            or len(seen_ids) != EXPECTED_SEEN_PARTS
            or len(unseen_ids) != EXPECTED_UNSEEN_PARTS):
        raise ValueError(
            "PP116 benchmark taxonomy/split mismatch: "
            f"missing unseen objects={sorted(missing)}; "
            f"seen objects={seen_objects}; "
            f"seen parts={len(seen_ids)}, unseen parts={len(unseen_ids)}. "
            "Do not train until taxonomy is verified."
        )
    return {
        "benchmark": "OV-PARTS PP116 generalized zero-shot (74 seen / 42 unseen)",
        "seen_objects": seen_objects,
        "unseen_objects": unseen_objects,
        "seen_ids": seen_ids,
        "unseen_ids": unseen_ids,
        "seen_names": [text_names[i] for i in seen_ids],
        "unseen_names": [text_names[i] for i in unseen_ids],
        "parts_by_object": {k: len(v) for k, v in sorted(ids_by_object.items())},
    }


def filter_seen_examples(examples: list[Any], report: dict, text_names: list[str]):
    """Exclude unseen object annotations from the ORIGINAL valid cached examples.

    No mutation or remapping of pid, foreground or DINO patch tensor.
    A mixed/foreign candidate part is an error, not silently erased.
    """
    split = split_from_text_names(text_names)
    seen_objects = set(split["seen_objects"])
    unseen_objects = set(split["unseen_objects"])
    seen_ids = set(split["seen_ids"])
    all_objects = seen_objects | unseen_objects
    kept, excluded = [], []
    source_counts, kept_counts = Counter(), Counter()
    for ex in examples:
        obj = str(ex.class_name)
        if obj not in all_objects:
            raise ValueError(f"Unknown cached object row={ex.row_index} class={obj!r}")
        pids = [int(i) for i in ex.pids.tolist()]
        if not pids or len(pids) != len(set(pids)):
            raise ValueError(f"Invalid candidate ids row={ex.row_index}: {pids}")
        for pid in pids:
            if pid < 0 or pid >= len(text_names):
                raise ValueError(f"Out-of-taxonomy pid row={ex.row_index}: {pid}")
            if text_names[pid].split("'s ", 1)[0] != obj:
                raise ValueError(
                    f"Unexpected foreign-object presence row={ex.row_index}, "
                    f"obj={obj!r} pid={pid} text={text_names[pid]!r}"
                )
        source_counts[obj] += 1
        if obj in seen_objects:
            if not set(pids).issubset(seen_ids):
                raise RuntimeError("Unseen part leaked into seen training row")
            kept.append(ex)
            kept_counts[obj] += 1
        else:
            excluded.append(int(ex.row_index))
    if not kept:
        raise ValueError("No seen-only annotations remained after filtering")
    if not (set(excluded).isdisjoint({int(e.row_index) for e in kept})):
        raise RuntimeError("Seen/unseen cache overlap")
    if len(kept) + len(excluded) != len(examples):
        raise RuntimeError("Filtering accounting invariant failed")
    result = dict(report)
    result["source_ready_annotations"] = len(examples)
    result["excluded_unseen_annotations"] = len(excluded)
    result["seen_training_annotations"] = len(kept)
    result["split_name"] = split["benchmark"]
    result["seen_objects"] = split["seen_objects"]
    result["unseen_objects"] = split["unseen_objects"]
    result["seen_part_ids"] = list(split["seen_ids"])
    result["unseen_part_ids"] = list(split["unseen_ids"])
    result["source_crops_by_object"] = dict(sorted(source_counts.items()))
    result["seen_training_crops_by_object"] = dict(sorted(kept_counts.items()))
    result["excluded_unseen_row_indices"] = excluded
    # Make downstream training_samples.csv and preflight.json reflect actual data.
    if isinstance(report.get("rows"), list):
        included_indices = {int(e.row_index) for e in kept}
        result["rows"] = [r for r in report["rows"]
                          if int(r.get("row_index", -1)) in included_indices]
    result["selected_annotations"] = len(kept)
    result["ready_annotations"] = len(kept)
    result["status_counts"] = {"ready": len(kept)}
    return kept, result, split
