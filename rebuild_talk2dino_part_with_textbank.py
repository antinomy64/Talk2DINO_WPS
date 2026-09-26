#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_bank", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    tbp = Path(args.text_bank).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")

    data = torch.load(inp, map_location="cpu")
    bank = torch.load(tbp, map_location="cpu")

    if "features" not in bank:
        raise KeyError("text bank has no 'features'")
    if "class_names" not in bank:
        raise KeyError("text bank has no 'class_names'")

    feats = bank["features"].float().contiguous()
    names = list(bank["class_names"])

    if feats.ndim != 2 or feats.shape[1] != 512:
        raise ValueError(
            f"Expected text bank [N,512], got {tuple(feats.shape)}"
        )

    if bank.get("template_set") != "sub_imagenet_template":
        raise ValueError(
            f"Wrong template_set: {bank.get('template_set')}"
        )

    if bank.get("stage") != "pre_projector_prompt_mean":
        raise ValueError(
            f"Wrong stage: {bank.get('stage')}"
        )

    if bank.get("normalized", False):
        raise ValueError(
            "Expected raw unnormalized prompt-mean CLIP bank."
        )

    anns = data["annotations"]

    used = set()

    for i, ann in enumerate(anns):
        pid = int(ann["part_id"])

        if not (0 <= pid < len(names)):
            raise ValueError(
                f"annotation {i}: bad part_id={pid}"
            )

        # Strong taxonomy check.
        caption = str(ann.get("caption", ""))
        expected = names[pid]

        if caption and caption != expected:
            raise ValueError(
                f"annotation {i}: caption mismatch: "
                f"{caption!r} != {expected!r}"
            )

        ann["caption"] = expected
        ann["ann_feats"] = feats[pid].clone()
        ann["text_feature_source"] = "sub_imagenet_template_prompt_mean"
        used.add(pid)

    meta = dict(data.get("meta", {}))

    meta.update({
        "text_bank": str(tbp),
        "text_feature": "ann_feats",
        "text_feature_source":
            "sub_imagenet_template_prompt_mean",
        "template_set": "sub_imagenet_template",
        "text_stage": "pre_projector_prompt_mean",
        "text_normalized": False,
        "clip_model": bank.get("clip_model", "ViT-B/16"),
        "num_text_classes": len(names),
        "text_class_names": names,
    })

    data["meta"] = meta

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)

    print("=" * 60)
    print("Talk2DINO-Part text-bank rebuild")
    print("=" * 60)
    print("input        :", inp)
    print("text bank    :", tbp)
    print("output       :", out)
    print("annotations  :", len(anns))
    print("classes      :", len(names))
    print("used classes :", len(used))
    print("feature shape:", tuple(feats.shape))
    print("template     :", bank.get("template_set"))
    print("stage        :", bank.get("stage"))
    print("normalized   :", bank.get("normalized"))
    print("=" * 60)


if __name__ == "__main__":
    main()
