#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UNCHANGED official 116-way PP116 evaluator + post-hoc 74/42 IoU means.

The evaluator and text vocabulary are not modified. This only orchestrates the
existing bake/eval binaries and aggregates their 116 per-class IoUs afterward.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from seen_split import UNSEEN_OBJECTS

PROJECTOR = "weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
TEXT = "feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"
MODEL_CONFIG = "configs/vitb_mlp_infonce.yaml"
EVAL_CFG = "src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml"
EVAL_BASE_CFG = "src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml"


def parse_official_output(txt: str):
    match = re.search(r"mIoU of\s+850\s+test images:\s*([0-9.]+)%", txt)
    if not match:
        raise ValueError("Cannot locate official 850-image 116-way mIoU in evaluator log")
    perclass = []
    in_table = False
    for line in txt.splitlines():
        if "per class results:" in line.lower():
            in_table = True
            continue
        if in_table and "Summary:" in line:
            break
        if in_table and line.strip().startswith("|") and "Class" not in line and "IoU" not in line:
            fields = [x.strip() for x in line.strip().split("|")[1:-1]]
            if len(fields) == 3:
                try:
                    perclass.append((fields[0], float(fields[1]), float(fields[2])))
                except ValueError:
                    pass
    if len(perclass) != 116:
        raise ValueError(f"Official per-class parser returned {len(perclass)} of 116 parts; refuse to fabricate seen/unseen means")
    if len({p[0] for p in perclass}) != 116:
        raise ValueError("Duplicate part names in official evaluator output")
    seen, unseen = [], []
    for name, iou, _ in perclass:
        if "'s " not in name:
            raise ValueError(f"Unexpected official class name {name!r}")
        obj = name.split("'s ", 1)[0]
        (unseen if obj in UNSEEN_OBJECTS else seen).append(iou)
    if len(seen) != 74 or len(unseen) != 42:
        raise ValueError(f"Seen/unseen class-count mismatch: seen={len(seen)} unseen={len(unseen)}")
    mean_seen = sum(seen) / len(seen)
    mean_unseen = sum(unseen) / len(unseen)
    h = 2 * mean_seen * mean_unseen / (mean_seen + mean_unseen) if (mean_seen + mean_unseen) else 0.
    return {
        "full116_miou": float(match.group(1)),
        "seen74_miou": mean_seen,
        "unseen42_miou": mean_unseen,
        "harmonic_seen_unseen": h,
    }, perclass


def run_logged(cmd, path: Path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    print("[EXEC]", " ".join(map(str, cmd)), flush=True)
    with path.open("w", encoding="utf8") as stream:
        p = subprocess.run([str(x) for x in cmd], stdout=stream,
                           stderr=subprocess.STDOUT, env=env, check=False)
    if p.returncode:
        lines = path.read_text(errors="replace").splitlines()[-20:]
        raise RuntimeError(f"Command returned {p.returncode}: {path}\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--epochs", default="3,10,20,30")
    parser.add_argument("--run_dir", default="final_exp/relproto_alignment/pp116_seen74_softrelative_structure_l4_seed123")
    parser.add_argument("--output", default="output/pp116_seen74_softrelative_structure_l4_eval")
    parser.add_argument("--master_port", type=int, default=30171)
    args = parser.parse_args()
    root = Path.cwd()
    for rel in ["bake_predobj_relproto_w_into_projector_final.py", "src/open_vocabulary_segmentation/main.py"]:
        if not (root / rel).exists():
            raise FileNotFoundError(f"Run from repository root, missing {rel}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    out = Path(args.output)
    rows = []
    perclass_rows = []
    for epoch in [int(x) for x in args.epochs.split(",") if x.strip()]:
        ck = Path(args.run_dir) / f"W_epoch_{epoch:03d}.pt"
        if not ck.exists():
            raise FileNotFoundError(f"Seen-only W checkpoint does not exist: {ck}")
        name = f"pp116_seen74_softrel_l4_e{epoch:03d}"
        baked = Path("weights") / (name + ".pth")
        ep_out = out / f"epoch_{epoch:03d}"
        log = ep_out / "evaluation_stdout.log"
        if not baked.exists():
            run_logged([
                sys.executable, "bake_predobj_relproto_w_into_projector_final.py",
                "--project_root", ".", "--projector", PROJECTOR,
                "--w_checkpoint", ck, "--text_bank", TEXT,
                "--model_config", MODEL_CONFIG, "--output", baked, "--device", "cuda",
            ], ep_out / "bake_stdout.log", env)
        cached = False
        if log.exists():
            try:
                parse_official_output(log.read_text(errors="replace"))
                cached = True
            except ValueError:
                pass
        if not cached:
            run_logged([
                sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
                "--master_port", str(args.master_port), "src/open_vocabulary_segmentation/main.py",
                "--eval", "--output", ep_out / "official_output",
                "--eval_cfg", EVAL_CFG, "--eval_base_cfg", EVAL_BASE_CFG,
                "--opts", f"model.proj_name={name}", "evaluate.pamr=false",
            ], log, env)
        means, perclass = parse_official_output(log.read_text(errors="replace"))
        rows.append({"epoch": epoch, **means})
        perclass_rows.extend({"epoch": epoch, "part": name_, "IoU_pct": iou,
                              "mAcc_pct": acc, "split": ("unseen" if name_.split("'s ", 1)[0] in UNSEEN_OBJECTS else "seen")}
                             for name_, iou, acc in perclass)
        print(f"[E{epoch:03d}] all116={means['full116_miou']:.3f}% "
              f"seen74={means['seen74_miou']:.3f}% "
              f"unseen42={means['unseen42_miou']:.3f}% "
              f"harmonic={means['harmonic_seen_unseen']:.3f}%", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data, fields in [
        ("seen_unseen_summary.csv", rows, ["epoch", "full116_miou", "seen74_miou", "unseen42_miou", "harmonic_seen_unseen"]),
        ("all116_part_iou.csv", perclass_rows, ["epoch", "part", "IoU_pct", "mAcc_pct", "split"]),
    ]:
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=fields)
            wr.writeheader()
            wr.writerows(data)
    print("[DONE]", out / "seen_unseen_summary.csv", flush=True)


if __name__ == "__main__":
    main()
