#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Create ADE20K-Part-234 files by minimally adapting the already-working PP116 path.
#
# Run from:
#   /home/master/code/lyx/Talk2DINO_official_bg
#
# This script NEVER edits PP116 source files. It only creates ADE234-specific files.

from __future__ import annotations

import importlib.util
import py_compile
import re
import ast
from pathlib import Path

ROOT = Path.cwd().resolve()

SRC_EXTRACT = ROOT / "extract_predobj_cropaug.py"
SRC_TRAIN = ROOT / "train_relproto_alignemt.py"
SRC_BAKE = ROOT / "bake_predobj_relproto_w_into_projector_final.py"
TAXONOMY = ROOT / "ade20kpart234_taxonomy.py"

DST_EXTRACT = ROOT / "extract_predobj_cropaug_ade20kpart234.py"
DST_TRAIN = ROOT / "train_relproto_alignment_ade20kpart234.py"
DST_BAKE = ROOT / "bake_ade20kpart234_relproto_w.py"

DST_DATASET = ROOT / "src/open_vocabulary_segmentation/segmentation/datasets/ade20kpart234_part.py"
DST_BASECFG = ROOT / "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/ade20kpart234_part.py"
DST_CFGDIR = ROOT / "src/open_vocabulary_segmentation/configs/ade20kpart234"
DST_EVALCFG = DST_CFGDIR / "eval_ade20kpart234.yml"
DST_MODEL_CFG = DST_CFGDIR / "dinotext_ade20kpart234_vitb_mlp_infonce.yml"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_taxonomy():
    spec = importlib.util.spec_from_file_location("ade20kpart234_taxonomy", TAXONOMY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TAXONOMY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    classes = tuple(mod.PART_CLASSES)
    objects = tuple(mod.OBJECT_CLASSES)
    groups = {str(k): list(v) for k, v in mod.OBJECT_GROUPS.items()}
    p2o = {int(k): int(v) for k, v in mod.PART_TO_OBJECT.items()}

    assert len(classes) == 234, len(classes)
    assert len(objects) == 44, len(objects)
    assert sorted(p2o) == list(range(234))
    flat = [pid for obj in objects for pid in groups[obj]]
    assert flat == list(range(234))
    assert classes[0] == "person's head"
    assert classes[11] == "door's handle"
    assert classes[193] == "van's license plate"
    assert classes[233] == "light's diffusor"
    return classes, objects, groups, p2o


def replace_once(s: str, old: str, new: str, name: str) -> str:
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: expected exactly 1 match, got {n}")
    return s.replace(old, new, 1)


def replace_between(s: str, start_marker: str, end_marker: str, new: str, name: str) -> str:
    a = s.find(start_marker)
    if a < 0:
        raise RuntimeError(f"{name}: start marker not found")
    b = s.find(end_marker, a)
    if b < 0:
        raise RuntimeError(f"{name}: end marker not found")
    return s[:a] + new + s[b:]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("[write]", path.relative_to(ROOT))


for p in (SRC_EXTRACT, SRC_TRAIN, SRC_BAKE, TAXONOMY):
    require(p)

PART_CLASSES, OBJECT_CLASSES, OBJECT_GROUPS, PART_TO_OBJECT = load_taxonomy()
print("[taxonomy] 234 parts / 44 parent objects: PASS")


# ---------------------------------------------------------------------
# 1. Pred-object cache extractor
#    Source: working PP116 extract_predobj_cropaug.py
# ---------------------------------------------------------------------

s = SRC_EXTRACT.read_text(encoding="utf-8")

taxonomy_block = '''# -----------------------------------------------------------------------------
# ADE20K-Part-234 taxonomy
# -----------------------------------------------------------------------------

from ade20kpart234_taxonomy import (
    NUM_PARTS,
    NUM_OBJECTS,
    IGNORE_LABEL,
    PART_CLASSES,
    OBJECT_CLASSES,
    OBJECT_GROUPS,
    PART_TO_OBJECT,
)

PART_CLASSES = list(PART_CLASSES)
OBJECT_CLASSES = list(OBJECT_CLASSES)

OBJ_NAME_TO_ID = {
    name: i for i, name in enumerate(OBJECT_CLASSES)
}

PART_NAME_TO_ID = {
    name: i for i, name in enumerate(PART_CLASSES)
}

PART_IDS_BY_OBJECT = {
    name: list(OBJECT_GROUPS[name])
    for name in OBJECT_CLASSES
}

assert NUM_PARTS == 234
assert NUM_OBJECTS == 44
assert len(PART_CLASSES) == 234
assert len(OBJECT_CLASSES) == 44
assert sorted(PART_TO_OBJECT) == list(range(234))

'''

s = replace_between(
    s,
    "# -----------------------------------------------------------------------------\n# Pascal-Part-116 taxonomy",
    "VISION_DIM = 768",
    taxonomy_block,
    "extract taxonomy",
)

s = s.replace(
    'description="Offline predicted-object cropaug cache for Pascal-Part-116"',
    'description="Offline predicted-object cropaug cache for ADE20K-Part-234"',
)

s = replace_once(
    s,
    '    p.add_argument("--obj_mask_root", default=None)\n',
    '',
    "remove obj_mask_root arg",
)

start = s.find('    # Original PascalPart116 masks used by the earlier pipeline are 0-based')
end = s.find('    p.add_argument(\n        "--projector_weight"', start)
if start < 0 or end < 0:
    raise RuntimeError("extract CLI mask block not found")

ade_mask_cli = '''    # ADE20K-Part-234 released masks:
    # semantic part IDs 0..233; background/unlabeled = 65535.
    # Only np.unique(mask) is used to obtain image-level presence.
    p.add_argument("--part_ignore", type=int, default=65535)

'''
s = s[:start] + ade_mask_cli + s[end:]

self_start = s.find("def self_test() -> None:")
self_end = s.find("# -----------------------------------------------------------------------------\n# Main", self_start)
if self_start < 0 or self_end < 0:
    raise RuntimeError("extract self_test block not found")

self_test = '''def self_test() -> None:
    assert len(PART_CLASSES) == 234
    assert len(OBJECT_CLASSES) == 44
    assert sorted(PART_TO_OBJECT) == list(range(234))

    assert PART_CLASSES[0] == "person's head"
    assert PART_CLASSES[11] == "door's handle"
    assert PART_CLASSES[193] == "van's license plate"
    assert PART_CLASSES[233] == "light's diffusor"

    part = np.full((6, 8), 65535, dtype=np.uint16)
    part[1:3, 1:4] = 0
    part[3:5, 2:6] = 1

    present = sorted(
        int(x) for x in np.unique(part)
        if int(x) != 65535
    )
    assert present == [0, 1]
    object_ids = sorted({int(PART_TO_OBJECT[x]) for x in present})
    assert object_ids == [0]

    pm = np.zeros((10, 20), dtype=np.uint8)
    pm[2:8, 5:15] = 1
    box = square_crop_box(pm, 20, 10, 1.2)
    patch = mask_to_patch_grid(pm, box)

    assert tuple(patch.shape) == (1024,)
    assert patch.dtype == torch.bool
    assert int(patch.sum().item()) > 0

    print("SELF_TEST_PASS")


'''
s = s[:self_start] + self_test + s[self_end:]

s = replace_once(
    s,
    '    for required_name in ("image_root", "obj_mask_root", "part_mask_root", "output_pth"):\n',
    '    for required_name in ("image_root", "part_mask_root", "output_pth"):\n',
    "extract required args",
)

old_roots = '    image_root = Path(args.image_root).expanduser().resolve()\n    obj_root = Path(args.obj_mask_root).expanduser().resolve()\n    part_root = Path(args.part_mask_root).expanduser().resolve()\n    for root in (image_root, obj_root, part_root):\n        if not root.is_dir():\n            raise FileNotFoundError(root)\n'

new_roots = '    image_root = Path(args.image_root).expanduser().resolve()\n    part_root = Path(args.part_mask_root).expanduser().resolve()\n    for root in (image_root, part_root):\n        if not root.is_dir():\n            raise FileNotFoundError(root)\n'

s = replace_once(s, old_roots, new_roots, "extract roots")

old_stems = '''    else:
        image_stems = {p.stem for p in image_root.iterdir() if p.is_file()}
        obj_stems = {p.stem for p in obj_root.iterdir() if p.is_file()}
        part_stems = {p.stem for p in part_root.iterdir() if p.is_file()}
        stems = sorted(image_stems & obj_stems & part_stems)
'''
new_stems = '''    else:
        image_stems = {p.stem for p in image_root.iterdir() if p.is_file()}
        part_stems = {p.stem for p in part_root.iterdir() if p.is_file()}
        stems = sorted(image_stems & part_stems)
'''
s = replace_once(s, old_stems, new_stems, "extract stems")

s = s.replace("VOC20_CLASSES", "OBJECT_CLASSES")

old_shape = '''    if tuple(object_text.shape) != (20, VISION_DIM):
        raise ValueError(f"bad object text bank shape: {tuple(object_text.shape)}")
'''
new_shape = '''    expected_object_text_shape = (len(OBJECT_CLASSES), VISION_DIM)
    if tuple(object_text.shape) != expected_object_text_shape:
        raise ValueError(
            f"bad object text bank shape: {tuple(object_text.shape)} "
            f"!= {expected_object_text_shape}"
        )
'''
s = replace_once(s, old_shape, new_shape, "extract object text shape")

loop_start = s.find(
    '    for stem in tqdm(stems, desc="weak labels -> pred obj -> crop -> DINO cache"):'
)
downstream = s.find(
    '        # From here onward no GT spatial array is used in any computation.',
    loop_start,
)
if loop_start < 0 or downstream < 0:
    raise RuntimeError("extract per-image PP116 prefix not found")

new_prefix = '''    for stem in tqdm(
        stems,
        desc="ADE weak labels -> pred obj -> crop -> DINO cache",
    ):
        stats["requested_images"] += 1

        img_path = find_by_stem(image_root, stem, image_suffixes)
        part_path = find_by_stem(part_root, stem, mask_suffixes)

        if img_path is None or part_path is None:
            stats["missing_pair"] += 1
            continue

        pil = load_rgb(img_path)
        part_mask = load_label_mask(part_path)

        if part_mask.shape != (pil.height, pil.width):
            raise ValueError(
                f"{stem}: part mask shape {part_mask.shape} "
                f"!= RGB {(pil.height, pil.width)}"
            )

        # Weak-label extraction ONLY: keep the set of IDs, never positions.
        present_part_ids = sorted(
            int(x)
            for x in np.unique(part_mask)
            if int(x) != int(args.part_ignore)
        )

        invalid = [
            pid for pid in present_part_ids
            if pid < 0 or pid >= len(PART_CLASSES)
        ]
        if invalid:
            raise ValueError(
                f"{stem}: invalid ADE234 part IDs {invalid}"
            )

        if not present_part_ids:
            stats["no_part_label"] += 1
            continue

        object_ids = sorted({
            int(PART_TO_OBJECT[pid])
            for pid in present_part_ids
        })

        if not object_ids:
            stats["no_object_label"] += 1
            continue

        object_names = [
            OBJECT_CLASSES[cid]
            for cid in object_ids
        ]

        weak_parts: dict[int, tuple[list[int], list[str]]] = {}

        for cid in object_ids:
            pids = [
                pid
                for pid in present_part_ids
                if int(PART_TO_OBJECT[pid]) == cid
            ]
            pnames = [
                PART_CLASSES[pid]
                for pid in pids
            ]
            if not pids:
                raise AssertionError(
                    f"{stem}: empty part set for {OBJECT_CLASSES[cid]}"
                )
            weak_parts[cid] = (pids, pnames)

        # Destroy the GT spatial array before any Talk2DINO/crop/DINO operation.
        del part_mask

        foreign_part_audit: dict[str, list[int]] = {}

'''
s = s[:loop_start] + new_prefix + s[downstream:]

s = s.replace(
    "# Weak labels derived from GT IDs only; no positions retained.",
    "# ADE image-level part-presence labels only; no GT positions retained.",
)

meta_start = s.find(
    '            "format": "pascalpart116_weaklabels_predobj_cropaug_v1",'
)
meta_end = s.find(
    '            "object_prediction_candidate_scope":',
    meta_start,
)
if meta_start < 0 or meta_end < 0:
    raise RuntimeError("extract metadata PP116 block not found")

ade_meta = '''            "format": "ade20kpart234_weaklabels_predobj_cropaug_v1",
            "protocol": (
                "ADE20K-Part-234 GT part masks are read only through np.unique "
                "to derive image-level semantic part presence and parent-object "
                "presence. No GT pixel location, object mask, point, or box is "
                "used by Talk2DINO foreground prediction, cropping, DINO patch "
                "selection, RelProto construction, or W training. "
                "cropaug_box_xyxy, pred_obj_mask_patch, and cropaug_patch_tokens "
                "are all derived from the frozen Talk2DINO predicted object mask."
            ),
            "part_ignore": int(args.part_ignore),
            "target_spatial_supervision": "none",
'''
s = s[:meta_start] + ade_meta + s[meta_end:]

s = re.sub(r'\n\s*"object_mask_encoding": args\.obj_mask_mode,', '', s)
s = re.sub(r'\n\s*"object_ignore": int\(args\.obj_ignore\),', '', s)
s = re.sub(r'\n\s*"part_id_offset": int\(args\.part_id_offset\),', '', s)

s = s.replace("Pascal-Part-116", "ADE20K-Part-234")
s = s.replace("PascalPart116", "ADE20KPart234")

write(DST_EXTRACT, s)


# ---------------------------------------------------------------------
# 2. W trainer: source is PP116; only cardinality/messages change.
# ---------------------------------------------------------------------

s = SRC_TRAIN.read_text(encoding="utf-8")
s = s.replace(
    "Final Pred-Obj W trainer for Pascal-Part-116.",
    "Final Pred-Obj W trainer for ADE20K-Part-234.",
)
s = s.replace("Tensor[116, 512]", "Tensor[234, 512]")
s = s.replace("length 116", "length 234")
s = s.replace("[116,512]", "[234,512]")
s = s.replace("[116,768]", "[234,768]")
s = replace_once(s, "NUM_PARTS = 116", "NUM_PARTS = 234", "trainer NUM_PARTS")
s = s.replace("expected 116", "expected 234")
s = s.replace("[116,D]", "[234,D]")
s = s.replace("(116,768)", "(234,768)")
s = s.replace("[0,115]", "[0,233]")
write(DST_TRAIN, s)


# ---------------------------------------------------------------------
# 3. Bake: same implementation, only full text-bank row count/path.
# ---------------------------------------------------------------------

s = SRC_BAKE.read_text(encoding="utf-8")
s = s.replace(
    "Full 116-part raw CLIP bank used in Train-W",
    "Full 234-part raw CLIP bank used in Train-W",
)
s = s.replace(
    'default="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt"',
    'default="feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt"',
)
s = s.replace(
    'if tuple(raw_clip.shape) != (116, 512):',
    'if tuple(raw_clip.shape) != (234, 512):',
)
s = s.replace(
    'raise ValueError(f"raw CLIP shape {tuple(raw_clip.shape)} != (116,512)")',
    'raise ValueError(f"raw CLIP shape {tuple(raw_clip.shape)} != (234,512)")',
)
s = s.replace(
    '"full_real_text_rows_checked": 116,',
    '"full_real_text_rows_checked": 234,',
)
write(DST_BAKE, s)


# ---------------------------------------------------------------------
# 4. ADE234 MMSeg dataset
# ---------------------------------------------------------------------

dataset_code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from mmseg.datasets import DATASETS, CustomDataset

# 0..233 valid semantic parts; 65535 background/unlabeled.
PART_CLASSES = {PART_CLASSES!r}


def _voc_palette(n: int):
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
class ADE20KPart234Dataset(CustomDataset):
    CLASSES = PART_CLASSES
    PALETTE = _voc_palette(len(PART_CLASSES))

    def __init__(self, **kwargs):
        requested_ignore = int(kwargs.pop("ignore_index", 65535))
        requested_reduce = bool(kwargs.pop("reduce_zero_label", False))

        if requested_ignore != 65535:
            raise ValueError(
                f"ADE20K-Part-234 requires ignore_index=65535, "
                f"got {{requested_ignore}}"
            )
        if requested_reduce:
            raise ValueError(
                "ADE20K-Part-234 class 0 is valid; reduce_zero_label must be False"
            )

        super().__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            ignore_index=65535,
            reduce_zero_label=False,
            **kwargs,
        )

        assert len(self.CLASSES) == 234
        assert self.CLASSES[0] == "person's head"
        assert self.ignore_index == 65535
        assert self.reduce_zero_label is False

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        if self.ann_dir is None or not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(self.ann_dir)
'''
write(DST_DATASET, dataset_code)


base_cfg = '''# ADE20K-Part-234 full-image semantic part evaluation.
# 0..233 valid semantic parts; 65535 ignore; no label shift.

custom_imports = dict(
    imports=["segmentation.datasets.ade20kpart234_part"],
    allow_failed_imports=False,
)

dataset_type = "ADE20KPart234Dataset"
data_root = "/home/master/dataset/lyx/ADE20KPart234"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="FloatImage"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="annotations_detectron2_part/val",
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=65535,
        reduce_zero_label=False,
    )
)

test_cfg = dict(
    mode="slide",
    stride=(224, 224),
    crop_size=(448, 448),
)
'''
write(DST_BASECFG, base_cfg)


eval_cfg = '''evaluate:
  pamr: false
  bg_thresh: 0.4
  kp_w: 0.0
  pred_qual_path: null
  gt_qual_path: null
  eval_only: true
  template: sub_imagenet_template
  task:
    - ade20kpart234
  ade20kpart234: src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/ade20kpart234_part.py
'''

model_cfg = '''_base_: default.yml

model:
  clip_model_name: ViT-B/16
  keep_cls: false
  keep_end_seq: false
  model_name: dinov2_vitb14_reg
  proj_class: vitb_mlp_infonce
  proj_model: ProjectionLayer
  proj_name: vitb_mlp_infonce
  resize_dim: 448
  type: DINOText
  use_avg_text_token: false
'''

write(DST_EVALCFG, eval_cfg)
write(DST_MODEL_CFG, model_cfg)


# ---------------------------------------------------------------------
# 5. Compile and guarantee trainer algorithm functions are unchanged.
# ---------------------------------------------------------------------

for p in (DST_EXTRACT, DST_TRAIN, DST_BAKE, DST_DATASET):
    py_compile.compile(str(p), doraise=True)
    print("[py_compile PASS]", p.relative_to(ROOT))


def function_source(path: Path, names: list[str]):
    txt = path.read_text(encoding="utf-8")
    tree = ast.parse(txt)
    lines = txt.splitlines(keepends=True)
    got = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            got[node.name] = "".join(lines[node.lineno - 1: node.end_lineno])
    return got


core = [
    "normalize_last",
    "relative_scores",
    "build_relproto",
    "orthogonal_retract_",
    "matrix_stats",
    "batch_forward",
    "do_train",
]
a = function_source(SRC_TRAIN, core)
b = function_source(DST_TRAIN, core)

missing = [x for x in core if x not in a or x not in b]
if missing:
    raise RuntimeError(f"core functions missing: {missing}")

changed = [x for x in core if a[x] != b[x]]
if changed:
    raise RuntimeError(
        "ADE trainer changed core algorithm functions: "
        + ", ".join(changed)
    )

print("[core trainer identical to PP116] PASS")

for p in (DST_EXTRACT, DST_TRAIN, DST_BAKE):
    if "partimagenet" in p.read_text(encoding="utf-8").lower():
        raise RuntimeError(f"PartImageNet residue found in {p.name}")

print("[no PartImageNet residue] PASS")
print("=" * 72)
print("ADE20K-Part-234 install from PP116 mainline: PASS")
print("=" * 72)
