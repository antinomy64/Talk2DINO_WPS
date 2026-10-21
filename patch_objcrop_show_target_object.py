#!/usr/bin/env python3
"""Idempotent patch for visualize_objcrop_gt_predfg_triptych.py.
Show the target object category from cached part_category_id + dataset PART_CLASSES.
Do not modify other repo files or any model/inference logic.
"""
from pathlib import Path
import py_compile

p = Path('visualize_objcrop_gt_predfg_triptych.py')
if not p.is_file():
    raise FileNotFoundError(p)
s = p.read_text(encoding='utf-8')
initial = s

def replace_exact(before, after, name):
    global s
    if after in s:
        print(f'[SKIP] {name}')
    elif s.count(before) == 1:
        s = s.replace(before, after, 1)
        print(f'[PATCH] {name}')
    else:
        raise RuntimeError(f'Cannot uniquely locate {name}: found {s.count(before)} matches')

replace_exact('import argparse\n', 'import argparse\nimport ast\n', 'ast import')
helper = '''

def get_class_names_for_crop_vis(dataset: str) -> list[str]:
    """Read exact dataset taxonomy, preserving the numeric part ID ordering."""
    if dataset == "pp116":
        taxonomy = Path("src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py")
        source = ast.parse(taxonomy.read_text(encoding="utf-8"), filename=str(taxonomy))
        classes = None
        for node in ast.walk(source):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PART_CLASSES" for t in node.targets):
                classes = list(ast.literal_eval(node.value))
                break
        if classes is None:
            raise RuntimeError("PART_CLASSES not found in PP116 taxonomy")
        n_expected = 116
    elif dataset == "ade234":
        from ade20kpart234_taxonomy import PART_CLASSES
        classes = list(PART_CLASSES)
        n_expected = 234
    else:
        raise ValueError(dataset)
    if len(classes) != n_expected or len(set(classes)) != n_expected:
        raise ValueError(f"{dataset}: expected {n_expected} unique canonical part names, found {len(classes)}")
    if any("'s " not in str(x) for x in classes):
        raise ValueError(f"{dataset}: unrecognized object-part label format")
    return [str(x) for x in classes]


def get_target_object_for_crop(part_ids: list[int], class_names: list[str], crop_idx: int) -> str:
    """Derive object from part IDs; fail on ambiguous cross-object crops."""
    if not part_ids:
        raise ValueError(f"Crop {crop_idx}: empty part_category_id, cannot determine object")
    if any(x < 0 or x >= len(class_names) for x in part_ids):
        raise ValueError(f"Crop {crop_idx}: invalid part_category_id: {part_ids}")
    objects = {class_names[p].split("'s ", 1)[0] for p in part_ids}
    if len(objects) != 1:
        raise ValueError(f"Crop {crop_idx}: multiple object categories {sorted(objects)} from part IDs {part_ids}")
    return next(iter(objects))
'''
replace_exact('\ndef main():\n', helper+'\ndef main():\n', 'canonical object lookup helper')
replace_exact('    anns = load_cache_as_list(args.cache)\n',
              '    anns = load_cache_as_list(args.cache)\n    class_names = get_class_names_for_crop_vis(args.dataset)\n',
              'load part taxonomy once')
replace_exact('        rgb_crop = crop_and_resize_rgb(img_path, box, args.side)\n',
              '        object_name = get_target_object_for_crop(present_parts, class_names, idx)\n'
              '        rgb_crop = crop_and_resize_rgb(img_path, box, args.side)\n',
              'label object per crop')
replace_exact('        left = add_title(Image.fromarray(rgb_crop), f"{image_id} | Obj Crop")\n',
              '        left = add_title(Image.fromarray(rgb_crop), f"Obj Crop | {object_name}")\n',
              'object in left title')
replace_exact('alpha=args.alpha)), "GT Foreground")\n',
              'alpha=args.alpha)), f"GT FG | {object_name}")\n',
              'object in GT title')
replace_exact('alpha=args.alpha)), "Pred Foreground")\n',
              'alpha=args.alpha)), f"Pred FG | {object_name}")\n',
              'object in Pred title')
replace_exact('        footer_lines = [\n',
              '        footer_lines = [\n            f"OBJECT={object_name}  |  image_id={image_id}  |  crop={idx}",\n',
              'add object ID footer')
replace_exact('        rows.append({\n            "idx": idx,\n',
              '        rows.append({\n            "idx": idx,\n            "object": object_name,\n',
              'CSV object value')
replace_exact('            "idx", "image_id", "out_name",\n',
              '            "idx", "object", "image_id", "out_name",\n',
              'CSV object header')
replace_exact('print(f"[SAVE] {saved} {out_name} IoU={iou:.4f} Dice={dice:.4f}")',
              'print(f"[SAVE] {saved} {out_name} object={object_name} IoU={iou:.4f} Dice={dice:.4f}")',
              'object in progress log')

if s != initial:
    # compile in memory before overwriting original generated script
    compile(s, str(p), 'exec')
    p.write_text(s, encoding='utf-8')
py_compile.compile(str(p), doraise=True)
print('[PASS] Syntax. Target object is printed on each panel, footer, CSV, and progress output.')
