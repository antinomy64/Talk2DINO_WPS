# Clean VOC116-part evaluator

Hard-locked semantic protocol:

- valid part IDs: `0..115`
- background / ignore: `255`
- `ignore_index=255`
- `reduce_zero_label=False`
- `CLASSES` has exactly 116 part names and no `"background"`
- Talk2DINO therefore infers `with_bg=False`
- final prediction has 116 part channels
- pixels with GT=255 are excluded from MMSeg mIoU
- inference geometry follows official Talk2DINO ViT-B Pascal VOC:
  448 crop, 224 stride, slide mode

Install from the extracted bundle:

```bash
bash install.sh ~/code/lyx/Talk2DINO_official_bg
```

Audit:

```bash
cd ~/code/lyx/Talk2DINO_official_bg
python tools/audit_voc116_part_eval.py --repo_root .
```

Expected final marker:

```text
VOC116_PART_EVAL_AUDIT_PASS
```
