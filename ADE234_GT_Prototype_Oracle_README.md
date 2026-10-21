# ADE234 GT Prototype Oracle (Orthogonal W) · Offline protocol

## What is being measured

- **Privileged training GT**: train split ADE234 `annotations_detectron2_part/train` global labels 0..233; 65535 is ignore; label 0 (`person's head`) is **valid**.
- From **all 7347 training images**, resize RGB to 448×448 bicubic + ImageNet normalise; DINOv2 ViT-B/14 reg (frozen and **local weights only**) returns `x_norm_patchtokens` 32×32×768. GT train part mask maps to grid using nearest neighbour; pool all tokens belonging to each part, then **L2-normalise the final global prototype**. No additional L2 token pre-normalisation.
- Project 234 raw CLIP vectors using frozen original COCO2014 text projector, then optimize a shared 768×768 **Orthogonal CE Oracle** map with **Adam 150 steps**, lr=1e-3, tau=0.07 and **SVD retraction after every step**. Valid GT prototypes are positive targets; **all 234 text classes** participate in logits. A missing-training-GT class has no positive training sample, and the runner prints missing parts.
- Bake W into the last projector layer; **1016 heldout val** images, 234-way complete vocabulary sliding 448/224, PAMR=false, GT labels NEVER read to fit W from val.
- Oracle supervision is stronger than weak part-presence and cannot be included as a weakly supervised main result. This design emulates the previously used PP116 **GT Orthogonal CE** diagnostic, although exact original PP116 feature-extraction source is not available for byte-for-byte verification.

## Put these files in repo root

`ade234_gtprototype_orthoracle.py`, `export_all_part_metrics.py`, `run_ade234_GTProto_OrthCE_oracle.sh`.

Run offline:

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

# The script refuses to download weights: inspect local checkpoint first.
find "$HOME/.cache/torch/hub/checkpoints" /home/master/weight/public -type f \
  \( -name 'dinov2_vitb14_reg4_pretrain.pth' -o -iname '*vitb14*reg*pretrain*.pth' \) 2>/dev/null

# If the pretrained checkpoint lives elsewhere, set:
# export DINO_CKPT=/EXISTING/local/dinov2_vitb14_reg4_pretrain.pth
export CUDA_VISIBLE_DEVICES=0
bash run_ade234_GTProto_OrthCE_oracle.sh \
  2>&1 | tee output/ade234_GTProto_oracle_pipeline.log
```

The bash pipeline is idempotent for fully completed phases (it will not overwrite bank/W/baked weights), but refuses to evaluate into an existing incomplete evaluation directory.

After evaluation, `export_all_part_metrics.py` produces:

- `output/pp116_ade234_six_methods_partwise/PP116_all_parts.csv` (116 rows)
- `output/pp116_ade234_six_methods_partwise/ADE234_all_parts.csv` (234 rows)
- `output/pp116_ade234_six_methods_partwise/summary.csv`.

Set `PP_GT_PROTO_JSON` to the actual **PP116 GT Orthogonal CE** validation JSON before launching the bash pipeline to populate every PP116 column; otherwise the PP116 GT Prototype per-part columns stay blank and you can rerun only `export_all_part_metrics.py --pp-gtproto <actual>.json` after finding it.

## Each Part metrics

- `*_IoU_pct`: IoU for one part. `mIoU` is the macro average of part IoU, **not** a classwise quantity.
- `*_Acc_pct`: classwise recall `TP/(TP+FN)`; `mAcc` is the macro average, **not** a classwise quantity.
- `*_aAcc_contribution_pct`: `TP_part / sum_all_valid_GT_pixels`, and **NOT** independent classwise aAcc. The contributions sum to global aAcc and need the **exact evaluated val GT subset**. For a mismatch between val GT mask counts and 850/1016 evaluated images, these cells stay blank, with a warning, rather than incorrect values.

## Important implementation notes

- Training prototypes use only GT **train** labels and frozen DINO model; they are not identical to GT Part Union foreground Oracle (which filters 766 empty ADE crops).
- No GT masks/label shifting at inference; main.py 234-way output protocol unchanged.
- Earlier experimental PP116 GT Prototype Oracle: mIoU 47.11 / mAcc 81.16 / aAcc 73.48. The ADE equivalent requires the above server run; do not report unmeasured values.
- All classes with absent validation GT have NaN/blank classwise recall, **not 0**.
