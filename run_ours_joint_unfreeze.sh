#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
DATASET="${1:?Usage: bash run_ours_joint_unfreeze.sh pp116|ade234 [smoke|full]}"
MODE="${2:-smoke}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo 'Mode must be smoke or full' >&2; exit 2
fi
if [[ "$DATASET" == pp116 ]]; then
  TRAIN_SCRIPT=train_pp116_ours_unfreeze_projector.py
  CACHE=feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth
  TEXT=feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt
  DATA_OPTIONS=(--part_setting presence)
elif [[ "$DATASET" == ade234 ]]; then
  TRAIN_SCRIPT=train_ade234_ours_unfreeze_projector.py
  CACHE=feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug.pth
  TEXT=feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt
  DATA_OPTIONS=()
else
  echo "Unsupported dataset $DATASET" >&2; exit 2
fi
EXTRA=()
if [[ "$MODE" == smoke ]]; then
  EXTRA=(--epochs 1 --max_annotations 64)
else
  EXTRA=(--epochs 100)
fi
OUT="final_exp/relproto_alignment/${DATASET}_ours_jointP_lr1e6_Wlr1e5_lambda4_${MODE}_seed123"
if [[ -e "$OUT" ]]; then echo "Output already exists, not overwriting: $OUT" >&2; exit 2; fi
for F in "$TRAIN_SCRIPT" "$CACHE" "$TEXT" weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth; do
  if [[ ! -f "$F" ]]; then echo "Missing: $F" >&2; exit 2; fi
done
python -u "$TRAIN_SCRIPT" \
  --project_root . \
  --train_dataset "$CACHE" \
  --text_bank "$TEXT" \
  --weights weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --model_config configs/vitb_mlp_infonce.yaml \
  --out_dir "$OUT" \
  --soft_temperature 0.05 \
  --nce_temperature 0.07 \
  --lambda_struct 4.0 \
  --rank_delta 0.05 \
  --rank_margin 0.05 \
  --prototype_max_patches 8 \
  --batch_size 16 \
  --lr 1e-5 \
  --projector_lr 1e-6 \
  --projector_weight_decay 0 \
  --seed 123 \
  --expected_bg_thresh 0.55 \
  --require_pamr \
  --require_cache_projector_match \
  --device cuda \
  "${DATA_OPTIONS[@]}" "${EXTRA[@]}"
