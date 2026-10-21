#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
CACHE="feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug.pth"
TEXT="feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt"
BASE="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
CFG="configs/vitb_mlp_infonce.yaml"
SIDE="output/ade234_GTPartUnionFG_oracle/gt_partunion_valid_masks.pt"
RUN="final_exp/relproto_alignment/ade234_GTPartUnionFG_ORACLE_soft_structure_lam4_100ep"
P_NAME="vitb_mlp_infonce_coco2014_ade234_GTPartUnionFG_ORACLE_lam4_e100"
P_OUT="weights/${P_NAME}.pth"
EVAL="output/ade234_GTPartUnionFG_ORACLE_lam4_e100"
IMAGE_ROOT="${ADE_IMAGE_ROOT:-/home/master/dataset/lyx/ADE20KPart234/images/train}"
DEV="${ADE_GPU:-0}"

build_args() {
  PART_ROOT="${ADE_GT_PART_ROOT:-/home/master/dataset/lyx/ADE20KPart234/annotations_detectron2_part/train}"
  PART_OFFSET="${ADE_PART_ID_OFFSET:-0}"
  MASK_ARGS=(--cache "$CACHE" --gt-part-root "$PART_ROOT" --mask-mode part_union --part-id-offset "$PART_OFFSET" --images-root "$IMAGE_ROOT")
}

case "${1:-}" in
  inspect)
    python - <<'PY'
from pathlib import Path
import torch
p=Path('feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug.pth')
if not p.is_file():raise FileNotFoundError(p)
x=torch.load(p,map_location='cpu',weights_only=False)
r=x if isinstance(x,(list,tuple)) else next((x[k] for k in ('annotations','records','samples','items') if isinstance(x.get(k),(list,tuple))),None)
if not r:raise RuntimeError('No ADE crop records')
print('[CACHE] records=',len(r))
for k,v in r[0].items():
    print(' ',k,tuple(v.shape) if torch.is_tensor(v) else repr(v)[:130])
PY
    echo '[SEARCH] ADE PART GT masks:'
    find -L /home/master/dataset/lyx/ADE20KPart234/annotations_detectron2_part -maxdepth 2 -type d -print 2>/dev/null || true
    ;;
  preflight)
    build_args
    python build_ade234_gtpartunion_valid_masks.py "${MASK_ARGS[@]}" --preflight "${ADE_PREFLIGHT_ROWS:-32}"
    ;;
  build)
    build_args
    python build_ade234_gtpartunion_valid_masks.py "${MASK_ARGS[@]}" \
      --empty-policy "${ADE_EMPTY_POLICY:-error}" --out "$SIDE"
    ;;
  train)
    if [[ ! -f "$SIDE" ]]; then echo "Missing $SIDE; run build first" >&2; exit 1; fi
    if [[ -e "$RUN" ]]; then echo "Refusing to overwrite training directory $RUN" >&2; exit 1; fi
    CUDA_VISIBLE_DEVICES="$DEV" PYTHONUNBUFFERED=1 python -u train_ade234_gtpartunion_fg_oracle.py \
      --gtmask-bank "$SIDE" \
      --project_root . --train_dataset "$CACHE" --text_bank "$TEXT" \
      --model_config "$CFG" --weights "$BASE" --out_dir "$RUN" \
      --soft_temperature 0.05 --nce_temperature 0.07 --lambda_struct 4.0 \
      --rank_delta 0.05 --rank_margin 0.05 --prototype_max_patches 8 \
      --epochs 100 --batch_size 16 --lr 1e-5 --seed 123 \
      --expected_bg_thresh 0.55 --require_pamr --require_cache_projector_match --device cuda
    ;;
  bake)
    if [[ ! -f "$RUN/W_epoch_100.pt" ]]; then echo "Missing $RUN/W_epoch_100.pt" >&2; exit 1; fi
    if [[ -e "$P_OUT" ]]; then echo "Refusing overwrite $P_OUT" >&2; exit 1; fi
    CUDA_VISIBLE_DEVICES="$DEV" python bake_ade20kpart234_relproto_w.py \
      --project_root . --projector "$BASE" --w_checkpoint "$RUN/W_epoch_100.pt" \
      --text_bank "$TEXT" --model_config "$CFG" --output "$P_OUT" --device cuda
    ;;
  eval)
    if [[ ! -f "$P_OUT" ]]; then echo "Missing $P_OUT; run bake first" >&2; exit 1; fi
    mkdir -p "$EVAL"
    CUDA_VISIBLE_DEVICES="$DEV" python -m torch.distributed.run \
      --nproc_per_node=1 --master_port=32643 \
      src/open_vocabulary_segmentation/main.py --eval --output "$EVAL" \
      --eval_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml \
      --eval_base_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml \
      --opts model.proj_name="$P_NAME" evaluate.pamr=false
    ;;
  *) echo 'Usage: bash run_ade234_gtpartunion_fg_oracle.sh {inspect|preflight|build|train|bake|eval}' >&2; exit 2;;
esac
