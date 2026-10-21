#!/usr/bin/env bash
# Offline diagnostic: ADE234 full-image GT prototypes -> orthogonal CE W -> bake -> evaluate -> classwise CSV.
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_DEVICE="cuda"
EXP="output/ade234_gt_prototype_orthogonal_CE_oracle"
mkdir -p "$EXP"

# Local weights only; NEVER re-download/rewrite official repo or weights.
DINO_REPO="${DINO_REPO:-$HOME/.cache/torch/hub/facebookresearch_dinov2_main}"
DINO_CKPT="${DINO_CKPT:-$HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth}"
GT_ROOT="${ADE_TRAIN_GT:-../../../dataset/lyx/ADE20KPart234/annotations_detectron2_part/train}"
IMG_ROOT="${ADE_TRAIN_IMG:-../../../dataset/lyx/ADE20KPart234/images/train}"

BANK="feature/ade20kpart234_gt_visual_structure/gt_dinov2_vitb14reg_train_fullimg_patchpool_ORACLE.pt"
W="$EXP/diagnostic_GT_Orthogonal_CE_W.pt"
PROJ="vitb_mlp_infonce_coco2014_ade234_GTProto_OrthCE_150step"
PROJ_FILE="weights/${PROJ}.pth"
EVAL_OUT="output/ade234_GTProto_OrthCE_150step"

for f in ade234_gtprototype_orthoracle.py export_all_part_metrics.py train_relproto_alignemt.py ade20kpart234_taxonomy.py; do
  test -f "$f" || { echo "Missing $f"; exit 2; }
done

python ade234_gtprototype_orthoracle.py self-test

if [ ! -f "$BANK" ]; then
  test -f "$DINO_REPO/hubconf.py" || { echo "Missing DINO hubconf: $DINO_REPO"; exit 2; }
  test -f "$DINO_CKPT" || { echo "Missing local DINO weights (no automatic download): $DINO_CKPT"; exit 2; }
  echo '============= 1. GT PROTOTYPE EXTRACT: TRAIN ONLY ============='
  python -u ade234_gtprototype_orthoracle.py extract \
      --images-root "$IMG_ROOT" --gtpart-root "$GT_ROOT" \
      --dino-repo "$DINO_REPO" --dino-weights "$DINO_CKPT" \
      --batch-size 8 --device "$GPU_DEVICE" --out "$BANK" \
      2>&1 | tee "$EXP/extract.log"
else
  echo "[RESUME] Existing GT training prototype bank: $BANK"
fi

if [ ! -f "$W" ]; then
  echo '============= 2. TRAIN GT ORTHOGONAL CE ORACLE ============='
  python -u ade234_gtprototype_orthoracle.py fit \
      --bank "$BANK" --device "$GPU_DEVICE" \
      --steps 150 --temperature 0.07 --lr 0.001 --seed 123 \
      --out "$W" \
      2>&1 | tee "$EXP/orthCE_fit.log"
else
  echo "[RESUME] Existing Oracle W: $W"
fi

if [ ! -f "$PROJ_FILE" ]; then
  echo '============= 3. BAKE PROJECTOR ============='
  python -u ade234_gtprototype_orthoracle.py bake \
      --w "$W" --device "$GPU_DEVICE" \
      --out "$PROJ_FILE" \
      2>&1 | tee "$EXP/bake.log"
else
  echo "[RESUME] Existing baked projector: $PROJ_FILE"
fi

EVAL_JSON="segmentation_results/${PROJ}.json"
if [ ! -f "$EVAL_JSON" ]; then
  if [ -e "$EVAL_OUT" ]; then
    echo "Refusing to mix output into existing evaluation dir lacking completed JSON: $EVAL_OUT"
    exit 3
  fi
  echo '============= 4. EVALUATE: 1016 VAL, 234-WAY, PAMR=FALSE ============='
  python -m torch.distributed.run \
      --nproc_per_node=1 --master_port=32946 \
      src/open_vocabulary_segmentation/main.py \
      --eval --output "$EVAL_OUT" \
      --eval_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml \
      --eval_base_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml \
      --opts "model.proj_name=$PROJ" "evaluate.pamr=false" \
      2>&1 | tee "$EXP/eval.log"
else
  echo "[RESUME] Existing Oracle evaluation JSON: $EVAL_JSON"
fi

echo '============= 5. EXPORT PER-PART RESULTS ============='
# Set PP_GT_PROTO_JSON to the verified PP116 trained-GT CE Oracle evaluation JSON if needed.
PP_ARGS=()
if [[ -n "${PP_GT_PROTO_JSON:-}" ]]; then PP_ARGS+=(--pp-gtproto "$PP_GT_PROTO_JSON"); fi
python export_all_part_metrics.py \
    "${PP_ARGS[@]}" \
    --out-dir output/pp116_ade234_six_methods_partwise

grep -A 5 'Summary:' "$EXP/eval.log" || true
echo 'ALL FINISHED: ADE234 GT Prototype Orthogonal CE Oracle + per-part reports'
