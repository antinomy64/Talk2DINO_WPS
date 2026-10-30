#!/usr/bin/env bash
# Strictly new files. Never edits official trainer, model weights, cached data or GT.
set -Eeuo pipefail
ROOT="$(pwd)"
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
: "${NIGHT_TAG:=ear_mechanism_20261021_v1}"
: "${RUN_TRAIN:=1}"
: "${RUN_PROBES:=1}"
: "${RUN_EVAL:=1}"
: "${RUN_ALLCROP:=0}"
: "${RUN_BASELINE_PROBE:=1}"
: "${EVAL_EPOCHS:=3 10}"
: "${TRAIN_VARIANTS:=drop_ear_nostruct drop_ear_struct only_ear_nostruct full_nostruct only_ear_struct structure_only}"
: "${BASELINE_RUN:=final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep}"
OUT="final_exp/relproto_alignment/${NIGHT_TAG}"
REPORT="output/${NIGHT_TAG}"
mkdir -p "$OUT" "$REPORT"
log(){ printf '\n[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "$REPORT/run_status.log"; }
for path in train_relproto_alignemt.py train_pp116_softsingle_dual.py \
    feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
    feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
    weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
    configs/vitb_mlp_infonce.yaml; do
  if [[ ! -f "$path" ]]; then echo "MISSING REQUIRED FILE: $path" >&2; exit 2; fi
done
[[ -f "$BASELINE_RUN/W_epoch_010.pt" ]] || { echo "Missing baseline E10: $BASELINE_RUN" >&2;exit 2; }
python "$PKG/test_loss_equivalence.py" | tee "$REPORT/unit_test.log"
(
  echo "UTC: $(date -u +%FT%TZ)"
  echo "root: $ROOT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  git rev-parse HEAD 2>/dev/null || true
  sha256sum "$PKG/night_ear_train.py" "$PKG/night_ear_gradient_probe.py" "$PKG/run_ear_night.sh"
  echo "baseline_run: $BASELINE_RUN"
  echo "train_variants: $TRAIN_VARIANTS"
) > "$REPORT/provenance.txt"

run_grad(){
  local name="$1" dir="$2" lam="$3"
  log "Fixed GT-ear OFFLINE gradient probe: $name"
  python -u "$PKG/night_ear_gradient_probe.py" \
    --run "$dir" --output "$REPORT/gradient/$name" \
    --lambda-struct "$lam" --device cuda \
    2>&1 | tee "$REPORT/gradient_${name}.log"
}
run_eval(){
  local name="$1" dir="$2"
  for ep in $EVAL_EPOCHS; do
    local padded proj_name val_dir
    printf -v padded '%03d' "$ep"
    [[ -f "$dir/W_epoch_${padded}.pt" ]] || { echo "missing $dir/W_epoch_${padded}.pt" >&2;exit 3; }
    proj_name="vitb_mlp_infonce_coco2014_ear_${NIGHT_TAG}_${name}_e${padded}"
    val_dir="$REPORT/val/${name}/e${padded}"
    mkdir -p "$val_dir"
    if [[ ! -f "weights/${proj_name}.pth" ]]; then
      log "Bake $name E$ep"
      python -u bake_predobj_relproto_w_into_projector_final.py \
        --project_root . \
        --projector weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
        --w_checkpoint "$dir/W_epoch_${padded}.pt" \
        --text_bank feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
        --model_config configs/vitb_mlp_infonce.yaml \
        --output "weights/${proj_name}.pth" --device cuda \
        2>&1 | tee "$val_dir/bake.log"
    fi
    if [[ -f "$val_dir/EVALUATION_DONE" ]]; then
      log "Skip finished evaluation $name E$ep"
    else
      log "Evaluation $name E$ep (PAMR false, all 116 classes)"
      python -u -m torch.distributed.run --nproc_per_node=1 \
        --master_port="${MASTER_PORT:-32672}" \
        src/open_vocabulary_segmentation/main.py \
        --eval --output "$val_dir" \
        --eval_cfg src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml \
        --eval_base_cfg src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml \
        --opts "model.proj_name=${proj_name}" evaluate.pamr=false \
        2>&1 | tee "$val_dir/evaluation_stdout.log"
      touch "$val_dir/EVALUATION_DONE"
    fi
  done
}

if [[ "$RUN_BASELINE_PROBE" == 1 && "$RUN_PROBES" == 1 ]];then
  run_grad baseline_struct "$BASELINE_RUN" 4.0
fi
for spec in $TRAIN_VARIANTS; do
  case "$spec" in
    full_nostruct)      pos=full;lam=0.0;;
    drop_ear_struct)     pos=drop_ear;lam=4.0;;
    drop_ear_nostruct)   pos=drop_ear;lam=0.0;;
    only_ear_struct)     pos=only_ear;lam=4.0;;
    only_ear_nostruct)   pos=only_ear;lam=0.0;;
    structure_only)      pos=none;lam=4.0;;
    *) echo "unknown TRAIN_VARIANTS item: $spec" >&2; exit 2;;
  esac
  run="$OUT/$spec"
  if [[ "$RUN_TRAIN" == 1 ]];then
    if [[ -f "$run/W_epoch_010.pt" && -f "$run/summary.json" ]];then
      log "Reuse finished training: $spec"
    elif [[ -e "$run" ]];then
      echo "INCOMPLETE EXISTING RUN $run; do not overwrite. Inspect logs and rename NIGHT_TAG." >&2
      exit 3
    else
      log "Start training $spec (pos=$pos, lambda=$lam), 10 epochs"
      python -u "$PKG/night_ear_train.py" \
        --part_setting presence --positive_mode "$pos" \
        --project_root . \
        --train_dataset feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
        --text_bank feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
        --model_config configs/vitb_mlp_infonce.yaml \
        --weights weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
        --out_dir "$run" --soft_temperature 0.05 --nce_temperature 0.07 \
        --lambda_struct "$lam" --rank_delta 0.05 --rank_margin 0.05 \
        --epochs 10 --batch_size 16 --lr 1e-5 --seed 123 \
        --expected_bg_thresh 0.55 --require_pamr --require_cache_projector_match \
        --device cuda 2>&1 | tee "$REPORT/train_${spec}.log"
    fi
  fi
  [[ -f "$run/W_epoch_010.pt" ]] || { log "Missing run checkpoint: $spec"; exit 3; }
  if [[ "$RUN_PROBES" == 1 ]];then run_grad "$spec" "$run" "$lam";fi
  if [[ "$RUN_EVAL" == 1 ]];then run_eval "$spec" "$run";fi
  if [[ "$RUN_ALLCROP" == 1 ]];then
    log "All-Crop E0-E10 Ear GT audit: $spec (read-only audit, can be slow/large)"
    python -u audit_pp116_ear_patch_e0_e10.py \
      --run "$run" --lambda-struct "$lam" --start-epoch 0 --end-epoch 10 \
      --soft-temperature 0.05 --plots-per-ear 0 \
      --out-dir "$REPORT/allcrop/$spec" --device cuda \
      2>&1 | tee "$REPORT/allcrop_${spec}.log"
  fi
done
log "ALL REQUESTED RUNS COMPLETED. Summarize: python $PKG/night_ear_summarize.py --report $REPORT"
python "$PKG/night_ear_summarize.py" --report "$REPORT" || true
