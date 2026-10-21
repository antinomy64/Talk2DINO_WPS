#!/usr/bin/env bash
# New script ONLY. No official repository file is overwritten.
# Usage: bash run_pp116_relation_ablation.sh train0|verify4|eval|all|train4
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODE="${1:-all}"
R0=final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda0_presence_100ep
R4=final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep
BASE=weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth
TEXT=feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt
CACHE=feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth
CFG=configs/vitb_mlp_infonce.yaml
PREFIX=vitb_mlp_infonce_coco2014_DIAGREL

check4() {
  test -s "$R4/W_epoch_100.pt" || { echo "Missing existing lambda=4 run: $R4" >&2; exit 1; }
  python - "$R4/W_epoch_100.pt" <<'PY'
import sys, math, torch
p=sys.argv[1]
c=torch.load(p,map_location='cpu',weights_only=False)   # User's own trusted checkpoint
args=c.get('args')
if args is None: raise SystemExit('Cannot verify checkpoint args')
def get(k): return args.get(k) if isinstance(args,dict) else getattr(args,k,None)
want={'lambda_struct':4.0,'soft_temperature':0.05,'nce_temperature':0.07,
      'rank_delta':0.05,'rank_margin':0.05,'seed':123,'batch_size':16,
      'epochs':100,'lr':1e-5,'part_setting':'presence'}
for key,v in want.items():
    observed=get(key)
    if observed is None or (isinstance(v,(int,float)) and not isinstance(v,bool) and not math.isclose(float(observed),float(v),rel_tol=1e-7,abs_tol=1e-9)) or (isinstance(v,str) and observed!=v):
        raise SystemExit(f'lambda4 checkpoint incompatible {key}: got={observed!r}, expected={v!r}')
print('[VERIFY] Existing lambda=4 checkpoint hyperparameters match')
print('[VERIFY] Check original training code provenance manually if it was modified after run')
PY
}

train_arm() {
  local lam="$1" run="$2"
  if [ -e "$run" ]; then
    echo "[REFUSE] Output path already exists: $run. Do not overwrite an existing run." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_pp116_softsingle_dual.py \
    --part_setting presence \
    --project_root . \
    --train_dataset "$CACHE" \
    --text_bank "$TEXT" \
    --model_config "$CFG" \
    --weights "$BASE" \
    --out_dir "$run" \
    --soft_temperature 0.05 \
    --nce_temperature 0.07 \
    --lambda_struct "$lam" \
    --rank_delta 0.05 \
    --rank_margin 0.05 \
    --prototype_max_patches 8 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-5 \
    --seed 123 \
    --expected_bg_thresh 0.55 \
    --require_pamr \
    --require_cache_projector_match \
    --device cuda
}

# Deterministic experiment names; evaluation artifacts are isolated by lambda and epoch.
evaluate_arm() {
  local tag="$1" run="$2" epoch="$3"
  printf -v ep '%03d' "$epoch"
  local name="${PREFIX}_${tag}_e${ep}"
  local result="segmentation_results/${name}.json"
  local baked="weights/${name}.pth"
  local out="output/pp116_DIAGREL_${tag}_e${ep}"
  if [ -s "$result" ]; then echo "[SKIP eval] $result"; return; fi
  if [ ! -e "$baked" ]; then
    if [ "$epoch" -eq 0 ]; then
      cp -- "$BASE" "$baked"  # NEW diagnostic name only; original is never overwritten.
    else
      CUDA_VISIBLE_DEVICES=0 python bake_predobj_relproto_w_into_projector_final.py \
        --project_root . \
        --projector "$BASE" \
        --w_checkpoint "$run/W_epoch_${ep}.pt" \
        --text_bank "$TEXT" \
        --model_config "$CFG" \
        --output "$baked" \
        --device cuda
    fi
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=32625 \
    src/open_vocabulary_segmentation/main.py \
    --eval --output "$out" \
    --eval_cfg src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml \
    --eval_base_cfg src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml \
    --opts "model.proj_name=${name}" evaluate.pamr=false \
    2>&1 | tee "$out/eval.log"
  if [ ! -s "$result" ]; then echo "[ERROR] Expected evaluation result missing: $result" >&2;exit 1;fi
}

case "$MODE" in
  train0)
    check4
    train_arm 0.0 "$R0"
    ;;
  train4)
    # Only if the original lambda=4 run cannot be verified; uses a NEW directory.
    R4=final_exp/relproto_alignment/pp116_DIAGNOSTIC_lambda4_presence_100ep
    train_arm 4.0 "$R4"
    echo "[INFO] For eval mode, replace R4 path with the new run in this script."
    ;;
  verify4)check4;;
  eval)
    check4
    test -s "$R0/W_epoch_100.pt" || { echo "lambda=0 run not trained" >&2;exit 1;}
    for epoch in 0 10 20 30 40 50 60 70 80 90 100;do
      if [ "$epoch" -gt 0 ];then
        printf -v ep '%03d' "$epoch"
        test -s "$R0/W_epoch_${ep}.pt" && test -s "$R4/W_epoch_${ep}.pt" || { echo "missing E${ep} checkpoint" >&2;exit 1; }
      fi
      evaluate_arm lam0 "$R0" "$epoch"
      evaluate_arm lam4 "$R4" "$epoch"
    done
    ;;
  all)
    check4
    if [ ! -e "$R0" ];then train_arm 0.0 "$R0";else echo '[SKIP training] Lambda0 path exists; validate checkpoints before reusing';fi
    bash "$0" eval
    ;;
  *) echo "Usage: bash $0 train0|verify4|eval|all|train4" >&2;exit 2;;
esac
