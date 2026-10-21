#!/usr/bin/env bash
# PRIVILEGED train-GT visual prototype direct dense segmentation (no trained W).
# Usage: bash run_direct_gt_visual_oracles.sh pp116 | ade234 | all
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WHAT="${1:-all}"
case "$WHAT" in
  all) SETS=(pp116 ade234) ;;
  pp116|ade234) SETS=("$WHAT") ;;
  *) echo "Usage: bash $0 {pp116|ade234|all}" >&2; exit 2 ;;
esac

test -f direct_gt_visual_proto_eval.py || { echo 'Missing direct_gt_visual_proto_eval.py'; exit 2; }
python direct_gt_visual_proto_eval.py self-test

for DATA in "${SETS[@]}"; do
    LOGDIR="output/direct_gt_visual_${DATA}_logs"
    mkdir -p "$LOGDIR"
    if [ -e "$LOGDIR/eval.log" ]; then
        echo "REFUSE: $LOGDIR/eval.log exists; choose a new output/report directory if rerunning." >&2
        exit 3
    fi
    echo "==============================================="
    echo "   DIRECT GT VISUAL PROTOTYPE — $DATA"
    echo "==============================================="
    python -u direct_gt_visual_proto_eval.py prepare --dataset "$DATA" \
      2>&1 | tee "$LOGDIR/prepare.log"

    # Calls the UNMODIFIED main.py evaluator internally in the same torchrun process.
    # No CLIP text/W are used to compute the classification part vectors.
    python -u -m torch.distributed.run \
      --nproc_per_node=1 --master_port="$([[ "$DATA" == pp116 ]] && echo 32971 || echo 32972)" \
      direct_gt_visual_proto_eval.py eval --dataset "$DATA" \
      2>&1 | tee "$LOGDIR/eval.log"

    python -u direct_gt_visual_proto_eval.py export \
      --dataset "$DATA" \
      --eval-log "$LOGDIR/eval.log" \
      --out-dir "$LOGDIR/reports"

    echo "[COMPLETE] $DATA; CSV: $LOGDIR/reports/${DATA^^}_GTVisualDirect_per_part.csv"
    echo "[COMPLETE] log: $LOGDIR/eval.log"
done
