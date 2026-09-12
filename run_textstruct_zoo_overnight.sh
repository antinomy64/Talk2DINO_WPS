#!/usr/bin/env bash
# =============================================================================
# Overnight Talk2DINO experiment — Spearman-first version
#
# NO LLaMA.
#
# What runs:
#
#   A) GTProto ORACLE
#      - train from current initial projector using TRAIN GT part prototypes
#      - report the same three Spearmans:
#          RAW CLIP vs GT
#          INITIAL  vs GT
#          ORACLE   vs GT
#      - run ONE official-style PascalPart116 direct mIoU evaluation
#
#   B) 11 repository-CLIP text-structure candidates
#      - same initial projector
#      - same COCO InfoNCE continuation
#      - same epochs/lr/optimizer/seed
#      - only structure loss changes
#      - NO segmentation evaluation
#      - NO RelProto-W
#      - for each candidate report only:
#          RAW CLIP vs GT
#          INITIAL  vs GT
#          CANDIDATE vs GT
#
# GT visual prototype bank is crop-balanced TRAIN GT DINO prototypes.
# For all 11 non-oracle candidates GT is audit-only and NEVER enters training.
# =============================================================================

set -uo pipefail
umask 022

ROOT="${ROOT:-$(pwd)}"
GPU="${GPU:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="${OUT_DIR:-output/textstruct_spearman_${TS}}"

EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
BASE_STRUCTURE_WEIGHT="${BASE_STRUCTURE_WEIGHT:-1e-4}"
CALIBRATE_GRAD="${CALIBRATE_GRAD:-1}"

ORACLE_STEPS="${ORACLE_STEPS:-2000}"
ORACLE_LR="${ORACLE_LR:-1e-5}"

RANK_TAU="${RANK_TAU:-0.05}"
NEIGHBOR_TAU="${NEIGHBOR_TAU:-0.07}"
CONF_DELTA="${CONF_DELTA:-0.03}"
CONF_ALPHA="${CONF_ALPHA:-0.5}"
CONF_MARGIN_MAX="${CONF_MARGIN_MAX:-0.10}"
CONF_MAX_TRIPLETS="${CONF_MAX_TRIPLETS:-256}"
RKD_ANGLE_TRIPLETS="${RKD_ANGLE_TRIPLETS:-16384}"
TOPK="${TOPK:-5}"
TOPK_MARGIN="${TOPK_MARGIN:-0.03}"

EVAL_BG_THRESH="${EVAL_BG_THRESH:-0.4}"
PORT="${PORT:-29920}"

cd "$ROOT" || exit 2
mkdir -p "$RUN"/{logs,train_metrics,audits,eval}
STATUS="$RUN/status.tsv"
: > "$STATUS"
exec > >(tee -a "$RUN/master.log") 2>&1

MODEL_CONFIG="configs/vitb_mlp_infonce.yaml"
TEXT_BANK="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt"
GT_BANK="feature/pascalpart116_gt_visual_structure/gt_dinov2_vitb14reg_train_objcrop_x1p2.pt"

INITIAL="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
EXISTING_PARTSTRUCT="weights/vitb_mlp_infonce_coco2014_clean_ft10_partstruct_rawpath_w1e4_lr1e5.pth"

COCO_TRAIN="feature/coco2014_official_reproduce/captions_train2014_ready.pth"
COCO_VAL="feature/coco2014_official_reproduce/captions_val2014_ready.pth"

EVAL_CFG="src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml"
EVAL116="src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml"

METHODS=(
  object_rank
  global_row_rank
  object_neighbor_kl
  global_neighbor_kl
  object_confidence_rank
  global_confidence_rank
  rkd_distance
  rkd_angle
  gram_mse
  gram_corr
  topk_margin
)

record() {
  local st="$1" name="$2" note="${3:-}"
  note="${note//$'\t'/ }"
  note="${note//$'\n'/ }"
  printf '%s\t%s\t%s\n' "$st" "$name" "$note" >> "$STATUS"
}

run_step() {
  local name="$1"; shift
  local log="$RUN/logs/${name}.log"
  echo
  echo "========================================================================"
  echo "[STEP] $name"
  echo "========================================================================"
  "$@" > >(tee "$log") 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    record PASS "$name" "$log"
    echo "[PASS] $name"
  elif [[ $rc -eq 3 ]]; then
    record WARN "$name" "$log"
    echo "[WARN/SKIP] $name"
  else
    record FAIL "$name" "$log"
    echo "[FAIL rc=$rc] $name"
  fi
  return 0
}

weight_for() {
  local label="$1"
  case "$label" in
    initial) echo "$INITIAL" ;;
    existing_partstruct) echo "$EXISTING_PARTSTRUCT" ;;
    gtproto_oracle) echo "weights/vitb_mlp_infonce_spearman_${TS}_gtproto_oracle.pth" ;;
    *) echo "weights/vitb_mlp_infonce_spearman_${TS}_${label}.pth" ;;
  esac
}

preflight() {
  local bad=0
  for p in \
    "$MODEL_CONFIG" "$TEXT_BANK" "$GT_BANK" "$INITIAL" \
    "$COCO_TRAIN" "$COCO_VAL" \
    "src/loss.py" \
    "src/loss_textstruct_zoo.py" \
    "src/train_util_textstruct_zoo.py" \
    "train_textstruct_zoo.py" \
    "audit_textstruct_checkpoint.py" \
    "train_gtproto_oracle.py" \
    "$EVAL_CFG" "$EVAL116"
  do
    if [[ -f "$p" ]]; then
      echo "[OK] $p"
    else
      echo "[MISS] $p"
      bad=1
    fi
  done

  CUDA_VISIBLE_DEVICES="$GPU" python - <<'PY' || bad=1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

  python -m py_compile \
    src/loss_textstruct_zoo.py \
    src/train_util_textstruct_zoo.py \
    train_textstruct_zoo.py \
    audit_textstruct_checkpoint.py \
    train_gtproto_oracle.py || bad=1

  # Actual checkpoint + actual text bank smoke-test for all 11 losses.
  CUDA_VISIBLE_DEVICES="$GPU" python - \
    "$MODEL_CONFIG" "$INITIAL" "$TEXT_BANK" "$GT_BANK" <<'PY' || bad=1
import importlib, math, sys
from pathlib import Path
import torch, yaml
from src.loss_textstruct_zoo import METHODS, build_text_structure_loss

cfg_path, init_path, bank_path, gt_path = map(Path, sys.argv[1:5])

with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
cls = getattr(
    importlib.import_module("src.model"),
    cfg["model"].get("model_class", "ProjectionLayer"),
)
model = cls.from_config(cfg["model"])

try:
    state = torch.load(str(init_path), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(init_path), map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=True)
model = model.cuda().eval()

try:
    gt = torch.load(str(gt_path), map_location="cpu", weights_only=False)
except TypeError:
    gt = torch.load(str(gt_path), map_location="cpu")
assert gt.get("protocol", {}).get("split") in (None, "", "train")
assert tuple(torch.as_tensor(gt["crop_balanced_prototypes"]).shape) == (116, 768)

for method in METHODS:
    crit = build_text_structure_loss(
        method, str(bank_path), seed=123
    ).cuda()
    model.zero_grad(set_to_none=True)
    loss = crit(model)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{method}: nonfinite loss {loss}")
    loss.backward()

    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f"{method}: nonfinite grad")
            g2 += float(p.grad.detach().float().pow(2).sum().item())
    gn = math.sqrt(g2)
    if not (math.isfinite(gn) and gn > 0):
        raise RuntimeError(f"{method}: invalid grad norm {gn}")
    print(
        f"[loss-smoke] {method:24s} "
        f"loss={float(loss.detach()):.8g} grad={gn:.8g}"
    )
    model.zero_grad(set_to_none=True)

print("ALL_11_TEXTSTRUCT_LOSSES_REAL_PROJECTOR_SMOKE_PASS")
PY

  [[ $bad -eq 0 ]]
}

echo
echo "========================================================================"
echo "[STEP] 00_preflight"
echo "========================================================================"
if preflight > >(tee "$RUN/logs/00_preflight.log") 2>&1; then
  record PASS 00_preflight "$RUN/logs/00_preflight.log"
  echo "[PASS] 00_preflight"
else
  record FAIL 00_preflight "$RUN/logs/00_preflight.log"
  echo "[FATAL] preflight failed; refusing overnight run."
  exit 2
fi

echo
echo "========================================================================"
echo "SPEARMAN-FIRST OVERNIGHT"
echo "========================================================================"
echo "start       : $(date -Is)"
echo "run         : $RUN"
echo "epochs/lr   : $EPOCHS / $LR"
echo "base lambda : $BASE_STRUCTURE_WEIGHT"
echo "grad calib  : $CALIBRATE_GRAD"
echo "Only GTProto Oracle gets PascalPart116 mIoU."
echo "All 11 text-only candidates get THREE Spearmans only."
echo "========================================================================"

train_method() {
  local method="$1"
  local weight; weight="$(weight_for "$method")"
  local extra=()
  [[ "$CALIBRATE_GRAD" == "1" ]] || extra+=(--no_calibrate_grad)

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train_textstruct_zoo.py \
    --repo_root . \
    --method "$method" \
    --model_config "$MODEL_CONFIG" \
    --train_dataset "$COCO_TRAIN" \
    --val_dataset "$COCO_VAL" \
    --feature_name disentangled_self_attn \
    --text_features ann_feats \
    --init_weight "$INITIAL" \
    --structure_bank "$TEXT_BANK" \
    --output "$weight" \
    --metrics_json "$RUN/train_metrics/${method}.json" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer Adam \
    --scheduler linear \
    --warmup 0 \
    --seed 123 \
    --base_structure_weight "$BASE_STRUCTURE_WEIGHT" \
    --rank_temperature "$RANK_TAU" \
    --neighbor_temperature "$NEIGHBOR_TAU" \
    --confidence_delta "$CONF_DELTA" \
    --confidence_alpha "$CONF_ALPHA" \
    --confidence_margin_max "$CONF_MARGIN_MAX" \
    --confidence_max_triplets_per_anchor "$CONF_MAX_TRIPLETS" \
    --rkd_angle_triplets "$RKD_ANGLE_TRIPLETS" \
    --topk "$TOPK" \
    --topk_margin "$TOPK_MARGIN" \
    --device cuda \
    "${extra[@]}"
}

train_oracle() {
  local weight; weight="$(weight_for gtproto_oracle)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train_gtproto_oracle.py \
    --repo_root . \
    --model_config "$MODEL_CONFIG" \
    --init_weight "$INITIAL" \
    --text_bank "$TEXT_BANK" \
    --gt_bank "$GT_BANK" \
    --gt_key crop_balanced_prototypes \
    --output "$weight" \
    --metrics_json "$RUN/train_metrics/gtproto_oracle.json" \
    --steps "$ORACLE_STEPS" \
    --lr "$ORACLE_LR" \
    --seed 123 \
    --device cuda
}

audit_three() {
  local label="$1" weight="$2"
  [[ -f "$weight" ]] || return 3

  CUDA_VISIBLE_DEVICES="$GPU" python audit_textstruct_checkpoint.py \
    --repo_root . \
    --model_config "$MODEL_CONFIG" \
    --weight "$weight" \
    --initial_weight "$INITIAL" \
    --text_bank "$TEXT_BANK" \
    --gt_bank "$GT_BANK" \
    --gt_key crop_balanced_prototypes \
    --output "$RUN/audits/${label}.json" \
    --device cuda
}

# ---------------------------------------------------------------------------
# Reference audits: cheap, no training/eval.
# ---------------------------------------------------------------------------
run_step 01_audit_initial audit_three initial "$INITIAL"

if [[ -f "$EXISTING_PARTSTRUCT" ]]; then
  run_step 02_audit_existing_partstruct \
    audit_three existing_partstruct "$EXISTING_PARTSTRUCT"
fi

# ---------------------------------------------------------------------------
# A) GTProto Oracle: this is the ONE experiment that receives mIoU.
# ---------------------------------------------------------------------------
run_step 03_train_gtproto_oracle train_oracle

ORACLE_WEIGHT="$(weight_for gtproto_oracle)"
if [[ -f "$ORACLE_WEIGHT" ]]; then
  run_step 04_audit_gtproto_oracle \
    audit_three gtproto_oracle "$ORACLE_WEIGHT"
fi

eval_oracle_116() {
  [[ -f "$ORACLE_WEIGHT" ]] || return 3
  local name out log
  name="$(basename "$ORACLE_WEIGHT" .pth)"
  out="$RUN/eval/gtproto_oracle_direct116"
  log="$RUN/logs/eval_gtproto_oracle_direct116.log"
  mkdir -p "$out"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$PORT" \
    src/open_vocabulary_segmentation/main.py \
    --eval \
    --output "$out" \
    --eval_cfg "$EVAL_CFG" \
    --eval_base_cfg "$EVAL116" \
    --opts \
      "model.proj_name=${name}" \
      evaluate.pamr=false \
      "evaluate.bg_thresh=${EVAL_BG_THRESH}" \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -eq 0 ]] || return $rc

  local val
  val="$(grep -E 'INFO  >>' "$log" | tail -n1 | \
    sed -E 's/.*INFO  >>[[:space:]]*([0-9.]+).*/\1/' || true)"
  [[ -n "$val" ]] || {
    echo "Could not parse oracle mIoU"
    return 1
  }
  printf '%s\n' "$val" > "$RUN/oracle_direct116_miou.txt"
  echo "[ORACLE DIRECT116 mIoU] $val"
}
run_step 05_eval_gtproto_oracle_116 eval_oracle_116

# ---------------------------------------------------------------------------
# B) Text-internal structure zoo:
#    train + THREE Spearmans only. No mIoU, no W.
# ---------------------------------------------------------------------------
step=10
for method in "${METHODS[@]}"; do
  run_step "${step}_train_${method}" train_method "$method"
  weight="$(weight_for "$method")"
  if [[ -f "$weight" ]]; then
    run_step "${step}_spearman_${method}" audit_three "$method" "$weight"
  fi
  step=$((step + 1))
done

# ---------------------------------------------------------------------------
# Final compact report.
# ---------------------------------------------------------------------------
generate_summary() {
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])

labels = [
    "initial",
    "existing_partstruct",
    "gtproto_oracle",
    "object_rank",
    "global_row_rank",
    "object_neighbor_kl",
    "global_neighbor_kl",
    "object_confidence_rank",
    "global_confidence_rank",
    "rkd_distance",
    "rkd_angle",
    "gram_mse",
    "gram_corr",
    "topk_margin",
]

def read_audit(label):
    p = run / "audits" / f"{label}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def fmt(x):
    return "NA" if x is None else f"{float(x):.8f}"

miou_path = run / "oracle_direct116_miou.txt"
oracle_miou = (
    miou_path.read_text().strip()
    if miou_path.is_file()
    else "NA"
)

lines = [
    "=== THREE-SPEARMAN SCREEN ===",
    "GT visual = crop-balanced TRAIN-GT DINO prototypes",
    "",
]

md = [
    "# Three-Spearman screen",
    "",
    "GT visual target for audit: **crop-balanced TRAIN-GT DINO part prototypes**.",
    "",
    "| model | Raw CLIP ↔ GT | Initial ↔ GT | Model ↔ GT |",
    "|---|---:|---:|---:|",
]

for label in labels:
    a = read_audit(label)
    if a is None:
        raw = init = cur = None
    else:
        raw = a.get("raw_clip_vs_gt_visual_macro_spearman")
        init = a.get("initial_vs_gt_visual_macro_spearman")
        cur = a.get("current_vs_gt_visual_macro_spearman")

    line = (
        f"{label}\t"
        f"raw={fmt(raw)}\t"
        f"initial={fmt(init)}\t"
        f"current={fmt(cur)}"
    )
    lines.append(line)
    md.append(
        f"| {label} | {fmt(raw)} | {fmt(init)} | {fmt(cur)} |"
    )

lines += [
    "",
    f"GTProto Oracle direct116 mIoU = {oracle_miou}",
]
md += [
    "",
    f"**GTProto Oracle direct116 mIoU:** {oracle_miou}",
    "",
    "Only the GTProto Oracle was evaluated for segmentation. "
    "All text-structure candidates were screened by the three Spearmans only.",
]

(run / "RESULTS_TO_SEND.txt").write_text("\n".join(lines) + "\n")
(run / "FINAL_SUMMARY.md").write_text("\n".join(md) + "\n")

print("\n".join(lines))
PY
}
run_step FINAL_summary generate_summary

echo
echo "========================================================================"
echo "DONE $(date -Is)"
echo "========================================================================"
echo "Send me:"
echo "  cat '$RUN/RESULTS_TO_SEND.txt'"
echo
echo "Status:"
echo "  cat '$RUN/status.tsv'"
echo "========================================================================"
