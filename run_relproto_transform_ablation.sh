#!/usr/bin/env bash
set -euo pipefail

# Strict overnight ablation:
#   Initial Talk2DINO projector (NO PartStruct fine-tune)
#       ├─ no Stage-3 transform
#       ├─ shared Orthogonal W (current method)
#       └─ shared Unconstrained Linear matrix (same 768x768, same init/Adam, no bias)
#
# Orth vs Linear differs ONLY by SVD/polar retraction.
#
# Default: 3 seeds. Override, e.g.
#   SEEDS="123" GPU=0 bash run_relproto_transform_ablation.sh

cd "$(dirname "$0")"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-123 456 789}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PROTO_MAX="${PROTO_MAX:-4}"
BG_THRESH="${BG_THRESH:-0.54}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

INIT_WEIGHT="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
MODEL_CFG="configs/vitb_mlp_infonce.yaml"
TEXT_BANK="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw.pt"

IMAGE_ROOT="data/PascalPart116/images/train"
OBJ_MASK_ROOT="data/PascalPart116/annotations_detectron2_obj/train"
PART_MASK_ROOT="data/PascalPart116/annotations_detectron2_part/train"

CACHE_DIR="feature/pascalpart116_predobj_clip_struct_initial"
CACHE="${CACHE_DIR}/train_predobj_cropaug_initial_bg054.pth"

RUN_ROOT="final_exp/ablation_w_vs_linear_initial/${RUN_TAG}"
EVAL_ROOT="output/ablation_w_vs_linear_initial/${RUN_TAG}"
SUMMARY_CSV="${RUN_ROOT}/results.csv"
SUMMARY_JSON="${RUN_ROOT}/results.json"

mkdir -p "$CACHE_DIR" "$RUN_ROOT" "$EVAL_ROOT" weights logs

echo "============================================================"
echo "RELPROTO TRANSFORM ABLATION"
echo "============================================================"
echo "projector  : $INIT_WEIGHT"
echo "PartStruct : NONE"
echo "seeds      : $SEEDS"
echo "epochs     : $EPOCHS"
echo "lr         : $LR"
echo "batch      : $BATCH_SIZE"
echo "RelProto K : $PROTO_MAX"
echo "cache bg   : $BG_THRESH"
echo "run root   : $RUN_ROOT"
echo "============================================================"

for f in \
  "$INIT_WEIGHT" \
  "$MODEL_CFG" \
  "$TEXT_BANK" \
  train_relproto_alignemt.py \
  train_relproto_linear_ablation.py \
  bake_relproto_transform_into_projector.py \
  extract_predobj_cropaug.py
do
  test -e "$f" || { echo "[ERROR] missing: $f"; exit 2; }
done

for d in "$IMAGE_ROOT" "$OBJ_MASK_ROOT" "$PART_MASK_ROOT"; do
  test -d "$d" || { echo "[ERROR] missing data dir: $d"; exit 2; }
done

# ---------------------------------------------------------------------------
# 0) Initial-projector-specific predicted-object cache.
# ---------------------------------------------------------------------------
if [[ ! -f "$CACHE" ]]; then
  echo "[cache] building initial-projector cache..."
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python extract_predobj_cropaug.py \
    --repo_root . \
    --image_root "$IMAGE_ROOT" \
    --obj_mask_root "$OBJ_MASK_ROOT" \
    --part_mask_root "$PART_MASK_ROOT" \
    --projector_weight "$INIT_WEIGHT" \
    --bg_thresh "$BG_THRESH" \
    --lambda_bg 0.2 \
    --pamr \
    --output_pth "$CACHE" \
    --device cuda \
    2>&1 | tee "${RUN_ROOT}/00_cache.log"
else
  echo "[cache] reusing: $CACHE"
fi

# Strictly verify cache projector SHA, PAMR, and threshold before ANY training.
python - "$CACHE" "$INIT_WEIGHT" "$BG_THRESH" <<'PY'
import sys, torch, hashlib, math
cache, weight, bg = sys.argv[1], sys.argv[2], float(sys.argv[3])

def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20), b""):
            h.update(b)
    return h.hexdigest()

x=load(cache)
m=x["pred_obj_cropaug_meta"]
assert m["projector_weight_sha256"] == sha(weight), (m["projector_weight"], weight)
assert m["pamr"] is True
assert math.isclose(float(m["bg_thresh"]), bg, abs_tol=1e-12, rel_tol=0)
print("[CACHE CONTRACT PASS]")
print("annotations:", len(x["annotations"]))
print("projector :", m["projector_weight"])
print("bg_thresh :", m["bg_thresh"])
print("pamr      :", m["pamr"])
PY

# ---------------------------------------------------------------------------
# Eval helper.
# ---------------------------------------------------------------------------
eval_one () {
  local proj_name="$1"
  local out_dir="$2"
  local port="$3"

  echo "[eval] $proj_name"
  rm -f "segmentation_results/${proj_name}.json"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$port" \
    src/open_vocabulary_segmentation/main.py \
      --eval \
      --output "$out_dir" \
      --eval_cfg \
        src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml \
      --eval_base_cfg \
        src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml \
      --opts \
        model.proj_name="$proj_name" \
        evaluate.pamr=false \
        evaluate.bg_thresh=0.4 \
    2>&1 | tee "${out_dir}.log"

  test -f "segmentation_results/${proj_name}.json"
}

# ---------------------------------------------------------------------------
# 1) Re-evaluate the exact initial projector once under the same dense protocol.
# ---------------------------------------------------------------------------
BASE_NAME="vitb_mlp_infonce_coco2014_reproduce_clean"
eval_one "$BASE_NAME" "${EVAL_ROOT}/baseline_initial" 30101
cp "segmentation_results/${BASE_NAME}.json" \
   "${RUN_ROOT}/eval_baseline_initial.json"

# ---------------------------------------------------------------------------
# 2) Orthogonal W and 3) Unconstrained Linear, same seed/data/loss.
# ---------------------------------------------------------------------------
seed_index=0
for SEED in $SEEDS; do
  seed_index=$((seed_index + 1))

  ORTH_DIR="${RUN_ROOT}/orth_seed${SEED}"
  LIN_DIR="${RUN_ROOT}/linear_seed${SEED}"

  echo
  echo "================ seed $SEED : ORTHOGONAL W ================"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train_relproto_alignemt.py \
    --project_root . \
    --train_dataset "$CACHE" \
    --text_bank "$TEXT_BANK" \
    --model_config "$MODEL_CFG" \
    --weights "$INIT_WEIGHT" \
    --out_dir "$ORTH_DIR" \
    --prototype_max_patches "$PROTO_MAX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --expected_bg_thresh "$BG_THRESH" \
    --require_pamr \
    --require_cache_projector_match \
    --device cuda \
    2>&1 | tee "${ORTH_DIR}.log"

  echo
  echo "================ seed $SEED : LINEAR ======================="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python train_relproto_linear_ablation.py \
    --project_root . \
    --train_dataset "$CACHE" \
    --text_bank "$TEXT_BANK" \
    --model_config "$MODEL_CFG" \
    --weights "$INIT_WEIGHT" \
    --out_dir "$LIN_DIR" \
    --prototype_max_patches "$PROTO_MAX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --expected_bg_thresh "$BG_THRESH" \
    --require_pamr \
    --require_cache_projector_match \
    --device cuda \
    2>&1 | tee "${LIN_DIR}.log"

  ORTH_NAME="vitb_mlp_infonce_ablate_initial_orth_s${SEED}_${RUN_TAG}"
  LIN_NAME="vitb_mlp_infonce_ablate_initial_linear_s${SEED}_${RUN_TAG}"

  python bake_relproto_transform_into_projector.py \
    --projector "$INIT_WEIGHT" \
    --transform_checkpoint "${ORTH_DIR}/W_last.pt" \
    --model_config "$MODEL_CFG" \
    --output "weights/${ORTH_NAME}.pth"

  python bake_relproto_transform_into_projector.py \
    --projector "$INIT_WEIGHT" \
    --transform_checkpoint "${LIN_DIR}/W_last.pt" \
    --model_config "$MODEL_CFG" \
    --output "weights/${LIN_NAME}.pth"

  eval_one "$ORTH_NAME" "${EVAL_ROOT}/orth_seed${SEED}" "$((30110 + seed_index * 2))"
  eval_one "$LIN_NAME" "${EVAL_ROOT}/linear_seed${SEED}" "$((30111 + seed_index * 2))"

  cp "segmentation_results/${ORTH_NAME}.json" \
     "${RUN_ROOT}/eval_orth_seed${SEED}.json"
  cp "segmentation_results/${LIN_NAME}.json" \
     "${RUN_ROOT}/eval_linear_seed${SEED}.json"
done

# ---------------------------------------------------------------------------
# 4) Final machine-readable summary.
# ---------------------------------------------------------------------------
python - "$RUN_ROOT" "$SEEDS" "$RUN_TAG" <<'PY'
import sys, json, csv, statistics
from pathlib import Path

root = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
tag = sys.argv[3]

def loadj(p):
    return json.loads(Path(p).read_text())

def miou(path):
    d=loadj(path)
    if "voc116_part" not in d:
        raise KeyError((path, d))
    return float(d["voc116_part"])

baseline = miou(root/"eval_baseline_initial.json")
rows=[]

for seed in seeds:
    for method in ("orth","linear"):
        summ=loadj(root/f"{method}_seed{seed}"/"summary.json")
        last=summ["history"][-1]
        e=miou(root/f"eval_{method}_seed{seed}.json")
        rows.append({
            "seed":seed,
            "method":method,
            "miou":e,
            "delta_vs_initial":e-baseline,
            "train_final_loss":float(last["mean_loss"]),
            "orthogonality_max_abs":float(last["orthogonality_max_abs"]),
            "text_cosine_structure_max_abs":float(last["text_cosine_structure_max_abs"]),
            "identity_rms":float(last["identity_rms"]),
            "condition_number":float(last.get("condition_number",1.0)),
        })

out = {
    "run_tag":tag,
    "initial_projector_miou":baseline,
    "rows":rows,
    "aggregate":{},
}
for method in ("orth","linear"):
    vals=[r["miou"] for r in rows if r["method"]==method]
    ds=[r["delta_vs_initial"] for r in rows if r["method"]==method]
    out["aggregate"][method]={
        "n":len(vals),
        "miou_mean":statistics.mean(vals),
        "miou_std":statistics.stdev(vals) if len(vals)>1 else 0.0,
        "delta_vs_initial_mean":statistics.mean(ds),
    }

(root/"results.json").write_text(json.dumps(out,indent=2)+"\n")
with (root/"results.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

print("\n============================================================")
print("FINAL ABLATION SUMMARY")
print("============================================================")
print(f"Initial projector: {baseline:.4f}")
for method in ("orth","linear"):
    a=out["aggregate"][method]
    print(
        f"{method:8s}: "
        f"{a['miou_mean']:.4f} ± {a['miou_std']:.4f} "
        f"(Δ initial {a['delta_vs_initial_mean']:+.4f})"
    )
print("============================================================")
print("CSV :", root/"results.csv")
print("JSON:", root/"results.json")
PY

echo
echo "[DONE] overnight W-vs-Linear ablation finished."
echo "Results: $RUN_ROOT"
