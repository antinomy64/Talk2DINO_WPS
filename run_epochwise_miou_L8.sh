#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
BASE_PROJ="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"
TEXT_BANK="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"
MODEL_CONFIG="configs/vitb_mlp_infonce.yaml"
EVAL_CFG="src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml"
EVAL_BASE_CFG="src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml"
BAKER="bake_epoch_projector_no_provenance.py"

LINEAR_DIR="nightly_ablation_pp116_seed123_20260921/runs/core/relative_linear_L8/train"
ORTH_DIR="nightly_ablation_pp116_seed123_20260921/runs/core/relative_orthogonal_L8/train"

OUT_ROOT="nightly_ablation_pp116_seed123_20260921/analysis/epochwise_miou_L8"
TMP_WEIGHT_DIR="weights/__epochwise_miou_tmp"
RAW_CSV="$OUT_ROOT/epochwise_miou_raw.csv"
FINAL_CSV="$OUT_ROOT/epochwise_miou.csv"

BG_THRESH="0.55"
START_EPOCH=0
END_EPOCH=30
KEEP_BAKED="${KEEP_BAKED:-0}"

EXPECTED_BASE=21.70
EXPECTED_LINEAR_E30=29.88
EXPECTED_ORTH_E30=32.60
TOL=0.08

mkdir -p "$OUT_ROOT" "$TMP_WEIGHT_DIR"
echo "mapping,epoch,mIoU,mACC,aAcc,eval_log" > "$RAW_CSV"

for f in "$BASE_PROJ" "$TEXT_BANK" "$MODEL_CONFIG" "$EVAL_CFG" "$EVAL_BASE_CFG" "$BAKER"; do
    [[ -f "$f" ]] || { echo "[ERROR] missing: $f"; exit 1; }
done
for d in "$LINEAR_DIR" "$ORTH_DIR"; do
    [[ -d "$d" ]] || { echo "[ERROR] missing: $d"; exit 1; }
done

parse_metrics() {
python - "$1" <<'PY'
import re, sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(errors="ignore")
matches = re.findall(
    r"'aAcc':\s*([0-9.eE+-]+),\s*'mIoU':\s*([0-9.eE+-]+),\s*'mAcc':\s*([0-9.eE+-]+)",
    s,
)
if matches:
    aacc, miou, macc = map(float, matches[-1])
    if miou <= 1.0:
        miou *= 100.0
    if macc <= 1.0:
        macc *= 100.0
    if aacc <= 1.0:
        aacc *= 100.0
    print(f"{miou:.4f},{macc:.4f},{aacc:.4f}")
    raise SystemExit

tables = re.findall(
    r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
    s,
)
if tables:
    aacc, miou, macc = map(float, tables[-1])
    print(f"{miou:.4f},{macc:.4f},{aacc:.4f}")
    raise SystemExit

raise SystemExit("Could not parse aAcc/mIoU/mAcc from eval log")
PY
}

check_close() {
python - "$1" "$2" "$3" "$4" <<'PY'
import sys
got, expected, tol = map(float, sys.argv[1:4])
name = sys.argv[4]
diff = abs(got - expected)
print(f"[CHECK] {name}: got={got:.4f}, expected={expected:.4f}, diff={diff:.4f}, tol={tol:.4f}")
if diff > tol:
    raise SystemExit(
        f"[ERROR] sanity check failed for {name}. "
        "Stop here: epoch-wise evaluation protocol does not reproduce the formal result."
    )
PY
}

run_eval() {
    local mapping="$1"
    local epoch="$2"
    local weight_dir="$3"

    local tag="${mapping}_epoch_$(printf '%03d' "$epoch")"
    local eval_dir="$OUT_ROOT/$tag"
    local log="$eval_dir/eval.log"
    local metrics_file="$eval_dir/metrics.txt"
    mkdir -p "$eval_dir"

    if [[ -f "$metrics_file" ]]; then
        cat "$metrics_file"
        return 0
    fi

    local proj_name
    local baked_path=""

    if [[ "$epoch" -eq 0 ]]; then
        proj_name="vitb_mlp_infonce_coco2014_reproduce_clean"
    else
        local w_path="$weight_dir/W_epoch_$(printf '%03d' "$epoch").pt"
        [[ -f "$w_path" ]] || { echo "[ERROR] missing $w_path" >&2; return 1; }

        proj_name="__epochwise_${mapping}_e$(printf '%03d' "$epoch")"
        baked_path="$TMP_WEIGHT_DIR/${proj_name}.pth"

        python "$BAKER" \
            --base_projector "$BASE_PROJ" \
            --w_checkpoint "$w_path" \
            --text_bank "$TEXT_BANK" \
            --model_config "$MODEL_CONFIG" \
            --output "$baked_path" \
            --device cuda

        cp -f "$baked_path" "weights/${proj_name}.pth"
    fi

    local port
    if [[ "$mapping" == "orthogonal" ]]; then
        port=$((32720 + epoch))
    else
        port=$((32620 + epoch))
    fi
    echo
    echo "================================================================================"
    echo "[EVAL] mapping=$mapping epoch=$epoch proj_name=$proj_name"
    echo "================================================================================"

    set +e
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port="$port" \
        src/open_vocabulary_segmentation/main.py \
        --eval \
        --output "$eval_dir/output" \
        --eval_cfg "$EVAL_CFG" \
        --eval_base_cfg "$EVAL_BASE_CFG" \
        --opts \
            model.proj_name="$proj_name" \
            evaluate.pamr=false \
            evaluate.bg_thresh="$BG_THRESH" \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "$status" -ne 0 ]]; then
        echo "[ERROR] eval failed: mapping=$mapping epoch=$epoch" >&2
        exit "$status"
    fi

    local metrics
    metrics="$(parse_metrics "$log")"
    echo "$metrics" > "$metrics_file"

    if [[ "$epoch" -gt 0 && "$KEEP_BAKED" != "1" ]]; then
        rm -f "$baked_path" "weights/${proj_name}.pth"
    fi

    echo "$metrics"
}

append_row() {
    local mapping="$1"
    local epoch="$2"
    local metrics="$3"
    local miou macc aacc
    IFS=',' read -r miou macc aacc <<< "$metrics"
    local log="$OUT_ROOT/${mapping}_epoch_$(printf '%03d' "$epoch")/eval.log"
    echo "$mapping,$epoch,$miou,$macc,$aacc,$log" >> "$RAW_CSV"
}

echo
echo "### PRECHECK 1/3: epoch 0 baseline"
base_metrics="$(run_eval linear 0 "$LINEAR_DIR" | tail -n 1)"
base_miou="${base_metrics%%,*}"
check_close "$base_miou" "$EXPECTED_BASE" "$TOL" "Talk2DINO epoch0"
append_row linear 0 "$base_metrics"
append_row orthogonal 0 "$base_metrics"

echo
echo "### PRECHECK 2/3: Linear epoch 30"
linear30="$(run_eval linear 30 "$LINEAR_DIR" | tail -n 1)"
linear30_miou="${linear30%%,*}"
check_close "$linear30_miou" "$EXPECTED_LINEAR_E30" "$TOL" "Relative+Linear L8 epoch30"
append_row linear 30 "$linear30"

echo
echo "### PRECHECK 3/3: Orthogonal epoch 30"
orth30="$(run_eval orthogonal 30 "$ORTH_DIR" | tail -n 1)"
orth30_miou="${orth30%%,*}"
check_close "$orth30_miou" "$EXPECTED_ORTH_E30" "$TOL" "Relative+Orthogonal L8 epoch30"
append_row orthogonal 30 "$orth30"

echo
echo "### PRECHECK PASSED"
echo "Baseline ~= $EXPECTED_BASE, Linear e30 ~= $EXPECTED_LINEAR_E30, Orthogonal e30 ~= $EXPECTED_ORTH_E30"
echo "Running epochs 1..29..."

for mapping in linear orthogonal; do
    if [[ "$mapping" == "linear" ]]; then
        weight_dir="$LINEAR_DIR"
    else
        weight_dir="$ORTH_DIR"
    fi

    for epoch in $(seq 1 29); do
        metrics="$(run_eval "$mapping" "$epoch" "$weight_dir" | tail -n 1)"
        append_row "$mapping" "$epoch" "$metrics"
    done
done

python - "$RAW_CSV" "$FINAL_CSV" <<'PY'
import csv, sys
from pathlib import Path

raw_path, out_path = map(Path, sys.argv[1:3])
rows = list(csv.DictReader(raw_path.open()))
by_key = {}
for r in rows:
    by_key[(r["mapping"], int(r["epoch"]))] = r

fieldnames = [
    "epoch",
    "linear_mIoU", "orthogonal_mIoU",
    "linear_mACC", "orthogonal_mACC",
    "linear_aAcc", "orthogonal_aAcc",
]
out = []
for epoch in range(31):
    l = by_key.get(("linear", epoch))
    o = by_key.get(("orthogonal", epoch))
    if l is None or o is None:
        raise SystemExit(f"Missing row at epoch {epoch}: linear={l is not None}, orthogonal={o is not None}")
    out.append({
        "epoch": epoch,
        "linear_mIoU": l["mIoU"],
        "orthogonal_mIoU": o["mIoU"],
        "linear_mACC": l["mACC"],
        "orthogonal_mACC": o["mACC"],
        "linear_aAcc": l["aAcc"],
        "orthogonal_aAcc": o["aAcc"],
    })

with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out)

print(f"[OK] saved combined CSV: {out_path}")
PY

echo
echo "================================================================================"
echo "DONE"
echo "================================================================================"
echo "Raw CSV   : $RAW_CSV"
echo "Final CSV : $FINAL_CSV"
echo
echo "Quick view:"
column -s, -t "$FINAL_CSV" 2>/dev/null || cat "$FINAL_CSV"
