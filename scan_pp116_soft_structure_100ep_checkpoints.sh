#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

export CUDA_VISIBLE_DEVICES=0

# ========== 配置 ==========

RUN="final_exp/relproto_alignment/pp116_soft_structure_lambda4p0_seed123_bg055_tau005_100ep"

SCAN_ROOT="output/pp116_soft_structure_lambda4p0_100ep_checkpoint_scan"

SOURCE_PROJ="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"

TEXT_BANK="feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt"

MODEL_CONFIG="configs/vitb_mlp_infonce.yaml"

EVAL_CFG="src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml"

EVAL_BASE_CFG="src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml"

# 默认每隔10轮扫描
EPOCHS="${SCAN_EPOCHS:-30 40 50 60 70 80 90 100}"

mkdir -p "$SCAN_ROOT"

echo "========================================"
echo "PP116 Checkpoint Scan"
echo "Epochs: $EPOCHS"
echo "GPU: 0"
echo "========================================"

# ========== 批量 Bake + Eval ==========

for EP in $EPOCHS; do

    E3=$(printf "%03d" "$EP")

    echo ""
    echo "========================================"
    echo "Evaluating Epoch $EP"
    echo "========================================"

    # 100 Epoch 已经有验证结果，直接复用
    if [ "$EP" -eq 100 ]; then

        EXISTING_LOG="output/pp116_soft_structure_lambda4p0_100ep/eval.log"

        if [ ! -f "$EXISTING_LOG" ]; then
            echo "ERROR: Epoch 100 的评估日志不存在"
            exit 1
        fi

        grep -q "mIoU of 850 test images:" "$EXISTING_LOG"

        echo "Epoch 100 已评估，直接复用"
        grep -A 5 "Summary:" "$EXISTING_LOG"

        continue
    fi

    W_CKPT="$RUN/W_epoch_${E3}.pt"

    PROJ_NAME="vitb_mlp_infonce_coco2014_reproduce_clean_pp116_soft_structure_scan_e${E3}"

    PROJ_PATH="weights/${PROJ_NAME}.pth"

    EVAL_DIR="$SCAN_ROOT/epoch_${E3}"

    if [ ! -f "$W_CKPT" ]; then
        echo "ERROR: 找不到 $W_CKPT"
        exit 1
    fi

    mkdir -p "$EVAL_DIR"

    # ===== Bake =====

    if [ ! -e "$PROJ_PATH" ]; then

        echo "[BAKE] Epoch $EP"

        python bake_predobj_relproto_w_into_projector_final.py \
            --project_root . \
            --projector "$SOURCE_PROJ" \
            --w_checkpoint "$W_CKPT" \
            --text_bank "$TEXT_BANK" \
            --model_config "$MODEL_CONFIG" \
            --output "$PROJ_PATH" \
            --device cuda \
            2>&1 | tee "$EVAL_DIR/bake.log"

    else
        if [ ! -f "${PROJ_PATH}.bake_report.json" ]; then
            echo "ERROR: 已有 Projector 但缺少 Bake Report，拒绝覆盖"
            exit 1
        fi

        echo "[SKIP BAKE] $PROJ_PATH"
    fi

    # ===== Evaluation =====

    if [ -f "$EVAL_DIR/eval.log" ]; then

        if grep -q "mIoU of 850 test images:" "$EVAL_DIR/eval.log"; then
            echo "[SKIP EVAL] Epoch $EP 已完成"
        else
            echo "ERROR: Epoch $EP 存在未完成日志，拒绝覆盖"
            exit 1
        fi

    else

        echo "[EVAL] Epoch $EP"

        python -m torch.distributed.run \
            --nproc_per_node=1 \
            --master_port=32910 \
            src/open_vocabulary_segmentation/main.py \
            --eval \
            --output "$EVAL_DIR" \
            --eval_cfg "$EVAL_CFG" \
            --eval_base_cfg "$EVAL_BASE_CFG" \
            --opts \
                model.proj_name="$PROJ_NAME" \
                evaluate.pamr=false \
            2>&1 | tee "$EVAL_DIR/eval.log"

    fi

    echo "[RESULT] Epoch $EP"

    grep -A 5 "Summary:" "$EVAL_DIR/eval.log"

done

# ========== 汇总所有已经完成的评估 ==========

export SCAN_ROOT

python - <<'PY'
import csv
import os
import re
from pathlib import Path

root = Path(os.environ["SCAN_ROOT"])

logs = {}

# 原有100ep评估
logs[100] = Path(
    "output/pp116_soft_structure_lambda4p0_100ep/eval.log"
)

# 所有已经扫描的epoch
for p in root.glob("epoch_*/eval.log"):
    ep = int(p.parent.name.split("_")[-1])
    logs[ep] = p

results = []

for ep, path in sorted(logs.items()):

    if not path.is_file():
        continue

    content = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    matches = re.findall(
        r'^\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$',
        content,
        flags=re.MULTILINE,
    )

    if not matches:
        print(f"[WARN] 没有找到Summary: {path}")
        continue

    aacc, miou, macc = map(float, matches[-1])

    results.append({
        "epoch": ep,
        "mIoU": miou,
        "mAcc": macc,
        "aAcc": aacc,
    })

# 保存CSV
csv_path = root / "results.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "mIoU", "mAcc", "aAcc"],
    )
    writer.writeheader()
    writer.writerows(results)

# 按mIoU排序
ranking = sorted(
    results,
    key=lambda x: x["mIoU"],
    reverse=True,
)

print("\n" + "=" * 58)
print("PP116 CHECKPOINT SCAN RANKING")
print("=" * 58)

print(f"{'Rank':<6}{'Epoch':<9}{'mIoU':<12}{'mAcc':<12}{'aAcc':<12}")

for i, r in enumerate(ranking, 1):
    print(
        f"{i:<6}"
        f"{r['epoch']:<9}"
        f"{r['mIoU']:<12.2f}"
        f"{r['mAcc']:<12.2f}"
        f"{r['aAcc']:<12.2f}"
    )

if ranking:
    best = ranking[0]
    print("\nBEST CHECKPOINT")
    print(f"Epoch : {best['epoch']}")
    print(f"mIoU  : {best['mIoU']:.2f}%")
    print(f"mAcc  : {best['mAcc']:.2f}%")
    print(f"aAcc  : {best['aAcc']:.2f}%")

print(f"\nCSV saved: {csv_path}")
PY

echo ""
echo "CHECKPOINT_SCAN_COMPLETE"
