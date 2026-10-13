#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

export CUDA_VISIBLE_DEVICES=1

# ==========================================
# ADE-234 Soft Relative + Structure
# lambda=4.0, seed=123, 100 Epoch Checkpoints
# ==========================================

RUN="final_exp/relproto_alignment/ade234_soft_structure_lambda4p0_seed123_bg055_tau005_100ep"

SCAN_ROOT="output/ade234_soft_structure_lambda4p0_100ep_checkpoint_scan"

SOURCE_PROJ="weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth"

TEXT_BANK="feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt"

MODEL_CONFIG="configs/vitb_mlp_infonce.yaml"

EVAL_CFG="src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml"

EVAL_BASE_CFG="src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml"

# 默认粗扫，每10轮一个Checkpoint
EPOCHS="${SCAN_EPOCHS:-30 40 50 60 70 80 90 100}"

mkdir -p "$SCAN_ROOT"

echo "========================================"
echo "ADE234 CHECKPOINT SCAN"
echo "Lambda : 4.0"
echo "Epochs : $EPOCHS"
echo "GPU    : 1"
echo "========================================"

# ==========================================
# 1. Batch Bake + Evaluation
# ==========================================

for EP in $EPOCHS; do

    E3=$(printf "%03d" "$EP")

    echo ""
    echo "========================================"
    echo "Evaluating Epoch $EP"
    echo "========================================"

    # Epoch 100 已经完成Evaluation
    if [ "$EP" -eq 100 ]; then

        EXISTING_LOG="output/ade234_soft_structure_lambda4p0_100ep/eval.log"

        if [ ! -f "$EXISTING_LOG" ]; then
            echo "ERROR: Epoch 100 评估日志不存在"
            exit 1
        fi

        grep -q "mIoU of 1016 test images:" "$EXISTING_LOG"

        echo "Epoch 100 已评估，直接复用"

        grep -A 5 "Summary:" "$EXISTING_LOG"

        continue
    fi

    # 本次100 Epoch训练保存的Checkpoint
    W_CKPT="$RUN/W_epoch_${E3}.pt"

    PROJ_NAME="vitb_mlp_infonce_coco2014_reproduce_clean_ade234_soft_structure_scan_e${E3}"

    PROJ_PATH="weights/${PROJ_NAME}.pth"

    EVAL_DIR="$SCAN_ROOT/epoch_${E3}"

    if [ ! -f "$W_CKPT" ]; then
        echo "ERROR: 找不到Checkpoint: $W_CKPT"
        exit 1
    fi

    mkdir -p "$EVAL_DIR"

    # ======================================
    # Bake W
    # ======================================

    if [ ! -e "$PROJ_PATH" ]; then

        echo "[BAKE] Epoch $EP"

        python bake_ade20kpart234_relproto_w.py \
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
            echo "ERROR: 已有Projector但缺少Bake Report"
            exit 1
        fi

        echo "[SKIP BAKE] $PROJ_PATH"

    fi

    # ======================================
    # Evaluation
    # ======================================

    if [ -f "$EVAL_DIR/eval.log" ]; then

        if grep -q "mIoU of 1016 test images:" "$EVAL_DIR/eval.log"; then

            echo "[SKIP EVAL] Epoch $EP 已完成"

        else

            echo "ERROR: 已存在未完成的Eval日志，拒绝覆盖"
            echo "$EVAL_DIR/eval.log"
            exit 1

        fi

    else

        echo "[EVAL] Epoch $EP"

        python -m torch.distributed.run \
            --nproc_per_node=1 \
            --master_port=32934 \
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

# ==========================================
# 2. 汇总所有Checkpoint的结果
# ==========================================

export SCAN_ROOT

python - <<'PY'
import csv
import os
import re
from pathlib import Path

root = Path(os.environ["SCAN_ROOT"])

logs = {}

# 复用已有100 Epoch评估
logs[100] = Path(
    "output/ade234_soft_structure_lambda4p0_100ep/eval.log"
)

# 收集所有已完成的Checkpoint评估
for path in root.glob("epoch_*/eval.log"):
    ep = int(path.parent.name.split("_")[-1])
    logs[ep] = path

results = []

for ep, path in sorted(logs.items()):

    if not path.is_file():
        continue

    content = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if "mIoU of 1016 test images:" not in content:
        print(f"[WARN] 未完成评估：{path}")
        continue

    # 只在最后一个Summary区块中提取指标
    if "INFO Summary:" not in content:
        print(f"[WARN] 找不到Summary：{path}")
        continue

    summary = content.rsplit("INFO Summary:", 1)[-1]

    match = re.search(
        r'^\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$',
        summary,
        flags=re.MULTILINE,
    )

    if not match:
        print(f"[WARN] Summary解析失败：{path}")
        continue

    aacc, miou, macc = map(float, match.groups())

    results.append({
        "epoch": ep,
        "mIoU": miou,
        "mAcc": macc,
        "aAcc": aacc,
    })

if not results:
    raise RuntimeError("没有有效的Evaluation结果")

# 保存CSV
csv_path = root / "results.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "mIoU", "mAcc", "aAcc"],
    )
    writer.writeheader()
    writer.writerows(sorted(results, key=lambda r: r["epoch"]))

# 按mIoU排序
ranking = sorted(
    results,
    key=lambda r: (-r["mIoU"], r["epoch"]),
)

print("\n" + "=" * 60)
print("ADE234 CHECKPOINT SCAN RANKING")
print("=" * 60)

print(f"{'Rank':<6}{'Epoch':<9}{'mIoU':<12}{'mAcc':<12}{'aAcc':<12}")

for i, r in enumerate(ranking, 1):
    print(
        f"{i:<6}"
        f"{r['epoch']:<9}"
        f"{r['mIoU']:<12.2f}"
        f"{r['mAcc']:<12.2f}"
        f"{r['aAcc']:<12.2f}"
    )

best = ranking[0]

print("\nBEST CHECKPOINT")
print(f"Epoch : {best['epoch']}")
print(f"mIoU  : {best['mIoU']:.2f}%")
print(f"mAcc  : {best['mAcc']:.2f}%")
print(f"aAcc  : {best['aAcc']:.2f}%")

print(f"\nCSV saved: {csv_path}")
PY

echo ""
echo "ADE234_CHECKPOINT_SCAN_COMPLETE"
