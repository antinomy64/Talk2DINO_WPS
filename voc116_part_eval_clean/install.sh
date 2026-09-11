#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$HOME/code/lyx/Talk2DINO_official_bg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO"

mkdir -p src/open_vocabulary_segmentation/configs/voc116_part
mkdir -p src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets
mkdir -p src/open_vocabulary_segmentation/segmentation/datasets
mkdir -p tools

cp -f "$HERE/src/open_vocabulary_segmentation/configs/voc116_part/default.yml" \
  src/open_vocabulary_segmentation/configs/voc116_part/default.yml
cp -f "$HERE/src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml" \
  src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml
cp -f "$HERE/src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml" \
  src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml
cp -f "$HERE/src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/pascalpart116_part.py" \
  src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/pascalpart116_part.py
cp -f "$HERE/src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py" \
  src/open_vocabulary_segmentation/segmentation/datasets/pascalpart116_part.py
cp -f "$HERE/tools/audit_voc116_part_eval.py" \
  tools/audit_voc116_part_eval.py

echo "[installed] clean VOC116-part evaluator into: $REPO"
