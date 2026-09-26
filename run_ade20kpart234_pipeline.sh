#!/usr/bin/env bash
set -euo pipefail

cd /home/master/code/lyx/Talk2DINO_official_bg

DATA_ROOT=/home/master/dataset/lyx/ADE20KPart234
TEXT_BANK=feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt
CACHE_DIR=feature/ade20kpart234_predobj_clip_struct
CACHE=${CACHE_DIR}/train_predobj_cropaug.pth
SMOKE=${CACHE_DIR}/train_predobj_cropaug_smoke10.pth
RUN_DIR=final_exp/relproto_alignment/ade20kpart234_seed123_bg055_L8
BASE_PROJ=weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth
BAKED=weights/vitb_mlp_infonce_ade20kpart234_L8_ep030_baked.pth

mkdir -p "${CACHE_DIR}"

echo "===== 1. SELF TEST ====="

python -m py_compile \
  extract_predobj_cropaug_ade20kpart234.py \
  train_relproto_alignment_ade20kpart234.py \
  bake_ade20kpart234_relproto_w.py \
  src/open_vocabulary_segmentation/segmentation/datasets/ade20kpart234_part.py

python extract_predobj_cropaug_ade20kpart234.py --self_test

python - <<'PY'
import torch
from ade20kpart234_taxonomy import PART_CLASSES, OBJECT_GROUPS

p = "feature/ade20kpart234_clip_text/ade20kpart234_clip_vitb16_subimagenet_raw.pt"
x = torch.load(p, map_location="cpu", weights_only=False)

assert tuple(x["features"].shape) == (234, 512)
assert list(x["class_names"]) == list(PART_CLASSES)
assert len(x["object_groups"]) == 44
for k, ids in OBJECT_GROUPS.items():
    assert list(x["object_groups"][k]) == list(ids)
assert x["clip_model"] == "ViT-B/16"
assert x["template_set"] == "sub_imagenet_template"
assert x["normalized"] is False
assert x["stage"] == "pre_projector_prompt_mean"

print("ADE234 TEXT BANK CONTRACT: PASS")
PY

echo "===== 2. CACHE SMOKE10 ====="

rm -f "${SMOKE}"

CUDA_VISIBLE_DEVICES=0 \
python extract_predobj_cropaug_ade20kpart234.py \
  --repo_root /home/master/code/lyx/Talk2DINO_official_bg \
  --image_root "${DATA_ROOT}/images/train" \
  --part_mask_root "${DATA_ROOT}/annotations_detectron2_part/train" \
  --projector_weight "${BASE_PROJ}" \
  --output_pth "${SMOKE}" \
  --part_ignore 65535 \
  --bg_thresh 0.55 \
  --max_images 10 \
  --device cuda

python - <<'PY'
import torch
p = "feature/ade20kpart234_predobj_clip_struct/train_predobj_cropaug_smoke10.pth"
x = torch.load(p, map_location="cpu", weights_only=False)

anns = x["annotations"]
meta = x["pred_obj_cropaug_meta"]

assert meta["bg_thresh"] == 0.55
assert meta["pamr"] is True
assert meta["part_ignore"] == 65535
assert meta["target_spatial_supervision"] == "none"

for a in anns:
    assert tuple(a["cropaug_patch_tokens"].shape) == (1024, 768)
    assert tuple(a["pred_obj_mask_patch"].shape) == (1024,)
    assert a["pred_obj_mask_patch"].dtype == torch.bool
    assert int(a["pred_obj_mask_patch"].sum()) > 0
    assert len(a["part_category_id"]) >= 1

print("SMOKE annotations:", len(anns))
print("ADE234 SMOKE CACHE: PASS")
PY

echo "===== 3. FULL CACHE ====="

if [[ ! -f "${CACHE}" ]]; then
  CUDA_VISIBLE_DEVICES=0 \
  python extract_predobj_cropaug_ade20kpart234.py \
    --repo_root /home/master/code/lyx/Talk2DINO_official_bg \
    --image_root "${DATA_ROOT}/images/train" \
    --part_mask_root "${DATA_ROOT}/annotations_detectron2_part/train" \
    --projector_weight "${BASE_PROJ}" \
    --output_pth "${CACHE}" \
    --part_ignore 65535 \
    --bg_thresh 0.55 \
    --device cuda
else
  echo "[skip] full cache exists: ${CACHE}"
fi

echo "===== 4. TRAINER PREFLIGHT ====="

CUDA_VISIBLE_DEVICES=0 \
python train_relproto_alignment_ade20kpart234.py \
  --project_root /home/master/code/lyx/Talk2DINO_official_bg \
  --train_dataset "${CACHE}" \
  --text_bank "${TEXT_BANK}" \
  --model_config configs/vitb_mlp_infonce.yaml \
  --weights "${BASE_PROJ}" \
  --out_dir "${RUN_DIR}" \
  --feature_name cropaug_patch_tokens \
  --foreground_key pred_obj_mask_patch \
  --part_id_key part_category_id \
  --part_name_key part_class_name \
  --clip_model ViT-B/16 \
  --template_set sub_imagenet_template \
  --prototype_max_patches 8 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-5 \
  --seed 123 \
  --expected_bg_thresh 0.55 \
  --require_pamr \
  --require_cache_projector_match \
  --device cuda \
  --preflight_only

echo "===== 5. TRAIN RELATIVE + ORTHOGONAL / L=8 ====="

CUDA_VISIBLE_DEVICES=0 \
python train_relproto_alignment_ade20kpart234.py \
  --project_root /home/master/code/lyx/Talk2DINO_official_bg \
  --train_dataset "${CACHE}" \
  --text_bank "${TEXT_BANK}" \
  --model_config configs/vitb_mlp_infonce.yaml \
  --weights "${BASE_PROJ}" \
  --out_dir "${RUN_DIR}" \
  --feature_name cropaug_patch_tokens \
  --foreground_key pred_obj_mask_patch \
  --part_id_key part_category_id \
  --part_name_key part_class_name \
  --clip_model ViT-B/16 \
  --template_set sub_imagenet_template \
  --prototype_max_patches 8 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-5 \
  --seed 123 \
  --expected_bg_thresh 0.55 \
  --require_pamr \
  --require_cache_projector_match \
  --device cuda

echo "===== 6. BAKE EPOCH 30 ====="

CUDA_VISIBLE_DEVICES=0 \
python bake_ade20kpart234_relproto_w.py \
  --project_root /home/master/code/lyx/Talk2DINO_official_bg \
  --projector "${BASE_PROJ}" \
  --w_checkpoint "${RUN_DIR}/W_epoch_030.pt" \
  --text_bank "${TEXT_BANK}" \
  --model_config configs/vitb_mlp_infonce.yaml \
  --output "${BAKED}" \
  --device cuda \
  --overwrite

echo "===== 7. FULL-IMAGE VAL EVAL ====="

CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=32734 \
  src/open_vocabulary_segmentation/main.py \
  --eval \
  --output output/ours_ade20kpart234_L8_ep030 \
  --eval_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml \
  --eval_base_cfg src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml \
  --opts model.proj_name=vitb_mlp_infonce_ade20kpart234_L8_ep030_baked

echo "ADE20K-PART-234 PIPELINE COMPLETE"
