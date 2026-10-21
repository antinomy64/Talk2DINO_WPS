#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
DATASET="${1:?Usage: bash eval_ours_joint_unfreeze.sh pp116|ade234 [checkpoint.pt]}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
if [[ "$DATASET" == pp116 ]]; then
  RUN="final_exp/relproto_alignment/pp116_ours_jointP_lr1e6_Wlr1e5_lambda4_full_seed123"
  OUT_MODEL=pp116_ours_jointP_W_e100
  ECFG=src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml
  BCFG=src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml
  PORT=33371
elif [[ "$DATASET" == ade234 ]]; then
  RUN="final_exp/relproto_alignment/ade234_ours_jointP_lr1e6_Wlr1e5_lambda4_full_seed123"
  OUT_MODEL=ade234_ours_jointP_W_e100
  ECFG=src/open_vocabulary_segmentation/configs/ade20kpart234/dinotext_ade20kpart234_vitb_mlp_infonce.yml
  BCFG=src/open_vocabulary_segmentation/configs/ade20kpart234/eval_ade20kpart234.yml
  PORT=33372
else echo 'Dataset must be pp116 or ade234' >&2;exit 2; fi
CHECKPOINT="${2:-${RUN}/W_epoch_100.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then echo "Missing trained checkpoint: $CHECKPOINT" >&2; exit 2; fi
if [[ -e "weights/${OUT_MODEL}.pth" ]]; then
  echo "Refusing to overwrite weights/${OUT_MODEL}.pth" >&2; exit 2
fi
python -u bake_joint_projector_W_for_eval.py \
  --dataset "$DATASET" \
  --joint_checkpoint "$CHECKPOINT" \
  --initial_projector weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
  --model_config configs/vitb_mlp_infonce.yaml \
  --output "weights/${OUT_MODEL}.pth" --device cpu
python -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  src/open_vocabulary_segmentation/main.py --eval \
  --output "output/${DATASET}_ours_jointP_W_e100_eval" \
  --eval_cfg "$ECFG" \
  --eval_base_cfg "$BCFG" \
  --opts "model.proj_name=${OUT_MODEL}" evaluate.pamr=false
