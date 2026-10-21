#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
case "${1:-}" in
  preflight)
    python build_pp116_gtobj_valid_masks.py --preflight 16 \
      --cache feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
      --gt-object-root data/PascalPart116/annotations_detectron2_obj/train
    ;;
  build)
    python build_pp116_gtobj_valid_masks.py \
      --cache feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
      --gt-object-root data/PascalPart116/annotations_detectron2_obj/train \
      --empty-policy drop \
      --out output/pp116_gtobj_fg_oracle/gt_object_valid_masks.pt
    ;;
  train)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_pp116_gtobj_fg_oracle.py \
      --gtmask-bank output/pp116_gtobj_fg_oracle/gt_object_valid_masks.pt \
      --part_setting presence \
      --project_root . \
      --train_dataset feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
      --text_bank feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
      --model_config configs/vitb_mlp_infonce.yaml \
      --weights weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
      --out_dir final_exp/relproto_alignment/pp116_GTObjFG_ORACLE_softrelative_structure_lam4_100ep \
      --soft_temperature 0.05 \
      --nce_temperature 0.07 \
      --lambda_struct 4.0 \
      --rank_delta 0.05 \
      --rank_margin 0.05 \
      --epochs 100 \
      --batch_size 16 \
      --lr 1e-5 \
      --seed 123 \
      --expected_bg_thresh 0.55 \
      --require_pamr \
      --require_cache_projector_match \
      --device cuda
    ;;
  bake)
    CUDA_VISIBLE_DEVICES=0 python bake_predobj_relproto_w_into_projector_final.py \
      --project_root . \
      --projector weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
      --w_checkpoint final_exp/relproto_alignment/pp116_GTObjFG_ORACLE_softrelative_structure_lam4_100ep/W_epoch_100.pt \
      --text_bank feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
      --model_config configs/vitb_mlp_infonce.yaml \
      --output weights/vitb_mlp_infonce_coco2014_GTObjFG_ORACLE_lam4_e100.pth \
      --device cuda
    ;;
  eval)
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
      --nproc_per_node=1 \
      --master_port=32628 \
      src/open_vocabulary_segmentation/main.py \
      --eval \
      --output output/pp116_GTObjFG_ORACLE_lam4_e100 \
      --eval_cfg src/open_vocabulary_segmentation/configs/voc116_part/dinotext_voc116_part_vitb_mlp_infonce.yml \
      --eval_base_cfg src/open_vocabulary_segmentation/configs/voc116_part/eval_voc116_part.yml \
      --opts \
        model.proj_name=vitb_mlp_infonce_coco2014_GTObjFG_ORACLE_lam4_e100 \
        evaluate.pamr=false
    ;;
  *) echo 'Usage: bash run_pp116_gtobj_fg_oracle.sh {preflight|build|train|bake|eval}' >&2; exit 2;;
esac
