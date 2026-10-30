PP116 OPN-InfoNCE 导入与短轮次测速修复（未改变训练数学实现）

本包以原始上传的 PP116_OPN_INFONCE_20261029(1).zip 为基础，仅做两个运行修复：
1) 将仓库根目录加入 sys.path，兼容 python opn_nce/train_pp116_opn_nce.py 和 python -m opn_nce.train_pp116_opn_nce。
2) 仅在 --probe 启用时检查 probe_epochs <= epochs，避免 --epochs 1 测速被默认 probe_epochs 0,1,3,5,10,20,30 拦截。
两处改动均不更改 InfoNCE、Soft Relative、Structure、Adam、SVD 或训练数据。

上传此 ZIP 至仓库根目录后执行：
  unzip -o PP116_OPN_INFONCE_importfix_20261029.zip
  python -m opn_nce.train_pp116_opn_nce --help

原版无需替换，也可从仓库根目录执行：
  PYTHONPATH=. python -u opn_nce/train_pp116_opn_nce.py --help
但是原版用 --epochs 1 时还需额外加 --probe_epochs 0,1。

同 GPU 同脚本测速（无 nohup），各 512 条 cache annotation：
  cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
  for variant in global taxonomy_union; do
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m opn_nce.train_pp116_opn_nce \
      --project_root . \
      --train_dataset feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth \
      --text_bank feature/pascalpart116_clip_text/pascalpart116_clip_vitb16_subimagenet_raw_reproduce.pt \
      --model_config configs/vitb_mlp_infonce.yaml \
      --weights weights/vitb_mlp_infonce_coco2014_reproduce_clean.pth \
      --part_setting presence --nce_variant "$variant" --proto_variant positive \
      --max_annotations 512 --epochs 1 --probe_epochs 0,1 \
      --batch_size 16 --lr 1e-5 --seed 123 \
      --soft_temperature 0.05 --nce_temperature 0.07 \
      --lambda_struct 4.0 --rank_delta 0.05 --rank_margin 0.05 \
      --expected_bg_thresh 0.55 --require_pamr --require_cache_projector_match \
      --out_dir "final_exp/relproto_alignment/speedtest_${variant}_pathfix" --device cuda
  done

日志的 [epoch] seconds 是训练阶段时长，排除前置加载；首次 JIT / GPU warmup 仍可能影响短测试。
