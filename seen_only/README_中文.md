# PP116 Seen-74 STRICT training (only requested modifications)

The **only two differences** from the baseline Soft Relative + Structure Ours training are:

1. Use **only Seen-object cached crops** and their original image-level Part Presence (no unseen Object/Crop). Original IDs remain unchanged.
2. The InfoNCE **positive AND negative** classes are **exactly the 74 Seen Parts**. A local 74-class CE is computed by selecting the 74 text embeddings and remapping the canonical target IDs. No Unseen Part participates in CE. Structure Loss is formed only among Part Presence inside Seen training crops.

EVALUATION IS UNCHANGED: Use the existing `seen_only/eval_seen_unseen.py` after training; its evaluator uses the original 116-part predictions and computes seen/unseen means *after* the official evaluation. The exact `eval_seen_unseen.py` file in this zip is byte-identical to the previous release.

Official OV-PARTS PP116 Seen 74 / Unseen 42: Unseen Objects = `bird(9), car(10), dog(10), motorbike(4), sheep(9)`; Seen Objects = `aeroplane, bicycle, bottle, bus, cat, cow, horse, person, pottedplant, train, tvmonitor` (11 Objects, 74 Parts). These are **counts, not accuracy scores**.

## Install
```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip -o PP116_SeenOnly_Strict74_20261028.zip
python seen_only/selftest_seen_only.py
```

## REAL-data preflight (writes to its own folder; not the train output)
```bash
CUDA_VISIBLE_DEVICES=3 python -u seen_only/train_pp116_seen_only.py --project_root . --part_setting presence --preflight_only --device cuda --out_dir final_exp/relproto_alignment/pp116_seen74_strict_preflight
```
**Do NOT reuse the preflight directory for actual training.** It will contain files and the original trainer refuses to overwrite it.

## Train on GPU 3
```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u seen_only/train_pp116_seen_only.py --project_root . --part_setting presence --epochs 30 --batch_size 16 --lr 1e-5 --seed 123 --soft_temperature 0.05 --nce_temperature 0.07 --lambda_struct 4.0 --rank_delta 0.05 --rank_margin 0.05 --out_dir final_exp/relproto_alignment/pp116_seen74_strict_nce_l4_seed123 --device cuda > pp116_seen74_strict_gpu3.log 2>&1 < /dev/null &
echo "PID=$!"
tail -f pp116_seen74_strict_gpu3.log
```

## Evaluation: unchanged 116 class inference, report Seen / Unseen mean separately
```bash
nohup env PYTHONUNBUFFERED=1 python -u seen_only/eval_seen_unseen.py --gpu 3 --epochs 3,10,20,30 --run_dir final_exp/relproto_alignment/pp116_seen74_strict_nce_l4_seed123 --output output/pp116_seen74_strict_eval > pp116_seen74_strict_eval.log 2>&1 < /dev/null &
echo "PID=$!"
tail -f pp116_seen74_strict_eval.log
```

## Verification
- `preflight.json`: `seen_training_annotations` and `excluded_unseen_annotations` (actual counts from YOUR cache, never guessed); `seen_part_ids` contains 74 original IDs.
- `training_samples.csv`: contains ONLY the 11 Seen Object names.
- Each minibatch: `seen_only_cross_entropy` maps original 116 IDs to [0,73] and computes CE logits [B,K,74] from seen text alone.
- Frozen text bank still contains all 116 to allow **unchanged** 116-way evaluation. The other 42 have no train CE gradient, though shared W learned from Seen rotates all 116 embeddings at evaluation.

This script is based on the previously supplied 29KB Ours softrelative/structure training implementation. Other hyperparams, prototype formation, W optimizer/SVD are unchanged relative to that supplied file. We cannot guarantee identical behavior to a different, subsequently edited server-side Ours trainer until its code is compared.
