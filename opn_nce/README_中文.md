# PP116 OPN-InfoNCE：同 Object + 跨 Object 同名 Part

## 1. 核心规则

对于正样本 `cat's eye`，候选负类为：

- 同物体完整 taxonomy 下的其他部件：`cat's head / nose / ear / torso ...`
- 跨物体同名部件：`bird's eye / cow's eye / dog's eye / horse's eye / person's eye / sheep's eye`
- 不包含所有其他类别（比如 `car's wheel`、`dog's leg`）

**不再按视觉相似度选 Hardest Top-12**；负类集合对给定正类固定，不使用任何 GT 像素标签。具体集合由项目真实 116 类文本名称在运行时严格生成。

精确匹配 Part 名称（去首尾空白、大小写无关），**没有同义词合并**：`paw` 和 `foot` 不视为同名，`lower arm` 和 `upper arm` 也不视为同名。

loss：

    z_j = cosine(stopgrad(v_c), normalize(t_j @ W)) / 0.07
    N(c) = {j != c | obj(j)==obj(c) OR suffix(j)==suffix(c)}
    L_c = log(exp(z_c)+sum_{j in N(c)} exp(z_j)) - z_c
    L = mean_over_crops(mean_over_positive_parts(L_c)) + 4.0 * relation_rank_loss

**特殊类别**：`tvmonitor's screen` 没有其他同物体 Part，也没有其他 Object 的 `screen`，所以该正类的 OPN-InfoNCE = 0，没有该项的梯度；这不是程序错误。对其他包含多个部件的训练 Crop，它仍可参与结构相关项。若以后要统一给予负类，需要设计单独 fallback 消融，不要在本实验偷偷添加 Hardest 负类。

不同类别拥有不同数量的负类；因此不能把与 Hardest-12 的差异单独归因于“负类语义来源”，还可能受到负类数量/损失尺度影响。

## 2. 文件说明

- `opn_nce/train_pp116_opn_nce.py`：从修复后的 `hardest_search/train_pp116_hardest.py` 派生的完整训练器，直接基于 `train_relproto_alignemt.py` 加载原始 cache 和文本 bank；**不会覆盖你已有脚本**。
- `opn_nce/selftest_opn_nce.py`：无需数据和 GPU 的纯 CPU 自检，验证负类集合、数学值、梯度及零负类情形。
- `opn_nce/eval_pp116_opn_nce.py`：与先前 Weighted InfoNCE 的 `bake + 官方 116 类评估` 使用同样流程。
- `opn_nce/taxonomy_negative_sets_reference_116.csv`：从之前 PP116 官方评估输出提取的 116 类名称做的**参考负类清单**。正式训练会读取自己的 text bank 重新生成，不依赖参考文件。

## 3. 训练指令（GPU 3，100 Epoch）

把 ZIP 上传到 `/mnt/sda/master/code/lyx/Talk2DINO_official_bg`，在该目录执行：

```bash
unzip -o PP116_OPN_INFONCE_20261029.zip
python opn_nce/selftest_opn_nce.py

nohup env CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. PYTHONUNBUFFERED=1 \
python -u opn_nce/train_pp116_opn_nce.py \
  --project_root . \
  --part_setting presence \
  --nce_variant taxonomy_union \
  --proto_variant positive \
  --soft_temperature 0.05 \
  --nce_temperature 0.07 \
  --lambda_struct 4.0 \
  --rank_delta 0.05 \
  --rank_margin 0.05 \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-5 \
  --seed 123 \
  --out_dir final_exp/relproto_alignment/pp116_opn_union_l4_seed123_100ep \
  --device cuda \
  > pp116_opn_union_l4_seed123_100ep.log 2>&1 < /dev/null &

echo "PID=$!"
tail -f pp116_opn_union_l4_seed123_100ep.log
```

**与 Hardest-12 保持一致的关键点**：训练缓存、116 类全局词表、Soft Relative（正 margin + fallback + 单 Part absolute pooling）、Structure ranking（lambda=4）、W 正交 SVD、batch 16、Adam lr 1e-5、seed 123。只有 NCE 负类选择规则发生变化。

训练开始时自动保存 `final_exp/relproto_alignment/pp116_opn_union_l4_seed123_100ep/taxonomy_negative_sets.csv`，逐正类列出同物体负类、跨物体同名负类及总数量。每个 Epoch 保存 `W_epoch_XXX.pt`；训练无改动既有 checkpoint。

如果只想与 Hardest-12 的 **E30** 进行同 Epoch 对比，可以把 `--epochs 100` 改为 `--epochs 30`，并使用另一个 `--out_dir`，不要覆盖 100E 目录。

## 4. 官方评估（GPU 3）

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. PYTHONUNBUFFERED=1 \
python -u opn_nce/eval_pp116_opn_nce.py \
  --gpu 3 \
  --run_dir final_exp/relproto_alignment/pp116_opn_union_l4_seed123_100ep \
  --epochs 3,10,20,30,50,80,100 \
  --output output/pp116_opn_union_l4_seed123_100ep_eval \
  --master_port 32640
```

直接复用已有 `bake_predobj_relproto_w_into_projector_final.py` 和 PP116 官方 `main.py --eval`，PAMR=false。输出：

- `output/pp116_opn_union_l4_seed123_100ep_eval/official_macro_metrics.csv`：每个 E 的官方 mIoU、重新汇总的 mACC、Seen/Unseen 分组及 hIoU。
- `.../all116_per_part.csv`：每个 Epoch 全部 116 类的 IoU 与 Acc。
- `.../epoch_XXX/official_eval_stdout.log`：完整官方验证日志。

## 5. 解释和实验风险

1. `cat's eye` 与 `dog's eye` 被作为**负类**，并不是要使它们语义完全不相似；这是在固定全词表下训练不同 *Object-Part* 分类标签更可区分的假设，可能与共享部件语义发生冲突，需用结果验证。
2. 一些部件负类很少（例如特殊唯一部件），loss 会随候选集合大小变化；`tvmonitor's screen` 则为零 NCE。因而与 Hardest-12 并非纯粹的“来源控制”比较。下一步可设计等量 quota 或加权组别作为独立消融。
3. 训练只使用 Part Presence，不用 GT mask；验证 116 类原样 joint evaluation，不启用 PAMR。
