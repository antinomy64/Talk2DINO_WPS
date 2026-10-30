# PP116 全负类难度加权 InfoNCE（2026-10-28）

## 实验目标

在此前 **Hardest-12 / Global-116** 所使用的相同训练设置上，仅修改 InfoNCE 负类损失：

- **Hardest-12**：每个伪视觉原型只保留相似度最高的 12 个负类；其他 103 个完全不参与 InfoNCE。
- **本实验**：不丢弃任何负类，**115 个负类全部参与**；某个负类与当前伪视觉原型越相似（更难区分），给它越高的权重；较容易的负类权重较低，但仍大于零。

**其余全部保持不变**：原 PP116 116-Part 弱监督训练缓存、图像级 Part Presence、COCO2014 冻结 Talk2DINO projector、冻结 CLIP/DINOv2、Soft Relative 正 Margin Patch 加权聚合、Fallback、单 Part 绝对 Softmax、Structure ranking loss λ=4、共享正交 W、Adam + 每步 SVD、Batch 16、LR 1e-5、Seed 123、30 Epoch、原始 850 图像/116 类官方 Evaluation。

**注意**：此版本为与历史 Hardest-12 (mIoU 37.44% E30) **严格同设置对比的全部 116 类训练**，不是最近的 Seen-74-only 实验。可用于单独考察“负类硬截断 vs 全负类连续加权”。

## 数学公式（每个当前正类 Part c 的视觉伪原型单独计算）

令：

```text
z_j = cosine(detach(visual_proto_c), adapted_text_j) / 0.07
```

对其余 115 个负类，先算动态困难程度分布：

```text
p_j = softmax(z_j.detach() / weight_temperature) over all j != c
```

然后计算负类权重：

```text
w_j = (1 - strength) + strength * 115 * p_j
```

负类以加权形式进入 InfoNCE 分母：

```text
L_c = log( exp(z_c) + sum_{j != c} w_j * exp(z_j) ) - z_c
```

默认：`--hard_weight_strength 0.5`，`--hard_weight_temperature 2.0`。

- `strength=0`：**严格退化为未修改的全 116 类 InfoNCE**，在代码中走原始 CrossEntropy 路径，数值完全相等。
- `strength=0.5`：所有负类权重 `>=0.5`，但负类权重总和固定为 115，越难的权重越高；正类固定权重 1。
- `strength` 越接近 1：权重越集中到难负类；要求 `<1`，确保所有负类权重严格为正。
- `weight_temperature` 越小：权重差距越大；越大：各负类权重越均匀。
- Hardness 由**当前伪视觉原型与当前投影 Part 文本的余弦**计算，而不是文本之间的余弦、也不是 GT / 固定近邻。
- 权重 `detach()`：只改变对负类的损失强调，阻断额外经权重计算的梯度路径；InfoNCE 正视觉原型依旧 `detach()`。Structure Loss 原样通过 Soft Relative 权重回传 W。
- 此方法**不保证高于 Hardest-12**；权重归一化使平均负类权重=1，但并不使总体 NCE loss 数值与 Global 相同。

## 安装和 CPU 验证

将整个 ZIP 放在服务器仓库根目录：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip -o PP116_WEIGHTED_INFONCE_20261028.zip
python -m compileall -q weighted_nce
python weighted_nce/selftest_weighted_nce.py
```

`selftest_weighted_nce.py` 会同时做 weighted loss 数值和梯度测试、0 强度与 Global-116 的等价性验证；在完整仓库环境中还会调用原始脚本的 Soft Relative / Structure self-test。

## 正式运行：GPU 3，30 Epoch

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
python -u weighted_nce/train_pp116_weighted_nce.py \
  --project_root . \
  --part_setting presence \
  --nce_variant weighted_hardness \
  --hard_weight_strength 0.5 \
  --hard_weight_temperature 2.0 \
  --proto_variant positive \
  --soft_temperature 0.05 \
  --nce_temperature 0.07 \
  --lambda_struct 4.0 \
  --rank_delta 0.05 \
  --rank_margin 0.05 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-5 \
  --seed 123 \
  --out_dir final_exp/relproto_alignment/pp116_weighted_nce_s050_t200_l4_seed123 \
  --device cuda \
  > pp116_weighted_nce_s050_t200_gpu3.log 2>&1 < /dev/null &

echo "PID=$!"
tail -f pp116_weighted_nce_s050_t200_gpu3.log
```

**路径**：`final_exp/relproto_alignment/pp116_weighted_nce_s050_t200_l4_seed123/W_epoch_030.pt`。与原来的 W checkpoint 格式兼容。请使用**不存在的新 `out_dir`**，避免原训练器的 `FileExistsError`，不要覆盖先前 run。

训练历史：`.../training_history.csv`，每个 Epoch W：`.../W_epoch_XXX.pt`。Checkpoint 内 `search_variant` 记录本次两个负类加权参数。

## 直接使用原官方 116 类 Evaluation

```bash
nohup env PYTHONUNBUFFERED=1 \
python -u weighted_nce/eval_weighted_nce.py \
  --gpu 3 \
  --run_dir final_exp/relproto_alignment/pp116_weighted_nce_s050_t200_l4_seed123 \
  --epochs 3,10,20,30 \
  --output output/pp116_weighted_nce_s050_t200_l4_eval \
  --master_port 30191 \
  > pp116_weighted_nce_s050_t200_eval_gpu3.log 2>&1 < /dev/null &

echo "PID=$!"
tail -f pp116_weighted_nce_s050_t200_eval_gpu3.log
```

评估脚本只负责自动 bake W 到与当前 COCO2014 原始 projector 相同的映射权重中，然后调用仓库原始 `src/open_vocabulary_segmentation/main.py --eval`：850 图，所有 116 类共同预测，PAMR=false。**不使用 GT Object / Part Presence 进行推理**。

输出包括：

- `output/.../epoch_030/official_eval_stdout.log`：逐类 IoU / Acc 与官方 mIoU/mAcc 终端日志。
- `output/.../epoch_030/official_output/`：官方 eval 目录。
- `output/.../official_macro_metrics.csv`：各 Epoch 的官方 116 类 mIoU、根据官方逐类 Acc 重新计算的 mACC、Seen 和 Unseen mIoU、hIoU。
- `output/.../all116_per_part.csv`：116 个 Part 的 IoU / Acc。

统计 NaN 时仅排除 NaN 类别；IoU=0 仍参与平均，不推断缺失类别为 0。

## 建议的最小三点参数比较

为避免把“难负类权重强度”当作已知最优，建议将下面 3 个设置分别放在**不同目录**，保持其他参数完全相同：

| 实验 | strength | weight_temperature | 含义 |
|---|---:|---:|---|
| 权重偏平滑 | 0.25 | 2.0 | 小幅提高困难负类权重 |
| 标准 | 0.50 | 2.0 | 默认 |
| 强调困难负类 | 0.75 | 2.0 | 更强区别对待 |

其中 `weight_temperature=2.0` 表示对 **InfoNCE logits（已经除过 0.07）** 再做软化，而不是对原始余弦直接除以 2.0。

与历史控制对比：

| 方法 | InfoNCE | E30 mIoU |
|---|---|---:|
| Global-116 | 115 个负类权重=1 | 36.66% |
| Hardest-12 | 只保留最高的 12 个 | 37.44% |
| 本实验 | 所有 115 个负类，难度连续加权 | **待运行** |

以上是历史同一实验系列中的参考值，本脚本尚未跑过服务器官方评估。
