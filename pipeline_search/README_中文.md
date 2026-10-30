# PP116 Ours Pipeline 改进 — 全部 100 Epoch 版本

## 核心说明（2026-10-23 修订）

**所有正式实验统一训练 100 Epoch**：20 个单变量实验 + 6 个组合实验，总计 26 个模型、2600 个训练 Epoch（单 GPU 顺序执行）。实验使用**完全相同的 Ours 监督协议和基础超参数**，使得不同方法可与原始 Ours 的 E100 结果、以及本包重新训练的 `baseline_retrain` E100 结果直接比较。

默认官方评估检查点：**E3、E10、E20、E30、E50、E80、E100**，全部进行 850 张 PP116 验证图像的完整 116 类分割，不使用 PAMR。E100 是主要比较指标；其余用于检查 B/C 组的早期峰值与后期回落。

> **运行规模提醒：** 2600 个训练 Epoch 并不是普通“一晚上”可以保证跑完的工作量。例如 80–120 秒/Epoch 时，仅训练约为 58–87 小时，不含 Bake/评估。建议按两个 profile 顺序执行，并预留 GPU 时间和磁盘空间，不要在同一 GPU 上并发启动。

## 0. 强制约束

- 不改变弱监督形式：只有图像级 Part Presence、冻结的 Talk2DINO 前景、冻结 DINOv2 Crop Patch 特征；**GT Part/Object Mask 不能进入训练**。
- 改进范围只在新文件 `pipeline_search/train_pp116_pipeline_search.py`：可变更伪原型合成、训练 Loss、无 GT 的可靠度代理、权重/温度、训练策略；不改仓库原始 `train_pp116_softsingle_dual.py`，不改提取脚本、缓存或官方评估器。
- 默认基线仍是 **116-way Global InfoNCE + Soft Relative Prototype + Structure Loss λ=4 + 共享 768×768 正交 W**，不是 Object-Local InfoNCE。
- 原始优化协议：Adam lr=1e-5、batch size=16、seed=123、W 从单位阵初始化、每步 SVD 正交恢复；soft τ=0.05、NCE τ=0.07、rank_delta=0.05、rank_margin=0.05。
- 原始每 Crop 的 K 个正类损失先均值，再对 Crop 均值；伪视觉原型对于 NCE 保持 `detach()`；关系损失的视觉原型梯度路径保持有效。
- A/B/C/D 历史分类仅用于**评估后离线汇总**，不参与训练、样本选择或可信度计算。
- 每个模型从头单独训练 100 Epoch；不从 10/30 Epoch 旧实验继续，以免优化器状态与先前参数选择产生混淆。

## 1. 文件清单

- `pipeline_search/train_pp116_pipeline_search.py`：唯一包含训练算法变体的代码，原版 Baseline 路径与原 Ours 等价。
- `pipeline_search/run_search.py`：单 GPU 顺序训练、每 Epoch 保存 W、指定 Epoch Bake + 官方评估、断点安全重复调用、自动结果汇总；**三个 profile 默认都是 100 Epoch**。
- `pipeline_search/summarize_search.py`：官方 116 类 IoU/Acc/推算 Precision、mIoU、A/B/C/D 汇总、与同 Epoch retrained baseline 对照。
- `pipeline_search/selftest_search.py`：CPU 合成数据的训练 Baseline 与原 Ours Loss/梯度等价性测试、变体前向反向检查。
- `pipeline_search/pp116_four_groups_per_part.csv`：历史 A/B/C/D 分组，**不在训练中使用**。
- `pipeline_search/reference_original_train_pp116_softsingle_dual.py`：旧训练器快照，仅供静态参照。
- `pipeline_search/train_vs_original.patch`：训练器修改代码 diff（不需要在服务器 patch）。

## 2. 单变量：20 个模型，每个 100 Epoch

| 名称 | 训练改动 | 目的 |
|---|---|---|
| `baseline_retrain` | 原始 Ours | 与原始 E100 结果核对 |
| `proto_top8` | 正 Margin 候选取 Top-8 | 伪原型支持范围 |
| `proto_top32` | 正 Margin 候选取 Top-32 | 支持范围敏感性 |
| `proto_margin_power` | 大 Margin 获得更高 Softmax 相对权重 | 竞争特异性 |
| `proto_fallback4` | 无正 Margin 时取 Top-4 Fallback | 错误单 Patch 自强化 |
| `proto_softgate` | 连续 sigmoid Margin gate | 避免硬阈值 |
| `trust_margin025` | Margin/数量决定的 GT-free NCE 可靠度，底限 0.25 | 错误伪标签降低权重 |
| `trust_coherence025` | 加视觉一致性项 | 不一致 Patch 降权 |
| `trust_struct` | 同时对 NCE 和 Structure 降权 | 减弱不可靠关系监督 |
| `balance_presence` | 图像级 Presence 频次平衡 | 高频部件偏置 |
| `nce_smooth005` | 全局 InfoNCE label smoothing=0.05 | 伪目标过度自信 |
| `nce_cross025` | 跨物体负类降至 0.25 倍，仍保留全局竞争 | 软化跨物体负类 |
| `nce_hard16` | 正类+最难 16 个全局负类 | Hard negatives |
| `nce_clip4` | 单 Part CE 在 4 截断（梯度可能归零） | 高损失离群样本 |
| `struct_l1` | λ=1 | 结构权重 |
| `struct_l2` | λ=2 | 结构权重 |
| `struct_l6` | λ=6 | 结构权重 |
| `softtau0025` | soft τ=0.025 | 聚合尖锐度 |
| `softtau010` | soft τ=0.10 | 聚合平滑度 |
| `ncetau010` | NCE τ=0.10 | 语义竞争梯度尺度 |

## 3. 组合：6 个模型，每个 100 Epoch

一个额外 `baseline_retrain`，以及：

1. `combo_top32_trust`：Top-32 + Margin Reliability；
2. `combo_softgate_trust`：Softgate + Margin Reliability；
3. `combo_cross_struct2`：Cross-object 0.25 + λ=2；
4. `combo_top8_smooth`：Top-8 + Label Smoothing；
5. `combo_balance_trust`：Presence Balance + Margin Reliability。

组合实验不能把性能差异归因于某一个单变量；以自己的同 Epoch Baseline 对照。

## 4. 服务器执行（直接 CMD/终端命令，无 .sh）

把 ZIP 放在 `/mnt/sda/master/code/lyx/Talk2DINO_official_bg` 下，然后：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip PP116_PIPELINE_SEARCH_100E_20261023.zip
python pipeline_search/selftest_search.py
```

**可选**：需要非常简短的实际缓存/官方评估烟测时，显式使用 `--epochs 2`，不会用于 E100 的科学对比：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u pipeline_search/run_search.py \
  --profile pilot --gpu 0 --epochs 2 --eval-epochs 2 \
  --tag pp116_pipeline_smoke_20261023
```

### 正式单变量 20 × 100 Epoch

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u pipeline_search/run_search.py \
  --profile overnight --gpu 0 \
  --tag pp116_pipeline_search_100e_20261023_v1 \
  --eval-epochs 3,10,20,30,50,80,100 \
  > pp116_pipeline_search_100e.log 2>&1 < /dev/null &
echo "PID=$!"
```

正式组合 6 × 100 Epoch，**在单变量结束后**运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u pipeline_search/run_search.py \
  --profile extended --gpu 0 \
  --tag pp116_pipeline_search_100e_20261023_v1 \
  --eval-epochs 3,10,20,30,50,80,100 \
  > pp116_pipeline_extended_100e.log 2>&1 < /dev/null &
echo "PID=$!"
```

查看进度：

```bash
tail -f pp116_pipeline_search_100e.log
```

**不能覆盖旧实验目录。** 调度器会为不同 profile 使用不同的后缀，且在 provenance 不匹配或某变体训练目录不完整时拒绝混用；新 tag 不与之前 E10/E30 目录冲突。

## 5. 100 Epoch 结果路径

单变量训练：

`final_exp/relproto_alignment/pp116_pipeline_search_100e_20261023_v1_overnight_e100/<variant>/W_epoch_100.pt`

单变量报告：

`output/pp116_pipeline_search_100e_20261023_v1_overnight_e100/`

组合训练：

`final_exp/relproto_alignment/pp116_pipeline_search_100e_20261023_v1_extended_e100/<variant>/W_epoch_100.pt`

组合报告：

`output/pp116_pipeline_search_100e_20261023_v1_extended_e100/`

每组完整保存 E1–E100 的 W Checkpoint（原训练器保存协议不变）。官方评估 Log 在 `val/<variant>/e100/evaluation_stdout.log` 等七个检查点目录中。

报告内 `summary/variant_scorecard.csv`、`summary/delta_vs_retrained_baseline.csv`、`summary/ABCD_official_summary.csv`、`summary/official_116_per_part.csv`、`summary/all_training_history.csv` 分别包含总体成绩、相同 Epoch 的差值、四类宏平均、逐类 IoU/Acc、逐 Epoch 训练统计。

每个 profile 完成后会自动生成轻量上传包 `review_upload.zip`，**不含模型权重、预提取特征或 GT**：

`output/pp116_pipeline_search_100e_20261023_v1_overnight_e100/review_upload.zip`

`output/pp116_pipeline_search_100e_20261023_v1_extended_e100/review_upload.zip`

## 6. 使用说明与审计边界

- 首要对照是**相同 Epoch、相同评估协议**的 E100 全部 116 类 mIoU，同时阅读 ABCD 分组和每类 IoU。每组重训练 Baseline 与用户原始 Ours 100ep 都可复核；严禁跨 Epoch 比较当成单变量收益。
- 所有模型仍只有 `seed=123`：排名靠前的结果需要额外随机种子复核。
- 可信度代理不代表 GT 可靠性，GT 只允许训练结束后另行离线审计。
- 这批工作量远超普通夜间任务，但每个变体顺序执行并写日志，完成组可以复用；中断的半成品目录会拒绝自动重新从头覆盖，需单独人工判断恢复策略。

## 7. 质量检查

- 本地已完成训练和调度器代码语法检查，以及默认 profile/epoch 配置静态测试。
- 已有上一版包的 CPU Baseline Loss/梯度等价测试和模拟变体前后向测试；**本次没有改变训练算法文件，只改变调度器的默认 Epoch 与评估时间点**。
- 真实 GPU 训练和服务器数据适配须在用户服务器执行；不要将语法测试称为完整训练复现。
