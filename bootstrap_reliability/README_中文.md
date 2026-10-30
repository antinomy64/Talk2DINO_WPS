# PP116 原始 Ours：跨 Epoch / 跨 Crop 可信度选择性加权实验（2026-10-27）

## 目标

从**原始 Ours 的训练文件** `train_pp116_softsingle_dual.py` 完整复制并只在独立训练文件里增加 GT-free 可信度证据/InfoNCE 权重。维持图像级 Part Presence 监督、原始前景缓存和视觉/文本编码器、Soft Relative Patch Competition、K=1 绝对 Softmax、116-way InfoNCE + Structure λ4、Adam+每步正交 SVD。原训练/评估代码不修改，**训练绝不读取 part GT、GT 原型或 A/B/C/D**。

所有变体的运行环境：PP116 8,379 个 Object Crops，116 类，seed=123、batch=16、lr=1e-5、soft τ=0.05、NCE τ=0.07、bg cache=0.55、PAMR(cache)=True、epochs=30。仅优化 W；不改变推理。验证依然 850 张全116类、no PAMR、no GT object mask。

## 新的无 GT 证据

每个训练 Crop–Part 有一个 q(p)（伪原型聚合 Patch Softmax 权重）。

- `support_bad`：无正 Margin 的 Fallback，或极度稀疏的正 Margin 与有效支持；K=1 不基于没有竞争者的 Margin 判断。
- `temporal`：上一**完整 Epoch**的 W 快照作为只读 Teacher，对同一 Crop 同一 Part 的 q(p) 计算归一化 JS 一致性。不会随着每个 Batch 即时更新成 Student；**稳定不代表正确**。
- `cross`：上一 Epoch 同类 Part 的 DINOv2 伪原型均值，按 row_index 奇偶分两组；当前样本只能访问**相反分组**的原型，避免精确的同一个 Crop 给自己验证。类内竞争只比较同一 Object taxonomy 内已有足够历史样本的 Part；不足时该指标无效而非“不可信”。同一原始图片如出现于不同 row_index，此方案不保证排除其所有增强视图。
- `perturb`：对预测前景做确定性 15% Patch 遮挡，用同一个 W 重建伪原型并检验余弦一致性；原始训练伪原型完全不变。小前景样本不靠该证据惩罚。
- `trend`：上一 Epoch 同一 Crop/Part 的无 GT 支持与跨 Crop 一致性代理是否改善。恢复的样本部分解除降权。

两种以上不同证据出现异常时才使用更小的 r（默认 floor=0.35）。**不把单 Patch Softmax 高置信度直接视为可信**。权重 r 停止梯度；伪原型依旧对 NCE 停止梯度，原始 Structure 损失仍然通过其原有分支更新 W。

样本内保持原有 `1/K` 平均：

`L_nce,new = (1-η) * mean_k(CE_k) + η * sum_k(r_k CE_k) / sum_k(r_k)`

这里的 `sum_k(r CE)/sum_k(r)` 是**相对重加权**，并非把不可信 Crop 整体删除（尤其 K=1 时不会发生变化）。E1–E5 η=0；E6–E12 从 0 升到默认 0.7，之后保持 0.7。Structure 保留原配置，不乘 r。

### 局限和可证伪条件

1. 这些指标只是无 GT **可靠性代理**，不会知道真伪；稳定错误、相似部件错配仍可能被接受。跨样本 Anchor 也可能被错误原型污染。
2. 这轮训练只测试**选择性 Loss 降权**，**没有修改伪原型位置**。因此即使有提升，不能直接说实现了 Patch Label 修正。
3. 第一轮所有变体均仅 seed 123；任何增益必须多种子/离线 GT 后验复查。
4. 可靠性权重可能全部接近1（没有足够异常证据）。训练 CSV 的 `evidence_suppressed`、`evidence_weight`、`evidence_temporal_valid`、`evidence_cross_valid`、`evidence_perturb_valid` 会明确展示是否真的激活。若几乎为0，则不能说证明机制无效，只能说阈值和代理在当前训练上未触发。
5. 本包通过离线模拟数值测试；真实服务器需要先做 Smoke Test。

## 实验配置

| 模式 | NCE | 可靠性证据 | 用途 |
|---|---|---|---|
| `baseline_global` | 原始116-way | 无 | 精确原版对照 |
| `temporal_only` | 116-way | support + temporal | 单独测试历史权重一致性 |
| `cross_only` | 116-way | support + cross | 单独测试跨样本原型共识 |
| `temporal_cross` | 116-way | support + temporal + cross | 双证据 |
| `temporal_cross_perturb` | 116-way | 上述 + perturb | 鲁棒性证据 |
| `temporal_cross_perturb_trend` | 116-way | 上述 + trend | 保护恢复中的 Part |
| `hard12_control` | Hardest-12 | 无 | 已发现的负样本改进对照 |
| `hard12_selective` | Hardest-12 | 上述全部 | 验证两机制能否结合（不能作为单变量结论） |

`--profile core` 前6组；`--profile all` 全8组；`--profile smoke` 两组（Baseline 和 Temporal+Cross）。

## GPU3：部署及指令

将 ZIP 放到 `/mnt/sda/master/code/lyx/Talk2DINO_official_bg`：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip -o PP116_BOOTSTRAP_RELIABILITY_20261027.zip
python bootstrap_reliability/selftest_reliability.py
```

预期输出 `SELF_TEST_PASS_BOOTSTRAP_RELIABILITY`。自测不读取训练特征/GT。

**实际真实数据 Smoke Test**（2组×2 Epoch，各官方 Bake+Evaluate E2；有独立结果目录）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u bootstrap_reliability/run_reliability.py --profile smoke --gpu 3 --epochs 2 --eval-epochs 2 --tag pp116_bootstrap_reliability_20261027_v1
```

**正式实验（8组×30 Epoch×4次官方Evaluation，GPU3，顺序运行，不需要 .sh 文件）**：

```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u bootstrap_reliability/run_reliability.py --profile all --gpu 3 --epochs 30 --eval-epochs 3,10,20,30 --tag pp116_bootstrap_reliability_20261027_v1 > pp116_bootstrap_reliability_gpu3.log 2>&1 < /dev/null &
echo "PID=$!"
```

查看：

```bash
tail -f pp116_bootstrap_reliability_gpu3.log
```

Checkpoint：`final_exp/relproto_alignment/pp116_bootstrap_reliability_20261027_v1_all_e30/<variant>/W_epoch_030.pt`

官方结果：`output/pp116_bootstrap_reliability_20261027_v1_all_e30/`

完成后上传：`output/pp116_bootstrap_reliability_20261027_v1_all_e30/review_upload.zip`

## 训练可靠性诊断与对比

每组 Checkpoint 下的 `training_history.csv` 新增：

- `evidence_suppressed`：异常证据>=2而判定可疑的比例；
- `evidence_weight`：平均无 GT 可信权重；
- `evidence_fallback`、`evidence_support_score`：原始支持状态；
- `evidence_temporal_consistency`、`evidence_temporal_valid`；
- `evidence_cross_gap`、`evidence_cross_valid`；
- `evidence_perturb_consistency`、`evidence_perturb_valid`；
- `evidence_recovering`：被判断为正在恢复的比例；
- `evidence_anchor_eligible_part_folds`：历史 Anchor 覆盖的部件数×fold。

官方结果归档有 `scorecard.csv`（E3/E10/E20/E30 mIoU）、`per_part_official.csv`（全116类 IoU/Acc）以及训练历史。

下一步应使用**只读离线 GT 后验审计**检查可靠性机制是否真的提高 Q_GT/覆盖率；严禁将 GT 回流至本训练文件。A/B/C/D 标签仅用于结果分析。

## 断点、复用和安全

- 每组训练一个独立目录，完成后的完整 `W_epoch_030.pt`+`summary.json` 才复用。
- 如果某组只有部分 Epoch，调度器会**拒绝覆盖**，避免无意混入不同训练。先查看日志后手动备份/移动该组的目录再续跑。
- `provenance.json` 包含训练脚本 SHA256、参数和各方案设置；不同源码不能悄然写入同一个实验目录。
- 不修改原始 `train_relproto_alignemt.py`、`train_pp116_softsingle_dual.py`、Bake、官方评估。
