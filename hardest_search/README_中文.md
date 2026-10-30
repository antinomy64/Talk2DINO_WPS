# PP116 Hardest Negative 深度实验包（2026-10-24）

## 0. 项目与实验协议

针对上一轮 PP116 26 组消融中 `Hardest-16` 在 E30 达到 **37.32 mIoU**、原始 Global-116 Baseline 为 **36.66 mIoU**（+0.66 pp）的现象，系统拆解：

1. 负类数量 K 的效应是否单调？16 是否最优？
2. 收益来自 **难负类选择** 还是仅来自减少负类数量？
3. 动态视觉—文本相似度 Hard Mining 是否优于 **固定文本邻居**？
4. 同 Object 的部件和跨 Object 的部件，哪个应该被优先保留？
5. “dog's ear / cat's ear”等跨对象同名部件作为负类，是否构成有害的语义竞争？
6. 硬截断、保留小比例 Global CE、连续加权负类三种方式，哪一种同时保证细粒度判别和全词表竞争？
7. 负类课程（先全局后 Hardest，先 Hardest 后全局）能否改善训练？
8. Hardest 选择如何改变全 116 类最终推理、A/B/C/D 特别是 B/C 的退化？
9. 固定训练 Crop 下，哪些 Part 的 Top-1 对手经常替换，正类与最强对手的 Logit 差值如何变化？

**保持不变：** PP116 同一原始训练缓存；只用图像级 Part Presence 监督；冻结 DINOv2 / CLIP 文本库 / COCO2014 Projector；Soft Relative 正 Margin 聚合与 argmax fallback；K=1 绝对 Soft Pool；StopGrad 视觉伪原型的 InfoNCE；Structure Ranking Loss 默认 λ=4；共享 768×768 W；Adam + 每步 SVD；seed=123；LR=1e-5；batch=16；30 Epoch；PP116 官方 116 类、850 张、无 PAMR / 无 GT object mask 的验证协议。

**绝不读取任何 GT Part Mask 构造训练伪目标、负类、置信度或在线样本采样。** 仅官方评估器在模型训练结束或中间 Epoch 加载后使用验证 GT 计算指标。A/B/C/D 只用于离线结果分析，不进入训练。

**不会修改** 仓库原始 `train_relproto_alignemt.py`、`train_pp116_softsingle_dual.py`、`main.py`、模型权重、训练缓存、官方评价器。新训练文件全部独立放在 `hardest_search/` 下；调度器仅负责执行和记录。

## 1. 负样本数学定义

视觉伪原型 `v` 与 116 个投影文本 `t_j` 的 logit 是 `z_j=cos(v.detach(),t_j)/0.07`。正类 `y`，负类集合 `N(v,y)`：

```
L = log(exp(z_y) + sum_{j in N} w_j * exp(z_j)) - z_y
```

- Global-116: `N` 为其余 115 类；`w=1`。
- Hard-K: 从全部 115 类中按 `z_j` 选出最大的 K 个；`w=1`。无 Object 限制，含同/跨物体类别。
- Quota-K: 优先强制最多 q 个同 Object 最难负类，剩余 K-q 在未入选的所有类别中按 logit 选择。少于 q 个同物体负类时会自动用全局最难负类补齐。**最终恰好 K 个、无重复。**
- Random-K: 从全部 115 类均匀随机选 K 个，使用独立生成器，不改变 DataLoader 全局随机数序列。
- Easy-K: 只保留最容易的 K 个负类（反证控制）。
- Static-Text-K: 文本相似度 `cos(t_y,t_j)` 最高的 K 个负类，选择与具体视觉伪原型无关。
- SameName-Discount-K: 仍从全局视觉 TopK 选择，但对与正类有相同部件后缀（例如不同物体的 `ear`）的负类设置 `w=0.5` 或 `0.25`。保留它们，不直接删除。
- Schedule: Epoch 级更改 K；`linear/cosine/delayed`，其他训练参数不变。
- GlobalMix: `L=(1-α)L_Hard16 + α L_Global116`，α=0.05/0.15/0.30。检验全局弱负类信息是否应该保留；视觉原型的 StopGrad 不变。
- SoftNegativeGate: 所有负类保留连续权重 `w_j=sigmoid((z_j-z_(第16难))/τ_gate)`，正类权重为 1；门控和 K-th 阈值 `detach()`。检验硬 TopK 的集合跳变与全词表判别之间是否可以折中。

**重要：** Hardest 是每个视觉伪原型**单独动态选择**的负类，不是全数据集单一固定负类词表；动态选择依据使用 `detach()`，梯度只沿选中的文本 logits 回传，视觉伪原型在 InfoNCE 分支保持 StopGrad；Structure Loss 的梯度路径不变。

## 2. 全部 31 组实验（每组 30 Epoch）

### CORE（13 组：基线 + 负类数量 + 负类选择性质）

| ID | 方法 | 核心问题 |
|---|---|---|
| `baseline_116` | 原始 Global 115 negatives | 独立复现 36.66 基线 |
| `hard_k04` | Hardest-4 | 极窄选择是否欠约束 |
| `hard_k08` | Hardest-8 | 更窄的 K |
| `hard_k12` | Hardest-12 | 16 左侧 |
| `hard_k16` | Hardest-16 | 复现 37.32 |
| `hard_k24` | Hardest-24 | 16 右侧 |
| `hard_k32` | Hardest-32 | 更广竞争 |
| `hard_k48` | Hardest-48 | 进一步扩大 |
| `hard_k64` | Hardest-64 | 大 K |
| `hard_k96` | Hardest-96 | 接近 Global |
| `hard_k115` | 全 115 个负类 | Global 等价性控制 |
| `random_k16` | Random-16 | 区分难度效应与数量效应 |
| `static_text_k16` | Text-neighbor-16 | 区分动态视觉难负类与静态文本关系 |

### MECHANISM（18 组：负类来源、语义误伤、时间调度、Structure 交互、全局混合与连续加权）

| ID | 方法 | 核心问题 |
|---|---|---|
| `easy_k16` | Easiest-16 | Hard 的反证控制 |
| `quota_k16_local02` | Hard16 + 至少 2 个同物体负类 | 补充同物体判别 |
| `quota_k16_local04` | Hard16 + 至少 4 个同物体负类 | q 的敏感性 |
| `quota_k16_local08` | Hard16 + 至多可得的 8 个同物体负类 | 大量局部负类 |
| `quota_k32_local04` | Hard32 + 至少 4 个同物体负类 | 宽负类池与局部约束交互 |
| `samename_down05` | Hard16，同名跨物体负类权重 0.5 | 减轻同名语义冲突 |
| `samename_down025` | Hard16，同名跨物体负类权重 0.25 | 更大幅度权重降低 |
| `schedule_115_to_16_linear` | 115→16 线性退火 | 逐步聚焦难负类 |
| `schedule_115_to_16_cosine` | 115→16 Cosine 退火 | 退火曲率 |
| `schedule_115_warm5_16` | 前 5 Epoch Global，之后 Hard16 | 全局热启动 |
| `schedule_16_to_115` | 16→115 线性扩大 | 反向课程对照 |
| `hard16_structure2` | Hard16 + λ=2 | 负类与 Structure 的相对影响 |
| `hard16_structure6` | Hard16 + λ=6 | 高结构权重下的稳定性 |
| `hard16_globalmix005` | 0.95 Hard16 + 0.05 Global CE | 保留少量全局负类监督 |
| `hard16_globalmix015` | 0.85 Hard16 + 0.15 Global CE | 混合强度敏感性 |
| `hard16_globalmix030` | 0.70 Hard16 + 0.30 Global CE | 更多全局负类监督 |
| `softneg_k16_tau03` | 全负类连续门控，按第 16 难负类位置 sigmoid，温度 0.3 | 避免 TopK 硬截断 |
| `softneg_k16_tau10` | 全负类连续门控，温度 1.0 | 更平滑地抑制较弱负类 |

注意 `quota16_local08` 中若某 Object 仅有 2 个 Part，就只能强制 1 个同物体负类，其余自动从跨 Object 中补齐，不会创造不存在的类别。

## 3. 文件

- `train_pp116_hardest.py`：**唯一修改了训练计算的脚本**，完整独立可运行，复用原始 `train_relproto_alignemt.py` 的加载和检查流程，不替换原始文件。
- `run_hardest.py`：顺序训练、每 Epoch 保存 W、Bak​e + 官方 Evaluation、同名同参数断点跳过已完成实验、记录哈希、错误即停止。
- `selftest_hardest.py`：CPU 校验基线 Loss/W 梯度一致、Hard16 与历史定义数值/梯度一致、Hard115=Global、Quota 正类和补齐、随机种子隔离、调度端点。
- `summarize_hardest.py`：官方 116 类解析、A/B/C/D、逐类 IoU/Recall、由 IoU+Recall 推导的 Precision、mIoU 排名、固定 Probe 竞争转移、自动打包。
- `plot_hardest.py`：自动绘制全部最终 mIoU、K 扫描曲线、B/C Top1 竞争者替换比例（matplotlib 可用时）。
- `pp116_four_groups_per_part.csv`：以前按验证 IoU 轨迹整理的 A/B/C/D **分析标签**，不输入训练文件。
- `reference_previous_results_E30.csv`：之前实验的参考结果（仅供事后对照），不参与选择负类或 W 更新。

## 4. 服务器部署、自测与启动（GPU 3）

将 ZIP 放到仓库根目录后：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip -o PP116_HARDEST_DEEP_SEARCH_20261024.zip
python -m compileall -q hardest_search
python hardest_search/selftest_hardest.py
```

正常应打印：

```
PASS global CE bitwise; hard115 exact; hard16 CE and text gradient identical to previous search
PASS same-object quota 4 + global-hard 12; no duplicates; one true positive
PASS random/easiest/static-text/same-name losses finite, RNG isolation
PASS dynamic K boundaries and delayed warmup
PASS full original trainer baseline: total, NCE, relation, W gradients bitwise
SELF_TEST_PASS_HARDEST_DEEP_SEARCH
```

先做 Pilot（两组、2 Epoch、E2 的 Bake+Eval，独立目录）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u hardest_search/run_hardest.py --profile pilot --gpu 3 --epochs 2 --eval-epochs 2 --tag pp116_hardest_deep_20261024_v1
```

Pilot 成功后，**一条命令顺序执行全部 31 组 × 30 Epoch**：

```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u hardest_search/run_hardest.py --profile all --gpu 3 --epochs 30 --eval-epochs 3,10,20,30 --tag pp116_hardest_deep_20261024_v1 > pp116_hardest_deep_gpu3.log 2>&1 < /dev/null &
echo "PID=$!"
```

实时进度：

```bash
tail -f pp116_hardest_deep_gpu3.log
```

中断后：确保旧进程已退出，使用相同启动命令，会跳过完成训练和验证的实验。**部分 Epoch 训练中断不支持从中间 Epoch 精确恢复**，故调度器会拒绝把不完整训练视为已完成。若只需重新分析已经输出的官方评估日志：

```bash
python hardest_search/summarize_hardest.py --report output/pp116_hardest_deep_20261024_v1_all_e30
```

其他执行模式（互不混合，输出独立）：

- `--profile core`：13 组。
- `--profile mechanism`：18 组（无同目录 Baseline，跨报告比较应另读 Core 基线）。
- `--profile all`：31 组、建议使用。
- `--max-variants 3`：只选该 profile 的前三组，独立 `--tag` 防止 provenance 不匹配。

## 5. 输出目录与每个文件的用途

### 训练 Checkpoints（不需要上传）

```
final_exp/relproto_alignment/pp116_hardest_deep_20261024_v1_all_e30/<variant>/
    W_epoch_001.pt ... W_epoch_030.pt
    W_last.pt
    summary.json
    training_history.csv
    hardest_probe/
        probe_epoch_000.csv ... probe_epoch_030.csv
        probe_instances_epoch_000.csv ... probe_instances_epoch_030.csv
```

固定 Probe 是基于**训练缓存中的 image-level Presence**，对各类 Part 尽量覆盖 3 个固定 Crop，且每个变体/每个 Epoch 都用同一个确定样本子集；特殊稀有部件的样本数可能不足 3，详见 `n_probe`。非训练 GT 对齐指标：

- `full116_p_positive`：伪原型对所有 116 个文本类别的预测正类概率。
- `positive_minus_top1_logit`：正确标签 logit 相对最难负类的差值；并非 GT 局部 Patch 正确率。
- `global_top1_pid`、`global_top16_ids`：当前伪原型在全词表最难的竞争类及 Top16 集合。
- `top16_same_object_fraction` 和 `top16_same_suffix_fraction`：当前负类构成。
- `positive_count`、`fallback` 和 `effective_patch_count`：原始 Relative Competition 产生的候选支持信息。

**注意：** 即使训练方法是 Hard4/Random16/Quota16，固定 Probe 仍对所有方法统一用 global top16 检查当下模型的竞争几何，不把训练集合和分析集合混淆。

### 官方评估与整合报告

```
output/pp116_hardest_deep_20261024_v1_all_e30/
    provenance.json
    registry.json
    run_status.log
    preflight_selftest.log
    train/<variant>.log
    bake/<variant>_eXXX.log
    val/<variant>/eXXX/evaluation_stdout.log
    summary/
       variant_scorecard.csv
       ranked_final.csv
       official_116_per_part.csv
       delta_vs_retrained_baseline.csv
       perpart_precision_recall_area.csv
       ABCD_official_summary.csv
       all_training_history.csv
       fixed_probe_per_part_epochs.csv
       fixed_probe_crop_part_epochs.csv
       fixed_probe_same_crop_competition_changes.csv
       fixed_probe_group_competition_stability.csv
       fixed_probe_negative_changes.csv
       probe_vs_official_perpart.csv
       auto_report.md
       plots/*.png
    review_upload.zip
```

**明天上传**：

`output/pp116_hardest_deep_20261024_v1_all_e30/review_upload.zip`

不必上传 31 组 `.pt` 权重、完整缓存或原始 DINOv2 视觉特征。分析所需的轻量固定 Probe CSV 都会被汇总打进 ZIP。

## 6. 如何全面判断实验结论

优先按下列顺序检查，而不能只看最终 mIoU：

1. **复现控制**：`baseline_116` 与历史 Baseline E3/E10/E20/E30 是否一致；`hard_k16` 与历史 Hardest-16 是否一致；`hard_k115` 与 baseline 是否一致。
2. **K 效应**：画 E3/E10/E20/E30 的 K–mIoU 曲线、B/C 与 D 的收益—损失权衡，观察是否存在 U 型，而不是先假定 K=16 最优。
3. **难度 vs 数量**：`hard_k16` 对比 `random_k16`、`easy_k16`；Hard16 vs `static_text_k16` 检验视觉实例自适应有无价值。
4. **同/跨对象负类**：同 K 下比较 Quota2/4/8，结合固定 Probe 的 Top16 同物体比例，解释是否只用局部负类会失败、但强制少量局部负类能改善。
5. **同名语义冲突**：Hard16 与 SameName 0.5/0.25 对比，检查 `ear, leg, tail, head` 相关 B/C 部件是否改善。
6. **时间动态**：固定 Crop–Part 的 Top1 Rival 转移率、Top16 Jaccard 与 global 正类 Logit Gap；与 E3/E10/E20/E30 的验证 IoU 同时查看。
7. **Pseudo 质量代理**：结合 Fallback / positive_count / effective_patch_count，区分“负类导致的 logit 改善”与“伪标签候选支持真正改变”。这些无 GT 代理不能证明伪标签 GT Purity 提升。
8. **失败分类**：按历史 A/B/C/D 及 per-Part Recall、从 IoU 推算的 Precision，分类真实部件漏检和过预测趋势。没有保存验证级混淆矩阵，无法声称具体从哪个 Part 抢像素。
9. **Loss 均衡**：Hard16 + λ2/λ6 与 Hard16 正交比较，防止简单归因于负类策略而忽略 Structure 相对权重变化。
10. **稳定性和多种子**：31 组默认均为 seed123，只能作为方法筛选；任何 0.x pp 提升都必须选前几名以至少 2–3 个 seed 复验，再讨论统计显著性。

## 7. 风险及限制

- 31 × 30 = **930 个训练 Epoch**，每组至少 4 次官方 Evaluation，总计 **124 次**。历史记录约 75 秒/epoch，单纯训练约 19–21 小时，还需固定 Probe 和 Bake/Eval，可能跨越夜间。
- 与旧实验 `Hardest-16` 的小幅提升可能受单个 Seed 波动影响，本包不会把它包装成已确定的稳定论文贡献。
- Fixed Probe 只覆盖训练 Crop、只使用训练可获得的特征及 image-level labels，不衡量 GT Purity。若后续需要真正检验伪标签真实可靠性，应调用之前独立、**只读** GT 审计程序对优胜变体做离线分析，不能把 GT 引入这些训练函数。
- 如果训练阶段出现 NaN/Inf，禁止自动吞掉核心 Loss 或梯度的异常。调度器只会恢复**E30 模型全完成且错误仅在最终 JSON 统计写入**的经过检查的情况；其余情况停机供检查。
- 同一个 `--tag` 加 `--profile` 与 `--epochs` 形成独立实验目录；恢复时不要改变源码、参数、tag 或 profile。

## 8. 技术核验情况

- Python 语法 / AST 编译通过。
- CPU 单元测试：Global CE 与历史 CE bitwise 相同；Hard115 完全相同；Hard16 与历史的 Top16 CE / 视觉 StopGrad / 文本梯度相同；同 Object Quota 可以补齐全局负类；Random16 不改变 Python/Torch 全局随机数流；课程 K 在边界 Epoch 正确。
- CPU 模拟的完整 batch 测试：与原始 `train_pp116_softsingle_dual.py` 的总 Loss、NCE、Structure Loss 及 W 梯度 bitwise 相同。
- 固定 Probe 路径通过随机 Tensor 前向测试，生成 Part 汇总与 Crop–Part 轨迹文件。
- 官方日志解析和报告生成器在历史官方 E30 日志的模拟报告目录上测试通过。
- **尚未在实际服务器完成真数据 CUDA 的端到端 Pilot**；正式运行前先执行上述服务器 Pilot，不应把 CPU 单测等同于已经验证完整生产训练。
