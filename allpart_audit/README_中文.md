# PP116 全部 116 Part 原始 Ours 伪标签与竞争关系审计（只读夜间包）

## 0. 审计对象与不可改变的条件

仅分析**已经完成训练**的原始 Ours：`train_pp116_softsingle_dual.py`，`--part_setting presence`，全 116 类 InfoNCE（**不是** Object-Local InfoNCE），`soft_temperature=0.05`，`nce_temperature=0.07`，`lambda_struct=4.0`，`rank_delta=0.05`，`rank_margin=0.05`，W 为 768×768 正交矩阵、Adam + 每步 SVD，冻结 DINOv2、CLIP、Projector 以及预测前景。

**不重新训练、不修改已有 Checkpoint、不修改仓库原有训练代码、不读取 GT 作为训练监督。** GT 仅用于后验审计；GT 从 `audit_pp116_gt_weighted_purity.py` 的 `GTLookup.crop32()` 重建到 **同一预测物体 Crop 的 32×32 网格**。这个辅助脚本是先前 Ear 审计使用过的服务器原文件，本包不覆盖它。第一次请检查它仍在仓库根目录。

全部类别 **以 `RUN/projected_part_bank.pt` 的 classnames 顺序定义 ID**；A/B/C/D 历史表只通过 `part` 字符串关联，不能把表格行号当作 PP116 类别 ID。类别顺序不匹配时直接报错。

之前冻结的分类：A=20，B=11，C=15，D=68，另外 2 类验证 GT 无效不参与 ABCD，但仍可被审计。直接复用历史 `pp116_four_groups_per_part.csv` 的原始类别、阈值和 IoU；**不会为了迎合机制重新划组**。原始历史 IoU 是独立的官方验证记录，须核对其对应原始实验运行版本。审计 Crop 质量不等于验证 IoU。

## 1. 分层审计设计：每个 Part 发生了什么

### 一层：前景和 Presence 能否提供学习机会（全部 Object Crop）

1. 对每种 Part 统计有 GT 的 Crop 数、GT Patch 数、GT 有多少落在冻结的预测 FG，`FG recall`。
2. 找出 GT 确实包含某 Part，但该 Part **未出现在图像级 Presence 候选集合**的 Crop：这类 Crop 的相对竞争压根不会给这个部件形成视觉原型。
3. 区分零数据、K=1 不存在相对竞争、K≥2 竞争但失败，以及 GT 主要是 Ignore 的情况。

输出：`summary/part_presence_and_foreground_coverage.csv`。

### 二层：动态伪标签/视觉原型质量（所有可训练 Part occurrence，多个 Epoch）

对于真实训练同一条 Cache Crop，在固定 Patch、FG、候选集合下，加载每个 Epoch 的 W，**完全照搬原始 trainer 生成伪原型**：

- `K>=2`: `margin = own cosine - maximum other present Part cosine`，只取正 Margin Patch；若全非正，Fallback 仅选最大 Margin Patch；对候选 patch 的 margin/0.05 做 Softmax，并池化、归一化。
- `K=1`: FG 上绝对 cosine/0.05 Softmax；没有“正 Margin”定义，不应统计为竞争失败。
- 完整 116 类 `[v.detach() @ text.T]/0.07` 的全局 InfoNCE；**禁止读取局部 InfoNCE 新实验的权重和代码**。
- Structure 排序损失按原始冻住的文本关系和动态视觉原型逐 Crop 重算。

每个 Part occurrence × Epoch 记录：

| 方面 | 变量 | 解释 |
|---|---|---|
| GT 覆盖 | `n_gt`, `n_gt_fg`, `fg_recall` | 部件 GT 与预测 FG 的交集、漏掉多少 |
| 伪证据是否真 | `q_gt`, `q_other`, `q_ignore` | **Softmax 权重总质量**分别落在真正目标/其他有效 GT/Ignore；三者约和为 1 |
| 条件纯度 | `q_gt_valid` | 在**有效标注**质量之中，目标占多少；可能因高 Ignore 而虚高，不等于 q_gt |
| 离散支持 | `selected_gt/other/ignore`, `selected_valid_precision` | 正 Margin/Fallback 选择数，无权重信息 |
| 覆盖 | `selected_gt_fraction`, `gt_margin_positive` | 正确 GT 有多少被选中/成为正 Margin；K=1 置空 |
| 失败分解 | `fallback`, `candidate_positive_count`, `pos_on_other_count` | 是否只能用 Fallback、错误区域也是否高响应 |
| 分数来源 | `mean_own_cos_gt`, `mean_rival_cos_gt`, `mean_margin_gt` | 目标上 own 与最强 rival 的相对优势 |
| 强竞争者 | `rival_on_gt`, `rival_on_other` 与直方图 | 实际同物体候选 Part 的最强竞争者，不是全 116 类 |
| 注意力集中 | `max_weight`, `entropy`, `effective_support` | 伪标签是否压在很少 Patch；**集中不必然正确** |
| GT Oracle | `pseudo_oracle_cos` | 动态伪原型 vs **同 Crop 真 Part & FG Patch 均值原型**；后验 GT 指标 |
| NCE 监督自洽 | `global_nce_p_true`, `global_nce_rank`, `global_nce_top1`, `global_nce_loss` | 原始 116 类伪原型上的真实正类 softmax；可高度自信但目标错 |
| Structure | `struct_loss_crop`, `struct_triplets`, `structure_rank_satisfaction` | 按原公式动态重算，统计关系排序成立比例；K<3 时 rank 满足率置空 |

以上 GT 质量指标**只供离线解释，训练时不存在这些 GT 权重或筛选条件**。

SQL 明细：`output/.../audit.sqlite`，表 `part_epoch`（一条记录是一个训练 Crop × Part × Epoch），`gt_presence_coverage`，`transition` 和 `epoch_done`。

### 三层：竞争机制演化

对同一个固定 GT Patch，跟踪相邻已保存 Epoch：

- `negative -> positive` 和 `positive -> nonpositive`，**得/失必须同时统计**；
- own cosine 增长、最强竞争者下降，还是 own 增长得更快；
- 最高竞争者是否换成了别的 Part；
- Support Jaccard 与权重 Total Variation；
- 新增正确证据和新增错误证据的数量；
- 即使 Margin 越来越正，是否同样在错误或 Ignore 区域取得正 Margin。

输出：`summary/per_part_epoch_transitions.csv`、`summary/per_part_E0_Eend_endpoint_changes.csv`（E0→最终 Epoch 的**净变化**，不重复计数中途来回翻转）与 `summary/part_competitor_edges.csv`。其中 rival 图同时区分 `GT_target` 和 `GT_other` 两种区域。**Cross-Object 类不会出现在 Relative Competition 竞争者列表里**，但其全局 InfoNCE 能通过 W 更新造成间接影响。

### 四层：无标签“自信度”与真实伪标签质量是否一致

`global_nce_p_true`、`max_weight`、`normalized_entropy` 都是不依赖 GT 可以计算的信号；按 10 个区间汇总平均 `q_gt`、`q_ignore`、`q_gt_valid`。输出：`summary/confidence_proxy_vs_GT_reliability.csv`。

**这些信号不是校准概率，只有后验 GT 测量才能检验真假。** 特别检验高 NCE 自信却低 Q_GT、高 Ignore、单 Patch Fallback、错误部件支持持续增长等失配。

### 五层：A/B/C/D 与官方 IoU 的对应

`summary/per_part_epoch_metrics.csv`：116 个 Part 在每个审计 Epoch 的训练 Crop 统计，并关联历史验证 IoU（E0、E10、E20...E100）。`summary/ABCD_epoch_macro.csv`：对 **Part 等权宏平均**，不是让训练 Crop 数量多的部件占更大权重。`summary/per_part_outcome_disagreement.csv`：重点寻找“Q_GT 上升但 IoU 下降”、初始有效反而退化、始终靠 Fallback、早峰后恶化等现象。

同一个 Part 可能存在不同 Crop 机制。`visuals` 会在每个 ABCD 组中自动选取高训练样本数的 Part，分别展示 Q 高/低的典型 Crop；图里将 GT 与权重、Margin、错证据、竞争者分开可视化。**这只是展示，不能替代全数据汇总**。

### 六层：其他 Part 通过共享 W 产生的间接作用（可选，较耗时）

`gradient_source_probe.py`：选固定采样的训练 Crop 构建**原始 116-way InfoNCE + 原始 Structure**，对每个**有出现的正类标签**分离 InfoNCE 梯度，再将所有来源按目标 Part 分成互斥四组：

1. SELF：目标 Part 本身正类 InfoNCE；
2. SAME_OBJECT_OTHER：同物体的其他 Part；
3. SAME_PART_OTHER_OBJECT：其他物体中同后缀 Part（例如 cow's ear 和 dog's ear）；
4. OTHER_PART_OTHER_OBJECT：其余所有非同物体、非同名 Part。

另外单独计算 `STRUCTURE` 梯度。目标函数为固定训练 Crop 的**GT Part Patch 的平均相对 Margin**，不进入训练 Loss。局部指标：

\[
D_j=-\langle\Pi_W(\nabla_W M_{\rm GT}),\Pi_W(\nabla_W L_j)\rangle.
\]

输出：`gradient/by_source_part.csv`（每个探针目标×有出现的来源 Part）和 `gradient/grouped_source_vs_target.csv`。

**注意：** 非目标正类的 116-way InfoNCE 分母仍包括全部文本类别。这是**按照损失正类来源拆分梯度**，不是说同源 Part 直接参加同物体 Patch Competition。探针先验证所有 source Loss 和 W 梯度相加重建原始 trainer Loss/梯度，再进行归因。仅是随机种子固定的**分层抽样/切空间一阶 SGD 方向**，不等于完整训练的 Adam+SVD 参数更新因果效应。没有出现在 probe 采样里的 Part，不得记为“全训练梯度为零”。

## 2. 推荐的服务器执行方法（没有 .sh）

把 ZIP 解压到**仓库根目录**，得到 `allpart_audit/` 文件夹：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip PP116_ALLPART_AUDIT_20261022.zip
```

先检查所需的**原始**代码和 GT 后验映射脚本存在：

```bash
ls -lh train_pp116_softsingle_dual.py train_relproto_alignemt.py audit_pp116_gt_weighted_purity.py
ls -lh final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep/projected_part_bank.pt
python allpart_audit/selftest_audit.py
```

正常应出现 `SELF_TEST_PASS_ALLPART`。此测试不需要训练数据或 GPU。

### A. 小规模真实缓存冒烟测试（24 Crop × E0/E1/E2）

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u allpart_audit/run_night.py --smoke --device cuda
```

本模式输出 `output/pp116_allpart_audit_ours_v1_smoke/`；不运行可视化/梯度探针，**不能拿来得出所有类别的统计结论**。

### B. 完整夜间审计（主推荐，全部 Crop、E0～E100 指定 15 个快照）

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u allpart_audit/run_night.py \
  --out output/pp116_allpart_audit_ours_v1 \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --device cuda \
  > allpart_audit_launcher.log 2>&1 < /dev/null &

echo "PID=$!"
```

**默认阶段**：①只读全量重算 + 断点写入 SQLite，②汇总与曲线，③分层梯度探针，④ABCD 各类典型 Patch 可视化，⑤自动压缩分析 ZIP。每步详细日志在 `output/pp116_allpart_audit_ours_v1/logs/`。

逐行看状态：

```bash
tail -f allpart_audit_launcher.log
```

如果夜间训练、系统重启或中断，要**先确认旧进程已经退出**，再运行同样的完整命令；审计读取 `epoch_done`，自动跳过已写入的 Crop×Epoch，参数/分类表变化会被拒绝混入旧数据。

### C. 如果只要全量前向审计、不需要梯度

在完整命令后加 `--no-gradient`。如果不需要示例 PNG，加 `--no-visuals`，还可加 `--no-plots`。也可以全部审计结束后单独运行：

```bash
python allpart_audit/gradient_source_probe.py \
  --out output/pp116_allpart_audit_ours_v1 \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --epochs 0,1,3,10,30,100 --device cuda
```

默认目标为 ABCD 每组约 4 个，source batch 64 个贪心覆盖样本；想用全部审计库挑选所有有 GT-FG Patch 的目标 Part，可指定 `--full-gradient-targets`（最多约 114 个有效 A–D 目标，部分缺乏可审计 GT 的 Part 会被跳过），**计算明显更重，source 仍然是固定采样探针**。

### D. 手工汇总/可视化/打包（都是直接 CMD，不需要 .sh）

```bash
python allpart_audit/summarize_all_116.py \
  --out output/pp116_allpart_audit_ours_v1 --plot

python allpart_audit/render_exemplars.py \
  --out output/pp116_allpart_audit_ours_v1 \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep --device cuda

python allpart_audit/pack_results.py \
  --out output/pp116_allpart_audit_ours_v1 \
  --zip output/pp116_allpart_mechanism_review_upload.zip
```

只需要上传 `output/pp116_allpart_mechanism_review_upload.zip`。**不需要上传**动辄数百 MB 甚至 GB 的 `audit.sqlite`、训练缓存、DINO 特征和 Checkpoint。

## 3. 输出检查与科学边界

- 所有 Checkpoint 由 `W_epoch_XXX.pt` 精确加载，E0 强制 W=单位矩阵；E1/E2 等可看早期弱伪标签自举。
- 本次**不改变任何训练参数**，只读重放历史状态。源码自测验证 `K>=2` 相对竞争，`K=1` 绝对 Softmax，原始 116-way CE，结构排序损失。使用 `--verify-source` 的真实缓存抽查再与原始 trainer 原型交叉校验。
- GT Ignore=255 必须单列；有 GT 但预测 FG 漏掉、以及 image-level Presence 缺失必须分开报告。
- 对于 K=1，Margin/最强竞争者不能定义，写空而不是 0。
- 对于 GT 不存在的 Crop，不能把 `q_gt=0` 当作伪标签错误的证据；要参考 `n_gt_observed_crops` 和 `n_gt_in_fg_crops`。
- 116-way NCE 的 `p_true` 是网络对伪原型的分类概率，不是伪标签空间正确率。
- 正确 Part Patch Margin 上升不代表伪标签质量提高；错误和 Ignore 区域上的 Margin、权重、false positives 也必须审计。
- **训练 Crop 的 32×32 GT Patch 指标不等于验证集 448 像素 IoU**。历史表里的验证 IoU 只作为独立参考，用之前的官方协议获取。
- 带有 GT 的局部梯度目标仅用于诊断导数方向，不是训练监督；不能据此断言所有 Adam Step 的因果归因。
- 所有历史模型结果来自单个 Seed/单次轨迹，分类和解释有样本覆盖差异，不能宣称统计显著性。
- 若某项工具失败则终止并保留已完成阶段，**不会用空值填充后假装完成**。

## 4. 文件说明

- `audit_all_116.py`：主审计、GT-分离、SQL 原子写入、可恢复；默认所有训练 Crop。
- `summarize_all_116.py`：逐类轨迹、ABCD 宏平均、竞争图、伪标签可信度、Q-vs-IoU 对照，内存随每个 Part 的 Crop 数增长，而非所有 Crop 总数。
- `render_exemplars.py`：回放 ABCD 样本的 GT、Margin、Weight、错误支持、竞争者图，并保存对应 NPZ（方便复核）。
- `gradient_source_probe.py`：分来源真实梯度、一阶方向/语义后缀迁移、原训练器重建一致性检查。
- `pack_results.py`：仅压缩轻量报告和视觉图；不带权重和 SQL 大库。
- `run_night.py`：Python 顺序编排和阶段日志；**不是 .sh**。
- `selftest_audit.py`：CPU 公式一致性测试。
- `pp116_four_groups_per_part.csv`：历史 A/B/C/D 分类及 E0～E100 验证 IoU，不重新制定分类标准。
- `reference_original_training_OURS.py`：随包只读参照代码，实际检查始终使用服务器仓库根目录的原训练模块，不覆盖任何原文件。
