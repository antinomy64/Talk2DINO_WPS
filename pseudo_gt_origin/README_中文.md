# PP116：每个 Part 的伪标签 Patch 来源 GT（E0、E10、…、E100）

## 统计目标

对 **116 个目标 Part**，针对原始 Ours 训练所用的每条 **预测物体 Crop**，在 **固定同一 Crop、同一预测前景、同一候选 Part Presence** 条件下，依次加载各个 Epoch 的共享正交 `W`，重算当时模型会选择哪些 Patch 来构建目标 Part 的伪视觉原型。**GT 在所有选择和权重计算完成以后才读取**，绝不用于选择、训练或反向传播。

**对每个目标 Part → 每个 Epoch → 每个来源 GT 类别**，统计：

- **Patch 数量占比**：该来源 GT 的选中 Patch 数量 / 目标 Part 所有 Crop 选中的 Patch 总数。对于数量占比，每个 Patch 权重相等。
- **伪原型 Softmax 权重占比**：每个 Crop 中落入该来源 GT 的 Softmax 权重之和，然后在当前目标 Part 的所有训练 Crop 上等权平均。每个 Crop 的全部权重合计为 100%；因此各 GT 来源权重占比总和为 100%。这沿用旧审计中“伪原型权重落在目标 GT 区域上的比例”的统计口径。
- **GT=255 Ignore** 单独列出，不能自动认为是错误 Part；其他类别保留准确的 **116 个 GT Part 名称**，不再只叫 Other。
- **特殊选择模式单列**：`正Margin全部Patch`、`Fallback单Patch`、`K1绝对相似度`；只有第一种是真正的“正 Margin 选中”；另外两种也参与总体伪原型，不能悄悄漏掉。

这里统计的是训练中的 **Crop-level 自生成 Pseudo Label**，并不是完整图片推理的 116 类结果，也不是官方验证 IoU。对于同一个 Patch，不同目标 Part 的独立伪原型可以重复选中；因此不同目标 Part 之间不能把 Patch 计数直接相加当作无重叠分割面积。

## 文件目录

主要脚本：`pseudo_gt_origin/audit_pseudo_gt_origin.py`。它依赖服务器仓库已有：

```
train_relproto_alignemt.py
train_pp116_softsingle_dual.py          # 可选：开启 --verify-trainer 时需要
audit_pp116_gt_weighted_purity.py       # 复用此前通过的 GTLookup.crop32
```

默认输入（在仓库根目录执行）：

```
训练 Crop 缓存： feature/pascalpart116_predobj_clip_struct/train_predobj_cropaug_bg055_reproduce.pth
原始 100Epoch Run： final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep
冻结的投影 Part 文本特征： <run>/projected_part_bank.pt
GT Mask： data/PascalPart116/annotations_detectron2_part/train
```

**不需要重新训练**。如果想研究 Hardest-12、Weighted InfoNCE、Seen-only 或其他模型，也可以把 `--run` 指向那次训练的 W checkpoint 目录，但要根据它已保存的 Epoch 修改 `--epochs`。该方法必须仍使用相同的 Soft Relative 伪原型生成规则，并且文本 bank、训练 Crop 缓存、Part ID 顺序必须对应。

## 1. 安装文件及 CPU 自测

把 ZIP 解压到 `/mnt/sda/master/code/lyx/Talk2DINO_official_bg` 根目录：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip -o PP116_PSEUDO_GT_ORIGIN_EVERY10_20261028.zip
python pseudo_gt_origin/selftest.py
```

## 2. 建议先跑少量 Crop 预检

```bash
CUDA_VISIBLE_DEVICES=3 python -u pseudo_gt_origin/audit_pseudo_gt_origin.py \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --epochs 0,10,20,30,40,50,60,70,80,90,100 \
  --max-crops 5 \
  --verify-trainer \
  --out output/_smoke_pp116_gt_origin_5crops \
  --device cuda
```

预检会测试实际训练器的伪原型数学一致性（至少第一个 K>=2 样本），以及选中 Patch 数量、Softmax 权重加总等守恒检查。

如果 `audit_pp116_gt_weighted_purity.py` 不在仓库根目录，请从之前 Ear 审计所使用的**原文件**恢复；不要临时替换 GT 变换算法。

## 3. 正式分析原始 Ours（E0–100，每隔 10 Epoch）

```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
python -u pseudo_gt_origin/audit_pseudo_gt_origin.py \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --epochs 0,10,20,30,40,50,60,70,80,90,100 \
  --tau 0.05 \
  --out output/pp116_pseudo_gt_origin_E0_E100_every10 \
  --progress-every 100 \
  --device cuda \
  > pp116_pseudo_gt_origin_E0_E100.log 2>&1 < /dev/null &

echo "PID=$!"
tail -f pp116_pseudo_gt_origin_E0_E100.log
```

中途意外中断时，**不要删除输出 SQLite**。在原命令后面加 `--resume` 即可安全续跑；它会验证运行配置一致并跳过已经完整处理过的 Crop。

## 4. 如果分析刚才训练的 Weighted InfoNCE（只能 E0–E30）

**只有在 Weighted InfoNCE 真正训练并保存了 W_epoch_*.pt 之后才能运行。**

```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
python -u pseudo_gt_origin/audit_pseudo_gt_origin.py \
  --run final_exp/relproto_alignment/pp116_weighted_nce_s050_t200_l4_seed123 \
  --epochs 0,10,20,30 \
  --tau 0.05 \
  --out output/pp116_weighted_nce_pseudo_gt_origin_E0_E30 \
  --device cuda \
  > pp116_weighted_nce_pseudo_gt_origin.log 2>&1 < /dev/null &

echo "PID=$!"
tail -f pp116_weighted_nce_pseudo_gt_origin.log
```

## 5. 输出文件

所有输出在 `--out` 目录下：

1. **`每个Part_每10Epoch_GT来源_权重占比_宽表.csv`**：每 Epoch × 116 行目标 Part，后面是 116 个来源 GT + Ignore 共 117 列；百分比，所有来源求和约 100%。适合直接打开分析“Cow Ear 伪标签的权重来自 Cow Head、Cow Neck、Cow Ear 各多少”。
2. **`每个Part_每10Epoch_GT来源_选中Patch数量占比_宽表.csv`**：同样 116×117 格式，但按选中的 Patch 个数（不是 Softmax 权重）计百分比。
3. **`每个Part_每10Epoch_GT来源_数量和权重占比_完整长表.csv`**：完整 E×目标 Part×GT 来源明细；含零数据行，保证不会漏掉 GT 类别。
4. **`每个Part_每10Epoch_来源GT权重Top5.csv`**：每个 Part 每个 Epoch 权重最大的前五个 GT 来源，方便快速寻找错误来源。
5. **`每个Part_每10Epoch_正确_错误_Ignore汇总.csv`**：目标 GT、自身以外的有效标注 GT、Ignore 的数量和权重比例。方便与之前审计对照。
6. **`按选择模式_每个Part_Epoch_GT来源.csv`**：把正常正 Margin、Fallback 和 K=1 拆开，防止把 Fallback 误认为正 Margin。
7. **`逐Crop_逐Part_逐Epoch_GT来源明细.csv.gz`**：实际每张 Crop、每个目标 Part、每个 Epoch、每个来源 GT 的 Patch 数与加权占比；解压后 CSV，可查具体失败样本。
8. `pseudo_gt_origin.sqlite`：中间逐 Crop 原始汇总及续跑账本。与 CSV 一致，可做更细 SQL 分析。
9. `audit_settings.json` 与 `complete.json`：运行输入、实际记录数、完成状态。

> Excel/CSV 中“目标”指正在生成视觉伪原型的 Part；“来源 GT”指模型选中 Patch 在离线 GT 里真正属于的 Part。对于确实没有训练 Crop 的 Part，相关比例留空，**不写成 0%**。

## 6. 统计口径实例

假设目标 `cow's ear` 某个 Crop 选中 4 个 Patch，它们在 GT 中分别是：

- 1 个 `cow's ear`，其 Softmax 权重为 0.6；
- 2 个 `cow's head`，权重分别为 0.1、0.2；
- 1 个 GT Ignore，权重为 0.1。

则该 Crop 内：

| GT 来源 | 数量占比 | Softmax 权重占比 |
|---|---:|---:|
| cow's ear | 25% | 60% |
| cow's head | 50% | 30% |
| GT Ignore | 25% | 10% |

这两个百分比都正确，回答的是不同的问题。若同一目标 Part 有多个训练 Crop，宽表的数量百分比会按全部选中 Patch 计数汇总，而 Softmax 权重百分比则先对每 Crop 加权，再按 Crop 等权平均。**不混淆这两种统计。**

## 7. 数据自检与风险

- 不存在任何权重在 GT 映射之外被丢弃：每个 Crop 伪原型 Softmax 权重在所有 116+Ignore 来源上的总和为 1；每个 Crop 选中 Patch 在所有来源上的计数恰好等于所选数量。
- Part 文本顺序严格来自同次训练保存的 `projected_part_bank.pt`，而不是靠猜 ID。
- `GT=255` 的比例只能称为“未标注区域获得的权重”，**不能直接叫错误比例**。
- K=1 用所有预测前景 Patch 进行绝对 Softmax，没有 Relative Margin 的概念。Fallback 使用最大 Margin 的 1 个 Patch，即便 Margin≤0 也会参与原型。
- 最终统计是 **PP116 训练 Crop**，不是 Val 850 图像的全图分割。验证集 IoU 必须独立从官方日志读取。
- 如果有运行中已有同名数据库，脚本会拒绝覆盖；只有 `--resume` 才可复用并且校验配置。
