# PP116 Ear 跨物体迁移与自举机制：夜间实验包 v2（2026-10-22）

## 0. 要回答的问题

`cow's ear` 的 E0 真正通过正 Margin 命中的 GT 耳朵 Patch 只有 **1 个**（固定前景与 Presence 下），E10 真实耳朵上的正 Margin 明显增加。究竟是 Cow 自己的噪声伪目标学习，还是 Cat/Dog/Horse/Sheep/Person 五种 Ear 的正类监督迁移？**另外五种 Ear 是否也有类似跨物体迁移？**

上一轮已经发现“六类 Ear 同时屏蔽正类监督时，Ear IoU 全部降为 0”，“只训练六类 Ear 时出现高召回低 IoU”。但无法从中识别 *单一* Ear 与其他五种 Ear 的贡献。本实验补上逐类 Leave-One-Out 和控制型正类集合。

## 1. 实验控制与准确的解释边界

- 完整 **116 类 InfoNCE 分母**永远保留（包括被屏蔽的 Ear）；只把某些 Part 作为正目标的交叉熵项乘 0。
- **所有** Part 仍然根据真实训练时的 image-level Presence 参与候选竞争（K≥2 正 Margin + argmax fallback；K=1 全前景 absolute Softmax）；从来不为了消融去删除候选类别。
- 每个 Crop 内部分部件损失始终保留原始 `1/K`，Batch 内所有 Crop 仍等权。**绝不对剩余正样本数量重归一**。
- 假如一批次完全没有当前要保留的正目标（尤其是 Only-Target），零损失保持可微但梯度为零；Adam 仍执行原脚本每步逻辑。运行时请查看 target_positive_terms/positive_terms_used。
- 冻结 CLIP、COCO2014 Talk2DINO projector、DINOv2 Patch Cache 和预测物体前景。只训练 768×768 正交 W、Adam lr=1e-5，每步 SVD 重正交；seed=123，batch=16，epochs=10，τsoft=.05，τnce=.07，rank delta=.05，margin=.05，λ=4 或 0。推理全 116 类，无 PAMR。
- Structure Loss **保持原始 K≥3 关系排序定义**；它对所有 Part 的可微 Soft Prototype 起作用，不因某 Part 正类 InfoNCE 被 mask 就将其从排序中删掉。
- 训练读取原始 **无空间监督**缓存 + 文本类别名。GT 空间 mask 只存在于 `audit_only_reference/*.npz`，仅由 `ear_transfer_probe.py` 离线读取，用于审计真实 Ear Patch Margin、Purity，不作为训练的 loss / batch 采样 / 伪标签 / optimizer 目标。
- 对比尽量复用上一夜的单变量控制模型（这些实验已实际完成）；如果服务器找不到对应完整模型，会在 **本次新目录**内重训该控制，不覆盖旧模型。主要 Baseline `full_l4` 必须已存在，绝不自动重训或替换。
- 这是一份 **训练归因/消融** 包，不是 GT 教师监督包；只报告从真实日志与真实权重生成的数据。

## 2. 模型正类集合（每个目标目标 c ∈ {Cat,Cow,Dog,Horse,Person,Sheep}）

| 名称 | c 正类 | 其他五种 Ear 正类 | 非 Ear 110 类正类 | 精确测试问题 |
|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | Ours 基线 |
| `loo_c_l4` | × | ✓ | ✓ | **不靠自己**，还恢复吗？ |
| `target_nonear_c_l4` | ✓ | × | ✓ | 其他五个 Ear 的贡献大小 |
| `other5_c_l4` | × | ✓ | × | 五个 Ear **独立**能带动目标 Ear 吗？ |
| `only_c_l4` | ✓ | × | × | 自己的噪声伪原型独立能学吗？ |
| NonEar-only | × | × | ✓ | 他类是否能独立带动？来自上一夜 |
| AllEars-only | ✓ | ✓ | × | 六种 Ear 监督容易过度预测？来自上一夜 |

`core`（推荐第一晚）新训练 **18 个模型**：六类各自的 `Leave-One-Out λ4` 和 `Target+NonEar λ4`（共12组）；Cow 单独的 `Other5 λ4`、`OnlyCow λ4`（2组）；以及 Cow 对应四种正类集合在 `λ0` 下的无结构对照（4组）。同时读取/复用六个上一夜/原始的公共对照（Full、NonEar、AllEars，各 λ=0/4），总计**24个对照行**；只有18个需要新训练。

`cow`：只运行 Cow 上的上述 8 个新训练实验（外加六组可复用公共对照），适合先做快速深入验证。

`extended`：在 `core` 的基础上，对另外五种 Ear 补做 `Other5 λ4 / OnlyTarget λ4 / DropTarget λ0 / Target+NonEar λ0 / Other5 λ0 / OnlyTarget λ0`，新训练总计 **48 个**，适合覆盖六种 Ear 的完整 λ=0/4 双因素正类监督集合。若服务器算力不足，建议第一晚先 `core`，第二晚使用相同 tag 运行 `extended`；已完成的 core 变体会直接复用。

## 3. 核心机制分析如何读数

对某目标 Ear c，在 `λ4` 下定义官方验证 IoU（同一 116 类词表）：

- **自己的边际必要性**：`IoU(Full) - IoU(LOO_c)`。较大正数表示缺乏 c 正类监督时显著退化。
- **其他五个 Ear 在 c 存在时的边际作用**：`IoU(Full) - IoU(Target+NonEar_c)`。
- **其他五个 Ear 在 c 不存在时能否提供作用**：`IoU(LOO_c) - IoU(NonEarOnly)`。
- **非加性交互**：`IoU(Full)-IoU(LOO_c)-IoU(Target+NonEar_c)+IoU(NonEarOnly)`。这是评价指标层面的交互，**不是**梯度的线性可加归因。
- **更强的跨物体 Ear 测试**：`Other5-only` 的 *官方验证 IoU* 是否能高于初始 E0，不能只看真实 Ear Patch 正 Margin/Recall。高覆盖但低 IoU 可能是过度预测。
- **“每种 Ear 都这样吗？”** 看六行指标和相同六个固定 Crop、无需自行推断：每种 Ear 的 IoU/Head IoU/召回率、E0–E10 Margin/Purity/Fallback 都输出。Person Ear 可作为恢复较慢的反例。
- **结构效应**：Cow 对应 λ0 与 λ4 同项比较，可区分 InfoNCE 自举与结构帮助；全面检查其他五种 Ear 请运行 `extended`。

`gradient_by_target.csv` 是离线对 `∇_W L_target`、`∇_W L_other5`、`∇_W L_nonEar`、`λ∇_W L_struct` 的精确损失梯度拆分与切空间 **局部 SGD 方向预测**（用于 six fixed Crop probes）。**不代表** Adam 动量/SVD 更新的精确因果贡献，且只在小样本批次上测量。可靠的机制结论以官方验证 IoU 和模型消融为核心。

## 4. 解压、预检（所有操作在服务器仓库根目录）

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
unzip PP116_EAR_TRANSFER_NIGHT_20261022.zip

# CPU 合成测试：原始 full mode loss / W grad 与新版一致，掩码拆分正确
PYTHONPATH="$PWD" python ear_transfer/test_transfer.py

# 不训练；用当前 Baseline E0 的真实缓存+历史 GT 审计样本检查审计兼容性
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python ear_transfer/ear_transfer_probe.py \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --epochs 0 \
  --output output/ear_transfer_probe_smoke \
  --device cuda
```

预计输出：`SELF_TEST_PASS_EAR_TRANSFER`、`[audit] ... epoch=0 fixed_Ear=6`。第一个测试不加载任何实际数据，第二个只读原始 6 份固定审计 NPZ 和当前训练缓存。若实盘缓存顺序与 NPZ provenance 不一致，必须失败，不应强行继续。

## 5. 一键夜间运行

推荐 **core**（18 个新模型+6 个复用对照，官方验证 E3、E10，固定六个 Crop 的 E0–E10 逐轮审计，附 Baseline 在六种 Ear 上的局部梯度分解）：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python -u ear_transfer/run_transfer_night.py \
  --profile core --gpu 0 --eval-epochs 3,10 --gradient-baseline \
  > ear_transfer_launcher.log 2>&1 < /dev/null &

echo "PID=$!"
```

查看总进度：

```bash
tail -f ear_transfer_launcher.log
```

每个实验内部训练进度/错误：

```bash
tail -f output/pp116_ear_transfer_20261022_v1/train/loo_cow_l4.log
```

谨慎执行 `--profile extended`（48 个新模型，总时长和占用显著增加）；它会自动复用已经完成的 core 模型，**不会重新训练或覆盖**。需要独立运行前，避免两个运行器同时操作相同 `--tag` 目录。

```bash
python -u ear_transfer/run_transfer_night.py --profile extended --gpu 0 --eval-epochs 3,10 --gradient-baseline
```

只关注 Cow：`--profile cow`。预检单模型：`--profile smoke --skip-eval`（但即使 smoke，训练仍是完整10 Epoch）。需要全官方验证，可改成 `--eval-epochs 1,3,5,10`；会增加耗时。可选 `--skip-probe`、`--skip-eval`、`--skip-train`。

**安全性与断点**：一个训练目录若已经完整保存 `W_epoch_010.pt` 和 `summary.json`，会直接复用；若存在但不完整，则拒绝覆盖，要求检查或换新 `--tag`。评估 `EVALUATION_DONE` 只有在真实官方日志可解析时才创建。实验失败时返回非零退出码；不要只看 shell 中的最终 DONE，要检查该目录的 `run_status.log`。

## 6. 产物说明

训练输出：

```text
final_exp/relproto_alignment/pp116_ear_transfer_20261022_v1/
  loo_cow_l4/
    W_epoch_001.pt ... W_epoch_010.pt
    training_history.csv, summary.json
  target_nonear_cow_l4/ ...
```

审计和对照汇总：

```text
output/pp116_ear_transfer_20261022_v1/
  provenance.json                  # git SHA + 源码 SHA + 协议
  registry.json                    # 每个变体实际权重路径（含旧实验复用信息）
  run_status.log                   # 失败/复用/训练/评估进度
  train/<variant>.log              # 完整训练日志
  probes/<variant>/
    fixed_crop_ear_epoch.csv        # 6 Ear × E0..E10 的 GT margin, Purity, Coverage, Rival Cosine
    gradient_by_target.csv          # Baseline 可选六种 Ear 的 4 路梯度分解
  val/<variant>/e003/evaluation_stdout.log
  val/<variant>/e010/evaluation_stdout.log
  summary/
    official_ear_head_iou_acc.csv    # 官方 850 图像的真实分割指标
    e10_six_ear_official_long.csv
    six_target_factorial_l4.csv     # 六种 Ear 的逐类对照，差中差
    cow_full_factorial_lambda0_lambda4.csv
    fixed_six_ear_crop_epoch.csv
    baseline_gradient_target_other5_nonear.csv
    e10_fixed_probe_vs_val_IoU.csv
    results_explained.md
    six_ear_official_loo_compare.png
```

**所有 Ear IoU 必须来自 `evaluation_stdout.log`，不能用训练 Crop 的 GT Margin / 覆盖率冒充。** 对应类别的 Head IoU 和 Ear 类别召回率会同时保留，以排查像素过度预测。

## 7. 汇总和回传结果

即使中途停止也可以手动汇总已有真实结果（不存在的结果保留 NaN，绝不填数）：

```bash
python ear_transfer/ear_transfer_summarize.py \
  --report output/pp116_ear_transfer_20261022_v1

python ear_transfer/pack_results.py \
  --report output/pp116_ear_transfer_20261022_v1 \
  --out output/pp116_ear_transfer_review_upload.zip
```

把 `output/pp116_ear_transfer_review_upload.zip` 传给 ChatGPT 即可。这份上传包不包含大型 `.pt` 权重、GPU特征缓存或 GT npz；保留代码、配置、数值、官方日志和对照矩阵。

## 8. 仍然存在的因果边界

1. 单一 seed=123，10 Epoch。建议在观察主效应后再重复 2–3 个 seed；不能先用 GT 筛样本或选超参。
2. 正类项 mask 后固定分母 1/K 不变，是严格保留每项原始系数的公平消融；但 **整体损失标量及相对 Structure 权重必然改变**。因此组合损失的贡献不是简单线性或 Shapley 因果分解。
3. `OnlyTarget`/`Other5-only` 会造成极端类别不平衡，可能产生高召回低 IoU，必须看 Head 和全体 mIoU，而非纯覆盖率。
4. 结构关系损失仍包含被 mask 正类的 Part；只屏蔽 CE 不等于从模型/候选集中移除这些部件。
5. 离线梯度探针对六个固定 Crop 和少量非 Ear 样本测量的是局部一阶方向，不是实际 Adam + SVD 的逐步参数更新归因。
6. 6 个 Ear 仅是 PP116 中的 6 个具体类别，不能将结论自动扩展到所有部件名称或其他数据集。

## 9. 可选：全部训练 Crop 的 E0–E10 正 Margin 审计（比固定六个 Crop 更全面）

需要服务器已经存在此前成功使用的 `audit_pp116_ear_patch_e0_e10.py`。**建议核心消融及官方验证完成后再执行**，因为每个变体会重新读取全部 Ear Crop、输出较大的 Patch NPZ 文件。

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

CUDA_VISIBLE_DEVICES=0 python -u ear_transfer/run_fullcrop_selected.py \
  --report output/pp116_ear_transfer_20261022_v1 \
  --variants full_l4,loo_cow_l4,target_nonear_cow_l4,other5_cow_l4,only_cow_l4 \
  --max-ear-crops 0 --gpu 0
```

前期探针可先 `--max-ear-crops 3` 测试。需要注意这会生成与全量实验同名的输出，因此如需在测试后再跑全量实验，请使用另一个 `--report` 路径或手动检查并清理该测试输出，**不要在正式输出上先测3个再跑全量**。在全量审计后查看 `output/<tag>/allcrop/<variant>/ear_epoch_summary.csv` 和 `per_crop_epoch.csv`，比较整个 Ear 类别而非仅单个代表 Crop。
