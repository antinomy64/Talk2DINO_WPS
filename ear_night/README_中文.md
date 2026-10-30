# PP116：六种 Ear 早期恢复的梯度归因与受控消融（夜间实验包 v1）

## 0. 研究问题和协议

目标不是再次证明 Ear 的 GT Patch Margin 从负变正，而是区分 **驱动共享 W 更新的来源**：

- **Ear-positive NCE**：六种 Ear 自身作为正目标的 InfoNCE 项；可包含大量噪声伪原型。
- **Other-positive NCE**：所有非 Ear 部件作为正目标的 InfoNCE 项，**仍然使用 116 类完整分母、Ear 仍是负类**；这可以通过共享 W 间接改变 Ear。
- **Structure**：原代码中的 visual-prototype ranking（原始文本两两关系作为排序先验）。在 K<3 时等于零；只有当前相对选区的 Softmax 聚合路径可微。

方法与上传的 `train_pp116_softsingle_dual.py` 保持一致：presence 候选、冻结特征与前景、K≥2 的全部正 Margin + 无正 Margin 时单 Patch fallback、K=1 全前景 Softmax、tau=0.05、InfoNCE tau=0.07、lambda=4/0、rank_delta=rank_margin=0.05、116-way CE、equal-annotation（每 Crop 内 1/K，然后 Crop 等权）、W[768×768] 单独 Adam lr=1e-5 + 每步 SVD、batch=16、seed=123。只训练 10 Epoch，推理关闭 PAMR。

**严格的 GT 边界**：`night_ear_train.py` 只读取与原训练相同的预提取视觉缓存和图像级 Part Presence/文本词表，不读取任何 GT Part mask。六份 `audit_only_reference/*.npz` 保存了历史审计中的 GT 对应，仅由独立的 `night_ear_gradient_probe.py` 读入，作为离线 **GT Ear Patch Margin 探针**；不用于构造伪目标、训练梯度、超参搜索或 Adam 更新。评估时的 GT 只计算指标。

## 1. 目录与安全

```text
Talk2DINO_official_bg/
  ear_night/
    night_ear_train.py                 新训练脚本（复制基线，加入只作用于 NCE 正类的开关）
    night_ear_gradient_probe.py        固定样本上的 loss/gradient/margin 归因，不更新参数
    night_ear_summarize.py             只基于实际输出产生表格与图
    run_ear_night.sh                   顺序实验运行器
    test_loss_equivalence.py           CPU 合成测试（验证基线函数及梯度完全一致）
    baseline_reference.py              用户提供的原始训练脚本快照（对照，不调用）
    audit_only_reference/              六个历史代表 Crop 的冻结 GT 探针（绝不训练）
    README_中文.md
```

**不会**修改仓库官方文件、不覆盖训练权重、不改变视觉缓存或训练标签。实验写入：

- `final_exp/relproto_alignment/ear_mechanism_20261021_v1/<variant>/`
- `output/ear_mechanism_20261021_v1/`
- 推理才会将各变体的 W bake 到 `weights/vitb_mlp_infonce_coco2014_ear_*.pth` 新文件中。

若有尚未完成的同名训练输出，脚本**拒绝覆盖/混合**，需要手动检查，或设置新的 `NIGHT_TAG`。已完整训练且有 `summary.json` 和 `W_epoch_010.pt` 时会跳过对应训练。

## 2. 消融实验矩阵

| variant | Ear 自身作为正类 | 其他 110 个非 Ear 类作为正类 | 结构 λ |
|---|---|---|---:|
| baseline_struct（使用已有 100ep 实验的 E0–E10） | 是 | 是 | 4 |
| full_nostruct | 是 | 是 | 0 |
| drop_ear_struct | 否 | 是 | 4 |
| drop_ear_nostruct | 否 | 是 | 0 |
| only_ear_struct | 是 | 否 | 4 |
| only_ear_nostruct | 是 | 否 | 0 |
| structure_only | 否 | 否 | 4 |

**`drop_ear` 并不从候选集合中移除 Ear**，也不从 116-way CE 的分类词表中移除 Ear。仅把“GT target id 为 Ear 的 CE 项”乘以零。`only_ear` 也只对正类项做互补操作。原始 K 分母、annotation 分母都保持不变；**不会把剩余损失重新归一化**。这是保证其他正类项系数不变的必要条件。只有 `structure_only` 关闭全部 CE 项，研究结构驱动是否足够。

`night_ear_train.py` 是用户上传原始源码的拷贝修改版；`baseline_reference.py` 保留原始文件以供检查差异。初始化、动态伪原型、正交更新、保存 Checkpoint 沿用原始实现。

## 3. 先做无 GPU 测试

在服务器仓库根目录（当前 Conda 环境）运行：

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg
PYTHONPATH="$PWD" python ear_night/test_loss_equivalence.py
```

应输出三条 PASS，第一条验证 **原始 `batch_forward` 与实验版 baseline 的 loss 和 W gradient 一致**，其余检查 CE 互补分解与固定 K 分母，以及 K 不同的情况下梯度路径仍有效。

建议先只做固定梯度探针 E0（加载现有 baseline 的真实缓存，不训练）：

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python ear_night/night_ear_gradient_probe.py \
  --run final_exp/relproto_alignment/pp116_softrelative_structure_reproduce_100ep \
  --epochs 0 \
  --output output/ear_probe_smoke \
  --device cuda
```

需成功输出 `gradient_components.csv` 和 `six_ear_margin_direction.csv`，并确保 `gradient_reconstruction_relative_error` 很小。该脚本默认使用 6 个历史代表 Crop 加上 10 个固定的非 Ear Crop 作为**抽样梯度探针**，不是整个训练集梯度普查。所有 Epoch 使用完全相同的一组 Crop，以排除样本组合变化。

## 4. 夜间一键运行（无交互）

```bash
cd /mnt/sda/master/code/lyx/Talk2DINO_official_bg

nohup env GPU=0 RUN_ALLCROP=0 \
  bash ear_night/run_ear_night.sh \
  > ear_night_launcher.log 2>&1 < /dev/null &
echo "PID=$!"
```

`RUN_ALLCROP=0` 是默认推荐的第一晚设置：6 种实验×10 Epoch，各实验对固定六个 Crop E0–E10 逐 Epoch 梯度+GT-Margin 审计、默认 E3/E10 做官方验证。运行器还会完成已有 baseline 的梯度探针，不重复训练 baseline。`RUN_ALLCROP=1` 则对六种新实验继续调用你服务器**已有**的 `audit_pp116_ear_patch_e0_e10.py` 全部 Ear Crop E0–E10 审计，会明显增加磁盘和运行时间；需要其依赖 `audit_pp116_gt_weighted_purity.py` 可用。强烈建议第一晚先用默认，第二晚只对关键变体做全量 Patch 审计。

实时查看：

```bash
tail -f ear_night_launcher.log
```

更详细的训练日志、每组评估 stdout、测试结果与 `provenance.txt` 都在 `output/ear_mechanism_20261021_v1/`。

**更紧凑的第一晚优先实验**（若机器较慢）：

```bash
GPU=0 TRAIN_VARIANTS="drop_ear_nostruct drop_ear_struct only_ear_nostruct full_nostruct" \
RUN_ALLCROP=0 bash ear_night/run_ear_night.sh
```

**增加每个模型的验证时点**（会明显增加评估时间）：

```bash
GPU=0 EVAL_EPOCHS="1 2 3 5 10" RUN_ALLCROP=0 \
  bash ear_night/run_ear_night.sh
```

**只执行已训练实验的梯度与评估，不重新训练**：

```bash
GPU=0 RUN_TRAIN=0 bash ear_night/run_ear_night.sh
```

`RUN_TRAIN=0` 只适合所有被请求变体都已存在完整 E10 Checkpoint 的场景；否则会安全报错。

## 5. 固定 GT Ear Margin 梯度归因的含义

在同一个固定 Crop 上，E0、E1……E10 的预测前景、DINO Patch 向量、候选 Part 集合都不变。对真实 Ear Patch（GT 仅审计）的 margin

\[
R_{e,p}(W)=\langle t_e W,x_p\rangle-\max_{q\in\mathrm{Presence}\setminus\{e\}}\langle t_q W,x_p\rangle
\]

求梯度 `grad_R`；对三类损失分别求 `grad_Lear`, `grad_Lother`, `grad_Lstruct`；在正交群的切空间投影

\[
P_W(G)=G-W\operatorname{sym}(W^T G)
\]

给出一阶下降方向预期 margin 增长：

\[
D_j=-\langle P_W(\nabla R), P_W(\nabla \mathcal L_j)\rangle
\]

对于 Structure 已乘 lambda。若 D>0，说明**以该梯度进行微小 SGD 更新的局部方向**倾向提升 Ear GT Margin；D<0 则相反。`gradient_reconstruction_relative_error` 检验 `grad_full == grad_ear + grad_other + lambda*grad_struct`，否则直接失败。这个值**不是** Adam/SVD 更新的精确贡献，也不能替代受控消融。

额外拆分 `other`：对所有非 Ear 正目标，Ear 的六个 Logit 在 softmax 中仍为负类，原始 CE 的 `dCE/d logit_ear = p(ear)`。将 p(ear) **detach** 后构建仅供梯度分析的 surrogate，得到 `other_as_ear_negative`；然后令 `other_remaining_logits = other - other_as_ear_negative`。这比只测三条梯度路径更细，能够检验非 Ear 样本是否主要通过把 Ear 当负类而间接影响其方向。

说明：如果 GT Margin 已接近饱和，当前瞬时 D 可以变小甚至为负，并不代表此前没有贡献；还需对照 E0→E10 的实际 Margin、覆盖率和 val IoU。

## 6. 自动产物与检查

- `<run>/training_history.csv`：每 Epoch loss、梯度范数、结构有效占比等，新增 `ear_positive_occurrences` 与 Ear/Other NCE 观测量。
- `<run>/W_epoch_001.pt` 至 `W_epoch_010.pt`：不覆盖基线的 W、Adam 状态、方法元数据。
- `output/<tag>/gradient/<arm>/gradient_components.csv`：原始梯度和正交切线梯度范数、不同来源的余弦、分解残差。
- `output/<tag>/gradient/<arm>/six_ear_margin_direction.csv`：六个固定 GT Ear Crop 每个 Epoch 的 Margin/覆盖率与三来源一阶方向；另有 Other-Ear-negative 梯度拆分。
- `output/<tag>/val/<arm>/e003/`、`e010/`：真实验证集评估。保留 `evaluation_stdout.log` 和评估器本身的日志；仅当评估命令返回 0 时创建 `EVALUATION_DONE`。
- `output/<tag>/summary/`：所有臂的六个固定 Crop Margin 曲线、E10 表、原始 eval 文本筛选结果。

**注意：`six_ear_margin_direction.csv` 反映六个固定 *训练* Crop，而不是全 PP116 的 mIoU。真正的部件 IoU 要以 `main.py --eval` 的验证集结果为准。** 自动汇总不会猜测日志中不认识的表格格式，`eval_lines_unparsed.csv` 的行必须在确认指标列定义后再解释为 IoU。

## 7. 如何判断原因

1. **DropEar-NoStruct 的 Ear E10 GT Margin 和 IoU 仍大幅恢复**：非 Ear 正类的 NCE（含 Ear 负类路径）足以推动恢复；Ear 自身正类损失不是必要条件。
2. **OnlyEar-NoStruct 恢复明显**：Ear 自身存在足够的训练信号，但不能声称 E0 的伪标签普遍正确；其中大量可能是含噪信号。
3. **DropEar-Struct 明显优于 DropEar-NoStruct**：在去掉 Ear 正类监督时，Structure 有正面作用（也要看 λ0 vs λ4 的对照和不同 Ear 的差异）。
4. **Structure-Only 仍有恢复**：没有任何 InfoNCE 时，结构排序损失可能单独推动原型竞争重组（它仍依赖相对竞争产生的视觉原型和冻结文本关系）。
5. **只有 Baseline 显著成功**：说明各来源之间的相互作用可能非常重要，不能简单给每类梯度划分固定“贡献百分比”。
6. **Person Ear 与其余 Ear 分化**：应该结合它更慢的响应变迁考察是否受不同梯度方向影响。研究对象是所有六类，不要事后只选成功例。

注意：单个因素从 Baseline 中移除，训练会使所有后续伪目标同步变化，因此属于 **端到端效应**；`OnlyEar`/`Structure-Only` 也不是“完全相同训练轨迹上的梯度因果分解”。为强结论至少需要更多 seed（建议 123/456/789）和训练阶段的直接梯度方向分析，初步夜间实验先 seed 123。

## 8. 已验证 / 尚未验证

本包已在构建环境中完成：Python 语法检查、Shell 语法检查、**synthetic CPU 测试对照用户原始 `batch_forward`：baseline loss 与 grad 一致**、Ear/Other 互补 loss/grad 分解以及不同 K 的训练路径测试。

**没有**在你的服务器上读取实际完整 cache、执行 GPU 训练、bake 或验证集评估。这些步骤必须由运行器在服务器完成，不能预先声称成功。若真实数据目录、仓库模块接口或 GPU 环境与上传版本不同，运行器会报错并保留日志，不会覆盖原模型。
