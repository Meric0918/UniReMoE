# Stage 8 的 PT 性能下降：证据、机制与改进实验

日期：2026-09-24。分析对象为 Stage 1 `best_ptcr_20260920_151134_400917` 与 Stage 8 `full/stage8_best_20260921_111731_349210`。本报告使用 checkpoint 内嵌配置、当前训练实现、对应训练日志与实际 train/validation 数据；没有重新训练或用 test 标签调参。

结论：三元建模具有生物学动机，但当前 Stage 8 的训练目标、表示更新与模型选择机制，没有保证原始 PT 任务受益。证据最充分的是：独立 PT 验证缺失、二元回放暴露量显著下降、PHT 数据来源与 peptide 频率改变、共享表示接受三元梯度，以及 Stage 1 正则化没有延续。损失梯度冲突和参数遗忘具有明确机制，但各因素贡献大小还需要受控消融；不能把观察到的全部下降归因于“约束太强”或“模型太复杂”。

## 1. 下降包含排序能力损失，不只是阈值问题

两份报告的 PT 样本数及正负样本数一致，总计 263,610 条，数据目录均为 `dataset/PT/new`。Stage 1 使用严格 `score > 0.5`；Stage 8 使用的边界比较需要统一后再做严格分类指标复现。这个差别只影响恰好位于阈值的样本，不能解释 AUROC/AUPR 的下降。

| PT 等权宏平均 | Stage 1 | Stage 8 | Stage 8 − Stage 1 |
|---|---:|---:|---:|
| AUROC | 0.848279 | 0.779403 | −0.068876 |
| AUPR | 0.821128 | 0.715759 | −0.105369 |
| Accuracy | 0.795419 | 0.728678 | −0.066741 |
| MCC | 0.563803 | 0.418071 | −0.145732 |
| F1 | 0.730244 | 0.591635 | −0.138609 |
| 五指标平均 | 0.751775 | 0.646709 | −0.105066 |

| Split | AUROC 变化 | Recall：Stage 1 → Stage 8 | Specificity：Stage 1 → Stage 8 |
|---|---:|---|---|
| unseen_pair | −0.052461 | 0.734661 → 0.478638 | 0.858854 → 0.917550 |
| unseen_peptide | −0.062127 | 0.544763 → 0.376134 | 0.869016 → 0.894587 |
| unseen_tcr | −0.092040 | 0.835716 → 0.620159 | 0.814857 → 0.843677 |

Stage 8 在阈值 0.5 下更少预测正类；但 AUROC、AUPR 也下降，表明重新校准阈值只能改善其中一部分。

## 2. 架构：新增的生物学上下文没有直接进入 PT 推理路径

上一轮回答把 Stage 1 描述为“专门的 pTCR 模型”，需要精确化：Stage 1 本身是共享 peptide Adapter 的 PH/PT 联合模型；报告测试的是从该模型导出的、按 PT 验证表现选择的任务分支。

实际前向见 [stage8_model.py](/home/mjp/Project/PLM/UnifyImmun/source/plm_unified/stage8_model.py:39)。

| 路径 | 实际计算 | 与 PT 的关系 |
|---|---|---|
| PT | 原始 peptide 表示与 TCR 表示 → PT cross-attention → pooling → PT classifier | 推理时不使用 HLA，也不使用 PHT head |
| PH | peptide 与 HLA → PH cross-attention → pooling → PH classifier | 与 PT 共享 peptide 编码器和 Adapter |
| PHT | PH cross-attention 产生的 peptide 序列表示再与 TCR 做 cross-attention | 与 PT 共享 peptide、TCR 编码器及 Adapter；不经过 PT cross-attention/classifier |

因此，PHT 可能学到了更好的 HLA 条件识别，但 PT 分支只能通过共享参数间接受益。Stage8B 联合步骤又会解冻这些共享参数，使其朝 PHT 分布移动。PT 头与输入表示之间已有的配合会改变，仅仅保留 PT classifier 的名字和结构不能保持能力。

联合监督权重为 `L_PHT + 0.3 L_PH + 0.2 L_PT`。PHT 在共享表示上的损失系数是 PT 辅助项的五倍；这不是实测梯度范数比例，不能直接解释为“梯度大五倍”。需要在真实 batch 上分别记录 `||g_PHT||`、`||g_PT||` 和 cosine similarity 才能定量判断。

尤其是 `010` 样本：它对 PT 是正例，对 PHT 是负例。两个任务的标签可以同时正确，但若没有任务专属表示，二者可能争用同一组 peptide/TCR 特征。更合理的架构方向是共享基础知识、为 HLA 条件识别保留专属 Adapter/残差分支。

## 3. 生物学蕴含不能直接等同于不同输入条件下的概率上界

TCR 识别的是 peptide–MHC 复合物。把 HLA 纳入模型有明确依据，但输入更完整并不自动保证在缺失 HLA 的 PT 测试任务上更好。[NetTCR-2.0 原始研究](https://www.nature.com/articles/s42003-021-02610-3)也展示了配对 TCR 链信息及数据质量对可预测性的影响。

如果把同一识别事件的 PT 输出理解为缺失 HLA 时的边缘概率，则有：

\[
P(Y=1\mid p,t)=\sum_h P(Y=1\mid p,h,t)\,P(h\mid p,t).
\]

特定 HLA 下的条件概率完全可能高于这个平均值。例如适配 HLA 下识别概率为 0.9，该 HLA 的权重为 0.1，其他 HLA 下均为 0，则边缘概率为 0.09。此时不满足 `PHT <= PT`，但并不违背生物学。

这只是解释条件概率的例子，并不证明当前 PT 标签严格等于该边缘事件。首先需要明确 PT 标签表示实验条件下识别、某种 HLA 下存在识别，还是其他定义。只有在标签事件、样本空间和校准条件一致时，事件蕴含才支持对应的概率上界。不同来源的二元和三元分类器分数不天然满足这些前提。

现有四类仅包含 `000/010/100/111`。如果 PH 和 PT 表示分别具有结合能力，而 PHT 表示在同一个 HLA 背景下形成有效识别，`110` 也可能是有意义的困难负例。是否添加必须依赖实验或明确的标签生成依据，不能把它随机生成后当作真阴性。

实际约束见 [stage8_loss.py](/home/mjp/Project/PLM/UnifyImmun/source/plm_unified/stage8_loss.py:96)：`111` 上 PT 正约束使用 `relu(pht.detach()-pt-margin)^2`，方向是提高 PT；`000/100` 上使用 `pt.detach()`，直接降低的是 PHT。因此，“负三元样本通过 PT 约束直接把 PT 压低”并不符合当前代码。PT 召回下降更可能涉及共享梯度、数据分布和训练/选择策略。仅看 full 的结果，不能判断去掉 PT implication 一定更好。

## 4. 数据：类间平衡掩盖了 peptide 分布偏移

可复现统计见 [audit.json](audit.json)，审计脚本为 [analyze_stage8_pt_retention.py](/home/mjp/Project/PLM/UnifyImmun/source/analyze_stage8_pt_retention.py)。指纹与原训练审计一致：`aaeb3ce53d879a54915da4e50c9e414b63b98e6d7d5915263c63a89b0d144920`。

| 数据 | 样本数 | 唯一 peptide | 唯一 TCR |
|---|---:|---:|---:|
| 原始 PT train | 664,952 | 1,427 | 154,259 |
| PHT train 投影到 PT | 72,190 | 1,294 | 56,859 |

PHT train 的四类计数为 `000=18,048、010=18,048、100=18,048、111=18,046`。强制等量采样只使每轮增加两个抽样位置，不存在大量少数类循环重复导致的有效数据膨胀。早期对采样倍率的怀疑已被实际数据排除。

但类内分布变化很大：`111` 中 `GILGFVFTL` 5,276 条、`NLVPMVATV` 3,942 条，合计占 **51.08%**。这两个 peptide 在全部 PHT train 中占 **17.04%**，在全部 PT train 中只占 **1.98%**。四类各占 25% 并不能平衡 peptide，模型仍可能依赖高频 peptide 的捷径。

在 72,190 条 PHT 训练记录中，57,073 条（79.06%）的 PT pair 已存在于 PT train。它们为原 pair 增加 HLA 条件信息，但从 PT 角度并不等于同等数量的新独立样本。重复投影也会改变监督权重。

已排除/尚未证明的事项：

- PT train 内、PHT train 投影到 PT 后的冲突 pair 数均为 0；重叠记录与 PT train 的标签冲突为 0。不能把“存在大量相互矛盾的 PT 标签”写成此次下降的事实。
- 三个 PT validation split 与 PHT train 的 exact PT pair 交集均为 0；unseen_peptide 的 peptide 交集为 0，unseen_tcr 的 TCR 交集为 0。没有发现这两类显式泄漏；没有进一步证明近同源序列、来源批次等完全独立。
- 标签一致不等于实验真值可靠。当前文件的 `source_file` 不能单独证明每个负例都经过实验验证；负样本来源仍应追溯。

外部证据也支持优先关注数据来源、负采样和泄漏控制，而不是只比较网络容量：[Nature Methods 的 TCR–epitope 方法评估](https://www.nature.com/articles/s41592-025-02910-0)。这类研究支持风险机制，并不能替代对本项目的因果消融。

## 5. 训练：PT 更新次数相近，但看到的数据少了约 16 倍

以下 Stage 1 参数以 checkpoint 内保存值为准。当前 `plm_stage1_ph_pt.yaml` 已与当时运行不同，例如当前 smoothing 为 0.015，而实际 PT 为 0.03。

| 项目 | Stage 1 实际运行 | Stage 8 实际运行 |
|---|---|---|
| 编码器 | 冻结 PLM，读取 embedding cache | Stage8B 解冻 peptide/HLA/TCR PLM 顶部各两层 |
| 有效 batch | 1,024 | microbatch 4 × accumulation 16 = 64；配置里的 256 不用于 online batch |
| 每轮 PT 原始数据暴露 | 285 × 1,024 = 291,840 条 | floor(72,190 × 0.25) = 18,047 条 |
| 每轮 PT 数据集覆盖 | 43.89% | 2.71% |
| 每轮 PT 更新 | 285 | 282 |
| 每轮 PHT 更新 | 无 | 1,128 |
| 顺序 | PH → PT | PH replay → PT replay → joint |
| PT label smoothing | 0.03 | replay 普通 CE，无 smoothing |
| 分组权重 | peptide × label 的 inverse-sqrt，clip 0.25–4 | replay 普通 CE，无分组权重 |
| 对抗正则 | FGM epsilon 0.5、权重 0.25，有 ramp | 无 FGM |
| Adapter LR | peptide 5e-5、TCR 1e-4 | 三者均 2.5e-5 |
| PT attention / classifier LR | 7.5e-5 / 7.5e-5 | 均 5e-5 |
| PT pooling LR | 1e-4 | 5e-5 |
| PLM LR | 0 | 顶层 5e-6，次顶层 2.5e-6 |
| 权重衰减 | 0.01，bias/norm 不衰减 | 0.01，当前 optimizer 没有同样的排除规则 |
| 验证与停止 | 独立 PT 三 split、校准 guard、task 级停止 | PHT 来源的三 head 加权分数；patience 20 |

Stage 1 本次启用了 task-step balance，所以不能把它误写为每轮遍历 664,952 条 PT。其 checkpoint 中的 `task_step_limits` 明确为 PH/PT 各 285。

Stage 8 的 25% 是“相对于 PHT train 的比例”，不是“覆盖 PT train 的 25%”。相对于三种训练输入的总行数，PT replay 占约 16.67%。35 轮时，每条 PHT 记录已经多次出现，而原始 PT 随机回放累计暴露 631,645 条，约为完整 PT 数据量的 0.95 倍；不同轮有重复，不能解释成每条原始 PT 样本都被见过。

长达 1,128 次更新的 joint pass 排在每轮末尾，因此保存时参数刚适配过 PHT，而不是刚完成 PT 回放。它是遗忘的高风险设置；要证明具体遗忘量，应在 PH 后、PT 后、joint 后使用同一组固定 PT validation 样本分别评估。

另外，online `_train_joint` 没有调用 `BalancedFourTypeBatchSampler.set_epoch()`，而 cached trainer 调用了。当前 online 路径会每轮复用相同的三元顺序，减少跨轮随机化。这是可确认的实现问题，但没有证据说明它单独造成了主要降幅。

## 6. 模型选择：PT 权重 0.4 不等于监控原始 PT 任务

Stage 8 的选择分数为：

\[
0.4 S_{PHT(PHT\ val)}+0.2 S_{PH(PHT\ val)}+0.4 S_{PT(PHT\ val)}.
\]

三项都来自 `PHT/classes/*/val.csv`，没有来自 `PT/benchmarks/val_unseen_*` 的分数。代码里虽然出现 `joint_degradation_tolerance=0.002`，它在 retained cached trainer 中被使用，当前 online checkpoint 选择并没有提供独立 PT 保持门槛。

PHT validation 的类型分布是 `000=15,491、010=533、100=4、111=16,027`，`000+111` 占 **98.32%**；train 则是四类各约 25%。因此，仅仅改善这组验证分布上的 PT head，不能证明改善了原始 PT 泛化，更不能充分评估 `100/010` 的行为。

对应日志中，full 第 35 轮：joint PT train AUROC **0.964248**，PHT validation 上 PT AUROC **0.834899**，而 PT/new test 宏平均 AUROC 为 **0.779403**。这三者来自不同分布，不能用前两者替代后者。

PHT 的 train loss 从第 1 轮 0.125032 降至第 35 轮 0.021654，而 val loss 从 0.167534 升至 0.626233，val AUROC 几乎未变（0.991957 → 0.991850）。这支持“训练拟合增强，但概率校准恶化”的判断。best checkpoint 的 PHT validation 阈值约 5.96e-6，也应触发校准检查；它与 PT 阈值无关，不应套用到 PT。

## 7. 初始化差异：需要对照，但不能当作全部原因

Stage 8 来自同一轮训练的 `best_joint` **round 9**，对比报告来自 `best_ptcr` **round 10**。两者的 PT 模块并非逐参数相同：例如 peptide Adapter 相对 L2 差异约 2.14%，TCR Adapter 约 3.15%。

二者原 PT validation selection score 为 0.616835 与 0.618615，仅差约 0.00178；这不等于 test 上也仅差同样数值。需要补测 exact `best_joint`，才能从总下降中分离初始化因素。

已用 tensor equality 核查 Stage8A best warmup 的所有二元模块，与 exact `best_joint` 完全相同；其被选中的是 warmup epoch 1。Stage8A 自身不改变这些模块，后续 Stage8B 才引入共享模型的持续更新。这一结论不依赖结构图推测。

## 8. 改进顺序及预期解决的问题

| 优先级 | 改进 | 针对的问题 | 验证方法 |
|---|---|---|---|
| P0 | 模型选择加入独立 PT 三 split，设置宏平均和单 split 下限 | PHT 分布改善掩盖 PT 退化 | 相对固定 Stage 1 教师的 AUROC/AUPR/MCC guard |
| P0 | 初始实验冻结所有 PLM，joint 仅训练 PHT 新模块 | 全链路表示漂移 | 检查 joint 前后二元参数与 PT 输出相同 |
| P0 | PT replay 增至 PHT 数量的 1 倍，恢复频率权重和 smoothing，加入教师 KL | 原始 PT 暴露不足、正则丢失 | teacher/student 同一 PT validation 对比 |
| P0 | PH → joint → PT，并修复 sampler epoch | 长三元 pass 覆盖 PT 适配 | 固定监控集的逐 pass 曲线 |
| P1 | 交错小段回放，按任务有效样本数而非笼统 epoch 预算 | 单任务连续更新过长 | 相同总样本和 optimizer-step 预算比较 |
| P1 | 给 PHT 单独的 peptide/TCR Adapter 或零初始化残差路径 | 增加三元能力同时减少对 PT 的扰动 | 梯度范数/cosine；PT 和 PHT 的同时改善 |
| P1 | 在 peptide/来源/负采样方式上分层，审计实验真负例 | 类平衡仍存在捷径与标签噪声 | peptide group macro、source-stratified 指标 |
| P1 | 在保留条件下逐步解冻一个 PLM 顶层，或使用低秩 Adapter | 全冻结限制适应能力 | 验证门槛通过后再扩展可训练参数 |
| P2 | 仅对可信事件添加一致性/对比项，明确 PT 的 HLA 边缘语义 | 概率上界与实际任务不完全匹配 | full / no-PT-imp / conditional-head 消融 |
| P2 | 多任务梯度冲突实测后应用 PCGrad 等方法 | 不同任务争用共享参数 | 在同预算下比较梯度冲突率和泛化 |

教师保持可以采用：

\[
L_{PT}=L_{CE,weighted,smooth}+0.5T^2 KL\bigl(p_{teacher}^{(T)}\Vert p_{student}^{(T)}\bigr),\quad T=2.
\]

教师使用 exact Stage 1 `best_ptcr` 并全程冻结，硬标签仍参与训练。学生如果使用被 Stage8B 改过的编码器，不能把其 hidden states 直接送给 Stage 1 教师；两者特征坐标不同。首个保护实验冻结原始 PLM，因此可以共享原始编码结果。知识蒸馏方法依据见 [Hinton 等](https://arxiv.org/abs/1503.02531)；梯度投影方法依据见 [PCGrad](https://arxiv.org/abs/2001.06782)。具体超参数在本项目上仍须验证。

更长期的结构可显式拆开：PT 使用独立的二元 expert；PHT 使用 `pHLA` 条件表示和专属 TCR 交互残差。若进一步采用“呈递概率 × 呈递成立后的识别概率”，必须先确认标签确实对应这两个事件，不应直接把三个独立 sigmoid 分数相乘。

若短期需要保持已知 PT 性能，可在应用层让缺失 HLA 的 PT 请求调用 exact Stage 1 PT expert，完整三元请求调用 Stage 8 PHT expert。这样复用的是已有能力，不构成“单一联合模型性能已提高”的新实验结果。

## 9. 已实现的保护性实验与验证状态

新增文件：

- [实验配置](/home/mjp/Project/PLM/UnifyImmun/configs/plm_stage8_pt_retention_20260924.yaml)
- [模型与训练策略](/home/mjp/Project/PLM/UnifyImmun/source/plm_unified/stage8_retention.py)
- [训练入口](/home/mjp/Project/PLM/UnifyImmun/source/train_plm_stage8_pt_retention.py)
- [最终测试入口](/home/mjp/Project/PLM/UnifyImmun/source/evaluate_plm_stage8_pt_retention.py)
- [CPU 回归测试](/home/mjp/Project/PLM/UnifyImmun/tests/test_stage8_pt_retention.py)

这是独立的实验入口，未修改已有 Stage 8 训练器、checkpoint 和原始数据。保留相同的 sequential 三元前向结构，从已有 Stage8A warmup 启动；PLM 全冻结，PH replay 不更新共享 peptide Adapter，joint 仅更新三元新模块，PT replay 更新 PT 路径并使用冻结教师。PT replay 每轮为 72,190 条，是原运行的四倍，仍只覆盖 PT train 的 10.86%，并非完整遍历。

训练计划为 10 轮。PT CE 使用 peptide inverse-sqrt 权重、全局样本均值归一化和 0.03 smoothing，教师 KL 权重 0.5、温度 2。该分组权重是实验实现，尚未逐式复刻 Stage 1 的 peptide×label 分组/批次归一化；FGM 也尚未加入本实验。阶段一目的是验证保持机制，而不是同时复制所有训练细节。

选择分数使用 `0.6 × independent PT macro + 0.3 × PHT + 0.1 × PHT来源PH`，并要求相对教师：PT AUROC/AUPR/MCC 宏平均下降不超过 0.01，每个 split 下降不超过 0.02；PHT 五指标平均相对初始 warmup 下降不超过 0.02。未通过门槛的模型不会写成 `best.pt`。这些是预先指定的实验容忍度，不代表已经保证优于 Stage 1。

在 joint 冻结二元分支时，原损失里的正向蕴含项不再更新二元参数，只作为诊断常量；负类约束仍可更新 PHT。这是有意设置的保守保持消融，要研究完全可训练的一致性模型，需要后续的任务专属 Adapter/残差实验。

已完成：完整真实数据/lineage dry-run；8 项 CPU 单元测试通过，覆盖 joint 参数隔离、PH 对 PT 的隔离、完整一轮 PT 参数更新、optimizer 包含全部下游参数、教师停止梯度、sample 权重、单 split / 非有限数值验证门槛，以及固定编码器/新下游权重的正确重建和来源校验。CPU 测试使用真实下游网络和小型替代编码器，未运行真实大型 PLM 的训练前向。

当前 GPU 已有三个运行中的任务，因此本次没有启动完整十轮训练；没有生成新的性能提升结论。实验输出是带来源信息的 frozen-PLM 下游 checkpoint，类型为 `stage8_pt_retention_frozen_plm`，并非原有 Stage-8 V2。`initial.pt/last.pt/best.pt` 保存推理权重与计划/验证指标；本入口尚不提供中断恢复。新增最终评测入口会校验 warmup SHA256，重建原始编码器和新下游权重，再复用原 Stage 8 的 unique-sequence 编码及 PH/PT/PHT 行级评测，实现固定 0.5 阈值的 JSON/Markdown 输出。仅接受通过 guard 且按 validation 选中的 `best` 模型。

检查入口：

```bash
OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage8_pt_retention.py --dry-run
```

启动独立训练：

```bash
OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage8_pt_retention.py \
  --config configs/plm_stage8_pt_retention_20260924.yaml --device cuda
```

训练完成、配置锁定后进行最终评测（替换实际运行目录）：

```bash
OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage8_pt_retention.py \
  --checkpoint trained_model/PLM_Stage8_PTRetention/RUN_TIMESTAMP/best.pt \
  --output-prefix tests/stage8/pt_retention_final
```

## 10. 用最少实验分离原因

第一组控制使用 exact Stage 1 joint、其 Stage8A warmup、原 Stage8B baseline 和 full。先在独立 PT validation 上评测，分离初始化、解冻/三元训练、约束三类影响；不要在 test 上选最优条件。

第二组在相同源模型、随机种子及计算预算下依次增加：独立 PT checkpoint guard；冻结 PLM/保护 joint 梯度；增大/交错 replay；教师保持及正则恢复。新入口组合了多个干预，只能评估组合效果，不能单凭一次运行归因到其中一个。

第三组在保持 PT 的条件下探索三元正迁移：PHT 专属 Adapter/残差、可信困难负例和同源分组划分、测量并缓解共享梯度冲突。这里才检验“生物学结构能否提供超越二元专家的泛化收益”。

开发阶段固定 train/validation。锁定模型和阈值后进行最终 PT/new 和 PHT benchmark 评估；报告每个 split 的 AUROC/AUPR/MCC/Recall、peptide 宏平均以及至少多随机种子重复。置信区间应考虑同 peptide/TCR 样本相关性，不能只做独立行级重采样。由于现有 test 结果已经被观察，严格的确认性结论最好再用未参与开发的新外部测试集。
