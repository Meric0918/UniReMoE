# Stage1 训练改进方案

生成日期：2026-09-14

## 1. 改进目标

本文基于以下两份分析提出 Stage1 的下一版训练方案：

- [模型维度、训练时长与测试集表现分析](stage1_dimension_training_test_analysis.md)
- [pTCR validation loss 持续升高分析](stage1_ptcr_validation_loss_rising_analysis.md)

改进目标不是单纯降低训练 loss，而是同时满足：

1. 保留 256 维模型对 pTCR unseen-peptide 召回和 pHLA 长训练的收益。
2. 减少 pTCR OOD validation loss 持续升高和高置信度假阴性。
3. 减少高频 peptide/HLA 对训练梯度和逐行验证指标的支配。
4. 缓解 pHLA 与 pTCR 对共享 `peptide_adapter` 的任务冲突。
5. 使不同最大训练轮数之间的学习率轨迹可以公平比较。
6. 用判别能力、校准能力和最差分布表现共同选择 checkpoint。
7. 对每轮刷新后的 task-best 模型进行固定样本 UMAP 可视化，观察表示空间、错误
   类型和 OOD 分布随训练轮次的变化。

## 2. 当前训练设置的主要问题

| 模块 | 当前设置 | 已观察到的问题 |
|---|---|---|
| 基础损失 | 普通 Cross-Entropy | 少量高置信度错误可导致 OOD validation loss 持续升高 |
| FGM | `epsilon=1.0`，clean 与 adversarial loss 等权相加 | 扰动及 adversarial 梯度可能偏强，且没有显式权重可调 |
| 学习率 | adapter/pooling/classifier `3e-4`，cross-attention `2e-4` | classifier 和共享 adapter 更新较激进，可能加速 logit 变大与训练实体专化 |
| optimizer 分组 | 按模块类型合并两个任务的参数 | 冻结的任务分支仍随全局 scheduler 改变学习率；无法设置任务特异 LR |
| scheduler | 总步数依赖 `max_rounds` | 10 轮和 100 轮实验改变的不只是训练时长，还改变了整个 LR 轨迹 |
| 数据采样 | 全局 shuffle、逐行等权 | pTCR 高频 peptide、pHLA 高频 HLA 对梯度贡献过大 |
| 正则化 | dropout `0.1/0.1/0.2`，weight decay `0.01` | 对 256 维模型的 OOD 过拟合约束不足；所有参数统一 weight decay |
| 验证选择 | 五个判别/阈值指标的宏平均，不含 NLL | loss 和校准恶化时仍可能更新最佳 checkpoint |
| 阈值 | 固定 `0.5` | 未校准模型的 Recall/F1 对阈值敏感，不同任务可能需要不同阈值 |
| 任务停止 | 每任务可单独停止，但共享 adapter 仍可被另一任务更新 | 已停止任务的联合模型状态会继续漂移，且停止后不再复评该任务 |

## 3. 总体改进优先级

### P0：应首先修改

1. 固定 scheduler 预算，使其不再由 `max_rounds` 隐式决定。
2. 拆分每个模块/任务的 optimizer 参数组和更新计数。
3. validation 增加 NLL、Brier、ECE、实体宏平均和 worst-split 指标。
4. 每轮保存可追踪的逐样本 validation 预测或至少实体级聚合。
5. 已停止任务仍需在共享 adapter 更新后复评，并设置 retention gate。
6. 保留 pHLA/pTCR 各自 checkpoint；联合 checkpoint 必须使用同一当前模型状态的两个任务分数。

### P1：建议作为下一轮主实验

1. 使用轻度 label smoothing。
2. 降低共享 peptide adapter、classifier 和 pooling 的学习率。
3. 将 FGM 权重和 epsilon 调低并独立配置。
4. 对 pTCR 采用 peptide-label 分组权重，对 pHLA 采用 HLA-label 分组权重。
5. 使用任务特异 early-stopping patience。

### P2：在 P0/P1 仍不能解决时再做

1. shared-base + task-specific peptide adapter/delta。
2. PCGrad、GradNorm 等多任务梯度冲突处理。
3. Brier 正则或显式 logit penalty。
4. 更复杂的 domain generalization 或 worst-group 优化。

## 4. 损失函数改进

### 4.1 不建议直接放弃 Cross-Entropy

当前问题不是 Cross-Entropy 实现错误，而是：

- 训练样本按行等权，高频实体主导梯度；
- 模型在 OOD 样本上变得过度自信；
- FGM adversarial loss 与 clean loss 等权相加；
- checkpoint 选择完全不约束校准。

因此下一版应保留 CE 的概率学习能力，在其上增加轻度校准、分组和 adversarial 权重控制。

### 4.2 推荐的第一版损失

对任务 `t`，建议使用：

```text
L_t = (L_group_weighted_label_smoothed_CE
       + lambda_adv(t) * L_adversarial_CE)
      / (1 + lambda_adv(t))
```

推荐起始值：

| 参数 | pHLA | pTCR | 当前值/状态 |
|---|---:|---:|---|
| Label smoothing | 0.02 | 0.03–0.05 | 0 |
| FGM epsilon | 0.5 | 0.25–0.5 | 1.0 |
| `lambda_adv` | 0.25–0.5 | 0.25 | 隐式为 1.0 |

原因：

- 轻度 label smoothing 可以抑制极端 logit，降低高置信度错误对 NLL 的破坏。
- pTCR 已出现明显概率质量恶化，建议使用略强的 smoothing 和较弱的 FGM。
- FGM 仍可保留局部鲁棒性，但不应默认与 clean loss 等权。
- 除以 `(1 + lambda_adv)` 可保持不同 adversarial 权重下的整体梯度尺度大致可比，避免把正则强度变化与有效学习率变化混在一起。

建议在 Stage1B 的前 1–2 个 cycle 中，将 `lambda_adv` 从 0 线性增加到目标值。当前 warmup 不使用 FGM，而 Stage1B 直接累加 clean 与 adversarial 两次反向梯度；即使有 gradient clipping，这种突变也可能造成共享表示和 logit 的早期漂移。

不建议一开始就使用较大的 `label_smoothing=0.1`，因为它可能明显压低正确阳性的置信度和 Recall。

### 4.3 实体分组权重

全局标签已经 1:1 平衡，不需要普通 class weight。应处理的是实体频次不平衡：

- pTCR train：1,427 个 peptide，单 peptide 样本数中位数 150、最大 19,928。
- pHLA train：99 个 HLA，单 HLA 样本数中位数 1,527、最大 22,971。

建议样本权重：

```text
pTCR: w_i ∝ 1 / sqrt(count(peptide_i, label_i))
pHLA: w_i ∝ 1 / sqrt(count(HLA_i, label_i))
```

随后将权重归一化为均值 1，并裁剪至 `[0.25, 4.0]`，避免极低频组获得过大梯度。

首轮实验应在“分组加权 loss”和“分组 sampler”中二选一，不要同时启用，否则会重复校正频率。项目中已有 `PeptideAwareBalancedSampler` 的平方根频率设计，可以复用其统计逻辑。

### 4.4 第二阶段可测试 Brier 正则

如果 label smoothing 后 NLL/ECE 仍持续恶化，可测试：

```text
L_t = (L_CE + lambda_adv * L_adv) / (1 + lambda_adv)
    + lambda_brier * L_Brier
lambda_brier ∈ {0.02, 0.05, 0.10}
```

Brier loss 是有界的概率误差，可对校准提供直接约束。它应作为第二阶段消融，而不是与所有改动一次性加入。

### 4.5 暂不推荐 Focal Loss

Focal loss 会进一步强调当前模型判错的困难样本。如果这些样本包含 OOD、标签噪声或极端高频实体，可能放大不稳定性；同时 focal loss 通常不以概率校准为优势。当前首要问题正是 OOD 概率质量，因此不建议把 focal loss 作为第一选择。

## 5. 学习率与 scheduler 改进

### 5.1 推荐起始学习率

当前所有 adapter 共用 `3e-4`，分类头也是 `3e-4`。建议拆分模块后使用：

| 参数组 | 推荐起始 LR | 当前 LR | 说明 |
|---|---:|---:|---|
| shared peptide adapter | `5e-5` | `3e-4` | 最需要稳定，避免两任务反复推动共享表示 |
| HLA adapter | `1e-4` | `3e-4` | pHLA 长训练仍需学习，但不必与共享层同速 |
| TCR adapter | `1e-4` | `3e-4` | 降低训练 TCR/peptide 组合专化速度 |
| pHLA cross-attention | `1e-4` | `2e-4` | 中等更新速度 |
| pTCR cross-attention | `7.5e-5–1e-4` | `2e-4` | pTCR OOD loss 更敏感 |
| pooling | `1e-4` | `3e-4` | 减少表示尺度快速变化 |
| classifier | `5e-5–1e-4` | `3e-4` | 直接控制 logit 变大和过度置信 |

这是一组保守起点，不应直接认定为最优值。建议优先比较统一缩放系数：`1.0、0.5、0.25`，再细调个别模块。

### 5.2 scheduler 必须与最大训练轮数解耦

当前：

```text
total_steps = steps_per_pair * (warmup_epochs_per_task + max_rounds)
```

因此把 `max_rounds` 从 10 改到 100，也把 cosine scheduler 拉长了 10 倍左右。训练时长消融被 LR 轨迹混杂。

建议新增独立配置：

```yaml
training:
  max_rounds: 50
  scheduler_total_rounds: 40
  scheduler_warmup_rounds: 1
  scheduler_min_lr_ratio: 0.05
```

所有时长消融固定 `scheduler_total_rounds`。达到调度终点后保持最小 LR，不因修改 `max_rounds` 重算历史轨迹。

### 5.3 使用模块更新计数，而不是全局 step 衰减所有参数

当前同一个 scheduler 在 pHLA 和 pTCR 的每个 batch 后都更新所有参数组的 LR，即使某任务分支被冻结。建议：

1. 每个实际可训练模块维护自己的 update count。
2. frozen 参数组不推进其 scheduler。
3. shared peptide adapter 在两个任务更新时均推进共享计数。
4. pHLA/pTCR 分支只在本任务 batch 中推进。

可通过“shared optimizer + 两个 task branch optimizer”，或自定义 task-aware scheduler 实现。不要为同一个共享参数建立两个互相独立的 Adam 状态。

还应处理任务提前停止后的调度长度：当前只要一个任务停止，每轮实际 optimizer step 数就减少，scheduler 可能无法在预计位置到达最低学习率。固定 step budget 与模块更新计数可以同时消除这一偏差。

### 5.4 任务更新比例

当前每轮各遍历一个完整 epoch，pTCR 约 665k 行，pHLA 约 291k 行，共享 adapter 每轮接受的 pTCR 更新明显更多。

建议将“round”改为固定 optimizer step 数，而不是每任务完整 epoch：

```text
每个 cycle：N 个 pHLA step + N 个 pTCR step
```

首轮使用 1:1 的共享更新次数，再消融 `pHLA:pTCR = 1:1、1:2`。这可以把任务权重与数据集大小解耦。

## 6. 正则化改进

### 6.1 Dropout

推荐从温和增大开始：

| 模块 | 当前 | 推荐起点 | 消融范围 |
|---|---:|---:|---:|
| adapter dropout | 0.10 | 0.15 | 0.10 / 0.15 / 0.20 |
| attention dropout | 0.10 | 0.15 | 0.10 / 0.15 / 0.20 |
| classifier dropout | 0.20 | 0.25 | 0.20 / 0.25 / 0.30 |

不建议一次增到很高。pTCR Recall 已经偏低，过强 dropout 可能进一步产生欠拟合。

### 6.2 Weight decay

保留 AdamW，但应拆分 decay/no-decay 参数：

- Linear/attention 权重：`weight_decay=0.01` 起步，可测试 `0.02`。
- bias、LayerNorm 参数：`weight_decay=0`。

当前统一对所有参数使用 `0.01` 不够精细。相比直接把全局 weight decay 提到很高，先排除 bias/normalization 更稳妥。

### 6.3 Gradient clipping 与大 batch

- 保留 `gradient_clip_norm=1.0`。
- 当前 batch size 为 1024，可补充比较 512；较小 batch 的梯度噪声可能改善泛化，但优先级低于实体分组加权。
- 记录 clipping 触发比例；如果大量 step 都被 clip，应先降低 LR/FGM，而不是继续收紧 clip。

## 7. 多任务共享表示改进

### 7.1 修正停止任务继续漂移的问题

当前 pTCR 停止后，pHLA 仍会更新共享 `peptide_adapter`；但 trainer 只评估 active task，因此 pTCR 不再被复评。对于 task-specific pTCR checkpoint 没有问题，但当前 unified state 的 pTCR 能力可能继续下降。

建议加入 retention gate：

1. 即使任务已 early-stop，每轮仍以只读方式复评其 validation。
2. 若 pTCR score 相对其最佳值下降超过 `0.005–0.01`，或 NLL 上升超过预设比例，则：
   - 冻结 shared peptide adapter，只继续训练 pHLA 私有分支；或
   - 回滚 shared adapter；或
   - 启用 task-specific peptide delta。

### 7.2 联合 checkpoint 必须使用当前状态分数

当前 joint round score 使用两个任务各自的历史最佳分数，这两个最佳分数可能来自不同轮次，但保存的是当前单一模型状态。建议改为：

```text
joint_current_score = mean(current_phla_score, current_ptcr_score)
```

并且每轮在所有更新结束后，对两个任务都重新评估。历史最佳分数只用于报告，不应用来代表当前 joint model。

### 7.3 如果冲突仍然明显

按实施复杂度排序：

1. shared peptide adapter 降低 LR。
2. pHLA/pTCR 固定 step 比例。
3. shared base + 两个轻量 task-specific residual/LoRA adapter。
4. 计算两个任务在 shared adapter 上的梯度余弦相似度。
5. 若负梯度冲突频繁，再引入 PCGrad/GradNorm。

不建议未经梯度诊断就直接加入复杂多任务优化算法。

## 8. 验证指标与 checkpoint 选择

### 8.1 每个 split 必须记录的指标

判别与阈值指标：

- AUROC、AUPR；
- Accuracy、MCC、F1；
- Precision、Recall、Specificity；
- confusion matrix。

概率与校准指标：

- NLL/Cross-Entropy；
- Brier score；
- ECE，并保存 reliability diagram；
- 正确/错误样本的平均置信度和 logit 绝对值；
- `p_true` 分位数和单样本 loss 的 P95/P99。

实体级稳健性指标：

- 对所有实体组均可计算：group-macro NLL、Brier、Accuracy，以及有对应标签时的 Recall/Specificity；
- AUROC/AUPR 只对同时含正负标签的实体组计算，并同时报告有效组数量与覆盖样本比例；
- pTCR unseen-peptide：peptide-macro 指标；
- pTCR unseen-TCR：TCR-macro 指标；
- pHLA unseen-HLA：HLA-macro 指标；
- worst-group、最差 10% group 平均和 group-CVaR；
- row-level 与 group-level 指标必须同时报告。

ECE 建议使用固定为 15 个等频 bins 的 adaptive ECE，并同时保留 reliability diagram。单独一个 ECE 数字依赖分箱规则，不应跨实现直接比较。

### 8.2 不再只保存一个“最佳”定义

建议每个任务保存三个 checkpoint：

| Checkpoint | 选择原则 | 用途 |
|---|---|---|
| `best_discrimination` | validation AUROC/AUPR 与分类综合表现 | 排序或普通分类 |
| `best_calibration` | 最低 NLL/Brier/ECE | 概率解释与风险控制 |
| `best_robust` | 最佳实体宏平均/worst-group | OOD 和长尾实体 |

若必须自动选择一个 checkpoint，推荐使用 Pareto/门控策略，而不是直接把所有指标任意线性相加：

```text
1. 找到 discrimination score 距最佳值不超过 0.005 的候选轮次；
2. 在候选中选择实体宏平均更高者；
3. 若仍接近，选择 NLL/Brier 更低的较早轮次。
```

这种策略通常会选择“判别性能几乎相同但校准更好”的 checkpoint。

### 8.3 split 权重

pHLA `val_unseen_pair` 只有约 60 条数据，不应与数万条的大 split 在模型选择中始终等权。建议：

- 同时报告等 split 宏平均和预先定义的 deployment-weighted 分数。
- tiny split 报告 bootstrap 置信区间，不以单次点估计主导 checkpoint。
- 权重必须在查看 test 结果前按真实应用需求固定，不能根据 test 表现反向调整。

### 8.4 阈值选择

固定 `0.5` 适合统一报告，但不一定适合部署。建议：

1. 保留 threshold-free 的 AUROC/AUPR。
2. 在独立 calibration validation 上为 pHLA、pTCR 分别选择阈值。
3. 可按 MCC、balanced accuracy，或业务规定的最低 Recall 选择。
4. 阈值和 temperature 一经选定，对 test 完全冻结。

不要在同一小 validation split 上同时选 checkpoint、阈值和 calibration temperature；应划分 calibration 子集或使用嵌套交叉验证。

### 8.5 每轮 task-best 模型的 UMAP 可视化

UMAP 应作为验证诊断工具加入 Stage1，而不是作为额外的训练损失或 checkpoint
选择指标。它的作用是把“validation loss 上升、Recall 下降、OOD 泛化变差”等
数值现象映射回表示空间，判断问题主要来自表示学习、分类边界、概率校准，还是
少数数据分布。

#### 设计原则

每轮验证前向时复用模型已有的 `pooled_embedding`，只对固定抽取的验证样本保留
向量、真实标签和预测概率。默认每个 validation split 最多抽取 1000 条样本，
并按真实标签进行可复现的分层抽样。这样不会为了可视化再次完整前向整个验证集，
也能保证不同轮次看到的是同一批样本。

当某一轮刷新 pHLA 或 pTCR 的 task-best 分数时，生成该任务的 UMAP 快照。首次
task-best 分别拟合任务独立的 `PCA(最多 50 维)+UMAP` 参考空间，后续 task-best
使用同一个参考降维器的 `transform()`。因此同一任务、同一次 run 中的轮次坐标
可以比较；pHLA 与 pTCR、不同 seed/run 之间的绝对坐标不能直接比较。

每个快照至少生成两张图：

- `umap_by_label.png`：按真实标签 0/1 着色，观察正负样本在表示空间中的分离；
- `umap_by_outcome.png`：按 TN/TP/FP/FN 着色，定位错误样本是否形成局部簇。

同时保存 `coordinates.csv.gz`，其中包含 validation split、原始数据集行索引、
真实标签、预测概率、TP/TN/FP/FN 和二维坐标；`summary.json` 保存轮次、task-best
分数、样本数、错误类型计数和参考空间信息。这样可以把 UMAP 中的 FN/FP 反查到
具体 peptide、HLA 或 TCR，再与 group-macro、worst-group 和单样本 loss 对照。

#### 如何解释 UMAP 与 loss/指标的联合变化

| UMAP 现象 | 可能原因 | 后续动作 |
|---|---|---|
| 正负标签逐渐分离，FP/FN 数量下降 | 表示和分类边界都在改善 | 保留当前训练方向，继续检查 calibration |
| 标签仍混杂，但 classifier 输出变得更自信 | 表示没有明显改善，主要是 logit 变大 | 降低 classifier/shared adapter LR，加入 label smoothing 或 logit/calibration 约束 |
| 训练相关样本分离，unseen split 仍混杂 | 训练实体专化、OOD 表示不足 | 检查 group weighting、任务采样比例和 shared adapter 更新 |
| FN 或 FP 在某些局部簇中集中 | 特定 peptide/HLA/TCR group 或数据分布造成系统性错误 | 结合坐标 CSV 做 group-level 统计，优先改采样、权重和 retention gate |
| UMAP 几乎不变但 validation loss 持续升高 | 表示变化有限，概率校准或分类头过度自信 | 重点查看 NLL、Brier、ECE、logit 绝对值和 P95/P99 单样本 loss |
| 不同轮次出现明显漂移且错误簇扩大 | 表示空间被后续任务更新破坏 | 检查 shared peptide adapter LR、任务冲突和停止任务复评 |

UMAP 的二维几何结构不应单独作为“模型变好”的证据，也不应用来替代 AUROC、
AUPR、NLL 或 group robustness。它只用于解释指标变化和筛选需要深入审计的数据
区域；最终 checkpoint 仍按第 8.2 节的判别、校准和稳健性门控规则选择。

#### 配置与输出

建议在 Stage1 配置中固定以下参数：

```yaml
training:
  best_umap_enabled: true
  best_umap_max_samples_per_split: 1000
  best_umap_random_state: 42
  best_umap_n_neighbors: 30
  best_umap_min_dist: 0.1
```

默认输出结构为：

```text
figures/stage1/best_model_umap/fold_<fold>/seed_<seed>/<timestamp>/
  index.json
  phla/
    reference_reducer.joblib
    round_XXXX/
      umap_by_label.png
      umap_by_outcome.png
      coordinates.csv.gz
      summary.json
  ptcr/
    ...
```

命令行可以使用 `--best-umap` 显式开启、使用 `--no-best-umap` 关闭，或通过
`--best-umap-output-dir` 指定输出目录。由于 UMAP 首次拟合会增加少量计算时间，
建议先在 seed 42 上启用，确认图和样本反查流程正确后再扩展到多个 seed。

## 9. Early stopping 改进

pTCR 在约 round 8 达到最佳，而 pHLA 长训练可到 round 38 才达到最佳。统一 patience 不合理。

建议：

| 任务 | 最小训练轮数 | Patience | 最大轮数 | 监控对象 |
|---|---:|---:|---:|---|
| pTCR | 4 | 3–5 | 20 | discrimination + group robustness，附 NLL gate |
| pHLA | 10 | 6–10 | 50 | unseen-HLA 与 unseen-peptide 的稳健综合分数 |

同时加入：

- NLL 连续恶化告警：例如连续 3 轮上升超过 `min_delta`。
- retention gate：另一任务更新共享参数后，本任务指标不能超过容忍退化。
- 选择较早轮次作为完全并列时的 tie-breaker。

## 10. 推荐的 Stage1-v2 起始配置

以下配置用于下一轮消融的起点，不代表已经验证的最优参数：

```yaml
model:
  model_dim: 256
  num_attention_heads: 8
  feedforward_dim: 1024
  adapter_dropout: 0.15
  attention_dropout: 0.15
  classifier_dropout: 0.25

training:
  max_rounds: 50
  scheduler_total_rounds: 40
  scheduler_warmup_rounds: 1
  scheduler_min_lr_ratio: 0.05

  shared_peptide_adapter_lr: 5.0e-5
  hla_adapter_lr: 1.0e-4
  tcr_adapter_lr: 1.0e-4
  phla_cross_attention_lr: 1.0e-4
  ptcr_cross_attention_lr: 7.5e-5
  pooling_lr: 1.0e-4
  classifier_lr: 7.5e-5

  weight_decay: 0.01
  no_decay_bias_and_norm: true
  gradient_clip_norm: 1.0

  label_smoothing:
    phla: 0.02
    ptcr: 0.03
  fgm_epsilon:
    phla: 0.5
    ptcr: 0.5
  adversarial_loss_weight:
    phla: 0.5
    ptcr: 0.25
  adversarial_warmup_cycles: 2

  group_weighting:
    phla: hla_label_inverse_sqrt
    ptcr: peptide_label_inverse_sqrt
    min_weight: 0.25
    max_weight: 4.0

  task_step_ratio:
    phla: 1
    ptcr: 1
  early_stopping_patience:
    phla: 8
    ptcr: 4

  best_umap_enabled: true
  best_umap_max_samples_per_split: 1000
  best_umap_random_state: 42
  best_umap_n_neighbors: 30
  best_umap_min_dist: 0.1
```

其中 scheduler、task-aware LR、loss/FGM、分组权重、task step ratio、retention 和
UMAP 字段已经加入当前 config/trainer；仍未纳入本轮的 P2 实验项见第 17 节。

## 11. 推荐的实验顺序

### 阶段 A：只增加观测，不改变训练行为

- 保存逐轮 validation predictions。
- 新增 NLL、Brier、ECE、group-macro、worst-group。
- 修正 joint-current score 和停止任务复评。
- 对每轮刷新后的 pHLA/pTCR task-best 生成固定样本 UMAP，并保存 FP/FN 对应的
  样本索引和预测概率。

目的：建立可信基线，并确定具体哪些 validation 实体驱动 loss 上升；同时区分
“表示空间变差”和“表示基本稳定但分类头过度自信”两类问题。

### 阶段 B：单因素筛选，先用 seed 42

| 实验 | 相对基线只改变 |
|---|---|
| B0 | 当前 256 配置，作为复现实验 |
| B1 | 固定 scheduler + 模块更新计数 |
| B2 | B1 + 全部 LR × 0.5 |
| B3 | B2 + label smoothing 0.03 |
| B4 | B3 + `lambda_adv=0.25, epsilon=0.5` |
| B5 | B4 + 实体 inverse-sqrt group weighting |

每一步只有在 validation discrimination 不显著下降、NLL/group robustness 改善时才保留。

### 阶段 C：多随机种子确认

将阶段 B 的当前基线和前两名候选使用 `42、3407、2026` 三个 seed 完整训练，报告均值、标准差和 bootstrap 置信区间。所有方案选择只使用 train/validation；test 仅在最终方案冻结后运行一次。

### 阶段 D：共享 adapter 消融

如果 pTCR retention 仍被 pHLA 长训练破坏，再比较：

1. shared adapter 低 LR；
2. pTCR 停止后冻结 shared adapter；
3. shared base + task-specific residual adapter；
4. PCGrad。

## 12. 成功标准

建议在实验前固定以下验收条件：

### pTCR

- unseen-peptide 和 unseen-TCR 的 row-level 与 entity-macro 指标不低于 256-early 基线。
- macro NLL 不再从早期最低点持续恶化超过预设容忍度。
- Recall 不因校准正则显著下降。
- worst 10% peptide group 的 NLL/Recall 改善。
- 三个 seed 的方向一致，不由单一 seed 驱动。

### pHLA

- 保留 256-long 对 unseen-HLA 和 unseen-peptide 的主要收益。
- unseen-HLA 的 HLA-macro 指标提高。
- tiny unseen-pair 不主导模型选择，且报告置信区间。

### 联合模型

- 当前同一 checkpoint 的 pHLA/pTCR 分数均满足 retention gate。
- 不再用不同轮次的历史最佳分数组合来代表当前 joint state。
- 修改 `max_rounds` 时，前 N 轮的 LR 序列保持一致。

## 13. 改动的保留/淘汰决策表

| 改动 | 预期直接改善 | 主要风险 | 保留条件 |
|---|---|---|---|
| scheduler 与 `max_rounds` 解耦 | 实验公平性、长短训练可比性 | 需要迁移 checkpoint 格式 | 属于逻辑修正；前 N 轮 LR 在不同时长实验中必须完全一致 |
| task-aware update count | 私有分支只按真实更新次数衰减 | 实现和恢复训练更复杂 | resume 后 LR、optimizer state、update count 与不中断训练一致 |
| shared adapter LR 降低 | 减少任务冲突与 OOD 表示漂移 | pHLA 收敛变慢 | pTCR NLL/retention 改善，pHLA 最终分数下降不超过预设容忍度 |
| classifier LR 降低 | 抑制 logit 过快增大 | 判别指标早期上升变慢 | discrimination 在更长预算内恢复，同时 NLL/ECE 改善 |
| label smoothing | 降低高置信度错误和 ECE | 可能降低 Recall 或正确样本置信度 | NLL/Brier/ECE 改善，Recall 与 AUPR 无显著下降 |
| 降低 FGM epsilon/权重 | 减少过强扰动和梯度尺度突变 | 鲁棒性收益可能减少 | OOD group 指标改善，且无明显 clean/robustness 回退 |
| inverse-sqrt 分组权重 | 改善低频实体与 group-macro | row-level 指标可能下降 | group-macro/worst-group 明显改善，row-level 分数下降不超过 0.005 |
| 增大 dropout | 减少 256 维过拟合 | 欠拟合、Recall 下降 | 多 seed 下 NLL 与 OOD 指标改善，不延长到不合理训练时间 |
| 1:1 task step ratio | 控制共享 adapter 的任务贡献 | pTCR 每个 cycle 数据覆盖减少 | 当前 joint state 的双任务 retention 改善 |
| retention gate | 防止已停止任务被共享更新破坏 | 可能限制另一任务继续提升 | 联合 checkpoint 的当前双任务指标优于原逻辑 |

### 13.1 单 seed 筛选的临时门槛

以 256-early 的 pTCR round 8 为参考：`validation performance=0.6630`，三个 split 等权 NLL 为 `0.8324`。建议候选方案首先满足：

- pTCR validation performance 不低于 `0.6580`，即距离基线不超过 `0.005`；
- 宏平均 NLL 至少改善约 3%，或在 Pareto 意义下明显优于基线；
- 任一主要大规模 split 的 discrimination score 不下降超过 `0.01`；
- Recall 不因 smoothing/正则化出现超过 `0.01` 的绝对下降；
- 新增的 peptide/TCR group-macro 和 worst-10% 指标优于基线。

这些只用于 seed 42 的预筛选。最终选择必须以三个 seed 的均值、方差和置信区间为准，不能把单 seed 门槛当成最终结论。

pHLA 可用长训练的最佳 validation score `0.7784` 作为参考，候选最终均值应尽量保持在其 `0.005` 范围内，同时重点检查 unseen-HLA，而不是用极小的 unseen-pair 决定去留。

### 13.2 Test 使用原则

此前 test 分析已经帮助定位了高频 unseen-peptide 风险，但下一轮方案搜索不能继续根据这些 test peptide 调参。后续流程应为：

```text
仅使用 train/validation 设计与筛选
  → 冻结损失、LR、正则化、checkpoint 规则、阈值和温度
  → 三个 seed 完成
  → 最终只运行一次 test
```

## 14. 最终推荐

最值得优先实施的组合是：

```text
256 维模型
+ 固定且任务感知的 LR scheduler
+ shared peptide adapter / classifier 降 LR
+ 轻度 label smoothing
+ 降低 FGM epsilon 和 adversarial loss 权重
+ peptide/HLA inverse-sqrt 分组加权
+ task-specific early stopping
+ discrimination、calibration、group robustness 三类 checkpoint
+ 停止任务 retention gate
+ 每轮 task-best 的固定样本 UMAP 与错误样本反查
```

其中，**scheduler 解耦、当前 joint score、停止任务复评属于训练逻辑正确性问题，应先于超参数搜索修正**；损失、学习率和 dropout 则应通过有顺序的消融实验决定，避免一次改动过多后无法归因。

## 15. 建议的代码实施顺序

### 15.1 配置层

修改 [`config.py`](../../source/plm_unified/config.py) 和 Stage1 YAML：

- 增加 scheduler 固定预算及最小 LR 比例；
- 将 LR 拆成 shared、pHLA-private、pTCR-private 参数；
- 增加 task-specific label smoothing、FGM epsilon、adversarial loss weight；
- 增加 group weighting、task step ratio、task-specific patience；
- 为所有新字段增加范围和互斥校验。

### 15.2 数据层

修改 [`data.py`](../../source/plm_unified/data.py)：

- 让 batch 在不影响 embedding cache 的情况下携带 peptide/receptor 标识或预计算 group id；
- 训练前统计 `(group, label)` 频次并生成归一化、裁剪后的样本权重；
- 明确 sampler 与 weighted loss 互斥；
- 保证固定 seed 和 resume 后的数据顺序可复现。

### 15.3 指标层

修改 [`metrics.py`](../../source/plm_unified/metrics.py)：

- 增加 Brier、adaptive ECE、置信度和单样本 loss 分位数；
- 增加 group-macro、worst-10%、CVaR 和有效组覆盖率；
- 对单标签组跳过 AUROC/AUPR，而不是产生 NaN 后静默求平均；
- 保留当前 `performance_avg` 作为兼容指标，但不再作为唯一 checkpoint 标准。

逐轮默认只保存 group 聚合、分位数和 top-loss 样本的 row id；完整逐样本预测使用压缩 Parquet/NPZ，只在预设轮次（例如 1、3、5、8、10）以及三类候选最佳轮次保存。这样既能定位错误来源，也避免数十轮 validation CSV 累积到数 GB。

### 15.4 Trainer 层

修改 [`trainer.py`](../../source/plm_unified/trainer.py)：

1. 将 criterion 改为 `reduction="none"`，再执行 label smoothing、样本加权和规范化求均值。
2. clean 与 adversarial loss 使用相同的 group weight，并按 `(clean + lambda*adv)/(1+lambda)` 合并。
3. 为 FGM 增加 task-specific epsilon、loss weight 和 warmup ramp。
4. 建立一个 shared optimizer 状态和两个 private optimizer 状态，或实现等价的 task-aware 参数组；同一 shared 参数只能有一份 Adam moments。
5. 每轮更新后始终评估 pHLA 和 pTCR，计算 current joint score。
6. 实现 discrimination/calibration/robust 三类 checkpoint 和 retention gate。
7. 保存 scheduler budget、各模块 update count、group-weight 配置、选择阈值和校准参数，确保可恢复训练。
8. 在每轮 validation forward 中复用 `pooled_embedding`，只缓存固定 UMAP 样本；
   当 task-best 刷新后拟合或复用任务参考降维器，生成 label/outcome 图、坐标表和
   summary，并把产物路径写入 tracker。
9. UMAP 生成失败时应保留已经保存的 checkpoint，并输出明确告警；可视化不能改变
   loss、指标和 checkpoint 选择逻辑。

### 15.5 测试层

在 [`test_plm_unified.py`](../../tests/test_plm_unified.py) 至少增加：

- 不同 `max_rounds` 下前 N 步 LR 完全相同；
- frozen private branch 不推进其 scheduler/update count；
- group weight 均值为 1、范围被正确裁剪；
- `lambda_adv=0` 与 clean loss 完全一致，改变 lambda 不隐式改变 loss 总尺度；
- 单标签 group 不计算 AUROC/AUPR且覆盖率正确；
- joint score 使用两个任务当前分数而非历史最佳分数；
- 已停止任务仍被复评且 retention gate 能触发；
- 中断恢复与连续训练产生相同的 LR、采样顺序和 checkpoint 选择结果。

## 16. 代码依据

- 当前损失、FGM、optimizer、scheduler 和 early stopping：[`trainer.py`](../../source/plm_unified/trainer.py)
- 当前 `performance_avg`：[`metrics.py`](../../source/plm_unified/metrics.py)
- 共享 peptide adapter 与任务模块：[`model.py`](../../source/plm_unified/model.py)
- 可复用的平方根 peptide 采样设计：[`stage12_data.py`](../../source/plm_unified/stage12_data.py)
- 当前 Stage1 配置：[`plm_stage1_ph_pt.yaml`](../../configs/plm_stage1_ph_pt.yaml)

## 17. 本次已实施的 Stage1-v2 内容

本方案中的第一阶段改进已接入训练代码和两个 Stage1 YAML：

- scheduler 使用独立的 `scheduler_total_rounds`、`scheduler_warmup_rounds` 和
  `scheduler_min_lr_ratio`，不再隐式随 `max_rounds` 改变；私有参数组只在对应
  任务 step 上推进 scheduler，shared peptide adapter 仍只保留一份 Adam 状态；
- optimizer 拆分 shared peptide adapter、pHLA/pTCR 私有模块的 LR，并对 bias 和
  LayerNorm 排除 weight decay；
- 训练损失支持任务特异 label smoothing、FGM epsilon、adversarial loss weight
  和 warmup；
- pHLA/pTCR 支持 entity-label inverse-sqrt 分组加权，并支持固定 task step ratio；
- validation 记录 NLL、Brier、ECE、置信度、P95/P99 单样本 loss、group-macro 和
  worst-group 指标；
- 每个任务分别保存 discrimination、calibration 和 robust 三类 task checkpoint；
- 已停止任务可继续进行只读 retention 复评，退化超过容忍值时冻结 shared peptide
  adapter；
- 当前 joint score 使用同一轮两个任务的当前分数，而不是不同轮次的历史最佳分数；
- 每轮刷新 task-best 时生成固定样本、固定参考空间的 UMAP，并保存 FP/FN 反查坐标。

PCGrad/GradNorm、Brier 作为训练正则项、temperature calibration 和更复杂的
task-specific residual adapter 保留为后续消融，不与本轮基础改动同时引入。
