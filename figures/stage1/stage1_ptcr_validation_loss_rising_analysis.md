# Stage1 中 128 与 256-early 的 pTCR Validation Loss 持续升高分析

生成日期：2026-09-14

## 1. 问题与结论

本文分析 [Stage1 pTCR loss 对比图](stage1_pTCR_loss_comparison.png) 中，`model_dim=128` 和 `model_dim=256 (early)` 的训练 loss 持续下降，而三个 validation split 的 loss 整体升高的原因。

> 核心结论：这是以 **OOD 验证集上的过拟合和概率置信度恶化** 为主、由 **验证集实体/来源分布差异、高频 peptide 的逐行加权、共享 peptide adapter 的任务干扰** 共同放大的现象。当前 checkpoint 选择分数不包含 loss，因此即使 validation loss 升高，只要 AUROC、AUPR 或固定阈值指标仍有改善，训练和最佳模型更新就可以继续。

曲线并非每一轮都严格单调增加，而是伴有波动的整体上升趋势。两个维度都出现相同模式，说明问题主要来自数据划分、训练目标和模型选择机制，而不是 256 维独有的问题。

![Stage1 pTCR loss comparison](stage1_pTCR_loss_comparison.png)

## 2. 逐轮数据证明了什么

### 2.1 128 维

| Round | Train loss | Val unseen-pair | Val unseen-peptide | Val unseen-TCR | 三个 val loss 等权平均 | pTCR 选择分数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3901 | 0.9665 | 0.8198 | 0.3785 | 0.7216 | 0.6255 |
| 2 | 0.3774 | 1.1247 | 0.8683 | 0.3765 | 0.7898 | 0.6323 |
| 3 | 0.3736 | 1.1530 | 0.8497 | 0.3794 | 0.7940 | 0.6317 |
| 4 | 0.3695 | 1.2019 | 0.9180 | 0.3795 | 0.8331 | 0.6361 |
| 5 | 0.3610 | 1.1583 | 0.9255 | 0.3831 | 0.8223 | 0.6392 |
| 6 | 0.3535 | 1.2351 | 0.9869 | 0.3842 | 0.8687 | 0.6427 |
| 7 | 0.3474 | 1.3049 | 0.9753 | 0.3868 | 0.8890 | 0.6445 |
| 8 | 0.3430 | 1.2136 | 0.9343 | 0.3868 | 0.8449 | **0.6504** |
| 9 | 0.3399 | 1.2440 | 0.9492 | 0.3887 | 0.8606 | 0.6495 |
| 10 | 0.3389 | 1.2387 | 0.9551 | 0.3886 | 0.8608 | 0.6489 |

从 round 1 到 round 10：

- Train loss 下降 `13.12%`。
- unseen-pair loss 上升 `28.16%`。
- unseen-peptide loss 上升 `16.51%`。
- unseen-TCR loss 上升 `2.66%`。
- 三个 validation loss 等权平均上升 `19.29%`。
- Train 与 validation 宏平均 loss 的差距从 `0.3315` 扩大到 `0.5219`。

### 2.2 256-early

| Round | Train loss | Val unseen-pair | Val unseen-peptide | Val unseen-TCR | 三个 val loss 等权平均 | pTCR 选择分数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3876 | 0.9071 | 0.9095 | 0.3783 | 0.7317 | 0.6522 |
| 2 | 0.3776 | 1.0441 | 0.8928 | 0.3861 | 0.7743 | 0.6490 |
| 3 | 0.3716 | 0.9764 | 0.8237 | 0.3790 | 0.7264 | 0.6545 |
| 4 | 0.3639 | 1.1116 | 0.9114 | 0.3795 | 0.8009 | 0.6451 |
| 5 | 0.3555 | 1.2186 | 0.9999 | 0.3867 | 0.8684 | 0.6500 |
| 6 | 0.3474 | 1.1466 | 0.9772 | 0.3864 | 0.8368 | 0.6507 |
| 7 | 0.3394 | 1.1753 | 1.0046 | 0.3901 | 0.8567 | 0.6584 |
| 8 | 0.3333 | 1.1166 | 0.9888 | 0.3918 | 0.8324 | **0.6630** |
| 9 | 0.3303 | 1.1620 | 1.0182 | 0.3949 | 0.8584 | 0.6606 |
| 10 | 0.3286 | 1.1601 | 1.0166 | 0.3956 | 0.8574 | 0.6612 |

从 round 1 到 round 10：

- Train loss 下降 `15.24%`。
- unseen-pair loss 上升 `27.88%`。
- unseen-peptide loss 上升 `11.78%`。
- unseen-TCR loss 上升 `4.55%`。
- 三个 validation loss 等权平均上升 `17.19%`。
- Train 与 validation 宏平均 loss 的差距从 `0.3440` 扩大到 `0.5288`。

### 2.3 两组实验的共同模式

两组实验使用同一数据、随机种子和 Stage1 训练超参数，Stage1 相关差异主要是模型宽度、注意力头数和 FFN 维度。它们都呈现：

1. Train loss 平稳下降。
2. unseen-pair 和 unseen-peptide loss 明显上升。
3. unseen-TCR loss 仅缓慢上升。
4. 最佳 pTCR 综合分数都出现在 round 8，而不是 validation loss 最低的 round 1–3。

因此，这是具有重复性的结构性现象，不像单轮随机抖动，也不能归因于某一个模型维度。

## 3. 第一主因：训练集与 validation 是实体级 OOD 分布

### 3.1 数据规模与训练集重叠

| 数据集 | 样本数 | 负/正样本 | 唯一 peptide | 唯一 TCR | 与训练集重叠 peptide | 与训练集重叠 TCR | 与训练集重叠 pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 664,952 | 332,476 / 332,476 | 1,427 | 154,259 | — | — | — |
| val unseen-pair | 10,382 | 5,191 / 5,191 | 200 | 9,377 | 0 | 0 | 0 |
| val unseen-peptide | 154,260 | 77,130 / 77,130 | 311 | 92,423 | 0 | 79,103 | 0 |
| val unseen-TCR | 143,028 | 71,514 / 71,514 | 1,410 | 33,022 | 1,404 | 0 | 0 |

这些 validation split 不是普通随机划分的 IID validation：

- `val_unseen_pair` 中 peptide、TCR 和 pair 与训练集都没有重叠，实际是最强的双实体分布迁移；其数据来源还主要是 `PHT_strict_unseen_triple:*`，而 pTCR train 全部来自 `PT.csv`。
- `val_unseen_peptide` 的所有 peptide 均未在训练中出现，但大部分 TCR 已见。
- `val_unseen_tcr` 的所有 TCR 均未见，但绝大部分 peptide 已见。

这解释了 loss 的分层：

- unseen-pair 的 loss 最高且增长最大，因为两个输入实体和主要数据来源都发生变化。
- unseen-peptide 次之，说明 peptide 的分布外迁移是当前 pTCR 分支的主要困难。
- unseen-TCR 最稳定，因为多数 peptide 已见，模型可以利用相对稳定的 peptide 表示。

随着训练继续，模型更充分地学习训练集中 1,427 个 peptide 及其搭配规律，训练 loss 自然下降；但这些规律不能完全迁移到新 peptide 或双实体均未见的数据，因此 OOD validation loss 反而升高。这是典型的训练分布专化。

### 3.2 不是简单的全局类别不平衡

Train 和三个 validation split 的正负样本总数均为 1:1，因此不能把曲线归因于普通的正负类别不平衡。真正的不均衡发生在实体组内部：不同 peptide/TCR 的样本量和组内正负比例差异很大。

## 4. 第二主因：Cross-Entropy 对高置信度错误极其敏感

Stage1 validation loss 使用普通 `CrossEntropyLoss`。对真实类别的预测概率记为 `p_true`，单样本 loss 为：

```text
loss = -log(p_true)
```

因此：

- 真实类别概率为 `0.6` 时，loss 约为 `0.511`。
- 真实类别概率为 `0.1` 时，loss 约为 `2.303`。
- 真实类别概率为 `0.01` 时，loss 约为 `4.605`。

少量越来越自信的错误，就能抵消大量普通样本上的小幅改善。AUROC/AUPR 主要评价排序，Accuracy/F1 等只看阈值两侧的类别，均没有 cross-entropy 对概率幅度敏感。

### 4.1 loss 与分类指标确实发生了分离

以 128 的 unseen-peptide 为例：

| 指标 | Round 1 | Round 8 | 变化方向 |
|---|---:|---:|---|
| Loss | 0.8198 | 0.9343 | 变差 |
| AUROC | 0.6928 | 0.7196 | 改善 |
| AUPR | 0.7199 | 0.7519 | 改善 |
| Accuracy | 0.6256 | 0.6447 | 改善 |
| MCC | 0.2994 | 0.3285 | 改善 |
| F1 | 0.4858 | 0.5343 | 改善 |
| Recall | 0.3537 | 0.4078 | 改善 |

模型把更多样本移动到了正确的排序或阈值一侧，但剩余错误样本获得了更差的真实类别概率，导致平均 cross-entropy 上升。

用 `exp(-mean_loss)` 表示模型分配给真实类别概率的几何平均，可更直观地看到这种变化：

| 配置与 split | Round 1 | Round 10 |
|---|---:|---:|
| 128 unseen-pair | 0.380 | 0.290 |
| 128 unseen-peptide | 0.441 | 0.385 |
| 256-early unseen-pair | 0.404 | 0.313 |
| 256-early unseen-peptide | 0.403 | 0.362 |

虽然该量不能定位具体错误样本，但它确认了整体概率质量在下降。结合 Accuracy/F1 的改善，最合理的解释是错误集中度和错误置信度在增加，即概率校准变差。

## 5. 第三主因：checkpoint 选择分数不包含 loss

Stage1 的 `performance_avg` 为以下五项指标的算术平均：

```text
(AUROC + Accuracy + MCC + F1 + AUPR) / 5
```

三个 validation split 的 `performance_avg` 再做等权宏平均，作为 pTCR early-stopping 分数。**Cross-entropy loss 不参与 checkpoint 选择。**

因此出现了看似矛盾但与实现完全一致的结果：

- 128：pTCR 选择分数从 round 1 的 `0.6255` 提升到 round 8 的 `0.6504`，同时宏平均 validation loss 从 `0.7216` 上升到 `0.8449`。
- 256-early：选择分数从 `0.6522` 提升到 round 8 的 `0.6630`，同时宏平均 validation loss 从 `0.7317` 上升到 `0.8324`。

两个配置在 round 8 后分数均不再改善，但 patience 为 5，而实验最多只训练到 round 10。round 9–10 只累计了两个未改善轮次，不足以触发 early stopping，所以训练按 `max_rounds=10` 结束。

这不是 early-stopping 代码失效，而是当前模型选择目标明确偏向排序与固定阈值分类指标，对概率校准退化没有约束。

## 6. 放大曲线的其他因素

### 6.1 validation 是逐行平均，高频 peptide 权重很大

| Split | 样本数 | 唯一 peptide | 每个 peptide 样本数中位数 | 最大样本数 |
|---|---:|---:|---:|---:|
| val unseen-pair | 10,382 | 200 | 9 | 2,245 |
| val unseen-peptide | 154,260 | 311 | 152 | 12,162 |
| val unseen-TCR | 143,028 | 1,410 | 32 | 4,290 |

当前 loss 按样本逐行平均，而不是先对每个 peptide 求平均。一个拥有 12,162 条记录的 peptide，其权重相当于数百个低频 peptide。只要少数高频 peptide 发生系统性高置信度误判，整条 validation loss 就会明显上升。

这与测试集中观察到的现象一致：少数高频 unseen peptide 可以贡献大部分假阴性和 loss。不过，当前训练日志没有保存每一轮的逐样本预测，因此“具体是哪几个 validation peptide 驱动曲线”仍需额外导出逐轮 logits 才能确认。

### 6.2 pHLA 与 pTCR 共同更新 peptide adapter

pHLA 和 pTCR 使用各自的 receptor adapter、cross-attention 和 classifier，但共享 `peptide_adapter`。每轮先后训练两个任务时，共享 peptide 表示会被两个目标反复更新。

如果 pHLA 与 pTCR 对 peptide 表示的最优梯度方向不一致，就可能出现：

- 当前任务训练 loss 继续下降；
- 共享表示更贴合训练实体；
- 某些 pTCR OOD peptide 的概率校准恶化；
- 曲线发生轮间振荡，而不是平滑单调变化。

该机制与现有架构和曲线波动相符，但尚未通过梯度余弦相似度或冻结 adapter 的消融实验直接验证，因此应视为可能的放大因素，不是已证实的唯一原因。

### 6.3 FGM 与图中 train loss 的口径

Stage1B 优化时同时反向传播 clean loss 和 FGM adversarial loss，但图中的 `round/ptcr/train/loss` 记录的是 clean loss，不是两者之和。Validation 则是在 `eval()` 模式下计算 clean cross-entropy。

此外，train loss 在 `train()` 模式下带 dropout，并由各 batch 更新前的预测累计；validation loss 在完整训练轮结束后、关闭 dropout 计算。因此 train 与 validation loss 的绝对值不是完全同口径。这个差异不能解释 validation 随轮次持续升高，但会影响两者绝对差距的精确解释。

## 7. 已确认事实与机制判断

| 判断 | 可信度 | 依据 |
|---|---|---|
| 已发生训练分布过拟合/专化 | 高 | Train loss 持续下降，三个 OOD val loss 整体上升，gap 显著扩大 |
| 概率校准或错误置信度恶化 | 高 | loss 上升但 AUROC/AUPR、Accuracy、F1 可同时改善；真实类别概率几何平均下降 |
| OOD 实体划分是主要诱因 | 高 | unseen-pair 双实体零重叠；unseen-peptide 的 peptide 零重叠；最强 OOD split loss 最高 |
| 当前 checkpoint 规则允许 loss 恶化 | 已确认 | `performance_avg` 明确不包含 loss，最佳分数在 round 8 |
| 高频 peptide 放大平均 loss | 中高 | 311 个 peptide 对应 154,260 行，单个 peptide 最大 12,162 行；尚缺逐轮样本归因 |
| pHLA/pTCR 共享 adapter 产生任务干扰 | 中 | 架构和轮间波动支持，但尚缺冻结/梯度消融 |
| 只是 256 模型太大 | 不支持 | 128 出现同样趋势，且相对升幅与 256-early 接近 |
| 只是正负类别不平衡 | 不支持 | Train 和各 validation split 全局均为 1:1 |
| loss 实现错误 | 暂无证据 | 训练与验证均使用标准 CrossEntropyLoss，样本加权计算正确 |

## 8. 对模型能力的含义

Validation loss 上升不等于模型的所有能力都在下降，它更准确地说明：

- 模型输出的概率越来越不可靠，尤其是在未见 peptide 和双实体未见场景。
- 排序能力或固定阈值分类能力在前 8 轮仍可能改善。
- 如果下游只使用排序，round 8 仍可能比 round 1 更好。
- 如果下游把预测值当作可信概率，或对高置信度错误非常敏感，则 round 8 的风险明显更高。
- round 8 之后选择分数和 loss 都不再改善，应视为已经进入没有实际收益的过拟合阶段。

因此，现有最佳 checkpoint 并非“选错”，而是优化目标偏向分类性能；如果研究目标还要求概率可解释性和 OOD 稳健性，现有选择标准不完整。

## 9. 建议的验证与改进

### 9.1 优先做的诊断

1. 每轮保存 validation 的 `label、probability、peptide、TCR、split`，计算每条样本和每个实体的 loss 变化。
2. 分别统计正确与错误样本的平均置信度、平均 logit 绝对值、NLL、Brier score 和 ECE。
3. 对每个 peptide/TCR 计算组内 loss，并列出 round 1、3、8、10 的最大退化组。
4. 同时报告 row-macro 与 peptide-macro/TCR-macro 指标，判断曲线是否由高频实体支配。
5. 至少重复 3 个随机种子；若趋势在不同种子复现，才能更稳健地确认结构性过拟合。

### 9.2 checkpoint 选择

可保留当前 `performance_avg` 主目标，但增加概率质量保护条件，例如：

```text
更新 checkpoint 的条件：
performance_avg 改善
并且 macro validation NLL 不超过历史最优值 + 容忍阈值
```

也可以保存两个检查点：

- `best_discrimination`：按当前 performance_avg 选择；
- `best_calibration`：按 validation NLL/Brier/ECE 选择。

这样可以根据下游是重排序还是概率决策选择模型，而不必把两个目标强行压成一个分数。

### 9.3 减少过拟合与过度置信

- 尝试较低的 classifier/cross-attention 学习率和更早停止。
- 对 classifier 使用适度 label smoothing；在 validation 上检查其是否降低 NLL/ECE，而不是只看 Accuracy。
- 增大 weight decay 或 dropout，但需要重新验证排序指标，避免过度正则化。
- 对训练数据按 peptide 进行分组采样或 loss reweight，减少高频 peptide 支配梯度。
- 使用 validation 数据做 temperature scaling；该方法改善校准但不改变排序，不应使用 test 数据拟合温度。
- 比较共享与 task-specific peptide adapter，或降低 pHLA 更新共享 adapter 时的学习率。

### 9.4 建议的消融实验

| 实验 | 目的 | 关键控制变量 |
|---|---|---|
| Round 1/3/5/8/10 逐样本预测 | 定位 loss 上升来自哪些样本和实体 | 同一 validation、同一阈值 |
| 固定 peptide adapter | 检验共享表示漂移 | 其他参数和训练顺序不变 |
| pTCR task-specific peptide adapter | 检验跨任务负迁移 | 参数量差异需单独报告 |
| 降低学习率 | 检验 logit/置信度增长过快 | 保持总步数与 scheduler 一致 |
| peptide-grouped loss | 检验高频实体支配 | 同时报告 row-level 与 group-level 指标 |
| label smoothing / temperature scaling | 检验校准问题 | 分开比较排序、阈值指标与 NLL/ECE |

## 10. 最终判断

128 和 256-early 的 validation loss 上升，主要不是因为模型“完全没有学到东西”，而是因为模型继续提高训练集拟合和部分分类指标时，对 OOD 验证样本中的剩余错误变得更加自信。

其逻辑链条为：

```text
继续训练
  → 更贴合训练 peptide/TCR 及其组合规律
  → train loss 下降
  → 部分 validation 样本跨过分类阈值，Accuracy/F1 等改善
  → 剩余 OOD 错误的真实类别概率进一步降低
  → Cross-Entropy 被高置信度错误主导并升高
  → checkpoint 选择分数因不包含 loss，仍可继续改善到 round 8
```

当前证据支持把该现象定义为：**OOD 场景下“判别指标局部改善、概率质量持续恶化”的过拟合，而不是普通训练失败。**

## 11. 数据与代码依据

- 原始曲线数据：
  - `swanlog/run-20260824_174207-atzgps52okvljt1gllrtg/backup.swanlab`
  - `swanlog/run-20260831_174047-frlqhls5o52qllgdm8q85/backup.swanlab`
- Stage1 validation 使用 `CrossEntropyLoss`，并按样本数求平均：[`trainer.py`](../../source/plm_unified/trainer.py)。
- `performance_avg` 不包含 loss：[`metrics.py`](../../source/plm_unified/metrics.py)。
- pHLA/pTCR 共享 `peptide_adapter`：[`model.py`](../../source/plm_unified/model.py)。
- 数据文件：
  - [`PT train`](../../dataset/PT/train.csv)
  - [`PT val unseen-pair`](../../dataset/PT/benchmarks/val_unseen_pair.csv)
  - [`PT val unseen-peptide`](../../dataset/PT/benchmarks/val_unseen_peptide.csv)
  - [`PT val unseen-TCR`](../../dataset/PT/benchmarks/val_unseen_tcr.csv)
- 综合模型能力和测试集分布分析：[`stage1_dimension_training_test_analysis.md`](stage1_dimension_training_test_analysis.md)。
