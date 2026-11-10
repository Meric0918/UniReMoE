# Stage 3：HLA–Peptide–TCR 三元微调

本文件集中整理 Stage 3 的训练管线、三方对比和 pTCR 外部分析。

## 阶段说明

> 来源：`PLM_STAGE3.md`

该管线可以在 PLM Stage 1 或 Stage 2 联合模型上增加显式三元分支：

```text
peptide ─┐
         ├─ peptide→HLA cross-attention ── pHLA-conditioned peptide ─┐
HLA ─────┘                                                           │
                                                                     ├─ pooling ─ classifier
TCR ───────────────────── pHLA-conditioned peptide→TCR attention ────┘
```

它保留 pHLA、pTCR 两个原分支，并在 Stage 3B 中通过受控二元回放降低灾难性遗忘。

## 1. 输入与数据审计

当前配置使用 `dataset/PHT/classes/` 下的两个类别目录：

```text
PHT_000/
  train.csv
  val.csv
PHT_111/
  train.csv
  val.csv
```

`PHT_000he` 按实际目录名 `PHT_000` 处理。两个目录的 train 文件在内存中合并用于训练，两个 val 文件在内存中合并用于验证，不改写原始数据集。文件包含 `peptide,HLA,tcr,PH,PT,PHT` 及相关审计列，PHT 标签类型使用 `000` 和 `111`。

加载时按以下规则构建内存数据集：

- peptide 长度为 8–15；
- HLA/TCR 不超过配置上限 34；
- 标签只能为 0/1；
- 相同三元组出现冲突标签时停止；
- 类别目录顺序固定为 `PHT_000`、`PHT_111`，源文件路径写入审计信息；
- 集合内重复只在加载时检查，不修改原 CSV；
- 非标准残基保留给现有 tokenizer，但数量写入审计信息。

## 2. 初始化与训练

### 2.1 从 Stage 2 初始化

正式运行：

```bash
cd /home/mjp/Project/PLM/UnifyImmun

/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

加载器要求源文件为 `format_version=1`、`checkpoint_type=stage2_training` 的完整联合检查点。它会校验模型配置、fold 和三个基础 PLM 的指纹；源 checkpoint 不必处于 `complete`，因此 best Stage 2B checkpoint 可以直接使用。

### 2.2 从 Stage 1 直接初始化

```bash
cd /home/mjp/Project/PLM/UnifyImmun

/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage1-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_42/last.pt \
  --fold 4 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

Stage 1 来源必须是 `stage=complete` 的完整 `last.pt`，且模型配置、fold、seed 和三个基础 PLM 指纹均与当前 Stage 3 运行一致。禁止使用 `warmup_last.pt`，也禁止拼接 `best_phla.pt` 与 `best_ptcr.pt`：两个单任务文件可能来自不同训练轮次，并包含不同版本的共享 Peptide Adapter。

Stage 1 只提供三个 Adapter 和 pHLA/pTCR 下游分支。三个在线基础 PLM 从配置路径加载原始预训练权重；triplet attention、pooling 和 classifier 从 Stage 1 的 pTCR 分支复制初始化。

两种初始化方式均使用全新的 optimizer 和 scheduler，不继承源 checkpoint 的优化器状态。之后执行相同的 Stage 3 训练流程：

训练分两步：

1. Stage 3A 固定 2 个 epoch，只训练由 pTCR 分支复制初始化的 triplet cross-attention、pooling 和 classifier；
2. Stage 3B 每轮按 pHLA replay、pTCR replay、完整 triplet epoch 的顺序训练，渐进解冻三个 PLM 顶部各 2 层。

两个二元 replay 各抽取约 `0.25 × triplet_train_size` 个样本。每个验证集、每个任务均记录 ROC-AUC、AUPR、Accuracy、MCC、F1、Sensitivity、Specificity、Precision、Recall 和 validation loss。离散指标使用严格阈值：

```text
y_pred = (y_prob > 0.5).astype(int)
```

每个任务的验证分数和联合 round score 为：

```text
performance_avg = (ROC-AUC + Accuracy + MCC + F1 + AUPR) / 5
round_score = (pHLA performance_avg + pTCR performance_avg + PHT performance_avg) / 3
```

三个任务等权，样本量大的验证集不会自动获得更高权重。loss 只记录，不参与 checkpoint 选择。Stage 3B 至少训练 3 轮，连续 5 轮 round score 没有至少 `0.001` 的提升时早停；最多训练 10 轮。

## 3. 调试与恢复

PHT 模式使用 `--limit-per-type` 限制审计完成后每个 PHT 类型返回的样本，用于 smoke test：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --seed 9999 \
  --device cuda \
  --num-workers 0 \
  --swanlab-mode disabled \
  --limit-per-type 64
```

恢复训练：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --resume \
    trained_model/PLM_Triplet/fold_1/seed_42/stage3_last_<timestamp>.pt
```

`--resume`、`--stage1-checkpoint` 与 `--stage2-checkpoint` 互斥；新训练必须指定其中一种初始化 checkpoint。恢复文件包含 optimizer、scheduler、GradScaler、DataLoader RNG、全局 RNG、SwanLab run id、阶段、patience 状态以及统一的 `initialization_source` 来源信息。旧版仅包含 `source_stage2` 的 Stage 3 checkpoint 仍可恢复。

## 4. 训练产物

同一运行共享时间戳：

```text
trained_model/PLM_Triplet/fold_1/seed_42/
  stage3a_best_triplet_<timestamp>.pt
  stage3b_best_joint_<timestamp>.pt
  best_round/
    stage3_best_joint_<timestamp>.pt
  stage3_last_<timestamp>.pt
  triplet_best_<timestamp>.pt
```

- `stage3_last` 是完整恢复点，保留最后一轮状态；
- `stage3a_best_triplet` 用于进入 Stage 3B 前回载最佳预热权重；
- `stage3b_best_joint` 是完整的联合最佳训练点；
- `best_round/stage3_best_joint` 是按 round score 选出的完整联合模型，包含 pHLA、pTCR、PHT 三个头，不拆分保存；
- `triplet_best` 去除 optimizer，但保留三个 PLM 和 pHLA/pTCR/triplet 三个推理分支。

所有新产物通过 `initialization_source.kind=stage1|stage2` 记录初始化分支，并同时保存来源路径、fold、seed、训练阶段和基础 PLM 指纹。Stage 2 来源产物继续保留 `source_stage2` 兼容字段。

## 5. Validation 阈值与独立测试

训练只使用 validation 选择阈值：先最大化 MCC，并列时依次选择更高 F1、离 0.5 更近的阈值。该阈值写入 `triplet_best`，测试时固定使用。

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --checkpoint \
    trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_<timestamp>.pt \
  --split test \
  --device cuda \
  --hla-shuffle
```

输出：

```text
evaluation_triplet_test_<timestamp>.json
predictions_triplet_test_<timestamp>.csv
```

JSON 同时包含固定 0.5 阈值和 validation 选择阈值的指标。CSV 保留原始 CSV 行号、三条序列、标签、阳性概率和预测类别。`--hla-shuffle` 会额外报告确定性 HLA 位移消融；若性能和概率几乎不变，说明模型没有充分学习 HLA restriction。

如需重新分析 validation 阈值：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --checkpoint <triplet_best.pt> \
  --split val \
  --select-threshold \
  --no-predictions
```

禁止在 test 上使用 `--select-threshold`。

## 6. 测试

合成测试不加载服务器大模型：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_unified \
  tests.test_plm_stage2 \
  tests.test_plm_triplet
```

正式训练前还应运行 `source/smoke_test_plm_encoders.py`，确认 ESM-C 和 TCR-BERT tokenizer 仍保持一残基一 token 对齐。

## Stage1/Stage3/Baseline 对比报告

> 来源：`paper/baseline_stage1_stage3.md`

> **对比目的**：在相同的测试数据集（COVID / Independent / Triple）上，对比三个模型的推理结果，评估不同训练阶段的增益与权衡。
> - **Baseline**：`model_TCR_fold4.pkl`，单一 TCR-肽结合二分类模型（fold4）
> - **Stage1**：`best_ptcr.pt`，PLM 统一模型 ptcr 单任务（**fold1**）
> - **Stage3**：`triplet_best_...pt`，多任务模型 ptcr 分支（fold4）
>
> ⚠️ **重要提示**：Baseline 与 Stage3 均为 **fold4**，而 Stage1 为 **fold1**。fold 之间的数据划分不同，因此 Stage1 与另外两个模型的比较并非严格同条件对照，结论需谨慎解读。

---

## 1. 测试配置对比

| 项目 | Baseline | Stage1 | Stage3 |
|---|---|---|---|
| 模型文件 | `trained_model/TCR_10/model_TCR_fold4.pkl` | `trained_model/PLM_Unified/fold_1/seed_2026/best_ptcr.pt` | `trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt` |
| Fold | **fold4** | **fold1** | **fold4** |
| 模型类型 | 单一 TCR-肽结合二分类 | PLM 统一模型，ptcr 单任务 | Stage3 多任务模型（ptcr 分支） |
| 推理方式 | 模型直接输出结合概率 | full ptcr 推理 | full_stage3_checkpoint_ptcr_branch |
| 阈值 | 0.5 | 0.5 | 0.5（checkpoint 自带 0.50390625） |
| 测试数据目录 | `data/data_TCR/test_data_set` | `data/data_TCR/test_data_set` | `data/data_TCR/test_data_set` |
| 测试数据集 | covid / independent / triple | 同左 | 同左 |
| 推理设备 | CUDA：RTX 4090 | CUDA：RTX 4090 | CUDA：RTX 4090 |
| 肽长度过滤 | 仅保留 [8, 15] | 仅保留 [8, 15] | 仅保留 [8, 15] |

> 注：三个模型均对 `triple_set.csv` 过滤掉 244 条（肽过长 190 + 肽过短 54），Triple 样本数均为 96,799；COVID 与 Independent 三个模型样本数一致。

---

## 2. 模型架构简介

本节简要介绍 Stage1 与 Stage3 的主体模型结构。两者均基于 **PLM-UnifyImmun** 框架，共享相同的编码器骨干（蛋白质语言模型 + TCR 语言模型），但在训练策略与任务分支上存在关键差异。Baseline 模型（`model_TCR_fold4.pkl`）为独立的单一二分类模型，不在此架构图范围内。

### 2.1 Stage1：单任务 ptcr 模型

> 原始报告引用了 `PLM-Unifyimmun_Stage1.png`，但该图片文件不在当前仓库中。

Stage1 是一个**单任务**模型，仅训练 **ptcr（peptide-TCR 结合预测）** 任务。主体结构包含：

- **肽段编码器**：采用 **ESMC-300M** 蛋白质语言模型（3 亿参数，30 层 Transformer，Pre-LN + RoPE + SwiGLU 架构），将肽段氨基酸序列编码为稠密向量表示。
- **TCR 编码器**：采用 **TCR-BERT**（12 层 BERT，768 隐藏维，氨基酸级 tokenizer），将 TCR CDR3 序列编码为序列嵌入。
- **结合预测头**：两个编码器的输出经池化后，通过交互模块（如拼接 + MLP）预测 peptide-TCR 结合概率。
- **训练方式**：仅使用 ptcr 二分类标签（结合 / 不结合）进行端到端训练，损失函数为二元交叉熵。

### 2.2 Stage3：多任务 Triplet 模型

> 原始报告引用了 `PLM-Unifyimmun_Stage3.png`，但该图片文件不在当前仓库中。

Stage3 是一个**多任务**模型，在共享编码器骨干之上同时训练多个免疫相关任务。主体结构包含：

- **共享编码器骨干**：与 Stage1 相同的 ESMC-300M（肽段）+ TCR-BERT（TCR）双编码器，权重在多个任务间共享。
- **多任务分支**：除了 ptcr（peptide-TCR 结合）分支外，还包含其他免疫预测分支（如 pmhc 等），各分支有独立的预测头但共享底层编码器。
- **Triplet 训练**：采用 triplet loss 策略，同时优化正样本对（结合）与负样本对（非结合）之间的距离，增强编码器的区分能力。
- **推理分支**：本次评估使用 `ptcr` 分支输出，即多任务模型中专门负责 peptide-TCR 结合预测的分支。

### 2.3 Stage1 与 Stage3 的核心差异

| 维度 | Stage1 | Stage3 |
|---|---|---|
| 训练任务数 | 单任务（仅 ptcr） | 多任务（ptcr + 其他免疫任务） |
| 编码器共享 | 仅 ptcr 使用 | 多任务共享同一组编码器权重 |
| 训练策略 | 端到端二元分类 | 多任务联合 + Triplet loss |
| 表示学习 | 任务特异 | 跨任务泛化（其他任务的知识可迁移至 ptcr） |
| 推理分支 | 整个模型即 ptcr | 从多任务模型中提取 ptcr 分支 |

> **架构与性能的关联**：Stage3 通过多任务共享编码器，使 ptcr 分支能利用其他免疫任务学到的蛋白质/TCR 表示知识，在困难数据集（如 COVID）上显著提升了正类召回能力。但共享编码器也可能在分布差异较大的数据集（如 Triple）上引入偏移，导致负类误判增加。

---

## 3. 总体指标对比

> 每个数据集中，**最优值加粗**；`▲/▼` 表示 Stage3 相对 Baseline 的变化方向。

### 3.1 COVID 数据集（样本数 1,089,310）

| 指标 | Baseline | Stage1 | Stage3 | 最优 | S3 vs Baseline |
|---|---:|---:|---:|---|---:|
| AUROC | 0.585461 | 0.535709 | **0.609340** | Stage3 | ▲ +4.08% |
| AUPR | 0.587099 | 0.561638 | **0.611107** | Stage3 | ▲ +4.09% |
| Accuracy | 0.529265 | 0.539297 | **0.586170** | Stage3 | ▲ +10.76% |
| MCC | 0.102005 | 0.098657 | **0.193442** | Stage3 | ▲ +89.64% |
| F1 | 0.227067 | 0.362247 | **0.474905** | Stage3 | ▲ +109.14% |
| Precision | **0.650662** | 0.595737 | 0.655820 | Baseline | ▲ +0.79% |
| Recall | 0.137531 | 0.260247 | **0.372224** | Stage3 | ▲ +170.66% |
| Specificity | **0.925342** | 0.821441 | 0.802488 | Baseline | ▼ −13.28% |

### 3.2 Independent 数据集（样本数 28,707）

| 指标 | Baseline | Stage1 | Stage3 | 最优 | S3 vs Baseline |
|---|---:|---:|---:|---|---:|
| AUROC | 0.945459 | **0.948968** | 0.947995 | Stage1 | ▲ +0.27% |
| AUPR | 0.903840 | 0.911458 | **0.911484** | Stage3 | ▲ +0.85% |
| Accuracy | 0.881527 | 0.886892 | **0.887310** | Stage3 | ▲ +0.66% |
| MCC | 0.737195 | 0.750608 | **0.753442** | Stage3 | ▲ +2.20% |
| F1 | 0.826824 | 0.837268 | **0.840397** | Stage3 | ▲ +1.64% |
| Precision | **0.843883** | 0.840765 | 0.830846 | Baseline | ▼ −1.55% |
| Recall | 0.810441 | 0.833799 | **0.850170** | Stage3 | ▲ +4.90% |
| Specificity | 0.919632 | **0.915351** | 0.907218 | Stage1 | ▼ −1.35% |

### 3.3 Triple 数据集（样本数 96,799）

| 指标 | Baseline | Stage1 | Stage3 | 最优 | S3 vs Baseline |
|---|---:|---:|---:|---|---:|
| AUROC | **0.862162** | 0.854428 | 0.833440 | Baseline | ▼ −3.33% |
| AUPR | **0.851110** | 0.827757 | 0.813461 | Baseline | ▼ −4.42% |
| Accuracy | **0.809006** | 0.780111 | 0.787394 | Baseline | ▼ −2.67% |
| MCC | **0.601771** | 0.534475 | 0.548974 | Baseline | ▼ −8.77% |
| F1 | **0.702095** | 0.654245 | 0.672772 | Baseline | ▼ −4.18% |
| Precision | **0.898503** | 0.847987 | 0.843574 | Baseline | ▼ −6.11% |
| Recall | **0.576151** | 0.532568 | 0.559490 | Baseline | ▼ −2.89% |
| Specificity | **0.958278** | 0.938799 | 0.933493 | Baseline | ▼ −2.59% |

---

## 4. 混淆矩阵对比

> Stage1 / Stage3 的混淆矩阵由 sensitivity / specificity 反推得到（与官方 Accuracy 完全吻合）。

### 4.1 COVID 数据集（负类 541,653 / 正类 547,657）

|  | Baseline 预测负 | Baseline 预测正 | Stage1 预测负 | Stage1 预测正 | Stage3 预测负 | Stage3 预测正 |
|---|---:|---:|---:|---:|---:|---:|
| 真实负类 | 501,214 | 40,439 | 444,936 | 96,717 | 434,670 | 106,983 |
| 真实正类 | 472,337 | 75,320 | 405,131 | 142,526 | 343,806 | 203,851 |

**解读**：从 Baseline → Stage1 → Stage3，正类检出量持续提升（75,320 → 142,526 → 203,851），但负类误判也随之增加（40,439 → 96,717 → 106,983）。Stage3 在 COVID 上正类检出量约为 Baseline 的 2.7 倍。

### 4.2 Independent 数据集（负类 18,689 / 正类 10,018）

|  | Baseline 预测负 | Baseline 预测正 | Stage1 预测负 | Stage1 预测正 | Stage3 预测负 | Stage3 预测正 |
|---|---:|---:|---:|---:|---:|---:|
| 真实负类 | 17,187 | 1,502 | 17,107 | 1,582 | 16,955 | 1,734 |
| 真实正类 | 1,899 | 8,119 | 1,665 | 8,353 | 1,501 | 8,517 |

**解读**：三个模型在 Independent 上表现接近，Stage3 正类检出最多（8,517），漏检最少（1,501），整体全面提升。

### 4.3 Triple 数据集（负 58,986 / 正 37,813；三个模型相同）

|  | Baseline 预测负 | Baseline 预测正 | Stage1 预测负 | Stage1 预测正 | Stage3 预测负 | Stage3 预测正 |
|---|---:|---:|---:|---:|---:|---:|
| 真实负类 | 56,525 | 2,461 | 55,376 | 3,610 | 55,063 | 3,923 |
| 真实正类 | 16,027 | 21,786 | 17,675 | 20,138 | 16,657 | 21,156 |

**解读**：Baseline 在 Triple 上仍保持最佳整体表现；在使用与 Stage1 / Stage3 相同的 96,799 条样本后，其误判最少（FP 为 2,461），Precision（0.8985）与 Specificity（0.9583）均高于 Stage1 / Stage3。Baseline 的正类检出量也最高（21,786），但相较原先未过滤数据的结果，指标会因测试样本变化而有所不同。

---

## 5. 三模型综合表现概览

| 数据集 | Baseline | Stage1 (fold1) | Stage3 | 趋势 |
|---|---|---|---|---|
| **COVID** | 区分能力弱，Recall 极低（13.8%） | Recall 提升近一倍，AUROC 略降 | **显著提升**，F1/MCC/Recall 翻倍 | 随训练阶段递进，Recall 持续提升，代价是 Specificity 下降 |
| **Independent** | 接近上限（AUROC 0.945） | 与 Baseline 持平略优 | **小幅全面提升**，Recall 最高 | 三者接近，Stage3 边际稳健增益 |
| **Triple** | **最优**，AUROC 0.862、Precision 0.899、Specificity 0.958 | 整体回落，Precision 降至 0.848 | 略优于 Stage1 但仍不及 Baseline | 在相同过滤后测试集上，Baseline 仍全面领先 |

---

## 6. 关键观察与结论

### 6.1 训练阶段的影响

1. **COVID（模型弱项数据集）**：从 Baseline → Stage1 → Stage3，Recall 持续提升（13.8% → 26.0% → 37.2%），F1 翻倍增长。多任务学习（Stage3）对困难分布下的正类识别有实质性帮助。
2. **Independent（已接近上限）**：三个模型表现接近，Stage3 在不牺牲整体性能前提下进一步降低漏检，属于"边际稳健增益"。
3. **Triple**：在统一使用 96,799 条过滤后样本的前提下，Baseline 仍全面领先；PLM 系列（Stage1/3）负类误判更多，导致 Precision/Specificity 下降。

### 6.2 需要注意的混淆因素

- **fold 不一致**：Stage1 使用 fold1，而 Baseline/Stage3 使用 fold4。严格对比 Stage1 与 Stage3 的训练阶段增益时，fold 差异是主要混淆因素，建议后续补充 fold4 的 Stage1 结果以做严格对照。
- **样本过滤**：三个模型均使用过滤后的 Triple 测试集（肽长 [8,15]），样本数均为 96,799，因此本次 Triple 指标处于相同样本基础上。

### 6.3 建议

1. **召回优先场景**（如候选筛选、湿实验验证前过滤）：推荐 Stage3，尤其 COVID 上 Recall 提升显著。
2. **精度优先 / 误报敏感场景**（如 Triple 高特异性需求）：Baseline 更优；若用 Stage3，建议对 ptcr 分支单独做阈值调优（当前统一 0.5 可能偏低）。
3. **后续实验**：补充 fold4 的 Stage1 评估，消除 fold 混淆，做严格的阶段间对照。

---

## 7. 相关文件

- Baseline 报告：`08.03/model_TCR_fold4_test_report.md`
- Stage1 评估结果：`08.03/evaluation_ptcr_sets.json`
- Stage3 评估结果：`08.03/evaluation_ptcr_data_TCR_sets_20260803_103409_300147.json`

## Stage3 Triplet checkpoint 的 pTCR 数据集分析

> 来源：`tests/PLM_Triplet_pTCR_data_TCR_sets_Analysis_20260801.md`

## 1. 分析概况

本报告整理使用 Stage 3 `triplet_best` 模型对 `data/data_TCR` 目录下三个
`*_set.csv` 数据集进行的独立评估。

| 项目 | 内容 |
|---|---|
| 评估日期 | 2026-08-01 |
| Checkpoint | `trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260731_145558_787556.pt` |
| Checkpoint 类型 | `stage3_multitask_model` |
| 使用分支 | pTCR 二元分支 |
| 输入字段 | `peptide,tcr,label` |
| HLA 输入 | 无 |
| 分类阈值 | 0.5 |

由于这些文件不包含 HLA 列，不能使用 HLA-peptide-TCR 三元分支；本次测试使用
同一个 checkpoint 中保留的 pTCR 分支。`triple_set.csv` 虽然名称包含
`triple`，但其文件本身仍是 peptide-TCR 二元数据。

## 2. 模型与推理方式

该 checkpoint 来自 PLM Stage 3，完整模型包含：

- peptide ESM-C 300M 编码器；
- HLA ESM-C 300M 编码器；
- TCR-BERT 编码器；
- pHLA、pTCR 和 triplet 三个预测分支。

本次只调用 pTCR 分支。为保证使用 checkpoint 中已经微调过的 PLM 参数，测试时
采用完整 checkpoint 编码器在线推理，而不是直接使用 Stage 1 的旧 embedding
cache。所有数据集中的唯一 peptide 和唯一 TCR 各编码一次，再复用其表示完成
样本级预测：

- 唯一 peptide：1,994 个；
- 唯一 TCR：200,202 个。

模型配置要求 peptide 长度为 8–15。评估没有在这些测试数据上重新选择阈值，
固定使用 0.5；checkpoint 中记录的 0.46875 是 triplet 验证集阈值，不用于本次
pTCR 测试。

## 3. 数据审计

| 数据集 | 原始有效行数 | 实际评估样本数 | 正样本 | 负样本 | 过滤情况 |
|---|---:|---:|---:|---:|---|
| `covid_set.csv` | 1,089,310 | 1,089,310 | 547,657 | 541,653 | 无过滤 |
| `independent_set.csv` | 28,707 | 28,707 | 10,018 | 18,689 | 无过滤 |
| `triple_set.csv` | 97,043 | 96,799 | 37,813 | 58,986 | 54 条 peptide 过短，190 条过长 |

`triple_set.csv` 共过滤 244 行，占原始数据的约 0.251%。过滤依据与项目现有
pTCR 数据加载逻辑一致。

## 4. 测试结果

所有指标均在固定阈值 0.5 下计算；AUROC 和 AUPR 不依赖该固定阈值。

| 数据集 | 样本数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `covid_set` | 1,089,310 | 0.547126 | 0.556212 | 0.533417 | 0.085926 | 0.345629 | 0.586017 | 0.245091 | 0.824940 | 1.415277 |
| `independent_set` | 28,707 | 0.951500 | 0.915163 | 0.892326 | 0.762469 | 0.844899 | 0.849460 | 0.840387 | 0.920167 | 0.273892 |
| `triple_set` | 96,799 | 0.834734 | 0.809739 | 0.777653 | 0.529339 | 0.648518 | 0.847750 | 0.525110 | 0.939545 | 0.677436 |

## 5. 分数据集解读

### 5.1 `independent_set`

该数据集表现最好：AUROC 为 0.951500，AUPR 为 0.915163，MCC 为 0.762469，
F1 为 0.844899。模型在固定阈值下同时保持了较高的 precision、recall 和
specificity，说明其排序能力和二分类决策均较稳定。

该结果与 checkpoint 中记录的 pTCR validation 结果接近：

| 指标 | Validation | `independent_set` |
|---|---:|---:|
| AUROC | 0.950573 | 0.951500 |
| AUPR | 0.913988 | 0.915163 |
| MCC | 0.757345 | 0.762469 |
| F1 | 0.841159 | 0.844899 |

这说明 `independent_set` 与训练/验证数据分布较接近，模型具有较好的外部泛化
表现。

### 5.2 `triple_set`

该数据集仍有一定预测能力，但明显低于 `independent_set`：AUROC 为 0.834734，
AUPR 为 0.809739，MCC 为 0.529339。

模型 specificity 较高（0.939545）、precision 较高（0.847750），但 recall
只有 0.525110。这表明模型预测为阳性时通常较可靠，但漏掉了较多真实阳性；在
当前固定阈值下，模型整体偏保守。

### 5.3 `covid_set`

该数据集上的表现接近随机：AUROC 为 0.547126，AUPR 为 0.556212，MCC 仅为
0.085926。Recall 只有 0.245091，loss 升高到 1.415277，说明模型在该数据集上
不仅阳性识别能力弱，概率输出也可能存在明显的域偏移或校准问题。

因此，当前 checkpoint 不适合直接用于 `covid_set` 上的高可靠 pTCR 阳性识别，
需要进一步进行 covid 域适配、重新校准或独立训练。

## 6. 总体结论

模型在三个数据集上的表现呈现清晰梯度：

```text
independent_set  >>  triple_set  >>  covid_set
```

- `independent_set`：接近 validation 表现，外部泛化良好；
- `triple_set`：具有中等排序能力，但固定阈值下阳性召回偏低；
- `covid_set`：接近随机，存在明显域偏移，不能直接作为可靠预测工具。

需要注意的是，以上结果来自固定阈值 0.5，未在测试集上调阈值。若业务目标是
提高阳性 recall 或 F1，应在独立验证集上重新选择阈值，而不能使用这些测试集
本身进行调参。

## 7. 输出文件与复现

本次详细 JSON 报告：

`evaluation_ptcr_data_TCR_sets_20260801_131457_433034.json`：该历史 JSON 文件不在当前仓库中。

评估脚本：

[`source/evaluate_plm_triplet_tcr_sets.py`](../source/evaluate_plm_triplet_tcr_sets.py)

默认复现命令：

```bash
CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun/cuda \
LD_LIBRARY_PATH=/home/mjp/miniconda3/envs/unifyimmun/cuda/lib64:/home/mjp/miniconda3/envs/unifyimmun/lib \
NVTE_TORCH_COMPILE=0 PYTHONPATH=source \
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
source/evaluate_plm_triplet_tcr_sets.py \
  --checkpoint trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260731_145558_787556.pt \
  --data-dir data \
  --sets covid,independent,triple \
  --threshold 0.5
```
