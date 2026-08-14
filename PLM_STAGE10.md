# PLM Stage10：Teacher–Student 抗遗忘训练

Stage10 在不修改 Stage8 任何代码、配置或训练行为的前提下，实现“方案一”：从同一个 fold 4、seed 2026 的 Stage3 checkpoint 分别构造冻结 Teacher 和可训练 Student，在保留 Stage8 三头任务及约束设计的同时，用 PH/PT 知识蒸馏抑制灾难性遗忘。

## 1. 模型与数据流

```text
Stage3 checkpoint
├── peptide / HLA / TCR encoders ──> 共享的只读 residue cache
├── binding_model ──> Stage3 Teacher（永久 eval、requires_grad=False）
└── binding_model ──> Stage10 Student（唯一被优化的模型）

同一批缓存张量
├── Teacher ──> PH logits、PT logits
└── Student ──> PH logits、PT logits、PHT logits
```

Teacher 和 Student 都从 Stage3 `stage3_multitask_model` 的 `binding_model.*` 权重初始化，不从 Stage8 初始化。三个 PLM encoder 不参与 Stage10 反向传播。Teacher 不进入优化器、不保存到 lean/full checkpoint，也不能切换到 train 模式。

Stage10 严格校验 Stage3 checkpoint 的格式、任务集合、模型配置、fold、seed、Stage1 上游血缘、三个 encoder 权重和 binding 权重。默认来源为：

```text
trained_model/PLM_Triplet/fold_4/seed_2026/
  triplet_best_20260802_215908_285880.pt
```

## 2. 损失函数

Student 继续使用 Stage8 原有损失，四种 condition 的含义和约束 ramp 均不改变：

```text
L_stage8 = L_supervised + λ_hier(epoch) L_hier + λ_pt(epoch) L_pt_imp
```

Stage10 新增 Teacher→Student 的 PH/PT 蒸馏：

```text
L_keep_PH = T² KL(softmax(z_teacher_PH / T) || softmax(z_student_PH / T))
L_keep_PT = T² KL(softmax(z_teacher_PT / T) || softmax(z_student_PT / T))

L_keep = w_PH L_keep_PH + w_PT L_keep_PT
L_stage10 = L_stage8 + λ_keep L_keep
```

默认值为：

- `distillation_temperature = 1.0`
- `phla_keep_weight = 1.0`
- `ptcr_keep_weight = 1.0`
- `keep_loss_weight = 1.0`

KL 使用 float32 计算，Teacher 概率被 detach，梯度只能进入 Student。

四种 condition 只控制 Stage8 原有约束，蒸馏在四种条件下始终开启：

| condition | hierarchy | PT implication | PH/PT distillation |
|---|---:|---:|---:|
| `baseline` | 关闭 | 关闭 | 开启 |
| `hier_only` | 开启 | 关闭 | 开启 |
| `pt_imp_only` | 关闭 | 开启 | 开启 |
| `full` | 开启 | 开启 | 开启 |

## 3. 验证保留门控与最佳模型

每次验证在完全相同且有序的 batch 上同时计算 Teacher 与 Student。单任务分数为：

```text
task_score = (AUROC + AUPR) / 2
```

训练 epoch 只有同时满足以下硬门控，才有资格成为 best：

```text
Student_PH_score >= Teacher_PH_score - 0.002
Student_PT_score >= Teacher_PT_score - 0.002
```

合格候选仍按 Stage8 原来的 joint score 选择：

```text
joint = 0.6 * PHT_score + 0.2 * PH_score + 0.2 * PT_score
```

并且必须超过当前 best 加 `early_stopping_min_delta`。不合格 epoch 计为未提升。epoch 0 的 Student 与 Teacher 相同，始终作为安全初始 best；如果后续 epoch 都不合格或 joint 未提升，最终 best 保持 epoch 0。

验证还报告 PH/PT KL、正类概率 MAE、0.5 阈值分类一致率、Student−Teacher task score 差值，以及 Student 原有三头/四类型/约束指标。

`primary_success` 同时要求：best 通过保留门控、joint 相对 epoch 0 降幅不超过 0.002、总体有向约束违例率低于 epoch 0。

## 4. 配置、审计与共享缓存

主配置是 `configs/plm_stage10.yaml`，输出目录为：

```text
trained_model/PLM_Stage10/fold_4/seed_2026/<condition>/
swanlog/PLM_Stage10/
```

Stage10 复用由同一 Stage3 SHA 和相同数据指纹生成的 Stage8 residue cache 命名空间：

```text
embedding_cache/plm_stage8/fold_4/seed_2026/<stage3-sha-prefix>/
```

这是输入特征复用，不是复用 Stage8 模型。`cache` 命令会先校验三个 manifest、hidden dimension、Stage3 SHA、数据指纹和 row-sized chunk 布局；缓存有效时直接返回，不重写 shard。只有缓存目录不存在或显式传入 `--overwrite` 才会加载 Stage3 encoder 重新计算。

```bash
./run_stage10.sh audit
./run_stage10.sh cache
```

审计规则与 Stage8 相同：去除同标签重复项、移除同标签 train/validation 重叠、报告冲突/非法标签/序列过滤，并要求清洗后四种标签类型在训练和验证中都存在。审计报告写入 Stage10 输出目录，不修改源 CSV。

## 5. 训练与消融

小规模流水线验证（`--limit-per-type` 仅限 debug，完整审计仍先执行）：

```bash
./run_stage10.sh train \
  --condition baseline \
  --limit-per-type 4 \
  --num-workers 0 \
  --swanlab-mode disabled \
  --device cuda
```

正式运行一个条件：

```bash
./run_stage10.sh train --condition full --device cuda
```

依次运行四种消融：

```bash
./run_stage10.sh ablations --device cuda
```

如需在消融前校验/构建缓存：

```bash
./run_stage10.sh ablations --precompute --device cuda
```

可覆盖 `--stage3-checkpoint`、`--device`、`--num-workers`、`--swanlab-mode`、`--run-timestamp`；恢复训练还可传入 `--resume`。

## 6. Checkpoint 与恢复规则

每个 run 写出：

```text
stage10_best_<timestamp>.pt
stage10_last_<timestamp>.pt
summary_<timestamp>.json
validation_predictions_<timestamp>.csv
```

lean checkpoint 类型为 `stage10_binding_model`，其 `model` 只包含 Student binding 权重。best/last 还记录 Stage3 来源血缘、Teacher policy、Stage10 配置、数据审计、严格训练契约、验证 Teacher/Student/retention 指标、自包含 best Student，以及优化器/调度器/scaler/RNG 状态；没有 Teacher 权重或 PLM encoder 权重。

只有 `artifact_role=last` 能恢复训练。恢复会拒绝：

- Stage3 SHA 不同；
- condition 不同；
- 数据指纹、有效样本数或 debug limit 不同；
- 模型配置或任一 Stage10 超参数不同；
- Teacher policy 不同；
- 非自包含或 best checkpoint。

示例：

```bash
./run_stage10.sh train \
  --condition full \
  --resume trained_model/PLM_Stage10/fold_4/seed_2026/full/stage10_last_<timestamp>.pt
```

## 7. 内部验证与 full 导出

对严格 validation split 重新进行成对 Teacher/Student 评估：

```bash
./run_stage10.sh evaluate \
  --checkpoint trained_model/PLM_Stage10/fold_4/seed_2026/full/stage10_best_<timestamp>.pt
```

JSON 包含 Student 三任务指标、Teacher PH/PT 指标、retention 指标与 eligibility、joint score、PHT threshold、Stage3 血缘和数据指纹。

将已完成的 best lean checkpoint 与其精确 Stage3 encoder 合并：

```bash
./run_stage10.sh export \
  --checkpoint trained_model/PLM_Stage10/fold_4/seed_2026/full/stage10_best_<timestamp>.pt \
  --output trained_model/PLM_Stage10/fold_4/seed_2026/full/stage10_full_<timestamp>.pt
```

full checkpoint 类型为 `stage10_multitask_model`，包含 Stage3 三个 encoder 和 Stage10 Student binding；不包含 Teacher 副本、优化器、调度器、scaler、RNG 或 best 训练副本。

## 8. 外部 pTCR 描述性比较

外部 `peptide,tcr,label` CSV 没有 HLA，因此只评价 pTCR 分支。Stage3、Stage10 和可选 Stage8 在同一批 Stage3 encoder residue 张量和相同行序上推理：

```bash
./run_stage10.sh evaluate-tcr \
  --checkpoint trained_model/PLM_Stage10/fold_4/seed_2026/baseline/stage10_best_<timestamp>.pt \
  --stage8-checkpoint trained_model/PLM_Stage8/fold_4/seed_2026/baseline/stage8_best_<timestamp>.pt \
  --threshold 0.5
```

可省略 `--stage8-checkpoint`，此时只比较 Stage10 与 Stage3。若提供 Stage8，必须引用完全相同的 Stage3 SHA。

输出包括：

- 每个数据集、每个模型的 AUROC、AUPR、Accuracy、F1、MCC、Precision、Recall、Specificity、loss、样本计数和概率摘要；
- `stage10_minus_stage3` 和可选 `stage10_minus_stage8` 的有符号差值；
- 含 dataset、源 CSV 行号、序列、标签及各模型概率的逐行 CSV；
- checkpoint 路径/SHA、Stage3 血缘、固定阈值来源、设备、batch size、过滤和唯一序列计数。

外部评估只做描述性报告：阈值默认并固定为 0.5，不提供测试集选阈值选项，不影响 checkpoint 选择，不设性能通过/失败门槛，也不能据其结果调整 Stage10 超参数。任何后续调参必须另建只使用验证集的实验设计。
