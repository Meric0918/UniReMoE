# Stage 5：三任务联合训练

本文件集中整理 Stage 5 的阶段说明。

## 阶段说明

> 来源：`PLM_STAGE5.md`

Stage 5 是与 Stage 3 平行的训练管线。模型参数结构与 Stage 3 完全一致，
但三个基础 PLM 全程冻结，并在同一个优化步中联合训练 pHLA、pTCR 和 PHT。

## 1. 数据与损失

- `data/data_HLA/{train,val}_fold_<fold>.csv` 提供 PH 标签；
- `data/data_TCR/{train,val}_fold_<fold>.csv` 提供 PT 标签；
- `data/data_triple/{train,val,test}_set.csv` 提供 PHT 标签。

一个 joint step 以 triplet batch 为锚，同时取得一个 PH batch 和一个 PT batch。
三元标签不会被当作 PH/PT 标签。每个 epoch 的 joint step 数由 PHT DataLoader
决定；每一步每个任务恰好贡献一个 batch。因此即使
`|D_PH|=1,000,000`、`|D_PT|=100,000`、`|D_PHT|=20,000`，PH 数据集也不会
因为样本总量更大而获得更多的 loss 项或优化步。

阶段 1A 固定训练 5 个 epoch，阶段 1B 固定训练 10 个 epoch；两个阶段
均先在各自 batch 内独立求均值，再使用相同的三任务联合损失：

```text
L_PH  = (1 / B_PH)  * sum_i CE_PH(i)
L_PT  = (1 / B_PT)  * sum_i CE_PT(i)
L_PHT = (1 / B_PHT) * sum_i CE_PHT(i)

L = 0.3 L_PH + 0.2 L_PT + 1.0 L_PHT
```

三个 batch 的样本不会拼接后求一次总平均。Stage 5 不包含额外的一致性损失
或层级损失。

阶段 1A 只使用上述干净联合损失。阶段 1B 仿照 Stage 1B：先对干净联合损失
反向传播，再根据其梯度用 FGM 同时扰动 Peptide、HLA、TCR 三个 Adapter，
计算一次相同权重、相同独立归一化方式的对抗联合损失并反向传播，随后立即
恢复 Adapter。FGM 不直接扰动 PLM、Cross-Attention、Pooling 或 Classifier；
默认 `fgm_epsilon=1.0`。因此 1B 的优化目标为干净联合损失与其 FGM 对抗版本
之和，而不是增加新的任务损失项。

## 2. 初始化与训练

Stage 5 不继承 Stage 1 或 Stage 2 checkpoint。新运行只加载预训练 ESM-C、
TCR-BERT 作为冻结的基础 PLM，Adapter、pHLA/pTCR 下游模块和 PHT 模块均按
随机参数从头训练：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage5.py \
  --config configs/plm_stage5.yaml \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

三个 PLM 不进入 optimizer。三个 Adapter、原 pHLA/pTCR 下游模块和 triplet
下游模块分别使用 `2.5e-5`、`5e-5` 和 `1e-4` 的默认学习率。

`--limit` 只用于调试：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage5.py \
  --config configs/plm_stage5.yaml \
  --seed 9999 \
  --limit 64 \
  --num-workers 0 \
  --device cuda \
  --swanlab-mode disabled
```

## 3. SwanLab 与恢复

默认使用 SwanLab online 模式，日志保存在 `swanlog/`，group 为
`plm-stage5-fold<fold>`。日志包含：

- `L_PHT`、`L_PH`、`L_PT`、干净联合损失及 FGM 对抗联合损失；
- 1B 的 FGM 扰动参数数量、梯度范数和 `epsilon`；
- 三任务 train/validation 指标和联合分数；
- 学习率、梯度范数、数据审计、随机初始化策略和模型指纹。

恢复训练时沿用原 timestamp 和 SwanLab run ID。只允许恢复由本随机初始化
流程创建的 Stage 5 checkpoint；继承 Stage 1/2 的旧 Stage 5 checkpoint 会被拒绝：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage5.py \
  --config configs/plm_stage5.yaml \
  --resume trained_model/PLM_Stage5/fold_1/seed_42/stage5_last_<timestamp>.pt \
  --device cuda
```

## 4. 产物与评估

一个运行生成：

```text
stage5_1a_complete_<timestamp>.pt
stage5_1b_best_joint_<timestamp>.pt
stage5_last_<timestamp>.pt
stage5_best_<timestamp>.pt
```

1B 固定跑满 10 个 epoch，部署模型按以下 validation 联合分数选择：

```text
0.6 * PHT_score + 0.2 * PH_score + 0.2 * PT_score
task_score = (AUROC + AUPR) / 2
```

评估 test：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage5.py \
  --config configs/plm_stage5.yaml \
  --checkpoint \
    trained_model/PLM_Stage5/fold_1/seed_42/stage5_best_<timestamp>.pt \
  --split test \
  --device cuda \
  --hla-shuffle
```

JSON 包含 PHT 指标；预测 CSV 包含 PHT 概率与预测标签。只有 validation
允许使用 `--select-threshold`。

## 5. 测试

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_unified \
  tests.test_plm_stage2 \
  tests.test_plm_triplet \
  tests.test_plm_stage5
```
