# Stage Adapter（PHT-only）训练说明

## 从随机权重开始训练

使用独立配置直接启动，程序自动依次完成 StageAdapterA warmup 和
StageAdapterB，并保存 warmup、best、last 检查点：

```bash
STAGE_ADAPTER_CONFIG=configs/plm_stage_adapter_random.yaml ./run_stage_adapter.sh train
```

该配置使用 `stage_adapter.initialization: random_binding`，不要求
Stage1/Stage2 任务检查点或其 SHA256。三个 PLM 从本地目录加载预训练权重
并保持冻结；共享 P/H/T Adapter 和全部 PHT 专属模块从随机权重开始训练。
模型权重、结构配置和 tokenizer 文件需存在于 `esmc_model`、`tcr_bert_model`
目录。

训练损失仍只使用 PHT 标签。PH/PT 独立预测头及未用于残基特征的
PLM 预测头没有相应训练目标，保持随机初始化，不作为已经训练好的
PH/PT 专家。随机模式的共享 Adapter 会更新，检查点冻结校验只覆盖
未参与训练的 PH/PT 独立预测头。

仅执行 warmup、恢复训练、验证或 benchmark 时，继续指定同一个配置：

```bash
STAGE_ADAPTER_CONFIG=configs/plm_stage_adapter_random.yaml ./run_stage_adapter.sh warmup
STAGE_ADAPTER_CONFIG=configs/plm_stage_adapter_random.yaml ./run_stage_adapter.sh resume <last.pt>
STAGE_ADAPTER_CONFIG=configs/plm_stage_adapter_random.yaml ./run_stage_adapter.sh evaluate --checkpoint <best.pt>
```

也可用 `./run_stage_adapter.sh train --from-scratch` 将现有配置的初始化
模式覆盖为 `random_binding`；后续恢复和评估时需将配置中的
`initialization` 也设为 `random_binding`。独立配置将模型输出放在
`trained_model/PLM_StageAdapter_Random`。

若需要连三个 PLM 也从随机权重开始，可单独设置 `initialization: random`。
该模式会训练三个 PLM 的 backbone，通常需要更多显存。原有任务检查点
初始化模式默认值为 `checkpoint`，使用流程如下。

## 训练边界

`stage_adapter` 使用显式任务路由，不训练 Router。PH、PT 和 PHT 分别通过
`forward_ph`、`forward_pt`、`forward_pht` 进入固定分支。

| 组件 | 初始化 | 训练状态 |
|---|---|---|
| ESM-C / TCR-BERT | 本地预训练模型 | 冻结 |
| Stage1 Peptide/HLA/TCR Adapter | Stage1 `last_20260831_174042_678934.pt` | 冻结 |
| Stage1 PH/PT Cross-Attention、Pooling、Classifier | Stage1 `last_20260831_174042_678934.pt` | 冻结 |
| PHT P/H/T Residual Adapter | 随机初始化 | 可训练 |
| PHT PH Cross-Attention、PHT Cross-Attention | 随机初始化 | 可训练 |
| PHT Pooling、Classifier | 随机初始化 | 可训练 |

训练过程不构造 PH/PT loss，不执行 PH/PT replay、蒸馏、约束或性能验证。
PH/PT 分支只通过参数冻结和 SHA256 校验和保护。训练、验证、早停和
benchmark 输出均只包含 PHT 指标。

## 数据与目标

数据沿用 Stage8 的四类目录：`PHT_000`、`PHT_010`、`PHT_100`、
`PHT_111`。训练 batch 四类等量采样，损失为：

```text
0.5 * mean(CE on 111) + 0.5 * mean(CE on 000/010/100)
```

验证集保持原始分布。模型选择分数为 PHT 的 AUROC、Accuracy、MCC、F1、
AUPR 五项平均值。

## 生命周期

- StageAdapterA：固定 2 epochs，训练全部 PHT 专属模块，保存最优 warmup。
- StageAdapterB：最少 5、最多 100 rounds，patience 20，只训练 PHT 专属模块。
- 检查点：V2 完整模型 `warmup`、`best`、`last`；`last` 额外保存优化器、
  scheduler、scaler 和随机数状态，可严格恢复。

## 命令

先审计数据：

```bash
./run_stage_adapter.sh audit
```

运行两轮 warmup：

```bash
./run_stage_adapter.sh warmup \
  --stage1-checkpoint trained_model/PLM_Unified_PH_PT/fold_1/seed_42/last_20260831_174042_678934.pt
```

从 warmup 开始 StageAdapterB：

```bash
./run_stage_adapter.sh train \
  --warmup-checkpoint trained_model/PLM_StageAdapter/fold_1/seed_42/warmup/stage_adapter_warmup_<timestamp>.pt
```

恢复中断训练：

```bash
./run_stage_adapter.sh resume \
  trained_model/PLM_StageAdapter/fold_1/seed_42/stage_adapter_last_<timestamp>.pt
```

验证和测试集 benchmark：

```bash
./run_stage_adapter.sh evaluate --checkpoint <best-or-last.pt>
./run_stage_adapter.sh benchmark --checkpoint <best-or-last.pt> \
  --output trained_model/PLM_StageAdapter/benchmark.json
```

正式训练默认开启 SwanLab。调试时可传 `--swanlab-mode disabled`；也可用
`--limit-per-type N` 做四类等量的小规模贯通测试。限量信息写入检查点，恢复和
评估时会自动复用。
