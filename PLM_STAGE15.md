# PLM Stage15：Stage1 + Shared Hyper-PMT

Stage15 是 PH/PT 架构对照实验：数据、冻结 PLM residue cache、Stage1A/1B
训练顺序、FGM、group weighting、retention gate、模型选择和三种子协议均沿用
Stage1；唯一的模型级变化是用 Stage14 的 Shared Hyper-PMT 替换两个独立的
cross-attention，并对更新后的 P+H 或 P+T 全部有效 token 做任务专属池化。

## 架构

- peptide/HLA/TCR 使用独立 residue adapter，peptide adapter 跨任务共享。
- PH hyperedge 为 `{peptide, hla}`，PT hyperedge 为 `{peptide, tcr}`。
- PH/PT 共用一个两层 Hyper-PMT；每层包含实体专属静态变换、排除自身实体的
  跨实体注意力、共享 compatibility MLP、实体门控和共享 post-FFN。
- PH/PT 仅保留各自的 attention pooling 和二分类头。
- 不包含 PHT head、PH prior、hierarchy loss、implication loss或 PLM 解冻。

## 初始化与训练

Stage15 不加载 Stage1/Stage14 下游权重。adapter、Hyper-PMT、pool 和 classifier
均随机初始化；ESM-C/TCR-BERT 保持冻结，并复用
`embedding_cache/plm_stage1_ph_pt/fold_<fold>`。

Hyper-PMT 学习率为 `1e-4`，属于 shared scheduler group，PH/PT step 都会推进。
FGM 只作用于当前任务 adapter。retention gate 冻结 shared 参数时会同时冻结
peptide adapter 和 Hyper-PMT。

## 使用

```bash
./run_stage15.sh train --fold 1 --seed 42
./run_stage15.sh resume /path/to/stage15_last_TIMESTAMP.pt
./run_stage15.sh evaluate --checkpoint /path/to/stage15_best_joint_TIMESTAMP.pt
./run_stage15.sh seeds --fold 1
./run_stage15.sh export --checkpoint /path/to/stage15_best_joint_TIMESTAMP.pt
./run_stage15.sh benchmark \
  --checkpoint /path/to/stage15_best_joint_TIMESTAMP.pt \
  --output-json tests/stage15_benchmark.json \
  --output-markdown tests/stage15_benchmark.md
```

配置位于 `configs/plm_stage15.yaml`。模型、日志、UMAP 和验证错误样本分别写入
`trained_model/PLM_Stage15`、`swanlog/PLM_Stage15`、`figures/stage15` 和
`csv/stage15`，不会覆盖 Stage1 产物。

Checkpoint 类型为 `stage15_training_model`、`stage15_joint_model` 和
`stage15_task_model`。只有 training checkpoint 可用于 resume；task checkpoint
包含目标 adapter/pool/head 和完整 Shared Hyper-PMT 状态。
