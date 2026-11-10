# Stage1 Encoder

`stage1_encoder` 是独立的 Stage1 实验版本，将三个逐残基 MLP adapter 替换为
Transformer encoder。保留共享 Peptide 分支、pHLA/pTCR 两套 cross-attention、
masked attention pooling 和分类头，以及原有任务交替训练策略。

## 模型

```text
Peptide: ESM-C 特征 [B, Lp, 960] → Peptide encoder → [B, Lp, 256]
HLA:     ESM-C 特征 [B, Lh, 960] → HLA encoder     → [B, Lh, 256]
TCR:     TCR-BERT   [B, Lt, 768] → TCR encoder     → [B, Lt, 256]

pHLA: Peptide → cross-attention(HLA) → pooling → classifier
pTCR: Peptide → cross-attention(TCR) → pooling → classifier
```

每个 encoder 使用 `Linear → LayerNorm → Dropout` 投影，再接三层
`self-attention → feedforward`。每层使用 Pre-LayerNorm 和残差连接，堆叠后
增加最终 LayerNorm。三个 encoder 参数独立，Peptide encoder 在两任务间共享。

| 参数 | 默认值 |
|---|---|
| model_dim | 256 |
| num_attention_heads | 8，每头 32 维 |
| encoder_num_layers | 3 |
| feedforward_dim | 1024，GELU |
| encoder_dropout / attention_dropout | 0.15 / 0.15 |
| classifier_dropout | 0.25 |
| 可训练参数总量 | 9,412,998 |

Self-attention 使用双向注意力和 padding mask，投影后、每层之后以及最终输出的
padding 位置均清零。全空序列报错。PLM 残基特征已经包含序列位置信息，因此
本版本不额外添加位置编码。

模型返回 `logits`、`pooled_embedding`，按需返回 cross-attention 和 pooling 权重。
`return_features=True` 时返回 `plm_embedding` 与 `sequence_encoder_embedding`：
前者拼接两条原始 PLM 序列的 masked mean，后者拼接两条 encoder 输出的
masked mean（512 维）。

## 配置与训练

独立配置为 `configs/plm_stage1_encoder.yaml`。使用与当前 Stage1 相同的
`dataset/PH`、`dataset/PT` 和 `embedding_cache/plm_stage1_ph_pt/fold_1` 残基缓存。
若缓存尚未生成，先使用原 Stage1 配置运行已有的缓存构建入口：

```bash
python source/precompute_plm_embeddings.py --config configs/plm_stage1.yaml --device cuda
```

两个上游 PLM 在本阶段冻结。训练设置沿用 Stage1，包括无 FGM 预热、PH → PT
交替、任务独立早停、联合最佳轮次选择、校准约束和共享分支保留机制。
FGM 作用于当前任务的 Peptide encoder 和 receptor encoder 全部可训练参数。

Encoder 默认学习率为 `encoder_lr=3e-4`，具体分支覆盖为
`shared_peptide_encoder_lr=5e-5`、`hla_encoder_lr=1e-4`、`tcr_encoder_lr=1e-4`。
其他模块的学习率和训练参数与 Stage1 一致。默认 batch size 为 1024，可通过
`--batch-size` 调整；配置中的三组 seeds 为 42、3407、2026。

```bash
# 有限样本调试；seed 9999 用于隔离调试产物
python source/train_plm_stage1_encoder.py \
  --config configs/plm_stage1_encoder.yaml \
  --seed 9999 --limit 256 --batch-size 32 --num-workers 0 \
  --swanlab-mode disabled --no-best-umap

# 正式训练一个 seed
python source/train_plm_stage1_encoder.py \
  --config configs/plm_stage1_encoder.yaml --seed 42 --device cuda

# 三随机种子，顺序训练并汇总
python source/run_plm_stage1_encoder_seeds.py \
  --config configs/plm_stage1_encoder.yaml \
  --seeds 42 3407 2026 --device cuda

# 恢复同一次实验，时间戳从完整 checkpoint 读取
python source/train_plm_stage1_encoder.py \
  --config configs/plm_stage1_encoder.yaml \
  --resume trained_model/PLM_Stage1_Encoder/fold_1/seed_42/last_<timestamp>.pt
```

无需缓存的调试沿用 `--no-cache --limit <N>`，在线 batch size 由
`training.online_batch_size` 控制。CLI 选项与 Stage1 保持一致。

## 评估与导出

默认评估两个任务的 `val_unseen_pair`，自动优先选择最新联合最佳模型。
PH 也支持 `val_unseen_hla`、`val_unseen_peptide`，PT 也支持
`val_unseen_tcr`、`val_unseen_peptide`，通过对应 split 参数选择。

```bash
# 联合最佳模型，评估 PH/PT unseen pair
python source/evaluate_plm_stage1_encoder.py \
  --config configs/plm_stage1_encoder.yaml --seed 42 --device cuda

# 导出的单任务模型，必须选择其对应任务
python source/evaluate_plm_stage1_encoder.py \
  --config configs/plm_stage1_encoder.yaml --tasks ptcr \
  --tcr-split val_unseen_tcr \
  --checkpoint trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_ptcr_<timestamp>.pt

# 手动将完整 checkpoint 拆成两份独立推理模型
python source/export_plm_task_models.py \
  --checkpoint trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_round/best_joint_<timestamp>.pt \
  --output-dir trained_model/PLM_Stage1_Encoder/fold_1/seed_42/exported
```

上面的 `<timestamp>` 是路径占位符，运行时替换为实际文件后缀。
对缓存未覆盖的 split，可添加 `--no-cache --online-batch-size 32` 进行在线 PLM 编码。
评估报告包含实际 checkpoint、split、架构及两任务指标。

训练自动保存完整 `last`、`warmup_last`、联合 `best_joint` 和独立任务最佳模型，
另有校准及鲁棒性最佳任务模型。独立任务 checkpoint 只包含共享 Peptide encoder、
对应 receptor encoder 和该任务预测头。

所有新 checkpoint 标记 `architecture=stage1_encoder`；恢复和评估校验架构与
模型配置。旧 adapter checkpoint 无法作为 encoder 版本的初始权重或恢复状态。

## 实验产物与验证

- 模型与报告：`trained_model/PLM_Stage1_Encoder/fold_<fold>/seed_<seed>/`。
- UMAP：`figures/stage1_encoder/best_model_umap/fold_<fold>/seed_<seed>/<timestamp>/`。
- 高置信错误 CSV：`csv/stage1_encoder/<timestamp>/`。
- SwanLab group：`plm-stage1-encoder-fold{fold}`，实验前缀 `PLM-Stage1-Encoder`。

UMAP 默认开启，特征阶段为 `plm`、`sequence_encoder`、`classifier_input`。
沿用固定 train/validation 样本及参考降维器，支持 `--no-best-umap` 关闭。

以下测试使用合成特征和临时缓存，无需真实 PLM checkpoint：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/stage1_encoder/test_stage1_encoder.py \
  tests/stage1/test_plm_unified.py \
  tests/stage1/test_plm_stage1_improvements.py \
  tests/stage1/test_plm_stage1_umap.py \
  tests/stage1/test_stage1_dimension.py \
  tests/stage15/test_plm_stage15.py
```

覆盖 padding 屏蔽、残基间信息交互、任务冻结、参数分组、FGM 恢复、模型导出
预测一致性、断点恢复、UMAP 阶段命名，以及实际 CLI/CSV/HDF5 训练评估流程。
此前两层版本的实现验证结果为 61 项测试通过，包括默认 PLM 输入维度的 CUDA bf16/fp16
前向及反向传播，以及 Stage1、Stage15 的回归测试。尚未运行完整数据集实验。
后续 Stage2 等阶段接入 encoder checkpoint 不属于本次实现。
