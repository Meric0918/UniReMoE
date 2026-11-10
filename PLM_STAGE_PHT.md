# Stage PHT

`stage_PHT` 是独立的 PHT 实验，使用三个随机初始化的序列 Encoder，
并沿用 `stage_adapter` PHT 分支的两级 cross-attention、pooling 和分类头。
原路径中的基础 Adapter 与 PHT 残差 Adapter 合并为每条序列一个 Encoder。

```text
Peptide → 冻结 ESM-C    → Peptide Encoder ─┐
HLA     → 冻结 ESM-C    → HLA Encoder ─────┴→ PH cross-attention ─┐
TCR     → 冻结 TCR-BERT → TCR Encoder ───────────────────────────┴→ PHT cross-attention
                                                                  → pooling → classifier
```

三个 Encoder 直接复用 `stage1_encoder.SequenceEncoder`，参数独立：
`Linear → LayerNorm → Dropout → 3 × Pre-LN TransformerEncoderLayer → LayerNorm`。
输入维度分别为 960、960、768，输出维度均为 256，8 个注意力头，
FFN 1024，GELU，Encoder dropout 0.15。保留 padding mask、padding 清零、
空序列检查，不额外添加位置编码。

ESM-C 和 TCR-BERT 加载预训练权重，始终冻结并处于评估模式。
三个 Encoder、两级 cross-attention、pooling、classifier 均随机初始化并训练。
attention/classifier dropout 分别为 0.1/0.2。模型只输出 PHT 二分类结果，
`forward_pht` 返回 `logits [B, 2]` 和 `pooled_embedding [B, 256]`。

## 数据和训练

配置文件为 `configs/plm_stage_PHT.yaml`，训练参数位于独立的 `stage_PHT` 节。
数据、审计、采样、损失和评估直接复用 Stage Adapter 的实现：

- 数据：`dataset/PHT/classes/{PHT_000,PHT_010,PHT_100,PHT_111}/{train,val}.csv`。
- 每个训练 batch 四类等量采样；损失为 `0.5 × mean(CE_111) + 0.5 × mean(CE_negatives)`。
- 验证集保持审计后的原始分布；仅 PHT 标签参与训练。
- 预热 2 epochs，保存验证分数最高的预热权重。正式训练加载该权重，并重建优化器和 scheduler。
- 正式训练最少 5、最多 100 rounds，patience 20，最小提升 `1e-4`。
- 选模分数为固定阈值 0.5 下 AUROC、Accuracy、MCC、F1、AUPR 的平均值。

| 参数 | 默认值 |
|---|---|
| micro batch / 梯度累积 / 评估 batch | 4 / 16 / 32 |
| Encoder 学习率 | `2.5e-5`，沿用原 Adapter 学习率 |
| cross-attention、pooling、classifier 学习率 | `1e-4` |
| 优化器 / weight decay | AdamW / 0.01 |
| scheduler | 与 Stage Adapter 相同的预热及余弦调度，预热比例 0.05 |
| 梯度裁剪 / mixed precision | 1.0 / bf16 |
| fold / seed | 1 / 42 |

模型使用在线 PLM 残基特征。预训练 PLM 路径在配置的 `paths` 节设置。

## 运行

```bash
# 审计数据
./run_stage_PHT.sh audit

# 从随机下游权重开始：自动完成预热和正式训练
./run_stage_PHT.sh train --device cuda

# 只运行预热
./run_stage_PHT.sh warmup --device cuda

# 从已保存的最佳预热权重开始正式训练
./run_stage_PHT.sh train --device cuda --warmup-checkpoint <warmup.pt>

# 从 last 完整恢复
./run_stage_PHT.sh resume <last.pt> --device cuda

# 验证集评估
./run_stage_PHT.sh evaluate --checkpoint <best.pt> --device cuda

# PHT benchmark：报告阈值 0.5 和验证集选定阈值的指标
./run_stage_PHT.sh benchmark --checkpoint <best.pt> --device cuda \
  --output reports/stage_PHT/benchmarks.json

# 小样本调试仍加载真实 PLM；不会更改默认训练配置
./run_stage_PHT.sh train --limit-per-type 8 --num-workers 0 \
  --swanlab-mode disabled --device cuda
```

命令中的 `<warmup.pt>`、`<last.pt>`、`<best.pt>` 需要替换成实际文件路径。
自定义配置使用 `STAGE_PHT_CONFIG=/path/to/config.yaml ./run_stage_PHT.sh train`。
也可直接调用 `source/train_plm_stage_PHT.py`、`source/evaluate_plm_stage_PHT.py`
和 `source/evaluate_plm_stage_PHT_benchmarks.py`。

## 检查点和记录

产物根目录为 `trained_model/PLM_Stage_PHT/fold_<fold>/seed_<seed>/`：

- `warmup/stage_PHT_warmup_<timestamp>.pt`：最佳预热模型。
- `stage_PHT_best_<timestamp>.pt`：最佳正式训练模型。
- `stage_PHT_last_<timestamp>.pt`：最近一轮完整恢复状态。
- `audit_<timestamp>/`：数据审计报告。

检查点包含完整 PLM 和下游模型权重，使用 `architecture=stage_PHT`、
`checkpoint_type=stage_PHT_full_model`、`format_version=1`。
`last` 额外保存优化器、scheduler、scaler、Python/NumPy/PyTorch/CUDA 随机数状态，
以及 best/warmup 配套文件路径及 SHA256。恢复时这些配套文件也必须保留。
恢复和评估校验架构、模型配置及 PLM 指纹；训练恢复还校验训练配置、
初始化 seed/fold、数据指纹和实际样本范围。旧 Adapter/Encoder 检查点不兼容。
小样本训练后从预热继续时，需要传入相同的 `--limit-per-type`；
从 last 恢复和验证集评估时自动读取保存的样本范围。

SwanLab 使用 `PLM-Stage-PHT` 前缀、`plm-stage-pht-fold{fold}` 分组，
日志目录为 `swanlog/PLM_Stage_PHT`，指标前缀为 `stage_PHT/`。

## 实现验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/stage_PHT tests/stage_adapter tests/stage1_encoder
```

新增测试使用合成 PLM 特征及临时 CSV，覆盖 padding 屏蔽、全部下游模块更新、
PLM 冻结、完整参数分组、预热权重加载、含 dropout 的精确断点恢复、
架构/配置/PLM 指纹校验及训练、验证、benchmark 的 CLI 流程。
正式全量训练需要另行启动。
