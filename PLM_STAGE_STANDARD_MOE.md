# Stage Standard MoE 消融实验

`stage_standard_moe` 用方案 A 的分支独立专家池、线性门控和 Top-2 路由替换 `stage_final` 的 MoE，比较两种 MoE 设计在同一 Full 网络中的表现。PH→PHT、PT→PHT 残差融合、辅助监督和定向一致性损失全部保留。本实验采用 seed 42，按 PH→PT→PHT 交替训练。

这是 [Standard MoE 消融对比](<Standard MoE消融对比.md>) 中方案 A 的 **MoE 子模块替换实验**。文档中同时移除跨分支信息注入和一致性损失的完整 Standard MoE 基线不属于本次实验。

## 对照设置

| 项目 | stage_final | stage_standard_moe |
|---|---|---|
| 专家池 | 2 个共享专家，每分支 2 个专属专家 | 每分支独立 4 个专家 |
| 独立专家模块总数 | 8 | 12 |
| 每个样本每分支激活专家数 | 4 | 2 |
| 门控输入 | 当前交互特征及输入序列摘要的拼接 | 当前 MoE 输入序列的 masked mean |
| 门控网络 | 两层 MLP | Linear |
| 路由 | 稠密 Softmax | Top-2 后归一化 |
| PH→PHT 和 PT 残差融合 | 保留 | 保留 |
| 辅助监督和一致性损失 | 保留 | 保留 |
| 总参数量 | 11,390,355 | 12,115,475 |
| 专家参数量 | 2,103,296 | 3,154,944 |

参数量对应默认配置。两组每分支均有 4 个候选专家，但共享方式、门控上下文、门控网络和激活数量一起变化，因此结果用于比较 MoE 设计整体，不能单独归因于专家共享。单种子实验不提供跨种子的显著性结论。

## 模型与路由

```text
Z_PH = StandardMoE_PH(Cross_PH(P, H))
Z_PT = StandardMoE_PT(Cross_PT(P, T))
U = Cross_PHT(Z_PH, T)
beta = sigmoid(MLP([LN(U), LN(Z_PT), masked_mean(H)]))
U_fused = U + beta * Linear(LN(Z_PT))
Z_PHT = StandardMoE_PHT(U_fused)
```

每个 Standard MoE 有 4 个独立的 `Linear(256,512) → GELU → Linear(512,256)` 专家。对进入当前 MoE 的序列 `H`，先计算 `Linear(masked_mean(H))/temperature`，选取最大的 2 个 logits 并做 Softmax。这等价于对全部 Softmax 概率取 Top-2 后重新归一化。每个样本的所有 token 使用同一组权重。

仅将被选中的样本送入对应专家，输出为 `Mask(H + Dropout(Σ w_i E_i(LN(H))))`。门控及累加使用 float32；输出 padding 为零。PHT 门控仅使用 `U_fused` 的序列摘要。路由权重形状为 `[batch, 4]`，每行 2 个非零值，和为 1。

模型类为 `StageStandardMoEBindingModel`。保留 pair `forward()`、`forward_triplet()`、`forward_all()` 和 attention/features 接口；输出包含 logits、routing、attention、pooling_weights、features 和 fusion_beta。PHT 训练与评估调用 `forward_all()`，同时计算三个分类头。

## 数据与交替训练

配置为 `configs/plm_stage_standard_moe.yaml`，公开设置节为 `stage_standard_moe`。默认使用 Full 实际运行配置中的数据和缓存：

- 数据：`training_logs/stage_final_20261110_160519/data`，其中 PHT 训练集已排除 5 条造成辅助监督泄漏的记录，原始 72,192 行变为 72,187 行。
- 缓存：`embedding_cache/plm_stage_MOE`，复用已有 ESM-C 300M 和 TCR-BERT 残基特征，检查模型身份、维度、dtype、长度与序列覆盖。
- Full 对照：`trained_model/PLM_Stage_final/seed_42/sequential_20261110_160519`。
- 新实验输出：`trained_model/PLM_Stage_standard_moe`，与 Full 分开保存。

数据审计沿用 Full 的完整监督与泄漏检查，先审计再截取调试样本。若数据副本或缓存不在本机，应通过显式配置恢复同一数据版本；不自动切回原始数据或重新清理。

冻结 PLM，下游随机初始化。每轮依次完整遍历 PH、PT、PHT，保留尾 batch，每个 batch 执行一次优化更新。三次遍历完成后验证、选模和早停。`max_steps_per_epoch` 在本阶段对每个任务分别限步。

| 设置 | 默认值 |
|---|---:|
| seed / num_experts / top_k | 42 / 4 / 2 |
| model_dim / expert_hidden_dim | 256 / 512 |
| sequence_encoder_num_layers / attention heads | 3 / 8 |
| batch_size / eval_batch_size | 32 / 256 |
| max_epochs（完整 rounds） / patience | 100 / 10 |
| learning_rate / router_learning_rate | 0.0002 / 0.0001 |
| weight_decay / gradient_clip_norm / dropout | 0.01 / 1.0 / 0.1 |
| warmup_ratio / mixed_precision | 0.05 / bf16 |
| temperature | 前 5 轮从 2 降至 1 |
| aux_ph_weight / aux_pt_weight | 0.1 / 0.1 |
| hierarchy_lambda / pt_implication_lambda | 0.05 / 0.03 |
| consistency_margin | 0.05 |
| consistency_warmup_epochs / consistency_ramp_epochs | 3 / 5 |
| balance_lambda | 0 |

损失、标签 mask、各分组的归一化及一致性项的定向 detach 与 [Stage Final](PLM_STAGE_FINAL.md) 相同。辅助 BCE 从首轮生效；一致性约束前 3 轮关闭，第 4 轮为目标权重的 1/5，第 8 轮达到完整权重。专家 router 和 fusion.gate 使用 router_learning_rate，其余模块使用 learning_rate。

## 运行和评估

```bash
# 审计实际训练和验证数据，并检查 benchmark
./run_stage_standard_moe.sh audit --splits train val --include-benchmarks \
  --output-dir reports/stage_standard_moe/preflight

# 正式交替训练
./run_stage_standard_moe.sh train-sequential \
  --output-dir trained_model/PLM_Stage_standard_moe/seed_42/sequential

# 完整 round 边界恢复
./run_stage_standard_moe.sh resume-sequential \
  --resume trained_model/PLM_Stage_standard_moe/seed_42/sequential/last.pt

# 测试与各 benchmark
./run_stage_standard_moe.sh evaluate \
  --checkpoint trained_model/PLM_Stage_standard_moe/seed_42/sequential/best.pt \
  --output-dir reports/stage_standard_moe/seed42_test
./run_stage_standard_moe.sh benchmark \
  --checkpoint trained_model/PLM_Stage_standard_moe/seed_42/sequential/best.pt \
  --output-dir reports/stage_standard_moe/seed42_benchmarks
```

入口还支持 `cache`，默认共享缓存无需重建。可使用 `STAGE_STANDARD_MOE_CONFIG`、`--config`、`PYTHON_BIN`，以及设备、种子、worker、调试样本和步数限制等参数。YAML 默认 SwanLab online；本地验证使用 `--swanlab-mode disabled`。SwanLab 分组为 `plm-stage_standard_moe`。

三个任务验证集 PR-AUC 的均值用于选模，PR-AUC 沿用 Full 的曲线梯形积分口径。每个任务仅在验证集按 MCC 选择阈值，测试及 benchmark 使用 checkpoint 保存的阈值；一致性 margin 同样读取 checkpoint。

`metrics.json` 和 `metrics.md` 保存 AUROC、PR-AUC、Accuracy、F1、MCC、参数量，以及 PHT 四类召回或假阳性率、一致性违反率和融合统计。预测 CSV 包含三个分类概率、有效标签、fusion_beta_mean 和稀疏路由权重。history 与 SwanLab 记录专家选择频率、平均权重、路由熵、耗时、峰值显存及各项损失；不再使用 shared/private 比例。

checkpoint 标识为 `stage_standard_moe_v1`，仅支持 sequential。last.pt 包含优化器、学习率调度、GradScaler、RNG、history、最佳模型和早停状态。恢复检查架构、专家数、Top-k、门控定义、损失、数据和缓存契约；不加载 Full 或其他阶段 checkpoint。

完成两组训练后，按相同 test 和 benchmark 文件比较各任务指标，并同时报告最优轮次、参数量、训练耗时和显存。新模型保留 Full 的所有任务协同设置，结果表中应继续将这些项标为启用。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_standard_moe -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_final -v
```

测试使用临时合成 HDF5 缓存，不加载 PLM。覆盖样本级稀疏派发、公式与梯度、padding、Full 周边拓扑等价性、融合后门控输入、完整损失、交替顺序、尾 batch、精确恢复、checkpoint 拒绝条件、CLI 评估及 SwanLab 字段。CUDA 可用时执行默认模型尺寸的 bf16 前向、反向和一次优化更新。

实现验证结果见 [验证记录](reports/stage_standard_moe/verification.md)：20 项新阶段测试、28 项 Full 回归测试全部通过，包含 CUDA bf16 更新；实际数据指纹与 Full 相同，所需序列的共享缓存缺失数为零。
