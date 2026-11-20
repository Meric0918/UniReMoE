# Stage Final Gated：样本级迁移与监督门控

独立架构 `stage_final_gated_v1` 在 Stage Final 的残基级 PT→PHT 融合上增加样本迁移门，并用另一个独立样本门重加权 PH/PT 辅助监督和两项一致性损失。两个门可以同时开启；是否改善 unseen-HLA 泛化，需要匹配预算的正式实验验证。

## 模型

保留冻结 PLM 缓存、三个 SequenceEncoder、共享／专属专家、PH→PHT 路径，以及原残基融合模块。令 `U = Cross_PHT(Z_PH, T)`，上下文为：

```text
C = concat(masked_mean(LN(U)), masked_mean(LN(Z_PT)), masked_mean(H))
gF = sigmoid(MLP_F(C))                         # [B]
gC = 0.05 + 0.95 * sigmoid(MLP_C(detach(C)))   # [B]
beta_effective = gF[:, None] * beta_residue   # [B, L_peptide]
U_fused = U + beta_effective[..., None] * Linear(LN(Z_PT))
```

两个 MLP 均为 `3d → 128 → 1`，激活函数为 GELU。输出层权重初始化为零；迁移门偏置为 `logit(0.9)`，约束门偏置为零，因此初始 `gF=0.9`、`gC=0.525`。有效残基外的融合系数为零。约束门输入复用残基融合的 LayerNorm，随后整体 detach；权重生成路径不会反向修改主干。监督损失自身仍正常更新主干。

只输入 PH/PT 二元样本时不计算新门。PHT 的 `forward_all()` 返回三个分类头；`forward_triplet()` 返回 PHT 分类头。迁移门通过 PHT 目标学习，约束门通过组内相对加权目标学习。模型不使用真值标签作为门控输入。

## 组内归一化监督

对每个有效监督组 `S`：

```text
w_i = gC_i / mean(gC[S])
weighted_loss = mean(w_i * raw_loss_i)
dispersion = mean((w_i - 1)^2)
term = original_coefficient * (weighted_loss + 0.001 * dispersion)
```

归一化分母保留梯度。约束门是相对权重，统一缩小门值无法整体关闭监督；正下限限制单个样本被完全排除。它仍可能偏向容易满足约束的样本，需结合 raw/weighted loss、分类效果和门值分布解释。

六个组分别归一化：

| 监督组 | 有效样本 | 原系数 |
|---|---|---:|
| PH 辅助 BCE | 有效 PH 标签 | 0.1 |
| PT 辅助 BCE | 有效 PT 标签 | 0.1 |
| hierarchy positive | `111` | `r(epoch) × 0.05` |
| hierarchy negative | `000/010` | `r(epoch) × 0.05` |
| PT implication positive | `111` | `r(epoch) × 0.03` |
| PT implication negative | `000/100` | `r(epoch) × 0.03` |

一致性保留原来的方向和概率 detach：正类用 `relu(detach(pPHT)-pPair-margin)^2`，负类用 `relu(pPHT-detach(pPair)-margin)^2`。margin 为 0.05；`r(e)=clip((e-3)/5,0,1)`。辅助 BCE 从首轮启用。PHT 原生 BCE 和独立 PH/PT 原生 BCE 不受约束门控制。

空组为零，单样本组的权重为一。关闭某一损失系数时，该组也不贡献门控正则或门控参数梯度。关闭所有相关系数时，AdamW 不会因为零梯度而对约束门执行权重衰减。

## 配置、运行与对照

默认配置为 `configs/plm_stage_final_gated.yaml`。使用 Final 实际训练的已审计数据快照 `training_logs/stage_final_20261110_160519/data`，读取兼容的 `embedding_cache/plm_stage_MOE` 缓存。该快照的 PHT 训练集为 72,185 条，避免原始数据中已发现的辅助监督 heldout 重叠。训练前仍执行完整审计，不自动过滤或修改数据。

```bash
./run_stage_final_gated.sh audit --include-benchmarks

# 顺序训练：每轮完整 PH → PT → PHT
./run_stage_final_gated.sh train-sequential

# 联合训练：每步累积 PH、PT、PHT 三个 microbatch 后更新
./run_stage_final_gated.sh train

./run_stage_final_gated.sh resume-sequential --resume PATH_TO_SEQUENTIAL_LAST.pt
./run_stage_final_gated.sh resume --resume PATH_TO_JOINT_LAST.pt
./run_stage_final_gated.sh evaluate --checkpoint PATH_TO_BEST.pt
./run_stage_final_gated.sh benchmark --checkpoint PATH_TO_BEST.pt
```

支持 `STAGE_FINAL_GATED_CONFIG`、`--config`、`PYTHON_BIN` 和原有设备／采样截取／SwanLab 参数。默认 SwanLab online；本地验证使用 `--swanlab-mode disabled`。`cache` 操作仅在显式 `--overwrite` 时重建；默认缓存路径与旧阶段共享，通常直接读取即可。

新增配置项：

| 参数 | 默认值 |
|---|---:|
| sample_fusion_enabled | true |
| sample_constraint_enabled | true |
| sample_gate_hidden_dim | 128 |
| sample_fusion_init | 0.9 |
| constraint_gate_floor | 0.05 |
| constraint_gate_regularization | 0.001 |

两个新门使用 `router_learning_rate=1e-4`，其余组件沿用原配置。禁用门时其参数仍注册，但不参与前向和更新。

| 消融配置前缀 `configs/` | 样本迁移门 | 样本约束门 | 输出子目录 |
|---|---|---|---|
| plm_stage_final_gated_no_sample_gates.yaml | 关 | 关 | no_sample_gates |
| plm_stage_final_gated_fusion_only.yaml | 开 | 关 | fusion_only |
| plm_stage_final_gated_constraint_only.yaml | 关 | 开 | constraint_only |
| plm_stage_final_gated.yaml | 开 | 开 | full |

四组均保留原残基 β、辅助 BCE 和一致性损失。用 `--config` 选择配置，统一数据、seed、训练方式和预算。两个新门关闭并使用相同基础权重时，eval 输出恢复原 Final。不同训练过程不保证逐位一致。

新 checkpoint 只由本阶段入口加载，模型契约包含门控开关和尺寸等设置；恢复还严格检查全部训练设置、数据／缓存身份和训练方式。旧 Final 继续使用原入口评估，不自动迁移或覆盖。模型选择仍使用三个原生验证任务的平均 AUPR，阈值仍由验证集选择。

## 可观测性

PHT 输出与预测 CSV 包含：

- `sample_fusion_gate`、`sample_constraint_gate`：样本门值；二元预测为空。
- `residue_beta`／CSV `residue_beta_mean`：原始残基门。
- `fusion_beta`／CSV `fusion_beta_mean`：实际相乘后的融合系数。

history、SwanLab、评估 JSON/Markdown 同时记录总体及 `000/010/100/111` 门值统计。均值与标准差按样本精确累计；分位数使用 256 个固定直方图箱估计，箱宽为 `1/256`；近零／近一定义为 `<0.05`／`>0.95`。约束门下限为 0.05，因此近零率通常为零，判断权重集中程度应结合分位数及归一化权重。

`loss_details` 包含各组 raw/weighted loss、权重离散程度和权重统计；组内最小／最大权重的 epoch 汇总是按有效样本数加权的 batch 统计平均，不是整个 epoch 的极值。归一化在每个 batch 内独立执行，因此这些诊断随 batch 划分变化。评估损失使用 checkpoint 保存的监督设置和 epoch，不参与选模；门值和预测本身不依赖辅助标签是否齐全。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_final_gated -v
```

覆盖公式、padding、禁用门与原 Final 等价、组内手算和缩放不变性、定向梯度、权重生成路径 detach、关闭损失项后的参数隔离、两种训练模式的精确恢复、审计、缺失辅助标签、CLI 和日志，以及默认尺寸 bf16 更新。验证记录位于 `reports/stage_final_gated/verification.json`；本次实现不启动正式训练。
