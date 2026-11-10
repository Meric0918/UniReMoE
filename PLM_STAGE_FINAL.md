# Stage Final：PT 特征融合与定向一致性联合训练

`stage_final` 将 `stage_MOE_PT` 的残基级 PT→PHT 融合与 `stage_MOE_consis` 的辅助监督、一致性损失组合为独立阶段。复用两阶段的核心算法，不加载旧阶段 checkpoint；冻结 PLM，下游随机初始化。支持联合与顺序训练，不自动启动正式训练。

## 模型与损失

```text
Z_PH = PH_MoE(Cross_PH(P, H))
Z_PT = PT_MoE(Cross_PT(P, T))
U = Cross_PHT(Z_PH, T)
beta = sigmoid(MLP([LN(U), LN(Z_PT), masked_mean(H)]))
U_fused = U + beta * Linear(LN(Z_PT))
Z_PHT = PHT_MoE(U_fused)
```

序列按 peptide 残基对齐，beta 是每个残基一个标量，padding 为零。保留原 PH→PHT 路径及 2 个共享、每分支 2 个专属专家。模型类为 `StageFinalBindingModel`，保留 pair `forward()`、`forward_triplet()`、`forward_all()`、attention/features 选项，输出携带 `fusion_beta`。PHT 训练和评估使用一次 `forward_all()`，同时计算 PH/PT/PHT 三个分类头；单独二元预测不执行融合。

PHT batch 的损失为：

```text
L = pht_weight * BCE_PHT + aux_ph_weight * BCE_PH + aux_pt_weight * BCE_PT
  + ramp(epoch) * (hierarchy_lambda * LH + pt_implication_lambda * LT)
  + balance_lambda * Lbalance_PHT
h(a,b) = relu(a-b-margin)^2
LH = mean_111 h(detach(pPHT), pPH) + mean_000,010 h(pPHT, detach(pPH))
LT = mean_111 h(detach(pPHT), pPT) + mean_000,100 h(pPHT, detach(pPT))
```

各分组单独取均值，空组为零。PHT BCE 按有效原生标签平均，不重采样；辅助监督不重复增加 PH/PT 路由正则。独立 PH/PT batch 仅使用原生标签监督。约束不会强制三个概率相等，也没有概率乘积或 PH&PT→PHT 反向规则。概率操作数的 detach 不冻结分支；PHT 损失仍通过融合与共享路径更新 PT 中间模块。

默认设置见 `configs/plm_stage_final.yaml`，唯一公开设置节为 `stage_final`：

| 设置 | 默认值 |
|---|---:|
| pt_fusion_enabled / fusion_hidden_dim / fusion_bias_init | true / 128 / -2 |
| aux_ph_weight / aux_pt_weight | 0.1 / 0.1 |
| consistency_margin | 0.05 |
| hierarchy_lambda / pt_implication_lambda | 0.05 / 0.03 |
| consistency_warmup_epochs / consistency_ramp_epochs | 3 / 5 |
| learning_rate / router_learning_rate | 0.0002 / 0.0001 |
| balance_lambda | 0 |

辅助 BCE 从第 1 轮启用；一致性项前 3 轮关闭，第 4 轮启用 1/5，第 8 轮达到全权重。顺序训练按完整 round 调度。专家 router 与 fusion.gate 使用 router_learning_rate，融合投影及 LayerNorm 使用普通学习率。沿用 AdamW、5% 学习率 warmup、线性衰减、路由温度前 5 轮 2→1、梯度裁剪和 bf16。

## 数据、缓存与运行

默认读取原始 `dataset/PH`、`dataset/PT`、`dataset/PHT` 聚合 CSV。PHT train/val 要求 `PH,PT,PHT,class_code`，仅允许 `000/010/100/111`；四类子目录不重复读取。完整审计先于调试截取，检查 split 重叠、二元标签冲突及三元组新增二元监督泄漏。失败写入 `audit_failure.json` 并停止，不自动改标、删样本或重新划分。

```bash
./run_stage_final.sh audit --include-benchmarks
./run_stage_final.sh cache --splits train val test --include-benchmarks
./run_stage_final.sh train --output-dir trained_model/PLM_Stage_final/joint_seed42
./run_stage_final.sh train-sequential --output-dir trained_model/PLM_Stage_final/sequential_seed42
./run_stage_final.sh resume --resume trained_model/PLM_Stage_final/joint_seed42/last.pt
./run_stage_final.sh resume-sequential --resume trained_model/PLM_Stage_final/sequential_seed42/last.pt
./run_stage_final.sh evaluate --checkpoint trained_model/PLM_Stage_final/joint_seed42/best.pt
./run_stage_final.sh benchmark --checkpoint trained_model/PLM_Stage_final/joint_seed42/best.pt
```

支持 `STAGE_FINAL_CONFIG`、`--config`、`PYTHON_BIN`，及设备、种子、worker、样本/步数截取与输出目录参数。YAML 默认 SwanLab online，可用 `--swanlab-mode disabled` 关闭。联合模式每步各取三个任务一个 microbatch 再更新，较短任务循环；顺序模式完整遍历 PH→PT→PHT，逐 batch 更新。两者均保留尾 batch，顺序模式的步数限制作用于每个任务。

默认缓存为 `embedding_cache/plm_stage_final`，输出为 `trained_model/PLM_Stage_final`。可显式将 cache_root 指向兼容的 MoE 缓存，校验编码器身份、维度、dtype、长度与覆盖范围；复用时不要运行带 `--overwrite` 的缓存重建。新训练输出目录必须为空。

原始数据此前存在 5 个 PT 配对的辅助监督泄漏。使用 `--config` 指向显式清理的数据副本可以开展训练；本阶段默认仍严格审计原始数据，不自动选择历史实验副本。

## 评估、恢复与消融

三个原生验证集 PR-AUC 的算术平均用于选模；PR-AUC 沿用 precision-recall 曲线梯形积分。各任务验证集按 MCC 选择阈值，测试与 benchmark 使用 checkpoint 阈值。一致性 margin 同样读取 checkpoint。

history、SwanLab 和评估 JSON 同时保存路由、beta、辅助 BCE、约束分量与有效样本数、当前权重以及一致性诊断。beta 按有效残基加权，排除 padding。PHT 诊断包含 111 召回、000/010/100 假阳性率，以及全体/适用/正向/负向组的违反率和平均超出幅度。预测 CSV 额外保存三个概率、可用标签、class_code、违反标记/幅度及 fusion_beta_mean。评估数据可只有 PHT 标签，缺失辅助标签时跳过对应标签感知诊断；二元预测的融合值为空。

checkpoint 标识为 `stage_final_v1`。last.pt 保存 optimizer、scheduler、scaler、RNG、history、最佳模型和早停状态；仅支持本阶段同模式完整 epoch 边界恢复。恢复严格检查融合、损失、数据及缓存契约；旧阶段 checkpoint 不可直接加载。

消融配置：关闭 pt_fusion_enabled 得到 consis 的网络路径；将两项辅助权重及两项一致性权重全部设为零，仅保留 PT 融合与原生监督；也可单独关闭一致性项。关闭融合保留已注册的融合参数，但不参与前向和更新。相同权重的 eval 输出可对照旧模型；不同随机初始化或训练过程不承诺逐位一致。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_final -v
```

测试使用临时合成 HDF5 缓存，不加载 PLM。覆盖融合公式与 padding、旧模型等价性、完整损失及梯度、四类约束、学习率分组、数据审计、尾 batch、联合/顺序精确恢复、CLI 评估和 checkpoint margin。CUDA 可用时执行默认模型尺寸的 bf16 前向、反向和一次优化更新。正式性能收益需另外进行全量训练和消融实验。

实现验证：stage_final 28 项测试、原 MoE 24 项、PT 33 项、consis 14 项全部通过，共 99 项；默认尺寸 CUDA bf16 完整损失更新通过。结果见 [验证记录](reports/stage_final/verification.json)。原始数据审计按预期拒绝 PT 辅助监督的 10 行 heldout 重叠，详见 [审计报告](reports/stage_final/preflight/audit_failure.json)。未启动正式全量训练。
