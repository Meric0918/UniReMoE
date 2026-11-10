# Stage MoE Consis：四类三元组完整一致性

`stage_MOE_consis` 复用 Stage MoE v1 的网络结构：冻结 ESM-C/TCR-BERT 残基缓存、三个 SequenceEncoder、PH/PT/PHT cross-attention、两个共享专家及各分支两个专属专家。下游随机初始化，不载入 Stage MoE 或 Stage 8 checkpoint。参数量不增加，默认下游共 11,224,978 个参数。

## 数据与损失

使用 `dataset/PH`、`dataset/PT`、`dataset/PHT` 中的聚合 CSV，沿用原划分与采样。四类子目录是聚合 PHT 的视图，不重复读入。PHT train/val 必须含 `PH,PT,PHT,class_code`，类型仅允许 `000/010/100/111`。PH/PT 独立数据仍只提供本任务监督。

三元组使用一次 `forward_all()` 得到三个单 logit，令其 Sigmoid 为 `pH,pT,p3`，`SG` 为 `detach()`，`h(a,b)=relu(a-b-m)^2`：

```text
LH = mean_111 h(SG(p3),pH) + mean_000,010 h(p3,SG(pH))
LT = mean_111 h(SG(p3),pT) + mean_000,100 h(p3,SG(pT))
Ltriple = pht_weight * BCE_PHT + aux_ph_weight * BCE_PH + aux_pt_weight * BCE_PT
        + r(epoch) * (hierarchy_lambda * LH + pt_implication_lambda * LT)
        + balance_lambda * Lbalance_PHT
```

每项在对应有效标签组内取均值，空组为零。PHT BCE 对 batch 直接平均，保留训练集约 1:3 正负比，不额外重加权或重采样。PH/PT 独立 batch 的任务权重仍默认 1。PHT batch 的门控均衡只统计 PHT 门控，辅助分类不重复增加 PH/PT 门控正则。

| 类别 | hierarchy | PT implication |
|---|---|---|
| 111 | 直接提高不足的 PH 输出 | 直接提高不足的 PT 输出 |
| 000 | 直接降低过高的 PHT 输出 | 直接降低过高的 PHT 输出 |
| 100 | 关闭 | 直接降低过高的 PHT 输出 |
| 010 | 直接降低过高的 PHT 输出 | 关闭 |

只惩罚超过容差的不等式违反。共享专家及 PH→PHT 特征路径仍接收梯度，停止某个概率操作数的梯度不代表冻结对应分支。没有概率乘积、三个输出相等或 `PH & PT => PHT` 约束；输出概率关系是软训练先验。

默认参数位于 `configs/plm_stage_MOE_consis.yaml`：

| 参数 | 默认值 |
|---|---:|
| aux_ph_weight / aux_pt_weight | 0.1 / 0.1 |
| consistency_margin | 0.05 |
| hierarchy_lambda / pt_implication_lambda | 0.05 / 0.03 |
| consistency_warmup_epochs | 3 |
| consistency_ramp_epochs | 5 |
| balance_lambda | 0 |

辅助 BCE 从第 1 轮开启；`r(e)=clip((e-3)/5,0,1)`，第 1–3 轮约束关闭，第 4 轮为最大权重的 1/5，第 8 轮达到最大。权重可设为零以进行对照。联合模式的 epoch 与顺序模式的 round 各自使用此调度。

## 运行与恢复

```bash
# 完整审计（也检查三元组新增监督与原二元 heldout 配对的重叠）
./run_stage_MOE_consis.sh audit --include-benchmarks

# 新建独立缓存
./run_stage_MOE_consis.sh cache --splits train val test --include-benchmarks

# 联合训练：每步分别累积 PH、PT、PHT microbatch 后更新
./run_stage_MOE_consis.sh train

# 顺序训练：每轮 PH → PT → PHT，各自完整遍历、逐 batch 更新
./run_stage_MOE_consis.sh train-sequential

# 必须使用同阶段、同训练模式的 last.pt
./run_stage_MOE_consis.sh resume --resume PATH_TO_JOINT_LAST.pt
./run_stage_MOE_consis.sh resume-sequential --resume PATH_TO_SEQUENTIAL_LAST.pt

./run_stage_MOE_consis.sh evaluate --checkpoint PATH_TO_BEST.pt
./run_stage_MOE_consis.sh benchmark --checkpoint PATH_TO_BEST.pt
```

支持 `STAGE_MOE_CONSIS_CONFIG`、`--config`、`PYTHON_BIN`，及原 Stage MoE 的设备、worker、调试截取、输出目录和 SwanLab 参数。默认 SwanLab online；本地测试使用 `--swanlab-mode disabled`。顺序模式的 `--max-steps-per-epoch` 分别限制每个任务。

缓存默认 `embedding_cache/plm_stage_MOE_consis`，产物默认 `trained_model/PLM_Stage_MOE_consis/seed_<seed>/<action>_<timestamp>`。可以把新配置的 `paths.cache_root` 指向已有 `embedding_cache/plm_stage_MOE`，直接读取兼容的原缓存，无需重新计算。缓存验证编码器指纹、维度、dtype、最大长度与序列覆盖。`cache` 默认拒绝覆盖已有缓存；仅显式 `--overwrite` 可重建，复用缓存时应特别注意该参数会修改所指目录。

独立 checkpoint 标识为 `stage_MOE_consis_v1`。恢复校验网络、训练模式、所有损失设置、数据及缓存身份；保存 optimizer、scheduler、scaler、随机状态和最佳模型。只承诺完整 epoch 边界恢复。旧 MoE/Stage 8 checkpoint 不能作为本阶段恢复点。

## 审计、评估与日志

完整数据审计先于调试截取，并检查三元组派生的 `(peptide,HLA)`、`(peptide,TCR)` 标签与独立二元监督的冲突，以及这些新增训练监督是否出现在二元 val/test/benchmark。发现冲突或泄漏即报错，不改标、不自动过滤或重新划分。二元 heldout 文件的检查不参与模型选优或阈值拟合。PHT 自身按完整三元组检查 split 重叠，不因其不同三元组共享一个二元配对而报错。

审计失败保存 `audit_failure.json`；泄漏报告按二元任务统计 heldout 文件行数，包含最多 20 条来源示例。同一配对若同时出现在 test 和 benchmark，会分别计数。成功报告为 `audit.json`。

验证选优仍为 PH/PT/PHT 原生验证集 PR-AUC 算术平均，PR-AUC 沿用 `precision_recall_curve + auc`。三个阈值分别在验证集最大化 MCC，测试读取 checkpoint 阈值，不重新拟合。评估一致性容差读取 checkpoint 中的训练配置。

新增 PHT 诊断包含 `111` 召回率、`000/010/100` 各自假阳性率，以及两个约束在全体、适用标签组、正向组和负向组的违反率、平均超出 margin 的幅度。平均超出幅度包含未违反样本的零值。无有效样本的诊断记为 null，一致性指标不参与选优。

测试及 benchmark 可只有 PHT 标签；仍输出三个概率和全体概率违反情况，缺失辅助标签时跳过无法计算的标签感知诊断。已提供标签必须合法；train/val 不允许缺失辅助标签。

逐样本 CSV 保存三个概率、可用标签、class_code、违反标记及幅度；history/SwanLab 保存原生及辅助 BCE、约束正负分量、对应有效样本数、当前约束权重、监督和加权总损失、路由统计。每个分量使用自己的有效样本数汇总，训练样本量不因三个标签重复计数。

## 验证与现有数据状态

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_MOE_consis -v
```

测试使用临时真实 HDF5 缓存，不加载基础 PLM，覆盖四类损失手算和梯度方向、warmup/ramp、辅助标签与泄漏审计、两种训练的精确恢复、模式拒绝、CLI 评估及缺失辅助标签的评估行为。

实现验证结果：新增 14 项测试及原 Stage MoE 的 24 项回归测试全部通过。

实现期间对真实数据的审计发现新增 PT 监督与二元 heldout 有 5 个配对重叠，在测试集与 unseen-TCR benchmark 中合计 10 行，PH 为 0。示例：`dataset/PHT/train.csv:2484` 的 PT 配对出现在 `dataset/PT/test.csv:187751`。因此当前默认数据会被新阶段拒绝训练，需要先明确处理这些配对的划分；实现没有修改源数据。完整来源见 [审计报告](reports/stage_MOE_consis/audit_failure.json)。

另以 RTX 4090、默认模型尺寸完成 bf16 三头前向、完整损失反向及 AdamW 更新，所有参数梯度与更新后参数均有限。这不代表正式全量训练或性能收益已验证。
