# Stage MOE 顺序训练模型：PH / PT / PHT Benchmark 测试报告

- 测试日期：2026-10-28 16:02（北京时间）
- 模型：Stage MOE，顺序训练，seed 42，第 22 轮最佳检查点。
- 检查点：[best.pt](../../../trained_model/PLM_Stage_MOE/seed_42/sequential_online_20261027_193829/best.pt)。
- 检查点 SHA-256：`fbdacf05b840e32238ab4f27ecef355c34bce16990619e3b0d259dd922b324c0`。
- 检查点按三个任务验证集 AUPR 的算术平均值选择，最佳值为 0.905633；以下结果全部来自单独的 benchmark 测试文件。
- 数据范围：PH 3 个文件 / 110,576 行；PT 3 个文件 / 298,034 行；PHT 4 个文件 / 42,146 行。共 10 个文件、450,756 行（按文件计数）。
- 推理：CUDA、bf16 混合精度，batch size 256，使用与检查点编码器指纹匹配的冻结 PLM 残基表示缓存。

## 评估方法

- 仅读取每个任务自己的监督标签：PH/PT 使用 CSV 的 `label`，PHT 使用 `PHT`；PHT 文件的 PH/PT 标签不参与 PHT 测试。
- 主表使用检查点保存的验证集阈值，分别为 PH **0.09807931631803513**、PT **0.6540113687515259**、PHT **6.893610589031596e-7**。阈值在验证集按 MCC 选择，没有根据测试集重新调整。预测概率大于或等于阈值时判为正。
- 同时报告固定阈值 0.5 的对照；两组结果使用同一批预测概率。AUROC、AUPR 和 Brier 不依赖分类阈值；本报告的 ECE 固定以 0.5 确定预测类别，不随上述阈值对照变化。
- AUPR 按仓库实现，以 precision-recall 曲线计算梯形面积（不是 average precision）。Brier 和 ECE 越小越好；ECE 使用 15 个等频分箱，比较预测类别置信度和准确率。
- 这 10 个测试集的正负样本均为 1:1，AUPR 会随正样本比例改变，不能直接视为自然人群比例下的结果。

## 排序与概率指标

| 任务 | 测试集（链接为原始 CSV） | 样本数（正 / 负） | AUROC | AUPR | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| PH | [test_unseen_hla](../../../dataset/PH/benchmarks/test_unseen_hla.csv) | 47,748（23,874 / 23,874） | 0.8449 | 0.8257 | 0.2368 | 0.2120 |
| PH | [test_unseen_pair](../../../dataset/PH/benchmarks/test_unseen_pair.csv) | 64（32 / 32） | 0.6592 | 0.6075 | 0.3776 | 0.3632 |
| PH | [test_unseen_peptide](../../../dataset/PH/benchmarks/test_unseen_peptide.csv) | 62,764（31,382 / 31,382） | 0.9565 | 0.9551 | 0.0928 | 0.0611 |
| PT | [test_unseen_pair](../../../dataset/PT/benchmarks/test_unseen_pair.csv) | 10,274（5,137 / 5,137） | 0.8125 | 0.8559 | 0.1979 | 0.1586 |
| PT | [test_unseen_peptide](../../../dataset/PT/benchmarks/test_unseen_peptide.csv) | 144,800（72,400 / 72,400） | 0.7098 | 0.7701 | 0.2541 | 0.1889 |
| PT | [test_unseen_tcr](../../../dataset/PT/benchmarks/test_unseen_tcr.csv) | 142,960（71,480 / 71,480） | 0.8664 | 0.8811 | 0.1481 | 0.0858 |
| PHT | [test_unseen_hla](../../../dataset/PHT/benchmarks/test_unseen_hla.csv) | 19,670（9,835 / 9,835） | 0.7800 | 0.6421 | 0.5102 | 0.5060 |
| PHT | [test_unseen_peptide](../../../dataset/PHT/benchmarks/test_unseen_peptide.csv) | 14,118（7,059 / 7,059） | 0.9816 | 0.9837 | 0.0784 | 0.0767 |
| PHT | [test_unseen_tcr](../../../dataset/PHT/benchmarks/test_unseen_tcr.csv) | 7,698（3,849 / 3,849） | 0.9994 | 0.9995 | 0.0159 | 0.0120 |
| PHT | [test_unseen_triple](../../../dataset/PHT/benchmarks/test_unseen_triple.csv) | 660（330 / 330） | 0.8083 | 0.7688 | 0.2698 | 0.2632 |

## 分类指标：检查点保存的验证集阈值

| 任务 | 测试集 | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| PH | test_unseen_hla | 0.7648 | 0.5317 | 0.7537 | 0.7909 | 0.7199 | 0.8096 |
| PH | test_unseen_pair | 0.5625 | 0.1390 | 0.4400 | 0.6111 | 0.3438 | 0.7812 |
| PH | test_unseen_peptide | 0.8913 | 0.7839 | 0.8943 | 0.8700 | 0.9201 | 0.8625 |
| PT | test_unseen_pair | 0.7724 | 0.5685 | 0.7346 | 0.8812 | 0.6297 | 0.9151 |
| PT | test_unseen_peptide | 0.6802 | 0.4103 | 0.5799 | 0.8451 | 0.4413 | 0.9191 |
| PT | test_unseen_tcr | 0.8101 | 0.6267 | 0.7955 | 0.8618 | 0.7386 | 0.8816 |
| PHT | test_unseen_hla | 0.5787 | 0.1750 | 0.4614 | 0.6396 | 0.3609 | 0.7966 |
| PHT | test_unseen_peptide | 0.9158 | 0.8316 | 0.9152 | 0.9211 | 0.9095 | 0.9221 |
| PHT | test_unseen_tcr | 0.9775 | 0.9559 | 0.9780 | 0.9581 | 0.9987 | 0.9564 |
| PHT | test_unseen_triple | 0.6621 | 0.3340 | 0.6162 | 0.7131 | 0.5424 | 0.7818 |

## 分类指标：固定阈值 0.5 对照

这组指标由相同的逐条预测概率复算，未使用测试标签选择阈值。

| 任务 | 测试集 | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| PH | test_unseen_hla | 0.7053 | 0.4509 | 0.6285 | 0.8499 | 0.4987 | 0.9119 |
| PH | test_unseen_pair | 0.6094 | 0.2582 | 0.4681 | 0.7333 | 0.3438 | 0.8750 |
| PH | test_unseen_peptide | 0.8824 | 0.7699 | 0.8752 | 0.9320 | 0.8250 | 0.9398 |
| PT | test_unseen_pair | 0.7672 | 0.5515 | 0.7343 | 0.8551 | 0.6434 | 0.8910 |
| PT | test_unseen_peptide | 0.6822 | 0.3950 | 0.6063 | 0.7966 | 0.4894 | 0.8751 |
| PT | test_unseen_tcr | 0.8136 | 0.6282 | 0.8081 | 0.8327 | 0.7849 | 0.8423 |
| PHT | test_unseen_hla | 0.4840 | -0.0844 | 0.0397 | 0.2857 | 0.0214 | 0.9466 |
| PHT | test_unseen_peptide | 0.9142 | 0.8381 | 0.9072 | 0.9883 | 0.8384 | 0.9901 |
| PHT | test_unseen_tcr | 0.9808 | 0.9620 | 0.9805 | 0.9968 | 0.9647 | 0.9969 |
| PHT | test_unseen_triple | 0.7273 | 0.4905 | 0.6642 | 0.8641 | 0.5394 | 0.9152 |

## 数据与 cold 划分检查

- 全部 10 个 CSV 的原始行均被保留；无无效序列、长度不合规或重复记录被剔除。评估程序检查了同任务测试记录与 train/val 监督记录的重叠。
- 当前六个 train/val CSV 的 SHA-256 与训练时审计一致；检查点中的训练数据指纹也与训练审计一致。所有测试序列均已覆盖缓存。
- 下表逐项统计测试集中不同实体序列在**任一任务训练池**出现的数量；仅审计该 split 所声称未见过的实体，pair/triple 则列出其所有实体。分母是该测试集中不同实体的数量。

| 任务 | 测试集 | 训练池中出现的不同实体数 / 测试集不同实体总数 |
|---|---|---|
| PH | test_unseen_hla | HLA 0 / 21 |
| PH | test_unseen_pair | peptide 0 / 55；HLA 0 / 16 |
| PH | test_unseen_peptide | peptide 0 / 41,668 |
| PT | test_unseen_pair | peptide 0 / 193；TCR 0 / 9,382 |
| PT | test_unseen_peptide | peptide 0 / 302 |
| PT | test_unseen_tcr | TCR 5 / 33,061 |
| PHT | test_unseen_hla | HLA 0 / 22 |
| PHT | test_unseen_peptide | peptide 0 / 268 |
| PHT | test_unseen_tcr | TCR 4 / 6,808 |
| PHT | test_unseen_triple | peptide 0 / 28；HLA 0 / 14；TCR 0 / 644 |

- PT 的 unseen-TCR 中有 **5 个不同 TCR、涉及 8 条测试记录**出现在 PHT 训练池；PHT 的 unseen-TCR 中有 **4 个不同 TCR、涉及 4 条测试记录**出现在 PT 或 PHT 训练池。因此这两个文件整体不能称为严格跨任务 cold-TCR 测试。
- PH unseen-pair 与 PHT unseen-triple 只有 64 和 660 条记录，尤其 PH unseen-pair 的指标可能受少量样本影响。
- 逐条预测与原始测试 CSV 已按文件、行号、序列和标签核对；并从预测概率独立复算 AUROC、AUPR、Accuracy、MCC、F1、Precision、Recall 和 Specificity，结果与原始评估一致。

## 结果解读

- PH 的 unseen-peptide 表现较好，AUROC / AUPR 分别为 0.9565 / 0.9551；unseen-pair 的 AUPR 为 0.6075，但该集仅 64 条。
- PT 的 unseen-peptide AUROC / AUPR 为 0.7098 / 0.7701，验证阈值下 Recall 为 0.4413；unseen-TCR 的 AUPR 为 0.8811，需结合上面的训练池实体重叠解释。
- PHT 的 unseen-TCR AUPR 为 0.9995，但仍有上述 4 个训练池 TCR；unseen-HLA 的 AUPR 为 0.6421、验证阈值下 MCC 为 0.1750。该集使用固定 0.5 时 Recall 降至 0.0214、MCC 为 -0.0844，表明其分类结果对阈值高度敏感。
- 这些结果反映各 benchmark 划分上的表现；由于样本平衡和不同划分构成，不能单凭结果断言自然样本分布下的性能。

## 结果文件与复现

- [完整指标与数据审计 JSON](metrics.json)
- [固定阈值 0.5 指标 JSON](fixed_threshold_0_5.json)
- [逐行对应及指标复算检查 JSON](verification.json)
- [评估程序生成的原始 Markdown](metrics.md)；[运行日志](evaluation.log)
- 每条预测包含原文件、行号、序列、标签、概率、专家路由权重、检查点阈值和分类结果：

| 任务 | 测试集 | 逐条预测 |
|---|---|---|
| PH | test_unseen_hla | [预测 CSV](ph_test_unseen_hla_predictions.csv) |
| PH | test_unseen_pair | [预测 CSV](ph_test_unseen_pair_predictions.csv) |
| PH | test_unseen_peptide | [预测 CSV](ph_test_unseen_peptide_predictions.csv) |
| PT | test_unseen_pair | [预测 CSV](pt_test_unseen_pair_predictions.csv) |
| PT | test_unseen_peptide | [预测 CSV](pt_test_unseen_peptide_predictions.csv) |
| PT | test_unseen_tcr | [预测 CSV](pt_test_unseen_tcr_predictions.csv) |
| PHT | test_unseen_hla | [预测 CSV](pht_test_unseen_hla_predictions.csv) |
| PHT | test_unseen_peptide | [预测 CSV](pht_test_unseen_peptide_predictions.csv) |
| PHT | test_unseen_tcr | [预测 CSV](pht_test_unseen_tcr_predictions.csv) |
| PHT | test_unseen_triple | [预测 CSV](pht_test_unseen_triple_predictions.csv) |

在项目根目录执行的评估命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/mjp/miniconda3/envs/unifyimmun/bin/python -u \
  source/evaluate_plm_stage_MOE_benchmarks.py \
  --config configs/plm_stage_MOE.yaml \
  --checkpoint trained_model/PLM_Stage_MOE/seed_42/sequential_online_20261027_193829/best.pt \
  --device cuda:0 --num-workers 2 \
  --output-dir reports/stage_MOE/sequential_20261027_193829_benchmarks_20261028
```
