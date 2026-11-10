# Stage MoE Consis benchmark 测试总结

- 评估日期：2026-11-08。使用同一数据快照、残基缓存及各自的最佳 checkpoint。
- 测试范围：PH 3 个、PT 3 个、PHT 4 个 benchmark；每种训练方式共评估 450,756 条记录。
- 阈值：读取对应 checkpoint 的验证集 MCC 最优阈值；未基于 benchmark 调阈值。
- 报告值格式：`联合 / 交替`；指标按每个 benchmark 独立计算，不跨数据集拼接或平均。

## 使用的模型

- 联合训练：第 12 轮，验证三任务平均 PR-AUC `0.9076`；checkpoint [`best.pt`](trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261106_152222/best.pt)；详细结果 [`metrics.json`](trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261106_152222/benchmarks_20261108_182931/metrics.json)。
- 交替训练（PH→PT→PHT）：第 29 轮，验证三任务平均 PR-AUC `0.8827`；checkpoint [`best.pt`](trained_model/PLM_Stage_MOE_consis/seed_42/sequential_20261106_152222/best.pt)；详细结果 [`metrics.json`](trained_model/PLM_Stage_MOE_consis/seed_42/sequential_20261106_152222/benchmarks_20261108_182931/metrics.json)。

## 分类性能

| 任务 / benchmark | 样本数 | AUROC 联合 / 交替 | AUPR 联合 / 交替 | MCC 联合 / 交替 | F1 联合 / 交替 |
|---|---:|---:|---:|---:|---:|
| PH / test_unseen_hla | 47748 | 0.8839 / 0.8510 | 0.8667 / 0.8529 | 0.6223 / 0.5500 | 0.8071 / 0.7573 |
| PH / test_unseen_pair | 64 | 0.6514 / 0.7222 | 0.6249 / 0.7097 | 0.1348 / 0.2329 | 0.4615 / 0.5283 |
| PH / test_unseen_peptide | 62764 | 0.9645 / 0.9436 | 0.9596 / 0.9416 | 0.8199 / 0.7578 | 0.9121 / 0.8803 |
| PHT / test_unseen_hla | 19670 | 0.9795 / 0.9942 | 0.9654 / 0.9931 | 0.8631 / 0.8788 | 0.9328 / 0.9402 |
| PHT / test_unseen_peptide | 14118 | 0.9913 / 0.9864 | 0.9911 / 0.9873 | 0.8943 / 0.8650 | 0.9475 / 0.9291 |
| PHT / test_unseen_tcr | 7698 | 0.9994 / 0.9991 | 0.9994 / 0.9992 | 0.9684 / 0.9752 | 0.9842 / 0.9876 |
| PHT / test_unseen_triple | 660 | 0.7746 / 0.9852 | 0.8260 / 0.9842 | 0.4361 / 0.8798 | 0.6486 / 0.9379 |
| PT / test_unseen_pair | 10274 | 0.7441 / 0.7661 | 0.8254 / 0.8162 | 0.5493 / 0.4948 | 0.7274 / 0.6726 |
| PT / test_unseen_peptide | 144800 | 0.7242 / 0.7331 | 0.7762 / 0.7651 | 0.3960 / 0.3855 | 0.5827 / 0.6134 |
| PT / test_unseen_tcr | 142960 | 0.8902 / 0.8004 | 0.8967 / 0.8275 | 0.6382 / 0.5316 | 0.8081 / 0.7168 |

## PHT 一致性与类别指标

| PHT benchmark | 111 召回 联合 / 交替 | 000 FPR 联合 / 交替 | 010 FPR 联合 / 交替 | 100 FPR 联合 / 交替 | PH 约束违反率 联合 / 交替 | PT 约束违反率 联合 / 交替 |
|---|---:|---:|---:|---:|---:|---:|
| test_unseen_hla | 0.9642 / 0.9876 | 0.1031 / 0.1133 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.4607 / 0.0019 | 0.0156 / 0.0163 |
| test_unseen_peptide | 0.9548 / 0.8977 | 0.0605 / 0.0346 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.0048 / 0.0031 | 0.0489 / 0.0718 |
| test_unseen_tcr | 0.9966 / 0.9961 | 0.0283 / 0.0208 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.0055 / 0.0035 | 0.0726 / 0.0816 |
| test_unseen_triple | 0.5424 / 0.9152 | 0.0000 / 0.0000 | 0.1280 / 0.0335 | 1.0000 / 1.0000 | 0.2671 / 0.0137 | 0.0060 / 0.0181 |

一致性违反率按 `pPHT > pPH/PT + 0.05` 计算，仅统计各约束适用标签组。完整的约束超出幅度、各组样本数和逐样本概率见运行目录中的 `metrics.json` 与预测 CSV。

## 数据与解释

- 两种模型使用同一份训练副本；从 `72,192` 条 PHT train 记录中排除了 5 条有泄漏风险的 `111` 记录。原始数据未修改。
- PHT `test_unseen_tcr` 的严格冷 TCR 审计发现 3 个唯一 TCR 已出现在某一任务的训练池；该 benchmark 的结果应按此重叠解读。
- `test_unseen_pair` 和 `test_unseen_triple` 表示配对或三元组合场景，不是单实体冷启动；完整 cold audit 随各运行 JSON 保存。
- 报告不将不同任务或 benchmark 的分数合并成总成绩；样本规模及阳性比例不同。

## 结果文件

- 联合训练：[`metrics.md`](trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261106_152222/benchmarks_20261108_182931/metrics.md)、[`metrics.json`](trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261106_152222/benchmarks_20261108_182931/metrics.json) 和该目录下各 benchmark 的逐样本预测 CSV。
- 交替训练（PH→PT→PHT）：[`metrics.md`](trained_model/PLM_Stage_MOE_consis/seed_42/sequential_20261106_152222/benchmarks_20261108_182931/metrics.md)、[`metrics.json`](trained_model/PLM_Stage_MOE_consis/seed_42/sequential_20261106_152222/benchmarks_20261108_182931/metrics.json) 和该目录下各 benchmark 的逐样本预测 CSV。
