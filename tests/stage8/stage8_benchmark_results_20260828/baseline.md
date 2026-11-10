# Stage 8 `baseline` 在 PH/PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本报告使用 Stage-8 V2 的 best checkpoint，对 `dataset/PH/benchmarks`、`dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **450,756** 条有效样本。

- PH benchmark 使用 `phla` head；PT benchmark 使用 `ptcr` head。
- PHT benchmark 同时报告 `phla`、`ptcr` 和 `pht` 三个 head。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集选择的 PHT 阈值仅作补充，不使用 test 标签调参。
- PHT 约束统计使用训练配置中的 margin，仅用于报告测试时的约束违反率和 hinge 强度。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Condition | `baseline` |
| Checkpoint | [`stage8_best_20260827_110610_283819.pt`](../../../trained_model/PLM_Stage8/fold_1/seed_42/baseline/stage8_best_20260827_110610_283819.pt) |
| Checkpoint 类型 | `stage8_online_model`，format `2` |
| Checkpoint SHA-256 | `2563b55aa2a39d03c7707bc9a5099d342c2454c8f798d5fefe94d4a8d78b36ad` |
| Best Stage8B round | `11`；selection score `0.8771615366929253` |
| Validation joint score | `0.8771615366929253` |
| 配置 | [`plm_stage8.yaml`](../../../configs/plm_stage8.yaml) |
| 设备 | `cuda:0` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `256` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.021240234375` |
| Constraint margin | `0.05` |

所有 head 均从同一个 Stage-8 V2 full online model 产生；PH/PT pair 只传入对应的 peptide-receptor 表示，PHT 则一次前向产生三个 head 的输出。

## 3. 数据审计

| 数据集 | 原始行数 | 纳入评测 | 过滤/去重差额 | PH 正/负 | PT 正/负 | PHT 正/负 |
|---|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | 47,748 | 47,748 | 0 | 23,874/23,874 | — | — |
| `PH/test_unseen_pair` | 64 | 64 | 0 | 32/32 | — | — |
| `PH/test_unseen_peptide` | 62,764 | 62,764 | 0 | 31,382/31,382 | — | — |
| `PHT/test_unseen_hla` | 19,670 | 19,670 | 0 | 9,836/9,834 | 9,836/9,834 | 9,835/9,835 |
| `PHT/test_unseen_peptide` | 14,118 | 14,118 | 0 | 7,060/7,058 | 7,060/7,058 | 7,059/7,059 |
| `PHT/test_unseen_tcr` | 7,698 | 7,698 | 0 | 3,850/3,848 | 3,850/3,848 | 3,849/3,849 |
| `PHT/test_unseen_triple` | 660 | 660 | 0 | 331/329 | 658/2 | 330/330 |
| `PT/test_unseen_pair` | 10,274 | 10,274 | 0 | — | 5,137/5,137 | — |
| `PT/test_unseen_peptide` | 144,800 | 144,800 | 0 | — | 72,400/72,400 | — |
| `PT/test_unseen_tcr` | 142,960 | 142,960 | 0 | — | 71,480/71,480 | — |
| **合计** | **450,756** | **450,756** | **0** | — | — | — |

测试集合并包含 **85,657** 个唯一 peptide、**122** 个唯一 HLA 和 **143,269** 个唯一 TCR。

## 4. 主结果：固定阈值 0.5

AUROC/AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 `0.5`。

| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | `phla` | 47,748 | 0.820273 | 0.800672 | 0.713978 | 0.449439 | 0.662415 | 0.808094 | 0.561238 | 0.866717 | 0.805200 |
| `PH/test_unseen_pair` | `phla` | 64 | 0.592285 | 0.671637 | 0.593750 | 0.216506 | 0.458333 | 0.687500 | 0.343750 | 0.843750 | 1.565470 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.929727 | 0.925201 | 0.858199 | 0.717048 | 0.855115 | 0.874126 | 0.836913 | 0.879485 | 0.392625 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.624737 | 0.429784 | 0.478139 | -0.087164 | 0.079950 | 0.337623 | 0.045344 | 0.911023 | 1.546613 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.993134 | 0.994526 | 0.942806 | 0.888775 | 0.945125 | 0.908392 | 0.984953 | 0.900651 | 0.144056 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.865336 | 0.757279 | 0.507270 | 0.063535 | 0.040016 | 0.773946 | 0.020539 | 0.994001 | 3.256030 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.998040 | 0.998194 | 0.968551 | 0.938433 | 0.969371 | 0.944863 | 0.995184 | 0.941910 | 0.101495 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.856618 | 0.900075 | 0.815696 | 0.667370 | 0.780125 | 0.966904 | 0.653824 | 0.977614 | 0.543486 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.986270 | 0.987068 | 0.916135 | 0.842488 | 0.909077 | 0.992621 | 0.838504 | 0.993767 | 0.535227 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.997549 | 0.997502 | 0.972850 | 0.946317 | 0.973338 | 0.956380 | 0.990909 | 0.954782 | 0.074880 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.950866 | 0.964506 | 0.893998 | 0.788254 | 0.892660 | 0.904318 | 0.881299 | 0.906705 | 0.263498 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.999336 | 0.999361 | 0.979865 | 0.960236 | 0.979533 | 0.995972 | 0.963627 | 0.996103 | 0.055742 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.805618 | 0.832705 | 0.760606 | 0.581709 | 0.693798 | 0.967568 | 0.540785 | 0.981763 | 0.866128 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.708587 | 0.998944 | 0.443939 | 0.048959 | 0.613277 | 1.000000 | 0.442249 | 1.000000 | 1.076758 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.733352 | 0.804323 | 0.768182 | 0.601127 | 0.700587 | 0.988950 | 0.542424 | 0.993939 | 3.215106 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.852268 | 0.871092 | 0.715203 | 0.494420 | 0.622257 | 0.923726 | 0.469145 | 0.961261 | 0.646122 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.740199 | 0.746939 | 0.616478 | 0.296769 | 0.444360 | 0.806142 | 0.306713 | 0.926243 | 0.831572 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.891441 | 0.887710 | 0.796216 | 0.598990 | 0.779981 | 0.847500 | 0.722426 | 0.870006 | 0.436143 |

## 5. benchmark 宏平均

同一数据族和 head 内为各 benchmark 的简单算术平均，不是拼接样本后的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.780762 | 0.799170 | 0.721976 | 0.460998 | 0.658621 | 0.684305 |
| `PHT` / `phla` | 0.856486 | 0.814546 | 0.795037 | 0.594824 | 0.679114 | 0.748001 |
| `PHT` / `pht` | 0.896074 | 0.887008 | 0.792863 | 0.616846 | 0.657303 | 0.770019 |
| `PHT` / `ptcr` | 0.877301 | 0.964513 | 0.774110 | 0.598339 | 0.807797 | 0.804412 |
| `PT` / `ptcr` | 0.827969 | 0.835247 | 0.709299 | 0.463393 | 0.615532 | 0.690288 |

## 6. PHT checkpoint 阈值补充

checkpoint 在 validation 上为 PHT head 记录的阈值为 `0.021240234375`。下表与固定阈值 `0.5` 对比；该阈值未由任何 test 标签重新选择。

| Benchmark | Head | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `pht` | `0.5` | 0.507270 | 0.063535 | 0.040016 | 0.773946 | 0.020539 | 0.994001 |
| `PHT/test_unseen_hla` | `pht` | `0.021240234375` | 0.495425 | -0.023681 | 0.063591 | 0.441099 | 0.034265 | 0.956584 |
| `PHT/test_unseen_peptide` | `pht` | `0.5` | 0.916135 | 0.842488 | 0.909077 | 0.992621 | 0.838504 | 0.993767 |
| `PHT/test_unseen_peptide` | `pht` | `0.021240234375` | 0.933135 | 0.869189 | 0.930281 | 0.971764 | 0.892194 | 0.974076 |
| `PHT/test_unseen_tcr` | `pht` | `0.5` | 0.979865 | 0.960236 | 0.979533 | 0.995972 | 0.963627 | 0.996103 |
| `PHT/test_unseen_tcr` | `pht` | `0.021240234375` | 0.986880 | 0.973838 | 0.986963 | 0.980759 | 0.993245 | 0.980514 |
| `PHT/test_unseen_triple` | `pht` | `0.5` | 0.768182 | 0.601127 | 0.700587 | 0.988950 | 0.542424 | 0.993939 |
| `PHT/test_unseen_triple` | `pht` | `0.021240234375` | 0.746970 | 0.541307 | 0.681905 | 0.917949 | 0.542424 | 0.951515 |

## 7. PHT 约束统计

违反率的分母只包含该约束适用的 label type；`hierarchy_*` 比较 `pht` 与 `phla`，`pt_*` 比较 `pht` 与 `ptcr`。

| Benchmark | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall | Hinge overall |
|---|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | 0.018302 | 0.008847 | 0.004779 | 0.014440 | 0.011592 | 0.001091 |
| `PHT/test_unseen_peptide` | 0.004958 | 0.005667 | 0.428106 | 0.015018 | 0.113445 | 0.023233 |
| `PHT/test_unseen_tcr` | 0.020005 | 0.004158 | 0.217199 | 0.009096 | 0.062622 | 0.011092 |
| `PHT/test_unseen_triple` | 0.000000 | 0.015198 | 0.015152 | 0.000000 | 0.010091 | 0.000838 |

## 8. 结果说明与可复核产物

- 全部 benchmark/head 中 performance average 最高的是 `PHT/test_unseen_tcr` 的 `pht` head：`0.983666`。
- PHT head 中 performance average 最高的是 `PHT/test_unseen_tcr`：`0.983666`。
- `test_unseen_pair` 样本量较小，指标方差可能较大，应结合样本数和混淆矩阵解读。
- checkpoint validation metrics 摘要：`{"by_type": {"000": {"mean_phla_probability": 0.08113202452659607, "mean_pht_probability": 0.010063805617392063, "mean_ptcr_probability": 0.1643112450838089, "phla_cross_entropy": 0.18675215342370655, "pht_cross_entropy": 0.06676580278838365, "ptcr_cross_entropy": 0.25731249537185324, "samples": 15491}, "010": {"mean_phla_probability": 0.10932601243257523, "mean_pht_probability": 0.03600739315152168, "mean_ptcr_probability": 0.43261340260505676, "phla_cross_entropy": 0.25407564531708476, "pht_cross_entropy": 0.5496017416011982, "ptcr_cross_entropy": 1.583065891529836, "samples": 533}, "100": {"mean_phla_probability": 0.9091796875, "mean_pht_probability": 0.000843878835439682, "mean_ptcr_probability": 0.07208251953125, "phla_cross_entropy": 0.1025260156747064, "pht_cross_entropy": 0.000844677636947649, "ptcr_cross_entropy": 0.07514993684832302, "samples": 4}, "111": {"mean_phla_probability": 0.9783844947814941, "mean_pht_probability": 0.8229917287826538, "mean_ptcr_probability": 0.7093371748924255, "phla_cross_entropy": 0.043988369357001525, "pht_cross_entropy": 0.9432934333849458, "ptcr_cross_entropy": 0.6456240582065722, "samples": 16027}}, "constraint_components": {"hierarchy_negative": 0.001781194587238133, "hierarchy_positive": 0.0011894513154402375, "pt_negative": 0.001852473709732294, "pt_positive": 0.07648367434740067}, "phla": {"accuracy": 0.9603805958508813, "aupr": 0.9945966874050461, "auroc": 0.9946248326032565, "f1": 0.9613441285688197, "loss": 0.11268672130907814, "mcc": 0.921886599128694, "performance_avg": 0.9665665687113394, "precision": 0.9387148546632587, "recall": 0.9850913854407086, "samples": 32055, "sensitivity": 0.9850913854407086, "specificity": 0.9356590114827759}, "pht": {"accuracy": 0.9134612384963344, "aupr": 0.9854590835913191, "auroc": 0.9851067668144048, "f1": 0.9062457753143166, "loss": 0.4913096630374312, "mcc": 0.8368869131406135, "performance_avg": 0.9254319554713977, "precision": 0.9886439053167171, "recall": 0.8365258626068509, "samples": 32055, "sensitivity": 0.8365258626068509, "specificity": 0.9903918143249314}, "pht_selected": {"accuracy": 0.9410388394946186, "aupr": 0.9854590835913191, "auroc": 0.9851067668144048, "f1": 0.9389613744994186, "loss": 0.0, "mcc": 0.8841243902338681, "performance_avg": 0.9469380909267258, "precision": 0.9732208609493205, "recall": 0.9070318836962625, "samples": 32055, "sensitivity": 0.9070318836962625, "specificity": 0.9750436735712503}, "ptcr": {"accuracy": 0.7823428482296053, "aupr": 0.9086723535997927, "auroc": 0.8842800798293979, "f1": 0.7644735509570266, "loss": 0.473488072258618, "mcc": 0.581174176910405, "performance_avg": 0.7841886019052455, "precision": 0.866799356962413, "recall": 0.683756038647343, "samples": 32055, "sensitivity": 0.683756038647343, "specificity": 0.8877057115198451}, "violations": {"hierarchy_negative": 0.007925611582626061, "hierarchy_positive": 0.012541336494665252, "overall": 0.09935035313733818, "pt_negative": 0.014262665375927719, "pt_positive": 0.359830286391714}}`。
- 本次没有重新训练、修改 checkpoint，且没有使用 test 标签进行阈值选择或模型选择。
- 每个 checkpoint 的 unique peptide/HLA/TCR 分别只编码一次，再复用 residue-level 表示完成行级推理。
- 原始 JSON：[`baseline.json`](baseline.json)
- 本报告：[`baseline.md`](baseline.md)

