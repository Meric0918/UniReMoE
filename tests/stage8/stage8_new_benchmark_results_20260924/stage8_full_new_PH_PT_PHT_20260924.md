# Stage 8 `full` 在 PH/PT/PHT test 数据上的评测结果

## 1. 结论摘要

本报告使用 Stage 8 的 best checkpoint，对 `dataset/PH/new`、`dataset/PT/new` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **393,267** 条有效样本。

- PH benchmark 使用 `phla` head；PT benchmark 使用 `ptcr` head。
- PHT benchmark 同时报告 `phla`、`ptcr` 和 `pht` 三个 head。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集选择的 PHT 阈值仅作补充，不使用 test 标签调参。
- PHT 约束统计使用训练配置中的 margin，仅用于报告测试时的约束违反率和 hinge 强度。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Condition | `full` |
| Checkpoint | [`stage8_best_20260921_111731_349210.pt`](../../../trained_model/PLM_Stage8/fold_1/seed_42/full/stage8_best_20260921_111731_349210.pt) |
| Checkpoint 类型 | `stage8_online_model`，format `2` |
| Checkpoint 角色 | `best` |
| Checkpoint SHA-256 | `a252996a23eb2420c7c8bcb1cc4bb12c7db796551a47b82475d3f553dcbfcdbb` |
| Best Stage8B round | `35`；selection score `0.8688053079943364` |
| Validation joint score | `0.8688053079943364` |
| 配置 | [`plm_stage8_20260920_best_joint.yaml`](../../../configs/plm_stage8_20260920_best_joint.yaml) |
| PH 数据目录 | `dataset/PH/new` |
| PT 数据目录 | `dataset/PT/new` |
| PHT 数据目录 | `dataset/PHT/benchmarks` |
| 设备 | `cuda` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `128` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `5.9604644775390625e-06` |
| Constraint margin | `0.05` |

所有 head 均从同一个 Stage 8 full online model 产生；PH/PT pair 只传入对应的 peptide-receptor 表示，PHT 则一次前向产生三个 head 的输出。

## 3. 数据审计

| 数据集 | 原始行数 | 纳入评测 | 过滤/去重差额 | PH 正/负 | PT 正/负 | PHT 正/负 |
|---|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | 24,696 | 24,696 | 0 | 9,271/15,425 | — | — |
| `PH/test_unseen_pair` | 51 | 51 | 0 | 22/29 | — | — |
| `PH/test_unseen_peptide` | 62,764 | 62,764 | 0 | 31,382/31,382 | — | — |
| `PHT/test_unseen_hla` | 19,670 | 19,670 | 0 | 9,836/9,834 | 9,836/9,834 | 9,835/9,835 |
| `PHT/test_unseen_peptide` | 14,118 | 14,118 | 0 | 7,060/7,058 | 7,060/7,058 | 7,059/7,059 |
| `PHT/test_unseen_tcr` | 7,698 | 7,698 | 0 | 3,850/3,848 | 3,850/3,848 | 3,849/3,849 |
| `PHT/test_unseen_triple` | 660 | 660 | 0 | 331/329 | 658/2 | 330/330 |
| `PT/test_unseen_pair` | 8,207 | 8,207 | 0 | — | 3,113/5,094 | — |
| `PT/test_unseen_peptide` | 112,443 | 112,443 | 0 | — | 41,541/70,902 | — |
| `PT/test_unseen_tcr` | 142,960 | 142,960 | 0 | — | 71,480/71,480 | — |
| **合计** | **393,267** | **393,267** | **0** | — | — | — |

测试集合并包含 **65,268** 个唯一 peptide、**122** 个唯一 HLA 和 **131,892** 个唯一 TCR。

## 4. 主结果：固定阈值 0.5

AUROC/AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 `0.5`。

| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | `phla` | 24,696 | 0.605857 | 0.484359 | 0.612164 | 0.109855 | 0.358731 | 0.472904 | 0.288966 | 0.806418 | 2.072427 |
| `PH/test_unseen_pair` | `phla` | 51 | 0.470219 | 0.377887 | 0.509804 | -0.091634 | 0.193548 | 0.333333 | 0.136364 | 0.793103 | 3.273802 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.869663 | 0.871944 | 0.792381 | 0.585407 | 0.787391 | 0.806781 | 0.768912 | 0.815850 | 0.845521 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.860178 | 0.613175 | 0.895780 | 0.795833 | 0.900918 | 0.858670 | 0.947540 | 0.844011 | 0.574916 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.992562 | 0.994080 | 0.918302 | 0.844465 | 0.923516 | 0.868188 | 0.986377 | 0.850214 | 0.177568 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.968958 | 0.965608 | 0.906101 | 0.817024 | 0.900715 | 0.955520 | 0.851856 | 0.960346 | 0.654585 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.991833 | 0.991686 | 0.968480 | 0.938299 | 0.969304 | 0.944736 | 0.995184 | 0.941768 | 0.137164 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.845759 | 0.895565 | 0.818813 | 0.651568 | 0.798074 | 0.901391 | 0.716006 | 0.921649 | 0.614302 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.989560 | 0.988629 | 0.898286 | 0.808418 | 0.888803 | 0.980188 | 0.813005 | 0.983567 | 0.698966 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.997312 | 0.997344 | 0.988179 | 0.976514 | 0.988287 | 0.979587 | 0.997143 | 0.979210 | 0.046314 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.961341 | 0.971775 | 0.891530 | 0.784571 | 0.894823 | 0.868672 | 0.922597 | 0.860447 | 0.261559 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.998691 | 0.998770 | 0.979995 | 0.960504 | 0.979662 | 0.996240 | 0.963627 | 0.996363 | 0.114189 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.660217 | 0.643721 | 0.696970 | 0.414773 | 0.642857 | 0.786026 | 0.543807 | 0.851064 | 2.489121 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.406155 | 0.997274 | 0.433333 | -0.062658 | 0.604651 | 0.993056 | 0.434650 | 0.000000 | 2.019187 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.832360 | 0.841133 | 0.746970 | 0.542938 | 0.680688 | 0.922280 | 0.539394 | 0.954545 | 3.940037 |
| `PT/test_unseen_pair` | `ptcr` | 8,207 | 0.811102 | 0.730240 | 0.751066 | 0.454922 | 0.593271 | 0.780105 | 0.478638 | 0.917550 | 0.644748 |
| `PT/test_unseen_peptide` | `ptcr` | 112,443 | 0.716221 | 0.595069 | 0.703050 | 0.323416 | 0.483447 | 0.676436 | 0.376134 | 0.894587 | 0.757773 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.810886 | 0.821968 | 0.731918 | 0.475876 | 0.698188 | 0.798678 | 0.620159 | 0.843677 | 0.667086 |

## 5. benchmark 宏平均

同一数据族和 head 内为各 benchmark 的简单算术平均，不是拼接样本后的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.648580 | 0.578063 | 0.638116 | 0.201209 | 0.446557 | 0.502505 |
| `PHT` / `phla` | 0.877385 | 0.811482 | 0.887352 | 0.781355 | 0.875342 | 0.846583 |
| `PHT` / `pht` | 0.947392 | 0.948535 | 0.882838 | 0.782221 | 0.862467 | 0.884691 |
| `PHT` / `ptcr` | 0.801454 | 0.964673 | 0.765495 | 0.554487 | 0.805266 | 0.778275 |
| `PT` / `ptcr` | 0.779403 | 0.715759 | 0.728678 | 0.418071 | 0.591635 | 0.646709 |

## 6. PHT checkpoint 阈值补充

checkpoint 在 validation 上为 PHT head 记录的阈值为 `5.9604644775390625e-06`。下表与固定阈值 `0.5` 对比；该阈值未由任何 test 标签重新选择。

| Benchmark | Head | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `pht` | `0.5` | 0.906101 | 0.817024 | 0.900715 | 0.955520 | 0.851856 | 0.960346 |
| `PHT/test_unseen_hla` | `pht` | `5.9604644775390625e-06` | 0.880224 | 0.768001 | 0.888055 | 0.833556 | 0.950178 | 0.810269 |
| `PHT/test_unseen_peptide` | `pht` | `0.5` | 0.898286 | 0.808418 | 0.888803 | 0.980188 | 0.813005 | 0.983567 |
| `PHT/test_unseen_peptide` | `pht` | `5.9604644775390625e-06` | 0.956014 | 0.913669 | 0.957293 | 0.930233 | 0.985975 | 0.926052 |
| `PHT/test_unseen_tcr` | `pht` | `0.5` | 0.979995 | 0.960504 | 0.979662 | 0.996240 | 0.963627 | 0.996363 |
| `PHT/test_unseen_tcr` | `pht` | `5.9604644775390625e-06` | 0.981943 | 0.964348 | 0.982218 | 0.967490 | 0.997402 | 0.966485 |
| `PHT/test_unseen_triple` | `pht` | `0.5` | 0.746970 | 0.542938 | 0.680688 | 0.922280 | 0.539394 | 0.954545 |
| `PHT/test_unseen_triple` | `pht` | `5.9604644775390625e-06` | 0.677273 | 0.368189 | 0.626970 | 0.742739 | 0.542424 | 0.812121 |

## 7. PHT 约束统计

违反率的分母只包含该约束适用的 label type；`hierarchy_*` 比较 `pht` 与 `phla`，`pt_*` 比较 `pht` 与 `ptcr`。

| Benchmark | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall | Hinge overall |
|---|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | 0.719980 | 0.011592 | 0.002745 | 0.047997 | 0.195587 | 0.005140 |
| `PHT/test_unseen_peptide` | 0.003400 | 0.013035 | 0.194645 | 0.019977 | 0.057767 | 0.021518 |
| `PHT/test_unseen_tcr` | 0.001819 | 0.003638 | 0.165238 | 0.005198 | 0.043978 | 0.007763 |
| `PHT/test_unseen_triple` | 0.000000 | 0.009119 | 0.012121 | 0.000000 | 0.007064 | 0.001162 |

## 8. 结果说明与可复核产物

- 全部 benchmark/head 中 performance average 最高的是 `PHT/test_unseen_tcr` 的 `phla` head：`0.989527`。
- PHT head 中 performance average 最高的是 `PHT/test_unseen_tcr`：`0.983524`。
- `test_unseen_pair` 样本量较小，指标方差可能较大，应结合样本数和混淆矩阵解读。
- checkpoint validation metrics 摘要：`{"by_type": {"000": {"mean_phla_probability": 0.06212020292878151, "mean_pht_probability": 0.011948627419769764, "mean_ptcr_probability": 0.16277384757995605, "phla_cross_entropy": 0.8005392259124545, "pht_cross_entropy": 0.16874988677468042, "ptcr_cross_entropy": 0.24341607380685873, "samples": 15491}, "010": {"mean_phla_probability": 0.09001121670007706, "mean_pht_probability": 0.026693182066082954, "mean_ptcr_probability": 0.39128947257995605, "phla_cross_entropy": 1.170295851517088, "pht_cross_entropy": 0.5153720317330283, "ptcr_cross_entropy": 2.161622545778655, "samples": 533}, "100": {"mean_phla_probability": 0.9873046875, "mean_pht_probability": 3.543570346664637e-05, "mean_ptcr_probability": 0.1313018798828125, "phla_cross_entropy": 0.013029000284753671, "pht_cross_entropy": 3.5438160638463775e-05, "ptcr_cross_entropy": 0.16454223661970044, "samples": 4}, "111": {"mean_phla_probability": 0.9901486039161682, "mean_pht_probability": 0.8191623091697693, "mean_ptcr_probability": 0.6965726017951965, "phla_cross_entropy": 0.03438965948508201, "pht_cross_entropy": 1.1854040004433148, "ptcr_cross_entropy": 0.9422041652690621, "samples": 16027}}, "constraint_components": {"hierarchy_negative": 0.0027711738366633654, "hierarchy_positive": 0.0004897608887404203, "pt_negative": 0.004633605945855379, "pt_positive": 0.11333805322647095}, "phla": {"accuracy": 0.9656215878958041, "aupr": 0.9874088197623948, "auroc": 0.9878074706915361, "brier": 0.029558574573375223, "ece": 0.02522759513310649, "f1": 0.9664454052737349, "group_count": 0, "group_coverage": NaN, "group_macro_accuracy": NaN, "group_macro_nll": NaN, "group_macro_recall": NaN, "loss": 0.15422431447170878, "loss_p95": 0.10438442968477084, "loss_p99": 8.234274863953788, "mcc": 0.9323472120373624, "mean_confidence": 0.9908404984134629, "nll": 0.23517995611088502, "performance_avg": 0.9679260991321664, "precision": 0.9440247457022188, "recall": 0.9899569583931134, "samples": 32055, "sensitivity": 0.9899569583931134, "specificity": 0.9412755866200699, "worst_group_nll": NaN}, "pht": {"accuracy": 0.9029480580252691, "aupr": 0.9912115082689419, "auroc": 0.9918498196104655, "brier": 0.0892819157149506, "ece": 0.08695513862163866, "f1": 0.8939275120188209, "group_count": 0, "group_coverage": NaN, "group_macro_accuracy": NaN, "group_macro_nll": NaN, "group_macro_recall": NaN, "loss": 0.6262333226267139, "loss_p95": 5.692192187441372, "loss_p99": 12.044373382893536, "mcc": 0.8178019487183789, "mean_confidence": 0.9899030633135747, "nll": 0.6400090225464666, "performance_avg": 0.9195477693283752, "precision": 0.9854909036235152, "recall": 0.8179322393461035, "samples": 32055, "sensitivity": 0.8179322393461035, "specificity": 0.9879585724981282, "worst_group_nll": NaN}, "pht_selected": {"accuracy": 0.9646545000779909, "aupr": 0.9912115082689419, "auroc": 0.9918498196104655, "brier": 0.0892819157149506, "ece": 0.08695513862163866, "f1": 0.9653315382026254, "group_count": 0, "group_coverage": NaN, "group_macro_accuracy": NaN, "group_macro_nll": NaN, "group_macro_recall": NaN, "loss": 0.0, "loss_p95": 5.692192187441372, "loss_p99": 12.044373382893536, "mcc": 0.9300210068111631, "mean_confidence": 0.9899030633135747, "nll": 0.6400090225464666, "performance_avg": 0.9686136745942374, "precision": 0.9471598414795245, "recall": 0.9842141386410432, "samples": 32055, "sensitivity": 0.9842141386410432, "specificity": 0.9450960818567506, "worst_group_nll": NaN}, "ptcr": {"accuracy": 0.7807206364061768, "aupr": 0.885926738080232, "auroc": 0.8348986524404788, "brier": 0.1732433474093227, "ece": 0.1312421136047906, "f1": 0.7603886142832793, "group_count": 0, "group_coverage": NaN, "group_macro_accuracy": NaN, "group_macro_nll": NaN, "group_macro_recall": NaN, "loss": 0.6248147105560755, "loss_p95": 3.6218416174627883, "loss_p99": 5.49193292996075, "mcc": 0.5805776142467464, "mean_confidence": 0.9050302891717847, "nll": 0.624684708365492, "performance_avg": 0.7685024510913826, "precision": 0.8730332681017613, "recall": 0.6734903381642512, "samples": 32055, "sensitivity": 0.6734903381642512, "specificity": 0.8953210713133268, "worst_group_nll": NaN}, "violations": {"hierarchy_negative": 0.006677483774338492, "hierarchy_positive": 0.0022462095214325823, "overall": 0.06951064130999009, "pt_negative": 0.013101000322684736, "pt_positive": 0.2541336494665252}}`。
- 本次没有重新训练、修改 checkpoint，且没有使用 test 标签进行阈值选择或模型选择。
- 每个 checkpoint 的 unique peptide/HLA/TCR 分别只编码一次，再复用 residue-level 表示完成行级推理。
- 原始 JSON：[`stage8_full_new_PH_PT_PHT_20260924.json`](stage8_full_new_PH_PT_PHT_20260924.json)
- 本报告：[`stage8_full_new_PH_PT_PHT_20260924.md`](stage8_full_new_PH_PT_PHT_20260924.md)

