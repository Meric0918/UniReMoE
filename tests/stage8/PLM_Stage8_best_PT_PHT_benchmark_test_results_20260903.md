# Stage 8 `full` 在 PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本报告使用 Stage 8 的 best checkpoint，对 `dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **340,180** 条有效样本。

- PT benchmark 使用 `ptcr` head。
- PHT benchmark 同时报告 `phla`、`ptcr` 和 `pht` 三个 head。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集选择的 PHT 阈值仅作补充，不使用 test 标签调参。
- PHT 约束统计使用训练配置中的 margin，仅用于报告测试时的约束违反率和 hinge 强度。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Condition | `full` |
| Checkpoint | [`stage8_best_20260901_214051_422800.pt`](../../trained_model/PLM_Stage8/fold_1/seed_42/full/stage8_best_20260901_214051_422800.pt) |
| Checkpoint 类型 | `stage8_online_model`，format `2` |
| Checkpoint 角色 | `best` |
| Checkpoint SHA-256 | `cbe98a6c050740a715948fd5551e0864a6015cbc52f621bb7a37d073db4e562f` |
| Best Stage8B round | `8`；selection score `0.8895528810588921` |
| Validation joint score | `0.8895528810588921` |
| 配置 | [`plm_stage8.yaml`](../../configs/plm_stage8.yaml) |
| 设备 | `cuda` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `256` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.0164794921875` |
| Constraint margin | `0.05` |

所有 head 均从同一个 Stage 8 full online model 产生；PT pair 只传入对应的 peptide-receptor 表示，PHT 则一次前向产生三个 head 的输出。

## 3. 数据审计

| 数据集 | 原始行数 | 纳入评测 | 过滤/去重差额 | PH 正/负 | PT 正/负 | PHT 正/负 |
|---|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | 19,670 | 19,670 | 0 | 9,836/9,834 | 9,836/9,834 | 9,835/9,835 |
| `PHT/test_unseen_peptide` | 14,118 | 14,118 | 0 | 7,060/7,058 | 7,060/7,058 | 7,059/7,059 |
| `PHT/test_unseen_tcr` | 7,698 | 7,698 | 0 | 3,850/3,848 | 3,850/3,848 | 3,849/3,849 |
| `PHT/test_unseen_triple` | 660 | 660 | 0 | 331/329 | 658/2 | 330/330 |
| `PT/test_unseen_pair` | 10,274 | 10,274 | 0 | — | 5,137/5,137 | — |
| `PT/test_unseen_peptide` | 144,800 | 144,800 | 0 | — | 72,400/72,400 | — |
| `PT/test_unseen_tcr` | 142,960 | 142,960 | 0 | — | 71,480/71,480 | — |
| **合计** | **340,180** | **340,180** | **0** | — | — | — |

测试集合并包含 **1,892** 个唯一 peptide、**117** 个唯一 HLA 和 **143,269** 个唯一 TCR。

## 4. 主结果：固定阈值 0.5

AUROC/AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 `0.5`。

| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.989394 | 0.991349 | 0.922877 | 0.848878 | 0.926054 | 0.889503 | 0.965738 | 0.880008 | 0.195705 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.992345 | 0.994131 | 0.931164 | 0.866947 | 0.934545 | 0.890876 | 0.982717 | 0.879601 | 0.167403 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.982971 | 0.987770 | 0.965989 | 0.933160 | 0.965111 | 0.990685 | 0.940824 | 0.991154 | 0.224579 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.992291 | 0.917153 | 0.956722 | 0.916137 | 0.958325 | 0.924220 | 0.995042 | 0.918390 | 0.143797 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.812197 | 0.858209 | 0.761864 | 0.569948 | 0.703423 | 0.932413 | 0.564731 | 0.959054 | 0.647074 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.974565 | 0.978291 | 0.911744 | 0.831178 | 0.905319 | 0.976397 | 0.843887 | 0.979601 | 0.514120 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.996785 | 0.996706 | 0.965575 | 0.932774 | 0.966570 | 0.939662 | 0.995065 | 0.936071 | 0.098415 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.950828 | 0.964683 | 0.891400 | 0.782816 | 0.891766 | 0.889004 | 0.894545 | 0.888254 | 0.271472 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.998539 | 0.998707 | 0.982203 | 0.964610 | 0.982019 | 0.992308 | 0.971941 | 0.992466 | 0.058610 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.852588 | 0.860673 | 0.707576 | 0.440398 | 0.650995 | 0.810811 | 0.543807 | 0.872340 | 0.971952 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.803571 | 0.999336 | 0.412121 | 0.045872 | 0.581897 | 1.000000 | 0.410334 | 1.000000 | 1.428867 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.575459 | 0.749763 | 0.759091 | 0.574969 | 0.692456 | 0.957219 | 0.542424 | 0.975758 | 2.996922 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.769596 | 0.808435 | 0.712089 | 0.479311 | 0.624715 | 0.896903 | 0.479268 | 0.944909 | 0.781890 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.711615 | 0.724510 | 0.624095 | 0.304411 | 0.470922 | 0.794777 | 0.334586 | 0.913605 | 0.871083 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.858450 | 0.855361 | 0.756330 | 0.523647 | 0.728689 | 0.821927 | 0.654449 | 0.858212 | 0.503407 |

## 5. benchmark 宏平均

同一数据族和 head 内为各 benchmark 的简单算术平均，不是拼接样本后的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PHT` / `phla` | 0.957765 | 0.941470 | 0.888188 | 0.784547 | 0.875486 | 0.889491 |
| `PHT` / `pht` | 0.882883 | 0.928633 | 0.904757 | 0.825979 | 0.886226 | 0.885696 |
| `PHT` / `ptcr` | 0.889735 | 0.954090 | 0.749138 | 0.566396 | 0.777908 | 0.787453 |
| `PT` / `ptcr` | 0.779887 | 0.796102 | 0.697505 | 0.435790 | 0.608109 | 0.663478 |

## 6. PHT checkpoint 阈值补充

checkpoint 在 validation 上为 PHT head 记录的阈值为 `0.0164794921875`。下表与固定阈值 `0.5` 对比；该阈值未由任何 test 标签重新选择。

| Benchmark | Head | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `pht` | `0.5` | 0.965989 | 0.933160 | 0.965111 | 0.990685 | 0.940824 | 0.991154 |
| `PHT/test_unseen_hla` | `pht` | `0.0164794921875` | 0.940214 | 0.880876 | 0.941153 | 0.926594 | 0.956177 | 0.924250 |
| `PHT/test_unseen_peptide` | `pht` | `0.5` | 0.911744 | 0.831178 | 0.905319 | 0.976397 | 0.843887 | 0.979601 |
| `PHT/test_unseen_peptide` | `pht` | `0.0164794921875` | 0.921589 | 0.844740 | 0.919132 | 0.948869 | 0.891203 | 0.951976 |
| `PHT/test_unseen_tcr` | `pht` | `0.5` | 0.982203 | 0.964610 | 0.982019 | 0.992308 | 0.971941 | 0.992466 |
| `PHT/test_unseen_tcr` | `pht` | `0.0164794921875` | 0.979735 | 0.959891 | 0.980031 | 0.965935 | 0.994544 | 0.964926 |
| `PHT/test_unseen_triple` | `pht` | `0.5` | 0.759091 | 0.574969 | 0.692456 | 0.957219 | 0.542424 | 0.975758 |
| `PHT/test_unseen_triple` | `pht` | `0.0164794921875` | 0.739394 | 0.520911 | 0.675472 | 0.895000 | 0.542424 | 0.936364 |

## 7. PHT 约束统计

违反率的分母只包含该约束适用的 label type；`hierarchy_*` 比较 `pht` 与 `phla`，`pt_*` 比较 `pht` 与 `ptcr`。

| Benchmark | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall | Hinge overall |
|---|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | 0.001017 | 0.010779 | 0.002135 | 0.020948 | 0.008719 | 0.001045 |
| `PHT/test_unseen_peptide` | 0.007508 | 0.016577 | 0.481513 | 0.030887 | 0.134129 | 0.046799 |
| `PHT/test_unseen_tcr` | 0.012731 | 0.010135 | 0.200052 | 0.014813 | 0.059439 | 0.011053 |
| `PHT/test_unseen_triple` | 0.003030 | 0.027356 | 0.015152 | 0.000000 | 0.015136 | 0.000605 |

## 8. 结果说明与可复核产物

- 全部 benchmark/head 中 performance average 最高的是 `PHT/test_unseen_tcr` 的 `pht` head：`0.985215`。
- PHT head 中 performance average 最高的是 `PHT/test_unseen_tcr`：`0.985215`。
- `test_unseen_pair` 样本量较小，指标方差可能较大，应结合样本数和混淆矩阵解读。
- checkpoint validation metrics 摘要：`{"by_type": {"000": {"mean_phla_probability": 0.10486211627721786, "mean_pht_probability": 0.018749801442027092, "mean_ptcr_probability": 0.1932152658700943, "phla_cross_entropy": 0.32373688763085434, "pht_cross_entropy": 0.058642750579529467, "ptcr_cross_entropy": 0.3412312762565097, "samples": 15491}, "010": {"mean_phla_probability": 0.1391753852367401, "mean_pht_probability": 0.06088738515973091, "mean_ptcr_probability": 0.503976583480835, "phla_cross_entropy": 0.5785822871004526, "pht_cross_entropy": 0.28875184405063, "ptcr_cross_entropy": 1.3318739041086087, "samples": 533}, "100": {"mean_phla_probability": 0.9892578125, "mean_pht_probability": 0.1911030411720276, "mean_ptcr_probability": 0.2265625, "phla_cross_entropy": 0.01090422662346454, "pht_cross_entropy": 0.3592498864221312, "ptcr_cross_entropy": 0.35206670042173405, "samples": 4}, "111": {"mean_phla_probability": 0.9903338551521301, "mean_pht_probability": 0.8362510800361633, "mean_ptcr_probability": 0.7777779698371887, "phla_cross_entropy": 0.02755799910809028, "pht_cross_entropy": 0.6612590738398691, "ptcr_cross_entropy": 0.5228849265918589, "samples": 16027}}, "constraint_components": {"hierarchy_negative": 0.002832236001268029, "hierarchy_positive": 0.0007068374543450773, "pt_negative": 0.0060693733394145966, "pt_positive": 0.04881061613559723}, "phla": {"accuracy": 0.9491810949929809, "aupr": 0.9928744403322783, "auroc": 0.9941797155492511, "f1": 0.9513658755038066, "loss": 0.1475981604493254, "mcc": 0.901972733485758, "performance_avg": 0.9579147719728149, "precision": 0.9123339441136051, "recall": 0.9938868442392864, "samples": 32055, "sensitivity": 0.9938868442392864, "specificity": 0.9044558162755866}, "pht": {"accuracy": 0.9150522539385432, "aupr": 0.9869728770465797, "auroc": 0.9866245138269525, "f1": 0.9090909090909091, "loss": 0.3596028362602028, "mcc": 0.8373293187679142, "performance_avg": 0.9270139745341798, "precision": 0.9776676719804682, "recall": 0.849503962064017, "samples": 32055, "sensitivity": 0.849503962064017, "specificity": 0.9805964562016471}, "pht_selected": {"accuracy": 0.9432849789424427, "aupr": 0.9869728770465797, "auroc": 0.9866245138269525, "f1": 0.9427509761934753, "loss": 0.0, "mcc": 0.8867231760655853, "performance_avg": 0.9492713044150071, "precision": 0.9516816072223282, "recall": 0.9339863979534535, "samples": 32055, "sensitivity": 0.9339863979534535, "specificity": 0.9525829797853755}, "ptcr": {"accuracy": 0.8191857744501638, "aupr": 0.9191288843770095, "auroc": 0.8924501767751168, "f1": 0.8174603174603174, "loss": 0.4476321215489003, "mcc": 0.641329057570607, "performance_avg": 0.8179108421266429, "precision": 0.8542654028436019, "recall": 0.783695652173913, "samples": 32055, "sensitivity": 0.783695652173913, "specificity": 0.8571151984511133}, "violations": {"hierarchy_negative": 0.013791812281577633, "hierarchy_positive": 0.0075497597803706245, "overall": 0.08280244757994747, "pt_negative": 0.024136818328493063, "pt_positive": 0.2837711362076496}}`。
- 本次没有重新训练、修改 checkpoint，且没有使用 test 标签进行阈值选择或模型选择。
- 每个 checkpoint 的 unique peptide/HLA/TCR 分别只编码一次，再复用 residue-level 表示完成行级推理。
- 原始 JSON：[`stage8_best_20260901_214051_422800_PT_PHT.json`](stage8_benchmark_results_20260903/stage8_best_20260901_214051_422800_PT_PHT.json)
- 本报告：[`PLM_Stage8_best_PT_PHT_benchmark_test_results_20260903.md`](PLM_Stage8_best_PT_PHT_benchmark_test_results_20260903.md)

