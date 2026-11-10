# Stage 8 `full` 在 PH/PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本报告使用 Stage-8 V2 的 best checkpoint，对 `dataset/PH/benchmarks`、`dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **450,756** 条有效样本。

- PH benchmark 使用 `phla` head；PT benchmark 使用 `ptcr` head。
- PHT benchmark 同时报告 `phla`、`ptcr` 和 `pht` 三个 head。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集选择的 PHT 阈值仅作补充，不使用 test 标签调参。
- PHT 约束统计使用训练配置中的 margin，仅用于报告测试时的约束违反率和 hinge 强度。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Condition | `full` |
| Checkpoint | [`stage8_best_20260827_110610_283819.pt`](../../../trained_model/PLM_Stage8/fold_1/seed_42/full/stage8_best_20260827_110610_283819.pt) |
| Checkpoint 类型 | `stage8_online_model`，format `2` |
| Checkpoint SHA-256 | `575af59f232ed1e8d86aaba4286953eb0d8194243054f492092c1725effad624` |
| Best Stage8B round | `11`；selection score `0.87814251824617` |
| Validation joint score | `0.87814251824617` |
| 配置 | [`plm_stage8.yaml`](../../../configs/plm_stage8.yaml) |
| 设备 | `cuda:0` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `256` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.032958984375` |
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
| `PH/test_unseen_hla` | `phla` | 47,748 | 0.820520 | 0.802157 | 0.716554 | 0.453558 | 0.667142 | 0.807995 | 0.568108 | 0.865000 | 0.801232 |
| `PH/test_unseen_pair` | `phla` | 64 | 0.590820 | 0.671106 | 0.578125 | 0.184428 | 0.425532 | 0.666667 | 0.312500 | 0.843750 | 1.566813 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.929487 | 0.924928 | 0.858087 | 0.716754 | 0.855178 | 0.873079 | 0.837996 | 0.878179 | 0.393239 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.639868 | 0.436885 | 0.477428 | -0.089506 | 0.080014 | 0.334331 | 0.045445 | 0.909498 | 1.467579 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.993069 | 0.994463 | 0.941179 | 0.885807 | 0.943668 | 0.905447 | 0.985258 | 0.897092 | 0.147304 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.867060 | 0.759948 | 0.507117 | 0.061853 | 0.040004 | 0.765152 | 0.020539 | 0.993696 | 3.197664 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.998072 | 0.998219 | 0.967984 | 0.937356 | 0.968836 | 0.943847 | 0.995184 | 0.940776 | 0.101491 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.858762 | 0.901471 | 0.819592 | 0.672932 | 0.786200 | 0.964970 | 0.663314 | 0.975914 | 0.534253 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.986645 | 0.987341 | 0.917127 | 0.844231 | 0.910249 | 0.992638 | 0.840487 | 0.993767 | 0.525460 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.997632 | 0.997589 | 0.974279 | 0.949260 | 0.974771 | 0.956728 | 0.993506 | 0.955042 | 0.074358 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.951743 | 0.965192 | 0.894778 | 0.789629 | 0.894090 | 0.900211 | 0.888052 | 0.901507 | 0.262216 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.999337 | 0.999361 | 0.979865 | 0.960236 | 0.979533 | 0.995972 | 0.963627 | 0.996103 | 0.055198 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.805159 | 0.831552 | 0.759091 | 0.577377 | 0.692456 | 0.962366 | 0.540785 | 0.978723 | 0.867702 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.714666 | 0.998970 | 0.463636 | 0.050946 | 0.632017 | 1.000000 | 0.462006 | 1.000000 | 1.042567 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.743017 | 0.808216 | 0.768182 | 0.601127 | 0.700587 | 0.988950 | 0.542424 | 0.993939 | 3.173604 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.853959 | 0.871728 | 0.718999 | 0.499109 | 0.630393 | 0.920718 | 0.479268 | 0.958731 | 0.634639 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.738996 | 0.745534 | 0.618052 | 0.296930 | 0.451851 | 0.799937 | 0.314848 | 0.921257 | 0.824196 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.890392 | 0.886553 | 0.797111 | 0.599677 | 0.782475 | 0.843310 | 0.729827 | 0.864396 | 0.435373 |

## 5. benchmark 宏平均

同一数据族和 head 内为各 benchmark 的简单算术平均，不是拼接样本后的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.780276 | 0.799397 | 0.717589 | 0.451580 | 0.649284 | 0.679625 |
| `PHT` / `phla` | 0.860183 | 0.816061 | 0.794695 | 0.593621 | 0.679019 | 0.748716 |
| `PHT` / `pht` | 0.899015 | 0.888716 | 0.793073 | 0.616862 | 0.657593 | 0.771052 |
| `PHT` / `ptcr` | 0.879560 | 0.965024 | 0.779796 | 0.599828 | 0.813994 | 0.807641 |
| `PT` / `ptcr` | 0.827782 | 0.834605 | 0.711388 | 0.465238 | 0.621573 | 0.692117 |

## 6. PHT checkpoint 阈值补充

checkpoint 在 validation 上为 PHT head 记录的阈值为 `0.032958984375`。下表与固定阈值 `0.5` 对比；该阈值未由任何 test 标签重新选择。

| Benchmark | Head | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `pht` | `0.5` | 0.507117 | 0.061853 | 0.040004 | 0.765152 | 0.020539 | 0.993696 |
| `PHT/test_unseen_hla` | `pht` | `0.032958984375` | 0.497204 | -0.015654 | 0.056658 | 0.457627 | 0.030198 | 0.964209 |
| `PHT/test_unseen_peptide` | `pht` | `0.5` | 0.917127 | 0.844231 | 0.910249 | 0.992638 | 0.840487 | 0.993767 |
| `PHT/test_unseen_peptide` | `pht` | `0.032958984375` | 0.933772 | 0.870835 | 0.930766 | 0.975023 | 0.890353 | 0.977192 |
| `PHT/test_unseen_tcr` | `pht` | `0.5` | 0.979865 | 0.960236 | 0.979533 | 0.995972 | 0.963627 | 0.996103 |
| `PHT/test_unseen_tcr` | `pht` | `0.032958984375` | 0.986100 | 0.972236 | 0.986160 | 0.981968 | 0.990387 | 0.981813 |
| `PHT/test_unseen_triple` | `pht` | `0.5` | 0.768182 | 0.601127 | 0.700587 | 0.988950 | 0.542424 | 0.993939 |
| `PHT/test_unseen_triple` | `pht` | `0.032958984375` | 0.753030 | 0.557973 | 0.687140 | 0.937173 | 0.542424 | 0.963636 |

## 7. PHT 约束统计

违反率的分母只包含该约束适用的 label type；`hierarchy_*` 比较 `pht` 与 `phla`，`pt_*` 比较 `pht` 与 `ptcr`。

| Benchmark | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall | Hinge overall |
|---|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | 0.018302 | 0.008847 | 0.004779 | 0.014440 | 0.011592 | 0.001020 |
| `PHT/test_unseen_peptide` | 0.004817 | 0.005809 | 0.422156 | 0.014877 | 0.111922 | 0.022321 |
| `PHT/test_unseen_tcr` | 0.020265 | 0.004418 | 0.212783 | 0.009096 | 0.061647 | 0.010363 |
| `PHT/test_unseen_triple` | 0.000000 | 0.018237 | 0.015152 | 0.000000 | 0.011100 | 0.000790 |

## 8. 结果说明与可复核产物

- 全部 benchmark/head 中 performance average 最高的是 `PHT/test_unseen_tcr` 的 `pht` head：`0.983666`。
- PHT head 中 performance average 最高的是 `PHT/test_unseen_tcr`：`0.983666`。
- `test_unseen_pair` 样本量较小，指标方差可能较大，应结合样本数和混淆矩阵解读。
- checkpoint validation metrics 摘要：`{"by_type": {"000": {"mean_phla_probability": 0.0815160796046257, "mean_pht_probability": 0.010107756592333317, "mean_ptcr_probability": 0.16842716932296753, "phla_cross_entropy": 0.18756169182440552, "pht_cross_entropy": 0.06504508564868494, "ptcr_cross_entropy": 0.2656473611006562, "samples": 15491}, "010": {"mean_phla_probability": 0.10925159603357315, "mean_pht_probability": 0.035954251885414124, "mean_ptcr_probability": 0.43787893652915955, "phla_cross_entropy": 0.2526981403802815, "pht_cross_entropy": 0.551273035729475, "ptcr_cross_entropy": 1.5587467799801853, "samples": 533}, "100": {"mean_phla_probability": 0.912109375, "mean_pht_probability": 0.0011528730392456055, "mean_ptcr_probability": 0.07244873046875, "phla_cross_entropy": 0.09930016446572942, "pht_cross_entropy": 0.0011545724212763636, "ptcr_cross_entropy": 0.07553534313963965, "samples": 4}, "111": {"mean_phla_probability": 0.9791033267974854, "mean_pht_probability": 0.8235910534858704, "mean_ptcr_probability": 0.7153913974761963, "phla_cross_entropy": 0.04266245763030468, "pht_cross_entropy": 0.9377973035761865, "ptcr_cross_entropy": 0.6273189890554184, "samples": 16027}}, "constraint_components": {"hierarchy_negative": 0.0017808714183047414, "hierarchy_positive": 0.0009917495772242546, "pt_negative": 0.0018025035969913006, "pt_positive": 0.07281434535980225}, "phla": {"accuracy": 0.9604741849945406, "aupr": 0.9947160918352278, "auroc": 0.9947366393488801, "f1": 0.9614483493077742, "loss": 0.11247979394546483, "mcc": 0.9221055651803534, "performance_avg": 0.9666961661333552, "precision": 0.9385172864441013, "recall": 0.9855280394236168, "samples": 32055, "sensitivity": 0.9855280394236168, "specificity": 0.9354093859211183}, "pht": {"accuracy": 0.9131804710653564, "aupr": 0.9855068196642482, "auroc": 0.9851274709733414, "f1": 0.9059002535925613, "loss": 0.4885286570078794, "mcc": 0.8364258088759237, "performance_avg": 0.9252281648342862, "precision": 0.988780631827576, "recall": 0.8358395208086354, "samples": 32055, "sensitivity": 0.8358395208086354, "specificity": 0.9905165959570751}, "pht_selected": {"accuracy": 0.9410700358758384, "aupr": 0.9855068196642482, "auroc": 0.9851274709733414, "f1": 0.9388297011107153, "loss": 0.0, "mcc": 0.8845118469881842, "performance_avg": 0.9470091749224656, "precision": 0.9758987478120371, "recall": 0.9044737006301865, "samples": 32055, "sensitivity": 0.9044737006301865, "specificity": 0.977664087846269}, "ptcr": {"accuracy": 0.7851193261581656, "aupr": 0.9094113085167547, "auroc": 0.8852820042463441, "f1": 0.7690915186054308, "loss": 0.46804527862267975, "mcc": 0.5849960810456104, "performance_avg": 0.7867800477144611, "precision": 0.8644310474755087, "recall": 0.6926932367149758, "samples": 32055, "sensitivity": 0.6926932367149758, "specificity": 0.8838980316231042}, "violations": {"hierarchy_negative": 0.007675986020968547, "hierarchy_positive": 0.012291757658950521, "overall": 0.09853239582841772, "pt_negative": 0.014391739270732494, "pt_positive": 0.35696012978099456}}`。
- 本次没有重新训练、修改 checkpoint，且没有使用 test 标签进行阈值选择或模型选择。
- 每个 checkpoint 的 unique peptide/HLA/TCR 分别只编码一次，再复用 residue-level 表示完成行级推理。
- 原始 JSON：[`full.json`](full.json)
- 本报告：[`full.md`](full.md)

