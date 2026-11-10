# Stage 3 `stage3_best_joint` 在 PH/PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本次使用指定的 Stage 3 joint checkpoint，对 `dataset/PH/benchmarks`、`dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **450,756** 条有效样本；所有数据行均通过当前模型的 peptide、HLA、TCR 长度及 canonical 残基约束。

- PH benchmark 使用 Stage 3 的 pHLA head；PT benchmark 使用 pTCR head。
- PHT benchmark 同时报告 pHLA、pTCR 和 PHT 三个 head，标签分别对应 CSV 中的 `PH`、`PT`、`PHT` 列。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集上为 PHT head 选择的阈值另作补充比较，没有在任何 test 数据上重新选择阈值。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`stage3_best_joint_20260825_161220_330205.pt`](../../trained_model/PLM_Triplet/fold_1/seed_42/best_round/stage3_best_joint_20260825_161220_330205.pt) |
| Checkpoint 类型 | `stage3_joint_model` |
| Checkpoint SHA-256 | `618e6a9fd566c25d6cdf7755bd91c290f89d693ee1a4eb7d2791f9a82c96a44c` |
| Checkpoint best round | `8`；best selection score `0.9083495540238027` |
| 配置 | [`plm_triplet.yaml`](../../configs/plm_triplet.yaml) |
| 设备 | `cuda` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `512` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.0025177001953125` |

PH/PT 的 pair 数据没有三元输入，因此仅调用对应的 Stage 3 pair head；PHT 数据则使用完整三元输入并一次前向得到三个 head 的输出。

## 3. 数据审计

| 数据集 | 原始行数 | 纳入评测 | 过滤/去重 | PH 正/负 | PT 正/负 | PHT 正/负 |
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
| `PH/test_unseen_hla` | `phla` | 47,748 | 0.822674 | 0.824584 | 0.708763 | 0.455127 | 0.636406 | 0.846785 | 0.509760 | 0.907766 | 1.172979 |
| `PH/test_unseen_pair` | `phla` | 64 | 0.657715 | 0.682736 | 0.625000 | 0.320256 | 0.454545 | 0.833333 | 0.312500 | 0.937500 | 2.619468 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.873720 | 0.879806 | 0.774728 | 0.572084 | 0.738288 | 0.880758 | 0.635492 | 0.913963 | 0.916232 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.990105 | 0.969499 | 0.976309 | 0.952902 | 0.976019 | 0.988224 | 0.964111 | 0.988509 | 0.127012 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.977053 | 0.984131 | 0.951957 | 0.904367 | 0.951191 | 0.966719 | 0.936153 | 0.967765 | 0.156632 |
| `PHT/test_unseen_hla` | `triplet` | 19,670 | 0.983876 | 0.987264 | 0.966345 | 0.933336 | 0.965707 | 0.984370 | 0.947738 | 0.984952 | 0.320259 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.995855 | 0.995881 | 0.983000 | 0.966136 | 0.983144 | 0.975063 | 0.991360 | 0.974639 | 0.066723 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.810157 | 0.877647 | 0.820938 | 0.676132 | 0.787599 | 0.967988 | 0.663881 | 0.978039 | 0.616100 |
| `PHT/test_unseen_peptide` | `triplet` | 14,118 | 0.991958 | 0.990719 | 0.982434 | 0.964989 | 0.982572 | 0.974899 | 0.990367 | 0.974501 | 0.093010 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.996508 | 0.996393 | 0.988698 | 0.977402 | 0.988720 | 0.987057 | 0.990390 | 0.987006 | 0.048863 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.965358 | 0.974496 | 0.929852 | 0.862330 | 0.927027 | 0.966197 | 0.890909 | 0.968815 | 0.205670 |
| `PHT/test_unseen_tcr` | `triplet` | 7,698 | 0.998703 | 0.998675 | 0.993115 | 0.986232 | 0.993121 | 0.992220 | 0.994024 | 0.992206 | 0.030216 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.749888 | 0.803045 | 0.766667 | 0.597211 | 0.700389 | 0.983607 | 0.543807 | 0.990881 | 1.780171 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.675912 | 0.998735 | 0.409091 | 0.045585 | 0.578834 | 1.000000 | 0.407295 | 1.000000 | 1.690745 |
| `PHT/test_unseen_triple` | `triplet` | 660 | 0.636772 | 0.726992 | 0.665152 | 0.340727 | 0.618307 | 0.718876 | 0.542424 | 0.787879 | 4.160226 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.740905 | 0.793916 | 0.632373 | 0.366755 | 0.437863 | 0.929836 | 0.286354 | 0.978392 | 0.948356 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.625774 | 0.654754 | 0.564109 | 0.200085 | 0.292561 | 0.775968 | 0.180262 | 0.947956 | 1.091642 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.749440 | 0.760786 | 0.630666 | 0.325771 | 0.473485 | 0.824283 | 0.332135 | 0.929197 | 0.806854 |

## 5. 各数据族的 benchmark 宏平均

以下为同一数据族内各 benchmark 的简单算术平均，不是将样本拼接后重新计算的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.784703 | 0.795708 | 0.702830 | 0.449156 | 0.609747 |
| `PHT` / `phla` | 0.933089 | 0.941205 | 0.928669 | 0.873413 | 0.912068 |
| `PHT` / `ptcr` | 0.857120 | 0.958752 | 0.777959 | 0.622104 | 0.811162 |
| `PHT` / `triplet` | 0.902827 | 0.925913 | 0.901761 | 0.806321 | 0.889927 |
| `PT` / `ptcr` | 0.705373 | 0.736485 | 0.609049 | 0.297537 | 0.401303 |

## 6. PHT head 的 checkpoint 阈值补充

checkpoint 在验证集上为 PHT head 记录的阈值为 `0.0025177001953125`。下表将它与 固定阈值 `0.5` 对比；该阈值只在验证集上确定，未使用任何 test 标签重新调节。

| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` / `PHT` | `0.5` | 0.966345 | 0.933336 | 0.965707 | 0.984370 | 0.947738 | 0.984952 |
| `PHT/test_unseen_hla` / `PHT` | `0.0025177001953125` | 0.937214 | 0.874665 | 0.937937 | 0.927266 | 0.948856 | 0.925572 |
| `PHT/test_unseen_peptide` / `PHT` | `0.5` | 0.982434 | 0.964989 | 0.982572 | 0.974899 | 0.990367 | 0.974501 |
| `PHT/test_unseen_peptide` / `PHT` | `0.0025177001953125` | 0.974713 | 0.950472 | 0.975292 | 0.953451 | 0.998158 | 0.951268 |
| `PHT/test_unseen_tcr` / `PHT` | `0.5` | 0.993115 | 0.986232 | 0.993121 | 0.992220 | 0.994024 | 0.992206 |
| `PHT/test_unseen_tcr` / `PHT` | `0.0025177001953125` | 0.985451 | 0.971243 | 0.985641 | 0.972918 | 0.998701 | 0.972201 |
| `PHT/test_unseen_triple` / `PHT` | `0.5` | 0.665152 | 0.340727 | 0.618307 | 0.718876 | 0.542424 | 0.787879 |
| `PHT/test_unseen_triple` / `PHT` | `0.0025177001953125` | 0.628788 | 0.261506 | 0.593698 | 0.655678 | 0.542424 | 0.715152 |

## 7. 结果说明

- 主表中 AUROC 最高的是 `PHT/test_unseen_tcr` 的 `triplet` head：`0.998703`；对应 AUPR 为 `0.998675`。
- PHT benchmark 的 PHT head 中，AUROC 最高的是 `PHT/test_unseen_tcr`：`0.998703`；PHT 正类比例差异会影响 AUPR 与阈值指标的直接比较。
- `test_unseen_pair` 等小样本 benchmark 的指标方差较大，应结合样本数和混淆矩阵谨慎解读。

## 8. 评测方法与可复核产物

- 使用项目现有 Stage 3 progressive model、checkpoint 加载、encoder 和 `compute_binary_metrics` 实现；没有重新训练、修改 checkpoint 或使用 test 数据调参。
- 对全部测试文件的唯一 peptide/HLA/TCR 分别编码一次，再复用 residue-level 表示完成行级推理；没有使用 Stage 1 embedding cache。
- PH/PT pair 的标签来自各自 CSV 的 `label`；PHT 三元测试的三个标签来自 `PH`、`PT`、`PHT` 列。
- 原始 JSON：[`evaluation_stage3_best_joint_ph_pt_pht_benchmark_test_20260826_000000.json`](../../trained_model/PLM_Triplet/fold_1/seed_42/best_round/evaluation_stage3_best_joint_ph_pt_pht_benchmark_test_20260826_000000.json)
- 本报告：[`PLM_Stage3_best_joint_PH_PT_PHT_benchmark_test_results_20260826.md`](PLM_Stage3_best_joint_PH_PT_PHT_benchmark_test_results_20260826.md)

