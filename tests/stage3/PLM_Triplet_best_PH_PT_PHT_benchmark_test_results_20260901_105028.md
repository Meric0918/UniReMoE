# Stage 3 `triplet_best` 在 PH/PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本次使用指定的 Stage 3 `triplet_best` 多任务 checkpoint，对 `dataset/PH/benchmarks`、`dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **450,756** 条有效样本；所有数据行均通过当前模型的 peptide、HLA、TCR 长度及 canonical 残基约束。

- PH benchmark 使用 Stage 3 的 pHLA head；PT benchmark 使用 pTCR head。
- PHT benchmark 同时报告 pHLA、pTCR 和 PHT 三个 head，标签分别对应 CSV 中的 `PH`、`PT`、`PHT` 列。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集上为 PHT head 选择的阈值另作补充比较，没有在任何 test 数据上重新选择阈值。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`triplet_best_20260831_205045_885619.pt`](../../trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260831_205045_885619.pt) |
| Checkpoint 类型 | `stage3_multitask_model` |
| Checkpoint SHA-256 | `6c2993adf918c994ab66df17fe62d11a432688c131e992a7cc76dcf016d6952f` |
| Checkpoint best round | `6`；best selection score `0.9316408952095859` |
| 配置 | [`plm_triplet.yaml`](../../configs/plm_triplet.yaml) |
| 设备 | `cuda` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `512` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.0074462890625` |

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
| `PH/test_unseen_hla` | `phla` | 47,748 | 0.766019 | 0.800135 | 0.645744 | 0.375677 | 0.482516 | 0.894814 | 0.330318 | 0.961171 | 1.652473 |
| `PH/test_unseen_pair` | `phla` | 64 | 0.599609 | 0.680969 | 0.593750 | 0.321634 | 0.315789 | 1.000000 | 0.187500 | 1.000000 | 2.729659 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.844583 | 0.861949 | 0.756803 | 0.546817 | 0.706428 | 0.890986 | 0.585208 | 0.928398 | 1.018685 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.994545 | 0.995060 | 0.968988 | 0.939023 | 0.968242 | 0.992211 | 0.945405 | 0.992577 | 0.124420 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.987293 | 0.990914 | 0.960447 | 0.920973 | 0.960192 | 0.966522 | 0.953945 | 0.966951 | 0.131898 |
| `PHT/test_unseen_hla` | `triplet` | 19,670 | 0.995417 | 0.995804 | 0.970005 | 0.941004 | 0.969300 | 0.992646 | 0.947026 | 0.992984 | 0.211751 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.998069 | 0.998146 | 0.981513 | 0.963102 | 0.981631 | 0.975521 | 0.987819 | 0.975205 | 0.064466 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.764653 | 0.837409 | 0.760731 | 0.558333 | 0.708743 | 0.905685 | 0.582153 | 0.939360 | 0.800537 |
| `PHT/test_unseen_peptide` | `triplet` | 14,118 | 0.991422 | 0.991604 | 0.935402 | 0.874752 | 0.932183 | 0.981061 | 0.887944 | 0.982859 | 0.234160 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.997794 | 0.997571 | 0.986880 | 0.973769 | 0.986854 | 0.989043 | 0.984675 | 0.989085 | 0.046469 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.959360 | 0.970849 | 0.927644 | 0.858869 | 0.924207 | 0.970563 | 0.882078 | 0.973233 | 0.222481 |
| `PHT/test_unseen_tcr` | `triplet` | 7,698 | 0.998825 | 0.998951 | 0.992076 | 0.984154 | 0.992067 | 0.993229 | 0.990907 | 0.993245 | 0.038256 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.759787 | 0.808934 | 0.765152 | 0.594842 | 0.697856 | 0.983516 | 0.540785 | 0.990881 | 1.763159 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.870441 | 0.999578 | 0.450000 | 0.049564 | 0.619098 | 1.000000 | 0.448328 | 1.000000 | 1.452103 |
| `PHT/test_unseen_triple` | `triplet` | 660 | 0.582250 | 0.704261 | 0.671212 | 0.354382 | 0.622609 | 0.730612 | 0.542424 | 0.800000 | 3.458684 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.725922 | 0.773677 | 0.682402 | 0.420940 | 0.576838 | 0.864025 | 0.432938 | 0.931867 | 0.905776 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.602234 | 0.645880 | 0.581492 | 0.228344 | 0.355949 | 0.771990 | 0.231298 | 0.931685 | 1.171181 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.728174 | 0.760767 | 0.635115 | 0.348845 | 0.466386 | 0.867560 | 0.318914 | 0.951315 | 0.902540 |

## 5. 各数据族的 benchmark 宏平均

以下为同一数据族内各 benchmark 的简单算术平均，不是将样本拼接后重新计算的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.736737 | 0.781018 | 0.665433 | 0.414709 | 0.501578 |
| `PHT` / `phla` | 0.937549 | 0.949928 | 0.925633 | 0.867684 | 0.908646 |
| `PHT` / `ptcr` | 0.895437 | 0.949688 | 0.774705 | 0.596935 | 0.803060 |
| `PHT` / `triplet` | 0.891979 | 0.922655 | 0.892174 | 0.788573 | 0.879040 |
| `PT` / `ptcr` | 0.685443 | 0.726774 | 0.633003 | 0.332709 | 0.466391 |

## 6. PHT head 的 checkpoint 阈值补充

checkpoint 在验证集上为 PHT head 记录的阈值为 `0.0074462890625`。下表将它与 固定阈值 `0.5` 对比；该阈值只在验证集上确定，未使用任何 test 标签重新调节。

| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` / `PHT` | `0.5` | 0.970005 | 0.941004 | 0.969300 | 0.992646 | 0.947026 | 0.992984 |
| `PHT/test_unseen_hla` / `PHT` | `0.0074462890625` | 0.963498 | 0.927427 | 0.962932 | 0.978081 | 0.948246 | 0.978749 |
| `PHT/test_unseen_peptide` / `PHT` | `0.5` | 0.935402 | 0.874752 | 0.932183 | 0.981061 | 0.887944 | 0.982859 |
| `PHT/test_unseen_peptide` / `PHT` | `0.0074462890625` | 0.976555 | 0.953166 | 0.976682 | 0.971413 | 0.982009 | 0.971101 |
| `PHT/test_unseen_tcr` / `PHT` | `0.5` | 0.992076 | 0.984154 | 0.992067 | 0.993229 | 0.990907 | 0.993245 |
| `PHT/test_unseen_tcr` / `PHT` | `0.0074462890625` | 0.990517 | 0.981090 | 0.990567 | 0.985347 | 0.995843 | 0.985191 |
| `PHT/test_unseen_triple` / `PHT` | `0.5` | 0.671212 | 0.354382 | 0.622609 | 0.730612 | 0.542424 | 0.800000 |
| `PHT/test_unseen_triple` / `PHT` | `0.0074462890625` | 0.642424 | 0.290722 | 0.602694 | 0.678030 | 0.542424 | 0.742424 |

## 7. 结果说明

- 主表中 AUROC 最高的是 `PHT/test_unseen_tcr` 的 `triplet` head：`0.998825`；对应 AUPR 为 `0.998951`。
- PHT benchmark 的 PHT head 中，AUROC 最高的是 `PHT/test_unseen_tcr`：`0.998825`；PHT 正类比例差异会影响 AUPR 与阈值指标的直接比较。
- `test_unseen_pair` 等小样本 benchmark 的指标方差较大，应结合样本数和混淆矩阵谨慎解读。

## 8. 评测方法与可复核产物

- 使用项目现有 Stage 3 progressive model、checkpoint 加载、encoder 和 `compute_binary_metrics` 实现；没有重新训练、修改 checkpoint 或使用 test 数据调参。
- 对全部测试文件的唯一 peptide/HLA/TCR 分别编码一次，再复用 residue-level 表示完成行级推理；没有使用 Stage 1 embedding cache。
- PH/PT pair 的标签来自各自 CSV 的 `label`；PHT 三元测试的三个标签来自 `PH`、`PT`、`PHT` 列。
- 原始 JSON：[`evaluation_stage3_triplet_best_ph_pt_pht_benchmark_test_20260901_105028.json`](../../trained_model/PLM_Triplet/fold_1/seed_42/evaluation_stage3_triplet_best_ph_pt_pht_benchmark_test_20260901_105028.json)
- 本报告：[`PLM_Triplet_best_PH_PT_PHT_benchmark_test_results_20260901_105028.md`](PLM_Triplet_best_PH_PT_PHT_benchmark_test_results_20260901_105028.md)
