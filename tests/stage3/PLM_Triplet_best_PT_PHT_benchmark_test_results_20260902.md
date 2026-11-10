# Stage 3 `triplet_best` 在 PT/PHT benchmark test 上的评测结果

## 1. 结论摘要

本次使用 Stage 3 最优部署 checkpoint，对 `dataset/PT/benchmarks` 和 `dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。共评测 **340,180** 条有效样本；所有数据行均通过当前模型的 peptide、HLA、TCR 长度及 canonical 残基约束。

- PT benchmark 使用 Stage 3 的 pTCR head。
- PHT benchmark 同时报告 pHLA、pTCR 和 PHT 三个 head，标签分别对应 CSV 中的 `PH`、`PT`、`PHT` 列。
- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集上为 PHT head 选择的阈值另作补充比较，没有在任何 test 数据上重新选择阈值。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`triplet_best_20260901_214051_422010.pt`](../../trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260901_214051_422010.pt) |
| Checkpoint 类型 | `stage3_multitask_model` |
| Checkpoint SHA-256 | `fca682f59eaec6feffe2ff81672996787193bdce799a3bf1c649dec36c9e5504` |
| Checkpoint best round | `7`；best selection score `0.9331164887095177` |
| 配置 | [`plm_triplet.yaml`](../../configs/plm_triplet.yaml) |
| 设备 | `cuda` |
| 精度 | `bf16` autocast |
| unique-sequence 编码 batch | `512` |
| 行级推理 batch | `2048` |
| 固定分类阈值 | `0.5` |
| checkpoint PHT 验证阈值 | `0.062255859375` |

PT 的 pair 数据没有三元输入，因此仅调用 Stage 3 的 pTCR head；PHT 数据则使用完整三元输入并一次前向得到 pHLA、pTCR 和 PHT 三个 head 的输出。

## 3. 数据审计

| 数据集 | 原始行数 | 纳入评测 | 过滤/去重 | PH 正/负 | PT 正/负 | PHT 正/负 |
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
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.992348 | 0.993341 | 0.968531 | 0.938014 | 0.967809 | 0.990631 | 0.946015 | 0.991051 | 0.124876 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.989075 | 0.991902 | 0.959837 | 0.919788 | 0.959525 | 0.967156 | 0.952013 | 0.967663 | 0.128317 |
| `PHT/test_unseen_hla` | `triplet` | 19,670 | 0.993642 | 0.994313 | 0.966904 | 0.934562 | 0.966226 | 0.986441 | 0.946823 | 0.986985 | 0.173141 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.997899 | 0.997932 | 0.982505 | 0.965108 | 0.982631 | 0.975702 | 0.989660 | 0.975347 | 0.064509 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.784142 | 0.849127 | 0.768027 | 0.577275 | 0.715242 | 0.926143 | 0.582578 | 0.953528 | 0.771267 |
| `PHT/test_unseen_peptide` | `triplet` | 14,118 | 0.990257 | 0.990312 | 0.940147 | 0.883168 | 0.937634 | 0.978737 | 0.899844 | 0.980450 | 0.203147 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.997372 | 0.997190 | 0.984022 | 0.968049 | 0.983999 | 0.985666 | 0.982338 | 0.985707 | 0.052855 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.958767 | 0.969748 | 0.924526 | 0.852731 | 0.920877 | 0.967936 | 0.878182 | 0.970894 | 0.231517 |
| `PHT/test_unseen_tcr` | `triplet` | 7,698 | 0.997978 | 0.998122 | 0.991167 | 0.982334 | 0.991162 | 0.991678 | 0.990647 | 0.991686 | 0.038832 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.785544 | 0.821471 | 0.766667 | 0.597211 | 0.700389 | 0.983607 | 0.543807 | 0.990881 | 1.739214 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.856003 | 0.999520 | 0.437879 | 0.048360 | 0.607407 | 1.000000 | 0.436170 | 1.000000 | 1.601863 |
| `PHT/test_unseen_triple` | `triplet` | 660 | 0.596373 | 0.722895 | 0.672727 | 0.357819 | 0.623693 | 0.733607 | 0.542424 | 0.803030 | 2.940304 |
| `PT/test_unseen_pair` | `ptcr` | 10,274 | 0.726938 | 0.780827 | 0.680358 | 0.423054 | 0.567325 | 0.877701 | 0.419116 | 0.941600 | 0.952272 |
| `PT/test_unseen_peptide` | `ptcr` | 144,800 | 0.615466 | 0.656096 | 0.584261 | 0.237249 | 0.358486 | 0.784552 | 0.232320 | 0.936202 | 1.203935 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.757942 | 0.781189 | 0.645621 | 0.367720 | 0.489922 | 0.873860 | 0.340375 | 0.950867 | 0.880795 |

## 5. 各数据族的 benchmark 宏平均

以下为同一数据族内各 benchmark 的简单算术平均，不是将样本拼接后重新计算的 pooled metric。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 |
|---|---:|---:|---:|---:|---:|
| `PHT` / `phla` | 0.943291 | 0.952484 | 0.925431 | 0.867095 | 0.908707 |
| `PHT` / `ptcr` | 0.896997 | 0.952574 | 0.772567 | 0.599539 | 0.800763 |
| `PHT` / `triplet` | 0.894562 | 0.926410 | 0.892736 | 0.789471 | 0.879679 |
| `PT` / `ptcr` | 0.700115 | 0.739371 | 0.636747 | 0.342674 | 0.471911 |

## 6. PHT head 的 checkpoint 阈值补充

checkpoint 在验证集上为 PHT head 记录的阈值为 `0.062255859375`。下表将它与 固定阈值 `0.5` 对比；该阈值只在验证集上确定，未使用任何 test 标签重新调节。

| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` / `PHT` | `0.5` | 0.966904 | 0.934562 | 0.966226 | 0.986441 | 0.946823 | 0.986985 |
| `PHT/test_unseen_hla` / `PHT` | `0.062255859375` | 0.960803 | 0.921888 | 0.960313 | 0.972477 | 0.948449 | 0.973157 |
| `PHT/test_unseen_peptide` / `PHT` | `0.5` | 0.940147 | 0.883168 | 0.937634 | 0.978737 | 0.899844 | 0.980450 |
| `PHT/test_unseen_peptide` / `PHT` | `0.062255859375` | 0.961326 | 0.922730 | 0.961072 | 0.967418 | 0.954809 | 0.967842 |
| `PHT/test_unseen_tcr` / `PHT` | `0.5` | 0.991167 | 0.982334 | 0.991162 | 0.991678 | 0.990647 | 0.991686 |
| `PHT/test_unseen_tcr` / `PHT` | `0.062255859375` | 0.989218 | 0.978476 | 0.989267 | 0.984809 | 0.993765 | 0.984671 |
| `PHT/test_unseen_triple` / `PHT` | `0.5` | 0.672727 | 0.357819 | 0.623693 | 0.733607 | 0.542424 | 0.803030 |
| `PHT/test_unseen_triple` / `PHT` | `0.062255859375` | 0.659091 | 0.327214 | 0.614065 | 0.707510 | 0.542424 | 0.775758 |

## 7. 结果分析

- 主表中 AUROC 最高的是 `PHT/test_unseen_tcr` 的 `triplet` head：`0.997978`；对应 AUPR 为 `0.998122`。
- PHT 的 unseen-HLA、unseen-peptide 和 unseen-TCR 上，triplet head 的 AUROC 分别为 `0.993642`、`0.990257` 和 `0.997978`，说明三元模型在这些受控分布外推任务上保持较强区分能力。
- `PHT/test_unseen_triple` 是主要困难集：triplet head AUROC=`0.596373`、MCC=`0.357819`；其样本量仅 660，且 PHT 标签为 330/330，不能仅依据 Accuracy 判断性能。
- PT 单独任务明显更难，pTCR head 的 AUROC 在三个 benchmark 上为 `0.726938`、`0.615466` 和 `0.757942`；其中 `PT/test_unseen_peptide` 最弱，提示对未见 peptide 的泛化仍是瓶颈。
- PHT 的 checkpoint 阈值为 `0.062255859375`，只在验证集确定。主结果统一使用固定阈值 `0.5`，避免使用 test 标签调阈值；两种阈值结果已在第 6 节并列给出。
- `test_unseen_pair` 等小样本 benchmark 的指标方差较大，应结合样本数和混淆矩阵谨慎解读。

## 8. 评测方法与可复核产物

- 使用项目现有 Stage 3 progressive model、checkpoint 加载、encoder 和 `compute_binary_metrics` 实现；没有重新训练、修改 checkpoint 或使用 test 数据调参。
- 对全部测试文件的唯一 peptide/HLA/TCR 分别编码一次，再复用 residue-level 表示完成行级推理；没有使用 Stage 1 embedding cache。
- PT pair 的标签来自各自 CSV 的 `label`；PHT 三元测试的三个标签来自 `PH`、`PT`、`PHT` 列。
- 原始 JSON：[`evaluation_stage3_triplet_best_pt_pht_benchmark_test_20260902.json`](../../trained_model/PLM_Triplet/fold_1/seed_42/evaluation_stage3_triplet_best_pt_pht_benchmark_test_20260902.json)
- 本报告：[`PLM_Triplet_best_PT_PHT_benchmark_test_results_20260902.md`](PLM_Triplet_best_PT_PHT_benchmark_test_results_20260902.md)
