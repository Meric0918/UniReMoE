# Stage Adapter PH/PT/PHT benchmark test 评测结果

## 1. 结论摘要

指定的 Stage Adapter best checkpoint 已在本次指定数据目录中的全部 `test_*.csv` 上完成评测，共计 **393,267** 条有效测试样本。

三项主任务的 benchmark 宏平均 performance average 再等权平均为 **0.664038**。

- PH/PT 使用源检查点中已训练并冻结的 `phla` / `ptcr` head。
- PHT 主结果使用训练后的 Stage Adapter `pht` head；同时补充冻结 `phla`/`ptcr` head 在 PHT 行上的投影结果。
- 主表统一使用固定阈值 `0.5`；验证集选择的 PHT 阈值只作为补充，未使用 test 标签重新选阈值。

## 2. 模型与评测设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`stage_adapter_best_20261007_205705_808542.pt`](../../../../trained_model/PLM_StageAdapter/fold_1/seed_3407/stage_adapter_best_20261007_205705_808542.pt) |
| Checkpoint SHA-256 | `5f2d9070f1b6cca0d935c9d9f9527ff2fc3a2b709a16b849960a0c85f200fd0b` |
| 类型 / 角色 | `stage_adapter_v2_full_model` / `best` |
| 初始化模式 | `checkpoint` |
| 训练 phase / round | `stage_adapter_b` / `12` |
| Validation PHT performance avg | `0.913582` |
| Validation-selected PHT threshold | `0.00506591796875` |
| 配置 | [`plm_stage_adapter_20261007_stage2b_round16.yaml`](../../../../configs/plm_stage_adapter_20261007_stage2b_round16.yaml) |
| PH 数据目录 | `/home/mjp/Project/PLM/UnifyImmun/dataset/PH/new` |
| PT 数据目录 | `/home/mjp/Project/PLM/UnifyImmun/dataset/PT/new` |
| PHT 数据目录 | `/home/mjp/Project/PLM/UnifyImmun/dataset/PHT/benchmarks` |
| 设备 / 精度 | `cuda` / `bf16` autocast |
| unique-sequence / row batch | `256` / `2048` |

## 3. 数据审计

| 数据集 | Source CSV | 原始行数 | 有效行数 | 过滤/去重 | PH 正/负 | PT 正/负 | PHT 正/负 |
|---|---|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | [`test_unseen_hla.csv`](../../../../dataset/PH/new/test_unseen_hla.csv) | 24,696 | 24,696 | 0 | 9,271/15,425 | — | — |
| `PH/test_unseen_pair` | [`test_unseen_pair.csv`](../../../../dataset/PH/new/test_unseen_pair.csv) | 51 | 51 | 0 | 22/29 | — | — |
| `PH/test_unseen_peptide` | [`test_unseen_peptide.csv`](../../../../dataset/PH/new/test_unseen_peptide.csv) | 62,764 | 62,764 | 0 | 31,382/31,382 | — | — |
| `PHT/test_unseen_hla` | [`test_unseen_hla.csv`](../../../../dataset/PHT/benchmarks/test_unseen_hla.csv) | 19,670 | 19,670 | 0 | 9,836/9,834 | 9,836/9,834 | 9,835/9,835 |
| `PHT/test_unseen_peptide` | [`test_unseen_peptide.csv`](../../../../dataset/PHT/benchmarks/test_unseen_peptide.csv) | 14,118 | 14,118 | 0 | 7,060/7,058 | 7,060/7,058 | 7,059/7,059 |
| `PHT/test_unseen_tcr` | [`test_unseen_tcr.csv`](../../../../dataset/PHT/benchmarks/test_unseen_tcr.csv) | 7,698 | 7,698 | 0 | 3,850/3,848 | 3,850/3,848 | 3,849/3,849 |
| `PHT/test_unseen_triple` | [`test_unseen_triple.csv`](../../../../dataset/PHT/benchmarks/test_unseen_triple.csv) | 660 | 660 | 0 | 331/329 | 658/2 | 330/330 |
| `PT/test_unseen_pair` | [`test_unseen_pair.csv`](../../../../dataset/PT/new/test_unseen_pair.csv) | 8,207 | 8,207 | 0 | — | 3,113/5,094 | — |
| `PT/test_unseen_peptide` | [`test_unseen_peptide.csv`](../../../../dataset/PT/new/test_unseen_peptide.csv) | 112,443 | 112,443 | 0 | — | 41,541/70,902 | — |
| `PT/test_unseen_tcr` | [`test_unseen_tcr.csv`](../../../../dataset/PT/new/test_unseen_tcr.csv) | 142,960 | 142,960 | 0 | — | 71,480/71,480 | — |
| **合计** | — | **393,267** | **393,267** | **0** | — | — | — |

测试集包含 **65,268** 个唯一 peptide、**122** 个唯一 HLA 和 **131,892** 个唯一 TCR。

## 4. 主结果：固定阈值 0.5

| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss | Performance avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PH/test_unseen_hla` | `phla` | 24,696 | 0.722916 | 0.571696 | 0.672700 | 0.248793 | 0.401037 | 0.640625 | 0.291878 | 0.901588 | 0.892688 | 0.523428 |
| `PH/test_unseen_pair` | `phla` | 51 | 0.348746 | 0.334419 | 0.509804 | -0.154020 | 0.074074 | 0.200000 | 0.045455 | 0.862069 | 1.950785 | 0.222605 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.967089 | 0.964784 | 0.910857 | 0.821714 | 0.910915 | 0.910320 | 0.911510 | 0.910203 | 0.230240 | 0.915072 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.780184 | 0.518387 | 0.426487 | -0.230534 | 0.067146 | 0.179885 | 0.041277 | 0.811775 | 0.746729 | 0.312334 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.992461 | 0.994314 | 0.940519 | 0.884815 | 0.943149 | 0.903295 | 0.986682 | 0.894346 | 0.155524 | 0.951051 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.397768 | 0.420530 | 0.461718 | -0.163767 | 0.035350 | 0.170026 | 0.019725 | 0.903711 | 5.748252 | 0.230320 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.995082 | 0.713079 | 0.945460 | 0.895180 | 0.948001 | 0.905911 | 0.994193 | 0.896713 | 0.137541 | 0.899360 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.587848 | 0.706108 | 0.657246 | 0.387503 | 0.515955 | 0.878107 | 0.365297 | 0.949277 | 1.208975 | 0.570932 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.979004 | 0.979806 | 0.876328 | 0.770589 | 0.861473 | 0.979080 | 0.769089 | 0.983567 | 0.687012 | 0.893440 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.986972 | 0.978473 | 0.944011 | 0.890785 | 0.946145 | 0.911630 | 0.983377 | 0.904626 | 0.157676 | 0.949277 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.936996 | 0.954054 | 0.881268 | 0.763165 | 0.878844 | 0.897401 | 0.861039 | 0.901507 | 0.313259 | 0.882865 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.998921 | 0.998919 | 0.981813 | 0.963796 | 0.981642 | 0.990998 | 0.972460 | 0.991167 | 0.053886 | 0.985018 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.680447 | 0.522823 | 0.619697 | 0.271282 | 0.506876 | 0.724719 | 0.389728 | 0.851064 | 1.091115 | 0.520225 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.708967 | 0.998853 | 0.390909 | 0.043887 | 0.560175 | 1.000000 | 0.389058 | 1.000000 | 1.863649 | 0.540558 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.742668 | 0.646709 | 0.557576 | 0.167774 | 0.304762 | 0.711111 | 0.193939 | 0.921212 | 3.771810 | 0.483898 |
| `PT/test_unseen_pair` | `ptcr` | 8,207 | 0.890294 | 0.872387 | 0.848788 | 0.674567 | 0.783156 | 0.858621 | 0.719884 | 0.927562 | 0.398093 | 0.813838 |
| `PT/test_unseen_peptide` | `ptcr` | 112,443 | 0.818964 | 0.790909 | 0.787350 | 0.530894 | 0.678057 | 0.769325 | 0.606148 | 0.893515 | 0.511750 | 0.721235 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.914038 | 0.914066 | 0.837423 | 0.674857 | 0.837890 | 0.835489 | 0.840305 | 0.834541 | 0.374896 | 0.835655 |

## 5. benchmark 宏平均

宏平均为同一数据族/head 下各 benchmark 的简单算术平均，不按样本量加权。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.679584 | 0.623633 | 0.697787 | 0.305496 | 0.462009 | 0.553702 |
| `PHT` / `phla` | 0.860671 | 0.683190 | 0.733914 | 0.456678 | 0.617042 | 0.670299 |
| `PHT` / `pht` | 0.779590 | 0.761491 | 0.719359 | 0.434598 | 0.545807 | 0.648169 |
| `PHT` / `ptcr` | 0.806568 | 0.913332 | 0.717485 | 0.519842 | 0.724531 | 0.736352 |
| `PT` / `ptcr` | 0.874432 | 0.859121 | 0.824520 | 0.626772 | 0.766368 | 0.790243 |

## 6. PHT 验证阈值补充

Checkpoint 在 validation 上选择的 PHT 阈值为 `0.00506591796875`。该阈值原样用于下表，没有通过 test 标签调整。

| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `0.5` | 0.461718 | -0.163767 | 0.035350 | 0.170026 | 0.019725 | 0.903711 |
| `PHT/test_unseen_hla` | `0.00506591796875` | 0.399542 | -0.295647 | 0.051706 | 0.122901 | 0.032740 | 0.766345 |
| `PHT/test_unseen_peptide` | `0.5` | 0.876328 | 0.770589 | 0.861473 | 0.979080 | 0.769089 | 0.983567 |
| `PHT/test_unseen_peptide` | `0.00506591796875` | 0.902394 | 0.807810 | 0.897986 | 0.940456 | 0.859187 | 0.945601 |
| `PHT/test_unseen_tcr` | `0.5` | 0.981813 | 0.963796 | 0.981642 | 0.990998 | 0.972460 | 0.991167 |
| `PHT/test_unseen_tcr` | `0.00506591796875` | 0.978176 | 0.957022 | 0.978577 | 0.960932 | 0.996882 | 0.959470 |
| `PHT/test_unseen_triple` | `0.5` | 0.557576 | 0.167774 | 0.304762 | 0.711111 | 0.193939 | 0.921212 |
| `PHT/test_unseen_triple` | `0.00506591796875` | 0.607576 | 0.241969 | 0.491159 | 0.698324 | 0.378788 | 0.836364 |

## 7. 结果解读与复核

- PH 主任务宏平均 performance average：`0.553702`。
- PT 主任务宏平均 performance average：`0.790243`。
- PHT 主任务宏平均 performance average：`0.648169`。
- PHT head 表现最高的单个 split 是 `PHT/test_unseen_tcr`，performance average 为 `0.985018`。
- `test_unseen_pair` 等小样本 split 的指标方差可能较大，应结合样本数、正负比例和混淆矩阵解读。
- 本次只执行推理；没有重新训练、修改 checkpoint，也没有使用测试标签做阈值或模型选择。
- 每个唯一 peptide/HLA/TCR 只编码一次，随后复用 residue-level 表示完成全部行级推理。
- 原始 JSON：[`summary.json`](summary.json)
- 本报告：[`summary.md`](summary.md)

