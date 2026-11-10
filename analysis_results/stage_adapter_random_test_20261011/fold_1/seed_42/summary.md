# Stage Adapter PH/PT/PHT benchmark test 评测结果

## 1. 结论摘要

指定的 Stage Adapter best checkpoint 已在本次指定数据目录中的全部 `test_*.csv` 上完成评测，共计 **393,267** 条有效测试样本。

已训练的 PHT head 在 PHT benchmark 上的宏平均 performance average 为 **0.774839**。PH/PT 头未接受对应任务监督，因此不与 PHT 分数合并。

- PH/PT 独立预测头从随机权重初始化，未接受 PH/PT 训练；其结果仅供观察。
- PHT 主结果使用训练后的 Stage Adapter `pht` head；同时补充冻结 `phla`/`ptcr` head 在 PHT 行上的投影结果。
- 主表统一使用固定阈值 `0.5`；验证集选择的 PHT 阈值只作为补充，未使用 test 标签重新选阈值。

## 2. 模型与评测设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`stage_adapter_best_20261010_213244_248955.pt`](../../../../trained_model/PLM_StageAdapter_Random/fold_1/seed_42/stage_adapter_best_20261010_213244_248955.pt) |
| Checkpoint SHA-256 | `0cd25653e8a55b8bc076736ceafa410f731a28a7134848f639e8ffc1143681d3` |
| 类型 / 角色 | `stage_adapter_v2_full_model` / `best` |
| 初始化模式 | `random_binding` |
| 训练 phase / round | `stage_adapter_b` / `5` |
| Validation PHT performance avg | `0.912896` |
| Validation-selected PHT threshold | `0.10986328125` |
| 配置 | [`plm_stage_adapter_random.yaml`](../../../../configs/plm_stage_adapter_random.yaml) |
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
| `PH/test_unseen_hla` | `phla` | 24,696 | 0.494996 | 0.362916 | 0.616375 | -0.020921 | 0.032871 | 0.306667 | 0.017366 | 0.976402 | 0.687985 | 0.297247 |
| `PH/test_unseen_pair` | `phla` | 51 | 0.412226 | 0.355772 | 0.568627 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.691483 | 0.267325 |
| `PH/test_unseen_peptide` | `phla` | 62,764 | 0.520893 | 0.516127 | 0.499618 | -0.002465 | 0.046280 | 0.492248 | 0.024281 | 0.974954 | 0.693129 | 0.316090 |
| `PHT/test_unseen_hla` | `phla` | 19,670 | 0.834315 | 0.530945 | 0.484901 | -0.123618 | 0.000000 | 0.000000 | 0.000000 | 0.969900 | 0.689707 | 0.345309 |
| `PHT/test_unseen_hla` | `ptcr` | 19,670 | 0.711578 | 0.721507 | 0.499949 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.692554 | 0.386607 |
| `PHT/test_unseen_hla` | `pht` | 19,670 | 0.880851 | 0.784878 | 0.507168 | 0.035514 | 0.091642 | 0.584229 | 0.049720 | 0.964616 | 1.184586 | 0.460011 |
| `PHT/test_unseen_peptide` | `phla` | 14,118 | 0.441876 | 0.379083 | 0.490721 | -0.062658 | 0.024688 | 0.291667 | 0.012890 | 0.968688 | 0.693388 | 0.254742 |
| `PHT/test_unseen_peptide` | `ptcr` | 14,118 | 0.569409 | 0.566179 | 0.499929 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.698342 | 0.327103 |
| `PHT/test_unseen_peptide` | `pht` | 14,118 | 0.979886 | 0.978443 | 0.906573 | 0.816713 | 0.901999 | 0.948438 | 0.859895 | 0.953251 | 0.213372 | 0.916723 |
| `PHT/test_unseen_tcr` | `phla` | 7,698 | 0.622353 | 0.554027 | 0.491556 | -0.075729 | 0.007606 | 0.159574 | 0.003896 | 0.979470 | 0.691430 | 0.319963 |
| `PHT/test_unseen_tcr` | `ptcr` | 7,698 | 0.511111 | 0.535837 | 0.499870 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.700361 | 0.309364 |
| `PHT/test_unseen_tcr` | `pht` | 7,698 | 0.995936 | 0.996219 | 0.970382 | 0.940770 | 0.970436 | 0.968677 | 0.972201 | 0.968563 | 0.080753 | 0.974749 |
| `PHT/test_unseen_triple` | `phla` | 660 | 0.767849 | 0.687166 | 0.489394 | -0.096073 | 0.000000 | 0.000000 | 0.000000 | 0.981763 | 0.689110 | 0.369667 |
| `PHT/test_unseen_triple` | `ptcr` | 660 | 0.312310 | 0.996467 | 0.003030 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.821893 | 0.262361 |
| `PHT/test_unseen_triple` | `pht` | 660 | 0.879421 | 0.882365 | 0.748485 | 0.547105 | 0.681992 | 0.927083 | 0.539394 | 0.957576 | 0.985883 | 0.747874 |
| `PT/test_unseen_pair` | `ptcr` | 8,207 | 0.443543 | 0.339250 | 0.620690 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.672648 | 0.280696 |
| `PT/test_unseen_peptide` | `ptcr` | 112,443 | 0.463483 | 0.345462 | 0.630559 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.669959 | 0.287901 |
| `PT/test_unseen_tcr` | `ptcr` | 142,960 | 0.451151 | 0.466840 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.702483 | 0.283598 |

## 5. benchmark 宏平均

宏平均为同一数据族/head 下各 benchmark 的简单算术平均，不按样本量加权。

| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |
|---|---:|---:|---:|---:|---:|---:|
| `PH` / `phla` | 0.476038 | 0.411605 | 0.561540 | -0.007795 | 0.026384 | 0.293554 |
| `PHT` / `phla` | 0.666598 | 0.537805 | 0.489143 | -0.089520 | 0.008074 | 0.322420 |
| `PHT` / `pht` | 0.934024 | 0.910476 | 0.783152 | 0.585025 | 0.661517 | 0.774839 |
| `PHT` / `ptcr` | 0.526102 | 0.704997 | 0.375695 | 0.000000 | 0.000000 | 0.321359 |
| `PT` / `ptcr` | 0.452725 | 0.383851 | 0.583750 | 0.000000 | 0.000000 | 0.284065 |

## 6. PHT 验证阈值补充

Checkpoint 在 validation 上选择的 PHT 阈值为 `0.10986328125`。该阈值原样用于下表，没有通过 test 标签调整。

| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PHT/test_unseen_hla` | `0.5` | 0.507168 | 0.035514 | 0.091642 | 0.584229 | 0.049720 | 0.964616 |
| `PHT/test_unseen_hla` | `0.10986328125` | 0.667463 | 0.377809 | 0.567366 | 0.811696 | 0.436096 | 0.898831 |
| `PHT/test_unseen_peptide` | `0.5` | 0.906573 | 0.816713 | 0.901999 | 0.948438 | 0.859895 | 0.953251 |
| `PHT/test_unseen_peptide` | `0.10986328125` | 0.927398 | 0.858209 | 0.930494 | 0.892430 | 0.971951 | 0.882845 |
| `PHT/test_unseen_tcr` | `0.5` | 0.970382 | 0.940770 | 0.970436 | 0.968677 | 0.972201 | 0.968563 |
| `PHT/test_unseen_tcr` | `0.10986328125` | 0.948558 | 0.900312 | 0.950636 | 0.913731 | 0.990647 | 0.906469 |
| `PHT/test_unseen_triple` | `0.5` | 0.748485 | 0.547105 | 0.681992 | 0.927083 | 0.539394 | 0.957576 |
| `PHT/test_unseen_triple` | `0.10986328125` | 0.722727 | 0.477587 | 0.661738 | 0.848341 | 0.542424 | 0.903030 |

## 7. 结果解读与复核

- PH 主任务宏平均 performance average：`0.293554`。
- PT 主任务宏平均 performance average：`0.284065`。
- PHT 主任务宏平均 performance average：`0.774839`。
- PHT head 表现最高的单个 split 是 `PHT/test_unseen_tcr`，performance average 为 `0.974749`。
- `test_unseen_pair` 等小样本 split 的指标方差可能较大，应结合样本数、正负比例和混淆矩阵解读。
- 本次只执行推理；没有重新训练、修改 checkpoint，也没有使用测试标签做阈值或模型选择。
- 每个唯一 peptide/HLA/TCR 只编码一次，随后复用 residue-level 表示完成全部行级推理。
- 原始 JSON：[`summary.json`](summary.json)
- 本报告：[`summary.md`](summary.md)

