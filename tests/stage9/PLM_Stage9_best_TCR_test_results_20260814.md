# PLM Stage 9 best 模型在外部 TCR 测试集上的结果

## 1. 测试概况

| 项目 | 内容 |
|---|---|
| 评测日期 | 2026-08-14 |
| Checkpoint | /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage9/fold_4/seed_2026/stage9_best_20260813_164522_523785.pt |
| Checkpoint 类型 | stage9_multitask_model |
| 测试分支 | pTCR 二分类分支 |
| 测试数据目录 | /home/mjp/Project/PLM/UnifyImmun/data/data_TCR/test_data_set |
| 输入字段 | peptide,tcr,label |
| HLA 输入 | 无 |
| 分类阈值 | 0.5（固定，未在测试集调参） |
| 推理设备 | NVIDIA GeForce RTX 4090 |

三个 CSV 文件没有 HLA 列，因此本次不能计算 Stage 9 的 PH-TCR 成对排序、pHLA 或 PHT 三元指标；评估使用 Stage 9 checkpoint 中保留并继续训练的 pTCR 分支。

## 2. 数据审计

| 数据集 | 原始行数 | 实际测试样本数 | 正样本 | 负样本 | 正样本比例 | peptide 长度过滤 |
|---|---:|---:|---:|---:|---:|---|
| covid_set.csv | 1,089,310 | 1,089,310 | 547,657 | 541,653 | 50.2756% | 无 |
| independent_set.csv | 28,707 | 28,707 | 10,018 | 18,689 | 34.8974% | 无 |
| triple_set_peptide_8_15aa.csv | 96,799 | 96,799 | 37,813 | 58,986 | 39.0634% | 无 |

三个文件合计使用 **1,214,816** 条样本（正样本 595,488，负样本 619,328）。并集包含 **1,994** 个唯一 peptide 和 **200,202** 个唯一 TCR；最长保留 TCR 为 33 aa，未超过配置上限 34 aa。

## 3. 测试指标

AUROC 和 AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 0.5。

| 数据集 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covid_set | 1,089,310 | 0.609422 | 0.611140 | 0.586233 | 0.193474 | 0.475311 | 0.655661 | 0.372773 | 0.802058 | 1.061361 |
| independent_set | 28,707 | 0.948000 | 0.911542 | 0.887449 | 0.753868 | 0.840735 | 0.830461 | 0.851268 | 0.906844 | 0.274848 |
| triple_set_peptide_8_15aa | 96,799 | 0.833463 | 0.813293 | 0.786971 | 0.547967 | 0.672167 | 0.842634 | 0.559067 | 0.933069 | 0.569959 |

## 4. 模型与推理方式

1. 加载指定的 Stage 9 stage9_best 部署 checkpoint，并校验模型配置、运行时 fold/seed 和 checkpoint 格式。
2. 对三个文件中的唯一 peptide 和唯一 TCR 各编码一次，再复用 residue-level 表示完成全部 pTCR 样本推理；没有使用旧的 Stage 1 embedding cache。
3. 使用项目配置中的 peptide 长度范围 [8, 15]；TCR 长度上限为 33 aa，实际数据未超过配置上限 34 aa。
4. 固定阈值为 0.5。checkpoint 中的 selected_pair_threshold 属于 Stage 9 成对 pair head，不用于本次无 HLA 的 pTCR 外部测试。

## 5. 结果概览

按 AUROC 排序，independent_set 最高（0.948000），covid_set 最低（0.609422）。
- covid_set：AUROC=0.609422，AUPR=0.611140，MCC=0.193474，Recall=0.372773，Specificity=0.802058。
- independent_set：AUROC=0.948000，AUPR=0.911542，MCC=0.753868，Recall=0.851268，Specificity=0.906844。
- triple_set_peptide_8_15aa：AUROC=0.833463，AUPR=0.813293，MCC=0.547967，Recall=0.559067，Specificity=0.933069。

这些指标反映的是不同外部数据集上的 pTCR 分支表现，不能据此声称 Stage 9 的成对对比学习目标在这些数据上已被直接验证。若要评价 Stage 9 的核心 pair ranking，需要提供包含 peptide,hla,tcr_positive,tcr_negative 的成对数据，并使用项目的 Stage 9 pair evaluator。

## 6. 相关文件

- Checkpoint：/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage9/fold_4/seed_2026/stage9_best_20260813_164522_523785.pt
- 配置：/home/mjp/Project/PLM/UnifyImmun/configs/plm_stage9.yaml
- 测试数据：/home/mjp/Project/PLM/UnifyImmun/data/data_TCR/test_data_set/covid_set.csv、/home/mjp/Project/PLM/UnifyImmun/data/data_TCR/test_data_set/independent_set.csv、/home/mjp/Project/PLM/UnifyImmun/data/data_TCR/test_data_set/triple_set_peptide_8_15aa.csv
- 原始 JSON 结果：/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage9/fold_4/seed_2026/evaluation_stage9_ptcr_test_data_set_20260814_155357_969158.json
- 本总结：/home/mjp/Project/PLM/UnifyImmun/tests/stage9/PLM_Stage9_best_TCR_test_results_20260814.md

## 7. 复现口径

本次使用项目的 OnlinePairDataset、Stage 9 模型构建与 checkpoint 加载逻辑，以及现有 pTCR 批量推理逻辑；运行参数为 device=cuda、encode_batch_size=512、pair_batch_size=2048、threshold=0.5。

