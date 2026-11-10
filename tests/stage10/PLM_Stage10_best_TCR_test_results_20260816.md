# PLM Stage10 baseline 模型外部 TCR 测试总结

## 1. 测试概况

本次使用指定的 Stage10 baseline 最佳 checkpoint，对 `data/data_TCR/test_data_set` 下的三个 CSV 进行完整推理。由于这些文件只包含 `peptide,tcr,label`，没有 HLA 输入，因此本次只评价 pTCR 二分类分支，不能计算 pHLA、PHT 三元头、成对排序或约束违例率。

| 项目 | 内容 |
|---|---|
| 评测日期 | 2026-08-16 |
| Stage10 checkpoint | `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage10/fold_4/seed_2026/baseline/stage10_best_20260814_210517_299513.pt` |
| Checkpoint 类型 | `stage10_binding_model` |
| Stage10 条件 | `baseline` |
| 最佳验证轮次 | 8 |
| 最佳验证 joint score | 0.983496 |
| 测试数据目录 | `/home/mjp/Project/PLM/UnifyImmun/data/data_TCR/test_data_set` |
| 输入字段 | `peptide,tcr,label` |
| 测试分支 | pTCR 二分类 |
| 分类阈值 | 0.5（固定，未在测试集上选择） |
| 推理设备 | NVIDIA GeForce RTX 4090 |
| 编码 / 推理 batch size | 512 / 2048 |

本报告以及比较结果只作外部测试集上的描述性统计；外部数据没有参与 checkpoint 选择、阈值选择或超参数调节。

## 2. 数据审计

三个文件均通过项目的 `peptide` 长度过滤（配置范围 `[8, 15]`），没有行被过滤。标签均为 0/1；总计使用 1,214,816 条样本，其中正样本 595,488 条、负样本 619,328 条，正样本比例为 49.0188%。

| 数据集 | 原始行数 | 实际测试样本数 | 正样本 | 负样本 | 正样本比例 | peptide 长度过滤 |
|---|---:|---:|---:|---:|---:|---|
| `covid_set.csv` | 1,089,310 | 1,089,310 | 547,657 | 541,653 | 50.2756% | 无 |
| `independent_set.csv` | 28,707 | 28,707 | 10,018 | 18,689 | 34.8974% | 无 |
| `triple_set_peptide_8_15aa.csv` | 96,799 | 96,799 | 37,813 | 58,986 | 39.0634% | 无 |
| **合计** | **1,214,816** | **1,214,816** | **595,488** | **619,328** | **49.0188%** | **无** |

三个数据集的并集包含 1,994 个唯一 peptide 和 200,202 个唯一 TCR；保留样本中的最长 TCR 为 33 aa，未超过配置上限 34 aa。

## 3. Stage10 pTCR 测试指标

AUROC 和 AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 0.5。

| 数据集 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `covid_set` | 1,089,310 | 0.637237 | 0.626871 | 0.570732 | 0.179748 | 0.392195 | 0.680556 | 0.275474 | 0.869263 | 0.983899 |
| `independent_set` | 28,707 | 0.946307 | 0.905954 | 0.884453 | 0.742460 | 0.827930 | 0.861864 | 0.796566 | 0.931564 | 0.285455 |
| `triple_set_peptide_8_15aa` | 96,799 | 0.910079 | 0.877399 | 0.807498 | 0.597329 | 0.701254 | 0.890436 | 0.578373 | 0.954379 | 0.413809 |

### 3.1 概率输出概览

| 数据集 | 平均预测概率 | 正样本平均概率 | 负样本平均概率 |
|---|---:|---:|---:|
| `covid_set` | 0.261405 | 0.316215 | 0.205988 |
| `independent_set` | 0.316791 | 0.707034 | 0.107607 |
| `triple_set_peptide_8_15aa` | 0.298025 | 0.593167 | 0.108824 |

## 4. 与 Stage3 pTCR 参照模型的差值

评估脚本在完全相同的样本顺序、编码器表示和阈值下，同时计算了 Stage3 来源模型作为参照。下表为 `Stage10 - Stage3`，正值表示 Stage10 更高；Loss 的负值表示 Stage10 更低。

| 数据集 | ΔAUROC | ΔAUPR | ΔAccuracy | ΔMCC | ΔF1 | ΔRecall | ΔSpecificity | ΔLoss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `covid_set` | +0.027895 | +0.015769 | -0.015367 | -0.013663 | -0.082225 | -0.096091 | +0.066253 | -0.077885 |
| `independent_set` | -0.001680 | -0.005293 | -0.002891 | -0.011011 | -0.012461 | -0.053304 | +0.024131 | +0.010618 |
| `triple_set_peptide_8_15aa` | +0.076636 | +0.063991 | +0.020042 | +0.048170 | +0.028500 | +0.019094 | +0.020649 | -0.156057 |

## 5. 结果摘要

- `independent_set` 上的排序性能最高：AUROC=0.946307、AUPR=0.905954，固定阈值下 MCC=0.742460。
- `triple_set_peptide_8_15aa` 上 Stage10 相比 Stage3 的提升最明显：AUROC 提升 0.076636，AUPR 提升 0.063991，MCC 提升 0.048170，Loss 降低 0.156057。
- `covid_set` 上 AUROC 和 AUPR 相比 Stage3 提升，但固定阈值 0.5 下 Recall=0.275474，低于 Stage3 的 0.371565；同时 Specificity=0.869263。该结果说明排序指标与固定决策阈值下的召回表现并不一致，不能仅依据 AUROC 判断阈值下的分类效果。
- `independent_set` 上 Stage10 的 AUROC 仅比 Stage3 低 0.001680，但固定阈值下 Recall 下降 0.053304、Specificity 上升 0.024131，表现出更保守的阳性判定倾向。

上述结论仅针对无 HLA 外部数据上的 pTCR 分支，不能外推为 Stage10 的 pHLA/PHT 三元任务或约束设计已在这些测试集上得到验证。

## 6. 推理方法与模型来源

1. 校验并加载指定的 Stage10 `stage10_binding_model` checkpoint，以及其记录的 Stage3 来源 SHA-256。
2. 使用 Stage10 checkpoint 记录的 Stage3 来源 checkpoint 加载微调后的 peptide ESM-C 和 TCR-BERT 编码器；Stage10 baseline checkpoint 本身保存的是 binding model 权重。
3. 对三个测试集并集中的每个唯一 peptide 和唯一 TCR 各编码一次，再复用 residue-level 表示完成所有样本的 pTCR 推理；本次未使用旧的 Stage1 embedding cache。
4. 固定使用阈值 0.5；没有在外部测试集上选择阈值、选择模型或调参。

关键指纹：

- Stage10 checkpoint SHA-256：`f60b4cadbc0be1c741a404c63293d2d75de3ef8dd7fe9c07ae4fb1a22623b91f`
- Stage3 来源 checkpoint：`/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt`
- Stage3 来源 SHA-256：`c21998c15d0a98d97f224a3ccd6b9824bc9720344d5e0d98eb0914d6f0446262`

## 7. 相关文件与复现命令

- 本总结：[PLM_Stage10_best_TCR_test_results_20260816.md](/home/mjp/Project/PLM/UnifyImmun/tests/stage10/PLM_Stage10_best_TCR_test_results_20260816.md)
- 原始 JSON 汇总：[evaluation_stage10_ptcr_test_data_set_20260816.json](/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage10/fold_4/seed_2026/baseline/evaluation_stage10_ptcr_test_data_set_20260816.json)
- 逐行概率输出：[evaluation_stage10_ptcr_rows_20260816.csv](/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage10/fold_4/seed_2026/baseline/evaluation_stage10_ptcr_rows_20260816.csv)（1,214,816 条数据记录，含表头共 1,214,817 行）
- 使用的 checkpoint：[stage10_best_20260814_210517_299513.pt](/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage10/fold_4/seed_2026/baseline/stage10_best_20260814_210517_299513.pt)
- 配置：[plm_stage10.yaml](/home/mjp/Project/PLM/UnifyImmun/configs/plm_stage10.yaml)

复现命令：

```bash
./run_stage10.sh evaluate-tcr \
  --checkpoint trained_model/PLM_Stage10/fold_4/seed_2026/baseline/stage10_best_20260814_210517_299513.pt \
  --device cuda \
  --encode-batch-size 512 \
  --pair-batch-size 2048 \
  --threshold 0.5
```
