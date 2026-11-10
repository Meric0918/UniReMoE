# Stage 2 最优 pHLA / pTCR 模型：PH / PT new 测试总结

## 训练运行与模型选择

目标运行：`swanlog/run-20260928_112411-vmvqnyub`。本次模型来自 Stage 2、fold 1、seed 3407 的 `--from-scratch` 训练；PLM 初始化权重仍为预训练 ESM-C / TCR-BERT。Stage 2B 最多训练 20 轮，整轮 joint score 最高为 0.8360（第 16 轮）。pHLA 和 pTCR 的部署 checkpoint 按各自验证集 `(AUROC + AUPR) / 2` 独立选择，因此各自选中的轮次不同。

| 任务 | 最优 checkpoint | 选择轮次 | Global step | 验证选择分数 | 验证 AUROC | 验证 AUPR |
|---|---|---:|---:|---:|---:|---:|
| pHLA | [`stage2_best_phla_20260928_112359_821461.pt`](../../../../trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_phla_20260928_112359_821461.pt) | 20 | 164,318 | 0.8675 | 0.8544 | 0.8806 |
| pTCR | [`stage2_best_ptcr_20260928_112359_821461.pt`](../../../../trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_ptcr_20260928_112359_821461.pt) | 10 | 89,628 | 0.8070 | 0.7876 | 0.8264 |

## PH new：pHLA 最优模型

测试共 87,511 条（负 46,836、正 40,675）。三个 CSV 之间没有重复 peptide–HLA pair。

| Split | 样本数 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | performance_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 0.7253 | 0.5679 | 0.6714 | 0.2456 | 0.4018 | 0.2939 | 0.8983 | 0.5224 |
| `test_unseen_pair` | 51 | 0.3401 | 0.3287 | 0.4902 | -0.1952 | 0.0714 | 0.0455 | 0.8276 | 0.2071 |
| `test_unseen_peptide` | 62,764 | 0.9702 | 0.9679 | 0.9169 | 0.8340 | 0.9178 | 0.9287 | 0.9050 | 0.9213 |

等权 macro AUROC/AUPR 为 **0.6785 / 0.6215**，合并 87,511 条后 pooled AUROC/AUPR 为 **0.9113 / 0.9122**。macro 将 51 条的 `test_unseen_pair` 与两个大 split 等权，容易被小 split 拉低；pooled 更符合当前测试样本总构成。

## PT new：pTCR 最优模型

测试共 263,610 条（负 147,476、正 116,134）。三个 CSV 之间没有重复 peptide–TCR pair。

| Split | 样本数 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | performance_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 0.8882 | 0.8711 | 0.8474 | 0.6715 | 0.7823 | 0.7228 | 0.9236 | 0.8121 |
| `test_unseen_peptide` | 112,443 | 0.8188 | 0.7860 | 0.7819 | 0.5191 | 0.6739 | 0.6100 | 0.8826 | 0.7159 |
| `test_unseen_tcr` | 142,960 | 0.9154 | 0.9146 | 0.8379 | 0.6760 | 0.8401 | 0.8516 | 0.8242 | 0.8368 |

等权 macro AUROC/AUPR 为 **0.8742 / 0.8572**，pooled AUROC/AUPR 为 **0.8831 / 0.8751**。

## 结果解读

- pHLA 在 `test_unseen_peptide` 表现突出（AUROC 0.9702），`test_unseen_hla` 为中等区分度（AUROC 0.7253）；固定阈值 0.5 时 unseen-HLA Recall 只有 0.2939、Specificity 为 0.8983，漏检较多。
- pHLA 的 `test_unseen_pair` 只有 51 条，AUROC 0.3401 且 MCC 为负，结果不稳定，不能据此单独判断模型整体能力。该小 split 也解释了 pHLA 等权 macro 低（0.6785），而所有样本 pooled AUROC 为 0.9113。
- pTCR 在 `test_unseen_tcr` 最好（AUROC/AUPR 0.9154/0.9146），`test_unseen_pair` 也较好（0.8882/0.8711）；`test_unseen_peptide` 是相对弱项（0.8188/0.7860），且有 112,443 条样本，值得后续优先改进。
- 以上测试指标均使用配置阈值 0.5；测试集没有参与 checkpoint 选择或阈值调整。`performance_avg` 是 AUROC、AUPR、Accuracy、MCC、F1 的算术平均。

## 完整报告

- [PH new / pHLA 明细与 JSON](ph_new_pHLA.md) / [JSON](ph_new_pHLA.json)
- [PT new / pTCR 明细与 JSON](pt_new_pTCR.md) / [JSON](pt_new_pTCR.json)
- 评估代码：[PH Stage 2 pHLA](../../../../source/evaluate_plm_stage2_ph_new.py)、[PT Stage 2 pTCR](../../../../source/evaluate_plm_stage2_pt_new.py)
