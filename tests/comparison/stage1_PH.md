# Stage 1 最优 pHLA checkpoint：PH new 测试结果

## 结果摘要

使用 Stage 1B 的最佳 pHLA checkpoint 对 `dataset/PH/new` 下 **3** 个测试集、共 **87,511** 条样本完成评估。

| 汇总 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | — | 0.7012 | 0.6082 | 0.6685 | 0.2637 | 0.4875 | 0.5511 | 0.4581 | 0.8126 | 0.5458 | 0.7725 | 0.2441 | 0.1863 |
| 全部样本 pooled | 87,511 | 0.8876 | 0.8624 | 0.8006 | 0.5987 | 0.7826 | 0.7934 | 0.7720 | 0.8254 | 0.7864 | 0.4241 | 0.1366 | 0.0284 |

测试样本正类 / 负类为 **40,675 / 46,836**。宏平均对三个 split 等权；pooled 指标先合并样本再计算。

## 各测试集指标

| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 9,271 / 15,425 | 37.54% | 0.7450 | 0.5437 | 0.6586 | 0.2348 | 0.4689 | 0.5636 | 0.4015 | 0.8132 | 0.5302 | 0.6285 | 0.1305 |
| `test_unseen_pair` | 51 | 22 / 29 | 43.14% | 0.4310 | 0.3609 | 0.4902 | -0.1580 | 0.1333 | 0.2500 | 0.0909 | 0.7931 | 0.2515 | 1.3460 | 0.4129 |
| `test_unseen_peptide` | 62,764 | 31,382 / 31,382 | 50.00% | 0.9277 | 0.9199 | 0.8567 | 0.7144 | 0.8603 | 0.8396 | 0.8820 | 0.8315 | 0.8558 | 0.3429 | 0.0155 |

固定阈值为 `0.5`，正类预测使用严格比较 `score > 0.5`。阈值没有根据测试标签重新选择。`performance_avg` 是 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。

## 结果解读

`test_unseen_peptide` 区分能力最高（AUROC=0.9277）；`test_unseen_hla` AUROC=0.7450，但固定阈值下 Recall=0.4015、Specificity=0.8132，说明在这些未见 HLA 上漏检仍较多。

`test_unseen_pair` AUROC=0.4310，但只有 51 条，结果波动可能较大。三个 split 等权宏平均 `performance_avg`=0.5458，pooled `performance_avg`=0.7864；前者给 51 条的 pair split 与大样本 split 相同权重，后者则主要受 unseen peptide 大样本影响，建议以分 split 结果为主。

### 分组指标与混淆矩阵

| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | HLA | 18 | 0.7535 | 0.3471 | 0.9928 | 0.5363 | 1.0316 | 12,543 | 2,882 | 5,549 | 3,722 |
| `test_unseen_pair` | peptide-HLA pair | 51 | 0.4902 | — | 0.0000 | 1.3460 | 4.9462 | 23 | 6 | 20 | 2 |
| `test_unseen_peptide` | peptide | 41,668 | 0.8655 | 0.8870 | 0.3591 | 0.3267 | 4.4466 | 26,093 | 5,289 | 3,703 | 27,679 |

`test_unseen_peptide` 按 peptide、`test_unseen_hla` 按 HLA、`test_unseen_pair` 按 peptide–HLA pair 分组。Group-macro recall 仅对同时含正负样本的组有定义。

## 数据完整性与训练集重叠

| Split | CSV | Rows | 正 / 负 | 唯一 peptide | 唯一 HLA | 唯一 pair | 重复 pair 行 | 标签冲突 | 长度过滤 | peptide 与 PH/train 重叠 | HLA 与 PH/train 重叠 | pair 与 PH/train 重叠 | CSV SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `test_unseen_hla` | [`test_unseen_hla.csv`](../../dataset/PH/new/test_unseen_hla.csv) | 24,696 | 9,271 / 15,425 | 22,700 | 18 | 24,696 | 0 | 0 | 0 | 13,910 / 22,700 | 0 / 18 | 0 / 24,696 | `e46f3e258bd8d2b62c1cf2748b286fdec2eeda0f96bfeab3c6187e4fa6beeb22` |
| `test_unseen_pair` | [`test_unseen_pair.csv`](../../dataset/PH/new/test_unseen_pair.csv) | 51 | 22 / 29 | 46 | 13 | 51 | 0 | 0 | 0 | 0 / 46 | 0 / 13 | 0 / 51 | `bc07bf5edb2ea31a7ce4222cfc3b9489863dfbb9905be1666647d84d6aefa652` |
| `test_unseen_peptide` | [`test_unseen_peptide.csv`](../../dataset/PH/new/test_unseen_peptide.csv) | 62,764 | 31,382 / 31,382 | 41,668 | 99 | 62,764 | 0 | 0 | 0 | 0 / 41,668 | 99 / 99 | 0 / 62,764 | `344100c2711fb3829ee9278514f7dd31e09b0ab3fab6735838770834494e42fb` |

PH new 三个 split 间共享 pair 行 **0** 条，其中标签冲突 **0** 条。PH/train 含 193,205 个唯一 peptide、99 个唯一 HLA 和 291,052 个唯一 pair。

## Checkpoint 与评测设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [`best_phla_20260928_021049_999878.pt`](../../trained_model/PLM_Unified/fold_1/seed_3407/best_phla_20260928_021049_999878.pt) |
| Evaluation config | [`plm_stage1.yaml`](../../configs/plm_stage1.yaml) |
| Checkpoint 类型 / 任务 | `task_model` / `phla` |
| Fold / seed | 1 / 3407 |
| Stage / round / global step | `stage1b` / 15 / 15,895 |
| 验证集 pHLA best score | 0.7195 |
| 训练 run timestamp | `20260928_021049_999878` |
| Checkpoint SHA-256 | `e87c30f13521b9b5431d6acaaeda8aaa4b87b97b7fc22edf6434fa426103d7fc` |
| ESM-C 模型 | `/home/mjp/model/ESMC-300M` |
| ESM-C 指纹 | `2edde85b1885b70dbdf5857a7e78946df6fa2932fdd4d7346f836bb71de16469` |
| 设备 | `NVIDIA GeForce RTX 4090` (`cuda`) |
| pHLA head 精度 | `bf16` |
| ESM-C encode batch / pair batch | 64 / 1024 |
| 阈值 | `0.5`，`score > threshold` |
| Peptide 长度 | 8–15 aa；本次过滤 0 行 |
| HLA 长度 | 最大 34 aa |
| 评测时间 | 2026-09-28T20:12:46+08:00 |

此 checkpoint 的验证 best score 用于训练期间选模；本报告 PH new 指标只作测试集评估，不参与选模或阈值调整。`test_unseen_pair` 仅 51 条样本，其指标不确定性较大，应结合两个大样本划分解读。

## 结果文件与复现

- 全量指标与数据审计：[`PLM_Unified_PH_new_test_results_best_phla_20260928_021049_999878.json`](PLM_Unified_PH_new_test_results_best_phla_20260928_021049_999878.json)
- 评测程序：[`evaluate_plm_stage1_ph_new.py`](../../source/evaluate_plm_stage1_ph_new.py)

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage1_ph_new.py \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/best_phla_20260928_021049_999878.pt \
  --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PH/new \
  --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage1/PLM_Unified_PH_new_test_results_best_phla_20260928_021049_999878.md \
  --output-json /home/mjp/Project/PLM/UnifyImmun/tests/stage1/PLM_Unified_PH_new_test_results_best_phla_20260928_021049_999878.json \
  --device cuda
```
