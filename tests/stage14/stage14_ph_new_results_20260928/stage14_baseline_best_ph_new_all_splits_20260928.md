# PLM-Stage14 baseline：PH new 全部测试集总结

## 结论

使用 Stage14 baseline 第 **21** 轮最佳 checkpoint，对 `dataset/PH/new` 下全部 **3** 个 CSV 测试集共 **87,511** 条 PHLA 样本完成评估。

| 汇总 | 样本总数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | 87,511 | 0.639087 | 0.596640 | 0.644182 | 0.222574 | 0.470956 | 0.554962 | 0.423482 | 0.800083 | 0.514688 |
| 全部样本 pooled | 87,511 | 0.808707 | 0.799506 | 0.740318 | 0.476782 | 0.704728 | 0.747327 | 0.666724 | 0.804232 | 0.706008 |

正类 / 负类合计为 **40,675 / 46,836**；`test_unseen_peptide` 的 `performance_avg` 最高（0.7632），`test_unseen_pair` 最低（0.2843）。

`performance_avg` 为 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。宏平均对三个 split 等权；pooled 指标将所有样本合并后重新计算。分类指标使用固定阈值 `score > 0.5`，未基于测试标签调阈值。

## 各测试集结果

| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | `performance_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 9,271 / 15,425 | 37.54% | 0.653944 | 0.534702 | 0.644558 | 0.202154 | 0.447368 | 0.383238 | 0.801621 | 0.496545 |
| `test_unseen_pair` | 51 | 22 / 29 | 43.14% | 0.409875 | 0.400141 | 0.509804 | -0.091634 | 0.193548 | 0.136364 | 0.793103 | 0.284347 |
| `test_unseen_peptide` | 62,764 | 31,382 / 31,382 | 50.00% | 0.853443 | 0.855077 | 0.778185 | 0.557204 | 0.771950 | 0.750844 | 0.805525 | 0.763172 |

### 分组鲁棒性与混淆矩阵

`unseen_peptide` 按 peptide 分组，`unseen_hla` 按 HLA 分组，`unseen_pair` 按 peptide–HLA pair 分组。Group-macro recall 仅在同时含正负标签的组上计算；unseen pair 的单 pair 组只有一个标签，recall 未定义。

| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | hla | 18 | 0.742564 | 0.360455 | 0.992752 | 1.317088 | 3.310240 | 12,365 | 3,060 | 5,718 | 3,553 |
| `test_unseen_pair` | pair | 51 | 0.509804 | NaN | 0.000000 | 3.443538 | 10.252026 | 23 | 6 | 19 | 3 |
| `test_unseen_peptide` | peptide | 41,668 | 0.787492 | 0.776704 | 0.359075 | 1.011348 | 16.118096 | 25,279 | 6,103 | 7,819 | 23,563 |

## 数据清单与完整性检查

| CSV 文件 | Split | Samples | 正 / 负 | 唯一 peptide | 唯一 HLA | 重复 pair | 标签冲突 | 长度过滤 | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [`test_unseen_hla.csv`](../../../dataset/PH/new/test_unseen_hla.csv) | `test_unseen_hla` | 24,696 | 9,271 / 15,425 | 22,700 | 18 | 0 | 0 | 0 | `e46f3e258bd8d2b62c1cf2748b286fdec2eeda0f96bfeab3c6187e4fa6beeb22` |
| [`test_unseen_pair.csv`](../../../dataset/PH/new/test_unseen_pair.csv) | `test_unseen_pair` | 51 | 22 / 29 | 46 | 13 | 0 | 0 | 0 | `bc07bf5edb2ea31a7ce4222cfc3b9489863dfbb9905be1666647d84d6aefa652` |
| [`test_unseen_peptide.csv`](../../../dataset/PH/new/test_unseen_peptide.csv) | `test_unseen_peptide` | 62,764 | 31,382 / 31,382 | 41,668 | 99 | 0 | 0 | 0 | `344100c2711fb3829ee9278514f7dd31e09b0ab3fab6735838770834494e42fb` |

合计 **87,511** 行，文件间共享 pair **0** 条，标签冲突 **0** 条；去重后共有 **64,370** 个 peptide、**118** 个 HLA。

所有纳入样本的 peptide/HLA 为标准氨基酸序列；报告列出各文件长度过滤数。评估时对全部文件的唯一 peptide/HLA 各编码一次，再计算 PHLA pair 分数。

### 测试实体与 Stage14 训练集重叠

| 测试 split | peptide 重叠 / 未出现 | HLA 重叠 / 未出现 |
|---|---:|---:|
| `test_unseen_hla` | 13,910 / 8,790 | 0 / 18 |
| `test_unseen_pair` | 0 / 46 | 0 / 13 |
| `test_unseen_peptide` | 0 / 41,668 | 99 / 0 |

`test_unseen_hla` 的 18 个 HLA 中，0 个出现在 PH/PHT 训练文件，18 个未出现。

## 结果解读

等权宏平均 `performance_avg` 为 0.5147，pooled 为 0.7060。`test_unseen_pair` 仅 51 条，其指标受样本量限制，宜结合另外两个大样本 split 判断泛化表现。

Pooled Recall=0.667、Specificity=0.804，假阴性 13,556、假阳性 9,169。若应用需要提高正类召回，应在独立验证集选择阈值。

## 评测设置与 Checkpoint

| 项目 | 内容 |
|---|---|
| 评测时间 | 2026-09-28T19:56:59+08:00 |
| 任务 | PHLA：peptide–HLA 二分类 |
| 测试目录 | [`dataset/PH/new`](../../../dataset/PH/new) |
| 设备 / 混合精度 | `cuda` / `bf16` |
| 编码 batch / pair batch | 256 / 2048 |
| Checkpoint 条件 / 类型 | `baseline` / `stage14_online_model` |
| Checkpoint | [`stage14_best_20260923_172315_002675.pt`](../../../trained_model/PLM_Stage14/fold_1/seed_42/baseline/stage14_best_20260923_172315_002675.pt) |
| Stage14 best round / global step | 21 / 35532 |
| Checkpoint 验证 joint score | 0.888381 |
| Checkpoint SHA-256 | `5800207f0e29aba90490719b488ea9d918ee2bdc2fe7a4d7e2c09e0d6115054b` |

验证 joint score 是训练验证集选模分数，与本报告 PH new 测试指标来自不同数据集，不应直接比较。

## 产物与复现

- JSON 全量结果：[`stage14_baseline_best_ph_new_all_splits_20260928.json`](stage14_baseline_best_ph_new_all_splits_20260928.json)
- 评估入口：[`evaluate_plm_stage14_ph_new.py`](../../../source/evaluate_plm_stage14_ph_new.py)

从项目根目录执行：

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage14_ph_new.py \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage14/fold_1/seed_42/baseline/stage14_best_20260923_172315_002675.pt \
  --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PH/new \
  --output-json /home/mjp/Project/PLM/UnifyImmun/tests/stage14/stage14_ph_new_results_20260928/stage14_baseline_best_ph_new_all_splits_20260928.json \
  --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage14/stage14_ph_new_results_20260928/stage14_baseline_best_ph_new_all_splits_20260928.md \
  --device cuda
```
