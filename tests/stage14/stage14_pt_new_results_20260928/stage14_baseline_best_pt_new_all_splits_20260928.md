# PLM-Stage14 baseline：PT new 全部测试集总结

## 结论

使用 Stage14 baseline 第 **21** 轮最佳 checkpoint，对 `dataset/PT/new` 下全部 **3** 个 CSV 测试集共 **263,610** 条 PTCR 样本完成评估。

| 汇总 | 样本总数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | 263,610 | 0.765406 | 0.733630 | 0.754255 | 0.479465 | 0.639977 | 0.780465 | 0.550345 | 0.894123 | 0.674547 |
| 全部样本 pooled | 263,610 | 0.766106 | 0.757935 | 0.726080 | 0.446712 | 0.624163 | 0.789017 | 0.516292 | 0.891284 | 0.664199 |

三个 split 的正负样本合计为 **116,134 / 147,476**；其中 `test_unseen_pair` 的 `performance_avg` 最高 （0.7454），`test_unseen_peptide` 最低（0.5756）。

`performance_avg` 为 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。宏平均对三个 split 等权；pooled 指标将所有测试样本合并后重新计算。分类指标使用固定阈值 `score > 0.5`，未基于测试标签调阈值。

## 各测试集结果

| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | `performance_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 3,113 / 5,094 | 37.93% | 0.797391 | 0.762079 | 0.818570 | 0.607813 | 0.741179 | 0.684870 | 0.900275 | 0.745406 |
| `test_unseen_peptide` | 112,443 | 41,541 / 70,902 | 36.94% | 0.701872 | 0.620287 | 0.713642 | 0.350368 | 0.491929 | 0.375244 | 0.911907 | 0.575620 |
| `test_unseen_tcr` | 142,960 | 71,480 / 71,480 | 50.00% | 0.796957 | 0.818523 | 0.730554 | 0.480214 | 0.686824 | 0.590921 | 0.870187 | 0.702614 |

### 分组鲁棒性与混淆矩阵

分组对象按测试设计确定：`unseen_peptide` 按 peptide、`unseen_tcr` 按 TCR、`unseen_pair` 按 peptide–TCR pair 分组。Group-macro 指标对组等权；Group-macro recall 仅在含正负标签的组上计算。`unseen_pair` 中每个 pair 组只有一个标签，因此该 split 的 Group-macro recall 未定义。

| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | pair | 8,207 | 0.818570 | NaN | 0.000000 | 0.640625 | 7.139111 | 4,586 | 508 | 981 | 2,132 |
| `test_unseen_peptide` | peptide | 299 | 0.837908 | 0.225504 | 0.977393 | 0.459791 | 5.357416 | 64,656 | 6,246 | 25,953 | 15,588 |
| `test_unseen_tcr` | tcr | 33,061 | 0.777294 | 0.540542 | 0.851945 | 0.626609 | 8.079479 | 62,201 | 9,279 | 29,241 | 42,239 |

## 数据清单与完整性检查

| CSV 文件 | Split | Samples | 正 / 负 | 唯一 peptide | 唯一 TCR | 重复 pair | 标签冲突 | 长度过滤 | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [`test_unseen_pair.csv`](../../../dataset/PT/new/test_unseen_pair.csv) | `test_unseen_pair` | 8,207 | 3,113 / 5,094 | 192 | 7,425 | 0 | 0 | 0 | `119edba70e2b28276048e96917460f25787ffa04ec2e520593bc71fa66374879` |
| [`test_unseen_peptide.csv`](../../../dataset/PT/new/test_unseen_peptide.csv) | `test_unseen_peptide` | 112,443 | 41,541 / 70,902 | 299 | 75,021 | 0 | 0 | 0 | `98e4e244b4564c8b432d8b7c12530236b5dc11c7969101b90dc662a14e998f39` |
| [`test_unseen_tcr.csv`](../../../dataset/PT/new/test_unseen_tcr.csv) | `test_unseen_tcr` | 142,960 | 71,480 / 71,480 | 1,410 | 33,061 | 0 | 0 | 0 | `bfca9a02552e7085fc0dee4c14c57a89950c2c5a95a1177c8816e1986c661a5a` |

合计 **263,610** 行，三个文件间共享 pair **0** 条，标签冲突 **0** 条。全目录 peptide/TCR 去重后分别为 **1,712 / 108,983** 条序列。

输入标签为二元值，评测序列均为标准氨基酸；报告列出各文件长度过滤数。模型将全部文件的唯一 peptide/TCR 各编码一次，再分别计算每个 split 的 PTCR 指标。

### 测试实体与 Stage14 训练集重叠

| 测试 split | peptide 重叠 / 未出现 | TCR 重叠 / 未出现 |
|---|---:|---:|
| `test_unseen_pair` | 0 / 192 | 0 / 7,425 |
| `test_unseen_peptide` | 0 / 299 | 70,791 / 4,230 |
| `test_unseen_tcr` | 1,407 / 3 | 5 / 33,056 |

`test_unseen_peptide` 的 299 个测试 peptide 在所检查的 PH/PT 二分类训练文件与四类 PHT 训练文件中均未出现。

`test_unseen_tcr` 的 33,061 个测试 TCR 中，5 个在 PT/PHT 训练文件中出现，33,056 个未出现。

## 结果解读

按 split 等权宏平均的 `performance_avg` 为 0.6745，按样本合并 pooled 的 `performance_avg` 为 0.6642。两者差异反映各 split 样本规模和类别构成不同；各 split 的明细应一并参考。

`test_unseen_peptide` 的 `performance_avg` 最低（0.5756），其按样本 Recall 为 0.375，按 peptide 等权的 Group-macro recall 为 0.226，显示正类检出率会随未见 peptide 组而变化。

Pooled Recall=0.516、Specificity=0.891，假阴性 56,175、假阳性 16,033。如应用侧需要提高正类召回，应在独立验证集上选择阈值，再用本测试结果报告固定阈值表现。

## 评测设置与 Checkpoint

| 项目 | 内容 |
|---|---|
| 评测时间 | 2026-09-28T16:30:24+08:00 |
| 任务 | PTCR：peptide–TCR 二分类 |
| 测试目录 | [`dataset/PT/new`](../../../dataset/PT/new) |
| 设备 / 混合精度 | `cuda` / `bf16` |
| 编码 batch / pair batch | 256 / 2048 |
| Checkpoint 条件 / 类型 | `baseline` / `stage14_online_model` |
| Checkpoint | [`stage14_best_20260923_172315_002675.pt`](../../../trained_model/PLM_Stage14/fold_1/seed_42/baseline/stage14_best_20260923_172315_002675.pt) |
| Stage14 best round / global step | 21 / 35532 |
| Checkpoint 验证 joint score | 0.888381 |
| Checkpoint SHA-256 | `5800207f0e29aba90490719b488ea9d918ee2bdc2fe7a4d7e2c09e0d6115054b` |

Checkpoint 的 validation joint score 是训练验证集上的选模分数，与本报告 PT new 的测试指标来自不同数据集，不应直接比较。

## 产物与复现

- JSON 全量结果：[`stage14_baseline_best_pt_new_all_splits_20260928.json`](stage14_baseline_best_pt_new_all_splits_20260928.json)
- 评估入口：[`evaluate_plm_stage14_pt_new.py`](../../../source/evaluate_plm_stage14_pt_new.py)

从项目根目录执行：

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage14_pt_new.py \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage14/fold_1/seed_42/baseline/stage14_best_20260923_172315_002675.pt \
  --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PT/new \
  --output-json /home/mjp/Project/PLM/UnifyImmun/tests/stage14/stage14_pt_new_results_20260928/stage14_baseline_best_pt_new_all_splits_20260928.json \
  --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage14/stage14_pt_new_results_20260928/stage14_baseline_best_pt_new_all_splits_20260928.md \
  --device cuda
```
