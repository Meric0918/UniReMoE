# Stage 1 pTCR 模型：PT new 测试结果

## 结果摘要

使用 Stage 1B pTCR checkpoint 对 [`dataset/PT/new`](../../dataset/PT/new) 下 3 个测试集、共 **263,610** 条保留样本完成评估。

| 汇总 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | 263,610 | 0.8617 | 0.8376 | 0.8045 | 0.5845 | 0.7471 | 0.7690 | 0.7289 | 0.8468 | 0.7671 | 0.4511 | 0.1419 | 0.0523 |
| 全部样本 pooled | 263,610 | 0.8699 | 0.8588 | 0.8000 | 0.5931 | 0.7690 | 0.7830 | 0.7555 | 0.8351 | 0.7782 | 0.4527 | 0.1436 | 0.0397 |

纳入 **263,610** 条样本：正类 **116,134**、负类 **147,476**，正类比例 **44.06%**。

`test_unseen_tcr` 的 `performance_avg` 最高 （0.8283）；`test_unseen_peptide` 最低（0.6845）。

宏平均对三个 split 等权；pooled 指标将样本合并后重新计算。`performance_avg` 是 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。

## 各测试集结果

| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 3,113 / 5,094 | 37.93% | 0.8786 | 0.8566 | 0.8234 | 0.6220 | 0.7615 | 0.7809 | 0.7430 | 0.8726 | 0.7884 | 0.4278 | 0.0683 |
| `test_unseen_peptide` | 112,443 | 41,541 / 70,902 | 36.94% | 0.7987 | 0.7490 | 0.7594 | 0.4699 | 0.6458 | 0.7078 | 0.5938 | 0.8564 | 0.6845 | 0.5400 | 0.0764 |
| `test_unseen_tcr` | 142,960 | 71,480 / 71,480 | 50.00% | 0.9078 | 0.9073 | 0.8307 | 0.6618 | 0.8339 | 0.8184 | 0.8500 | 0.8113 | 0.8283 | 0.3855 | 0.0121 |

固定阈值为 `0.5`，正类预测使用严格比较 `score > 0.5`；未依据测试标签重新选择阈值。

### 分组鲁棒性与混淆矩阵

| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | peptide–TCR pair | 8,207 | 0.8234 | NaN | 0.0000 | 0.4278 | 4.5861 | 4,445 | 649 | 800 | 2,313 |
| `test_unseen_peptide` | peptide | 299 | 0.8379 | 0.2661 | 0.9774 | 0.3998 | 3.4855 | 60,721 | 10,181 | 16,874 | 24,667 |
| `test_unseen_tcr` | TCR | 33,061 | 0.8366 | 0.7735 | 0.8519 | 0.3688 | 5.5452 | 57,995 | 13,485 | 10,725 | 60,755 |

Group-macro 指标在各 split 的主要未见实体上等权计算：unseen peptide 按 peptide、unseen TCR 按 TCR、unseen pair 按 peptide–TCR pair 分组。Group-macro recall 仅对同时包含正负标签的组有定义。

## 数据检查

| Split | CSV | 原始行数 | 纳入行数 | 正 / 负 | 唯一 peptide | 唯一 TCR | 唯一 pair | 重复 pair 行 | 标签冲突 | peptide 长度 | TCR 长度 | 长度过滤 | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `test_unseen_pair` | [test_unseen_pair.csv](../../dataset/PT/new/test_unseen_pair.csv) | 8,207 | 8,207 | 3,113 / 5,094 | 192 | 7,425 | 8,207 | 0 | 0 | 8–13 aa | 7–32 aa | 0 | `119edba70e2b28276048e96917460f25787ffa04ec2e520593bc71fa66374879` |
| `test_unseen_peptide` | [test_unseen_peptide.csv](../../dataset/PT/new/test_unseen_peptide.csv) | 112,443 | 112,443 | 41,541 / 70,902 | 299 | 75,021 | 112,443 | 0 | 0 | 8–15 aa | 3–32 aa | 0 | `98e4e244b4564c8b432d8b7c12530236b5dc11c7969101b90dc662a14e998f39` |
| `test_unseen_tcr` | [test_unseen_tcr.csv](../../dataset/PT/new/test_unseen_tcr.csv) | 142,960 | 142,960 | 71,480 / 71,480 | 1,410 | 33,061 | 142,960 | 0 | 0 | 8–15 aa | 5–32 aa | 0 | `bfca9a02552e7085fc0dee4c14c57a89950c2c5a95a1177c8816e1986c661a5a` |

三个文件合计重复 pair 行 **0**，标签冲突 **0**；跨文件共享 pair **0**，跨文件标签冲突 **0**。

## Checkpoint 与评测设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [best_ptcr_20260928_021049_999878.pt](../../trained_model/PLM_Unified/fold_1/seed_3407/best_ptcr_20260928_021049_999878.pt) |
| 类型 / 任务 | `task_model` / `ptcr` |
| Fold / seed | 1 / 3407 |
| Stage / best round / global step | `stage1b` / 12 / 13,090 |
| 验证集最佳分数 | 0.639196 |
| Checkpoint SHA-256 | `14d4d1204c7c3be4acc590c1e3d20e84ba4b4c0d66a24a56305f14e5676b54f4` |
| ESM-C encoder | `/home/mjp/model/ESMC-300M` |
| TCR-BERT encoder | `/home/mjp/model/TCR-Bert` |
| 设备 | `NVIDIA GeForce RTX 4090` (`cuda`) |
| 混合精度 | pTCR head `bf16`；encoder `float16` |
| Encoder batch / pair batch | 128 / 1024 |
| Peptide / TCR 长度上限 | 15 / 34 aa |
| 评测时间 | 2026-09-28T20:23:02+08:00 |

Checkpoint 的 `best_score` 是训练验证集上的选模分数；PT new 指标来自独立测试文件，不参与选模或阈值选择。Stage 1 checkpoint 保存 pTCR head 和 adapters；ESM-C 与 TCR-BERT 使用配置中本地的冻结编码器权重，先对测试集唯一序列在线编码，再进行 pTCR 推理。

## 结果文件与复现

- 全量指标与数据审计：[evaluation_pt_new_20260928_201628.json](../../trained_model/PLM_Unified/fold_1/seed_3407/evaluation_pt_new_20260928_201628.json)
- 评估程序：[evaluate_plm_stage1_pt_new.py](../../source/evaluate_plm_stage1_pt_new.py)

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage1_pt_new.py \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/best_ptcr_20260928_021049_999878.pt \
  --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PT/new \
  --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage1/evaluation_pt_new_20260928_201628.md \
  --output-json /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/evaluation_pt_new_20260928_201628.json \
  --device cuda
```
