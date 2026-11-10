# Stage1 Encoder 最佳 pHLA 模型：PH/new 测试结果

使用 **第 10 轮、seed 42、3 层 encoder** 的独立 pHLA 最佳模型，
对 [dataset/PH/new](../../../../dataset/PH/new) 下 3 个 `test_*.csv` 文件完成全量推理，合计 **87,511** 条样本（正类 40,675，负类 46,836）。

模型：[best_phla_20261013_191934_818965.pt](../../../../trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_phla_20261013_191934_818965.pt)；训练验证选模分数 **0.781363**。
测试固定使用 `probability > 0.5`，没有依据测试标签调整权重、阈值或选模。

## 测试指标

| Split | N | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 37.54% | 0.8388 | 0.7101 | 0.7685 | 0.4983 | 0.6763 | 0.7117 | 0.6443 | 0.8431 | 0.6984 |
| `test_unseen_pair` | 51 | 43.14% | 0.4655 | 0.4203 | 0.4118 | -0.2965 | 0.1176 | 0.1667 | 0.0909 | 0.6552 | 0.2238 |
| `test_unseen_peptide` | 62,764 | 50.00% | 0.9639 | 0.9611 | 0.9042 | 0.8094 | 0.9065 | 0.8859 | 0.9280 | 0.8805 | 0.9090 |

综合分为 `(AUROC + AUPR + Accuracy + MCC + F1) / 5`。

| 汇总方式 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | 0.7561 | 0.6972 | 0.6948 | 0.3371 | 0.5668 | 0.5881 | 0.5544 | 0.7929 | 0.6104 |
| 全部样本合并后重新计算 | 0.9377 | 0.9275 | 0.8656 | 0.7303 | 0.8565 | 0.8503 | 0.8629 | 0.8681 | 0.8635 |

宏平均给每个 split 相同权重；pooled 指标将全部预测合并后计算，不是各 split 指标的加权平均。

## 混淆矩阵与概率校准

| Split | TP | FN | TN | FP | 高置信 FN（p<0.2） | 高置信 FP（p>0.8） | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 5,973 | 3,298 | 13,005 | 2,420 | 1,658 | 1,090 | 0.5383 | 0.1682 | 0.0903 |
| `test_unseen_pair` | 2 | 20 | 19 | 10 | 17 | 3 | 1.4790 | 0.4220 | 0.4776 |
| `test_unseen_peptide` | 29,122 | 2,260 | 27,632 | 3,750 | 1,049 | 1,913 | 0.2459 | 0.0722 | 0.0212 |

NLL、Brier、ECE 越小越好。ECE 使用项目实现的 15 个等频概率 bin 的置信度校准误差；其 0.5 平局内部使用 `>=`，分类指标使用严格 `>`。

## 结果解读

综合分最高的是 `test_unseen_peptide`（0.9090），最低的是 `test_unseen_pair`（0.2238）。

- `test_unseen_hla`：AUROC 0.8388，Precision 71.17%，Recall 64.43%，ECE 0.0903；9,271 个阳性中漏检 3,298 个，其中 1,658 个阳性概率低于 0.2。
- `test_unseen_pair`：AUROC 0.4655，Precision 16.67%，Recall 9.09%，ECE 0.4776；22 个阳性中漏检 20 个，其中 17 个阳性概率低于 0.2。
- `test_unseen_peptide`：AUROC 0.9639，Precision 88.59%，Recall 92.80%，ECE 0.0212；31,382 个阳性中漏检 2,260 个，其中 1,049 个阳性概率低于 0.2。

这些测试集的正类比例并不完全相同，AUPR 和 Accuracy 应结合各自类别比例解读。

`test_unseen_pair` 只有 **51** 条样本，单条样本即可使 Accuracy 变化 1.96 个百分点，且与大样本 split 等权参与宏平均；应结合样本量和逐 split 结果判断。

数据与训练集的实体、配对重叠如下。新实体划分名称仅作文件标识，实际重叠数用于解释测试难度。

| Split | peptide 行数在训练中出现 | receptor 行数在训练中出现 | 配对行数在训练中出现 |
|---|---:|---:|---:|
| `test_unseen_hla` | 15,699 | 0 | 0 |
| `test_unseen_pair` | 0 | 0 | 0 |
| `test_unseen_peptide` | 0 | 62,764 | 0 |

## 数据检查

| Split | 原始 / 纳入 | 正 / 负 | 唯一 peptide | 唯一 receptor | 重复配对行 | 标签冲突行 |
|---|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 / 24,696 | 9,271 / 15,425 | 22,700 | 18 | 0 | 0 |
| `test_unseen_pair` | 51 / 51 | 22 / 29 | 46 | 13 | 0 | 0 |
| `test_unseen_peptide` | 62,764 / 62,764 | 31,382 / 31,382 | 41,668 | 99 | 0 | 0 |

跨文件重复配对行 0，跨文件标签冲突行 0。具体来源、长度范围、过滤统计和文件 SHA256 保存在 JSON。

### 与前次 benchmark 数据的差异

| Split | benchmark 配对数 | new 配对数 | 移除阳性 | 移除阴性 | 新增配对 | 共有配对标签变化 |
|---|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 47,748 | 24,696 | 14,603 | 8,449 | 0 | 0 |
| `test_unseen_pair` | 64 | 51 | 10 | 3 | 0 | 0 |
| `test_unseen_peptide` | 62,764 | 62,764 | 0 | 0 | 0 | 0 |

本次使用同一批训练 checkpoint。`new` 与前次 benchmark 的测试样本集和正类比例不同，比较指标时应同时考虑这些数据变化。

## 分组指标

| Split | 分组对象 | 组数 | Group accuracy | Group recall | Recall coverage | Group NLL |
|---|---|---:|---:|---:|---:|---:|
| `test_unseen_hla` | HLA | 18 | 0.7624 | 0.5980 | 0.9928 | 0.5269 |
| `test_unseen_pair` | peptide-HLA pair | 51 | 0.4118 | — | 0.0000 | 1.4790 |
| `test_unseen_peptide` | peptide | 41,668 | 0.9113 | 0.9302 | 0.3591 | 0.2293 |

分组 Recall 仅对同时包含正负类的组计算；没有符合条件的组时显示 `—`。

## 模型与评估设置

| 项目 | 值 |
|---|---|
| 架构 | `stage1_encoder`；3 层、8 头、256 维 |
| Checkpoint round / seed / SHA256 | 10 / 42 / `cc30664aafdbe8436d45d43710af1850d0c31153a9bed76489da564d790b04b1` |
| 配置快照 | [stage1_encoder_20261013_191934_818965.yaml](../../../../training_logs/stage1_encoder_20261013_191934_818965.yaml) |
| 精度 / batch size / workers | bf16 / 1024 / 2 |
| 设备 | NVIDIA GeForce RTX 4090 |
| 评估时间 | 2026-10-14T01:07:59+08:00 |

本次使用已有冻结 PLM 残基缓存，按序列精确查找并校验 hidden dimension、PLM checkpoint fingerprint 及有效残基 mask。Peptide/HLA 输入 960 维，TCR 输入 768 维；新加的 sequence encoder 参数来自对应任务最佳 checkpoint。

## 结果文件与复现

- [完整指标和审计 JSON](./stage1_encoder_PH_new.json)
- [评估程序](../../../../source/evaluate_plm_stage1_encoder_new.py)

逐行预测保存为当前目录下对应 split 的 `predictions.csv.gz`，可用于进一步错误分析。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage1_encoder_new.py \
  --config /home/mjp/Project/PLM/UnifyImmun/training_logs/stage1_encoder_20261013_191934_818965.yaml \
  --tasks phla \
  --phla-checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_phla_20261013_191934_818965.pt \
  --ph-data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PH/new \
  --cache-root /home/mjp/Project/PLM/UnifyImmun/embedding_cache/plm_stage1_ph_pt/fold_1 \
  --output-dir /home/mjp/Project/PLM/UnifyImmun/reports/stage1_encoder/20261013_191934_818965/new_tests_20261014_010727_036600 --device cuda --batch-size 1024 --num-workers 2
```
