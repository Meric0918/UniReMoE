# Stage1 Encoder 最佳 pTCR 模型：PT/new 测试结果

使用 **第 6 轮、seed 42、3 层 encoder** 的独立 pTCR 最佳模型，
对 [dataset/PT/new](../../../../dataset/PT/new) 下 3 个 `test_*.csv` 文件完成全量推理，合计 **263,610** 条样本（正类 116,134，负类 147,476）。

模型：[best_ptcr_20261013_191934_818965.pt](../../../../trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_ptcr_20261013_191934_818965.pt)；训练验证选模分数 **0.645711**。
测试固定使用 `probability > 0.5`，没有依据测试标签调整权重、阈值或选模。

## 测试指标

| Split | N | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 37.93% | 0.8835 | 0.8563 | 0.8462 | 0.6714 | 0.7697 | 0.8910 | 0.6775 | 0.9494 | 0.8054 |
| `test_unseen_peptide` | 112,443 | 36.94% | 0.8259 | 0.7927 | 0.7915 | 0.5406 | 0.6679 | 0.8116 | 0.5674 | 0.9228 | 0.7237 |
| `test_unseen_tcr` | 142,960 | 50.00% | 0.9146 | 0.9128 | 0.8375 | 0.6752 | 0.8394 | 0.8298 | 0.8492 | 0.8258 | 0.8359 |

综合分为 `(AUROC + AUPR + Accuracy + MCC + F1) / 5`。

| 汇总方式 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 split 等权宏平均 | 0.8747 | 0.8539 | 0.8251 | 0.6291 | 0.7590 | 0.8441 | 0.6980 | 0.8993 | 0.7884 |
| 全部样本合并后重新计算 | 0.8858 | 0.8760 | 0.8182 | 0.6297 | 0.7828 | 0.8261 | 0.7438 | 0.8767 | 0.7985 |

宏平均给每个 split 相同权重；pooled 指标将全部预测合并后计算，不是各 split 指标的加权平均。

## 混淆矩阵与概率校准

| Split | TP | FN | TN | FP | 高置信 FN（p<0.2） | 高置信 FP（p>0.8） | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 2,109 | 1,004 | 4,836 | 258 | 747 | 170 | 0.4236 | 0.1251 | 0.0643 |
| `test_unseen_peptide` | 23,570 | 17,971 | 65,430 | 5,472 | 11,640 | 1,774 | 0.5103 | 0.1583 | 0.0793 |
| `test_unseen_tcr` | 60,702 | 10,778 | 59,031 | 12,449 | 4,630 | 4,402 | 0.3747 | 0.1177 | 0.0233 |

NLL、Brier、ECE 越小越好。ECE 使用项目实现的 15 个等频概率 bin 的置信度校准误差；其 0.5 平局内部使用 `>=`，分类指标使用严格 `>`。

## 结果解读

综合分最高的是 `test_unseen_tcr`（0.8359），最低的是 `test_unseen_peptide`（0.7237）。

- `test_unseen_pair`：AUROC 0.8835，Precision 89.10%，Recall 67.75%，ECE 0.0643；3,113 个阳性中漏检 1,004 个，其中 747 个阳性概率低于 0.2。
- `test_unseen_peptide`：AUROC 0.8259，Precision 81.16%，Recall 56.74%，ECE 0.0793；41,541 个阳性中漏检 17,971 个，其中 11,640 个阳性概率低于 0.2。
- `test_unseen_tcr`：AUROC 0.9146，Precision 82.98%，Recall 84.92%，ECE 0.0233；71,480 个阳性中漏检 10,778 个，其中 4,630 个阳性概率低于 0.2。

这些测试集的正类比例并不完全相同，AUPR 和 Accuracy 应结合各自类别比例解读。

数据与训练集的实体、配对重叠如下。新实体划分名称仅作文件标识，实际重叠数用于解释测试难度。

| Split | peptide 行数在训练中出现 | receptor 行数在训练中出现 | 配对行数在训练中出现 |
|---|---:|---:|---:|
| `test_unseen_pair` | 0 | 0 | 0 |
| `test_unseen_peptide` | 0 | 107,598 | 0 |
| `test_unseen_tcr` | 142,954 | 0 | 0 |

## 数据检查

| Split | 原始 / 纳入 | 正 / 负 | 唯一 peptide | 唯一 receptor | 重复配对行 | 标签冲突行 |
|---|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 / 8,207 | 3,113 / 5,094 | 192 | 7,425 | 0 | 0 |
| `test_unseen_peptide` | 112,443 / 112,443 | 41,541 / 70,902 | 299 | 75,021 | 0 | 0 |
| `test_unseen_tcr` | 142,960 / 142,960 | 71,480 / 71,480 | 1,410 | 33,061 | 0 | 0 |

跨文件重复配对行 0，跨文件标签冲突行 0。具体来源、长度范围、过滤统计和文件 SHA256 保存在 JSON。

### 与前次 benchmark 数据的差异

| Split | benchmark 配对数 | new 配对数 | 移除阳性 | 移除阴性 | 新增配对 | 共有配对标签变化 |
|---|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 10,274 | 8,207 | 2,024 | 43 | 0 | 0 |
| `test_unseen_peptide` | 144,800 | 112,443 | 30,859 | 1,498 | 0 | 0 |
| `test_unseen_tcr` | 142,960 | 142,960 | 0 | 0 | 0 | 0 |

本次使用同一批训练 checkpoint。`new` 与前次 benchmark 的测试样本集和正类比例不同，比较指标时应同时考虑这些数据变化。

`test_unseen_pair` 移除配对主要来自 peptide：`YVLDHLIVV`（2,067 条）。

`test_unseen_peptide` 移除配对主要来自 peptide：`YVLDHLIVV`（12,323 条）、`LLFLVLIML`（12,263 条）、`HTTDPSFLGRY`（7,771 条）。

## 分组指标

| Split | 分组对象 | 组数 | Group accuracy | Group recall | Recall coverage | Group NLL |
|---|---|---:|---:|---:|---:|---:|
| `test_unseen_pair` | peptide–TCR pair | 8,207 | 0.8462 | — | 0.0000 | 0.4236 |
| `test_unseen_peptide` | peptide | 299 | 0.8721 | 0.2354 | 0.9774 | 0.3387 |
| `test_unseen_tcr` | TCR | 33,061 | 0.8439 | 0.7706 | 0.8519 | 0.3565 |

分组 Recall 仅对同时包含正负类的组计算；没有符合条件的组时显示 `—`。

## 模型与评估设置

| 项目 | 值 |
|---|---|
| 架构 | `stage1_encoder`；3 层、8 头、256 维 |
| Checkpoint round / seed / SHA256 | 6 / 42 / `38bd1791775a878664fd8c8da22464985d34e609c45b29fc23c9b2902bba9e52` |
| 配置快照 | [stage1_encoder_20261013_191934_818965.yaml](../../../../training_logs/stage1_encoder_20261013_191934_818965.yaml) |
| 精度 / batch size / workers | bf16 / 1024 / 2 |
| 设备 | NVIDIA GeForce RTX 4090 |
| 评估时间 | 2026-10-14T01:10:09+08:00 |

本次使用已有冻结 PLM 残基缓存，按序列精确查找并校验 hidden dimension、PLM checkpoint fingerprint 及有效残基 mask。Peptide/HLA 输入 960 维，TCR 输入 768 维；新加的 sequence encoder 参数来自对应任务最佳 checkpoint。

## 结果文件与复现

- [完整指标和审计 JSON](./stage1_encoder_PT_new.json)
- [评估程序](../../../../source/evaluate_plm_stage1_encoder_new.py)

逐行预测保存为当前目录下对应 split 的 `predictions.csv.gz`，可用于进一步错误分析。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage1_encoder_new.py \
  --config /home/mjp/Project/PLM/UnifyImmun/training_logs/stage1_encoder_20261013_191934_818965.yaml \
  --tasks ptcr \
  --ptcr-checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_ptcr_20261013_191934_818965.pt \
  --pt-data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PT/new \
  --cache-root /home/mjp/Project/PLM/UnifyImmun/embedding_cache/plm_stage1_ph_pt/fold_1 \
  --output-dir /home/mjp/Project/PLM/UnifyImmun/reports/stage1_encoder/20261013_191934_818965/new_tests_20261014_010727_036600 --device cuda --batch-size 1024 --num-workers 2
```
