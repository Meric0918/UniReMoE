# Stage1 Encoder：PH / PT Benchmark 测试报告

- 测试完成时间：2026-10-28 16:18（北京时间）；fold 1、seed 42。
- **PH 使用第 6 轮联合最佳模型**：[best_joint_20261013_191934_818965.pt](../../../../trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_round/best_joint_20261013_191934_818965.pt)。检查点中保存的联合验证选模分数为 0.706774。
- PH 联合模型 SHA-256：`8cb86393dd5fcfa0f407161b4c630a676b80d2db4cfcfd18db1aaadc9aeb6947`。
- **PT 使用第 6 轮任务最佳模型**：[best_ptcr_20261013_191934_818965.pt](../../../../trained_model/PLM_Stage1_Encoder/fold_1/seed_42/best_ptcr_20261013_191934_818965.pt)。检查点中保存的任务验证选模分数为 0.645711。
- PT 任务模型 SHA-256：`38bd1791775a878664fd8c8da22464985d34e609c45b29fc23c9b2902bba9e52`。
- PT 任务最佳模型与联合最佳模型的 PT 分支及其共享 peptide encoder 权重逐项一致；两个模型在本次 PT 测试中的逐条预测概率也完全一致。
- 测试范围：PH 3 个文件、110,576 条；PT 3 个文件、298,034 条；总计 **6 个文件、408,610 条记录**。

## 测试方法

- 使用各任务 benchmark 目录下全部 `test_*.csv`；PH 使用 `label`、peptide 和 HLA，PT 使用 `label`、peptide 和 TCR。
- 使用与检查点编码器指纹匹配的冻结 PLM 残基缓存，在 CUDA 上以 bf16 和 batch size 1024 推理。没有重新训练，也没有使用测试集选择模型或阈值。
- 模型输出经 softmax 得到正类概率；按照该 Stage1 流程的固定规则，**概率 > 0.5** 判为正。表中 Accuracy、MCC、F1、Precision、Recall 和 Specificity 均基于该规则。
- AUPR 按仓库实现计算 precision-recall 曲线的梯形面积，而非 average precision；AUROC 和 AUPR 不依赖分类阈值。Brier 和 ECE 越低越好。
- ECE 使用 15 个等频分箱，按 0.5 确定预测类别并比较类别置信度与准确率；实现中 ECE 对概率恰好等于 0.5 的样本采用 `>= 0.5`，与上面的分类指标严格 `> 0.5` 规则有所区别。
- 6 个测试集正负样本均为 1:1。AUPR 和阈值相关指标会随样本分布改变，使用时应考虑目标数据的正样本比例。

## 排序与概率指标

| 任务及使用模型 | 测试集（链接为原始 CSV） | 样本数（正 / 负） | AUROC | AUPR | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| PH（联合最佳） | [test_unseen_hla](../../../../dataset/PH/benchmarks/test_unseen_hla.csv) | 47,748（23,874 / 23,874） | 0.8477 | 0.8184 | 0.1714 | 0.1023 |
| PH（联合最佳） | [test_unseen_pair](../../../../dataset/PH/benchmarks/test_unseen_pair.csv) | 64（32 / 32） | 0.5415 | 0.5670 | 0.3657 | 0.3440 |
| PH（联合最佳） | [test_unseen_peptide](../../../../dataset/PH/benchmarks/test_unseen_peptide.csv) | 62,764（31,382 / 31,382） | 0.9550 | 0.9513 | 0.0807 | 0.0218 |
| PT（任务最佳） | [test_unseen_pair](../../../../dataset/PT/benchmarks/test_unseen_pair.csv) | 10,274（5,137 / 5,137） | 0.7021 | 0.7603 | 0.2699 | 0.2281 |
| PT（任务最佳） | [test_unseen_peptide](../../../../dataset/PT/benchmarks/test_unseen_peptide.csv) | 144,800（72,400 / 72,400） | 0.7034 | 0.7365 | 0.2891 | 0.2445 |
| PT（任务最佳） | [test_unseen_tcr](../../../../dataset/PT/benchmarks/test_unseen_tcr.csv) | 142,960（71,480 / 71,480） | 0.9146 | 0.9129 | 0.1177 | 0.0233 |

## 分类指标：固定阈值 0.5

| 任务及使用模型 | 测试集 | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| PH（联合最佳） | test_unseen_hla | 0.7629 | 0.5333 | 0.7411 | 0.8159 | 0.6789 | 0.8469 |
| PH（联合最佳） | test_unseen_pair | 0.5469 | 0.1026 | 0.4314 | 0.5789 | 0.3438 | 0.7500 |
| PH（联合最佳） | test_unseen_peptide | 0.8935 | 0.7878 | 0.8959 | 0.8760 | 0.9166 | 0.8703 |
| PT（任务最佳） | test_unseen_pair | 0.6863 | 0.4384 | 0.5741 | 0.8938 | 0.4228 | 0.9498 |
| PT（任务最佳） | test_unseen_peptide | 0.6280 | 0.3178 | 0.4714 | 0.8141 | 0.3317 | 0.9243 |
| PT（任务最佳） | test_unseen_tcr | 0.8375 | 0.6752 | 0.8394 | 0.8298 | 0.8492 | 0.8258 |

## 数据与验证

- 六个文件的所有原始行均参与测试，长度、规范氨基酸、二元标签和重复序列对检查均通过；没有发现无效或重复记录。
- 使用当前 PH/PT 训练及 benchmark 验证文件核对时，六个测试集与相同任务的训练、验证数据均无相同的 peptide–HLA 或 peptide–TCR 监督对。当前训练/验证文件与测试文件的 SHA-256 记录在数据审计 JSON 中。
- 下表比较测试集所声称未见实体与 **PH/PT 当前训练池**的序列；pair 场景同时检查两类实体。分母为测试集中不同实体的数量。

| 任务 | 测试集 | 训练池已出现的不同实体 / 测试集不同实体总数 |
|---|---|---|
| PH | test_unseen_hla | HLA 0 / 21 |
| PH | test_unseen_pair | peptide 0 / 55；HLA 0 / 16 |
| PH | test_unseen_peptide | peptide 0 / 41,668 |
| PT | test_unseen_pair | peptide 0 / 193；TCR 0 / 9,382 |
| PT | test_unseen_peptide | peptide 0 / 302 |
| PT | test_unseen_tcr | TCR 0 / 33,061 |

- 这里的训练池仅包含该 Stage1 Encoder 训练所用的 PH/PT，不包含 PHT；因此 cold 审计口径与使用 PH/PT/PHT 三任务训练池的 MOE 报告不同。
- 每条预测与原始测试数据的文件、行号、序列及标签已核对；对输出概率独立复算了 AUROC、AUPR、Accuracy、MCC、F1、Precision、Recall、Specificity、NLL、Brier 和 ECE，均与评估结果一致。

## 结果解读

- PH 联合模型在 unseen-peptide 上的 AUROC / AUPR 为 **0.9550 / 0.9513**，unseen-HLA 为 **0.8477 / 0.8184**。
- PH unseen-pair 只有 64 条样本，联合模型 AUPR 为 **0.5670**、MCC 为 **0.1026**；小样本场景下结果可能波动较大。
- PT unseen-TCR 的 AUROC / AUPR 为 **0.9146 / 0.9129**；unseen-peptide 的 AUPR 为 **0.7365**、Recall 为 **0.3317**，该场景的阳性检出率相对较低。
- 结果对应单个 seed 和这些平衡 benchmark 划分，不能直接视作其他数据分布下的性能。

## 文件与复现

- [完整评估指标 JSON](metrics.json)、[数据审计 JSON](data_audit.json)、[逐行与指标独立复核 JSON](verification.json)。
- [评估脚本](run_benchmarks.py)、[复核脚本](verify_benchmarks.py)、[运行日志](evaluation.log)。原始指标 JSON 也保留了额外评估的 PH 任务最佳模型结果；**本报告 PH 的所有结果均使用联合最佳模型**。

| 任务 | 测试集 | 本报告使用的逐条预测 |
|---|---|---|
| PH（联合最佳） | test_unseen_hla | [预测 CSV.gz](joint_phla_test_unseen_hla_predictions.csv.gz) |
| PH（联合最佳） | test_unseen_pair | [预测 CSV.gz](joint_phla_test_unseen_pair_predictions.csv.gz) |
| PH（联合最佳） | test_unseen_peptide | [预测 CSV.gz](joint_phla_test_unseen_peptide_predictions.csv.gz) |
| PT（任务最佳） | test_unseen_pair | [预测 CSV.gz](task_best_ptcr_test_unseen_pair_predictions.csv.gz) |
| PT（任务最佳） | test_unseen_peptide | [预测 CSV.gz](task_best_ptcr_test_unseen_peptide_predictions.csv.gz) |
| PT（任务最佳） | test_unseen_tcr | [预测 CSV.gz](task_best_ptcr_test_unseen_tcr_predictions.csv.gz) |

从项目根目录复现推理及复核：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/mjp/miniconda3/envs/unifyimmun/bin/python -u \
  reports/stage1_encoder/20261013_191934_818965/benchmark_test_20261028/run_benchmarks.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/mjp/miniconda3/envs/unifyimmun/bin/python -u \
  reports/stage1_encoder/20261013_191934_818965/benchmark_test_20261028/verify_benchmarks.py
```
