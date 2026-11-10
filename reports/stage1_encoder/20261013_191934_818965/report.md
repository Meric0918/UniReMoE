# Stage1 Encoder：两个分支最佳模型分析

实验 `20261013_191934_818965`：3 层 encoder，seed 42，fold 1；训练退出码为 0，在第 11 轮、12,155 step 正常结束。
本次对两个任务独立最佳 checkpoint 评估了六个独立 test splits，共 408,610 条样本。冻结权重、固定阈值 `p>0.5`、bf16 推理；缓存 PLM 指纹与训练 checkpoint 一致。

| 分支 | 最佳轮次 | 验证等权综合分 | 测试等权综合分 | 分析 |
| --- | --- | --- | --- | --- |
| pHLA | 10 | 0.7814 | 0.6804 | [pHLA 详细报告](./pHLA.md) |
| pTCR | 6 | 0.6457 | 0.6799 | [pTCR 详细报告](./pTCR.md) |

## 独立测试结果

| 分支 | 划分 | N | AUROC | AUPR | F1 | Recall | ECE | 综合分 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pHLA | 未见 HLA | 47,748 | 0.8582 | 0.8266 | 0.7427 | 0.6732 | 0.1242 | 0.7475 |
| pHLA | 双实体未见（unseen_pair） | 64 | 0.5532 | 0.5418 | 0.3774 | 0.3125 | 0.3898 | 0.3847 |
| pHLA | 未见 peptide | 62,764 | 0.9639 | 0.9610 | 0.9064 | 0.9280 | 0.0214 | 0.9090 |
| pTCR | 双实体未见（unseen_pair） | 10,274 | 0.7021 | 0.7603 | 0.5741 | 0.4228 | 0.2281 | 0.6322 |
| pTCR | 未见 peptide | 144,800 | 0.7034 | 0.7365 | 0.4714 | 0.3317 | 0.2445 | 0.5714 |
| pTCR | 未见 TCR | 142,960 | 0.9146 | 0.9129 | 0.8394 | 0.8492 | 0.0233 | 0.8359 |

## 结论

pHLA 的未见 peptide 结果稳定；未见 HLA 的召回与校准还有差距，双实体未见表现弱。双实体未见测试集仅 64 条，其中 22 个假阴性有 17 个集中在两个 HLA 伪序列；其验证集也仅 60 条。应扩大这一验证场景，避免小样本结果对等权选模产生过大影响。

pTCR 在已有 peptide 上的新 TCR 泛化稳定；未见 peptide 和双实体未见的主要问题是高置信度假阴性。未见 peptide 测试集漏检 48,382 个阳性，其中三个 peptide 贡献 30,395 个漏检。第 6 轮之后，训练指标改善而冷启动验证集下降，当前任务最佳选在第 6 轮有依据。

`unseen_pair` 测试样本中的两类实体均没有出现在对应任务训练集；该划分实际评估双实体冷启动。两任务未见 peptide 测试集中的 peptide 也未出现在另一个任务训练集中。

两分支测试宏平均综合分相近，但能力分布不同。pHLA 的小样本双实体划分与六万多条未见 peptide 样本等权，因此宏平均不能替代逐划分结果。

## 训练与模型使用

联合最佳模型在第 6 轮，联合验证分 0.706774。第 9/10 轮联合即时分数更高，但 pTCR 校准约束未通过，因此联合最佳未更新；第 11 轮触发联合 patience=5，同时 pTCR 任务 patience=5。pHLA 的独立任务 patience 未耗尽。

pHLA 第 10 轮和 pTCR 第 6 轮的独立 checkpoint 各自包含所属轮次的 Peptide encoder，分析与测试均使用对应文件。它们的共享 encoder 权重不同；若需要一套共享 encoder 同时运行两任务，应使用第 6 轮的联合 checkpoint，不能直接把两个任务最佳权重按同名参数覆盖拼接。

当前评估没有重新训练模型，也没有用 test 选择阈值。结果只对应 seed 42；多种子重复和同设置对照仍是评估架构稳定性的必要后续工作。

## 图表与复现

- [pHLA 诊断图](./phla_diagnostics.png)
- [pTCR 诊断图](./ptcr_diagnostics.png)
- [训练轨迹](./training_dynamics.png)
- [UMAP 特征展示](./feature_umap.png)
- [完整结构化分析](./analysis.json)
- [原始推理指标](./evaluation.json)

UMAP 来自训练期间固定 train/validation 子样本，三阶段分别建立参考空间；图中距离不能跨任务或跨阶段比较，也不能作为独立测试性能证据。

从项目根目录复现：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/20261013_191934_818965/analyze.py
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/20261013_191934_818965/summarize.py
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/20261013_191934_818965/write_report.py
```
