# PLM Stage MOE 最佳模型：PH / PT / PHT Benchmark Test 总结

## 模型与评估设置

- 模型：joint 训练，seed 42，Epoch 12 的 `best.pt`
- checkpoint 验证集选择分数：三项任务 mean task PR-AUC = **0.900117**
- checkpoint SHA-256：`1ea07b480c05658beff7b5b9519b426dfeeb9529fb8efddab9fe8d2eed9e7f58`
- 测试范围：PH 3 个、PT 3 个、PHT 4 个 benchmark，共 **450,756** 条记录
- 标签策略：每个 benchmark 只使用对应任务的原生标签；PHT 文件使用 `PHT` 标签
- 分类阈值：使用 checkpoint 中保存的验证集阈值，没有在 test 数据上重新调阈值
- 数据检查：10 个文件均无重复记录、无因序列格式无效而过滤的记录；test 记录与对应任务 train/val 监督样本无重叠

## 各 Benchmark 结果

| 任务 / Benchmark | 样本数 | 正 / 负 | AUROC | AUPRC | Accuracy | MCC | F1 | 验证集阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PH / unseen HLA | 47,748 | 23,874 / 23,874 | 0.872767 | 0.858858 | 0.809353 | 0.619021 | 0.806266 | 0.057493 |
| PH / unseen pair | 64 | 32 / 32 | 0.563477 | 0.510273 | 0.531250 | 0.063628 | 0.482759 | 0.057493 |
| PH / unseen peptide | 62,764 | 31,382 / 31,382 | 0.965484 | 0.961858 | 0.906634 | 0.816643 | 0.910690 | 0.057493 |
| PT / unseen pair | 10,274 | 5,137 / 5,137 | 0.762409 | 0.830802 | 0.762215 | 0.560940 | 0.710922 | 0.622000 |
| PT / unseen peptide | 144,800 | 72,400 / 72,400 | 0.661672 | 0.739841 | 0.655338 | 0.384559 | 0.511343 | 0.622000 |
| PT / unseen TCR | 142,960 | 71,480 / 71,480 | 0.892945 | 0.901076 | 0.817138 | 0.641386 | 0.802472 | 0.622000 |
| PHT / unseen HLA | 19,670 | 9,835 / 9,835 | 0.864188 | 0.735581 | 0.869548 | 0.743150 | 0.876015 | 1.691898e-10 |
| PHT / unseen peptide | 14,118 | 7,059 / 7,059 | 0.979166 | 0.980995 | 0.924777 | 0.851019 | 0.922504 | 1.691898e-10 |
| PHT / unseen TCR | 7,698 | 3,849 / 3,849 | 0.999271 | 0.999333 | 0.983632 | 0.967553 | 0.983830 | 1.691898e-10 |
| PHT / unseen triple | 660 | 330 / 330 | 0.800115 | 0.774932 | 0.680303 | 0.373831 | 0.631763 | 1.691898e-10 |

## 结果解读

- PH unseen peptide 表现较强（AUROC 0.965484，AUPRC 0.961858）；unseen HLA 的 AUROC/AUPRC 为 0.872767/0.858858。
- PH unseen pair 只有 64 条样本，AUROC 0.563477、MCC 0.063628；该结果受样本量限制，应谨慎解读。
- PT unseen TCR 的 AUROC/AUPRC 为 0.892945/0.901076。PT unseen peptide 相对较弱，AUROC 0.661672、MCC 0.384559。
- PHT unseen peptide 和 unseen TCR 的排序指标较高；PHT unseen triple 有 660 条样本，AUROC 0.800115、MCC 0.373831，样本量相对较小。
- Cold audit 发现 PT unseen TCR 有 **5 / 33,061** 个 unique TCR、PHT unseen TCR 有 **4 / 6,808** 个 unique TCR 出现在任一训练任务中，因此这两个文件不是完全严格的 unseen-TCR split。其余标注为 unseen HLA/peptide 的 benchmark 没有发现对应实体与训练集重合。
- PHT 使用的验证集阈值约为 `1.69e-10`。AUROC/AUPRC 反映分数排序；Accuracy、MCC、F1 则依赖该阈值。

## 产物

- 指标明细：[metrics.json](metrics.json)
- 评估器生成的表格：[metrics.md](metrics.md)
- 逐样本概率、预测标签和 MoE routing 权重：本目录下的 10 个 `*_predictions.csv` 文件
- 最佳 checkpoint：[best.pt](/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE/seed_42/joint_online_20261027_193829/best.pt)
