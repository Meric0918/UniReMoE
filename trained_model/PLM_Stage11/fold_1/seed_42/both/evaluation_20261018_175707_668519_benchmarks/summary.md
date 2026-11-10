# Stage11 PHT benchmarks 测试总结

## 测试设置

- 模型：`stage11_best_20261018_175707_668519.pt`，第 9 轮最佳 checkpoint；验证集选择分数为 0.989305。
- 数据：`dataset/PHT/benchmarks/` 下的 unseen peptide、unseen HLA、unseen TCR、unseen triple 四个 test split，共 42,146 条样本。
- 判定阈值：固定 0.5；报告 AUROC、AUPR 及阈值指标。
- 四个 split 均通过序列和标签审计；没有发现与训练集重复的完整 PHT 三元组。

## 结果

| 测试集 | 样本数 | PHT 正例 `111` | PHT 负例 | AUROC | AUPR | MCC | F1 | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unseen peptide | 14,118 | 7,059 | 7,059 | 0.982465 | 0.979637 | 0.726467 | 0.824979 | 0.711149 | 0.987109 |
| Unseen HLA | 19,670 | 9,835 | 9,835 | 0.705099 | 0.579301 | 0.012966 | 0.038269 | 0.019827 | 0.983630 |
| Unseen TCR | 7,698 | 3,849 | 3,849 | 0.997913 | 0.998052 | 0.943868 | 0.970966 | 0.951416 | 0.991686 |
| Unseen triple | 660 | 330 | 330 | 0.838531 | 0.845706 | 0.475379 | 0.593814 | 0.436364 | 0.966667 |

## 结果解读

模型在 unseen TCR 上表现最好，unseen peptide 上也保持较强区分能力。unseen triple 的区分能力中等，阈值 0.5 下正例召回率为 0.436。

unseen HLA 上的阈值表现较弱：虽然 AUROC 为 0.705，固定阈值下 Recall 仅 0.020，MCC 为 0.013。这表示当前 checkpoint 在该 split 上很少将样本判为正例；阈值敏感指标不支持其具备可靠的 unseen HLA 正例检出能力。

每个 split 都有 `010`、`100` 类各 1 条样本（unseen triple 则 `000`、`100` 各 1 条）。因此，单独按这两类计算的假阳性率不稳定；总体结果主要反映各 split 中占比最大的类别。

## 输出文件

- `summary.json`：完整评估指标、数据审计信息与训练三元组重叠计数。
- `test_unseen_peptide_predictions.csv`
- `test_unseen_hla_predictions.csv`
- `test_unseen_tcr_predictions.csv`
- `test_unseen_triple_predictions.csv`

预测文件逐行保留序列、原始标签、类别代码、预测概率和阈值判定结果。
