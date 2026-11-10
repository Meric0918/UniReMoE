# ERGO-II PHT benchmarks 测试集指标

- 数据：`/home/mjp/Project/PLM/UnifyImmun/dataset/PHT/benchmarks` 中的四个 `test_unseen_*` split。
- Checkpoint：`/home/mjp/Project/ERGO-II/PHT_logs/ergo_ii_pht/version_pht_lstm_hla_20261011_r1/checkpoints/epoch=3.ckpt`。
- AUPR 使用 `average_precision_score`（Average Precision）计算；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 `PHT probability >= 0.5`。

| Test split | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_unseen_hla.csv | 0.631198 | 0.520606 | 0.485257 | -0.080931 | 0.036723 | 0.285503 | 0.019624 | 0.950890 |
| test_unseen_peptide.csv | 0.944568 | 0.952802 | 0.869670 | 0.742171 | 0.863724 | 0.905013 | 0.826038 | 0.913302 |
| test_unseen_tcr.csv | 0.988567 | 0.989790 | 0.941803 | 0.883623 | 0.941984 | 0.939065 | 0.944921 | 0.938685 |
| test_unseen_triple.csv | 0.855684 | 0.809295 | 0.713636 | 0.454765 | 0.654479 | 0.824885 | 0.542424 | 0.884848 |
| All test splits (pooled) | 0.776662 | 0.804195 | 0.700992 | 0.454896 | 0.609628 | 0.877866 | 0.466948 | 0.935035 |

注：pooled 行将四个 test split 的样本合并后重新计算。使用的是当前保存的最低验证集 loss checkpoint；训练中验证 AUC 最高轮的权重没有保存。
