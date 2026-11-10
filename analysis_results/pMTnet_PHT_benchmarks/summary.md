# pMTnet 最佳模型 PHT Test 集指标

模型为第 5 轮最佳 checkpoint；PHT 概率阈值为 0.5。AUPR 使用 average precision 计算；Specificity = TN / (TN + FP)。

| 数据集 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_unseen（总体） | 0.7439 | 0.7938 | 0.7107 | 0.4973 | 0.6061 | 0.9493 | 0.4452 | 0.9762 |
| test_unseen_hla.csv | 0.5017 | 0.4478 | 0.4903 | -0.0947 | 0.0018 | 0.0431 | 0.0009 | 0.9797 |
| test_unseen_peptide.csv | 0.9840 | 0.9824 | 0.8849 | 0.7801 | 0.8748 | 0.9593 | 0.8039 | 0.9659 |
| test_unseen_tcr.csv | 0.9959 | 0.9963 | 0.9587 | 0.9190 | 0.9574 | 0.9873 | 0.9293 | 0.9880 |
| test_unseen_triple.csv | 0.7006 | 0.6600 | 0.6606 | 0.3993 | 0.5172 | 0.8955 | 0.3636 | 0.9576 |
