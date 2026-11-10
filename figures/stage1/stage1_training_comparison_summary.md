# Stage1：128、256_early 与 256 训练曲线对比

生成日期：2026-09-13

## 1. 对比对象与数据来源

三组图像均使用监控脚本读取训练记录生成，曲线含义保持一致：实线为训练集，虚线为验证集；指标分别为 loss、AUROC 和 AUPR。

| 图像目录 | 模型/训练设置 | 监控来源 | Stage1B 终点 |
|---|---:|---|---|
| `128` | `model_dim=128` | `swanlog/run-20260824_174207-atzgps52okvljt1gllrtg` | pHLA/pTCR 均为 round 10 |
| `256_early` | `model_dim=256`，短训练 | `swanlog/run-20260831_174047-frlqhls5o52qllgdm8q85` | pHLA/pTCR 均为 round 10 |
| `256` | `model_dim=256`，长训练 | `swanlog/run-20260901_214057-0nycpeulwmebbp1bdhjze` | pHLA round 58，pTCR round 36 |

对应图像：

| 指标 | 128 | 256_early | 256 |
|---|---|---|---|
| Loss | [查看](128/stage1_loss_training_curve.png) | [查看](256_early/stage1_loss_training_curve.png) | [查看](256/stage1_loss_training_curve.png) |
| AUROC | [查看](128/stage1_auroc_training_curve.png) | [查看](256_early/stage1_auroc_training_curve.png) | [查看](256/stage1_auroc_training_curve.png) |
| AUPR | [查看](128/stage1_aupr_training_curve.png) | [查看](256_early/stage1_aupr_training_curve.png) | [查看](256/stage1_aupr_training_curve.png) |

## 2. 结论摘要

1. **训练集拟合能力：256 长训练最强。** 在各自最终点，256 的 pHLA/pTCR 训练 loss 最低，AUROC 和 AUPR 最高；但它经历了 36–58 个 round，不能与只训练 10 个 round 的结果直接等价比较。
2. **10 轮公平比较：256_early 综合表现最好。** 六个验证 split 的最终平均 loss/AUROC/AUPR 为 `0.7310 / 0.7894 / 0.8049`，优于 128 的 `0.7532 / 0.7881 / 0.7937`；256 在第 10 轮时为 `0.7882 / 0.7621 / 0.7832`。
3. **长训练主要提升 pHLA 的排序指标。** 256 长训练最终 pHLA 验证 AUROC/AUPR 明显高于另外两组，尤其是 `val_unseen_peptide`；但 `val_unseen_hla` 和 `val_unseen_pair` 的 loss 明显升高，说明概率校准/置信度变差。
4. **pTCR 存在任务分化。** 256 长训练的 `val_unseen_pair` 和 `val_unseen_peptide` AUROC/AUPR 略有优势，但 `val_unseen_tcr` 从约 `0.91/0.91` 降至 `0.8627/0.8780`，同时三个 pTCR 验证 loss 均明显升高。
5. **建议：** 若目标是平衡 pHLA 与 pTCR 的泛化，优先使用 256_early 或基于验证集宏平均指标选择 checkpoint；若重点是 pHLA 排序性能，可在 256 长训练的约 round 45–56 选择 pHLA 最优 checkpoint，而不要直接使用最终 checkpoint。

## 3. Stage1B 训练集最终指标

数值顺序均为 `loss / AUROC / AUPR`。

| 图像目录 | 终点 | pHLA | pTCR |
|---|---|---:|---:|
| `128` | round 10 | `0.2903 / 0.9475 / 0.9425` | `0.3389 / 0.9292 / 0.9270` |
| `256_early` | round 10 | `0.3048 / 0.9424 / 0.9373` | `0.3286 / 0.9335 / 0.9307` |
| `256` | pHLA round 58；pTCR round 36 | `0.1868 / 0.9780 / 0.9761` | `0.2336 / 0.9663 / 0.9644` |

与 128 相比，256 长训练最终训练 loss 分别下降约 35.6%（pHLA）和 31.1%（pTCR）；训练 AUROC 分别提高约 3.1 和 3.7 个百分点，AUPR 分别提高约 3.4 和 3.7 个百分点。这说明长训练显著增强了训练集拟合，但不能单独据此判断泛化最好。

## 4. 各验证 split 最终指标对比

数值顺序为 `loss / AUROC / AUPR`；这里的“最终”是每个目录图像中最后一个可用点。

### pHLA

| 验证 split | 128 | 256_early | 256 |
|---|---:|---:|---:|
| `val_unseen_hla` | `0.8594 / 0.7682 / 0.7420` | `0.7246 / 0.7633 / 0.7311` | `1.0151 / 0.8159 / 0.8307` |
| `val_unseen_pair` | `0.7977 / 0.7589 / 0.7404` | `0.7927 / 0.7900 / 0.8117` | `1.0084 / 0.8544 / 0.8784` |
| `val_unseen_peptide` | `0.2796 / 0.9515 / 0.9468` | `0.2967 / 0.9456 / 0.9410` | `0.2217 / 0.9703 / 0.9678` |

### pTCR

| 验证 split | 128 | 256_early | 256 |
|---|---:|---:|---:|
| `val_unseen_pair` | `1.2387 / 0.6264 / 0.6752` | `1.1601 / 0.6500 / 0.7008` | `1.4032 / 0.6699 / 0.7215` |
| `val_unseen_peptide` | `0.9551 / 0.7148 / 0.7489` | `1.0166 / 0.6803 / 0.7369` | `1.2910 / 0.7005 / 0.7553` |
| `val_unseen_tcr` | `0.3886 / 0.9090 / 0.9091` | `0.3956 / 0.9071 / 0.9079` | `0.6504 / 0.8627 / 0.8780` |

## 5. 验证集宏平均

宏平均对六个验证 split 等权，不代表官方模型选择分数，仅用于横向观察。

### 10 轮公平比较

| 设置 | 平均 loss | 平均 AUROC | 平均 AUPR |
|---|---:|---:|---:|
| 128，round 10 | 0.7532 | 0.7881 | 0.7937 |
| 256_early，round 10 | **0.7310** | **0.7894** | **0.8049** |
| 256，round 10 | 0.7882 | 0.7621 | 0.7832 |

### 各自最终点

| 设置 | 平均 loss | 平均 AUROC | 平均 AUPR |
|---|---:|---:|---:|
| 128，round 10 | 0.7532 | 0.7881 | 0.7937 |
| 256_early，round 10 | **0.7310** | 0.7894 | 0.8049 |
| 256，pHLA round 58 / pTCR round 36 | 0.9316 | **0.8123** | **0.8386** |

256 最终点的平均 AUROC/AUPR 最高，但平均 loss 也最高。结合曲线看，这更像是模型排序能力增强同时输出置信度变得过激，而不是所有维度上的全面改善。

## 6. 256 长训练中的最佳验证点

256 的最终点并不是所有验证指标的最佳点：

| 任务/split | 最佳 AUROC（round） | 最佳 AUPR（round） | 终点表现 |
|---|---:|---:|---:|
| pHLA `val_unseen_hla` | 0.8180（45） | 0.8316（45） | 0.8159 / 0.8307 |
| pHLA `val_unseen_pair` | 0.8678（52） | 0.8947（52） | 0.8544 / 0.8784 |
| pHLA `val_unseen_peptide` | 0.9708（56） | 0.9685（56） | 0.9703 / 0.9678 |
| pTCR `val_unseen_pair` | 0.6905（35） | 0.7326（35） | 0.6699 / 0.7215 |
| pTCR `val_unseen_peptide` | 0.7137（34） | 0.7623（34） | 0.7005 / 0.7553 |
| pTCR `val_unseen_tcr` | 0.9094（7） | 0.9100（7） | 0.8627 / 0.8780 |

## 7. 总体判断

- **训练 loss：** 256 长训练下降最充分；128 与 256_early 在 10 轮时都已明显收敛，但 256_early 的 pTCR 训练 loss 略低，128 的 pHLA 训练 loss 略低。
- **AUROC/AUPR：** 256 长训练对 pHLA 的验证排序提升最明显；pTCR 的 pair/peptide split 有小幅收益，但 unseen-TCR split 明显退化。
- **泛化与校准：** 256 长训练的高 AUROC/AUPR 伴随更高验证 loss，建议同时监控排序指标和 loss，不要只按单一指标选最终模型。
- **推荐模型选择策略：** 对 pHLA 与 pTCR 分别保存最佳 checkpoint，再计算六个验证 split 的宏平均；如果必须选择一个统一 checkpoint，256_early 是较稳妥的 10 轮平衡基线，256 长训练则应选择中途验证最优点而不是最后一轮。

## 8. 注意事项

三组实验的训练轮数不同，且来自不同时间的训练记录；因此本文件适合做曲线和结果对比，不应视为严格的单变量消融实验。尤其是 256 的最终点包含额外训练轮数，和前两组的 round 10 终点需分开解读。
