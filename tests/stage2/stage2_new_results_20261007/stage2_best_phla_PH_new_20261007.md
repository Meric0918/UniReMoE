# PLM Unified Stage 2：PH new 数据集 pHLA 测试结果

## 汇总

| 汇总方式 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | performance_avg | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三组等权宏平均 | 87,511 | 0.679014 | 0.621338 | 0.692564 | 0.294184 | 0.463497 | 0.550119 | 1.089427 | 0.263753 | 0.246382 |
| 全部样本 pooled | 87,511 | 0.911196 | 0.911962 | 0.847002 | 0.693688 | 0.826390 | 0.838048 | 0.439237 | 0.121889 | 0.076205 |

合计 **87,511** 条：负样本 46,836，正样本 40,675（46.48%）。
宏平均对三个 split 等权；pooled 将样本合并后重新计算。固定使用配置阈值 0.5，未根据测试标签调整阈值。

## 评测设置与 checkpoint

- 评测时间：2026-10-07T17:35:44+08:00；fold 1 / seed 3407。
- 目录：`/home/mjp/Project/PLM/UnifyImmun/dataset/PH/new`；Stage2 pHLA task checkpoint：`/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_phla_20260928_112359_821461.pt`。
- 选择验证分数：0.867491；Stage `stage2b` 第 20 轮；global step 164318；SHA-256 `178fb8ccf43906582cdb7396fa1f88a2f9720be717d1a5a33c2214125eb14ad4`。
- 验证集 AUROC / AUPR：0.854413 / 0.880570；初始化方式 `scratch`。
- 设备 `cuda`，精度 `bf16`，batch size 32；peptide 长度 8–15 aa，HLA 最大长度 34 aa。
- pHLA 分支使用 checkpoint 中微调后的 peptide ESM-C、HLA ESM-C 和分类模块在线推理。

## 分 split 指标

| Split | Samples | Neg | Pos | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | performance_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 15,425 | 9,271 | 0.725254 | 0.567099 | 0.670918 | 0.244321 | 0.401502 | 0.294035 | 0.897439 | 0.521819 |
| `test_unseen_pair` | 51 | 29 | 22 | 0.341693 | 0.329078 | 0.490196 | -0.195161 | 0.071429 | 0.045455 | 0.827586 | 0.207447 |
| `test_unseen_peptide` | 62,764 | 31,382 | 31,382 | 0.970096 | 0.967838 | 0.916576 | 0.833391 | 0.917561 | 0.928526 | 0.904627 | 0.921093 |

## 结果解读

- `test_unseen_peptide` 表现最好（AUROC 0.970096、AUPR 0.967838），占总测试样本约 71.7%，因此 pooled 指标主要反映该 split。
- `test_unseen_hla` 表现明显较弱（AUROC 0.725254、AUPR 0.567099）；阈值 0.5 下 recall 为 0.294、specificity 为 0.897，呈现较低召回的取舍。
- `test_unseen_pair` 只有 51 条（22 正、29 负），AUROC 0.341693 的估计不稳定，应视作探索性结果，不能据此单独判断模型泛化。
- 三组等权宏平均给这个 51 条 split 与 62,764 条 split 相同权重，所以宏平均 AUROC（0.679014）远低于 pooled AUROC（0.911196）；两者回答的是不同汇总问题。

## 数据校准与分组鲁棒性

| Split | NLL | Brier | ECE | Group count | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 0.991424 | 0.269081 | 0.237372 | 18 | 0.992752 | 0.770774 | 0.370376 | 1.830154 |
| `test_unseen_pair` | 2.056204 | 0.458478 | 0.489188 | 13 | 0.352941 | 0.512088 | 0.500000 | 3.973539 |
| `test_unseen_peptide` | 0.220652 | 0.063699 | 0.012585 | 99 | 0.988274 | 0.931896 | 0.785635 | 0.423489 |

测试划分对应的未见实体分别为 HLA、peptide–HLA pair 和 peptide；各 split 的样本构成不同，跨 split 比较应结合划分定义解读。

## 复现命令

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage2_ph_new.py --config /home/mjp/Project/PLM/UnifyImmun/configs/plm_stage2.yaml \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_phla_20260928_112359_821461.pt --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PH/new \
  --output-json /home/mjp/Project/PLM/UnifyImmun/tests/stage2/stage2_new_results_20261007/stage2_best_phla_PH_new_20261007.json --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage2/stage2_new_results_20261007/stage2_best_phla_PH_new_20261007.md \
  --fold 1 --seed 3407 --batch-size 32 --device cuda
```

- 汇总 JSON：[stage2_best_phla_PH_new_20261007.json](stage2_best_phla_PH_new_20261007.json)
- 原始数据集与 checkpoint 均未修改。
