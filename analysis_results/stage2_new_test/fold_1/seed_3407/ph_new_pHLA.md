# PLM Unified Stage 2：PH new 数据集 pHLA 测试结果

## 汇总

| 汇总方式 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | performance_avg | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三组等权宏平均 | 87,511 | 0.678540 | 0.621505 | 0.692821 | 0.294791 | 0.463677 | 0.550267 | 1.089451 | 0.263128 | 0.246342 |
| 全部样本 pooled | 87,511 | 0.911264 | 0.912172 | 0.847345 | 0.694409 | 0.826729 | 0.838384 | 0.438848 | 0.121771 | 0.075879 |

合计 **87,511** 条：负样本 46,836，正样本 40,675（46.48%）。
宏平均对三个 split 等权；pooled 将样本合并后重新计算。固定使用配置阈值 0.5，未根据测试标签调整阈值。

## 评测设置与 checkpoint

- 评测时间：2026-09-30T17:51:20+08:00；fold 1 / seed 3407。
- 目录：`/home/mjp/Project/PLM/UnifyImmun/dataset/PH/new`；Stage2 pHLA task checkpoint：`/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_phla_20260928_112359_821461.pt`。
- 选择验证分数：0.867491；Stage `stage2b` 第 20 轮；global step 164318；SHA-256 `178fb8ccf43906582cdb7396fa1f88a2f9720be717d1a5a33c2214125eb14ad4`。
- 验证集 AUROC / AUPR：0.854413 / 0.880570；初始化方式 `scratch`。
- 设备 `cuda`，精度 `bf16`，batch size 128；peptide 长度 8–15 aa，HLA 最大长度 34 aa。
- pHLA 分支使用 checkpoint 中微调后的 peptide ESM-C、HLA ESM-C 和分类模块在线推理。

## 分 split 指标

| Split | Samples | Neg | Pos | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | performance_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 24,696 | 15,425 | 9,271 | 0.725328 | 0.567908 | 0.671404 | 0.245574 | 0.401769 | 0.293927 | 0.898282 | 0.522397 |
| `test_unseen_pair` | 51 | 29 | 22 | 0.340125 | 0.328717 | 0.490196 | -0.195161 | 0.071429 | 0.045455 | 0.827586 | 0.207061 |
| `test_unseen_peptide` | 62,764 | 31,382 | 31,382 | 0.970166 | 0.967891 | 0.916863 | 0.833959 | 0.917835 | 0.928685 | 0.905041 | 0.921343 |

## 数据校准与分组鲁棒性

| Split | NLL | Brier | ECE | Group count | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 0.990735 | 0.268936 | 0.236780 | 18 | 0.992752 | 0.772265 | 0.370352 | 1.828988 |
| `test_unseen_pair` | 2.057238 | 0.456854 | 0.489910 | 13 | 0.352941 | 0.512088 | 0.500000 | 3.937691 |
| `test_unseen_peptide` | 0.220380 | 0.063593 | 0.012337 | 99 | 0.988274 | 0.932482 | 0.786298 | 0.423611 |

测试划分对应的未见实体分别为 HLA、peptide–HLA pair 和 peptide；各 split 的样本构成不同，跨 split 比较应结合划分定义解读。

## 复现命令

```bash
PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage2_ph_new.py --config /home/mjp/Project/PLM/UnifyImmun/configs/plm_stage2.yaml \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_phla_20260928_112359_821461.pt --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PH/new \
  --output-json /home/mjp/Project/PLM/UnifyImmun/analysis_results/stage2_new_test/fold_1/seed_3407/ph_new_pHLA.json --output-md /home/mjp/Project/PLM/UnifyImmun/analysis_results/stage2_new_test/fold_1/seed_3407/ph_new_pHLA.md \
  --fold 1 --seed 3407 --batch-size 128 --device cuda
```

- 汇总 JSON：[ph_new_pHLA.json](ph_new_pHLA.json)
- 原始数据集与 checkpoint 均未修改。
