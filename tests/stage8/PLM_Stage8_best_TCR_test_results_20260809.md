# PLM Stage 8 `best` 模型 pTCR 测试结果

## 1. 测试概况

| 项目 | 内容 |
|---|---|
| 评测日期 | 2026-08-09 |
| Checkpoint | `trained_model/PLM_Stage8/fold_4/seed_2026/baseline/stage8_best_20260809_011230_337330.pt` |
| Checkpoint 类型 | `stage8_binding_model` |
| Stage 8 条件 | `baseline` |
| 最优验证轮次 | epoch 9 |
| 最优验证 joint score | 0.993786 |
| 测试分支 | pTCR 二元分类分支 |
| 测试数据目录 | `data/data_TCR/test_data_set` |
| 分类阈值 | 0.5 |
| 推理设备 | NVIDIA GeForce RTX 4090 |

测试文件只包含 `peptide,tcr,label`，没有 HLA 列，因此本次只调用 checkpoint 中的
pTCR 分支；不能使用需要 HLA 输入的 pHLA 或 PHT 三元分支。

## 2. 数据审计

| 数据集 | 原始行数 | 实际测试样本数 | 正样本 | 负样本 | peptide 长度过滤 |
|---|---:|---:|---:|---:|---|
| `covid_set.csv` | 1,089,310 | 1,089,310 | 547,657 | 541,653 | 无 |
| `independent_set.csv` | 28,707 | 28,707 | 10,018 | 18,689 | 无 |
| `triple_set_peptide_8_15aa.csv` | 96,799 | 96,799 | 37,813 | 58,986 | 无 |

三个文件合计使用 1,214,816 条样本。三个数据集的并集包含 1,994 个唯一 peptide
和 200,202 个唯一 TCR；所有保留的 TCR 长度均不超过配置上限 34。

## 3. 测试指标

所有指标均在固定阈值 0.5 下计算；AUROC 和 AUPR 不依赖该阈值。

| 数据集 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `covid_set` | 1,089,310 | 0.658411 | 0.623923 | 0.519750 | 0.079256 | 0.182174 | 0.633216 | 0.106391 | 0.937691 | 1.514839 |
| `independent_set` | 28,707 | 0.791757 | 0.744588 | 0.782353 | 0.507353 | 0.587646 | 0.867160 | 0.444400 | 0.963508 | 0.810759 |
| `triple_set_peptide_8_15aa` | 96,799 | 0.949274 | 0.927132 | 0.896456 | 0.781138 | 0.863642 | 0.889303 | 0.839420 | 0.933018 | 0.286915 |

### 3.1 概率输出概览

| 数据集 | 平均预测概率 | 正样本平均概率 | 负样本平均概率 |
|---|---:|---:|---:|
| `covid_set` | 0.151912 | 0.194444 | 0.108908 |
| `independent_set` | 0.199728 | 0.447998 | 0.066647 |
| `triple_set_peptide_8_15aa` | 0.386345 | 0.806999 | 0.116685 |

## 4. 推理方法与模型来源

1. 加载 Stage 8 `baseline` 的 `stage8_best_20260809_011230_337330.pt`，只使用其 pTCR
   binding state。
2. 按 checkpoint 中记录的 Stage 3 来源加载微调后的 peptide ESM-C 和 TCR-BERT 编码器：
   `trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt`。
3. 对三个测试集并集中的每个唯一 peptide/TCR 编码一次，再复用其 residue-level 表示进行
   样本级 pTCR 推理；没有把训练/验证缓存中的表示错误地用于未覆盖的测试序列。
4. 使用配置中的 peptide 长度范围 `[8, 15]`；本次三个文件没有行被过滤。
5. 阈值固定为 0.5，没有在测试集上重新选择阈值。checkpoint 中的
   `selected_pht_threshold=0.484375` 属于独立的 PHT 头，本次 pTCR 测试不使用。

## 5. 结果解读

- `triple_set_peptide_8_15aa` 表现最好，AUROC 为 0.949274、AUPR 为 0.927132，固定阈值下
  MCC 为 0.781138，说明排序能力和二分类判别均较强。
- `independent_set` 仍有较好的排序能力（AUROC 0.791757），但固定阈值下 Recall 为
  0.444400，阳性漏检较多，整体弱于 triple 测试集。
- `covid_set` 的 AUROC 为 0.658411，具有有限的高于随机排序能力；但固定阈值下 Recall
  仅为 0.106391、Specificity 为 0.937691，且 Loss 为 1.514839，提示该数据集上的概率
  分布和决策阈值存在明显不匹配。若需要提高 covid 数据上的阳性召回，应在独立验证集上
  重新校准阈值或进行域适配，不能直接使用测试集调参。

## 6. 相关文件

- Checkpoint：`trained_model/PLM_Stage8/fold_4/seed_2026/baseline/stage8_best_20260809_011230_337330.pt`
- Stage 3 来源：`trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt`
- 测试数据：`data/data_TCR/test_data_set/covid_set.csv`、`independent_set.csv`、`triple_set_peptide_8_15aa.csv`
- Stage 3 来源 SHA-256：`c21998c15d0a98d97f224a3ccd6b9824bc9720344d5e0d98eb0914d6f0446262`
