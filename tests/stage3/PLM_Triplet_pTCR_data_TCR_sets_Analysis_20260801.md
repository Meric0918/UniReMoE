# PLM Triplet Checkpoint 在 `data/data_TCR/*_set.csv` 上的 pTCR 测试分析

## 1. 分析概况

本报告整理使用 Stage 3 `triplet_best` 模型对 `data/data_TCR` 目录下三个
`*_set.csv` 数据集进行的独立评估。

| 项目 | 内容 |
|---|---|
| 评估日期 | 2026-08-01 |
| Checkpoint | `trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260731_145558_787556.pt` |
| Checkpoint 类型 | `stage3_multitask_model` |
| 使用分支 | pTCR 二元分支 |
| 输入字段 | `peptide,tcr,label` |
| HLA 输入 | 无 |
| 分类阈值 | 0.5 |

由于这些文件不包含 HLA 列，不能使用 HLA-peptide-TCR 三元分支；本次测试使用
同一个 checkpoint 中保留的 pTCR 分支。`triple_set.csv` 虽然名称包含
`triple`，但其文件本身仍是 peptide-TCR 二元数据。

## 2. 模型与推理方式

该 checkpoint 来自 PLM Stage 3，完整模型包含：

- peptide ESM-C 300M 编码器；
- HLA ESM-C 300M 编码器；
- TCR-BERT 编码器；
- pHLA、pTCR 和 triplet 三个预测分支。

本次只调用 pTCR 分支。为保证使用 checkpoint 中已经微调过的 PLM 参数，测试时
采用完整 checkpoint 编码器在线推理，而不是直接使用 Stage 1 的旧 embedding
cache。所有数据集中的唯一 peptide 和唯一 TCR 各编码一次，再复用其表示完成
样本级预测：

- 唯一 peptide：1,994 个；
- 唯一 TCR：200,202 个。

模型配置要求 peptide 长度为 8–15。评估没有在这些测试数据上重新选择阈值，
固定使用 0.5；checkpoint 中记录的 0.46875 是 triplet 验证集阈值，不用于本次
pTCR 测试。

## 3. 数据审计

| 数据集 | 原始有效行数 | 实际评估样本数 | 正样本 | 负样本 | 过滤情况 |
|---|---:|---:|---:|---:|---|
| `covid_set.csv` | 1,089,310 | 1,089,310 | 547,657 | 541,653 | 无过滤 |
| `independent_set.csv` | 28,707 | 28,707 | 10,018 | 18,689 | 无过滤 |
| `triple_set.csv` | 97,043 | 96,799 | 37,813 | 58,986 | 54 条 peptide 过短，190 条过长 |

`triple_set.csv` 共过滤 244 行，占原始数据的约 0.251%。过滤依据与项目现有
pTCR 数据加载逻辑一致。

## 4. 测试结果

所有指标均在固定阈值 0.5 下计算；AUROC 和 AUPR 不依赖该固定阈值。

| 数据集 | 样本数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `covid_set` | 1,089,310 | 0.547126 | 0.556212 | 0.533417 | 0.085926 | 0.345629 | 0.586017 | 0.245091 | 0.824940 | 1.415277 |
| `independent_set` | 28,707 | 0.951500 | 0.915163 | 0.892326 | 0.762469 | 0.844899 | 0.849460 | 0.840387 | 0.920167 | 0.273892 |
| `triple_set` | 96,799 | 0.834734 | 0.809739 | 0.777653 | 0.529339 | 0.648518 | 0.847750 | 0.525110 | 0.939545 | 0.677436 |

## 5. 分数据集解读

### 5.1 `independent_set`

该数据集表现最好：AUROC 为 0.951500，AUPR 为 0.915163，MCC 为 0.762469，
F1 为 0.844899。模型在固定阈值下同时保持了较高的 precision、recall 和
specificity，说明其排序能力和二分类决策均较稳定。

该结果与 checkpoint 中记录的 pTCR validation 结果接近：

| 指标 | Validation | `independent_set` |
|---|---:|---:|
| AUROC | 0.950573 | 0.951500 |
| AUPR | 0.913988 | 0.915163 |
| MCC | 0.757345 | 0.762469 |
| F1 | 0.841159 | 0.844899 |

这说明 `independent_set` 与训练/验证数据分布较接近，模型具有较好的外部泛化
表现。

### 5.2 `triple_set`

该数据集仍有一定预测能力，但明显低于 `independent_set`：AUROC 为 0.834734，
AUPR 为 0.809739，MCC 为 0.529339。

模型 specificity 较高（0.939545）、precision 较高（0.847750），但 recall
只有 0.525110。这表明模型预测为阳性时通常较可靠，但漏掉了较多真实阳性；在
当前固定阈值下，模型整体偏保守。

### 5.3 `covid_set`

该数据集上的表现接近随机：AUROC 为 0.547126，AUPR 为 0.556212，MCC 仅为
0.085926。Recall 只有 0.245091，loss 升高到 1.415277，说明模型在该数据集上
不仅阳性识别能力弱，概率输出也可能存在明显的域偏移或校准问题。

因此，当前 checkpoint 不适合直接用于 `covid_set` 上的高可靠 pTCR 阳性识别，
需要进一步进行 covid 域适配、重新校准或独立训练。

## 6. 总体结论

模型在三个数据集上的表现呈现清晰梯度：

```text
independent_set  >>  triple_set  >>  covid_set
```

- `independent_set`：接近 validation 表现，外部泛化良好；
- `triple_set`：具有中等排序能力，但固定阈值下阳性召回偏低；
- `covid_set`：接近随机，存在明显域偏移，不能直接作为可靠预测工具。

需要注意的是，以上结果来自固定阈值 0.5，未在测试集上调阈值。若业务目标是
提高阳性 recall 或 F1，应在独立验证集上重新选择阈值，而不能使用这些测试集
本身进行调参。

## 7. 输出文件与复现

本次详细 JSON 报告：

[`evaluation_ptcr_data_TCR_sets_20260801_131457_433034.json`](../../trained_model/PLM_Triplet/fold_1/seed_42/evaluation_ptcr_data_TCR_sets_20260801_131457_433034.json)

评估脚本：

[`source/evaluate_plm_triplet_tcr_sets.py`](../../source/evaluate_plm_triplet_tcr_sets.py)

默认复现命令：

```bash
CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun/cuda \
LD_LIBRARY_PATH=/home/mjp/miniconda3/envs/unifyimmun/cuda/lib64:/home/mjp/miniconda3/envs/unifyimmun/lib \
NVTE_TORCH_COMPILE=0 PYTHONPATH=source \
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
source/evaluate_plm_triplet_tcr_sets.py \
  --checkpoint trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_20260731_145558_787556.pt \
  --data-dir data \
  --sets covid,independent,triple \
  --threshold 0.5
```

