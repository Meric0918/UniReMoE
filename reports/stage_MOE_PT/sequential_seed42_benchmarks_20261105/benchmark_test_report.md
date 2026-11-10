# Stage MOE PT sequential：PH / PT / PHT Benchmark Test 结果

- 评测时间：2026-11-05 20:29:11 CST
- 检查点：[best.pt](../../../trained_model/PLM_Stage_MOE_PT/sequential_seed42_20261103_202153/best.pt)（sequential，seed 42，第 12 轮最佳检查点）
- 检查点 SHA-256：`ef7c62a0d91c2447535ff4ef8ec93239624e304e00fdafd5b09dd4f4f0e7fbae`
- 测试范围：PH 3 个、PT 3 个、PHT 4 个 `test_*.csv` benchmark，共 450,756 条有效记录。
- 推理：CUDA:0、bf16、eval batch size 256；温度 1.0。
- 标签和输出头：PH/PT 使用各自任务的 `label` 与原生任务 head；PHT 使用 `PHT` 标签和 PHT head。其他标签不参与该数据集的监督评估。
- 分类阈值直接读取 checkpoint 中由验证集确定的任务阈值；没有使用 benchmark test 标签选阈值。AUROC/AUPR 按预测概率计算，分类指标按表中阈值计算。

## 任务级汇总

以下为每个任务组内 benchmark 指标的简单算术平均（unweighted macro mean），不是合并所有样本后重新计算的 pooled metric。

| 任务 | Benchmark 数 | 样本数合计 | Macro AUROC | Macro AUPR | Macro MCC | Macro F1 | 验证阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PH | 3 | 110,576 | 0.802316 | 0.800436 | 0.463819 | 0.685906 | 0.163445 |
| PT | 3 | 298,034 | 0.818729 | 0.849501 | 0.547185 | 0.718004 | 0.579876 |
| PHT | 4 | 42,146 | 0.941874 | 0.912058 | 0.636867 | 0.735826 | 0.009126 |

## 各 benchmark 结果

| 任务 | 测试集 | 样本数（正 / 负） | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | BCE | 阈值 | Cold audit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PH | [`test_unseen_hla`](../../../dataset/PH/benchmarks/test_unseen_hla.csv) | 47,748（23,874 / 23,874） | 0.831504 | 0.813522 | 0.753686 | 0.509550 | 0.741761 | 0.779501 | 0.707506 | 0.799866 | 0.785777 | 0.163445 | hla: 0/21 seen; strict |
| PH | [`test_unseen_pair`](../../../dataset/PH/benchmarks/test_unseen_pair.csv) | 64（32 / 32） | 0.630371 | 0.643402 | 0.562500 | 0.139010 | 0.440000 | 0.611111 | 0.343750 | 0.781250 | 2.224256 | 0.163445 | 组合划分（无单一声称冷实体） |
| PH | [`test_unseen_peptide`](../../../dataset/PH/benchmarks/test_unseen_peptide.csv) | 62,764（31,382 / 31,382） | 0.945072 | 0.944384 | 0.869304 | 0.742896 | 0.875958 | 0.833520 | 0.922949 | 0.815659 | 0.331882 | 0.163445 | peptide: 0/41668 seen; strict |
| PT | [`test_unseen_pair`](../../../dataset/PT/benchmarks/test_unseen_pair.csv) | 10,274（5,137 / 5,137） | 0.781982 | 0.838997 | 0.774090 | 0.568558 | 0.739535 | 0.873079 | 0.641425 | 0.906755 | 0.701961 | 0.579876 | 组合划分（无单一声称冷实体） |
| PT | [`test_unseen_peptide`](../../../dataset/PT/benchmarks/test_unseen_peptide.csv) | 144,800（72,400 / 72,400） | 0.762624 | 0.797722 | 0.686782 | 0.417727 | 0.596495 | 0.838075 | 0.463025 | 0.910539 | 0.680237 | 0.579876 | peptide: 0/302 seen; strict |
| PT | [`test_unseen_tcr`](../../../dataset/PT/benchmarks/test_unseen_tcr.csv) | 142,960（71,480 / 71,480） | 0.911581 | 0.911785 | 0.826273 | 0.655271 | 0.817982 | 0.858977 | 0.780722 | 0.871824 | 0.380302 | 0.579876 | tcr: 5/33061 seen; not strict |
| PHT | [`test_unseen_hla`](../../../dataset/PHT/benchmarks/test_unseen_hla.csv) | 19,670（9,835 / 9,835） | 0.864497 | 0.748353 | 0.564820 | 0.174990 | 0.344764 | 0.697430 | 0.228978 | 0.900661 | 3.026247 | 0.009126 | hla: 0/22 seen; strict |
| PHT | [`test_unseen_peptide`](../../../dataset/PHT/benchmarks/test_unseen_peptide.csv) | 14,118（7,059 / 7,059） | 0.986442 | 0.987204 | 0.935614 | 0.872767 | 0.933645 | 0.963102 | 0.905936 | 0.965293 | 0.528728 | 0.009126 | peptide: 0/268 seen; strict |
| PHT | [`test_unseen_tcr`](../../../dataset/PHT/benchmarks/test_unseen_tcr.csv) | 7,698（3,849 / 3,849） | 0.999492 | 0.999517 | 0.989348 | 0.978799 | 0.989425 | 0.982330 | 0.996622 | 0.982073 | 0.057921 | 0.009126 | tcr: 4/6808 seen; not strict |
| PHT | [`test_unseen_triple`](../../../dataset/PHT/benchmarks/test_unseen_triple.csv) | 660（330 / 330） | 0.917066 | 0.913160 | 0.739394 | 0.520911 | 0.675472 | 0.895000 | 0.542424 | 0.936364 | 1.892425 | 0.009126 | 组合划分（无单一声称冷实体） |

## 数据完整性与结果解读

- 10 个 CSV 共 450,756 条样本，正负样本各 225,378 条。全部输入行均保留；重复行和无效序列均为 0。
- 已逐条核验 10 个预测 CSV：样本行数、原始 CSV 行号、native label、概率及分类阈值均一致；从导出的概率复算的 AUROC、AUPR 和阈值分类指标与 `metrics.json` 一致。
- AUROC 最高为 `pht/test_unseen_tcr`（0.999492）；最低为 `ph/test_unseen_pair`（0.630371，N=64），小样本 benchmark 的估计不确定性较大。
- Cold audit 将声称的冷实体与 PH、PT、PHT 全部训练池比较：PT `test_unseen_tcr` 有 5/33,061 个 TCR 在训练池出现，PHT `test_unseen_tcr` 有 4/6,808 个 TCR 出现；这两组不满足严格跨任务 cold-TCR 条件。其他声称冷实体的数据见结果表及原始 JSON；pair/triple 组合划分没有单一声称冷实体。
- Cache 中覆盖本次全部 benchmark 所需序列；评估器同时核对了 cache encoder fingerprint 与 checkpoint encoder fingerprint。每个数据文件 SHA-256、完整 cold audit、路由统计和融合统计保存在原始 JSON 中。

## 结果文件与复现

- 评估器原始指标：[metrics.json](metrics.json)；评估器简表：[metrics.md](metrics.md)。
- 逐样本预测：
  - [PH / test_unseen_hla](ph_test_unseen_hla_predictions.csv)
  - [PH / test_unseen_pair](ph_test_unseen_pair_predictions.csv)
  - [PH / test_unseen_peptide](ph_test_unseen_peptide_predictions.csv)
  - [PT / test_unseen_pair](pt_test_unseen_pair_predictions.csv)
  - [PT / test_unseen_peptide](pt_test_unseen_peptide_predictions.csv)
  - [PT / test_unseen_tcr](pt_test_unseen_tcr_predictions.csv)
  - [PHT / test_unseen_hla](pht_test_unseen_hla_predictions.csv)
  - [PHT / test_unseen_peptide](pht_test_unseen_peptide_predictions.csv)
  - [PHT / test_unseen_tcr](pht_test_unseen_tcr_predictions.csv)
  - [PHT / test_unseen_triple](pht_test_unseen_triple_predictions.csv)

复现命令（在项目根目录运行）：

```bash
./run_stage_MOE_PT.sh benchmark \
  --checkpoint /home/mjp/UniPHT/trained_model/PLM_Stage_MOE_PT/sequential_seed42_20261103_202153/best.pt \
  --device cuda:0 \
  --output-dir reports/stage_MOE_PT/sequential_seed42_benchmarks_20261105
```

使用配置：[plm_stage_MOE_PT.yaml](../../../configs/plm_stage_MOE_PT.yaml)。
