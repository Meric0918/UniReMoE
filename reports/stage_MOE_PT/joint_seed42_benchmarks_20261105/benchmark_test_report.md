# Stage MOE PT joint：PH / PT / PHT Benchmark Test 结果

- 评测时间：2026-11-05 20:38:10 CST
- 检查点：[best.pt](../../../trained_model/PLM_Stage_MOE_PT/joint_seed42_20261103_192126/best.pt)（joint，seed 42，第 9 轮最佳检查点）
- 检查点 SHA-256：`d63e0c1b4ad27c50163cdbb00640804d67ca73ba8cbbef6625e80246fe0a0fe4`
- 测试范围：PH 3 个、PT 3 个、PHT 4 个 `test_*.csv` benchmark，共 450,756 条有效记录。
- 推理：CUDA:0、bf16、eval batch size 256；温度 1.0。
- 标签和输出头：PH/PT 使用各自任务的 `label` 与原生任务 head；PHT 使用 `PHT` 标签和 PHT head。其他标签不参与该数据集的监督评估。
- 分类阈值直接读取 checkpoint 中由验证集确定的任务阈值；没有使用 benchmark test 标签选阈值。AUROC/AUPR 按预测概率计算，分类指标按表中阈值计算。

## 任务级汇总

以下为每个任务组内 benchmark 指标的简单算术平均（unweighted macro mean），不是合并所有样本后重新计算的 pooled metric。

| 任务 | Benchmark 数 | 样本数合计 | Macro AUROC | Macro AUPR | Macro MCC | Macro F1 | 验证阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PH | 3 | 110,576 | 0.751007 | 0.732819 | 0.387107 | 0.611426 | 0.0940101817 |
| PT | 3 | 298,034 | 0.833626 | 0.854787 | 0.535987 | 0.712801 | 0.497878999 |
| PHT | 4 | 42,146 | 0.840377 | 0.837579 | 0.506090 | 0.618354 | 1.85539108e-07 |

## 各 benchmark 结果

| 任务 | 测试集 | 样本数（正 / 负） | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | BCE | 阈值 | Cold audit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PH | [`test_unseen_hla`](../../../dataset/PH/benchmarks/test_unseen_hla.csv) | 47,748（23,874 / 23,874） | 0.785092 | 0.759485 | 0.689285 | 0.397615 | 0.633208 | 0.772656 | 0.536399 | 0.842171 | 1.656145 | 0.0940101817 | hla: 0/21 seen; strict |
| PH | [`test_unseen_pair`](../../../dataset/PH/benchmarks/test_unseen_pair.csv) | 64（32 / 32） | 0.500000 | 0.472669 | 0.484375 | -0.036886 | 0.297872 | 0.466667 | 0.218750 | 0.750000 | 3.154118 | 0.0940101817 | 组合划分（无单一声称冷实体） |
| PH | [`test_unseen_peptide`](../../../dataset/PH/benchmarks/test_unseen_peptide.csv) | 62,764（31,382 / 31,382） | 0.967929 | 0.966302 | 0.897011 | 0.800591 | 0.903199 | 0.852009 | 0.960933 | 0.833089 | 0.267253 | 0.0940101817 | peptide: 0/41668 seen; strict |
| PT | [`test_unseen_pair`](../../../dataset/PT/benchmarks/test_unseen_pair.csv) | 10,274（5,137 / 5,137） | 0.812605 | 0.850260 | 0.759587 | 0.541052 | 0.720208 | 0.861284 | 0.618844 | 0.900331 | 0.690587 | 0.497878999 | 组合划分（无单一声称冷实体） |
| PT | [`test_unseen_peptide`](../../../dataset/PT/benchmarks/test_unseen_peptide.csv) | 144,800（72,400 / 72,400） | 0.774155 | 0.801193 | 0.674786 | 0.391711 | 0.580043 | 0.818488 | 0.449185 | 0.900387 | 0.692736 | 0.497878999 | peptide: 0/302 seen; strict |
| PT | [`test_unseen_tcr`](../../../dataset/PT/benchmarks/test_unseen_tcr.csv) | 142,960（71,480 / 71,480） | 0.914119 | 0.912908 | 0.837591 | 0.675198 | 0.838152 | 0.835265 | 0.841060 | 0.834121 | 0.374148 | 0.497878999 | tcr: 5/33061 seen; not strict |
| PHT | [`test_unseen_hla`](../../../dataset/PHT/benchmarks/test_unseen_hla.csv) | 19,670（9,835 / 9,835） | 0.795538 | 0.641364 | 0.452771 | -0.215719 | 0.005911 | 0.032226 | 0.003254 | 0.902288 | 15.912522 | 1.85539108e-07 | hla: 0/22 seen; strict |
| PHT | [`test_unseen_peptide`](../../../dataset/PHT/benchmarks/test_unseen_peptide.csv) | 14,118（7,059 / 7,059） | 0.990289 | 0.989761 | 0.941989 | 0.884065 | 0.941579 | 0.948276 | 0.934977 | 0.949001 | 1.355169 | 1.85539108e-07 | peptide: 0/268 seen; strict |
| PHT | [`test_unseen_tcr`](../../../dataset/PHT/benchmarks/test_unseen_tcr.csv) | 7,698（3,849 / 3,849） | 0.999435 | 0.999456 | 0.984412 | 0.969125 | 0.984604 | 0.972624 | 0.996882 | 0.971941 | 0.159208 | 1.85539108e-07 | tcr: 4/6808 seen; not strict |
| PHT | [`test_unseen_triple`](../../../dataset/PHT/benchmarks/test_unseen_triple.csv) | 660（330 / 330） | 0.576244 | 0.719735 | 0.663636 | 0.386891 | 0.541322 | 0.850649 | 0.396970 | 0.930303 | 13.862628 | 1.85539108e-07 | 组合划分（无单一声称冷实体） |

## 数据完整性与结果解读

- 10 个 CSV 共 450,756 条样本，正负样本各 225,378 条。全部输入行均保留；重复行和无效序列均为 0。
- 已逐条核验 10 个预测 CSV：样本行数、原始 CSV 行号、native label、概率及分类阈值均一致；从导出的概率复算的 AUROC、AUPR 和阈值分类指标与 `metrics.json` 一致。
- AUROC 最高为 `pht/test_unseen_tcr`（0.999435）；最低为 `ph/test_unseen_pair`（0.500000，N=64），小样本 benchmark 的估计不确定性较大。
- Cold audit 将声称的冷实体与 PH、PT、PHT 全部训练池比较：PT `test_unseen_tcr` 有 5/33,061 个 TCR 在训练池出现，PHT `test_unseen_tcr` 有 4/6,808 个 TCR 出现；这两组不满足严格跨任务 cold-TCR 条件。其他声称冷实体的数据见结果表及原始 JSON；pair/triple 组合划分没有单一声称冷实体。
- Cache 覆盖本次 benchmark 所需序列；评估器核对了 cache encoder fingerprint 与 checkpoint encoder fingerprint。每个数据文件 SHA-256、完整 cold audit、路由统计和融合统计保存在原始 JSON 中。

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
  --checkpoint /home/mjp/UniPHT/trained_model/PLM_Stage_MOE_PT/joint_seed42_20261103_192126/best.pt \
  --device cuda:0 \
  --output-dir reports/stage_MOE_PT/joint_seed42_benchmarks_20261105
```

使用配置：[plm_stage_MOE_PT.yaml](../../../configs/plm_stage_MOE_PT.yaml)。
