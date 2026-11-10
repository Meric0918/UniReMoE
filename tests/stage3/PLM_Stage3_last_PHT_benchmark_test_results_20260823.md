# Stage 3 `stage3_last` PHT benchmark test 评测总结

## 1. 结论摘要

本次使用指定的 Stage 3 完整 checkpoint，对 `dataset/PHT/benchmarks/test_*.csv` 四个自然分布测试集进行了完整评测。有效样本共 **2,970,530** 条；`test_unseen_tcr.csv` 中有 1 条 TCR 长度为 36，超过模型配置上限 34，因此排除后进入评测的样本为 859,704 条。其余数据均通过长度、canonical 残基和标签校验。

主要结果如下：

- PHT（三元）头在 unseen-TCR 上最好：AUROC=`0.999406`、AUPR=`0.965711`、MCC=`0.887391`。
- PHT 头在 unseen-peptide 上也较强：AUROC=`0.978468`、AUPR=`0.810486`、MCC=`0.816777`。
- unseen-HLA 和 unseen-triple 的 PHT AUPR 分别为 `0.003898` 和 `0.004062`；这两个测试集的 PHT 正样本比例只有 `0.1122%` 和 `0.0689%`，应优先结合 AUPR、混淆矩阵和阈值策略解读，不能只看 Accuracy。
- checkpoint 的 validation 选择阈值为 `0.99609375`。在该阈值下模型更保守；固定阈值 `0.5` 会提高召回，但通常降低 precision/specificity。两种口径均已写入原始 JSON。

## 2. 评测对象与设置

| 项目 | 内容 |
|---|---|
| Checkpoint | `trained_model/PLM_Triplet_PHT/fold_1/seed_42/stage3_last_20260821_182020_363567.pt` |
| Checkpoint 类型 | `stage3_training`，format version `1` |
| Checkpoint SHA-256 | `173222f63267a0cd306d84a3e21e066cd2c552804eb0d3ef49118aa1f9e2940c` |
| 配置 | `configs/plm_stage3_pht.yaml` |
| 设备 | NVIDIA GeForce RTX 4090，CUDA 12.4，PyTorch 2.6.0 |
| 精度 | BF16 autocast |
| 评测 head | pHLA、pTCR、PHT 三个 head |
| 阈值 | checkpoint 阈值 `0.99609375`；固定阈值 `0.5` |
| 测试集阈值选择 | 未在 test 上重新选择 |
| checkpoint 记录 | Stage 3B round `6`，global step `34,664` |

模型选择指标为 `macro_mean_pht_auroc_aupr`，checkpoint 记录的 best selection score 为 `0.656414`。测试集没有参与模型选择、阈值选择或调参。

本次运行设置 `CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun`，仅用于使 `transformer_engine` 找到环境内的 NVRTC 动态库，不改变模型、权重或数据。

## 3. 数据审计

主结果使用 `dataset/PHT/benchmarks/test_*.csv`，没有使用 `dataset/PHT/balanced_test`。类别编码为 `PH/PT/PHT`。

| Benchmark | 原始行数 | 纳入评测 | 过滤 | `111` | `100` | `010` | `000` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_hla` | 1,117,711 | 1,117,711 | 0 | 1,254 | 70,281 | 456,864 | 589,312 |
| `test_unseen_peptide` | 980,043 | 980,043 | 0 | 10,817 | 47,133 | 552,043 | 370,050 |
| `test_unseen_tcr` | 859,705 | 859,704 | 1 | 5,290 | 55,251 | 351,631 | 447,532 |
| `test_unseen_triple` | 13,072 | 13,072 | 0 | 9 | 674 | 6,063 | 6,326 |
| **合计** | **2,970,531** | **2,970,530** | **1** | **17,370** | **173,339** | **1,366,601** | **1,413,220** |

合计 PHT 正样本（`111`）仅占 `0.5847%`。过滤的 1 行位于 `test_unseen_tcr.csv` 第 648 行，原因是 TCR 长度为 36；没有发现同标签重复行或标签冲突。四个 benchmark 合计包含 1,501 个唯一 peptide、136 个唯一 HLA 和 228,746 个唯一 TCR。

## 4. 主结果：checkpoint validation 阈值 `0.99609375`

AUROC/AUPR 与阈值无关；MCC、F1、Recall 和 Specificity 使用表中阈值。

| Benchmark | Head | AUROC | AUPR | MCC | F1 | Recall | Specificity | Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unseen-HLA | pHLA | 0.708349 | 0.117057 | 0.045278 | 0.076311 | 0.053722 | 0.975777 | 0.703779 |
| unseen-HLA | pTCR | 0.950771 | 0.937053 | 0.361916 | 0.338750 | 0.203963 | 0.999830 | 0.277328 |
| unseen-HLA | PHT | 0.870364 | 0.003898 | -0.001629 | 0.001431 | 0.014354 | 0.978606 | 0.348973 |
| unseen-peptide | pHLA | 0.979167 | 0.841736 | 0.681148 | 0.664355 | 0.515755 | 0.997681 | 0.109985 |
| unseen-peptide | pTCR | 0.817406 | 0.871713 | 0.410364 | 0.487690 | 0.322496 | 0.999935 | 0.778795 |
| unseen-peptide | PHT | 0.978468 | 0.810486 | 0.816777 | 0.802716 | 0.672183 | 0.999971 | 0.023666 |
| unseen-TCR | pHLA | 0.995869 | 0.956644 | 0.628366 | 0.587422 | 0.418014 | 0.999606 | 0.044653 |
| unseen-TCR | pTCR | 0.942048 | 0.928343 | 0.342103 | 0.313062 | 0.185657 | 0.999706 | 0.300628 |
| unseen-TCR | PHT | 0.999406 | 0.965711 | 0.887391 | 0.883990 | 0.807372 | 0.999881 | 0.007436 |
| unseen-triple | pHLA | 0.611239 | 0.112834 | 0.187705 | 0.229630 | 0.226940 | 0.958673 | 0.813129 |
| unseen-triple | pTCR | 0.809508 | 0.817284 | 0.358630 | 0.355363 | 0.216074 | 1.000000 | 0.667280 |
| unseen-triple | PHT | 0.910915 | 0.004062 | -0.000230 | 0.000000 | 0.000000 | 0.999923 | 0.038560 |

## 5. 合并测试集结果：阈值对比

四个 benchmark 合并后，样本数为 2,970,530。下表比较 checkpoint 阈值与固定 0.5 阈值；AUROC、AUPR 和 Loss 在两种阈值下相同。

| Head | 阈值 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pHLA | 0.996094 | 0.907302 | 0.541595 | 0.946197 | 0.435191 | 0.425531 | 0.676499 | 0.310384 | 0.989817 | 0.317596 |
| pHLA | 0.500000 | 0.907302 | 0.541595 | 0.913202 | 0.451888 | 0.481711 | 0.390590 | 0.628282 | 0.932749 | 0.317596 |
| pTCR | 0.996094 | 0.893273 | 0.898042 | 0.649314 | 0.386127 | 0.396731 | 0.999163 | 0.247502 | 0.999819 | 0.451232 |
| pTCR | 0.500000 | 0.893273 | 0.898042 | 0.814231 | 0.632620 | 0.779514 | 0.871881 | 0.704842 | 0.909652 | 0.451232 |
| PHT | 0.996094 | 0.972346 | 0.218156 | 0.989959 | 0.460768 | 0.436671 | 0.324938 | 0.665515 | 0.991868 | 0.141436 |
| PHT | 0.500000 | 0.972346 | 0.218156 | 0.978795 | 0.368169 | 0.296236 | 0.183786 | 0.763212 | 0.980063 | 0.141436 |

在自然分布合并集上，0.996094 阈值使 pTCR precision/specificity 很高但 recall 仅 `0.247502`；固定 0.5 将 pTCR recall 提升至 `0.704842`，MCC 从 `0.386127` 提升至 `0.632620`。PHT 的固定阈值 recall 从 `0.665515` 提升至 `0.763212`，但 precision 从 `0.324938` 降至 `0.183786`。

## 6. PHT 四类机制的补充观察

合并四个 benchmark 后，PHT 概率按类别的均值和预测为阳性的比例如下。`111` 是 PHT 正类；其余三类在 PHT 头上均为负类。

| 类别 | 样本数 | 平均 PHT 概率 | 阳性率（0.996094） | 阳性率（0.5） |
|---|---:|---:|---:|---:|
| `111` | 17,370 | 0.768953 | 0.665515 | 0.763212 |
| `100` | 173,339 | 0.028072 | 0.000300 | 0.020128 |
| `010` | 1,366,601 | 0.034649 | 0.017510 | 0.033206 |
| `000` | 1,413,220 | 0.011619 | 0.000025 | 0.007082 |

模型对 `111` 与负类有明显概率分离，但 `010` 的误报率高于 `000`，说明仅凭 PHT 头的最终概率仍会受到单实体标签机制影响。

## 7. 评测方法与可复核产物

- 使用项目现有 `plm_unified` 的 Stage 3 progressive model、encoder 和 `compute_binary_metrics` 实现；没有重新训练或修改 checkpoint。
- 对四个 benchmark 的唯一 peptide/HLA/TCR 分别编码一次，再调用 Stage 3 的 pHLA、pTCR、PHT 三头进行行级推理。
- 推理 batch size 为 2,048，唯一序列 encoder batch size 为 512；完整运行约 181 秒。
- 原始 JSON 汇总：[`evaluation_stage3_last_pht_benchmark_test_20260823.json`](../../trained_model/PLM_Triplet_PHT/fold_1/seed_42/evaluation_stage3_last_pht_benchmark_test_20260823.json)
- 指定 checkpoint：[`stage3_last_20260821_182020_363567.pt`](../../trained_model/PLM_Triplet_PHT/fold_1/seed_42/stage3_last_20260821_182020_363567.pt)
- 数据口径说明：[`dataset/run.md`](../../dataset/run.md)

本次未输出逐行 predictions CSV；原始 JSON 已包含逐 benchmark、逐 head、两种阈值、混淆矩阵、类别分布和过滤审计结果。
