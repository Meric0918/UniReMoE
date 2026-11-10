# Stage MOE：PH / PT / PHT Benchmark Test 评测

- 评测时间：2026-11-03 16:10 CST
- 检查点：[best.pt](../../../trained_model/PLM_Stage_MOE/seed_42/joint_online_20261027_193829/best.pt)（joint，seed 42，第 12 轮最佳检查点）
- 检查点 SHA-256：`1ea07b480c05658beff7b5b9519b426dfeeb9529fb8efddab9fe8d2eed9e7f58`
- 测试集：PH 3 个、PT 3 个、PHT 4 个；共 450,756 条保留记录。
- 分类阈值固定为 **0.5**：预测概率 `>= 0.5` 判为阳性。阈值没有根据 benchmark test 标签调整。
- PH/PT 仅使用各自 CSV 的 `label` 监督；PHT 仅使用 `PHT` 列。其他任务标签不参与评测。
- 推理使用 CUDA、bf16 混合精度、batch size 256 和与 checkpoint 编码器指纹一致的冻结 PLM 缓存。

## 固定阈值 0.5 结果

| 任务 | Benchmark test 集 | 样本数（正 / 负） | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PH | [test_unseen_hla](../../../dataset/PH/benchmarks/test_unseen_hla.csv) | 47,748（23,874 / 23,874） | 0.8728 | 0.8589 | 0.7874 | 0.5864 | 0.7641 | 0.8582 | 0.6885 | 0.8863 | 0.1865 | 0.1716 |
| PH | [test_unseen_pair](../../../dataset/PH/benchmarks/test_unseen_pair.csv) | 64（32 / 32） | 0.5635 | 0.5103 | 0.5469 | 0.0998 | 0.4528 | 0.5714 | 0.3750 | 0.7188 | 0.4384 | 0.4303 |
| PH | [test_unseen_peptide](../../../dataset/PH/benchmarks/test_unseen_peptide.csv) | 62,764（31,382 / 31,382） | 0.9655 | 0.9619 | 0.9146 | 0.8293 | 0.9144 | 0.9171 | 0.9117 | 0.9176 | 0.0721 | 0.0574 |
| PT | [test_unseen_pair](../../../dataset/PT/benchmarks/test_unseen_pair.csv) | 10,274（5,137 / 5,137） | 0.7624 | 0.8308 | 0.7613 | 0.5523 | 0.7153 | 0.8861 | 0.5998 | 0.9229 | 0.2129 | 0.1759 |
| PT | [test_unseen_peptide](../../../dataset/PT/benchmarks/test_unseen_peptide.csv) | 144,800（72,400 / 72,400） | 0.6617 | 0.7398 | 0.6581 | 0.3688 | 0.5398 | 0.8256 | 0.4010 | 0.9153 | 0.2795 | 0.2214 |
| PT | [test_unseen_tcr](../../../dataset/PT/benchmarks/test_unseen_tcr.csv) | 142,960（71,480 / 71,480） | 0.8929 | 0.9011 | 0.8227 | 0.6469 | 0.8165 | 0.8463 | 0.7887 | 0.8567 | 0.1330 | 0.0578 |
| PHT | [test_unseen_hla](../../../dataset/PHT/benchmarks/test_unseen_hla.csv) | 19,670（9,835 / 9,835） | 0.8642 | 0.7356 | 0.4838 | -0.1209 | 0.0039 | 0.0559 | 0.0020 | 0.9656 | 0.5151 | 0.5148 |
| PHT | [test_unseen_peptide](../../../dataset/PHT/benchmarks/test_unseen_peptide.csv) | 14,118（7,059 / 7,059） | 0.9792 | 0.9810 | 0.8772 | 0.7754 | 0.8613 | 0.9902 | 0.7620 | 0.9925 | 0.1196 | 0.1190 |
| PHT | [test_unseen_tcr](../../../dataset/PHT/benchmarks/test_unseen_tcr.csv) | 7,698（3,849 / 3,849） | 0.9993 | 0.9993 | 0.9748 | 0.9507 | 0.9742 | 0.9986 | 0.9509 | 0.9987 | 0.0239 | 0.0235 |
| PHT | [test_unseen_triple](../../../dataset/PHT/benchmarks/test_unseen_triple.csv) | 660（330 / 330） | 0.8001 | 0.7749 | 0.7439 | 0.5379 | 0.6756 | 0.9215 | 0.5333 | 0.9545 | 0.2529 | 0.2532 |

说明：AUROC/AUPR、Brier 与 ECE 根据概率计算；Accuracy、MCC、F1、Precision、Recall、Specificity 使用固定阈值 0.5。AUPR 使用仓库实现的 `precision_recall_curve + auc`。ECE 为 15 个等频分箱的自适应 ECE，按 0.5 计算类别置信度。

## 数据完整性与复现

- 10 个测试 CSV 共计 450,756 条记录，评估程序没有报告重复记录或无效序列剔除；每个预测 CSV 的行数、原始行号、序列和 native label 均与原始测试文件核对一致。
- 所有 test 集均为正负样本各半；不同 benchmark 划分的样本构成不同，不宜将合计行数上的单一分数解释为自然样本分布性能。
- 冷启动实体交叉训练池检查详见 `metrics.json` 中的 `cold_audit`。PT unseen-TCR 有 5 个不同 TCR、PHT unseen-TCR 有 4 个不同 TCR 出现在任一任务训练池；这两个集合不能整体称为严格跨任务 cold-TCR。
- `metrics.md` 保留评估器直接输出的 checkpoint 验证阈值结果；本报告主表及 `fixed_threshold_0_5.json` 才是固定 0.5 结果。逐条预测 CSV 中的分类列由原评估器按 checkpoint 阈值产生，固定阈值指标已从概率列独立复算。

复现推理命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python -u source/evaluate_plm_stage_MOE_benchmarks.py \
  --config configs/plm_stage_MOE.yaml \
  --checkpoint trained_model/PLM_Stage_MOE/seed_42/joint_online_20261027_193829/best.pt \
  --device cuda:0 --num-workers 2 \
  --output-dir reports/stage_MOE/joint_20261027_193829_benchmarks_20261103
```

## 结果文件

- [固定阈值 0.5 完整指标 JSON](fixed_threshold_0_5.json)
- [评估器原始指标 JSON](metrics.json)
- [逐行与固定阈值复算核对](verification_fixed_threshold.json)
- 逐条预测 CSV：见本目录下 `ph_*_predictions.csv`、`pt_*_predictions.csv`、`pht_*_predictions.csv`。
