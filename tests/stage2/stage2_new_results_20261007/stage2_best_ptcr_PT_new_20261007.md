# PLM Unified Stage 2：PT new 数据集 pTCR 测试结果

## 1. 结论摘要

本报告使用指定的 Stage 2 pTCR task checkpoint，对当前工作区 `dataset/PT/new/` 下的 3 个测试 split 完成在线推理。

| 汇总方式 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | `performance_avg` | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三组等权宏平均 | 263,610 | 0.874310 | 0.857196 | 0.822086 | 0.621533 | 0.764700 | 0.787965 | 0.421921 | 0.130665 | 0.040471 |
| 全部样本 pooled | 263,610 | 0.883132 | 0.875077 | 0.813816 | 0.620831 | 0.782713 | 0.795114 | 0.427167 | 0.134032 | 0.030949 |

三个 split 共纳入 **263,610** 条样本，其中负样本 **147,476** 条、正样本 **116,134** 条，正样本比例 **44.0552%**。

`performance_avg` 是项目实现中 AUROC、Accuracy、MCC、F1 和 AUPR 五项指标的算术平均。宏平均对三个 split 等权计算；pooled 行将全部样本合并后重新计算。

## 2. 评测设置

| 项目 | 内容 |
|---|---|
| 评测时间 | 2026-10-07T17:28:46+08:00 |
| Fold / seed | fold 1 / seed 3407 |
| 任务 | pTCR（peptide–TCR 二分类） |
| 测试目录 | [new](../../../dataset/PT/new) |
| 评测脚本 | [evaluate_plm_stage2_pt_new.py](../../../source/evaluate_plm_stage2_pt_new.py) |
| 配置 | [plm_stage2.yaml](../../../configs/plm_stage2.yaml) |
| 输入模式 | Stage 2 微调 PLM 在线编码 |
| 推理设备 | `cuda` |
| 混合精度 | `bf16` autocast |
| 推理 batch size | `32` |
| Peptide 长度约束 | `8–15` aa |
| TCR 最大长度 | `34` aa |
| 分类阈值 | `0.5`，严格比较 `score > 0.5` |

### 2.1 Checkpoint 信息

| 项目 | 内容 |
|---|---|
| Checkpoint | [stage2_best_ptcr_20260928_112359_821461.pt](../../../trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_ptcr_20260928_112359_821461.pt) |
| 类型 / 任务 | `stage2_task_model` / `ptcr` |
| Stage | `stage2b` |
| 选中轮次 | `10` |
| Global step | `89628` |
| 验证集最佳分数 | `0.806979` |
| 验证 AUROC / AUPR | `0.787588` / `0.826370` |
| 初始化方式 | `scratch` |
| SHA-256 | `e97b0964f6a24abc577866d36a1a404ab45847758af532973ed21a6683a0f40c` |

本次只调用 pTCR 分支；pHLA 分支和 HLA 编码器不参与推理。Checkpoint 中的 peptide ESM-C 和 TCR-BERT 权重均为 Stage 2 task checkpoint 内保存的微调后权重。

## 3. 测试数据统计

| Benchmark | 数据文件 | Samples | 负样本 | 正样本 | 正样本比例 | 唯一 peptide | 唯一 TCR |
|---|---|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | [test_unseen_pair.csv](../../../dataset/PT/new/test_unseen_pair.csv) | 8,207 | 5,094 | 3,113 | 37.9310% | 192 | 7,425 |
| `test_unseen_peptide` | [test_unseen_peptide.csv](../../../dataset/PT/new/test_unseen_peptide.csv) | 112,443 | 70,902 | 41,541 | 36.9441% | 299 | 75,021 |
| `test_unseen_tcr` | [test_unseen_tcr.csv](../../../dataset/PT/new/test_unseen_tcr.csv) | 142,960 | 71,480 | 71,480 | 50.0000% | 1,410 | 33,061 |
| **合计** | **3 个 new benchmark** | **263,610** | **147,476** | **116,134** | **44.0552%** | — | — |

## 4. 逐 benchmark 主结果

| Benchmark | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity | `performance_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 0.888525 | 0.871035 | 0.847569 | 0.671850 | 0.781637 | 0.855887 | 0.719242 | 0.925991 | 0.812123 |
| `test_unseen_peptide` | 112,443 | 0.818970 | 0.785983 | 0.781000 | 0.517112 | 0.672553 | 0.751263 | 0.608772 | 0.881907 | 0.715124 |
| `test_unseen_tcr` | 142,960 | 0.915435 | 0.914571 | 0.837689 | 0.675638 | 0.839911 | 0.828569 | 0.851567 | 0.823811 | 0.836649 |

## 5. 校准与分组鲁棒性指标

| Benchmark | NLL | Brier | ECE | Mean confidence | Group count | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 0.393290 | 0.117903 | 0.052080 | 0.887652 | 192 | 0.710126 | 0.899429 | 0.100893 | 3.058181 |
| `test_unseen_peptide` | 0.503033 | 0.157929 | 0.060460 | 0.838755 | 299 | 0.977393 | 0.865864 | 0.250632 | 3.038198 |
| `test_unseen_tcr` | 0.369440 | 0.116163 | 0.008873 | 0.844833 | 1,410 | 0.929064 | 0.901214 | 0.244118 | 3.748223 |

## 6. 结果摘要

- `test_unseen_tcr` 的 `performance_avg` 最高，为 `0.836649`；AUROC=`0.915435`，AUPR=`0.914571`。
- 三个 split 的实体未见设置不同：`test_unseen_pair` 同时约束 peptide/TCR 组合，`test_unseen_peptide` 约束 peptide，`test_unseen_tcr` 约束 TCR；跨 split 比较应结合各自样本构成解读。
- 本次使用固定阈值 `0.5` 计算 Accuracy、MCC、F1、Precision、Recall 和 Specificity，没有利用 test 标签重新选择阈值。

## 7. 原始结果与复现

- 原始汇总 JSON：[stage2_best_ptcr_PT_new_20261007.json](stage2_best_ptcr_PT_new_20261007.json)

```bash
export PYTHONPATH=source
PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
$PYTHON_BIN source/evaluate_plm_stage2_pt_new.py \
  --config /home/mjp/Project/PLM/UnifyImmun/configs/plm_stage2.yaml \
  --checkpoint /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Unified/fold_1/seed_3407/stage2_best_ptcr_20260928_112359_821461.pt \
  --data-dir /home/mjp/Project/PLM/UnifyImmun/dataset/PT/new \
  --output-json /home/mjp/Project/PLM/UnifyImmun/tests/stage2/stage2_new_results_20261007/stage2_best_ptcr_PT_new_20261007.json \
  --output-md /home/mjp/Project/PLM/UnifyImmun/tests/stage2/stage2_new_results_20261007/stage2_best_ptcr_PT_new_20261007.md \
  --batch-size 32 \
  --device cuda
```

本报告中的结果来自当前工作区的 `dataset/PT/new/*.csv`；未修改原始数据集或 checkpoint 文件。
