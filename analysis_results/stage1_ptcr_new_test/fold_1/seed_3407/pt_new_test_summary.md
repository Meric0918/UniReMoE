# Stage 1 pTCR：PT/new 测试集评估总结

## 1. 结果摘要

使用 Stage 1 fold 1 / seed 3407 的 pTCR checkpoint，在 `dataset/PT/new/` 的三个 test split 上完成在线推理。总计 **263,610** 条样本（负样本 147,476，正样本 116,134，正样本比例 44.06%）。三个文件中的样本均通过 Stage 1 的 peptide 长度约束（8–15 aa），且全部纳入评估。

| 汇总（各 split 等权宏平均） | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | performance_avg | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三个 test split | 0.861736 | 0.837599 | 0.804637 | 0.584893 | 0.747394 | 0.768997 | 0.729537 | 0.846651 | 0.767252 | 0.451098 | 0.141924 | 0.051884 |

`performance_avg` 为 AUROC、Accuracy、MCC、F1、AUPR 五项的算术平均；此处所有宏平均均对三个 split 等权，不按样本数加权，也不是把所有样本合并后重算的 pooled 指标。

## 2. 评测设置

| 项目 | 内容 |
|---|---|
| Checkpoint | [trained_model/PLM_Unified/fold_1/seed_3407/best_ptcr_20260928_021049_999878.pt](../../../../trained_model/PLM_Unified/fold_1/seed_3407/best_ptcr_20260928_021049_999878.pt) |
| Checkpoint 类型 / 任务 | `task_model` / `ptcr` |
| Stage / 训练轮次 | `stage1b` / 第 `12` 轮 |
| Seed / Global step | `3407` / `13090` |
| 训练 run | `20260928_021049_999878`；验证集选择分数 `0.639196` |
| Checkpoint SHA-256 | `14d4d1204c7c3be4acc590c1e3d20e84ba4b4c0d66a24a56305f14e5676b54f4` |
| 配置基准 | [configs/plm_stage1.yaml](../../../../configs/plm_stage1.yaml)；评测路径配置 [analysis_results/stage1_ptcr_new_test/plm_stage1_pt_new_eval.yaml](../../plm_stage1_pt_new_eval.yaml) |
| 评估代码 | [source/evaluate_plm_stage1.py](../../../../source/evaluate_plm_stage1.py) |
| 输入方式 | ESM-C 与 TCR-BERT 在线编码；使用 checkpoint 中训练好的 pTCR task head |
| 设备 / 推理 batch | CUDA / 128 |
| 精度 / 分类阈值 | PLM 编码 float16，模型评估 bf16 autocast；阈值 0.5，按 `score > 0.5` 判正 |
| ESM-C encoder fingerprint | `2edde85b1885b70dbdf5857a7e78946df6fa2932fdd4d7346f836bb71de16469`（与 checkpoint 记录一致） |
| TCR-BERT encoder fingerprint | `2ef7e9ef2ed25e46d93097f21b5122eea8070ae4809c5d432818dc7326596056`（与 checkpoint 记录一致） |

评估使用了 checkpoint 训练时对应的本地 ESM-C / TCR-BERT；其 fingerprint 与 checkpoint 中的缓存元数据匹配。

## 3. 测试集构成

| Split | 样本数 | 负样本 | 正样本 | 正样本比例 | 唯一 peptide | 唯一 TCR | Peptide 长度 | TCR 长度 | 数据 SHA-256 | 原始评估 JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `test_unseen_pair` | 8,207 | 5,094 | 3,113 | 37.93% | 192 | 7,425 | 8–13 aa | 7–32 aa | `119edba70e2b28276048e96917460f25787ffa04ec2e520593bc71fa66374879` | [evaluation-hla_val-tcr_test_unseen_pair_20260928_104825_626299.json](evaluation-hla_val-tcr_test_unseen_pair_20260928_104825_626299.json) |
| `test_unseen_peptide` | 112,443 | 70,902 | 41,541 | 36.94% | 299 | 75,021 | 8–15 aa | 3–32 aa | `98e4e244b4564c8b432d8b7c12530236b5dc11c7969101b90dc662a14e998f39` | [evaluation-hla_val-tcr_test_unseen_peptide_20260928_105237_453112.json](evaluation-hla_val-tcr_test_unseen_peptide_20260928_105237_453112.json) |
| `test_unseen_tcr` | 142,960 | 71,480 | 71,480 | 50.00% | 1,410 | 33,061 | 8–15 aa | 5–32 aa | `bfca9a02552e7085fc0dee4c14c57a89950c2c5a95a1177c8816e1986c661a5a` | [evaluation-hla_val-tcr_test_unseen_tcr_20260928_105804_400158.json](evaluation-hla_val-tcr_test_unseen_tcr_20260928_105804_400158.json) |

所有 CSV 原始数据行均被保留；peptide 长度过滤剔除数为 0。表中的分组指标按 peptide 分组计算（Stage 1 pTCR 数据加载器以 peptide 生成 group id），不能解释成按 TCR 分组的宏平均。

## 4. 分 split 测试结果

| Split | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | performance_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 8,207 | 0.878640 | 0.856507 | 0.823931 | 0.623035 | 0.762218 | 0.781377 | 0.743977 | 0.872792 | 0.788866 |
| `test_unseen_peptide` | 112,443 | 0.798737 | 0.748986 | 0.759300 | 0.469792 | 0.646084 | 0.707203 | 0.594690 | 0.855745 | 0.684580 |
| `test_unseen_tcr` | 142,960 | 0.907831 | 0.907303 | 0.830680 | 0.661851 | 0.833880 | 0.818412 | 0.849944 | 0.811416 | 0.828309 |
| **等权宏平均** | **263,610** | 0.861736 | 0.837599 | 0.804637 | 0.584893 | 0.747394 | 0.768997 | 0.729537 | 0.846651 | 0.767252 |

### 校准与分组指标

| Split | NLL | Brier | ECE | 平均置信度 | 分组数（peptide） | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_unseen_pair` | 0.427757 | 0.131612 | 0.067306 | 0.887162 | 192 | 0.710126 | 0.869767 | 0.096692 | 3.121230 |
| `test_unseen_peptide` | 0.540042 | 0.172623 | 0.076449 | 0.831872 | 299 | 0.977393 | 0.837963 | 0.266338 | 3.497485 |
| `test_unseen_tcr` | 0.385494 | 0.121538 | 0.011896 | 0.841605 | 1,410 | 0.929064 | 0.891425 | 0.224152 | 3.130380 |

## 5. 简要解读

- `test_unseen_tcr` 的 `performance_avg` 最高（0.828309；AUROC 0.907831，AUPR 0.907303）。
- `test_unseen_peptide` 的 `performance_avg` 为 0.684580，低于另外两个 split；`test_unseen_pair` 为 0.788866。
- 三种 split 的样本来源和正负比例不同，分别报告更适合解释其不同未见条件下的表现；等权宏平均仅作总览。
- Accuracy、MCC、F1、Precision、Recall 与 Specificity 使用固定 0.5 阈值；本次未用 test 标签调阈值。

## 6. 复跑

```bash
export PYTHONPATH=source
for split in test_unseen_pair test_unseen_peptide test_unseen_tcr; do
  CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun/cuda \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python source/evaluate_plm_stage1.py \
    --config analysis_results/stage1_ptcr_new_test/plm_stage1_pt_new_eval.yaml \
    --checkpoint trained_model/PLM_Unified/fold_1/seed_3407/best_ptcr_20260928_021049_999878.pt \
    --fold 1 --seed 3407 --device cuda --hla-split val \
    --tcr-split "$split" --tasks ptcr --no-cache --online-batch-size 128
done
```

评估输入 CSV：
- [dataset/PT/new/test_unseen_pair.csv](../../../../dataset/PT/new/test_unseen_pair.csv)
- [dataset/PT/new/test_unseen_peptide.csv](../../../../dataset/PT/new/test_unseen_peptide.csv)
- [dataset/PT/new/test_unseen_tcr.csv](../../../../dataset/PT/new/test_unseen_tcr.csv)

