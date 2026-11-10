# Stage11：串联 PHT 路径的两级对比学习

Stage11 继承 Stage8 的 **模型结构**：先由 peptide 查询 HLA 得到 PH 残基表示，再由 PH 查询 TCR 得到 PHT 表示。Stage11 不载入 Stage8/Stage1 下游权重；ESM-C 和 TCR-BERT 使用本地预训练权重、始终冻结，adapter、交叉注意力、分类头和对比头从随机参数开始训练。最终任务只输出 PHT 结合概率。

```text
peptide -> frozen ESM-C -> peptide adapter --+--> PH cross attention: Q=peptide, K/V=HLA
                                             |       |
                                             |       +--> PH sequence --------+--> PHT cross attention: Q=PH, K/V=TCR
                                             |                                 |            |
HLA -----> frozen ESM-C -> HLA adapter ------+                                 |            +--> attention pooling -> classifier -> PHT
                                             |                                 |
                peptide / HLA pooled projections -> P↔H contrastive loss        |
                                                                               |
TCR ---> frozen TCR-BERT -> TCR adapter ---------------------------------------+
                                     PH / TCR pooled projections -> PH↔T contrastive loss
```

投影是独立的侧支：每侧使用 masked attention pooling、无偏置线性层、L2 归一化。交叉注意力的输入始终是 256 维残基序列；不会把单个投影向量当作 attention 输入。PH 与 TCR、peptide 与 HLA 分别去重，因此相似度矩阵可以是矩形。对比头只在训练时运行。`forward_ph_contrastive`、`forward_pht_contrastive` 返回归一化表示；`forward_pht` 返回 `[B,2]` logits，可选择返回两级注意力权重和最终池化权重。

## 训练数据与目标

- P↔H：取四类 PHT 训练数据中 `PH=1` 的配对，以及 `dataset/PH/train.csv` 中的正例。按 `(peptide,HLA序列)` 去重。补充 PH 数据若与任何 PHT 验证集 PH 配对重叠，则整对排除；跨任务同一配对标签冲突时停止。
- PH↔T：只取四类 PHT 训练数据中 `PHT=1` 的三元组，左侧实体是完整 `(peptide,HLA序列)`，右侧实体是 TCR。
- 分类：沿用 `PHT_000/010/100/111` 的 train/val 划分与四类平衡 batch。`111` 与其余三类分别贡献 0.5 倍平均交叉熵。
- 对比正例矩阵由全部审计后的**训练正例关系**构建。批内未标为正例的组合承担对比负例作用，同时分别记录其中已知阴性与未标注组合的数量。验证和测试关系不进入此矩阵。

对比损失使用温度 `0.2` 的双向多正例 CLIP。每个有效 anchor 对所有已知正例的 `log_softmax` 平均；没有正例或没有负例的 anchor 不参与对应方向。若整批无有效 anchor，该批不更新模型。相似度、归一化和 `log_softmax` 均以 FP32 计算。单正例方阵时退化为 TRAP 所用的双向 CLIP 损失。[TRAP 源码](https://github.com/gejingxuan/TRAP/blob/main/TRAP/train.py)

## 交替训练

每个 round 按以下顺序执行：

1. **P↔H 对比**：使用 PH 正例池按无放回轮转取样，只更新 peptide/HLA adapter 和第一级对比头。
2. **PH↔T 对比**：完整遍历 PHT 正例一次，更新三个 adapter、第一级交叉注意力和第二级对比头。PH 梯度经过第一级交叉注意力回传。
3. **PHT 分类**：完整遍历四类平衡 batch，更新三个 adapter、两级交叉注意力、PHT 池化和分类头。对比头不参与此阶段。
4. 在 PHT 验证集评估，以 `(AUROC+AUPR)/2` 保存 best。阈值固定为 `0.5`；每类阴性的假阳性率单独报告。

前两个阶段计划步数均为 `ceil(PHT正例数/contrastive_batch_size)`；即使 PH 正例池更大，两级对比每轮的数据量也一致。三个阶段共享一个 AdamW 优化器及全局 warmup/cosine 调度器；阶段外参数没有梯度。默认 batch 为对比 256、分类 256（每类 64）、验证 256，学习率 `1e-4`，最多 100 rounds，至少 5 rounds，patience 20。下游计算默认 bf16，冻结 PLM 缓存为 float16。

四个消融条件由 `--condition none|ph|pht|both` 指定；其余配置及分类数据一致。消融时每轮总更新步数会因对比阶段数量不同而变化，历史文件逐阶段记录实际步数。

## 运行

默认配置在 [`configs/plm_stage11.yaml`](configs/plm_stage11.yaml)。脚本使用项目的 `unifyimmun` Python 环境；可通过 `PYTHON_BIN` 覆盖。数据审计不会加载 PLM。预计算需要本地 ESM-C 与 TCR-BERT 模型，并分别写入 `embedding_cache/plm_stage11/fold_<fold>/train`、`test`。

```bash
./run_stage11.sh audit
./run_stage11.sh precompute --scope train
./run_stage11.sh train --condition both
./run_stage11.sh precompute --scope test
./run_stage11.sh evaluate --checkpoint trained_model/PLM_Stage11/fold_1/seed_42/both/stage11_best_TIMESTAMP.pt --split all
```

恢复训练使用 `./run_stage11.sh train --resume PATH_TO_STAGE11_LAST.pt`。不同消融条件使用独立输出目录。`best` 保存模型与最佳验证结果；`last` 额外保存优化器、调度器、AMP scaler、随机状态、PH 轮转采样位置及最佳模型，可在 round 边界继续训练。数据、模型配置、PLM/cache 指纹或训练条件变更时拒绝恢复。测试评估读取 `dataset/PHT/test.csv` 及四个 unseen peptide/HLA/TCR/triple benchmark 文件，输出逐样本 CSV、JSON 和 Markdown 汇总；测试三元组与训练三元组的重叠会如实报告。

可在预计算前用 `./run_stage11.sh precompute --scope train --audit-only` 查看所需序列数量。若指定缓存目录已含同名实体，只有明确传入 `--overwrite` 才覆盖该实体；非 Stage11 缓存目录会被拒绝。
