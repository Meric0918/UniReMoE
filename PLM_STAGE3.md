# PLM Stage 3：HLA-peptide-TCR 三元微调

该管线可以在 PLM Stage 1 或 Stage 2 联合模型上增加显式三元分支：

```text
peptide ─┐
         ├─ peptide→HLA cross-attention ── pHLA-conditioned peptide ─┐
HLA ─────┘                                                           │
                                                                     ├─ pooling ─ classifier
TCR ───────────────────── pHLA-conditioned peptide→TCR attention ────┘
```

它保留 pHLA、pTCR 两个原分支，并在 Stage 3B 中通过受控二元回放降低灾难性遗忘。

## 1. 输入与数据审计

当前配置使用 `dataset/PHT/classes/` 下的两个类别目录：

```text
PHT_000/
  train.csv
  val.csv
PHT_111/
  train.csv
  val.csv
```

`PHT_000he` 按实际目录名 `PHT_000` 处理。两个目录的 train 文件在内存中合并用于训练，两个 val 文件在内存中合并用于验证，不改写原始数据集。文件包含 `peptide,HLA,tcr,PH,PT,PHT` 及相关审计列，PHT 标签类型使用 `000` 和 `111`。

加载时按以下规则构建内存数据集：

- peptide 长度为 8–15；
- HLA/TCR 不超过配置上限 34；
- 标签只能为 0/1；
- 相同三元组出现冲突标签时停止；
- 类别目录顺序固定为 `PHT_000`、`PHT_111`，源文件路径写入审计信息；
- 集合内重复只在加载时检查，不修改原 CSV；
- 非标准残基保留给现有 tokenizer，但数量写入审计信息。

## 2. 初始化与训练

### 2.1 从 Stage 2 初始化

正式运行：

```bash
cd /home/mjp/Project/PLM/UnifyImmun

/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

加载器要求源文件为 `format_version=1`、`checkpoint_type=stage2_training` 的完整联合检查点。它会校验模型配置、fold 和三个基础 PLM 的指纹；源 checkpoint 不必处于 `complete`，因此 best Stage 2B checkpoint 可以直接使用。

### 2.2 从 Stage 1 直接初始化

```bash
cd /home/mjp/Project/PLM/UnifyImmun

/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage1-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_42/last.pt \
  --fold 4 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

Stage 1 来源必须是 `stage=complete` 的完整 `last.pt`，且模型配置、fold、seed 和三个基础 PLM 指纹均与当前 Stage 3 运行一致。禁止使用 `warmup_last.pt`，也禁止拼接 `best_phla.pt` 与 `best_ptcr.pt`：两个单任务文件可能来自不同训练轮次，并包含不同版本的共享 Peptide Adapter。

Stage 1 只提供三个 Adapter 和 pHLA/pTCR 下游分支。三个在线基础 PLM 从配置路径加载原始预训练权重；triplet attention、pooling 和 classifier 从 Stage 1 的 pTCR 分支复制初始化。

两种初始化方式均使用全新的 optimizer 和 scheduler，不继承源 checkpoint 的优化器状态。之后执行相同的 Stage 3 训练流程：

训练分两步：

1. Stage 3A 固定 2 个 epoch，只训练由 pTCR 分支复制初始化的 triplet cross-attention、pooling 和 classifier；
2. Stage 3B 每轮按 pHLA replay、pTCR replay、完整 triplet epoch 的顺序训练，渐进解冻三个 PLM 顶部各 2 层。

两个二元 replay 各抽取约 `0.25 × triplet_train_size` 个样本。每个验证集、每个任务均记录 ROC-AUC、AUPR、Accuracy、MCC、F1、Sensitivity、Specificity、Precision、Recall 和 validation loss。离散指标使用严格阈值：

```text
y_pred = (y_prob > 0.5).astype(int)
```

每个任务的验证分数和联合 round score 为：

```text
performance_avg = (ROC-AUC + Accuracy + MCC + F1 + AUPR) / 5
round_score = (pHLA performance_avg + pTCR performance_avg + PHT performance_avg) / 3
```

三个任务等权，样本量大的验证集不会自动获得更高权重。loss 只记录，不参与 checkpoint 选择。Stage 3B 至少训练 3 轮，连续 5 轮 round score 没有至少 `0.001` 的提升时早停；最多训练 10 轮。

## 3. 调试与恢复

PHT 模式使用 `--limit-per-type` 限制审计完成后每个 PHT 类型返回的样本，用于 smoke test：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --seed 9999 \
  --device cuda \
  --num-workers 0 \
  --swanlab-mode disabled \
  --limit-per-type 64
```

恢复训练：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --resume \
    trained_model/PLM_Triplet/fold_1/seed_42/stage3_last_<timestamp>.pt
```

`--resume`、`--stage1-checkpoint` 与 `--stage2-checkpoint` 互斥；新训练必须指定其中一种初始化 checkpoint。恢复文件包含 optimizer、scheduler、GradScaler、DataLoader RNG、全局 RNG、SwanLab run id、阶段、patience 状态以及统一的 `initialization_source` 来源信息。旧版仅包含 `source_stage2` 的 Stage 3 checkpoint 仍可恢复。

## 4. 训练产物

同一运行共享时间戳：

```text
trained_model/PLM_Triplet/fold_1/seed_42/
  stage3a_best_triplet_<timestamp>.pt
  stage3b_best_joint_<timestamp>.pt
  best_round/
    stage3_best_joint_<timestamp>.pt
  stage3_last_<timestamp>.pt
  triplet_best_<timestamp>.pt
```

- `stage3_last` 是完整恢复点，保留最后一轮状态；
- `stage3a_best_triplet` 用于进入 Stage 3B 前回载最佳预热权重；
- `stage3b_best_joint` 是完整的联合最佳训练点；
- `best_round/stage3_best_joint` 是按 round score 选出的完整联合模型，包含 pHLA、pTCR、PHT 三个头，不拆分保存；
- `triplet_best` 去除 optimizer，但保留三个 PLM 和 pHLA/pTCR/triplet 三个推理分支。

所有新产物通过 `initialization_source.kind=stage1|stage2` 记录初始化分支，并同时保存来源路径、fold、seed、训练阶段和基础 PLM 指纹。Stage 2 来源产物继续保留 `source_stage2` 兼容字段。

## 5. Validation 阈值与独立测试

训练只使用 validation 选择阈值：先最大化 MCC，并列时依次选择更高 F1、离 0.5 更近的阈值。该阈值写入 `triplet_best`，测试时固定使用。

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --checkpoint \
    trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_<timestamp>.pt \
  --split test \
  --device cuda \
  --hla-shuffle
```

输出：

```text
evaluation_triplet_test_<timestamp>.json
predictions_triplet_test_<timestamp>.csv
```

JSON 同时包含固定 0.5 阈值和 validation 选择阈值的指标。CSV 保留原始 CSV 行号、三条序列、标签、阳性概率和预测类别。`--hla-shuffle` 会额外报告确定性 HLA 位移消融；若性能和概率几乎不变，说明模型没有充分学习 HLA restriction。

如需重新分析 validation 阈值：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_triplet.py \
  --config configs/plm_triplet.yaml \
  --checkpoint <triplet_best.pt> \
  --split val \
  --select-threshold \
  --no-predictions
```

禁止在 test 上使用 `--select-threshold`。

## 6. 测试

合成测试不加载服务器大模型：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_unified \
  tests.test_plm_stage2 \
  tests.test_plm_triplet
```

正式训练前还应运行 `source/smoke_test_plm_encoders.py`，确认 ESM-C 和 TCR-BERT tokenizer 仍保持一残基一 token 对齐。
