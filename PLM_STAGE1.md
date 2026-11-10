# Unified PLM Stage 1

该管线实现《Embedding论文问题与相关研究》第 6–12 章确认后的版本：

- Peptide/HLA：`/home/mjp/model/ESMC-300M`；
- TCR：`/home/mjp/model/TCR-Bert`；
- HLA 第一版直接编码现有 33/34 位 pseudo-sequence；
- 三个独立 Adapter，共享一个 Peptide Adapter；
- 256 维统一表示、8-head cross-attention、masked attention pooling、双分类头；
- 1A 无 FGM 预热，1B pHLA/pTCR 逐 epoch 交替；
- 1B 只对当前任务的 Peptide Adapter 与 receptor Adapter 使用 FGM；
- SwanLab 按 fold 分组，三个随机种子分别记录。

## 1. 环境

服务器已有 CUDA PyTorch 环境时，只安装缺失依赖：

```bash
cd /home/mjp/Project/PLM/UnifyImmun
pip install -r requirements-plm.txt
```

`transformers==4.57.6` 与 ESM-C checkpoint 的导出版本一致。正式运行前先验证
两个本地 checkpoint、tokenizer 和逐残基对齐：

```bash
python source/smoke_test_plm_encoders.py \
  --config configs/plm_stage1.yaml \
  --device cuda
```

预期关键输出：

```text
ESM-C hidden size：960
TCR-BERT hidden size：768
ESM-C lengths：[9, 34]
TCR-BERT lengths：[13, 15]
```

任一长度不一致都应停止，不要跳过 tokenizer 对齐检查。

## 2. 构建残基级缓存

默认只收集 Fold 1 的 train 和 validation（固定宽度 float16 预计约 21 GB），
过滤不在 8–15 范围内的 peptide，再生成分片 HDF5 和 SQLite 索引：

```bash
python source/precompute_plm_embeddings.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --device cuda
```

缓存目录：

```text
embedding_cache/plm_stage1/fold_1/
  filter_stats_<YYYYMMDD_HHMMSS_ffffff>.json
  peptide/
  hla/
  tcr/
```

如需重建，必须显式传入 `--overwrite`。可使用 `--entities peptide hla` 或
`--entities tcr` 单独重建实体缓存。

## 3. 调试训练

调试时限制每个 split 的行数并关闭远端记录：

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 9999 \
  --limit 4096 \
  --num-workers 0 \
  --swanlab-mode local
```

`--limit` 只用于 smoke test，正式实验禁止使用。调试时使用
`--num-workers 0` 可将 HDF5 缓存读取留在主进程，便于定位数据问题。
调试 seed 使用 9999，避免覆盖正式实验的 seed 42/3407/2026。

无需缓存的在线编码调试必须同时指定 `--no-cache` 和 `--limit`，并自动使用
`training.online_batch_size`：

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42 \
  --no-cache \
  --limit 256 \
  --swanlab-mode local
```

## 4. 正式三随机种子实验

```bash
python source/run_plm_stage1_seeds.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seeds 42 3407 2026 \
  --device cuda \
  --swanlab-mode online
```

每个 seed 的产物：

```text
trained_model/PLM_Unified/fold_1/seed_<seed>/
  warmup_last_<YYYYMMDD_HHMMSS_ffffff>.pt
  last_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_phla_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_ptcr_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_phla_calibration_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_ptcr_calibration_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_phla_robust_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_ptcr_robust_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_round/
    best_joint_<YYYYMMDD_HHMMSS_ffffff>.pt
```

同一次训练的所有文件共享同一个时间戳，重复运行相同 fold/seed 不会覆盖旧结果。
`last_<timestamp>.pt` 保留优化器、scheduler、两个任务分支及各任务早停状态，
用于恢复训练。`best_phla_<timestamp>.pt` 与 `best_ptcr_<timestamp>.pt`
是精简的独立推理模型，分别只包含共享
Peptide Adapter 和对应任务的 receptor Adapter、cross-attention、pooling
及分类头。`best_round/best_joint_<timestamp>.pt` 是唯一的轮级最佳模型，
包含 PHLA 与 PTCR 两个任务头的联合模型；不会在 `best_round/` 下再拆分出
`phla` 和 `ptcr` 两个文件，也不会把上游 peptide PLM encoder 写入 checkpoint。

三次训练完成后生成：

```text
trained_model/PLM_Unified/fold_1/seed_summary_<YYYYMMDD_HHMMSS_ffffff>.json
```

每个 benchmark 使用以下分数：

```text
performance_avg = (AUROC + Accuracy + MCC + F1 + AUPR) / 5
```

离散分类指标固定使用严格阈值 `y_pred = (y_prob > 0.5).astype(int)`。
每个任务把三个 benchmark 的 `performance_avg` 做简单平均，作为该任务当前
验证分数；三个 benchmark 等权，样本量不会带来额外权重。Validation loss
只记录，不参与 checkpoint 选择。

每个任务分别维护最佳分数和 patience：当前分数必须比历史最佳至少高 `0.001`
才保存任务 checkpoint 并清零计数；连续达到配置中该任务 patience 的有效训练
round 没有显著提升时，只停止该任务，另一任务可继续训练。当前 Stage1-v2 配置
为 pHLA `5`、pTCR `5` 个 round。

每轮按 HLA(PH) → TCR(PT) 训练，轮级分数使用同一轮当前模型重新评估得到的：

```text
round_score = (当前 HLA 验证分 + 当前 TCR 验证分) / 2
```

轮级同样使用 `0.001` 的显著提升阈值；连续达到 `round_early_stopping_patience`
（当前配置为 `5`）轮没有显著提升时停止整个渐进训练。最佳轮次的 PHLA/PTCR 联合模型写入唯一的
`best_round/best_joint_<timestamp>.pt`。

### 每轮 task-best 的 UMAP 可视化

当某轮刷新 pHLA 或 pTCR 的 task-best 时，训练器会从各验证 split 的固定分层
样本中收集 `pooled_embedding`，并同步生成两类 UMAP 图：按真实标签着色，以及
按 TP/TN/FP/FN 着色。未刷新 task-best 的轮次不会重复生成相同模型的图。

首次 best 为每个任务分别拟合一个 PCA+UMAP 参考空间；后续 best 使用同一个
降维器的 `transform()`。因此同一任务、同一次训练中各轮坐标可以直接比较，
但 pHLA 与 pTCR 之间、不同 seed/run 之间不应比较绝对坐标。

`configs/plm_stage1.yaml` 和 `configs/plm_stage1_ph_pt.yaml` 默认开启该功能，
也可以在命令行覆盖：

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1_ph_pt.yaml \
  --fold 1 \
  --seed 42 \
  --best-umap \
  --best-umap-max-samples-per-split 1000
```

临时关闭时使用 `--no-best-umap`；输出目录可通过
`--best-umap-output-dir` 指定。默认产物结构为：

```text
figures/stage1/best_model_umap/fold_<fold>/seed_<seed>/<timestamp>/
  index.json
  phla/
    reference_reducer.joblib
    round_0001/
      umap_by_label.png
      umap_by_outcome.png
      coordinates.csv.gz
      summary.json
  ptcr/
    ...
```

其中 `coordinates.csv.gz` 保留 split、原数据集行索引、标签、预测概率、
TP/TN/FP/FN 和二维坐标，便于进一步定位拉低表现的样本。UMAP 在验证前向中
复用已有输出，仅把固定抽样行复制到 CPU；首次拟合会增加少量训练耗时。

## 5. 断点续训与评估

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42 \
  --resume trained_model/PLM_Unified/fold_1/seed_42/last_20260726_120000_000001.pt
```

验证集评估：

```bash
python source/evaluate_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42
```

未传 `--checkpoint` 时，评估脚本优先选择最新的
`best_round/best_joint_<timestamp>.pt`；若不存在则回退到最新一组时间戳一致
的 pHLA/pTCR 权重。生成的评估 JSON 也会自动添加新的时间戳后缀。

单任务 checkpoint 必须配合对应的 `--tasks` 使用：

```bash
python source/evaluate_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --checkpoint trained_model/PLM_Unified/fold_1/seed_42/best_phla_20260726_120000_000001.pt \
  --tasks phla \
  --fold 1 \
  --device cuda
```

旧版本已有的联合训练 checkpoint 无需重新训练，可直接拆分为任务推理模型：

```bash
python source/export_plm_task_models.py \
  --checkpoint trained_model/PLM_Unified/fold_1/seed_42/best_joint.pt
```

如需评估 independent/external/covid/triple，先把相应 split 加入 YAML 的
`cache.hla_splits` 或 `cache.tcr_splits` 并重建缓存，再向评估命令传入
`--hla-split`/`--tcr-split`。全部测试集同时使用固定宽度缓存可能超过 30 GB，
应按磁盘预算分批构建。

## 6. SwanLab 记录

默认 project 为 `unifyimmun`，group 为 `plm-stage1-fold1`。每个 run 记录：

- 配置、模型路径和 cache checkpoint 指纹；
- 数据保留/过滤数量；
- clean/adversarial loss；
- FGM epsilon、攻击参数数量和 Adapter 梯度范数；
- 每任务样本数、optimizer step 数及比例；
- AUROC、AUPR、accuracy、MCC、F1；
- NLL、Brier、ECE、置信度分布、P95/P99 单样本 loss；
- group-macro 指标、worst-group NLL、retention gate 和 shared adapter 冻结状态；
- joint score、best joint score、early-stopping 状态；
- 各参数组学习率和 checkpoint 恢复所需 run id。

Stage1-v2 将 scheduler 总预算与 `max_rounds` 解耦，并按 shared peptide adapter、
pHLA/pTCR 私有模块分别设置学习率和更新计数（私有分支只在对应任务 step 上推进，
shared 参数保持一份 Adam 状态）；训练损失支持轻度 label smoothing、FGM
权重/epsilon warmup 和 entity-label inverse-sqrt 分组加权。`best_*_calibration`
按验证 NLL 选择，`best_*_robust` 优先按 group-macro accuracy 选择，普通
`best_phla/best_ptcr` 仍按判别综合分数选择。
