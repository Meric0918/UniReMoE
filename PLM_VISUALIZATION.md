# Stage 1/Stage 2 embedding 可视化

该分析在相同 Fold、seed 和验证样本上比较 Stage 1 与 Stage 2：

- 单分子空间默认使用 `SequenceAdapter` 后、masked mean pooling 的共同表示。
  当前 Stage-1/Stage-Adapter 配置为 256 维；实际维度始终从所用 checkpoint 对应的
  `model.model_dim` 读取（旧 Stage-2 checkpoint 仍可能是 128 维）；
- 配对空间：提取 cross-attention 与 masked attention pooling 后、分类器前的
  `model_dim` 维 pHLA/pTCR 向量（当前 Stage-1/Stage-Adapter 配置为 256 维）；
- 如需复现旧版分析，可增加 `--entity-feature-space raw`，将单分子空间切回
  原始 PLM 的 Peptide/HLA 960 维或 TCR 768 维向量；
- 每个实体和每个任务分别拟合降维器，不把 ESM-C 与 TCR-BERT 的原始坐标空间
  直接混合；
- 同一个分析空间内，Stage 1 和 Stage 2 的匹配样本拼接后共同拟合 PCA/UMAP，
  因而左右面板可以直接比较。

## 1. 安装依赖

```bash
cd /home/mjp/Project/PLM/UnifyImmun
/home/mjp/miniconda3/envs/unifyimmun/bin/pip install -r requirements-plm.txt
```

## 2. Fold 1 / seed 42 正式分析

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/visualize_plm_embeddings.py \
  --config configs/plm_stage2.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --split val \
  --tasks phla,ptcr \
  --reducers pca,umap \
  --entity-feature-space adapter \
  --max-pairs-per-task 20000 \
  --max-entity-sequences 20000 \
  --metric-sample-size 5000 \
  --random-state 42 \
  --device cuda \
  --output-dir analysis_results/plm_embeddings/fold_1_seed_42
```

Stage 1 checkpoint 默认从 Stage 2 checkpoint 的
`source_stage1_checkpoint` 字段读取，并要求二者严格属于同一训练路径。正式主图
使用验证集原始类别比例的分层抽样，不进行过采样。

少量数据 smoke test：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/visualize_plm_embeddings.py \
  --config configs/plm_stage2.yaml \
  --stage2-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/stage2b_best_joint_20260726_235238_711877.pt \
  --tasks phla,ptcr \
  --reducers pca \
  --max-pairs-per-task 128 \
  --max-entity-sequences 128 \
  --metric-sample-size 64 \
  --device cuda \
  --output-dir analysis_results/plm_embeddings/smoke
```

## 3. 输出

```text
analysis_results/plm_embeddings/fold_1_seed_42/
  entity_coordinates.csv.gz  # 默认是 adapter 后的 model_dim 维实体表示
  pair_coordinates.csv.gz
  summary.json
  figures/
    png/
      pca/
      umap/
    svg/
      pca/
      umap/
```

`summary.json` 同时记录实体/配对 feature space 与实际维度、checkpoint、抽样数量、降维参数、PCA 方差解释率、
trustworthiness、Stage 1→2 高维余弦变化、标签 silhouette 和抽样集预测指标。
如需保存用于当前分析的高维向量，可增加 `--save-high-dimensional`；实体 NPZ
保存的空间由 `--entity-feature-space` 决定，配对 NPZ 保存分类器输入向量。
输出目录非空时脚本会停止，防止混入旧坐标；确认要替换脚本生成的分析文件时增加
`--overwrite`。

## 4. 解释边界

- UMAP/PCA 图用于观察邻域、连续梯度、异常点和 Stage 1→2 表征迁移，二维坐标轴
  本身没有固定生物学含义；
- 验证样本的阳性比例、TP/TN/FP/FN 和预测概率应结合查看，不能只依据视觉上是否
  出现簇来判断模型优劣；
- UMAP 是随机降维方法，正式结果必须固定 `--random-state`，并保留坐标 CSV 和
  `summary.json`；
- 二维邻近或较高 attention 不等于残基存在因果接触。若要提出残基机制结论，应再
  使用 integrated gradients、attention 随机化以及已知结构进行交叉验证。
