# PLM Stage 9：同一 pHLA 下的 TCR 成对对比学习

Stage 9 从完整 Stage 3 `triplet_best` 初始化，在保留 pHLA、pTCR、PHT 三个分类分支的同时，对同一 peptide–HLA 对应的已知结合与不结合 TCR 进行显式排序。

## 1. 数据格式与审计

默认基础输入文件为审计清洗后的 `data/data_stage9/pht_pairs_clean.csv`；配置还会加载 `data/data_stage9/pht_pairs_full_valtest_supplement.csv`，将 full 数据中的新 PH 组加入 val/test：

```csv
peptide,hla,tcr_positive,tcr_negative,pair_id
SIINFEKL,YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY,CASSLGQETQYF,CASSIRSSYEQYF,pair-1
```

- `hla` 是 Stage 3 使用的最长 34 aa pseudo-sequence，而不是等位基因名称。
- `pair_id` 可省略，加载器会生成 `row:<CSV行号>`。
- peptide 长度必须为 8–15；HLA、两个 TCR 均不得超过 34 aa。
- `tcr_positive` 与 `tcr_negative` 必须不同。
- 同一 `(peptide,hla,tcr)` 不得同时出现在正、负角色。
- 完全重复 pair 只保留首行；一个 PH 对应多个不同阴性可以保留。
- 若同一 `(P,H,T)` 同时出现在正、负角色，清洗步骤会删除包含该冲突键的全部 pair 行；原始 `pht_pairs.csv` 保留不覆盖。
- full 数据补充只接受 Stage3 train 从未出现过的 PH、现有数据中不存在的 PH、合法序列和无角色冲突的 pair；补充文件显式记录 `split=val/test`，按 PH 组平衡分配且同一 PH 不跨 split。

审计器读取 Stage 3 train 三元组。只要一个 PH 组中任一正/负 TCR 曾被 Stage 3 训练见过，整个 PH 组都强制放入 Stage 9 train；其余 PH 使用 `split_seed=2026` 稳定哈希拆成 80/10/10。这样同一 PH 不会跨 train/validation/test。

```bash
./run_stage9.sh audit
```

审计失败会以退出码 2 停止，并在 JSON 中列出错误。训练不会自动修复正负角色冲突。

## 2. 表示与损失

```text
P/H Encoder + Adapter → pHLA cross-attention → PH pooling/projector → z_PH
TCR Encoder + Adapter → shared TCR pooling/projector                → z_T
```

投影结果为 128 维 L2 单位向量。每个 batch 的 P/H 只编码一次，正负 TCR 合并为一个 `2B` batch 编码。模型同时复用 Stage 3 triplet attention、pooling 和 classifier，分别输出正负三元组的结合 logits。

成对对比损失只使用行内明确阴性：

```text
s+ = cosine(z_PH, z_T+)
s- = cosine(z_PH, z_T-)
L_con = softplus((s- - s+) / 0.07)
L_cls = (CE(logits+, 1) + CE(logits-, 0)) / 2
```

Stage 9A 只优化 `L_con`；Stage 9B 使用 `L_con + L_cls`。其他 batch 中的 TCR 不会被自动当作负样本。

## 3. 训练

配置文件：`configs/plm_stage9.yaml`。

```bash
./run_stage9.sh train \
  --stage3-checkpoint \
    trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt
```

调试运行：

```bash
./run_stage9.sh train \
  --stage3-checkpoint <triplet_best.pt> \
  --limit 64 \
  --num-workers 0 \
  --swanlab-mode disabled
```

`--limit` 只在完整审计和 PH 拆分后限制每个 split，不能用于正式实验。

训练过程：

1. **Stage 9A**：2 epochs，冻结整个 Stage 3，只训练两个新 pooling 和两个 projector；按 validation 对比损失、排序准确率、相似度间隔依次选模。
2. **Stage 9B**：每轮依次进行 pHLA、pTCR、原 PHT replay，然后训练完整 pair epoch。三个 replay 各上限为 `0.25 × pair_train_rows`。
3. Stage 9A/9B 全程冻结 peptide、HLA、TCR 三个 PLM encoder，不解冻任何 encoder 层，也不将 encoder 参数加入 optimizer。
4. Stage 9B 只更新任务作用域内的 adapter、Stage 3 attention/pooling/classifier，以及四个新增 contrastive pooling/projector 模块。
5. 候选模型必须满足原三任务 validation joint score 相对来源 Stage 3 下降不超过 0.01；合格模型按 pair ranking accuracy、mean similarity gap、paired classification score 依次选优。
6. 最少训练 3 rounds，最多 10 rounds，连续 3 rounds 无合格提升后早停。

恢复训练：

```bash
./run_stage9.sh train \
  --resume trained_model/PLM_Stage9/fold_4/seed_2026/stage9_last_<timestamp>.pt
```

恢复会校验来源 Stage 3 SHA-256、pair CSV SHA-256、fold、seed、全部 Stage 9 超参数和有效数据规模，并恢复 optimizer、scheduler、AMP、全局 RNG 与 DataLoader RNG。

训练产物：

```text
trained_model/PLM_Stage9/fold_<fold>/seed_<seed>/
  data_audit_<timestamp>.json
  stage9a_best_contrastive_<timestamp>.pt
  stage9b_best_constrained_<timestamp>.pt
  stage9_last_<timestamp>.pt
  stage9_best_<timestamp>.pt
```

`stage9_last` 用于恢复；`stage9_best` 是包含三个冻结 PLM、原四个推理接口和新 embedding heads 的部署 checkpoint。

## 4. 评估与 embedding 导出

测试集使用 checkpoint 中仅由 validation 选择的分类阈值：

```bash
./run_stage9.sh evaluate \
  --checkpoint trained_model/PLM_Stage9/fold_4/seed_2026/stage9_best_<timestamp>.pt \
  --split test \
  --save-embeddings
```

默认还会重新加载来源 Stage 3，在同一 pair split 上比较正负分类概率排序。若只需 Stage 9 结果，可加 `--no-source-comparison`。输出包括 JSON 指标、逐 pair CSV，以及可选的 `z_PH/z_T+/z_T-` NPZ。

如需重新选择 validation 阈值：

```bash
./run_stage9.sh evaluate \
  --checkpoint <stage9_best.pt> \
  --split val \
  --select-threshold
```

程序禁止在 test 上使用 `--select-threshold`。

任意 PHT 长表 embedding 导出，输入列必须为 `peptide,hla,tcr`：

```bash
./run_stage9.sh export \
  --checkpoint <stage9_best.pt> \
  --input-csv input_pht.csv \
  --output-npz embeddings.npz
```

NPZ 包含两侧单位向量、余弦相似度、Stage 3 triplet 结合概率和原序列标识。

## 5. 测试

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage9 \
  tests.test_plm_triplet \
  tests.test_plm_stage8
```

正式训练前还应运行现有 encoder smoke test，确认本机 ESM-C/TCR-BERT checkpoint 与 tokenizer residue 对齐。
