# PLM Stage 13：GraphSAGE 融合的 Stage1 双任务模型

Stage13 是独立于 Stage1 的实验管线。Stage1 的代码、配置和文档保持不变。

## 模型顺序

```text
ESM-C / TCR-BERT residue hidden states
    ↓ masked mean（分子节点初始特征）
PH/PT train-positive heterogeneous GraphSAGE
    ↓ 256 维关系感知节点表示
graph token + residue tokens
    ↓
Stage1 adapters → cross-attention → masked pooling → pHLA/pTCR classifier
```

图只使用 `dataset/PH/train.csv` 与 `dataset/PT/train.csv` 的去重正例。负例不建边，
验证/测试 pair 不建边。训练一个 batch 时会同时从邻接表中移除该 batch 的查询正边，
避免模型通过“边是否存在”直接读取标签；未知节点使用自身 encoder 特征并作为孤立节点。

默认 GraphSAGE 为两层关系感知 mean aggregation，fanout 为 `[16, 8]`。HLA、peptide、
TCR 的 encoder 特征先分别投影到共同图空间；最终图表示投影回实体输入维度，作为
一个 token 前置到残基序列，因此保留 Stage1 的残基级交互能力。

## 使用

先确认 Stage1 residue cache 已存在，然后构建图：

```bash
./run_stage13.sh build-graph
```

调试建图可以限制每个训练 CSV 的行数并显式覆盖输出：

```bash
./run_stage13.sh build-graph --overwrite --limit 256
```

训练、恢复和验证：

```bash
./run_stage13.sh train --swanlab-mode disabled
./run_stage13.sh resume trained_model/PLM_Stage13/fold_1/seed_42/last_<timestamp>.pt
./run_stage13.sh evaluate --checkpoint <best_joint.pt> --split validation
```

训练默认沿用 Stage1 的两轮任务 warmup、pHLA/pTCR 交替 round、FGM、任务级早停和
联合 checkpoint 逻辑。Stage13 checkpoint 依赖 graph manifest 与 residue cache 指纹，
只能由 Stage13 evaluator 加载。

## 产物与审计

图产物位于 `embedding_cache/plm_stage13/fold_<fold>/`，包括节点序列、CSR 邻接、
masked-mean 特征和 manifest。训练产物位于 `trained_model/PLM_Stage13/`；正式训练
前应使用不带 `--limit` 的命令重建完整图，并保存 manifest、checkpoint 与评估 JSON。
