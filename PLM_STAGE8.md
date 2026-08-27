# PLM Stage 8：Stage 1 同源的四类型约束训练

Stage 8 与 Stage 3 位于同一层级，二者都从同一个 Stage 1 `best_joint` 初始化：

```text
Stage 1 best_joint
├── Stage 3A → Stage 3B
└── Stage 8A → Stage 8B
```

Stage 8 不读取 Stage 3 checkpoint，也不读取 Stage 3 或 Stage 8 的 embedding cache。
当前唯一允许的 Stage 1 来源是：

```text
trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/
best_joint_20260824_174202_510837.pt
SHA256: 5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96
```

初始化时会校验 checkpoint 类型、模型配置、fold/seed、三个基础 PLM fingerprint 和完整文件 SHA256；随后把 Stage 1 pTCR 的 triplet cross-attention、pooling、classifier 精确复制到 Stage 8 的新 PHT 模块。

## 数据审计与在线输入

训练/验证数据位于 `dataset/PHT/classes/`，必须包含 `PHT_000`、`PHT_010`、`PHT_100`、`PHT_111` 四个目录，每个目录包含 `train.csv` 和 `val.csv`。审计会校验目录名、`class_code`、PH/PT/PHT 标签、重复三元组、跨目录冲突、序列长度和氨基酸字符；冲突会停止流程。

`--limit-per-type` 只能在完整审计成功后应用。Stage 8 训练直接将原始 peptide/HLA/TCR 序列交给在线 PLM；Stage 8 的 cache/precompute/export 公共入口已移除。`stage8_data.py` 中仍保留 Stage 10 所需的通用 cache helper，Stage 10 的缓存流程不变。

```bash
./run_stage8.sh audit
```

## Stage 8A：共享 PHT warmup

Stage 8A 从上述 Stage 1 checkpoint 开始，训练两轮；只开放新 triplet/PHT 模块，三个在线 PLM、已有 pHLA/pTCR adapter、attention 和 classifier 均冻结。Stage 8A 不使用 PH/PT replay，也不启用任何一致性约束。

PHT 监督采用四类平衡 batch。`111` 的总交叉熵权重为 0.5，`000`、`010`、`100` 三类合并后的总权重为 0.5；另保留 PH/PT 监督项的配置权重 `0.3/0.2`。micro batch 为 4，gradient accumulation 为 16，有效 batch 为 64。

## Stage 8B：四条件在线训练

四个条件 `baseline`、`hier_only`、`pt_imp_only`、`full` 共享同一个、字节一致的 Stage 8A warmup checkpoint。每个 Stage 8B round 的固定次序为：

```text
PH replay → PT replay → balanced four-class joint pass
```

PH/PT replay 比例为 0.25；在线 PLM 的 peptide、HLA、TCR 分支各解冻顶层两层，joint pass 训练相关 PLM 层和全部下游模块。默认设置为最少 5、最多 15 个 round，patience 5，minimum delta `0.0001`。选择分数保持当前 Stage 8 定义：

```text
score(task) = (AUROC + Accuracy + MCC + F1 + AUPR) / 5
joint = 0.4 * score(PHT) + 0.2 * score(PH) + 0.4 * score(PT)
```

只有 Stage 8B 启用约束，并按前 5 个 round 线性 ramp：

- hierarchy：最大权重 `0.05`；
- PT implication：最大权重 `0.03`；
- `PH & PT => PHT` 不加入，避免错误约束 `010`/`100`。

四类数据的严格审计、平衡采样、PHT 交叉熵和上述约束方向共同构成 Stage 8 的新增限制；约束违反率只作为报告指标，不改变选优分数。

## 运行命令

先审计，再生成一次共享 warmup：

```bash
./run_stage8.sh audit
./run_stage8.sh warmup --stage1-checkpoint \
  trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260824_174202_510837.pt \
  --device cuda
```

从指定 warmup 运行单个条件：

```bash
./run_stage8.sh train --condition full \
  --warmup-checkpoint trained_model/PLM_Stage8/fold_1/seed_42/warmup/stage8a_best_pht_TIMESTAMP.pt \
  --device cuda
```

一次 warmup、四个条件：

```bash
./run_stage8.sh ablations --stage1-checkpoint \
  trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260824_174202_510837.pt \
  --device cuda
```

恢复只能使用同一条件目录、同一 timestamp 的 `stage8_last_TIMESTAMP.pt`；它必须找到并验证同目录的 `stage8_best_TIMESTAMP.pt` 和 warmup SHA。旧 Stage 8 V1 checkpoint 会被明确拒绝。

## V2 产物

由于 Stage 8B 会微调 PLM，三类 V2 artifact 都保存完整的 encoder-inclusive model state：

- `stage8a_best_pht_TIMESTAMP.pt`：共享 Stage 8A warmup，无 optimizer state；
- `stage8_best_TIMESTAMP.pt`：当前条件的完整部署模型，无 optimizer state；
- `stage8_last_TIMESTAMP.pt`：完整当前模型和 optimizer/scheduler/scaler/RNG，可恢复训练，但不重复嵌入 `best_model`。

每个 artifact 都记录 Stage 1 SHA、Stage 8A SHA、数据 fingerprint、effective data（包括 debug limit 和样本数）、condition、training contract 和验证指标。V2 public evaluator 直接加载完整模型并在线编码，不依赖任何 Stage 3 来源。

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python source/evaluate_plm_stage8.py \
  --checkpoint trained_model/PLM_Stage8/fold_1/seed_42/full/stage8_best_TIMESTAMP.pt \
  --device cuda

/home/mjp/miniconda3/envs/unifyimmun/bin/python source/evaluate_plm_stage8_tcr_sets.py \
  --checkpoint trained_model/PLM_Stage8/fold_1/seed_42/full/stage8_best_TIMESTAMP.pt \
  --device cuda
```

## 与 Stage 3 的比较边界

两者共享 Stage 1 起点和 triplet 模块的 pTCR 初始化，但训练目的不同：Stage 3 以 triplet 任务为主并进行二分类 replay；Stage 8 在四种 PHT 类型上进行 PH/PT/PHT joint pass，并新增有向 hierarchy、PT implication 约束。四个 Stage 8 条件使用相同 warmup，便于把差异归因到约束项。
