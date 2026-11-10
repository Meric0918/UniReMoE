# Stage 8：Stage 1 同源的四类型约束训练

Stage 8 与 Stage 3 是同一层级的并行实验分支，均直接从 Stage 1 的固定 `best_joint` 初始化，不使用 Stage 3 checkpoint 或 Stage 3/Stage 8 embedding cache。完整流程和命令见 [PLM_STAGE8.md](../PLM_STAGE8.md)。

```text
Stage 1 best_joint
├── Stage 3A → Stage 3B
└── Stage 8A → Stage 8B
```

固定来源：

```text
trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260824_174202_510837.pt
SHA256 5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96
```

Stage 8A 训练两轮，仅训练新 PHT triplet 模块；四个 Stage 8B 条件共享这份 warmup，并按 `PH replay → PT replay → balanced four-class joint pass` 训练。micro batch/accumulation 为 `4/16`，online PLM 各解冻顶层两层，replay 比例为 `0.25`。约束只在 Stage 8B 生效，前 5 round ramp 到 hierarchy `0.05` 和 PT implication `0.03`，不加入 `PH & PT => PHT`。

Stage 8 V2 保存完整 online model：`stage8a_best_pht_TIMESTAMP.pt`、`stage8_best_TIMESTAMP.pt`、`stage8_last_TIMESTAMP.pt`。last 不嵌入重复的 `best_model`，恢复时必须有同目录同 timestamp 的 best companion，并验证 Stage 1、warmup、数据和 contract identity。旧 V1 artifact 不接受。

```bash
./run_stage8.sh audit
./run_stage8.sh warmup --stage1-checkpoint /path/to/best_joint_20260824_174202_510837.pt
./run_stage8.sh train --condition full --warmup-checkpoint /path/to/stage8a_best_pht_TIMESTAMP.pt
./run_stage8.sh ablations --stage1-checkpoint /path/to/best_joint_20260824_174202_510837.pt
```

`ablations` 默认并行启动四个 Stage 8B 条件；每个条件独立写 checkpoint 和 SwanLab run。每个完成的 round 都会记录训练指标、验证指标、joint score、约束分量与违反率。已有 warmup/ baseline last 时，可用 `--warmup-checkpoint ... --resume-baseline ...` 跳过 warmup 并恢复 baseline；资源受限时用 `--sequential` 串行执行。

Stage 10 仍使用 `stage8_data.py` 中保留的通用 cache helper 和 cached Stage 8 binding evaluator；移除的仅是 Stage 8 自己的 precompute/export 公共入口。
