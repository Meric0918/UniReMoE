# PLM Stage 14：Hyper-PMT 异构超图融合

Stage14 是独立于 Stage8 的结构优化分支。它从当前 Stage1 `best_joint`
初始化，保留旧 Stage8 的四类型 PHT 数据、PH/PT replay 和约束损失，但将
顺序 `PH → PHT` 交互替换为共享的两层 Hyper-PMT 融合编码器。

## 模型

```text
ESM-C / TCR-BERT residue states
        ↓
Peptide / HLA / TCR adapters
        ↓
shared 2-layer Hyper-PMT fusion
  static + cross-entity dynamic attention
  relation bias + [S,D,S×D,(S−D)²] compatibility
        ↓
PH / PT / PHT edge pooling and task heads
```

PH、PT 和 PHT 使用同一个融合模块；dynamic attention 的 KV 只来自其它实体，
不会把同一实体的 token 作为自己的上下文。PHT 额外使用一个小权重、可学习
的 Stage1 PH prior 残差，但三元超图仍是主路径。

## 运行

```bash
./run_stage14.sh audit
./run_stage14.sh warmup --stage1-checkpoint \
  trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260831_174042_678934.pt
./run_stage14.sh ablations --stage1-checkpoint \
  trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260831_174042_678934.pt
```

`ablations` 默认使用一份 Stage14A warmup 并启动 `baseline`、`full` 两个
Stage14B 条件。baseline 不启用一致性约束；full 同时启用 hierarchy 和
PT implication 约束。资源有限时增加 `--sequential`。

恢复与评测：

```bash
./run_stage14.sh resume trained_model/PLM_Stage14/fold_1/seed_42/full/stage14_last_TIMESTAMP.pt
./run_stage14.sh evaluate --checkpoint trained_model/PLM_Stage14/fold_1/seed_42/full/stage14_best_TIMESTAMP.pt
./run_stage14.sh benchmarks --checkpoint <stage14_best.pt> \
  --output-json tests/stage14_benchmark.json \
  --output-markdown tests/PLM_Stage14_benchmark.md
```

Stage14 checkpoint 类型为 `stage14_online_model`，不接受 Stage8 V2 checkpoint。
warmup、best、last 均保存完整 online model，并验证 Stage1 SHA、PLM fingerprint、
数据指纹与训练合同。
