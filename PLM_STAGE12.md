# PLM Stage12：Stage8-aligned constrained retraining

Stage12 与 Stage8 位于同一层级，二者都从固定的 Stage1 `best_joint` 初始化。Stage8 full 只作为验证集参考，不作为初始化、Teacher 或蒸馏目标。

```text
Stage1 best_joint
├── Stage8 full
└── Stage12 full（Stage8-aligned）
```

## 固定训练合同

- fold/seed：`fold_1 / seed_42`；仅运行一个 `full` 条件。
- 模型：完全复用 Stage8 的 `ProgressiveStage8FineTuningModel`，不增加参数或分支。
- Stage12A：PHT-only warmup 2 epochs，不启用约束。
- Stage12B：每轮按 `PH replay → PT replay → PHT joint` 串行训练。
- joint loss：PH=`0.3`、PT=`0.2`、PHT=`1.0`，与 Stage8 一致。单任务 replay 保持未缩放交叉熵。
- replay：PH 和 PT 均为 PHT train 的 25%，使用与 Stage8 相同的普通随机 DataLoader；训练器在每轮对 loader 做样本数截断。
- 约束：hierarchy lambda=`0.05`、PT implication lambda=`0.03`、margin=`0.05`、5 rounds ramp。
- 其余设置沿用 Stage8：top-2 PLM layers、micro batch 4、gradient accumulation 16、bf16、gradient checkpointing、min/max rounds 5/15、patience 5。

## 验证与选模

每个 Stage12B round 都使用固定阈值 0.5 评估并写入同一个 SwanLab run：

- PT：`val_unseen_pair`、`val_unseen_peptide`、`val_unseen_tcr` 的等权 macro；
- PH：`val_unseen_hla`、`val_unseen_pair`、`val_unseen_peptide` 的等权 macro；
- PHT：Stage8 原四类型 validation joint。
- 总体选模：`PHT joint`、`PH macro`、`PT macro` 三项等权平均。
- PHT 内部 head 选模：`pht`、`phla`、`ptcr` 三项等权平均。

Stage12B 只按总体等权 joint score 进行 best checkpoint 选择和早停，不使用相对 Stage8 的 retention gate，也不要求 PT macro 单独提升。

## 运行

先执行严格数据审计：

```bash
./run_stage12.sh audit
```

前台训练：

```bash
./run_stage12.sh train
```

tmux 后台训练：

```bash
tmux new-session -d -s stage12 \
  "cd /home/mjp/Project/PLM/UnifyImmun && ./run_stage12.sh train 2>&1 | tee training_logs/stage12_$(date +%Y%m%d_%H%M%S).log"
```

恢复训练：

```bash
./run_stage12.sh resume trained_model/PLM_Stage12/fold_1/seed_42/full/stage12_last_TIMESTAMP.pt
```

训练完成并存在 `stage12_best` 后运行一次 test：

```bash
./run_stage12.sh evaluate \
  --checkpoint trained_model/PLM_Stage12/fold_1/seed_42/full/stage12_best_TIMESTAMP.pt \
  --output-json tests/stage12_benchmark_results_TIMESTAMP.json \
  --output-markdown tests/PLM_Stage12_PH_PT_PHT_benchmark_test_results_TIMESTAMP.md \
  --device cuda
```

## 产物

- `stage12_warmup_TIMESTAMP.pt`：Stage12A 完整在线模型；
- `stage12_best_TIMESTAMP.pt`：按总体等权 joint score 选出的最佳模型；
- `stage12_last_TIMESTAMP.pt`：包含 optimizer、scheduler、scaler 和 RNG 的恢复点；
- `summary_TIMESTAMP.json`：最终状态、全部谱系和 checkpoint 路径。

所有 checkpoint 都记录 Stage1 来源 SHA、Stage8 reference SHA、训练/验证数据指纹和完整 Stage12 训练合同。
