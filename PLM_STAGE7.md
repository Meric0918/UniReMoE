# PLM Stage 7：Stage 3 与 Stage 1 的 HLA 条件增益评估

Stage 7 是独立的只读评估阶段，用于检验 Stage 3 三元模型加入 HLA 后，
是否比 Stage 1 的 peptide–TCR 二元模型获得更好的识别性能。它不训练模型、
不创建 checkpoint、不使用 SwanLab，也不修改 Stage 1、Stage 3 或 Stage 5
的配置、源码和产物。

Stage 7 读取同 fold/seed 的一个 `stage3_multitask_model` checkpoint，以及
Stage 1 的 pHLA、pTCR 两个 `task_model` checkpoint。Stage 3 在训练中微调过
三个基础 PLM 的顶部层，因此评估会加载 checkpoint 中的 Peptide、HLA 和 TCR
encoder 权重；Stage 1 仍使用原始冻结 PLM。两者不会错误地共享 residue
embedding。为保证比较具有明确继承关系，Stage 3 checkpoint 必须记录为由同
fold/seed 的 Stage 1 初始化。

## 1. 主要性能实验

在严格清理后的原始 triplet validation/test 标签上比较：

- `Stage1 pTCR`：不输入 HLA，是主要二元基线；
- `Stage1 late fusion`：validation 标准化后的 pHLA+pTCR 可加基线；
- `Stage3 pTCR`：同一个 Stage 3 checkpoint 中移除 HLA 路径的消融；
- `Stage3 late fusion`：Stage 3 二元分支的可加基线；
- `Stage3 PHT`：完整 HLA–peptide–TCR 三元分支。

AUROC、AUPR 是主要阈值无关指标。每个模型的分类阈值只在 triplet validation
上按 MCC 选择，然后原样用于 test。报告同时给出全局指标，以及按 peptide
等权的 accuracy gain bootstrap 95% CI。

只有当 Stage3 PHT 相对 Stage1 pTCR 的 AUROC、AUPR 均提高，且 peptide-macro
accuracy gain 的 95% CI 下界大于 0，才判定 Stage 3 优于 Stage 1。相对同一
Stage 3 checkpoint 的 pTCR 消融也满足该条件，才能更有力地把收益归因于三元
分支中的 HLA 条件化。

## 2. HLA 使用证据

性能提高本身可能来自目标任务训练或模型容量，因此同时执行三项 HLA
检验：

1. 固定阳性 peptide–TCR，把原始 HLA 与最多 5 个 pHLA-valid 错配 HLA 排序；
2. 对同一 peptide 构造 2×2 TCR–HLA tetrad，计算非加性交互：

   ```text
   D = (m(p,t1,h1) + m(p,t2,h2) - m(p,t1,h2) - m(p,t2,h1)) / 2
   ```

3. 复现确定性的 HLA 循环位移，观察 Stage3 PHT 性能下降。

Stage 1/3 的可加 pHLA+pTCR 分数在 tetrad 中理论上 `D=0`。Stage3 PHT 只有在
peptide-macro D 的 95% CI 下界大于 0、单侧 sign-flip `p<0.05`，且相对 Stage3
late fusion 的差值 CI 下界大于 0 时，才支持正向非加性 TCR×HLA 相容性。

合成 HLA decoy 是未观测组合，不是实验确认的阴性三元组。Stage3 还接受了
Stage1 没有接受的 PHT 监督，因此 Stage3–Stage1 差异同时包含 HLA 条件化、
三元架构、PLM 微调和目标任务训练效应；Stage 7 会在结论中保留这一限制。

## 3. 运行

当前 fold 4 / seed 2026：

```bash
cd /home/mjp/Project/PLM/UnifyImmun
./run_stage7.sh
```

也可以显式指定 checkpoint：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage7.py \
  --config configs/plm_stage7.yaml \
  --stage3-checkpoint \
    trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt \
  --stage1-phla-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_2026/best_phla.pt \
  --stage1-ptcr-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_2026/best_ptcr.pt \
  --device cuda
```

调试时可限制派生数据；原始三元组分类集也会同步限制：

```bash
./run_stage7.sh --limit-pairs 64 --limit-tetrads 16
```

## 4. 产物与多 seed

新版 manifest 使用 `format_version=2` 和
`comparison=stage3_vs_stage1`。旧版 Stage5-vs-Stage1 结果不会被聚合器误读。

```text
trained_model/PLM_Stage7/fold_<fold>/seed_<seed>/<timestamp>/
  stage7_manifest.json
  stage7_data_audit.json
  stage7_counterfactual_val.csv
  stage7_counterfactual_test.csv
  stage7_summary.json
  stage7_report.md
```

多个相同 fold、不同 seed 且派生行严格对齐的运行可聚合：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/analyze_plm_stage7.py \
  --run-dir <seed_1_stage7_run> \
  --run-dir <seed_2_stage7_run> \
  --run-dir <seed_3_stage7_run>
```

少于 3 个运行时，结论始终标记为 `preliminary=true`。

## 5. 测试

```bash
cd /home/mjp/Project/PLM/UnifyImmun
PYTHONPATH=source:. /home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage7 \
  tests.test_plm_triplet
```
