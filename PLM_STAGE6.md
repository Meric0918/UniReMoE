# PLM Stage 6：PH覆盖率与多任务冲突消融

Stage 6 是独立于 Stage 5 的诊断实验。它不修改、不迁移也不覆盖任何
Stage 5 配置、源码、checkpoint或SwanLab日志，只读取现有Stage 5结果作为
`stage5_joint_matched`基线。

## 1. 实验设计

固定fold 4、seed 2026、随机下游初始化、冻结基础PLM、45,533个逻辑步/epoch、
5个1A epoch和10个1B epoch。主指标为PH
`task_score=(AUROC+AUPR)/2`。

| 条件 | 活跃任务 | 每个epoch的PH覆盖 |
|---|---|---:|
| `stage5_joint_matched` | PH+PT+PHT | 182,132 / 803,048 |
| `stage6_phla_only_matched` | PH | 182,132 / 803,048 |
| `stage6_joint_phla_full` | PH+PT+PHT | 803,048 / 803,048 |
| `stage6_phla_only_full` | PH | 803,048 / 803,048 |

Stage 5基线不重训。三个Stage 6条件都使用相同逻辑步数和梯度累积设置。
full条件把200,762个PH batch精确分配到45,533个逻辑步：18,630步各5个，
26,903步各4个。同一逻辑步的PH样本先合并为一次样本均值，再乘固定权重0.3，
不会因为放入多个PH batch而放大loss。

joint仍使用：

```text
L = 0.3 L_PH + 0.2 L_PT + 1.0 L_PHT
```

PH-only仍使用`L=0.3 L_PH`以保持梯度尺度。它只更新Peptide/HLA Adapter、
PH Cross-Attention、Pooling和Classifier；PT/PHT模块、TCR Adapter和三个PLM
保持冻结。1B的FGM在joint中攻击三个Adapter，在PH-only中只攻击Peptide/HLA
Adapter。

## 2. 训练

依次执行三个新条件：

```bash
cd /home/mjp/Project/PLM/UnifyImmun
./run_stage6.sh all
```

也可以单独运行一个条件：

```bash
./run_stage6.sh stage6_joint_phla_full
```

调试时必须同时缩小逻辑步数，防止误跑45,533步：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage6.py \
  --condition stage6_phla_only_full \
  --limit 64 \
  --logical-steps 4 \
  --num-workers 0 \
  --device cuda \
  --swanlab-mode disabled
```

每个条件写入独立目录：

```text
trained_model/PLM_Stage6/<condition>/fold_4/seed_2026/
```

产物名为：

```text
stage6_1a_complete_<timestamp>.pt
stage6_1b_best_<timestamp>.pt
stage6_last_<timestamp>.pt
stage6_best_<timestamp>.pt
```

可用`--resume`恢复同一条件的`stage6_training` checkpoint。Stage 5 checkpoint
不能作为Stage 6训练恢复点。

## 3. PH评估

对现有Stage 5最终checkpoint和三个Stage 6最终checkpoint分别执行：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage6_phla.py \
  --checkpoint \
    trained_model/PLM_Stage5/fold_4/seed_2026/stage5_last_20260805_205301_895002.pt \
  --split all \
  --fold 4 \
  --seed 2026 \
  --device cuda \
  --joint-val-metrics \
  --output-dir trained_model/PLM_Stage6/evaluations/stage5_joint_matched
```

将`--checkpoint`替换为各Stage 6的`stage6_last_*.pt`。对
`stage6_joint_phla_full`也添加`--joint-val-metrics`，用于检查PT/PHT保持情况。

validation保持原始fold划分。independent和external默认执行以下清洗：

1. 删除与`train_fold_4.csv`重叠的peptide-HLA pair；
2. 删除在评估集内部具有冲突标签的整个pair；
3. 相同标签的重复pair仅保留第一条。

JSON会记录清洗审计，CSV保存逐样本概率和稳定`sample_id`。四个条件必须使用
相同清洗设置，分析脚本会拒绝样本或标签未对齐的输入。

## 4. 配对Bootstrap分析

每个split单独运行，四个`--prediction`参数分别指向对应预测CSV：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/analyze_plm_stage6.py \
  --split independent \
  --bootstrap-samples 2000 \
  --seed 2026 \
  --prediction stage5_joint_matched=/path/stage5_independent.csv \
  --prediction stage6_phla_only_matched=/path/phla_only_matched_independent.csv \
  --prediction stage6_joint_phla_full=/path/joint_full_independent.csv \
  --prediction stage6_phla_only_full=/path/phla_only_full_independent.csv \
  --output-dir trained_model/PLM_Stage6/analysis
```

在validation分析中可额外传入Stage 5和joint-full评估JSON：

```text
--stage5-evaluation <stage5_val_evaluation.json>
--joint-full-evaluation <joint_full_val_evaluation.json>
```

分析输出覆盖率效应、任务冲突效应和交互效应的AUROC、AUPR、task score差值及
95%置信区间。task score提升至少0.01且置信区间下界大于0记为明确改善；
0.005–0.01记为趋势性改善；绝对值小于0.005记为影响有限。joint-full的
PT/PHT task score下降不超过0.005可接受，超过0.01不可接受。

## 5. 测试

```bash
cd /home/mjp/Project/PLM/UnifyImmun
PYTHONPATH=source:. /home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage6
```

测试覆盖PH batch精确分配、逻辑步一致、样本均值loss、PH-only参数隔离、FGM
目标、Stage 6路径命名、外部集清洗以及配对bootstrap复现性。
