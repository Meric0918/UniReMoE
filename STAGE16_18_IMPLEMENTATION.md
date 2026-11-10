# Stage16–18：实现、运行与验收

对应计划：[Stage16](PLM_STAGE16.md)、[Stage17](PLM_STAGE17.md)、[Stage18](PLM_STAGE18.md)。本次完成代码实现和工程验证，不包含正式35轮训练、完整消融、多种子实验或新的性能提升结论。

## 1. 三个可运行入口

| 阶段 | 配置 | 训练入口 | 默认行为 |
|---|---|---|---|
| 16 | [plm_stage16.yaml](configs/plm_stage16.yaml) | [train_plm_stage16.py](source/train_plm_stage16.py) | 原三元训练，同轨迹运行legacy/独立PT/guarded三个选择器 |
| 17 | [plm_stage17.yaml](configs/plm_stage17.yaml) | [train_plm_stage17.py](source/train_plm_stage17.py) | S17-J：PLM冻结，joint只更新PHT模块 |
| 18 | [plm_stage18.yaml](configs/plm_stage18.yaml) | [train_plm_stage18.py](source/train_plm_stage18.py) | S18-R3：PT比例1.0、group权重、smoothing0.03、KL alpha0.5/T2 |

默认35轮训练、额外round0完整验证；scheduler horizon保持100轮。三个主实验都从同一个warmup开始。Stage18默认仍微调原来的PLM顶部两层，只有COMB类条件继承Stage17的冻结。

共同实现集中在 `stage16_config/data/evaluation/selection/trainer/checkpoint/pipeline.py`。Stage17提供独立条件、参数权限和类型校验；Stage18提供独立条件、教师缓存、加权CE/KL。它们共享训练/选模基础设施，防止复制代码后比较规则漂移。

此前的 `train_plm_stage8_pt_retention.py` 是另一套组合原型，不应拿来替代这三个正式入口。已有Stage8源文件、checkpoint与CSV没有被改写。

## 2. 环境与审计

在项目根目录执行，使用已有 `unifyimmun` 环境；新增实现没有引入新的运行时依赖。

```bash
source /home/mjp/miniconda3/etc/profile.d/conda.sh
conda activate unifyimmun
export OMP_NUM_THREADS=2
export MKL_THREADING_LAYER=GNU

python source/train_plm_stage16.py --dry-run
python source/train_plm_stage17.py --dry-run
python source/train_plm_stage18.py --dry-run
```

`--audit-only` 与 `--dry-run` 均做完整train/validation审计、三个来源SHA及原始encoder/tokenizer fingerprint校验，写入数据/运行manifest，但不分配PLM或训练。manifest包含文件SHA、有效行ID、标签数、拒绝原因、规范化输入别名、投影标签冲突及跨任务holdout交集。原始数据不被修改。

PH/PT/PHT train有效数目前为291,052 / 664,952 / 72,190；PT三个validation为10,382 / 154,260 / 143,028。debug限制在完整审计之后应用。训练/参考缓存准备拒绝test路径；真正的跨任务实体暴露会显式记录，而不是掩盖为全局unseen。

## 3. 正式训练与消融

```bash
python source/train_plm_stage16.py
python source/train_plm_stage17.py --condition S17-J
python source/train_plm_stage18.py --condition S18-R3
```

三个命令是独立运行，不要求先训练完前一个阶段。正式实验建议按资源串行启动。

Stage17支持 `S17-U / S17-F / S17-J / S17-P / S17-Z`；Stage18支持 `S18-R0 / S18-R1 / S18-W / S18-S / S18-R2 / S18-R3 / S18-COMB / S18-COMB-P`。

```bash
python source/train_plm_stage17.py --condition S17-U
python source/train_plm_stage17.py --condition S17-F
python source/train_plm_stage18.py --condition S18-R1
python source/train_plm_stage18.py --condition S18-R2
python source/train_plm_stage18.py --condition S18-COMB
```

S17-J允许PH replay更新共享peptide Adapter；S17-P才额外冻结它。COMB对应J而不是P；COMB-P为单列条件。所有条件保持 `PH → PT → joint`。Z没有二元更新，是保持不变量的诊断，不是公平同预算的主对照。

Stage18剂量探索通过新配置的 `policy_overrides` 明确记录，例如：

```yaml
policy_overrides:
  pt_replay_ratio: 2.0
  distillation_weight: 0.1
  temperature: 2.0
```

其余可覆盖项为 `label_smoothing`、`group_weighting`；Stage16/17拒绝这些额外干预。标准对照保持空overrides。`training.sampler_policy: legacy_order` 只让joint重复旧采样顺序，其他工程规则不变；它不是对历史Stage8全部实现的逐bit复现。

等更新数辅助实验需单独配置：R0的rounds=30，r=1.0条件rounds=20，两者 `training.scheduler_total_steps: 50760`。主实验仍是35轮、相同PHT暴露量；不可把更多计算的效果全部归因于KD。

## 4. 教师预计算与数值约定

```bash
python source/precompute_plm_stage18_teacher.py --role both
```

支持 `--role train / reference / both`。训练入口发现所需缓存缺失时也会自动生成；显式预计算更便于分离一次性成本。

缓存只使用Stage1教师和原始冻结PLM，按唯一peptide/TCR编码，然后按行计算原始两类logits。train和reference分目录、分角色。缓存身份绑定teacher/warmup SHA、encoder/tokenizer fingerprint、实际CPU/CUDA与精度、数据SHA、row ID和规范化输入key；sidecar保存字节SHA和抽查记录。每次加载验证identity、字节SHA、shape、finite和row key，拒绝缺失/错序/改动。

默认抽查最多1,024行，以原始教师在线前向复核缓存。小于1,024行的debug集全部抽查。缓存被修改或参数/精度改变时，不能继续原run；需要重新登记实验。学生在缓存准备后重置到固定训练RNG，缓存命中和未命中不会改变后续dropout/训练轨迹。

PT loss采用训练集 `(peptide,label)` inverse-sqrt权重、group均值归一化并clip到[0.25,4]。CE按完整有效累积窗口的权重总和归一化；KD按该窗口的样本数归一化，包含T²，教师detach。主分类与评测均使用T=1、FP32 softmax、严格 `probability > 0.5`，ECE也采用相同tie规则。

Stage16/18学生的验证特征不能跨参数更新复用。最终推理只在同一个锁定checkpoint内缓存residue特征；没有把微调后的学生hidden喂给Stage1教师。

## 5. 小规模接线测试与断点恢复

```bash
python source/train_plm_stage17.py \
  --rounds 1 --debug-pair-limit 8 --debug-limit-per-type 2 \
  --output-dir tests/stage17/my_smoke \
  --stop-after-steps 1

python source/train_plm_stage17.py \
  --resume tests/stage17/my_smoke/last.pt
```

输出目录必须不存在或为空，避免覆盖旧run。`--stop-after-steps`是绝对optimizer步数，保存后正常返回 `checkpointed_stop`。SIGINT/SIGTERM也请求在当前累积窗口结束后保存。正式训练默认每250个optimizer steps保存last，round结束也保存。

last包含完整学生（包括微调encoder）、optimizer/scheduler/scaler、RNG、round/pass/batch游标、累计PT覆盖、监控状态、各selector最佳artifact与SHA、固定参考和teacher缓存身份。采样使用独立生成器及可重建的epoch排列，恢复时跳过已提交窗口，不重复反向传播。若进程被强杀，最多回退到最后一个成功提交的last。

恢复只接收同阶段training checkpoint；科学参数、数据、教师缓存和best companion身份必须一致。恢复使用内嵌配置，不依赖之后改动的YAML，也不允许切换CPU/CUDA类别来冒充精确续训。完成或已满足早停的run不会多训练一轮。恢复会重建日志和best索引，意外未提交的索引另存为 `uncommitted_*`，不暴露成有效best。

debug模型不能锁定进入最终test，也不能冒充正式模型。这里的小规模验证与35轮正式实验是不同层次的证据。

## 6. 选模与输出

每个run位于 `trained_model/PLM_StageNN/fold_<fold>/seed_<seed>/<condition>/<timestamp>/`，或者显式 `--output-dir`。

| 产物 | 内容 |
|---|---|
| `data_manifest.json` / `run_manifest.json` | 数据与来源、配置/策略、实现文件SHA、环境、耗时/显存与状态 |
| `reference_metrics.json` | 固定Stage1 PT参考和warmup PHT参考 |
| `initial.pt` | 完整起始模型 |
| `selected_round_NNN.pt` | 当轮被任一selector选中的完整推理模型，不含teacher |
| `best_legacy.json` / `best_pt_validation.json` / `best_guarded.json` | 指向不可变模型artifact的SHA索引；无可行解时guarded索引不存在 |
| `last.pt` | 可恢复训练状态 |
| `round_metrics.jsonl` / `selection_decisions.jsonl` | 全split指标、三个selector、gate失败原因、最差PT差值 |
| `pass_metrics.jsonl` / `monitor_metrics.jsonl` / `training_steps.jsonl` | 样本/steps、分项loss、梯度诊断、逐pass概率变化和参数漂移 |
| `group_weights.json` / `exposure_stats.jsonl` / `teacher_cache_manifest.json` | 分组权重、真实暴露及唯一覆盖、teacher身份 |
| `validation_predictions_round_NNN.pt` | 同一轨迹的逐行预测，可用于配对统计 |
| `training_summary.md` | 选择结果与是否存在可行checkpoint |

三个selector分别使用旧PHT来源分数、独立PT加权分数、同分数加保留门槛。guard采用PT AUROC/AUPR/MCC宏平均最多下降0.01、单split最多0.02、PHT S最多下降0.02。门槛判断前校验样本/正负数及有限指标。round0可以最优，但会明确记录；全程无可行解不会自动放宽门槛或用诊断best替代。

这些门槛保证的是定义的经验保留标准，不是统计显著性或优于Stage1的证明。

## 7. 锁定模型后最终评测

待三个阶段需要比较的候选及控制全部固定，再创建一次选择清单；以下路径中的 `RUN16/RUN17/RUN18` 应替换为实际目录。

```bash
python source/lock_plm_stage16_18_benchmarks.py \
  --checkpoints RUN16/best_guarded.json RUN17/best_guarded.json RUN18/best_guarded.json \
  --output tests/stage16_18_final_selection.json

python source/evaluate_plm_stage16_benchmarks.py \
  --checkpoint RUN16/best_guarded.json \
  --selection-manifest tests/stage16_18_final_selection.json \
  --output-prefix tests/stage16/final

python source/evaluate_plm_stage17_benchmarks.py \
  --checkpoint RUN17/best_guarded.json \
  --selection-manifest tests/stage16_18_final_selection.json \
  --output-prefix tests/stage17/final

python source/evaluate_plm_stage18_benchmarks.py \
  --checkpoint RUN18/best_guarded.json \
  --selection-manifest tests/stage16_18_final_selection.json \
  --output-prefix tests/stage18/final
```

可把legacy/独立PT selector或initial控制一起列入清单；无guarded候选时明确报告失败，而非强行提供不存在的路径。锁文件不可覆盖，改选模型需新清单。评测器先验证锁、模型类型、内嵌配置与SHA，再读取 `PH/new`、`PT/new`、`PHT/benchmarks` 的test文件。它重建完整学生encoder state，不需要teacher，不在test选择阈值。

输出JSON、Markdown与 `.predictions.pt`；可设置 `--encode-batch-size`、`--row-batch-size` 和 `--device`。CPU推理使用FP32；报告记录实际设备/精度，不能把不同数值协议的结果当作严格逐bit复现。

## 8. 消融统计与其他seed

```bash
python source/analyze_plm_stage16_18.py \
  --runs RUN16 RUN17 RUN18 --output-prefix tests/stage16_18_ablation_summary

python source/analyze_plm_stage16_18.py \
  --baseline-predictions tests/stage16/final.predictions.pt \
  --candidate-predictions tests/stage18/final.predictions.pt \
  --bootstrap 2000 --output-prefix tests/stage16_18_paired
```

消融汇总包含完整逐split指标、选中轮和可行比例。配对bootstrap校验行ID、输入key和标签，按peptide簇重采样；unseen_tcr另外提供TCR簇敏感性分析。单标签重采样不会伪造AUROC，报告有效重复数。该区间不替代独立种子重复或新外部测试集。

其他seed先准备该seed的Stage1 joint/PT专家路径与SHA配置，再生成共同warmup：

```bash
python source/create_plm_stage16_18_warmup.py \
  --config configs/YOUR_SAME_SEED_CONFIG.yaml \
  --output trained_model/YOUR_COMMON_WARMUP.pt
```

生成器训练两轮、只更新PHT新模块，以PHT validation选best，保留二元和原始PLM权重；输出带实际warmup SHA的 `.ready_config.json`，可直接作为新训练入口的 `--config`。之后为三个stage分别设置stage/condition/output_root，复用同一个新warmup。配置中的旧warmup字段在生成阶段不作来源使用；生成前须提供正确同seed的joint和teacher，不能修改seed数字后复用seed42来源。

## 9. 已验证内容与证据

本轮最终结果：30项新增测试与8项原型回归测试通过，共38项；10个CLI的启动/帮助接口通过检查。需求到实现的对应关系见 [工程验收记录](tests/stage16_18_implementation_verification.md)。

CPU测试覆盖配置/数据角色、三个selector、非有限数/计数/无可行解、eval状态恢复、连续与中断训练一致、普通及加权CE/KD的完整窗口等价、条件权限矩阵、带历史动量的joint隔离、零解冻层、Z保持、KL方向/温度/teacher无梯度、缓存身份/字节校验、缓存命中与否不改变学生轨迹、推理缓存与在线输出、锁定后的临时benchmark端到端评测，以及共同warmup生成。

运行回归测试：

```bash
python -m unittest discover -s tests -p 'test_plm_stage1*.py' -v
python -m unittest discover -s tests -p 'test_stage8_pt_retention.py' -v
```

真实PLM接线验证使用审计后的debug数据（每个PT split 8行、PHT每类型2行），没有读取正式test。以下是实施过程保留的证据，均不是可报告性能的训练实验：

| 验证 | 证据 |
|---|---|
| Stage16完整PH/PT/joint轮与梯度诊断 | [运行记录](tests/stage16/implementation_smoke_20260925/run_manifest.json) |
| Stage17窗口边界停止后重新进程恢复，完成剩余轮 | [运行记录](tests/stage17/implementation_smoke_v2_20260925/run_manifest.json) |
| Stage18原始教师缓存 + R3微调 + KL完整轮 | [运行记录](tests/stage18/implementation_smoke_20260925/run_manifest.json) |
| Stage18 COMB冻结/隔离 + KL完整轮 | [运行记录](tests/stage18/implementation_comb_smoke_20260925/run_manifest.json) |
| Stage18当前schema3缓存、PT更新后保存、重新进程恢复并完成joint | [最终运行记录](tests/stage18/implementation_final_smoke_20260925/run_manifest.json) |

这些记录来自开发中的对应实现版本；当前缓存schema为3，校验包含实际计算设备。旧debug缓存/断点不应改写身份强行迁入新协议，正式实验应新建run。首个Stage17接线尝试发现Transformer Engine的`_extra_state`是BytesIO而非权重张量，已将运行时元数据与需要逐tensor校验的权重区分，并完成后续验证。

完整35轮、全部消融、完整teacher train预计算、多seed与最终真实benchmark均尚未运行。本次验证证明了代码链路和关键不变量，没有证明PT性能已经改善。
