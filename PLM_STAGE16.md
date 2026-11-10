# Stage 16 计划：独立 PT 验证、保留门槛与 checkpoint 选择

日期：2026-09-25。状态：代码已实现，并完成 CPU 回归测试与真实 PLM 小规模验证；35轮正式实验尚未运行。运行方式与验收证据见 [Stage16–18 使用说明](STAGE16_18_IMPLEMENTATION.md)。

Stage16 对应第一优先项一：将独立 PT 三个 validation split 纳入模型选择，对宏平均和每个 split 设置退化门槛。核心问题是：在训练轨迹相同的条件下，改进选模能否找出兼顾 PT 与 PHT 的 checkpoint？本阶段不改变模型结构、训练数据、损失或回放量。

关联计划：[Stage17](PLM_STAGE17.md) 研究冻结与梯度隔离；[Stage18](PLM_STAGE18.md) 研究回放、正则和教师保持。三阶段主实验共享起点；后两者继承 Stage16 的评测协议，而不加载 Stage16/17 选中的权重继续训练。组合收益单独作为 `S18-COMB` 报告。

## 1. 证据与可证伪假设

证据来自 [PT 退化分析](tests/stage8/pt_retention_analysis_20260924/analysis.md)、[实际数据审计](tests/stage8/pt_retention_analysis_20260924/audit.json)、[当前运行配置](configs/plm_stage8_20260920_best_joint.yaml) 和 [online trainer](source/plm_unified/stage8_trainer.py)。旧的 [Stage8 总说明](PLM_STAGE8.md) 含较早的来源 checkpoint 和轮数，不能替代本轮实验的内嵌配置。

当前 Stage8 的选择分数为 `0.4*S(PHT)+0.2*S(PH)+0.4*S(PT)`，三个 head 全部在 PHT 来源的验证数据上计算。该验证集的 `000/111` 占约 98.32%，不能代表独立 PT 泛化。配置中的 `joint_degradation_tolerance` 不构成当前 online 入口的独立 PT 保留门槛。

假设 H16：同一训练轨迹中可能存在未被旧选择器选中的 checkpoint，能更好保持独立 PT 性能。若所有候选均无法满足保留门槛，则 H16 在当前轨迹上不成立，必须如实输出 `no_feasible_checkpoint`；这也是支持继续 Stage17/18 的有效结果。选择器不能创造训练轨迹上不存在的能力。

## 2. 三阶段共同实验契约

### 2.1 初始化与来源

首轮固定 fold 1 / seed 42。以下路径必须在运行前校验 SHA256，不能只按文件名识别：

| 角色 | 项目根目录下路径 | SHA256 |
|---|---|---|
| Stage1 联合来源，round 9 | `trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260920_151134_400917.pt` | `ea95033e9e228a9e8b514ee802bd27d1911b02b22447157473fcb477b9722d59` |
| 三阶段共同 warmup，选中的 epoch 1 | `trained_model/PLM_Stage8/fold_1/seed_42/warmup/stage8a_best_pht_20260921_111731_349210.pt` | `404d7aad11d17bb15480fea15380b290b7d8249b6c008f649b202fe5dc35d3a9` |
| PT 保留参考；Stage18 另作教师，round 10 | `trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_ptcr_20260920_151134_400917.pt` | `aac5a63d0de2789c21f11505915b635dfc687108d2c3e77fb26bb3c5abc5ea91` |

warmup 是两轮 Stage8A 尝试中被选中的 epoch 1，不应在日志中写成“加载 epoch 2 权重”。它的二元模块与 exact Stage1 joint 一致，但不等于 round 10 的 PT 专家。运行时重新验证，不通过则停止该实验。

所有学生都从共同 warmup 初始化。PT 参考模型只提供验证参考，不将其 peptide Adapter 拼接进学生；Stage18 教师也保持独立。多随机种子确认阶段采用 42/3407/2026，各自使用同 seed 的 Stage1 joint、PT 专家和重新生成的 warmup，并锁定来源清单；如果只有 seed42 warmup 可用，改变 replay seed 只能称“固定初始化的采样重复”，不能称独立训练种子实验。

### 2.2 数据角色与审计

| 用途 | 路径 | 参考有效样本量 |
|---|---|---:|
| PH replay | `dataset/PH/train.csv` | 291,052 |
| PT replay | `dataset/PT/train.csv` | 664,952 |
| PHT joint train | `dataset/PHT/classes/PHT_{000,010,100,111}/train.csv` 四文件 | 72,190 |
| PHT validation | 同四目录的 `val.csv`，经完整审计合并 | 32,055 |
| 独立 PT pair validation | `dataset/PT/benchmarks/val_unseen_pair.csv` | 10,382 |
| 独立 PT peptide validation | `dataset/PT/benchmarks/val_unseen_peptide.csv` | 154,260 |
| 独立 PT TCR validation | `dataset/PT/benchmarks/val_unseen_tcr.csv` | 143,028 |
| 最终 PT test | `dataset/PT/new/test_*.csv` | 263,610 合计 |
| 最终 PH / PHT test | `dataset/PH/new/test_*.csv`、`dataset/PHT/benchmarks/test_*.csv` | 由最终审计记录 |

花括号路径是四个明确文件的简写，正式 manifest 必须展开，禁止把 glob 顺序当稳定 ID。训练和选择器不得读取 `test_*`。不因为已观察到 PT/new 的表现而再次挑选要删除的 peptide。三阶段共用最终test解锁流程：待需要比较的主配置、选择规则和checkpoint清单全部锁定后再批量评测，不能先观察Stage16新test结果再据此调整Stage17/18。

运行前重做序列规范化、长度、非法字符、三元标签类型、重复、冲突、训练/验证交集审计。当前 PHT 指纹为 `aaeb3ce53d879a54915da4e50c9e414b63b98e6d7d5915263c63a89b0d144920`；三个 PT validation 与 PHT train 的 exact PT pair 无交集。后续还要检查 PH/PT/PHT 全部训练输入对各自保留实体的交集、截断后输入碰撞；新增过滤必须版本化，并让所有比较条件使用同一 manifest。

每个文件保存 SHA256、原始/有效行数、标签计数、行 ID、序列规则和拒绝原因。PHT 的三元去重不代替 PT 投影去重审计。发现问题时报告而不修改原始 CSV。

### 2.3 网络与训练预算

统一维度：peptide/HLA 输入 960、TCR 输入 768，Adapter/model dim 256，attention heads 8，FFN 1024，dropout 0.15/0.15/0.25；长度 peptide 8–15、HLA/TCR 最长 34。PT 推理仍只传 peptide/TCR。

统一主条件为原 Stage8 `full`。Stage16 及 Stage18 主条件均解冻三种 PLM 顶部各两层；Stage17 的冻结是被研究的干预。共同基线的 PH/PT replay 各 `floor(72190*0.25)=18047`，joint 平衡采样 72,192 次，microbatch 4、累积 16，有效 batch 64；顺序保持 `PH → PT → joint`。

主比较固定训练35个StageB rounds，每轮完整验证，另加训练前round0，共36个验证时点。warmup已完成，不在35轮中重复计算。沿用历史配置的 scheduler horizon 100 rounds、warmup ratio 0.05，避免把缩短日程偷偷混入改进；达到35轮时学习率无需衰减到零。Stage18 按各自每轮计划 steps 将同一 horizon 映射到不同的总步数，记录映射公式。预算扩展到100轮必须作为另一个预先登记的实验。

先跑1轮接线检查，再跑10轮 pilot，最后35轮主比较；pilot 与主比较使用同一 scheduler horizon。pilot 只能排查失败与资源，不用其 test 指标选配置。主比较不因某一个选择器早停而截断其他选择器的候选；独立的部署早停实验另行报告。

下游 LR：Adapter 2.5e-5、已有 attention/pooling/classifier 5e-5、新三元模块 1e-4；在线 PLM 顶层5e-6、次顶层2.5e-6。weight decay 0.01，clip norm 1，bf16 autocast，约束 margin 0.05，hierarchy 0.05、PT implication 0.03，前5轮 ramp。没有研究到的参数保持基线一致。

### 2.4 比较中的工程修正

共同的 `controlled_v1` runner 要保证评估不改变训练 RNG、模式和 sampler 状态，checkpoint 可恢复、梯度累积按真实样本数加权，并在每轮调用 joint sampler 的 `set_epoch(round)`。三阶段及新基线都使用这些规则。为避免把 sampler 修正误认为 Stage16 收益，必须保留一个显式 `legacy_order` 配对控制，并记录共同修正与历史 Stage8 的差别。历史分数仅作参考，不能当成与新 runner 完全同轨迹的反事实。

本计划不包含训练顺序改为 `PH → joint → PT`、冻结 PH 的共享 Adapter、FGM、PCGrad 或全新结构；如需探索，使用独立条件名。

## 3. 验证输出与统一度量

每轮输出两个有明确数据来源的命名空间：`pht_val/{pht,phla,ptcr}` 与 `pt_independent/{val_unseen_pair,val_unseen_peptide,val_unseen_tcr}`。不要都命名为 `val/ptcr`。

每个完整 split 记录 AUROC、AUPR、Accuracy、MCC、F1、Precision、Recall、Specificity、NLL、Brier、ECE、正负数和混淆矩阵。AUPR 沿用项目 PR 曲线积分实现，不换成另一种 average precision 定义。阈值统一为严格 `probability > 0.5`，logits 先转 FP32 再 softmax，指标聚合使用 FP64；另外记录 `probability == 0.5` 的样本数。重算 Stage1/Stage8 参考时使用相同规则，历史报表中不一致的部分单独标注。

定义 `S=(AUROC+AUPR+Accuracy+MCC+F1)/5`；`M_PT` 是三个独立 PT split 的 S 等权平均。样本拼接后的 pooled 指标可以附录提供，不能替代宏平均。逐 peptide group macro 作为次要诊断；没有双类的 group 不计算其 AUROC，并报告覆盖率。

PHT 继续提供按四类型及 split_type 的诊断；其中 `100` 验证样本只有4条，不作为稳定单类型选模指标。独立 PH validation 可在 round0、候选best和最终轮报告以监测副作用，不进入当前主选择器。

独立 PT 全验证约30.8万条，必须在 round0及每轮结束完成。固定小监控子集只用于训练过程内的逐 pass 诊断，不能决定 best。其行ID在训练前锁定，禁止按预测错误重新抽样。

## 4. 三个并行选择器与硬门槛

同一训练进程、同一组每轮预测，同时运行：

| 条件ID | 分数 | 是否受保留门槛限制 |
|---|---|---|
| S16-L | 旧公式：`0.4*S(PHT_pht)+0.2*S(PHT_phla)+0.4*S(PHT_ptcr)` | 否，作旧规则控制 |
| S16-V | `J16=0.6*M_PT+0.3*S(PHT_pht)+0.1*S(PHT_phla)` | 否，用于分离验证分布替换效果 |
| S16-G，主方案 | 与 S16-V 相同 | 是，见下式 |

0.1 的 PH 项仍来自 PHT validation，必须注明，不能称独立 PH 泛化。分数中 PT 的替换和权重变化是选模策略的一部分；若需拆分二者，可另加保留原0.4/0.2/0.4权重的独立PT替换选择器，不再增加训练。

独立参考 `R_PT` 来自冻结的 exact Stage1 PT 专家，使用它自己的原始 PLM 编码；不能用正在微调的学生 encoder 给参考模型提取特征。参考预测可预计算一次，但缓存必须绑定模型、编码器、精度、数据行ID和SHA。学生 round0 PT 分数另存，仅用于区分初始化差异。

PHT 参考 `R_PHT` 来自共同 warmup 的 round0 PHT validation。历史 S 约0.933961，仅用于核对；门槛使用统一评估器重算后的值。

候选必须同时满足：

```text
对 m ∈ {AUROC, AUPR, MCC}：
  macro_PT(candidate,m) >= macro_PT(R_PT,m) - 0.01
  对每个 PT split：candidate(split,m) >= R_PT(split,m) - 0.02
S(PHT_pht,candidate) >= S(R_PHT) - 0.02
所有 gate 使用的指标和 J16 均为有限数；样本量、标签数与 manifest 一致
```

门槛是预设的经验容忍度，不是统计显著性检验。第一阶段不自动加入 NLL/ECE gate，以免同时改变可行域；记录它们，后续以单独协议试验。test、PHT验证阈值优化结果均不用于上述 gate。

best 更新要求通过 gate 且 `J16 > best_J16 + 1e-4`；差值不超过1e-4时保留更早轮次。round0可被选中，但报告应明确“未训练基线仍最优”。无候选通过时不写 `best_guarded`，照常保存 `last`、诊断best、每个失败条件；不把初始模型静默标为通过，也不自动放宽容忍度。

主比较只记录 shadow early-stop 状态，不提前结束。部署扩展模式可以采用 min_rounds=5、patience=20、max_rounds=100；无可行解时失败轮也消耗patience，重启后不得重置。

## 5. 实现拆解与文件职责

以下文件/接口均已实现；本节保留原设计职责，实际命令见使用说明。

| 文件 | 责任 |
|---|---|
| `configs/plm_stage16.yaml` | 来源、数据清单、预算、三个selector及gate，拒绝未知字段 |
| `source/plm_unified/stage16_config.py` | 独立配置类；解除对 Stage8 固定数值的错误继承，保留数据/类型验证 |
| `source/plm_unified/stage16_evaluation.py` | 独立 PT 完整评估、FP32概率、来源命名空间、参考缓存 |
| `source/plm_unified/stage16_selection.py` | 无副作用纯函数，分数/门槛/失败原因/同分规则 |
| `source/plm_unified/stage16_trainer.py` | 复用 Stage8 前向和损失；共同runner、shadow selectors、预算与状态管理 |
| `source/plm_unified/stage16_checkpoint.py` | 来源校验、原子保存、可恢复状态、部署类型 |
| `source/train_plm_stage16.py`、`source/evaluate_plm_stage16_benchmarks.py` | 审计/训练/恢复与锁定模型最终评测 |
| `tests/test_plm_stage16.py` | gate、轨迹不变性、累积、评估状态恢复、resume一致性 |

优先复用现有 [metrics](source/plm_unified/metrics.py)、[数据审计](source/plm_unified/stage8_data.py)、[参考原型](source/plm_unified/stage8_retention.py) 的独立函数；不能直接继承原型整套训练行为，因为它同时启用了冻结、KD、加大replay和改变顺序。

Stage8 的 config/trainer 当前把 replay=0.25、unfrozen_layers=2 等写入硬契约。新阶段必须拥有显式配置与checkpoint类型，不能只改YAML或运行时偷偷篡改旧config后仍保存成Stage8 V2。

## 6. checkpoint、记录与恢复

输出目录拟为 `trained_model/PLM_Stage16/fold_1/seed_42/full/<run_id>/`。类型拟为 `stage16_training_model`（last）、`stage16_inference_model`（best/initial），format version 1。

`last` 保存当前完整模型（含被微调的PLM）、optimizer/scheduler/scaler、round/global/pass step、各selector最佳分数/路径/SHA和patience、所有RNG及loader generator、sampler epoch、参考预测缓存身份、数据manifest与配置哈希。没有通过gate时，guarded best companion允许为null，resume不能假定best必然存在。

独立保存 `initial`、`best_legacy`、`best_pt_validation`、`best_guarded`；同一轮被多个selector选中可用一个不可变模型artifact和多个带SHA的索引，避免重复写入大型PLM。推理checkpoint使用内嵌配置恢复，不依赖之后可能改动的YAML。

`round_metrics.jsonl`、`selection_decisions.jsonl`、`reference_metrics.json`、`data_manifest.json`、`run_manifest.json`、`training_summary.md` 为必需产物。日志明确 `guard_pass`、`guard_failures`、最差split差值、每轮实际样本/steps、LR和评估耗时。

## 7. 实施顺序和测试验收

| 步骤 | 工作 | 通过条件 |
|---|---|---|
| S16-1 | 锁定来源、重做数据审计、重算参考 | SHA/标签/样本量一致；test未被访问 |
| S16-2 | 实现评估器和纯选择器 | 手算样例、NaN/Inf、等分、缺split均按契约处理 |
| S16-3 | 接入同轨迹三selector | 仅开启/关闭额外评估时，相同随机种子的下一步参数与采样一致 |
| S16-4 | 实现last/best/无可行解状态与resume | 中断恢复后采样、下一步LR、模型更新、选择结果一致 |
| S16-5 | 1轮接线测试与10轮pilot | 真实PLM前向、三split完整指标、没有缓存跨模型污染 |
| S16-6 | 固定35轮主比较 | 三selector候选轮集合相同，输出可行率、选中轮与所有差值 |
| S16-7 | 锁定方案，最终test及多种子确认 | 固定阈值、完整split、同seed配对；不按test回选模型 |

评估状态恢复测试必须覆盖 Python/NumPy/CPU/CUDA RNG、模块train/eval、encoder active_training、requires_grad及loader generator。评价时若需要unique-sequence缓存，对学生只能在同一checkpoint内复用；模型更新后必须失效。

梯度累积用不同大小microbatch对照一个完整有效batch，验证梯度等价；门槛测试包含“宏平均提高但某个split跌破底线”、所有候选无可行解、PHT掉点、round0最优，以及文件来源错配。

## 8. 成功标准与风险处理

工程完成标准是计划中的独立验证、shadow选择、门槛、日志、恢复和最终评测全部可运行并通过测试。科学成功标准是 S16-G 在保留条件成立时，独立PT验证及锁定后的test表现优于同轨迹 S16-L；若只有“拒绝坏模型”而没有更好的可行模型，应报告保护有效、性能提升未证实。

主要结果报告逐split AUROC/AUPR/MCC/Recall、PT宏平均S、PHT S/NLL、选中轮和计算成本。给出seed配对差值，采用以peptide为簇的配对bootstrap辅助区间；unseen_tcr另做TCR簇敏感性分析，说明两种相关结构未被单一重采样完全覆盖。先报告seed42探索，再报告多seed确认。已经观察过的PT/new是回顾性benchmark，更强的确认结论需要新的外部保留集。

独立验证的计算量可能大于原验证；先测单轮用时和显存，再按 `T_total≈rounds*(T_train+T_PHTval+T_PTval)` 估算。不得为了省时用不固定的小子集替代主gate。真实资源不足时降低评估batch或分块编码，保证split完整。Stage16 无可行解不妨碍 Stage17/18 从同一warmup开始。
