# Stage 17 计划：冻结 PLM 与三元训练的梯度隔离

日期：2026-09-25。状态：代码已实现，并完成 CPU 回归测试、真实 PLM 小规模训练及断点恢复；正式消融实验尚未运行。运行方式与验收证据见 [Stage16–18 使用说明](STAGE16_18_IMPLEMENTATION.md)。

Stage17 对应第一优先项二：冻结 PLM，并让 joint pass 仅更新 PHT 新模块，验证共享表示漂移是否是 PT 退化的关键因素。核心目标是在保留现有三元前向结构的同时，明确哪些训练步骤会改变二元能力。

本阶段采用 [Stage16](PLM_STAGE16.md) 的数据、统一评估、保留门槛和checkpoint协议，从共同 Stage8A warmup重新初始化，不接着 Stage16 best 继续训练。[Stage18](PLM_STAGE18.md) 的加大回放、smoothing、分组权重及教师蒸馏不进入 Stage17 主方案。

## 1. 假设、结论边界与设计原则

H17a：冻结基础 PLM 可以缓解微调后的 residue 表示偏移。H17b：仅冻结 PLM 仍不足，因为三元梯度继续更新共享 Adapter/PH attention；进一步隔离 joint 更新可以降低 PT 漂移。

需要区分两个层次的保持：

- **单个 joint pass 的保持**：其前后二元路径参数和固定输入的eval输出不变，属于可严格测试的工程不变量。
- **整轮/整个训练的保持**：PH/PT replay 仍可能更新共享 peptide Adapter，PT replay 也会更新PT head；只能通过独立验证确认，不能由第一个不变量推出。

如果冻结后 PT 保持而 PHT 明显下降，只能说明存在任务适应权衡。若二者都未改善，需转向回放分布、正则化与标签/任务语义，而非自动扩大冻结范围。Stage17 是保护性训练消融，不预设冻结一定最优。

## 2. 初始化、数据和不可混入的变化

共同来源使用 [Stage16 第2节](PLM_STAGE16.md#2-三阶段共同实验契约) 的三文件SHA清单：Stage1 joint round9 → Stage8A best warmup；Stage1 best_ptcr round10只作为独立PT验证参考。

学生起点是 `trained_model/PLM_Stage8/fold_1/seed_42/warmup/stage8a_best_pht_20260921_111731_349210.pt`，不是已经微调PLM的Stage8B best。冻结一个已经漂移的Stage8B encoder不会恢复原始坐标系。重新构造PLM后逐参数检查warmup中的encoder权重与所声明的原始权重一致，并检查warmup二元权重与exact Stage1 joint一致。

PH/PT train、四类PHT train/val、独立PT三split与最终PH/PT/PHT test路径及序列规则完全沿用Stage16。每轮仍是PH replay 18,047条、PT replay 18,047条、joint采样72,192次；三个PT验证集分别10,382/154,260/143,028条。不得为了弥补冻结而在本阶段主方案增加PT数据。

维持同一256维Adapter和attention、8 heads、1024 FFN、原来的顺序三元网络，不引入双向attention、Hyper-PMT、额外残差Adapter或新分类头。所有条件保持 `PH → PT → joint`。原型中的 `PH → joint → PT` 是另一个干预，不直接沿用。

## 3. 模型前向与参数分组

使用 [Stage8BindingModel](source/plm_unified/stage8_model.py) 的前向关系：

```text
peptide PLM → peptide Adapter ─┬→ PT attention(peptide,TCR) → PT pool/head
                              └→ PH attention(peptide,HLA) → PH pool/head
                                             └→ PHT attention(PH sequence,TCR) → PHT pool/head
```

TCR PLM/Adapter由PT和PHT共享。PHT不经过PT attention、pool或classifier，不能用“冻结PT head”替代对共享参数的保护。

主方案 `S17-J` 的逐pass可训练参数如下。✓表示requires_grad=True并处于训练模式；—表示冻结且eval。

| 模块 | PH replay | PT replay | joint |
|---|---|---|---|
| peptide/HLA/TCR全部PLM | — | — | — |
| peptide Adapter | ✓ | ✓ | — |
| HLA Adapter | ✓ | — | — |
| TCR Adapter | — | ✓ | — |
| PH attention/pool/classifier | ✓ | — | — |
| PT attention/pool/classifier | — | ✓ | — |
| PHT attention/pool/classifier | — | — | ✓ |

主方案允许PH replay更新peptide Adapter，与原始训练语义一致。因此它仍可能影响PT；`S17-P` 单独研究在PH pass冻结共享peptide Adapter的额外收益。

冻结必须同时保证权重和行为：PLM dropout关闭，encoder的active_training=False，LayerNorm等保持推理行为；joint时二元Adapter/attention的dropout也关闭。不能只设置requires_grad=False然后对整个model调用train()覆盖局部模式。

## 4. 损失与有向约束的实际梯度

PH/PT replay 的主方案仍使用普通CE，不启用KD、group weighting、smoothing或FGM。

joint前向和损失定义保留Stage8：

\[
L_{joint}=L_{PHT}+0.3L_{PH}+0.2L_{PT}+\lambda_HL_H+\lambda_TL_T.
\]

PHT CE由111总权重0.5、其余三类合并总权重0.5组成。由于PH/PT路径冻结，它们的辅助CE只记录数值，不更新参数。日志应分别给出 `reported_total_loss` 和对当前可训练参数有贡献的 `effective_train_loss`，避免把常量项当作优化能力。

对111：现有正约束使用 `relu(pht.detach()-pair-margin)^2`，其梯度原本流向二元分支；在S17-J中该分支被冻结，因此没有可训练参数收到这个约束的梯度。对000/010的PH负约束、000/100的PT负约束，pair.detach()作为锚点，PHT仍可收到梯度。

这意味着S17-J保持原损失公式，但其有效梯度范围改变。必须报告各约束的active参数和梯度范数；不能称为“完全相同的约束优化”。不为让正约束重新有梯度而偷偷反转detach方向，也不把所有样本强制成 `PHT≤PT` 的硬概率模型。

PHT新模块接收固定上游表示，仍可学习HLA条件下的TCR识别。这一阶段刻意限制跨任务参数适应，后续专属Adapter/残差路径属于新的结构实验。

## 5. 对照条件与可归因比较

| 条件ID | PLM策略 | joint下游策略 | 二元replay | 用途 |
|---|---|---|---|---|
| S17-U | 顶部各两层可训练 | 全部相关下游可训练 | 基线CE | 与Stage16 controlled_v1共享的训练控制 |
| S17-F | 全冻结 | 全部下游可训练 | 基线CE | U→F：仅PLM冻结的效果 |
| S17-J，主方案 | 全冻结 | 仅PHT新模块 | 基线CE | F→J：进一步隔离joint的效果 |
| S17-P，次要 | 全冻结 | 仅PHT新模块 | PH另冻peptide Adapter | J→P：PH对PT共享表示的影响 |
| S17-Z，诊断 | 全冻结 | 仅PHT新模块 | 无二元更新 | 检验全程PT输出严格保持，不作同预算优劣主证据 |

主证据先完成U/F/J。U可复用Stage16同run的完整轨迹和预测，但必须确认初始化、每轮抽样、LR日程、预算、工程修正版本完全相同。P和Z标为次要；Z改变数据更新预算，不能把其低成本或保持能力当作泛化优势。

另可对J做 `constraint=baseline` 对照，研究仅剩负向PHT约束的影响；这不是替换主条件full。所有条件接受Stage16的同一独立PT选择规则，并保留每轮固定位置（joint结束后）的指标。

## 6. 训练参数、optimizer与缓存

microbatch=4、accumulation=16，正常有效batch=64。每轮PH/PT各282个optimizer steps，joint 1,128个，共1,692个；末尾partial window按实际样本数归一化。主预算35轮，对应59,220个StageB更新，不包含继承的warmup。seed42先pilot，之后按共同协议多seed确认。

下游LR延续Adapter 2.5e-5、已有attention/pool/head 5e-5、PHT模块1e-4；weight decay0.01、clip norm1、bf16。PLM组从optimizer中排除，不能仅设置学习率为0然后误以为没有状态/衰减风险。scheduler沿用共同100-round horizon、5% warmup，按实际有效更新推进。

optimizer初始化必须收集PH/PT/joint各pass将使用的参数并集。若直接在“只开放PHT”的模式下构造optimizer，会漏掉replay参数；当前保护原型已经遇到并处理这一点，新实现必须有覆盖测试。

每次切换pass先 `zero_grad(set_to_none=True)`，再设置trainability/mode。被冻结的参数必须grad=None，防止上一pass残留梯度和AdamW权重衰减造成变化。optimizer已包含的参数可以暂时冻结，不需要每个pass重新初始化动量。

首个对照实现保留在线冻结PLM编码，便于与Stage16对齐。之后可增加冻结residue cache作为效率模式，但必须单独通过等价性验收：缓存绑定原始PLM SHA/fingerprint、tokenizer、序列规范化、长度、输出层、dtype、特殊token处理和mask；禁止直接复用Stage8B微调encoder缓存。

缓存不得保存正在训练的Adapter输出。相同PLM residue可以跨round复用；下游特征不可以。在线/缓存等价性在固定真实样本上比较mask、residue和logits，先FP32，再记录bf16/存储fp16误差。主结论只比较同一推理/缓存协议；不能按效果挑选缓存精度。

## 7. 诊断：把“遗忘”定位到训练步骤

从独立PT validation每个split预先锁定最多2,048条的标签分层监控子集，保存行ID、seed与SHA。监控子集只作诊断，完整split才用于gate。

在固定的 round0、1、5、10、20、35，分别于PH前、PH后、PT后、joint后记录：PT AUROC/AUPR/MCC/Recall、同一行概率变化分位数、平均预测正率、各模块参数相对L2漂移。checkpoint选择始终使用joint后完整验证，避免对某个条件挑选有利的pass。

S17-J必须满足单个joint前后PT相关参数逐tensor相等、固定eval输入输出在确定性执行下相等。GPU非确定性核允许预先声明的数值误差，但参数本身仍应逐bit相同；先在CPU确定性测试中证明严格相等。输出差异超出重复推理本底时检查dropout、encoder模式、stale gradients和buffer更新。

对于S17-F/U，记录joint中PHT监督、PT辅助监督及约束分别对共享Adapter/PLM的梯度范数与cosine。诊断只在固定少量训练batch进行，使用autograd.grad且不覆盖主梯度，不从validation反向传播，不改变采样或optimizer步数。

## 8. 实现模块与状态契约

| 文件 | 计划责任 |
|---|---|
| `configs/plm_stage17.yaml` | explicit condition、encoder_policy、joint_trainability、PH共享Adapter策略 |
| `source/plm_unified/stage17_config.py` | 校验主/次要条件及来源，允许真正冻结，不套用Stage8层数必须2的规则 |
| `source/plm_unified/stage17_model.py` | 复用Stage8前向；实现逐pass训练权限与模式管理 |
| `source/plm_unified/stage17_trainer.py` | optimizer参数并集、有效梯度日志、逐pass漂移诊断、共同runner |
| `source/plm_unified/stage17_checkpoint.py` | 自包含或显式依赖bundle的冻结模型保存/恢复 |
| `source/train_plm_stage17.py`、`source/evaluate_plm_stage17_benchmarks.py` | 审计/训练/恢复/最终评测 |
| `tests/test_plm_stage17.py` | 权重与输出不变量、optimizer覆盖、模式、缓存及恢复测试 |

复用Stage16的evaluation/selection，不能复制后悄悄改gate。现有 [ProtectedStage8Model](source/plm_unified/stage8_retention.py) 可借鉴，但它在PH pass额外冻结peptide Adapter，等价于这里P的部分策略；复用时必须拆成显式开关，主J默认关闭该额外保护。现有PTRetentionTrainer同时加KD/权重/replay/改顺序，不能直接作为Stage17主trainer。

建议初版推理artifact包含完整模型state，类型 `stage17_inference_model`；last为 `stage17_training_model`，完整保存optimizer/scheduler/scaler/RNG/sampler/selector状态。冻结虽可节省训练显存，不意味着可以省掉恢复所需的来源信息。

若为节省磁盘导出binding-only，必须同时导出不可变encoder bundle及SHA映射；缺失或错配依赖时拒绝加载。重建依据内嵌模型配置，不依赖可变的外部YAML。输出位于 `trained_model/PLM_Stage17/fold_<fold>/seed_<seed>/<condition>/<run_id>/`，不覆盖Stage8或Stage16。

## 9. 执行步骤与验收清单

| 步骤 | 实施内容 | 验收证据 |
|---|---|---|
| S17-1 | 拆出独立config和U/F/J/P/Z权限表 | 每个条件的可训练参数清单、numel、optimizer覆盖无重复 |
| S17-2 | 实现模式/梯度保护 | 真实下游网络的CPU测试；一次joint只改变PHT新模块 |
| S17-3 | 完整三pass更新 | PH/PT仍能更新各自允许的参数；joint不能污染冻结参数 |
| S17-4 | 真实PLM与缓存可选验证 | GPU单batch前后原始encoder权重相等，dropout关闭，mask对齐 |
| S17-5 | 接入Stage16 gate、日志和resume | 同一checkpoint评估一致；中断下一步与不中断一致 |
| S17-6 | 1轮接线、10轮pilot、35轮U/F/J | 完整独立PT+PHT验证，逐pass漂移报告、每轮实际成本 |
| S17-7 | 依证据追加P/Z与条件约束对照 | 全程PT严格保持与实际PT泛化提升分别报告 |
| S17-8 | 锁定方案、最终test、多seed | 与同seed U比较，未通过gate不作为合格提升模型 |

必须额外测试：已存在optimizer动量时切换冻结、连续多个round的sampler epoch、零解冻层不会因为Python切片 `[-0:]` 变成解冻全部层、评估后恢复正确pass模式、joint前后PT预测相等但整轮允许变化、导出恢复后输出一致。

约束测试分别对111/000/010/100构造输入，检查理论梯度方向和冻结后的零梯度；不能只测试loss数值有限。PHT有效loss必须确实更新triplet模块，避免“全部冻结”也通过保持测试。

## 10. 科学验收与后续分支

工程通过要求：梯度路由、模式控制、可训练参数并集、完整独立验证、恢复、最终评测全部完成。科学验收使用Stage16固定gate：PT三指标宏平均相对Stage1专家最多下降0.01、每split最多0.02，PHT的S相对warmup最多下降0.02；门槛通过仅说明可接受保持，不等于优于专家。

如果J优于U且F介于两者之间，可支持“PLM漂移与下游共享更新都值得控制”；它不是对所有可能数据/模型的普遍因果结论。如果J的joint局部保持测试通过但独立PT仍下降，优先比较PH后与PT后的漂移，利用P分离PH共享影响，再进入Stage18检查回放与正则。

如果PHT掉点超过门槛，记录不可行，不通过放松阈值掩盖代价。后续可以研究PHT专属残差Adapter或一个PLM顶层的受保护解冻，但不把这些结构变化混入本计划的主结果。

资源按相同数据/round预算比较，另报告冻结带来的峰值显存、吞吐、训练/验证耗时。U/F/J相同optimizer-step数量不意味着FLOPs相同；不将计算节省自动解释为预测能力提升。Stage17即使尚无可行模型，Stage18的独立主实验仍可从共同warmup开始。
