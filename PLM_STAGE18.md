# Stage 18 计划：增强 PT 回放、恢复正则化与 Stage1 教师保持

日期：2026-09-25。状态：代码已实现，并完成 CPU 回归测试及真实 PLM 的 R3/COMB 小规模验证；正式35轮实验尚未运行。运行方式与验收证据见 [Stage16–18 使用说明](STAGE16_18_IMPLEMENTATION.md)。

Stage18 对应第一优先项三：增加原始 PT 数据暴露，恢复频率加权和label smoothing，并使用冻结Stage1 PT专家的知识保持。核心问题是：在不改变学生结构和原有PLM解冻策略的前提下，能否减少三元训练后的二元能力损失？

主实验从共同Stage8A warmup初始化，继承 [Stage16](PLM_STAGE16.md) 的独立验证、gate和比较协议，但不继承 [Stage17](PLM_STAGE17.md) 的冻结策略。合并Stage17保护的版本单列为 `S18-COMB`，避免把三个第一优先项的贡献混为一谈。

## 1. 依据与假设

实际PT train有664,952条。历史Stage8每轮只回放18,047条，占2.71%；Stage1本次训练的PT步数上限为285，有效batch1,024，即每轮约291,840条。Stage8的282次PT更新虽然与Stage1接近，但有效batch64，原始数据暴露少约16倍。

Stage1 checkpoint保存的PT smoothing为0.03、group weighting为peptide×label的inverse-sqrt，clip范围0.25–4，并使用FGM。Stage8 replay是普通CE。PHT train的peptide频率与原PT差别明显，因此仅添加更多PHT监督不能替代原始PT覆盖。

H18a：回放量提高可以减轻分布偏移。H18b：在相同回放量下恢复频率权重与smoothing，能改善校准及罕见peptide表现。H18c：在相同数据和正则下，教师KL能减少已学知识的漂移。三项逐项消融，不用一次组合成功推断每项都有效。

新增回放不会保证提升；教师也可能保留旧模型偏差。硬标签仍是监督依据，教师提供软约束。主验证必须同时检查PT排序、MCC/Recall与PHT保持，不能只看KD loss变小。

## 2. 共同起点、学生与教师

### 2.1 学生

采用Stage8原有顺序三元结构，维度/attention/dropout沿用 [Stage16共同契约](PLM_STAGE16.md#2-三阶段共同实验契约)。初始化为 `stage8a_best_pht_20260921_111731_349210.pt`，来源Stage1 joint round9；共同warmup SHA为 `404d7aad11d17bb15480fea15380b290b7d8249b6c008f649b202fe5dc35d3a9`。

**独立主实验**仍解冻peptide/HLA/TCR PLM顶部各两层，joint更新全部相关下游模块；PH、PT、PHT的共享关系与Stage16控制一致。顺序仍为 `PH → PT → joint`。加大replay的效果不能与“把PT放到最后”同时归因。

### 2.2 教师

使用 `trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_ptcr_20260920_151134_400917.pt`，round10、`task_model`、input_mode=cache；SHA为 `aac5a63d0de2789c21f11505915b635dfc687108d2c3e77fb26bb3c5abc5ea91`。核对模型维度、tokenizer、原始PLM fingerprint和序列处理方式。seed扩展必须用同seed教师并记录manifest。

教师全程eval、requires_grad=False，使用原始冻结PLM表示。主学生PLM在训练中变化，因此禁止：

```text
学生微调encoder的hidden → Stage1教师下游 → 教师软标签
```

这会改变教师的输入坐标，使“保持原始知识”的目标失效。正确路径是独立的冻结教师encoder，或在训练前由原始encoder+教师生成缓存。主方案优先使用后者，以控制显存与重复编码成本。

学生不加载教师PT参数，也不替换共享peptide Adapter；教师只参与软标签监督与固定验证参考。这样初始化差异在所有条件中一致，不与KD混淆。

## 3. 数据与教师 logits 缓存

训练数据仍为 `dataset/PH/train.csv`、`dataset/PT/train.csv` 和 `dataset/PHT/classes/PHT_{000,010,100,111}/train.csv`。四类PHT有效72,190条，独立PT三个validation和最终test分工完全沿用Stage16。KD默认只作用于原始PT replay，不直接监督PHT样本，以免扩大教师目标的覆盖域后难以归因。

已实现 `source/precompute_plm_stage18_teacher.py`，在完整训练审计之后生成教师logits。只生成train软标签的缓存；独立PT validation参考预测另存，角色明确为reference，永不进入训练loader。test不参与缓存准备和超参选择。

缓存协议：

| 字段 | 要求 |
|---|---|
| 数据身份 | PT train文件SHA、规范化版本、有效row ID和过滤报告 |
| 输入键 | peptide/TCR实际输入序列与mask/长度策略；重复输入可共享logits但保留row映射 |
| 教师身份 | task checkpoint SHA、原始PLM/tokenizer fingerprint、特征层、精度与max lengths |
| 数值 | 两类原始logits，存FP32，不预先绑定温度；稳定softmax/对数计算在FP32 |
| 索引 | `row_id → cache_index`，不能依赖当前batch顺序或被截断前的行号 |
| 校验 | 行数、两个logit维度、finite、输入key一致；缺失/错配不得默认为0或跳过 |
| 复现 | 生成器代码版本、时间、随机/精度设置、manifest SHA |

只存664,952行×2个FP32 logits约5.1 MiB，另有索引/manifest；这不包含生成时的residue特征与原始模型占用。可按唯一序列编码并分块生成，不要求把整个residue cache装入GPU。训练过程中按row ID取logits，避免重复计算教师PLM。

缓存验收包括随机抽取固定1,024条有效训练记录，现场用教师原始编码器重新前向，核对logits/softmax与缓存误差；误差容限在数值校验阶段固定，不按下游表现调整。多温度实验复用同一原始logits。更换教师、序列处理、dtype或文件内容时缓存失效。

## 4. 回放配额与采样

定义回放比例为相对于**实际PHT train记录数**，而不是相对于PT train或balanced sampler抽样数：

\[
n_{PT}(r)=\min(N_{PT},\lfloor rN_{PHT}\rfloor).
\]

| PT比例r | 每轮PT行数 | PT train暴露比例 | PT更新数（有效batch64） | 加上PH+joint的总更新数 |
|---|---:|---:|---:|---:|
| 0.25，控制 | 18,047 | 2.71% | 282 | 1,692 |
| 0.5，可选剂量 | 36,095 | 5.43% | 564 | 1,974 |
| 1.0，主方案 | 72,190 | 10.86% | 1,128 | 2,538 |
| 2.0，可选剂量 | 144,380 | 21.71% | 2,256 | 3,666 |

PH配额始终18,047，joint仍采样72,192次。比例2.0是可选后续剂量，需要Stage18独立配置允许>1；不能被Stage8原来的`binary_replay_ratio≤1`错误拒绝，也不能绕过验证偷偷写入。

主方案每round对完整PT train做固定seed的均匀无放回排列，取前n条；不同round使用不同seed/generator状态。ratio 0.25/0.5/1.0/2.0使用同一round排列的嵌套前缀，以减少采样差异。跨round允许重复，不将累计暴露数误写为unique覆盖数；记录累计唯一pair/peptide/TCR覆盖。

不同时启用peptide加权采样，频率校正先只体现在loss中，避免双重加权。连续游标遍历整份PT train可作为额外条件，但主方案不把“改变抽样方式”混入回放量比较。

## 5. PT 监督与保持损失

### 5.1 频率权重

精确统计原始PT train中的 `g=(peptide,label)`，频次为n_g，原始权重a_g=n_g^(-1/2)。以所有实际存在group的a_g算术平均归一化，再clip到[0.25,4]，与当前Stage1实现的group定义和clip规则一致。负/正类分别成group，没有出现的group不伪造样本。

group统计只能来自train，缓存其SHA和计数。不要把当前原型的“只按peptide计数、按样本平均全局归一化”描述为Stage1同款；两者必须显式区分。当前 [peptide_weights原型](source/plm_unified/stage8_retention.py) 需要改为新的group函数才可用于本主方案。

每个有效累积窗口W的监督损失为：

\[
L_{sup}(W)=\frac{\sum_{i\in W}w_i CE_{\epsilon}(z_i,y_i)}{\sum_{i\in W}w_i},\quad\epsilon=0.03.
\]

这对应在有效batch上归一化的加权CE，避免microbatch=4时每个小batch单独归一化后再平均造成另一种目标。smoothing使用PyTorch两类均匀平滑定义：正确类目标0.985，另一类0.015；它不是把二分类正标签直接改成0.97。

权重clip保证单样本在归一化前有界，但归一化后的梯度还依赖窗口构成。记录权重分布、各peptide/label有效损失贡献和有效样本数，不能只记录平均loss。

### 5.2 教师 KL

\[
q_i^{(T)}=softmax(z_i^{teacher}/T),\quad p_i^{(T)}=softmax(z_i^{student}/T)
\]

\[
L_{KD}(W)=\frac{T^2}{|W|}\sum_{i\in W}KL(q_i^{(T)}\Vert p_i^{(T)}),\quad
L_{PT}=L_{sup}+\alpha L_{KD}.
\]

主参数T=2、alpha=0.5，温度只用于KD。评估始终使用T=1的学生输出；不得把温度2的概率直接拿去算主指标。教师logits detach，KL方向是teacher到student。学生和教师logits先float再进行log_softmax/softmax，避免bf16饱和影响KL。

主KD按样本均匀平均，不额外乘group权重，以明确其“保持原预测行为”的目标。对KD也加group权重、只在教师高置信样本上蒸馏或置信度门控，均作为另行登记的条件。硬标签CE一直保留；如果教师与标签分歧，记录该子集指标，而非删除困难样本。

### 5.3 累积与其他任务

在前向前确定窗口真实行数及权重总和。每个microbatch反向贡献为 `sum(weight*CE)/window_weight_sum + alpha*T²*sum(KL)/window_samples`，窗口最后一步统一clip/step。不得简单按microbatch个数平均；最后partial batch同样适用。

PH replay CE、joint的PHT/PH/PT监督权重1/0.3/0.2、约束0.05/0.03、margin0.05和5轮ramp全部沿用共同基线。新增KD不加入PHT损失。FGM在主消融中关闭；恢复epsilon0.5/weight0.25属于后续 `S18-FGM`，需要单独计费并与相同CE/KD条件比较。

## 6. 消融矩阵与顺序

| 条件ID | PT比例 | group weighting | smoothing | KD | 比较解释 |
|---|---:|---|---:|---|---|
| S18-R0 | 0.25 | 无 | 0 | 无 | Stage16同训练控制，可复用合格轨迹 |
| S18-R1 | 1.0 | 无 | 0 | 无 | R0→R1仅加大回放 |
| S18-W | 1.0 | peptide×label | 0 | 无 | R1→W仅恢复频率权重 |
| S18-S | 1.0 | 无 | 0.03 | 无 | R1→S仅恢复smoothing |
| S18-R2 | 1.0 | peptide×label | 0.03 | 无 | 验证两种正则的组合 |
| S18-R3，主方案 | 1.0 | peptide×label | 0.03 | alpha0.5/T2 | R2→R3在同回放与正则下检验KD |
| S18-COMB，组合 | 1.0 | peptide×label | 0.03 | alpha0.5/T2 | R3+Stage17-J冻结/隔离，不计入独立KD收益 |

执行优先R0/R1/R2/R3，W/S用于分离正则贡献，COMB在Stage17-J接口测试完成后运行。全部从同一个warmup重新开始，不在R1 best上继续训练R2，否则会把额外训练步数混入差异。

R3与R2采用完全相同的抽样行序、学生初始化、batch、LR和预算。KD缓存查找不得消耗训练随机数。固定teacher在不参与KD的条件中不需要训练时加载，但评估参考完全相同。

超参探索次于主消融：仅在独立validation上按顺序试alpha∈{0.1,0.5,1.0}；确认保留行为后再试T∈{1,2,4}或回放比例。记录每次试验，避免默认运行3×3×4的全网格。最终结果需区分预注册主参数与验证后调参结果。

COMB先与Stage17-J比较（同冻结策略，新增回放/正则/KD），再与R3比较（同回放策略，新增冻结/隔离）。如要叠加Stage17-P对PH共享Adapter的保护，另用`S18-COMB-P`；默认COMB不默默引入它。

## 7. 预算、LR日程和公平性

沿用microbatch4、accumulation16、bf16、clip1及Stage16各模块LR。独立主学生仍有两个PLM学习率5e-6/2.5e-6。optimizer参数覆盖和weight decay规则保持共同控制，不同时恢复Stage1的其他优化器细节。

主比较以**相同PHT暴露量**为预算：35轮，每条件PHT采样次数相同。R0共59,220个更新，R3共88,830个更新，增加50%；PT replay累计暴露从631,645条增至2,526,650条。它们是含重复的行暴露数，不是独立样本总数。

scheduler仍按100-round horizon映射总计划更新、warmup5%；r不同会改变每轮总更新，因此必须存储每组LR随实际step/round的曲线。论文或结果表必须说明R1/R3花费了更多计算，不能把收益全部描述为更优算法。R0→R1评价的是回放策略整体干预，包含更多更新及其日程映射，不是已经独立分离了样本覆盖、更新次数和LR轨迹的因果贡献。

增加一个**等更新数**辅助实验：R0跑30轮、r=1.0条件跑20轮，均50,760个更新。该辅助实验单独从共同warmup启动，两者scheduler_total_steps均50,760、warmup比例相同，不直接截取主实验中LR不同的中间checkpoint。它减少了r=1.0条件的PHT暴露，反映预算权衡，不替代主消融。

相同optimizer-step不等于相同FLOPs/GPU时间，因为joint与pair前向成本不同。报告实际GPU时间、编码序列数、峰值显存、教师预计算和验证成本；教师缓存可跨消融复用，但一次性成本及摊销方式都要列出。先测一轮吞吐再估算总时间，不预写未经测量的小时数。

## 8. 选择、诊断和副作用监测

所有条件采用Stage16的J16和guard：独立PT AUROC/AUPR/MCC宏平均相对固定Stage1参考最多下降0.01，每split最多0.02，PHT的S相对共同warmup最多下降0.02；同分与无可行解行为完全一致。teacher训练缓存不能替代独立PT validation参考预测。

每轮记录 `replay/PT/supervised_CE`、`KD_T2`、`alpha*KD`、权重范围、CE/KD分别对共享参数的梯度范数、teacher-student概率差、teacher-label分歧率及该子集表现。高频与低频peptide的分组阈值只按训练频率预定义；validation/test不能参与频率估计。

完整验证额外报告NLL/ECE/Brier、Recall/Specificity、最差split及peptide group覆盖率。若AUROC改善但Recall继续下降，要区分概率校准与排序变化；主结果阈值固定0.5。可在validation另做阈值校准作补充，阈值随模型封存，不在test重新选择。

PHT各类型与PH的变化继续记录。更多PT更新也可能让共享peptide/TCR表示向PT偏移，损害PHT；发生时gate应拒绝该checkpoint，而不是只展示PT的最好一项。

## 9. 实现文件与产物

| 文件 | 计划职责 |
|---|---|
| `configs/plm_stage18.yaml` | condition、分任务replay配额、group权重、smoothing、alpha/T、预算协议 |
| `source/plm_unified/stage18_config.py` | 允许独立比例及消融开关，拒绝非法teacher/输入模式组合 |
| `source/plm_unified/stage18_teacher.py` | 教师加载、冻结、logits缓存与identity验证 |
| `source/precompute_plm_stage18_teacher.py` | train缓存/validation reference分角色生成与抽查 |
| `source/plm_unified/stage18_loss.py` | effective-window加权CE、FP32 KL、逐项诊断 |
| `source/plm_unified/stage18_trainer.py` | 独立replay配额、嵌套采样、共同runner、正确累积与状态 |
| `source/plm_unified/stage18_checkpoint.py` | 保存学生全模型、teacher/cache身份、resume状态及推理导出 |
| `source/train_plm_stage18.py`、`source/evaluate_plm_stage18_benchmarks.py` | 审计/训练/恢复/最终评测，复用Stage16选择器 |
| `tests/test_plm_stage18.py` | teacher身份、KD方向、窗口梯度、配额、采样/恢复、消融等价性 |

上述接口均已实现，正式Stage18使用完整resume状态与独立教师缓存。较早的 [PTRetention原型](source/plm_unified/stage8_retention.py) 仅作历史参考；其PLM全冻结、PH共享Adapter冻结、训练顺序以及peptide-only权重不等价于独立S18-R3，不能替代新入口。

输出根目录为 `trained_model/PLM_Stage18/fold_<fold>/seed_<seed>/<condition>/<run_id>/`。主方案保存encoder-inclusive的 `stage18_training_model` last与 `stage18_inference_model` best；不能仅保存binding权重后靠warmup原始encoder恢复，因为主学生的PLM已经微调。

last必须保存模型、optimizer、scheduler、scaler、各阶段步数、累计样本数/采样游标、RNG/loader/sampler状态、各selector与gate参考、teacher/cache/manifest SHA、group统计哈希、配置和预算定义。断点恢复必须验证teacher/cache与原运行一致；训练允许外部只读teacher缓存依赖，推理artifact不依赖教师。

必需日志/报告包括 `teacher_cache_manifest.json`、`group_weights.json`、`exposure_stats.jsonl`、`round_metrics.jsonl`、`selection_decisions.jsonl`、`ablation_summary.md`，以及共同data/run manifests。每个最终模型保留选中轮的完整validation证据与失败门槛为空的记录。

## 10. 实施步骤与测试要求

| 步骤 | 工作 | 验收证据 |
|---|---|---|
| S18-1 | 锁定教师、训练数据、group统计与缓存协议 | 来源SHA、只读train/reference角色、行ID覆盖一致 |
| S18-2 | 小规模缓存→完整train缓存 | 抽查1,024行对照真实教师；错序/错模型/缺行均拒绝 |
| S18-3 | effective-window CE/KD | 与一次完整batch梯度等价，支持末尾短batch |
| S18-4 | 分任务回放、嵌套采样与LR映射 | 实际样本/steps与配额表一致，不发生双重加权 |
| S18-5 | 主学生在线微调与缓存teacher隔离 | 改动学生encoder后缓存teacher logits不变，学生仍收到KD梯度 |
| S18-6 | Stage16 gate和完整resume | teacher固定；中断后下一步采样、LR、KD与不中断一致 |
| S18-7 | 1轮接线、10轮pilot、35轮R0/R1/R2/R3 | 完整独立PT与PHT验证、消融和实际资源表 |
| S18-8 | W/S、等更新预算与COMB | 对应控制条件明确，不混用采样与起点 |
| S18-9 | 锁定配置、多seed与最终test | 所有split、固定阈值、seed配对及teacher对照完整 |

单元测试必须覆盖alpha=0退化为无KD、T=1与T²因子、teacher无梯度、KL方向、teacher=student时KD近0、极端logits不NaN、label smoothing目标正确、group按(peptide,label)而不是仅peptide统计、cache错序和来源变化、缺失数据fail-fast、回放上限min行为。

整轮测试不能只检查loss下降：需证明允许的学生encoder/Adapter/PT head确实更新，教师逐参数未变，joint未被意外冻住，COMB则满足Stage17-J不变量。Stage18配置中的`freeze_plm=false`与`teacher_features=student_hidden`组合必须被明确拒绝。

checkpoint roundtrip用同一真实输入比较恢复前后PT/PHT输出；部署路径只保留学生，不要求运行教师。benchmark加载器必须识别Stage18新类型、内嵌模型配置和fine-tuned encoder state，禁止套用原型binding-only恢复方式。

## 11. 验收标准、风险与决策

工程完成：回放配额可审计、加权CE/KD数学目标正确、教师特征独立、数据无误用、恢复与最终评估可运行。科学结果分开报告R0→R1（更多回放）、R1→R2（组合正则）、R2→R3（同预算下KD），以及W/S的单项贡献。通过gate仅表示保留，不能宣称已超过Stage1。

若R1优于R0而R3不优于R2，优先保留更多回放和有效正则，报告当前KD没有额外收益；不因方法复杂而强行选择R3。若KD很小但验证下降，检查teacher缓存身份、学生特征是否错用、KD是否被稀释，以及教师在罕见peptide上的偏差。

若独立R3仍损害PHT而COMB可兼顾，说明组合保护更合适，但它的收益应归于明确的组合条件。若所有条件都不可行，保留失败证据，进一步审视数据/标签与任务定义；不要继续通过观察test调整alpha或gate。

资源上先生成可复用teacher logits，避免与学生同时常驻两套大PLM。完整训练前做真实显存/吞吐测量，按可用GPU串行或调度运行。训练中断恢复属于正式实现验收内容，不能依赖人工从best重新启动来替代。
