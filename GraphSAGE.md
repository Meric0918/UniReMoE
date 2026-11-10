# 论文分析与预测模型（Hyper-SAGNN）

**User:** Anonymous  
**Created:** 8/3/2026 19:44:00  
**Updated:** 8/31/2026 16:52:57  
**Exported:** 8/31/2026 16:59:28  
**Link:** [https://chatgpt.com/c/6a707eef-bdf4-83ec-93e3-50299b8dbd2e](https://chatgpt.com/c/6a707eef-bdf4-83ec-93e3-50299b8dbd2e)  

## Prompt:
2026/8/3 19:43:57

Q1:这篇论文试图解决什么问题?
Q2:有哪些相关研究?
Q3:论文如何解决这个问题?
Q4:论文做了哪些实验?
Q5:有什么可以进一步探索的点?
Q6:总结一下论文的主要内容

[Attachment: s42256-025-01002-0.pdf]

## Response:
2026/8/3 19:43:59

> **总结UniPMT论文方法、基准与实验结果**
>
> 
>
> **读取 PDF 处理技能指南**
>
>
> **梳理研究方向**
>
> 我会区分论文已提出的后续工作与基于结果的延伸建议，重点覆盖多任务学习、P-M、P-T、P-M-T预测，消融实验、可解释性、泛化能力和临床转化验证。
>
> **梳理了研究方向**
>
> 
>
> **区分后续方向**
>
> 我会把论文明确提出的方向与我的延伸建议分开。重点包括数据扩充、α链与其他链信息、三维结构、更严格数据划分、可靠负样本、外部实验验证，以及组织环境、表达量和不确定性建模。
>
> **区分后续方向**
>
> 
>
> **完成PDF指南审查、页面渲染及论文方法内容提取**
>
> 
>
> 思考了 1m 35s

下面基于论文 **《A unified deep framework for peptide-major histocompatibility complex-T cell receptor binding prediction》** 进行梳理。文中将 peptide、MHC、TCR 分别简称为 **P、M、T**，模型名称为 **UniPMT**。`s42256-025-01002-0.pdf`

---

## Q1：这篇论文试图解决什么问题？

论文试图解决的核心问题是：

> **如何更准确地预测抗原肽、Ⅰ类 MHC 和 TCR CDR3β 之间的特异性结合，尤其是 P-M-T 三元结合。**

这一问题的重要性在于：抗原肽首先被 MHC-I 呈递，然后由 TCR 识别。准确判断这个过程可用于：

- 肿瘤新抗原筛选；
- 抗原特异性 TCR 发现；
- TCR-T 细胞治疗；
- 肿瘤疫苗设计；
- 免疫治疗响应预测。

传统实验方法成本高、耗时长，因此需要计算预测模型。

### 现有方法的主要不足

现有研究通常分别处理：

- P-M：肽能否被 MHC 呈递；
- P-T：肽能否被某个 TCR 识别；
- P-M-T：某个 MHC 呈递的肽能否被 TCR 识别。

但这三个任务在生物学上互相关联，现有模型却大多只使用其中一种关系。例如，P-T 模型通常不利用 P-M 呈递信息，造成：

1. 多种结合信息没有被充分利用；
2. P-M-T 数据稀缺时，模型难以学习稳定规律；
3. 容易学习到单一数据集中的偏差；
4. 对新肽、新 TCR 和新抗原的泛化能力不足。

因此，论文的核心假设是：

> 联合学习 P-M、P-T 和 P-M-T，可以让三个任务相互提供辅助信息，从而提高整体预测性能。

论文重点预测的是：

- **Ⅰ类 MHC**；
- **TCR β 链的 CDR3 序列**；
- 输出 0～1 之间的结合概率。

---

## Q2：有哪些相关研究？

论文中的相关研究大致分为三类。

### 1. Peptide-MHC 结合预测

P-M 预测主要包括两种路线。

**打分函数方法**

通过统计打分、序列 motif 或亲和力矩阵估计 peptide 与 MHC 的结合概率。

**深度学习方法**

利用 CNN、LSTM、注意力或 Transformer 学习 peptide 和 MHC 的序列表示。论文讨论或比较的方法包括：

- Anthem；
- CapsNet-MHC；
- DeepAttentionPan；
- DeepSeqPan；
- ACME；
- NetMHCpan；
- HLAB。

这类模型通常能够较好地预测抗原呈递，但并不直接回答 TCR 是否识别该 pMHC。

### 2. Peptide-TCR 结合预测

P-T 模型主要学习 peptide 与 TCR CDR3 序列之间的关系，包括：

- TEIM；
- TCR-AI；
- PanPep；
- NetTCR 2.0；
- NetTCR 2.2；
- ImRex；
- DeepTCR；
- DLpTCR。

其中：

- TEIM 强调 peptide 与 TCR 的残基交互；
- PanPep 使用元学习提高对未见 peptide 的泛化能力；
- NetTCR 系列使用卷积网络及序列相似性信息；
- ImRex 将 peptide-TCR 关系表示成类似图像的交互矩阵。

这些方法的问题是，通常忽略了 peptide 是否能够被对应 MHC 正确呈递。

### 3. Peptide-MHC-TCR 三元预测

代表性方法是 **pMTnet**。

pMTnet同时输入 peptide、MHC 和 TCR，利用迁移学习预测特定 MHC 呈递条件下的 TCR 特异性。

但论文认为，pMTnet 等方法仍主要把 P-M-T 当作一个独立任务，没有系统联合大规模 P-M 和 P-T 数据。

### 4. UniPMT所借鉴的方法

UniPMT还结合了多个通用机器学习方向：

- ESM蛋白质语言模型；
- GraphSAGE图神经网络；
- 异质图表示学习；
- 多任务学习；
- 深度矩阵分解 DMF；
- 对比学习与信息噪声对比估计；
- 负采样。

其主要创新并不是提出一种全新的序列编码器，而是把这些技术组织成一个统一的 P-M-T 学习框架。`s42256-025-01002-0.pdf`

---

## Q3：论文如何解决这个问题？

UniPMT从**数据、图结构和训练任务**三个层面进行统一。

### 1. 数据层面：构建统一关系数据集

作者整合多个公开数据来源：

- P-M 数据：BigMHC、IEDB 等；
- P-T 数据：PanPep、DLpTCR、NetTCR、IEDB TCR-full-v3；
- P-M-T 数据：ERGO、pMTnet。

经过清洗、去重和格式统一后，基础数据图包含：

- 291,632 个 peptide；
- 208 个 MHC；
- 144,053 个 TCR；
- 593,109 条 P-M-T 关系；
- 70,112 条 P-M 关系；
- 155,479 条 P-T 关系。

这种设计使一个任务的数据可以辅助其他任务。

### 2. 表示层面：构建异质图

UniPMT把三种生物实体表示为节点：

- peptide 节点；
- MHC 节点；
- TCR 节点。

把三种已知结合关系纳入图中：

- P-M；
- P-T；
- P-M-T。

初始表示方面：

- peptide 使用 ESM 获得 embedding；
- TCR CDR3β 使用 ESM 获得 embedding；
- MHC 使用由 TEIM 方法提取的 MHC pseudo-sequence 表示。

这里的 MHC pseudo-sequence 主要保留与 peptide 接触或与结合相关的位置，而不是直接使用完整 MHC 序列。

### 3. 图学习：GraphSAGE信息传播

作者使用 GraphSAGE 聚合邻居信息，更新 P、M、T 节点表示。

因此，一个 peptide 的表示不仅包含自身序列信息，还能够包含：

- 与它结合过的 MHC 信息；
- 与它结合过的 TCR 信息；
- 相关 P-M-T 三元关系信息。

论文第3页的图1将模型分为三个阶段：

1. Graph construction；
2. Graph structure learning；
3. Unified binding learning。

这也是 UniPMT 与单纯序列拼接模型的关键区别。

### 4. 统一学习三个任务

#### P-M任务

将 peptide 表示和 MHC 表示输入神经网络，得到 P-M 联合向量，再经过线性层和 sigmoid，输出 P-M 结合概率。

该任务采用二分类交叉熵损失。

#### P-M-T任务

首先得到：

- P-M 联合表示 $v_{pm}$；
- M-T 联合表示 $v_{mt}$。

随后利用 DMF 思想，对两个表示进行逐元素乘积，再使用多层感知机建模非线性交互：

$$
P_{pmt}=\sigma\left(f_{\mathrm{DMF}}(v_{pm}\odot v_{mt})\right)
$$

由于 P-M-T 数据中可靠负样本不足，作者通过随机打乱生成负样本，并使用类似 InfoNCE 的对比学习损失，让正三元组的分数高于负三元组。

#### P-T任务

P-T 结合并不是完全脱离 MHC 直接计算，而是考虑不同可能 MHC 下的 P-M-T 结合结果，再对多个 MHC 的预测进行聚合：

$$
P_{pt}=\frac{1}{M}\sum_{j=1}^{M}P_{pm_jt}
$$

这意味着 P-T 预测隐式考虑了 MHC 条件。

#### 总损失

三个任务进行联合优化：

$$
\mathcal{L}
=
\lambda_{pm}\mathcal{L}_{pm}
+
\lambda_{pt}\mathcal{L}_{pt}
+
\lambda_{pmt}\mathcal{L}_{pmt}
$$

这样，P-M、P-T 和 P-M-T 可以共享表示并相互辅助。`s42256-025-01002-0.pdf`

---

## Q4：论文做了哪些实验？

论文实验可以分为八组。

| 实验 | 数据与对比方法 | 主要结果 |
|---|---|---|
| P-M-T 基准预测 | pMTnet 数据集；对比 PanPep、pMTnet、TEIM | UniPMT 的 ROC-AUC 约 0.96，PR-AUC 约 0.72；至少提升 4 和 15 个百分点 |
| 困难的新抗原 P-M-T 测试 | 固定 P-M，仅替换新抗原特异性 TCR | ROC-AUC 0.7214，PR-AUC 0.2836 |
| P-M 预测 | IEDB BD2013；对比 CapsNet-MHC、DeepAttentionPan、Anthem、DeepSeqPan | ROC-AUC 93.55%，PR-AUC 84.35% |
| P-T 一般场景 | TEIM 数据集；五折评估 | ROC-AUC 约 0.78，PR-AUC 约 0.63 |
| P-T zero-shot | PanPep 数据集，测试 peptide 在训练中未出现 | ROC-AUC 约 0.62，PR-AUC 约 0.84 |
| 消融实验 | 分别只保留 P-M-T、P-M 或 P-T 任务 | 证明多任务关系对 P-M-T 和 zero-shot P-T 特别重要 |
| 临床相关验证 | 实验验证的新抗原特异性 P-M-T 三元组 | ROC-AUC 94.29%，PR-AUC 72.15% |
| 结构与可解释性实验 | 丙氨酸替换、同源建模、结合能分析 | 预测排名与结构结合能的相关系数为 0.796 |

### 1. P-M-T结合预测

训练集包含 29,342 个正三元组，测试集包含 551 个正三元组。

UniPMT结果：

- ROC-AUC：约 96%；
- PR-AUC：约 72%。

相较基线：

- ROC-AUC 至少提高约 4 个百分点；
- PR-AUC 至少提高约 15 个百分点。

PR-AUC 的提升尤其重要，因为该任务负样本多、类别不平衡。

### 2. 新抗原困难测试集

作者构建了更困难的测试方式：

- 固定 peptide 和 MHC；
- 只替换 TCR；
- 候选 TCR 也限制为已知新抗原特异性 TCR。

这样可避免模型仅依赖“P-M 不结合”就轻易判断三元组为负。

结果：

- UniPMT：ROC-AUC 0.7214，PR-AUC 0.2836；
- pMTnet：ROC-AUC 0.6020，PR-AUC 0.2278；
- TEIM：ROC-AUC 0.6328，PR-AUC 0.1556。

不过，该测试集只有 12 个正样本和 96 个负样本，因此结果的不确定性较高。

### 3. P-M预测

在 156,844 个 P-M 样本上进行五折实验。

UniPMT：

- ROC-AUC：93.55%；
- PR-AUC：84.35%。

相比 DeepAttentionPan，仅提升：

- ROC-AUC 约 0.17 个百分点；
- PR-AUC 约 0.16 个百分点。

这说明 UniPMT 在 P-M 任务上达到最优或接近最优，但多任务带来的增益有限。

### 4. P-T一般场景

在 TEIM 数据集上：

- ROC-AUC 约 0.78；
- PR-AUC 约 0.63。

相比已有模型：

- ROC-AUC 至少提高约 5 个百分点；
- PR-AUC 至少提高约 18 个百分点。

### 5. P-T zero-shot场景

测试集 peptide 在训练集中未出现，用于检验未见抗原泛化。

UniPMT：

- ROC-AUC 约 0.62；
- PR-AUC 约 0.84。

相比基线：

- ROC-AUC 至少提高约 2 个百分点；
- PR-AUC 至少提高约 20 个百分点。

### 6. 消融实验

作者训练了三个简化模型：

- P-M-T only；
- P-M only；
- P-T only。

结果显示：

- 对 P-M-T 预测，P-T 信息和 P-M-T 信息最重要；
- P-M-only 在 P-M 任务上已经接近完整模型，说明其他任务对 P-M 的反向帮助有限；
- 在 zero-shot P-T 中，P-T-only 表现较差，说明 P-M 和 P-M-T 信息有助于未见 peptide 泛化。

例如完整 UniPMT 在 P-M-T 上的 PR-AUC 为 72.36%，而 P-T-only 为 62.34%，P-M-T-only 为 33.72%。

### 7. 临床应用与克隆扩增分析

在 312 个经过实验验证的新抗原 P-M-T 三元组上：

- ROC-AUC：94.29%；
- PR-AUC：72.15%。

相较此前最好结果，PR-AUC 提高 17.62 个百分点。

作者还分析预测分数与 T 细胞克隆扩增比例的相关性：

- UniPMT 分数与克隆扩增的 Spearman 相关系数：0.2553；
- UMI count 与克隆扩增的相关系数：-0.07。

这说明预测分数对克隆扩增具有一定指示作用，但 0.2553 仍属于较弱相关，不能视为精确的克隆扩增预测。

### 8. 关键位点和结构验证

作者使用丙氨酸扫描：

- 将每个氨基酸依次替换成 alanine；
- 观察 embedding 和预测结果变化；
- 识别关键 peptide/TCR 位点。

随后在 1QRN 和 2PYE 等三维复合物结构上分析：

- 氢键距离；
- α 碳距离；
- 静电能；
- 范德华能；
- 总结合能。

此外，作者从模型预测结果中选择 15 个 P-M-T 三元组进行结构建模。预测排名与结合能之间的相关系数达到 0.796，表明较高的 UniPMT 分数通常对应更有利的结构结合能。

论文第4页图2集中展示三个预测任务的性能，第5～6页图3和图4展示临床验证、克隆扩增、关键位点与三维结构分析。`s42256-025-01002-0.pdf`

---

## Q5：有什么可以进一步探索的点？

### 一、论文作者明确提出的方向

#### 1. 增加训练数据

UniPMT依赖异质图学习。更多高质量 P-M-T 数据可以：

- 增加节点之间的可靠连接；
- 减少图中的稀疏性；
- 提高新 peptide 和新 TCR 的泛化能力。

#### 2. 加入TCR α链

当前模型只使用 TCR CDR3β。

但真实 TCR 通常由 α、β 两条链共同参与识别。未来可以加入：

- CDR3α；
- CDR3β；
- V/J 基因；
- α-β 配对关系；
- CDR1/CDR2 等区域。

#### 3. 引入三维结构

随着 pMHC-TCR 复合物结构数据增加，可以联合使用：

- 残基坐标；
- 接触图；
- 距离矩阵；
- 表面电荷；
- 氢键和范德华相互作用；
- 结构蛋白质语言模型。

`s42256-025-01002-0.pdf`

### 二、基于论文设计还可以继续探索的方向

以下属于对论文的进一步分析，而不是作者全部明确写出的结论。

#### 1. 改进负样本构造

论文大量使用随机打乱生成负样本，存在两个风险：

- 随机负样本过于容易；
- 某些被标记为负的组合实际上可能尚未实验验证，而不是真正不结合。

可以采用：

- hard negative mining；
- 同一 P-M 下替换相似 TCR；
- 同一 TCR 下替换相似 peptide；
- 实验验证的 non-binder；
- 正例-未标注学习；
- 带噪标签学习。

#### 2. 使用更严格的数据划分

除了随机划分和 peptide zero-shot，还应测试：

- unseen peptide；
- unseen TCR；
- unseen MHC allele；
- peptide 与 TCR 同时未见；
- 按患者划分；
- 按疾病或病毒划分；
- 按序列相似性聚类后划分。

否则训练集和测试集中的近似序列可能使性能偏高。

#### 3. 增加生物一致性约束

可以在三个任务之间添加一致性损失。例如：

- 一个 P-M-T 三元组预测为强结合时，其 P-M 分数不应很低；
- P-T 分数应与对不同 MHC 条件下 P-M-T 分数的聚合结果一致；
- 改变无关 MHC 时，不应无规律地大幅改变 P-T 预测；
- 对同一 pMHC，相似且关键位点一致的 TCR 应保持局部预测连续性。

这类约束可以减少三个预测头之间互相矛盾的结果。

#### 4. 从节点级图学习转向残基级交互

当前方法较依赖整体 embedding 和图邻居信息。未来可进一步建立：

- peptide residue-MHC residue cross-attention；
- pMHC-TCR residue cross-attention；
- 三方残基交互图；
- 接触位点监督；
- 多尺度序列-结构联合表示。

这样有机会更直接地解释“哪个 peptide 位点与哪个 TCR 位点发生作用”。

#### 5. 加强外部和前瞻性验证

目前主要使用整理后的公开数据集和结构模拟。更强的验证方式包括：

- 独立实验室数据；
- 不同测序平台的数据；
- 不同癌种或病毒数据；
- prospective wet-lab validation；
- 用模型筛选候选组合，再进行 tetramer、SPR 或细胞功能实验。

#### 6. 研究概率校准和不确定性

ROC-AUC 或 PR-AUC 较高，不代表输出的 0.8 就一定对应 80% 结合概率。

可增加：

- calibration curve；
- expected calibration error；
- temperature scaling；
- deep ensemble；
- conformal prediction；
- 高置信和低置信候选区分。

这对于临床候选筛选尤其重要。

---

## Q6：总结一下论文的主要内容

这篇论文提出了 **UniPMT**，一个面向 peptide、MHC-I 和 TCR CDR3β 特异性结合预测的统一深度学习框架。

其核心思想是：P-M、P-T 和 P-M-T 并不是三个相互独立的问题，应在同一个模型中联合学习。

模型首先使用 ESM 和 MHC pseudo-sequence 获得 P、M、T 的初始表示，再把三类分子及其结合关系构造成异质图，通过 GraphSAGE 聚合图邻居信息。随后，模型利用 DMF、多任务学习和对比学习，同时完成 P-M、P-T 和 P-M-T 预测。

实验表明，UniPMT在三个任务上均达到或接近最优水平，其中提升最明显的是：

- P-M-T 预测；
- P-T 一般预测；
- 未见 peptide 的 P-T zero-shot 预测；
- 新抗原特异性 TCR 识别。

论文还通过 T 细胞克隆扩增、丙氨酸扫描、关键残基分析和三维结合能模拟，验证了模型的临床应用潜力与一定的可解释性。

它最主要的贡献可以概括为：

> **将原本分散的 P-M、P-T 和 P-M-T 数据、图结构及预测任务统一起来，利用多任务间的生物学关联，提高数据稀缺条件下的特异性结合预测能力。**

但模型仍受限于随机负采样、TCR β 链单链输入、结构信息不足，以及独立临床验证规模有限等问题。`s42256-025-01002-0.pdf`

## Prompt:
2026/8/4 15:54:30

GraphSAGE图神经网络是什么，在本论文中起到什么作用

## Response:
2026/8/4 15:54:54

> **解释了GraphSAGE及其在UniPMT中的模块作用**
>
> 
>
> 思考了 21s

## 一句话理解

**GraphSAGE是一种通过聚合邻居节点信息来更新节点表示的图神经网络。**

在这篇论文中，它把原本彼此独立的 peptide、MHC 和 TCR 序列表示，转换成包含**已知结合关系上下文**的表示，再交给后续多任务模块预测 P-M、P-T 和 P-M-T 是否结合。

可以把它理解为：

> ESM负责回答“这个序列本身长什么样”，GraphSAGE负责补充“这个序列过去与哪些分子发生过结合”。

---

# 一、GraphSAGE是什么？

GraphSAGE全称是：

> **Graph Sample and Aggregate**

它属于图神经网络中的一种**邻居聚合模型**。

对于图中的一个节点，GraphSAGE不会只使用节点自身的特征，而是执行三个步骤：

1. 找到该节点的邻居；
2. 聚合邻居的特征；
3. 使用神经网络生成新的节点表示。

例如，某个 peptide 节点连接了多个 MHC 和 TCR：

```text
           MHC-1
             │
TCR-1 ── Peptide-A ── MHC-2
             │
           TCR-2
```

普通序列模型只能看到 Peptide-A 自身的氨基酸序列。

GraphSAGE还能看到：

- Peptide-A 与哪些 MHC 有结合关系；
- Peptide-A 与哪些 TCR 有结合关系；
- 这些 MHC、TCR 分别还与哪些 peptide 相关。

因此，更新后的 Peptide-A 表示不再只是一个序列 embedding，而是一个包含图关系的 embedding。

---

# 二、GraphSAGE如何更新节点表示？

论文给出的更新方式是：

$$
h_{n_i}^{(l+1)}
=
\operatorname{ReLU}
\left(
W^{(l)}
\cdot
\operatorname{mean}
\left(
\left\{
h_{n_j}^{(l)}
\mid
n_j\in\operatorname{Neighbours}(n_i)
\right\}
\right)
\right)
$$

其中：

- $n_i$：当前需要更新的节点；
- $n_j$：当前节点的邻居；
- $h_{n_j}^{(l)}$：第 $l$ 层时邻居节点的表示；
- $\operatorname{mean}$：对邻居 embedding 求平均；
- $W^{(l)}$：可学习的线性变换矩阵；
- ReLU：非线性激活函数；
- $h_{n_i}^{(l+1)}$：聚合邻居信息后的新表示。

论文采用的是一种较直接的 **mean aggregation**：把邻居向量取平均，再经过线性变换和 ReLU，生成更新后的节点表示。`s42256-025-01002-0.pdf`

直观上相当于：

```text
当前节点的新表示
    =
邻居节点信息的平均
    +
可学习的特征变换
```

需要注意，经典 GraphSAGE 常见实现还会把“当前节点自身表示”和“邻居聚合表示”拼接或相加；而论文正文给出的公式主要展示了邻居均值聚合。论文没有在正文公式中进一步展开更细的实现差异。

---

# 三、UniPMT中的图是怎样构建的？

UniPMT把三类生物实体作为节点：

| 节点类型 | 含义 | 初始表示 |
|---|---|---|
| P节点 | Peptide | ESM embedding |
| M节点 | MHC | MHC pseudo-sequence |
| T节点 | TCR CDR3β | ESM embedding |

图中包含三类生物关系：

- P-M结合关系；
- P-T结合关系；
- P-M-T结合关系。

可以概括为：

```text
Peptide节点 ── P-M关系 ── MHC节点
     │
     ├──── P-T关系 ───── TCR节点
     │
     └──── P-M-T三元关系 ── MHC、TCR
```

论文把它称为异质图，因为图中存在：

- 多种节点类型；
- 多种关系类型；
- 二元和三元生物关系。

作者先构造图，再使用 GraphSAGE 进行 message passing，得到更新后的 peptide、MHC 和 TCR 节点表示。之后，这些表示才进入 DMF 和多任务预测模块。这个流程在论文第3页图1中表现为：

```text
图构建
  ↓
GraphSAGE图结构学习
  ↓
更新后的P、M、T embedding
  ↓
P-M、P-M-T、P-T统一预测
```

---

# 四、GraphSAGE在论文中起到什么作用？

## 作用1：把序列表示变成关系感知表示

GraphSAGE之前：

```text
Peptide embedding = Peptide自身序列信息
TCR embedding     = TCR自身序列信息
MHC embedding     = MHC pseudo-sequence信息
```

GraphSAGE之后：

```text
Peptide embedding
= 自身序列信息
+ 相关MHC信息
+ 相关TCR信息
+ 图中的结合关系信息
```

因此，更新后的 embedding 不只是描述分子自身，还描述它在整个免疫结合网络中的位置。

例如，两个 peptide 的序列可能并不完全相似，但如果它们：

- 都能被相似的 HLA 呈递；
- 都被一组相似 TCR 识别；

那么经过图信息传播后，它们的表示可能会更接近。

---

## 作用2：在P-M、P-T和P-M-T任务之间传递信息

UniPMT的核心思想是三个任务不应完全分开训练。

假设某个 peptide-TCR 对的直接训练数据很少，但已知：

- 该 peptide 能与某个 MHC 结合；
- 该 MHC 参与了一些已知 P-M-T 关系；
- 该 TCR 也识别过其他相似 peptide。

GraphSAGE就可以沿着图传播这些信息：

```text
P-M数据
  ↓
更新Peptide和MHC表示
  ↓
辅助P-M-T预测
  ↓
进一步辅助P-T预测
```

所以 GraphSAGE 实际上是三个任务共享信息的“桥梁”。

如果没有图神经网络，三个任务即使同时训练，也可能只是共享部分模型参数；有了 GraphSAGE，它们还能通过图中的节点与边直接交换关系信息。

---

## 作用3：缓解P-M-T数据稀缺

高质量 P-M-T 三元组数据相对有限，但 P-M 数据和 P-T 数据相对更多。

GraphSAGE允许一个 P-M-T 节点表示间接吸收：

- 大规模 P-M 数据中的抗原呈递规律；
- P-T 数据中的 TCR识别规律；
- P-M-T 数据中的完整结合规律。

因此，即使直接三元组训练样本较少，模型仍可以从二元关系中获得辅助监督。

论文的消融实验也支持这一点：完整 UniPMT 在 P-M-T 任务上的 PR-AUC 为 72.36%，而只使用单一关系的模型明显下降；其中 P-M-T-only 的 PR-AUC 为 33.72%，P-T-only 为 62.34%。这说明联合图关系为三元结合预测提供了额外信息。`s42256-025-01002-0.pdf`

---

## 作用4：提高zero-shot泛化能力

在 P-T zero-shot 场景中，测试 peptide 在训练集中没有出现。

如果只依赖 peptide-TCR 已知配对，遇到新 peptide 时，模型很难判断。

GraphSAGE可以借助新 peptide 的其他关系进行推断，例如：

```text
新Peptide
   │
   ├── 与某些MHC序列或呈递模式相似
   │
   └── 与图中已有Peptide具有相似的序列表征
             ↓
       间接获得关系信息
             ↓
       预测其可能结合的TCR
```

论文中完整模型在 P-T zero-shot 数据集上达到约：

- ROC-AUC：0.62；
- PR-AUC：0.84。

消融结果中，P-T-only 模型的 ROC-AUC 只有 44.74%，而加入 P-M 和 P-M-T 信息后明显提高。作者据此认为，其他关系中包含对未见 peptide 的 P-T 预测有帮助的隐藏信息。`s42256-025-01002-0.pdf`

---

## 作用5：为后续DMF预测模块提供输入

GraphSAGE本身主要负责**表示学习**，并不直接完成最终的三类结合预测。

完整流程是：

```text
原始序列
   ↓
ESM / MHC pseudo-sequence
   ↓
初始节点embedding
   ↓
GraphSAGE邻居聚合
   ↓
关系感知的节点embedding
   ↓
DMF和多任务预测头
   ↓
P-M、P-T、P-M-T结合概率
```

例如，在 P-M 预测中，GraphSAGE输出的 peptide 表示 $h_p$ 和 MHC 表示 $h_m$ 会被输入神经网络：

$$
v_{pm}=f_{pm}(h_p,h_m)
$$

随后经过线性映射和 sigmoid，得到 P-M 结合概率：

$$
P_{pm}=\sigma(wv_{pm})
$$

因此，GraphSAGE承担的是：

> **学习更有信息量的 P、M、T 表示。**

DMF和预测头承担的是：

> **根据这些表示计算最终结合概率。**

`s42256-025-01002-0.pdf`

---

# 五、用一个具体例子理解

假设图中已有以下关系：

```text
Peptide-A ── MHC-1
Peptide-A ── TCR-1
Peptide-B ── MHC-1
Peptide-B ── TCR-2
```

现在需要判断：

```text
Peptide-A是否与TCR-2结合？
```

只看序列的模型会比较 Peptide-A 和 TCR-2 的序列特征。

GraphSAGE还会发现：

1. Peptide-A 和 Peptide-B 都连接 MHC-1；
2. Peptide-B 已知能与 TCR-2 结合；
3. 因此，Peptide-A 和 Peptide-B 可能共享某些呈递或识别模式。

信息传播路径可以表示为：

```text
Peptide-A
   ↓
MHC-1
   ↓
Peptide-B
   ↓
TCR-2
```

经过多层图传播后，Peptide-A 的表示可能间接包含 TCR-2 附近的关系特征，从而辅助判断未知 P-T 关系。

这不代表模型简单地认为“共享 MHC 就一定共享 TCR”，而是将这种图结构模式作为一种可学习信号，与序列 embedding 共同参与预测。

---

# 六、GraphSAGE不负责什么？

需要区分 GraphSAGE 与序列交互模型的能力。

UniPMT中的 GraphSAGE主要处理的是：

- 哪些 peptide 与哪些 MHC 相连；
- 哪些 peptide 与哪些 TCR 相连；
- 图中节点之间的整体关系。

它并不直接精细建模：

- peptide第几个残基与MHC第几个残基接触；
- peptide与TCR之间的残基级交叉注意力；
- 三维空间距离；
- 氢键和静电相互作用；
- TCR α链与β链的联合识别。

因此，它更接近：

> **分子级关系图学习**

而不是：

> **残基级结构交互学习**

这也是论文可以进一步改进的方向：在 GraphSAGE 的全局关系信息之外，加入 peptide-MHC、pMHC-TCR 的残基级 cross-attention 或三维结构图。

---

# 七、核心总结

GraphSAGE在UniPMT中的作用可以概括为四句话：

1. **把 peptide、MHC 和 TCR 构造成一个统一的异质关系图。**
2. **通过邻居聚合，让每个节点吸收与其相关的结合信息。**
3. **将单纯的序列 embedding 更新为包含生物关系上下文的 embedding。**
4. **为后续 P-M、P-T 和 P-M-T 多任务预测提供共享表示。**

所以，GraphSAGE不是UniPMT最后的分类器，而是整个模型中的**关系信息编码器**：

$$
\boxed{
\text{序列信息}
\xrightarrow{\text{ESM}}
\text{初始表示}
\xrightarrow{\text{GraphSAGE}}
\text{关系感知表示}
\xrightarrow{\text{DMF与多任务学习}}
\text{结合概率}
}
$$

从模型设计角度看，UniPMT性能提升的关键不是单独使用 GraphSAGE，而是将 **ESM序列表示、GraphSAGE关系传播和多任务学习** 结合起来。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)