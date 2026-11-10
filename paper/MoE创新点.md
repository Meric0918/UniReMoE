# 分支 · MoE模型架构优化

**User:** Anonymous  
**Created:** 11/6/2026 15:15:46  
**Updated:** 11/6/2026 16:01:05  
**Exported:** 11/6/2026 16:04:57  
**Link:** [https://chatgpt.com/c/6aed7e97-b288-83ec-a18f-ae7a0f753d6e](https://chatgpt.com/c/6aed7e97-b288-83ec-a18f-ae7a0f753d6e)  

## Prompt:
2026/11/6 15:16:51

我已经按照版本二实现了上述MoE架构，我需要在论文method中新增关于MoE的创新点，帮我分析至少3点

## Response:
2026/11/6 15:17:03 · gpt-6-pro

我会结合你已经实现的版本二，提炼适合写进 Method 的创新点，并对照 MMoE、PLE 和相关 TCR 预测方法，明确各点的贡献边界与所需的验证实验。

目前最值得强调的是三个相互配合的设计：共享专家如何处理不同关系、二元特征如何传入三元分支，以及如何保留肽残基的对应关系。已有工作分别覆盖了三任务联合预测、共享与专属专家、TCR 领域的 MoE，因此论文需要把贡献落到你这套具体的信息流和路由机制上。

进一步对照后，版本二最清楚的新机制是：在 HLA 条件下，按肽位置选择性地把 PT 特征送入 PHT，再进行 PHT 专家混合。我会将它作为重点，同时区分“结构上支持任务间迁移”和“实验已证明三个任务互相提升”，避免论文结论超出证据。

三个贡献可以分别围绕“专家共享方式”“残基对齐的层级表示”“HLA 条件门控迁移”展开。需要特别注明：PH→PHT 连接在原模型中就已存在，新增贡献应写成对这条层级路径进行专家增强。相应消融也要区分参数扩容、前向特征复用和反向联合训练各自的收益。

> **已搜索 39 个网站**
>
> 
>
> **已搜索 28 个网站**
>
> 
>
> **已搜索 12 个网站**
>
> 
>
> 思考了 4m 1s

**按照你已经实现的版本二，我建议将 MoE 相关内容组织成 3 个相互支撑的方法贡献：关系感知的共享-专属专家建模、肽残基对齐的分层交互专家化，以及 HLA 条件化的 PT→PHT 门控迁移。**

其中，**第一点是 MoE 模块的核心设计，第三点是版本二最明确的新增机制，第二点则解释了这套 MoE 为什么适合你的分子交互任务。**

需要先明确：这三点可以作为论文中的**方法贡献**，但不能直接等同于“三项全新的通用 MoE 算法”。更有说服力的写法，是说明你针对 PH、PT、PHT 的特殊关系，如何设计专家共享、交互表示和信息传递，并用消融实验支持相应主张。

下面以你前面描述的版本二为准进行分析。

---

# 一、先明确：你的创新应该落在哪个层面？

你的模型具有以下几个关键特征：

| 结构特征 | 具体实现 | 可以提炼的研究问题 |
|---|---|---|
| 三任务专家协同 | PH、PT、PHT 共用共享专家，同时保留分支专家 | 不同分子关系之间，哪些知识应该共享，哪些应该保留差异？ |
| 交互后、池化前的 MoE | 专家处理 Cross-Attention 输出的残基表示 | 如何让专家直接建模分子交互，而不仅处理单条序列的特征？ |
| PH→PHT 层级路径 | PHT 在经过专家增强的 PH 表示上继续整合 TCR | 二元交互知识如何参与三元识别？ |
| PT→PHT 门控路径 | 在 HLA 条件下，将 PT 表示注入 PHT | 如何利用 PT 的补充信息，同时保留具体 HLA 背景？ |
| 肽残基位置对齐 | 各路径保留相同的 $L_P$ 维度 | 如何在不同关系之间进行具有位置对应的信息传递？ |

这里有几个必须考虑的相关工作：

| 相关工作 | 已经覆盖的思想 | 你需要进一步说明的区别 |
|---|---|---|
| **MMoE** | 跨任务共享专家、任务独立门控。([research.google](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/?utm_source=chatgpt.com)) | 你的专家处理什么交互表示，如何适配输入实体组合不同的任务 |
| **PLE** | 显式区分共享成分与任务专属成分，并进行渐进式信息路由。([recsys.acm.org](https://recsys.acm.org/recsys20/session-4/?utm_source=chatgpt.com)) | 你的关系层级、专家部署位置和二元→三元信息流 |
| **NeoMUST** | 已在抗原呈递与肽-MHC 亲和力任务中使用共享专家、任务专家和门控。([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC10828515/?utm_source=chatgpt.com)) | 三种不同输入关系之间的交互级共享，以及面向 PHT 的定向迁移 |
| **UniPMT** | 已统一建模 PH、PT、PHT，并复用二元关系表示支持三元预测。([nature.com](https://www.nature.com/articles/s42256-025-01002-0?utm_source=chatgpt.com)) | 肽残基对齐的表示、独立 PT 分支、条件化迁移门控和交互层 MoE |
| **MoE-TCR** | 已将顶层稀疏 MoE 用于 TCR-epitope 预测，以完整的 pan-specific 预测模型作为专家。([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0031320326007028?utm_source=chatgpt.com)) | 你的专家位于交互网络内部，并服务于三任务的特征共享与层级融合 |

因此，你的论文主线可以定位为：

**针对 PH、PT、PHT 输入条件不同、任务依赖不对称的问题，构建以肽残基为共同索引的交互级多任务 MoE，通过选择性专家共享和条件化二元→三元迁移组织信息。**

这一定位能够把下面三点连成一个完整的方法。

---

# 二、创新点一：关系感知的共享-专属交互专家建模

建议的小节标题：

**Relation-aware Shared-Specific Interaction Experts**

中文可以写为：

**关系感知的共享与专属交互专家建模**

## 2.1 这一点解决什么问题？

PH、PT、PHT 虽然共享部分输入，但学习目标和输入条件不同：

- PH 根据 peptide 和 HLA 判断二元关系。
- PT 根据 peptide 和 TCR 判断未显式指定 HLA 的关系。
- PHT 根据 peptide、HLA 和 TCR 判断具体三元关系。

因此，它们既存在可迁移的信息，也存在任务差异。

从建模角度看，有两种需要权衡的情况：

**共享过少**时，每个任务分别学习交互模式，辅助任务的监督难以有效影响其他任务。

**共享过多**时，不同任务被迫使用相同的特征变换，模型缺少保留关系差异的空间。

你的 MoE 在这两者之间提供了一个可学习的调节机制：

> 每个任务可以根据当前样本，自适应地组合共享专家和自身的关系专属专家。

这里更值得强调的细节是：**专家接收的已经是分子交互表示。**

例如：

$$
U_{PH}=\operatorname{Cross}_{PH}(X_P,X_H)
$$

$$
U_{PT}=\operatorname{Cross}_{PT}(X_P,X_T)
$$

因此，共享专家接触到的是不同关系条件下的肽表示，而不是简单复用一层序列编码器。

## 2.2 如何在 Method 中形式化？

设关系集合为：

$$
\mathcal R=\{PH,PT,PHT\}
$$

对于关系 $r$，定义可访问的专家集合：

$$
\mathcal E_r=\mathcal E^{S}\cup\mathcal E^{r}
$$

其中：

- $\mathcal E^{S}$：所有任务共享的专家；
- $\mathcal E^{r}$：关系 $r$ 的专属专家。

按照前面的配置：

$$
|\mathcal E^{S}|=2,\qquad
|\mathcal E^{r}|=2
$$

每个任务的路由器计算：

$$
g_r(c_r)
=
\operatorname{Softmax}
\left(
\frac{f_r^{\mathrm{gate}}(c_r)}{\tau}
\right)
$$

设 $V_r$ 为进入该分支 MoE 的表示，则：

$$
Z_r
=
V_r+
\operatorname{Dropout}
\left[
\sum_{e\in\mathcal E_r}
g_{r,e}(c_r)
E_e\!\left(\operatorname{LN}_r(V_r)\right)
\right]
$$

这里：

- PH 的 $V_r$ 是 $U_{PH}$；
- PT 的 $V_r$ 是 $U_{PT}$；
- PHT 的 $V_r$ 是融合 PT 信息后的三元交互表示。

**共享专家必须是同一组参数，但它们在不同任务中处理的输入不同，得到的输出也不同。**

例如：

$$
E^S_1(U_{PH}),\quad
E^S_1(U_{PT}),\quad
E^S_1(\widetilde U_{PHT})
$$

三者共享的是 $E^S_1$ 的函数参数，而不是共享某一份固定的输出特征。

这个区别应当在论文中说明清楚。

## 2.3 哪些地方体现了针对任务的设计？

### 第一，专家共享发生在“交互表示层”

同一个共享专家在不同分子关系的表示上接受训练，有机会学习能够跨关系复用的特征变换。

与此同时，关系专属专家保留各任务独立调整表示的能力。

因此，你可以讨论的是：

**跨关系知识共享与关系特化之间的自适应平衡。**

不能仅根据网络结构，就认定共享专家已经学习到了某种具体生物作用力；那需要额外证据。

### 第二，路由器遵守任务输入边界

你的路由上下文应满足：

| 任务 | 路由器可以使用的信息 |
|---|---|
| PH | P、H 和 PH 交互表示 |
| PT | P、T 和 PT 交互表示 |
| PHT | P、H、T 和三元交互表示 |

这样，PH 和 PT 仍然可以只使用各自的二元输入完成预测。

共享参数允许任务间迁移知识，但不会要求 PH 推理时额外输入 TCR，也不会要求 PT 推理时额外输入 HLA。

### 第三，路由随样本变化

同一个任务中，不同样本可以对共享专家和专属专家分配不同权重。

因此，模型不仅学习“这个任务应该共享多少”，还学习“这个任务的当前样本应该共享多少”。

这是你的可学习路由需要验证的实际价值。

## 2.4 这一点的创新边界

共享专家、专属专家和任务门控本身已有先例。因此，建议将贡献表述集中在：

**针对 PH、PT、PHT 的异构输入关系，在交互后的残基表示上实施受任务输入约束的选择性专家共享。**

“缓解负迁移”可以作为设计目标；只有当完整模型相较于共享编码器基线、全部共享专家基线等表现更好时，才能进一步写成实验结论。

---

# 三、创新点二：肽残基对齐的分层交互专家化

建议的小节标题：

**Peptide-aligned Hierarchical Interaction Expertization**

中文可以写为：

**肽残基对齐的分层交互专家建模**

这一点回答的是：**为什么专家放在这些位置，以及不同关系的表示为什么能够相互传递。**

## 3.1 你的模型具有一个很有价值的共同索引

由于 PH、PT 都以 peptide 作为 Query：

$$
U_{PH}
=
\operatorname{Cross}_{PH}(X_P,X_H)
\in\mathbb R^{B\times L_P\times d}
$$

$$
U_{PT}
=
\operatorname{Cross}_{PT}(X_P,X_T)
\in\mathbb R^{B\times L_P\times d}
$$

经过 MoE 后：

$$
Z_{PH},Z_{PT}
\in\mathbb R^{B\times L_P\times d}
$$

二者的第 $i$ 个位置，都对应同一条肽的第 $i$ 个残基，只是包含的交互上下文不同。

可以这样理解：

| 表示 | 第 $i$ 个位置的含义 |
|---|---|
| $X_{P,i}$ | 肽的第 $i$ 个残基的序列表示 |
| $Z_{PH,i}$ | 融合 HLA 信息并经过专家变换的该残基表示 |
| $Z_{PT,i}$ | 融合 TCR 信息并经过专家变换的该残基表示 |
| $U_{PHT,i}$ | 在 PH 表示基础上进一步整合 TCR 的该残基表示 |

**这种对齐是模型中的残基索引对应，不代表已经获得了真实三维结构中的接触对齐。**

但它确实为跨关系的信息融合提供了明确的位置依据。

## 3.2 为什么 MoE 放在 Cross-Attention 后、Pooling 前有意义？

这让每个专家处理：

$$
E_j(U_{r,i})
$$

其中，$U_{r,i}$ 已经包含其他分子的上下文。

这样，专家变换发生在两个重要操作之间：

$$
\text{分子间交互}
\rightarrow
\text{专家变换}
\rightarrow
\text{位置聚合}
$$

模型在聚合成单个向量之前，还能够保留和变换每个肽位置上的信息。

相较于先 Pooling，再在全局向量上应用 MoE，你的设计保留了显式的逐位置访问能力。这使后续的 PH→PHT 和 PT→PHT 可以利用未池化的交互表示。

这里应强调的是：

**专家处理的粒度与跨关系迁移的粒度保持一致。**

## 3.3 PH→PHT 路径让专家参与不同交互阶段

你的 PH 路径先得到：

$$
Z_{PH}
=
\operatorname{MoE}_{PH}
\left(
\operatorname{Cross}_{PH}(X_P,X_H)
\right)
$$

随后：

$$
U_{PHT}
=
\operatorname{Cross}_{PHT}(Z_{PH},X_T)
$$

如果展开 Query 残差，其核心形式为：

$$
U_{PHT}
=
Z_{PH}
+
\operatorname{MHA}_{PHT}
\left(
\operatorname{LN}(Z_{PH}),
\operatorname{LN}(X_T),
\operatorname{LN}(X_T)
\right)
$$

这里存在一个有意义的建模顺序：

1. 建立 PH 交互表示。
2. 通过共享与专属专家对 PH 表示进行变换。
3. 在保留 PH 信息的基础上整合 TCR。
4. 在三元交互阶段再次进行专家变换。

所以，你的专家分布在具有不同关系语义的交互阶段中。

这种结构可以作为一种**受分子关系启发的归纳偏置**：模型被引导着先组织 PH 信息，再用于三元预测。

## 3.4 如何避免把已有结构重新包装为创新？

这里需要分清两个参照对象。

**相对于你自己的原始模型：**

如果原模型已经包含 PH→PHT，以及 peptide Query，那么这两项本身不能算“加入 MoE 后的新设计”。

新增部分应当明确为：

**在原有肽对齐的层级交互路径中，加入跨阶段共享的专家变换，使 PHT 接收经过关系专家处理的二元表示。**

**相对于领域现有方法：**

UniPMT 已经复用 PH 对应的向量表示参与三元预测，因此“首次利用二元特征预测三元关系”也不成立。你的区别需要落到**未池化的肽残基表示、交互阶段内的专家变换，以及后面的条件化 PT 迁移**。([nature.com](https://www.nature.com/articles/s42256-025-01002-0?utm_source=chatgpt.com))

因此，这一点更适合作为完整架构的结构性贡献。其独立创新强度通常弱于版本二新增的迁移门控，但它能够很好地解释整个设计为什么成立。

## 3.5 一个容易写错的术语

如果你采用的是：

$$
g_r\in\mathbb R^{B\times4}
$$

那么你的实现是：

**样本级路由、残基级专家计算。**

同一个样本的不同肽位置使用同一组专家比例，但每个位置的输入不同，因此专家输出不同。

只有当：

$$
g_r\in\mathbb R^{B\times L_P\times4}
$$

时，才能称为“残基级专家路由”。

这个术语区分建议在 Method 中写准确。

---

# 四、创新点三：HLA 条件化的 PT→PHT 门控迁移

建议的小节标题：

**HLA-conditioned Transfer of Peptide-TCR Representations**

中文可以写为：

**HLA 条件化的肽-TCR 表征迁移**

**这是我最建议你重点展开的创新点。**它既是版本二相对版本一的明确变化，也有清楚的动机、公式和消融方式。

## 4.1 为什么不能只写“加入 PT 特征”？

因为 PT 和 PHT 的条件不同：

$$
PT:(P,T)\rightarrow y_{PT}
$$

$$
PHT:(P,H,T)\rightarrow y_{PHT}
$$

PT 分支没有显式接收 HLA，它学习到的统计模式受到训练数据中 HLA 背景的影响。因此，将 PT 特征迁移到某个具体的 PHT 样本时，合理的问题是：

**当前 HLA 条件下，PT 表征中的哪些信息值得使用，应该使用多少？**

由此，你的融合模块可以被解释为一种**条件化的信息迁移机制**。

同时，PT 更适合被表述为“未显式指定 HLA 的预测任务”，不能直接解释成不依赖 HLA 的独立生物识别过程。

## 4.2 门控迁移的完整定义

先将 PT 表示映射到适合注入的空间：

$$
R_{PT,i}
=
W_{PT}\operatorname{LN}(Z_{PT,i})
$$

对于每个肽位置 $i$，计算：

$$
\beta_i
=
\sigma
\left[
f_{\mathrm{fusion}}
\left(
[
\operatorname{LN}(U_{PHT,i});
\operatorname{LN}(Z_{PT,i});
\overline X_H
]
\right)
\right]
$$

其中：

- $U_{PHT,i}$：当前三元主路径的表示；
- $Z_{PT,i}$：对应位置的 PT 表示；
- $\overline X_H$：HLA 的汇总表示；
- $\beta_i$：当前肽位置的迁移强度。

然后进行残差注入：

$$
\widetilde U_{PHT,i}
=
U_{PHT,i}
+
\beta_iR_{PT,i}
$$

最后：

$$
Z_{PHT}
=
\operatorname{MoE}_{PHT}
\left(
\widetilde U_{PHT},
c_{PHT}
\right)
$$

这里，PH→PHT 保留为三元建模的主路径，PT 提供经过条件选择的补充表示。

## 4.3 这一设计为什么比“多加一条连接”更有研究价值？

### 第一，它让迁移决策依赖目标任务的上下文

门控同时读取：

$$
U_{PHT,i},\quad Z_{PT,i},\quad \overline X_H
$$

因此，PT 信息的使用方式由当前三元预测需求决定。

同一个 PT 表示，在不同 HLA 条件下，可以获得不同的注入强度。

这是结构允许的行为；实际是否学到了有意义的条件差异，需要实验检验。

### 第二，它保留逐位置的迁移能力

若：

$$
\beta\in\mathbb R^{B\times L_P\times1}
$$

模型可以对不同肽位置分别控制 PT 信息的注入。

这比为整条肽使用一个统一融合系数具有更细的控制粒度，也利用了第二个贡献中的残基对应关系。

### 第三，它包含两种功能不同的门控

| 门控 | 控制对象 | 作用 |
|---|---|---|
| 迁移门控 $\beta_i$ | PT→PHT 的特征通路 | 决定当前肽位置接收多少 PT 信息 |
| 专家门控 $g_{PHT,j}$ | PHT 的专家组合 | 决定融合后的三元表示如何被变换 |

两者分别控制**跨关系信息传递**和**关系内部的专家处理**。

这可以作为一个完整的协同门控机制展开，但没有必要再拆成两项独立创新。

### 第四，它使用中间特征，不要求三个预测概率满足人为的乘积关系

你的设计并不要求：

$$
\hat y_{PHT}
=
\hat y_{PH}\hat y_{PT}
$$

而是让 PH、PT 的中间表示参与 PHT 的学习。

这样，模型能够利用辅助任务的信息，同时由 PHT 自己的监督目标决定最终预测。

## 4.4 论文中应当控制哪些结论？

这套设计可以支持以下主张：

- 迁移由 HLA 和三元交互上下文共同控制。
- PT 信息在肽位置上被选择性注入。
- 跨关系迁移与专家选择共同参与三元表示学习。

但仅根据架构，还不能断言：

- 门控保证了模型充分利用 HLA。
- $\beta_i$ 是某个残基真实生物贡献的比例。
- 模型已经显式建模了所有 HLA-TCR 接触。
- 模型已经满足 $PHT\Rightarrow PH\land PT$ 的逻辑约束。

尤其最后一点，**特征依赖关系不会自动变成输出层面的逻辑约束**。如果你之后加入一致性损失，需要另行定义，并核查三个任务标签的语义。

---

# 五、三任务联合训练应该怎样融入这些创新点？

你提到的“共享 encoder、共享专家，使三个任务相互促进”，可以成为第一点的重要机制解释，并在 Training 小节中展开。

不过，建议区分以下三种作用。

## 5.1 参数共享：不同任务更新同一组参数

对于共享专家参数 $\theta_S$：

$$
\nabla_{\theta_S}\mathcal L
=
\lambda_{PH}\nabla_{\theta_S}\mathcal L_{PH}
+
\lambda_{PT}\nabla_{\theta_S}\mathcal L_{PT}
+
\lambda_{PHT}\nabla_{\theta_S}\mathcal L_{PHT}
$$

这说明三个任务的监督可以共同塑造共享专家。

编码器共享也需要写具体：

- peptide 编码模块在三个任务间复用；
- HLA 编码模块在 PH、PHT 间复用；
- TCR 编码模块在 PT、PHT 间复用。

如果预训练编码器被冻结，那么联合训练更新的是可训练的投影层、交互层和专家，不能写成“联合优化了冻结的预训练编码器”。

## 5.2 表征复用：二元特征直接参与当前样本的三元预测

这是：

$$
Z_{PH}\rightarrow PHT
$$

以及：

$$
Z_{PT}\rightarrow PHT
$$

与参数共享不同，这两条路径让当前样本的二元交互结果直接参与三元计算。

因此，你的架构同时包含：

| 机制 | 共享或传递的对象 |
|---|---|
| 参数共享 | 编码器和共享专家的参数 |
| 表征迁移 | 当前样本的 $Z_{PH}$、$Z_{PT}$ |
| 联合监督 | 多个任务损失对可训练模块的更新 |

这三个层面应当分别说明。

## 5.3 梯度反馈：PHT 的监督能够更新二元交互模块

如果两条特征通路没有使用 `detach`，则 PHT 损失能够沿着它们反向传播：

$$
\nabla_{\theta_{PH}^{\mathrm{int}}}\mathcal L_{PHT}
=
\left(
\frac{\partial Z_{PH}}
{\partial\theta_{PH}^{\mathrm{int}}}
\right)^\top
\frac{\partial\mathcal L_{PHT}}
{\partial Z_{PH}}
$$

PT 路径同理。

这里的 $\theta_{PH}^{\mathrm{int}}$ 指 PH 的交互模块参数。

一个很容易忽略的细节是：

**PHT 损失可以更新被复用的 PH/PT 交互层和专家，但不会自动更新没有参与 PHT 计算的 PH/PT 独立分类头。**

同样，“分支专属专家”表示它只被相应分支直接调用，不代表它只接受该任务标签的监督。

## 5.4 “互相促进”应当怎样证明？

结构上存在迁移通路，只能说明模型**支持联合学习**。

要写“三个任务相互促进”，需要比较：

$$
\Delta_r
=
\operatorname{Metric}_r(\text{联合训练})
-
\operatorname{Metric}_r(\text{对应单任务训练})
$$

并分别报告：

$$
\Delta_{PH},\quad
\Delta_{PT},\quad
\Delta_{PHT}
$$

如果只有 PHT 提升，就应表述为“辅助任务改善了三元预测”。

如果三个任务都在可靠评估中提升，才有依据进一步讨论三任务协同收益。

交替训练或将三个任务的损失累积后更新，都可以实现联合学习，但二者的优化过程并不完全相同。Method 应写清你实际采用的采样和更新方式，不宜仅把“交替训练”作为新的优化算法贡献。

---

# 六、为了让这三点成立，最值得做哪些消融？

建议按照“一个主张对应一个对照”的方式组织实验。

| 要验证的主张 | 建议的对照实验 | 能回答的问题 |
|---|---|---|
| MoE 的收益超出单纯增加参数 | 完整模型 vs 参数量接近的普通 FFN，保持其他连接一致 | 提升是否只是网络变大？ |
| 共享专家提供额外的跨任务迁移 | 保留编码器共享，将共享专家改为各分支独立参数 | 收益是否来自交互专家共享？ |
| 专属专家有必要 | 共享＋专属专家 vs 全部共享专家，并尽量匹配容量 | 关系特化是否有价值？ |
| 样本自适应路由有必要 | 可学习路由 vs 均匀权重、仅随任务变化的固定权重 | 模型是否真的受益于样本条件路由？ |
| PH 专家增强对 PHT 有价值 | PHT 接收 $Z_{PH}$ vs 接收 MoE 前的 $U_{PH}$，其余尽量一致 | 二元专家变换是否帮助三元建模？ |
| 肽位置对应有价值 | 传递逐位置 $Z_{PT,i}$ vs 将 PT 池化后广播到各位置 | 局部对应是否优于仅传递全局信息？ |
| PT→PHT 迁移有价值 | 版本一 vs 版本二 | 新增 PT 路径是否有效？ |
| 门控迁移优于简单融合 | 无门控相加、拼接投影、可学习门控 | 提升是否来自选择性迁移？ |
| 显式 HLA 条件有价值 | 完整融合门控 vs 去除门控中的 $\overline X_H$ | HLA 上下文是否帮助迁移决策？ |
| 反向联合优化有价值 | 完整梯度 vs 对输入 PHT 的 $Z_{PH}$、$Z_{PT}$ 分别停止梯度 | 收益来自前向复用，还是也来自梯度反馈？ |

这里有两个实验解释上的细节。

**第一，去掉融合门控里的 $\overline X_H$，不等于移除模型中的 HLA。**主路径 $U_{PHT}$ 仍然包含 HLA 信息。这个消融只检验显式向融合门控提供 HLA 摘要的作用。

**第二，固定 P、T 后替换 HLA，可以检查模型是否依赖 HLA，但不能自动把替换后的样本当成真实阴性。**要判断这种依赖是否正确，仍需有标签的 HLA 限制性样本。

评估时，建议至少做到：

- PH、PT、PHT 分别报告 AUROC 和 AUPRC。
- 使用多个随机种子报告结果波动。
- 同时报告参数量和计算开销，特别是你的稠密 MoE 会计算所有候选专家。
- 将 cold-peptide 划分统一应用于三个任务：如果测试肽出现在 PH 或 PT 辅助训练集中，就不能称为严格的三任务 cold-peptide 测试。

如果论文要强调泛化能力，这些划分控制比随机划分上小幅提升更有说服力。

---

# 七、在 Method 中如何组织这三点？

建议按照实际前向计算顺序组织，而不是直接使用“创新点一、二、三”作为论文小节标题。

| Method 小节 | 主要内容 | 对应贡献 |
|---|---|---|
| **Problem formulation and sequence representation** | 定义 PH、PT、PHT 输入输出，说明序列编码和共享范围 | 问题设置 |
| **Peptide-aligned hierarchical interaction modeling** | 定义 PH、PT 交互、肽位置对应，以及 PH→PHT 主路径 | 创新点二 |
| **Relation-aware shared-specific experts** | 定义专家集合、门控上下文、残差专家混合及参数共享方式 | 创新点一 |
| **HLA-conditioned cross-relation transfer** | 定义 PT→PHT 的投影、逐位置门控、残差注入及后续 PHT MoE | 创新点三 |
| **Joint optimization** | 定义有效标签、任务损失、采样策略和梯度传播方式 | 三项贡献的训练基础 |

在 Introduction 的贡献列表中，可以分别突出以下内容：

1. **关系感知的交互专家共享：**在 PH、PT、PHT 不同输入关系的交互表示上，结合跨任务共享专家与关系专属专家，实现样本条件下的选择性知识共享。
2. **肽残基对齐的层级表示：**在二元与三元交互阶段保留共同的肽位置索引，使专家增强后的二元表示能够以未池化形式参与三元建模。
3. **HLA 条件化的表征迁移：**通过逐位置迁移门控，将独立 PT 分支的信息选择性地注入 PHT，并与三元专家路由共同完成表示整合。

**我建议将第 1 点和第 3 点作为论文最主要的方法贡献，将第 2 点作为连接二者的结构性贡献。**尤其要把版本二的价值写具体：它增加了一条由 HLA 与当前三元上下文控制的 PT 信息通路；对应实验应证明，这种条件化迁移相较于没有迁移、简单融合和停止梯度的替代方案，究竟带来了什么收益。

---

**Sources:**

- [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/?utm_source=chatgpt.com)
- [RecSys 2020 - Session P3: Novel Machine Learning Approaches I](https://recsys.acm.org/recsys20/session-4/?utm_source=chatgpt.com)
- [NeoMUST: an accurate and efficient multi-task learning model for neoantigen presentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10828515/?utm_source=chatgpt.com)
- [A unified deep framework for peptide–major histocompatibility complex–T cell receptor binding prediction](https://www.nature.com/articles/s42256-025-01002-0?utm_source=chatgpt.com)
- [MoE-TCR: Mixture-of-experts framework for pan-specific TCR-epitope binding prediction](https://www.sciencedirect.com/science/article/pii/S0031320326007028?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)