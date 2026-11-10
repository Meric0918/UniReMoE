# 优化三元嵌入融合

**User:** Anonymous  
**Created:** 9/1/2026 20:14:20  
**Updated:** 9/1/2026 20:17:51  
**Exported:** 9/1/2026 20:18:11  
**Link:** [https://chatgpt.com/c/6a96c173-3d88-83ec-bffb-5330cb47d137](https://chatgpt.com/c/6a96c173-3d88-83ec-bffb-5330cb47d137)  

## Prompt:
2026/9/1 20:14:17

![stage5_multitask_architecture_gpt-image-2(1).png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000f004821189ba620ddb721186&fn=stage5_multitask_architecture_gpt-image-2%281%29.png&cd=attachment&ts=496740&p=fs&cid=1&sig=0690dd030631be532524a3ace8925896886d844ebc6af2278b18d188fd5189c8&v=0)

[Attachment: Hypergraph Transformer.pdf]

使用Hypergraph Transformer论文中的设计与思想对我上述图中的enbedding进行优化，使得peptide，tcr与hla的enbedding可以进行相互交融，具体应该怎么设计

## Response:
2026/9/1 20:14:19

> **梳理架构融合**
>
> 我会把论文中的静态—动态嵌入思想改造成类型感知的多方交互：先分别编码肽、HLA、TCR，再以实体级超边完成互相更新，同时保留残基级交叉注意力。PH、PT、PHT共享编码器，并用任务头区分。
>
> **梳理了架构融合方案**
>
> 
>
> **优化融合方案**
>
> 我会把静态—动态嵌入用于正则化，而非替代现有交叉注意力。PHT采用三方同步超边，避免先融合P-H、再接入T造成的顺序偏置；PH/PT共享同一融合块。FGM先作用于适配器输出，再考虑扩展至融合投影。
>
> **设计Hyper-SAGNN多任务超图距离学习架构**
>
> 
>
> 思考了 1m 36s

可以，而且我认为 **Hyper-SAGNN 的思想非常适合你现在的 PH / PT / PHT 多任务结构**。不过先说明一点：你上传的这篇 PDF 虽然文件名是 *Hypergraph Transformer.pdf*，正文实际是 **Hyper-SAGNN: a self-attention based graph neural network for hypergraphs**。它最值得迁移到你模型里的不是它原始的图特征生成方式，而是两个核心思想：

1. **把 $P,H,T$ 看成一个 heterogeneous hyperedge，而不是先拆成 PH，再让 PH 与 T 交互；**
2. **同一个实体同时维护 static embedding 和 context-dependent dynamic embedding，dynamic embedding 必须由“其他实体”产生。**

论文的 Hyper-SAGNN 本身就是为了避免把高阶关系简单拆成多个 pairwise 关系，因为一些 heterogeneous hyperedge 本身可能是不可分解的；同时它可以用统一模型处理不同大小的 hyperedge。`Hypergraph Transformer.pdf`

对于你现在这个模型，我最推荐把当前三个任务中的多个 Cross-Attention，升级成一个**共享的 Hyper-PMT Fusion Encoder**。

---

# 一、你当前 PHT 结构最大的限制在哪里

现在你的 PHT 是：

```text
Peptide ─┐
         ├─ PH Cross-Attention ─→ Z_pHLA ─┐
HLA ─────┘                                ├─ PHT Cross-Attention → prediction
TCR ──────────────────────────────────────┘
```

即：

$$
P,H\rightarrow Z_{pHLA}
$$

然后：

$$
Z_{pHLA},T\rightarrow Z_{PHT}
$$

这个结构在生物学上是合理的，因为确实存在：

$$
HLA \rightarrow peptide\ presentation \rightarrow TCR\ recognition
$$

这样的层级。

但从 representation learning 角度，它存在一个很明显的问题：

> **信息流是单向和顺序性的。**

例如第二步 TCR 进入以后：

- TCR 可以看到 pHLA；
- 但是 HLA 不会再根据 TCR 更新；
- Peptide 也不会根据 TCR 更新；
- HLA 与 TCR 之间不存在直接的信息交换；
- 最终 PHT representation 很容易偏向“先学好 PH，再让 T 去适配 PH”。

也就是说现在更像：

$$
(P,H)\rightarrow T
$$

而不是：

$$
P\leftrightarrow H\leftrightarrow T
$$

这正是 Hyper-SAGNN 思想可以解决的地方。

---

# 二、我建议把 P、H、T 直接看成一个三节点超边

定义：

$$
e_{PHT}=\{P,H,T\}
$$

三个 heterogeneous nodes：

```text
vP = Peptide
vH = HLA
vT = TCR
```

你的三个任务就可以统一成：

$$
e_{PH}=\{P,H\}
$$

$$
e_{PT}=\{P,T\}
$$

$$
e_{PHT}=\{P,H,T\}
$$

这其实非常漂亮，因为你的三个任务天然对应：

```text
2-node hyperedge: PH
2-node hyperedge: PT
3-node hyperedge: PHT
```

Hyper-SAGNN 恰好强调一个模型应该能够处理 **variable-sized heterogeneous hyperedges**。`Hypergraph Transformer.pdf`

因此你甚至不需要 PH / PT / PHT 三套 interaction encoder。

可以变成：

```text
               ┌───────────────┐
Peptide ──────→│               │
HLA ──────────→│ Hyper-PMT     │──→ PH / PT / PHT embedding
TCR ──────────→│ Fusion Block  │
               └───────────────┘
```

缺哪个实体，就通过 mask 去掉哪个实体。

---

# 三、不要直接对 ESM-C / TCR-BERT 输出做 pooling

这一点很关键。

建议你保持：

$$
P_0\in R^{L_P\times128}
$$

$$
H_0\in R^{L_H\times128}
$$

$$
T_0\in R^{L_T\times128}
$$

也就是说：

```text
Peptide ESM-C
     ↓
S1 Adapter
     ↓
Lp × 128

HLA ESM-C
     ↓
S2 Adapter
     ↓
Lh × 128

TCR-BERT
     ↓
S3 Adapter
     ↓
Lt × 128
```

不要过早变成：

```text
Peptide → 128维向量
HLA     → 128维向量
TCR     → 128维向量
```

因为真正重要的是：

- peptide residue ↔ HLA residue；
- peptide residue ↔ TCR residue；
- TCR residue ↔ HLA residue；

这些细粒度关系。

所以我的建议是：

> **Hyper-SAGNN 在 entity level 上提供思想，实际实现时把 dynamic embedding 扩展到 residue-token level。**

---

# 四、最关键：建立 Static / Dynamic 两套 embedding

这是整个改造的核心。

Hyper-SAGNN 对每个节点产生两种 representation：

### Static embedding

只依赖节点自己：

$$
s_i=f_s(x_i)
$$

### Dynamic embedding

由当前 hyperedge 中的**其他节点**生成：

$$
d_i=f_d(x_{j\neq i})
$$

论文明确把 static embedding 定义成与当前 tuple 无关，而 dynamic embedding 则依赖当前 tuple 中其他节点的特征。`Hypergraph Transformer.pdf`

并且特别强调：

$$
j\neq i
$$

也就是计算 dynamic embedding 时排除自身。`Hypergraph Transformer.pdf`

这个思想放到你的 PMT 模型里非常有意义。

---

# 五、Static branch 怎么设计

你的 S1/S2/S3 Adapter 输出就可以作为 static branch 的基础。

例如：

$$
P_0=S1(ESMC(P))
$$

$$
H_0=S2(ESMC(H))
$$

$$
T_0=S3(TCRBERT(T))
$$

然后：

$$
S_P=FFN_P(P_0)
$$

$$
S_H=FFN_H(H_0)
$$

$$
S_T=FFN_T(T_0)
$$

得到：

```text
Static Peptide embedding   SP
Static HLA embedding       SH
Static TCR embedding       ST
```

它们表达的是：

> 这个 peptide / HLA / TCR 本身是什么。

而不是：

> 它在当前 PHT 组合中是什么。

---

# 六、Dynamic branch：实现真正的 P/H/T 相互交融

这一步最重要。

对于 PHT：

### Peptide dynamic embedding

禁止 peptide 看自己：

$$
D_P=
MHA(
Q=S_P,\,
K=[S_H;S_T],\,
V=[S_H;S_T])
$$

即：

```text
                HLA
                 ↓
Peptide ← Attention
                 ↑
                TCR
```

---

### HLA dynamic embedding

$$
D_H=
MHA(
Q=S_H,\,
K=[S_P;S_T],\,
V=[S_P;S_T])
$$

即：

```text
              Peptide
                 ↓
HLA ←──── Attention
                 ↑
                TCR
```

---

### TCR dynamic embedding

$$
D_T=
MHA(
Q=S_T,\,
K=[S_P;S_H],\,
V=[S_P;S_H])
$$

即：

```text
              Peptide
                 ↓
TCR ←──── Attention
                 ↑
                HLA
```

最终：

```text
             ┌──────── HLA ────────┐
             ↓                     ↓
Peptide ←──────────────→ TCR
   ↑                             ↑
   └──────────── HLA ────────────┘
```

这时候才真正变成：

$$
P\leftrightarrow H
$$

$$
P\leftrightarrow T
$$

$$
H\leftrightarrow T
$$

并且三者是在**同一层同时发生交互**。

论文 Figure 2 的核心正是 multi-head attention 产生 dynamic embeddings，并让每个节点利用 tuple 内其他节点的信息。`Hypergraph Transformer.pdf`

---

# 七、我建议不要直接照搬普通 Self-Attention

如果直接：

```text
[P,H,T] → Transformer Self-Attention
```

那么：

$$
P\rightarrow P
$$

也允许发生。

但 Hyper-SAGNN 的关键设计恰恰是：

$$
i\not\rightarrow i
$$

所以应该做一个 **cross-entity attention mask**。

例如 PHT：

| Query \ KV | P | H | T |
|---|---:|---:|---:|
| P | ❌ | ✓ | ✓ |
| H | ✓ | ❌ | ✓ |
| T | ✓ | ✓ | ❌ |

注意这里的 ❌ 最好是：

> **禁止同一 entity 的 token 作为 dynamic branch 的 KV。**

而不是简单禁止 token 自己。

也就是说 Peptide 的 dynamic embedding 完全由 HLA + TCR 推导。

这比普通 Transformer 的：

```text
P/H/T 全部 concat
      ↓
self-attention
```

更符合 Hyper-SAGNN 的思想。

---

# 八、但是 static 和 dynamic 最后不能直接相加

Hyper-SAGNN 还有一个非常值得你借鉴的设计：

它不是简单：

$$
z=s+d
$$

而是看：

$$
(d-s)^{\circ2}
$$

论文把 static/dynamic embedding 的 element-wise squared difference 输入一个网络，让这个差异表示当前节点和整个 hyperedge 是否兼容。`Hypergraph Transformer.pdf`

直观上：

> 如果一个 TCR 本身的 representation，能够被当前 peptide + HLA 上下文“解释”，那么这个 TCR 就更可能属于这个 PHT hyperedge。

论文也明确解释 dynamic embedding 是邻居特征的加权组合，因此 static/dynamic 的差异表示“当前节点能够被 tuple 中其他节点解释得多好”。`Hypergraph Transformer.pdf`

这个思想放到你的任务上非常漂亮。

---

# 九、我建议不要完全照抄 Hyper-SAGNN 的 distance，而是做增强版

原论文：

$$
R_P=(D_P-S_P)^2
$$

对于生物 sequence interaction，我更建议：

$$
R_i=
[
S_i;
D_i;
S_i\odot D_i;
(S_i-D_i)^2
]
$$

也就是说保留四种信息：

```text
Static
Dynamic
Static × Dynamic
(Static - Dynamic)²
```

其中：

- $S_i$：实体自身特征；
- $D_i$：其它实体给它的 contextual representation；
- $S_i\odot D_i$：一致性；
- $(S_i-D_i)^2$：不一致性。

然后：

$$
Z_i=MLP(R_i)
$$

最终得到：

$$
Z_P,Z_H,Z_T
$$

我认为这会比只使用：

$$
(D-S)^2
$$

更适合你的任务。

---

# 十、完整的 PHT 模块可以设计成这样

我会把你现在的：

```text
P + H
 ↓
PH Cross-Attention
 ↓
ZpHLA + T
 ↓
PHT Cross-Attention
```

直接替换成：

```text
Peptide ESM-C
      ↓
 S1 Adapter
      ↓
     SP ─────────────┐
                     │
HLA ESM-C            │
      ↓               │
 S2 Adapter           │
      ↓               │
     SH ───────┐      │
               │      │
TCR-BERT       │      │
      ↓        │      │
 S3 Adapter    │      │
      ↓        │      │
     ST ───────┼──────┘
               ↓
      ┌──────────────────────┐
      │ Hyper-PMT Fusion     │
      │                      │
      │ DP ← SH + ST         │
      │ DH ← SP + ST         │
      │ DT ← SP + SH         │
      └──────────────────────┘
               ↓
       Static-Dynamic
        Interaction
               ↓
       ZP    ZH    ZT
          \   |   /
           \  |  /
        Hyperedge Pooling
               ↓
            ZPHT
               ↓
      PHT Classifier
```

这就是我最推荐的主干。

---

# 十一、最好堆叠 2 层，而不是只做一次

第一层：

$$
S_P,S_H,S_T
\rightarrow
D_P^{(1)},D_H^{(1)},D_T^{(1)}
$$

得到：

$$
F_i^{(1)}
=
LN(S_i+Gate_i(D_i^{(1)}))
$$

第二层再次：

$$
D_P^{(2)}
=
Attn(F_P^{(1)},[F_H^{(1)},F_T^{(1)}])
$$

等等。

这样信息路径就变成：

第一层：

$$
P\rightarrow H
$$

第二层可以形成：

$$
P\rightarrow H\rightarrow T
$$

以及：

$$
T\rightarrow P\rightarrow H
$$

等更复杂的高阶交互。

我建议最开始：

```text
num_layers = 2
hidden_dim = 128
num_heads = 4 or 8
FFN_dim = 256 or 512
dropout = 0.1
```

不建议上来就 4～6 层，因为 P/H/T 都是短序列，而你的基础模型已经是 pretrained protein language model。

---

# 十二、异构性问题：P、H、T不能被完全同等对待

这里我会在原始 Hyper-SAGNN 基础上做一个重要修改。

原论文可以统一处理 heterogeneous hypergraph，但你的三类 biological entity 差异非常大。

因此加入：

$$
E_{\text{type}}^P,\quad
E_{\text{type}}^H,\quad
E_{\text{type}}^T
$$

例如：

$$
\tilde P=P+E_P
$$

$$
\tilde H=H+E_H
$$

$$
\tilde T=T+E_T
$$

让 attention 明确知道：

```text
这是 Peptide
这是 HLA
这是 TCR
```

此外，我不建议一开始为六种方向：

```text
P ← H
P ← T
H ← P
H ← T
T ← P
T ← H
```

各设计完整的一套 Q/K/V。

参数量会明显增加。

更稳妥的是：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

共享主 projection，然后加入 relation bias：

$$
A_{ij}
=
\frac{Q_iK_j^T}{\sqrt d}
+
b_{\tau_i,\tau_j}
$$

其中：

$$
b_{P,H},
b_{P,T},
b_{H,P},
...
$$

只有很小的一组 relation parameters。

这样既保留 parameter sharing，也具备 heterogeneous awareness。

---

# 十三、PH、PT、PHT 三个任务应该全部使用同一个 Fusion Block

这是这个设计另一个非常大的优势。

## PH

输入：

$$
\{P,H\}
$$

attention mask：

```text
P ← H
H ← P
```

得到：

$$
D_P^{PH},D_H^{PH}
$$

然后：

$$
\hat y_{PH}=Head_{PH}(Z_P,Z_H)
$$

---

## PT

输入：

$$
\{P,T\}
$$

```text
P ← T
T ← P
```

得到：

$$
\hat y_{PT}=Head_{PT}(Z_P,Z_T)
$$

---

## PHT

输入：

$$
\{P,H,T\}
$$

```text
P ← H,T
H ← P,T
T ← P,H
```

得到：

$$
\hat y_{PHT}=Head_{PHT}(Z_P,Z_H,Z_T)
$$

最重要的是：

> **这三个任务的 Hyper-PMT Fusion Encoder 是同一个 parameter instance。**

于是你的 Parameter Sharing 可以从现在：

```text
S1: PH PT PHT
S2: PH PHT
S3: PT PHT
S4: PH PHT
```

进一步改成：

```text
S1 Peptide Adapter : PH PT PHT
S2 HLA Adapter     : PH PHT
S3 TCR Adapter     : PT PHT

S4 Hyper-PMT Fusion:
PH · PT · PHT
```

然后只有：

```text
PH Head
PT Head
PHT Head
```

是 task-specific。

---

# 十四、这和你的 Multi-Task Learning 非常契合

这一点有论文依据。

Hyper-SAGNN 专门实验了：

- pairwise edges；
- higher-order hyperedges；
- 两者同时训练。

结果发现混合 edge + hyperedge 训练，并不会明显破坏 hyperedge prediction；在 hyperedge 数据较少时，同时利用 edge 与 hyperedge 反而能提升性能或降低过拟合，因为模型可以学习不同 hyperedge size 之间共享的结构信息。`Hypergraph Transformer.pdf`

映射到你的任务就是：

```text
PH  = pair edge
PT  = pair edge
PHT = triple hyperedge
```

所以你现在：

$$
L_{\text{clean}}
=
0.3L_{PH}
+
0.2L_{PT}
+
1.0L_{PHT}
$$

这个 Multi-Task 思路不仅可以保留，**反而更适合新架构**。

因为原来 PH/PT/PHT 只有 Adapter 层真正大量共享。

改造之后：

$$
L_{PH},L_{PT},L_{PHT}
$$

都会同时优化：

```text
Adapters
    +
Hyper-PMT Fusion Encoder
```

这才形成更强的“关系层联合优化”。

---

# 十五、训练时我建议继续用你之前那种“同一步三个 batch”的方式

也就是：

```python
PH_batch  -> HyperFusion({P,H})   -> L_PH
PT_batch  -> HyperFusion({P,T})   -> L_PT
PHT_batch -> HyperFusion({P,H,T}) -> L_PHT

loss = 0.3 * L_PH + 0.2 * L_PT + 1.0 * L_PHT

loss.backward()
optimizer.step()
```

而不是：

```text
step1 PH optimizer.step()
step2 PT optimizer.step()
step3 PHT optimizer.step()
```

因为现在 S4 是三任务共享的 Hyper-PMT Fusion。

在同一个 optimization step 聚合梯度，会更加明确地实现：

$$
\nabla L
=
0.3\nabla L_{PH}
+
0.2\nabla L_{PT}
+
\nabla L_{PHT}
$$

也就是：

> PH/PT 的 pairwise interaction 信息和 PHT 的 higher-order interaction 信息同时塑造同一个 embedding space。

---

# 十六、PHT千万不要变成 PH × PT 的简单组合

这也是 Hypergraph 思想非常重要的一点。

不要做：

$$
P(PHT)=P(PH)\times P(PT)
$$

也不要简单：

$$
Z_{PHT}=Z_{PH}+Z_{PT}
$$

因为 Hyper-SAGNN 的出发点之一就是：

> higher-order hyperedge 可能是 **indecomposable** 的，并不能完全由 pairwise edges 推导出来。`Hypergraph Transformer.pdf`

你的 PHT 同样应该允许存在一个真正的：

$$
I_{PHT}
$$

三阶 interaction。

可以理解成：

$$
Score_{PHT}
=
f(
I_{PH},
I_{PT},
I_{HT},
I_{PHT}
)
$$

而其中：

$$
I_{PHT}
$$

必须由三者共同进入 HyperFusion 才能学习。

这也是我为什么不太建议保留：

```text
PH → ZpHLA → T
```

作为 PHT 唯一主路径。

因为这实际上仍然比较接近把三元关系分解成两个二元过程。

---

# 十七、但是 PH 的生物学先验不要完全丢掉

这里需要做一个折中。

从生物学逻辑上：

```text
HLA presents peptide
        ↓
TCR recognizes pHLA
```

这个先验是有价值的。

所以我不建议完全删除你现有的 PH inductive bias。

有两种做法，我更推荐第二种。

### 方案 A：直接完全 Hypergraph 化

```text
P
H → HyperFusion → PHT
T
```

结构最干净。

### 方案 B：Hypergraph + PH prior

保留一个轻量 PH branch：

$$
Z_{PH}^{prior}
=
CrossAttn(P,H)
$$

但是不要让它成为 PHT 的唯一入口。

而是：

```text
P ───────────────┐
H ───────────────┤
T ───────────────┤
                 ↓
          Hyper-PMT Fusion
                 ↓
              Z_PHT
                 ↑
        PH prior embedding
```

最终：

$$
Z_{PHT}^{final}
=
Z_{PHT}^{hyper}
+
Gate(Z_{PH}^{prior})
$$

这样：

- Hypergraph branch 学 higher-order interaction；
- PH branch 提供生物学先验；
- 不会强迫 PHT 必须通过 PH 单一路径。

如果你的 PH 数据量很大、质量也高，我尤其推荐这个版本。

---

# 十八、还有一个非常适合你负样本设计的 Hyper-SAGNN 思想

Hyper-SAGNN 不只是输出整个 hyperedge probability。

它会给每个节点一个自己的：

$$
p_i
$$

论文甚至利用最小的 $p_i$ 来识别整个 tuple 中哪个节点是 “outsider”，并报告了比较好的 outsider identification 结果。`Hypergraph Transformer.pdf`

这个思想非常适合 PMT。

例如一个负样本：

```text
P：正确
H：正确
T：被随机替换
```

你不仅监督：

$$
y_{PHT}=0
$$

还可以监督：

```text
T is outsider
```

让模型产生：

$$
c_P,c_H,c_T
$$

三个 node compatibility scores。

对于上面的样本：

$$
c_T<c_P,c_H
$$

可以加：

$$
L_{outlier}
=
CE(
[-c_P,-c_H,-c_T],
T
)
$$

如果换的是 HLA：

$$
target=H
$$

这会极大提升模型的可解释性。

最终你甚至可以告诉用户：

```text
PHT prediction = 0.08

Peptide compatibility = 0.91
HLA compatibility     = 0.87
TCR compatibility     = 0.14
```

于是模型不只是：

> “不结合”

还可以说：

> “主要不匹配来源是 TCR。”

这对于你的研究方向其实非常有价值。

---

# 十九、最终损失函数我建议设计为

第一版先保持简单：

$$
\boxed{
L=
0.3L_{PH}
+
0.2L_{PT}
+
1.0L_{PHT}
}
$$

跑通并得到 baseline。

然后增加 Hyper-SAGNN auxiliary compatibility：

$$
L=
0.3L_{PH}
+
0.2L_{PT}
+
L_{PHT}
+
\lambda_h L_{hyper}
$$

如果有明确的“替换哪个实体”的负样本，再加入：

$$
L=
0.3L_{PH}
+
0.2L_{PT}
+
L_{PHT}
+
\lambda_hL_{hyper}
+
\lambda_oL_{outlier}
$$

初始可以：

$$
\lambda_h=0.05
$$

$$
\lambda_o=0.05
$$

不要一开始设太大。

---

# 二十、FGM 应该放在哪里

你现在图片里是：

```text
Stage5-1B FGM targets:
S1
S2
S3
```

换成 HyperFusion 后，我第一轮实验仍然建议：

```text
FGM
 ↓
S1/S2/S3 Adapter output
 ↓
Hyper-PMT Fusion
```

也就是说不要一开始同时扰动：

```text
Adapter
+
HyperFusion
+
Classifier
```

否则不容易判断提升到底来自 Hypergraph 模块还是 adversarial training。

推荐实验顺序：

```text
Baseline:
原模型

Experiment 1:
原模型 + Hyper-PMT Fusion
无 FGM

Experiment 2:
Hyper-PMT Fusion
+ Adapter FGM

Experiment 3:
Hyper-PMT Fusion
+ outsider auxiliary loss

Experiment 4:
Hyper-PMT Fusion
+ PH prior branch
```

这样 ablation 会非常清楚。

---

# 二十一、我最推荐的最终 Stage 5 架构

如果让我直接修改你这张图，我会变成：

```text
                 Frozen encoders
────────────────────────────────────────────

Peptide ── ESM-C ── S1 Adapter ── P tokens
                                      │
HLA ───── ESM-C ── S2 Adapter ─── H tokens
                                      │
TCR ─── TCR-BERT ─ S3 Adapter ─── T tokens
                                      │
                                      ▼

                 Shared S4
       ┌──────────────────────────┐
       │ Hyper-PMT Fusion Encoder │
       │                          │
       │ Static branch            │
       │ SP    SH    ST            │
       │                          │
       │ Dynamic branch           │
       │ DP ← H,T                 │
       │ DH ← P,T                 │
       │ DT ← P,H                 │
       │                          │
       │ self-entity masked       │
       └──────────────────────────┘
             │        │       │
             ▼        ▼       ▼
             ZP       ZH      ZT
              \       |       /
               \      |      /
                ▼     ▼     ▼

────────────────────────────────────────────
PH:
HyperFusion(P,H)
      ↓
PH Pooling
      ↓
PH Classifier

PT:
HyperFusion(P,T)
      ↓
PT Pooling
      ↓
PT Classifier

PHT:
HyperFusion(P,H,T)
      ↓
Hyperedge Pooling
      ↓
PHT Classifier
────────────────────────────────────────────
```

Parameter sharing：

```text
S1 : PH · PT · PHT
S2 : PH · PHT
S3 : PT · PHT
S4 : PH · PT · PHT
```

Task-specific：

```text
PH  : Pooling + Classifier
PT  : Pooling + Classifier
PHT : Pooling + Classifier
```

这比你现在的 parameter sharing 更完整。

---

# 二十二、Static / Dynamic 具体公式可以直接这样写进论文

Adapter：

$$
X_P=A_P(E_P(P))
$$

$$
X_H=A_H(E_H(H))
$$

$$
X_T=A_T(E_T(T))
$$

Static representations：

$$
S_i=FFN_s(X_i),\qquad
i\in\{P,H,T\}
$$

Dynamic representations：

$$
D_i=
MHA
\left(
Q=S_i,\,
K=Concat_{j\neq i}(S_j),\,
V=Concat_{j\neq i}(S_j)
\right)
$$

Hyper-SAGNN-inspired compatibility：

$$
C_i=
[
S_i;
D_i;
S_i\odot D_i;
(S_i-D_i)^2
]
$$

$$
Z_i=FFN_c(C_i)
$$

Hyperedge representation：

$$
Z_e=
AttentionPool
\left(
Z_i:i\in e
\right)
$$

最终：

$$
\hat y_e=
\sigma(
MLP_{task}(Z_e)
)
$$

其中：

$$
e\in
\{
(P,H),
(P,T),
(P,H,T)
\}
$$

这套数学表达非常适合作为你的 Method 部分。

---

# 二十三、为什么我认为这个版本比“直接增加一个三元 Cross-Attention”更好

可以把三种模型理解成：

### 你现在

$$
P\leftrightarrow H
\rightarrow
T
$$

属于**层级交互**。

### 普通 triple self-attention

$$
[P,H,T]
\rightarrow
SelfAttention
$$

属于**全局混合**。

### 我建议的 Hyper-SAGNN-inspired 版本

$$
\text{自身身份 Static}
+
\text{其它实体提供的上下文 Dynamic}
$$

属于：

$$
\boxed{
\text{identity}+\text{contextual compatibility}
}
$$

这对于 binding prediction 很合适，因为问题实际上就是：

> “这个 peptide / HLA / TCR 自身的分子表示，与另外两个分子共同形成的上下文是否兼容？”

这正好对应：

$$
S_i
\quad vs.\quad
D_i
$$

---

# 二十四、对于你的项目，我最终会选择这个版本

如果考虑性能、创新性、生物解释性和实现复杂度的平衡，我建议最终采用：

$$
\boxed{
\text{Frozen PLM}
\rightarrow
\text{Adapters}
\rightarrow
\textbf{2-layer Heterogeneous Hyper-PMT Fusion}
\rightarrow
\text{Static-Dynamic Compatibility}
\rightarrow
\text{Task Heads}
}
$$

其中最关键的四个设计是：

1. **P/H/T 同时交互，而不是 P+H 后再接 T；**
2. **保留 Static embedding，防止三个模态全部混成一个无法解释的表示；**
3. **Dynamic embedding 只能从其他 entity 获得，实施 $j\neq i$ 的 cross-entity mask；**
4. **PH、PT、PHT 共用完全相同的 Hyper-PMT Fusion 参数，让 pairwise 数据真正辅助 PHT higher-order learning。**

如果要给这个模块起论文里的名字，我反而不建议直接叫 Hyper-SAGNN，因为你已经进行了 residue-level 和 heterogeneous biological adaptation。可以叫：

> **Heterogeneous Hypergraph Interaction Encoder (HHIE)**

或者更针对你的任务：

> **PMT Hypergraph Fusion Module (PHFM)**

你的整套模型创新点就可以非常清楚地表述成：

$$
\boxed{
\text{Pairwise interaction learning}
\quad+\quad
\text{Higher-order hyperedge interaction learning}
}
$$

而不是简单的“三个 Cross-Attention 拼起来”。

尤其值得强调的是，你上传的论文已经证明 Hyper-SAGNN 可以通过混合 edge 与 hyperedge 训练学习不同规模 hyperedge 之间的共识结构，正好对应你这里 **PH/PT 两个二元任务辅助 PHT 三元任务** 的核心逻辑。`Hypergraph Transformer.pdf`

如果继续沿这个方向设计，我会优先把你图里的 **S4 PH Cross-Attention、PT Cross-Attention、PHT Cross-Attention 三部分合并成一个共享的 S4 Hyper-PMT Fusion Encoder**；这会是整张 Stage 5 架构图最重要的一次结构升级。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)