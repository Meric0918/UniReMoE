# Adapter设计方案

**User:** Anonymous  
**Created:** 9/1/2026 20:53:59  
**Updated:** 9/7/2026 11:18:05  
**Exported:** 9/11/2026 17:21:41  
**Link:** [https://chatgpt.com/c/6a96ca8e-cc28-83ec-b906-a9ddb905d603](https://chatgpt.com/c/6a96ca8e-cc28-83ec-b906-a9ddb905d603)  

## Prompt:
2026/9/1 20:53:57

![stage5_multitask_architecture_gpt-image-2(2).png](https://chatgpt.com/backend-api/estuary/content?id=file_0000000070a482118f8e6fc5bf6c8acb&fn=stage5_multitask_architecture_gpt-image-2%282%29.png&cd=attachment&ts=496977&p=fs&cid=1&sig=12a13c3b3e402dd768f52d02a173dc6c7ba1dc5b642531ce47022a691d378be3&v=0)

我在Stage1中只有PH和PT进行交替训练，在stage3中加入了PHT任务，stage3是在stage1的best_model上继续训练的，但是stage3在PT上的表现不如stage1，我可以设计一个adapter根据不同的输入（PT，PH，PHT）选择不同stage的模型进行预测吗，进行详细分析设计

## Response:
2026/9/1 20:54:11

> **设计输入路由**
>
> 我会把 PH、PT、PHT 当作显式任务标签：先用确定性路由选择对应阶段适配器，而不是训练复杂路由器。共享骨干保持不变，分别挂载 Stage1、Stage3 适配器和任务头；PHT 继续训练时用蒸馏、回放或冻结机制保护 PT 性能，并用验证集比较硬路由与联合微调。
>
> **设计了输入路由**
>
> 
>
> **完善路由架构**
>
> 我会为 PH、PT、PHT 分别保存验证集最优路径，并用任务适配器隔离参数更新。共享编码器冻结，Stage1 专家保护 PT，Stage3 专家学习 PHT；再通过蒸馏和校准减少跨阶段性能退化。
>
> **完善了路由架构**
>
> 
>
> 思考了 1m 40s

可以，而且从你现在的 Stage1 → Stage3 训练方式以及图中的参数共享关系来看，**这是一个非常合理的设计方向**。但我建议不要简单做成“一个 Adapter 判断输入后加载整个 Stage1 或 Stage3 checkpoint”，而是进一步设计成：

> **Task-aware Hard Router + Stage-specific Adapter/Expert Bank**

核心思想是：**保留 Stage1 已经学得很好的 PT 能力，不允许 Stage3 的 PHT 训练破坏它；而 PHT 则通过自己的增量 Adapter 学习新的三元交互。**

你现在 Stage3 的 PT 下降，很可能不是偶然，而是典型的 **multi-task negative transfer / catastrophic forgetting**。

---

# 1. 先分析为什么 Stage3 加入 PHT 后 PT 反而下降

从你的图里，当前参数共享关系是：

| 模块 | PH | PT | PHT |
|---|---:|---:|---:|
| S1 Peptide Adapter | ✓ | ✓ | ✓ |
| S2 HLA Adapter | ✓ |  | ✓ |
| S3 TCR Adapter |  | ✓ | ✓ |
| S4 PH Cross-Attention | ✓ |  | ✓ |
| PT Cross-Attention |  | ✓ |  |
| PHT Cross-Attention |  |  | ✓ |

这里最值得注意的是：

> **PT 和 PHT 虽然 Cross-Attention 不共享，但是它们共享 S1 Peptide Adapter 和 S3 TCR Adapter。**

Stage1 时：

$$
P \xrightarrow{S1} z_P,\qquad
T \xrightarrow{S3}z_T
$$

是完全围绕：

$$
P-T
$$

二元特异性学习出来的。

到了 Stage3，PHT loss 开始更新：

$$
S1,\quad S2,\quad S3,\quad S4
$$

尤其：

$$
L_{PHT}\rightarrow S1,S3
$$

于是就出现：

$$
\nabla L_{PT}
$$

和

$$
\nabla L_{PHT}
$$

可能并不一致。

---

## 举一个非常直观的例子

Stage1 中 S3-TCR Adapter 可能学到：

> “TCR 哪些残基特征最适合直接和 peptide 匹配。”

但是 Stage3 中 PHT 的目标变成：

> “TCR 哪些残基特征适合和 pHLA 表示匹配。”

这两个目标并不完全相同。

所以可能产生：

$$
z_T^{Stage1}
$$

特别适合：

$$
P\leftrightarrow T
$$

但训练 PHT 后变成：

$$
z_T^{Stage3}
$$

更适合：

$$
pHLA\leftrightarrow T
$$

于是 PT Cross-Attention 本身即使没有改变，输入已经发生 representation drift：

$$
z_P^{(1)},z_T^{(1)}
\rightarrow
z_P^{(3)},z_T^{(3)}
$$

最终：

$$
Performance_{PT}^{Stage3}
<
Performance_{PT}^{Stage1}
$$

这实际上就是你当前结果非常合理的一种解释。

---

# 2. 所以能不能“根据 PH/PT/PHT 选择不同 Stage”？

完全可以。

而且因为你的任务类型本身就是确定的：

- 输入 P + H → PH
- 输入 P + T → PT
- 输入 P + H + T → PHT

所以甚至**根本不需要训练一个复杂 Router 来猜是什么任务**。

直接使用输入 mask 即可。

例如：

$$
m=(H,T)
$$

其中：

- H=1：存在 HLA
- T=1：存在 TCR

那么：

$$
(1,0)\rightarrow PH
$$

$$
(0,1)\rightarrow PT
$$

$$
(1,1)\rightarrow PHT
$$

然后：

$$
R(m)=
\begin{cases}
M_{PH}, & (1,0)\\
M_{PT}, & (0,1)\\
M_{PHT}, & (1,1)
\end{cases}
$$

这是一个 **Hard Router**。

---

# 3. 最简单版本：Stage Checkpoint Router

你现在甚至可以先不修改任何网络。

假设：

$$
M_1 = Stage1\ best
$$

$$
M_3 = Stage3\ best
$$

如果你的验证结果是：

- PH：Stage3 最好
- PT：Stage1 最好
- PHT：Stage3 最好

那么直接：

```text
                 Input
                   │
           ┌──── Task Router ────┐
           │          │          │
          PH         PT         PHT
           │          │          │
       Stage3      Stage1      Stage3
           │          │          │
       PH Head      PT Head     PHT Head
```

即：

$$
f(x)=
\begin{cases}
f_{PH}^{Stage3}(P,H)\\
f_{PT}^{Stage1}(P,T)\\
f_{PHT}^{Stage3}(P,H,T)
\end{cases}
$$

---

## 这个方法最大的优点

你可以保证：

> PT 不会因为 Stage3 PHT 训练发生任何性能下降。

因为 PT 根本没有使用 Stage3。

---

## 但它也存在问题

这更接近：

> **checkpoint routing / multi-expert inference**

而不是真正意义上的 Adapter。

需要保留：

$$
M_{Stage1}+M_{Stage3}
$$

两套模型参数。

如果 ESM-C、TCR-BERT 都冻结，并且只是保存上层网络，其实参数量可能还可以接受。

但是论文方法层面会稍微显得“不够统一”。

所以我更建议你进一步改造成下面这个版本。

---

# 4. 更推荐：Stage-specific Adapter Bank

这是我认为最适合你当前研究的方案。

不要：

> PT 调 Stage1 整个模型  
> PHT 调 Stage3 整个模型

而应该拆成：

$$
\text{Frozen backbone}
+
\text{Stage1 base}
+
\text{Task-specific Adapter}
$$

---

# 5. 一个非常关键的修改：把 Stage1 作为 Base Model

你已经有：

$$
M_{Stage1}^{best}
$$

而且 PT 性能很好。

那么 Stage3 不应该继续随意修改 Stage1 学到的表示。

可以把 Stage1 最优参数定义为：

$$
\theta_{base}
=
\theta_{Stage1}^{best}
$$

然后：

> **冻结 Stage1 的关键共享参数。**

比如：

```text
ESM-C        Frozen
TCR-BERT     Frozen

S1_base      Frozen
S2_base      Frozen
S3_base      Frozen
S4_base      Frozen
```

再针对不同 task 添加小型 Adapter。

---

# 6. 从你现在的架构改成这样

建议把你的：

```text
Peptide ESM-C
      │
     S1
```

改成：

```text
Peptide ESM-C
      │
 S1 Stage1 Base
    Frozen
      │
 ┌────┼─────────┐
 │    │         │
PH   PT        PHT
 │    │         │
A_P_PH A_P_PT A_P_PHT
```

HLA：

```text
HLA ESM-C
    │
S2 Stage1 Base
  Frozen
    │
 ┌──┴─────────┐
PH           PHT
 │             │
A_H_PH      A_H_PHT
```

TCR：

```text
TCR-BERT
    │
S3 Stage1 Base
  Frozen
    │
 ┌──┴────────┐
PT          PHT
 │            │
A_T_PT     A_T_PHT
```

这样就是一个真正的：

> **Task-aware Adapter Bank**

---

# 7. Adapter 不需要很大

比如你目前：

$$
960\rightarrow256\rightarrow128
$$

得到：

$$
z_P\in R^{L_P\times128}
$$

之后增加：

$$
Adapter(z)
=
W_{up}
\sigma(
W_{down}z
)
$$

再做 residual：

$$
\tilde z
=
z+\alpha Adapter(z)
$$

比如：

$$
128\rightarrow32\rightarrow128
$$

就足够。

所以：

```text
z
│
├──────────────────┐
│                  │
│        128 → 32 → 128
│                  │
└──────── + ───────┘
           │
          z'
```

---

# 8. 最关键的技巧：PHT Adapter 零初始化

假设：

$$
z_P^{base}=S1(P)
$$

那么 PHT：

$$
z_P^{PHT}
=
z_P^{base}
+
A_P^{PHT}(z_P^{base})
$$

初始化的时候令：

$$
A_P^{PHT}\approx0
$$

那么初始阶段：

$$
z_P^{PHT}\approx z_P^{Stage1}
$$

随着训练：

$$
A_P^{PHT}
$$

逐渐学会：

> “为了 PHT 三元交互，peptide representation 应该怎样在 Stage1 representation 上进行增量调整。”

这样就不会直接破坏 Stage1 的空间。

---

# 9. 对你来说甚至可以进一步简化

实际上你不一定需要：

$$
A_{PT}
$$

因为 Stage1 的 PT 已经最好。

所以直接：

$$
A_P^{PT}=0
$$

$$
A_T^{PT}=0
$$

即：

### PT

完全使用 Stage1 representation：

$$
z_P^{PT}=S1_{Stage1}(P)
$$

$$
z_T^{PT}=S3_{Stage1}(T)
$$

然后：

$$
z_{PT}
=
CrossAttention_{PT}
(z_P^{PT},z_T^{PT})
$$

于是 PT 能力被“锁死”。

---

# 10. PHT 才使用新的 Adapter

例如：

$$
z_P^{PHT}
=
S1_{Stage1}(P)
+
A_P^{PHT}
$$

$$
z_H^{PHT}
=
S2_{Stage1}(H)
+
A_H^{PHT}
$$

$$
z_T^{PHT}
=
S3_{Stage1}(T)
+
A_T^{PHT}
$$

然后：

$$
z_{PHLA}
=
CrossAttn_{PH}^{PHT}
(z_P^{PHT},z_H^{PHT})
$$

再：

$$
z_{PHT}
=
CrossAttn_{PHT}
(z_{PHLA},z_T^{PHT})
$$

这样 PHT 可以随便学习：

$$
A_P^{PHT},A_H^{PHT},A_T^{PHT}
$$

而不会污染 PT。

这点特别重要。

---

# 11. 我会把你的架构修改成这个逻辑

整体可以变成：

```text
                     Frozen Encoders
               ┌────────┬─────────┐
               │        │         │
            Peptide    HLA       TCR
               │        │         │
              S1       S2        S3
          Stage1 Base Stage1   Stage1
             Frozen   Frozen    Frozen
               │        │         │
       ┌───────┼────────┼─────────┼───────┐
       │       │        │         │       │
      PH      PT       PHT       ...
       │       │        │
   PH Adapter  │    PHT Adapter
       │       │        │
       ↓       ↓        ↓

PH:
P + H
↓
PH-specific Adapter
↓
PH Cross Attention
↓
PH classifier

PT:
P + T
↓
Stage1 representation
↓
Stage1 PT Cross Attention
↓
Stage1 PT classifier

PHT:
P + H + T
↓
PHT-specific Adapter
↓
PH Cross Attention
↓
pHLA
↓
PHT Cross Attention with TCR
↓
PHT classifier
```

其中 PT 这条路径基本就是：

> **Stage1 expert**

PHT 是：

> **Stage1 base + PHT delta expert**

---

# 12. 你的 Router 应该放在哪里？

这里很重要。

我不建议：

```text
Embedding
↓
Neural Network Router
↓
判断 PH/PT/PHT
```

因为完全没必要。

任务类型本身是先验信息。

例如：

```python
if peptide is not None and hla is not None and tcr is None:
    task = "PH"

elif peptide is not None and hla is None and tcr is not None:
    task = "PT"

elif peptide is not None and hla is not None and tcr is not None:
    task = "PHT"
```

然后：

```python
if task == "PH":
    return model.forward_ph(...)

elif task == "PT":
    return model.forward_pt(...)

elif task == "PHT":
    return model.forward_pht(...)
```

这比学一个 Router 更好。

原因是你的 routing label 是 **100% 已知的**。

不应该浪费模型容量去学习：

$$
R(x)
$$

---

# 13. 可以把这个东西叫做什么？

论文里我建议避免简单叫：

> Adapter

因为实际上包含两个概念。

可以叫：

### Task-aware Stage Router

或者更完整一些：

> **Task-aware Stage-specific Adapter Routing**

简称：

$$
TSAR
$$

结构就是：

$$
Task\ ID
\rightarrow
Router
\rightarrow
Task/Stage-specific Adapter
\rightarrow
Task Head
$$

---

# 14. 为什么这种方案比“Stage1/Stage3 两个模型直接切换”更适合论文？

因为简单切 checkpoint：

```text
PT → Stage1
PHT → Stage3
```

reviewer 很容易问：

> 那不就是两个独立模型吗？

但 Adapter Bank 不一样。

你的底层仍然：

$$
ESM-C_P
$$

$$
ESM-C_H
$$

$$
TCR-BERT
$$

以及 Stage1 base representation 都共享。

只有少量：

$$
\Delta\theta_{task}
$$

不同。

可以写成：

$$
\theta_t
=
\theta_{base}
+
\Delta\theta_t
$$

其中：

$$
t\in\{PH,PT,PHT\}
$$

这就是典型参数高效 task specialization。

---

# 15. 对你现在模型，我最推荐的具体参数共享策略

可以把你图右侧的 Parameter sharing 从现在：

```text
S1   PH · PT · PHT
S2   PH · PHT
S3   PT · PHT
S4   PH · PHT
```

改成：

| 模块 | Base | PH Adapter | PT Adapter | PHT Adapter |
|---|---|---|---|---|
| S1 peptide | Stage1 Frozen | optional | none | trainable |
| S2 HLA | Stage1 Frozen | optional | - | trainable |
| S3 TCR | Stage1 Frozen | - | none | trainable |
| S4 PH attention | Stage1 Frozen/clone | optional | - | trainable delta |
| PT attention | Stage1 Frozen | - | Stage1 | - |
| PHT attention | - | - | - | trainable |
| PT classifier | Stage1 Frozen | - | Stage1 | - |
| PHT classifier | - | - | - | trainable |

这样你几乎可以**理论上保证 Stage1 PT 不退化**。

---

# 16. PH 怎么处理则取决于你的实验结果

你目前只说：

> Stage3 PT 不如 Stage1。

所以 PH 不一定需要 Stage1。

假设你的结果是：

$$
PH_{Stage3}>PH_{Stage1}
$$

那么可以设置：

```text
PH  → Stage3 Adapter
PT  → Stage1 frozen
PHT → Stage3/PHT Adapter
```

如果：

$$
PH_{Stage1}>PH_{Stage3}
$$

则：

```text
PH  → Stage1
PT  → Stage1
PHT → PHT Adapter
```

千万不要为了模型形式上的对称强行统一。

应该根据 validation set 确定：

$$
k_t
=
\arg\max_k Metric(t,k)
$$

其中：

$$
k\in\{Stage1,Stage3\}
$$

---

# 17. 还有一个很值得做的实验：确认是不是 gradient conflict

如果你想把这个方案做成论文上的一个充分论证，我特别建议你分析 Stage3 时：

$$
g_{PT}
=
\nabla_{\theta_{shared}}L_{PT}
$$

和：

$$
g_{PHT}
=
\nabla_{\theta_{shared}}L_{PHT}
$$

尤其计算 S1、S3 上的：

$$
cos(g_{PT},g_{PHT})
=
\frac{
g_{PT}^Tg_{PHT}
}{
\Vert g_{PT}\Vert
\Vert g_{PHT}\Vert
}
$$

如果：

$$
cos<0
$$

意味着：

> PT 和 PHT 在这个参数上的梯度方向互相冲突。

你可以分别统计：

```text
S1 Peptide Adapter
cos(gPT,gPHT)

S3 TCR Adapter
cos(gPT,gPHT)
```

如果发现：

```text
S1: -0.18
S3: -0.31
```

或者训练过程中大量 batch：

$$
cos<0
$$

那几乎就能直接支持你的假设：

> **PHT 引入导致共享实体表示出现负迁移。**

这样 Adapter Router 的动机就非常完整了。

---

# 18. 还可以测 representation drift

比如固定一批 PT validation：

Stage1：

$$
z_P^{(1)},z_T^{(1)}
$$

Stage3：

$$
z_P^{(3)},z_T^{(3)}
$$

计算：

$$
D_P
=
E[
\Vert
z_P^{(3)}-z_P^{(1)}
\Vert_2^2
]
$$

以及：

$$
D_T
=
E[
\Vert
z_T^{(3)}-z_T^{(1)}
\Vert_2^2
]
$$

如果 Stage3 后 S1/S3 发生明显 drift，同时 PT AUC/AUPR 下降，那么你的故事线就会非常清楚：

```text
PHT加入
 ↓
S1/S3 gradient conflict
 ↓
PT representation drift
 ↓
PT performance degradation
 ↓
Task-specific adapter isolation
 ↓
恢复 PT + 保留 PHT
```

这个会比单纯说“Stage3 PT下降，所以用了Adapter”有说服力得多。

---

# 19. 另外一个可以和 Adapter 一起使用的机制：Stage1 Teacher

即使你不完全冻结 PT，也可以把：

$$
M_{Stage1}^{best}
$$

当 Teacher。

Stage3 student：

$$
M_{Stage3}
$$

在 PT 数据上增加：

$$
L_{KD}^{PT}
=
KL(
p_{Stage1}^{PT}
\Vert
p_{Stage3}^{PT}
)
$$

例如温度：

$$
p_T
=
softmax(z/T)
$$

则：

$$
L_{KD}
=
T^2
KL
\left(
softmax(z_{teacher}/T)
\Vert
softmax(z_{student}/T)
\right)
$$

最终：

$$
L =
\lambda_{PH}L_{PH}
+
\lambda_{PT}L_{PT}
+
\lambda_{PHT}L_{PHT}
+
\beta L_{KD}^{PT}
$$

这个机制就是：

> PHT 可以学习，但不要忘掉 Stage1 的 PT decision boundary。

---

# 20. 还可以进一步加入 representation distillation

比 logits KD 更适合你这个问题，因为你的问题很可能出现在 S1/S3 embedding drift。

Teacher：

$$
z_P^T,z_T^T
$$

Student：

$$
z_P^S,z_T^S
$$

加：

$$
L_{rep}
=
\Vert z_P^S-z_P^T\Vert_2^2
+
\Vert z_T^S-z_T^T\Vert_2^2
$$

那么：

$$
L_{Stage3}
=
0.3L_{PH}
+
0.2L_{PT}
+
1.0L_{PHT}
+
\beta L_{KD}^{PT}
+
\gamma L_{rep}^{PT}
$$

你的图现在底部：

$$
L_{clean}
=
0.3L_{PH}
+
0.2L_{PT}
+
1.0L_{PHT}
$$

就可以扩展成：

$$
\boxed{
L=
0.3L_{PH}
+
0.2L_{PT}
+
1.0L_{PHT}
+
\beta L_{KD}^{PT}
+
\gamma L_{rep}^{PT}
}
$$

不过如果你已经对 PT 路径完全冻结，那么 KD 就不是必需的。

---

# 21. 三种方案我给你的推荐级别

### 方案 A：Checkpoint Router

```text
PH  → best PH stage
PT  → Stage1
PHT → Stage3
```

优点：

- 马上就能做
- 性能最容易保证
- 不需要重新训练
- 是非常好的 baseline

缺点：

- 两套模型
- 不够 elegant

我建议：**一定做，作为 baseline。**

---

### 方案 B：Task-specific Adapter Router

```text
             Stage1 Base
                Frozen
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
      PH A       PT       PHT A
        │         │         │
       PH        PT        PHT
```

其中：

```text
PT:
Stage1 → 完全保留

PHT:
Stage1 + ΔPHT

PH:
Stage1 + ΔPH 或 Stage3
```

这是我**最推荐作为最终模型**的。

优点：

- 少量参数
- 避免负迁移
- 统一模型
- task-aware
- 解释性强
- 很适合写论文

---

### 方案 C：Learned Soft Router / Mixture-of-Experts

还可以做到：

$$
\alpha_1,\alpha_3
=
Router(x)
$$

然后：

$$
z
=
\alpha_1z^{Stage1}
+
\alpha_3z^{Stage3}
$$

例如 PT：

$$
p_{PT}
=
\alpha(x)p_{PT}^{Stage1}
+
(1-\alpha(x))p_{PT}^{Stage3}
$$

这个更高级，但是我**暂时不建议你上来就用**。

因为你已经知道输入是什么任务。

学习一个 Router：

$$
P,T\rightarrow \alpha
$$

容易引入：

- router overfitting
- validation leakage
- expert collapse
- 训练复杂度
- 可解释性下降

而且如果 Stage1 在 PT 上整体明显优于 Stage3：

$$
Stage1_{PT}>Stage3_{PT}
$$

那根本没有必要让 Router 学着选。

直接：

$$
PT\rightarrow Stage1
$$

即可。

---

# 22. 我认为你下一版 Stage5 图应该怎么修改

你现在这张图的核心问题在：

> S1/S3 仍然是 PH/PT/PHT 全局共享。

建议改成：

```text
                     Frozen encoder
                           │
                     Stage1 Base
                        Frozen
                           │
                Task-aware Hard Router
                  /         |        \
                PH          PT        PHT
                │           │          │
            PH Adapter      │      PHT Adapter
                │           │          │
            PH branch    Stage1     PHT branch
                         PT branch
```

特别把当前：

```text
S1
PH · PT · PHT

S3
PT · PHT
```

改成类似：

```text
S1-base
PH · PT · PHT
FROZEN

ΔS1-PH
PH

ΔS1-PHT
PHT
```

以及：

```text
S3-base
PT · PHT
FROZEN

ΔS3-PHT
PHT
```

PT 不需要 ΔS3。

于是：

$$
\boxed{
z_T^{PT}=S3_{base}(T)
}
$$

而：

$$
\boxed{
z_T^{PHT}
=
S3_{base}(T)
+
\Delta S3_{PHT}(T)
}
$$

这两个公式实际上就是整个方案最重要的设计。

---

# 23. 推荐你最终采用的完整训练流程

你的整个训练流程可以重新整理为：

### Stage1：PH/PT representation pretraining

交替：

$$
PH\rightarrow PT\rightarrow PH\rightarrow PT...
$$

优化：

$$
L_{Stage1}
=
\lambda_{PH}L_{PH}
+
\lambda_{PT}L_{PT}
$$

保存：

$$
\boxed{M_{Stage1}^{best}}
$$

---

### Stage2

如果你前面还有一致性约束、FGM 等，可以保持原设计。

---

### Stage3：PHT task expansion

从：

$$
M_{Stage1}^{best}
$$

初始化。

但不要继续全量修改 S1/S3。

改成：

$$
\theta_{base}
\leftarrow
M_{Stage1}^{best}
$$

冻结：

$$
\theta_{PT-critical}
$$

增加：

$$
A_P^{PHT}
$$

$$
A_H^{PHT}
$$

$$
A_T^{PHT}
$$

$$
CrossAttn_{PHT}
$$

$$
Classifier_{PHT}
$$

训练：

$$
L_{PHT}
$$

也可以继续混合 PH/PHT：

$$
L =
\lambda_{PH}L_{PH}
+
\lambda_{PHT}L_{PHT}
$$

而 PT Stage1 expert 保持不动。

---

### 最终 inference

Router：

$$
R(P,H,T)
$$

执行：

$$
(P,H,-)
\rightarrow
PH\ Expert
$$

$$
(P,-,T)
\rightarrow
Stage1\ PT\ Expert
$$

$$
(P,H,T)
\rightarrow
PHT\ Adapter\ Expert
$$

这样整个逻辑非常干净。

---

# 24. 最后，我尤其建议你不要把问题理解成“Stage3 不好，所以回退 Stage1”

更好的科研叙事应该是：

> **不同任务对共享生物实体表示存在不同优化偏好。PHT 的引入使 Peptide/TCR 表示由直接 PT-specific representation 向 context-dependent pHLA-TCR representation 偏移，从而导致负迁移。为此，引入 task-aware stage-specific adapters，在共享生物语言模型表示的基础上实现任务特异的参数增量，并通过 deterministic routing 保留 Stage1 的二元识别能力，同时允许 PHT 学习三元交互。**

这会比：

> “哪个 Stage 好就用哪个”

强很多。

---

## 我建议你最终采用这个版本

$$
\boxed{
\text{Frozen Stage1 Base}
+
\text{Task-aware Hard Router}
+
\text{Task-specific Residual Adapters}
}
$$

具体：

$$
\boxed{
PT=Stage1\ frozen\ expert
}
$$

$$
\boxed{
PHT=Stage1\ base+PHT\ adapter
}
$$

PH 根据验证集决定：

$$
\boxed{
PH=Stage1\ base
\quad or\quad
Stage1\ base+PH\ adapter
}
$$

同时用 **gradient cosine + representation drift** 去证明为什么需要这样的 Adapter isolation。

如果你的目标是既保住 Stage1 的 PT 最优性能，又继续提升 PHT，我认为这比让 Stage3 继续同时更新 S1/S3 要更加合理。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)