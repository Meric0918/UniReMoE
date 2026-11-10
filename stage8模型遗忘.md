# 八、stage8只使用PHT数据可能产生任务遗忘

这是当前方案最大的工程风险。
经过stage3，模型已经学会：

- PH任务；
- PT任务；
- PHT任务。

stage8只输入PHT数据，如果继续更新共享编码器，那么共享表示会逐渐适应PHT数据分布，可能造成：

$$
PH\text{性能下降}
$$

$$
PT\text{性能下降}
$$

这就是多任务微调中的灾难性遗忘。

仅仅冻结：

```python
ph_head.requires_grad = False
pt_head.requires_grad = False
```

还不完全够，因为共享的peptide、HLA和TCR编码器仍然会更新，PH/PT预测仍可能漂移。

---

# 九、解决遗忘的三种方案

## 方案一：stage3模型作为冻结教师，最推荐

stage3训练完成后保存：

```text
Teacher：冻结的Stage 3模型
Student：从Stage 3权重初始化的Stage 8模型
```

在stage8中：

```text
PHT样本
   ├── Teacher → pPH_teacher、pPT_teacher
   └── Student → pPHT、pPH、pPT
```

加入保持损失：

$$
\mathcal L_{keep}
=
KL(p_{PH}^{teacher}\Vert p_{PH}^{student})
+
KL(p_{PT}^{teacher}\Vert p_{PT}^{student})
$$

这样既能加入新的约束，又不会让PH/PT能力大幅漂移。

不过教师输出不要直接作为绝对真值，只用于保持阶段1已有知识。

---

## 方案二：少量PH/PT数据回放

stage8仍以PHT为主，但偶尔加入PH和PT batch：

```text
80% PHT batch
10% PH batch
10% PT batch
```

或者每训练4个PHT batch：

```text
4 × PHT batch
1 × PH batch
1 × PT batch
```

这通常是最容易实现、效果也比较稳定的方法。

stage8变成：

$$
\begin{aligned}
\mathcal L_{\text{stage2}}
=&
\mathcal L_{PHT}^{weighted}
+\lambda_{hier}\mathcal L_{hier}\\
&+\lambda_{PT-imp}\mathcal L_{PT-imp}\\
&+\lambda_{replay}
\left(
\mathcal L_{PH}^{replay}
+
\mathcal L_{PT}^{replay}
\right)
\end{aligned}
$$

---

## 方案三：部分冻结

可以冻结：

- ESMC主体；
- TCR-BERT主体；
- PH/PT预测头；

只训练：

- 交互层；
- PHT融合层；
- 一致性适配层；
- PHT预测头。

这种方法最稳定，但模型适应PHT任务的空间也最小。

对于你的模型，我更推荐：

> 冻结ESMC和TCR-BERT大部分参数，只解冻顶部1～2层和交互模块，同时保留少量PH/PT数据回放。

---