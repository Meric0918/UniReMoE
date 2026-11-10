# PLM Stage 4：pHLA–TCR 双向 Cross-Attention 消融实验

> 状态：设计与实现规格，尚未实现  
> 日期：2026-07-29  
> 定位：Stage 3 的并行架构消融，不替换、不覆盖现有 Stage 3  
> 主实验：保留 peptide→HLA 单向条件化，仅将第二级 pHLA–TCR 交互扩展为双向

## 1. 研究问题与结论边界

Stage 3 已经实现显式的 HLA-peptide-TCR 三元预测，但两级 cross-attention 都只更新
peptide 位置轴上的表示：

```text
peptide --queries--> HLA
    │
    └── HLA-conditioned peptide --queries--> TCR
                                      │
                                      └── peptide轴表示 → pooling → classifier
```

Stage 4 要回答的核心问题是：

> 在保持 Stage 3 的 pHLA 建模方式、数据、优化预算和二元回放策略不变时，显式增加
> `TCR queries pHLA` 的反向分支，能否稳定提高三元识别性能？

Stage 4 只验证第二级 pHLA–TCR 双向交互，不包含以下改动：

- 不把 peptide–HLA 第一级改成双向；
- 不引入 BLIP-2 式固定数量 learned query；
- 不加入 Q-Former 的多层 query self-attention；
- 不拼接 HLA、peptide、TCR 后做全局 self-attention；
- 不改变三元数据清理、split、标签和负样本定义；
- 不把 Stage 4 自动视为 Stage 3 的替代模型。

只有在预注册的三种子消融标准满足后，才能把双向结构列为后续正式模型候选。

## 2. Stage 3 基线的代码事实

### 2.1 三个序列编码器

Stage 3 在线调用三个独立编码器：

| 输入 | 编码器 | PLM输出维度 | Adapter输出维度 |
|---|---|---:|---:|
| peptide | 独立 ESM-C | 960 | 128 |
| HLA | 另一个独立 ESM-C | 960 | 128 |
| TCR | TCR-BERT backbone | 768 | 128 |

Peptide 和 HLA 的 ESM-C 从相同基础 checkpoint 构建，但属于两个独立实例，不共享
参数。TCR-BERT 只使用去除原分类头后的 backbone。Stage 3 从 `triplet_best` 推理
checkpoint 加载时，三个编码器、三个 Adapter、pHLA/pTCR 二元分支和三元正向分支
都继承 Stage 3 联合最佳权重。

三个 Adapter 统一将 residue hidden states 映射到 `model_dim=128`：

```text
Linear(input_dim, 256)
GELU
Dropout(0.1)
Linear(256, 128)
LayerNorm(128)
```

特殊 token 会在进入 Adapter 前移除，padding 位置通过 residue mask 清零。

### 2.2 Stage 3 的单向注意力

`source/plm_unified/attention.py` 中的 `CrossAttentionBlock` 固定为：

```python
attended, weights = self.attention(
    query=peptide,
    key=receptor,
    value=receptor,
    key_padding_mask=~receptor_mask,
)
```

因此第一级 pHLA 条件化为：

$$
P=\operatorname{Adapter}_{pep}(H_{pep}),\quad
M=\operatorname{Adapter}_{HLA}(H_{HLA}),\quad
T=\operatorname{Adapter}_{TCR}(H_{TCR})
$$

$$
P_{HLA}
=
\operatorname{CrossAttn}_{pHLA}
\left(
Q=P,\ K=M,\ V=M
\right)
$$

其中：

```text
P      : [B, Lp, 128]
M      : [B, Lh, 128]
P_HLA  : [B, Lp, 128]
A_pHLA : [B, heads, Lp, Lh]
```

第二级三元正向注意力为：

$$
C_{forward}
=
\operatorname{CrossAttn}_{forward}
\left(
Q=P_{HLA},\ K=T,\ V=T
\right)
$$

其中：

```text
T          : [B, Lt, 128]
C_forward  : [B, Lp, 128]
A_forward  : [B, heads, Lp, Lt]
```

这里“单向”指只有 Query 一侧产生更新后的输出。HLA/TCR 仍然可以通过 K/V 路径获得
梯度，但模型不会显式保留更新后的 HLA/TCR token 序列。

### 2.3 当前设计的优势与限制

当前单向串行结构具有明确的生物学归纳偏置：

```text
HLA提供呈递环境
    ↓
形成HLA-conditioned peptide
    ↓
peptide位置从TCR中提取识别相关信息
```

它计算量低，并能输出 peptide–HLA 和 peptide–TCR 的残基注意力图。但最终分类只
池化 peptide 位置轴，TCR 作为独立交互主体的条件化表示没有保留下来。Stage 4
针对这一限制增加反向流。

## 3. 与 Q-Former 的关系

BLIP-2 Q-Former 使用固定数量的可学习 query token，通过多层 self-attention 和
cross-attention，从冻结图像编码器中提取固定长度信息，再连接冻结 LLM。它主要解决
大型冻结视觉编码器与大型冻结语言模型之间的信息瓶颈和模态对齐问题。

参考：

- BLIP-2: <https://proceedings.mlr.press/v202/li23q.html>

Stage 4 只借鉴“使用轻量 cross-attention 桥接预训练编码器”的思想，不复制标准
Q-Former，原因如下：

1. peptide、HLA 和 TCR 都是较短的 residue sequence，不需要把数百个 image patch
   压缩成固定 query 数；
2. 当前任务是二分类，没有需要接收固定长度 soft prompt 的冻结 LLM；
3. peptide residue 本身是样本相关、具有生物学含义的自然 Query；
4. 固定 learned query 会削弱 peptide residue 与 HLA/TCR residue 的直接对齐；
5. Q-Former 的收益还依赖对比、匹配和生成等预训练目标，不能只归因于模块形状。

因此 Stage 4 采用 residue-level 双流 co-attention，而不是 learned-query
Q-Former。

## 4. Stage 4 主架构

### 4.1 总体数据流

```text
                           ┌─────────────────────────────────────┐
peptide ── ESM-C ─ Adapter ──┐                                  │
                              ├─ peptide queries HLA ── P_HLA ───┤
HLA ────── ESM-C ─ Adapter ──┘                                  │
                                                                 │
                       ┌─ P_HLA queries TCR ── C_forward ─ pool ─┤
P_HLA ─────────────────┤                                         ├─ concat ─ classifier
                       └─ TCR queries P_HLA ── C_reverse ─ pool ─┤
                                                                 │
TCR ─── TCR-BERT ─ Adapter ──────────────────────────────────────┘
```

第一级 peptide→HLA 完全沿用 Stage 3：

$$
P_{HLA}
=
\operatorname{CrossAttn}_{pHLA}(Q=P,K=M,V=M)
$$

第二级保留 Stage 3 正向分支：

$$
C_f
=
\operatorname{CrossAttn}_{f}(Q=P_{HLA},K=T,V=T)
$$

并新增反向分支：

$$
C_r
=
\operatorname{CrossAttn}_{r}(Q=T,K=P_{HLA},V=P_{HLA})
$$

对应张量为：

```text
C_f : [B, Lp, 128]
C_r : [B, Lt, 128]
A_f : [B, heads, Lp, Lt]
A_r : [B, heads, Lt, Lp]
```

`A_f` 和 `A_r` 具有不同的 Q/K/V 投影和 softmax 归一化维度，不能将其中一个简单
转置后视为另一个。

### 4.2 双流池化

正向流沿用 Stage 3 的 `triplet_pooling`：

$$
u_f
=
\operatorname{AttnPool}_{f}(C_f,\ mask=P_{mask})
\in \mathbb{R}^{B\times128}
$$

反向流新增独立 `reverse_triplet_pooling`：

$$
u_r
=
\operatorname{AttnPool}_{r}(C_r,\ mask=T_{mask})
\in \mathbb{R}^{B\times128}
$$

两个 pooling 不共享参数。正向池化权重长度为 `Lp`，反向池化权重长度为 `Lt`。
padding residue 的权重必须严格为零。

### 4.3 拼接融合与分类

Stage 4 不使用标量或向量 gate，固定采用拼接：

$$
u_{bi}
=
[u_f;u_r]
\in \mathbb{R}^{B\times256}
$$

分类器为：

```text
Linear(256, 64)
GELU
Dropout(0.2)
Linear(64, 2)
```

最终输出：

$$
\hat y
=
\operatorname{Classifier}_{bi}(u_{bi})
$$

选择拼接的原因：

- 不强制两个方向在同一128维空间逐元素对齐；
- 比向量门控更容易解释和初始化；
- 可以精确复制 Stage 3 分类决策作为初始状态；
- 反向流的独立贡献可以通过分类器后128列和消融实验分析。

### 4.4 建议的模型输出接口

未来实现中的输出对象应至少包含：

```python
@dataclass
class Stage4BindingOutput:
    logits: torch.Tensor
    phla_cross_attention: torch.Tensor | None = None
    triplet_cross_attention: torch.Tensor | None = None
    reverse_triplet_cross_attention: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None
    reverse_pooling_weights: torch.Tensor | None = None
```

兼容性约定：

- 原 `triplet_cross_attention` 字段继续表示正向 `pHLA queries TCR`；
- 新字段 `reverse_triplet_cross_attention` 表示 `TCR queries pHLA`；
- 原 `pooling_weights` 继续表示 peptide 轴权重；
- 新字段 `reverse_pooling_weights` 表示 TCR 轴权重；
- pHLA/pTCR 二元推理接口保持不变。

## 5. Stage 3 权重继承与初始化

### 5.1 初始化源

每个 Stage 4 运行必须从相同 fold/seed 的 Stage 3 部署 checkpoint 初始化：

```text
trained_model/PLM_Triplet/
  fold_<fold>/
    seed_<seed>/
      triplet_best_<timestamp>.pt
```

要求源 checkpoint：

```text
format_version  = 1
checkpoint_type = stage3_multitask_model
tasks           = [phla, ptcr, triplet]
```

加载器必须校验：

- Stage 3 模型配置；
- fold 和 seed；
- peptide/HLA/TCR 三个基础 PLM 的 fingerprint；
- Adapter、二元分支和三元正向分支参数完整；
- source Stage 2 元数据存在；
- 数据审计信息和 triplet split 定义兼容。

Stage 4 使用全新的 optimizer、scheduler 和 GradScaler，不加载 Stage 3 的训练状态。

### 5.2 参数继承

以下模块完整继承 Stage 3：

- peptide ESM-C；
- HLA ESM-C；
- TCR-BERT；
- peptide/HLA/TCR Adapter；
- pHLA/pTCR 两个二元分支；
- `phla_cross_attention`；
- 正向 `triplet_cross_attention`；
- 正向 `triplet_pooling`；
- Stage 3 `triplet_classifier` 的可兼容部分。

以下模块为 Stage 4 新增：

- `reverse_triplet_cross_attention`；
- `reverse_triplet_pooling`；
- 输入扩展到256维的 `bidirectional_triplet_classifier`。

反向 attention 和 pooling 使用 PyTorch 模块默认初始化，并依靠 Stage 4A 预热。
不从正向 attention 直接复制，因为正向模块的 Q 投影学习的是 pHLA 表示，而反向模块
的 Q 投影需要处理 TCR 表示，二者语义不等价。

### 5.3 分类器的基线保持初始化

设 Stage 3 分类器第一层参数为：

```text
W3 : [64, 128]
b3 : [64]
```

Stage 4 第一层为：

```text
W4 : [64, 256]
b4 : [64]
```

初始化规则固定为：

```python
W4[:, :128] = W3
W4[:, 128:] = 0
b4 = b3
```

最后一个 `Linear(64, 2)` 的 weight 和 bias 完整复制。由此在 `eval()` 模式下：

$$
\operatorname{logits}_{stage4,init}
=
\operatorname{logits}_{stage3}
$$

初始化等价性必须通过严格单元测试验证。由于反向部分权重初始为零，第一次反向传播
主要更新分类器后128列；这些权重变为非零后，梯度再进入反向 pooling 和 attention。
Stage 4A 的两个完整 epoch 足以覆盖这一自然启动过程，不额外引入手工 bootstrap。

## 6. Stage 4 的实验定位

Stage 4 是消融分支，而不是新的默认生产阶段：

```text
Stage 3 triplet_best
    ├── 原始Stage 3直接评估
    ├── Stage 4单向续训control
    └── Stage 4双向实验模型
```

三个模型必须从同一个 Stage 3 checkpoint 和相同数据定义出发。Stage 4 的所有输出
写入独立目录，不得覆盖 Stage 3 checkpoint、指标或预测文件。

初始实验固定：

```text
fold  = 1
seeds = [42, 3407, 2026]
```

当前仓库尚未产生对应的 Stage 3 `triplet_best` 文件，因此 Stage 4 正式训练的先决
条件是先完成三个种子的 Stage 3 训练。文档中的 Stage 3 路径在此之前均为占位符。

## 7. 三模型三种子对照设计

### 7.1 模型组

| 组别 | 初始化 | 架构 | 是否继续训练 | 目的 |
|---|---|---|---|---|
| `stage3_baseline` | Stage 3 `triplet_best` | 原单向 | 否 | 记录原始基线 |
| `unidirectional_control` | 同一 `triplet_best` | 原单向 | 是 | 排除额外训练预算收益 |
| `bidirectional` | 同一 `triplet_best` | 二级双向 | 是 | 测量反向分支增益 |

`unidirectional_control` 和 `bidirectional` 必须使用：

- 相同 triplet/pHLA/pTCR 数据；
- 相同 batch size 和 gradient accumulation；
- 相同 epoch/round 数量上限；
- 相同 optimizer-step 预算；
- 相同 scheduler 类型和 warmup 比例；
- 相同 early stopping 和模型选择规则；
- 相同随机种子；
- 相同 validation 频率；
- 相同 PLM 解冻范围。

二者参数量不同是架构消融本身的一部分，必须同时报告可训练参数量、总参数量、显存
峰值和每 optimizer step 耗时。

### 7.2 Stage 4A：新增交互头预热

默认固定训练2个 triplet epoch。

双向实验组冻结：

- 三个 PLM；
- 三个 Adapter；
- pHLA/pTCR 二元分支；
- `phla_cross_attention`；
- 正向 `triplet_cross_attention` 和正向 pooling。

只训练：

- `reverse_triplet_cross_attention`；
- `reverse_triplet_pooling`；
- `bidirectional_triplet_classifier`。

单向续训 control 在相同数据和 optimizer steps 下只训练原有 triplet head：

- 正向 `triplet_cross_attention`；
- 正向 `triplet_pooling`；
- 原 `triplet_classifier`。

Stage 4A 每个 epoch 后在 triplet validation 上计算指标，以
`triplet_score=(AUROC+AUPR)/2` 保存各自最佳预热 checkpoint。完成后回载最佳权重
进入 Stage 4B。

### 7.3 Stage 4B：渐进联合微调

每轮训练顺序保持：

```text
pHLA replay
    ↓
pTCR replay
    ↓
完整 triplet epoch
    ↓
三个 validation task
```

二元 replay 各请求：

$$
N_{replay}
=
\left\lceil
0.25\times N_{triplet\_train}
\right\rceil
$$

Stage 4B 的 triplet 任务激活：

- 三个 Adapter；
- `phla_cross_attention`；
- 正向 triplet attention/pooling；
- 双向组的反向 attention/pooling；
- 当前组对应 classifier；
- peptide/HLA/TCR 三个 PLM 顶部各2层。

pHLA replay 只训练 pHLA 路径及 peptide/HLA 对应顶部层；pTCR replay 只训练 pTCR
路径及 peptide/TCR 对应顶部层。反向三元分支不能使用二元标签更新。

默认训练控制：

```text
min_rounds              = 3
max_rounds              = 10
early_stopping_patience = 3
early_stopping_min_delta= 1e-4
binary_replay_ratio     = 0.25
```

联合选模分数保持：

$$
S_t=\frac{AUROC_t+AUPR_t}{2}
$$

$$
S_{joint}
=
0.6S_{triplet}
+0.2S_{pHLA}
+0.2S_{pTCR}
$$

### 7.4 优化器默认值

Stage 4 计划沿用 Stage 3 的默认超参数：

| 参数组 | 学习率 |
|---|---:|
| 三个 PLM 最顶层 | `5e-6` |
| 三个 PLM 倒数第二层 | `2.5e-6` |
| 三个 Adapter | `2.5e-5` |
| 继承的 pHLA/pTCR attention、pooling、classifier | `5e-5` |
| 正向 triplet 模块 | `1e-4` |
| 新反向 attention/pooling | `1e-4` |
| 双向融合 classifier | `1e-4` |

其他默认值：

```text
optimizer               = AdamW
weight_decay            = 0.01
scheduler               = linear warmup + cosine decay
scheduler_warmup_ratio  = 0.05
gradient_clip_norm      = 1.0
mixed_precision         = bf16
gradient_checkpointing  = true
micro_batch_size        = 4
eval_batch_size         = 32
gradient_accumulation   = 16
```

Stage 4A 和 Stage 4B 因 trainable 参数范围不同，必须分别创建 optimizer 和
scheduler。

## 8. 计划配置接口

建议未来新增 `configs/plm_stage4.yaml`，继承 Stage 3 的 paths/model/training 数据
定义，并新增独立的 `stage4` 节：

```yaml
stage4:
  architecture_variant: bidirectional
  fusion: concat
  source_checkpoint_type: stage3_multitask_model
  head_warmup_epochs: 2
  min_rounds: 3
  max_rounds: 10
  early_stopping_patience: 3
  early_stopping_min_delta: 0.0001
  binary_replay_ratio: 0.25
  triplet_score_weight: 0.6
  phla_score_weight: 0.2
  ptcr_score_weight: 0.2
  peptide_unfrozen_layers: 2
  hla_unfrozen_layers: 2
  tcr_unfrozen_layers: 2
  plm_top_lr: 0.000005
  plm_lower_lr: 0.0000025
  adapter_lr: 0.000025
  pretrained_attention_lr: 0.00005
  triplet_module_lr: 0.0001
  reverse_module_lr: 0.0001
  fusion_classifier_lr: 0.0001
  weight_decay: 0.01
  scheduler_warmup_ratio: 0.05
  gradient_clip_norm: 1.0
  mixed_precision: bf16
  gradient_checkpointing: true
```

`architecture_variant` 只允许：

```text
unidirectional_control
bidirectional
```

不提供隐式 fallback。checkpoint 中保存的 variant 必须与恢复时配置一致。

## 9. 计划训练与评估入口

以下接口为未来实现约定，当前仓库中尚不存在对应脚本。

双向实验：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage4.py \
  --config configs/plm_stage4.yaml \
  --stage3-checkpoint \
    trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_<timestamp>.pt \
  --variant bidirectional \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

单向续训 control：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage4.py \
  --config configs/plm_stage4.yaml \
  --stage3-checkpoint \
    trained_model/PLM_Triplet/fold_1/seed_42/triplet_best_<timestamp>.pt \
  --variant unidirectional_control \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

恢复：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/train_plm_stage4.py \
  --config configs/plm_stage4.yaml \
  --resume <stage4_last_checkpoint> \
  --device cuda
```

评估：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage4.py \
  --config configs/plm_stage4.yaml \
  --checkpoint <stage4_best_model> \
  --split test \
  --device cuda \
  --hla-shuffle \
  --tcr-shuffle
```

## 10. 计划产物与 checkpoint 规范

建议使用独立目录：

```text
trained_model/PLM_Stage4_Ablation/
  fold_1/
    seed_<seed>/
      unidirectional_control/
      bidirectional/
```

每个续训组产生：

```text
stage4a_best_triplet_<timestamp>.pt
stage4b_best_joint_<timestamp>.pt
stage4_last_<timestamp>.pt
stage4_<variant>_best_<timestamp>.pt
```

完整恢复 checkpoint：

```text
format_version  = 1
checkpoint_type = stage4_ablation_training
```

部署/评估 checkpoint：

```text
format_version  = 1
checkpoint_type = stage4_ablation_model
```

checkpoint 至少保存：

- `architecture_variant`；
- `fusion=concat`；
- 完整模型 state dict；
- source Stage 3 checkpoint 绝对路径及 metadata；
- source Stage 2 metadata；
- 三个 PLM fingerprint；
- config；
- fold、seed、run timestamp；
- Stage 4A epoch、Stage 4B round；
- optimizer、scheduler、GradScaler（仅恢复点）；
- global RNG 和 DataLoader RNG（仅恢复点）；
- validation metrics、joint score 和选定阈值；
- 数据审计和 shuffle 配置；
- 正向/反向模块参数量。

恢复训练必须验证 variant、fusion、fold、seed、模型配置和 encoder fingerprint。

## 11. Validation、阈值与独立测试

### 11.1 阈值规则

训练和模型选择只使用 validation：

1. 最大化 MCC；
2. MCC 并列时最大化 F1；
3. 再并列时选择离0.5更近的阈值；
4. 最终阈值写入部署 checkpoint；
5. test 固定使用该阈值，禁止在 test 重新选择。

阈值无关指标 `AUROC/AUPR` 是架构比较的主要依据。ACC、MCC、F1、precision、
recall 和 specificity 作为补充。

### 11.2 Shuffle 依赖性分析

保留现有确定性 HLA cyclic-shift：

```text
(peptide_i, TCR_i, HLA_i)
→
(peptide_i, TCR_i, HLA_{i+offset})
```

新增同规则 TCR shuffle：

```text
(peptide_i, TCR_i, HLA_i)
→
(peptide_i, TCR_{i+offset}, HLA_i)
```

再增加 HLA+TCR 双重 shuffle。每种 shuffle 必须记录：

- changed fraction；
- AUROC/AUPR/MCC/F1；
- 原始与 shuffle 后的平均绝对概率变化；
- 阳性和阴性样本分别的概率变化；
- 正向和反向 attention entropy；
- 正向/反向 pooling entropy。

Shuffle 只用于判断模型是否依赖相应输入，不能把 shuffle 后样本解释为具有可靠阴性
标签的新数据。

### 11.3 注意力解释边界

正向注意力：

```text
A_forward[b, h, peptide_position, tcr_position]
```

表示某个 HLA-conditioned peptide residue 从 TCR residue 中提取信息的相对权重。

反向注意力：

```text
A_reverse[b, h, tcr_position, peptide_position]
```

表示某个 TCR residue 从 HLA-conditioned peptide residue 中提取信息的相对权重。

二者是模型内部关联权重，不等价于：

- 原子接触概率；
- 结合自由能贡献；
- 因果重要性；
- 已验证的生物学 interaction map。

需要与突变扫描、integrated gradients、结构距离或独立实验联合验证后，才可进行更强
解释。

## 12. 三种子汇总与预注册验收标准

每个 seed 分别报告三组模型结果，再汇总均值、标准差和逐 seed 差值：

```text
seeds = [42, 3407, 2026]
```

主比较不是 `bidirectional - stage3_baseline`，而是：

$$
\Delta S_{triplet}
=
S_{triplet}^{bidirectional}
-
S_{triplet}^{unidirectional\_control}
$$

这样可以降低额外训练轮次造成的混淆。

双向结构被判定为“值得后续正式验证”必须同时满足：

1. 三种子的平均 `Δ triplet_score >= 0.005`；
2. 至少2/3个 seed 的 `Δ triplet_score > 0`；
3. pHLA 平均 task score 相对 control 下降不超过 `0.005`；
4. pTCR 平均 task score 相对 control 下降不超过 `0.005`；
5. 训练过程无 NaN/Inf、异常梯度和系统性 early collapse；
6. 三种子结论不由单个异常 seed 单独驱动。

以下作为重要诊断，但不设置硬 promotion threshold：

- HLA shuffle 性能下降；
- TCR shuffle 性能下降；
- 双重 shuffle 性能下降；
- 反向分类器权重从零产生稳定非零更新；
- 反向注意力没有全部坍缩到同一 residue 或 padding；
- 增加的训练时间和显存可接受。

架构选择必须仅依据 validation。冻结架构、超参数和阈值规则后，才能运行一次独立
test。达到上述标准也不会自动替换 Stage 3，只形成下一阶段候选结论。

可选的统计补充固定为：

```text
paired bootstrap samples = 10,000
bootstrap seed           = 20260729
confidence interval      = 95%
```

bootstrap 在同一 split 的样本级预测上成对重采样，报告 AUROC、AUPR 和 MCC 差值
区间，不用三个位点 seed 直接估计正态置信区间。

## 13. 实现测试规格

### 13.1 模型形状

合成输入：

```text
B=2, heads=4, Lp=5, Lh=6, Lt=7, D=128
```

必须验证：

```text
logits.shape                          == [2, 2]
phla_cross_attention.shape            == [2, 4, 5, 6]
triplet_cross_attention.shape         == [2, 4, 5, 7]
reverse_triplet_cross_attention.shape == [2, 4, 7, 5]
pooling_weights.shape                 == [2, 5]
reverse_pooling_weights.shape         == [2, 7]
```

### 13.2 Mask

必须验证：

- HLA padding 不获得 pHLA cross-attention 权重；
- TCR padding 不获得正向 attention 权重；
- peptide padding 不获得反向 attention 权重；
- peptide padding 不参与正向 pooling；
- TCR padding 不参与反向 pooling；
- 任一输入不存在真实 residue 时立即报错；
- 特殊 token 不进入 residue mask。

### 13.3 初始化等价性

对相同 Stage 3 checkpoint 和相同 batch，在 `eval()` 且 dropout 关闭时：

```text
stage4_initial_logits == stage3_logits
```

使用严格 tensor close 断言，并验证：

```text
W4[:, :128] == W3
W4[:, 128:] == 0
```

### 13.4 梯度与可训练性

Stage 4A 必须验证：

- 三个 PLM 无梯度；
- 三个 Adapter 无梯度；
- 正向 triplet attention/pooling 无梯度；
- 反向 classifier 后半权重在首个 step 更新；
- 后续 step 中反向 pooling/attention 获得有限非零梯度。

Stage 4B 必须验证：

- 三个 PLM 仅顶部2层可训练；
- 其他 PLM 层保持冻结；
- triplet 训练激活正向和反向分支；
- pHLA/pTCR replay 不更新反向三元分支；
- 所有梯度范数为有限值。

### 13.5 Checkpoint 与恢复

必须覆盖：

- 从合法 Stage 3 `triplet_best` 初始化；
- 拒绝 Stage 2、错误 fold/seed 和 encoder fingerprint 不匹配；
- Stage 4A→Stage 4B 最佳权重回载；
- `stage4_last` 恢复 optimizer/scheduler/scaler；
- 恢复 DataLoader RNG 和全局 RNG；
- 连续训练与中断恢复得到一致的 batch 顺序和状态；
- control checkpoint 不能以 bidirectional 配置恢复，反之亦然。

### 13.6 消融与评估

必须覆盖：

- HLA shuffle 保留 peptide/TCR/label/source row；
- TCR shuffle 保留 peptide/HLA/label/source row；
- shuffle 可复现并记录 changed fraction；
- validation 可选阈值、test 禁止选阈值；
- 三模型汇总按 fold/seed/variant 对齐；
- 缺少任一 seed 时汇总明确标记 incomplete，而不是静默计算。

## 14. 风险与失败模式

### 14.1 反向分支被分类器忽略

由于分类器后128列初始为零，可能长期保持较小。训练日志必须记录：

- 前128列与后128列 weight norm；
- 两部分 gradient norm；
- `u_f` 和 `u_r` 的均值、方差和范数；
- 打乱或置零 `u_r` 后的概率变化。

如果反向部分始终接近零，应判定为模型未使用新分支，而不是把相同指标解释为双向
结构无条件等价。

### 14.2 过拟合

双向模型增加 attention、pooling 和分类器参数。若训练指标上升但 validation 下降，
不应继续增加 Q-Former 层数或改成两级双向；应优先检查：

- dropout 和 weight decay；
- 早停；
- 反向分支学习率；
- 负样本构造偏差；
- HLA/TCR/peptide 重复和同源泄漏。

### 14.3 额外训练预算混淆

如果双向模型优于未经续训的 Stage 3、但不优于单向续训 control，则不能把收益归因
于双向架构。最终结论必须以双向相对等预算 control 的差值为准。

### 14.4 顺序归纳偏置仍然存在

Stage 4 虽增加第二级双向流，第一级仍先计算 HLA-conditioned peptide，因此并非完全
对称的三元图模型。若 Stage 4 无收益，只能说明当前数据和训练条件下“第二级反向流”
未产生稳定提升，不能证明所有双向、迭代或结构感知模型都无效。

## 15. 后续扩展顺序

只有 Stage 4 消融得到稳定证据后，才按以下顺序考虑更复杂结构：

1. 在 pHLA→TCR 与 TCR→pHLA 之间做多轮迭代；
2. 在 HLA 条件化之后加入 peptide query self-attention；
3. 将 peptide–HLA 第一级扩展为双向；
4. 引入少量 interaction tokens，但保留 peptide residue queries；
5. 最后才考虑完整 learned-query Q-Former。

每次只改变一个主要结构因素，并保留等预算 control。

## 16. 实现清单

未来实际开发按以下顺序进行：

1. 增加 Stage 4 配置和严格校验；
2. 实现反向 attention、反向 pooling、拼接分类器和基线保持初始化；
3. 实现 Stage 3 `triplet_best` 加载与兼容性审计；
4. 实现 Stage 4A/4B trainability 和优化器参数组；
5. 实现单向续训 control；
6. 实现 checkpoint、恢复和独立输出目录；
7. 增加 TCR shuffle、双重 shuffle 和 attention 导出；
8. 完成合成单元测试和 `--limit` smoke test；
9. 依次运行 fold 1 的三个 seed；
10. 只使用 validation 汇总架构结论；
11. 冻结方案后运行一次独立 test；
12. 将结果和是否满足预注册标准追加回本文档。

在代码实现和实验结果完成前，本文所有 Stage 4 脚本名、配置项和 checkpoint 类型均
属于已锁定的设计接口，而不是对当前仓库已有功能的描述。
