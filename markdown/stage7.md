# Stage 7：Stage 3 与 Stage 1 的 HLA 条件增益评估

本文件集中整理 Stage 7 的评估方案和 fold4/seed2026 实际报告。

## 阶段说明与评估方案

> 来源：`PLM_STAGE7.md`

Stage 7 是独立的只读评估阶段，用于检验 Stage 3 三元模型加入 HLA 后，
是否比 Stage 1 的 peptide–TCR 二元模型获得更好的识别性能。它不训练模型、
不创建 checkpoint、不使用 SwanLab，也不修改 Stage 1、Stage 3 或 Stage 5
的配置、源码和产物。

Stage 7 读取同 fold/seed 的一个 `stage3_multitask_model` checkpoint，以及
Stage 1 的 pHLA、pTCR 两个 `task_model` checkpoint。Stage 3 在训练中微调过
三个基础 PLM 的顶部层，因此评估会加载 checkpoint 中的 Peptide、HLA 和 TCR
encoder 权重；Stage 1 仍使用原始冻结 PLM。两者不会错误地共享 residue
embedding。为保证比较具有明确继承关系，Stage 3 checkpoint 必须记录为由同
fold/seed 的 Stage 1 初始化。

## 1. 主要性能实验

在严格清理后的原始 triplet validation/test 标签上比较：

- `Stage1 pTCR`：不输入 HLA，是主要二元基线；
- `Stage1 late fusion`：validation 标准化后的 pHLA+pTCR 可加基线；
- `Stage3 pTCR`：同一个 Stage 3 checkpoint 中移除 HLA 路径的消融；
- `Stage3 late fusion`：Stage 3 二元分支的可加基线；
- `Stage3 PHT`：完整 HLA–peptide–TCR 三元分支。

AUROC、AUPR 是主要阈值无关指标。每个模型的分类阈值只在 triplet validation
上按 MCC 选择，然后原样用于 test。报告同时给出全局指标，以及按 peptide
等权的 accuracy gain bootstrap 95% CI。

只有当 Stage3 PHT 相对 Stage1 pTCR 的 AUROC、AUPR 均提高，且 peptide-macro
accuracy gain 的 95% CI 下界大于 0，才判定 Stage 3 优于 Stage 1。相对同一
Stage 3 checkpoint 的 pTCR 消融也满足该条件，才能更有力地把收益归因于三元
分支中的 HLA 条件化。

## 2. HLA 使用证据

性能提高本身可能来自目标任务训练或模型容量，因此同时执行三项 HLA
检验：

1. 固定阳性 peptide–TCR，把原始 HLA 与最多 5 个 pHLA-valid 错配 HLA 排序；
2. 对同一 peptide 构造 2×2 TCR–HLA tetrad，计算非加性交互：

   ```text
   D = (m(p,t1,h1) + m(p,t2,h2) - m(p,t1,h2) - m(p,t2,h1)) / 2
   ```

3. 复现确定性的 HLA 循环位移，观察 Stage3 PHT 性能下降。

Stage 1/3 的可加 pHLA+pTCR 分数在 tetrad 中理论上 `D=0`。Stage3 PHT 只有在
peptide-macro D 的 95% CI 下界大于 0、单侧 sign-flip `p<0.05`，且相对 Stage3
late fusion 的差值 CI 下界大于 0 时，才支持正向非加性 TCR×HLA 相容性。

合成 HLA decoy 是未观测组合，不是实验确认的阴性三元组。Stage3 还接受了
Stage1 没有接受的 PHT 监督，因此 Stage3–Stage1 差异同时包含 HLA 条件化、
三元架构、PLM 微调和目标任务训练效应；Stage 7 会在结论中保留这一限制。

## 3. 运行

当前 fold 4 / seed 2026：

```bash
cd /home/mjp/Project/PLM/UnifyImmun
./run_stage7.sh
```

也可以显式指定 checkpoint：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/evaluate_plm_stage7.py \
  --config configs/plm_stage7.yaml \
  --stage3-checkpoint \
    trained_model/PLM_Triplet/fold_4/seed_2026/triplet_best_20260802_215908_285880.pt \
  --stage1-phla-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_2026/best_phla.pt \
  --stage1-ptcr-checkpoint \
    trained_model/PLM_Unified/fold_4/seed_2026/best_ptcr.pt \
  --device cuda
```

调试时可限制派生数据；原始三元组分类集也会同步限制：

```bash
./run_stage7.sh --limit-pairs 64 --limit-tetrads 16
```

## 4. 产物与多 seed

新版 manifest 使用 `format_version=2` 和
`comparison=stage3_vs_stage1`。旧版 Stage5-vs-Stage1 结果不会被聚合器误读。

```text
trained_model/PLM_Stage7/fold_<fold>/seed_<seed>/<timestamp>/
  stage7_manifest.json
  stage7_data_audit.json
  stage7_counterfactual_val.csv
  stage7_counterfactual_test.csv
  stage7_summary.json
  stage7_report.md
```

多个相同 fold、不同 seed 且派生行严格对齐的运行可聚合：

```bash
/home/mjp/miniconda3/envs/unifyimmun/bin/python \
  source/analyze_plm_stage7.py \
  --run-dir <seed_1_stage7_run> \
  --run-dir <seed_2_stage7_run> \
  --run-dir <seed_3_stage7_run>
```

少于 3 个运行时，结论始终标记为 `preliminary=true`。

## 5. 测试

```bash
cd /home/mjp/Project/PLM/UnifyImmun
PYTHONPATH=source:. /home/mjp/miniconda3/envs/unifyimmun/bin/python -m unittest \
  tests.test_plm_stage7 \
  tests.test_plm_triplet
```

## fold4/seed2026 实际结果报告

> 来源：`trained_model/PLM_Stage7/fold_4/seed_2026/20260807_181402_082643/stage7_report.md`

- 运行数：1
- Fold：4
- Seed：[2026]
- 合成 decoy 为未观测组合，不等同于实验真阴性。

## test

- 原始三元组：22796；peptides：11502
- Stage1 pTCR：AUROC=0.8385，AUPR=0.8654，MCC=0.5282
- Stage1 pHLA+pTCR late fusion：AUROC=0.9790，AUPR=0.9791，MCC=0.8738
- Stage3 pTCR path ablation：AUROC=0.8442，AUPR=0.8709，MCC=0.5490
- Stage3 PHT：AUROC=0.9990，AUPR=0.9991，MCC=0.9744
- Stage3−Stage1 peptide-macro accuracy gain：0.1336 (95% CI 0.1271–0.1405)
- pHLA-valid anchors：10753；decoys：23260
- Stage3 PHT peptide-macro paired win rate：0.8625 (95% CI 0.8257–0.8976)
- 独立 tetrads：155；peptides：25
- Stage3 PHT peptide-macro D：0.0766 (95% CI -0.0708–0.2178, p=0.154028)

## val

- 原始三元组：22791；peptides：11495
- Stage1 pTCR：AUROC=0.8406，AUPR=0.8688，MCC=0.5341
- Stage1 pHLA+pTCR late fusion：AUROC=0.9795，AUPR=0.9794，MCC=0.8726
- Stage3 pTCR path ablation：AUROC=0.8466，AUPR=0.8743，MCC=0.5562
- Stage3 PHT：AUROC=0.9989，AUPR=0.9988，MCC=0.9778
- Stage3−Stage1 peptide-macro accuracy gain：0.1376 (95% CI 0.1309–0.1443)
- pHLA-valid anchors：10724；decoys：23404
- Stage3 PHT peptide-macro paired win rate：0.8641 (95% CI 0.8259–0.9010)
- 独立 tetrads：134；peptides：25
- Stage3 PHT peptide-macro D：0.1765 (95% CI -0.0188–0.4203, p=0.0671893)

## 结论分级

- 判定：`supports_stage3_hla_conditioned_ptcr_improvement`
- Stage3 PHT 优于 Stage1 pTCR：True
- Stage3 PHT 优于同 checkpoint 的 pTCR 消融：True
- HLA 反事实敏感：True
- 初步结果：True
- 限制：Stage3 received target PHT supervision that Stage1 did not; the comparison therefore combines HLA conditioning, architecture, fine-tuning, and target-task training effects. Synthetic decoys are not experimentally validated non-binding triples.

## 模型与评分器说明

Stage7 报告中的 7 个 score 并不是 7 个独立训练的模型。实际加载的是 3 个模型对象：

1. Stage1 pHLA checkpoint；
2. Stage1 pTCR checkpoint；
3. Stage3 多任务三元组 checkpoint。

其余评分器是同一模型不同分支或分数级组合。

| 评分器 | 输入 | HLA 是否进入 TCR 决策 | 说明 |
|---|---|---|---|
| `stage1_phla_margin` | peptide + HLA | 是 | Stage1 peptide–HLA 二分类分支 |
| `stage1_ptcr_margin` | peptide + TCR | 否 | Stage1 HLA-free peptide–TCR 分支 |
| `stage1_late_margin` | Stage1 pHLA + Stage1 pTCR 分数 | 仅加法使用 | 两个二元模型的验证集标准化分数相加，不是新训练模型 |
| `stage3_phla_margin` | peptide + HLA | 是 | Stage3 微调 encoder 后的 pHLA 分支 |
| `stage3_ptcr_margin` | peptide + TCR | 否 | 同一个 Stage3 checkpoint 的 HLA-free pTCR 分支 |
| `stage3_late_margin` | Stage3 pHLA + Stage3 pTCR 分数 | 仅加法使用 | Stage3 两个二元分支的分数级组合 |
| `stage3_pht_margin` | peptide + HLA + TCR | 是，且条件化 | 先用 HLA 条件化 peptide，再与 TCR 匹配的三元组分支 |

各二分类分数均采用正类 logit margin：

```text
score = logit(label=1) - logit(label=0)
```

### Stage1 模型

Stage1 pHLA 模型输入 peptide 和 HLA pseudo-sequence。两条序列分别经过 PLM 和 SequenceAdapter，随后以 peptide 为 query 对 HLA 残基表示进行 cross-attention，再经过 masked pooling 和二分类头输出 peptide–HLA 结合分数。

Stage1 pTCR 模型的结构类似，但输入替换为 peptide 和 TCR。该分支完全不读取 HLA，因此无法在内部表示中表达“同一个 peptide 在不同 HLA 背景下对应不同 TCR 识别能力”。Stage1 阶段底层 PLM 冻结，主要训练 adapter、cross-attention、pooling 和分类头。实现见 `source/plm_unified/model.py` 和 `source/plm_unified/attention.py`。

Stage1 的 late fusion 为验证集标准化后的分数相加：

```text
Z(stage1_pHLA) + Z(stage1_pTCR)
```

因此它虽然间接使用了 HLA 分数，但 HLA 没有进入 pTCR 分支的内部计算。

### Stage3 模型

Stage3 从同一 fold/seed 的 Stage1 完整 checkpoint 初始化，保留 pHLA 和 pTCR 二元分支，并加入 PHT 三元组分支。PHT 分支的计算顺序为：

```text
peptide
  -> peptide--HLA cross-attention
  -> HLA-conditioned peptide representation
  -> TCR cross-attention
  -> triplet classifier
```

因此 TCR 匹配的不是与 HLA 无关的原始 peptide 表示，而是当前 HLA 背景下得到的 peptide 表示。Stage3 warm-up 阶段先训练新增的三元组模块，随后使用三元组任务和 pHLA/pTCR 二元任务 replay 联合训练；当前配置的二元 replay ratio 为 0.25，并对 peptide、HLA、TCR 三个 PLM 的顶部 2 层进行渐进式微调。实现见 `source/plm_unified/triplet_model.py`。

这里的 `stage3_ptcr_margin` 是同一 Stage3 checkpoint 的 pTCR 路径，输入仍只有 peptide 和 TCR，但使用的是 Stage3 微调后的 encoder。它是 HLA-free 的结构对照，不是重新训练的独立 Stage3-no-HLA 模型。

## 基础 PLM 和序列表示

| 序列 | PLM | 原始隐藏维度 | Stage7 中的处理 |
|---|---|---:|---|
| peptide | `/home/mjp/model/ESMC-300M` | 960 | adapter 映射到 128 维 |
| HLA pseudo-sequence | `/home/mjp/model/ESMC-300M` | 960 | adapter 映射到 128 维 |
| TCR | `/home/mjp/model/TCR-Bert` | 768 | adapter 映射到 128 维 |

SequenceAdapter 的主要维度为 `input_dim -> 256 -> 128`，中间使用 GELU、dropout 和 LayerNorm；cross-attention 使用 4 个头，feed-forward 维度为 512。当前实验使用 HLA pseudo-sequence，而不是完整 allele 名称；Stage7 配置中的元数据模式为 `pseudo_only`，没有额外的 allele 对齐表。

## 训练数据来源

### Stage1 pHLA 数据

路径为 `data/data_HLA/`，记录描述 `peptide, hla, label`。

当前 fold 4 的 checkpoint 数据审计为：

- train：803,146 条，清洗后 803,048 条；
- validation：200,786 条，清洗后 200,761 条。

这些数据用于训练 peptide–HLA 二分类分支，也在 Stage7 中被有限地用作候选 HLA decoy 的来源，见下文。Stage7 没有使用 pHLA validation/test 标签构造 decoy。

### Stage1 pTCR 数据

路径为 `data/data_TCR/`，记录描述 `peptide, tcr, label`。

- train：206,690 条；
- validation：51,672 条。

这些数据用于训练 HLA-free 的 peptide–TCR 二分类分支。

### Stage3 三元组数据

路径为：

```text
data/data_triple/merged_pseudo.csv
data/data_triple/train_set.csv
data/data_triple/val_set.csv
data/data_triple/test_set.csv
```

每条记录包含 `peptide, tcr, hla, label`。原始数据共 228,056 条，并且 `merged_pseudo.csv` 与三个 split 的多重集合完全一致。

当前 fold 4 严格清洗后的数量为：

- train：182,131 条（原始 182,444）；
- validation：22,791 条（原始 22,806）；
- test：22,796 条（原始 22,806）。

清洗包括 peptide 长度 8–15、HLA/TCR 长度不超过 34、删除非标准氨基酸、重复记录和冲突记录。清洗后的 train/validation/test 标签基本各占一半，Stage3 使用这些三元组数据进行 PHT 训练和验证。

## Stage7 评估数据和反事实构造

Stage7 本身不更新模型参数，主要使用清洗后的三元组 validation/test 数据进行以下三类评估。

### 1. 原始三元组分类

直接使用真实记录：

```text
peptide + HLA + TCR + observed label
```

该部分可以计算 AUROC、AUPR、accuracy、MCC、F1 和 peptide-macro accuracy，是本报告中唯一具有真实三元组标签的主分类实验。

### 2. `phla_valid_pair` HLA 反事实

固定已知正样本 `(peptide, TCR, original HLA)`，再为同一个 peptide–TCR 寻找替代 HLA。候选 HLA 必须满足：

- 在 fold 4 pHLA train 中与同一 peptide 构成已知正 pHLA；
- 在 triplet train 中出现过；
- 与原始 HLA 不同；
- 不能在任一 triplet split 中构成已知正 triplet；
- 最多选择 5 个，并按训练频次接近程度匹配。

Stage7 读取 `data/data_HLA/train_fold_4.csv` 的 803,146 条记录，得到 800,338 个 unique pHLA pair；2,808 个标签冲突的 pair 被排除，最终保留 255,233 个无冲突正 pHLA pair。

test 中共有 11,394 个正 anchor，其中 10,753 个找到至少一个 decoy，共 23,260 个 decoy。decoy 是未观测组合，不应直接解释为实验真阴性；这里主要测试模型是否会随 HLA 改变而改变 peptide–TCR 排名。

### 3. `tetrad` 交叉组合

对同一个 peptide 的两个正三元组：

```text
(peptide, TCR1, HLA1)
(peptide, TCR2, HLA2)
```

构造两个 off-diagonal 组合 `(TCR1, HLA2)` 和 `(TCR2, HLA1)`，用于测量 HLA–TCR 是否存在超出简单加法的非加性相互作用。 `tetrad_disjoint` 会限制 pair 复用，避免单个实体过度贡献结果；test 中有 155 个独立 tetrads，涉及 25 个 peptide。

`legacy_hla_shift` 则是确定性替换 HLA 的压力测试，用于观察模型对 HLA 改变的敏感度，但替换后的组合同样不具有实验验证标签。

## 四类评估的结果分析

本节对当前运行中的四类结果分别解释。前三类是 Stage7 的核心评估，分别对应原始三元组分类、phla_valid_pair HLA 反事实排序和 tetrad 交互评估；第四类 legacy_hla_shift 是额外的 HLA 敏感性压力测试。所有数值均来自当前 fold 4 / seed 2026 运行，结论仍因只有一个 seed 而标记为 preliminary。

### 1. 原始三元组分类：PHT 的整体识别性能明显提高

test 集上的主要结果如下：

| 模型/评分器 | AUROC | AUPR | Accuracy | MCC |
|---|---:|---:|---:|---:|
| Stage1 pTCR | 0.8385 | 0.8654 | 0.7572 | 0.5282 |
| Stage1 late fusion | 0.9790 | 0.9791 | 0.9359 | 0.8738 |
| Stage3 pTCR path | 0.8442 | 0.8709 | 0.7717 | 0.5490 |
| Stage3 late fusion | 0.9786 | 0.9790 | 0.9331 | 0.8688 |
| Stage3 PHT | 0.9990 | 0.9991 | 0.9872 | 0.9744 |

Stage3 PHT 相比 Stage1 pTCR 的 test 增益为：

- AUROC：+0.1606；
- AUPR：+0.1337；
- Accuracy：+0.2300；
- MCC：+0.4462；
- peptide-macro accuracy gain：+0.1336，95% CI 为 [0.1271, 0.1405]。

相对于同一个 Stage3 checkpoint 的 HLA-free pTCR path，Stage3 PHT 仍然提高：

- AUROC：+0.1548；
- AUPR：+0.1282；
- Accuracy：+0.2155；
- MCC：+0.4254；
- peptide-macro accuracy gain：+0.1592，95% CI 为 [0.1521, 0.1662]。

相比已经包含 HLA 分数的 Stage1 late fusion，Stage3 PHT 仍有小但稳定的增益：AUROC +0.0200、AUPR +0.0200、MCC +0.1006，peptide-macro accuracy gain 为 +0.0855，95% CI 为 [0.0802, 0.0908]。validation 结果具有相同趋势：Stage3 PHT AUROC=0.9989，而 Stage1 pTCR AUROC=0.8406、Stage3 pTCR path AUROC=0.8466；PHT 相对 Stage1 pTCR 的 peptide-macro accuracy gain 为 0.1376，95% CI [0.1309, 0.1443]。

这说明三元组分支不仅比完全不使用 HLA 的 pTCR 模型好，也比简单把 pHLA 和 pTCR 分数相加的 late fusion 更好。Stage1 pTCR 到 Stage3 pTCR path 只有小幅提升，说明大部分增益并不能简单解释为“Stage3 encoder 微调后自然变强”，而与三元组分支及其目标任务监督有关。

但是，原始三元组分类结果仍不能单独证明 HLA 条件化是唯一原因。Stage3 同时改变了模型结构、接受了 PHT 监督，并微调了 PLM 顶部层。因此该实验支持的是“完整 Stage3 PHT 方案提高 peptide–TCR 识别”，而不是已经完成 HLA 单因素因果归因。

### 2. phla_valid_pair：PHT 能根据 HLA 改变 peptide–TCR 排名

test 中有 10,753 个可用 positive anchor 和 23,260 个 decoy，覆盖 306 个 peptide。对于每个 anchor，原始 HLA 是已知正 triplet 中的 HLA，decoy HLA 只满足 pHLA 为正和其他过滤条件，并没有实验验证的 triplet 标签。

Stage3 PHT 的 test 结果为：

- anchor-level paired win rate：0.9713；
- anchor-level hits@1：0.9495；
- peptide-macro paired win rate：0.8625，95% CI [0.8257, 0.8976]；
- peptide-macro hits@1：0.8378，95% CI [0.7979, 0.8769]；
- peptide-macro MRR：0.9158，95% CI [0.8950, 0.9359]。

validation 结果基本一致：peptide-macro paired win rate 为 0.8641，95% CI [0.8259, 0.9010]；peptide-macro hits@1 为 0.8437，95% CI [0.8033, 0.8831]。这说明该现象不是只在 test 中偶然出现。

对照模型的行为具有解释性：

- Stage1 pTCR 和 Stage3 pTCR path 的 paired win rate 都是 0.5，分数差为 0；这是因为它们不输入 HLA，原始 HLA 和 decoy HLA 的分数完全相同；
- Stage1/Stage3 pHLA 及 late fusion 的 anchor-level win rate 约为 0.7016，但 peptide-macro win rate 约为 0.4775，说明单独的 pHLA 分数或加法融合不能稳定恢复原始 triplet HLA；
- Stage3 PHT 的 peptide-macro win rate 达到 0.8625，明显高于上述加法基线。

因此，phla_valid_pair 提供了比较直接的 HLA 使用证据：在固定 peptide 和 TCR 的情况下，只有 PHT 分支能够系统地偏好原始 cognate HLA。需要注意的是，decoy 是“未观测组合”而不是真实负样本；该结果说明模型具备 HLA 条件排序能力，不等价于证明所有 decoy 都不结合。

### 3. tetrad：存在正向趋势，但尚未证明稳定的非加性 HLA–TCR 交互

tetrad_disjoint test 包含 155 个 tetrad、25 个 peptide；tetrad_all 包含 35,288 个有效 tetrad、同样涉及 25 个 peptide。对每个 tetrad 计算：

    D = (S(p,t1,h1) + S(p,t2,h2) - S(p,t1,h2) - S(p,t2,h1)) / 2

test 结果如下：

| 实验 | Stage3 PHT peptide-macro D | 95% CI | 单侧 sign-flip p |
|---|---:|---:|---:|
| tetrad_disjoint | 0.0766 | [-0.0708, 0.2178] | 0.1540 |
| tetrad_all | 0.1045 | [-0.0572, 0.2752] | 0.1184 |

相对于 Stage3 late fusion，PHT 的 D 差值与上述结果相同；其 CI 也跨越 0。validation 的趋势略强，但仍未达到预设显著性标准：

- tetrad_disjoint：D=0.1765，95% CI [-0.0188, 0.4203]，p=0.0672；
- tetrad_all：D=0.1178，95% CI [-0.0329, 0.2845]，p=0.0829。

所有 pTCR、pHLA 和 late fusion 对照的 D 均为数值精度范围内的 0，验证了加法模型的理论抵消性质，也说明 tetrad 计算本身没有引入系统性伪交互。Stage3 PHT 的点估计为正，说明模型在部分 peptide 上可能学习到了 HLA–TCR 条件兼容性；但 25 个 peptide 的有效样本规模较小，置信区间较宽，不能拒绝 D=0。

因此，tetrad 结果支持“可能存在非加性相互作用的趋势”，但目前不能宣称已经证明稳定的非加性 TCR×HLA 相容性。这也是结论中 non_additive_tcr_hla_compatibility=false 的原因。

### 4. legacy_hla_shift：大幅改变 HLA 后 PHT 性能显著下降

该压力测试在每个 validation/test 三元组中，用同一 split 内确定性半偏移位置的 HLA 替换原始 HLA。test 共 22,796 条记录，其中 20,502 条（89.94%）的 HLA 被改变；负样本中改变比例为 97.92%，正样本中为 81.95%。

Stage3 PHT 的 test 指标为：

| 指标 | 原始 HLA | shifted HLA | 变化（shifted−original） |
|---|---:|---:|---:|
| AUROC | 0.9990 | 0.7132 | -0.2859 |
| AUPR | 0.9991 | 0.7331 | -0.2660 |
| Accuracy | 0.9869 | 0.6016 | -0.3853 |
| MCC | 0.9739 | 0.2720 | -0.7019 |
| F1 | 0.9869 | 0.4029 | -0.5840 |
| Recall | 0.9874 | 0.2689 | -0.7185 |

validation 结果也一致：AUROC 从 0.9989 降至 0.7066，Accuracy 从 0.9885 降至 0.5950，MCC 从 0.9770 降至 0.2563。性能下降主要体现在正样本召回率，说明替换 HLA 后大量原本被判定为正的三元组分数下降；specificity 仍保持在 0.931 左右。

该结果说明 Stage3 PHT 的预测确实依赖 HLA，而不是只依赖 peptide–TCR 二元信息。可是，shifted 组合的原始 label 是被复制过来的，并不代表替换后的 HLA 在生物学上仍应保持相同标签。因此该实验只能作为 HLA 敏感性压力测试，不能当作严格的有标签泛化测试，也不能单独证明分数下降一定对应真实结合丧失。

## 四类结果的综合结论

四类评估形成了由强到弱的证据链：

1. **原始三元组分类**：Stage3 PHT 在真实标签上显著优于 Stage1 pTCR、Stage3 pTCR path 和 late fusion；
2. **HLA-valid pair**：固定 peptide–TCR 后，PHT 能稳定偏好原始 HLA，证明模型会使用 HLA 改变 TCR 识别评分；
3. **Tetrad**：PHT 的 D 点估计为正，但置信区间跨 0，尚不足以证明稳定的非加性 TCR–HLA 相互作用；
4. **Legacy HLA shift**：改变 HLA 后性能大幅下降，进一步证明 PHT 不是 HLA-invariant 的 peptide–TCR 模型。

因此当前最稳妥的结论是：

> Stage3 PHT 能利用 HLA 条件化信息提高 peptide–TCR 三元组识别，并对 HLA 改变保持明显敏感性；但当前数据尚不足以证明所有性能提升均由 HLA 条件化单独造成，也不足以证明稳定的非加性 HLA–TCR 交互。

要进一步增强结论，需要至少增加多 seed 运行、peptide/HLA/TCR entity-disjoint split、未见 HLA 测试，以及经过实验验证的 HLA–peptide–TCR 负样本或配对实验。
