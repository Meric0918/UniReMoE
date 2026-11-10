# Stage MoE PT 分层交互融合

`stage_MOE_PT` 实现《MoE模型架构优化.md》的版本二，继续执行 PH、PT、PHT 分类。新增 PT→PHT 残基级门控融合，保留 PH→PHT 条件路径。旧 `stage_MOE` 的代码、配置和产物保持独立。

## 模型与接口

编码器、Cross-Attention、MoE 专家、池化和分类头复用版本一基础组件：冻结 ESM-C/TCR-BERT，三组 SequenceEncoder 输出 256 维表示；2 个共享专家加每任务 2 个专属专家，各分支采用样本级稠密 Softmax 路由。

PHT 输入经过以下计算，所有序列按同一肽残基位置对齐：

```text
Z_PH = PH_MoE(Cross_PH(P, H))
Z_PT = PT_MoE(Cross_PT(P, T))
U = Cross_PHT(Z_PH, T)              # 保留 Z_PH 的 Query 残差
R_PT = Linear(LayerNorm(Z_PT))
β = sigmoid(MLP([LN(U), LN(Z_PT), masked_mean(H)]))
U_fused = U + β * R_PT
Z_PHT = PHT_MoE(U_fused, context=[mean(U_fused), mean(P), mean(H), mean(T)])
logit_PHT = Classifier(AttentionPool(Z_PHT))
```

HLA 摘要广播到每个肽位置。融合门控为 `768→128→1` GELU MLP，输出偏置初始化为 −2；β 为 `[B,L_P]` 标量门控，padding 位置为零。PHT 损失通过融合连接更新 PT 中间路径，不截断梯度。单独的 PH/PT 预测不执行融合。

模型位于 `source/plm_unified/stage_MOE_PT_model.py`，类名 `StageMOEPTBindingModel`。保留 `forward('ph'/'pt', ...)`、`forward_triplet(...)` 和 `forward_all(...)` 接口及原有 `return_attention`、`return_features` 选项。输出包含：

- `logits`：对应分类任务的 `[B]` logit。
- `routing`：实际执行分支的 `[B,4]` 专家权重。
- `fusion_beta`：PHT 融合的 `[B,L_P]` β，pair 预测或禁用融合时为 `None`。

开启融合时，`forward_triplet` 返回 PH/PT/PHT 路由，仅计算 PHT 分类头；`forward_all` 计算三个分类头。`pt_fusion_enabled: false` 恢复版本一计算路径，融合参数保留在模型中但不参与前向和更新。相同基础权重下，其评估输出与版本一一致。

## 配置与训练

配置文件为 `configs/plm_stage_MOE_PT.yaml`，新增设置位于 `stage_MOE_PT`：

| 配置 | 默认值 | 用途 |
|---|---:|---|
| pt_fusion_enabled | true | 开启 PT→PHT 融合 |
| fusion_hidden_dim | 128 | β 门控隐藏维度 |
| fusion_bias_init | -2.0 | β 门控输出层偏置初值 |

数据审计、缓存读取和带掩码 BCE 复用 Stage MOE。沿用 `dataset/PH`、`dataset/PT`、`dataset/PHT` 当前划分，仅使用各自原生任务标签；PHT 阴性不派生 PH/PT 阴性。

默认联合训练每个 optimizer step 各取 PH/PT/PHT 一个 microbatch，累积梯度后更新一次。顺序训练每轮完整遍历 PH→PT→PHT，每个 batch 更新一次。两种模式均保留尾 batch；联合模式循环较短任务，顺序模式不补齐。

训练默认每任务 batch size 32，最多 100 轮、patience 10；三任务损失权重均为 1，均衡正则默认 0。β 门控与专家路由器使用 `1e-4` 学习率，其他模块使用 `2e-4`。沿用 AdamW、5% warmup、线性学习率衰减及前五轮 2→1 的专家路由温度；温度不作用于 β。

每轮按三任务验证 PR-AUC 均值选模，阈值仅在验证集按 MCC 选择。测试和 benchmark 读取保存的阈值。PR-AUC 使用仓库的 precision-recall 曲线梯形积分。

## 运行

脚本优先使用 `/home/mjp/miniconda3/envs/unifyimmun/bin/python`，可通过 `PYTHON_BIN` 覆盖。使用 `STAGE_MOE_PT_CONFIG` 或 `--config` 指定配置。

```bash
# 审计现有数据与跨任务 cold 资格
./run_stage_MOE_PT.sh audit --include-benchmarks

# 预计算冻结 PLM 的残基缓存，覆盖后续训练和评估
./run_stage_MOE_PT.sh cache --splits train val test --include-benchmarks

# 默认联合训练；也可改为 train-sequential
./run_stage_MOE_PT.sh train \
  --output-dir trained_model/PLM_Stage_MOE_PT/joint_seed42

./run_stage_MOE_PT.sh train-sequential \
  --output-dir trained_model/PLM_Stage_MOE_PT/sequential_seed42

# 恢复联合训练；顺序模式使用 resume-sequential
./run_stage_MOE_PT.sh resume \
  --resume trained_model/PLM_Stage_MOE_PT/joint_seed42/last.pt

./run_stage_MOE_PT.sh evaluate \
  --checkpoint trained_model/PLM_Stage_MOE_PT/joint_seed42/best.pt

./run_stage_MOE_PT.sh benchmark \
  --checkpoint trained_model/PLM_Stage_MOE_PT/joint_seed42/best.pt
```

训练需先构建所需缓存。默认缓存目录为 `embedding_cache/plm_stage_MOE_PT`，输出为 `trained_model/PLM_Stage_MOE_PT`。已有缓存需显式 `--overwrite` 才会重建；训练输出目录必须为空。YAML 默认启用 SwanLab online，可在训练命令添加 `--swanlab-mode disabled` 关闭。

checkpoint 架构标识为 `stage_MOE_PT_v2`，包含融合配置、温度、三任务阈值和编码器身份。`last.pt` 另含优化器、调度器、RNG、历史与早停状态，仅支持完整 epoch 边界恢复。融合设置、训练/数据/缓存契约必须匹配；联合与顺序模式不能交叉恢复，版本一 checkpoint 不能直接加载为版本二。

## 诊断与对照

`history.json` 和 SwanLab 记录路由权重、熵以及 β 统计。`fusion` 包含有效残基数量、均值、总体标准差、β<0.05 与 β>0.95 的比例，按有效残基数聚合，排除 padding。评估同样输出这些统计，逐样本预测 CSV 增加 `fusion_beta_mean`；不执行融合的预测该列为空。

无融合消融：复制配置，将 `pt_fusion_enabled` 设为 false，在新的输出目录从头训练；两组使用相同划分、种子、超参数和验证选模规则。比较 PHT、PH、PT 的 AUROC/PR-AUC，以及 cold-peptide/cold-TCR benchmark 的结果。禁用融合模型仍注册融合参数，因此总注册参数量包含未使用的融合模块。

HLA 敏感性检查：在 `model.eval()` 下固定同批 P/T，调用 `forward_all`；替换 HLA hidden 和对应 mask 后再次调用。记录 PHT 概率与 β 的变化，并确认 PT 输出保持一致。该变化仅说明模型是否依赖 HLA；判断这种依赖是否正确需要真实 HLA 限制性标签，不能把任意替换当作阴性。

benchmark 的 `cold_audit` 检查测试实体是否出现在任一任务训练池。若 `strict_cold_eligible` 为 false，不能将整个数据集结果表述为严格跨任务 cold 泛化。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_MOE_PT -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mjp/miniconda3/envs/unifyimmun/bin/python \
  -m unittest discover -s tests/stage_MOE -v
```

测试使用临时合成 HDF5 缓存，覆盖公式、padding、输入边界、梯度路径、禁用融合等价性、优化器分组、配置与 checkpoint 契约、两种训练模式、精确恢复、CLI 评估及 benchmark。CUDA 可用时运行默认尺寸 bf16 前向、反向与更新检查；无需加载 PLM 或创建正式训练缓存。性能收益仍需正式训练和消融结果验证。

2026-11-03 验证：版本二 33 项测试通过，版本一 24 项回归测试通过；默认尺寸 bf16 检查在 RTX 4090 上通过。未启动正式训练。
