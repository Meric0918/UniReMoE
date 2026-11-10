## 为什么要分ab:训练分两步：

  1. Stage 3A 固定 2 个 epoch，只训练由 pTCR 分支复制初始化的 triplet cross-attention、pooling 和
  classifier；
  2. Stage 3B 每轮按 pHLA replay、pTCR replay、完整 triplet epoch 的顺序训练，渐进解冻三个 PLM 顶部各 2
  层。这样设计目的是什么


## 分成 Stage 3A 和 Stage 3B，核心目的是：先让新增三元模块适应输入，再安全地调整整个模型，同时避免破坏Stage 2 已学到的二元能力。

  ### Stage 3A：稳定三元头

  Stage 3 的新增路径是：

  pHLA 条件化 peptide
          ↓
  triplet cross-attention
          ↓
  pooling
          ↓
  classifier

  虽然这些模块从 pTCR 分支复制初始化，但输入已经变了：

  原 pTCR 分支输入：原始 peptide 表示
  三元分支输入：经过 HLA 条件化的 peptide 表示

  因此复制参数只是一个较好的起点，不能直接认为已经适合三元任务。

  Stage 3A 冻结三个 PLM、adapter 和原有二元分支，只训练新增模块，主要作用是：

  - 让三元注意力适应 pHLA 条件表示；
  - 让分类器适应三元标签分布；
  - 建立合理的梯度和输出尺度；
  - 防止尚未训练成熟的三元分类头产生不稳定梯度；
  - 避免一开始就破坏 Stage 2 学好的 peptide、HLA、TCR 表征。

  可以把它理解为“先安装并校准新头”。

  如果跳过 Stage 3A，一开始就解冻整个网络，随机或尚未适配的三元头可能通过梯度反向修改三个大型 PLM，导致：

  三元 loss 很大
      ↓
  梯度不稳定
      ↓
  共享编码器快速漂移
      ↓
  pHLA/pTCR 能力下降，训练波动增大

  ### Stage 3B：学习真正的三元特征

  Stage 3A 只训练新头，能够利用已有特征，但不能改变底层表示。真正的三元相互作用可能要求模型重新调整：

  - peptide 的 HLA 锚定位点表示；
  - HLA 对 peptide 的条件化方式；
  - TCR 对 pHLA 复合物的识别特征；
  - 三个序列之间的信息对齐。

  因此 Stage 3B 解冻：

  - 三个 adapter；
  - pHLA cross-attention；
  - triplet cross-attention；
  - 三个 PLM 顶部各 2 层。

  只解冻顶部 2 层，是一种折中：

   方式            问题
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PLM 全部冻结    三元任务只能重组旧特征，适应能力有限
  ──────────────  ────────────────────────────────────────────────
   PLM 全部解冻    参数量大、显存高、容易过拟合和遗忘
  ──────────────  ────────────────────────────────────────────────
   只解冻顶部层    保留通用序列表示，同时允许高层特征适应三元任务

  所以 Stage 3B 相当于“安装完成后，再有限度地调整整个系统”。

  ### 为什么加入 pHLA 和 pTCR replay

  三个任务共享 peptide、HLA、TCR 编码器。若 Stage 3B 只训练三元数据，模型参数会不断朝三元任务移动，可能发
  生灾难性遗忘：

  Stage 2 学到的 pHLA/pTCR 能力
                ↓
  只训练 triplet
                ↓
  共享编码器发生偏移
                ↓
  二元任务性能下降

  因此每轮重新加入一部分 pHLA、pTCR 数据：

  pHLA replay → 巩固 peptide-HLA 相互作用
  pTCR replay → 巩固 peptide-TCR 相互作用
  triplet     → 学习主要的三元目标

  Replay 的目的包括：

  - 保留 Stage 2 的二元能力；
  - 稳定共享 PLM 和 adapter；
  - 防止三元数据的偏差完全覆盖原有知识；
  - 让三元模型仍能单独执行 pHLA 和 pTCR 预测；
  - 起到多任务正则化作用，降低过拟合。

  ### 为什么 triplet epoch 放在每轮最后

  当前每轮顺序为：

  pHLA replay → pTCR replay → 完整 triplet epoch

  二元 replay 使用较少样本，完整 triplet 放最后，意味着每轮结束时参数最终更偏向主要目标——三元预测。换句话
  说：

  - replay 是辅助约束；
  - triplet 是主任务；
  - 最后的 triplet epoch 防止二元任务在本轮结束时占据主导。

  不过这个顺序不是理论上唯一正确的。还可以比较：

  - 三种任务 batch 随机混合；
  - 一个 batch 内联合计算三个 loss；
  - triplet → replay → triplet；
  - PCGrad、GradNorm 等梯度冲突方法。

  当前顺序的优势是实现简单、显存需求低、每种任务的数据加载和指标清晰。

  ### 两阶段设计总结

  Stage 3A
  冻结旧模型，只训练新增头
  目的：稳定初始化、适应新输入、避免破坏旧知识
                  ↓
  Stage 3B
  渐进解冻高层参数，加入二元 replay
  目的：学习深层三元特征，同时防止灾难性遗忘

  其中“Stage 3A 固定 2 个 epoch”和“解冻顶部 2 层”是当前合理的工程默认值，不是理论上的固定最优值。论文中最
  好进行消融：

  - 无 Stage 3A；
  - Stage 3A 训练 1/2/3 epoch；
  - PLM 全冻结；
  - 解冻顶部 1/2/4 层；
  - 有/无二元 replay。

  如果“无 Stage 3A”与当前方案性能接近，可以简化训练；如果无 replay 时二元性能明显下降，就能直接证明两阶段
  训练和 replay 的必要性。

