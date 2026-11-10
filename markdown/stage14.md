# Stage 14：Hyper-PMT 异构超图融合

Stage14 是 Stage8 的独立结构优化分支，使用当前 Stage1 best_joint 初始化。
Peptide、HLA、TCR 在 residue-token level 通过共享的两层 Hyper-PMT encoder
同步交互；PH/PT 二元 edge 与 PHT 三元 hyperedge 使用相同参数。

模型保留 Stage1 PH attention 作为 PHT 的门控先验，训练继续采用四类型 PHT
数据、PH/PT replay 以及 Stage8 的 hierarchy/PT implication 约束。完整运行方式
见仓库根目录的 `PLM_STAGE14.md`。
