# UnifyImmun Stage 文档索引

本目录集中整理项目中与模型 Stage1–Stage8 直接相关的说明、实验报告和分析文档。
原始文件没有删除或移动；每个汇总文档中的“来源”路径指向原始 Markdown 或报告文件。

## 阶段文档

| 阶段 | 集中文档 | 主要内容 |
|---|---|---|
| Stage 1 | [stage1.md](stage1.md) | PLM 统一模型、Adapter 训练 |
| Stage 2 | [stage2.md](stage2.md) | Encoder 顶层渐进式微调 |
| Stage 3 | [stage3.md](stage3.md) | 三元 Cross-Attention、对比实验、pTCR 分析 |
| Stage 4 | [stage4.md](stage4.md) | 双向 Cross-Attention |
| Stage 5 | [stage5.md](stage5.md) | pHLA、pTCR、PHT 三任务联合训练 |
| Stage 6 | [stage6.md](stage6.md) | Stage 5 欠拟合原因分析 |
| Stage 7 | [stage7.md](stage7.md) | Stage 3 与 Stage 1 的 HLA 条件增益 |
| Stage 8 | [stage8.md](stage8.md) | 三联合一致性约束 |
| Stage 14 | [stage14.md](stage14.md) | Hyper-PMT 异构超图融合 |

## 跨阶段资料

- [cross_stage.md](cross_stage.md)：Stage 总览、Stage1/Stage2 可视化和多 fold/seed 分析。

## 说明

`paper/plan/` 下的 stage-gates、task-packets 和 review 文件属于论文写作流程管理，不属于模型阶段技术文档，因此没有复制进本目录；原文件仍保持原位置。
