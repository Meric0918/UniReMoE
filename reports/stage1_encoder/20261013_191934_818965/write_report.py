"""Write Chinese branch reports using the recorded evaluation and diagnostics."""

import json
from pathlib import Path

from summarize import OUT, ROOT, STAMP, table


NAMES = {
    "unseen_hla": "未见 HLA", "unseen_pair": "双实体未见（unseen_pair）",
    "unseen_peptide": "未见 peptide", "unseen_tcr": "未见 TCR",
}


def name(split):
    return NAMES[split.split("_", 1)[1]]


def metric_table(metrics):
    return table(
        ["划分", "N", "AUROC", "AUPR", "Accuracy", "MCC", "F1", "Recall", "ECE", "综合分"],
        [[name(split), f"{values['samples']:,}",
          *[f"{values[key]:.4f}" for key in ("auroc", "aupr", "accuracy", "mcc", "f1", "recall", "ece", "performance_avg")]]
         for split, values in metrics.items()],
    )


def branch_report(task, info, all_results):
    label = "pHLA" if task == "phla" else "pTCR"
    test = {split: entry["metrics"] for split, entry in info["test"].items()}
    best = info["best_round"]
    score = info["score"]
    parts = [f"# {label} 分支最佳模型分析", "",
             f"实验：`{STAMP}`；seed 42，fold 1，三个独立的 3 层、8 头、256 维 sequence encoder。",
             f"分支判别最佳模型为第 **{best} 轮**，验证集三种划分等权综合分 **{score:.6f}**，独立分支参数量 **{info['parameters']:,}**。",
             f"模型：[best_{task}_{STAMP}.pt]({info['checkpoint']})。", "",
             "## 验证与独立测试", "",
             "综合分为 `(AUROC + AUPR + Accuracy + MCC + F1) / 5`；每个任务的三个划分等权平均。分类阈值固定为 `probability > 0.5`。以下 test 指标来自本次对冻结任务最佳模型的完整评估，未调整模型参数或阈值。", "",
             "验证集：", "", metric_table(info["validation"]), "", "独立测试集：", "", metric_table(test), "",
             f"验证集等权综合分：**{score:.4f}**；测试集等权综合分：**{info['test_macro']['performance_avg']:.4f}**。", "",
             f"![{label} 验证/测试、ROC、PR 和概率可靠性](./{task}_diagnostics.png)", "",
             "## 分支表现与错误", ""]
    if task == "phla":
        ci = info["test"]["test_unseen_pair"]["auroc_stratified_row_bootstrap_95"]
        acc_ci = info["test"]["test_unseen_pair"]["accuracy_wilson_95"]
        parts.extend([
            "**未见 peptide 是最稳定的场景。** 验证/测试 AUROC 分别为 0.9641/0.9639，综合分为 0.9088/0.9090。测试召回率 92.80%，ECE 0.0214，结果支持对当前分布下新 peptide 的迁移。", "",
            "**未见 HLA 存在泛化和校准差距。** 测试 AUROC 0.8582、召回率 67.32%、特异度 86.04%；漏检 7,801 个阳性，其中 3,995 个被赋予低于 0.2 的阳性概率。验证集该场景的召回率仅 50.48%，ECE 0.2035；测试结果更好，但新的 HLA 集合与验证集不同，仍需按 HLA 分组看误差。", "",
            "**双实体未见是当前主要风险。** 测试 AUROC 0.5532、MCC −0.0333、Accuracy 48.44%，32 个阳性只检出 10 个，漏检 22 个；其中 19 个的阳性概率低于 0.2。NLL 1.3820、ECE 0.3898，说明该场景同时存在排序、分类和置信度问题。", "",
            "该场景验证集只有 60 条、测试集只有 64 条样本。验证 AUROC 0.8511 到测试 AUROC 0.5532 的差距不能据此推广到所有新 HLA/peptide 组合。",
            f"测试 Accuracy 的近似 Wilson 95% 区间为 [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]；按正负类别分层、2,000 次行重采样的 AUROC 95% 区间为 [{ci[0]:.4f}, {ci[1]:.4f}]，包含 0.5。这两个区间没有校正相同 HLA/peptide 的相关性，应结合按实体分组的重复验证。", "",
            "错误集中：`TLGFFDAQKGPPWEQPEDQDAGHIGINDWAAAK` 与 `TLGFFDAPRGPPWEQPEDQDEGHLGINDWAAAQ` 两个 HLA 伪序列合计贡献 **17/22（77.27%）** 个假阴性，其组内阳性召回率均为 0。分组表保留了完整序列和样本数；这里不据伪序列推断 HLA 等位基因名称。", "",
        ])
    else:
        parts.extend([
            "**未见 TCR 表现稳定。** 验证/测试 AUROC 分别为 0.9147/0.9146，综合分 0.8368/0.8359。测试召回率 84.92%，ECE 0.0233。该划分的 142,960 条测试样本中，142,954 条的 peptide 已出现在 pTCR 训练集，因此结果主要支持对已有 peptide 上新 TCR 的泛化。", "",
            "**未见 peptide 是主要短板。** 测试 AUROC 0.7034、AUPR 0.7365，说明仍有一定排序能力；固定阈值下 precision 81.41%、recall 33.17%，72,400 个阳性中漏检 **48,382** 个。漏检中的 **37,356（77.21%）** 个阳性概率低于 0.2；ECE 0.2445、NLL 0.9156，预测呈现明显的高置信度假阴性。", "",
            "漏检集中在少量 peptide：`YVLDHLIVV`、`LLFLVLIML`、`HTTDPSFLGRY` 分别漏检 11,759、11,608、7,028 个阳性，合计 **30,395/48,382（62.82%）**。三者的组内漏检率约为 96.71%、99.99%、99.11%。这些组在测试集中的支持度较高，值得优先检查其训练覆盖、相近序列、预测概率和标签来源。", "",
            "**双实体未见仍以漏检为主。** 测试 AUROC 0.7021、综合分 0.6322；precision 89.38%、recall 42.28%、specificity 94.98%。5,137 个阳性漏检 2,965 个，其中 2,573 个阳性概率低于 0.2。ECE 0.2281，不能只用较高 precision 或 specificity 判断整体表现。", "",
            "验证到测试的等权综合分从 0.6457 提升到 0.6799，主要来自双实体未见测试集比对应验证集更容易；未见 peptide 和未见 TCR 的指标没有同步改善。应逐划分判断能力。", "",
        ])
    parts.extend(["测试集混淆矩阵及高置信度错误：", "", table(
        ["划分", "TP", "FN", "TN", "FP", "高置信 FN（p<0.2）", "高置信 FP（p>0.8）"],
        [[name(split), *[entry["confusion"][key] for key in ("TP", "FN", "TN", "FP")],
          entry["high_confidence_FN"], entry["high_confidence_FP"]] for split, entry in info["test"].items()]), "",
        "验证集的高置信度错误来自训练器保存的完整错误 CSV，条件为 `abs(label - probability) > 0.8`：", "", table(
            ["划分", "高置信 FN", "高置信 FP"],
            [[name(split), entry["high_confidence_FN"], entry["high_confidence_FP"]]
             for split, entry in info["validation_diagnostics"].items()]), "",
        "## 最佳轮次与其他选模目标", ""])
    alternatives = [["判别最佳", best, f"{score:.4f}", f"{info['validation_macro']['nll']:.4f}", f"{info['validation_macro']['ece']:.4f}"]]
    for role, title in (("calibration", "校准最佳（最小 NLL）"), ("robust", "组宏平均准确率最佳")):
        alt = info["alternatives"][role]
        alternatives.append([title, alt["round"], f"{alt['macro']['performance_avg']:.4f}",
                             f"{alt['macro']['nll']:.4f}", f"{alt['macro']['ece']:.4f}"])
    parts.extend([table(["选模目标", "轮次", "验证综合分", "验证 NLL", "验证 ECE"], alternatives), ""])
    if task == "phla":
        parts.extend([
            "训练 AUROC 在第 10 轮为 0.9655；未见 peptide 验证 AUROC 0.9641，但未见 HLA 只有 0.7891。训练在第 11 轮结束时 pHLA 的任务早停标志仍为 false；整体停止来自联合轮次早停，pHLA 没有单独耗尽 5 轮 patience。", "",
            "第 6 轮校准模型的平均 NLL 0.5684，低于第 10 轮的 0.5938；平均 ECE 0.1410 却高于第 10 轮的 0.1374。这里的校准选模目标是 NLL，使用该模型不能直接推断 ECE 更好。鲁棒模型与第 10 轮判别模型一致。", "",
        ])
    else:
        parts.extend([
            "第 6 轮以后训练 AUROC 从 0.9168 继续升至第 11 轮的 0.9348，但双实体未见验证 AUROC 从 0.6932 降至 0.5463；训练 NLL 下降而该验证集 NLL 上升。这个轨迹与对训练分布的过拟合一致。仅凭日志还不能确定是模型容量、共享分支干扰还是数据分布造成。", "",
            "第 5 轮校准模型的验证 NLL/ECE 为 0.7236/0.1627，优于第 6 轮的 0.7732/0.1804；综合分降低约 0.87 个百分点。第 8 轮鲁棒模型优化按 peptide 分组的宏平均准确率，验证综合分为 0.6316，低于判别最佳的 0.6457。各目标模型的 test 指标尚未做对照评估。", "",
        ])
    parts.extend(["## 数据划分检查与后续重点", "", table(
        ["测试划分", "训练中见过的 peptide 行数", "训练中见过的 receptor 行数", "训练中见过的配对", "重复配对行"],
        [[name(split), audit["peptide_rows_seen_in_task_train"], audit["receptor_rows_seen_in_task_train"],
          audit["pairs_in_task_train"], audit["duplicate_pair_rows"]] for split, audit in info["audits"].items()]), "",
        "所有比较均使用本任务训练集；同时检查了两个任务训练集的 peptide 并集。在双实体未见和未见 peptide 两种测试划分中，peptide 也未出现在另一个任务训练集，因而没有被另一任务的共享 Peptide encoder 训练标签直接覆盖。六个测试划分均没有训练配对重叠、重复配对行或同配对标签冲突。", ""])
    if task == "phla":
        parts.append("优先补充双实体未见验证样本，并按 HLA/peptide 分组做重复验证；对上述两组集中漏检的 HLA 伪序列进行数据来源、长度及分布检查。对未见 HLA 的概率输出，可在专用验证集上研究校准和阈值设置。")
    else:
        parts.append("优先提高未见 peptide 的训练覆盖和表征泛化，检查三个大量漏检 peptide 的近邻与数据来源；在验证集上评估以召回为目标的阈值及概率校准，同时报告 precision 的变化。训练集继续收敛不足以支持继续延长训练，应由冷启动验证集控制选模。")
    parts.extend(["", "当前结论来自 seed 42 的一次训练。层数效应需要同设置对照及多种子重复实验。概率曲线使用 15 个等频概率 bin；ECE 数值沿用仓库实现（置信度 ECE，其内部 0.5 平局使用 `>=`），分类指标统一使用严格 `>`。", "",
                  "## 产物", "", f"- [诊断图](./{task}_diagnostics.png)",
                  "- [训练轨迹](./training_dynamics.png)", "- [固定 train/validation 样本 UMAP](./feature_umap.png)",
                  "- [完整指标、模型 SHA256 与数据划分审计](./analysis.json)"])
    for split in info["test"]:
        parts.append(f"- {name(split)}：[逐行预测](./{task}_{split}_predictions.csv.gz)、[分组错误](./{task}_{split}_groups.csv)")
    return "\n".join(parts) + "\n"


def main():
    results = json.loads((OUT / "analysis.json").read_text())
    for task, info in results["tasks"].items():
        label = "pHLA" if task == "phla" else "pTCR"
        (OUT / f"{label}.md").write_text(branch_report(task, info, results))
    summary = ["# Stage1 Encoder：两个分支最佳模型分析", "",
               f"实验 `{STAMP}`：3 层 encoder，seed 42，fold 1；训练退出码为 0，在第 11 轮、12,155 step 正常结束。",
               "本次对两个任务独立最佳 checkpoint 评估了六个独立 test splits，共 408,610 条样本。冻结权重、固定阈值 `p>0.5`、bf16 推理；缓存 PLM 指纹与训练 checkpoint 一致。", "",
               table(["分支", "最佳轮次", "验证等权综合分", "测试等权综合分", "分析"],
                     [["pHLA", 10, "0.7814", "0.6804", "[pHLA 详细报告](./pHLA.md)"],
                      ["pTCR", 6, "0.6457", "0.6799", "[pTCR 详细报告](./pTCR.md)"]]), "",
               "## 独立测试结果", "", table(
                   ["分支", "划分", "N", "AUROC", "AUPR", "F1", "Recall", "ECE", "综合分"],
                   [["pHLA" if task == "phla" else "pTCR", name(split), f"{entry['metrics']['samples']:,}",
                     *[f"{entry['metrics'][key]:.4f}" for key in ("auroc", "aupr", "f1", "recall", "ece", "performance_avg")]]
                    for task, info in results["tasks"].items() for split, entry in info["test"].items()]), "",
               "## 结论", "",
               "pHLA 的未见 peptide 结果稳定；未见 HLA 的召回与校准还有差距，双实体未见表现弱。双实体未见测试集仅 64 条，其中 22 个假阴性有 17 个集中在两个 HLA 伪序列；其验证集也仅 60 条。应扩大这一验证场景，避免小样本结果对等权选模产生过大影响。", "",
               "pTCR 在已有 peptide 上的新 TCR 泛化稳定；未见 peptide 和双实体未见的主要问题是高置信度假阴性。未见 peptide 测试集漏检 48,382 个阳性，其中三个 peptide 贡献 30,395 个漏检。第 6 轮之后，训练指标改善而冷启动验证集下降，当前任务最佳选在第 6 轮有依据。", "",
               "`unseen_pair` 测试样本中的两类实体均没有出现在对应任务训练集；该划分实际评估双实体冷启动。两任务未见 peptide 测试集中的 peptide 也未出现在另一个任务训练集中。", "",
               "两分支测试宏平均综合分相近，但能力分布不同。pHLA 的小样本双实体划分与六万多条未见 peptide 样本等权，因此宏平均不能替代逐划分结果。", "",
               "## 训练与模型使用", "",
               "联合最佳模型在第 6 轮，联合验证分 0.706774。第 9/10 轮联合即时分数更高，但 pTCR 校准约束未通过，因此联合最佳未更新；第 11 轮触发联合 patience=5，同时 pTCR 任务 patience=5。pHLA 的独立任务 patience 未耗尽。", "",
               "pHLA 第 10 轮和 pTCR 第 6 轮的独立 checkpoint 各自包含所属轮次的 Peptide encoder，分析与测试均使用对应文件。它们的共享 encoder 权重不同；若需要一套共享 encoder 同时运行两任务，应使用第 6 轮的联合 checkpoint，不能直接把两个任务最佳权重按同名参数覆盖拼接。", "",
               "当前评估没有重新训练模型，也没有用 test 选择阈值。结果只对应 seed 42；多种子重复和同设置对照仍是评估架构稳定性的必要后续工作。", "",
               "## 图表与复现", "",
               "- [pHLA 诊断图](./phla_diagnostics.png)", "- [pTCR 诊断图](./ptcr_diagnostics.png)",
               "- [训练轨迹](./training_dynamics.png)", "- [UMAP 特征展示](./feature_umap.png)",
               "- [完整结构化分析](./analysis.json)", "- [原始推理指标](./evaluation.json)", "",
               "UMAP 来自训练期间固定 train/validation 子样本，三阶段分别建立参考空间；图中距离不能跨任务或跨阶段比较，也不能作为独立测试性能证据。", "",
               "从项目根目录复现：", "", "```bash",
               f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/{STAMP}/analyze.py",
               f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/{STAMP}/summarize.py",
               f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python reports/stage1_encoder/{STAMP}/write_report.py",
               "```", ""]
    (OUT / "report.md").write_text("\n".join(summary))
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
