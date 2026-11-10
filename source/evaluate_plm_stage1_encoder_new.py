#!/usr/bin/env python3
"""Evaluate independent Stage1 encoder task-best models on PH/new and PT/new."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from evaluate_plm_stage1_ph_new import (
    audit_dataset as audit_phla,
    group_ids_for as phla_groups,
    json_safe,
    sha256_file,
)
from evaluate_plm_stage1_pt_new import audit_dataset as audit_ptcr, group_ids_for as ptcr_groups
from plm_unified.cache import EmbeddingCache
from plm_unified.data import make_cached_dataloader, normalise_sequence
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from plm_unified.stage1_encoder_model import TaskSpecificStage1EncoderBindingModel, validate_encoder_checkpoint
from plm_unified.trainer import load_training_checkpoint, resolve_device

ROOT = Path(__file__).resolve().parents[1]
RUN_TIMESTAMP = "20261013_191934_818965"
METRICS = (
    "auroc", "aupr", "accuracy", "mcc", "f1", "precision", "recall", "specificity",
    "performance_avg", "nll", "brier", "ece",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"training_logs/stage1_encoder_{RUN_TIMESTAMP}.yaml")
    models = ROOT / "trained_model/PLM_Stage1_Encoder/fold_1/seed_42"
    parser.add_argument("--phla-checkpoint", type=Path, default=models / f"best_phla_{RUN_TIMESTAMP}.pt")
    parser.add_argument("--ptcr-checkpoint", type=Path, default=models / f"best_ptcr_{RUN_TIMESTAMP}.pt")
    parser.add_argument("--ph-data-dir", type=Path, default=ROOT / "dataset/PH/new")
    parser.add_argument("--pt-data-dir", type=Path, default=ROOT / "dataset/PT/new")
    parser.add_argument("--tasks", nargs="+", choices=("phla", "ptcr"), default=["phla", "ptcr"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, help="Fold-level cache containing peptide/hla/tcr directories.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def format_number(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.4f}"


def average(metrics):
    result = {}
    for name in METRICS:
        values = np.asarray([entry[name] for entry in metrics], dtype=float)
        finite = values[np.isfinite(values)]
        result[name] = float(finite.mean()) if len(finite) else None
    return result


def metrics_with_confusion(labels, scores, groups, loss):
    values = compute_binary_metrics(
        labels, scores, loss=loss, threshold=.5, strict_threshold=True, group_ids=groups,
    ).to_dict()
    predictions = scores > .5
    positive = labels == 1
    values.update({
        "tp": int(np.sum(predictions & positive)), "fn": int(np.sum(~predictions & positive)),
        "tn": int(np.sum(~predictions & ~positive)), "fp": int(np.sum(predictions & ~positive)),
        "high_confidence_fn": int(np.sum(positive & (scores < .2))),
        "high_confidence_fp": int(np.sum(~positive & (scores > .8))),
    })
    return values


@torch.inference_mode()
def predict(model, task, loader, device, precision):
    use_autocast = device.type == "cuda" and precision != "none"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    feature_dtype = dtype if use_autocast else torch.float32
    scores, labels = [], []
    loss_sum = samples = 0
    started = time.perf_counter()
    for index, batch in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            output = model(
                task,
                batch["peptide_hidden"].to(device, dtype=feature_dtype), batch["peptide_mask"].to(device),
                batch["receptor_hidden"].to(device, dtype=feature_dtype), batch["receptor_mask"].to(device),
            )
            loss = torch.nn.functional.cross_entropy(output.logits, batch["labels"].to(device))
        count = len(batch["labels"])
        loss_sum += float(loss.cpu()) * count
        samples += count
        labels.append(batch["labels"].numpy())
        scores.append(output.logits.softmax(dim=1)[:, 1].float().cpu().numpy())
        if index == 1 or index % 50 == 0 or index == len(loader):
            print(f"[EVAL {task}] batches={index}/{len(loader)} rows={samples:,} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return np.concatenate(labels), np.concatenate(scores), loss_sum / samples


def training_overlap(config, task, rows):
    import csv
    folder, column = ("PH", "HLA") if task == "phla" else ("PT", "tcr")
    path = config.paths.data_root / folder / "train.csv"
    peptides, receptors, pairs = set(), set(), set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            peptide, receptor = normalise_sequence(row["peptide"]), normalise_sequence(row[column])
            if not config.model.peptide_min_length <= len(peptide) <= config.model.peptide_max_length:
                continue
            peptides.add(peptide)
            receptors.add(receptor)
            pairs.add((peptide, receptor))
    return {
        "training_file": str(path.resolve()), "training_sha256": sha256_file(path),
        "peptide_rows_seen_in_train": sum(p in peptides for p, r, y in rows),
        "receptor_rows_seen_in_train": sum(r in receptors for p, r, y in rows),
        "pair_rows_seen_in_train": sum((p, r) in pairs for p, r, y in rows),
    }


def dataset_comparison(config, task, data_path, current_rows):
    """Record which labelled pairs differ from the earlier benchmark file."""
    import csv
    from collections import Counter
    folder, column = ("PH", "HLA") if task == "phla" else ("PT", "tcr")
    benchmark = config.paths.data_root / folder / "benchmarks" / data_path.name
    if not benchmark.is_file():
        return None
    previous = {}
    with benchmark.open(newline="") as handle:
        for row in csv.DictReader(handle):
            peptide, receptor = normalise_sequence(row["peptide"]), normalise_sequence(row[column])
            if config.model.peptide_min_length <= len(peptide) <= config.model.peptide_max_length:
                previous[peptide, receptor] = int(row["label"])
    current = {(p, r): y for p, r, y in current_rows}
    removed = previous.keys() - current.keys()
    groups = Counter(pair[0] if task == "ptcr" else pair[1] for pair in removed)
    return {
        "benchmark_file": str(benchmark), "benchmark_samples": len(previous),
        "new_samples": len(current), "removed_pairs": len(removed),
        "added_pairs": len(current.keys() - previous.keys()),
        "removed_positive": sum(previous[pair] for pair in removed),
        "removed_negative": sum(1 - previous[pair] for pair in removed),
        "common_pair_label_changes": sum(current[pair] != previous[pair] for pair in current.keys() & previous.keys()),
        "top_removed_groups": groups.most_common(5),
    }


def markdown(report, args):
    label = "pHLA" if report["task"] == "phla" else "pTCR"
    folder = "PH" if report["task"] == "phla" else "PT"
    checkpoint = report["checkpoint"]
    splits = report["splits"]
    total = report["data"]
    def link(path, text):
        relative = os.path.relpath(path, args.output_dir.resolve()).replace(os.sep, "/")
        return f"[{text}]({relative})"
    lines = [
        f"# Stage1 Encoder 最佳 {label} 模型：{folder}/new 测试结果", "",
        f"使用 **第 {checkpoint['round']} 轮、seed {checkpoint['seed']}、{report['encoder_layers']} 层 encoder** 的独立 {label} 最佳模型，",
        f"对 {link(report['data_dir'], f'dataset/{folder}/new')} 下 {len(splits)} 个 `test_*.csv` 文件完成全量推理，合计 **{total['samples']:,}** 条样本（正类 {total['positive']:,}，负类 {total['negative']:,}）。", "",
        f"模型：{link(checkpoint['path'], Path(checkpoint['path']).name)}；训练验证选模分数 **{checkpoint['best_score']:.6f}**。",
        "测试固定使用 `probability > 0.5`，没有依据测试标签调整权重、阈值或选模。", "",
        "## 测试指标", "",
        "| Split | N | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, entry in splits.items():
        m, d = entry["metrics"], entry["data"]
        numbers = [m[name] for name in METRICS[:9]]
        lines.append(f"| `{split}` | {m['samples']:,} | {d['positive_rate']:.2%} | " + " | ".join(format_number(v) for v in numbers) + " |")
    lines += ["", "综合分为 `(AUROC + AUPR + Accuracy + MCC + F1) / 5`。", "",
              "| 汇总方式 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | 综合分 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, title in (("macro_average", "三个 split 等权宏平均"), ("pooled_metrics", "全部样本合并后重新计算")):
        lines.append(f"| {title} | " + " | ".join(format_number(report[key][name]) for name in METRICS[:9]) + " |")
    lines += ["", "宏平均给每个 split 相同权重；pooled 指标将全部预测合并后计算，不是各 split 指标的加权平均。", "",
              "## 混淆矩阵与概率校准", "",
              "| Split | TP | FN | TN | FP | 高置信 FN（p<0.2） | 高置信 FP（p>0.8） | NLL | Brier | ECE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, entry in splits.items():
        m = entry["metrics"]
        lines.append(f"| `{split}` | " + " | ".join(f"{m[k]:,}" for k in ("tp", "fn", "tn", "fp", "high_confidence_fn", "high_confidence_fp"))
                     + " | " + " | ".join(format_number(m[k]) for k in ("nll", "brier", "ece")) + " |")
    lines += ["", "NLL、Brier、ECE 越小越好。ECE 使用项目实现的 15 个等频概率 bin 的置信度校准误差；其 0.5 平局内部使用 `>=`，分类指标使用严格 `>`。", "",
              "## 结果解读", ""]
    best = max(splits, key=lambda key: splits[key]["metrics"]["performance_avg"])
    weakest = min(splits, key=lambda key: splits[key]["metrics"]["performance_avg"])
    lines += [f"综合分最高的是 `{best}`（{splits[best]['metrics']['performance_avg']:.4f}），最低的是 `{weakest}`（{splits[weakest]['metrics']['performance_avg']:.4f}）。", ""]
    for split, entry in splits.items():
        m, d = entry["metrics"], entry["data"]
        lines.append(f"- `{split}`：AUROC {m['auroc']:.4f}，Precision {m['precision']:.2%}，Recall {m['recall']:.2%}，ECE {m['ece']:.4f}；{d['positive']:,} 个阳性中漏检 {m['fn']:,} 个，其中 {m['high_confidence_fn']:,} 个阳性概率低于 0.2。")
    lines += ["", "这些测试集的正类比例并不完全相同，AUPR 和 Accuracy 应结合各自类别比例解读。"]
    if "test_unseen_pair" in splits:
        n = splits["test_unseen_pair"]["metrics"]["samples"]
        if n < 100:
            lines += ["", f"`test_unseen_pair` 只有 **{n}** 条样本，单条样本即可使 Accuracy 变化 {100/n:.2f} 个百分点，且与大样本 split 等权参与宏平均；应结合样本量和逐 split 结果判断。"]
    lines += ["", "数据与训练集的实体、配对重叠如下。新实体划分名称仅作文件标识，实际重叠数用于解释测试难度。", "",
              "| Split | peptide 行数在训练中出现 | receptor 行数在训练中出现 | 配对行数在训练中出现 |",
              "|---|---:|---:|---:|"]
    for split, entry in splits.items():
        overlap = entry["training_overlap"]
        lines.append(f"| `{split}` | {overlap['peptide_rows_seen_in_train']:,} | {overlap['receptor_rows_seen_in_train']:,} | {overlap['pair_rows_seen_in_train']:,} |")
    lines += ["", "## 数据检查", "",
              "| Split | 原始 / 纳入 | 正 / 负 | 唯一 peptide | 唯一 receptor | 重复配对行 | 标签冲突行 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for split, entry in splits.items():
        d = entry["data"]
        receptor_count = d["unique_hlas"] if report["task"] == "phla" else d["unique_tcrs"]
        lines.append(f"| `{split}` | {d['raw_rows']:,} / {d['retained_rows']:,} | {d['positive']:,} / {d['negative']:,} | {d['unique_peptides']:,} | {receptor_count:,} | {d['duplicate_pair_rows']:,} | {d['conflicting_pair_rows']:,} |")
    lines += ["", f"跨文件重复配对行 {total['cross_split_pair_duplicates']:,}，跨文件标签冲突行 {total['cross_split_pair_conflicts']:,}。具体来源、长度范围、过滤统计和文件 SHA256 保存在 JSON。"]
    comparisons = {split: entry.get("dataset_comparison") for split, entry in splits.items() if entry.get("dataset_comparison") is not None}
    if comparisons:
        lines += ["", "### 与前次 benchmark 数据的差异", "",
                  "| Split | benchmark 配对数 | new 配对数 | 移除阳性 | 移除阴性 | 新增配对 | 共有配对标签变化 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for split, entry in comparisons.items():
            lines.append(f"| `{split}` | " + " | ".join(f"{entry[k]:,}" for k in ("benchmark_samples", "new_samples", "removed_positive", "removed_negative", "added_pairs", "common_pair_label_changes")) + " |")
        lines += ["", "本次使用同一批训练 checkpoint。`new` 与前次 benchmark 的测试样本集和正类比例不同，比较指标时应同时考虑这些数据变化。"]
        if report["task"] == "ptcr":
            for split, entry in comparisons.items():
                if entry["removed_pairs"]:
                    group_description = "、".join(f"`{group}`（{count:,} 条）" for group, count in entry["top_removed_groups"])
                    lines += ["", f"`{split}` 移除配对主要来自 peptide：{group_description}。"]
    lines += ["",
              "## 分组指标", "",
              "| Split | 分组对象 | 组数 | Group accuracy | Group recall | Recall coverage | Group NLL |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split, entry in splits.items():
        m = entry["metrics"]
        lines.append(f"| `{split}` | {entry['group_by']} | {m['group_count']:,} | {format_number(m['group_macro_accuracy'])} | {format_number(m['group_macro_recall'])} | {format_number(m['group_coverage'])} | {format_number(m['group_macro_nll'])} |")
    lines += ["", "分组 Recall 仅对同时包含正负类的组计算；没有符合条件的组时显示 `—`。", "",
              "## 模型与评估设置", "",
              "| 项目 | 值 |", "|---|---|",
              f"| 架构 | `{report['architecture']}`；{report['encoder_layers']} 层、{report['model_config']['num_attention_heads']} 头、{report['model_config']['model_dim']} 维 |",
              f"| Checkpoint round / seed / SHA256 | {checkpoint['round']} / {checkpoint['seed']} / `{checkpoint['sha256']}` |",
              f"| 配置快照 | {link(report['config'], Path(report['config']).name)} |",
              f"| 精度 / batch size / workers | {report['mixed_precision']} / {report['batch_size']} / {report['num_workers']} |",
              f"| 设备 | {report['device_name']} |",
              f"| 评估时间 | {report['evaluation_time']} |", "",
              "本次使用已有冻结 PLM 残基缓存，按序列精确查找并校验 hidden dimension、PLM checkpoint fingerprint 及有效残基 mask。Peptide/HLA 输入 960 维，TCR 输入 768 维；新加的 sequence encoder 参数来自对应任务最佳 checkpoint。", "",
              "## 结果文件与复现", "",
              f"- [完整指标和审计 JSON](./stage1_encoder_{folder}_new.json)",
              f"- {link(ROOT / 'source/evaluate_plm_stage1_encoder_new.py', '评估程序')}", "",
              "逐行预测保存为当前目录下对应 split 的 `predictions.csv.gz`，可用于进一步错误分析。", "",
              "```bash",
              "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
              "  source/evaluate_plm_stage1_encoder_new.py \\",
              f"  --config {report['config']} \\",
              f"  --tasks {report['task']} \\",
              f"  --{'phla' if report['task']=='phla' else 'ptcr'}-checkpoint {checkpoint['path']} \\",
              f"  --{'ph' if report['task']=='phla' else 'pt'}-data-dir {report['data_dir']} \\",
              f"  --cache-root {report['cache_root']} \\",
              f"  --output-dir {args.output_dir.resolve()} --device {report['device']} --batch-size {report['batch_size']} --num-workers {report['num_workers']}",
              "```", ""]
    return "\n".join(lines)


def main():
    import csv
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or number of workers")
    config = load_stage1_encoder_config(args.config)
    device = resolve_device(args.device)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (args.cache_root or config.cache_dir()).expanduser().resolve()
    for task in dict.fromkeys(args.tasks):
        started = time.perf_counter()
        checkpoint_path = (args.phla_checkpoint if task == "phla" else args.ptcr_checkpoint).expanduser().resolve()
        checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
        validate_encoder_checkpoint(checkpoint, config)
        if checkpoint.get("checkpoint_type") != "task_model" or checkpoint.get("task") != task:
            raise ValueError(f"Expected a {task} task-model checkpoint")
        saved_precision = checkpoint["config"]["training"]["mixed_precision"]
        if checkpoint["config"]["training"]["threshold"] != .5:
            raise ValueError("This evaluator requires the recorded 0.5 threshold")
        data_dir = (args.ph_data_dir if task == "phla" else args.pt_data_dir).expanduser().resolve()
        data_paths = sorted(data_dir.glob("test_*.csv"))
        if not data_paths:
            raise FileNotFoundError(f"No test CSVs in {data_dir}")
        folder, receptor = ("PH", "hla") if task == "phla" else ("PT", "tcr")
        audit_fn = audit_phla if task == "phla" else audit_ptcr
        grouping = phla_groups if task == "phla" else ptcr_groups
        caches = {}
        for entity in ("peptide", receptor):
            manifest = checkpoint["cache_metadata"][entity]
            caches[entity] = EmbeddingCache(
                cache_root / entity, expected_hidden_dim=manifest["hidden_dim"],
                expected_fingerprint=manifest["model"]["checkpoint_fingerprint"],
            )
        model = TaskSpecificStage1EncoderBindingModel(config.model, task)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        split_results, all_labels, all_scores = {}, [], []
        pair_owner = {}
        duplicate_pairs = conflicting_pairs = 0
        for data_path in data_paths:
            split = data_path.stem
            dataset, audit = audit_fn(data_path, config, None)
            group_name, groups = grouping(split, dataset.rows)
            loader = make_cached_dataloader(
                data_path, task, caches["peptide"], caches[receptor],
                batch_size=args.batch_size, shuffle=False, seed=checkpoint["config"]["training"]["seed"],
                num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=False,
                peptide_min_length=config.model.peptide_min_length, peptide_max_length=config.model.peptide_max_length,
            )
            print(f"[DATA {task}/{split}] rows={len(dataset):,} positive={audit['positive']:,} negative={audit['negative']:,}", flush=True)
            labels, scores, loss = predict(model, task, loader, device, saved_precision)
            if not np.array_equal(labels, np.asarray([r[2] for r in dataset.rows])):
                raise RuntimeError("Prediction order does not match audited CSV rows")
            metrics = metrics_with_confusion(labels, scores, groups, loss)
            public = {k: v for k, v in audit.items() if not isinstance(v, set)}
            split_results[split] = {"data": public, "metrics": metrics, "group_by": group_name,
                                    "training_overlap": training_overlap(config, task, dataset.rows),
                                    "dataset_comparison": dataset_comparison(config, task, data_path, dataset.rows)}
            with gzip.open(args.output_dir / f"{task}_{split}_predictions.csv.gz", "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["peptide", receptor, "label", "probability", "prediction", "group_id"])
                for (p, r, y), score, group in zip(dataset.rows, scores, groups):
                    writer.writerow([p, r, y, float(score), int(score > .5), int(group)])
                    if (p, r) in pair_owner:
                        old_split, old_label = pair_owner[p, r]
                        if old_split != split:
                            duplicate_pairs += 1
                            conflicting_pairs += int(old_label != y)
                    else:
                        pair_owner[p, r] = split, y
            all_labels.append(labels)
            all_scores.append(scores)
            print(f"[RESULT {task}/{split}] " + json.dumps({k: metrics[k] for k in METRICS}), flush=True)
        labels = np.concatenate(all_labels)
        scores = np.concatenate(all_scores)
        weighted_loss = sum(e["metrics"]["loss"] * e["metrics"]["samples"] for e in split_results.values()) / len(labels)
        pooled = metrics_with_confusion(labels, scores, None, weighted_loss)
        report = {
            "task": task, "architecture": "stage1_encoder", "encoder_layers": config.model.encoder_num_layers,
            "model_config": config.to_dict()["model"],
            "data_dir": str(data_dir), "cache_root": str(cache_root), "config": str(args.config.resolve()),
            "evaluation_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "mixed_precision": saved_precision, "threshold_comparison": "probability > 0.5",
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                           "round": checkpoint["current_round"], "seed": checkpoint["config"]["training"]["seed"],
                           "best_score": checkpoint["best_score"], "run_timestamp": checkpoint["run_timestamp"]},
            "cache_manifests": {entity: cache.manifest for entity, cache in caches.items()},
            "data": {"samples": len(labels), "positive": int(labels.sum()), "negative": int(len(labels)-labels.sum()),
                     "cross_split_pair_duplicates": duplicate_pairs, "cross_split_pair_conflicts": conflicting_pairs},
            "splits": split_results, "macro_average": average([e["metrics"] for e in split_results.values()]),
            "pooled_metrics": pooled, "elapsed_seconds": time.perf_counter() - started,
        }
        path = args.output_dir / f"stage1_encoder_{folder}_new.json"
        path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        md = args.output_dir / f"stage1_encoder_{folder}_new.md"
        md.write_text(markdown(report, args))
        print(f"[DONE {task}] {md}", flush=True)
        for cache in caches.values():
            cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
