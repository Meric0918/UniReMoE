#!/usr/bin/env python3
"""Evaluate a Stage-1 pTCR checkpoint on the CSV benchmarks in dataset/PT/new."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import numpy as np
import torch
import torch.nn as nn

_SOURCE = str(_ROOT / "source")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from plm_unified.config import apply_overrides, load_config
from plm_unified.data import OnlinePairDataset, normalise_sequence, stable_group_id
from plm_unified.encoders import ESMCResidueEncoder, TCRBertResidueEncoder
from plm_unified.metrics import compute_binary_metrics
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


DEFAULT_CHECKPOINT = (
    _ROOT
    / "trained_model/PLM_Unified/fold_1/seed_3407/"
    "best_ptcr_20260928_021049_999878.pt"
)
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_METRIC_FIELDS = (
    "loss", "auroc", "aupr", "accuracy", "mcc", "f1", "precision",
    "recall", "sensitivity", "specificity", "nll", "brier", "ece",
    "mean_confidence", "loss_p95", "loss_p99", "group_macro_nll",
    "group_macro_accuracy", "group_macro_recall", "worst_group_nll",
    "group_coverage", "performance_avg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_ROOT / "configs/plm_stage1.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "dataset/PT/new")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NaN" if not np.isfinite(number) else f"{number:.{digits}f}"


def sorted_unique(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value))


def audit_dataset(path: Path, config, limit: int | None):
    dataset = OnlinePairDataset(
        path,
        "ptcr",
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        limit=limit,
    )
    peptides: set[str] = set()
    tcrs: set[str] = set()
    pairs: dict[tuple[str, str], int] = {}
    peptide_lengths: Counter[int] = Counter()
    tcr_lengths: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    split_type_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and row_number > limit + 1:
                break
            peptide = normalise_sequence(row["peptide"])
            tcr = normalise_sequence(row["tcr"])
            if not set(peptide).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical peptide at {path}:{row_number}")
            if not set(tcr).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical TCR at {path}:{row_number}")
            if len(tcr) > config.model.tcr_max_length:
                raise ValueError(f"TCR exceeds model limit at {path}:{row_number}")
            source_counts[(row.get("source") or "").strip() or "(empty)"] += 1
            split_type_counts[(row.get("split_type") or "").strip() or "(empty)"] += 1

    duplicate_pair_rows = 0
    conflicting_pair_rows = 0
    for peptide, tcr, label in dataset.rows:
        peptides.add(peptide)
        tcrs.add(tcr)
        peptide_lengths[len(peptide)] += 1
        tcr_lengths[len(tcr)] += 1
        pair = (peptide, tcr)
        if pair in pairs:
            duplicate_pair_rows += 1
            conflicting_pair_rows += int(pairs[pair] != label)
        else:
            pairs[pair] = label
    positive = sum(label for _, _, label in dataset.rows)
    total = len(dataset)
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "raw_rows": dataset.filter_stats["total"],
        "retained_rows": total,
        "positive": positive,
        "negative": total - positive,
        "positive_rate": positive / total,
        "unique_peptides": len(peptides),
        "unique_tcrs": len(tcrs),
        "unique_pairs": len(pairs),
        "duplicate_pair_rows": duplicate_pair_rows,
        "conflicting_pair_rows": conflicting_pair_rows,
        "peptide_length_min": min(peptide_lengths),
        "peptide_length_max": max(peptide_lengths),
        "tcr_length_min": min(tcr_lengths),
        "tcr_length_max": max(tcr_lengths),
        "filter_stats": dict(dataset.filter_stats),
        "source_counts": dict(source_counts),
        "split_type_counts": dict(split_type_counts),
        "peptides": peptides,
        "tcrs": tcrs,
        "pairs": set(pairs),
    }
    return dataset, audit


def encode_unique(encoder, sequences: list[str], batch_size: int, name: str):
    hidden = np.empty(
        (len(sequences), encoder.max_length, encoder.hidden_size), dtype=np.float16
    )
    masks = np.empty((len(sequences), encoder.max_length), dtype=np.uint8)
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_id, start in enumerate(range(0, len(sequences), batch_size), start=1):
            end = min(start + batch_size, len(sequences))
            encoded = encoder.encode(sequences[start:end])
            hidden[start:end] = encoded.hidden.detach().to("cpu", dtype=torch.float16).numpy()
            masks[start:end] = encoded.mask.detach().to("cpu", dtype=torch.uint8).numpy()
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if encoder.device.type == "cuda":
                    torch.cuda.synchronize(encoder.device)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[ENCODE {name}] {batch_id:,}/{total_batches:,} "
                    f"sequences={end:,}/{len(sequences):,} "
                    f"rate={end / elapsed:,.1f}/s",
                    flush=True,
                )
            del encoded
    return hidden, masks


def group_ids_for(name: str, rows: list[tuple[str, str, int]]) -> tuple[str, np.ndarray]:
    if "unseen_peptide" in name:
        group_by = "peptide"
        values = (row[0] for row in rows)
    elif "unseen_tcr" in name:
        group_by = "TCR"
        values = (row[1] for row in rows)
    else:
        group_by = "peptide–TCR pair"
        values = (f"{row[0]}\x1f{row[1]}" for row in rows)
    groups = np.fromiter(
        (stable_group_id(value) for value in values), dtype=np.int64, count=len(rows)
    )
    return group_by, groups


def evaluate_split(
    model: TaskSpecificBindingModel,
    rows: list[tuple[str, str, int]],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    tcr_hidden: np.ndarray,
    tcr_mask: np.ndarray,
    peptide_index: dict[str, int],
    tcr_index: dict[str, int],
    *,
    name: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    threshold: float,
):
    p_locations = np.fromiter((peptide_index[row[0]] for row in rows), dtype=np.int64)
    t_locations = np.fromiter((tcr_index[row[1]] for row in rows), dtype=np.int64)
    labels = np.fromiter((row[2] for row in rows), dtype=np.int64)
    group_by, groups = group_ids_for(name, rows)
    scores: list[np.ndarray] = []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    use_autocast = device.type == "cuda" and precision != "none"
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    feature_dtype = autocast_dtype if use_autocast else torch.float32
    total_batches = (len(rows) + batch_size - 1) // batch_size
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch_id, start in enumerate(range(0, len(rows), batch_size), start=1):
            end = min(start + batch_size, len(rows))
            p_locs = p_locations[start:end]
            t_locs = t_locations[start:end]
            batch_labels = torch.from_numpy(labels[start:end]).to(device, dtype=torch.long)
            peptide = torch.from_numpy(peptide_hidden[p_locs]).to(device, dtype=feature_dtype)
            p_mask = torch.from_numpy(peptide_mask[p_locs].astype(bool, copy=False)).to(device)
            tcr = torch.from_numpy(tcr_hidden[t_locs]).to(device, dtype=feature_dtype)
            t_mask = torch.from_numpy(tcr_mask[t_locs].astype(bool, copy=False)).to(device)
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with context:
                output = model("ptcr", peptide, p_mask, tcr, t_mask)
                loss = criterion(output.logits, batch_labels)
            batch_count = end - start
            loss_sum += float(loss.detach().cpu()) * batch_count
            scores.append(torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy())
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[EVAL {name}] {batch_id:,}/{total_batches:,} "
                    f"samples={end:,}/{len(rows):,} rate={end / elapsed:,.1f}/s",
                    flush=True,
                )
    probabilities = np.concatenate(scores)
    metrics = compute_binary_metrics(
        labels,
        probabilities,
        loss=loss_sum / len(rows),
        threshold=threshold,
        strict_threshold=True,
        group_ids=groups,
    ).to_dict()
    predictions = probabilities > threshold
    metrics.update(
        {
            "positive_samples": int(labels.sum()),
            "negative_samples": int(len(labels) - labels.sum()),
            "positive_rate": float(labels.mean()),
            "threshold": threshold,
            "threshold_comparison": "score > threshold",
            "tp": int(np.sum(predictions & (labels == 1))),
            "tn": int(np.sum(~predictions & (labels == 0))),
            "fp": int(np.sum(predictions & (labels == 0))),
            "fn": int(np.sum(~predictions & (labels == 1))),
            "group_by": group_by,
        }
    )
    return metrics, labels, probabilities, groups


def pooled_metrics(labels, scores, groups, threshold: float):
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    y_groups = np.concatenate(groups)
    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    nll = -np.log(np.where(y_true == 1, clipped, 1.0 - clipped))
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=float(np.mean(nll)),
        threshold=threshold,
        strict_threshold=True,
        group_ids=y_groups,
    ).to_dict()
    predictions = y_score > threshold
    metrics.update(
        {
            "positive_samples": int(y_true.sum()),
            "negative_samples": int(len(y_true) - y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "threshold": threshold,
            "threshold_comparison": "score > threshold",
            "tp": int(np.sum(predictions & (y_true == 1))),
            "tn": int(np.sum(~predictions & (y_true == 0))),
            "fp": int(np.sum(predictions & (y_true == 0))),
            "fn": int(np.sum(~predictions & (y_true == 1))),
        }
    )
    return metrics


def markdown(report: dict[str, Any], output_md: Path, output_json: Path) -> str:
    def link(path: str | Path, label: str | None = None) -> str:
        target = Path(path).resolve()
        relative = os.path.relpath(target, output_md.parent.resolve()).replace(os.sep, "/")
        return f"[{label or target.name}]({relative})"

    data = report["data"]
    splits = report["splits"]
    macro = report["macro_average"]
    pooled = report["pooled_metrics"]
    checkpoint = report["checkpoint"]
    best_split = max(splits, key=lambda name: splits[name]["metrics"]["performance_avg"])
    weakest_split = min(splits, key=lambda name: splits[name]["metrics"]["performance_avg"])
    lines = [
        "# Stage 1 pTCR 模型：PT new 测试结果",
        "",
        "## 结果摘要",
        "",
        (
            f"使用 Stage 1B pTCR checkpoint 对 [`dataset/PT/new`]("
            f"{link(data['directory'], 'dataset/PT/new').split('](', 1)[1]} 下 "
            f"{data['split_count']} 个测试集、共 **{data['total_rows']:,}** 条保留样本完成评估。"
        ),
        "",
        "| 汇总 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 三个 split 等权宏平均 | {data['total_rows']:,} | {fmt(macro['auroc'])} | "
            f"{fmt(macro['aupr'])} | {fmt(macro['accuracy'])} | {fmt(macro['mcc'])} | "
            f"{fmt(macro['f1'])} | {fmt(macro['precision'])} | {fmt(macro['recall'])} | "
            f"{fmt(macro['specificity'])} | {fmt(macro['performance_avg'])} | "
            f"{fmt(macro['nll'])} | {fmt(macro['brier'])} | {fmt(macro['ece'])} |"
        ),
        (
            f"| 全部样本 pooled | {pooled['samples']:,} | {fmt(pooled['auroc'])} | "
            f"{fmt(pooled['aupr'])} | {fmt(pooled['accuracy'])} | {fmt(pooled['mcc'])} | "
            f"{fmt(pooled['f1'])} | {fmt(pooled['precision'])} | {fmt(pooled['recall'])} | "
            f"{fmt(pooled['specificity'])} | {fmt(pooled['performance_avg'])} | "
            f"{fmt(pooled['nll'])} | {fmt(pooled['brier'])} | {fmt(pooled['ece'])} |"
        ),
        "",
        (
            f"纳入 **{data['total_rows']:,}** 条样本：正类 **{data['total_positive']:,}**、"
            f"负类 **{data['total_negative']:,}**，正类比例 **{data['positive_rate']:.2%}**。"
        ),
        "",
        (
            f"`{best_split}` 的 `performance_avg` 最高 "
            f"（{fmt(splits[best_split]['metrics']['performance_avg'])}）；"
            f"`{weakest_split}` 最低（{fmt(splits[weakest_split]['metrics']['performance_avg'])}）。"
        ),
        "",
        "宏平均对三个 split 等权；pooled 指标将样本合并后重新计算。`performance_avg` 是 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。",
        "",
        "## 各测试集结果",
        "",
        "| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in splits.items():
        m, d = item["metrics"], item["data"]
        lines.append(
            f"| `{name}` | {m['samples']:,} | {d['positive']:,} / {d['negative']:,} | "
            f"{d['positive_rate']:.2%} | {fmt(m['auroc'])} | {fmt(m['aupr'])} | "
            f"{fmt(m['accuracy'])} | {fmt(m['mcc'])} | {fmt(m['f1'])} | "
            f"{fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['specificity'])} | "
            f"{fmt(m['performance_avg'])} | {fmt(m['nll'])} | {fmt(m['ece'])} |"
        )
    lines += [
        "",
        "固定阈值为 `0.5`，正类预测使用严格比较 `score > 0.5`；未依据测试标签重新选择阈值。",
        "",
        "### 分组鲁棒性与混淆矩阵",
        "",
        "| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in splits.items():
        m = item["metrics"]
        lines.append(
            f"| `{name}` | {m['group_by']} | {m['group_count']:,} | "
            f"{fmt(m['group_macro_accuracy'])} | {fmt(m['group_macro_recall'])} | "
            f"{fmt(m['group_coverage'])} | {fmt(m['group_macro_nll'])} | "
            f"{fmt(m['worst_group_nll'])} | {m['tn']:,} | {m['fp']:,} | "
            f"{m['fn']:,} | {m['tp']:,} |"
        )
    lines += [
        "",
        "Group-macro 指标在各 split 的主要未见实体上等权计算：unseen peptide 按 peptide、unseen TCR 按 TCR、unseen pair 按 peptide–TCR pair 分组。Group-macro recall 仅对同时包含正负标签的组有定义。",
        "",
        "## 数据检查",
        "",
        "| Split | CSV | 原始行数 | 纳入行数 | 正 / 负 | 唯一 peptide | 唯一 TCR | 唯一 pair | 重复 pair 行 | 标签冲突 | peptide 长度 | TCR 长度 | 长度过滤 | SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in splits.items():
        d = item["data"]
        filtered = d["filter_stats"]["peptide_too_short"] + d["filter_stats"]["peptide_too_long"]
        lines.append(
            f"| `{name}` | {link(d['path'])} | {d['raw_rows']:,} | {d['retained_rows']:,} | "
            f"{d['positive']:,} / {d['negative']:,} | {d['unique_peptides']:,} | "
            f"{d['unique_tcrs']:,} | {d['unique_pairs']:,} | "
            f"{d['duplicate_pair_rows']:,} | {d['conflicting_pair_rows']:,} | "
            f"{d['peptide_length_min']}–{d['peptide_length_max']} aa | "
            f"{d['tcr_length_min']}–{d['tcr_length_max']} aa | {filtered:,} | `{d['sha256']}` |"
        )
    lines += [
        "",
        (
            f"三个文件合计重复 pair 行 **{data['duplicate_pair_rows']:,}**，"
            f"标签冲突 **{data['conflicting_pair_rows']:,}**；跨文件共享 pair **"
            f"{data['cross_split_pair_duplicates']:,}**，跨文件标签冲突 **"
            f"{data['cross_split_pair_conflicts']:,}**。"
        ),
        "",
        "## Checkpoint 与评测设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Checkpoint | {link(checkpoint['path'])} |",
        f"| 类型 / 任务 | `{checkpoint['checkpoint_type']}` / `{checkpoint['task']}` |",
        f"| Fold / seed | {checkpoint['fold']} / {checkpoint['seed']} |",
        f"| Stage / best round / global step | `{checkpoint['stage']}` / {checkpoint['round']} / {checkpoint['global_step']:,} |",
        f"| 验证集最佳分数 | {fmt(checkpoint['best_score'], 6)} |",
        f"| Checkpoint SHA-256 | `{checkpoint['sha256']}` |",
        f"| ESM-C encoder | `{report['encoders']['esmc']['checkpoint_path']}` |",
        f"| TCR-BERT encoder | `{report['encoders']['tcr_bert']['checkpoint_path']}` |",
        f"| 设备 | `{report['device_name']}` (`{report['device']}`) |",
        f"| 混合精度 | pTCR head `{report['mixed_precision']}`；encoder `{report['encoder_dtype']}` |",
        f"| Encoder batch / pair batch | {report['encode_batch_size']} / {report['row_batch_size']} |",
        f"| Peptide / TCR 长度上限 | {report['model_config']['peptide_max_length']} / {report['model_config']['tcr_max_length']} aa |",
        f"| 评测时间 | {report['evaluation_time']} |",
        "",
        "Checkpoint 的 `best_score` 是训练验证集上的选模分数；PT new 指标来自独立测试文件，不参与选模或阈值选择。Stage 1 checkpoint 保存 pTCR head 和 adapters；ESM-C 与 TCR-BERT 使用配置中本地的冻结编码器权重，先对测试集唯一序列在线编码，再进行 pTCR 推理。",
        "",
        "## 结果文件与复现",
        "",
        f"- 全量指标与数据审计：{link(output_json)}",
        f"- 评估程序：{link(_ROOT / 'source/evaluate_plm_stage1_pt_new.py')}",
        "",
        "```bash",
        "PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
        "  source/evaluate_plm_stage1_pt_new.py \\",
        f"  --checkpoint {checkpoint['path']} \\",
        f"  --data-dir {data['directory']} \\",
        f"  --output-md {output_md.resolve()} \\",
        f"  --output-json {output_json.resolve()} \\",
        f"  --device {report['device']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "task_model" or checkpoint.get("task") != "ptcr":
        raise ValueError("Expected a Stage-1 pTCR task_model checkpoint")
    saved_config = checkpoint.get("config", {})
    saved_training = saved_config.get("training", {})
    config = apply_overrides(
        load_config(config_path),
        fold=saved_training.get("fold", 1),
        seed=saved_training.get("seed", 3407),
        device=args.device,
    )
    if saved_config.get("model") is not None and saved_config["model"] != config.to_dict()["model"]:
        raise ValueError("Checkpoint model configuration does not match evaluation config")
    device = resolve_device(config.training.device)
    for path in (config.paths.esmc_model, config.paths.tcr_bert_model):
        if not path.is_dir():
            raise FileNotFoundError(path)

    model = TaskSpecificBindingModel(config.model, "ptcr")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    data_paths = sorted(data_dir.glob("test_*.csv"))
    if not data_paths:
        raise FileNotFoundError(f"No test_*.csv files found in {data_dir}")
    rows_by_split: dict[str, list[tuple[str, str, int]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    pair_owner: dict[tuple[str, str], tuple[str, int]] = {}
    cross_split_pair_duplicates = 0
    cross_split_pair_conflicts = 0
    total_duplicate_rows = 0
    total_conflicting_rows = 0
    for path in data_paths:
        name = path.stem
        dataset, audit = audit_dataset(path, config, args.limit)
        rows_by_split[name] = dataset.rows
        audits[name] = audit
        total_duplicate_rows += audit["duplicate_pair_rows"]
        total_conflicting_rows += audit["conflicting_pair_rows"]
        for peptide, tcr, label in dataset.rows:
            pair = (peptide, tcr)
            if pair in pair_owner:
                old_split, old_label = pair_owner[pair]
                if old_split != name:
                    cross_split_pair_duplicates += 1
                    cross_split_pair_conflicts += int(old_label != label)
            else:
                pair_owner[pair] = (name, label)
        print(
            f"[DATA] {name}: rows={len(dataset):,} positive={audit['positive']:,} "
            f"negative={audit['negative']:,} unique_peptide={audit['unique_peptides']:,} "
            f"unique_tcr={audit['unique_tcrs']:,}",
            flush=True,
        )

    all_peptides = sorted_unique({row[0] for rows in rows_by_split.values() for row in rows})
    all_tcrs = sorted_unique({row[1] for rows in rows_by_split.values() for row in rows})
    esmc = ESMCResidueEncoder(
        config.paths.esmc_model,
        max_length=config.model.peptide_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    tcr_bert = TCRBertResidueEncoder(
        config.paths.tcr_bert_model,
        max_length=config.model.tcr_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    if esmc.hidden_size != config.model.peptide_input_dim:
        raise ValueError("ESM-C hidden size does not match checkpoint model config")
    if tcr_bert.hidden_size != config.model.tcr_input_dim:
        raise ValueError("TCR-BERT hidden size does not match checkpoint model config")
    print(
        f"[ENCODERS] unique peptide={len(all_peptides):,} unique TCR={len(all_tcrs):,} "
        f"device={device}",
        flush=True,
    )
    peptide_hidden, peptide_mask = encode_unique(
        esmc, all_peptides, args.encode_batch_size, "peptide"
    )
    tcr_hidden, tcr_mask = encode_unique(
        tcr_bert, all_tcrs, args.encode_batch_size, "TCR"
    )
    peptide_index = {sequence: index for index, sequence in enumerate(all_peptides)}
    tcr_index = {sequence: index for index, sequence in enumerate(all_tcrs)}

    splits: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    for name, rows in rows_by_split.items():
        metrics, labels, scores, groups = evaluate_split(
            model,
            rows,
            peptide_hidden,
            peptide_mask,
            tcr_hidden,
            tcr_mask,
            peptide_index,
            tcr_index,
            name=name,
            device=device,
            precision=config.training.mixed_precision,
            batch_size=args.row_batch_size,
            threshold=config.training.threshold,
        )
        splits[name] = {"data": audits[name], "metrics": metrics}
        all_labels.append(labels)
        all_scores.append(scores)
        all_groups.append(groups)

    total_rows = sum(len(rows) for rows in rows_by_split.values())
    total_positive = sum(audit["positive"] for audit in audits.values())
    encoder_metadata = {
        "esmc": {
            "checkpoint_path": str(esmc.checkpoint_path),
            "checkpoint_fingerprint": esmc.checkpoint_fingerprint,
            "hidden_size": esmc.hidden_size,
        },
        "tcr_bert": {
            "checkpoint_path": str(tcr_bert.checkpoint_path),
            "checkpoint_fingerprint": tcr_bert.checkpoint_fingerprint,
            "hidden_size": tcr_bert.hidden_size,
        },
    }
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "ptcr",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "mixed_precision": config.training.mixed_precision,
        "encoder_dtype": config.cache.dtype,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "threshold": config.training.threshold,
        "config_path": str(config_path),
        "model_config": config.to_dict()["model"],
        "encoders": encoder_metadata,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "checkpoint_type": checkpoint.get("checkpoint_type"),
            "task": checkpoint.get("task"),
            "fold": saved_training.get("fold", config.training.fold),
            "seed": saved_training.get("seed", config.training.seed),
            "stage": checkpoint.get("stage"),
            "round": checkpoint.get("current_round"),
            "global_step": checkpoint.get("global_step"),
            "best_score": checkpoint.get("best_score"),
            "selection_role": checkpoint.get("selection_role"),
            "run_timestamp": checkpoint.get("run_timestamp"),
        },
        "data": {
            "directory": str(data_dir),
            "files": [str(path.resolve()) for path in data_paths],
            "split_count": len(data_paths),
            "total_rows": total_rows,
            "total_positive": total_positive,
            "total_negative": total_rows - total_positive,
            "positive_rate": total_positive / total_rows,
            "unique_peptides": len(all_peptides),
            "unique_tcrs": len(all_tcrs),
            "duplicate_pair_rows": total_duplicate_rows,
            "conflicting_pair_rows": total_conflicting_rows,
            "cross_split_pair_duplicates": cross_split_pair_duplicates,
            "cross_split_pair_conflicts": cross_split_pair_conflicts,
        },
        "splits": {
            name: {
                "data": {key: value for key, value in audit.items() if key not in {"peptides", "tcrs", "pairs"}},
                "metrics": splits[name]["metrics"],
            }
            for name, audit in audits.items()
        },
        "macro_average": {
            field: float(np.nanmean([item["metrics"][field] for item in splits.values()]))
            for field in _METRIC_FIELDS
        },
        "pooled_metrics": pooled_metrics(
            all_labels, all_scores, all_groups, config.training.threshold
        ),
        "outputs": {
            "markdown": str(args.output_md.expanduser().resolve()),
            "json": str(args.output_json.expanduser().resolve()),
        },
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    output_md.write_text(markdown(report, output_md, output_json), encoding="utf-8")
    print(f"[DONE] JSON written to: {output_json}", flush=True)
    print(f"[DONE] Markdown written to: {output_md}", flush=True)
    print(json.dumps({"macro_average": report["macro_average"], "pooled_metrics": report["pooled_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
