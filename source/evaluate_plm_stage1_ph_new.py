#!/usr/bin/env python3
"""Evaluate a Stage-1 pHLA task checkpoint on every CSV in dataset/PH/new."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
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
from plm_unified.encoders import ESMCResidueEncoder
from plm_unified.metrics import compute_binary_metrics
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


DEFAULT_CHECKPOINT = (
    _ROOT
    / "trained_model/PLM_Unified/fold_1/seed_3407/"
    "best_phla_20260928_021049_999878.pt"
)
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_METRIC_FIELDS = (
    "loss", "auroc", "aupr", "accuracy", "mcc", "f1", "precision",
    "recall", "specificity", "nll", "brier", "ece", "performance_avg",
)


def parse_args(argv=None, *, default_config="plm_stage1.yaml", default_checkpoint=DEFAULT_CHECKPOINT,
               automatic_outputs=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_ROOT / "configs" / default_config
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "dataset/PH/new")
    parser.add_argument("--output-md", type=Path, required=not automatic_outputs)
    parser.add_argument("--output-json", type=Path, required=not automatic_outputs)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not np.isfinite(numeric) else f"{numeric:.4f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sorted_unique(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value))


def audit_dataset(path: Path, config, limit: int | None) -> tuple[OnlinePairDataset, dict[str, Any]]:
    dataset = OnlinePairDataset(
        path,
        "phla",
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        limit=limit,
    )
    peptide_lengths: Counter[int] = Counter()
    hla_lengths: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    split_type_counts: Counter[str] = Counter()
    peptides: set[str] = set()
    hlas: set[str] = set()
    positives = 0
    retained_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "HLA", "label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and retained_rows >= limit:
                break
            peptide = normalise_sequence(row["peptide"])
            hla = normalise_sequence(row["HLA"])
            label = int(row["label"])
            if not set(peptide).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical peptide at {path}:{row_number}")
            if not set(hla).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical HLA at {path}:{row_number}")
            if len(hla) > config.model.hla_max_length:
                raise ValueError(f"HLA exceeds model limit at {path}:{row_number}")
            if len(peptide) < config.model.peptide_min_length or len(peptide) > config.model.peptide_max_length:
                continue
            if label not in {0, 1}:
                raise ValueError(f"Label must be binary at {path}:{row_number}")
            retained_rows += 1
            peptides.add(peptide)
            hlas.add(hla)
            peptide_lengths[len(peptide)] += 1
            hla_lengths[len(hla)] += 1
            positives += label
            source_counts[(row.get("source") or "").strip() or "(empty)"] += 1
            split_type_counts[(row.get("split_type") or "").strip() or "(empty)"] += 1

    duplicate_pair_rows = 0
    conflicting_pair_rows = 0
    seen_pairs: dict[tuple[str, str], int] = {}
    for peptide, hla, label in dataset.rows:
        pair = (peptide, hla)
        if pair in seen_pairs:
            duplicate_pair_rows += 1
            conflicting_pair_rows += int(seen_pairs[pair] != label)
        else:
            seen_pairs[pair] = label

    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "raw_rows": dataset.filter_stats["total"],
        "retained_rows": len(dataset),
        "positive": positives,
        "negative": len(dataset) - positives,
        "positive_rate": positives / len(dataset),
        "unique_peptides": len(peptides),
        "unique_hlas": len(hlas),
        "unique_pairs": len(seen_pairs),
        "duplicate_pair_rows": duplicate_pair_rows,
        "conflicting_pair_rows": conflicting_pair_rows,
        "peptide_length_min": min(peptide_lengths),
        "peptide_length_max": max(peptide_lengths),
        "hla_length_min": min(hla_lengths),
        "hla_length_max": max(hla_lengths),
        "filter_stats": dict(dataset.filter_stats),
        "source_counts": dict(source_counts),
        "split_type_counts": dict(split_type_counts),
        "peptides": peptides,
        "hlas": hlas,
        "pairs": set(seen_pairs),
    }
    return dataset, audit


def audit_training_overlap(data_root: Path, audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = (data_root / "PH" / "train.csv", data_root / "train.csv")
    training_path = next((path for path in candidates if path.is_file()), candidates[0])
    training_peptides: set[str] = set()
    training_hlas: set[str] = set()
    training_pairs: set[tuple[str, str]] = set()
    with training_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        label_column = "label" if "label" in columns else "PH"
        if not {"peptide", "HLA", label_column}.issubset(columns):
            raise ValueError(f"Unexpected PH training columns in {training_path}")
        for row in reader:
            peptide = normalise_sequence(row["peptide"])
            hla = normalise_sequence(row["HLA"])
            training_peptides.add(peptide)
            training_hlas.add(hla)
            training_pairs.add((peptide, hla))
    result: dict[str, Any] = {
        "training_file": str(training_path.resolve()),
        "training_sha256": sha256_file(training_path),
        "unique_training_peptides": len(training_peptides),
        "unique_training_hlas": len(training_hlas),
        "unique_training_pairs": len(training_pairs),
        "splits": {},
    }
    for name, audit in audits.items():
        test_peptides = audit["peptides"]
        test_hlas = audit["hlas"]
        test_pairs = audit["pairs"]
        result["splits"][name] = {
            "unique_test_peptides": len(test_peptides),
            "overlapping_peptides": len(test_peptides & training_peptides),
            "unique_test_hlas": len(test_hlas),
            "overlapping_hlas": len(test_hlas & training_hlas),
            "unique_test_pairs": len(test_pairs),
            "overlapping_pairs": len(test_pairs & training_pairs),
        }
    return result


def encode_unique(
    encoder: ESMCResidueEncoder,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.max_length = max_length
    hidden = np.empty((len(sequences), max_length, encoder.hidden_size), dtype=np.float16)
    masks = np.empty((len(sequences), max_length), dtype=np.uint8)
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
                    f"sequences={end:,}/{len(sequences):,} rate={end / elapsed:,.1f}/s",
                    flush=True,
                )
            del encoded
    return hidden, masks


def group_ids_for(name: str, rows: list[tuple[str, str, int]]) -> tuple[str, np.ndarray]:
    if "unseen_peptide" in name:
        group_by = "peptide"
        values = (row[0] for row in rows)
    elif "unseen_hla" in name:
        group_by = "HLA"
        values = (row[1] for row in rows)
    else:
        group_by = "peptide-HLA pair"
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
    hla_hidden: np.ndarray,
    hla_mask: np.ndarray,
    peptide_index: dict[str, int],
    hla_index: dict[str, int],
    *,
    name: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    p_locations = np.fromiter((peptide_index[row[0]] for row in rows), dtype=np.int64, count=len(rows))
    h_locations = np.fromiter((hla_index[row[1]] for row in rows), dtype=np.int64, count=len(rows))
    labels = np.fromiter((row[2] for row in rows), dtype=np.int64, count=len(rows))
    group_by, groups = group_ids_for(name, rows)
    scores: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    use_autocast = device.type == "cuda" and precision != "none"
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    feature_dtype = autocast_dtype if use_autocast else torch.float32
    total_batches = (len(rows) + batch_size - 1) // batch_size
    started = time.perf_counter()
    samples = 0
    model.eval()
    with torch.inference_mode():
        for batch_id, start in enumerate(range(0, len(rows), batch_size), start=1):
            end = min(start + batch_size, len(rows))
            p_locs = p_locations[start:end]
            h_locs = h_locations[start:end]
            batch_labels = torch.from_numpy(labels[start:end]).to(device, dtype=torch.long)
            peptide = torch.from_numpy(peptide_hidden[p_locs]).to(device, dtype=feature_dtype)
            p_mask = torch.from_numpy(peptide_mask[p_locs].astype(bool, copy=False)).to(device)
            hla = torch.from_numpy(hla_hidden[h_locs]).to(device, dtype=feature_dtype)
            h_mask = torch.from_numpy(hla_mask[h_locs].astype(bool, copy=False)).to(device)
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                output = model("phla", peptide, p_mask, hla, h_mask)
                loss = criterion(output.logits, batch_labels)
            batch_count = end - start
            samples += batch_count
            loss_sum += float(loss.detach().cpu()) * batch_count
            scores.append(torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy())
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[EVAL {name}] {batch_id:,}/{total_batches:,} "
                    f"samples={samples:,}/{len(rows):,} rate={samples / elapsed:,.1f}/s",
                    flush=True,
                )
    probabilities = np.concatenate(scores)
    metrics = compute_binary_metrics(
        labels,
        probabilities,
        loss=loss_sum / samples,
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
    return metrics, labels, probabilities


def markdown(report: dict[str, Any], output_md: Path, output_json: Path) -> str:
    def link(path: str | Path) -> str:
        target = Path(path).resolve()
        relative = os.path.relpath(target, output_md.parent.resolve()).replace(os.sep, "/")
        return f"[`{target.name}`]({relative})"

    data = report["data"]
    splits = report["splits"]
    macro = report["macro_average"]
    pooled = report["pooled_metrics"]
    checkpoint = report["checkpoint"]
    lines = [
        "# Stage 1 最优 pHLA checkpoint：PH new 测试结果",
        "",
        "## 结果摘要",
        "",
        (
            f"使用 Stage 1B 的最佳 pHLA checkpoint 对 `dataset/PH/new` 下 "
            f"**{data['split_count']}** 个测试集、共 **{data['total_rows']:,}** 条样本完成评估。"
        ),
        "",
        "| 汇总 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
        f"| 三个 split 等权宏平均 | — | {fmt(macro['auroc'])} | "
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
            f"测试样本正类 / 负类为 **{data['total_positive']:,} / {data['total_negative']:,}**。"
            "宏平均对三个 split 等权；pooled 指标先合并样本再计算。"
        ),
        "",
        "## 各测试集指标",
        "",
        "| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in splits.items():
        m = item["metrics"]
        d = item["data"]
        lines.append(
            f"| `{name}` | {m['samples']:,} | {d['positive']:,} / {d['negative']:,} | "
            f"{d['positive_rate']:.2%} | {fmt(m['auroc'])} | {fmt(m['aupr'])} | "
            f"{fmt(m['accuracy'])} | {fmt(m['mcc'])} | {fmt(m['f1'])} | "
            f"{fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['specificity'])} | "
            f"{fmt(m['performance_avg'])} | {fmt(m['nll'])} | {fmt(m['ece'])} |"
        )
    lines += [
        "",
        "固定阈值为 `0.5`，正类预测使用严格比较 `score > 0.5`。阈值没有根据测试标签重新选择。`performance_avg` 是 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。",
        "",
        "## 结果解读",
        "",
        (
            f"`test_unseen_peptide` 区分能力最高（AUROC={splits['test_unseen_peptide']['metrics']['auroc']:.4f}）；"
            f"`test_unseen_hla` AUROC={splits['test_unseen_hla']['metrics']['auroc']:.4f}，"
            f"但固定阈值下 Recall={splits['test_unseen_hla']['metrics']['recall']:.4f}、"
            f"Specificity={splits['test_unseen_hla']['metrics']['specificity']:.4f}，"
            "说明在这些未见 HLA 上漏检仍较多。"
        ),
        "",
        (
            f"`test_unseen_pair` AUROC={splits['test_unseen_pair']['metrics']['auroc']:.4f}，"
            f"但只有 {splits['test_unseen_pair']['metrics']['samples']} 条，结果波动可能较大。"
            f"三个 split 等权宏平均 `performance_avg`={macro['performance_avg']:.4f}，"
            f"pooled `performance_avg`={pooled['performance_avg']:.4f}；前者给 51 条的 pair split 与大样本 split 相同权重，"
            "后者则主要受 unseen peptide 大样本影响，建议以分 split 结果为主。"
        ),
        "",
        "### 分组指标与混淆矩阵",
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
        "`test_unseen_peptide` 按 peptide、`test_unseen_hla` 按 HLA、`test_unseen_pair` 按 peptide–HLA pair 分组。Group-macro recall 仅对同时含正负样本的组有定义。",
        "",
        "## 数据完整性与训练集重叠",
        "",
        "| Split | CSV | Rows | 正 / 负 | 唯一 peptide | 唯一 HLA | 唯一 pair | 重复 pair 行 | 标签冲突 | 长度过滤 | peptide 与 PH/train 重叠 | HLA 与 PH/train 重叠 | pair 与 PH/train 重叠 | CSV SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in splits.items():
        d = item["data"]
        overlap = report["training_overlap"]["splits"][name]
        filtered = d["filter_stats"]["peptide_too_short"] + d["filter_stats"]["peptide_too_long"]
        lines.append(
            f"| `{name}` | {link(d['path'])} | {d['raw_rows']:,} | {d['positive']:,} / {d['negative']:,} | "
            f"{d['unique_peptides']:,} | {d['unique_hlas']:,} | {d['unique_pairs']:,} | "
            f"{d['duplicate_pair_rows']:,} | {d['conflicting_pair_rows']:,} | {filtered:,} | "
            f"{overlap['overlapping_peptides']:,} / {overlap['unique_test_peptides']:,} | "
            f"{overlap['overlapping_hlas']:,} / {overlap['unique_test_hlas']:,} | "
            f"{overlap['overlapping_pairs']:,} / {overlap['unique_test_pairs']:,} | `{d['sha256']}` |"
        )
    lines += [
        "",
        (
            f"PH new 三个 split 间共享 pair 行 **{data['cross_split_pair_duplicates']:,}** 条，"
            f"其中标签冲突 **{data['cross_split_pair_conflicts']:,}** 条。"
            f"PH/train 含 {report['training_overlap']['unique_training_peptides']:,} 个唯一 peptide、"
            f"{report['training_overlap']['unique_training_hlas']:,} 个唯一 HLA 和 "
            f"{report['training_overlap']['unique_training_pairs']:,} 个唯一 pair。"
        ),
        "",
        "## Checkpoint 与评测设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Checkpoint | {link(checkpoint['path'])} |",
        f"| Evaluation config | {link(report['config'])} |",
        f"| Checkpoint 类型 / 任务 | `{checkpoint['checkpoint_type']}` / `{checkpoint['task']}` |",
        f"| Fold / seed | {checkpoint['fold']} / {checkpoint['seed']} |",
        f"| Stage / round / global step | `{checkpoint['stage']}` / {checkpoint['round']} / {checkpoint['global_step']:,} |",
        f"| 验证集 pHLA best score | {fmt(checkpoint['best_score'])} |",
        f"| 训练 run timestamp | `{checkpoint['run_timestamp']}` |",
        f"| Checkpoint SHA-256 | `{checkpoint['sha256']}` |",
        f"| ESM-C 模型 | `{report['encoder']['checkpoint_path']}` |",
        f"| ESM-C 指纹 | `{report['encoder']['checkpoint_fingerprint']}` |",
        f"| 设备 | `{report['device_name']}` (`{report['device']}`) |",
        f"| pHLA head 精度 | `{report['mixed_precision']}` |",
        f"| ESM-C encode batch / pair batch | {report['encode_batch_size']} / {report['row_batch_size']} |",
        f"| 阈值 | `{report['threshold']}`，`score > threshold` |",
        f"| Peptide 长度 | {report['peptide_min_length']}–{report['peptide_max_length']} aa；本次过滤 {sum(item['data']['filter_stats']['peptide_too_short'] + item['data']['filter_stats']['peptide_too_long'] for item in splits.values()):,} 行 |",
        f"| HLA 长度 | 最大 {report['hla_max_length']} aa |",
        f"| 评测时间 | {report['evaluation_time']} |",
        "",
        "此 checkpoint 的验证 best score 用于训练期间选模；本报告 PH new 指标只作测试集评估，不参与选模或阈值调整。`test_unseen_pair` 仅 51 条样本，其指标不确定性较大，应结合两个大样本划分解读。",
        "",
        "## 结果文件与复现",
        "",
        f"- 全量指标与数据审计：{link(output_json)}",
        f"- 评测程序：{link(_ROOT / 'source/evaluate_plm_stage1_ph_new.py')}",
        "",
        "```bash",
        "PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
        "  source/evaluate_plm_stage1_ph_new.py \\",
        f"  --checkpoint {checkpoint['path']} \\",
        f"  --data-dir {data['directory']} \\",
        f"  --output-md {output_md.resolve()} \\",
        f"  --output-json {output_json.resolve()} \\",
        f"  --device {report['device']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None, *, default_config="plm_stage1.yaml", config_loader=load_config,
         model_class=TaskSpecificBindingModel, checkpoint_validator=None,
         artifact_namespace=None) -> None:
    args = parse_args(
        argv, default_config=default_config,
        default_checkpoint=None if artifact_namespace else DEFAULT_CHECKPOINT,
        automatic_outputs=artifact_namespace is not None,
    )
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = apply_overrides(
        config_loader(args.config),
        fold=args.fold if args.fold is not None else (None if artifact_namespace else 1),
        seed=args.seed if args.seed is not None else (None if artifact_namespace else 3407),
        device=args.device,
    )
    device = resolve_device(config.training.device)
    if not config.paths.esmc_model.is_dir():
        raise FileNotFoundError(config.paths.esmc_model)
    if args.checkpoint is None:
        from plm_unified.artifacts import find_latest_artifact
        args.checkpoint = find_latest_artifact(config.run_dir(), "best_phla.pt")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "task_model" or checkpoint.get("task") != "phla":
        raise ValueError("Expected a Stage-1 pHLA task_model checkpoint")
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint, config)
    saved_model_config = checkpoint.get("config", {}).get("model")
    if saved_model_config is not None and saved_model_config != config.to_dict()["model"]:
        raise ValueError("Checkpoint model configuration does not match the evaluation config")
    saved_config = checkpoint.get("config", {})
    saved_training = saved_config.get("training", {})
    saved_cache = saved_config.get("cache", {})
    config.training.fold = int(saved_training.get("fold", config.training.fold))
    config.training.seed = int(saved_training.get("seed", config.training.seed))
    config.training.threshold = float(
        saved_training.get("threshold", config.training.threshold)
    )
    config.training.mixed_precision = saved_training.get(
        "mixed_precision", config.training.mixed_precision
    )
    config.cache.dtype = saved_cache.get("dtype", config.cache.dtype)
    model = model_class(config.model, "phla")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    data_dir = args.data_dir.expanduser().resolve()
    data_paths = sorted(data_dir.glob("test_*.csv"))
    if not data_paths:
        raise FileNotFoundError(f"No test_*.csv files found in {data_dir}")
    audits: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[tuple[str, str, int]]] = {}
    pair_owner: dict[tuple[str, str], tuple[str, int]] = {}
    cross_split_pair_duplicates = 0
    cross_split_pair_conflicts = 0
    for path in data_paths:
        name = path.stem
        dataset, audit = audit_dataset(path, config, args.limit)
        audits[name] = audit
        rows_by_split[name] = dataset.rows
        for peptide, hla, label in dataset.rows:
            pair = (peptide, hla)
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
            f"unique_HLA={audit['unique_hlas']:,}",
            flush=True,
        )

    all_peptides = sorted_unique({row[0] for rows in rows_by_split.values() for row in rows})
    all_hlas = sorted_unique({row[1] for rows in rows_by_split.values() for row in rows})
    encoder = ESMCResidueEncoder(
        config.paths.esmc_model,
        max_length=config.model.hla_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    if encoder.hidden_size != config.model.peptide_input_dim or encoder.hidden_size != config.model.hla_input_dim:
        raise ValueError("ESM-C hidden size does not match the checkpoint model config")
    cache_metadata = checkpoint.get("cache_metadata", {})
    expected_fingerprints = {
        cache_metadata.get(entity, {}).get("model", {}).get("checkpoint_fingerprint")
        for entity in ("peptide", "hla")
    }
    expected_fingerprints.discard(None)
    if expected_fingerprints and expected_fingerprints != {encoder.checkpoint_fingerprint}:
        raise ValueError("Loaded ESM-C fingerprint does not match the checkpoint cache metadata")
    print(
        f"[ENCODER] unique peptide={len(all_peptides):,} unique HLA={len(all_hlas):,} "
        f"device={device}",
        flush=True,
    )
    peptide_hidden, peptide_mask = encode_unique(
        encoder, all_peptides, config.model.peptide_max_length,
        args.encode_batch_size, "peptide",
    )
    hla_hidden, hla_mask = encode_unique(
        encoder, all_hlas, config.model.hla_max_length,
        args.encode_batch_size, "HLA",
    )
    peptide_index = {sequence: index for index, sequence in enumerate(all_peptides)}
    hla_index = {sequence: index for index, sequence in enumerate(all_hlas)}

    splits: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for name, rows in rows_by_split.items():
        metrics, labels, scores = evaluate_split(
            model,
            rows,
            peptide_hidden,
            peptide_mask,
            hla_hidden,
            hla_mask,
            peptide_index,
            hla_index,
            name=name,
            device=device,
            precision=config.training.mixed_precision,
            batch_size=args.row_batch_size,
            threshold=config.training.threshold,
        )
        all_labels.append(labels)
        all_scores.append(scores)
        splits[name] = {"data": audits[name], "metrics": metrics}

    pooled_labels = np.concatenate(all_labels)
    pooled_scores = np.concatenate(all_scores)
    clipped = np.clip(pooled_scores, 1e-7, 1.0 - 1e-7)
    pooled_nll = -np.log(np.where(pooled_labels == 1, clipped, 1.0 - clipped))
    pooled_metrics = compute_binary_metrics(
        pooled_labels,
        pooled_scores,
        loss=float(np.mean(pooled_nll)),
        threshold=config.training.threshold,
        strict_threshold=True,
    ).to_dict()
    pooled_predictions = pooled_scores > config.training.threshold
    pooled_metrics.update(
        {
            "positive_samples": int(pooled_labels.sum()),
            "negative_samples": int(len(pooled_labels) - pooled_labels.sum()),
            "positive_rate": float(pooled_labels.mean()),
            "threshold": config.training.threshold,
            "threshold_comparison": "score > threshold",
            "tp": int(np.sum(pooled_predictions & (pooled_labels == 1))),
            "tn": int(np.sum(~pooled_predictions & (pooled_labels == 0))),
            "fp": int(np.sum(pooled_predictions & (pooled_labels == 0))),
            "fn": int(np.sum(~pooled_predictions & (pooled_labels == 1))),
        }
    )
    macro_average = {
        field: float(np.nanmean([item["metrics"][field] for item in splits.values()]))
        for field in _METRIC_FIELDS
    }
    total_positive = int(pooled_labels.sum())
    public_audits = {
        name: {key: value for key, value in audit.items() if key not in {"peptides", "hlas", "pairs"}}
        for name, audit in audits.items()
    }
    training_overlap = audit_training_overlap(config.paths.data_root, audits)
    if artifact_namespace:
        from plm_unified.artifacts import create_artifact_timestamp
        output_base = (
            config.paths.project_root / "reports" / artifact_namespace
            / f"fold_{config.training.fold}" / f"seed_{config.training.seed}"
            / f"PH_new_{create_artifact_timestamp()}"
        )
        args.output_md = args.output_md or output_base.with_suffix(".md")
        args.output_json = args.output_json or output_base.with_suffix(".json")
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "phla",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "mixed_precision": config.training.mixed_precision,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "threshold": config.training.threshold,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "hla_max_length": config.model.hla_max_length,
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": {
            "path": str(checkpoint_path),
            "checkpoint_type": checkpoint["checkpoint_type"],
            "task": checkpoint["task"],
            "fold": checkpoint.get("config", {}).get("training", {}).get("fold", config.training.fold),
            "seed": checkpoint.get("config", {}).get("training", {}).get("seed", config.training.seed),
            "stage": checkpoint.get("stage"),
            "round": checkpoint.get("current_round"),
            "global_step": checkpoint.get("global_step"),
            "best_score": checkpoint.get("best_score"),
            "selection_role": checkpoint.get("selection_role"),
            "run_timestamp": checkpoint.get("run_timestamp"),
            "source_checkpoint": checkpoint.get("source_checkpoint"),
            "sha256": sha256_file(checkpoint_path),
        },
        "encoder": {
            "checkpoint_path": str(encoder.checkpoint_path),
            "checkpoint_fingerprint": encoder.checkpoint_fingerprint,
            "hidden_size": encoder.hidden_size,
            "dtype": config.cache.dtype,
        },
        "data": {
            "directory": str(data_dir),
            "files": [str(path.resolve()) for path in data_paths],
            "split_count": len(data_paths),
            "total_rows": int(len(pooled_labels)),
            "total_positive": total_positive,
            "total_negative": int(len(pooled_labels) - total_positive),
            "positive_rate": float(pooled_labels.mean()),
            "unique_peptides": len(all_peptides),
            "unique_hlas": len(all_hlas),
            "cross_split_pair_duplicates": cross_split_pair_duplicates,
            "cross_split_pair_conflicts": cross_split_pair_conflicts,
        },
        "training_overlap": training_overlap,
        "splits": {
            name: {"data": public_audits[name], "metrics": item["metrics"]}
            for name, item in splits.items()
        },
        "macro_average": macro_average,
        "pooled_metrics": pooled_metrics,
        "outputs": {
            "markdown": str(args.output_md.expanduser().resolve()),
            "json": str(args.output_json.expanduser().resolve()),
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if artifact_namespace:
        report["architecture"] = artifact_namespace
    output_json.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    rendered_markdown = markdown(report, output_md, output_json)
    if artifact_namespace:
        rendered_markdown = rendered_markdown.replace(
            "  --checkpoint", f"  --config {config.source_path} \\\n  --checkpoint"
        )
    output_md.write_text(rendered_markdown, encoding="utf-8")
    print(f"JSON written to: {output_json}", flush=True)
    print(f"Markdown written to: {output_md}", flush=True)
    print(json.dumps({"macro_average": macro_average, "pooled_metrics": pooled_metrics}, indent=2))


if __name__ == "__main__":
    main()
