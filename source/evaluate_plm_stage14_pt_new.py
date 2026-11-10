#!/usr/bin/env python3
"""Evaluate a Stage-14 checkpoint on every CSV split in dataset/PT/new."""

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

import numpy as np

# Make the conda-bundled CUDA toolkit visible to Transformer Engine before any
# Transformers ESM-C module is imported.
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "source")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from plm_unified.data import OnlinePairDataset, stable_group_id
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage14_checkpoint import (
    checkpoint_sha256,
    load_stage14_checkpoint,
    validate_stage14_for_evaluation,
)
from plm_unified.stage14_config import (
    STAGE14_APPROVED_STAGE1_SHA256S,
    apply_stage14_overrides,
    load_stage14_config,
)
from plm_unified.stage14_model import build_progressive_stage14_model
from plm_unified.trainer import resolve_device


_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_ROOT / "configs" / "plm_stage14.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "dataset" / "PT" / "new",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def feature_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def sorted_unique(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value))


def encode_unique(
    name: str,
    encoder,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.empty((len(sequences), max_length, encoder.hidden_size), dtype=np.float16)
    masks = np.empty((len(sequences), max_length), dtype=np.uint8)
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_id, start in enumerate(range(0, len(sequences), batch_size), start=1):
            end = min(start + batch_size, len(sequences))
            with autocast(device, precision):
                encoded = encoder(sequences[start:end])
            hidden[start:end] = encoded.hidden.detach().to("cpu", dtype=torch.float16).numpy()
            masks[start:end] = encoded.mask.detach().to("cpu", dtype=torch.uint8).numpy()
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[ENCODE {name}] {batch_id:,}/{total_batches:,} "
                    f"sequences={end:,}/{len(sequences):,} "
                    f"rate={end / elapsed:,.1f}/s",
                    flush=True,
                )
            del encoded
    return hidden, masks


def audit_data(path: Path, config, limit: int | None) -> dict[str, Any]:
    counts = {"rows": 0, "positive": 0, "negative": 0}
    split_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    peptide_lengths: Counter[int] = Counter()
    tcr_lengths: Counter[int] = Counter()
    peptides: set[str] = set()
    tcrs: set[str] = set()
    seen_pairs: dict[tuple[str, str], int] = {}
    duplicate_rows = 0
    conflicting_pairs = 0
    required = {"peptide", "tcr", "label"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and counts["rows"] >= limit:
                break
            peptide = "".join((row.get("peptide") or "").split()).replace("-", "").upper()
            tcr = "".join((row.get("tcr") or "").split()).replace("-", "").upper()
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be binary at {path}:{row_number}")
            if not peptide or not tcr:
                raise ValueError(f"Empty peptide/TCR at {path}:{row_number}")
            if not set(peptide).issubset(_AMINO_ACIDS) or not set(tcr).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical sequence at {path}:{row_number}")
            if len(tcr) > config.model.tcr_max_length:
                raise ValueError(f"TCR exceeds max length at {path}:{row_number}")
            pair = (peptide, tcr)
            if pair in seen_pairs:
                duplicate_rows += 1
                conflicting_pairs += int(seen_pairs[pair] != label)
            else:
                seen_pairs[pair] = label
            counts["rows"] += 1
            counts["positive" if label else "negative"] += 1
            peptides.add(peptide)
            tcrs.add(tcr)
            split_types[(row.get("split_type") or "").strip()] += 1
            sources[(row.get("source") or "").strip()] += 1
            peptide_lengths[len(peptide)] += 1
            tcr_lengths[len(tcr)] += 1
    return {
        **counts,
        "split_types": dict(split_types),
        "sources": dict(sources),
        "peptide_length_min": min(peptide_lengths) if peptide_lengths else None,
        "peptide_length_max": max(peptide_lengths) if peptide_lengths else None,
        "tcr_length_min": min(tcr_lengths) if tcr_lengths else None,
        "tcr_length_max": max(tcr_lengths) if tcr_lengths else None,
        "unique_peptides": len(peptides),
        "unique_tcrs": len(tcrs),
        "duplicate_pair_rows": duplicate_rows,
        "conflicting_pair_rows": conflicting_pairs,
    }


def audit_training_peptide_overlap(config, test_peptides: set[str]) -> dict[str, Any]:
    data_root = config.paths.data_root
    paths = [data_root / "PH" / "train.csv", data_root / "PT" / "train.csv"]
    paths.extend(
        data_root / "PHT" / "classes" / kind / config.stage14.train_filename
        for kind in config.stage14.class_dirs
    )
    sources: dict[str, dict[str, int]] = {}
    training_union: set[str] = set()
    for path in paths:
        if not path.is_file():
            return {"available": False, "missing_source": str(path), "sources": sources}
        sequences: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "peptide" not in (reader.fieldnames or []):
                return {"available": False, "invalid_source": str(path), "sources": sources}
            for row in reader:
                value = "".join((row.get("peptide") or "").split()).replace("-", "").upper()
                if value:
                    sequences.add(value)
        overlap = test_peptides & sequences
        relative = path.resolve().relative_to(_ROOT).as_posix()
        sources[relative] = {
            "unique_train_peptides": len(sequences),
            "overlapping_test_peptides": len(overlap),
        }
        training_union.update(sequences)
    overlap_union = test_peptides & training_union
    return {
        "available": True,
        "unique_train_peptides_union": len(training_union),
        "test_peptides": len(test_peptides),
        "overlapping_test_peptides": len(overlap_union),
        "test_peptides_not_seen_in_train": len(test_peptides - training_union),
        "sources": sources,
    }


def audit_training_tcr_overlap(config, test_tcrs: set[str]) -> dict[str, Any]:
    data_root = config.paths.data_root
    paths = [data_root / "PT" / "train.csv"]
    paths.extend(
        data_root / "PHT" / "classes" / kind / config.stage14.train_filename
        for kind in config.stage14.class_dirs
    )
    sources: dict[str, dict[str, int]] = {}
    training_union: set[str] = set()
    for path in paths:
        if not path.is_file():
            return {"available": False, "missing_source": str(path), "sources": sources}
        sequences: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "tcr" not in (reader.fieldnames or []):
                return {"available": False, "invalid_source": str(path), "sources": sources}
            for row in reader:
                value = "".join((row.get("tcr") or "").split()).replace("-", "").upper()
                if value:
                    sequences.add(value)
        overlap = test_tcrs & sequences
        relative = path.resolve().relative_to(_ROOT).as_posix()
        sources[relative] = {
            "unique_train_tcrs": len(sequences),
            "overlapping_test_tcrs": len(overlap),
        }
        training_union.update(sequences)
    overlap_union = test_tcrs & training_union
    return {
        "available": True,
        "unique_train_tcrs_union": len(training_union),
        "test_tcrs": len(test_tcrs),
        "overlapping_test_tcrs": len(overlap_union),
        "test_tcrs_not_seen_in_train": len(test_tcrs - training_union),
        "sources": sources,
    }


def evaluate_rows(
    model,
    rows: list[tuple[str, str, int]],
    peptides: list[str],
    tcrs: list[str],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    tcr_hidden: np.ndarray,
    tcr_mask: np.ndarray,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    group_by: str,
):
    peptide_index = {sequence: index for index, sequence in enumerate(peptides)}
    tcr_index = {sequence: index for index, sequence in enumerate(tcrs)}
    peptide_locations = np.fromiter(
        (peptide_index[row[0]] for row in rows), dtype=np.int64, count=len(rows)
    )
    tcr_locations = np.fromiter(
        (tcr_index[row[1]] for row in rows), dtype=np.int64, count=len(rows)
    )
    labels = np.fromiter((row[2] for row in rows), dtype=np.int64, count=len(rows))
    if group_by == "peptide":
        group_sequences = (row[0] for row in rows)
    elif group_by == "tcr":
        group_sequences = (row[1] for row in rows)
    elif group_by == "pair":
        group_sequences = (f"{row[0]}\x1f{row[1]}" for row in rows)
    else:
        raise ValueError(f"Unsupported group_by value: {group_by}")
    groups = np.fromiter(
        (stable_group_id(value) for value in group_sequences),
        dtype=np.int64,
        count=len(rows),
    )
    probabilities: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    row_dtype = feature_dtype(device, precision)
    started = time.perf_counter()
    samples = 0
    with torch.inference_mode():
        for batch_id, start in enumerate(range(0, len(rows), batch_size), start=1):
            end = min(start + batch_size, len(rows))
            p_locs = peptide_locations[start:end]
            t_locs = tcr_locations[start:end]
            batch_labels = torch.from_numpy(labels[start:end]).to(
                device, dtype=torch.long, non_blocking=device.type == "cuda"
            )
            peptide = torch.from_numpy(peptide_hidden[p_locs]).to(
                device, dtype=row_dtype, non_blocking=device.type == "cuda"
            )
            p_mask = torch.from_numpy(peptide_mask[p_locs].astype(bool, copy=False)).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            tcr = torch.from_numpy(tcr_hidden[t_locs]).to(
                device, dtype=row_dtype, non_blocking=device.type == "cuda"
            )
            t_mask = torch.from_numpy(tcr_mask[t_locs].astype(bool, copy=False)).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            with autocast(device, precision):
                output = model.binding_model("ptcr", peptide, p_mask, tcr, t_mask)
                loss = criterion(output.logits, batch_labels)
            batch_count = end - start
            samples += batch_count
            loss_sum += float(loss.detach().cpu()) * batch_count
            probabilities.append(torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy())
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[EVAL rows] {batch_id:,}/{total_batches:,} "
                    f"samples={samples:,}/{len(rows):,} "
                    f"rate={samples / elapsed:,.1f}/s",
                    flush=True,
                )
    scores = np.concatenate(probabilities)
    metrics = compute_binary_metrics(
        labels,
        scores,
        loss=loss_sum / samples,
        threshold=0.5,
        strict_threshold=True,
        group_ids=groups,
    ).to_dict()
    predictions = scores > 0.5
    metrics.update(
        {
            "positive_samples": int(labels.sum()),
            "negative_samples": int(len(labels) - labels.sum()),
            "positive_rate": float(labels.mean()),
            "threshold": 0.5,
            "threshold_comparison": "score > 0.5",
            "tp": int(np.sum(predictions & (labels == 1))),
            "tn": int(np.sum(~predictions & (labels == 0))),
            "fp": int(np.sum(predictions & (labels == 0))),
            "fn": int(np.sum(~predictions & (labels == 1))),
        }
    )
    return metrics, labels, scores


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return "NaN" if not np.isfinite(value) else f"{float(value):.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def markdown(report: dict[str, Any], output_md: Path, output_json: Path) -> str:
    checkpoint = report["checkpoint"]
    data = report["data"]
    metrics_by_split = report["splits"]
    macro = report["macro_average"]
    pooled = report["pooled_metrics"]
    benchmark_rows = []
    for split_name, item in metrics_by_split.items():
        metrics = item["metrics"]
        benchmark_rows.append(
            f"| `{split_name}` | {fmt(metrics['samples'])} | "
            f"{fmt(item['data']['positive'])} / {fmt(item['data']['negative'])} | "
            f"{item['data']['positive_rate']:.2%} | {fmt(metrics['auroc'])} | "
            f"{fmt(metrics['aupr'])} | {fmt(metrics['accuracy'])} | "
            f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
            f"{fmt(metrics['recall'])} | {fmt(metrics['specificity'])} | "
            f"{fmt(metrics['performance_avg'])} |"
        )
    split_names = list(metrics_by_split)
    best_split = max(split_names, key=lambda key: metrics_by_split[key]["metrics"]["performance_avg"])
    weakest_split = min(split_names, key=lambda key: metrics_by_split[key]["metrics"]["performance_avg"])
    lines = [
        "# PLM-Stage14 baseline：PT new 全部测试集总结",
        "",
        "## 结论",
        "",
        (
            f"使用 Stage14 baseline 第 **{checkpoint['best_round']}** 轮最佳 checkpoint，"
            f"对 `dataset/PT/new` 下全部 **{data['split_count']}** 个 CSV 测试集共 "
            f"**{data['total_rows']:,}** 条 PTCR 样本完成评估。"
        ),
        "",
        "| 汇总 | 样本总数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 三个 split 等权宏平均 | {fmt(data['total_rows'])} | "
            f"{fmt(macro['auroc'])} | {fmt(macro['aupr'])} | {fmt(macro['accuracy'])} | "
            f"{fmt(macro['mcc'])} | {fmt(macro['f1'])} | {fmt(macro['precision'])} | "
            f"{fmt(macro['recall'])} | {fmt(macro['specificity'])} | {fmt(macro['performance_avg'])} |"
        ),
        (
            f"| 全部样本 pooled | {fmt(pooled['samples'])} | {fmt(pooled['auroc'])} | "
            f"{fmt(pooled['aupr'])} | {fmt(pooled['accuracy'])} | {fmt(pooled['mcc'])} | "
            f"{fmt(pooled['f1'])} | {fmt(pooled['precision'])} | {fmt(pooled['recall'])} | "
            f"{fmt(pooled['specificity'])} | {fmt(pooled['performance_avg'])} |"
        ),
        "",
        (
            f"三个 split 的正负样本合计为 **{data['total_positive']:,} / "
            f"{data['total_negative']:,}**；其中 `{best_split}` 的 `performance_avg` 最高 "
            f"（{metrics_by_split[best_split]['metrics']['performance_avg']:.4f}），"
            f"`{weakest_split}` 最低（{metrics_by_split[weakest_split]['metrics']['performance_avg']:.4f}）。"
        ),
        "",
        "`performance_avg` 为 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。宏平均对三个 split 等权；pooled 指标将所有测试样本合并后重新计算。分类指标使用固定阈值 `score > 0.5`，未基于测试标签调阈值。",
        "",
        "## 各测试集结果",
        "",
        "| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | `performance_avg` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *benchmark_rows,
        "",
        "### 分组鲁棒性与混淆矩阵",
        "",
        "分组对象按测试设计确定：`unseen_peptide` 按 peptide、`unseen_tcr` 按 TCR、`unseen_pair` 按 peptide–TCR pair 分组。Group-macro 指标对组等权；Group-macro recall 仅在含正负标签的组上计算。`unseen_pair` 中每个 pair 组只有一个标签，因此该 split 的 Group-macro recall 未定义。",
        "",
        "| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, item in metrics_by_split.items():
        metrics = item["metrics"]
        lines.append(
            f"| `{split_name}` | {item['group_by']} | {fmt(metrics['group_count'])} | "
            f"{fmt(metrics['group_macro_accuracy'])} | {fmt(metrics['group_macro_recall'])} | "
            f"{fmt(metrics['group_coverage'])} | {fmt(metrics['group_macro_nll'])} | "
            f"{fmt(metrics['worst_group_nll'])} | {fmt(metrics['tn'])} | "
            f"{fmt(metrics['fp'])} | {fmt(metrics['fn'])} | {fmt(metrics['tp'])} |"
        )
    lines += [
        "",
        "## 数据清单与完整性检查",
        "",
        "| CSV 文件 | Split | Samples | 正 / 负 | 唯一 peptide | 唯一 TCR | 重复 pair | 标签冲突 | 长度过滤 | SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name, item in metrics_by_split.items():
        values = item["data"]
        path = Path(values["path"])
        relative = path.relative_to(_ROOT).as_posix()
        filters = values["filter_stats"]
        lines.append(
            f"| [`{path.name}`](../../../{relative}) | `{split_name}` | "
            f"{fmt(values['raw_rows'])} | {fmt(values['positive'])} / {fmt(values['negative'])} | "
            f"{fmt(values['unique_peptides'])} | {fmt(values['unique_tcrs'])} | "
            f"{fmt(values['duplicate_pair_rows'])} | {fmt(values['conflicting_pair_rows'])} | "
            f"{fmt(filters['peptide_too_short'] + filters['peptide_too_long'])} | "
            f"`{values['sha256']}` |"
        )
    lines += [
        "",
        (
            f"合计 **{data['total_rows']:,}** 行，三个文件间共享 pair **{data['cross_split_pair_duplicates']:,}** 条，"
            f"标签冲突 **{data['cross_split_pair_conflicts']:,}** 条。全目录 peptide/TCR 去重后分别为 "
            f"**{data['unique_peptides']:,} / {data['unique_tcrs']:,}** 条序列。"
        ),
        "",
        "输入标签为二元值，评测序列均为标准氨基酸；报告列出各文件长度过滤数。模型将全部文件的唯一 peptide/TCR 各编码一次，再分别计算每个 split 的 PTCR 指标。",
        "",
        "### 测试实体与 Stage14 训练集重叠",
        "",
        "| 测试 split | peptide 重叠 / 未出现 | TCR 重叠 / 未出现 |",
        "|---|---:|---:|",
    ]
    for split_name, overlaps in data["training_entity_overlap"].items():
        peptide_overlap = overlaps["peptide"]
        tcr_overlap = overlaps["tcr"]
        if peptide_overlap.get("available") and tcr_overlap.get("available"):
            lines.append(
                f"| `{split_name}` | "
                f"{fmt(peptide_overlap['overlapping_test_peptides'])} / "
                f"{fmt(peptide_overlap['test_peptides_not_seen_in_train'])} | "
                f"{fmt(tcr_overlap['overlapping_test_tcrs'])} / "
                f"{fmt(tcr_overlap['test_tcrs_not_seen_in_train'])} |"
            )
        else:
            lines.append(f"| `{split_name}` | — | — |")
    unseen_peptide_overlap = data["training_entity_overlap"].get("test_unseen_peptide", {})
    unseen_tcr_overlap = data["training_entity_overlap"].get("test_unseen_tcr", {})
    if unseen_peptide_overlap.get("peptide", {}).get("available"):
        lines += [
            "",
            (
                f"`test_unseen_peptide` 的 {unseen_peptide_overlap['peptide']['test_peptides']:,} 个测试 peptide "
                "在所检查的 PH/PT 二分类训练文件与四类 PHT 训练文件中均未出现。"
            ),
        ]
    if unseen_tcr_overlap.get("tcr", {}).get("available"):
        lines += [
            "",
            (
                f"`test_unseen_tcr` 的 {unseen_tcr_overlap['tcr']['test_tcrs']:,} 个测试 TCR 中，"
                f"{unseen_tcr_overlap['tcr']['overlapping_test_tcrs']:,} 个在 PT/PHT 训练文件中出现，"
                f"{unseen_tcr_overlap['tcr']['test_tcrs_not_seen_in_train']:,} 个未出现。"
            ),
        ]
    lines += [
        "",
        "## 结果解读",
        "",
        (
            f"按 split 等权宏平均的 `performance_avg` 为 {macro['performance_avg']:.4f}，"
            f"按样本合并 pooled 的 `performance_avg` 为 {pooled['performance_avg']:.4f}。"
            "两者差异反映各 split 样本规模和类别构成不同；各 split 的明细应一并参考。"
        ),
        "",
        (
            f"`test_unseen_peptide` 的 `performance_avg` 最低（{metrics_by_split['test_unseen_peptide']['metrics']['performance_avg']:.4f}），"
            f"其按样本 Recall 为 {metrics_by_split['test_unseen_peptide']['metrics']['recall']:.3f}，"
            f"按 peptide 等权的 Group-macro recall 为 {metrics_by_split['test_unseen_peptide']['metrics']['group_macro_recall']:.3f}，"
            "显示正类检出率会随未见 peptide 组而变化。"
        ),
        "",
        (
            f"Pooled Recall={pooled['recall']:.3f}、Specificity={pooled['specificity']:.3f}，"
            f"假阴性 {pooled['fn']:,}、假阳性 {pooled['fp']:,}。如应用侧需要提高正类召回，"
            "应在独立验证集上选择阈值，再用本测试结果报告固定阈值表现。"
        ),
        "",
        "## 评测设置与 Checkpoint",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 评测时间 | {report['evaluation_time']} |",
        "| 任务 | PTCR：peptide–TCR 二分类 |",
        f"| 测试目录 | [`dataset/PT/new`](../../../dataset/PT/new) |",
        f"| 设备 / 混合精度 | `{report['device']}` / `{report['mixed_precision']}` |",
        f"| 编码 batch / pair batch | {report['encode_batch_size']} / {report['row_batch_size']} |",
        f"| Checkpoint 条件 / 类型 | `{checkpoint['condition']}` / `{checkpoint['checkpoint_type']}` |",
        f"| Checkpoint | [`{Path(checkpoint['path']).name}`](../../../{Path(checkpoint['path']).relative_to(_ROOT).as_posix()}) |",
        f"| Stage14 best round / global step | {checkpoint['best_round']} / {checkpoint['global_step']} |",
        f"| Checkpoint 验证 joint score | {fmt(checkpoint['validation_joint_score'])} |",
        f"| Checkpoint SHA-256 | `{checkpoint['sha256']}` |",
        "",
        "Checkpoint 的 validation joint score 是训练验证集上的选模分数，与本报告 PT new 的测试指标来自不同数据集，不应直接比较。",
        "",
        "## 产物与复现",
        "",
        f"- JSON 全量结果：[`{output_json.name}`]({output_json.name})",
        "- 评估入口：[`evaluate_plm_stage14_pt_new.py`](../../../source/evaluate_plm_stage14_pt_new.py)",
        "",
        "从项目根目录执行：",
        "",
        "```bash",
        "PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
        "  source/evaluate_plm_stage14_pt_new.py \\",
        f"  --checkpoint {checkpoint['path']} \\",
        f"  --data-dir {data['directory']} \\",
        f"  --output-json {output_json} \\",
        f"  --output-md {output_md} \\",
        f"  --device {report['device']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def add_confusion_counts(metrics: dict[str, Any], labels: np.ndarray, scores: np.ndarray) -> None:
    predictions = scores > 0.5
    metrics.update(
        {
            "positive_samples": int(labels.sum()),
            "negative_samples": int(len(labels) - labels.sum()),
            "positive_rate": float(labels.mean()),
            "threshold": 0.5,
            "threshold_comparison": "score > 0.5",
            "tp": int(np.sum(predictions & (labels == 1))),
            "tn": int(np.sum(~predictions & (labels == 0))),
            "fp": int(np.sum(predictions & (labels == 0))),
            "fn": int(np.sum(~predictions & (labels == 1))),
        }
    )


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = apply_stage14_overrides(
        load_stage14_config(args.config), device=args.device, num_workers=0
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = load_stage14_checkpoint(checkpoint_path, artifact_role="best")
    source_sha = checkpoint["source_stage1"]["sha256"]
    if source_sha not in STAGE14_APPROVED_STAGE1_SHA256S:
        raise ValueError("Checkpoint references an unapproved Stage-1 source")
    # Restore the checkpoint's approved Stage-1 hash in memory, then require the
    # complete Stage-14 model/training contract to match the current config.
    config.paths.stage1_sha256 = source_sha
    validate_stage14_for_evaluation(checkpoint, config=config)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    data_paths = sorted(data_dir.glob("*.csv"))
    if not data_paths:
        raise FileNotFoundError(f"No CSV test datasets found in {data_dir}")
    device = resolve_device(config.runtime.device)

    audits: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[tuple[str, str, int]]] = {}
    for data_path in data_paths:
        split_name = data_path.stem
        audit = audit_data(data_path, config, args.limit)
        dataset = OnlinePairDataset(
            data_path,
            "ptcr",
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            limit=args.limit,
        )
        if len(dataset) != audit["rows"]:
            raise ValueError(
                f"{data_path} row count changed after peptide length filtering: "
                f"{audit['rows']} audited, {len(dataset)} retained"
            )
        audit.update(
            {
                "path": str(data_path),
                "sha256": sha256_file(data_path),
                "raw_rows": audit["rows"],
                "retained_rows": len(dataset),
                "filter_stats": dict(dataset.filter_stats),
                "positive_rate": audit["positive"] / audit["rows"],
                "training_entity_overlap": {
                    "peptide": audit_training_peptide_overlap(
                        config, {row[0] for row in dataset.rows}
                    ),
                    "tcr": audit_training_tcr_overlap(
                        config, {row[1] for row in dataset.rows}
                    ),
                },
            }
        )
        audits[split_name] = audit
        rows_by_split[split_name] = dataset.rows
        print(
            f"[DATA] {split_name}: rows={len(dataset):,} "
            f"peptides={audit['unique_peptides']:,} tcrs={audit['unique_tcrs']:,} "
            f"positive={audit['positive']:,} negative={audit['negative']:,}",
            flush=True,
        )

    pair_owner: dict[tuple[str, str], tuple[str, int]] = {}
    cross_split_pair_duplicates = 0
    cross_split_pair_conflicts = 0
    for split_name, rows in rows_by_split.items():
        for peptide, tcr, label in rows:
            pair = (peptide, tcr)
            if pair in pair_owner:
                cross_split_pair_duplicates += 1
                old_split, old_label = pair_owner[pair]
                cross_split_pair_conflicts += int(old_label != label)
            else:
                pair_owner[pair] = (split_name, label)

    peptides = sorted_unique(
        {row[0] for rows in rows_by_split.values() for row in rows}
    )
    tcrs = sorted_unique({row[1] for rows in rows_by_split.values() for row in rows})
    model = build_progressive_stage14_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        current = current_metadata[entity].get("checkpoint_fingerprint")
        if saved and saved != current:
            raise ValueError(f"Base PLM fingerprint mismatch for {entity}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_evaluation()
    precision = config.stage14.mixed_precision
    peptide_hidden, peptide_mask = encode_unique(
        "peptide", model.peptide_encoder, peptides, config.model.peptide_max_length,
        args.encode_batch_size, device, precision,
    )
    tcr_hidden, tcr_mask = encode_unique(
        "TCR", model.tcr_encoder, tcrs, config.model.tcr_max_length,
        args.encode_batch_size, device, precision,
    )

    split_results: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for split_name, rows in rows_by_split.items():
        if "unseen_peptide" in split_name:
            group_by = "peptide"
        elif "unseen_tcr" in split_name:
            group_by = "tcr"
        else:
            group_by = "pair"
        metrics, labels, scores = evaluate_rows(
            model,
            rows,
            peptides,
            tcrs,
            peptide_hidden,
            peptide_mask,
            tcr_hidden,
            tcr_mask,
            device=device,
            precision=precision,
            batch_size=args.row_batch_size,
            group_by=group_by,
        )
        all_labels.append(labels)
        all_scores.append(scores)
        split_results[split_name] = {
            "group_by": group_by,
            "data": audits[split_name],
            "metrics": metrics,
        }

    pooled_labels = np.concatenate(all_labels)
    pooled_scores = np.concatenate(all_scores)
    clipped = np.clip(pooled_scores, 1e-7, 1.0 - 1e-7)
    pooled_nll = -np.log(np.where(pooled_labels == 1, clipped, 1.0 - clipped))
    pooled_metrics = compute_binary_metrics(
        pooled_labels,
        pooled_scores,
        loss=float(np.mean(pooled_nll)),
        threshold=0.5,
        strict_threshold=True,
    ).to_dict()
    add_confusion_counts(pooled_metrics, pooled_labels, pooled_scores)

    macro_fields = (
        "loss", "auroc", "aupr", "accuracy", "mcc", "f1", "precision",
        "recall", "specificity", "nll", "brier", "ece", "performance_avg",
    )
    macro_average = {
        field: float(np.nanmean([item["metrics"][field] for item in split_results.values()]))
        for field in macro_fields
    }
    total_rows = int(len(pooled_labels))
    total_positive = int(pooled_labels.sum())
    unique_peptides_all = set(peptides)
    unique_tcrs_all = set(tcrs)
    training_overlaps = {
        split_name: audit["training_entity_overlap"]
        for split_name, audit in audits.items()
    }
    data_info = {
        "directory": str(data_dir),
        "files": [str(path) for path in data_paths],
        "split_count": len(data_paths),
        "total_rows": total_rows,
        "total_positive": total_positive,
        "total_negative": total_rows - total_positive,
        "positive_rate": total_positive / total_rows,
        "unique_peptides": len(unique_peptides_all),
        "unique_tcrs": len(unique_tcrs_all),
        "cross_split_pair_duplicates": cross_split_pair_duplicates,
        "cross_split_pair_conflicts": cross_split_pair_conflicts,
        "training_entity_overlap": training_overlaps,
    }
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "ptcr",
        "device": str(device),
        "mixed_precision": precision,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "checkpoint": {
            "path": str(checkpoint_path),
            "condition": checkpoint["condition"],
            "checkpoint_type": checkpoint["checkpoint_type"],
            "artifact_role": checkpoint["artifact_role"],
            "best_round": checkpoint["training_state"].get("stage14b_round"),
            "global_step": checkpoint["training_state"].get("global_step"),
            "validation_joint_score": checkpoint["metrics"].get("validation_joint_score"),
            "sha256": checkpoint_sha256(checkpoint_path),
            "stage1_sha256": source_sha,
        },
        "config": str(args.config.expanduser().resolve()),
        "data": data_info,
        "splits": split_results,
        "macro_average": macro_average,
        "pooled_metrics": pooled_metrics,
        "outputs": {
            "json": str(args.output_json.expanduser().resolve()),
            "markdown": str(args.output_md.expanduser().resolve()),
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(markdown(report, output_md, output_json), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
