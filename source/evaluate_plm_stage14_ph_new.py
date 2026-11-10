#!/usr/bin/env python3
"""Evaluate a Stage-14 checkpoint on every CSV split in dataset/PH/new."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "source") not in sys.path:
    sys.path.insert(0, str(_ROOT / "source"))

from evaluate_plm_stage14_pt_new import (
    audit_training_peptide_overlap,
    autocast,
    encode_unique,
    feature_dtype,
    fmt,
    sha256_file,
    sorted_unique,
)
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
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "dataset" / "PH" / "new")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def audit_training_hla_overlap(config, test_hlas: set[str]) -> dict[str, Any]:
    data_root = config.paths.data_root
    paths = [data_root / "PH" / "train.csv"]
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
            if "HLA" not in (reader.fieldnames or []):
                return {"available": False, "invalid_source": str(path), "sources": sources}
            for row in reader:
                value = "".join((row.get("HLA") or "").split()).replace("-", "").upper()
                if value:
                    sequences.add(value)
        overlap = test_hlas & sequences
        relative = path.resolve().relative_to(_ROOT).as_posix()
        sources[relative] = {
            "unique_train_hlas": len(sequences),
            "overlapping_test_hlas": len(overlap),
        }
        training_union.update(sequences)
    overlap_union = test_hlas & training_union
    return {
        "available": True,
        "unique_train_hlas_union": len(training_union),
        "test_hlas": len(test_hlas),
        "overlapping_test_hlas": len(overlap_union),
        "test_hlas_not_seen_in_train": len(test_hlas - training_union),
        "sources": sources,
    }


def audit_data(path: Path, config, limit: int | None) -> dict[str, Any]:
    counts = {"rows": 0, "positive": 0, "negative": 0}
    split_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    peptide_lengths: Counter[int] = Counter()
    hla_lengths: Counter[int] = Counter()
    peptides: set[str] = set()
    hlas: set[str] = set()
    seen_pairs: dict[tuple[str, str], int] = {}
    duplicate_rows = 0
    conflicting_pairs = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "HLA", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and counts["rows"] >= limit:
                break
            peptide = "".join((row.get("peptide") or "").split()).replace("-", "").upper()
            hla = "".join((row.get("HLA") or "").split()).replace("-", "").upper()
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be binary at {path}:{row_number}")
            if not peptide or not hla:
                raise ValueError(f"Empty peptide/HLA at {path}:{row_number}")
            if not set(peptide).issubset(_AMINO_ACIDS) or not set(hla).issubset(_AMINO_ACIDS):
                raise ValueError(f"Non-canonical sequence at {path}:{row_number}")
            if len(hla) > config.model.hla_max_length:
                raise ValueError(f"HLA exceeds max length at {path}:{row_number}")
            pair = (peptide, hla)
            if pair in seen_pairs:
                duplicate_rows += 1
                conflicting_pairs += int(seen_pairs[pair] != label)
            else:
                seen_pairs[pair] = label
            counts["rows"] += 1
            counts["positive" if label else "negative"] += 1
            peptides.add(peptide)
            hlas.add(hla)
            split_types[(row.get("split_type") or "").strip()] += 1
            sources[(row.get("source") or "").strip()] += 1
            peptide_lengths[len(peptide)] += 1
            hla_lengths[len(hla)] += 1
    return {
        **counts,
        "split_types": dict(split_types),
        "sources": dict(sources),
        "peptide_length_min": min(peptide_lengths) if peptide_lengths else None,
        "peptide_length_max": max(peptide_lengths) if peptide_lengths else None,
        "hla_length_min": min(hla_lengths) if hla_lengths else None,
        "hla_length_max": max(hla_lengths) if hla_lengths else None,
        "unique_peptides": len(peptides),
        "unique_hlas": len(hlas),
        "duplicate_pair_rows": duplicate_rows,
        "conflicting_pair_rows": conflicting_pairs,
    }


def evaluate_rows(
    model,
    rows: list[tuple[str, str, int]],
    peptides: list[str],
    hlas: list[str],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    hla_hidden: np.ndarray,
    hla_mask: np.ndarray,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    group_by: str,
):
    peptide_index = {sequence: index for index, sequence in enumerate(peptides)}
    hla_index = {sequence: index for index, sequence in enumerate(hlas)}
    peptide_locations = np.fromiter(
        (peptide_index[row[0]] for row in rows), dtype=np.int64, count=len(rows)
    )
    hla_locations = np.fromiter(
        (hla_index[row[1]] for row in rows), dtype=np.int64, count=len(rows)
    )
    labels = np.fromiter((row[2] for row in rows), dtype=np.int64, count=len(rows))
    if group_by == "peptide":
        group_sequences = (row[0] for row in rows)
    elif group_by == "hla":
        group_sequences = (row[1] for row in rows)
    elif group_by == "pair":
        group_sequences = (f"{row[0]}\x1f{row[1]}" for row in rows)
    else:
        raise ValueError(f"Unsupported group_by value: {group_by}")
    groups = np.fromiter(
        (stable_group_id(value) for value in group_sequences), dtype=np.int64, count=len(rows)
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
            h_locs = hla_locations[start:end]
            batch_labels = torch.from_numpy(labels[start:end]).to(
                device, dtype=torch.long, non_blocking=device.type == "cuda"
            )
            peptide = torch.from_numpy(peptide_hidden[p_locs]).to(
                device, dtype=row_dtype, non_blocking=device.type == "cuda"
            )
            p_mask = torch.from_numpy(peptide_mask[p_locs].astype(bool, copy=False)).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            hla = torch.from_numpy(hla_hidden[h_locs]).to(
                device, dtype=row_dtype, non_blocking=device.type == "cuda"
            )
            h_mask = torch.from_numpy(hla_mask[h_locs].astype(bool, copy=False)).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            with autocast(device, precision):
                output = model.binding_model("phla", peptide, p_mask, hla, h_mask)
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
                    f"[EVAL PHLA] {batch_id:,}/{total_batches:,} "
                    f"samples={samples:,}/{len(rows):,} rate={samples / elapsed:,.1f}/s",
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


def markdown(report: dict[str, Any], output_md: Path, output_json: Path) -> str:
    checkpoint = report["checkpoint"]
    data = report["data"]
    splits = report["splits"]
    macro = report["macro_average"]
    pooled = report["pooled_metrics"]
    best = max(splits, key=lambda name: splits[name]["metrics"]["performance_avg"])
    weakest = min(splits, key=lambda name: splits[name]["metrics"]["performance_avg"])
    lines = [
        "# PLM-Stage14 baseline：PH new 全部测试集总结",
        "",
        "## 结论",
        "",
        (
            f"使用 Stage14 baseline 第 **{checkpoint['best_round']}** 轮最佳 checkpoint，"
            f"对 `dataset/PH/new` 下全部 **{data['split_count']}** 个 CSV 测试集共 "
            f"**{data['total_rows']:,}** 条 PHLA 样本完成评估。"
        ),
        "",
        "| 汇总 | 样本总数 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | `performance_avg` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 三个 split 等权宏平均 | {fmt(data['total_rows'])} | {fmt(macro['auroc'])} | "
            f"{fmt(macro['aupr'])} | {fmt(macro['accuracy'])} | {fmt(macro['mcc'])} | "
            f"{fmt(macro['f1'])} | {fmt(macro['precision'])} | {fmt(macro['recall'])} | "
            f"{fmt(macro['specificity'])} | {fmt(macro['performance_avg'])} |"
        ),
        (
            f"| 全部样本 pooled | {fmt(pooled['samples'])} | {fmt(pooled['auroc'])} | "
            f"{fmt(pooled['aupr'])} | {fmt(pooled['accuracy'])} | {fmt(pooled['mcc'])} | "
            f"{fmt(pooled['f1'])} | {fmt(pooled['precision'])} | {fmt(pooled['recall'])} | "
            f"{fmt(pooled['specificity'])} | {fmt(pooled['performance_avg'])} |"
        ),
        "",
        (
            f"正类 / 负类合计为 **{data['total_positive']:,} / {data['total_negative']:,}**；"
            f"`{best}` 的 `performance_avg` 最高（{splits[best]['metrics']['performance_avg']:.4f}），"
            f"`{weakest}` 最低（{splits[weakest]['metrics']['performance_avg']:.4f}）。"
        ),
        "",
        "`performance_avg` 为 AUROC、Accuracy、MCC、F1、AUPR 五项指标的算术平均。宏平均对三个 split 等权；pooled 指标将所有样本合并后重新计算。分类指标使用固定阈值 `score > 0.5`，未基于测试标签调阈值。",
        "",
        "## 各测试集结果",
        "",
        "| Split | Samples | 正 / 负 | 正类比例 | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | `performance_avg` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in splits.items():
        m = item["metrics"]
        d = item["data"]
        lines.append(
            f"| `{name}` | {fmt(m['samples'])} | {fmt(d['positive'])} / {fmt(d['negative'])} | "
            f"{d['positive_rate']:.2%} | {fmt(m['auroc'])} | {fmt(m['aupr'])} | "
            f"{fmt(m['accuracy'])} | {fmt(m['mcc'])} | {fmt(m['f1'])} | "
            f"{fmt(m['recall'])} | {fmt(m['specificity'])} | {fmt(m['performance_avg'])} |"
        )
    lines += [
        "",
        "### 分组鲁棒性与混淆矩阵",
        "",
        "`unseen_peptide` 按 peptide 分组，`unseen_hla` 按 HLA 分组，`unseen_pair` 按 peptide–HLA pair 分组。Group-macro recall 仅在同时含正负标签的组上计算；unseen pair 的单 pair 组只有一个标签，recall 未定义。",
        "",
        "| Split | 分组对象 | 组数 | Group-macro accuracy | Group-macro recall | Recall coverage | Group-macro NLL | Worst-group NLL | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in splits.items():
        m = item["metrics"]
        lines.append(
            f"| `{name}` | {item['group_by']} | {fmt(m['group_count'])} | "
            f"{fmt(m['group_macro_accuracy'])} | {fmt(m['group_macro_recall'])} | "
            f"{fmt(m['group_coverage'])} | {fmt(m['group_macro_nll'])} | "
            f"{fmt(m['worst_group_nll'])} | {fmt(m['tn'])} | {fmt(m['fp'])} | "
            f"{fmt(m['fn'])} | {fmt(m['tp'])} |"
        )
    lines += [
        "",
        "## 数据清单与完整性检查",
        "",
        "| CSV 文件 | Split | Samples | 正 / 负 | 唯一 peptide | 唯一 HLA | 重复 pair | 标签冲突 | 长度过滤 | SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in splits.items():
        d = item["data"]
        path = Path(d["path"])
        relative = path.relative_to(_ROOT).as_posix()
        filters = d["filter_stats"]
        lines.append(
            f"| [`{path.name}`](../../../{relative}) | `{name}` | {fmt(d['raw_rows'])} | "
            f"{fmt(d['positive'])} / {fmt(d['negative'])} | {fmt(d['unique_peptides'])} | "
            f"{fmt(d['unique_hlas'])} | {fmt(d['duplicate_pair_rows'])} | "
            f"{fmt(d['conflicting_pair_rows'])} | "
            f"{fmt(filters['peptide_too_short'] + filters['peptide_too_long'])} | `{d['sha256']}` |"
        )
    lines += [
        "",
        (
            f"合计 **{data['total_rows']:,}** 行，文件间共享 pair **{data['cross_split_pair_duplicates']:,}** 条，"
            f"标签冲突 **{data['cross_split_pair_conflicts']:,}** 条；去重后共有 **{data['unique_peptides']:,}** 个 peptide、"
            f"**{data['unique_hlas']:,}** 个 HLA。"
        ),
        "",
        "所有纳入样本的 peptide/HLA 为标准氨基酸序列；报告列出各文件长度过滤数。评估时对全部文件的唯一 peptide/HLA 各编码一次，再计算 PHLA pair 分数。",
        "",
        "### 测试实体与 Stage14 训练集重叠",
        "",
        "| 测试 split | peptide 重叠 / 未出现 | HLA 重叠 / 未出现 |",
        "|---|---:|---:|",
    ]
    for name, overlaps in data["training_entity_overlap"].items():
        p = overlaps["peptide"]
        h = overlaps["hla"]
        if p.get("available") and h.get("available"):
            lines.append(
                f"| `{name}` | {fmt(p['overlapping_test_peptides'])} / {fmt(p['test_peptides_not_seen_in_train'])} | "
                f"{fmt(h['overlapping_test_hlas'])} / {fmt(h['test_hlas_not_seen_in_train'])} |"
            )
        else:
            lines.append(f"| `{name}` | — | — |")
    hla_overlap = data["training_entity_overlap"].get("test_unseen_hla", {}).get("hla", {})
    if hla_overlap.get("available"):
        lines += [
            "",
            (
                f"`test_unseen_hla` 的 {hla_overlap['test_hlas']:,} 个 HLA 中，"
                f"{hla_overlap['overlapping_test_hlas']:,} 个出现在 PH/PHT 训练文件，"
                f"{hla_overlap['test_hlas_not_seen_in_train']:,} 个未出现。"
            ),
        ]
    lines += [
        "",
        "## 结果解读",
        "",
        (
            f"等权宏平均 `performance_avg` 为 {macro['performance_avg']:.4f}，pooled 为 "
            f"{pooled['performance_avg']:.4f}。`test_unseen_pair` 仅 {splits['test_unseen_pair']['metrics']['samples']:,} 条，"
            "其指标受样本量限制，宜结合另外两个大样本 split 判断泛化表现。"
        ),
        "",
        (
            f"Pooled Recall={pooled['recall']:.3f}、Specificity={pooled['specificity']:.3f}，"
            f"假阴性 {pooled['fn']:,}、假阳性 {pooled['fp']:,}。若应用需要提高正类召回，应在独立验证集选择阈值。"
        ),
        "",
        "## 评测设置与 Checkpoint",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 评测时间 | {report['evaluation_time']} |",
        "| 任务 | PHLA：peptide–HLA 二分类 |",
        f"| 测试目录 | [`dataset/PH/new`](../../../dataset/PH/new) |",
        f"| 设备 / 混合精度 | `{report['device']}` / `{report['mixed_precision']}` |",
        f"| 编码 batch / pair batch | {report['encode_batch_size']} / {report['row_batch_size']} |",
        f"| Checkpoint 条件 / 类型 | `{checkpoint['condition']}` / `{checkpoint['checkpoint_type']}` |",
        f"| Checkpoint | [`{Path(checkpoint['path']).name}`](../../../{Path(checkpoint['path']).relative_to(_ROOT).as_posix()}) |",
        f"| Stage14 best round / global step | {checkpoint['best_round']} / {checkpoint['global_step']} |",
        f"| Checkpoint 验证 joint score | {fmt(checkpoint['validation_joint_score'])} |",
        f"| Checkpoint SHA-256 | `{checkpoint['sha256']}` |",
        "",
        "验证 joint score 是训练验证集选模分数，与本报告 PH new 测试指标来自不同数据集，不应直接比较。",
        "",
        "## 产物与复现",
        "",
        f"- JSON 全量结果：[`{output_json.name}`]({output_json.name})",
        "- 评估入口：[`evaluate_plm_stage14_ph_new.py`](../../../source/evaluate_plm_stage14_ph_new.py)",
        "",
        "从项目根目录执行：",
        "",
        "```bash",
        "PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
        "  source/evaluate_plm_stage14_ph_new.py \\",
        f"  --checkpoint {checkpoint['path']} \\",
        f"  --data-dir {data['directory']} \\",
        f"  --output-json {output_json} \\",
        f"  --output-md {output_md} \\",
        f"  --device {report['device']}",
        "```",
        "",
    ]
    return "\n".join(lines)


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
    config.paths.stage1_sha256 = source_sha
    validate_stage14_for_evaluation(checkpoint, config=config)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    data_paths = sorted(data_dir.glob("*.csv"))
    if not data_paths:
        raise FileNotFoundError(f"No CSV datasets found in {data_dir}")
    device = resolve_device(config.runtime.device)

    audits: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[tuple[str, str, int]]] = {}
    for data_path in data_paths:
        split_name = data_path.stem
        audit = audit_data(data_path, config, args.limit)
        dataset = OnlinePairDataset(
            data_path,
            "phla",
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            limit=args.limit,
        )
        if len(dataset) != audit["rows"]:
            raise ValueError(
                f"{data_path} row count changed after peptide length filtering: "
                f"{audit['rows']} audited, {len(dataset)} retained"
            )
        peptides_in_file = {row[0] for row in dataset.rows}
        hlas_in_file = {row[1] for row in dataset.rows}
        audit.update(
            {
                "path": str(data_path),
                "sha256": sha256_file(data_path),
                "raw_rows": audit["rows"],
                "retained_rows": len(dataset),
                "filter_stats": dict(dataset.filter_stats),
                "positive_rate": audit["positive"] / audit["rows"],
                "training_entity_overlap": {
                    "peptide": audit_training_peptide_overlap(config, peptides_in_file),
                    "hla": audit_training_hla_overlap(config, hlas_in_file),
                },
            }
        )
        audits[split_name] = audit
        rows_by_split[split_name] = dataset.rows
        print(
            f"[DATA] {split_name}: rows={len(dataset):,} "
            f"peptides={audit['unique_peptides']:,} hlas={audit['unique_hlas']:,} "
            f"positive={audit['positive']:,} negative={audit['negative']:,}",
            flush=True,
        )

    pair_owner: dict[tuple[str, str], tuple[str, int]] = {}
    cross_split_pair_duplicates = 0
    cross_split_pair_conflicts = 0
    for split_name, rows in rows_by_split.items():
        for peptide, hla, label in rows:
            pair = (peptide, hla)
            if pair in pair_owner:
                cross_split_pair_duplicates += 1
                _, old_label = pair_owner[pair]
                cross_split_pair_conflicts += int(old_label != label)
            else:
                pair_owner[pair] = (split_name, label)

    peptides = sorted_unique({row[0] for rows in rows_by_split.values() for row in rows})
    hlas = sorted_unique({row[1] for rows in rows_by_split.values() for row in rows})
    model = build_progressive_stage14_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla"):
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
    hla_hidden, hla_mask = encode_unique(
        "HLA", model.hla_encoder, hlas, config.model.hla_max_length,
        args.encode_batch_size, device, precision,
    )

    split_results: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for split_name, rows in rows_by_split.items():
        if "unseen_peptide" in split_name:
            group_by = "peptide"
        elif "unseen_hla" in split_name:
            group_by = "hla"
        else:
            group_by = "pair"
        metrics, labels, scores = evaluate_rows(
            model,
            rows,
            peptides,
            hlas,
            peptide_hidden,
            peptide_mask,
            hla_hidden,
            hla_mask,
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
    predictions = pooled_scores > 0.5
    pooled_metrics.update(
        {
            "positive_samples": int(pooled_labels.sum()),
            "negative_samples": int(len(pooled_labels) - pooled_labels.sum()),
            "positive_rate": float(pooled_labels.mean()),
            "threshold": 0.5,
            "threshold_comparison": "score > 0.5",
            "tp": int(np.sum(predictions & (pooled_labels == 1))),
            "tn": int(np.sum(~predictions & (pooled_labels == 0))),
            "fp": int(np.sum(predictions & (pooled_labels == 0))),
            "fn": int(np.sum(~predictions & (pooled_labels == 1))),
        }
    )
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
    data_info = {
        "directory": str(data_dir),
        "files": [str(path) for path in data_paths],
        "split_count": len(data_paths),
        "total_rows": total_rows,
        "total_positive": total_positive,
        "total_negative": total_rows - total_positive,
        "positive_rate": total_positive / total_rows,
        "unique_peptides": len(peptides),
        "unique_hlas": len(hlas),
        "cross_split_pair_duplicates": cross_split_pair_duplicates,
        "cross_split_pair_conflicts": cross_split_pair_conflicts,
        "training_entity_overlap": {
            name: audit["training_entity_overlap"] for name, audit in audits.items()
        },
    }
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "phla",
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
