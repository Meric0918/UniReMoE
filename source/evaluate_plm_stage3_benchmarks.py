#!/usr/bin/env python3
"""Evaluate a Stage-3 joint checkpoint on PH/PT/PHT benchmark test files.

PH and PT benchmark files are evaluated with the corresponding Stage-3 pair
head. PHT benchmark files are evaluated with all three Stage-3 heads. The
script encodes each unique peptide, HLA, and TCR once, then reuses the
residue-level representations for row-level inference.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from plm_unified.config import load_config
from plm_unified.data import OnlinePairDataset
from plm_unified.metrics import compute_binary_metrics
from plm_unified.trainer import load_training_checkpoint, resolve_device
from plm_unified.triplet_data import PHTMultitaskDataset
from plm_unified.triplet_model import (
    build_progressive_triplet_model,
    load_triplet_model_state,
)


_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_PAIR_SPECS = (
    ("PH", "phla", "phla"),
    ("PT", "ptcr", "ptcr"),
)
_PHT_HEADS = ("phla", "ptcr", "triplet")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "plm_triplet.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap rows per benchmark file after validation.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def encode_unique(
    label: str,
    encoder,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.empty(
        (len(sequences), max_length, encoder.hidden_size),
        dtype=np.float16,
    )
    mask = np.empty((len(sequences), max_length), dtype=np.uint8)
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(sequences), batch_size),
            start=1,
        ):
            end = min(start + batch_size, len(sequences))
            with _autocast(device):
                encoded = encoder(sequences[start:end])
            hidden[start:end] = (
                encoded.hidden.detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
            mask[start:end] = (
                encoded.mask.detach()
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                log(
                    f"[ENCODE {label}] {batch_id:,}/{total_batches:,} "
                    f"sequences={end:,}/{len(sequences):,} "
                    f"rate={end / max(elapsed, 1e-9):.1f}/s"
                )
    return hidden, mask


def finish_metrics(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
    loss_sum: float,
    samples: int,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_score = np.concatenate(probabilities).astype(np.float64, copy=False)
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=loss_sum / samples,
        threshold=threshold,
        strict_threshold=True,
    ).to_dict()
    y_pred = y_score > threshold
    metrics.update(
        {
            "positive_samples": int(y_true.sum()),
            "negative_samples": int((y_true == 0).sum()),
            "positive_rate": float(y_true.mean()),
            "tp": int(np.sum(y_pred & (y_true == 1))),
            "tn": int(np.sum(~y_pred & (y_true == 0))),
            "fp": int(np.sum(y_pred & (y_true == 0))),
            "fn": int(np.sum(~y_pred & (y_true == 1))),
            "threshold": float(threshold),
        }
    )
    return metrics


def evaluate_pair(
    model,
    rows: list[tuple[str, str, int]],
    peptide_index: dict[str, int],
    receptor_index: dict[str, int],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    receptor_hidden: np.ndarray,
    receptor_mask: np.ndarray,
    task: str,
    device: torch.device,
    row_batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    criterion = nn.CrossEntropyLoss()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    total_batches = (len(rows) + row_batch_size - 1) // row_batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(rows), row_batch_size),
            start=1,
        ):
            batch_rows = rows[start : start + row_batch_size]
            p_locations = np.fromiter(
                (peptide_index[row[0]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            r_locations = np.fromiter(
                (receptor_index[row[1]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            batch_labels_cpu = np.fromiter(
                (row[2] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            peptide = torch.from_numpy(peptide_hidden[p_locations]).to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[p_locations]
            ).to(device, dtype=torch.bool, non_blocking=True)
            receptor = torch.from_numpy(receptor_hidden[r_locations]).to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            receptor_mask_batch = torch.from_numpy(
                receptor_mask[r_locations]
            ).to(device, dtype=torch.bool, non_blocking=True)
            batch_labels = torch.from_numpy(batch_labels_cpu).to(
                device, dtype=torch.long, non_blocking=True
            )
            with _autocast(device):
                output = model.binding_model(
                    task,
                    peptide,
                    peptide_mask_batch,
                    receptor,
                    receptor_mask_batch,
                )
                loss = criterion(output.logits, batch_labels)
            batch_samples = len(batch_rows)
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch_labels_cpu)
            if batch_id == 1 or batch_id % 50 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                log(
                    f"[EVAL {task}] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(rows):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s"
                )
    return finish_metrics(labels, probabilities, loss_sum, samples, threshold)


def evaluate_pht(
    model,
    records,
    peptide_index: dict[str, int],
    hla_index: dict[str, int],
    tcr_index: dict[str, int],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    hla_hidden: np.ndarray,
    hla_mask: np.ndarray,
    tcr_hidden: np.ndarray,
    tcr_mask: np.ndarray,
    device: torch.device,
    row_batch_size: int,
    thresholds: dict[str, float],
) -> dict[str, dict[str, dict[str, float | int]]]:
    criterion = nn.CrossEntropyLoss()
    labels = {task: [] for task in _PHT_HEADS}
    probabilities = {task: [] for task in _PHT_HEADS}
    loss_sums = {task: 0.0 for task in _PHT_HEADS}
    samples = 0
    total_batches = (len(records) + row_batch_size - 1) // row_batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(records), row_batch_size),
            start=1,
        ):
            batch_records = records[start : start + row_batch_size]
            p_locations = np.fromiter(
                (peptide_index[record.peptide] for record in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            h_locations = np.fromiter(
                (hla_index[record.hla] for record in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            t_locations = np.fromiter(
                (tcr_index[record.tcr] for record in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            peptide = torch.from_numpy(peptide_hidden[p_locations]).to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[p_locations]
            ).to(device, dtype=torch.bool, non_blocking=True)
            hla = torch.from_numpy(hla_hidden[h_locations]).to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            hla_mask_batch = torch.from_numpy(hla_mask[h_locations]).to(
                device, dtype=torch.bool, non_blocking=True
            )
            tcr = torch.from_numpy(tcr_hidden[t_locations]).to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            tcr_mask_batch = torch.from_numpy(tcr_mask[t_locations]).to(
                device, dtype=torch.bool, non_blocking=True
            )
            batch_labels = {
                "phla": torch.tensor(
                    [record.ph for record in batch_records],
                    dtype=torch.long,
                    device=device,
                ),
                "ptcr": torch.tensor(
                    [record.pt for record in batch_records],
                    dtype=torch.long,
                    device=device,
                ),
                "triplet": torch.tensor(
                    [record.pht for record in batch_records],
                    dtype=torch.long,
                    device=device,
                ),
            }
            with _autocast(device):
                output = model.binding_model.forward_all(
                    peptide,
                    peptide_mask_batch,
                    hla,
                    hla_mask_batch,
                    tcr,
                    tcr_mask_batch,
                )
                logits = {
                    "phla": output.phla_logits,
                    "ptcr": output.ptcr_logits,
                    "triplet": output.pht_logits,
                }
                losses = {
                    task: criterion(logits[task], batch_labels[task])
                    for task in _PHT_HEADS
                }
            batch_samples = len(batch_records)
            samples += batch_samples
            for task in _PHT_HEADS:
                loss_sums[task] += (
                    float(losses[task].detach().cpu()) * batch_samples
                )
                probabilities[task].append(
                    torch.softmax(logits[task], dim=1)[:, 1]
                    .float()
                    .cpu()
                    .numpy()
                )
                labels[task].append(batch_labels[task].cpu().numpy())
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                log(
                    f"[EVAL PHT] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(records):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s"
                )
    return {
        task: {
            threshold_name: finish_metrics(
                labels[task],
                probabilities[task],
                loss_sums[task],
                samples,
                threshold,
            )
            for threshold_name, threshold in thresholds.items()
        }
        for task in _PHT_HEADS
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_link(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def make_markdown(report: dict, markdown_path: Path) -> str:
    checkpoint = Path(report["checkpoint"])
    config = Path(report["config"])
    json_path = Path(report["outputs"]["json"])
    results = report["results"]
    fixed_rows = []
    for key, item in results.items():
        if item["family"] == "PHT":
            for head in _PHT_HEADS:
                fixed_rows.append(
                    (key, head, item["metrics"][head]["fixed_0.5"])
                )
        else:
            fixed_rows.append(
                (key, item["head"], item["metrics"]["fixed_0.5"])
            )

    lines = [
        "# Stage 3 `stage3_best_joint` 在 PH/PT/PHT benchmark test 上的评测结果",
        "",
        "## 1. 结论摘要",
        "",
        (
            "本次使用指定的 Stage 3 joint checkpoint，对 "
            "`dataset/PH/benchmarks`、`dataset/PT/benchmarks` 和 "
            "`dataset/PHT/benchmarks` 下全部 `test_*.csv` 完成测试。"
            f"共评测 **{report['totals']['samples']:,}** 条有效样本；"
            "所有数据行均通过当前模型的 peptide、HLA、TCR 长度及 "
            "canonical 残基约束。"
        ),
        "",
        "- PH benchmark 使用 Stage 3 的 pHLA head；PT benchmark 使用 pTCR head。",
        "- PHT benchmark 同时报告 pHLA、pTCR 和 PHT 三个 head，标签分别对应 CSV 中的 `PH`、`PT`、`PHT` 列。",
        "- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集上为 PHT head 选择的阈值另作补充比较，没有在任何 test 数据上重新选择阈值。",
        "",
        "## 2. 评测对象与设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Checkpoint | [`{checkpoint.name}`]({relative_link(checkpoint, markdown_path.parent)}) |",
        f"| Checkpoint 类型 | `{report['checkpoint_type']}` |",
        f"| Checkpoint SHA-256 | `{report['checkpoint_sha256']}` |",
        f"| Checkpoint best round | `{report.get('best_round')}`；best selection score `{report.get('best_round_score')}` |",
        f"| 配置 | [`{config.name}`]({relative_link(config, markdown_path.parent)}) |",
        f"| 设备 | `{report['device']}` |",
        f"| 精度 | `{report['mixed_precision']}` autocast |",
        f"| unique-sequence 编码 batch | `{report['encode_batch_size']}` |",
        f"| 行级推理 batch | `{report['row_batch_size']}` |",
        f"| 固定分类阈值 | `{report['fixed_threshold']}` |",
        f"| checkpoint PHT 验证阈值 | `{report['checkpoint_selected_threshold']}` |",
        "",
        "PH/PT 的 pair 数据没有三元输入，因此仅调用对应的 Stage 3 pair head；PHT 数据则使用完整三元输入并一次前向得到三个 head 的输出。",
        "",
        "## 3. 数据审计",
        "",
        "| 数据集 | 原始行数 | 纳入评测 | 过滤/去重 | PH 正/负 | PT 正/负 | PHT 正/负 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(results.items()):
        raw_rows = item["raw_rows"]
        samples = item["samples"]
        if item["family"] == "PHT":
            counts = item["label_counts"]
            ph = f"{counts['phla']['positive']:,}/{counts['phla']['negative']:,}"
            pt = f"{counts['ptcr']['positive']:,}/{counts['ptcr']['negative']:,}"
            pht = f"{counts['triplet']['positive']:,}/{counts['triplet']['negative']:,}"
        else:
            metrics = item["metrics"]["fixed_0.5"]
            counts = f"{metrics['positive_samples']:,}/{metrics['negative_samples']:,}"
            ph = counts if item["head"] == "phla" else "—"
            pt = counts if item["head"] == "ptcr" else "—"
            pht = "—"
        lines.append(
            f"| `{key}` | {raw_rows:,} | {samples:,} | "
            f"{raw_rows - samples:,} | {ph} | {pt} | {pht} |"
        )
    lines.extend(
        [
            f"| **合计** | **{report['totals']['raw_rows']:,}** | **{report['totals']['samples']:,}** | **{report['totals']['filtered']:,}** | — | — | — |",
            "",
            (
                "测试集合并包含 "
                f"**{report['unique_sequences']['peptide']:,}** 个唯一 peptide、"
                f"**{report['unique_sequences']['hla']:,}** 个唯一 HLA 和 "
                f"**{report['unique_sequences']['tcr']:,}** 个唯一 TCR。"
            ),
            "",
            "## 4. 主结果：固定阈值 0.5",
            "",
            "AUROC/AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 `0.5`。",
            "",
            "| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, head, metrics in sorted(fixed_rows):
        lines.append(
            f"| `{key}` | `{head}` | {metrics['samples']:,} | "
            f"{fmt(metrics['auroc'])} | {fmt(metrics['aupr'])} | "
            f"{fmt(metrics['accuracy'])} | {fmt(metrics['mcc'])} | "
            f"{fmt(metrics['f1'])} | {fmt(metrics['precision'])} | "
            f"{fmt(metrics['recall'])} | {fmt(metrics['specificity'])} | "
            f"{fmt(metrics['loss'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 各数据族的 benchmark 宏平均",
            "",
            "以下为同一数据族内各 benchmark 的简单算术平均，不是将样本拼接后重新计算的 pooled metric。",
            "",
            "| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    macro_groups: dict[tuple[str, str], list[dict]] = {}
    for key, head, metrics in fixed_rows:
        family = key.split("/", 1)[0]
        macro_groups.setdefault((family, head), []).append(metrics)
    for (family, head), values in sorted(macro_groups.items()):
        averages = {
            name: sum(float(value[name]) for value in values) / len(values)
            for name in ("auroc", "aupr", "accuracy", "mcc", "f1")
        }
        lines.append(
            f"| `{family}` / `{head}` | {fmt(averages['auroc'])} | "
            f"{fmt(averages['aupr'])} | {fmt(averages['accuracy'])} | "
            f"{fmt(averages['mcc'])} | {fmt(averages['f1'])} |"
        )

    lines.extend(
        [
            "",
            "## 6. PHT head 的 checkpoint 阈值补充",
            "",
            (
                "checkpoint 在验证集上为 PHT head 记录的阈值为 "
                f"`{report['checkpoint_selected_threshold']}`。下表将它与 "
                "固定阈值 `0.5` 对比；该阈值只在验证集上确定，未使用任何 "
                "test 标签重新调节。"
            ),
            "",
            "| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in sorted(results.items()):
        if item["family"] != "PHT":
            continue
        for threshold_name, label in (
            ("fixed_0.5", "0.5"),
            ("checkpoint_selected", str(report["checkpoint_selected_threshold"])),
        ):
            metrics = item["metrics"]["triplet"][threshold_name]
            lines.append(
                f"| `{key}` / `PHT` | `{label}` | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
                f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
                f"{fmt(metrics['specificity'])} |"
            )

    best = max(fixed_rows, key=lambda row: row[2]["auroc"])
    pht_rows = [row for row in fixed_rows if row[1] == "triplet"]
    pht_best = max(pht_rows, key=lambda row: row[2]["auroc"])
    lines.extend(
        [
            "",
            "## 7. 结果说明",
            "",
            (
                f"- 主表中 AUROC 最高的是 `{best[0]}` 的 `{best[1]}` head："
                f"`{best[2]['auroc']:.6f}`；对应 AUPR 为 "
                f"`{best[2]['aupr']:.6f}`。"
            ),
            (
                f"- PHT benchmark 的 PHT head 中，AUROC 最高的是 `{pht_best[0]}`："
                f"`{pht_best[2]['auroc']:.6f}`；PHT 正类比例差异会影响 "
                "AUPR 与阈值指标的直接比较。"
            ),
            "- `test_unseen_pair` 等小样本 benchmark 的指标方差较大，应结合样本数和混淆矩阵谨慎解读。",
            "",
            "## 8. 评测方法与可复核产物",
            "",
            "- 使用项目现有 Stage 3 progressive model、checkpoint 加载、encoder 和 `compute_binary_metrics` 实现；没有重新训练、修改 checkpoint 或使用 test 数据调参。",
            "- 对全部测试文件的唯一 peptide/HLA/TCR 分别编码一次，再复用 residue-level 表示完成行级推理；没有使用 Stage 1 embedding cache。",
            "- PH/PT pair 的标签来自各自 CSV 的 `label`；PHT 三元测试的三个标签来自 `PH`、`PT`、`PHT` 列。",
            f"- 原始 JSON：[`{json_path.name}`]({relative_link(json_path, markdown_path.parent)})",
            f"- 本报告：[`{markdown_path.name}`]({relative_link(markdown_path, markdown_path.parent)})",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output_json = args.output_json.resolve()
    output_markdown = args.output_markdown.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = load_config(config_path)
    config.paths.data_root = root / "dataset"
    config.training.device = args.device
    config.validate(require_models=True)
    device = resolve_device(args.device)

    log(f"[INIT] checkpoint={checkpoint_path}")
    log(
        f"[INIT] device={device} "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}"
    )
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") not in {
        "stage3_joint_model",
        "stage3_training",
        "stage3_multitask_model",
    }:
        raise ValueError(
            f"Unexpected checkpoint type: {checkpoint.get('checkpoint_type')!r}"
        )
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            f"Unexpected checkpoint format: {checkpoint.get('format_version')!r}"
        )
    if checkpoint.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("Checkpoint model config does not match evaluation config")
    checkpoint_selected_threshold = float(
        checkpoint.get("selected_threshold", 0.5)
    )
    if not 0 <= checkpoint_selected_threshold <= 1:
        raise ValueError("Checkpoint selected threshold is outside [0, 1]")

    pair_data: dict[str, dict] = {}
    pht_data: dict[str, dict] = {}
    for family, task, head in _PAIR_SPECS:
        paths = sorted((root / "dataset" / family / "benchmarks").glob("test_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No benchmark test files found for {family}")
        for path in paths:
            key = f"{family}/{path.stem}"
            dataset = OnlinePairDataset(
                path,
                task,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=args.limit,
            )
            rows = dataset.rows
            receptor_limit = (
                config.model.hla_max_length
                if task == "phla"
                else config.model.tcr_max_length
            )
            invalid = [
                row
                for row in rows
                if len(row[1]) > receptor_limit
                or not set(row[0]).issubset(_CANONICAL_AMINO_ACIDS)
                or not set(row[1]).issubset(_CANONICAL_AMINO_ACIDS)
            ]
            if invalid:
                raise ValueError(
                    f"{key} has {len(invalid)} rows outside receptor/canonical constraints"
                )
            pair_data[key] = {
                "path": path,
                "task": task,
                "head": head,
                "rows": rows,
                "filter_stats": dict(dataset.filter_stats),
            }
            log(
                f"[DATA] {key} rows={len(rows):,} "
                f"filter_stats={dataset.filter_stats}"
            )

    pht_paths = sorted((root / "dataset" / "PHT" / "benchmarks").glob("test_*.csv"))
    if not pht_paths:
        raise FileNotFoundError("No PHT benchmark test files found")
    for path in pht_paths:
        key = f"PHT/{path.stem}"
        dataset = PHTMultitaskDataset(
            path,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            reject_noncanonical_sequences=True,
            allowed_label_types=("111", "100", "010", "000"),
        )
        records = dataset.records
        if args.limit is not None:
            records = records[: args.limit]
        pht_data[key] = {
            "path": path,
            "dataset": dataset,
            "records": records,
            "filter_stats": dict(dataset.filter_stats),
        }
        log(
            f"[DATA] {key} rows={len(records):,} "
            f"filter_stats={dataset.filter_stats}"
        )

    all_peptides = sorted(
        {
            row[0]
            for item in pair_data.values()
            for row in item["rows"]
        }
        | {
            record.peptide
            for item in pht_data.values()
            for record in item["records"]
        },
        key=lambda value: (-len(value), value),
    )
    all_hlas = sorted(
        {
            row[1]
            for item in pair_data.values()
            if item["task"] == "phla"
            for row in item["rows"]
        }
        | {
            record.hla
            for item in pht_data.values()
            for record in item["records"]
        },
        key=lambda value: (-len(value), value),
    )
    all_tcrs = sorted(
        {
            row[1]
            for item in pair_data.values()
            if item["task"] == "ptcr"
            for row in item["rows"]
        }
        | {
            record.tcr
            for item in pht_data.values()
            for record in item["records"]
        },
        key=lambda value: (-len(value), value),
    )
    for label, values, limit in (
        ("peptide", all_peptides, config.model.peptide_max_length),
        ("HLA", all_hlas, config.model.hla_max_length),
        ("TCR", all_tcrs, config.model.tcr_max_length),
    ):
        if any(len(value) > limit for value in values):
            raise ValueError(f"{label} exceeds configured length limit")
    log(
        f"[UNIQUE] peptide={len(all_peptides):,} "
        f"HLA={len(all_hlas):,} TCR={len(all_tcrs):,}"
    )

    model = build_progressive_triplet_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get(
            "checkpoint_fingerprint"
        )
        current_fingerprint = current_metadata[entity].get(
            "checkpoint_fingerprint"
        )
        if saved_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(
                f"Checkpoint base-PLM fingerprint mismatch for {entity}"
            )
    load_triplet_model_state(model, checkpoint["model"])
    model.set_evaluation()
    log("[INIT] model loaded and checkpoint state verified")

    peptide_hidden, peptide_mask = encode_unique(
        "peptide",
        model.peptide_encoder,
        all_peptides,
        config.model.peptide_max_length,
        args.encode_batch_size,
        device,
    )
    hla_hidden, hla_mask = encode_unique(
        "HLA",
        model.hla_encoder,
        all_hlas,
        config.model.hla_max_length,
        args.encode_batch_size,
        device,
    )
    tcr_hidden, tcr_mask = encode_unique(
        "TCR",
        model.tcr_encoder,
        all_tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
    )
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    hla_index = {value: index for index, value in enumerate(all_hlas)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}

    results: dict[str, dict] = {}
    for key, item in sorted(pair_data.items()):
        if item["task"] == "phla":
            receptor_index = hla_index
            receptor_hidden = hla_hidden
            receptor_mask = hla_mask
        else:
            receptor_index = tcr_index
            receptor_hidden = tcr_hidden
            receptor_mask = tcr_mask
        metrics = evaluate_pair(
            model,
            item["rows"],
            peptide_index,
            receptor_index,
            peptide_hidden,
            peptide_mask,
            receptor_hidden,
            receptor_mask,
            item["task"],
            device,
            args.row_batch_size,
            0.5,
        )
        results[key] = {
            "family": key.split("/", 1)[0],
            "benchmark": key.split("/", 1)[1],
            "path": str(item["path"].resolve()),
            "head": item["head"],
            "raw_rows": item["filter_stats"]["total"],
            "samples": len(item["rows"]),
            "filter_stats": item["filter_stats"],
            "metrics": {"fixed_0.5": metrics},
        }
        log(f"[DONE] {key} {json.dumps(metrics, sort_keys=True)}")

    for key, item in sorted(pht_data.items()):
        threshold_results = evaluate_pht(
            model,
            item["records"],
            peptide_index,
            hla_index,
            tcr_index,
            peptide_hidden,
            peptide_mask,
            hla_hidden,
            hla_mask,
            tcr_hidden,
            tcr_mask,
            device,
            args.row_batch_size,
            {
                "fixed_0.5": 0.5,
                "checkpoint_selected": checkpoint_selected_threshold,
            },
        )
        label_counts = {}
        for task, attribute in (
            ("phla", "ph"),
            ("ptcr", "pt"),
            ("triplet", "pht"),
        ):
            positives = sum(
                int(getattr(record, attribute)) for record in item["records"]
            )
            label_counts[task] = {
                "positive": positives,
                "negative": len(item["records"]) - positives,
            }
        results[key] = {
            "family": "PHT",
            "benchmark": key.split("/", 1)[1],
            "path": str(item["path"].resolve()),
            "head": ",".join(_PHT_HEADS),
            "raw_rows": item["filter_stats"]["total"],
            "samples": len(item["records"]),
            "filter_stats": item["filter_stats"],
            "label_counts": label_counts,
            "metrics": threshold_results,
        }
        log(f"[DONE] {key}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    del model
    gc.collect()

    totals = {
        "raw_rows": sum(item["raw_rows"] for item in results.values()),
        "samples": sum(item["samples"] for item in results.values()),
    }
    totals["filtered"] = totals["raw_rows"] - totals["samples"]
    report = {
        "status": "complete",
        "report_timestamp": datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_%f"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "format_version": 1,
        "config": str(config_path),
        "device": str(device),
        "mixed_precision": config.triplet.mixed_precision,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "fixed_threshold": 0.5,
        "checkpoint_selected_threshold": checkpoint_selected_threshold,
        "best_round": checkpoint.get("best_round"),
        "best_round_score": checkpoint.get(
            "round_score", checkpoint.get("best_joint_score")
        ),
        "stage3b_round": checkpoint.get("stage3b_round"),
        "results": results,
        "totals": totals,
        "unique_sequences": {
            "peptide": len(all_peptides),
            "hla": len(all_hlas),
            "tcr": len(all_tcrs),
        },
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_markdown),
        },
        "notes": [
            "PH benchmark rows were evaluated with the Stage-3 pHLA head.",
            "PT benchmark rows were evaluated with the Stage-3 pTCR head.",
            "PHT benchmark rows were evaluated with all three Stage-3 heads using PH/PT/PHT labels.",
            "Fixed threshold 0.5 was used for the main table. The checkpoint-selected threshold was evaluated only as a PHT-head supplement.",
            "Each unique peptide, HLA, and TCR was encoded once; no Stage-1 embedding cache was used.",
            "No test-set threshold selection or training was performed.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(
        make_markdown(report, output_markdown) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_markdown),
                "samples": totals["samples"],
                "checkpoint_selected_threshold": checkpoint_selected_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
