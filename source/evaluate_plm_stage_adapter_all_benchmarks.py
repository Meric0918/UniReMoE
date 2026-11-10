#!/usr/bin/env python3
"""Evaluate a Stage Adapter checkpoint on all PH/PT/PHT benchmark tests.

The Stage Adapter keeps the Stage-1 PH/PT experts frozen and trains an
independent PHT branch.  Pair benchmarks are therefore evaluated with the
frozen PH/PT heads, while every PHT benchmark is evaluated with all three
heads.  Unique residue representations are encoded once and reused for all
row-level predictions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from evaluate_plm_stage8_benchmarks import (
    _autocast,
    _feature_dtype,
    encode_unique,
    evaluate_pair,
    finish_metrics,
    load_benchmark_data,
    log,
    sha256_file,
)
from plm_unified.stage_adapter_checkpoint import load_stage_adapter_checkpoint, restore_stage_adapter_model
from plm_unified.stage_adapter_config import (
    apply_stage_adapter_overrides,
    load_stage_adapter_config,
)
from plm_unified.stage_adapter_model import build_progressive_stage_adapter_model
from plm_unified.trainer import resolve_device


_PHT_HEADS = ("phla", "ptcr", "pht")
_PHT_LABEL_ATTRIBUTES = {"phla": "ph", "ptcr": "pt", "pht": "pht"}
_METRIC_NAMES = (
    "auroc",
    "aupr",
    "accuracy",
    "mcc",
    "f1",
    "performance_avg",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/plm_stage_adapter.yaml"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ph-data-dir",
        type=Path,
        help="Directory containing PH test_*.csv files; defaults to dataset/PH/benchmarks.",
    )
    parser.add_argument(
        "--pt-data-dir",
        type=Path,
        help="Directory containing PT test_*.csv files; defaults to dataset/PT/benchmarks.",
    )
    parser.add_argument(
        "--pht-data-dir",
        type=Path,
        help="Directory containing PHT test_*.csv files; defaults to dataset/PHT/benchmarks.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument(
        "--limit", type=int, help="Debug only: cap rows per benchmark file."
    )
    return parser.parse_args()


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
    mixed_precision: str,
    row_batch_size: int,
    pht_threshold: float,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    feature_dtype = _feature_dtype(device, mixed_precision)
    labels: dict[str, list[np.ndarray]] = {head: [] for head in _PHT_HEADS}
    probabilities: dict[str, list[np.ndarray]] = {
        head: [] for head in _PHT_HEADS
    }
    loss_sums = {head: 0.0 for head in _PHT_HEADS}
    samples = 0
    total_batches = (len(records) + row_batch_size - 1) // row_batch_size
    started = time.time()

    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(records), row_batch_size), start=1
        ):
            batch_records = records[start : start + row_batch_size]
            p_locations = np.fromiter(
                (peptide_index[row.peptide] for row in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            h_locations = np.fromiter(
                (hla_index[row.hla] for row in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            t_locations = np.fromiter(
                (tcr_index[row.tcr] for row in batch_records),
                dtype=np.int64,
                count=len(batch_records),
            )
            peptide = torch.from_numpy(peptide_hidden[p_locations]).to(
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            peptide_mask_batch = torch.from_numpy(peptide_mask[p_locations]).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            hla = torch.from_numpy(hla_hidden[h_locations]).to(
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            hla_mask_batch = torch.from_numpy(hla_mask[h_locations]).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            tcr = torch.from_numpy(tcr_hidden[t_locations]).to(
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            tcr_mask_batch = torch.from_numpy(tcr_mask[t_locations]).to(
                device, dtype=torch.bool, non_blocking=device.type == "cuda"
            )
            batch_labels_cpu = {
                head: np.fromiter(
                    (
                        getattr(row, _PHT_LABEL_ATTRIBUTES[head])
                        for row in batch_records
                    ),
                    dtype=np.int64,
                    count=len(batch_records),
                )
                for head in _PHT_HEADS
            }
            batch_labels = {
                head: torch.from_numpy(values).to(
                    device,
                    dtype=torch.long,
                    non_blocking=device.type == "cuda",
                )
                for head, values in batch_labels_cpu.items()
            }
            with _autocast(device, mixed_precision):
                phla = model.binding_model(
                    "phla", peptide, peptide_mask_batch, hla, hla_mask_batch
                )
                ptcr = model.binding_model(
                    "ptcr", peptide, peptide_mask_batch, tcr, tcr_mask_batch
                )
                pht = model.binding_model.forward_pht(
                    peptide,
                    peptide_mask_batch,
                    hla,
                    hla_mask_batch,
                    tcr,
                    tcr_mask_batch,
                )
                logits = {
                    "phla": phla.logits,
                    "ptcr": ptcr.logits,
                    "pht": pht.logits,
                }
                losses = {
                    head: criterion(logits[head], batch_labels[head])
                    for head in _PHT_HEADS
                }

            batch_samples = len(batch_records)
            samples += batch_samples
            for head in _PHT_HEADS:
                loss_sums[head] += (
                    float(losses[head].detach().cpu()) * batch_samples
                )
                probabilities[head].append(
                    torch.softmax(logits[head], dim=1)[:, 1]
                    .float()
                    .cpu()
                    .numpy()
                )
                labels[head].append(batch_labels_cpu[head])
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.time() - started
                log(
                    f"[EVAL PHT] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(records):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s"
                )

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for head in _PHT_HEADS:
        metrics[head] = {
            "fixed_0.5": finish_metrics(
                labels[head], probabilities[head], loss_sums[head], samples, 0.5
            )
        }
        if head == "pht":
            metrics[head]["checkpoint_selected"] = finish_metrics(
                labels[head],
                probabilities[head],
                loss_sums[head],
                samples,
                pht_threshold,
            )
    return {"metrics": metrics}


def fixed_rows(report: Mapping[str, Any]):
    rows = []
    for key, item in sorted(report["results"].items()):
        if item["family"] == "PHT":
            for head in _PHT_HEADS:
                rows.append((key, head, item["metrics"][head]["fixed_0.5"]))
        else:
            rows.append((key, item["head"], item["metrics"]["fixed_0.5"]))
    return rows


def macro_rows(report: Mapping[str, Any]):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, head, metrics in fixed_rows(report):
        family = key.split("/", 1)[0]
        grouped.setdefault((family, head), []).append(metrics)
    return {
        group: {
            metric: float(np.mean([float(row[metric]) for row in values]))
            for metric in _METRIC_NAMES
        }
        for group, values in grouped.items()
    }


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def relative_link(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def make_markdown(report: Mapping[str, Any], output_path: Path) -> str:
    checkpoint = Path(report["checkpoint"])
    config = Path(report["config"])
    json_path = Path(report["outputs"]["json"])
    results = report["results"]
    rows = fixed_rows(report)
    macros = macro_rows(report)
    primary_groups = (("PH", "phla"), ("PT", "ptcr"), ("PHT", "pht"))
    primary_score = float(
        np.mean([macros[group]["performance_avg"] for group in primary_groups])
    )
    pht_score = float(macros[("PHT", "pht")]["performance_avg"])
    if report.get("initialization", "checkpoint") == "checkpoint":
        summary_score = (
            "三项主任务的 benchmark 宏平均 performance average 再等权平均为 "
            f"**{primary_score:.6f}**。"
        )
    else:
        summary_score = (
            "已训练的 PHT head 在 PHT benchmark 上的宏平均 performance average 为 "
            f"**{pht_score:.6f}**。"
            "PH/PT 头未接受对应任务监督，因此不与 PHT 分数合并。"
        )
    lines = [
        "# Stage Adapter PH/PT/PHT benchmark test 评测结果",
        "",
        "## 1. 结论摘要",
        "",
        (
            "指定的 Stage Adapter best checkpoint 已在本次指定数据目录中的全部 "
            "`test_*.csv` 上完成评测，"
            f"共计 **{report['totals']['samples']:,}** 条有效测试样本。"
        ),
        "",
        summary_score,
        "",
        (
            "- PH/PT 独立预测头从随机权重初始化，未接受 PH/PT 训练；其结果仅供观察。"
            if report.get("initialization", "checkpoint") != "checkpoint"
            else "- PH/PT 使用源检查点中已训练并冻结的 `phla` / `ptcr` head。"
        ),
        "- PHT 主结果使用训练后的 Stage Adapter `pht` head；同时补充冻结 `phla`/`ptcr` head 在 PHT 行上的投影结果。",
        "- 主表统一使用固定阈值 `0.5`；验证集选择的 PHT 阈值只作为补充，未使用 test 标签重新选阈值。",
        "",
        "## 2. 模型与评测设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Checkpoint | [`{checkpoint.name}`]({relative_link(checkpoint, output_path.parent)}) |",
        f"| Checkpoint SHA-256 | `{report['checkpoint_sha256']}` |",
        f"| 类型 / 角色 | `{report['checkpoint_type']}` / `{report['artifact_role']}` |",
        f"| 初始化模式 | `{report.get('initialization', 'checkpoint')}` |",
        f"| 训练 phase / round | `{report['phase']}` / `{report['training_state'].get('stage_adapter_b_round')}` |",
        f"| Validation PHT performance avg | `{fmt(report['checkpoint_validation']['pht']['performance_avg'])}` |",
        f"| Validation-selected PHT threshold | `{report['checkpoint_selected_threshold']}` |",
        f"| 配置 | [`{config.name}`]({relative_link(config, output_path.parent)}) |",
        f"| PH 数据目录 | `{report['data_dirs']['PH']}` |",
        f"| PT 数据目录 | `{report['data_dirs']['PT']}` |",
        f"| PHT 数据目录 | `{report['data_dirs']['PHT']}` |",
        f"| 设备 / 精度 | `{report['device']}` / `{report['mixed_precision']}` autocast |",
        f"| unique-sequence / row batch | `{report['encode_batch_size']}` / `{report['row_batch_size']}` |",
        "",
        "## 3. 数据审计",
        "",
        "| 数据集 | Source CSV | 原始行数 | 有效行数 | 过滤/去重 | PH 正/负 | PT 正/负 | PHT 正/负 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(results.items()):
        if item["family"] == "PHT":
            counts = item["label_counts"]
            ph = f"{counts['phla']['positive']:,}/{counts['phla']['negative']:,}"
            pt = f"{counts['ptcr']['positive']:,}/{counts['ptcr']['negative']:,}"
            pht = f"{counts['pht']['positive']:,}/{counts['pht']['negative']:,}"
        else:
            metric = item["metrics"]["fixed_0.5"]
            counts = f"{metric['positive_samples']:,}/{metric['negative_samples']:,}"
            ph = counts if item["family"] == "PH" else "—"
            pt = counts if item["family"] == "PT" else "—"
            pht = "—"
        lines.append(
            f"| `{key}` | [`{Path(item['path']).name}`]({relative_link(Path(item['path']), output_path.parent)}) | "
            f"{item['raw_rows']:,} | {item['samples']:,} | "
            f"{item['raw_rows'] - item['samples']:,} | {ph} | {pt} | {pht} |"
        )
    lines.extend(
        [
            f"| **合计** | — | **{report['totals']['raw_rows']:,}** | **{report['totals']['samples']:,}** | **{report['totals']['filtered']:,}** | — | — | — |",
            "",
            (
                "测试集包含 "
                f"**{report['unique_sequences']['peptide']:,}** 个唯一 peptide、"
                f"**{report['unique_sequences']['hla']:,}** 个唯一 HLA 和 "
                f"**{report['unique_sequences']['tcr']:,}** 个唯一 TCR。"
            ),
            "",
            "## 4. 主结果：固定阈值 0.5",
            "",
            "| 数据集 | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity | Loss | Performance avg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, head, metrics in rows:
        lines.append(
            f"| `{key}` | `{head}` | {metrics['samples']:,} | "
            f"{fmt(metrics['auroc'])} | {fmt(metrics['aupr'])} | "
            f"{fmt(metrics['accuracy'])} | {fmt(metrics['mcc'])} | "
            f"{fmt(metrics['f1'])} | {fmt(metrics['precision'])} | "
            f"{fmt(metrics['recall'])} | {fmt(metrics['specificity'])} | "
            f"{fmt(metrics['loss'])} | {fmt(metrics['performance_avg'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. benchmark 宏平均",
            "",
            "宏平均为同一数据族/head 下各 benchmark 的简单算术平均，不按样本量加权。",
            "",
            "| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (family, head), metrics in sorted(macros.items()):
        lines.append(
            f"| `{family}` / `{head}` | {fmt(metrics['auroc'])} | "
            f"{fmt(metrics['aupr'])} | {fmt(metrics['accuracy'])} | "
            f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
            f"{fmt(metrics['performance_avg'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. PHT 验证阈值补充",
            "",
            f"Checkpoint 在 validation 上选择的 PHT 阈值为 `{report['checkpoint_selected_threshold']}`。该阈值原样用于下表，没有通过 test 标签调整。",
            "",
            "| Benchmark | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in sorted(results.items()):
        if item["family"] != "PHT":
            continue
        for name, label in (
            ("fixed_0.5", "0.5"),
            ("checkpoint_selected", str(report["checkpoint_selected_threshold"])),
        ):
            metrics = item["metrics"]["pht"][name]
            lines.append(
                f"| `{key}` | `{label}` | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
                f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
                f"{fmt(metrics['specificity'])} |"
            )
    primary = {group: macros[group] for group in primary_groups}
    best_pht = max(
        (
            (key, item["metrics"]["pht"]["fixed_0.5"])
            for key, item in results.items()
            if item["family"] == "PHT"
        ),
        key=lambda row: float(row[1]["performance_avg"]),
    )
    lines.extend(
        [
            "",
            "## 7. 结果解读与复核",
            "",
            f"- PH 主任务宏平均 performance average：`{primary[('PH', 'phla')]['performance_avg']:.6f}`。",
            f"- PT 主任务宏平均 performance average：`{primary[('PT', 'ptcr')]['performance_avg']:.6f}`。",
            f"- PHT 主任务宏平均 performance average：`{primary[('PHT', 'pht')]['performance_avg']:.6f}`。",
            f"- PHT head 表现最高的单个 split 是 `{best_pht[0]}`，performance average 为 `{best_pht[1]['performance_avg']:.6f}`。",
            "- `test_unseen_pair` 等小样本 split 的指标方差可能较大，应结合样本数、正负比例和混淆矩阵解读。",
            "- 本次只执行推理；没有重新训练、修改 checkpoint，也没有使用测试标签做阈值或模型选择。",
            "- 每个唯一 peptide/HLA/TCR 只编码一次，随后复用 residue-level 表示完成全部行级推理。",
            f"- 原始 JSON：[`{json_path.name}`]({relative_link(json_path, output_path.parent)})",
            f"- 本报告：[`{output_path.name}`]({relative_link(output_path, output_path.parent)})",
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
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = apply_stage_adapter_overrides(
        load_stage_adapter_config(config_path), device=args.device, num_workers=0
    )
    config.paths.data_root = root / "dataset"
    checkpoint = load_stage_adapter_checkpoint(checkpoint_path)
    if checkpoint.get("artifact_role") not in {"best", "last"}:
        raise ValueError("Benchmark evaluation requires a best or last checkpoint")
    saved_config = checkpoint.get("config") or {}
    if saved_config.get("model") != config.to_dict().get("model"):
        raise ValueError("Checkpoint model configuration mismatch")
    selected_threshold = float(checkpoint["metrics"]["selected_pht_threshold"])
    if not 0 <= selected_threshold <= 1:
        raise ValueError("Checkpoint-selected PHT threshold is outside [0, 1]")

    device = resolve_device(config.runtime.device)
    mixed_precision = config.stage_adapter.mixed_precision
    log(f"[INIT] checkpoint={checkpoint_path}")
    log(
        f"[INIT] device={device}"
        + (
            f" gpu={torch.cuda.get_device_name(device)}"
            if device.type == "cuda"
            else ""
        )
    )
    benchmarks, peptides, hlas, tcrs = load_benchmark_data(
        root,
        config,
        args.limit,
        ph_data_dir=args.ph_data_dir,
        pt_data_dir=args.pt_data_dir,
        pht_data_dir=args.pht_data_dir,
    )
    data_dirs = {
        "PH": str((args.ph_data_dir or root / "dataset" / "PH" / "benchmarks").resolve()),
        "PT": str((args.pt_data_dir or root / "dataset" / "PT" / "benchmarks").resolve()),
        "PHT": str((args.pht_data_dir or root / "dataset" / "PHT" / "benchmarks").resolve()),
    }
    model = build_progressive_stage_adapter_model(config, device)
    restore_stage_adapter_model(checkpoint, model)
    del checkpoint["model"]
    model.set_evaluation()
    log("[INIT] model loaded and checkpoint contract verified")

    peptide_hidden, peptide_mask = encode_unique(
        "peptide",
        model.peptide_encoder,
        peptides,
        config.model.peptide_max_length,
        args.encode_batch_size,
        device,
        mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    hla_hidden, hla_mask = encode_unique(
        "HLA",
        model.hla_encoder,
        hlas,
        config.model.hla_max_length,
        args.encode_batch_size,
        device,
        mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tcr_hidden, tcr_mask = encode_unique(
        "TCR",
        model.tcr_encoder,
        tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
        mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    peptide_index = {value: index for index, value in enumerate(peptides)}
    hla_index = {value: index for index, value in enumerate(hlas)}
    tcr_index = {value: index for index, value in enumerate(tcrs)}

    results: dict[str, dict[str, Any]] = {}
    for key, item in sorted(benchmarks.items()):
        if item["family"] == "PHT":
            continue
        if item["task"] == "phla":
            receptor_index, receptor_hidden, receptor_mask = (
                hla_index,
                hla_hidden,
                hla_mask,
            )
        else:
            receptor_index, receptor_hidden, receptor_mask = (
                tcr_index,
                tcr_hidden,
                tcr_mask,
            )
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
            mixed_precision,
            args.row_batch_size,
            0.5,
        )
        results[key] = {
            "family": item["family"],
            "benchmark": item["benchmark"],
            "path": str(item["path"].resolve()),
            "head": item["head"],
            "raw_rows": item["raw_rows"],
            "samples": len(item["rows"]),
            "filter_stats": item["filter_stats"],
            "metrics": {"fixed_0.5": metrics},
        }
        log(f"[DONE] {key} {json.dumps(metrics, sort_keys=True)}")

    for key, item in sorted(benchmarks.items()):
        if item["family"] != "PHT":
            continue
        evaluated = evaluate_pht(
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
            mixed_precision,
            args.row_batch_size,
            selected_threshold,
        )
        label_counts = {}
        for head, attribute in _PHT_LABEL_ATTRIBUTES.items():
            positives = sum(
                int(getattr(record, attribute)) for record in item["records"]
            )
            label_counts[head] = {
                "positive": positives,
                "negative": len(item["records"]) - positives,
            }
        results[key] = {
            "family": "PHT",
            "benchmark": item["benchmark"],
            "path": str(item["path"].resolve()),
            "head": ",".join(_PHT_HEADS),
            "raw_rows": item["raw_rows"],
            "samples": len(item["records"]),
            "filter_stats": item["filter_stats"],
            "label_counts": label_counts,
            "metrics": evaluated["metrics"],
        }
        log(
            f"[DONE] {key} pht_performance_avg="
            f"{evaluated['metrics']['pht']['fixed_0.5']['performance_avg']:.6f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del model
    del peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    totals = {
        "raw_rows": sum(int(item["raw_rows"]) for item in results.values()),
        "samples": sum(int(item["samples"]) for item in results.values()),
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
        "initialization": config.stage_adapter.initialization,
        "format_version": checkpoint.get("format_version"),
        "artifact_role": checkpoint.get("artifact_role"),
        "phase": checkpoint.get("phase"),
        "config": str(config_path),
        "project_root": str(root),
        "data_dirs": data_dirs,
        "device": str(device),
        "mixed_precision": mixed_precision,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "fixed_threshold": 0.5,
        "checkpoint_selected_threshold": selected_threshold,
        "checkpoint_validation": checkpoint.get("metrics"),
        "training_state": checkpoint.get("training_state", {}),
        "effective_data": checkpoint.get("effective_data"),
        "source_stage1": checkpoint.get("source_stage1"),
        "source_warmup": checkpoint.get("source_warmup"),
        "frozen_binding_checksum": checkpoint.get("frozen_binding_checksum"),
        "results": results,
        "totals": totals,
        "unique_sequences": {
            "peptide": len(peptides),
            "hla": len(hlas),
            "tcr": len(tcrs),
        },
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_markdown),
        },
        "notes": [
            (
                "PH/PT heads are randomly initialized and have not received PH/PT supervision."
                if config.stage_adapter.initialization != "checkpoint"
                else "PH/PT pair benchmarks use the trained frozen experts stored in the source checkpoint."
            ),
            "PHT benchmarks use the trained Stage Adapter PHT head; frozen PH/PT projections are supplementary.",
            "The main table uses threshold 0.5, with the validation-selected PHT threshold reported separately.",
            "No test-set threshold selection or training was performed.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            json_safe(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(
        make_markdown(report, output_markdown) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_markdown),
                "samples": totals["samples"],
                "checkpoint_selected_threshold": selected_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
