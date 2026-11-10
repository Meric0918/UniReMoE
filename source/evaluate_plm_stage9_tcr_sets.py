#!/usr/bin/env python3
"""Evaluate the pTCR branch of a Stage-9 checkpoint on external pair sets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from evaluate_plm_triplet_tcr_sets import _encode_unique, _evaluate_rows
from plm_unified.data import OnlinePairDataset
from plm_unified.stage9_checkpoint import (
    load_stage9_deploy_checkpoint,
    load_torch_checkpoint,
)
from plm_unified.stage9_config import (
    load_stage9_config,
    restore_stage9_checkpoint_config,
)
from plm_unified.stage9_model import build_stage9_model
from plm_unified.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/plm_stage9.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data/data_TCR/test_data_set",
    )
    parser.add_argument(
        "--files",
        help="Comma-separated CSV filenames; default is every CSV in --data-dir.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def resolve_files(data_dir: Path, value: str | None) -> list[Path]:
    paths = (
        [data_dir / item.strip() for item in value.split(",") if item.strip()]
        if value
        else sorted(data_dir.glob("*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset file not found: {missing[0]}")
    return paths


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown(
    *,
    report: dict,
    markdown_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    data_dir: Path,
) -> str:
    datasets = report["datasets"]
    checkpoint = report["checkpoint"]
    metric_labels = [
        "Samples",
        "AUROC",
        "AUPR",
        "Accuracy",
        "MCC",
        "F1",
        "Precision",
        "Recall / Sensitivity",
        "Specificity",
        "Loss",
    ]
    lines = [
        "# PLM Stage 9 best 模型在外部 TCR 测试集上的结果",
        "",
        "## 1. 测试概况",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 评测日期 | {report['report_timestamp'][:4]}-{report['report_timestamp'][4:6]}-{report['report_timestamp'][6:8]} |",
        f"| Checkpoint | {checkpoint_path} |",
        f"| Checkpoint 类型 | {checkpoint['checkpoint_type']} |",
        "| 测试分支 | pTCR 二分类分支 |",
        f"| 测试数据目录 | {data_dir} |",
        "| 输入字段 | peptide,tcr,label |",
        "| HLA 输入 | 无 |",
        f"| 分类阈值 | {report['threshold_used']:.1f}（固定，未在测试集调参） |",
        f"| 推理设备 | {report['inference_device']} |",
        "",
        "三个 CSV 文件没有 HLA 列，因此本次不能计算 Stage 9 的 PH-TCR 成对排序、pHLA 或 PHT 三元指标；评估使用 Stage 9 checkpoint 中保留并继续训练的 pTCR 分支。",
        "",
        "## 2. 数据审计",
        "",
        "| 数据集 | 原始行数 | 实际测试样本数 | 正样本 | 负样本 | 正样本比例 | peptide 长度过滤 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in sorted(datasets):
        item = datasets[name]
        stats = item["filter_stats"]
        metrics = item["metrics"]
        filtered = []
        if stats.get("peptide_too_short", 0):
            filtered.append(f"过短 {stats['peptide_too_short']:,}")
        if stats.get("peptide_too_long", 0):
            filtered.append(f"过长 {stats['peptide_too_long']:,}")
        lines.append(
            f"| {name}.csv | {stats['total']:,} | {metrics['samples']:,} | "
            f"{metrics['positive_samples']:,} | {metrics['negative_samples']:,} | "
            f"{metrics['positive_rate']:.4%} | {', '.join(filtered) if filtered else '无'} |"
        )
    counts = report["aggregate_counts"]
    unique = report["unique_sequences"]
    lines += [
        "",
        f"三个文件合计使用 **{counts['samples']:,}** 条样本（正样本 {counts['positive_samples']:,}，负样本 {counts['negative_samples']:,}）。并集包含 **{unique['peptide']:,}** 个唯一 peptide 和 **{unique['tcr']:,}** 个唯一 TCR；最长保留 TCR 为 {unique['max_tcr_length']} aa，未超过配置上限 34 aa。",
        "",
        "## 3. 测试指标",
        "",
        "AUROC 和 AUPR 与分类阈值无关；Accuracy、MCC、F1、Precision、Recall 和 Specificity 使用固定阈值 0.5。",
        "",
        "| 数据集 | " + " | ".join(metric_labels) + " |",
        "|---|" + "---:|" * len(metric_labels),
    ]
    for name in sorted(datasets):
        metrics = datasets[name]["metrics"]
        values = [
            f"{metrics['samples']:,}",
            _fmt(metrics["auroc"]),
            _fmt(metrics["aupr"]),
            _fmt(metrics["accuracy"]),
            _fmt(metrics["mcc"]),
            _fmt(metrics["f1"]),
            _fmt(metrics["precision"]),
            _fmt(metrics["recall"]),
            _fmt(metrics["specificity"]),
            _fmt(metrics["loss"]),
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    ranked = sorted(
        datasets.items(),
        key=lambda item: item[1]["metrics"]["auroc"],
        reverse=True,
    )
    lines += [
        "",
        "## 4. 模型与推理方式",
        "",
        "1. 加载指定的 Stage 9 stage9_best 部署 checkpoint，并校验模型配置、运行时 fold/seed 和 checkpoint 格式。",
        "2. 对三个文件中的唯一 peptide 和唯一 TCR 各编码一次，再复用 residue-level 表示完成全部 pTCR 样本推理；没有使用旧的 Stage 1 embedding cache。",
        f"3. 使用项目配置中的 peptide 长度范围 [8, 15]；TCR 长度上限为 {unique['max_tcr_length']} aa，实际数据未超过配置上限 34 aa。",
        "4. 固定阈值为 0.5。checkpoint 中的 selected_pair_threshold 属于 Stage 9 成对 pair head，不用于本次无 HLA 的 pTCR 外部测试。",
        "",
        "## 5. 结果概览",
        "",
    ]
    if ranked:
        best_name, best_item = ranked[0]
        worst_name, worst_item = ranked[-1]
        lines.append(
            f"按 AUROC 排序，{best_name} 最高（{best_item['metrics']['auroc']:.6f}），"
            f"{worst_name} 最低（{worst_item['metrics']['auroc']:.6f}）。"
        )
    for name in sorted(datasets):
        metrics = datasets[name]["metrics"]
        lines.append(
            f"- {name}：AUROC={metrics['auroc']:.6f}，AUPR={metrics['aupr']:.6f}，"
            f"MCC={metrics['mcc']:.6f}，Recall={metrics['recall']:.6f}，"
            f"Specificity={metrics['specificity']:.6f}。"
        )
    lines += [
        "",
        "这些指标反映的是不同外部数据集上的 pTCR 分支表现，不能据此声称 Stage 9 的成对对比学习目标在这些数据上已被直接验证。若要评价 Stage 9 的核心 pair ranking，需要提供包含 peptide,hla,tcr_positive,tcr_negative 的成对数据，并使用项目的 Stage 9 pair evaluator。",
        "",
        "## 6. 相关文件",
        "",
        f"- Checkpoint：{checkpoint_path}",
        f"- 配置：{config_path}",
        f"- 测试数据：{data_dir}/covid_set.csv、{data_dir}/independent_set.csv、{data_dir}/triple_set_peptide_8_15aa.csv",
        f"- 原始 JSON 结果：{report['json_path']}",
        f"- 本总结：{markdown_path}",
        "",
        "## 7. 复现口径",
        "",
        "本次使用项目的 OnlinePairDataset、Stage 9 模型构建与 checkpoint 加载逻辑，以及现有 pTCR 批量推理逻辑；运行参数为 device=cuda、encode_batch_size=512、pair_batch_size=2048、threshold=0.5。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if args.encode_batch_size < 1 or args.pair_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    started = time.time()
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    config_path = args.config.resolve()
    device = resolve_device(args.device)
    print(
        f"[INIT] checkpoint={checkpoint_path} data_dir={data_dir} "
        f"device={device} threshold={args.threshold}",
        flush=True,
    )

    config = load_stage9_config(config_path)
    checkpoint_metadata = load_torch_checkpoint(checkpoint_path, mmap=False)
    restore_stage9_checkpoint_config(config, checkpoint_metadata.get("config", {}))
    if checkpoint_metadata.get("checkpoint_type") != "stage9_multitask_model":
        raise ValueError(
            "Expected stage9_multitask_model, got "
            f"{checkpoint_metadata.get('checkpoint_type')!r}"
        )

    dataset_rows = {}
    filter_stats = {}
    dataset_paths = resolve_files(data_dir, args.files)
    for path in dataset_paths:
        name = path.stem
        dataset = OnlinePairDataset(
            path,
            "ptcr",
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        dataset_rows[name] = dataset.rows
        filter_stats[name] = dict(dataset.filter_stats)
        labels = np.asarray([row[2] for row in dataset.rows], dtype=np.int64)
        print(
            f"[{name}] rows={len(dataset.rows):,} "
            f"positive={int(labels.sum()):,} "
            f"negative={int((labels == 0).sum()):,} "
            f"filter_stats={dataset.filter_stats}",
            flush=True,
        )

    all_peptides = sorted(
        {row[0] for rows in dataset_rows.values() for row in rows},
        key=lambda value: (-len(value), value),
    )
    all_tcrs = sorted(
        {row[1] for rows in dataset_rows.values() for row in rows},
        key=lambda value: (-len(value), value),
    )
    max_tcr_length = max(map(len, all_tcrs), default=0)
    if max_tcr_length > config.model.tcr_max_length:
        raise ValueError(
            f"TCR length {max_tcr_length} exceeds configured maximum "
            f"{config.model.tcr_max_length}"
        )
    print(
        f"[UNIQUE] peptides={len(all_peptides):,} "
        f"tcrs={len(all_tcrs):,} max_tcr_length={max_tcr_length}",
        flush=True,
    )

    model = build_stage9_model(config, device)
    load_stage9_deploy_checkpoint(model, checkpoint_metadata, config)
    del checkpoint_metadata
    gc.collect()
    model.set_evaluation()
    device_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    )
    print(f"[MODEL] loaded device={device_name}", flush=True)

    peptide_hidden, peptide_mask = _encode_unique(
        "peptide",
        model.peptide_encoder,
        all_peptides,
        config.model.peptide_max_length,
        args.encode_batch_size,
        device,
    )
    tcr_hidden, tcr_mask = _encode_unique(
        "tcr",
        model.tcr_encoder,
        all_tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
    )
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}

    results = {}
    for name in sorted(dataset_rows):
        print(f"[{name}] evaluating", flush=True)
        metrics = _evaluate_rows(
            model,
            dataset_rows[name],
            peptide_index,
            tcr_index,
            peptide_hidden,
            peptide_mask,
            tcr_hidden,
            tcr_mask,
            device=device,
            pair_batch_size=args.pair_batch_size,
            threshold=args.threshold,
        )
        labels = np.asarray([row[2] for row in dataset_rows[name]], dtype=np.int64)
        metrics["positive_samples"] = int(labels.sum())
        metrics["negative_samples"] = int((labels == 0).sum())
        metrics["positive_rate"] = float(labels.mean())
        results[name] = {
            "source_file": str((data_dir / f"{name}.csv").resolve()),
            "filter_stats": filter_stats[name],
            "metrics": metrics,
            "threshold_used": args.threshold,
            "inference": "stage9_checkpoint_ptcr_branch_on_external_peptide_tcr_pairs",
        }
        print(
            f"[{name}] DONE {json.dumps(metrics, sort_keys=True)}",
            flush=True,
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    json_path = (
        args.output.resolve()
        if args.output
        else checkpoint_path.parent
        / f"evaluation_stage9_ptcr_test_data_set_{timestamp}.json"
    )
    markdown_path = (
        args.markdown.resolve()
        if args.markdown
        else Path(__file__).resolve().parents[1]
        / "tests"
        / f"PLM_Stage9_best_TCR_test_results_{timestamp[:8]}.md"
    )
    if markdown_path.exists() and not args.markdown:
        markdown_path = markdown_path.with_name(
            f"PLM_Stage9_best_TCR_test_results_{timestamp}.md"
        )

    counts = {
        "samples": sum(item["metrics"]["samples"] for item in results.values()),
        "positive_samples": sum(
            item["metrics"]["positive_samples"] for item in results.values()
        ),
        "negative_samples": sum(
            item["metrics"]["negative_samples"] for item in results.values()
        ),
    }
    report = {
        "status": "complete",
        "report_timestamp": timestamp,
        "checkpoint": {
            "path": str(checkpoint_path),
            "checkpoint_type": "stage9_multitask_model",
            "format_version": 1,
        },
        "config": str(config_path),
        "data_directory": str(data_dir),
        "branch": "ptcr",
        "input_schema": ["peptide", "tcr", "label"],
        "threshold_used": args.threshold,
        "threshold_source": "fixed_external_test_threshold",
        "inference_device": device_name,
        "batch_sizes": {
            "encode_batch_size": args.encode_batch_size,
            "pair_batch_size": args.pair_batch_size,
        },
        "unique_sequences": {
            "peptide": len(all_peptides),
            "tcr": len(all_tcrs),
            "max_tcr_length": max_tcr_length,
        },
        "aggregate_counts": counts,
        "datasets": results,
        "notes": [
            "The external CSV files contain peptide,tcr,label and no HLA; only the Stage-9 checkpoint pTCR branch was evaluated.",
            "Stage-9 pair ranking and pHLA/PHT metrics cannot be computed for these files without HLA and explicit positive/negative TCR pairs.",
            "Peptides outside the configured length range [8, 15] were excluded by OnlinePairDataset.",
            "The threshold was fixed at 0.5 and was not selected or tuned on the external test sets.",
            "The checkpoint selected_pair_threshold belongs to the Stage-9 pair head and was not used for the pTCR branch.",
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report["json_path"] = str(json_path)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown(
            report=report,
            markdown_path=markdown_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_dir=data_dir,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] json={json_path}", flush=True)
    print(f"[DONE] markdown={markdown_path}", flush=True)
    print(f"[DONE] elapsed_min={(time.time() - started) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
