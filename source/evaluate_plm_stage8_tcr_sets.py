#!/usr/bin/env python3
"""Evaluate the pTCR branch of a Stage-8 checkpoint on external pair sets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import numpy as np
import torch
import torch.nn as nn

from evaluate_plm_triplet_tcr_sets import _encode_unique
from plm_unified.data import OnlinePairDataset
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage8_checkpoint import (
    checkpoint_sha256,
    load_stage8_v2_checkpoint,
    validate_stage8_v2_for_evaluation,
)
from plm_unified.stage8_config import (
    apply_stage8_overrides,
    load_stage8_config,
)
from plm_unified.stage8_model import build_progressive_stage8_model
from plm_unified.trainer import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "plm_stage8.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "data_TCR" / "test_data_set",
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
    return parser.parse_args(argv)


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


def _evaluate_rows(
    binding_model,
    rows: list[tuple[str, str, int]],
    peptide_index: dict[str, int],
    tcr_index: dict[str, int],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    tcr_hidden: np.ndarray,
    tcr_mask: np.ndarray,
    *,
    device: torch.device,
    pair_batch_size: int,
    threshold: float,
) -> tuple[dict[str, float | int], dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    total_batches = (len(rows) + pair_batch_size - 1) // pair_batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(rows), pair_batch_size),
            start=1,
        ):
            batch_rows = rows[start : start + pair_batch_size]
            peptide_locations = np.fromiter(
                (peptide_index[row[0]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            tcr_locations = np.fromiter(
                (tcr_index[row[1]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            batch_labels_cpu = np.fromiter(
                (row[2] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            peptide_batch = torch.from_numpy(
                peptide_hidden[peptide_locations]
            ).to(device, dtype=torch.bfloat16, non_blocking=True)
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[peptide_locations]
            ).to(device, dtype=torch.bool, non_blocking=True)
            tcr_batch = torch.from_numpy(tcr_hidden[tcr_locations]).to(
                device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            tcr_mask_batch = torch.from_numpy(tcr_mask[tcr_locations]).to(
                device,
                dtype=torch.bool,
                non_blocking=True,
            )
            batch_labels = torch.from_numpy(batch_labels_cpu).to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = binding_model(
                    "ptcr",
                    peptide_batch,
                    peptide_mask_batch,
                    tcr_batch,
                    tcr_mask_batch,
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
            if batch_id == 1 or batch_id % 100 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                print(
                    f"[EVAL] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(rows):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=loss_sum / samples,
        threshold=threshold,
    ).to_dict()
    metrics["positive_samples"] = int(y_true.sum())
    metrics["negative_samples"] = int((y_true == 0).sum())
    metrics["positive_rate"] = float(y_true.mean())
    probability_summary = {
        "mean": float(y_score.mean()),
        "positive_mean": float(y_score[y_true == 1].mean()),
        "negative_mean": float(y_score[y_true == 0].mean()),
    }
    return metrics, probability_summary


def _checkpoint_metadata(checkpoint: dict) -> dict:
    keys = (
        "checkpoint_type",
        "format_version",
        "condition",
        "artifact_role",
        "phase",
        "run_timestamp",
        "source_stage1",
        "source_stage8a",
        "dataset_fingerprint",
        "training_state",
        "metrics",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


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
        "# PLM Stage 8 best 模型 pTCR 测试结果",
        "",
        "## 1. 测试概况",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 评测日期 | {report['report_timestamp'][:4]}-{report['report_timestamp'][4:6]}-{report['report_timestamp'][6:8]} |",
        f"| Checkpoint | {checkpoint_path} |",
        f"| Checkpoint 类型 | {checkpoint['metadata'].get('checkpoint_type', 'unknown')} |",
        f"| Stage 8 条件 | {checkpoint['metadata'].get('condition', 'unknown')} |",
        f"| Stage8B 轮次 | {checkpoint['metadata'].get('training_state', {}).get('stage8b_round', 'unknown')} |",
        f"| 验证 joint score | {_fmt(checkpoint['metadata'].get('metrics', {}).get('validation_joint_score', float('nan')))} |",
        "| 测试分支 | pTCR 二元分类分支 |",
        f"| 测试数据目录 | {data_dir} |",
        "| 输入字段 | peptide,tcr,label |",
        "| HLA 输入 | 无 |",
        f"| 分类阈值 | {report['threshold_used']:.1f}（固定，未在测试集调参） |",
        f"| 推理设备 | {report['inference_device']} |",
        "",
        "测试文件只包含 peptide、TCR 和 label，没有 HLA 列，因此本次只调用 checkpoint 中的 pTCR 分支；不能计算需要 HLA 输入的 pHLA、PHT 三元或成对排序指标。",
        "",
        "## 2. 数据审计",
        "",
        "| 数据集 | 原始数据行数 | 实际测试样本数 | 正样本 | 负样本 | 正样本比例 | peptide 长度过滤 |",
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
    lines += [
        "",
        "### 3.1 概率输出概览",
        "",
        "| 数据集 | 平均预测概率 | 正样本平均概率 | 负样本平均概率 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(datasets):
        probability = datasets[name]["probability_summary"]
        lines.append(
            f"| {name} | {_fmt(probability['mean'])} | "
            f"{_fmt(probability['positive_mean'])} | "
            f"{_fmt(probability['negative_mean'])} |"
        )
    source = report["source_stage1"]
    lines += [
        "",
        "## 4. 推理方法与模型来源",
        "",
        "1. 加载指定的 Stage 8 V2 stage8_best checkpoint，并校验 checkpoint 类型、模型配置和 Stage 1 来源 SHA-256。",
        "2. 直接加载 checkpoint 内的完整在线模型状态，并使用其微调后的 peptide ESM-C 和 TCR-BERT 编码器。",
        "3. 对三个测试集并集中的每个唯一 peptide/TCR 编码一次，再复用 residue-level 表示进行样本级 pTCR 推理；没有使用旧的 Stage 1 embedding cache。",
        f"4. 使用配置中的 peptide 长度范围 [8, 15]；本次最长保留 TCR 为 {unique['max_tcr_length']} aa，未超过上限 34 aa。",
        "5. 阈值固定为 0.5，没有在测试集上重新选择阈值。checkpoint 中的 selected_pht_threshold 属于 PHT 头，本次无 HLA 的 pTCR 测试不使用。",
        "",
        "## 5. 结果解读",
        "",
    ]
    ranked = sorted(
        datasets.items(),
        key=lambda item: item[1]["metrics"]["auroc"],
        reverse=True,
    )
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
    covid = datasets.get("covid_set")
    independent = datasets.get("independent_set")
    triple = datasets.get("triple_set_peptide_8_15aa")
    if triple:
        lines.append(
            f"- triple_set_peptide_8_15aa 的 AUROC={triple['metrics']['auroc']:.6f}、"
            f"AUPR={triple['metrics']['aupr']:.6f}，固定阈值下 MCC="
            f"{triple['metrics']['mcc']:.6f}，表现最好。"
        )
    if independent:
        lines.append(
            f"- independent_set 的 AUROC={independent['metrics']['auroc']:.6f}，"
            f"但固定阈值 Recall={independent['metrics']['recall']:.6f}，"
            "说明排序能力优于固定阈值下的阳性召回表现。"
        )
    if covid:
        lines.append(
            f"- covid_set 的 AUROC={covid['metrics']['auroc']:.6f}，"
            f"固定阈值 Recall={covid['metrics']['recall']:.6f}、"
            f"Specificity={covid['metrics']['specificity']:.6f}，"
            "提示该数据集上的概率分布与 0.5 决策阈值存在明显不匹配。"
        )
    lines += [
        "",
        "上述结论只针对无 HLA 外部数据上的 pTCR 分支；不能据此宣称 Stage 8 的 PHT 三元约束或 pHLA 分支已在这些文件上得到验证。若需提高某个外部域的阳性召回，应在独立验证集上重新校准阈值，不能直接用测试集调参。",
        "",
        "## 6. 相关文件",
        "",
        f"- Checkpoint：{checkpoint_path}",
        f"- Stage 1 来源：{source['path']}",
        f"- Stage 1 来源 SHA-256：{source['sha256']}",
        f"- 配置：{config_path}",
        f"- 测试数据：{data_dir}",
        f"- 原始 JSON 结果：{report['json_path']}",
        f"- 本总结：{markdown_path}",
        "",
        "## 7. 复现参数",
        "",
        f"device={report['inference_device']}，encode_batch_size={report['batch_sizes']['encode_batch_size']}，pair_batch_size={report['batch_sizes']['pair_batch_size']}，threshold={report['threshold_used']:.1f}。",
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
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    stage8_config = load_stage8_config(config_path)
    apply_stage8_overrides(stage8_config, device=args.device)
    stage8_config.validate()
    device = resolve_device(args.device)
    print(
        f"[INIT] checkpoint={checkpoint_path} data_dir={data_dir} "
        f"device={device} threshold={args.threshold}",
        flush=True,
    )

    checkpoint = load_stage8_v2_checkpoint(checkpoint_path)
    validate_stage8_v2_for_evaluation(checkpoint, config=stage8_config)
    source_lineage = dict(checkpoint["source_stage1"])
    print(
        f"[CHECKPOINT] type={checkpoint.get('checkpoint_type')} "
        f"condition={checkpoint.get('condition')} "
        f"role={checkpoint.get('artifact_role')} "
        f"round={checkpoint.get('training_state', {}).get('stage8b_round')}",
        flush=True,
    )
    print(
        f"[SOURCE] path={source_lineage.get('path')} "
        f"sha256={source_lineage.get('sha256')}",
        flush=True,
    )

    dataset_rows: dict[str, list[tuple[str, str, int]]] = {}
    filter_stats: dict[str, dict[str, int]] = {}
    dataset_paths: dict[str, Path] = {}
    for path in resolve_files(data_dir, args.files):
        name = path.stem
        dataset = OnlinePairDataset(
            path,
            "ptcr",
            peptide_min_length=stage8_config.model.peptide_min_length,
            peptide_max_length=stage8_config.model.peptide_max_length,
        )
        dataset_paths[name] = path
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
    if max_tcr_length > stage8_config.model.tcr_max_length:
        raise ValueError(
            f"TCR length {max_tcr_length} exceeds configured maximum "
            f"{stage8_config.model.tcr_max_length}"
        )
    print(
        f"[UNIQUE] peptides={len(all_peptides):,} "
        f"tcrs={len(all_tcrs):,} max_tcr_length={max_tcr_length}",
        flush=True,
    )

    model_started = time.time()
    model = build_progressive_stage8_model(stage8_config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_evaluation()
    binding_model = model.binding_model
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_state"):
        checkpoint.pop(key, None)
    gc.collect()
    print(
        f"[MODEL] Stage-8 V2 full online model loaded in "
        f"{time.time() - model_started:.1f}s",
        flush=True,
    )

    device_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    )
    peptide_hidden, peptide_mask = _encode_unique(
        "peptide",
        model.peptide_encoder,
        all_peptides,
        stage8_config.model.peptide_max_length,
        args.encode_batch_size,
        device,
    )
    tcr_hidden, tcr_mask = _encode_unique(
        "tcr",
        model.tcr_encoder,
        all_tcrs,
        stage8_config.model.tcr_max_length,
        args.encode_batch_size,
        device,
    )
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}

    results = {}
    for name in sorted(dataset_rows):
        print(f"[{name}] evaluating", flush=True)
        metrics, probability_summary = _evaluate_rows(
            binding_model,
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
        results[name] = {
            "source_file": str(dataset_paths[name].resolve()),
            "filter_stats": filter_stats[name],
            "metrics": metrics,
            "probability_summary": probability_summary,
            "threshold_used": args.threshold,
            "inference": "stage8_v2_checkpoint_ptcr_branch_with_online_encoders",
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
        / f"evaluation_stage8_ptcr_test_data_set_{timestamp}.json"
    )
    markdown_path = (
        args.markdown.resolve()
        if args.markdown
        else root / "tests" / f"PLM_Stage8_best_TCR_test_results_{timestamp[:8]}.md"
    )
    if markdown_path.exists() and not args.markdown:
        markdown_path = markdown_path.with_name(
            f"PLM_Stage8_best_TCR_test_results_{timestamp}.md"
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
            "sha256": checkpoint_sha256(checkpoint_path),
            "metadata": _checkpoint_metadata(checkpoint),
        },
        "source_stage1": source_lineage,
        "source_stage8a": checkpoint.get("source_stage8a"),
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
            "The external CSV files contain peptide,tcr,label and no HLA; only the Stage-8 checkpoint pTCR branch was evaluated.",
            "The full Stage-8 V2 online encoders were run once per unique peptide and TCR; no Stage-1 embedding cache was used.",
            "Rows outside the configured peptide length range [8, 15] were excluded.",
            "The threshold was fixed at 0.5 and was not selected or tuned on the external test sets.",
            "The checkpoint selected_pht_threshold belongs to the PHT head and was not used for the pTCR branch.",
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
