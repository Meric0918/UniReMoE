#!/usr/bin/env python3
"""Evaluate a Stage-2 pTCR task checkpoint on dataset/PT/new."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any


# Keep the same CUDA overlay setup as the Stage-2 training entry point.
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import numpy as np
import torch
import torch.nn as nn

from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_online_dataloader
from plm_unified.finetuning import (
    ESMCFineTuningEncoder,
    TCRBertFineTuningEncoder,
)
from plm_unified.metrics import compute_binary_metrics
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


SPLITS = ("test_unseen_pair", "test_unseen_peptide", "test_unseen_tcr")
METRIC_FIELDS = (
    "loss",
    "auroc",
    "aupr",
    "accuracy",
    "mcc",
    "f1",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "nll",
    "brier",
    "ece",
    "mean_confidence",
    "loss_p95",
    "loss_p99",
    "group_macro_nll",
    "group_macro_accuracy",
    "group_macro_recall",
    "worst_group_nll",
    "group_coverage",
    "performance_avg",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "plm_stage2.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage-2 pTCR task checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "dataset" / "PT" / "new",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "stage2_task_model":
        raise ValueError(
            "Expected checkpoint_type=stage2_task_model, got "
            f"{checkpoint.get('checkpoint_type')!r}"
        )
    if checkpoint.get("task") != "ptcr":
        raise ValueError(
            f"Expected a pTCR checkpoint, got task={checkpoint.get('task')!r}"
        )
    return checkpoint


def load_state(module: nn.Module, state: dict[str, torch.Tensor], name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [key for key in unexpected if not key.endswith("._extra_state")]
    if missing or unexpected:
        raise RuntimeError(
            f"{name} state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def model_config_matches(checkpoint: dict[str, Any], config) -> None:
    saved = checkpoint.get("config", {}).get("model")
    current = config.to_dict()["model"]
    if saved is not None and saved != current:
        raise ValueError(
            "Checkpoint model configuration does not match the evaluation config"
        )


def build_model(checkpoint: dict[str, Any], config, device: torch.device):
    precision = config.stage2.mixed_precision
    encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }[precision]
    encoder_kwargs = {
        "device": device,
        "dtype": encoder_dtype,
        "gradient_checkpointing": config.stage2.gradient_checkpointing,
    }
    print("[MODEL] loading fine-tuned ESM-C peptide encoder", flush=True)
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.peptide_max_length,
        **encoder_kwargs,
    )
    print("[MODEL] loading fine-tuned TCR-BERT encoder", flush=True)
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model,
        config.model.tcr_max_length,
        **encoder_kwargs,
    )
    if peptide_encoder.hidden_size != config.model.peptide_input_dim:
        raise ValueError("Peptide ESM-C hidden size does not match model config")
    if tcr_encoder.hidden_size != config.model.tcr_input_dim:
        raise ValueError("TCR-BERT hidden size does not match model config")

    states = checkpoint["model"]
    load_state(peptide_encoder, states["peptide_encoder"], "peptide_encoder")
    load_state(tcr_encoder, states["receptor_encoder"], "tcr_encoder")
    binding_model = TaskSpecificBindingModel(config.model, "ptcr")
    load_state(binding_model, states["binding_model"], "ptcr binding_model")
    binding_model.to(device)

    peptide_encoder.set_active_training(False)
    tcr_encoder.set_active_training(False)
    peptide_encoder.eval()
    tcr_encoder.eval()
    binding_model.eval()
    peptide_encoder.requires_grad_(False)
    tcr_encoder.requires_grad_(False)
    binding_model.requires_grad_(False)
    return peptide_encoder, tcr_encoder, binding_model


def evaluate_split(
    split: str,
    csv_path: Path,
    peptide_encoder,
    tcr_encoder,
    binding_model,
    config,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    limit: int | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    loader = make_online_dataloader(
        csv_path,
        "ptcr",
        batch_size=batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=num_workers,
        pin_memory=config.training.pin_memory,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        limit=limit,
    )
    dataset = loader.dataset
    print(
        f"[EVAL START] {split}: rows={len(dataset):,} batches={len(loader):,}",
        flush=True,
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch_labels = batch["labels"].to(
                device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )
            with autocast_context(device, config.stage2.mixed_precision):
                peptide = peptide_encoder(batch["peptide_sequences"])
                tcr = tcr_encoder(batch["receptor_sequences"])
                output = binding_model(
                    "ptcr",
                    peptide.hidden,
                    peptide.mask,
                    tcr.hidden,
                    tcr.mask,
                )
                loss = criterion(output.logits, batch_labels)
            batch_samples = int(batch_labels.shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            labels.append(batch_labels.cpu().numpy())
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
            )
            group_ids.append(batch["group_ids"].numpy())
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                elapsed = time.perf_counter() - started
                speed = samples / max(elapsed, 1e-9)
                print(
                    f"[EVAL] {split}: {batch_index:,}/{len(loader):,} "
                    f"samples={samples:,} speed={speed:,.0f}/s",
                    flush=True,
                )

    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    y_group = np.concatenate(group_ids)
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=loss_sum / samples,
        threshold=config.training.threshold,
        strict_threshold=True,
        group_ids=y_group,
    ).to_dict()
    row_stats = {
        "path": str(csv_path.resolve()),
        "filter_stats": dict(dataset.filter_stats),
        "samples": int(len(dataset)),
        "negative": int(np.sum(y_true == 0)),
        "positive": int(np.sum(y_true == 1)),
        "positive_rate": float(np.mean(y_true)),
        "unique_peptides": len({row[0] for row in dataset.rows}),
        "unique_tcrs": len({row[1] for row in dataset.rows}),
        "metrics": metrics,
    }
    print(
        f"[EVAL END] {split}: AUROC={metrics['auroc']:.6f} "
        f"AUPR={metrics['aupr']:.6f} performance_avg="
        f"{metrics['performance_avg']:.6f}",
        flush=True,
    )
    return row_stats, y_true, y_score, y_group


def macro_average(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        field: float(np.nanmean([results[split]["metrics"][field] for split in SPLITS]))
        for field in METRIC_FIELDS
    }


def pooled_metrics(
    labels: list[np.ndarray],
    scores: list[np.ndarray],
    groups: list[np.ndarray],
) -> dict[str, Any]:
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    y_group = np.concatenate(groups)
    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    nll = -np.log(np.where(y_true == 1, clipped, 1.0 - clipped))
    return compute_binary_metrics(
        y_true,
        y_score,
        loss=float(np.mean(nll)),
        threshold=0.5,
        strict_threshold=True,
        group_ids=y_group,
    ).to_dict()


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def link(path: Path, root: Path, label: str | None = None) -> str:
    relative = os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")
    return f"[{label or path.name}]({relative})"


def render_markdown(
    report: dict[str, Any],
    *,
    output_md: Path,
    output_json: Path,
    project_root: Path,
) -> str:
    checkpoint_path = Path(report["checkpoint"]["path"])
    data_dir = Path(report["data"]["directory"])
    macro = report["macro_average"]
    pooled = report["pooled_metrics"]
    lines = [
        "# PLM Unified Stage 2：PT new 数据集 pTCR 测试结果",
        "",
        "## 1. 结论摘要",
        "",
        (
            "本报告使用指定的 Stage 2 pTCR task checkpoint，对当前工作区 "
            "`dataset/PT/new/` 下的 3 个测试 split 完成在线推理。"
        ),
        "",
        "| 汇总方式 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | `performance_avg` | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 三组等权宏平均 | {report['data']['total_samples']:,} | "
            f"{fmt(macro['auroc'])} | {fmt(macro['aupr'])} | "
            f"{fmt(macro['accuracy'])} | {fmt(macro['mcc'])} | "
            f"{fmt(macro['f1'])} | {fmt(macro['performance_avg'])} | "
            f"{fmt(macro['nll'])} | {fmt(macro['brier'])} | {fmt(macro['ece'])} |"
        ),
        (
            f"| 全部样本 pooled | {pooled['samples']:,} | "
            f"{fmt(pooled['auroc'])} | {fmt(pooled['aupr'])} | "
            f"{fmt(pooled['accuracy'])} | {fmt(pooled['mcc'])} | "
            f"{fmt(pooled['f1'])} | {fmt(pooled['performance_avg'])} | "
            f"{fmt(pooled['nll'])} | {fmt(pooled['brier'])} | {fmt(pooled['ece'])} |"
        ),
        "",
        (
            f"三个 split 共纳入 **{report['data']['total_samples']:,}** 条样本，"
            f"其中负样本 **{report['data']['total_negative']:,}** 条、"
            f"正样本 **{report['data']['total_positive']:,}** 条，"
            f"正样本比例 **{pct(report['data']['positive_rate'])}**。"
        ),
        "",
        "`performance_avg` 是项目实现中 AUROC、Accuracy、MCC、F1 和 AUPR 五项指标的算术平均。宏平均对三个 split 等权计算；pooled 行将全部样本合并后重新计算。",
        "",
        "## 2. 评测设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 评测时间 | {report['evaluation_time']} |",
        f"| Fold / seed | fold {report['fold']} / seed {report['seed']} |",
        "| 任务 | pTCR（peptide–TCR 二分类） |",
        f"| 测试目录 | {link(data_dir, output_md.parent)} |",
        f"| 评测脚本 | {link(project_root / 'source' / 'evaluate_plm_stage2_pt_new.py', output_md.parent)} |",
        f"| 配置 | {link(Path(report['config_path']), output_md.parent)} |",
        "| 输入模式 | Stage 2 微调 PLM 在线编码 |",
        f"| 推理设备 | `{report['device']}` |",
        f"| 混合精度 | `{report['mixed_precision']}` autocast |",
        f"| 推理 batch size | `{report['batch_size']}` |",
        f"| Peptide 长度约束 | `{report['model_config']['peptide_min_length']}–{report['model_config']['peptide_max_length']}` aa |",
        f"| TCR 最大长度 | `{report['model_config']['tcr_max_length']}` aa |",
        f"| 分类阈值 | `{report['threshold']}`，严格比较 `score > 0.5` |",
        "",
        "### 2.1 Checkpoint 信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Checkpoint | {link(checkpoint_path, output_md.parent)} |",
        f"| 类型 / 任务 | `{report['checkpoint']['type']}` / `{report['checkpoint']['task']}` |",
        f"| Stage | `{report['checkpoint']['stage']}` |",
        f"| 选中轮次 | `{report['checkpoint']['current_round']}` |",
        f"| Global step | `{report['checkpoint']['global_step']}` |",
        f"| 验证集最佳分数 | `{fmt(report['checkpoint']['best_score'])}` |",
        f"| 验证 AUROC / AUPR | `{fmt(report['checkpoint']['validation_metrics'].get('auroc'))}` / `{fmt(report['checkpoint']['validation_metrics'].get('aupr'))}` |",
        f"| 初始化方式 | `{report['checkpoint']['initialization_mode']}` |",
        f"| SHA-256 | `{report['checkpoint']['sha256']}` |",
        "",
        "本次只调用 pTCR 分支；pHLA 分支和 HLA 编码器不参与推理。Checkpoint 中的 peptide ESM-C 和 TCR-BERT 权重均为 Stage 2 task checkpoint 内保存的微调后权重。",
        "",
        "## 3. 测试数据统计",
        "",
        "| Benchmark | 数据文件 | Samples | 负样本 | 正样本 | 正样本比例 | 唯一 peptide | 唯一 TCR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        lines.append(
            f"| `{split}` | {link(Path(item['path']), output_md.parent)} | "
            f"{item['samples']:,} | {item['negative']:,} | {item['positive']:,} | "
            f"{pct(item['positive_rate'])} | {item['unique_peptides']:,} | "
            f"{item['unique_tcrs']:,} |"
        )
    lines += [
        f"| **合计** | **3 个 new benchmark** | **{report['data']['total_samples']:,}** | **{report['data']['total_negative']:,}** | **{report['data']['total_positive']:,}** | **{pct(report['data']['positive_rate'])}** | — | — |",
        "",
        "## 4. 逐 benchmark 主结果",
        "",
        "| Benchmark | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall / Sensitivity | Specificity | `performance_avg` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        metrics = report["splits"][split]["metrics"]
        lines.append(
            f"| `{split}` | {metrics['samples']:,} | {fmt(metrics['auroc'])} | "
            f"{fmt(metrics['aupr'])} | {fmt(metrics['accuracy'])} | "
            f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
            f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
            f"{fmt(metrics['specificity'])} | {fmt(metrics['performance_avg'])} |"
        )
    lines += [
        "",
        "## 5. 校准与分组鲁棒性指标",
        "",
        "| Benchmark | NLL | Brier | ECE | Mean confidence | Group count | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        metrics = report["splits"][split]["metrics"]
        lines.append(
            f"| `{split}` | {fmt(metrics['nll'])} | {fmt(metrics['brier'])} | "
            f"{fmt(metrics['ece'])} | {fmt(metrics['mean_confidence'])} | "
            f"{metrics['group_count']:,} | {fmt(metrics['group_coverage'])} | "
            f"{fmt(metrics['group_macro_accuracy'])} | "
            f"{fmt(metrics['group_macro_recall'])} | "
            f"{fmt(metrics['worst_group_nll'])} |"
        )
    best_split = max(
        SPLITS,
        key=lambda split: report["splits"][split]["metrics"]["performance_avg"],
    )
    lines += [
        "",
        "## 6. 结果摘要",
        "",
        (
            f"- `{best_split}` 的 `performance_avg` 最高，为 "
            f"`{fmt(report['splits'][best_split]['metrics']['performance_avg'])}`；"
            f"AUROC=`{fmt(report['splits'][best_split]['metrics']['auroc'])}`，"
            f"AUPR=`{fmt(report['splits'][best_split]['metrics']['aupr'])}`。"
        ),
        (
            "- 三个 split 的实体未见设置不同：`test_unseen_pair` 同时约束 peptide/TCR "
            "组合，`test_unseen_peptide` 约束 peptide，`test_unseen_tcr` 约束 TCR；"
            "跨 split 比较应结合各自样本构成解读。"
        ),
        (
            "- 本次使用固定阈值 `0.5` 计算 Accuracy、MCC、F1、Precision、Recall 和 "
            "Specificity，没有利用 test 标签重新选择阈值。"
        ),
        "",
        "## 7. 原始结果与复现",
        "",
        f"- 原始汇总 JSON：{link(output_json, output_md.parent)}",
        "",
        "```bash",
        "export PYTHONPATH=source",
        "PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python",
        f"$PYTHON_BIN source/evaluate_plm_stage2_pt_new.py \\",
        f"  --config {report['config_path']} \\",
        f"  --checkpoint {checkpoint_path} \\",
        f"  --data-dir {data_dir} \\",
        f"  --output-json {output_json} \\",
        f"  --output-md {output_md} \\",
        f"  --batch-size {report['batch_size']} \\",
        f"  --device {report['device']}",
        "```",
        "",
        "本报告中的结果来自当前工作区的 `dataset/PT/new/*.csv`；未修改原始数据集或 checkpoint 文件。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    device = resolve_device(config.training.device)
    if device.type != "cuda":
        print(f"[WARN] using {device}; Stage-2 inference is intended for CUDA", flush=True)
    checkpoint = load_checkpoint(checkpoint_path)
    model_config_matches(checkpoint, config)
    print(
        f"[RUN] checkpoint={checkpoint_path} device={device} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    started = time.perf_counter()
    peptide_encoder, tcr_encoder, binding_model = build_model(
        checkpoint,
        config,
        device,
    )
    print(
        f"[MODEL] ready in {time.perf_counter() - started:.1f}s; "
        f"checkpoint_sha256={sha256_file(checkpoint_path)}",
        flush=True,
    )

    split_results: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    for split in SPLITS:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        item, labels, scores, groups = evaluate_split(
            split,
            csv_path,
            peptide_encoder,
            tcr_encoder,
            binding_model,
            config,
            device,
            args.batch_size,
            args.num_workers,
            args.limit,
        )
        split_results[split] = item
        all_labels.append(labels)
        all_scores.append(scores)
        all_groups.append(groups)

    total_samples = sum(item["samples"] for item in split_results.values())
    total_negative = sum(item["negative"] for item in split_results.values())
    total_positive = sum(item["positive"] for item in split_results.values())
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_path": str(args.config.resolve()),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "type": checkpoint.get("checkpoint_type"),
            "task": checkpoint.get("task"),
            "stage": checkpoint.get("stage"),
            "current_round": checkpoint.get("current_round"),
            "global_step": checkpoint.get("global_step"),
            "best_score": checkpoint.get("best_score"),
            "validation_metrics": checkpoint.get("validation_metrics", {}),
            "initialization_mode": checkpoint.get("initialization_mode"),
            "run_timestamp": checkpoint.get("run_timestamp"),
            "source_checkpoint": checkpoint.get("source_checkpoint"),
        },
        "fold": config.training.fold,
        "seed": config.training.seed,
        "task": "ptcr",
        "device": str(device),
        "mixed_precision": config.stage2.mixed_precision,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "threshold": config.training.threshold,
        "data": {
            "directory": str(data_dir),
            "splits": list(SPLITS),
            "total_samples": total_samples,
            "total_negative": total_negative,
            "total_positive": total_positive,
            "positive_rate": total_positive / total_samples,
        },
        "model_config": config.to_dict()["model"],
        "encoder_metadata": checkpoint.get("encoder_metadata", {}),
        "splits": split_results,
        "macro_average": macro_average(split_results),
        "pooled_metrics": pooled_metrics(all_labels, all_scores, all_groups),
        "report_json": str(output_json),
        "report_md": str(output_md),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        render_markdown(
            report,
            output_md=output_md,
            output_json=output_json,
            project_root=project_root,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] JSON: {output_json}", flush=True)
    print(f"[DONE] Markdown: {output_md}", flush=True)


if __name__ == "__main__":
    main()
