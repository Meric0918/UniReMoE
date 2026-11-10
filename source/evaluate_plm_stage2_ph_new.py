#!/usr/bin/env python3
"""Evaluate a Stage-2 pHLA task checkpoint on dataset/PH/new."""

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

_ROOT = Path(__file__).resolve().parents[1]
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import numpy as np
import torch
import torch.nn as nn

from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_online_dataloader
from plm_unified.finetuning import ESMCFineTuningEncoder
from plm_unified.metrics import compute_binary_metrics
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import resolve_device

SPLITS = ("test_unseen_hla", "test_unseen_pair", "test_unseen_peptide")
METRIC_FIELDS = (
    "loss", "auroc", "aupr", "accuracy", "mcc", "f1", "precision",
    "recall", "sensitivity", "specificity", "nll", "brier", "ece",
    "mean_confidence", "loss_p95", "loss_p99", "group_macro_nll",
    "group_macro_accuracy", "group_macro_recall", "worst_group_nll",
    "group_coverage", "performance_avg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_ROOT / "configs/plm_stage2.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "dataset/PH/new")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
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
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "stage2_task_model":
        raise ValueError(
            "Expected checkpoint_type=stage2_task_model, got "
            f"{checkpoint.get('checkpoint_type')!r}"
        )
    if checkpoint.get("task") != "phla":
        raise ValueError(f"Expected a pHLA checkpoint, got task={checkpoint.get('task')!r}")
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


def build_model(checkpoint: dict[str, Any], config, device: torch.device):
    saved_model = checkpoint.get("config", {}).get("model")
    if saved_model is not None and saved_model != config.to_dict()["model"]:
        raise ValueError("Checkpoint model configuration does not match evaluation config")
    encoder_dtype = {
        "none": "float32", "fp16": "float16", "bf16": "bfloat16",
    }[config.stage2.mixed_precision]
    kwargs = {
        "device": device,
        "dtype": encoder_dtype,
        "gradient_checkpointing": config.stage2.gradient_checkpointing,
    }
    print("[MODEL] loading fine-tuned ESM-C peptide encoder", flush=True)
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.peptide_max_length, **kwargs
    )
    print("[MODEL] loading fine-tuned ESM-C HLA encoder", flush=True)
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.hla_max_length, **kwargs
    )
    if peptide_encoder.hidden_size != config.model.peptide_input_dim:
        raise ValueError("Peptide ESM-C hidden size does not match model config")
    if hla_encoder.hidden_size != config.model.hla_input_dim:
        raise ValueError("HLA ESM-C hidden size does not match model config")

    states = checkpoint["model"]
    load_state(peptide_encoder, states["peptide_encoder"], "peptide_encoder")
    load_state(hla_encoder, states["receptor_encoder"], "hla_encoder")
    binding_model = TaskSpecificBindingModel(config.model, "phla")
    load_state(binding_model, states["binding_model"], "phla binding_model")
    binding_model.to(device)
    for module in (peptide_encoder, hla_encoder):
        module.set_active_training(False)
        module.eval()
        module.requires_grad_(False)
    binding_model.eval()
    binding_model.requires_grad_(False)
    return peptide_encoder, hla_encoder, binding_model


def evaluate_split(
    split: str,
    csv_path: Path,
    peptide_encoder,
    hla_encoder,
    binding_model,
    config,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    limit: int | None,
):
    loader = make_online_dataloader(
        csv_path,
        "phla",
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
    print(f"[EVAL START] {split}: rows={len(dataset):,} batches={len(loader):,}", flush=True)
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch_labels = batch["labels"].to(
                device, dtype=torch.long, non_blocking=device.type == "cuda"
            )
            with autocast_context(device, config.stage2.mixed_precision):
                peptide = peptide_encoder(batch["peptide_sequences"])
                hla = hla_encoder(batch["receptor_sequences"])
                output = binding_model(
                    "phla", peptide.hidden, peptide.mask, hla.hidden, hla.mask
                )
                loss = criterion(output.logits, batch_labels)
            batch_samples = int(batch_labels.shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            labels.append(batch_labels.cpu().numpy())
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
            )
            groups.append(batch["group_ids"].numpy())
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
    group_ids = np.concatenate(groups)
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=loss_sum / samples,
        threshold=config.training.threshold,
        strict_threshold=True,
        group_ids=group_ids,
    ).to_dict()
    item = {
        "path": str(csv_path.resolve()),
        "filter_stats": dict(dataset.filter_stats),
        "samples": int(len(dataset)),
        "negative": int(np.sum(y_true == 0)),
        "positive": int(np.sum(y_true == 1)),
        "positive_rate": float(np.mean(y_true)),
        "unique_peptides": len({row[0] for row in dataset.rows}),
        "unique_hlas": len({row[1] for row in dataset.rows}),
        "metrics": metrics,
    }
    print(
        f"[EVAL END] {split}: AUROC={metrics['auroc']:.6f} "
        f"AUPR={metrics['aupr']:.6f} "
        f"performance_avg={metrics['performance_avg']:.6f}",
        flush=True,
    )
    return item, y_true, y_score, group_ids


def macro_average(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        field: float(np.nanmean([results[split]["metrics"][field] for split in SPLITS]))
        for field in METRIC_FIELDS
    }


def pooled_metrics(labels, scores, groups) -> dict[str, Any]:
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    group_ids = np.concatenate(groups)
    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    nll = -np.log(np.where(y_true == 1, clipped, 1.0 - clipped))
    return compute_binary_metrics(
        y_true,
        y_score,
        loss=float(np.mean(nll)),
        threshold=0.5,
        strict_threshold=True,
        group_ids=group_ids,
    ).to_dict()


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not np.isfinite(numeric) else f"{numeric:.6f}"


def link(path: Path, root: Path) -> str:
    relative = os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")
    return f"[{path.name}]({relative})"


def render_markdown(report: dict[str, Any], output_md: Path, output_json: Path) -> str:
    checkpoint_path = Path(report["checkpoint"]["path"])
    lines = [
        "# PLM Unified Stage 2：PH new 数据集 pHLA 测试结果",
        "",
        "## 汇总",
        "",
        "| 汇总方式 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | performance_avg | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("三组等权宏平均", report["macro_average"]), ("全部样本 pooled", report["pooled_metrics"])):
        n = report["data"]["total_samples"] if name == "三组等权宏平均" else metrics["samples"]
        lines.append(
            f"| {name} | {n:,} | {fmt(metrics['auroc'])} | {fmt(metrics['aupr'])} | "
            f"{fmt(metrics['accuracy'])} | {fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
            f"{fmt(metrics['performance_avg'])} | {fmt(metrics['nll'])} | "
            f"{fmt(metrics['brier'])} | {fmt(metrics['ece'])} |"
        )
    lines += [
        "",
        f"合计 **{report['data']['total_samples']:,}** 条：负样本 {report['data']['total_negative']:,}，正样本 {report['data']['total_positive']:,}（{report['data']['positive_rate']:.2%}）。",
        "宏平均对三个 split 等权；pooled 将样本合并后重新计算。固定使用配置阈值 0.5，未根据测试标签调整阈值。",
        "",
        "## 评测设置与 checkpoint",
        "",
        f"- 评测时间：{report['evaluation_time']}；fold {report['fold']} / seed {report['seed']}。",
        f"- 目录：`{report['data']['directory']}`；Stage2 pHLA task checkpoint：`{checkpoint_path}`。",
        f"- 选择验证分数：{fmt(report['checkpoint']['best_score'])}；Stage `{report['checkpoint']['stage']}` 第 {report['checkpoint']['current_round']} 轮；global step {report['checkpoint']['global_step']}；SHA-256 `{report['checkpoint']['sha256']}`。",
        f"- 验证集 AUROC / AUPR：{fmt(report['checkpoint']['validation_metrics'].get('auroc'))} / {fmt(report['checkpoint']['validation_metrics'].get('aupr'))}；初始化方式 `{report['checkpoint']['initialization_mode']}`。",
        f"- 设备 `{report['device']}`，精度 `{report['mixed_precision']}`，batch size {report['batch_size']}；peptide 长度 {report['model_config']['peptide_min_length']}–{report['model_config']['peptide_max_length']} aa，HLA 最大长度 {report['model_config']['hla_max_length']} aa。",
        "- pHLA 分支使用 checkpoint 中微调后的 peptide ESM-C、HLA ESM-C 和分类模块在线推理。",
        "",
        "## 分 split 指标",
        "",
        "| Split | Samples | Neg | Pos | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | performance_avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        m = item["metrics"]
        lines.append(
            f"| `{split}` | {item['samples']:,} | {item['negative']:,} | {item['positive']:,} | "
            f"{fmt(m['auroc'])} | {fmt(m['aupr'])} | {fmt(m['accuracy'])} | "
            f"{fmt(m['mcc'])} | {fmt(m['f1'])} | {fmt(m['recall'])} | "
            f"{fmt(m['specificity'])} | {fmt(m['performance_avg'])} |"
        )
    lines += [
        "",
        "## 数据校准与分组鲁棒性",
        "",
        "| Split | NLL | Brier | ECE | Group count | Group coverage | Group-macro accuracy | Group-macro recall | Worst-group NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        m = report["splits"][split]["metrics"]
        lines.append(
            f"| `{split}` | {fmt(m['nll'])} | {fmt(m['brier'])} | {fmt(m['ece'])} | "
            f"{m['group_count']:,} | {fmt(m['group_coverage'])} | "
            f"{fmt(m['group_macro_accuracy'])} | {fmt(m['group_macro_recall'])} | "
            f"{fmt(m['worst_group_nll'])} |"
        )
    lines += [
        "",
        "测试划分对应的未见实体分别为 HLA、peptide–HLA pair 和 peptide；各 split 的样本构成不同，跨 split 比较应结合划分定义解读。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "PYTHONPATH=source /home/mjp/miniconda3/envs/unifyimmun/bin/python \\",
        f"  source/evaluate_plm_stage2_ph_new.py --config {report['config_path']} \\",
        f"  --checkpoint {checkpoint_path} --data-dir {report['data']['directory']} \\",
        f"  --output-json {output_json} --output-md {output_md} \\",
        f"  --fold {report['fold']} --seed {report['seed']} --batch-size {report['batch_size']} --device {report['device']}",
        "```",
        "",
        f"- 汇总 JSON：{link(output_json, output_md.parent)}",
        "- 原始数据集与 checkpoint 均未修改。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    if not checkpoint_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError(checkpoint_path if not checkpoint_path.is_file() else data_dir)
    device = resolve_device(config.training.device)
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"[RUN] checkpoint={checkpoint_path} device={device} batch_size={args.batch_size}", flush=True)
    started = time.perf_counter()
    peptide_encoder, hla_encoder, binding_model = build_model(checkpoint, config, device)
    split_results: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    for split in SPLITS:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        item, labels, scores, groups = evaluate_split(
            split, csv_path, peptide_encoder, hla_encoder, binding_model,
            config, device, args.batch_size, args.num_workers, args.limit,
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
            "validation_metrics": checkpoint.get("validation_metrics"),
            "initialization_mode": checkpoint.get("initialization_mode"),
            "run_timestamp": checkpoint.get("run_timestamp"),
            "source_checkpoint": checkpoint.get("source_checkpoint"),
        },
        "fold": config.training.fold,
        "seed": config.training.seed,
        "task": "phla",
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
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report, output_md, output_json), encoding="utf-8")
    print(f"[DONE] JSON: {output_json}", flush=True)
    print(f"[DONE] Markdown: {output_md}", flush=True)


if __name__ == "__main__":
    main()
