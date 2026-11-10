#!/usr/bin/env python3
"""Compare Stage3/Stage10 and optional Stage8 on external pTCR sets."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import numpy as np
import torch

from evaluate_plm_triplet_tcr_sets import _encode_unique
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import normalise_sequence, resolve_task_columns
from plm_unified.stage8_checkpoint import extract_stage8_binding_state
from plm_unified.stage8_model import Stage8BindingModel
from plm_unified.stage10_checkpoint import (
    checkpoint_sha256,
    extract_stage10_binding_state,
    load_stage3_teacher_source,
    load_torch_checkpoint,
    validate_stage10_full_checkpoint,
    validate_stage10_lean_checkpoint,
)
from plm_unified.stage10_config import (
    apply_stage10_overrides,
    load_stage10_config,
)
from plm_unified.stage10_external import (
    EncodedPairSequences,
    ExternalPairRow,
    build_external_comparison_report,
    evaluate_binding_models_on_rows,
    validate_stage8_reference_source,
    write_external_row_csv,
)
from plm_unified.stage10_model import Stage10BindingModel
from plm_unified.trainer import resolve_device
from plm_unified.triplet_model import (
    build_progressive_triplet_model,
    load_triplet_model_state,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/plm_stage10.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage8-checkpoint", type=Path)
    parser.add_argument(
        "--stage3-config",
        type=Path,
        default=root / "configs/plm_triplet.yaml",
    )
    parser.add_argument("--stage3-checkpoint", type=Path)
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed classification threshold; never selected on test sets.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--rows-output", type=Path)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.encode_batch_size < 1 or args.pair_batch_size < 1:
        raise ValueError("Batch sizes must be positive")


def resolve_files(data_dir: Path, value: str | None) -> list[Path]:
    paths = (
        [
            data_dir / item.strip()
            for item in value.split(",")
            if item.strip()
        ]
        if value
        else sorted(data_dir.glob("*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
    return paths


def load_external_rows(
    path: Path,
    *,
    peptide_min_length: int,
    peptide_max_length: int,
) -> tuple[list[ExternalPairRow], dict[str, int]]:
    """Apply the same peptide/label filtering as OnlinePairDataset."""

    rows: list[ExternalPairRow] = []
    stats = {
        "total": 0,
        "kept": 0,
        "peptide_too_short": 0,
        "peptide_too_long": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        peptide_column, tcr_column, label_column = resolve_task_columns(
            "ptcr",
            reader.fieldnames,
        )
        required = {peptide_column, tcr_column, label_column}
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for source_row, raw in enumerate(reader, start=2):
            stats["total"] += 1
            peptide = normalise_sequence(raw[peptide_column])
            if len(peptide) < peptide_min_length:
                stats["peptide_too_short"] += 1
                continue
            if len(peptide) > peptide_max_length:
                stats["peptide_too_long"] += 1
                continue
            tcr = normalise_sequence(raw[tcr_column])
            try:
                label = int(raw[label_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{source_row}") from exc
            if label not in {0, 1}:
                raise ValueError(
                    f"Label must be 0 or 1 at {path}:{source_row}"
                )
            rows.append(ExternalPairRow(peptide, tcr, label, source_row))
            stats["kept"] += 1
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows, stats


def _checkpoint_metadata(checkpoint: dict) -> dict:
    keys = (
        "checkpoint_type",
        "condition",
        "stage10_condition",
        "artifact_role",
        "phase",
        "epoch",
        "best_epoch",
        "best_joint_score",
        "validation_joint_score",
        "run_timestamp",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def build_markdown(report: dict) -> str:
    lines = [
        "# Stage10 外部 pTCR 保留对比",
        "",
        "该报告只做描述性比较。分类阈值固定为 "
        f"{report['threshold_used']:.3f}，未在外部测试集上选阈值、选模型或调参。",
        "",
        "| 数据集 | 模型 | Samples | AUROC | AUPR | Accuracy | MCC | F1 | Recall | Specificity | Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(report["datasets"]):
        item = report["datasets"][dataset]
        for model in ("stage3", "stage8", "stage10"):
            if model not in item["models"]:
                continue
            metrics = item["models"][model]["metrics"]
            lines.append(
                f"| {dataset} | {model} | {metrics['samples']} | "
                f"{metrics['auroc']:.6f} | {metrics['aupr']:.6f} | "
                f"{metrics['accuracy']:.6f} | {metrics['mcc']:.6f} | "
                f"{metrics['f1']:.6f} | {metrics['recall']:.6f} | "
                f"{metrics['specificity']:.6f} | {metrics['loss']:.6f} |"
            )
        lines.extend(["", f"`{dataset}` 的有符号差值：", ""])
        for comparison, deltas in item["deltas"].items():
            summary = ", ".join(
                f"{name}={value:+.6f}"
                for name, value in deltas.items()
                if name in {"auroc", "aupr", "accuracy", "mcc", "f1"}
            )
            lines.append(f"- {comparison}: {summary}")
        lines.append("")
    lines.extend(
        [
            "外部 CSV 不含 HLA，因此这里只评价 pTCR 分支，不能用于评价 PHT 三元头或约束违例率。",
            "",
            f"逐行概率：`{report['row_csv']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    started = time.time()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    stage8_path = (
        args.stage8_checkpoint.expanduser().resolve()
        if args.stage8_checkpoint
        else None
    )
    data_dir = args.data_dir.expanduser().resolve()
    config = apply_stage10_overrides(
        load_stage10_config(args.config),
        device=args.device,
        stage3_checkpoint=args.stage3_checkpoint,
    )
    config.validate(require_files=True)
    device = resolve_device(config.runtime.device)

    source = load_stage3_teacher_source(config)
    stage10_checkpoint = load_torch_checkpoint(checkpoint_path, mmap=False)
    if stage10_checkpoint.get("checkpoint_type") == "stage10_multitask_model":
        validate_stage10_full_checkpoint(
            stage10_checkpoint,
            config,
            source.sha256,
        )
    else:
        validate_stage10_lean_checkpoint(
            stage10_checkpoint,
            config,
            source.sha256,
        )
    stage8_checkpoint = None
    if stage8_path is not None:
        stage8_checkpoint = load_torch_checkpoint(stage8_path, mmap=False)
        validate_stage8_reference_source(stage8_checkpoint, source.sha256)
        if stage8_checkpoint.get("config", {}).get("model") != config.to_dict()[
            "model"
        ]:
            raise ValueError("Optional Stage-8 model configuration mismatch")

    rows_by_dataset = {}
    filters = {}
    dataset_paths = {}
    for path in resolve_files(data_dir, args.files):
        rows, stats = load_external_rows(
            path,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        rows_by_dataset[path.stem] = rows
        filters[path.stem] = stats
        dataset_paths[path.stem] = path

    all_peptides = sorted(
        {
            row.peptide
            for rows in rows_by_dataset.values()
            for row in rows
        },
        key=lambda value: (-len(value), value),
    )
    all_tcrs = sorted(
        {row.tcr for rows in rows_by_dataset.values() for row in rows},
        key=lambda value: (-len(value), value),
    )
    max_tcr_length = max(map(len, all_tcrs), default=0)
    if max_tcr_length > config.model.tcr_max_length:
        raise ValueError(
            f"TCR length {max_tcr_length} exceeds configured maximum "
            f"{config.model.tcr_max_length}"
        )

    stage3_config = load_config(args.stage3_config)
    stage3_config.paths.esmc_model = config.paths.esmc_model
    stage3_config.paths.tcr_bert_model = config.paths.tcr_bert_model
    stage3_config.model = config.model
    stage3_config.triplet.mixed_precision = config.stage10.mixed_precision
    apply_overrides(
        stage3_config,
        fold=config.runtime.fold,
        seed=config.runtime.seed,
        device=args.device,
    )
    stage3_config.validate()
    stage3_model = build_progressive_triplet_model(stage3_config, device)
    load_triplet_model_state(stage3_model, source.checkpoint["model"])
    stage3_model.set_evaluation()

    stage10_model = Stage10BindingModel(config.model).to(device)
    stage10_model.load_state_dict(
        extract_stage10_binding_state(stage10_checkpoint),
        strict=True,
    )
    stage10_model.eval()
    models = {
        "stage3": stage3_model.binding_model,
        "stage10": stage10_model,
    }
    if stage8_checkpoint is not None:
        saved_stage8_config = stage8_checkpoint.get("config", {})
        saved_stage8_training = saved_stage8_config.get("stage8", {})
        stage8_model = Stage8BindingModel(
            config.model,
            encoder_num_layers=saved_stage8_training.get(
                "sequence_encoder_num_layers", 3
            ),
            encoder_dropout=saved_stage8_training.get(
                "sequence_encoder_dropout", 0.15
            ),
        ).to(device)
        stage8_model.load_state_dict(
            extract_stage8_binding_state(stage8_checkpoint),
            strict=True,
        )
        stage8_model.eval()
        models["stage8"] = stage8_model

    source_lineage = dict(source.lineage)
    source.checkpoint = {}
    source.binding_state = {}
    gc.collect()
    peptide_hidden, peptide_mask = _encode_unique(
        "peptide",
        stage3_model.peptide_encoder,
        all_peptides,
        config.model.peptide_max_length,
        args.encode_batch_size,
        device,
    )
    tcr_hidden, tcr_mask = _encode_unique(
        "tcr",
        stage3_model.tcr_encoder,
        all_tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
    )
    encoded = EncodedPairSequences(
        peptide_index={value: index for index, value in enumerate(all_peptides)},
        tcr_index={value: index for index, value in enumerate(all_tcrs)},
        peptide_hidden=peptide_hidden,
        peptide_mask=peptide_mask,
        tcr_hidden=tcr_hidden,
        tcr_mask=tcr_mask,
    )

    datasets = {}
    probabilities_by_dataset = {}
    for name in sorted(rows_by_dataset):
        reports, probabilities = evaluate_binding_models_on_rows(
            models,
            rows_by_dataset[name],
            encoded,
            device=device,
            pair_batch_size=args.pair_batch_size,
            threshold=args.threshold,
        )
        labels = np.asarray(
            [row.label for row in rows_by_dataset[name]],
            dtype=np.int64,
        )
        comparison = build_external_comparison_report(
            labels,
            probabilities,
            threshold=args.threshold,
        )
        probabilities_by_dataset[name] = probabilities
        datasets[name] = {
            "source_file": str(dataset_paths[name]),
            "filter_stats": filters[name],
            "models": reports,
            "deltas": comparison["deltas"],
        }

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint_path.parent
        / f"evaluation_stage10_ptcr_comparison_{timestamp}.json"
    )
    row_output = (
        args.rows_output.expanduser().resolve()
        if args.rows_output
        else checkpoint_path.parent
        / f"evaluation_stage10_ptcr_rows_{timestamp}.csv"
    )
    markdown = (
        args.markdown.expanduser().resolve()
        if args.markdown
        else checkpoint_path.parent
        / f"evaluation_stage10_ptcr_comparison_{timestamp}.md"
    )
    write_external_row_csv(
        row_output,
        rows_by_dataset,
        probabilities_by_dataset,
    )
    aggregate_samples = sum(
        item["models"]["stage10"]["metrics"]["samples"]
        for item in datasets.values()
    )
    aggregate_positive = sum(
        item["models"]["stage10"]["metrics"]["positive_samples"]
        for item in datasets.values()
    )
    aggregate_negative = sum(
        item["models"]["stage10"]["metrics"]["negative_samples"]
        for item in datasets.values()
    )
    report = {
        "status": "complete",
        "report_timestamp": timestamp,
        "evaluation_role": "descriptive_external_test_only",
        "checkpoint_stage10": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256(checkpoint_path),
            "metadata": _checkpoint_metadata(stage10_checkpoint),
        },
        "checkpoint_stage8": (
            {
                "path": str(stage8_path),
                "sha256": checkpoint_sha256(stage8_path),
                "metadata": _checkpoint_metadata(stage8_checkpoint),
            }
            if stage8_path is not None and stage8_checkpoint is not None
            else None
        ),
        "source_stage3": source_lineage,
        "config": str(Path(args.config).expanduser().resolve()),
        "stage3_config": str(Path(args.stage3_config).expanduser().resolve()),
        "data_directory": str(data_dir),
        "input_schema": ["peptide", "tcr", "label"],
        "branch": "ptcr",
        "threshold_used": args.threshold,
        "threshold_source": "fixed_external_test_threshold",
        "checkpoint_selection_from_external_data": False,
        "hyperparameter_tuning_from_external_data": False,
        "hard_performance_threshold": None,
        "inference_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "batch_sizes": {
            "encode_batch_size": args.encode_batch_size,
            "pair_batch_size": args.pair_batch_size,
        },
        "unique_sequences": {
            "peptide": len(all_peptides),
            "tcr": len(all_tcrs),
            "max_tcr_length": max_tcr_length,
        },
        "aggregate_counts": {
            "datasets": len(datasets),
            "samples": aggregate_samples,
            "positive_samples": aggregate_positive,
            "negative_samples": aggregate_negative,
            "positive_rate": aggregate_positive / aggregate_samples,
        },
        "datasets": datasets,
        "row_csv": str(row_output),
        "json_path": str(output),
        "markdown_path": str(markdown),
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"[DONE] json={output}", flush=True)
    print(f"[DONE] rows={row_output}", flush=True)
    print(f"[DONE] markdown={markdown}", flush=True)


if __name__ == "__main__":
    main()
