#!/usr/bin/env python3
"""Evaluate a Stage-8 V2 checkpoint on PH/PT/PHT benchmark test files.

PH and PT benchmark files use their corresponding pair head.  PHT benchmark
files are evaluated with all three Stage-8 heads and also receive the directed
hierarchy/PT-implication constraint report used during Stage-8 validation.
Each unique peptide, HLA, and TCR is encoded once per checkpoint; the cached
residue representations are then reused for row-level inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from plm_unified.data import OnlinePairDataset
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage8_checkpoint import (
    load_stage8_v2_checkpoint,
    load_torch_checkpoint,
    validate_stage8_v2_for_evaluation,
)
from plm_unified.stage8_config import (
    apply_stage8_overrides,
    load_stage8_config,
)
from plm_unified.stage8_evaluation import stage8_joint_score
from plm_unified.stage8_model import build_progressive_stage8_model
from plm_unified.stage12_checkpoint import validate_stage12_checkpoint
from plm_unified.stage12_config import (
    apply_stage12_overrides,
    load_stage12_config,
)
from plm_unified.trainer import resolve_device
from plm_unified.triplet_data import PHTMultitaskDataset


_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_PAIR_SPECS = (
    ("PH", "phla", "phla"),
    ("PT", "ptcr", "ptcr"),
)
_PHT_HEADS = ("phla", "ptcr", "pht")
_PHT_LABEL_ATTRIBUTES = {
    "phla": "ph",
    "ptcr": "pt",
    "pht": "pht",
}
_METRIC_NAMES = (
    "auroc",
    "aupr",
    "accuracy",
    "mcc",
    "f1",
    "precision",
    "recall",
    "specificity",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "plm_stage8.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ph-data-dir",
        type=Path,
        help="Directory containing PH test_*.csv files. Defaults to dataset/PH/benchmarks.",
    )
    parser.add_argument(
        "--pt-data-dir",
        type=Path,
        help="Directory containing PT test_*.csv files. Defaults to dataset/PT/benchmarks.",
    )
    parser.add_argument(
        "--pht-data-dir",
        type=Path,
        help="Directory containing PHT test_*.csv files. Defaults to dataset/PHT/benchmarks.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
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


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _feature_dtype(device: torch.device, mixed_precision: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def encode_unique(
    label: str,
    encoder,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a de-duplicated sequence list and move the result to CPU."""

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
            with _autocast(device, mixed_precision):
                encode = getattr(encoder, "encode", None)
                encoded = (
                    encode(sequences[start:end])
                    if callable(encode)
                    else encoder(sequences[start:end])
                )
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
                    torch.cuda.synchronize(device)
                elapsed = time.time() - started
                log(
                    f"[ENCODE {label}] {batch_id:,}/{total_batches:,} "
                    f"sequences={end:,}/{len(sequences):,} "
                    f"rate={end / max(elapsed, 1e-9):.1f}/s"
                )
            del encoded
    return hidden, mask


def finish_metrics(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
    loss_sum: float,
    samples: int,
    threshold: float,
) -> dict[str, float | int]:
    if samples < 1 or not labels or not probabilities:
        raise ValueError("Cannot compute metrics for an empty benchmark")
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


def compute_constraint_metrics(
    label_types: Sequence[str] | np.ndarray,
    *,
    phla: np.ndarray,
    ptcr: np.ndarray,
    pht: np.ndarray,
    margin: float,
) -> dict[str, Any]:
    """Compute the four directed Stage-8 implication constraint statistics."""

    types = np.asarray(label_types).astype(str)
    probabilities = {
        "phla": np.asarray(phla, dtype=np.float64),
        "ptcr": np.asarray(ptcr, dtype=np.float64),
        "pht": np.asarray(pht, dtype=np.float64),
    }
    length = len(types)
    if length == 0 or any(len(values) != length for values in probabilities.values()):
        raise ValueError("Constraint inputs must be non-empty and aligned")

    violation_masks = {
        "hierarchy_positive": (types == "111")
        & (probabilities["pht"] > probabilities["phla"] + margin),
        "hierarchy_negative": np.isin(types, ("000", "010"))
        & (probabilities["pht"] > probabilities["phla"] + margin),
        "pt_positive": (types == "111")
        & (probabilities["pht"] > probabilities["ptcr"] + margin),
        "pt_negative": np.isin(types, ("000", "100"))
        & (probabilities["pht"] > probabilities["ptcr"] + margin),
    }
    eligible = {
        "hierarchy_positive": types == "111",
        "hierarchy_negative": np.isin(types, ("000", "010")),
        "pt_positive": types == "111",
        "pt_negative": np.isin(types, ("000", "100")),
    }
    violation_counts = {
        name: int(mask.sum()) for name, mask in violation_masks.items()
    }
    eligible_counts = {
        name: int(mask.sum()) for name, mask in eligible.items()
    }
    violations = {
        name: float(violation_counts[name] / max(1, eligible_counts[name]))
        for name in violation_masks
    }
    violations["overall"] = float(
        sum(violation_counts.values())
        / max(1, sum(eligible_counts.values()))
    )

    hinge_values = {
        "hierarchy_positive": np.maximum(
            probabilities["pht"] - probabilities["phla"] - margin, 0.0
        ) ** 2,
        "hierarchy_negative": np.maximum(
            probabilities["pht"] - probabilities["phla"] - margin, 0.0
        ) ** 2,
        "pt_positive": np.maximum(
            probabilities["pht"] - probabilities["ptcr"] - margin, 0.0
        ) ** 2,
        "pt_negative": np.maximum(
            probabilities["pht"] - probabilities["ptcr"] - margin, 0.0
        ) ** 2,
    }
    constraint_components = {
        name: float(values[eligible[name]].mean())
        if eligible[name].any()
        else 0.0
        for name, values in hinge_values.items()
    }
    constraint_components["overall"] = float(
        sum(
            constraint_components[name] * eligible_counts[name]
            for name in hinge_values
        )
        / max(1, sum(eligible_counts.values()))
    )

    by_type: dict[str, dict[str, float | int]] = {}
    for label_type in ("111", "100", "010", "000"):
        selected = types == label_type
        if not selected.any():
            continue
        by_type[label_type] = {
            "samples": int(selected.sum()),
            "mean_phla_probability": float(probabilities["phla"][selected].mean()),
            "mean_ptcr_probability": float(probabilities["ptcr"][selected].mean()),
            "mean_pht_probability": float(probabilities["pht"][selected].mean()),
        }
    return {
        "margin": float(margin),
        "eligible": eligible_counts,
        "violation_counts": violation_counts,
        "violations": violations,
        "constraint_components": constraint_components,
        "by_type": by_type,
    }


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
    mixed_precision: str,
    row_batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    criterion = nn.CrossEntropyLoss()
    feature_dtype = _feature_dtype(device, mixed_precision)
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
                (row[0] for row in batch_rows), dtype=object, count=len(batch_rows)
            )
            p_locations = np.fromiter(
                (peptide_index[value] for value in p_locations),
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
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[p_locations]
            ).to(device, dtype=torch.bool, non_blocking=device.type == "cuda")
            receptor = torch.from_numpy(receptor_hidden[r_locations]).to(
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            receptor_mask_batch = torch.from_numpy(
                receptor_mask[r_locations]
            ).to(device, dtype=torch.bool, non_blocking=device.type == "cuda")
            batch_labels = torch.from_numpy(batch_labels_cpu).to(
                device, dtype=torch.long, non_blocking=device.type == "cuda"
            )
            with _autocast(device, mixed_precision):
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
                    torch.cuda.synchronize(device)
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
    mixed_precision: str,
    row_batch_size: int,
    thresholds: dict[str, float],
    margin: float,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    feature_dtype = _feature_dtype(device, mixed_precision)
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
                device, dtype=feature_dtype, non_blocking=device.type == "cuda"
            )
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[p_locations]
            ).to(device, dtype=torch.bool, non_blocking=device.type == "cuda")
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
                task: np.fromiter(
                    (
                        getattr(record, _PHT_LABEL_ATTRIBUTES[task])
                        for record in batch_records
                    ),
                    dtype=np.int64,
                    count=len(batch_records),
                )
                for task in _PHT_HEADS
            }
            batch_labels = {
                task: torch.from_numpy(values).to(
                    device, dtype=torch.long, non_blocking=device.type == "cuda"
                )
                for task, values in batch_labels_cpu.items()
            }
            with _autocast(device, mixed_precision):
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
                    "pht": output.pht_logits,
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
                labels[task].append(batch_labels_cpu[task])
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.time() - started
                log(
                    f"[EVAL PHT] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(records):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s"
                )

    metrics = {
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
    all_probabilities = {
        task: np.concatenate(probabilities[task]).astype(np.float64, copy=False)
        for task in _PHT_HEADS
    }
    all_types = np.asarray([record.label_type for record in records])
    return {
        "metrics": metrics,
        "constraints": compute_constraint_metrics(
            all_types,
            phla=all_probabilities["phla"],
            ptcr=all_probabilities["ptcr"],
            pht=all_probabilities["pht"],
            margin=margin,
        ),
    }


def _sorted_sequences(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value))


def load_benchmark_data(
    root: Path,
    config,
    limit: int | None = None,
    *,
    ph_data_dir: Path | None = None,
    pt_data_dir: Path | None = None,
    pht_data_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[str]]:
    """Read selected PH/PT/PHT ``test_*.csv`` files and collect unique sequences."""

    benchmarks: dict[str, dict[str, Any]] = {}
    peptides: set[str] = set()
    hlas: set[str] = set()
    tcrs: set[str] = set()
    data_dirs = {
        "PH": (ph_data_dir or root / "dataset" / "PH" / "benchmarks").resolve(),
        "PT": (pt_data_dir or root / "dataset" / "PT" / "benchmarks").resolve(),
        "PHT": (pht_data_dir or root / "dataset" / "PHT" / "benchmarks").resolve(),
    }

    for family, task, head in _PAIR_SPECS:
        paths = sorted(data_dirs[family].glob("test_*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No benchmark test files found for {family} in {data_dirs[family]}"
            )
        receptor_limit = (
            config.model.hla_max_length
            if task == "phla"
            else config.model.tcr_max_length
        )
        for path in paths:
            key = f"{family}/{path.stem}"
            dataset = OnlinePairDataset(
                path,
                task,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=limit,
            )
            rows = dataset.rows
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
            benchmarks[key] = {
                "family": family,
                "benchmark": path.stem,
                "path": path,
                "task": task,
                "head": head,
                "rows": rows,
                "raw_rows": int(dataset.filter_stats["total"]),
                "filter_stats": dict(dataset.filter_stats),
            }
            peptides.update(row[0] for row in rows)
            if task == "phla":
                hlas.update(row[1] for row in rows)
            else:
                tcrs.update(row[1] for row in rows)
            log(
                f"[DATA] {key} rows={len(rows):,} "
                f"filter_stats={dataset.filter_stats}"
            )

    pht_paths = sorted(data_dirs["PHT"].glob("test_*.csv"))
    if not pht_paths:
        raise FileNotFoundError(
            f"No PHT benchmark test files found in {data_dirs['PHT']}"
        )
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
        records = dataset.records if limit is None else dataset.records[:limit]
        if not records:
            raise ValueError(f"No usable rows found in {path}")
        benchmarks[key] = {
            "family": "PHT",
            "benchmark": path.stem,
            "path": path,
            "task": "pht",
            "head": ",".join(_PHT_HEADS),
            "records": records,
            "raw_rows": int(dataset.filter_stats["total"]),
            "filter_stats": dict(dataset.filter_stats),
        }
        peptides.update(record.peptide for record in records)
        hlas.update(record.hla for record in records)
        tcrs.update(record.tcr for record in records)
        log(
            f"[DATA] {key} rows={len(records):,} "
            f"filter_stats={dataset.filter_stats}"
        )

    all_peptides = _sorted_sequences(peptides)
    all_hlas = _sorted_sequences(hlas)
    all_tcrs = _sorted_sequences(tcrs)
    for label, values, maximum in (
        ("peptide", all_peptides, config.model.peptide_max_length),
        ("HLA", all_hlas, config.model.hla_max_length),
        ("TCR", all_tcrs, config.model.tcr_max_length),
    ):
        if any(len(value) > maximum for value in values):
            raise ValueError(f"{label} exceeds configured length limit")
    log(
        f"[UNIQUE] peptide={len(all_peptides):,} "
        f"HLA={len(all_hlas):,} TCR={len(all_tcrs):,}"
    )
    return benchmarks, all_peptides, all_hlas, all_tcrs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_link(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def fixed_metric_rows(report: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for key, item in sorted(report.get("results", {}).items()):
        if item.get("family") == "PHT":
            for head in _PHT_HEADS:
                rows.append((key, head, item["metrics"][head]["fixed_0.5"]))
        else:
            rows.append((key, str(item["head"]), item["metrics"]["fixed_0.5"]))
    return rows


def _macro_rows(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, head, metrics in fixed_metric_rows(report):
        family = key.split("/", 1)[0]
        grouped.setdefault((family, head), []).append(metrics)
    return {
        group: {
            name: float(np.mean([float(item[name]) for item in values]))
            for name in (*_METRIC_NAMES[:5], "performance_avg")
        }
        for group, values in grouped.items()
    }


def make_markdown(report: Mapping[str, Any], markdown_path: Path) -> str:
    """Render one self-contained Chinese summary for a Stage-8 checkpoint."""

    checkpoint = Path(str(report["checkpoint"]))
    config = Path(str(report["config"]))
    json_path = Path(str(report["outputs"]["json"]))
    results = report["results"]
    fixed_rows = fixed_metric_rows(report)
    condition = report.get("condition", "unknown")
    stage_label = report.get("stage_label", "Stage 8")
    artifact_role = report.get("artifact_role", "best")
    checkpoint_role_label = f"{artifact_role} checkpoint"
    checkpoint_metrics = report.get("checkpoint_validation_metrics", {})
    data_directories = report.get("data_directories", {})
    ph_data_dir = data_directories.get("PH", "dataset/PH/benchmarks")
    pt_data_dir = data_directories.get("PT", "dataset/PT/benchmarks")
    pht_data_dir = data_directories.get("PHT", "dataset/PHT/benchmarks")
    lines = [
        f"# {stage_label} `{condition}` 在 PH/PT/PHT test 数据上的评测结果",
        "",
        "## 1. 结论摘要",
        "",
        (
            f"本报告使用 {stage_label} 的 {checkpoint_role_label}，对 "
            f"`{ph_data_dir}`、`{pt_data_dir}` 和 `{pht_data_dir}` 下全部 "
            "`test_*.csv` 完成测试。"
            f"共评测 **{report['totals']['samples']:,}** 条有效样本。"
        ),
        "",
        "- PH benchmark 使用 `phla` head；PT benchmark 使用 `ptcr` head。",
        "- PHT benchmark 同时报告 `phla`、`ptcr` 和 `pht` 三个 head。",
        "- 主表统一使用固定阈值 `0.5`；checkpoint 在验证集选择的 PHT 阈值仅作补充，不使用 test 标签调参。",
        "- PHT 约束统计使用训练配置中的 margin，仅用于报告测试时的约束违反率和 hinge 强度。",
        "",
        "## 2. 评测对象与设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| Condition | `{condition}` |",
        f"| Checkpoint | [`{checkpoint.name}`]({relative_link(checkpoint, markdown_path.parent)}) |",
        f"| Checkpoint 类型 | `{report.get('checkpoint_type')}`，format `{report.get('format_version')}` |",
        f"| Checkpoint 角色 | `{artifact_role}` |",
        f"| Checkpoint SHA-256 | `{report.get('checkpoint_sha256')}` |",
        f"| {artifact_role.capitalize()} {stage_label.replace(' ', '')}B round | `{report.get('best_round')}`；selection score `{report.get('best_round_score')}` |",
        f"| Validation joint score | `{report.get('validation_joint_score')}` |",
        f"| 配置 | [`{config.name}`]({relative_link(config, markdown_path.parent)}) |",
        f"| PH 数据目录 | `{ph_data_dir}` |",
        f"| PT 数据目录 | `{pt_data_dir}` |",
        f"| PHT 数据目录 | `{pht_data_dir}` |",
        f"| 设备 | `{report.get('device')}` |",
        f"| 精度 | `{report.get('mixed_precision')}` autocast |",
        f"| unique-sequence 编码 batch | `{report.get('encode_batch_size')}` |",
        f"| 行级推理 batch | `{report.get('row_batch_size')}` |",
        f"| 固定分类阈值 | `{report.get('fixed_threshold')}` |",
        f"| checkpoint PHT 验证阈值 | `{report.get('checkpoint_selected_threshold')}` |",
        f"| Constraint margin | `{report.get('constraint_margin')}` |",
        "",
        f"所有 head 均从同一个 {stage_label} full online model 产生；PH/PT pair 只传入对应的 peptide-receptor 表示，PHT 则一次前向产生三个 head 的输出。",
        "",
        "## 3. 数据审计",
        "",
        "| 数据集 | 原始行数 | 纳入评测 | 过滤/去重差额 | PH 正/负 | PT 正/负 | PHT 正/负 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(results.items()):
        raw_rows = int(item["raw_rows"])
        samples = int(item["samples"])
        if item.get("family") == "PHT":
            counts = item["label_counts"]
            ph = f"{counts['phla']['positive']:,}/{counts['phla']['negative']:,}"
            pt = f"{counts['ptcr']['positive']:,}/{counts['ptcr']['negative']:,}"
            pht = f"{counts['pht']['positive']:,}/{counts['pht']['negative']:,}"
        else:
            metric = item["metrics"]["fixed_0.5"]
            counts = f"{metric['positive_samples']:,}/{metric['negative_samples']:,}"
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
    for key, head, metrics in fixed_rows:
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
            "## 5. benchmark 宏平均",
            "",
            "同一数据族和 head 内为各 benchmark 的简单算术平均，不是拼接样本后的 pooled metric。",
            "",
            "| 数据族 / Head | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (family, head), averages in sorted(_macro_rows(report).items()):
        lines.append(
            f"| `{family}` / `{head}` | {fmt(averages['auroc'])} | "
            f"{fmt(averages['aupr'])} | {fmt(averages['accuracy'])} | "
            f"{fmt(averages['mcc'])} | {fmt(averages['f1'])} | "
            f"{fmt(averages['performance_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## 6. PHT checkpoint 阈值补充",
            "",
            (
                "checkpoint 在 validation 上为 PHT head 记录的阈值为 "
                f"`{report.get('checkpoint_selected_threshold')}`。下表与固定阈值 "
                "`0.5` 对比；该阈值未由任何 test 标签重新选择。"
            ),
            "",
            "| Benchmark | Head | 阈值 | Accuracy | MCC | F1 | Precision | Recall | Specificity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in sorted(results.items()):
        if item.get("family") != "PHT":
            continue
        for threshold_name, label in (
            ("fixed_0.5", "0.5"),
            ("checkpoint_selected", str(report.get("checkpoint_selected_threshold"))),
        ):
            metrics = item["metrics"]["pht"][threshold_name]
            lines.append(
                f"| `{key}` | `pht` | `{label}` | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['mcc'])} | {fmt(metrics['f1'])} | "
                f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
                f"{fmt(metrics['specificity'])} |"
            )

    lines.extend(
        [
            "",
            "## 7. PHT 约束统计",
            "",
            "违反率的分母只包含该约束适用的 label type；`hierarchy_*` 比较 `pht` 与 `phla`，`pt_*` 比较 `pht` 与 `ptcr`。",
            "",
            "| Benchmark | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall | Hinge overall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in sorted(results.items()):
        if item.get("family") != "PHT":
            continue
        constraints = item["constraints"]
        violations = constraints["violations"]
        components = constraints["constraint_components"]
        lines.append(
            f"| `{key}` | {fmt(violations['hierarchy_positive'])} | "
            f"{fmt(violations['hierarchy_negative'])} | "
            f"{fmt(violations['pt_positive'])} | {fmt(violations['pt_negative'])} | "
            f"{fmt(violations['overall'])} | {fmt(components['overall'])} |"
        )

    best = max(fixed_rows, key=lambda row: float(row[2]["performance_avg"]))
    pht_rows = [row for row in fixed_rows if row[1] == "pht"]
    pht_best = max(pht_rows, key=lambda row: float(row[2]["performance_avg"]))
    lines.extend(
        [
            "",
            "## 8. 结果说明与可复核产物",
            "",
            (
                f"- 全部 benchmark/head 中 performance average 最高的是 `{best[0]}` 的 "
                f"`{best[1]}` head：`{best[2]['performance_avg']:.6f}`。"
            ),
            (
                f"- PHT head 中 performance average 最高的是 `{pht_best[0]}`："
                f"`{pht_best[2]['performance_avg']:.6f}`。"
            ),
            "- `test_unseen_pair` 样本量较小，指标方差可能较大，应结合样本数和混淆矩阵解读。",
            f"- checkpoint validation metrics 摘要：`{json.dumps(checkpoint_metrics, ensure_ascii=False, sort_keys=True)}`。",
            "- 本次没有重新训练、修改 checkpoint，且没有使用 test 标签进行阈值选择或模型选择。",
            "- 每个 checkpoint 的 unique peptide/HLA/TCR 分别只编码一次，再复用 residue-level 表示完成行级推理。",
            f"- 原始 JSON：[`{json_path.name}`]({relative_link(json_path, markdown_path.parent)})",
            f"- 本报告：[`{markdown_path.name}`]({relative_link(markdown_path, markdown_path.parent)})",
            "",
        ]
    )
    return "\n".join(lines)


def _report_metric_map(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (key, head): metrics
        for key, head, metrics in fixed_metric_rows(report)
    }


def _signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def make_comparison_markdown(
    reports: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> str:
    """Render a cross-condition comparison from per-model JSON reports."""

    if not reports:
        raise ValueError("At least one Stage-8 report is required")
    conditions = list(reports)
    metric_maps = {condition: _report_metric_map(report) for condition, report in reports.items()}
    keys = sorted(set().union(*(mapping for mapping in metric_maps.values())))
    macro_maps = {condition: _macro_rows(report) for condition, report in reports.items()}

    overall_scores = {
        condition: float(
            np.mean([
                float(metrics["performance_avg"])
                for metrics in metric_maps[condition].values()
            ])
        )
        for condition in conditions
    }
    best_condition = max(overall_scores, key=overall_scores.get)
    baseline = "baseline" if "baseline" in reports else conditions[0]

    lines = [
        "# Stage 8 baseline/full benchmark test 综合对比",
        "",
        "本报告对 `baseline` 和 `full` 两种 Stage-8 V2 best checkpoint 结果进行横向比较。主比较全部使用固定阈值 `0.5`；PHT validation-selected threshold 不参与主排名。",
        "",
        "## 1. 模型与总体结果",
        "",
        "| Condition | Checkpoint | Best round | Validation joint | Test macro performance avg | Test rows | 独立报告 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for condition in conditions:
        report = reports[condition]
        report_path = report.get("outputs", {}).get("markdown")
        report_link = (
            f"[`{Path(str(report_path)).name}`]("
            f"{relative_link(Path(str(report_path)), output_path.parent)})"
            if report_path
            else "—"
        )
        lines.append(
            f"| `{condition}` | `{Path(str(report.get('checkpoint', ''))).name}` | "
            f"{report.get('best_round', '—')} | {fmt(report.get('validation_joint_score'))} | "
            f"{fmt(overall_scores[condition])} | "
            f"{int(report.get('totals', {}).get('samples', 0)):,} | {report_link} |"
        )
    lines.extend(
        [
            "",
            f"按所有 benchmark/head 的等权 test `performance_avg`，当前最佳/最高的是 `{best_condition}`（{overall_scores[best_condition]:.6f}）。这只是测试集描述性汇总，不替代 checkpoint 原有的 validation 选择分数。",
            "",
            "## 2. 数据一致性",
            "",
            "| Condition | 原始行数 | 有效行数 | 过滤/去重 | peptide | HLA | TCR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        report = reports[condition]
        totals = report.get("totals", {})
        unique = report.get("unique_sequences", {})
        lines.append(
            f"| `{condition}` | {int(totals.get('raw_rows', 0)):,} | "
            f"{int(totals.get('samples', 0)):,} | {int(totals.get('filtered', 0)):,} | "
            f"{int(unique.get('peptide', 0)):,} | {int(unique.get('hla', 0)):,} | "
            f"{int(unique.get('tcr', 0)):,} |"
        )

    lines.extend(
        [
            "",
            "## 3. 各 benchmark/head 主结果",
            "",
            "单元格格式为 `AUROC / AUPR / Accuracy / MCC / F1 / Performance avg`。同一行中数值越高越好；loss、混淆矩阵和阈值补充见各 condition 的独立报告。",
            "",
            "| Dataset / Head | " + " | ".join(f"{condition}" for condition in conditions) + " |",
            "|---|" + "---:|" * len(conditions),
        ]
    )
    for key, head in keys:
        cells = []
        for condition in conditions:
            metrics = metric_maps[condition].get((key, head))
            if metrics is None:
                cells.append("—")
            else:
                cells.append(
                    " / ".join(
                        fmt(metrics[name], 4)
                        for name in (
                            "auroc",
                            "aupr",
                            "accuracy",
                            "mcc",
                            "f1",
                            "performance_avg",
                        )
                    )
                )
        lines.append(f"| `{key}` / `{head}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 4. 数据族/head 宏平均",
            "",
            "宏平均在每个 condition 内独立计算；`Δ` 部分为相对 baseline 的有符号差值。",
            "",
            "| Dataset family / Head | Condition | AUROC | AUPR | Accuracy | MCC | F1 | Performance avg |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in sorted(set().union(*(macro for macro in macro_maps.values()))):
        for condition in conditions:
            values = macro_maps[condition].get(group)
            if values is None:
                continue
            lines.append(
                f"| `{group[0]}` / `{group[1]}` | `{condition}` | "
                f"{fmt(values['auroc'])} | {fmt(values['aupr'])} | "
                f"{fmt(values['accuracy'])} | {fmt(values['mcc'])} | "
                f"{fmt(values['f1'])} | {fmt(values['performance_avg'])} |"
            )

    lines.extend(
        [
            "",
            f"### 相对 `{baseline}` 的差值",
            "",
            "| Dataset family / Head | Condition | ΔAUROC | ΔAUPR | ΔAccuracy | ΔMCC | ΔF1 | ΔPerformance avg |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    baseline_macro = macro_maps[baseline]
    for group in sorted(set().union(*(macro for macro in macro_maps.values()))):
        for condition in conditions:
            if condition == baseline or group not in baseline_macro or group not in macro_maps[condition]:
                continue
            current = macro_maps[condition][group]
            reference = baseline_macro[group]
            lines.append(
                f"| `{group[0]}` / `{group[1]}` | `{condition}` | "
                f"{_signed(current['auroc'] - reference['auroc'])} | "
                f"{_signed(current['aupr'] - reference['aupr'])} | "
                f"{_signed(current['accuracy'] - reference['accuracy'])} | "
                f"{_signed(current['mcc'] - reference['mcc'])} | "
                f"{_signed(current['f1'] - reference['f1'])} | "
                f"{_signed(current['performance_avg'] - reference['performance_avg'])} |"
            )

    lines.extend(
        [
            "",
            "## 5. PHT 约束对比",
            "",
            "以下为所有 PHT benchmark 的样本加权平均约束统计；分母仍按各约束适用的 label type 计算。",
            "",
            "| Condition | Hierarchy+ | Hierarchy− | PT+ | PT− | Overall violation | Hinge overall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        weighted: dict[str, float] = {}
        denominators: dict[str, int] = {}
        hinge_weighted: dict[str, float] = {}
        for item in reports[condition].get("results", {}).values():
            if item.get("family") != "PHT" or "constraints" not in item:
                continue
            constraints = item["constraints"]
            for name, denominator in constraints["eligible"].items():
                weighted[name] = weighted.get(name, 0.0) + constraints["violations"][name] * denominator
                denominators[name] = denominators.get(name, 0) + denominator
            for name, value in constraints["constraint_components"].items():
                if name == "overall":
                    continue
                hinge_weighted[name] = hinge_weighted.get(name, 0.0) + value * constraints["eligible"][name]
        directional_rates = {
            name: weighted.get(name, 0.0) / max(1, denominators.get(name, 0))
            for name in (
                "hierarchy_positive",
                "hierarchy_negative",
                "pt_positive",
                "pt_negative",
            )
        }
        total_violations = sum(weighted.values())
        total_eligible = sum(denominators.values())
        total_hinge = sum(hinge_weighted.values()) / max(1, total_eligible)
        lines.append(
            f"| `{condition}` | {fmt(directional_rates['hierarchy_positive'])} | "
            f"{fmt(directional_rates['hierarchy_negative'])} | "
            f"{fmt(directional_rates['pt_positive'])} | {fmt(directional_rates['pt_negative'])} | "
            f"{fmt(total_violations / max(1, total_eligible))} | {fmt(total_hinge)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 综合解读",
            "",
            f"- 按 test 宏观 performance average，最佳/最高 condition 为 `{best_condition}`；请同时查看 validation joint score，避免把 test 集结果误当成训练期模型选择依据。",
            "- `full` 同时启用 hierarchy 和 PT implication 两类约束；约束降低并不自动等价于分类效果提升。",
            "- PHT benchmark 的三个 head 共享同一组 encoder 和下游模块前向；PH/PT 的 pair benchmark 只评估相应 head，因此应优先比较同一 benchmark/head 行。",
            "- 各独立报告保存了 checkpoint SHA-256、过滤统计、混淆矩阵、验证阈值补充和 JSON 原始结果，可用于复核。",
            "",
        ]
    )
    return "\n".join(lines)


def _checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    training_state = checkpoint.get("training_state", {})
    metrics = checkpoint.get("metrics", {})
    if checkpoint.get("checkpoint_type") == "stage12_online_model":
        validation = metrics.get("validation", {})
        return {
            "best_round": training_state.get("stage12b_round"),
            "best_round_score": validation.get("joint_score"),
            "validation_joint_score": validation.get("joint_score"),
            "checkpoint_selected_threshold": 0.5,
            "validation_metrics": validation,
            "training_state": {
                key: training_state.get(key)
                for key in ("stage12b_round", "global_step", "best_joint_score_seen")
                if key in training_state
            },
        }
    validation_metrics = metrics.get("validation_metrics", {})
    return {
        "best_round": training_state.get("stage8b_round"),
        "best_round_score": training_state.get("best_joint"),
        "validation_joint_score": metrics.get("validation_joint_score"),
        "checkpoint_selected_threshold": metrics.get("selected_pht_threshold"),
        "validation_metrics": validation_metrics,
        "training_state": {
            key: training_state.get(key)
            for key in ("stage8b_round", "global_step", "best_joint")
            if key in training_state
        },
    }


def _stage_label(checkpoint: Mapping[str, Any]) -> str:
    return (
        "Stage 12"
        if checkpoint.get("checkpoint_type") == "stage12_online_model"
        else "Stage 8"
    )


def _validate_checkpoint_for_evaluation(checkpoint: Mapping[str, Any]) -> None:
    """Validate an evaluation-safe checkpoint role for either Stage 8 or 12."""

    if checkpoint.get("checkpoint_type") == "stage12_online_model":
        artifact_role = checkpoint.get("artifact_role")
        if artifact_role not in {"best", "last"}:
            raise ValueError(
                "Stage12 benchmark evaluation requires a best or last checkpoint"
            )
        validate_stage12_checkpoint(checkpoint, artifact_role=artifact_role)
        return
    raise ValueError("Stage8 benchmark validation must use the Stage8 evaluation path")


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
    def resolve_data_dir(value: Path | None, default: Path) -> Path:
        path = default if value is None else value.expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    ph_data_dir = resolve_data_dir(
        args.ph_data_dir,
        root / "dataset" / "PH" / "benchmarks",
    )
    pt_data_dir = resolve_data_dir(
        args.pt_data_dir,
        root / "dataset" / "PT" / "benchmarks",
    )
    pht_data_dir = resolve_data_dir(
        args.pht_data_dir,
        root / "dataset" / "PHT" / "benchmarks",
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    for family, data_dir in (
        ("PH", ph_data_dir),
        ("PT", pt_data_dir),
        ("PHT", pht_data_dir),
    ):
        if not data_dir.is_dir():
            raise FileNotFoundError(f"{family} data directory not found: {data_dir}")

    checkpoint = load_torch_checkpoint(checkpoint_path)
    is_stage12 = checkpoint.get("checkpoint_type") == "stage12_online_model"
    if is_stage12:
        config = load_stage12_config(config_path)
        config.paths.data_root = root / "dataset"
        apply_stage12_overrides(config, device=args.device, num_workers=0)
        _validate_checkpoint_for_evaluation(checkpoint)
    else:
        config = load_stage8_config(config_path)
        config.paths.data_root = root / "dataset"
        apply_stage8_overrides(config, device=args.device, num_workers=0)
        validate_stage8_v2_for_evaluation(checkpoint, config=config)
    device = resolve_device(args.device)
    mixed_precision = config.stage8.mixed_precision

    log(f"[INIT] checkpoint={checkpoint_path}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        log(f"[INIT] device={device} gpu={gpu_name}")
    else:
        log(f"[INIT] device={device}")

    summary = _checkpoint_summary(checkpoint)
    checkpoint_selected_threshold = float(summary["checkpoint_selected_threshold"])
    if not 0 <= checkpoint_selected_threshold <= 1:
        raise ValueError("Checkpoint selected PHT threshold is outside [0, 1]")
    condition = "full" if is_stage12 else str(checkpoint.get("condition"))
    if condition not in {"baseline", "full"}:
        raise ValueError(f"Unexpected checkpoint condition: {condition!r}")

    benchmarks, all_peptides, all_hlas, all_tcrs = load_benchmark_data(
        root,
        config,
        args.limit,
        ph_data_dir=ph_data_dir,
        pt_data_dir=pt_data_dir,
        pht_data_dir=pht_data_dir,
    )
    model = build_progressive_stage8_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        current_fingerprint = current_metadata[entity].get("checkpoint_fingerprint")
        if saved_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(f"Checkpoint base-PLM fingerprint mismatch for {entity}")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint["model"]
    model.set_evaluation()
    log(f"[INIT] model loaded and {_stage_label(checkpoint)} checkpoint contract verified")

    peptide_hidden, peptide_mask = encode_unique(
        "peptide",
        model.peptide_encoder,
        all_peptides,
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
        all_hlas,
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
        all_tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
        mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    hla_index = {value: index for index, value in enumerate(all_hlas)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}

    results: dict[str, dict[str, Any]] = {}
    for key, item in sorted(benchmarks.items()):
        if item["family"] == "PHT":
            continue
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
            {
                "fixed_0.5": 0.5,
                "checkpoint_selected": checkpoint_selected_threshold,
            },
            config.stage8.constraint_margin,
        )
        label_counts = {}
        for task, attribute in _PHT_LABEL_ATTRIBUTES.items():
            positives = sum(int(getattr(record, attribute)) for record in item["records"])
            label_counts[task] = {
                "positive": positives,
                "negative": len(item["records"]) - positives,
            }
        fixed_metrics = {
            task: evaluated["metrics"][task]["fixed_0.5"]
            for task in _PHT_HEADS
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
            "constraints": evaluated["constraints"],
            "joint_score_fixed_0.5": stage8_joint_score(
                fixed_metrics,
                config.stage8.selection_weights,
            ),
        }
        log(
            f"[DONE] {key} "
            f"joint_score_fixed_0.5={results[key]['joint_score_fixed_0.5']:.6f}"
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
    def report_path(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    report = {
        "status": "complete",
        "report_timestamp": datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"),
        "stage_label": _stage_label(checkpoint),
        "condition": condition,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "format_version": checkpoint.get("format_version"),
        "artifact_role": checkpoint.get("artifact_role"),
        "config": str(config_path),
        "data_directories": {
            "PH": report_path(ph_data_dir),
            "PT": report_path(pt_data_dir),
            "PHT": report_path(pht_data_dir),
        },
        "device": str(device),
        "mixed_precision": mixed_precision,
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "fixed_threshold": 0.5,
        "checkpoint_selected_threshold": checkpoint_selected_threshold,
        "constraint_margin": config.stage8.constraint_margin,
        "selection_weights": list(config.stage8.selection_weights),
        "best_round": summary["best_round"],
        "best_round_score": summary["best_round_score"],
        "validation_joint_score": summary["validation_joint_score"],
        "checkpoint_validation_metrics": summary["validation_metrics"],
        "checkpoint_training_state": summary["training_state"],
        "effective_data": checkpoint.get("effective_data"),
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
            f"PH rows from {report_path(ph_data_dir)} were evaluated with the {_stage_label(checkpoint)} phla head.",
            f"PT rows from {report_path(pt_data_dir)} were evaluated with the {_stage_label(checkpoint)} ptcr head.",
            f"PHT rows from {report_path(pht_data_dir)} were evaluated with phla, ptcr, and pht heads.",
            "Fixed threshold 0.5 was used for the main table; the checkpoint-selected threshold was evaluated only as a PHT supplement.",
            "Each unique peptide, HLA, and TCR was encoded once per checkpoint; no Stage-1 embedding cache was used.",
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
                "condition": condition,
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
