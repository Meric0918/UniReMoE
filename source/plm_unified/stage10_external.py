"""Shared-row external pTCR comparison utilities for Stage 10."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .metrics import compute_binary_metrics


_DELTA_METRICS = (
    "auroc",
    "aupr",
    "accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
    "loss",
)


@dataclass(frozen=True)
class ExternalPairRow:
    peptide: str
    tcr: str
    label: int
    source_row: int


@dataclass(frozen=True)
class EncodedPairSequences:
    peptide_index: dict[str, int]
    tcr_index: dict[str, int]
    peptide_hidden: np.ndarray
    peptide_mask: np.ndarray
    tcr_hidden: np.ndarray
    tcr_mask: np.ndarray


def _validate_threshold(threshold: float) -> None:
    if not 0.0 < threshold < 1.0:
        raise ValueError("External fixed threshold must be in (0, 1)")


def _binary_cross_entropy(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    epsilon = np.finfo(np.float64).eps
    scores = np.clip(
        probabilities.astype(np.float64),
        epsilon,
        1.0 - epsilon,
    )
    values = -labels * np.log(scores) - (1 - labels) * np.log(1 - scores)
    return float(values.mean())


def _probability_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    return {
        "mean": float(probabilities.mean()),
        "positive_mean": (
            float(positive.mean()) if positive.size else None
        ),
        "negative_mean": (
            float(negative.mean()) if negative.size else None
        ),
        "minimum": float(probabilities.min()),
        "maximum": float(probabilities.max()),
    }


def build_external_comparison_report(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    threshold: float,
) -> dict:
    """Build descriptive metrics and signed Stage-10 deltas."""

    _validate_threshold(threshold)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("External labels must be a non-empty 1-D array")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("External labels must be binary")
    required = {"stage3", "stage10"}
    missing = required - set(probabilities)
    if missing:
        raise ValueError(
            "External comparison requires probabilities for "
            + ", ".join(sorted(missing))
        )

    metrics: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    normalised: dict[str, np.ndarray] = {}
    for name in sorted(probabilities):
        scores = np.asarray(probabilities[name], dtype=np.float64)
        if scores.shape != labels.shape:
            raise ValueError(
                f"{name} probabilities do not match the shared row order"
            )
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"{name} probabilities must be finite in [0, 1]")
        normalised[name] = scores
        item = compute_binary_metrics(
            labels,
            scores,
            loss=_binary_cross_entropy(labels, scores),
            threshold=threshold,
        ).to_dict()
        item.update(
            {
                "positive_samples": int(labels.sum()),
                "negative_samples": int((labels == 0).sum()),
                "positive_rate": float(labels.mean()),
            }
        )
        metrics[name] = item
        summaries[name] = _probability_summary(labels, scores)

    deltas = {}
    for reference in ("stage3", "stage8"):
        if reference not in metrics:
            continue
        deltas[f"stage10_minus_{reference}"] = {
            name: float(metrics["stage10"][name] - metrics[reference][name])
            for name in _DELTA_METRICS
        }
        deltas[f"stage10_minus_{reference}"].update(
            {
                f"probability_{name}": float(
                    summaries["stage10"][name]
                    - summaries[reference][name]
                )
                for name in ("mean", "positive_mean", "negative_mean")
                if summaries["stage10"][name] is not None
                and summaries[reference][name] is not None
            }
        )
    return {
        "threshold_used": threshold,
        "threshold_source": "fixed_external_test_threshold",
        "metrics": metrics,
        "probability_summary": summaries,
        "deltas": deltas,
    }


def validate_stage8_reference_source(
    checkpoint: Mapping,
    stage3_sha256: str,
) -> None:
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in {
        "stage8_binding_model",
        "stage8_multitask_model",
    }:
        raise ValueError("Optional Stage-8 reference has an unsupported type")
    source = checkpoint.get("source_stage3")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != stage3_sha256
    ):
        raise ValueError(
            "Optional Stage-8 reference uses a different Stage-3 source"
        )


def evaluate_binding_models_on_rows(
    models: Mapping[str, torch.nn.Module],
    rows: Sequence[ExternalPairRow],
    encoded: EncodedPairSequences,
    *,
    device: torch.device,
    pair_batch_size: int,
    threshold: float,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Evaluate all models on the exact same contiguous row batches."""

    _validate_threshold(threshold)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    if not rows:
        raise ValueError("External row set is empty")
    if not {"stage3", "stage10"}.issubset(models):
        raise ValueError("External inference requires Stage-3 and Stage-10 models")
    model_names = sorted(models)
    for model in models.values():
        model.eval()
    labels = np.fromiter(
        (row.label for row in rows),
        dtype=np.int64,
        count=len(rows),
    )
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in model_names
    }
    hidden_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.inference_mode():
        for start in range(0, len(rows), pair_batch_size):
            batch_rows = rows[start : start + pair_batch_size]
            peptide_locations = np.fromiter(
                (encoded.peptide_index[row.peptide] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            tcr_locations = np.fromiter(
                (encoded.tcr_index[row.tcr] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            peptide_hidden = torch.from_numpy(
                encoded.peptide_hidden[peptide_locations]
            ).to(device, dtype=hidden_dtype, non_blocking=device.type == "cuda")
            peptide_mask = torch.from_numpy(
                encoded.peptide_mask[peptide_locations]
            ).to(device, dtype=torch.bool, non_blocking=device.type == "cuda")
            tcr_hidden = torch.from_numpy(
                encoded.tcr_hidden[tcr_locations]
            ).to(device, dtype=hidden_dtype, non_blocking=device.type == "cuda")
            tcr_mask = torch.from_numpy(
                encoded.tcr_mask[tcr_locations]
            ).to(device, dtype=torch.bool, non_blocking=device.type == "cuda")

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                for name in model_names:
                    output = models[name](
                        "ptcr",
                        peptide_hidden,
                        peptide_mask,
                        tcr_hidden,
                        tcr_mask,
                    )
                    chunks[name].append(
                        output.logits.softmax(-1)[:, 1]
                        .float()
                        .cpu()
                        .numpy()
                    )

    probabilities = {
        name: np.concatenate(chunks[name]) for name in model_names
    }
    comparison = build_external_comparison_report(
        labels,
        probabilities,
        threshold=threshold,
    )
    reports = {
        name: {
            "metrics": comparison["metrics"][name],
            "probability_summary": comparison["probability_summary"][name],
            "threshold_used": threshold,
        }
        for name in model_names
    }
    return reports, probabilities


def write_external_row_csv(
    destination: str | Path,
    rows_by_dataset: Mapping[str, Sequence[ExternalPairRow]],
    probabilities_by_dataset: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_stage8 = any(
        "stage8" in probabilities
        for probabilities in probabilities_by_dataset.values()
    )
    model_names = ["stage3"]
    if has_stage8:
        model_names.append("stage8")
    model_names.append("stage10")
    header = ["dataset", "source_row", "peptide", "tcr", "label"] + [
        f"prob_{name}" for name in model_names
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for dataset in sorted(rows_by_dataset):
            rows = rows_by_dataset[dataset]
            probabilities = probabilities_by_dataset.get(dataset)
            if probabilities is None:
                raise ValueError(f"No probabilities for dataset {dataset!r}")
            for name in model_names:
                if name not in probabilities:
                    raise ValueError(
                        f"No {name} probabilities for dataset {dataset!r}"
                    )
                if len(probabilities[name]) != len(rows):
                    raise ValueError(
                        f"{dataset}/{name} probabilities do not match rows"
                    )
            for index, row in enumerate(rows):
                writer.writerow(
                    [
                        dataset,
                        row.source_row,
                        row.peptide,
                        row.tcr,
                        row.label,
                        *[
                            float(probabilities[name][index])
                            for name in model_names
                        ],
                    ]
                )
    return path
