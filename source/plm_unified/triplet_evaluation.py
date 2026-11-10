"""Metrics, threshold selection, and inference helpers for Stage 3."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .metrics import BinaryMetrics, compute_binary_metrics
from .triplet_model import ProgressiveTripletFineTuningModel

EvaluationTask = Literal["phla", "ptcr", "triplet"]


@dataclass
class PredictionResult:
    metrics: BinaryMetrics
    labels: np.ndarray
    probabilities: np.ndarray
    source_rows: np.ndarray | None = None


def select_mcc_threshold(
    labels,
    probabilities,
) -> tuple[float, BinaryMetrics]:
    """Choose validation threshold by MCC, then F1, then distance to 0.5."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and probabilities must be aligned 1-D arrays")
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both classes")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_score = y_score[order]
    sorted_label = y_true[order]
    positives = int(sorted_label.sum())
    negatives = len(sorted_label) - positives
    cumulative_tp = np.cumsum(sorted_label)
    cumulative_fp = np.cumsum(1 - sorted_label)
    last_in_group = np.r_[
        sorted_score[:-1] != sorted_score[1:],
        True,
    ]
    candidate_indices = np.flatnonzero(last_in_group)

    best_key: tuple[float, float, float, float] | None = None
    best_threshold = 0.5
    for index in candidate_indices:
        tp = int(cumulative_tp[index])
        fp = int(cumulative_fp[index])
        fn = positives - tp
        tn = negatives - fp
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (
            (tp * tn - fp * fn) / math.sqrt(denominator)
            if denominator
            else 0.0
        )
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
        threshold = float(sorted_score[index])
        key = (mcc, f1, -abs(threshold - 0.5), -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold

    # Explicitly consider 0.5 because it can sit between two observed scores.
    predictions = y_score >= 0.5
    tp = int(np.sum(predictions & (y_true == 1)))
    fp = int(np.sum(predictions & (y_true == 0)))
    fn = int(np.sum(~predictions & (y_true == 1)))
    tn = int(np.sum(~predictions & (y_true == 0)))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (
        (tp * tn - fp * fn) / math.sqrt(denominator)
        if denominator
        else 0.0
    )
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    key_at_half = (mcc, f1, 0.0, -0.5)
    if best_key is None or key_at_half > best_key:
        best_threshold = 0.5

    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=0.0,
        threshold=best_threshold,
    )
    return best_threshold, metrics


def select_composite_threshold(
    labels,
    probabilities,
) -> tuple[float, BinaryMetrics, float]:
    """Choose a threshold from normalized MCC, F1, and balanced accuracy."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and probabilities must be aligned 1-D arrays")
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both classes")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_score = y_score[order]
    sorted_label = y_true[order]
    positives = int(sorted_label.sum())
    negatives = len(sorted_label) - positives
    cumulative_tp = np.cumsum(sorted_label, dtype=np.float64)
    cumulative_fp = np.cumsum(1 - sorted_label, dtype=np.float64)
    last_in_group = np.r_[sorted_score[:-1] != sorted_score[1:], True]
    candidate_indices = np.flatnonzero(last_in_group)
    thresholds = sorted_score[candidate_indices]
    tp = cumulative_tp[candidate_indices]
    fp = cumulative_fp[candidate_indices]
    fn = positives - tp
    tn = negatives - fp

    half_predictions = y_score >= 0.5
    half_tp = float(np.sum(half_predictions & (y_true == 1)))
    half_fp = float(np.sum(half_predictions & (y_true == 0)))
    half_fn = float(np.sum(~half_predictions & (y_true == 1)))
    half_tn = float(np.sum(~half_predictions & (y_true == 0)))
    thresholds = np.r_[thresholds, 0.5]
    tp = np.r_[tp, half_tp]
    fp = np.r_[fp, half_fp]
    fn = np.r_[fn, half_fn]
    tn = np.r_[tn, half_tn]

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.zeros_like(tp)
    valid = denominator > 0
    mcc[valid] = (
        (tp[valid] * tn[valid] - fp[valid] * fn[valid])
        / np.sqrt(denominator[valid])
    )
    f1_denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        f1_denominator,
        out=np.zeros_like(tp),
        where=f1_denominator > 0,
    )
    sensitivity = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) > 0,
    )
    specificity = np.divide(
        tn,
        tn + fp,
        out=np.zeros_like(tp),
        where=(tn + fp) > 0,
    )
    balanced_accuracy = (sensitivity + specificity) / 2.0
    composite = ((mcc + 1.0) / 2.0 + f1 + balanced_accuracy) / 3.0

    best_score = float(np.max(composite))
    candidates = np.flatnonzero(
        np.isclose(composite, best_score, rtol=0.0, atol=1e-12)
    )
    distances = np.abs(thresholds[candidates] - 0.5)
    closest = candidates[np.isclose(distances, np.min(distances), atol=1e-12)]
    best_index = int(closest[np.argmin(thresholds[closest])])
    best_threshold = float(thresholds[best_index])
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=0.0,
        threshold=best_threshold,
    )
    return best_threshold, metrics, float(composite[best_index])


@torch.inference_mode()
def evaluate_stage3_model(
    model: ProgressiveTripletFineTuningModel,
    loader,
    task: EvaluationTask,
    *,
    device: torch.device,
    mixed_precision: str,
    threshold: float = 0.5,
    collect_source_rows: bool = False,
) -> PredictionResult:
    model.set_evaluation()
    criterion = nn.CrossEntropyLoss()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = (
        torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    )

    for batch in loader:
        batch_labels = batch["labels"].to(
            device,
            dtype=torch.long,
            non_blocking=device.type == "cuda",
        )
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            if task == "triplet":
                output = model.forward_triplet(
                    batch["peptide_sequences"],
                    batch["hla_sequences"],
                    batch["tcr_sequences"],
                )
            else:
                output = model(
                    task,
                    batch["peptide_sequences"],
                    batch["receptor_sequences"],
                )
            loss = criterion(output.logits, batch_labels)
        batch_samples = int(batch_labels.shape[0])
        samples += batch_samples
        loss_sum += float(loss.detach().cpu()) * batch_samples
        probabilities.append(
            torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
        )
        labels.append(batch_labels.cpu().numpy())
        if collect_source_rows:
            if "source_rows" not in batch:
                raise ValueError("Loader does not provide source_rows")
            source_rows.append(batch["source_rows"].cpu().numpy())
    if not samples:
        raise RuntimeError(f"{task} evaluation loader is empty")

    label_values = np.concatenate(labels)
    probability_values = np.concatenate(probabilities)
    metrics = compute_binary_metrics(
        label_values,
        probability_values,
        loss=loss_sum / samples,
        threshold=threshold,
        strict_threshold=True,
    )
    return PredictionResult(
        metrics=metrics,
        labels=label_values,
        probabilities=probability_values,
        source_rows=(
            np.concatenate(source_rows) if collect_source_rows else None
        ),
    )


@torch.inference_mode()
def evaluate_stage3_multitask(
    model: ProgressiveTripletFineTuningModel,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
    threshold: float = 0.5,
) -> dict[str, PredictionResult]:
    """Evaluate PH, PT, and PHT heads from one encoder pass per batch."""

    model.set_evaluation()
    criterion = nn.CrossEntropyLoss()
    task_fields = {
        "phla": ("phla_logits", "phla_labels"),
        "ptcr": ("ptcr_logits", "ptcr_labels"),
        "triplet": ("pht_logits", "pht_labels"),
    }
    probabilities: dict[str, list[np.ndarray]] = {
        task: [] for task in task_fields
    }
    labels: dict[str, list[np.ndarray]] = {task: [] for task in task_fields}
    loss_sums = {task: 0.0 for task in task_fields}
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = (
        torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    )
    for batch in loader:
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            output = model.forward_all(
                batch["peptide_sequences"],
                batch["hla_sequences"],
                batch["tcr_sequences"],
            )
            batch_losses = {}
            batch_labels = {}
            for task, (logit_field, label_field) in task_fields.items():
                current_labels = batch[label_field].to(
                    device,
                    dtype=torch.long,
                    non_blocking=device.type == "cuda",
                )
                batch_labels[task] = current_labels
                batch_losses[task] = criterion(
                    getattr(output, logit_field), current_labels
                )
        batch_samples = int(batch["pht_labels"].shape[0])
        samples += batch_samples
        for task, (logit_field, _) in task_fields.items():
            loss_sums[task] += (
                float(batch_losses[task].detach().cpu()) * batch_samples
            )
            probabilities[task].append(
                torch.softmax(getattr(output, logit_field), dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels[task].append(batch_labels[task].cpu().numpy())
    if not samples:
        raise RuntimeError("PHT multitask evaluation loader is empty")
    results = {}
    for task in task_fields:
        task_labels = np.concatenate(labels[task])
        task_probabilities = np.concatenate(probabilities[task])
        results[task] = PredictionResult(
            metrics=compute_binary_metrics(
                task_labels,
                task_probabilities,
                loss=loss_sums[task] / samples,
                threshold=threshold,
                strict_threshold=True,
            ),
            labels=task_labels,
            probabilities=task_probabilities,
        )
    return results
