"""Binary classification metrics shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass
class BinaryMetrics:
    loss: float
    auroc: float
    aupr: float
    accuracy: float
    mcc: float
    f1: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    samples: int
    nll: float = float("nan")
    brier: float = float("nan")
    ece: float = float("nan")
    mean_confidence: float = float("nan")
    loss_p95: float = float("nan")
    loss_p99: float = float("nan")
    group_macro_nll: float = float("nan")
    group_macro_accuracy: float = float("nan")
    group_macro_recall: float = float("nan")
    worst_group_nll: float = float("nan")
    group_count: int = 0
    group_coverage: float = float("nan")

    @property
    def performance_avg(self) -> float:
        return performance_avg(self)

    def to_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        values["performance_avg"] = self.performance_avg
        return values


def compute_binary_metrics(
    labels,
    probabilities,
    *,
    loss: float,
    threshold: float = 0.5,
    strict_threshold: bool = False,
    group_ids=None,
    ece_bins: int = 15,
) -> BinaryMetrics:
    try:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for training metrics. "
            "Install requirements-plm.txt."
        ) from exc

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and probabilities must be aligned 1-D arrays")
    if group_ids is not None:
        y_group = np.asarray(group_ids)
        if y_group.ndim != 1 or len(y_group) != len(y_true):
            raise ValueError("group_ids must be aligned with labels")
    else:
        y_group = None
    if strict_threshold:
        y_pred = (y_score > threshold).astype(np.int64)
    else:
        y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_score))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        aupr = float(auc(recall_curve, precision_curve))
    else:
        # A single-label split has no threshold-free discrimination metric. Keep
        # the row-level metrics usable and let callers exclude this split/group.
        auroc = float("nan")
        aupr = float("nan")

    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    true_probability = np.where(y_true == 1, clipped, 1.0 - clipped)
    sample_nll = -np.log(true_probability)
    confidence = np.maximum(clipped, 1.0 - clipped)
    brier = float(np.mean((clipped - y_true) ** 2))
    ece = _adaptive_ece(y_true, clipped, bins=ece_bins)
    group_metrics = _group_metrics(y_true, y_score, y_pred, y_group)
    return BinaryMetrics(
        loss=float(loss),
        auroc=auroc,
        aupr=aupr,
        accuracy=float(accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        sensitivity=float(tp / (tp + fn)) if tp + fn else 0.0,
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        samples=int(len(y_true)),
        nll=float(np.mean(sample_nll)),
        brier=brier,
        ece=ece,
        mean_confidence=float(np.mean(confidence)),
        loss_p95=float(np.percentile(sample_nll, 95)),
        loss_p99=float(np.percentile(sample_nll, 99)),
        **group_metrics,
    )


def _adaptive_ece(labels, probabilities, *, bins: int = 15) -> float:
    """Equal-frequency expected calibration error."""

    y_true = np.asarray(labels, dtype=np.float64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    chunks = np.array_split(order, min(max(1, bins), len(order)))
    weighted_error = 0.0
    for indices in chunks:
        if len(indices) == 0:
            continue
        confidence = np.maximum(y_score[indices], 1.0 - y_score[indices])
        predictions = (y_score[indices] >= 0.5).astype(np.float64)
        accuracy = np.mean(predictions == y_true[indices])
        weighted_error += len(indices) / len(y_true) * abs(
            float(np.mean(confidence)) - float(accuracy)
        )
    return float(weighted_error)


def _group_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    group_ids: np.ndarray | None,
) -> dict[str, float | int]:
    if group_ids is None or len(group_ids) == 0:
        return {
            "group_macro_nll": float("nan"),
            "group_macro_accuracy": float("nan"),
            "group_macro_recall": float("nan"),
            "worst_group_nll": float("nan"),
            "group_count": 0,
            "group_coverage": float("nan"),
        }
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    sample_nll = -np.log(np.where(labels == 1, clipped, 1.0 - clipped))
    nll_values: list[float] = []
    accuracy_values: list[float] = []
    recall_values: list[float] = []
    valid_recall_rows = 0
    for group in np.unique(group_ids):
        mask = group_ids == group
        nll_values.append(float(np.mean(sample_nll[mask])))
        accuracy_values.append(float(np.mean(predictions[mask] == labels[mask])))
        if len(np.unique(labels[mask])) == 2:
            tp = int(np.sum((predictions[mask] == 1) & (labels[mask] == 1)))
            positives = int(np.sum(labels[mask] == 1))
            recall_values.append(float(tp / positives) if positives else 0.0)
            valid_recall_rows += int(np.sum(mask))
    return {
        "group_macro_nll": float(np.mean(nll_values)),
        "group_macro_accuracy": float(np.mean(accuracy_values)),
        "group_macro_recall": (
            float(np.mean(recall_values)) if recall_values else float("nan")
        ),
        "worst_group_nll": float(np.max(nll_values)),
        "group_count": int(len(nll_values)),
        "group_coverage": float(valid_recall_rows / len(labels))
        if recall_values
        else 0.0,
    }


def performance_avg(metrics: BinaryMetrics | Mapping[str, float]) -> float:
    """Average the five predictive metrics used for Stage 1 selection."""

    def value(name: str) -> float:
        if isinstance(metrics, BinaryMetrics):
            return float(getattr(metrics, name))
        return float(metrics[name])

    values = np.asarray(
        [
            value(name)
            for name in ("auroc", "accuracy", "mcc", "f1", "aupr")
        ],
        dtype=np.float64,
    )
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def joint_score(
    phla_metrics: BinaryMetrics | Mapping[str, float],
    ptcr_metrics: BinaryMetrics | Mapping[str, float],
) -> float:
    """Legacy reporting helper; training no longer uses it for early stopping."""

    def value(metrics, name: str) -> float:
        if isinstance(metrics, BinaryMetrics):
            return float(getattr(metrics, name))
        return float(metrics[name])

    return (
        value(phla_metrics, "auroc")
        + value(phla_metrics, "aupr")
        + value(ptcr_metrics, "auroc")
        + value(ptcr_metrics, "aupr")
    ) / 4.0


def task_score(metrics: BinaryMetrics | Mapping[str, float]) -> float:
    """Threshold-independent score used for one task's early stopping."""

    if isinstance(metrics, BinaryMetrics):
        auroc = metrics.auroc
        aupr = metrics.aupr
    else:
        auroc = metrics["auroc"]
        aupr = metrics["aupr"]
    return (float(auroc) + float(aupr)) / 2.0


def prefixed_metrics(prefix: str, metrics: BinaryMetrics) -> dict[str, float | int]:
    return {f"{prefix}/{key}": value for key, value in metrics.to_dict().items()}
