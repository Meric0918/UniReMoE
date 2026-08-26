"""Stage-8 three-head metrics and directed violation reporting."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import BinaryMetrics, compute_binary_metrics


def select_mcc_threshold(labels, probabilities) -> tuple[float, BinaryMetrics]:
    """Choose a validation threshold without importing an optional Stage3 CLI."""

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
    candidate_indices = np.flatnonzero(np.r_[
        sorted_score[:-1] != sorted_score[1:], True,
    ])

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
            if denominator else 0.0
        )
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
        threshold = float(sorted_score[index])
        key = (mcc, f1, -abs(threshold - 0.5), -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold

    predictions = y_score >= 0.5
    tp = int(np.sum(predictions & (y_true == 1)))
    fp = int(np.sum(predictions & (y_true == 0)))
    fn = int(np.sum(~predictions & (y_true == 1)))
    tn = int(np.sum(~predictions & (y_true == 0)))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (
        (tp * tn - fp * fn) / math.sqrt(denominator)
        if denominator else 0.0
    )
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    if best_key is None or (mcc, f1, 0.0, -0.5) > best_key:
        best_threshold = 0.5

    return best_threshold, compute_binary_metrics(
        y_true, y_score, loss=0.0, threshold=best_threshold
    )


def stage8_task_score(metrics: Mapping[str, float]) -> float:
    """Score one task using all five monitored predictive metrics."""
    return sum(
        float(metrics[name])
        for name in ("auroc", "accuracy", "mcc", "f1", "aupr")
    ) / 5.0


def stage8_joint_score(
    metrics: Mapping[str, Mapping[str, float]],
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> float:
    return (
        weights[0] * stage8_task_score(metrics["pht"])
        + weights[1] * stage8_task_score(metrics["phla"])
        + weights[2] * stage8_task_score(metrics["ptcr"])
    )


@dataclass
class Stage8Evaluation:
    metrics: dict
    joint_score: float
    selected_pht_threshold: float
    predictions: dict[str, np.ndarray]


@torch.inference_mode()
def evaluate_online_stage8_model(
    model,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
    margin: float = 0.05,
    selection_weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> Stage8Evaluation:
    """Evaluate the online Stage-8 model without touching cached inputs.

    The cached :func:`evaluate_stage8_model` remains deliberately separate for
    Stage10, whose models consume residue-cache tensors rather than sequences.
    """

    model.set_evaluation()
    values = {name: [] for name in (
        "ph_labels", "pt_labels", "pht_labels", "phla_prob", "ptcr_prob",
        "pht_prob", "source_rows",
    )}
    label_types: list[str] = []
    loss_sums = {"phla": 0.0, "ptcr": 0.0, "pht": 0.0}
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    for batch in loader:
        moved = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        context = torch.autocast("cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
        with context:
            output = model.forward_all(
                batch["peptide_sequences"], batch["hla_sequences"],
                batch["tcr_sequences"],
            )
        count = int(moved["pht_labels"].shape[0])
        samples += count
        for task, logits, label_key in (
            ("phla", output.phla_logits, "ph_labels"),
            ("ptcr", output.ptcr_logits, "pt_labels"),
            ("pht", output.pht_logits, "pht_labels"),
        ):
            loss_sums[task] += float(F.cross_entropy(logits, moved[label_key], reduction="sum").cpu())
            values[f"{task}_prob"].append(logits.softmax(-1)[:, 1].float().cpu().numpy())
        for key in ("ph_labels", "pt_labels", "pht_labels", "source_rows"):
            values[key].append(moved[key].cpu().numpy())
        label_types.extend(batch["label_types"])
    if not samples:
        raise RuntimeError("Stage-8 online validation loader is empty")
    arrays = {key: np.concatenate(chunks) for key, chunks in values.items()}
    metrics: dict = {}
    for task, label_key in (("phla", "ph_labels"), ("ptcr", "pt_labels"), ("pht", "pht_labels")):
        metrics[task] = compute_binary_metrics(
            arrays[label_key], arrays[f"{task}_prob"], loss=loss_sums[task] / samples
        ).to_dict()
    threshold, selected = select_mcc_threshold(arrays["pht_labels"], arrays["pht_prob"])
    metrics["pht_selected"] = selected.to_dict()
    ph, pt, pht = arrays["phla_prob"], arrays["ptcr_prob"], arrays["pht_prob"]
    types = np.asarray(label_types)
    violation_masks = {
        "hierarchy_positive": (types == "111") & (pht > ph + margin),
        "hierarchy_negative": np.isin(types, ("000", "010")) & (pht > ph + margin),
        "pt_positive": (types == "111") & (pht > pt + margin),
        "pt_negative": np.isin(types, ("000", "100")) & (pht > pt + margin),
    }
    eligible = {
        "hierarchy_positive": types == "111",
        "hierarchy_negative": np.isin(types, ("000", "010")),
        "pt_positive": types == "111",
        "pt_negative": np.isin(types, ("000", "100")),
    }
    metrics["violations"] = {
        name: float(mask.sum() / max(1, eligible[name].sum()))
        for name, mask in violation_masks.items()
    }
    metrics["violations"]["overall"] = float(
        sum(mask.sum() for mask in violation_masks.values())
        / max(1, sum(mask.sum() for mask in eligible.values()))
    )
    hinge_components = {
        "hierarchy_positive": np.maximum(pht - ph - margin, 0.0) ** 2,
        "hierarchy_negative": np.maximum(pht - ph - margin, 0.0) ** 2,
        "pt_positive": np.maximum(pht - pt - margin, 0.0) ** 2,
        "pt_negative": np.maximum(pht - pt - margin, 0.0) ** 2,
    }
    metrics["constraint_components"] = {
        name: float(values[eligible[name]].mean()) if eligible[name].any() else 0.0
        for name, values in hinge_components.items()
    }
    epsilon = np.finfo(np.float64).eps
    clipped = {
        "phla": np.clip(ph.astype(np.float64), epsilon, 1.0 - epsilon),
        "ptcr": np.clip(pt.astype(np.float64), epsilon, 1.0 - epsilon),
        "pht": np.clip(pht.astype(np.float64), epsilon, 1.0 - epsilon),
    }
    labels_by_task = {
        "phla": arrays["ph_labels"],
        "ptcr": arrays["pt_labels"],
        "pht": arrays["pht_labels"],
    }
    metrics["by_type"] = {
        kind: {
            "samples": int((types == kind).sum()),
            "mean_phla_probability": float(ph[types == kind].mean()),
            "mean_ptcr_probability": float(pt[types == kind].mean()),
            "mean_pht_probability": float(pht[types == kind].mean()),
            **{
                f"{task}_cross_entropy": float((
                    -labels_by_task[task][types == kind] * np.log(clipped[task][types == kind])
                    -(1 - labels_by_task[task][types == kind]) * np.log(1 - clipped[task][types == kind])
                ).mean())
                for task in ("phla", "ptcr", "pht")
            },
        }
        for kind in ("111", "100", "010", "000") if np.any(types == kind)
    }
    return Stage8Evaluation(
        metrics, stage8_joint_score(metrics, selection_weights), threshold,
        {**arrays, "label_types": types},
    )


@torch.inference_mode()
def evaluate_stage8_model(
    model,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
    margin: float = 0.05,
    selection_weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> Stage8Evaluation:
    model.eval()
    values = {name: [] for name in (
        "ph_labels", "pt_labels", "pht_labels", "phla_prob", "ptcr_prob",
        "pht_prob", "source_rows",
    )}
    label_types: list[str] = []
    loss_sums = {"phla": 0.0, "ptcr": 0.0, "pht": 0.0}
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    for batch in loader:
        moved = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        context = torch.autocast("cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
        with context:
            output = model.forward_all(
                moved["peptide_hidden"], moved["peptide_mask"],
                moved["hla_hidden"], moved["hla_mask"],
                moved["tcr_hidden"], moved["tcr_mask"],
            )
        count = int(moved["pht_labels"].shape[0])
        samples += count
        for task, logits, label_key in (
            ("phla", output.phla_logits, "ph_labels"),
            ("ptcr", output.ptcr_logits, "pt_labels"),
            ("pht", output.pht_logits, "pht_labels"),
        ):
            loss_sums[task] += float(F.cross_entropy(logits, moved[label_key], reduction="sum").cpu())
            values[f"{task}_prob"].append(logits.softmax(-1)[:, 1].float().cpu().numpy())
            if label_key not in values:
                raise AssertionError(label_key)
        for key in ("ph_labels", "pt_labels", "pht_labels", "source_rows"):
            values[key].append(moved[key].cpu().numpy())
        label_types.extend(batch["label_types"])
    if not samples:
        raise RuntimeError("Stage-8 validation loader is empty")
    arrays = {key: np.concatenate(chunks) for key, chunks in values.items()}
    metrics: dict = {}
    for task, label_key in (("phla", "ph_labels"), ("ptcr", "pt_labels"), ("pht", "pht_labels")):
        metrics[task] = compute_binary_metrics(
            arrays[label_key], arrays[f"{task}_prob"], loss=loss_sums[task] / samples
        ).to_dict()
    threshold, selected = select_mcc_threshold(arrays["pht_labels"], arrays["pht_prob"])
    metrics["pht_selected"] = selected.to_dict()
    ph, pt, pht = arrays["phla_prob"], arrays["ptcr_prob"], arrays["pht_prob"]
    types = np.asarray(label_types)
    violation_masks = {
        "hierarchy_positive": (types == "111") & (pht > ph + margin),
        "hierarchy_negative": np.isin(types, ("000", "010")) & (pht > ph + margin),
        "pt_positive": (types == "111") & (pht > pt + margin),
        "pt_negative": np.isin(types, ("000", "100")) & (pht > pt + margin),
    }
    eligible = {
        "hierarchy_positive": types == "111",
        "hierarchy_negative": np.isin(types, ("000", "010")),
        "pt_positive": types == "111",
        "pt_negative": np.isin(types, ("000", "100")),
    }
    metrics["violations"] = {
        name: float(mask.sum() / max(1, eligible[name].sum()))
        for name, mask in violation_masks.items()
    }
    metrics["violations"]["overall"] = float(
        sum(mask.sum() for mask in violation_masks.values())
        / max(1, sum(mask.sum() for mask in eligible.values()))
    )
    hinge_components = {
        "hierarchy_positive": np.maximum(pht - ph - margin, 0.0) ** 2,
        "hierarchy_negative": np.maximum(pht - ph - margin, 0.0) ** 2,
        "pt_positive": np.maximum(pht - pt - margin, 0.0) ** 2,
        "pt_negative": np.maximum(pht - pt - margin, 0.0) ** 2,
    }
    metrics["constraint_components"] = {
        name: float(values[eligible[name]].mean()) if eligible[name].any() else 0.0
        for name, values in hinge_components.items()
    }
    epsilon = np.finfo(np.float64).eps
    clipped = {
        "phla": np.clip(ph.astype(np.float64), epsilon, 1.0 - epsilon),
        "ptcr": np.clip(pt.astype(np.float64), epsilon, 1.0 - epsilon),
        "pht": np.clip(pht.astype(np.float64), epsilon, 1.0 - epsilon),
    }
    labels_by_task = {
        "phla": arrays["ph_labels"],
        "ptcr": arrays["pt_labels"],
        "pht": arrays["pht_labels"],
    }
    metrics["by_type"] = {
        kind: {
            "samples": int((types == kind).sum()),
            "mean_phla_probability": float(ph[types == kind].mean()),
            "mean_ptcr_probability": float(pt[types == kind].mean()),
            "mean_pht_probability": float(pht[types == kind].mean()),
            **{
                f"{task}_cross_entropy": float((
                    -labels_by_task[task][types == kind] * np.log(clipped[task][types == kind])
                    -(1 - labels_by_task[task][types == kind]) * np.log(1 - clipped[task][types == kind])
                ).mean())
                for task in ("phla", "ptcr", "pht")
            },
        }
        for kind in ("111", "100", "010", "000") if np.any(types == kind)
    }
    return Stage8Evaluation(
        metrics, stage8_joint_score(metrics, selection_weights), threshold,
        {**arrays, "label_types": types},
    )
