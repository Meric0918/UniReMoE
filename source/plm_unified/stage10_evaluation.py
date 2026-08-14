"""Paired teacher-student validation and retention metrics for Stage 10."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_binary_metrics, task_score
from .stage8_evaluation import stage8_joint_score
from .triplet_evaluation import select_mcc_threshold


@dataclass
class Stage10Evaluation:
    student_metrics: dict
    teacher_metrics: dict
    retention: dict[str, float]
    joint_score: float
    selected_pht_threshold: float
    retention_eligible: bool
    predictions: dict[str, np.ndarray]

    @property
    def metrics(self) -> dict:
        """Compatibility alias for consumers of Stage-8 evaluations."""

        return self.student_metrics


def retention_is_eligible(
    student_metrics: Mapping[str, Mapping[str, float]],
    teacher_metrics: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float,
) -> bool:
    if tolerance < 0:
        raise ValueError("retention degradation tolerance cannot be negative")
    epsilon = 1e-12
    return all(
        task_score(student_metrics[task])
        >= task_score(teacher_metrics[task]) - tolerance - epsilon
        for task in ("phla", "ptcr")
    )


def _student_metrics(
    arrays: Mapping[str, np.ndarray],
    label_types: np.ndarray,
    loss_sums: Mapping[str, float],
    samples: int,
    *,
    margin: float,
) -> tuple[dict, float]:
    metrics: dict = {}
    for task, label_key in (
        ("phla", "ph_labels"),
        ("ptcr", "pt_labels"),
        ("pht", "pht_labels"),
    ):
        metrics[task] = compute_binary_metrics(
            arrays[label_key],
            arrays[f"{task}_prob"],
            loss=loss_sums[task] / samples,
        ).to_dict()

    threshold, selected = select_mcc_threshold(
        arrays["pht_labels"],
        arrays["pht_prob"],
    )
    metrics["pht_selected"] = selected.to_dict()
    ph = arrays["phla_prob"]
    pt = arrays["ptcr_prob"]
    pht = arrays["pht_prob"]
    types = label_types
    violation_masks = {
        "hierarchy_positive": (types == "111") & (pht > ph + margin),
        "hierarchy_negative": np.isin(types, ("000", "010"))
        & (pht > ph + margin),
        "pt_positive": (types == "111") & (pht > pt + margin),
        "pt_negative": np.isin(types, ("000", "100"))
        & (pht > pt + margin),
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
        name: (
            float(component[eligible[name]].mean())
            if eligible[name].any()
            else 0.0
        )
        for name, component in hinge_components.items()
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
                f"{task}_cross_entropy": float(
                    (
                        -labels_by_task[task][types == kind]
                        * np.log(clipped[task][types == kind])
                        - (1 - labels_by_task[task][types == kind])
                        * np.log(1 - clipped[task][types == kind])
                    ).mean()
                )
                for task in ("phla", "ptcr", "pht")
            },
        }
        for kind in ("111", "100", "010", "000")
        if np.any(types == kind)
    }
    return metrics, threshold


@torch.inference_mode()
def evaluate_stage10_models(
    teacher,
    student,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
    margin: float = 0.05,
    selection_weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
    degradation_tolerance: float = 0.002,
) -> Stage10Evaluation:
    teacher.eval()
    student.eval()
    values = {
        name: []
        for name in (
            "ph_labels",
            "pt_labels",
            "pht_labels",
            "phla_prob",
            "ptcr_prob",
            "pht_prob",
            "teacher_phla_prob",
            "teacher_ptcr_prob",
            "source_rows",
        )
    }
    label_types: list[str] = []
    student_loss_sums = {"phla": 0.0, "ptcr": 0.0, "pht": 0.0}
    teacher_loss_sums = {"phla": 0.0, "ptcr": 0.0}
    retention_sums = {
        "phla_kl": 0.0,
        "ptcr_kl": 0.0,
        "phla_probability_mae": 0.0,
        "ptcr_probability_mae": 0.0,
        "phla_agreement": 0.0,
        "ptcr_agreement": 0.0,
    }
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = (
        torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    )

    for batch in loader:
        moved = {
            key: value.to(
                device,
                non_blocking=device.type == "cuda",
            )
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        context = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            teacher_output = teacher.forward_legacy_heads(
                moved["peptide_hidden"],
                moved["peptide_mask"],
                moved["hla_hidden"],
                moved["hla_mask"],
                moved["tcr_hidden"],
                moved["tcr_mask"],
            )
            student_output = student.forward_all(
                moved["peptide_hidden"],
                moved["peptide_mask"],
                moved["hla_hidden"],
                moved["hla_mask"],
                moved["tcr_hidden"],
                moved["tcr_mask"],
            )
        count = int(moved["pht_labels"].shape[0])
        samples += count

        for task, logits, label_key in (
            ("phla", student_output.phla_logits, "ph_labels"),
            ("ptcr", student_output.ptcr_logits, "pt_labels"),
            ("pht", student_output.pht_logits, "pht_labels"),
        ):
            student_loss_sums[task] += float(
                F.cross_entropy(
                    logits,
                    moved[label_key],
                    reduction="sum",
                ).cpu()
            )
            values[f"{task}_prob"].append(
                logits.softmax(-1)[:, 1].float().cpu().numpy()
            )

        teacher_logits = {
            "phla": teacher_output.phla_logits,
            "ptcr": teacher_output.ptcr_logits,
        }
        student_logits = {
            "phla": student_output.phla_logits,
            "ptcr": student_output.ptcr_logits,
        }
        for task, label_key in (
            ("phla", "ph_labels"),
            ("ptcr", "pt_labels"),
        ):
            teacher_loss_sums[task] += float(
                F.cross_entropy(
                    teacher_logits[task],
                    moved[label_key],
                    reduction="sum",
                ).cpu()
            )
            teacher_probability = teacher_logits[task].float().softmax(-1)
            student_probability = student_logits[task].float().softmax(-1)
            values[f"teacher_{task}_prob"].append(
                teacher_probability[:, 1].cpu().numpy()
            )
            per_row_kl = F.kl_div(
                F.log_softmax(student_logits[task].float(), dim=-1),
                teacher_probability,
                reduction="none",
            ).sum(dim=-1)
            retention_sums[f"{task}_kl"] += float(per_row_kl.sum().cpu())
            retention_sums[f"{task}_probability_mae"] += float(
                (
                    student_probability[:, 1]
                    - teacher_probability[:, 1]
                )
                .abs()
                .sum()
                .cpu()
            )
            retention_sums[f"{task}_agreement"] += float(
                (
                    (student_probability[:, 1] >= 0.5)
                    == (teacher_probability[:, 1] >= 0.5)
                )
                .sum()
                .cpu()
            )

        for key in (
            "ph_labels",
            "pt_labels",
            "pht_labels",
            "source_rows",
        ):
            values[key].append(moved[key].cpu().numpy())
        label_types.extend(batch["label_types"])

    if not samples:
        raise RuntimeError("Stage-10 validation loader is empty")
    arrays = {
        key: np.concatenate(chunks) for key, chunks in values.items()
    }
    types = np.asarray(label_types)
    student_metrics, threshold = _student_metrics(
        arrays,
        types,
        student_loss_sums,
        samples,
        margin=margin,
    )
    teacher_metrics = {
        task: compute_binary_metrics(
            arrays[label_key],
            arrays[f"teacher_{task}_prob"],
            loss=teacher_loss_sums[task] / samples,
        ).to_dict()
        for task, label_key in (
            ("phla", "ph_labels"),
            ("ptcr", "pt_labels"),
        )
    }
    retention = {
        name: total / samples for name, total in retention_sums.items()
    }
    retention.update(
        {
            f"{task}_task_score_delta": (
                task_score(student_metrics[task])
                - task_score(teacher_metrics[task])
            )
            for task in ("phla", "ptcr")
        }
    )
    return Stage10Evaluation(
        student_metrics=student_metrics,
        teacher_metrics=teacher_metrics,
        retention=retention,
        joint_score=stage8_joint_score(student_metrics, selection_weights),
        selected_pht_threshold=threshold,
        retention_eligible=retention_is_eligible(
            student_metrics,
            teacher_metrics,
            tolerance=degradation_tolerance,
        ),
        predictions={**arrays, "label_types": types},
    )
