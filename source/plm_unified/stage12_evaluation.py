"""Validation metrics and equal-weight Stage12 model selection."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_binary_metrics
from .stage8_evaluation import evaluate_online_stage8_model
from .stage12_config import Stage12ExperimentConfig


SCORE_METRICS = ("auroc", "aupr", "accuracy", "mcc", "f1")


def task_score(metrics: Mapping[str, float]) -> float:
    return sum(float(metrics[name]) for name in SCORE_METRICS) / len(SCORE_METRICS)


@torch.inference_mode()
def evaluate_pair_validation_splits(
    model,
    loaders: Mapping[str, object],
    *,
    task: str,
    device: torch.device,
    mixed_precision: str,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Evaluate raw-sequence PH or PT splits at one fixed threshold."""

    if task not in {"phla", "ptcr"}:
        raise ValueError("pair validation task must be phla or ptcr")
    model.set_evaluation()
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    result: dict[str, dict[str, float]] = {}
    for split, loader in loaders.items():
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        for batch in loader:
            target = batch["labels"].to(device, non_blocking=device.type == "cuda")
            context = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast else nullcontext()
            )
            with context:
                output = model(
                    task,
                    batch["peptide_sequences"],
                    batch["receptor_sequences"],
                )
                loss = F.cross_entropy(output.logits, target, reduction="sum")
            count = int(target.shape[0])
            samples += count
            loss_sum += float(loss.detach().cpu())
            labels.append(target.cpu().numpy())
            probabilities.append(
                output.logits.softmax(-1)[:, 1].float().cpu().numpy()
            )
        if not samples:
            raise RuntimeError(f"Stage12 validation split is empty: {split}")
        result[str(split)] = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=loss_sum / samples,
            threshold=threshold,
        ).to_dict()
    return result


def evaluate_stage12_validation(
    model,
    *,
    pht_loader,
    ph_loaders: Mapping[str, object],
    pt_loaders: Mapping[str, object],
    config: Stage12ExperimentConfig,
    device: torch.device,
) -> "Stage12ValidationScores":
    values = config.stage12
    ph_metrics = evaluate_pair_validation_splits(
        model,
        ph_loaders,
        task="phla",
        device=device,
        mixed_precision=values.mixed_precision,
        threshold=values.fixed_threshold,
    )
    pt_metrics = evaluate_pair_validation_splits(
        model,
        pt_loaders,
        task="ptcr",
        device=device,
        mixed_precision=values.mixed_precision,
        threshold=values.fixed_threshold,
    )
    pht = evaluate_online_stage8_model(
        model,
        pht_loader,
        device=device,
        mixed_precision=values.mixed_precision,
        margin=values.constraint_margin,
        selection_weights=values.pht_joint_weights,
    )
    return Stage12ValidationScores.from_metrics(
        pt_metrics=pt_metrics,
        ph_metrics=ph_metrics,
        pht_joint=pht.joint_score,
    )


@dataclass(frozen=True)
class Stage12ValidationScores:
    pt_splits: dict[str, float]
    pt_macro: float
    ph_splits: dict[str, float]
    ph_macro: float
    pht_joint: float
    pt_metrics: dict[str, dict[str, float]]
    ph_metrics: dict[str, dict[str, float]]

    @property
    def joint_score(self) -> float:
        """Equal-weight overall score for PHT, PH, and PT validation."""

        return (self.pht_joint + self.ph_macro + self.pt_macro) / 3.0

    @classmethod
    def from_metrics(
        cls,
        *,
        pt_metrics: Mapping[str, Mapping[str, float]],
        ph_metrics: Mapping[str, Mapping[str, float]],
        pht_joint: float,
    ) -> "Stage12ValidationScores":
        if not pt_metrics or not ph_metrics:
            raise ValueError("Stage12 validation requires non-empty PH and PT splits")
        pt = {name: task_score(metrics) for name, metrics in pt_metrics.items()}
        ph = {name: task_score(metrics) for name, metrics in ph_metrics.items()}
        return cls(
            pt_splits=pt,
            pt_macro=sum(pt.values()) / len(pt),
            ph_splits=ph,
            ph_macro=sum(ph.values()) / len(ph),
            pht_joint=float(pht_joint),
            pt_metrics={name: dict(values) for name, values in pt_metrics.items()},
            ph_metrics={name: dict(values) for name, values in ph_metrics.items()},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage12ValidationScores":
        return cls(
            pt_splits={key: float(item) for key, item in dict(value["pt_splits"]).items()},
            pt_macro=float(value["pt_macro"]),
            ph_splits={key: float(item) for key, item in dict(value["ph_splits"]).items()},
            ph_macro=float(value["ph_macro"]),
            pht_joint=float(value["pht_joint"]),
            pt_metrics={
                key: {name: float(item) for name, item in dict(metrics).items()}
                for key, metrics in dict(value["pt_metrics"]).items()
            },
            ph_metrics={
                key: {name: float(item) for name, item in dict(metrics).items()}
                for key, metrics in dict(value["ph_metrics"]).items()
            },
        )

    def to_dict(self) -> dict:
        values = asdict(self)
        values["joint_score"] = self.joint_score
        return values


@dataclass(frozen=True)
class Stage12SelectionDecision:
    improved: bool
    joint_score_delta: float
    pt_macro_delta: float
    ph_macro_delta: float
    pht_joint_delta: float

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage12SelectionDecision":
        return cls(
            improved=bool(value["improved"]),
            joint_score_delta=float(value["joint_score_delta"]),
            pt_macro_delta=float(value["pt_macro_delta"]),
            ph_macro_delta=float(value["ph_macro_delta"]),
            pht_joint_delta=float(value["pht_joint_delta"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Stage12TrackerUpdate:
    decision: Stage12SelectionDecision
    improved: bool
    scores: Stage12ValidationScores


class Stage12BestTracker:
    """Select the best Stage12B round by equal-weight joint score."""

    def __init__(
        self,
        reference: Stage12ValidationScores,
        *,
        selection_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        min_delta: float = 0.0001,
    ):
        self.reference = reference
        if len(selection_weights) != 3 or abs(sum(selection_weights) - 1.0) > 1e-8:
            raise ValueError("Stage12 selection weights must be three values summing to one")
        self.selection_weights = selection_weights
        self.min_delta = min_delta
        self.best_round: int | None = None
        self.best_scores: Stage12ValidationScores | None = None

    def _joint_score(self, scores: Stage12ValidationScores) -> float:
        return (
            self.selection_weights[0] * scores.pht_joint
            + self.selection_weights[1] * scores.ph_macro
            + self.selection_weights[2] * scores.pt_macro
        )

    def consider(
        self, round_index: int, scores: Stage12ValidationScores
    ) -> Stage12TrackerUpdate:
        if round_index < 1:
            raise ValueError("Only Stage12B rounds can participate in selection")
        previous = self.best_scores or self.reference
        current_score = self._joint_score(scores)
        previous_score = self._joint_score(previous)
        improved = current_score > previous_score + self.min_delta
        if improved:
            self.best_round = round_index
            self.best_scores = scores
        decision = Stage12SelectionDecision(
            improved=improved,
            joint_score_delta=current_score - previous_score,
            pt_macro_delta=scores.pt_macro - self.reference.pt_macro,
            ph_macro_delta=scores.ph_macro - self.reference.ph_macro,
            pht_joint_delta=scores.pht_joint - self.reference.pht_joint,
        )
        return Stage12TrackerUpdate(decision, improved, scores)
