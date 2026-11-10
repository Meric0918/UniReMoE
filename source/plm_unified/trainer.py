"""Stage 1A warm-up and stage 1B alternating dual-task training."""

from __future__ import annotations

import csv
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .config import ExperimentConfig
from .data import normalise_sequence, resolve_task_columns, stable_group_id
from .encoders import OnlineEncoderBundle
from .fgm import FGMStats, ModuleFGM
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    performance_avg,
    prefixed_metrics,
    task_score,
)
from .model import Task, UnifiedBindingModel, extract_task_state_dict
from .tracking import SwanLabTracker

Stage = Literal["warmup", "stage1b", "complete"]
TASK_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "best_phla.pt",
    "ptcr": "best_ptcr.pt",
}
CALIBRATION_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "best_phla_calibration.pt",
    "ptcr": "best_ptcr_calibration.pt",
}
ROBUST_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "best_phla_robust.pt",
    "ptcr": "best_ptcr_robust.pt",
}
BEST_ROUND_CHECKPOINT_FILENAME = "best_joint.pt"


def _metric_summary(metrics: BinaryMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} "
        f"AUROC={metrics.auroc:.4f} "
        f"AUPR={metrics.aupr:.4f} "
        f"ACC={metrics.accuracy:.4f} "
        f"MCC={metrics.mcc:.4f} "
        f"F1={metrics.f1:.4f} "
        f"NLL={metrics.nll:.4f} "
        f"ECE={metrics.ece:.4f}"
    )


def _macro_validation_score(
    metrics_by_split: Mapping[str, BinaryMetrics],
) -> float:
    """Average configured benchmark scores equally for one task."""

    if not metrics_by_split:
        raise ValueError("At least one validation benchmark is required")
    values = np.asarray(
        [performance_avg(metrics) for metrics in metrics_by_split.values()],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else -float("inf")


def _macro_metric(
    metrics_by_split: Mapping[str, BinaryMetrics],
    name: str,
    *,
    lower_is_better: bool = False,
) -> float:
    def value(metrics: BinaryMetrics | Mapping[str, float]) -> float:
        if isinstance(metrics, Mapping):
            return float(metrics[name])
        return float(getattr(metrics, name))

    values = np.asarray(
        [value(metrics) for metrics in metrics_by_split.values()],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    if len(finite):
        return float(np.mean(finite))
    return float("inf" if lower_is_better else "-inf")


def _selection_calibration_guard(
    current_nll: float,
    current_ece: float,
    best_nll: float,
    best_ece: float,
    *,
    enabled: bool,
    nll_tolerance: float,
    ece_tolerance: float,
) -> tuple[bool, bool, bool]:
    """Check calibration regressions before accepting a score improvement."""

    if not enabled:
        return True, True, True

    def within_tolerance(current: float, best: float, tolerance: float) -> bool:
        # A non-finite current calibration value is never safe.  An infinite
        # best value means that no finite baseline has been selected yet.
        if not math.isfinite(current):
            return False
        if not math.isfinite(best):
            return True
        return current <= best + tolerance

    nll_ok = within_tolerance(current_nll, best_nll, nll_tolerance)
    ece_ok = within_tolerance(current_ece, best_ece, ece_tolerance)
    return nll_ok, ece_ok, nll_ok and ece_ok


def _robust_validation_score(
    metrics_by_split: Mapping[str, BinaryMetrics],
) -> float:
    """Prefer group-macro accuracy and fall back to the legacy score."""

    group_values = np.asarray(
        [float(metrics.group_macro_accuracy) for metrics in metrics_by_split.values()],
        dtype=np.float64,
    )
    finite_group = group_values[np.isfinite(group_values)]
    if len(finite_group):
        return float(np.mean(finite_group))
    return _macro_validation_score(metrics_by_split)


@dataclass
class EpochResult:
    metrics: BinaryMetrics
    clean_loss: float
    adversarial_loss: float
    total_loss: float
    optimizer_steps: int
    samples: int
    attacked_parameter_count: float
    fgm_mean_gradient_norm: float
    fgm_max_gradient_norm: float
    model_gradient_norm: float

    def to_log_dict(self, prefix: str) -> dict[str, float | int]:
        values = prefixed_metrics(prefix, self.metrics)
        values.update(
            {
                f"{prefix}/clean_loss": self.clean_loss,
                f"{prefix}/adversarial_loss": self.adversarial_loss,
                f"{prefix}/total_loss": self.total_loss,
                # Explicit aliases make the train/eval comparison unambiguous:
                # these are computed from the unsmoothed probabilities rather
                # than from the weighted, label-smoothed training objective.
                f"{prefix}/unweighted_nll": self.metrics.nll,
                f"{prefix}/unweighted_ece": self.metrics.ece,
                f"{prefix}/optimizer_steps": self.optimizer_steps,
                f"{prefix}/samples": self.samples,
                f"{prefix}/fgm_attacked_parameters": self.attacked_parameter_count,
                f"{prefix}/fgm_mean_gradient_norm": self.fgm_mean_gradient_norm,
                f"{prefix}/fgm_max_gradient_norm": self.fgm_max_gradient_norm,
                f"{prefix}/model_gradient_norm": self.model_gradient_norm,
            }
        )
        return values


class ValidationErrorCSVWriter:
    """Persist high-confidence validation errors for each Stage-1 round."""

    THRESHOLD = 0.8
    _TASK_LAYOUT = {
        "phla": ("pHLA", ("hla", "peptide", "label", "prediction_score", "source")),
        "ptcr": ("pTCR", ("peptide", "tcr", "label", "prediction_score", "source")),
    }

    def __init__(
        self,
        project_root: str | Path,
        run_timestamp: str,
        *,
        peptide_min_length: int,
        peptide_max_length: int,
        artifact_namespace: str = "stage1",
        tasks: tuple[Task, ...] = ("phla", "ptcr"),
    ) -> None:
        self.tasks = tuple(tasks)
        self.output_dir = (
            Path(project_root).expanduser().resolve()
            / "csv"
            / artifact_namespace
            / run_timestamp
        )
        self.peptide_min_length = int(peptide_min_length)
        self.peptide_max_length = int(peptide_max_length)
        self._source_cache: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
        self._round_id: int | None = None
        self._round_rows: dict[str, list[dict[str, Any]]] = {}
        self._current_split: dict[str, Any] | None = None

    @staticmethod
    def _coerce_rows(rows: Any) -> list[tuple[str, str, int]]:
        return [
            (str(row[0]), str(row[1]), int(row[2]))
            for row in rows
        ]

    def _load_source_rows(
        self,
        task: str,
        loader: Any,
    ) -> list[tuple[str, str, int]] | None:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None

        csv_path = getattr(dataset, "csv_path", None)
        cache_key: tuple[str, str] | None = None
        if csv_path is not None:
            cache_key = (task, str(Path(csv_path).expanduser().resolve()))
            if cache_key in self._source_cache:
                return self._source_cache[cache_key]

        # OnlinePairDataset already retains the canonical raw rows.
        raw_rows = getattr(dataset, "rows", None)
        if raw_rows is not None:
            records = self._coerce_rows(raw_rows)
            if cache_key is not None:
                self._source_cache[cache_key] = records
            return records

        # CachedPairDataset intentionally keeps only embedding locations. Read
        # its CSV once in the main process and cache the filtered row order so
        # it can be aligned with the shuffle=False validation loader.
        if csv_path is None:
            return None
        csv_path = Path(csv_path).expanduser().resolve()
        if cache_key is None:
            cache_key = (task, str(csv_path))

        expected_count = len(dataset) if hasattr(dataset, "__len__") else None
        records: list[tuple[str, str, int]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            peptide_col, receptor_col, label_col = resolve_task_columns(
                task, reader.fieldnames
            )
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{csv_path} must contain columns: {sorted(required)}"
                )
            for row in reader:
                peptide = normalise_sequence(row[peptide_col])
                if not self.peptide_min_length <= len(peptide) <= self.peptide_max_length:
                    continue
                receptor = normalise_sequence(row[receptor_col])
                label = int(row[label_col])
                if label not in {0, 1}:
                    raise ValueError(f"Label must be 0 or 1 in {csv_path}")
                records.append((peptide, receptor, label))
                if expected_count is not None and len(records) >= expected_count:
                    break

        if expected_count is not None and len(records) != expected_count:
            raise RuntimeError(
                f"Validation source rows do not match loader length for {csv_path}: "
                f"{len(records)} != {expected_count}"
            )
        self._source_cache[cache_key] = records
        return records

    def begin_round(self, round_id: int) -> None:
        if round_id < 1:
            raise ValueError("round_id must be positive")
        self._round_id = int(round_id)
        self._round_rows = {task: [] for task in self.tasks}
        self._current_split = None

    def begin_split(self, task: str, split_name: str, loader: Any) -> None:
        if self._round_id is None:
            self._current_split = None
            return
        if task not in self.tasks:
            raise ValueError(f"Unknown validation task: {task}")
        self._current_split = {
            "task": task,
            "source": split_name,
            "source_rows": self._load_source_rows(task, loader),
            "offset": 0,
            "rows": [],
        }

    def discard_split(self) -> None:
        self._current_split = None

    def record_batch(self, labels: np.ndarray, prediction_scores: np.ndarray) -> None:
        current = self._current_split
        if current is None or current["source_rows"] is None:
            return
        label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
        score_values = np.asarray(prediction_scores, dtype=np.float64).reshape(-1)
        if len(label_values) != len(score_values):
            raise RuntimeError("Validation labels and prediction scores are misaligned")

        source_rows = current["source_rows"]
        start = int(current["offset"])
        end = start + len(label_values)
        if end > len(source_rows):
            raise RuntimeError(
                f"Validation predictions exceed source rows for {current['source']}"
            )
        task = current["task"]
        for index, (label, score) in enumerate(zip(label_values, score_values)):
            peptide, receptor, source_label = source_rows[start + index]
            if int(label) != int(source_label):
                raise RuntimeError(
                    f"Validation labels are out of order for {current['source']}"
                )
            if abs(int(label) - float(score)) <= self.THRESHOLD:
                continue
            if task == "phla":
                current["rows"].append(
                    {
                        "hla": receptor,
                        "peptide": peptide,
                        "label": int(label),
                        "prediction_score": f"{float(score):.10f}",
                        "source": current["source"],
                    }
                )
            else:
                current["rows"].append(
                    {
                        "peptide": peptide,
                        "tcr": receptor,
                        "label": int(label),
                        "prediction_score": f"{float(score):.10f}",
                        "source": current["source"],
                    }
                )
        current["offset"] = end

    def finish_split(self) -> None:
        current = self._current_split
        if current is None:
            return
        source_rows = current["source_rows"]
        if source_rows is not None and int(current["offset"]) != len(source_rows):
            raise RuntimeError(
                f"Validation predictions do not cover source rows for "
                f"{current['source']}: {current['offset']} != {len(source_rows)}"
            )
        self._round_rows[current["task"]].extend(current["rows"])
        self._current_split = None

    def write_round(self, round_id: int) -> None:
        if self._round_id != round_id:
            raise RuntimeError("Validation CSV round state is not initialized")
        if self._current_split is not None:
            raise RuntimeError("Cannot write validation CSVs with an active split")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for task in self.tasks:
            directory_name, fieldnames = self._TASK_LAYOUT[task]
            path = (
                self.output_dir
                / directory_name
                / f"{directory_name}_round_{round_id}.csv"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._round_rows[task])
            temporary.replace(path)


def _train_metric_summary(result: EpochResult) -> str:
    """Render train losses with their distinct optimization meanings."""

    metrics = result.metrics
    return (
        f"loss={metrics.loss:.4f} "
        f"clean_loss={result.clean_loss:.4f} "
        f"adversarial_loss={result.adversarial_loss:.4f} "
        f"total_loss={result.total_loss:.4f} "
        f"AUROC={metrics.auroc:.4f} "
        f"AUPR={metrics.aupr:.4f} "
        f"ACC={metrics.accuracy:.4f} "
        f"MCC={metrics.mcc:.4f} "
        f"F1={metrics.f1:.4f} "
        f"NLL={metrics.nll:.4f} "
        f"ECE={metrics.ece:.4f}"
    )


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_optimizer(
    model: UnifiedBindingModel,
    config: ExperimentConfig,
    *,
    module_specs: tuple | None = None,
) -> torch.optim.Optimizer:
    training = config.training
    parameter_groups: list[dict[str, Any]] = []

    def configured_lr(name: str, fallback: float) -> float:
        value = getattr(training, name, None)
        return float(fallback if value is None else value)

    if module_specs is None:
        module_specs = (
            (
                "shared_peptide_adapter",
                model.peptide_adapter,
                configured_lr("shared_peptide_adapter_lr", training.adapter_lr),
            ),
            (
                "hla_adapter",
                model.hla_adapter,
                configured_lr("hla_adapter_lr", training.adapter_lr),
            ),
            (
                "tcr_adapter",
                model.tcr_adapter,
                configured_lr("tcr_adapter_lr", training.adapter_lr),
            ),
            (
                "phla_cross_attention",
                model.phla_cross_attention,
                configured_lr("phla_cross_attention_lr", training.cross_attention_lr),
            ),
            (
                "ptcr_cross_attention",
                model.ptcr_cross_attention,
                configured_lr("ptcr_cross_attention_lr", training.cross_attention_lr),
            ),
            (
                "phla_pooling",
                model.phla_pooling,
                configured_lr("phla_pooling_lr", training.pooling_lr),
            ),
            (
                "ptcr_pooling",
                model.ptcr_pooling,
                configured_lr("ptcr_pooling_lr", training.pooling_lr),
            ),
            (
                "phla_classifier",
                model.phla_classifier,
                configured_lr("phla_classifier_lr", training.classifier_lr),
            ),
            (
                "ptcr_classifier",
                model.ptcr_classifier,
                configured_lr("ptcr_classifier_lr", training.classifier_lr),
            ),
        )

    def scheduler_task(module_name: str) -> str:
        if module_name in {"shared_peptide_adapter", "shared_peptide_encoder"}:
            return "shared"
        if module_name in {
            "hla_adapter",
            "hla_encoder",
            "phla_cross_attention",
            "phla_pooling",
            "phla_classifier",
        }:
            return "phla"
        return "ptcr"

    for module_name, module, learning_rate in module_specs:
        decay_parameters: list[nn.Parameter] = []
        no_decay_parameters: list[nn.Parameter] = []
        for name, parameter in module.named_parameters():
            is_no_decay = (
                training.no_decay_bias_and_norm
                and (
                    parameter.ndim <= 1
                    or name.endswith("bias")
                    or "norm" in name.lower()
                )
            )
            (no_decay_parameters if is_no_decay else decay_parameters).append(parameter)
        if decay_parameters:
            parameter_groups.append(
                {
                    "name": f"{module_name}/decay",
                    "params": decay_parameters,
                    "lr": learning_rate,
                    "weight_decay": training.weight_decay,
                    "scheduler_task": scheduler_task(module_name),
                }
            )
        if no_decay_parameters:
            parameter_groups.append(
                {
                    "name": f"{module_name}/no_decay",
                    "params": no_decay_parameters,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                    "scheduler_task": scheduler_task(module_name),
                }
            )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.0,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float = 0.05,
    warmup_steps: Optional[int] = None,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    resolved_warmup_steps = (
        max(0, int(warmup_steps))
        if warmup_steps is not None
        else max(1, round(total_steps * warmup_ratio))
    )
    if resolved_warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")

    def multiplier(step: int) -> float:
        if resolved_warmup_steps and step < resolved_warmup_steps:
            return max(step, 1) / resolved_warmup_steps
        progress = min(
            1.0,
            (step - resolved_warmup_steps)
            / max(1, total_steps - resolved_warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class TaskAwareCosineScheduler:
    """Cosine scheduler whose private groups advance only on their task steps.

    Stage 1 uses one AdamW optimizer so shared peptide parameters keep one set of
    moments.  A regular LambdaLR would nevertheless advance every parameter
    group's schedule on every pHLA and pTCR batch.  This small scheduler keeps
    the optimizer shared while tracking a separate update count for shared,
    pHLA-private, and pTCR-private groups.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_ratio: float,
        warmup_steps: Optional[int],
        min_lr_ratio: float,
        task_total_steps: Mapping[str, int],
        task_warmup_steps: Mapping[str, Optional[int]],
    ):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        resolved_warmup_steps = (
            max(0, int(warmup_steps))
            if warmup_steps is not None
            else max(1, round(total_steps * warmup_ratio))
        )
        if resolved_warmup_steps > total_steps:
            raise ValueError("warmup_steps cannot exceed total_steps")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.warmup_steps = int(resolved_warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.task_total_steps = {
            task: int(steps) for task, steps in task_total_steps.items()
        }
        if any(steps < 1 for steps in self.task_total_steps.values()):
            raise ValueError("task_total_steps must be positive")
        self.task_warmup_steps = {
            task: (
                max(0, int(value))
                if value is not None
                else max(1, round(self.task_total_steps[task] * warmup_ratio))
            )
            for task, value in (
                (task, task_warmup_steps.get(task))
                for task in self.task_total_steps
            )
        }
        if any(
            self.task_warmup_steps[task] > self.task_total_steps[task]
            for task in self.task_total_steps
        ):
            raise ValueError("task warmup steps cannot exceed task total steps")
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.group_steps = [0 for _ in optimizer.param_groups]
        self.global_step = 0
        self.task_steps = {task: 0 for task in self.task_total_steps}
        self.last_epoch = 0
        self._set_learning_rates()

    @staticmethod
    def _multiplier(
        step: int,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, total_steps - warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def _schedule_for_group(self, group: Mapping[str, Any]) -> tuple[int, int]:
        task = group.get("scheduler_task")
        if task in self.task_total_steps:
            return self.task_total_steps[task], self.task_warmup_steps[task]
        return self.total_steps, self.warmup_steps

    def _set_learning_rates(self) -> None:
        for index, group in enumerate(self.optimizer.param_groups):
            total_steps, warmup_steps = self._schedule_for_group(group)
            multiplier = self._multiplier(
                self.group_steps[index],
                total_steps,
                warmup_steps,
                self.min_lr_ratio,
            )
            group["lr"] = self.base_lrs[index] * multiplier
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, task: str | None = None) -> None:
        if task is not None and task not in self.task_total_steps:
            raise ValueError(f"Unknown scheduler task: {task}")
        self.global_step += 1
        if task is not None:
            self.task_steps[task] = self.task_steps.get(task, 0) + 1
        for index, group in enumerate(self.optimizer.param_groups):
            group_task = group.get("scheduler_task")
            should_step = (
                task is None
                or group_task is None
                or group_task == "shared"
                or group_task == task
            )
            if should_step:
                self.group_steps[index] += 1
        self.last_epoch = self.global_step
        self._set_learning_rates()

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "total_steps": self.total_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "task_total_steps": dict(self.task_total_steps),
            "task_warmup_steps": dict(self.task_warmup_steps),
            "base_lrs": list(self.base_lrs),
            "group_steps": list(self.group_steps),
            "global_step": self.global_step,
            "task_steps": dict(self.task_steps),
            "last_epoch": self.last_epoch,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        group_steps = state_dict.get("group_steps")
        if group_steps is None:
            # Best-effort compatibility with the old LambdaLR state.  New
            # Stage1 checkpoints always contain the per-group counters.
            legacy_step = int(state_dict.get("last_epoch", 0))
            group_steps = [legacy_step for _ in self.optimizer.param_groups]
        if len(group_steps) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler group count does not match optimizer")
        self.group_steps = [int(value) for value in group_steps]
        self.global_step = int(state_dict.get("global_step", state_dict.get("last_epoch", 0)))
        saved_task_steps = state_dict.get("task_steps", {})
        self.task_steps = {
            task: int(saved_task_steps.get(task, 0))
            for task in self.task_total_steps
        }
        saved_base_lrs = state_dict.get("base_lrs")
        if saved_base_lrs is not None and len(saved_base_lrs) == len(self.base_lrs):
            self.base_lrs = [float(value) for value in saved_base_lrs]
        self.last_epoch = int(state_dict.get("last_epoch", self.global_step))
        self._set_learning_rates()


class Stage1Trainer:
    tasks: tuple[Task, ...] = ("phla", "ptcr")
    display_name = "Stage 1"
    artifact_namespace = "stage1"
    checkpoint_architecture = "stage1"
    shared_freeze_state_key = "freeze_shared_adapter"
    joint_checkpoint_type = "stage1_joint_model"

    def __init__(
        self,
        model: UnifiedBindingModel,
        config: ExperimentConfig,
        train_loaders: Mapping[Task, Any],
        val_loaders: Mapping[Task, Any],
        tracker: SwanLabTracker,
        cache_metadata: Optional[Mapping[str, Any]] = None,
        online_encoders: Optional[OnlineEncoderBundle] = None,
        input_mode: str = "cache",
        run_timestamp: Optional[str] = None,
        best_umap_visualizer: Optional[Any] = None,
        umap_train_loaders: Optional[Mapping[Task, Any]] = None,
    ):
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = {
            task: dict(loaders) if isinstance(loaders, Mapping) else {"val": loaders}
            for task, loaders in val_loaders.items()
        }
        self.data_filter_stats = {}
        for task, loader in self.train_loaders.items():
            dataset = getattr(loader, "dataset", None)
            stats = getattr(dataset, "filter_stats", None)
            if stats:
                self.data_filter_stats[f"{task}/train"] = dict(stats)
        for task, split_loaders in self.val_loaders.items():
            for split_name, loader in split_loaders.items():
                dataset = getattr(loader, "dataset", None)
                stats = getattr(dataset, "filter_stats", None)
                if stats:
                    self.data_filter_stats[f"{task}/{split_name}"] = dict(stats)
        self.tracker = tracker
        self.cache_metadata = dict(cache_metadata or {})
        self.online_encoders = online_encoders
        self.input_mode = input_mode
        self.umap_train_loaders = dict(umap_train_loaders or {})
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.train_criteria = {
            task: nn.CrossEntropyLoss(
                reduction="none",
                label_smoothing=getattr(config.training, f"{task}_label_smoothing"),
            )
            for task in self.tasks
        }
        # Validation always reports unsmoothed negative log likelihood.
        self.eval_criterion = nn.CrossEntropyLoss()
        self.criterion = self.eval_criterion
        self.optimizer = self._build_optimizer(model, config)
        self.task_step_limits = self._build_task_step_limits()
        steps_per_pair = sum(self.task_step_limits.values())
        scheduler_rounds = config.training.scheduler_total_rounds
        if scheduler_rounds is None:
            scheduler_rounds = config.training.max_rounds
        scheduled_rounds = config.training.warmup_epochs_per_task + scheduler_rounds
        total_steps = steps_per_pair * scheduled_rounds
        scheduler_warmup_steps = None
        if config.training.scheduler_warmup_rounds is not None:
            scheduler_warmup_steps = steps_per_pair * (
                config.training.warmup_epochs_per_task
                + config.training.scheduler_warmup_rounds
            )
        scheduler_task_total_steps = {
            task: self.task_step_limits[task] * scheduled_rounds
            for task in self.task_step_limits
        }
        scheduler_task_warmup_steps = {
            task: (
                self.task_step_limits[task]
                * (
                    config.training.warmup_epochs_per_task
                    + config.training.scheduler_warmup_rounds
                )
                if config.training.scheduler_warmup_rounds is not None
                else None
            )
            for task in self.task_step_limits
        }
        self.scheduler = TaskAwareCosineScheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=config.training.scheduler_warmup_ratio,
            warmup_steps=scheduler_warmup_steps,
            min_lr_ratio=config.training.scheduler_min_lr_ratio,
            task_total_steps=scheduler_task_total_steps,
            task_warmup_steps=scheduler_task_warmup_steps,
        )
        self.scheduler_total_steps = total_steps
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_task_total_steps = scheduler_task_total_steps
        self.scheduler_task_warmup_steps = scheduler_task_warmup_steps
        self._group_weight_maps = {
            task: self._build_group_weight_map(task) for task in self.tasks
        }
        scaler_enabled = (
            self.device.type == "cuda"
            and config.training.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.fgm = ModuleFGM(model)

        self.stage: Stage = "warmup"
        self.warmup_completed: dict[Task, int] = {task: 0 for task in self.tasks}
        self.current_round = 0
        self.best_task_scores: dict[Task, float] = {
            task: -float("inf") for task in self.tasks
        }
        self.best_selected_nll: dict[Task, float] = {
            task: float("inf") for task in self.tasks
        }
        self.best_selected_ece: dict[Task, float] = {
            task: float("inf") for task in self.tasks
        }
        self.best_val_metrics: dict[Task, dict[str, float | int]] = {}
        self.no_improve_rounds: dict[Task, int] = {task: 0 for task in self.tasks}
        self.stopped_tasks: dict[Task, bool] = {task: False for task in self.tasks}
        self.best_round_id = 0
        self.best_round_score = -float("inf")
        self.round_no_improve_rounds = 0
        self.round_stopped = False
        self.current_task_scores: dict[Task, float] = {
            task: -float("inf") for task in self.tasks
        }
        self.retention_violations: dict[Task, int] = {task: 0 for task in self.tasks}
        self.freeze_shared_sequence = False
        self.global_step = 0
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.validation_error_csv = ValidationErrorCSVWriter(
            self.config.paths.project_root,
            self.run_timestamp,
            peptide_min_length=self.config.model.peptide_min_length,
            peptide_max_length=self.config.model.peptide_max_length,
            artifact_namespace=self.artifact_namespace,
            tasks=self.tasks,
        )
        self.best_umap_visualizer = best_umap_visualizer
        self.last_checkpoint_path = self.run_dir / timestamped_filename(
            "last.pt", self.run_timestamp
        )
        self.warmup_checkpoint_path = self.run_dir / timestamped_filename(
            "warmup_last.pt", self.run_timestamp
        )
        self.task_checkpoint_paths: dict[Task, Path] = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in TASK_CHECKPOINT_FILENAMES.items() if task in self.tasks
        }
        self.calibration_checkpoint_paths: dict[Task, Path] = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in CALIBRATION_CHECKPOINT_FILENAMES.items() if task in self.tasks
        }
        self.robust_checkpoint_paths: dict[Task, Path] = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in ROBUST_CHECKPOINT_FILENAMES.items() if task in self.tasks
        }
        self.best_calibration_scores: dict[Task, float] = {
            task: float("inf") for task in self.tasks
        }
        self.best_robust_scores: dict[Task, float] = {
            task: -float("inf") for task in self.tasks
        }
        self.best_calibration_metrics: dict[Task, dict[str, Any]] = {}
        self.best_robust_metrics: dict[Task, dict[str, Any]] = {}
        self.best_round_checkpoint_path = (
            self.run_dir
            / "best_round"
            / timestamped_filename(
                BEST_ROUND_CHECKPOINT_FILENAME,
                self.run_timestamp,
            )
        )

    def _build_optimizer(
        self,
        model: UnifiedBindingModel,
        config: ExperimentConfig,
    ) -> torch.optim.Optimizer:
        """Optimizer factory hook for architecture-compatible Stage1 variants."""

        return build_optimizer(model, config)

    @property
    def freeze_shared_adapter(self) -> bool:
        """Compatibility view for callers of the original Stage1 trainer."""
        return self.freeze_shared_sequence

    @freeze_shared_adapter.setter
    def freeze_shared_adapter(self, value: bool) -> None:
        self.freeze_shared_sequence = value

    def _feature_embeddings(self, output) -> dict[str, torch.Tensor]:
        return {
            "encoder": output.encoder_embedding,
            "adapter": output.adapter_embedding,
            "classifier_input": output.pooled_embedding,
        }

    def _autocast(self):
        precision = self.config.training.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _move_batch(
        self,
        task: Task,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        hidden_dtype = torch.float32
        if self.device.type == "cuda":
            if self.config.training.mixed_precision == "fp16":
                hidden_dtype = torch.float16
            elif self.config.training.mixed_precision == "bf16":
                hidden_dtype = torch.bfloat16
        if "peptide_sequences" in batch:
            if self.online_encoders is None:
                raise RuntimeError("Raw sequences require online_encoders")
            encoded = self.online_encoders.encode(
                task,
                batch["peptide_sequences"],
                batch["receptor_sequences"],
            )
        else:
            encoded = batch
        moved = {
            "peptide_hidden": encoded["peptide_hidden"].to(
                self.device, dtype=hidden_dtype, non_blocking=non_blocking
            ),
            "peptide_mask": encoded["peptide_mask"].to(
                self.device, dtype=torch.bool, non_blocking=non_blocking
            ),
            "receptor_hidden": encoded["receptor_hidden"].to(
                self.device, dtype=hidden_dtype, non_blocking=non_blocking
            ),
            "receptor_mask": encoded["receptor_mask"].to(
                self.device, dtype=torch.bool, non_blocking=non_blocking
            ),
            "labels": batch["labels"].to(
                self.device, dtype=torch.long, non_blocking=non_blocking
            ),
        }
        group_ids = batch.get("group_ids")
        if group_ids is not None:
            moved["group_ids"] = group_ids.to(
                self.device, dtype=torch.long, non_blocking=non_blocking
            )
        else:
            moved["group_ids"] = None
        sample_weights = self._sample_weights_for_batch(task, batch)
        moved["sample_weights"] = (
            sample_weights.to(self.device, non_blocking=non_blocking)
            if sample_weights is not None
            else None
        )
        return moved

    def _build_task_step_limits(self) -> dict[Task, int]:
        """Compute fixed per-task steps for an optional balanced cycle."""

        limits = {
            task: max(1, len(loader))
            for task, loader in self.train_loaders.items()
        }
        if not self.config.training.task_step_ratio_enabled:
            return limits
        available = [limits[task] for task in self.tasks if task in limits]
        if not available:
            return limits
        base_steps = min(available)
        result: dict[Task, int] = {}
        for task, ratio_name in (
            ("phla", "phla_task_step_ratio"),
            ("ptcr", "ptcr_task_step_ratio"),
        ):
            if task not in limits:
                continue
            requested = max(
                1,
                int(round(base_steps * getattr(self.config.training, ratio_name))),
            )
            result[task] = min(limits[task], requested)
        return result

    def _build_group_weight_map(self, task: Task) -> dict[tuple[int, int], float]:
        """Build inverse-sqrt group/label weights from the training dataset."""

        strategy = getattr(self.config.training, f"{task}_group_weighting", "none")
        if strategy == "none":
            return {}
        loader = self.train_loaders.get(task)
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return {}
        group_ids = getattr(dataset, "group_ids", None)
        labels = getattr(dataset, "labels", None)
        if group_ids is None:
            rows = getattr(dataset, "rows", None)
            if rows is None:
                return {}
            group_ids = np.asarray(
                [
                    stable_group_id(row[0] if task == "ptcr" else row[1])
                    for row in rows
                ],
                dtype=np.int64,
            )
        if labels is None:
            rows = getattr(dataset, "rows", None)
            if rows is None:
                return {}
            labels = np.asarray([int(row[2]) for row in rows], dtype=np.int64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        if len(group_ids) != len(labels) or not len(labels):
            return {}
        keys = list(zip(group_ids.tolist(), labels.tolist()))
        counts: dict[tuple[int, int], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        raw = {key: 1.0 / math.sqrt(count) for key, count in counts.items()}
        mean_weight = float(np.mean(list(raw.values())))
        if mean_weight <= 0:
            return {}
        weights = {
            key: value / mean_weight for key, value in raw.items()
        }
        minimum = self.config.training.group_weight_min
        maximum = self.config.training.group_weight_max
        return {
            key: float(np.clip(value, minimum, maximum))
            for key, value in weights.items()
        }

    def _sample_weights_for_batch(
        self,
        task: Task,
        batch: Mapping[str, Any],
    ) -> torch.Tensor | None:
        weight_map = self._group_weight_maps.get(task, {})
        group_ids = batch.get("group_ids")
        labels = batch.get("labels")
        if not weight_map or group_ids is None or labels is None:
            return None
        return torch.tensor(
            [
                weight_map.get((int(group), int(label)), 1.0)
                for group, label in zip(group_ids, labels)
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if weights is None:
            return values.mean()
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _task_patience(self, task: Task) -> int:
        configured = getattr(
            self.config.training,
            f"{task}_early_stopping_patience",
            None,
        )
        return int(
            self.config.training.early_stopping_patience
            if configured is None
            else configured
        )

    def _round_patience(self) -> int:
        configured = self.config.training.round_early_stopping_patience
        if configured is not None:
            return int(configured)
        return int(self.config.training.early_stopping_patience)

    def _adversarial_settings(
        self,
        task: Task,
        phase_name: str,
    ) -> tuple[float, float, float]:
        epsilon = getattr(self.config.training, f"{task}_fgm_epsilon", None)
        epsilon = float(
            self.config.training.fgm_epsilon if epsilon is None else epsilon
        )
        configured_weight = float(
            getattr(
                self.config.training,
                f"{task}_adversarial_loss_weight",
                1.0,
            )
        )
        round_id = 0
        if phase_name.startswith("round_"):
            try:
                round_id = int(phase_name.rsplit("_", 1)[1])
            except ValueError:
                round_id = 0
        warmup_rounds = self.config.training.adversarial_warmup_rounds
        ramp = (
            1.0
            if warmup_rounds <= 0
            else min(1.0, max(0, round_id) / warmup_rounds)
        )
        return epsilon, configured_weight * ramp, ramp

    def _forward(
        self,
        task: Task,
        batch: Mapping[str, torch.Tensor],
        *,
        return_features: bool = False,
    ):
        return self.model(
            task=task,
            peptide_hidden=batch["peptide_hidden"],
            peptide_mask=batch["peptide_mask"],
            receptor_hidden=batch["receptor_hidden"],
            receptor_mask=batch["receptor_mask"],
            return_features=return_features,
        )

    def train_epoch(
        self,
        task: Task,
        loader,
        *,
        phase_name: str,
        use_fgm: bool,
    ) -> EpochResult:
        self.model.set_trainable_task(
            task,
            freeze_shared=self.freeze_shared_sequence,
        )
        self.model.train()
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        clean_loss_sum = 0.0
        adversarial_loss_sum = 0.0
        total_loss_sum = 0.0
        samples = 0
        optimizer_steps = 0
        attacked_counts: list[int] = []
        fgm_mean_norms: list[float] = []
        fgm_max_norms: list[float] = []
        model_gradient_norms: list[float] = []
        fgm_epsilon, adversarial_weight, adversarial_ramp = (
            self._adversarial_settings(task, phase_name)
        )
        train_criterion = self.train_criteria[task]

        step_limit = self.task_step_limits.get(task)
        batches = islice(loader, step_limit) if step_limit is not None else loader
        for batch_index, cpu_batch in enumerate(batches, start=1):
            batch = self._move_batch(task, cpu_batch)
            batch_samples = int(batch["labels"].shape[0])
            self.optimizer.zero_grad(set_to_none=True)
            gradient_scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0

            with self._autocast():
                clean_output = self._forward(task, batch)
                clean_per_sample = train_criterion(
                    clean_output.logits,
                    batch["labels"],
                )
                clean_loss = self._weighted_mean(
                    clean_per_sample,
                    batch["sample_weights"],
                )
            loss_normalizer = 1.0 + adversarial_weight
            self.scaler.scale(clean_loss / loss_normalizer).backward()

            adversarial_loss_value = 0.0
            total_loss = clean_loss
            fgm_stats: Optional[FGMStats] = None
            if use_fgm and adversarial_weight > 0:
                fgm_stats = self.fgm.attack(
                    self.model.fgm_modules(task),
                    epsilon=fgm_epsilon,
                )
                try:
                    with self._autocast():
                        adversarial_output = self._forward(task, batch)
                        adversarial_per_sample = train_criterion(
                            adversarial_output.logits,
                            batch["labels"],
                        )
                        adversarial_loss = self._weighted_mean(
                            adversarial_per_sample,
                            batch["sample_weights"],
                        )
                    self.scaler.scale(
                        adversarial_loss * adversarial_weight / loss_normalizer
                    ).backward()
                    adversarial_loss_value = float(adversarial_loss.detach().cpu())
                    total_loss = (
                        clean_loss + adversarial_weight * adversarial_loss
                    ) / loss_normalizer
                finally:
                    self.fgm.restore()

            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                self.config.training.gradient_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(task)
            self.global_step += 1
            optimizer_steps += 1

            clean_loss_value = float(clean_loss.detach().cpu())
            clean_loss_sum += clean_loss_value * batch_samples
            adversarial_loss_sum += adversarial_loss_value * batch_samples
            total_loss_sum += float(total_loss.detach().cpu()) * batch_samples
            samples += batch_samples
            probabilities.append(
                torch.softmax(clean_output.logits.detach(), dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch["labels"].detach().cpu().numpy())
            model_gradient_norms.append(float(gradient_norm.detach().cpu()))
            if fgm_stats is not None:
                attacked_counts.append(fgm_stats.attacked_parameter_count)
                fgm_mean_norms.append(
                    fgm_stats.mean_gradient_norm / gradient_scale
                )
                fgm_max_norms.append(fgm_stats.max_gradient_norm / gradient_scale)

            if (
                batch_index % self.config.training.log_every_steps == 0
                or batch_index == (step_limit or len(loader))
            ):
                learning_rates = {
                    f"learning_rate/{group.get('name', index)}": group["lr"]
                    for index, group in enumerate(self.optimizer.param_groups)
                }
                self.tracker.log(
                    {
                        f"{phase_name}/{task}/batch_clean_loss": clean_loss_value,
                        f"{phase_name}/{task}/batch_adversarial_loss": (
                            adversarial_loss_value
                        ),
                        f"{phase_name}/{task}/batch_gradient_norm": float(
                            gradient_norm.detach().cpu()
                        ),
                        f"{phase_name}/{task}/fgm_epsilon": (
                            fgm_epsilon if use_fgm and adversarial_weight > 0 else 0.0
                        ),
                        f"{phase_name}/{task}/adversarial_loss_weight": (
                            adversarial_weight
                        ),
                        f"{phase_name}/{task}/adversarial_ramp": (
                            adversarial_ramp
                        ),
                        **learning_rates,
                    },
                    step=self.global_step,
                )

        average_clean_loss = clean_loss_sum / samples
        average_adversarial_loss = adversarial_loss_sum / samples
        average_total_loss = total_loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=average_clean_loss,
            threshold=self.config.training.threshold,
            strict_threshold=True,
        )
        return EpochResult(
            metrics=metrics,
            clean_loss=average_clean_loss,
            adversarial_loss=average_adversarial_loss,
            total_loss=average_total_loss,
            optimizer_steps=optimizer_steps,
            samples=samples,
            attacked_parameter_count=_mean(attacked_counts),
            fgm_mean_gradient_norm=_mean(fgm_mean_norms),
            fgm_max_gradient_norm=max(fgm_max_norms, default=0.0),
            model_gradient_norm=_mean(model_gradient_norms),
        )

    @torch.inference_mode()
    def evaluate(
        self,
        task: Task,
        loader,
        *,
        visualization_split: Optional[str] = None,
    ) -> BinaryMetrics:
        self.model.eval()
        collect_umap = (
            self.best_umap_visualizer is not None
            and visualization_split is not None
        )
        if collect_umap:
            self.best_umap_visualizer.begin_split(task, visualization_split)
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        group_ids: list[np.ndarray] = []
        has_group_ids = True
        loss_sum = 0.0
        samples = 0
        for cpu_batch in loader:
            batch = self._move_batch(task, cpu_batch)
            with self._autocast():
                output = self._forward(
                    task,
                    batch,
                    return_features=collect_umap,
                )
                loss = self.eval_criterion(output.logits, batch["labels"])
            batch_samples = int(batch["labels"].shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            positive_probabilities = torch.softmax(output.logits, dim=1)[:, 1]
            score_values = positive_probabilities.float().cpu().numpy()
            label_values = batch["labels"].cpu().numpy()
            probabilities.append(score_values)
            labels.append(label_values)
            self.validation_error_csv.record_batch(label_values, score_values)
            if batch["group_ids"] is None:
                has_group_ids = False
            else:
                group_ids.append(batch["group_ids"].cpu().numpy())
            if collect_umap:
                self.best_umap_visualizer.observe(
                    task,
                    visualization_split,
                    self._feature_embeddings(output),
                    batch["labels"],
                    positive_probabilities,
                )
        return compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=loss_sum / samples,
            threshold=self.config.training.threshold,
            strict_threshold=True,
            group_ids=np.concatenate(group_ids) if has_group_ids else None,
        )

    @torch.inference_mode()
    def _collect_umap_train(self, task: Task) -> None:
        """Collect fixed training rows for a task-best UMAP snapshot."""

        if self.best_umap_visualizer is None:
            return
        loader = self.umap_train_loaders.get(task)
        if loader is None:
            raise RuntimeError(
                f"No sequential UMAP training loader is configured for {task}"
            )
        split_name = "train"
        self.model.eval()
        self.best_umap_visualizer.begin_split(task, split_name)
        for cpu_batch in loader:
            batch = self._move_batch(task, cpu_batch)
            with self._autocast():
                output = self._forward(
                    task,
                    batch,
                    return_features=True,
                )
            positive_probabilities = torch.softmax(output.logits, dim=1)[:, 1]
            self.best_umap_visualizer.observe(
                task,
                split_name,
                self._feature_embeddings(output),
                batch["labels"],
                positive_probabilities,
            )
        self.best_umap_visualizer.finish_split(task, split_name)

    def _evaluate_validation_split(self, task: Task, split_name: str, loader):
        self.validation_error_csv.begin_split(task, split_name, loader)
        try:
            if self.best_umap_visualizer is None:
                metrics = self.evaluate(task, loader)
            else:
                try:
                    metrics = self.evaluate(
                        task,
                        loader,
                        visualization_split=split_name,
                    )
                except Exception as exc:
                    # UMAP is a diagnostic side effect. Never let a plotting or
                    # reducer failure change validation metrics, checkpoint
                    # selection, or stop a potentially expensive training run.
                    self.validation_error_csv.discard_split()
                    self._disable_umap(
                        f"validation collection failed for {task}/{split_name}: {exc}"
                    )
                    self.validation_error_csv.begin_split(task, split_name, loader)
                    metrics = self.evaluate(task, loader)
            if self.best_umap_visualizer is not None:
                self.best_umap_visualizer.finish_split(task, split_name)
            self.validation_error_csv.finish_split()
            return metrics
        except Exception:
            self.validation_error_csv.discard_split()
            raise

    def _disable_umap(self, reason: str) -> None:
        if self.best_umap_visualizer is None:
            return
        print(
            f"[{self.display_name} UMAP][warning] disabled: {reason}",
            flush=True,
        )
        try:
            self.tracker.log(
                {
                    "visualization/best_umap/disabled": 1,
                    "visualization/best_umap/disabled_reason": str(reason),
                },
                step=self.global_step,
            )
        except Exception:
            # Tracking is also auxiliary to the training result.
            pass
        self.best_umap_visualizer = None

    def fit(self) -> dict[str, Any]:
        if self.stage == "complete":
            return self.summary()

        if self.stage == "warmup":
            for task in self.tasks:
                first_epoch = self.warmup_completed[task] + 1
                for epoch in range(
                    first_epoch,
                    self.config.training.warmup_epochs_per_task + 1,
                ):
                    result = self.train_epoch(
                        task,
                        self.train_loaders[task],
                        phase_name="warmup",
                        use_fgm=False,
                    )
                    self.warmup_completed[task] = epoch
                    self.tracker.log(
                        {
                            "stage": "warmup",
                            "task": task,
                            "warmup_epoch": epoch,
                            **result.to_log_dict(f"warmup/{task}/train"),
                        },
                        step=self.global_step,
                    )
                    print(
                        f"[{self.display_name}A][seed={self.config.training.seed}]"
                        f"[task={'pHLA' if task == 'phla' else 'pTCR'}]"
                        f"[epoch={epoch}/"
                        f"{self.config.training.warmup_epochs_per_task}]"
                        f"[step={self.global_step}] "
                        f"{_train_metric_summary(result)}",
                        flush=True,
                    )
                    self.save_checkpoint(self.last_checkpoint_path)

            self.stage = "stage1b"
            self.save_checkpoint(self.warmup_checkpoint_path)
            self.save_checkpoint(self.last_checkpoint_path)

        start_round = self.current_round + 1
        for round_id in range(start_round, self.config.training.max_rounds + 1):
            active_tasks: list[Task] = [
                task
                for task in self.tasks
                if not self.stopped_tasks[task]
            ]
            if not active_tasks:
                break

            train_results: dict[Task, EpochResult] = {}
            for task in active_tasks:
                train_results[task] = self.train_epoch(
                    task,
                    self.train_loaders[task],
                    phase_name=f"round_{round_id}",
                    use_fgm=True,
                )

            if self.best_umap_visualizer is not None:
                try:
                    self.best_umap_visualizer.start_round(round_id)
                except Exception as exc:
                    self._disable_umap(
                        f"round {round_id} initialization failed: {exc}"
                    )
            evaluation_tasks: tuple[Task, ...] = (
                self.tasks
                if self.config.training.retention_eval_enabled
                else tuple(active_tasks)
            )
            self.validation_error_csv.begin_round(round_id)
            val_metrics: dict[Task, dict[str, BinaryMetrics]] = {}
            for task in evaluation_tasks:
                val_metrics[task] = {}
                for split_name, loader in self.val_loaders[task].items():
                    val_metrics[task][split_name] = self._evaluate_validation_split(
                        task,
                        split_name,
                        loader,
                    )
            self.validation_error_csv.write_round(round_id)
            scores: dict[Task, float] = {
                task: _macro_validation_score(metrics_by_split)
                for task, metrics_by_split in val_metrics.items()
            }
            calibration_scores: dict[Task, float] = {
                task: _macro_metric(
                    metrics_by_split,
                    "nll",
                    lower_is_better=True,
                )
                for task, metrics_by_split in val_metrics.items()
            }
            ece_scores: dict[Task, float] = {
                task: _macro_metric(
                    metrics_by_split,
                    "ece",
                    lower_is_better=True,
                )
                for task, metrics_by_split in val_metrics.items()
            }
            nll_guard_ok: dict[Task, bool] = {}
            ece_guard_ok: dict[Task, bool] = {}
            calibration_guard_ok: dict[Task, bool] = {}
            for task in evaluation_tasks:
                nll_guard_ok[task], ece_guard_ok[task], calibration_guard_ok[task] = (
                    _selection_calibration_guard(
                        calibration_scores[task],
                        ece_scores[task],
                        self.best_selected_nll[task],
                        self.best_selected_ece[task],
                        enabled=self.config.training.selection_calibration_guard_enabled,
                        nll_tolerance=self.config.training.selection_nll_tolerance,
                        ece_tolerance=self.config.training.selection_ece_tolerance,
                    )
                )
            robust_scores: dict[Task, float] = {
                task: _robust_validation_score(metrics_by_split)
                for task, metrics_by_split in val_metrics.items()
            }
            improved: dict[Task, bool] = {}
            score_improved: dict[Task, bool] = {}
            calibration_improved: dict[Task, bool] = {}
            robust_improved: dict[Task, bool] = {}
            newly_stopped: list[Task] = []
            for task in active_tasks:
                score_improved[task] = (
                    scores[task]
                    > self.best_task_scores[task]
                    + self.config.training.early_stopping_min_delta
                )
                improved[task] = score_improved[task] and calibration_guard_ok[task]
                if improved[task]:
                    self.best_task_scores[task] = scores[task]
                    self.best_selected_nll[task] = calibration_scores[task]
                    self.best_selected_ece[task] = ece_scores[task]
                    self.best_val_metrics[task] = {
                        split_name: metrics.to_dict()
                        for split_name, metrics in val_metrics[task].items()
                    }
                    self.no_improve_rounds[task] = 0
                else:
                    self.no_improve_rounds[task] += 1
                    if (
                        self.no_improve_rounds[task]
                        >= self._task_patience(task)
                    ):
                        self.stopped_tasks[task] = True
                        newly_stopped.append(task)
                calibration_improved[task] = (
                    np.isfinite(calibration_scores[task])
                    and calibration_scores[task]
                    < self.best_calibration_scores[task]
                    - self.config.training.early_stopping_min_delta
                )
                robust_improved[task] = (
                    robust_scores[task]
                    > self.best_robust_scores[task]
                    + self.config.training.early_stopping_min_delta
                )
                if calibration_improved[task]:
                    self.best_calibration_scores[task] = calibration_scores[task]
                    self.best_calibration_metrics[task] = {
                        split_name: metrics.to_dict()
                        for split_name, metrics in val_metrics[task].items()
                    }
                if robust_improved[task]:
                    self.best_robust_scores[task] = robust_scores[task]
                    self.best_robust_metrics[task] = {
                        split_name: metrics.to_dict()
                        for split_name, metrics in val_metrics[task].items()
                    }
            round_calibration_guard_ok = (
                not self.config.training.selection_calibration_guard_enabled
                or all(
                    calibration_guard_ok.get(task, False)
                    for task in evaluation_tasks
                )
            )
            if self.config.training.retention_eval_enabled:
                for task in self.tasks:
                    if task not in active_tasks:
                        retention_failed = (
                            scores[task]
                            < self.best_task_scores[task]
                            - self.config.training.retention_tolerance
                        )
                        if retention_failed:
                            self.retention_violations[task] += 1
                            self.freeze_shared_sequence = True
            self.current_round = round_id
            self.current_task_scores = {
                task: float(scores.get(task, self.best_task_scores[task]))
                for task in self.tasks
            }
            round_score = float(np.mean(list(self.current_task_scores.values())))
            round_improved = (
                round_score
                > self.best_round_score
                + self.config.training.early_stopping_min_delta
                and round_calibration_guard_ok
            )
            if round_improved:
                self.best_round_id = round_id
                self.best_round_score = round_score
                self.round_no_improve_rounds = 0
            else:
                self.round_no_improve_rounds += 1
                if (
                    self.round_no_improve_rounds
                    >= self._round_patience()
                ):
                    self.round_stopped = True

            log_values: dict[str, Any] = {
                "stage": "stage1b",
                "round": round_id,
                "early_stopping/round_score": round_score,
                "early_stopping/round_best_score": self.best_round_score,
                "early_stopping/round_no_improve_rounds": (
                    self.round_no_improve_rounds
                ),
                "early_stopping/round_improved": int(round_improved),
                "early_stopping/round_stopped": int(self.round_stopped),
            }
            log_values.update({
                f"round/{task}_optimizer_steps": (
                    train_results[task].optimizer_steps if task in train_results else 0
                )
                for task in self.tasks
            })
            if "phla" in self.tasks and "ptcr" in self.tasks:
                log_values["round/phla_to_ptcr_step_ratio"] = (
                    log_values["round/phla_optimizer_steps"]
                    / max(1, log_values["round/ptcr_optimizer_steps"])
                )
            for task in active_tasks:
                log_values.update(
                    train_results[task].to_log_dict(f"round/{task}/train")
                )
            for task in evaluation_tasks:
                for split_name, metrics in val_metrics[task].items():
                    log_values.update(
                        prefixed_metrics(
                            f"round/{task}/val/{split_name}",
                            metrics,
                        )
                    )
                log_values[f"early_stopping/{task}/score"] = scores[task]
                log_values[f"early_stopping/{task}/best_score"] = (
                    self.best_task_scores[task]
                )
                log_values[f"early_stopping/{task}/score_improved"] = int(
                    score_improved.get(task, False)
                )
                log_values[f"early_stopping/{task}/no_improve_rounds"] = (
                    self.no_improve_rounds[task]
                )
                log_values[f"early_stopping/{task}/stopped"] = int(
                    self.stopped_tasks[task]
                )
                log_values[f"retention/{task}/current_score"] = scores[task]
                log_values[f"retention/{task}/best_score"] = (
                    self.best_task_scores[task]
                )
                log_values[f"retention/{task}/violations"] = (
                    self.retention_violations[task]
                )
                log_values[f"retention/{task}/shared_frozen"] = int(
                    self.freeze_shared_sequence
                )
                log_values[f"selection/{task}/calibration_nll"] = (
                    calibration_scores[task]
                )
                log_values[f"selection/{task}/calibration_ece"] = (
                    ece_scores[task]
                )
                log_values[f"selection/{task}/calibration_improved"] = int(
                    calibration_improved.get(task, False)
                )
                log_values[f"selection/{task}/nll_guard_ok"] = int(
                    nll_guard_ok.get(task, True)
                )
                log_values[f"selection/{task}/ece_guard_ok"] = int(
                    ece_guard_ok.get(task, True)
                )
                log_values[f"selection/{task}/calibration_guard_ok"] = int(
                    calibration_guard_ok.get(task, True)
                )
                log_values[f"selection/{task}/best_selected_nll"] = (
                    self.best_selected_nll[task]
                )
                log_values[f"selection/{task}/best_selected_ece"] = (
                    self.best_selected_ece[task]
                )
                log_values[f"selection/{task}/robust_score"] = robust_scores[task]
                log_values[f"selection/{task}/robust_improved"] = int(
                    robust_improved.get(task, False)
                )
            self.tracker.log(log_values, step=self.global_step)
            print(
                f"[{self.display_name}B][seed={self.config.training.seed}]"
                f"[round={round_id}/{self.config.training.max_rounds}]"
                f"[step={self.global_step}] "
                f"active={','.join(active_tasks)}",
                flush=True,
            )
            for task in active_tasks:
                task_name = "pHLA" if task == "phla" else "pTCR"
                for split_name, metrics in val_metrics[task].items():
                    print(
                        f"  {task_name} {split_name}: "
                        f"{_metric_summary(metrics)}",
                        flush=True,
                    )
                print(
                    f"  {task_name} train: "
                    f"{_train_metric_summary(train_results[task])} | "
                    f"macro_val_score={scores[task]:.4f} "
                    f"best={self.best_task_scores[task]:.4f} "
                    f"score_improved={'yes' if score_improved[task] else 'no'} "
                    f"calibration_guard={'ok' if calibration_guard_ok[task] else 'blocked'} "
                    f"improved={'yes' if improved[task] else 'no'} "
                    f"patience={self.no_improve_rounds[task]}/"
                    f"{self._task_patience(task)} "
                    f"stopped={'yes' if self.stopped_tasks[task] else 'no'}",
                    flush=True,
                )
            print(
                f"  {'Joint' if len(self.tasks) > 1 else 'PH'} round: score={round_score:.4f} "
                f"best={self.best_round_score:.4f} "
                f"improved={'yes' if round_improved else 'no'} "
                f"patience={self.round_no_improve_rounds}/"
                f"{self._round_patience()} "
                f"stopped={'yes' if self.round_stopped else 'no'}",
                flush=True,
            )

            self.save_checkpoint(self.last_checkpoint_path)
            for task in active_tasks:
                if improved[task]:
                    self.save_best_task_checkpoint(task)
                    if self.best_umap_visualizer is not None:
                        try:
                            self._collect_umap_train(task)
                            artifact = self.best_umap_visualizer.render_task(
                                task,
                                round_id=round_id,
                                score=scores[task],
                            )
                            visualization_log = {
                                f"visualization/best_umap/{task}/round": round_id,
                                f"visualization/best_umap/{task}/score": scores[task],
                            }
                            for stage, path in artifact.figures.items():
                                visualization_log[
                                    f"visualization/best_umap/{task}/{stage}_figure"
                                ] = str(path)
                            for stage, path in artifact.coordinates.items():
                                visualization_log[
                                    f"visualization/best_umap/{task}/{stage}_coordinates"
                                ] = str(path)
                            self.tracker.log(
                                visualization_log,
                                step=self.global_step,
                            )
                        except Exception as exc:
                            self._disable_umap(
                                f"round {round_id} render failed for {task}: {exc}"
                            )
                if calibration_improved.get(task, False):
                    self.save_selection_checkpoint(
                        task,
                        role="calibration",
                        score=calibration_scores[task],
                        metrics={
                            split_name: metrics.to_dict()
                            for split_name, metrics in val_metrics[task].items()
                        },
                    )
                if robust_improved.get(task, False):
                    self.save_selection_checkpoint(
                        task,
                        role="robust",
                        score=robust_scores[task],
                        metrics={
                            split_name: metrics.to_dict()
                            for split_name, metrics in val_metrics[task].items()
                        },
                    )
            if round_improved:
                self.save_best_round_checkpoint(round_id, round_score)
            for task in newly_stopped:
                task_name = "pHLA" if task == "phla" else "pTCR"
                print(
                    f"[Early stopping][seed={self.config.training.seed}]"
                    f"[task={task_name}] round={round_id} "
                    f"best_score={self.best_task_scores[task]:.4f}",
                    flush=True,
                )
            if self.round_stopped or all(self.stopped_tasks.values()):
                break

        self.stage = "complete"
        self.save_checkpoint(self.last_checkpoint_path)
        print(
            f"[Complete][seed={self.config.training.seed}] "
            f"round={self.current_round} step={self.global_step} "
            + " ".join(
                f"best_{task}={self.best_task_scores[task]:.4f}" for task in self.tasks
            ),
            flush=True,
        )
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "format_version": 4,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "warmup_completed": dict(self.warmup_completed),
            "current_round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_selected_nll": dict(self.best_selected_nll),
            "best_selected_ece": dict(self.best_selected_ece),
            "selection_calibration_guard": {
                "enabled": self.config.training.selection_calibration_guard_enabled,
                "nll_tolerance": self.config.training.selection_nll_tolerance,
                "ece_tolerance": self.config.training.selection_ece_tolerance,
            },
            "best_val_metrics": self.best_val_metrics,
            "best_calibration_scores": dict(self.best_calibration_scores),
            "best_robust_scores": dict(self.best_robust_scores),
            "best_calibration_metrics": self.best_calibration_metrics,
            "best_robust_metrics": self.best_robust_metrics,
            "no_improve_rounds": dict(self.no_improve_rounds),
            "stopped_tasks": dict(self.stopped_tasks),
            "best_round": self.best_round_id,
            "best_round_score": self.best_round_score,
            "current_task_scores": dict(self.current_task_scores),
            "round_no_improve_rounds": self.round_no_improve_rounds,
            "round_stopped": self.round_stopped,
            "retention_violations": dict(self.retention_violations),
            self.shared_freeze_state_key: self.freeze_shared_sequence,
            "scheduler_total_steps": self.scheduler_total_steps,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_task_total_steps": dict(self.scheduler_task_total_steps),
            "scheduler_task_warmup_steps": dict(self.scheduler_task_warmup_steps),
            "task_step_limits": dict(self.task_step_limits),
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "data_filter_stats": self.data_filter_stats,
            "cache_metadata": self.cache_metadata,
            "input_mode": self.input_mode,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": self.tracker.run_id,
            "dataloader_rng_state": {
                task: loader.generator.get_state()
                for task, loader in self.train_loaders.items()
                if getattr(loader, "generator", None) is not None
            },
            "rng_state": _capture_rng_state(),
        }
        architecture = getattr(self.model, "architecture", None)
        if architecture is not None:
            state["architecture"] = architecture
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        _atomic_torch_save(self.checkpoint_state(), path)

    def save_best_task_checkpoint(self, task: Task) -> None:
        """Save only the task whose own validation score just improved."""

        training_state = self.checkpoint_state()
        task_state = build_task_checkpoint_state(
            training_state,
            task,
            source_checkpoint=self.last_checkpoint_path,
        )
        _atomic_torch_save(
            task_state,
            self.task_checkpoint_paths[task],
        )

    def save_selection_checkpoint(
        self,
        task: Task,
        *,
        role: str,
        score: float,
        metrics: Mapping[str, Any],
    ) -> None:
        """Save a task checkpoint for calibration or robustness selection."""

        if role == "calibration":
            target = self.calibration_checkpoint_paths[task]
        elif role == "robust":
            target = self.robust_checkpoint_paths[task]
        else:
            raise ValueError(f"Unknown Stage1 selection role: {role}")
        state = build_task_checkpoint_state(
            self.checkpoint_state(),
            task,
            source_checkpoint=self.last_checkpoint_path,
            selection_role=role,
            selection_score=score,
            selection_metrics=metrics,
        )
        _atomic_torch_save(state, target)

    def save_best_round_checkpoint(self, round_id: int, round_score: float) -> None:
        """Save one unified PHLA+PTCR checkpoint for the best progressive round."""

        state = self.checkpoint_state()
        state.update(
            {
                "checkpoint_type": self.joint_checkpoint_type,
                "best_round": int(round_id),
                "round_score": float(round_score),
                "model_scope": "joint_heads" if len(self.tasks) > 1 else self.tasks[0],
            }
        )
        _atomic_torch_save(state, self.best_round_checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        expected_architecture = self.checkpoint_architecture
        if checkpoint.get("architecture", "stage1") != expected_architecture:
            raise ValueError(f"Checkpoint architecture does not match {expected_architecture}")
        if checkpoint.get("checkpoint_type") == "task_model":
            raise ValueError("A task model cannot resume training; use a full training checkpoint")
        format_version = checkpoint.get("format_version")
        if format_version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported checkpoint format")
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint.get("input_mode", "cache") != self.input_mode:
            raise ValueError(
                f"Checkpoint input mode {checkpoint.get('input_mode')} does not match "
                f"current mode {self.input_mode}"
            )
        checkpoint_training = checkpoint_config.get("training", {})
        for key in ("fold", "seed"):
            current_value = getattr(self.config.training, key)
            if checkpoint_training.get(key) != current_value:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint_training.get(key)} does not match "
                    f"current {key}={current_value}"
                )
        if checkpoint_config.get("model") != self.config.to_dict()["model"]:
            raise ValueError("Checkpoint model configuration does not match current config")
        checkpoint_cache = checkpoint.get("cache_metadata", {})
        for entity, current_manifest in self.cache_metadata.items():
            saved_manifest = checkpoint_cache.get(entity, {})
            current_fingerprint = current_manifest.get("model", {}).get(
                "checkpoint_fingerprint"
            )
            saved_fingerprint = saved_manifest.get("model", {}).get(
                "checkpoint_fingerprint"
            )
            if saved_fingerprint and saved_fingerprint != current_fingerprint:
                raise ValueError(
                    f"Checkpoint cache fingerprint mismatch for entity '{entity}'"
                )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.stage = checkpoint["stage"]
        self.warmup_completed = {
            task: int(checkpoint["warmup_completed"][task]) for task in self.tasks
        }
        self.current_round = int(checkpoint["current_round"])
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        saved_selected_nll = checkpoint.get("best_selected_nll")
        saved_selected_ece = checkpoint.get("best_selected_ece")
        self.best_selected_nll = {
            task: float(
                saved_selected_nll.get(task, float("inf"))
                if isinstance(saved_selected_nll, Mapping)
                else _macro_metric(
                    self.best_val_metrics.get(task, {}),
                    "nll",
                    lower_is_better=True,
                )
            )
            for task in self.tasks
        }
        self.best_selected_ece = {
            task: float(
                saved_selected_ece.get(task, float("inf"))
                if isinstance(saved_selected_ece, Mapping)
                else _macro_metric(
                    self.best_val_metrics.get(task, {}),
                    "ece",
                    lower_is_better=True,
                )
            )
            for task in self.tasks
        }
        self.best_calibration_scores = {
            task: float(
                checkpoint.get("best_calibration_scores", {}).get(
                    task,
                    float("inf"),
                )
            )
            for task in self.tasks
        }
        self.best_robust_scores = {
            task: float(
                checkpoint.get("best_robust_scores", {}).get(
                    task,
                    -float("inf"),
                )
            )
            for task in self.tasks
        }
        self.best_calibration_metrics = dict(
            checkpoint.get("best_calibration_metrics", {})
        )
        self.best_robust_metrics = dict(checkpoint.get("best_robust_metrics", {}))
        if format_version in {2, 3, 4}:
            self.best_task_scores = {
                task: float(checkpoint["best_task_scores"][task])
                for task in self.tasks
            }
            self.no_improve_rounds = {
                task: int(checkpoint["no_improve_rounds"][task])
                for task in self.tasks
            }
            self.stopped_tasks = {
                task: bool(checkpoint["stopped_tasks"][task])
                for task in self.tasks
            }
        else:
            self.best_task_scores = {
                task: (
                    task_score(self.best_val_metrics[task])
                    if task in self.best_val_metrics
                    else -float("inf")
                )
                for task in self.tasks
            }
            self.no_improve_rounds = {task: 0 for task in self.tasks}
            self.stopped_tasks = {task: False for task in self.tasks}
        self.best_round_id = int(checkpoint.get("best_round", 0))
        self.best_round_score = float(
            checkpoint.get("best_round_score", -float("inf"))
        )
        self.round_no_improve_rounds = int(
            checkpoint.get("round_no_improve_rounds", 0)
        )
        self.round_stopped = bool(checkpoint.get("round_stopped", False))
        self.current_task_scores = {
            task: float(
                checkpoint.get("current_task_scores", {}).get(
                    task,
                    self.best_task_scores[task],
                )
            )
            for task in self.tasks
        }
        self.retention_violations = {
            task: int(checkpoint.get("retention_violations", {}).get(task, 0))
            for task in self.tasks
        }
        self.freeze_shared_sequence = bool(
            checkpoint.get(self.shared_freeze_state_key, False)
        )
        self.global_step = int(checkpoint["global_step"])
        for task, generator_state in checkpoint.get(
            "dataloader_rng_state", {}
        ).items():
            loader = self.train_loaders.get(task)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generator.set_state(generator_state)
        _restore_rng_state(checkpoint.get("rng_state"))

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_selected_nll": dict(self.best_selected_nll),
            "best_selected_ece": dict(self.best_selected_ece),
            "selection_calibration_guard": {
                "enabled": self.config.training.selection_calibration_guard_enabled,
                "nll_tolerance": self.config.training.selection_nll_tolerance,
                "ece_tolerance": self.config.training.selection_ece_tolerance,
            },
            "best_val_metrics": self.best_val_metrics,
            "best_calibration_scores": dict(self.best_calibration_scores),
            "best_robust_scores": dict(self.best_robust_scores),
            "no_improve_rounds": dict(self.no_improve_rounds),
            "stopped_tasks": dict(self.stopped_tasks),
            "best_round": self.best_round_id,
            "best_round_score": self.best_round_score,
            "round_no_improve_rounds": self.round_no_improve_rounds,
            "round_stopped": self.round_stopped,
            "current_task_scores": dict(self.current_task_scores),
            "retention_violations": dict(self.retention_violations),
            self.shared_freeze_state_key: self.freeze_shared_sequence,
            "scheduler_total_steps": self.scheduler_total_steps,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_task_total_steps": dict(self.scheduler_task_total_steps),
            "scheduler_task_warmup_steps": dict(self.scheduler_task_warmup_steps),
            "task_step_limits": dict(self.task_step_limits),
            "data_filter_stats": self.data_filter_stats,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "run_dir": str(self.run_dir),
            "last_checkpoint": str(self.last_checkpoint_path),
            "warmup_checkpoint": str(self.warmup_checkpoint_path),
            "task_checkpoints": {
                task: str(path)
                for task, path in self.task_checkpoint_paths.items()
            },
            "calibration_checkpoints": {
                task: str(path)
                for task, path in self.calibration_checkpoint_paths.items()
            },
            "robust_checkpoints": {
                task: str(path)
                for task, path in self.robust_checkpoint_paths.items()
            },
            "best_round_checkpoint": str(self.best_round_checkpoint_path),
            "best_umap_output_dir": (
                str(self.best_umap_visualizer.output_dir)
                if self.best_umap_visualizer is not None
                else None
            ),
        }


def peek_swanlab_run_id(path: str | Path) -> Optional[str]:
    checkpoint = load_training_checkpoint(path, map_location="cpu")
    return checkpoint.get("swanlab_run_id")


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location,
) -> dict[str, Any]:
    """Load trusted local training state across PyTorch 2.1+ defaults."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_task_checkpoint_state(
    training_checkpoint: Mapping[str, Any],
    task: Task,
    *,
    source_checkpoint: str | Path,
    selection_role: str = "discrimination",
    selection_score: float | None = None,
    selection_metrics: Mapping[str, Any] | None = None,
    state_extractor: Callable | None = None,
) -> dict[str, Any]:
    """Build a lean inference checkpoint containing exactly one task branch."""

    if training_checkpoint.get("checkpoint_type") == "task_model":
        raise ValueError("Expected a training checkpoint, got a task model")
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Training checkpoint does not contain a model state")
    architecture = training_checkpoint.get("architecture", "stage1")
    if state_extractor is None:
        if architecture == "stage1_encoder":
            from .stage1_encoder_model import extract_encoder_task_state_dict
            state_extractor = extract_encoder_task_state_dict
        elif architecture == "stage1":
            state_extractor = extract_task_state_dict
        else:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    best_metrics = training_checkpoint.get("best_val_metrics", {})
    best_task_scores = training_checkpoint.get("best_task_scores", {})
    validation_metrics = best_metrics.get(task, {})
    best_score = best_task_scores.get(task)
    if best_score is None and validation_metrics:
        best_score = task_score(validation_metrics)
    no_improve = training_checkpoint.get("no_improve_rounds", {})
    stopped = training_checkpoint.get("stopped_tasks", {})
    state = {
        "format_version": 2,
        "checkpoint_type": "task_model",
        "task": task,
        "model": state_extractor(model_state, task),
        "config": training_checkpoint.get("config", {}),
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_format_version": training_checkpoint.get("format_version"),
        "stage": training_checkpoint.get("stage"),
        "current_round": training_checkpoint.get("current_round"),
        "global_step": training_checkpoint.get("global_step"),
        "run_timestamp": training_checkpoint.get("run_timestamp"),
        "best_score": best_score,
        "selection_role": selection_role,
        "selection_score": selection_score,
        "selection_metrics": dict(selection_metrics or {}),
        "validation_metrics": validation_metrics,
        "early_stopping": {
            "no_improve_rounds": (
                no_improve.get(task) if isinstance(no_improve, Mapping) else None
            ),
            "stopped": stopped.get(task) if isinstance(stopped, Mapping) else None,
        },
        "data_filter_stats": training_checkpoint.get("data_filter_stats", {}),
        "cache_metadata": training_checkpoint.get("cache_metadata", {}),
        "input_mode": training_checkpoint.get("input_mode", "cache"),
    }
    if "architecture" in training_checkpoint:
        state["architecture"] = architecture
    return state


def export_task_checkpoints(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    artifact_timestamp: str | None = None,
) -> dict[Task, Path]:
    """Split a trusted full Stage1 checkpoint into pHLA and pTCR model files."""

    source = Path(checkpoint_path).resolve()
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    destination = Path(output_dir).resolve() if output_dir else source.parent
    timestamp = validate_artifact_timestamp(
        artifact_timestamp or create_artifact_timestamp()
    )
    targets = {
        task: destination / timestamped_filename(filename, timestamp)
        for task, filename in TASK_CHECKPOINT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Task checkpoint already exists: {names}")
    for task, target in targets.items():
        state = build_task_checkpoint_state(
            checkpoint,
            task,
            source_checkpoint=source,
        )
        _atomic_torch_save(state, target)
    return targets


def _atomic_torch_save(state: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0
