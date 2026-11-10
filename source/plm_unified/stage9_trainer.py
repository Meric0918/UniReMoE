"""Two-stage optimization and constrained model selection for Stage 9."""

from __future__ import annotations

import math
import random
import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .metrics import BinaryMetrics, task_score
from .stage9_checkpoint import atomic_torch_save, load_torch_checkpoint
from .stage9_config import Stage9ExperimentConfig
from .stage9_evaluation import (
    Stage9PairPredictionResult,
    evaluate_stage9_pairs,
)
from .stage9_loss import compute_stage9_pair_loss
from .stage9_model import ProgressiveStage9FineTuningModel
from .trainer import build_scheduler
from .triplet_evaluation import evaluate_stage3_model, select_mcc_threshold


Stage9Phase = Literal["stage9a", "stage9b"]


@dataclass(frozen=True)
class Stage9SelectionMetrics:
    ranking_accuracy: float
    mean_similarity_gap: float
    paired_classification_score: float
    legacy_joint_score: float


def stage9a_candidate_is_better(
    *,
    contrastive_loss: float,
    ranking_accuracy: float,
    mean_similarity_gap: float,
    best_contrastive_loss: float,
    best_ranking_accuracy: float,
    best_mean_similarity_gap: float,
    min_delta: float,
) -> bool:
    if contrastive_loss < best_contrastive_loss - min_delta:
        return True
    if contrastive_loss > best_contrastive_loss + min_delta:
        return False
    if ranking_accuracy > best_ranking_accuracy + min_delta:
        return True
    if ranking_accuracy < best_ranking_accuracy - min_delta:
        return False
    return mean_similarity_gap > best_mean_similarity_gap + min_delta


def candidate_is_better(
    candidate: Stage9SelectionMetrics,
    best: Stage9SelectionMetrics | None,
    *,
    source_legacy_joint: float,
    degradation_tolerance: float,
    min_delta: float,
) -> bool:
    if candidate.legacy_joint_score < source_legacy_joint - degradation_tolerance:
        return False
    if best is None:
        return True
    if candidate.ranking_accuracy > best.ranking_accuracy + min_delta:
        return True
    if candidate.ranking_accuracy < best.ranking_accuracy - min_delta:
        return False
    if candidate.mean_similarity_gap > best.mean_similarity_gap + min_delta:
        return True
    if candidate.mean_similarity_gap < best.mean_similarity_gap - min_delta:
        return False
    return (
        candidate.paired_classification_score
        > best.paired_classification_score + min_delta
    )


def bounded_replay_batches(
    *,
    pair_samples: int,
    replay_ratio: float,
    replay_loader_batches: int,
    replay_batch_size: int,
) -> int:
    if pair_samples < 1 or replay_loader_batches < 1 or replay_batch_size < 1:
        raise ValueError("Replay sizes must be positive")
    requested_samples = replay_sample_cap(
        pair_samples,
        replay_ratio,
        replay_loader_batches * replay_batch_size,
    )
    if requested_samples == 0:
        return 0
    return min(replay_loader_batches, math.ceil(requested_samples / replay_batch_size))


def replay_sample_cap(
    pair_samples: int,
    replay_ratio: float,
    replay_dataset_samples: int,
) -> int:
    """Return a hard upper bound on replay examples for one legacy task."""
    if pair_samples < 0 or replay_dataset_samples < 0:
        raise ValueError("Replay sample counts cannot be negative")
    if replay_ratio < 0:
        raise ValueError("replay_ratio cannot be negative")
    requested_samples = math.floor(pair_samples * replay_ratio)
    return min(replay_dataset_samples, requested_samples)


def _batch_sample_count(batch: Any) -> int:
    if isinstance(batch, Mapping):
        for value in batch.values():
            count = _batch_sample_count(value)
            if count:
                return count
        return 0
    if isinstance(batch, (torch.Tensor, np.ndarray)):
        return int(batch.shape[0]) if batch.ndim else 0
    if isinstance(batch, tuple):
        return _batch_sample_count(batch[0]) if batch else 0
    if isinstance(batch, list):
        if not batch:
            return 0
        if isinstance(batch[0], str):
            return len(batch)
        nested_count = _batch_sample_count(batch[0])
        return nested_count or len(batch)
    return 0


def _slice_batch(batch: Any, size: int) -> Any:
    if isinstance(batch, Mapping):
        return {key: _slice_batch(value, size) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch[:size]
    if isinstance(batch, np.ndarray):
        return batch[:size]
    if isinstance(batch, tuple):
        return tuple(_slice_batch(value, size) for value in batch)
    if isinstance(batch, list):
        if not batch or isinstance(batch[0], str):
            return batch[:size]
        return [_slice_batch(value, size) for value in batch]
    return batch


def iter_capped_replay_batches(
    loader: Iterable[Any],
    max_samples: int,
) -> Iterator[Any]:
    """Yield replay batches without ever crossing ``max_samples`` examples."""
    if max_samples < 0:
        raise ValueError("max_samples cannot be negative")
    remaining = max_samples
    for batch in loader:
        if remaining == 0:
            break
        batch_samples = _batch_sample_count(batch)
        if batch_samples < 1:
            raise ValueError("Could not infer replay batch size")
        if batch_samples > remaining:
            batch = _slice_batch(batch, remaining)
            batch_samples = remaining
        yield batch
        remaining -= batch_samples


def build_stage9_optimizer(
    model: ProgressiveStage9FineTuningModel,
    config: Stage9ExperimentConfig,
    phase: Stage9Phase,
) -> torch.optim.Optimizer:
    values = config.stage9
    groups: list[dict[str, Any]] = []

    def add_group(name: str, parameters, lr: float) -> None:
        parameters = list(parameters)
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr})

    binding = model.binding_model
    if phase == "stage9a":
        add_group(
            "contrastive_modules",
            (
                parameter
                for module in binding.contrastive_modules().values()
                for parameter in module.parameters()
            ),
            values.contrastive_module_lr,
        )
        return torch.optim.AdamW(groups, weight_decay=values.weight_decay)
    if phase != "stage9b":
        raise ValueError(f"Unknown Stage-9 optimizer phase: {phase}")

    add_group(
        "adapters",
        (
            parameter
            for module in (
                binding.peptide_adapter,
                binding.hla_adapter,
                binding.tcr_adapter,
            )
            for parameter in module.parameters()
        ),
        values.adapter_lr,
    )
    add_group(
        "pretrained_modules",
        (
            parameter
            for module in (
                binding.phla_cross_attention,
                binding.ptcr_cross_attention,
                binding.phla_pooling,
                binding.ptcr_pooling,
                binding.phla_classifier,
                binding.ptcr_classifier,
                binding.triplet_cross_attention,
                binding.triplet_pooling,
                binding.triplet_classifier,
            )
            for parameter in module.parameters()
        ),
        values.pretrained_module_lr,
    )
    add_group(
        "contrastive_modules",
        (
            parameter
            for module in binding.contrastive_modules().values()
            for parameter in module.parameters()
        ),
        values.contrastive_module_lr,
    )
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


def stage9_legacy_joint_score(
    metrics: Mapping[str, BinaryMetrics | Mapping[str, float]],
) -> float:
    return (
        0.6 * task_score(metrics["triplet"])
        + 0.2 * task_score(metrics["phla"])
        + 0.2 * task_score(metrics["ptcr"])
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def stage9_training_contract(
    config: Stage9ExperimentConfig,
    data_audit: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = {
        "model": config.to_dict()["model"],
        "stage9": config.to_dict()["stage9"],
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
        "dataset_fingerprint": data_audit.get("dataset_fingerprint"),
        "effective_data": data_audit.get("effective_data"),
    }
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "descriptor": descriptor,
    }


class Stage9Trainer:
    """Warm contrastive heads, then jointly fine-tune with legacy replay."""

    def __init__(
        self,
        model: ProgressiveStage9FineTuningModel,
        config: Stage9ExperimentConfig,
        pair_train_loader,
        pair_val_loader,
        legacy_train_loaders: Mapping[str, Any],
        legacy_val_loaders: Mapping[str, Any],
        *,
        source_stage3: Mapping[str, Any],
        data_audit: Mapping[str, Any],
        tracker=None,
        run_timestamp: str | None = None,
    ):
        required = {"phla", "ptcr", "triplet"}
        if set(legacy_train_loaders) != required or set(legacy_val_loaders) != required:
            raise ValueError("Stage 9 requires phla, ptcr, and triplet replay loaders")
        self.model = model
        self.config = config
        self.pair_train_loader = pair_train_loader
        self.pair_val_loader = pair_val_loader
        self.legacy_train_loaders = dict(legacy_train_loaders)
        self.legacy_val_loaders = dict(legacy_val_loaders)
        self.source_stage3 = dict(source_stage3)
        self.data_audit = dict(data_audit)
        self.training_contract = stage9_training_contract(config, data_audit)
        self.tracker = tracker
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.device = next(model.parameters()).device
        self.criterion = nn.CrossEntropyLoss()
        self.phase: str = "stage9a"
        self.stage9a_epoch = 0
        self.stage9b_round = 0
        self.best_stage9a_loss = float("inf")
        self.best_stage9a_ranking = -float("inf")
        self.best_stage9a_gap = -float("inf")
        self.best_selection: Stage9SelectionMetrics | None = None
        self.source_legacy_joint: float | None = None
        self.source_legacy_metrics: dict[str, dict] = {}
        self.best_validation: dict[str, Any] = {}
        self.selected_threshold = 0.5
        self.bad_rounds = 0
        self.global_step = 0
        self.optimizer_phase: Stage9Phase = "stage9a"
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage9a")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.stage9.mixed_precision == "fp16"
            ),
        )
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage9a_best_path = self.run_dir / timestamped_filename(
            "stage9a_best_contrastive.pt", self.run_timestamp
        )
        self.stage9b_best_path = self.run_dir / timestamped_filename(
            "stage9b_best_constrained.pt", self.run_timestamp
        )
        self.last_path = self.run_dir / timestamped_filename(
            "stage9_last.pt", self.run_timestamp
        )
        self.deploy_path = self.run_dir / timestamped_filename(
            "stage9_best.pt", self.run_timestamp
        )

    def _autocast(self):
        values = self.config.stage9
        if self.device.type != "cuda" or values.mixed_precision == "none":
            return nullcontext()
        dtype = torch.float16 if values.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _planned_optimizer_steps(self, phase: Stage9Phase) -> int:
        accumulation = self.config.stage9.gradient_accumulation_steps
        pair_steps = math.ceil(len(self.pair_train_loader) / accumulation)
        if phase == "stage9a":
            return max(1, pair_steps * self.config.stage9.stage9a_epochs)
        replay_steps = sum(
            math.ceil(self._replay_batches(task) / accumulation)
            for task in ("phla", "ptcr", "triplet")
        )
        return max(
            1,
            (pair_steps + replay_steps) * self.config.stage9.max_rounds,
        )

    def _build_optimization(self, phase: Stage9Phase) -> None:
        self.optimizer_phase = phase
        self.optimizer = build_stage9_optimizer(self.model, self.config, phase)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self._planned_optimizer_steps(phase),
            warmup_ratio=self.config.stage9.scheduler_warmup_ratio,
        )

    def _replay_batches(self, task: str) -> int:
        loader = self.legacy_train_loaders[task]
        return bounded_replay_batches(
            pair_samples=len(self.pair_train_loader.dataset),
            replay_ratio=self.config.stage9.replay_ratio,
            replay_loader_batches=len(loader),
            replay_batch_size=loader.batch_size or self.config.stage9.micro_batch_size,
        )

    def _optimizer_step(self, trainable_parameters) -> float:
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, self.config.stage9.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return float(gradient_norm.detach().cpu())

    def train_pair_epoch(
        self,
        epoch_id: int,
        *,
        phase: Stage9Phase,
    ) -> dict[str, float | int]:
        self.model.set_trainable_stage9("pair", phase)
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        accumulation = self.config.stage9.gradient_accumulation_steps
        total_batches = len(self.pair_train_loader)
        totals = {
            "total_loss": 0.0,
            "contrastive_loss": 0.0,
            "classification_loss": 0.0,
            "ranking_correct": 0.0,
            "similarity_gap": 0.0,
        }
        samples = 0
        optimizer_steps = 0
        for batch_index, batch in enumerate(self.pair_train_loader, start=1):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model.forward_pair(
                    batch["peptide_sequences"],
                    batch["hla_sequences"],
                    batch["positive_tcr_sequences"],
                    batch["negative_tcr_sequences"],
                )
                loss = compute_stage9_pair_loss(
                    output,
                    temperature=self.config.stage9.temperature,
                    contrastive_weight=self.config.stage9.contrastive_loss_weight,
                    classification_weight=self.config.stage9.classification_loss_weight,
                    include_classification=phase == "stage9b",
                )
            self.scaler.scale(loss.total / window_size).backward()
            batch_samples = loss.samples
            samples += batch_samples
            totals["total_loss"] += float(loss.total.detach().cpu()) * batch_samples
            totals["contrastive_loss"] += (
                float(loss.contrastive.detach().cpu()) * batch_samples
            )
            totals["classification_loss"] += (
                float(loss.classification.detach().cpu()) * batch_samples
            )
            totals["ranking_correct"] += loss.ranking_correct
            totals["similarity_gap"] += (
                float(loss.mean_similarity_gap.detach().cpu()) * batch_samples
            )
            if batch_index - window_start + 1 == window_size:
                gradient_norm = self._optimizer_step(trainable)
                optimizer_steps += 1
                if self.tracker and (
                    optimizer_steps % self.config.stage9.log_every_optimizer_steps == 0
                    or batch_index == total_batches
                ):
                    self.tracker.log(
                        {
                            "phase": phase,
                            f"{phase}/pair/epoch_or_round": epoch_id,
                            f"{phase}/pair/running_loss": totals["total_loss"] / samples,
                            f"{phase}/pair/gradient_norm": gradient_norm,
                        },
                        step=self.global_step,
                    )
        if not samples:
            raise RuntimeError("Stage-9 pair training loader is empty")
        return {
            **{name: total / samples for name, total in totals.items()},
            "samples": samples,
            "optimizer_steps": optimizer_steps,
        }

    def _legacy_forward(self, task: str, batch):
        if task == "triplet":
            return self.model.forward_triplet(
                batch["peptide_sequences"],
                batch["hla_sequences"],
                batch["tcr_sequences"],
            )
        return self.model(
            task,
            batch["peptide_sequences"],
            batch["receptor_sequences"],
        )

    def train_replay_epoch(self, task: str, round_id: int) -> dict[str, float | int]:
        self.model.set_trainable_stage9(task, "stage9b")  # type: ignore[arg-type]
        loader = self.legacy_train_loaders[task]
        max_batches = self._replay_batches(task)
        max_samples = replay_sample_cap(
            len(self.pair_train_loader.dataset),
            self.config.stage9.replay_ratio,
            len(loader.dataset),
        )
        if max_samples == 0:
            return {
                "loss": 0.0,
                "samples": 0,
                "optimizer_steps": 0,
                "round": round_id,
            }
        accumulation = self.config.stage9.gradient_accumulation_steps
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        samples = 0
        loss_sum = 0.0
        optimizer_steps = 0
        capped_batches = iter_capped_replay_batches(loader, max_samples)
        for batch_index, batch in enumerate(capped_batches, start=1):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, max_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(
                self.device,
                dtype=torch.long,
                non_blocking=self.device.type == "cuda",
            )
            with self._autocast():
                output = self._legacy_forward(task, batch)
                loss = self.criterion(output.logits, labels)
            self.scaler.scale(loss / window_size).backward()
            batch_samples = int(labels.shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            if batch_index - window_start + 1 == window_size:
                self._optimizer_step(trainable)
                optimizer_steps += 1
        if not samples:
            raise RuntimeError(f"Stage-9 {task} replay loader is empty")
        return {
            "loss": loss_sum / samples,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
            "round": round_id,
        }

    def evaluate_pairs(self) -> Stage9PairPredictionResult:
        return evaluate_stage9_pairs(
            self.model,
            self.pair_val_loader,
            device=self.device,
            mixed_precision=self.config.stage9.mixed_precision,
            temperature=self.config.stage9.temperature,
            threshold=self.selected_threshold,
        )

    def evaluate_legacy(self) -> tuple[dict[str, BinaryMetrics], float]:
        metrics = {
            task: evaluate_stage3_model(
                self.model,
                self.legacy_val_loaders[task],
                task,  # type: ignore[arg-type]
                device=self.device,
                mixed_precision=self.config.stage9.mixed_precision,
                threshold=0.5,
            ).metrics
            for task in ("phla", "ptcr", "triplet")
        }
        return metrics, stage9_legacy_joint_score(metrics)

    @staticmethod
    def _threshold_from_pairs(
        result: Stage9PairPredictionResult,
    ) -> tuple[float, BinaryMetrics]:
        samples = result.metrics.samples
        labels = np.concatenate(
            [np.ones(samples, dtype=np.int64), np.zeros(samples, dtype=np.int64)]
        )
        probabilities = np.concatenate(
            [result.positive_probabilities, result.negative_probabilities]
        )
        threshold, metrics = select_mcc_threshold(labels, probabilities)
        metrics.loss = result.metrics.classification_loss
        return threshold, metrics

    @staticmethod
    def _selection(
        pair: Stage9PairPredictionResult,
        legacy_joint: float,
    ) -> Stage9SelectionMetrics:
        classification = pair.metrics.classification
        return Stage9SelectionMetrics(
            ranking_accuracy=pair.metrics.ranking_accuracy,
            mean_similarity_gap=pair.metrics.mean_similarity_gap,
            paired_classification_score=(classification.auroc + classification.aupr) / 2,
            legacy_joint_score=legacy_joint,
        )

    def _dataloader_rng_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        loaders = {
            "pair/train": self.pair_train_loader,
            "pair/val": self.pair_val_loader,
            **{
                f"legacy/train/{task}": loader
                for task, loader in self.legacy_train_loaders.items()
            },
            **{
                f"legacy/val/{task}": loader
                for task, loader in self.legacy_val_loaders.items()
            },
        }
        for name, loader in loaders.items():
            generator = getattr(loader, "stage9_generator", None) or getattr(
                loader, "generator", None
            )
            if generator is not None:
                state[name] = generator.get_state()
        return state

    def checkpoint_state(self, *, artifact_role: str) -> dict[str, Any]:
        return {
            "format_version": 1,
            "checkpoint_type": "stage9_training",
            "artifact_role": artifact_role,
            "tasks": ["phla", "ptcr", "triplet", "pair"],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_phase": self.optimizer_phase,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "phase": self.phase,
            "stage9a_epoch": self.stage9a_epoch,
            "stage9b_round": self.stage9b_round,
            "best_stage9a_loss": self.best_stage9a_loss,
            "best_stage9a_ranking": self.best_stage9a_ranking,
            "best_stage9a_gap": self.best_stage9a_gap,
            "best_selection": (
                self.best_selection.__dict__ if self.best_selection else None
            ),
            "source_legacy_joint": self.source_legacy_joint,
            "source_legacy_metrics": self.source_legacy_metrics,
            "best_validation": self.best_validation,
            "selected_pair_threshold": self.selected_threshold,
            "bad_rounds": self.bad_rounds,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "source_stage3": self.source_stage3,
            "data_audit": self.data_audit,
            "training_contract": self.training_contract,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": getattr(self.tracker, "run_id", None),
            "rng_state": _rng_state(),
            "dataloader_rng_state": self._dataloader_rng_state(),
        }

    def save_checkpoint(self, path: Path, *, artifact_role: str) -> None:
        atomic_torch_save(self.checkpoint_state(artifact_role=artifact_role), path)

    def save_deploy_checkpoint(self) -> None:
        payload = {
            "format_version": 1,
            "checkpoint_type": "stage9_multitask_model",
            "tasks": ["phla", "ptcr", "triplet", "pair"],
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "source_stage3": self.source_stage3,
            "data_audit": self.data_audit,
            "phase": "complete",
            "stage9b_round": self.stage9b_round,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "selected_pair_threshold": self.selected_threshold,
            "source_legacy_joint": self.source_legacy_joint,
            "best_selection": (
                self.best_selection.__dict__ if self.best_selection else None
            ),
            "validation_metrics": self.best_validation,
        }
        atomic_torch_save(payload, self.deploy_path)

    def _record_best_validation(
        self,
        pair: Stage9PairPredictionResult,
        legacy: Mapping[str, BinaryMetrics],
        legacy_joint: float,
        threshold_metrics: BinaryMetrics,
    ) -> None:
        self.best_validation = {
            "pair": pair.metrics.to_dict(),
            "pair_selected_threshold": self.selected_threshold,
            "pair_selected_threshold_metrics": threshold_metrics.to_dict(),
            "legacy": {
                task: metrics.to_dict() for task, metrics in legacy.items()
            },
            "legacy_joint_score": legacy_joint,
            "legacy_forgetting": (
                legacy_joint - self.source_legacy_joint
                if self.source_legacy_joint is not None
                else 0.0
            ),
        }

    def _log_round(
        self,
        round_id: int,
        pair: Stage9PairPredictionResult,
        selection: Stage9SelectionMetrics,
        improved: bool,
    ) -> None:
        if not self.tracker:
            return
        payload = {
            "phase": "stage9b",
            "stage9b/round": round_id,
            "stage9b/improved": int(improved),
            "stage9b/bad_rounds": self.bad_rounds,
            "stage9b/pair/ranking_accuracy": selection.ranking_accuracy,
            "stage9b/pair/mean_similarity_gap": selection.mean_similarity_gap,
            "stage9b/pair/classification_score": selection.paired_classification_score,
            "stage9b/legacy_joint_score": selection.legacy_joint_score,
            "stage9b/legacy_forgetting": selection.legacy_joint_score
            - float(self.source_legacy_joint),
        }
        self.tracker.log(payload, step=self.global_step)

    def fit(self) -> dict[str, Any]:
        values = self.config.stage9
        if self.phase == "complete":
            if not self.deploy_path.is_file() and self.stage9b_best_path.is_file():
                best = load_torch_checkpoint(self.stage9b_best_path, mmap=False)
                self.model.load_state_dict(best["model"], strict=True)
                self.save_deploy_checkpoint()
            return self.summary()
        if self.source_legacy_joint is None:
            source_metrics, self.source_legacy_joint = self.evaluate_legacy()
            self.source_legacy_metrics = {
                task: metrics.to_dict() for task, metrics in source_metrics.items()
            }

        if self.phase == "stage9a":
            for epoch in range(self.stage9a_epoch + 1, values.stage9a_epochs + 1):
                train_metrics = self.train_pair_epoch(epoch, phase="stage9a")
                validation = self.evaluate_pairs()
                improved = stage9a_candidate_is_better(
                    contrastive_loss=validation.metrics.contrastive_loss,
                    ranking_accuracy=validation.metrics.ranking_accuracy,
                    mean_similarity_gap=validation.metrics.mean_similarity_gap,
                    best_contrastive_loss=self.best_stage9a_loss,
                    best_ranking_accuracy=self.best_stage9a_ranking,
                    best_mean_similarity_gap=self.best_stage9a_gap,
                    min_delta=values.early_stopping_min_delta,
                )
                if improved:
                    self.best_stage9a_loss = validation.metrics.contrastive_loss
                    self.best_stage9a_ranking = validation.metrics.ranking_accuracy
                    self.best_stage9a_gap = validation.metrics.mean_similarity_gap
                self.stage9a_epoch = epoch
                if self.tracker:
                    self.tracker.log(
                        {
                            **{
                                f"stage9a/pair/train/{name}": value
                                for name, value in train_metrics.items()
                            },
                            "stage9a/pair/val/contrastive_loss": validation.metrics.contrastive_loss,
                            "stage9a/pair/val/ranking_accuracy": validation.metrics.ranking_accuracy,
                            "stage9a/pair/val/mean_similarity_gap": validation.metrics.mean_similarity_gap,
                        },
                        step=self.global_step,
                    )
                if improved:
                    self.save_checkpoint(self.stage9a_best_path, artifact_role="stage9a_best")
                self.save_checkpoint(self.last_path, artifact_role="last")
            if not self.stage9a_best_path.is_file():
                raise RuntimeError("Stage 9A did not produce a best checkpoint")
            best = load_torch_checkpoint(self.stage9a_best_path, mmap=False)
            self.model.load_state_dict(best["model"], strict=True)
            self.phase = "stage9b"
            self.stage9b_round = 0
            self.bad_rounds = 0
            self._build_optimization("stage9b")
            self.model.set_trainable_stage9("pair", "stage9b")

            pair = self.evaluate_pairs()
            legacy, legacy_joint = self.evaluate_legacy()
            threshold, threshold_metrics = self._threshold_from_pairs(pair)
            self.selected_threshold = threshold
            self.best_selection = self._selection(pair, legacy_joint)
            self._record_best_validation(
                pair, legacy, legacy_joint, threshold_metrics
            )
            self._log_round(0, pair, self.best_selection, True)
            self.save_checkpoint(self.stage9b_best_path, artifact_role="stage9b_best")
            self.save_deploy_checkpoint()
            self.save_checkpoint(self.last_path, artifact_role="last")

        for round_id in range(self.stage9b_round + 1, values.max_rounds + 1):
            for task in ("phla", "ptcr", "triplet"):
                self.train_replay_epoch(task, round_id)
            self.train_pair_epoch(round_id, phase="stage9b")
            pair = self.evaluate_pairs()
            legacy, legacy_joint = self.evaluate_legacy()
            selection = self._selection(pair, legacy_joint)
            improved = candidate_is_better(
                selection,
                self.best_selection,
                source_legacy_joint=float(self.source_legacy_joint),
                degradation_tolerance=values.legacy_degradation_tolerance,
                min_delta=values.early_stopping_min_delta,
            )
            if improved:
                threshold, threshold_metrics = self._threshold_from_pairs(pair)
                self.selected_threshold = threshold
                self.best_selection = selection
                self.bad_rounds = 0
                self._record_best_validation(
                    pair, legacy, legacy_joint, threshold_metrics
                )
            else:
                self.bad_rounds += 1
            self.stage9b_round = round_id
            self._log_round(round_id, pair, selection, improved)
            if improved:
                self.save_checkpoint(
                    self.stage9b_best_path, artifact_role="stage9b_best"
                )
                self.save_deploy_checkpoint()
            self.save_checkpoint(self.last_path, artifact_role="last")
            if (
                round_id >= values.min_rounds
                and self.bad_rounds >= values.early_stopping_patience
            ):
                break

        self.phase = "complete"
        self.save_checkpoint(self.last_path, artifact_role="last")
        if self.stage9b_best_path.is_file():
            best = load_torch_checkpoint(self.stage9b_best_path, mmap=False)
            self.model.load_state_dict(best["model"], strict=True)
            self.save_deploy_checkpoint()
        return self.summary()

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(path, mmap=False)
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("checkpoint_type") != "stage9_training"
            or checkpoint.get("artifact_role") != "last"
        ):
            raise ValueError("Stage-9 resume requires a format-1 last training checkpoint")
        saved_config = checkpoint.get("config", {})
        if saved_config.get("model") != self.config.to_dict()["model"]:
            raise ValueError("Stage-9 resume model configuration mismatch")
        saved_runtime = saved_config.get("runtime", {})
        for key in ("fold", "seed"):
            if saved_runtime.get(key) != getattr(self.config.runtime, key):
                raise ValueError(f"Stage-9 resume {key} mismatch")
        if checkpoint.get("source_stage3", {}).get("sha256") != self.source_stage3.get(
            "sha256"
        ):
            raise ValueError("Stage-9 resume source Stage-3 checkpoint mismatch")
        if checkpoint.get("data_audit", {}).get(
            "dataset_fingerprint"
        ) != self.data_audit.get("dataset_fingerprint"):
            raise ValueError("Stage-9 resume pair dataset fingerprint mismatch")
        if checkpoint.get("training_contract", {}).get(
            "sha256"
        ) != self.training_contract["sha256"]:
            raise ValueError("Stage-9 resume training contract mismatch")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.phase = str(checkpoint["phase"])
        self.stage9a_epoch = int(checkpoint["stage9a_epoch"])
        self.stage9b_round = int(checkpoint["stage9b_round"])
        self.best_stage9a_loss = float(checkpoint["best_stage9a_loss"])
        self.best_stage9a_ranking = float(
            checkpoint.get("best_stage9a_ranking", -float("inf"))
        )
        self.best_stage9a_gap = float(
            checkpoint.get("best_stage9a_gap", -float("inf"))
        )
        saved_selection = checkpoint.get("best_selection")
        self.best_selection = (
            Stage9SelectionMetrics(**saved_selection) if saved_selection else None
        )
        source_joint = checkpoint.get("source_legacy_joint")
        self.source_legacy_joint = float(source_joint) if source_joint is not None else None
        self.source_legacy_metrics = dict(checkpoint.get("source_legacy_metrics", {}))
        self.best_validation = dict(checkpoint.get("best_validation", {}))
        self.selected_threshold = float(
            checkpoint.get("selected_pair_threshold", 0.5)
        )
        self.bad_rounds = int(checkpoint["bad_rounds"])
        self.global_step = int(checkpoint["global_step"])
        optimizer_phase = checkpoint.get(
            "optimizer_phase", "stage9a" if self.phase == "stage9a" else "stage9b"
        )
        self._build_optimization(optimizer_phase)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        _restore_rng_state(checkpoint.get("rng_state"))
        saved_loader_state = checkpoint.get("dataloader_rng_state", {})
        loaders = {
            "pair/train": self.pair_train_loader,
            "pair/val": self.pair_val_loader,
            **{
                f"legacy/train/{task}": loader
                for task, loader in self.legacy_train_loaders.items()
            },
            **{
                f"legacy/val/{task}": loader
                for task, loader in self.legacy_val_loaders.items()
            },
        }
        for name, state in saved_loader_state.items():
            loader = loaders.get(name)
            if loader is None:
                continue
            generator = getattr(loader, "stage9_generator", None) or getattr(
                loader, "generator", None
            )
            if generator is not None:
                generator.set_state(state)

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "stage9a_epoch": self.stage9a_epoch,
            "stage9b_round": self.stage9b_round,
            "global_step": self.global_step,
            "source_stage3": self.source_stage3,
            "source_legacy_joint": self.source_legacy_joint,
            "best_selection": (
                self.best_selection.__dict__ if self.best_selection else None
            ),
            "selected_pair_threshold": self.selected_threshold,
            "stage9a_best_checkpoint": str(self.stage9a_best_path),
            "stage9b_best_checkpoint": str(self.stage9b_best_path),
            "last_checkpoint": str(self.last_path),
            "deploy_checkpoint": str(self.deploy_path),
        }
