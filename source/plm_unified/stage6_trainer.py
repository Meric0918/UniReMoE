"""Stage-6 trainer for the PH coverage and task-conflict ablation."""

from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .fgm import FGMStats, ModuleFGM
from .metrics import BinaryMetrics, compute_binary_metrics, prefixed_metrics, task_score
from .stage6_config import Stage6ExperimentConfig
from .stage6_model import Stage6FineTuningModel, Stage6Phase
from .trainer import (
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    build_scheduler,
    load_training_checkpoint,
    resolve_device,
)
from .triplet_evaluation import PredictionResult, evaluate_stage3_model, select_mcc_threshold
from .triplet_model import load_triplet_model_state
from .triplet_trainer import checkpoint_initialization_source


def stage6_scratch_initialization_source() -> dict[str, Any]:
    return {
        "kind": "scratch",
        "stage": "stage6",
        "downstream_modules": "random",
        "base_plms": "pretrained_frozen",
    }


def require_stage6_scratch_source(source: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(source)
    if normalized.get("kind") != "scratch" or normalized.get("stage") != "stage6":
        raise ValueError(
            "Stage 6 may only resume a scratch-initialized Stage-6 checkpoint"
        )
    return normalized


def phla_batches_per_step(
    total_phla_batches: int,
    logical_steps: int,
    coverage: str,
) -> tuple[int, ...]:
    """Return a deterministic allocation of PH batches to logical steps."""

    if total_phla_batches < 1 or logical_steps < 1:
        raise ValueError("PH batches and logical steps must be positive")
    if coverage == "matched":
        return (1,) * logical_steps
    if coverage != "full":
        raise ValueError(f"Unknown PH coverage: {coverage}")
    if total_phla_batches < logical_steps:
        raise ValueError(
            "Full PH coverage requires at least one PH batch per logical step"
        )
    base, extra = divmod(total_phla_batches, logical_steps)
    return (base + 1,) * extra + (base,) * (logical_steps - extra)


def stage6_weighted_loss(
    per_sample_losses: Mapping[str, torch.Tensor],
    config: Stage6ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample-mean each active task once, then apply fixed Stage-5 weights."""

    expected = (
        {"phla", "ptcr", "triplet"}
        if config.stage6.task_mode == "joint"
        else {"phla"}
    )
    if set(per_sample_losses) != expected:
        raise ValueError(f"Stage-6 {config.stage6.task_mode} loss requires {expected}")
    task_losses: dict[str, torch.Tensor] = {}
    for task, losses in per_sample_losses.items():
        if losses.ndim != 1 or losses.numel() == 0:
            raise ValueError(f"Stage-6 {task} loss must be a non-empty vector")
        task_losses[task] = losses.mean()
    values = config.stage6
    total = values.phla_loss_weight * task_losses["phla"]
    if values.task_mode == "joint":
        total = (
            total
            + values.ptcr_loss_weight * task_losses["ptcr"]
            + values.pht_loss_weight * task_losses["triplet"]
        )
    return total, task_losses


def stage6_selection_score(
    metrics: Mapping[str, BinaryMetrics | Mapping[str, float]],
    config: Stage6ExperimentConfig,
) -> float:
    if config.stage6.task_mode == "phla_only":
        return task_score(metrics["phla"])
    values = config.stage6
    return (
        values.triplet_score_weight * task_score(metrics["triplet"])
        + values.phla_score_weight * task_score(metrics["phla"])
        + values.ptcr_score_weight * task_score(metrics["ptcr"])
    )


def _console_log(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


@dataclass
class Stage6EpochResult:
    metrics: dict[str, BinaryMetrics]
    losses: dict[str, float]
    optimizer_steps: int
    logical_steps: int
    task_samples: dict[str, int]
    task_batches: dict[str, int]
    model_gradient_norm: float
    fgm_attacked_parameter_count: float
    fgm_mean_gradient_norm: float
    fgm_max_gradient_norm: float

    def to_log_dict(self, prefix: str) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            f"{prefix}/loss_total": self.losses["total"],
            f"{prefix}/loss_clean_total": self.losses["clean_total"],
            f"{prefix}/loss_adversarial_total": self.losses[
                "adversarial_total"
            ],
            f"{prefix}/optimizer_steps": self.optimizer_steps,
            f"{prefix}/logical_steps": self.logical_steps,
            f"{prefix}/model_gradient_norm": self.model_gradient_norm,
            f"{prefix}/fgm_attacked_parameters": self.fgm_attacked_parameter_count,
            f"{prefix}/fgm_mean_gradient_norm": self.fgm_mean_gradient_norm,
            f"{prefix}/fgm_max_gradient_norm": self.fgm_max_gradient_norm,
        }
        for task, loss in self.losses.items():
            if task in {"phla", "ptcr", "triplet"}:
                values[f"{prefix}/loss_{task}"] = loss
        for task, count in self.task_samples.items():
            values[f"{prefix}/samples_{task}"] = count
        for task, count in self.task_batches.items():
            values[f"{prefix}/batches_{task}"] = count
        for task, metrics in self.metrics.items():
            values.update(prefixed_metrics(f"{prefix}/{task}", metrics))
        return values


def build_stage6_optimizer(
    model: Stage6FineTuningModel,
    config: Stage6ExperimentConfig,
) -> torch.optim.Optimizer:
    values = config.stage6
    binding = model.binding_model
    groups: list[dict[str, Any]] = []

    def add_group(name: str, modules: Sequence[nn.Module], lr: float) -> None:
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr})

    if values.task_mode == "joint":
        add_group(
            "adapters",
            (binding.peptide_adapter, binding.hla_adapter, binding.tcr_adapter),
            values.adapter_lr,
        )
        add_group(
            "pretrained_attention",
            (
                binding.phla_cross_attention,
                binding.ptcr_cross_attention,
                binding.phla_pooling,
                binding.ptcr_pooling,
                binding.phla_classifier,
                binding.ptcr_classifier,
            ),
            values.pretrained_attention_lr,
        )
        add_group(
            "triplet_modules",
            tuple(binding.new_triplet_modules().values()),
            values.triplet_module_lr,
        )
    else:
        add_group(
            "phla_adapters",
            (binding.peptide_adapter, binding.hla_adapter),
            values.adapter_lr,
        )
        add_group(
            "phla_attention",
            (
                binding.phla_cross_attention,
                binding.phla_pooling,
                binding.phla_classifier,
            ),
            values.pretrained_attention_lr,
        )
    if not groups:
        raise RuntimeError("Stage-6 optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=values.weight_decay)
    encoder_ids = {
        id(parameter)
        for encoder in (
            model.peptide_encoder,
            model.hla_encoder,
            model.tcr_encoder,
        )
        for parameter in encoder.parameters()
    }
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if encoder_ids & optimized_ids:
        raise RuntimeError("Stage-6 optimizer must not contain PLM parameters")
    return optimizer


class Stage6Trainer:
    """Run one Stage-6 condition with fixed logical optimizer exposure."""

    def __init__(
        self,
        model: Stage6FineTuningModel,
        config: Stage6ExperimentConfig,
        train_loaders: Mapping[str, Any],
        val_loaders: Mapping[str, Any],
        tracker,
        *,
        initialization_source: Mapping[str, Any],
        data_audit: Mapping[str, Any],
        run_timestamp: Optional[str] = None,
    ):
        expected = (
            {"phla", "ptcr", "triplet"}
            if config.stage6.task_mode == "joint"
            else {"phla"}
        )
        if set(train_loaders) != expected or set(val_loaders) != expected:
            raise ValueError(f"Stage-6 {config.stage6.task_mode} requires {expected}")
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = dict(val_loaders)
        self.tracker = tracker
        self.initialization_source = require_stage6_scratch_source(
            initialization_source
        )
        self.data_audit = dict(data_audit)
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.fgm = ModuleFGM(model)
        self.phase: Stage6Phase | str = "stage6_1a"
        self.stage1a_epoch = 0
        self.stage1b_epoch = 0
        self.best_stage1a_score = -float("inf")
        self.best_selection_score = -float("inf")
        self.best_val_metrics: dict[str, Any] = {}
        self.selected_threshold = 0.5
        self.global_step = 0
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.run_dir / timestamped_filename(
            "stage6_last.pt", self.run_timestamp
        )
        self.stage1a_complete_path = self.run_dir / timestamped_filename(
            "stage6_1a_complete.pt", self.run_timestamp
        )
        self.stage1b_best_path = self.run_dir / timestamped_filename(
            "stage6_1b_best.pt", self.run_timestamp
        )
        self.deploy_path = self.run_dir / timestamped_filename(
            "stage6_best.pt", self.run_timestamp
        )
        self.optimizer_phase: Stage6Phase = "stage6_1a"
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage6_1a")
        scaler_enabled = (
            self.device.type == "cuda" and config.stage6.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        schedule = self._phla_schedule()
        _console_log(
            "[STAGE6 INIT] "
            f"condition={config.stage6.condition} fold={config.training.fold} "
            f"seed={config.training.seed} mode={config.stage6.task_mode} "
            f"coverage={config.stage6.phla_coverage} "
            f"logical_steps={len(schedule):,} phla_batches={sum(schedule):,}"
        )

    def _active_tasks(self) -> tuple[str, ...]:
        if self.config.stage6.task_mode == "joint":
            return ("phla", "ptcr", "triplet")
        return ("phla",)

    def _phla_schedule(self) -> tuple[int, ...]:
        return phla_batches_per_step(
            len(self.train_loaders["phla"]),
            self.config.stage6.logical_steps_per_epoch,
            self.config.stage6.phla_coverage,
        )

    def _planned_optimizer_steps(self, phase: Stage6Phase) -> int:
        epochs = (
            self.config.stage6.stage1a_epochs
            if phase == "stage6_1a"
            else self.config.stage6.stage1b_epochs
        )
        per_epoch = math.ceil(
            self.config.stage6.logical_steps_per_epoch
            / self.config.stage6.gradient_accumulation_steps
        )
        return max(1, per_epoch * epochs)

    def _build_optimization(self, phase: Stage6Phase) -> None:
        self.optimizer_phase = phase
        self.model.set_trainable_stage6(phase, self.config.stage6.task_mode)
        self.optimizer = build_stage6_optimizer(self.model, self.config)
        total_steps = self._planned_optimizer_steps(phase)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=self.config.stage6.scheduler_warmup_ratio,
        )
        groups = ", ".join(
            f"{group.get('name', index)}={group['lr']:.2e}"
            for index, group in enumerate(self.optimizer.param_groups)
        )
        _console_log(
            f"[OPTIMIZER][{phase.upper()}] total_steps={total_steps:,} "
            f"lr_groups=({groups})"
        )

    def _autocast(self):
        precision = self.config.stage6.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _next_or_restart(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _labels(self, batch) -> torch.Tensor:
        return batch["labels"].to(
            self.device,
            dtype=torch.long,
            non_blocking=self.device.type == "cuda",
        )

    def _encode_batches(self, phla_batches, ptcr_batch=None, triplet_batch=None):
        encoded: dict[str, Any] = {
            "phla": [
                (
                    self.model.peptide_encoder(batch["peptide_sequences"]),
                    self.model.hla_encoder(batch["receptor_sequences"]),
                )
                for batch in phla_batches
            ]
        }
        if self.config.stage6.task_mode == "joint":
            encoded["ptcr"] = (
                self.model.peptide_encoder(ptcr_batch["peptide_sequences"]),
                self.model.tcr_encoder(ptcr_batch["receptor_sequences"]),
            )
            encoded["triplet"] = (
                self.model.peptide_encoder(triplet_batch["peptide_sequences"]),
                self.model.hla_encoder(triplet_batch["hla_sequences"]),
                self.model.tcr_encoder(triplet_batch["tcr_sequences"]),
            )
        return encoded

    def _forward_encoded(self, encoded) -> dict[str, torch.Tensor]:
        binding = self.model.binding_model
        phla_logits = []
        for peptide, hla in encoded["phla"]:
            phla_logits.append(
                binding(
                    task="phla",
                    peptide_hidden=peptide.hidden,
                    peptide_mask=peptide.mask,
                    receptor_hidden=hla.hidden,
                    receptor_mask=hla.mask,
                ).logits
            )
        outputs = {"phla": torch.cat(phla_logits, dim=0)}
        if self.config.stage6.task_mode == "joint":
            peptide, tcr = encoded["ptcr"]
            outputs["ptcr"] = binding(
                task="ptcr",
                peptide_hidden=peptide.hidden,
                peptide_mask=peptide.mask,
                receptor_hidden=tcr.hidden,
                receptor_mask=tcr.mask,
            ).logits
            peptide, hla, tcr = encoded["triplet"]
            outputs["triplet"] = binding.forward_triplet(
                peptide.hidden,
                peptide.mask,
                hla.hidden,
                hla.mask,
                tcr.hidden,
                tcr.mask,
            ).logits
        return outputs

    def _loss(self, outputs, labels):
        per_sample = {
            task: self.criterion(outputs[task], labels[task])
            for task in self._active_tasks()
        }
        total, task_losses = stage6_weighted_loss(per_sample, self.config)
        return total, task_losses, per_sample

    def _capture_fgm_gradients(self):
        snapshots = []
        seen: set[int] = set()
        for module in self.model.fgm_modules(self.config.stage6.task_mode).values():
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                snapshots.append(
                    (
                        parameter,
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone(),
                    )
                )
        return snapshots

    @staticmethod
    def _use_current_clean_fgm_gradients(snapshots):
        accumulated = []
        for parameter, previous in snapshots:
            current = parameter.grad
            accumulated.append((parameter, current))
            if current is None:
                continue
            parameter.grad = (
                current.detach().clone()
                if previous is None
                else current.detach() - previous
            )
        return accumulated

    @staticmethod
    def _restore_accumulated_fgm_gradients(accumulated) -> None:
        for parameter, gradient in accumulated:
            parameter.grad = gradient

    def train_epoch(
        self,
        *,
        phase: Stage6Phase,
        epoch_id: int,
    ) -> Stage6EpochResult:
        self.model.set_trainable_stage6(phase, self.config.stage6.task_mode)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        values = self.config.stage6
        accumulation = values.gradient_accumulation_steps
        schedule = self._phla_schedule()
        total_steps = len(schedule)
        iterators = {task: iter(loader) for task, loader in self.train_loaders.items()}
        tasks = self._active_tasks()
        probabilities: dict[str, list[np.ndarray]] = {task: [] for task in tasks}
        labels_history: dict[str, list[np.ndarray]] = {task: [] for task in tasks}
        logical_loss_sums = {task: 0.0 for task in tasks}
        sample_loss_sums = {task: 0.0 for task in tasks}
        sample_counts = {task: 0 for task in tasks}
        batch_counts = {task: 0 for task in tasks}
        total_loss_sum = 0.0
        clean_loss_sum = 0.0
        adversarial_loss_sum = 0.0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        attacked_counts: list[int] = []
        fgm_mean_norms: list[float] = []
        fgm_max_norms: list[float] = []
        started_at = time.perf_counter()
        _console_log(
            f"[TRAIN START][{phase.upper()}] epoch={epoch_id} "
            f"logical_steps={total_steps:,} phla_batches={sum(schedule):,}"
        )

        for step_index, phla_batch_count in enumerate(schedule, start=1):
            phla_batches = []
            for _ in range(phla_batch_count):
                if values.phla_coverage == "full":
                    try:
                        phla_batch = next(iterators["phla"])
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Full PH schedule exhausted its loader early"
                        ) from exc
                else:
                    phla_batch, iterators["phla"] = self._next_or_restart(
                        self.train_loaders["phla"], iterators["phla"]
                    )
                phla_batches.append(phla_batch)
            batch_counts["phla"] += phla_batch_count

            ptcr_batch = triplet_batch = None
            if values.task_mode == "joint":
                ptcr_batch, iterators["ptcr"] = self._next_or_restart(
                    self.train_loaders["ptcr"], iterators["ptcr"]
                )
                triplet_batch, iterators["triplet"] = self._next_or_restart(
                    self.train_loaders["triplet"], iterators["triplet"]
                )
                batch_counts["ptcr"] += 1
                batch_counts["triplet"] += 1

            window_start = ((step_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_steps - window_start + 1)
            if step_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            previous_fgm = (
                self._capture_fgm_gradients() if phase == "stage6_1b" else None
            )
            step_labels = {
                "phla": torch.cat([self._labels(batch) for batch in phla_batches])
            }
            if values.task_mode == "joint":
                step_labels["ptcr"] = self._labels(ptcr_batch)
                step_labels["triplet"] = self._labels(triplet_batch)
            encoded = self._encode_batches(phla_batches, ptcr_batch, triplet_batch)
            with self._autocast():
                clean_outputs = self._forward_encoded(encoded)
                clean_loss, task_losses, per_sample_losses = self._loss(
                    clean_outputs, step_labels
                )
            self.scaler.scale(clean_loss / window_size).backward()

            adversarial_value = 0.0
            fgm_stats: Optional[FGMStats] = None
            if phase == "stage6_1b":
                gradient_scale = (
                    self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                )
                accumulated = self._use_current_clean_fgm_gradients(previous_fgm)
                try:
                    fgm_stats = self.fgm.attack(
                        self.model.fgm_modules(values.task_mode),
                        epsilon=values.fgm_epsilon,
                    )
                finally:
                    self._restore_accumulated_fgm_gradients(accumulated)
                try:
                    with self._autocast():
                        adversarial_outputs = self._forward_encoded(encoded)
                        adversarial_loss, _, _ = self._loss(
                            adversarial_outputs, step_labels
                        )
                    self.scaler.scale(adversarial_loss / window_size).backward()
                    adversarial_value = float(adversarial_loss.detach().cpu())
                finally:
                    self.fgm.restore()
                attacked_counts.append(fgm_stats.attacked_parameter_count)
                fgm_mean_norms.append(fgm_stats.mean_gradient_norm / gradient_scale)
                fgm_max_norms.append(fgm_stats.max_gradient_norm / gradient_scale)

            for task in tasks:
                logits = clean_outputs[task]
                task_samples = int(step_labels[task].shape[0])
                sample_counts[task] += task_samples
                probabilities[task].append(
                    torch.softmax(logits.detach(), dim=1)[:, 1]
                    .float()
                    .cpu()
                    .numpy()
                )
                labels_history[task].append(
                    step_labels[task].detach().cpu().numpy()
                )
                logical_loss_sums[task] += float(task_losses[task].detach().cpu())
                sample_loss_sums[task] += float(
                    per_sample_losses[task].detach().sum().cpu()
                )
            clean_value = float(clean_loss.detach().cpu())
            clean_loss_sum += clean_value
            adversarial_loss_sum += adversarial_value
            total_loss_sum += clean_value + adversarial_value

            if step_index - window_start + 1 == window_size:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, values.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                if (
                    optimizer_steps % values.log_every_optimizer_steps == 0
                    or step_index == total_steps
                ):
                    log_values: dict[str, Any] = {
                        "phase": phase,
                        "epoch": epoch_id,
                        "train/logical_step": step_index,
                        "train/phla_batches_consumed": batch_counts["phla"],
                        "train/loss_total_running": total_loss_sum / step_index,
                        "train/loss_clean_total_running": clean_loss_sum / step_index,
                        "train/loss_adversarial_total_running": (
                            adversarial_loss_sum / step_index
                        ),
                        "train/gradient_norm": gradient_norms[-1],
                        "train/fgm_epsilon": (
                            values.fgm_epsilon if phase == "stage6_1b" else 0.0
                        ),
                        "train/fgm_attacked_parameters": (
                            fgm_stats.attacked_parameter_count if fgm_stats else 0
                        ),
                    }
                    for task in tasks:
                        log_values[f"train/loss_{task}_running"] = (
                            logical_loss_sums[task] / step_index
                        )
                    for group in self.optimizer.param_groups:
                        log_values[f"train/lr/{group.get('name', 'unknown')}"] = (
                            group["lr"]
                        )
                    self.tracker.log(log_values, step=self.global_step)
                    _console_log(
                        f"[TRAIN PROGRESS][{phase.upper()}] epoch={epoch_id} "
                        f"step={step_index:,}/{total_steps:,} "
                        f"ph_batches={batch_counts['phla']:,} "
                        f"loss={total_loss_sum / step_index:.4f} "
                        f"elapsed={_format_duration(time.perf_counter() - started_at)}"
                    )

        if values.phla_coverage == "full":
            try:
                next(iterators["phla"])
            except StopIteration:
                pass
            else:
                raise RuntimeError("Full PH schedule did not consume the entire loader")
        metrics = {
            task: compute_binary_metrics(
                np.concatenate(labels_history[task]),
                np.concatenate(probabilities[task]),
                loss=sample_loss_sums[task] / sample_counts[task],
                threshold=self.config.training.threshold,
            )
            for task in tasks
        }
        losses = {
            "total": total_loss_sum / total_steps,
            "clean_total": clean_loss_sum / total_steps,
            "adversarial_total": adversarial_loss_sum / total_steps,
            **{
                task: logical_loss_sums[task] / total_steps
                for task in tasks
            },
        }
        return Stage6EpochResult(
            metrics=metrics,
            losses=losses,
            optimizer_steps=optimizer_steps,
            logical_steps=total_steps,
            task_samples=sample_counts,
            task_batches=batch_counts,
            model_gradient_norm=float(np.mean(gradient_norms)),
            fgm_attacked_parameter_count=(
                float(np.mean(attacked_counts)) if attacked_counts else 0.0
            ),
            fgm_mean_gradient_norm=(
                float(np.mean(fgm_mean_norms)) if fgm_mean_norms else 0.0
            ),
            fgm_max_gradient_norm=max(fgm_max_norms, default=0.0),
        )

    def evaluate_all(
        self,
    ) -> tuple[dict[str, PredictionResult], float, float, BinaryMetrics, dict[str, Any]]:
        results = {
            task: evaluate_stage3_model(
                self.model,
                self.val_loaders[task],
                task,
                device=self.device,
                mixed_precision=self.config.stage6.mixed_precision,
                threshold=self.config.training.threshold,
            )
            for task in self._active_tasks()
        }
        phla_result = results["phla"]
        threshold, threshold_metrics = select_mcc_threshold(
            phla_result.labels, phla_result.probabilities
        )
        threshold_metrics.loss = phla_result.metrics.loss
        selection = stage6_selection_score(
            {task: result.metrics for task, result in results.items()}, self.config
        )
        serialised = {
            task: result.metrics.to_dict() for task, result in results.items()
        }
        serialised["phla_selected_threshold"] = threshold
        serialised["phla_selected_threshold_metrics"] = threshold_metrics.to_dict()
        return results, selection, threshold, threshold_metrics, serialised

    def _record_epoch(
        self,
        *,
        phase: Stage6Phase,
        epoch_id: int,
        train_result: Stage6EpochResult,
        val_results: Mapping[str, PredictionResult],
        selection: float,
        threshold: float,
        threshold_metrics: BinaryMetrics,
        improved: bool,
    ) -> None:
        values: dict[str, Any] = {
            "phase": phase,
            f"{phase}/epoch": epoch_id,
            f"{phase}/selection_score": selection,
            f"{phase}/improved": int(improved),
            f"{phase}/phla_selected_threshold": threshold,
            **train_result.to_log_dict(f"{phase}/train"),
        }
        for task, result in val_results.items():
            values.update(prefixed_metrics(f"{phase}/{task}/val", result.metrics))
        values.update(
            prefixed_metrics(f"{phase}/phla/val_selected", threshold_metrics)
        )
        self.tracker.log(values, step=self.global_step)
        _console_log(
            f"[{phase.upper()}][epoch={epoch_id}][step={self.global_step}] "
            f"selection={selection:.4f} improved={'yes' if improved else 'no'}"
        )

    def fit(self) -> dict[str, Any]:
        if self.phase == "complete":
            if not self.deploy_path.is_file():
                best = load_stage6_checkpoint(self.stage1b_best_path, map_location="cpu")
                load_triplet_model_state(self.model, best["model"])
                del best
                gc.collect()
                self.save_deploy_checkpoint()
            return self.summary()
        if self.phase == "stage6_1a":
            for epoch_id in range(
                self.stage1a_epoch + 1, self.config.stage6.stage1a_epochs + 1
            ):
                train_result = self.train_epoch(phase="stage6_1a", epoch_id=epoch_id)
                val_results, selection, threshold, threshold_metrics, _ = (
                    self.evaluate_all()
                )
                improved = (
                    selection
                    > self.best_stage1a_score + self.config.stage6.selection_min_delta
                )
                if improved:
                    self.best_stage1a_score = selection
                self.stage1a_epoch = epoch_id
                self._record_epoch(
                    phase="stage6_1a",
                    epoch_id=epoch_id,
                    train_result=train_result,
                    val_results=val_results,
                    selection=selection,
                    threshold=threshold,
                    threshold_metrics=threshold_metrics,
                    improved=improved,
                )
                self.save_checkpoint(self.last_path)
            self.save_checkpoint(self.stage1a_complete_path)
            self.phase = "stage6_1b"
            self.stage1b_epoch = 0
            self._build_optimization("stage6_1b")
            self.save_checkpoint(self.last_path)
        for epoch_id in range(
            self.stage1b_epoch + 1, self.config.stage6.stage1b_epochs + 1
        ):
            train_result = self.train_epoch(phase="stage6_1b", epoch_id=epoch_id)
            val_results, selection, threshold, threshold_metrics, serialised = (
                self.evaluate_all()
            )
            improved = (
                selection
                > self.best_selection_score + self.config.stage6.selection_min_delta
            )
            if improved:
                self.best_selection_score = selection
                self.best_val_metrics = serialised
                self.selected_threshold = threshold
            self.stage1b_epoch = epoch_id
            self._record_epoch(
                phase="stage6_1b",
                epoch_id=epoch_id,
                train_result=train_result,
                val_results=val_results,
                selection=selection,
                threshold=threshold,
                threshold_metrics=threshold_metrics,
                improved=improved,
            )
            if improved:
                self.save_checkpoint(self.stage1b_best_path)
                self.save_deploy_checkpoint()
            self.save_checkpoint(self.last_path)
        self.phase = "complete"
        self.save_checkpoint(self.last_path)
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "checkpoint_type": "stage6_training",
            "condition": self.config.stage6.condition,
            "tasks": list(self._active_tasks()),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_phase": self.optimizer_phase,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "phase": self.phase,
            "stage1a_epoch": self.stage1a_epoch,
            "stage1b_epoch": self.stage1b_epoch,
            "best_stage1a_score": self.best_stage1a_score,
            "best_selection_score": self.best_selection_score,
            "best_val_metrics": self.best_val_metrics,
            "selected_threshold": self.selected_threshold,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "initialization_source": self.initialization_source,
            "data_audit": self.data_audit,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": self.tracker.run_id,
            "dataloader_rng_state": {
                f"{split}/{task}": loader.generator.get_state()
                for split, loaders in (
                    ("train", self.train_loaders),
                    ("val", self.val_loaders),
                )
                for task, loader in loaders.items()
                if getattr(loader, "generator", None) is not None
            },
            "rng_state": _capture_rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        _console_log(
            f"[CHECKPOINT SAVE][STAGE6] condition={self.config.stage6.condition} "
            f"phase={self.phase} epoch1a={self.stage1a_epoch} "
            f"epoch1b={self.stage1b_epoch} path={path}"
        )
        _atomic_torch_save(self.checkpoint_state(), Path(path))

    def save_deploy_checkpoint(self) -> None:
        state = {
            "format_version": 1,
            "checkpoint_type": "stage6_model",
            "condition": self.config.stage6.condition,
            "tasks": list(self._active_tasks()),
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "initialization_source": self.initialization_source,
            "data_audit": self.data_audit,
            "validation_metrics": self.best_val_metrics,
            "selection_score": self.best_selection_score,
            "selected_threshold": self.selected_threshold,
            "run_timestamp": self.run_timestamp,
            "global_step": self.global_step,
        }
        _atomic_torch_save(state, self.deploy_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_stage6_checkpoint(path, map_location="cpu")
        source = require_stage6_scratch_source(
            checkpoint_initialization_source(checkpoint)
        )
        saved = checkpoint.get("config", {})
        current = self.config.to_dict()
        for key in ("fold", "seed"):
            if saved.get("training", {}).get(key) != current["training"][key]:
                raise ValueError(f"Stage-6 checkpoint {key} mismatch")
        if saved.get("model") != current["model"]:
            raise ValueError("Stage-6 checkpoint model configuration mismatch")
        saved_stage6 = dict(saved.get("stage6", {}))
        current_stage6 = dict(current["stage6"])
        for runtime_key in ("eval_batch_size", "log_every_optimizer_steps"):
            saved_stage6.pop(runtime_key, None)
            current_stage6.pop(runtime_key, None)
        if saved_stage6 != current_stage6:
            raise ValueError("Stage-6 checkpoint configuration mismatch")
        saved_metadata = checkpoint.get("encoder_metadata", {})
        current_metadata = self.model.encoder_metadata()
        for entity in ("peptide", "hla", "tcr"):
            fingerprint = saved_metadata.get(entity, {}).get(
                "checkpoint_fingerprint"
            )
            if fingerprint and fingerprint != current_metadata[entity][
                "checkpoint_fingerprint"
            ]:
                raise ValueError(f"Stage-6 PLM fingerprint mismatch for {entity}")
        load_triplet_model_state(self.model, checkpoint["model"])
        self.phase = checkpoint["phase"]
        self.stage1a_epoch = int(checkpoint["stage1a_epoch"])
        self.stage1b_epoch = int(checkpoint["stage1b_epoch"])
        self.best_stage1a_score = float(checkpoint["best_stage1a_score"])
        self.best_selection_score = float(checkpoint["best_selection_score"])
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        self.selected_threshold = float(checkpoint.get("selected_threshold", 0.5))
        self.global_step = int(checkpoint["global_step"])
        self.initialization_source = source
        self.data_audit = dict(checkpoint.get("data_audit", {}))
        optimizer_phase = checkpoint.get(
            "optimizer_phase",
            "stage6_1a" if self.phase == "stage6_1a" else "stage6_1b",
        )
        self._build_optimization(optimizer_phase)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        for name, generator_state in checkpoint.get(
            "dataloader_rng_state", {}
        ).items():
            split, task = name.split("/", 1)
            loaders = self.train_loaders if split == "train" else self.val_loaders
            generator = getattr(loaders.get(task), "generator", None)
            if generator is not None:
                generator.set_state(generator_state)
        _restore_rng_state(checkpoint.get("rng_state"))

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "condition": self.config.stage6.condition,
            "task_mode": self.config.stage6.task_mode,
            "phla_coverage": self.config.stage6.phla_coverage,
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "stage1a_epoch": self.stage1a_epoch,
            "stage1b_epoch": self.stage1b_epoch,
            "best_stage1a_score": self.best_stage1a_score,
            "best_selection_score": self.best_selection_score,
            "best_val_metrics": self.best_val_metrics,
            "selected_threshold": self.selected_threshold,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "last_checkpoint": str(self.last_path),
            "stage1a_complete_checkpoint": str(self.stage1a_complete_path),
            "stage1b_best_checkpoint": str(self.stage1b_best_path),
            "deploy_checkpoint": str(self.deploy_path),
            "initialization_source": self.initialization_source,
        }


def load_stage6_checkpoint(path: str | Path, *, map_location) -> dict[str, Any]:
    checkpoint = load_training_checkpoint(path, map_location=map_location)
    if (
        checkpoint.get("checkpoint_type") != "stage6_training"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Unsupported Stage-6 training checkpoint format")
    return checkpoint
