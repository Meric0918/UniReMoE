"""Scratch Stage-5 training with frozen PLMs and balanced task losses."""

from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .config import ExperimentConfig
from .fgm import FGMStats, ModuleFGM
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    prefixed_metrics,
    task_score,
)
from .stage5_model import Stage5FineTuningModel, Stage5Phase
from .tracking import SwanLabTracker
from .trainer import (
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    build_scheduler,
    load_training_checkpoint,
    resolve_device,
)
from .triplet_evaluation import (
    PredictionResult,
    evaluate_stage3_model,
    select_mcc_threshold,
)
from .triplet_model import load_triplet_model_state
from .triplet_trainer import checkpoint_initialization_source

STAGE5_TASKS = ("phla", "ptcr", "triplet")


def stage5_scratch_initialization_source() -> dict[str, Any]:
    """Describe Stage-5's random downstream initialization policy."""

    return {
        "kind": "scratch",
        "downstream_modules": "random",
        "base_plms": "pretrained_frozen",
    }


def require_stage5_scratch_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject Stage-5 lineages inherited from Stage 1 or Stage 2."""

    normalized = dict(source)
    if normalized.get("kind") != "scratch":
        raise ValueError(
            "Stage 5 may only start from random downstream parameters or "
            "resume a scratch-initialized Stage-5 checkpoint; Stage-1/2 "
            "model inheritance is disabled"
        )
    return normalized


def independently_normalized_stage5_loss(
    per_sample_losses: Mapping[str, torch.Tensor],
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean-reduce each task independently before applying task weights."""

    if set(per_sample_losses) != set(STAGE5_TASKS):
        raise ValueError("Stage-5 loss requires phla, ptcr, and triplet tasks")
    task_losses: dict[str, torch.Tensor] = {}
    for task in STAGE5_TASKS:
        values = per_sample_losses[task]
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(
                f"Stage-5 {task} loss must contain one value per sample"
            )
        task_losses[task] = values.mean()
    weights = config.stage5
    total_loss = (
        weights.phla_loss_weight * task_losses["phla"]
        + weights.ptcr_loss_weight * task_losses["ptcr"]
        + weights.pht_loss_weight * task_losses["triplet"]
    )
    return total_loss, task_losses


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
class Stage5EpochResult:
    metrics: dict[str, BinaryMetrics]
    losses: dict[str, float]
    optimizer_steps: int
    joint_steps: int
    model_gradient_norm: float
    fgm_attacked_parameter_count: float
    fgm_mean_gradient_norm: float
    fgm_max_gradient_norm: float

    def to_log_dict(self, prefix: str) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            f"{prefix}/loss_total": self.losses["total"],
            f"{prefix}/loss_clean_total": self.losses["clean_total"],
            f"{prefix}/loss_adversarial_total": self.losses["adversarial_total"],
            f"{prefix}/loss_pht": self.losses["pht"],
            f"{prefix}/loss_ph": self.losses["phla"],
            f"{prefix}/loss_pt": self.losses["ptcr"],
            f"{prefix}/optimizer_steps": self.optimizer_steps,
            f"{prefix}/joint_steps": self.joint_steps,
            f"{prefix}/model_gradient_norm": self.model_gradient_norm,
            f"{prefix}/fgm_attacked_parameters": (
                self.fgm_attacked_parameter_count
            ),
            f"{prefix}/fgm_mean_gradient_norm": self.fgm_mean_gradient_norm,
            f"{prefix}/fgm_max_gradient_norm": self.fgm_max_gradient_norm,
        }
        for task, metrics in self.metrics.items():
            values.update(prefixed_metrics(f"{prefix}/{task}", metrics))
        return values


def stage5_joint_score(
    metrics: Mapping[str, BinaryMetrics | Mapping[str, float]],
    config: ExperimentConfig,
) -> float:
    weights = {
        "triplet": config.stage5.triplet_score_weight,
        "phla": config.stage5.phla_score_weight,
        "ptcr": config.stage5.ptcr_score_weight,
    }
    return sum(weights[task] * task_score(metrics[task]) for task in weights)


def build_stage5_optimizer(
    model: Stage5FineTuningModel,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    values = config.stage5
    binding = model.binding_model
    groups: list[dict[str, Any]] = []

    def add_group(name: str, modules, lr: float) -> None:
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr})

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
    optimizer = torch.optim.AdamW(groups, weight_decay=values.weight_decay)
    encoder_parameter_ids = {
        id(parameter)
        for encoder in (
            model.peptide_encoder,
            model.hla_encoder,
            model.tcr_encoder,
        )
        for parameter in encoder.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if encoder_parameter_ids & optimizer_parameter_ids:
        raise RuntimeError("Stage-5 optimizer must not contain PLM parameters")
    return optimizer


class Stage5Trainer:
    """Train all three downstream branches while keeping every PLM frozen."""

    def __init__(
        self,
        model: Stage5FineTuningModel,
        config: ExperimentConfig,
        train_loaders: Mapping[str, Any],
        val_loaders: Mapping[str, Any],
        tracker: SwanLabTracker,
        *,
        initialization_source: Mapping[str, Any],
        data_audit: Mapping[str, Any],
        run_timestamp: Optional[str] = None,
    ):
        required_tasks = set(STAGE5_TASKS)
        if (
            set(train_loaders) != required_tasks
            or set(val_loaders) != required_tasks
        ):
            raise ValueError("Stage 5 requires phla, ptcr, and triplet loaders")
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = dict(val_loaders)
        self.tracker = tracker
        self.initialization_source = require_stage5_scratch_source(
            initialization_source
        )
        self.data_audit = dict(data_audit)
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.fgm = ModuleFGM(model)
        self.phase: Stage5Phase | str = "stage5_1a"
        self.stage1a_epoch = 0
        self.stage1b_epoch = 0
        self.best_stage1a_joint_score = -float("inf")
        self.best_joint_score = -float("inf")
        self.best_val_metrics: dict[str, Any] = {}
        self.selected_threshold = 0.5
        self.global_step = 0
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.run_dir / timestamped_filename(
            "stage5_last.pt", self.run_timestamp
        )
        self.stage1a_complete_path = self.run_dir / timestamped_filename(
            "stage5_1a_complete.pt", self.run_timestamp
        )
        self.stage1b_best_path = self.run_dir / timestamped_filename(
            "stage5_1b_best_joint.pt", self.run_timestamp
        )
        self.deploy_path = self.run_dir / timestamped_filename(
            "stage5_best.pt", self.run_timestamp
        )
        self.optimizer_phase: Stage5Phase = "stage5_1a"
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage5_1a")
        scaler_enabled = (
            self.device.type == "cuda"
            and config.stage5.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        _console_log(
            "[STAGE5 INIT] "
            f"fold={config.training.fold} seed={config.training.seed} "
            f"device={self.device} precision={config.stage5.mixed_precision} "
            f"micro_batch={config.stage5.micro_batch_size} "
            f"accumulation={config.stage5.gradient_accumulation_steps} "
            f"fgm_1b=three_adapters epsilon={config.stage5.fgm_epsilon:g} "
            "plms=frozen "
            f"initialization={self.initialization_source.get('kind')} "
            f"downstream={self.initialization_source.get('downstream_modules')}"
        )

    def _planned_optimizer_steps(self, phase: Stage5Phase) -> int:
        epochs = (
            self.config.stage5.stage1a_epochs
            if phase == "stage5_1a"
            else self.config.stage5.stage1b_epochs
        )
        per_epoch = math.ceil(
            len(self.train_loaders["triplet"])
            / self.config.stage5.gradient_accumulation_steps
        )
        return max(1, per_epoch * epochs)

    def _build_optimization(self, phase: Stage5Phase) -> None:
        self.optimizer_phase = phase
        self.model.set_trainable_stage5(phase)
        self.optimizer = build_stage5_optimizer(self.model, self.config)
        total_steps = self._planned_optimizer_steps(phase)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=self.config.stage5.scheduler_warmup_ratio,
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
        precision = self.config.stage5.mixed_precision
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

    def _encode_joint_batches(self, phla_batch, ptcr_batch, triplet_batch):
        """Encode frozen PLM inputs once for clean and adversarial passes."""

        return {
            "phla": (
                self.model.peptide_encoder(phla_batch["peptide_sequences"]),
                self.model.hla_encoder(phla_batch["receptor_sequences"]),
            ),
            "ptcr": (
                self.model.peptide_encoder(ptcr_batch["peptide_sequences"]),
                self.model.tcr_encoder(ptcr_batch["receptor_sequences"]),
            ),
            "triplet": (
                self.model.peptide_encoder(triplet_batch["peptide_sequences"]),
                self.model.hla_encoder(triplet_batch["hla_sequences"]),
                self.model.tcr_encoder(triplet_batch["tcr_sequences"]),
            ),
        }

    def _forward_encoded_joint(self, encoded):
        binding = self.model.binding_model
        phla_peptide, phla_hla = encoded["phla"]
        ptcr_peptide, ptcr_tcr = encoded["ptcr"]
        triplet_peptide, triplet_hla, triplet_tcr = encoded["triplet"]
        return {
            "phla": binding(
                task="phla",
                peptide_hidden=phla_peptide.hidden,
                peptide_mask=phla_peptide.mask,
                receptor_hidden=phla_hla.hidden,
                receptor_mask=phla_hla.mask,
            ),
            "ptcr": binding(
                task="ptcr",
                peptide_hidden=ptcr_peptide.hidden,
                peptide_mask=ptcr_peptide.mask,
                receptor_hidden=ptcr_tcr.hidden,
                receptor_mask=ptcr_tcr.mask,
            ),
            "triplet": binding.forward_triplet(
                triplet_peptide.hidden,
                triplet_peptide.mask,
                triplet_hla.hidden,
                triplet_hla.mask,
                triplet_tcr.hidden,
                triplet_tcr.mask,
            ),
        }

    def _joint_loss(self, outputs, batch_labels):
        per_sample_losses = {
            task: self.criterion(outputs[task].logits, batch_labels[task])
            for task in STAGE5_TASKS
        }
        total_loss, task_losses = independently_normalized_stage5_loss(
            per_sample_losses,
            self.config,
        )
        return total_loss, task_losses, per_sample_losses

    def _capture_fgm_gradients(self):
        """Snapshot accumulated adapter gradients before the clean backward."""

        snapshots = []
        seen_parameters: set[int] = set()
        for module in self.model.fgm_modules().values():
            for parameter in module.parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
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
        """Expose only this micro-batch's clean gradient to ModuleFGM."""

        accumulated_gradients = []
        for parameter, previous_gradient in snapshots:
            accumulated_gradient = parameter.grad
            accumulated_gradients.append((parameter, accumulated_gradient))
            if accumulated_gradient is None:
                continue
            if previous_gradient is None:
                parameter.grad = accumulated_gradient.detach().clone()
            else:
                parameter.grad = accumulated_gradient.detach() - previous_gradient
        return accumulated_gradients

    @staticmethod
    def _restore_accumulated_fgm_gradients(accumulated_gradients) -> None:
        for parameter, accumulated_gradient in accumulated_gradients:
            parameter.grad = accumulated_gradient

    def train_joint_epoch(
        self,
        *,
        phase: Stage5Phase,
        epoch_id: int,
    ) -> Stage5EpochResult:
        self.model.set_trainable_stage5(phase)
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        accumulation = self.config.stage5.gradient_accumulation_steps
        total_batches = len(self.train_loaders["triplet"])
        binary_iterators = {
            task: iter(self.train_loaders[task]) for task in ("phla", "ptcr")
        }
        probabilities: dict[str, list[np.ndarray]] = {
            task: [] for task in ("phla", "ptcr", "triplet")
        }
        labels: dict[str, list[np.ndarray]] = {
            task: [] for task in ("phla", "ptcr", "triplet")
        }
        batch_loss_sums = {task: 0.0 for task in STAGE5_TASKS}
        sample_loss_sums = {task: 0.0 for task in STAGE5_TASKS}
        sample_counts = {"phla": 0, "ptcr": 0, "triplet": 0}
        total_loss_sum = 0.0
        clean_total_loss_sum = 0.0
        adversarial_total_loss_sum = 0.0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        attacked_counts: list[int] = []
        fgm_mean_norms: list[float] = []
        fgm_max_norms: list[float] = []
        started_at = time.perf_counter()
        values = self.config.stage5
        _console_log(
            f"[TRAIN START][{phase.upper()}] epoch={epoch_id} "
            f"batches={total_batches:,}"
        )

        for batch_index, triplet_batch in enumerate(
            self.train_loaders["triplet"], start=1
        ):
            phla_batch, binary_iterators["phla"] = self._next_or_restart(
                self.train_loaders["phla"], binary_iterators["phla"]
            )
            ptcr_batch, binary_iterators["ptcr"] = self._next_or_restart(
                self.train_loaders["ptcr"], binary_iterators["ptcr"]
            )
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(
                accumulation,
                total_batches - window_start + 1,
            )
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            previous_fgm_gradients = (
                self._capture_fgm_gradients()
                if phase == "stage5_1b"
                else None
            )
            batch_labels = {
                "phla": self._labels(phla_batch),
                "ptcr": self._labels(ptcr_batch),
                "triplet": self._labels(triplet_batch),
            }
            encoded = self._encode_joint_batches(
                phla_batch,
                ptcr_batch,
                triplet_batch,
            )
            with self._autocast():
                clean_outputs = self._forward_encoded_joint(encoded)
                clean_loss, task_losses, per_sample_losses = self._joint_loss(
                    clean_outputs,
                    batch_labels,
                )
            self.scaler.scale(clean_loss / window_size).backward()

            adversarial_loss_value = 0.0
            fgm_stats: Optional[FGMStats] = None
            if phase == "stage5_1b":
                gradient_scale = (
                    self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                )
                accumulated_fgm_gradients = (
                    self._use_current_clean_fgm_gradients(
                        previous_fgm_gradients
                    )
                )
                try:
                    fgm_stats = self.fgm.attack(
                        self.model.fgm_modules(),
                        epsilon=values.fgm_epsilon,
                    )
                finally:
                    self._restore_accumulated_fgm_gradients(
                        accumulated_fgm_gradients
                    )
                try:
                    with self._autocast():
                        adversarial_outputs = self._forward_encoded_joint(encoded)
                        adversarial_loss, _, _ = self._joint_loss(
                            adversarial_outputs,
                            batch_labels,
                        )
                    self.scaler.scale(
                        adversarial_loss / window_size
                    ).backward()
                    adversarial_loss_value = float(
                        adversarial_loss.detach().cpu()
                    )
                finally:
                    self.fgm.restore()
                attacked_counts.append(fgm_stats.attacked_parameter_count)
                fgm_mean_norms.append(
                    fgm_stats.mean_gradient_norm / gradient_scale
                )
                fgm_max_norms.append(
                    fgm_stats.max_gradient_norm / gradient_scale
                )

            outputs = {
                task: output.logits for task, output in clean_outputs.items()
            }
            for task, logits in outputs.items():
                task_samples = int(batch_labels[task].shape[0])
                sample_counts[task] += task_samples
                probabilities[task].append(
                    torch.softmax(logits.detach(), dim=1)[:, 1]
                    .float()
                    .cpu()
                    .numpy()
                )
                labels[task].append(batch_labels[task].detach().cpu().numpy())
            for task in STAGE5_TASKS:
                batch_loss_sums[task] += float(
                    task_losses[task].detach().cpu()
                )
                sample_loss_sums[task] += float(
                    per_sample_losses[task].detach().sum().cpu()
                )
            clean_loss_value = float(clean_loss.detach().cpu())
            clean_total_loss_sum += clean_loss_value
            adversarial_total_loss_sum += adversarial_loss_value
            total_loss_sum += clean_loss_value + adversarial_loss_value

            if batch_index - window_start + 1 == window_size:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    values.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                if (
                    optimizer_steps % values.log_every_optimizer_steps == 0
                    or batch_index == total_batches
                ):
                    log_values: dict[str, Any] = {
                        "phase": phase,
                        "epoch": epoch_id,
                        "train/joint_batch": batch_index,
                        "train/loss_total_running": total_loss_sum / batch_index,
                        "train/loss_clean_total_running": (
                            clean_total_loss_sum / batch_index
                        ),
                        "train/loss_adversarial_total_running": (
                            adversarial_total_loss_sum / batch_index
                        ),
                        "train/loss_pht_running": (
                            batch_loss_sums["triplet"] / batch_index
                        ),
                        "train/loss_ph_running": (
                            batch_loss_sums["phla"] / batch_index
                        ),
                        "train/loss_pt_running": (
                            batch_loss_sums["ptcr"] / batch_index
                        ),
                        "train/gradient_norm": gradient_norms[-1],
                        "train/fgm_epsilon": (
                            values.fgm_epsilon
                            if phase == "stage5_1b"
                            else 0.0
                        ),
                        "train/fgm_attacked_parameters": (
                            fgm_stats.attacked_parameter_count
                            if fgm_stats is not None
                            else 0
                        ),
                    }
                    for group in self.optimizer.param_groups:
                        log_values[
                            f"train/lr/{group.get('name', 'unknown')}"
                        ] = group["lr"]
                    self.tracker.log(log_values, step=self.global_step)
                    _console_log(
                        f"[TRAIN PROGRESS][{phase.upper()}] "
                        f"epoch={epoch_id} batch={batch_index:,}/{total_batches:,} "
                        f"loss={total_loss_sum / batch_index:.4f} "
                        f"elapsed={_format_duration(time.perf_counter() - started_at)}"
                    )

        metrics = {
            task: compute_binary_metrics(
                np.concatenate(labels[task]),
                np.concatenate(probabilities[task]),
                loss=sample_loss_sums[task] / sample_counts[task],
                threshold=self.config.training.threshold,
            )
            for task in ("phla", "ptcr", "triplet")
        }
        return Stage5EpochResult(
            metrics=metrics,
            losses={
                "total": total_loss_sum / total_batches,
                "clean_total": clean_total_loss_sum / total_batches,
                "adversarial_total": (
                    adversarial_total_loss_sum / total_batches
                ),
                "pht": batch_loss_sums["triplet"] / total_batches,
                "phla": batch_loss_sums["phla"] / total_batches,
                "ptcr": batch_loss_sums["ptcr"] / total_batches,
            },
            optimizer_steps=optimizer_steps,
            joint_steps=total_batches,
            model_gradient_norm=(
                float(np.mean(gradient_norms)) if gradient_norms else 0.0
            ),
            fgm_attacked_parameter_count=(
                float(np.mean(attacked_counts)) if attacked_counts else 0.0
            ),
            fgm_mean_gradient_norm=(
                float(np.mean(fgm_mean_norms)) if fgm_mean_norms else 0.0
            ),
            fgm_max_gradient_norm=max(fgm_max_norms, default=0.0),
        )

    def evaluate_binary(self, task: str) -> PredictionResult:
        return evaluate_stage3_model(
            self.model,
            self.val_loaders[task],
            task,
            device=self.device,
            mixed_precision=self.config.stage5.mixed_precision,
            threshold=self.config.training.threshold,
        )

    def evaluate_all(
        self,
    ) -> tuple[
        dict[str, PredictionResult],
        float,
        float,
        BinaryMetrics,
        dict[str, Any],
    ]:
        results: dict[str, PredictionResult] = {
            task: self.evaluate_binary(task) for task in ("phla", "ptcr")
        }
        results["triplet"] = evaluate_stage3_model(
            self.model,
            self.val_loaders["triplet"],
            "triplet",
            device=self.device,
            mixed_precision=self.config.stage5.mixed_precision,
            threshold=self.config.training.threshold,
        )
        triplet_result = results["triplet"]
        threshold, threshold_metrics = select_mcc_threshold(
            triplet_result.labels,
            triplet_result.probabilities,
        )
        threshold_metrics.loss = triplet_result.metrics.loss
        joint = stage5_joint_score(
            {task: result.metrics for task, result in results.items()},
            self.config,
        )
        serialised: dict[str, Any] = {
            task: result.metrics.to_dict() for task, result in results.items()
        }
        serialised["triplet_selected_threshold"] = threshold
        serialised["triplet_selected_threshold_metrics"] = (
            threshold_metrics.to_dict()
        )
        return results, joint, threshold, threshold_metrics, serialised

    def _record_epoch(
        self,
        *,
        phase: Stage5Phase,
        epoch_id: int,
        train_result: Stage5EpochResult,
        val_results: Mapping[str, PredictionResult],
        joint: float,
        threshold: float,
        threshold_metrics: BinaryMetrics,
        improved: bool,
    ) -> None:
        values: dict[str, Any] = {
            "phase": phase,
            f"{phase}/epoch": epoch_id,
            f"{phase}/joint_score": joint,
            f"{phase}/improved": int(improved),
            f"{phase}/selected_threshold": threshold,
            **train_result.to_log_dict(f"{phase}/train"),
        }
        for task, result in val_results.items():
            values.update(prefixed_metrics(f"{phase}/{task}/val", result.metrics))
        values.update(
            prefixed_metrics(
                f"{phase}/triplet/val_selected", threshold_metrics
            )
        )
        self.tracker.log(values, step=self.global_step)
        _console_log(
            f"[{phase.upper()}][epoch={epoch_id}][step={self.global_step}] "
            f"joint={joint:.4f} improved={'yes' if improved else 'no'}"
        )

    def fit(self) -> dict[str, Any]:
        if self.phase == "complete":
            if not self.deploy_path.is_file():
                best = load_stage5_checkpoint(
                    self.stage1b_best_path, map_location="cpu"
                )
                load_triplet_model_state(self.model, best["model"])
                del best
                gc.collect()
                self.save_deploy_checkpoint()
            return self.summary()

        if self.phase == "stage5_1a":
            for epoch_id in range(
                self.stage1a_epoch + 1,
                self.config.stage5.stage1a_epochs + 1,
            ):
                train_result = self.train_joint_epoch(
                    phase="stage5_1a",
                    epoch_id=epoch_id,
                )
                (
                    val_results,
                    joint,
                    threshold,
                    threshold_metrics,
                    _,
                ) = self.evaluate_all()
                improved = (
                    joint
                    > self.best_stage1a_joint_score
                    + self.config.stage5.selection_min_delta
                )
                if improved:
                    self.best_stage1a_joint_score = joint
                self.stage1a_epoch = epoch_id
                self._record_epoch(
                    phase="stage5_1a",
                    epoch_id=epoch_id,
                    train_result=train_result,
                    val_results=val_results,
                    joint=joint,
                    threshold=threshold,
                    threshold_metrics=threshold_metrics,
                    improved=improved,
                )
                self.save_checkpoint(self.last_path)

            self.save_checkpoint(self.stage1a_complete_path)
            self.phase = "stage5_1b"
            self.stage1b_epoch = 0
            self._build_optimization("stage5_1b")
            self.save_checkpoint(self.last_path)

        for epoch_id in range(
            self.stage1b_epoch + 1,
            self.config.stage5.stage1b_epochs + 1,
        ):
            train_result = self.train_joint_epoch(
                phase="stage5_1b",
                epoch_id=epoch_id,
            )
            (
                val_results,
                joint,
                threshold,
                threshold_metrics,
                serialised,
            ) = self.evaluate_all()
            improved = (
                joint
                > self.best_joint_score + self.config.stage5.selection_min_delta
            )
            if improved:
                self.best_joint_score = joint
                self.best_val_metrics = serialised
                self.selected_threshold = threshold
            self.stage1b_epoch = epoch_id
            self._record_epoch(
                phase="stage5_1b",
                epoch_id=epoch_id,
                train_result=train_result,
                val_results=val_results,
                joint=joint,
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
        state = {
            "format_version": 1,
            "checkpoint_type": "stage5_training",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_phase": self.optimizer_phase,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "phase": self.phase,
            "stage1a_epoch": self.stage1a_epoch,
            "stage1b_epoch": self.stage1b_epoch,
            "best_stage1a_joint_score": self.best_stage1a_joint_score,
            "best_joint_score": self.best_joint_score,
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
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        _console_log(
            f"[CHECKPOINT SAVE][STAGE5] phase={self.phase} "
            f"epoch1a={self.stage1a_epoch} epoch1b={self.stage1b_epoch} "
            f"path={path}"
        )
        _atomic_torch_save(self.checkpoint_state(), Path(path))

    def save_deploy_checkpoint(self) -> None:
        state = {
            "format_version": 1,
            "checkpoint_type": "stage5_multitask_model",
            "tasks": ["phla", "ptcr", "triplet"],
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "initialization_source": self.initialization_source,
            "data_audit": self.data_audit,
            "validation_metrics": self.best_val_metrics,
            "joint_score": self.best_joint_score,
            "selected_threshold": self.selected_threshold,
            "run_timestamp": self.run_timestamp,
            "global_step": self.global_step,
        }
        _atomic_torch_save(state, self.deploy_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_stage5_checkpoint(path, map_location="cpu")
        checkpoint_source = require_stage5_scratch_source(
            checkpoint_initialization_source(checkpoint)
        )
        saved_config = checkpoint.get("config", {})
        current_config = self.config.to_dict()
        saved_training = saved_config.get("training", {})
        for key in ("fold", "seed"):
            if saved_training.get(key) != current_config["training"][key]:
                raise ValueError(f"Stage-5 checkpoint {key} mismatch")
        if saved_config.get("model") != current_config["model"]:
            raise ValueError("Stage-5 checkpoint model configuration mismatch")
        saved_stage5 = dict(saved_config.get("stage5", {}))
        current_stage5 = dict(current_config["stage5"])
        for runtime_key in ("eval_batch_size", "log_every_optimizer_steps"):
            saved_stage5.pop(runtime_key, None)
            current_stage5.pop(runtime_key, None)
        if saved_stage5 != current_stage5:
            raise ValueError("Stage-5 checkpoint configuration mismatch")
        saved_metadata = checkpoint.get("encoder_metadata", {})
        current_metadata = self.model.encoder_metadata()
        for entity in ("peptide", "hla", "tcr"):
            saved_fingerprint = saved_metadata.get(entity, {}).get(
                "checkpoint_fingerprint"
            )
            current_fingerprint = current_metadata[entity][
                "checkpoint_fingerprint"
            ]
            if saved_fingerprint and saved_fingerprint != current_fingerprint:
                raise ValueError(
                    f"Stage-5 base-PLM fingerprint mismatch for {entity}"
                )
        load_triplet_model_state(self.model, checkpoint["model"])
        self.phase = checkpoint["phase"]
        self.stage1a_epoch = int(checkpoint["stage1a_epoch"])
        self.stage1b_epoch = int(checkpoint["stage1b_epoch"])
        self.best_stage1a_joint_score = float(
            checkpoint["best_stage1a_joint_score"]
        )
        self.best_joint_score = float(checkpoint["best_joint_score"])
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        self.selected_threshold = float(
            checkpoint.get("selected_threshold", 0.5)
        )
        self.global_step = int(checkpoint["global_step"])
        self.initialization_source = checkpoint_source
        self.data_audit = dict(checkpoint.get("data_audit", {}))
        optimizer_phase = checkpoint.get(
            "optimizer_phase",
            "stage5_1a" if self.phase == "stage5_1a" else "stage5_1b",
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
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "stage1a_epoch": self.stage1a_epoch,
            "stage1b_epoch": self.stage1b_epoch,
            "best_stage1a_joint_score": self.best_stage1a_joint_score,
            "best_joint_score": self.best_joint_score,
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


def load_stage5_checkpoint(
    path: str | Path,
    *,
    map_location,
) -> dict[str, Any]:
    checkpoint = load_training_checkpoint(path, map_location=map_location)
    if (
        checkpoint.get("checkpoint_type") != "stage5_training"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Unsupported Stage-5 training checkpoint format")
    return checkpoint
