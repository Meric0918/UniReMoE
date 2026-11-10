"""Stage-3 triplet warm-up and binary-replay fine-tuning."""

from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
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
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    performance_avg,
    prefixed_metrics,
)
from .tracking import SwanLabTracker
from .trainer import (
    EpochResult,
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    build_scheduler,
    load_training_checkpoint,
    resolve_device,
)
from .triplet_evaluation import (
    PredictionResult,
    evaluate_stage3_multitask,
    evaluate_stage3_model,
    select_composite_threshold,
    select_mcc_threshold,
)
from .triplet_model import (
    ProgressiveTripletFineTuningModel,
    Stage3Phase,
    Stage3Task,
    load_triplet_model_state,
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


def _metric_summary(metrics: BinaryMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} AUROC={metrics.auroc:.4f} "
        f"AUPR={metrics.aupr:.4f} ACC={metrics.accuracy:.4f} "
        f"MCC={metrics.mcc:.4f} F1={metrics.f1:.4f} "
        f"performance_avg={metrics.performance_avg:.4f}"
    )


STAGE3_TASKS = ("phla", "ptcr", "triplet")


def stage3_joint_score(
    metrics: Mapping[str, BinaryMetrics | Mapping[str, float]],
    config: ExperimentConfig,
) -> float:
    del config
    missing = [task for task in STAGE3_TASKS if task not in metrics]
    if missing:
        raise ValueError(
            "Stage-3 joint score is missing task metrics: "
            + ", ".join(missing)
        )
    return float(np.mean([performance_avg(metrics[task]) for task in STAGE3_TASKS]))


def pht_macro_selection_score(
    metrics_by_split: Mapping[
        str, Mapping[str, BinaryMetrics | Mapping[str, float]]
    ],
    selection_splits,
) -> float:
    """Macro-average Stage-3 performance over selected validation splits.

    Each selected split and each of the three Stage-3 heads has equal weight.
    """

    splits = list(selection_splits)
    if not splits:
        raise ValueError("selection_splits cannot be empty")
    missing = [split for split in splits if split not in metrics_by_split]
    if missing:
        raise ValueError(
            "Missing PHT selection splits: " + ", ".join(missing)
        )
    missing = [
        f"{split}/{task}"
        for split in splits
        for task in STAGE3_TASKS
        if task not in metrics_by_split[split]
    ]
    if missing:
        raise ValueError(
            "Missing PHT task metrics: " + ", ".join(missing)
        )
    return float(
        np.mean(
            [
                performance_avg(metrics_by_split[split][task])
                for split in splits
                for task in STAGE3_TASKS
            ]
        )
    )


def checkpoint_initialization_source(
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read generic source metadata, including legacy Stage-2 checkpoints."""

    if not checkpoint:
        return {}
    source = checkpoint.get("initialization_source")
    if isinstance(source, Mapping):
        return dict(source)
    legacy = checkpoint.get("source_stage2")
    if isinstance(legacy, Mapping) and legacy:
        normalized = dict(legacy)
        normalized.setdefault("kind", "stage2")
        return normalized
    return {}


def build_stage3_optimizer(
    model: ProgressiveTripletFineTuningModel,
    config: ExperimentConfig,
    phase: Stage3Phase,
) -> torch.optim.Optimizer:
    values = config.triplet
    groups: list[dict[str, Any]] = []

    def add_group(name: str, parameters, lr: float) -> None:
        parameters = list(parameters)
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr})

    binding = model.binding_model
    if phase == "stage3a":
        add_group(
            "triplet_modules",
            (
                parameter
                for module in binding.new_triplet_modules().values()
                for parameter in module.parameters()
            ),
            values.triplet_module_lr,
        )
        return torch.optim.AdamW(groups, weight_decay=values.weight_decay)

    def add_encoder(name: str, encoder, layer_count: int) -> None:
        blocks = encoder.transformer_blocks()
        add_group(
            f"{name}_plm_top",
            blocks[-1].parameters(),
            values.plm_top_lr,
        )
        if layer_count > 1:
            add_group(
                f"{name}_plm_lower",
                (
                    parameter
                    for block in blocks[-layer_count:-1]
                    for parameter in block.parameters()
                ),
                values.plm_lower_lr,
            )

    add_encoder(
        "peptide",
        model.peptide_encoder,
        model.peptide_unfrozen_layers,
    )
    add_encoder("hla", model.hla_encoder, model.hla_unfrozen_layers)
    add_encoder("tcr", model.tcr_encoder, model.tcr_unfrozen_layers)
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
        "pretrained_attention",
        (
            parameter
            for module in (
                binding.phla_cross_attention,
                binding.ptcr_cross_attention,
                binding.phla_pooling,
                binding.ptcr_pooling,
                binding.phla_classifier,
                binding.ptcr_classifier,
            )
            for parameter in module.parameters()
        ),
        values.pretrained_attention_lr,
    )
    add_group(
        "triplet_modules",
        (
            parameter
            for module in binding.new_triplet_modules().values()
            for parameter in module.parameters()
        ),
        values.triplet_module_lr,
    )
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


class Stage3Trainer:
    """Warm the new head, then train triplet-first with bounded binary replay."""

    def __init__(
        self,
        model: ProgressiveTripletFineTuningModel,
        config: ExperimentConfig,
        train_loaders: Mapping[Stage3Task, Any],
        val_loaders: Mapping[Stage3Task, Any],
        tracker: SwanLabTracker,
        *,
        initialization_source: Mapping[str, Any],
        triplet_data_audit: Mapping[str, Any],
        run_timestamp: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = dict(val_loaders)
        self.tracker = tracker
        self.initialization_source = dict(initialization_source)
        self.triplet_data_audit = dict(triplet_data_audit)
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.phase: Stage3Phase | str = "stage3a"
        self.stage3a_epoch = 0
        self.stage3b_round = 0
        self.best_stage3a_score = -float("inf")
        self.best_joint_score = -float("inf")
        self.best_round_id = 0
        self.best_val_metrics: dict[str, Any] = {}
        self.selected_threshold = 0.5
        self.no_improve_rounds = 0
        self.global_step = 0
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.run_dir / timestamped_filename(
            "stage3_last.pt",
            self.run_timestamp,
        )
        self.stage3a_best_path = self.run_dir / timestamped_filename(
            "stage3a_best_triplet.pt",
            self.run_timestamp,
        )
        self.stage3b_best_path = self.run_dir / timestamped_filename(
            "stage3b_best_joint.pt",
            self.run_timestamp,
        )
        self.best_round_checkpoint_path = self.run_dir / "best_round" / (
            timestamped_filename("stage3_best_joint.pt", self.run_timestamp)
        )
        self.deploy_path = self.run_dir / timestamped_filename(
            "triplet_best.pt",
            self.run_timestamp,
        )
        self.optimizer_phase: Stage3Phase = "stage3a"
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage3a")
        scaler_enabled = (
            self.device.type == "cuda"
            and config.triplet.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        _console_log(
            "[STAGE3 INIT] "
            f"fold={config.training.fold} seed={config.training.seed} "
            f"device={self.device} precision={config.triplet.mixed_precision} "
            f"micro_batch={config.triplet.micro_batch_size} "
            f"accumulation={config.triplet.gradient_accumulation_steps} "
            f"source_kind={self.initialization_source.get('kind')} "
            f"source={self.initialization_source.get('path')}"
        )
        for split_name, loaders in (
            ("train", self.train_loaders),
            ("val", self.val_loaders),
        ):
            summary = ", ".join(
                f"{task}={len(loader.dataset):,}"
                for task, loader in loaders.items()
            )
            _console_log(f"[DATA][{split_name}] {summary}")

    def _planned_optimizer_steps(self, phase: Stage3Phase) -> int:
        accumulation = self.config.triplet.gradient_accumulation_steps
        if phase == "stage3a":
            return (
                math.ceil(len(self.train_loaders["triplet"]) / accumulation)
                * self.config.triplet.head_warmup_epochs
            )
        steps_per_round = sum(
            math.ceil(self._binary_replay_batches(task) / accumulation)
            for task in ("phla", "ptcr")
        ) + math.ceil(
            len(self.train_loaders["triplet"]) / accumulation
        )
        return steps_per_round * self.config.triplet.max_rounds

    def _build_optimization(self, phase: Stage3Phase) -> None:
        self.optimizer_phase = phase
        self.optimizer = build_stage3_optimizer(self.model, self.config, phase)
        total_steps = max(1, self._planned_optimizer_steps(phase))
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=self.config.triplet.scheduler_warmup_ratio,
        )
        groups = ", ".join(
            f"{group.get('name', index)}={group['lr']:.2e}"
            for index, group in enumerate(self.optimizer.param_groups)
        )
        _console_log(
            f"[OPTIMIZER][{phase.upper()}] total_steps={total_steps:,} "
            f"lr_groups=({groups})"
        )

    def _binary_replay_batches(self, task: Stage3Task) -> int:
        if task not in {"phla", "ptcr"}:
            raise ValueError(f"Replay is only defined for binary tasks: {task}")
        triplet_loader = self.train_loaders["triplet"]
        if self.config.triplet.data_mode == "pht_multitask":
            triplet_samples = len(triplet_loader.sampler)
        else:
            triplet_samples = len(triplet_loader.dataset)
        requested_samples = max(
            1,
            math.ceil(
                triplet_samples * self.config.triplet.binary_replay_ratio
            ),
        )
        batch_size = (
            self.train_loaders[task].batch_size
            or self.config.triplet.micro_batch_size
        )
        return min(
            len(self.train_loaders[task]),
            math.ceil(requested_samples / batch_size),
        )

    def _set_train_sampler_epoch(self, task: Stage3Task, epoch: int) -> None:
        sampler = getattr(self.train_loaders[task], "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _autocast(self):
        precision = self.config.triplet.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _forward(self, task: Stage3Task, batch):
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

    def train_epoch(
        self,
        task: Stage3Task,
        loader,
        *,
        phase: Stage3Phase,
        epoch_id: int,
        max_batches: int | None = None,
    ) -> EpochResult:
        self.model.set_trainable_stage3(task, phase)
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        accumulation = self.config.triplet.gradient_accumulation_steps
        total_batches = min(len(loader), max_batches or len(loader))
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        started_at = time.perf_counter()
        _console_log(
            f"[TRAIN START][{phase.upper()}][{task.upper()}] "
            f"id={epoch_id} batches={total_batches:,}"
        )

        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > total_batches:
                break
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(
                accumulation,
                total_batches - window_start + 1,
            )
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            batch_labels = batch["labels"].to(
                self.device,
                dtype=torch.long,
                non_blocking=self.device.type == "cuda",
            )
            with self._autocast():
                output = self._forward(task, batch)
                loss = self.criterion(output.logits, batch_labels)
            self.scaler.scale(loss / window_size).backward()
            batch_samples = int(batch_labels.shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            probabilities.append(
                torch.softmax(output.logits.detach(), dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch_labels.detach().cpu().numpy())

            if batch_index - window_start + 1 == window_size:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    self.config.triplet.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                if (
                    optimizer_steps
                    % self.config.triplet.log_every_optimizer_steps
                    == 0
                    or batch_index == total_batches
                ):
                    elapsed = time.perf_counter() - started_at
                    running_loss = loss_sum / max(1, samples)
                    self.tracker.log(
                        {
                            "phase": phase,
                            "task": task,
                            "epoch_or_round": epoch_id,
                            f"{phase}/{task}/batch": batch_index,
                            f"{phase}/{task}/running_loss": running_loss,
                            f"{phase}/{task}/gradient_norm": gradient_norms[-1],
                        },
                        step=self.global_step,
                    )
                    _console_log(
                        f"[TRAIN PROGRESS][{phase.upper()}][{task.upper()}] "
                        f"batch={batch_index:,}/{total_batches:,} "
                        f"loss={running_loss:.4f} "
                        f"elapsed={_format_duration(elapsed)}"
                    )

        if not samples:
            raise RuntimeError(f"{task} training loader is empty")
        average_loss = loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=average_loss,
            threshold=self.config.training.threshold,
        )
        _console_log(
            f"[TRAIN END][{phase.upper()}][{task.upper()}] "
            f"samples={samples:,} optimizer_steps={optimizer_steps:,} "
            f"{_metric_summary(metrics)}"
        )
        return EpochResult(
            metrics=metrics,
            clean_loss=average_loss,
            adversarial_loss=0.0,
            total_loss=average_loss,
            optimizer_steps=optimizer_steps,
            samples=samples,
            attacked_parameter_count=0.0,
            fgm_mean_gradient_norm=0.0,
            fgm_max_gradient_norm=0.0,
            model_gradient_norm=(
                float(np.mean(gradient_norms)) if gradient_norms else 0.0
            ),
        )

    def evaluate(self, task: Stage3Task, loader) -> PredictionResult:
        started_at = time.perf_counter()
        _console_log(
            f"[VAL START][{task.upper()}] samples={len(loader.dataset):,}"
        )
        result = evaluate_stage3_model(
            self.model,
            loader,
            task,
            device=self.device,
            mixed_precision=self.config.triplet.mixed_precision,
            threshold=self.config.training.threshold,
        )
        _console_log(
            f"[VAL END][{task.upper()}] "
            f"elapsed={_format_duration(time.perf_counter() - started_at)} "
            f"{_metric_summary(result.metrics)}"
        )
        return result

    def _evaluate_all(
        self,
    ) -> tuple[
        dict[str, PredictionResult],
        float,
        float,
        BinaryMetrics,
        dict,
    ]:
        if self.config.triplet.data_mode == "pht_multitask":
            results: dict[str, dict[str, PredictionResult]] = {}
            for split_name, loader in self.val_loaders.items():
                started_at = time.perf_counter()
                _console_log(
                    f"[VAL START][PHT][{split_name}] "
                    f"samples={len(loader.dataset):,}"
                )
                results[split_name] = evaluate_stage3_multitask(
                    self.model,
                    loader,
                    device=self.device,
                    mixed_precision=self.config.triplet.mixed_precision,
                    threshold=self.config.training.threshold,
                )
                _console_log(
                    f"[VAL END][PHT][{split_name}] "
                    f"elapsed={_format_duration(time.perf_counter() - started_at)} "
                    + " | ".join(
                        f"{task}({_metric_summary(result.metrics)})"
                        for task, result in results[split_name].items()
                    )
                )
            metrics_by_split = {
                split: {
                    task: result.metrics for task, result in task_results.items()
                }
                for split, task_results in results.items()
            }
            selection_splits = list(self.config.triplet.selection_splits)
            score = pht_macro_selection_score(
                metrics_by_split, selection_splits
            )
            selected_results = [
                results[split]["triplet"] for split in selection_splits
            ]
            labels = np.concatenate(
                [result.labels for result in selected_results]
            )
            probabilities = np.concatenate(
                [result.probabilities for result in selected_results]
            )
            threshold, threshold_metrics, threshold_score = (
                select_composite_threshold(labels, probabilities)
            )
            total_samples = sum(
                result.metrics.samples for result in selected_results
            )
            threshold_metrics.loss = sum(
                result.metrics.loss * result.metrics.samples
                for result in selected_results
            ) / total_samples
            serialised = {
                split: {
                    task: result.metrics.to_dict()
                    for task, result in task_results.items()
                }
                for split, task_results in results.items()
            }
            serialised.update(
                {
                    "selection_splits": selection_splits,
                    "selection_score": score,
                    "selection_metric": "macro_performance_avg",
                    "pht_selected_threshold": threshold,
                    "pht_selected_threshold_score": threshold_score,
                    "pht_selected_threshold_metrics": (
                        threshold_metrics.to_dict()
                    ),
                    "threshold_metric": (
                        "mean(normalized_mcc,f1,balanced_accuracy)"
                    ),
                }
            )
            return results, score, threshold, threshold_metrics, serialised

        results = {
            task: self.evaluate(task, self.val_loaders[task])
            for task in ("phla", "ptcr", "triplet")
        }
        threshold, threshold_metrics = select_mcc_threshold(
            results["triplet"].labels,
            results["triplet"].probabilities,
        )
        threshold_metrics.loss = results["triplet"].metrics.loss
        metrics = {task: result.metrics for task, result in results.items()}
        joint = stage3_joint_score(metrics, self.config)
        serialised = {
            task: result.metrics.to_dict() for task, result in results.items()
        }
        serialised["triplet_selected_threshold"] = threshold
        serialised["triplet_selected_threshold_metrics"] = (
            threshold_metrics.to_dict()
        )
        return results, joint, threshold, threshold_metrics, serialised

    def _record_stage3b(
        self,
        round_id: int,
        train_results: Mapping[str, EpochResult] | None,
        val_results: Mapping[str, PredictionResult],
        joint: float,
        current_threshold: float,
        threshold_metrics: BinaryMetrics,
        improved: bool,
    ) -> None:
        values: dict[str, Any] = {
            "phase": "stage3b",
            "stage3b/round": round_id,
            "stage3b/round_score": joint,
            "stage3b/joint_score": joint,
            "stage3b/best_joint_score": self.best_joint_score,
            "stage3b/improved": int(improved),
            "stage3b/no_improve_rounds": self.no_improve_rounds,
            "stage3b/selected_threshold": current_threshold,
        }
        if self.config.triplet.data_mode == "pht_multitask":
            for split_name, task_results in val_results.items():
                for task, result in task_results.items():
                    values.update(
                        prefixed_metrics(
                            f"stage3b/{split_name}/{task}/val",
                            result.metrics,
                        )
                    )
            if train_results:
                for task, result in train_results.items():
                    values.update(
                        result.to_log_dict(f"stage3b/{task}/train")
                    )
        else:
            for task, result in val_results.items():
                values.update(
                    prefixed_metrics(f"stage3b/{task}/val", result.metrics)
                )
                if train_results and task in train_results:
                    values.update(
                        train_results[task].to_log_dict(
                            f"stage3b/{task}/train"
                        )
                    )
        values.update(
            prefixed_metrics(
                "stage3b/triplet/val_selected",
                threshold_metrics,
            )
        )
        self.tracker.log(values, step=self.global_step)
        _console_log(
            f"[STAGE3B][round={round_id}][step={self.global_step}] "
            f"joint={joint:.4f} best={self.best_joint_score:.4f} "
            f"improved={'yes' if improved else 'no'} "
            f"threshold={current_threshold:.6f}"
        )

    def fit(self) -> dict[str, Any]:
        if self.phase == "complete":
            if not self.deploy_path.is_file():
                best = load_stage3_checkpoint(
                    self.stage3b_best_path,
                    map_location="cpu",
                )
                load_triplet_model_state(self.model, best["model"])
                del best
                gc.collect()
                self.save_deploy_checkpoint()
            return self.summary()

        if self.phase == "stage3a":
            for epoch_id in range(
                self.stage3a_epoch + 1,
                self.config.triplet.head_warmup_epochs + 1,
            ):
                self._set_train_sampler_epoch("triplet", epoch_id)
                train_result = self.train_epoch(
                    "triplet",
                    self.train_loaders["triplet"],
                    phase="stage3a",
                    epoch_id=epoch_id,
                )
                if self.config.triplet.data_mode == "pht_multitask":
                    (
                        val_results,
                        score,
                        _,
                        _,
                        _,
                    ) = self._evaluate_all()
                else:
                    val_result = self.evaluate(
                        "triplet",
                        self.val_loaders["triplet"],
                    )
                    val_results = {"triplet": val_result}
                    score = performance_avg(val_result.metrics)
                improved = (
                    score
                    > self.best_stage3a_score
                    + self.config.triplet.early_stopping_min_delta
                )
                if improved:
                    self.best_stage3a_score = score
                self.stage3a_epoch = epoch_id
                log_values = {
                    **train_result.to_log_dict("stage3a/triplet/train"),
                    "stage3a/triplet/score": score,
                    "stage3a/triplet/improved": int(improved),
                }
                if self.config.triplet.data_mode == "pht_multitask":
                    for split_name, task_results in val_results.items():
                        for task, result in task_results.items():
                            log_values.update(
                                prefixed_metrics(
                                    f"stage3a/{split_name}/{task}/val",
                                    result.metrics,
                                )
                            )
                else:
                    log_values.update(
                        prefixed_metrics(
                            "stage3a/triplet/val",
                            val_results["triplet"].metrics,
                        )
                    )
                self.tracker.log(log_values, step=self.global_step)
                if improved:
                    self.save_checkpoint(self.stage3a_best_path)
                self.save_checkpoint(self.last_path)

            _console_log(
                "[STAGE TRANSITION] loading best Stage 3A triplet weights "
                f"from {self.stage3a_best_path}"
            )
            best = load_stage3_checkpoint(
                self.stage3a_best_path,
                map_location="cpu",
            )
            load_triplet_model_state(self.model, best["model"])
            del best
            gc.collect()
            self.phase = "stage3b"
            self.stage3b_round = 0
            self.no_improve_rounds = 0
            self._build_optimization("stage3b")
            self.model.set_trainable_stage3("triplet", "stage3b")
            self.model.set_evaluation()

            (
                val_results,
                joint,
                threshold,
                threshold_metrics,
                serialised,
            ) = self._evaluate_all()
            self.best_joint_score = joint
            self.best_round_id = 0
            self.best_val_metrics = serialised
            self.selected_threshold = threshold
            self._record_stage3b(
                0,
                None,
                val_results,
                joint,
                threshold,
                threshold_metrics,
                True,
            )
            self.save_checkpoint(self.stage3b_best_path)
            self.save_best_round_checkpoint(0, joint)
            self.save_deploy_checkpoint()
            self.save_checkpoint(self.last_path)

        for round_id in range(
            self.stage3b_round + 1,
            self.config.triplet.max_rounds + 1,
        ):
            train_results = {}
            for task in ("phla", "ptcr"):
                self._set_train_sampler_epoch(
                    task,
                    self.config.triplet.head_warmup_epochs + round_id,
                )
                train_results[task] = self.train_epoch(
                    task,
                    self.train_loaders[task],
                    phase="stage3b",
                    epoch_id=round_id,
                    max_batches=self._binary_replay_batches(task),
                )
            self._set_train_sampler_epoch(
                "triplet",
                self.config.triplet.head_warmup_epochs + round_id,
            )
            train_results["triplet"] = self.train_epoch(
                "triplet",
                self.train_loaders["triplet"],
                phase="stage3b",
                epoch_id=round_id,
            )
            (
                val_results,
                joint,
                threshold,
                threshold_metrics,
                serialised,
            ) = self._evaluate_all()
            improved = (
                joint
                > self.best_joint_score
                + self.config.triplet.early_stopping_min_delta
            )
            if improved:
                self.best_joint_score = joint
                self.best_round_id = round_id
                self.best_val_metrics = serialised
                self.selected_threshold = threshold
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
            self.stage3b_round = round_id
            self._record_stage3b(
                round_id,
                train_results,
                val_results,
                joint,
                threshold,
                threshold_metrics,
                improved,
            )
            if improved:
                self.save_checkpoint(self.stage3b_best_path)
                self.save_best_round_checkpoint(round_id, joint)
                self.save_deploy_checkpoint()
            self.save_checkpoint(self.last_path)
            if (
                round_id >= self.config.triplet.min_rounds
                and self.no_improve_rounds
                >= self.config.triplet.early_stopping_patience
            ):
                _console_log(
                    "[Stage 3B early stopping] "
                    f"round={round_id} best_joint={self.best_joint_score:.4f}"
                )
                break

        self.phase = "complete"
        self.save_checkpoint(self.last_path)
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "format_version": 1,
            "checkpoint_type": "stage3_training",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_phase": self.optimizer_phase,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "phase": self.phase,
            "stage3a_epoch": self.stage3a_epoch,
            "stage3b_round": self.stage3b_round,
            "best_stage3a_score": self.best_stage3a_score,
            "best_joint_score": self.best_joint_score,
            "best_round": self.best_round_id,
            "round_score": self.best_joint_score,
            "best_selection_score": self.best_joint_score,
            "selection_metric": "macro_performance_avg",
            "model_scope": "joint_heads",
            "best_val_metrics": self.best_val_metrics,
            "selected_threshold": self.selected_threshold,
            "no_improve_rounds": self.no_improve_rounds,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "initialization_source": self.initialization_source,
            "triplet_data_audit": self.triplet_data_audit,
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
        if self.initialization_source.get("kind") == "stage2":
            state["source_stage2"] = self.initialization_source
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        _console_log(
            f"[CHECKPOINT SAVE][STAGE3] phase={self.phase} "
            f"round={self.stage3b_round} path={path}"
        )
        _atomic_torch_save(self.checkpoint_state(), path)

    def save_best_round_checkpoint(
        self,
        round_id: int,
        round_score: float,
    ) -> None:
        """Save the complete joint model and resumable Stage-3 state."""

        state = self.checkpoint_state()
        state.update(
            {
                "checkpoint_type": "stage3_joint_model",
                "best_round": int(round_id),
                "round_score": float(round_score),
                "model_scope": "joint_heads",
                "selection_metric": "macro_performance_avg",
            }
        )
        self.best_round_id = int(round_id)
        _console_log(
            f"[CHECKPOINT SAVE][STAGE3][BEST ROUND] round={round_id} "
            f"score={round_score:.4f} path={self.best_round_checkpoint_path}"
        )
        _atomic_torch_save(state, self.best_round_checkpoint_path)

    def save_deploy_checkpoint(self) -> None:
        state = {
            "format_version": 1,
            "checkpoint_type": "stage3_multitask_model",
            "tasks": ["phla", "ptcr", "triplet"],
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "initialization_source": self.initialization_source,
            "triplet_data_audit": self.triplet_data_audit,
            "validation_metrics": self.best_val_metrics,
            "joint_score": self.best_joint_score,
            "selection_score": self.best_joint_score,
            "selection_metric": "macro_performance_avg",
            "best_round": self.best_round_id,
            "round_score": self.best_joint_score,
            "model_scope": "joint_heads",
            "selected_threshold": self.selected_threshold,
            "run_timestamp": self.run_timestamp,
            "global_step": self.global_step,
        }
        if self.initialization_source.get("kind") == "stage2":
            state["source_stage2"] = self.initialization_source
        _atomic_torch_save(state, self.deploy_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_stage3_checkpoint(path, map_location="cpu")
        saved_config = checkpoint.get("config", {})
        current_config = self.config.to_dict()
        saved_training = saved_config.get("training", {})
        for key in ("fold", "seed"):
            current_value = current_config["training"][key]
            if saved_training.get(key) != current_value:
                raise ValueError(
                    f"Stage-3 checkpoint {key}={saved_training.get(key)} "
                    f"does not match current {key}={current_value}"
                )
        if saved_config.get("model") != current_config["model"]:
            raise ValueError("Stage-3 checkpoint model configuration mismatch")
        saved_triplet = dict(saved_config.get("triplet", {}))
        current_triplet = dict(current_config["triplet"])
        for runtime_key in ("eval_batch_size", "log_every_optimizer_steps"):
            saved_triplet.pop(runtime_key, None)
            current_triplet.pop(runtime_key, None)
        if saved_triplet != current_triplet:
            raise ValueError("Stage-3 checkpoint triplet configuration mismatch")
        saved_audit = dict(checkpoint.get("triplet_data_audit", {}))
        if self.config.triplet.data_mode == "pht_multitask":
            saved_fingerprint = saved_audit.get("dataset_fingerprint")
            current_fingerprint = self.triplet_data_audit.get(
                "dataset_fingerprint"
            )
            if saved_fingerprint != current_fingerprint:
                raise ValueError(
                    "Stage-3 PHT dataset fingerprint mismatch: "
                    f"saved={saved_fingerprint}, current={current_fingerprint}"
                )
            if saved_audit.get("limit_per_type") != self.triplet_data_audit.get(
                "limit_per_type"
            ):
                raise ValueError(
                    "Stage-3 PHT limit_per_type does not match resume data"
                )
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
                    f"Stage-3 base-PLM fingerprint mismatch for {entity}"
                )
        load_triplet_model_state(self.model, checkpoint["model"])
        self.phase = checkpoint["phase"]
        self.stage3a_epoch = int(checkpoint["stage3a_epoch"])
        self.stage3b_round = int(checkpoint["stage3b_round"])
        self.best_stage3a_score = float(checkpoint["best_stage3a_score"])
        self.best_joint_score = float(checkpoint["best_joint_score"])
        self.best_round_id = int(
            checkpoint.get("best_round", checkpoint.get("stage3b_round", 0))
        )
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        self.selected_threshold = float(
            checkpoint.get("selected_threshold", 0.5)
        )
        self.no_improve_rounds = int(checkpoint["no_improve_rounds"])
        self.global_step = int(checkpoint["global_step"])
        self.initialization_source = checkpoint_initialization_source(
            checkpoint
        )
        self.triplet_data_audit = dict(
            checkpoint.get("triplet_data_audit", {})
        )
        optimizer_phase = checkpoint.get(
            "optimizer_phase",
            "stage3a" if self.phase == "stage3a" else "stage3b",
        )
        self._build_optimization(optimizer_phase)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        for name, generator_state in checkpoint.get(
            "dataloader_rng_state",
            {},
        ).items():
            split, task = name.split("/", 1)
            loaders = (
                self.train_loaders if split == "train" else self.val_loaders
            )
            generator = getattr(loaders.get(task), "generator", None)
            if generator is not None:
                generator.set_state(generator_state)
        _restore_rng_state(checkpoint.get("rng_state"))

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "stage3a_epoch": self.stage3a_epoch,
            "stage3b_round": self.stage3b_round,
            "best_stage3a_score": self.best_stage3a_score,
            "best_joint_score": self.best_joint_score,
            "best_round": self.best_round_id,
            "best_val_metrics": self.best_val_metrics,
            "selected_threshold": self.selected_threshold,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "last_checkpoint": str(self.last_path),
            "stage3a_best_checkpoint": str(self.stage3a_best_path),
            "stage3b_best_checkpoint": str(self.stage3b_best_path),
            "best_round_checkpoint": str(self.best_round_checkpoint_path),
            "deploy_checkpoint": str(self.deploy_path),
            "initialization_source": self.initialization_source,
        }


def load_stage3_checkpoint(
    path: str | Path,
    *,
    map_location,
) -> dict[str, Any]:
    checkpoint = load_training_checkpoint(path, map_location=map_location)
    if (
        checkpoint.get("checkpoint_type")
        not in {"stage3_training", "stage3_joint_model"}
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Unsupported Stage-3 training checkpoint format")
    return checkpoint
