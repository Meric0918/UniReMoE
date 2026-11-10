"""Training loop for cached Stage-8 downstream modules."""

from __future__ import annotations

import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

import numpy as np
import torch

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .stage8_checkpoint import (
    Stage3Source,
    atomic_torch_save,
    load_torch_checkpoint,
    stage8_training_contract,
    validate_stage8_resume_checkpoint,
)
from .stage8_config import Stage8Condition, Stage8ExperimentConfig
from .stage8_evaluation import (
    Stage8Evaluation,
    evaluate_online_stage8_model,
    evaluate_stage8_model,
)
from .stage8_loss import compute_stage8_loss
from .metrics import compute_binary_metrics
from .trainer import build_scheduler


def build_stage8_optimizer(model, config: Stage8ExperimentConfig) -> torch.optim.Optimizer:
    values = config.stage8
    groups: list[dict[str, Any]] = []

    def add(name: str, modules, lr: float) -> None:
        parameters = [parameter for module in modules for parameter in module.parameters()]
        groups.append({"name": name, "params": parameters, "lr": lr})

    add(
        "sequence_encoders",
        (model.peptide_encoder, model.hla_encoder, model.tcr_encoder),
        values.sequence_encoder_lr,
    )
    add("pretrained_attention", (
        model.phla_cross_attention, model.ptcr_cross_attention,
        model.phla_pooling, model.ptcr_pooling,
        model.phla_classifier, model.ptcr_classifier,
    ), values.pretrained_attention_lr)
    add("triplet_modules", tuple(model.new_triplet_modules().values()), values.triplet_module_lr)
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Stage-8 optimizer contains duplicate parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


Stage8OnlinePhase = Literal["stage8a", "stage8b"]


def validate_stage8_online_trainer_contract(
    config: Stage8ExperimentConfig,
) -> None:
    """Reject schedule drift before constructing the online Stage-8 lifecycle."""

    values = config.stage8
    contract = {
        "stage8a_epochs": 2,
        "binary_replay_ratio": 0.25,
        "peptide_unfrozen_layers": 2,
        "hla_unfrozen_layers": 2,
        "tcr_unfrozen_layers": 2,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "min_rounds": 5,
        "max_rounds": 100,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-4,
        "constraint_ramp_epochs": 5,
        "pht_score_weight": 0.4,
        "phla_score_weight": 0.2,
        "ptcr_score_weight": 0.4,
    }
    for name, expected in contract.items():
        if getattr(values, name) != expected:
            raise ValueError(f"stage8.{name} must equal {expected}")


def build_stage8_online_optimizer(
    model,
    config: Stage8ExperimentConfig,
    phase: Stage8OnlinePhase,
) -> torch.optim.Optimizer:
    """Build the online optimizer from the phase's real trainable modules."""

    if phase not in {"stage8a", "stage8b"}:
        raise ValueError(f"Unknown Stage-8 online optimizer phase: {phase}")
    values = config.stage8
    binding = model.binding_model
    groups: list[dict[str, Any]] = []

    def add(name: str, parameters, lr: float) -> None:
        active = [parameter for parameter in parameters if parameter.requires_grad]
        if active:
            groups.append({"name": name, "params": active, "lr": lr})

    if phase == "stage8a":
        add(
            "triplet_modules",
            (
                parameter
                for module in binding.new_triplet_modules().values()
                for parameter in module.parameters()
            ),
            values.triplet_module_lr,
        )
    else:
        def add_encoder(name: str, encoder, layer_count: int) -> None:
            blocks = encoder.transformer_blocks()
            add(f"{name}_plm_top", blocks[-1].parameters(), 5e-6)
            if layer_count > 1:
                add(
                    f"{name}_plm_lower",
                    (
                        parameter
                        for block in blocks[-layer_count:-1]
                        for parameter in block.parameters()
                    ),
                    2.5e-6,
                )

        add_encoder("peptide", model.peptide_encoder, model.peptide_unfrozen_layers)
        add_encoder("hla", model.hla_encoder, model.hla_unfrozen_layers)
        add_encoder("tcr", model.tcr_encoder, model.tcr_unfrozen_layers)
        add(
            "sequence_encoders",
            (
                parameter
                for module in (
                    binding.peptide_encoder,
                    binding.hla_encoder,
                    binding.tcr_encoder,
                )
                for parameter in module.parameters()
            ),
            values.sequence_encoder_lr,
        )
        add(
            "pretrained_attention",
            (
                parameter
                for module in (
                    binding.phla_cross_attention, binding.ptcr_cross_attention,
                    binding.phla_pooling, binding.ptcr_pooling,
                    binding.phla_classifier, binding.ptcr_classifier,
                )
                for parameter in module.parameters()
            ),
            values.pretrained_attention_lr,
        )
        add(
            "triplet_modules",
            (
                parameter
                for module in binding.new_triplet_modules().values()
                for parameter in module.parameters()
            ),
            values.triplet_module_lr,
        )
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Stage-8 online optimizer contains duplicate parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


def _online_batch_samples(batch: Mapping[str, Any]) -> int:
    for key in ("labels", "pht_labels", "ph_labels", "pt_labels"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    for key in ("peptide_sequences", "receptor_sequences"):
        value = batch.get(key)
        if isinstance(value, list):
            return len(value)
    raise ValueError("Could not infer Stage-8 online batch size")


def _online_training_metrics(
    labels: list[np.ndarray], probabilities: list[np.ndarray], loss: float
) -> dict[str, float]:
    """Summarize one online training pass for the live training monitor."""

    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_score = np.concatenate(probabilities).astype(np.float64, copy=False)
    if len(np.unique(y_true)) < 2:
        # AUROC/AUPR are undefined for a one-class pass; keep loss usable.
        return {"loss": float(loss)}
    metrics = compute_binary_metrics(y_true, y_score, loss=loss)
    return {
        "loss": metrics.loss,
        "auroc": metrics.auroc,
        "aupr": metrics.aupr,
    }


def _slice_online_batch(batch: Mapping[str, Any], size: int) -> dict[str, Any]:
    return {
        key: value[:size] if isinstance(value, (torch.Tensor, list, tuple)) else value
        for key, value in batch.items()
    }


def _iter_capped_online_batches(
    loader: Iterable[Mapping[str, Any]],
    max_samples: int,
) -> Iterator[Mapping[str, Any]]:
    remaining = max_samples
    for batch in loader:
        if remaining == 0:
            return
        batch_samples = _online_batch_samples(batch)
        if batch_samples < 1:
            raise ValueError("Stage-8 online replay batch is empty")
        if batch_samples > remaining:
            batch = _slice_online_batch(batch, remaining)
            batch_samples = remaining
        yield batch
        remaining -= batch_samples


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Stage8Trainer:
    def __init__(
        self,
        model,
        config: Stage8ExperimentConfig,
        train_loader,
        val_loader,
        *,
        condition: Stage8Condition | str,
        source: Stage3Source,
        data_audit: dict,
        tracker=None,
        run_timestamp: str | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.condition = Stage8Condition(condition)
        self.source = source
        self.data_audit = data_audit
        self.tracker = tracker
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.device = next(model.parameters()).device
        self.optimizer = build_stage8_optimizer(model, config)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and config.stage8.mixed_precision == "fp16"),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=max(1, config.stage8.max_epochs * len(train_loader)),
            warmup_ratio=config.stage8.scheduler_warmup_ratio,
        )
        self.global_step = 0
        self.start_epoch = 1
        self.best_epoch = 0
        self.best_joint = float("-inf")
        self.bad_epochs = 0
        self.baseline: Stage8Evaluation | None = None
        self.best_evaluation: Stage8Evaluation | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.training_contract = stage8_training_contract(
            config, data_audit.get("effective_data")
        )
        self.run_dir = config.run_dir(self.condition)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.run_dir / f"stage8_best_{self.run_timestamp}.pt"
        self.last_path = self.run_dir / f"stage8_last_{self.run_timestamp}.pt"
        self.prediction_path = self.run_dir / f"validation_predictions_{self.run_timestamp}.csv"
        self.summary_path = self.run_dir / f"summary_{self.run_timestamp}.json"

    def _move(self, batch: dict) -> dict:
        return {
            key: (
                value.to(self.device, non_blocking=self.device.type == "cuda")
                if isinstance(value, torch.Tensor) else value
            )
            for key, value in batch.items()
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.model.train()
        totals: dict[str, float] = {}
        samples = 0
        values = self.config.stage8
        use_autocast = self.device.type == "cuda" and values.mixed_precision != "none"
        dtype = torch.float16 if values.mixed_precision == "fp16" else torch.bfloat16
        for batch_index, batch in enumerate(self.train_loader, start=1):
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()
            with context:
                output = self.model.forward_all(
                    batch["peptide_hidden"], batch["peptide_mask"],
                    batch["hla_hidden"], batch["hla_mask"],
                    batch["tcr_hidden"], batch["tcr_mask"],
                )
                loss = compute_stage8_loss(
                    output.phla_logits, output.ptcr_logits, output.pht_logits,
                    batch["ph_labels"], batch["pt_labels"], batch["pht_labels"],
                    batch["label_types"], condition=self.condition, epoch=epoch,
                    margin=values.constraint_margin,
                    hierarchy_lambda=values.hierarchy_lambda,
                    pt_implication_lambda=values.pt_implication_lambda,
                    ramp_epochs=values.constraint_ramp_epochs,
                    pht_weight=values.pht_loss_weight,
                    phla_weight=values.phla_loss_weight,
                    ptcr_weight=values.ptcr_loss_weight,
                )
            self.scaler.scale(loss.total).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), values.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            count = int(batch["pht_labels"].shape[0])
            samples += count
            scalars = {
                "total": loss.total, "supervised": loss.supervised,
                "pht": loss.pht, "phla": loss.phla, "ptcr": loss.ptcr,
                "hierarchy": loss.hierarchy, "pt_implication": loss.pt_implication,
            }
            for name, tensor in scalars.items():
                totals[name] = totals.get(name, 0.0) + float(tensor.detach().cpu()) * count
            if self.tracker and (
                self.global_step == 1 or self.global_step % values.log_every_steps == 0
            ):
                self.tracker.log({
                    **{f"train/{name}": totals[name] / samples for name in totals},
                    "train/gradient_norm": float(gradient_norm),
                    "train/epoch": epoch,
                    "train/hierarchy_lambda": loss.hierarchy_lambda,
                    "train/pt_implication_lambda": loss.pt_implication_lambda,
                }, step=self.global_step)
        return {name: total / samples for name, total in totals.items()}

    def evaluate(self) -> Stage8Evaluation:
        return evaluate_stage8_model(
            self.model, self.val_loader, device=self.device,
            mixed_precision=self.config.stage8.mixed_precision,
            margin=self.config.stage8.constraint_margin,
            selection_weights=self.config.stage8.selection_weights,
        )

    def _checkpoint(
        self,
        epoch: int,
        evaluation: Stage8Evaluation,
        phase: str,
        artifact_role: str,
    ) -> dict:
        return {
            "format_version": 1,
            "checkpoint_type": "stage8_binding_model",
            "condition": self.condition.value,
            "phase": phase,
            "artifact_role": artifact_role,
            "epoch": epoch,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": getattr(self.tracker, "run_id", None),
            "config": self.config.to_dict(),
            "source_stage3": self.source.lineage,
            "data_audit": self.data_audit,
            "training_contract": self.training_contract,
            "model": self.model.state_dict(),
            "best_model": self.best_model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng_state": _rng_state(),
            "best_epoch": self.best_epoch,
            "best_joint_score": self.best_joint,
            "bad_epochs": self.bad_epochs,
            "baseline_metrics": self.baseline.metrics if self.baseline else None,
            "validation_metrics": evaluation.metrics,
            "validation_joint_score": evaluation.joint_score,
            "selected_pht_threshold": evaluation.selected_pht_threshold,
        }

    def save(
        self,
        path: Path,
        epoch: int,
        evaluation: Stage8Evaluation,
        phase: str,
        artifact_role: str,
    ) -> None:
        atomic_torch_save(
            self._checkpoint(epoch, evaluation, phase, artifact_role), path
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(checkpoint_path, mmap=False)
        if checkpoint.get("artifact_role") != "last":
            raise ValueError(
                "Stage-8 resume requires a self-consistent last checkpoint, not best"
            )
        validate_stage8_resume_checkpoint(
            checkpoint, self.config, self.source.sha256,
            self.data_audit.get("effective_data"),
        )
        if checkpoint.get("condition") != self.condition.value:
            raise ValueError("Cannot resume Stage 8 under a different ablation condition")
        saved_fingerprint = checkpoint.get("data_audit", {}).get("dataset_fingerprint")
        current_fingerprint = self.data_audit.get("dataset_fingerprint")
        if saved_fingerprint != current_fingerprint:
            raise ValueError("Stage-8 resume dataset fingerprint mismatch")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.global_step = int(checkpoint["global_step"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_epoch = int(checkpoint["best_epoch"])
        self.best_joint = float(checkpoint["best_joint_score"])
        self.bad_epochs = int(checkpoint["bad_epochs"])
        best_model = checkpoint.get("best_model")
        if not isinstance(best_model, dict):
            raise ValueError("Stage-8 resume checkpoint has no self-contained best model")
        self.best_model_state = best_model
        _restore_rng_state(checkpoint.get("rng_state"))

    def _capture_best_model(self) -> None:
        self.best_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _write_predictions(self, evaluation: Stage8Evaluation) -> None:
        arrays = evaluation.predictions
        with self.prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "source_row", "label_type", "PH", "PT", "PHT",
                "prob_PH", "prob_PT", "prob_PHT",
            ])
            for index in range(len(arrays["pht_labels"])):
                writer.writerow([
                    int(arrays["source_rows"][index]), arrays["label_types"][index],
                    int(arrays["ph_labels"][index]), int(arrays["pt_labels"][index]),
                    int(arrays["pht_labels"][index]), float(arrays["phla_prob"][index]),
                    float(arrays["ptcr_prob"][index]), float(arrays["pht_prob"][index]),
                ])

    def _log_evaluation(self, epoch: int, evaluation: Stage8Evaluation) -> None:
        if not self.tracker:
            return
        payload = {"validation/epoch": epoch, "validation/joint_score": evaluation.joint_score}
        for task in ("pht", "phla", "ptcr"):
            for name, value in evaluation.metrics[task].items():
                payload[f"validation/{task}/{name}"] = value
        for name, value in evaluation.metrics["violations"].items():
            payload[f"validation/violations/{name}"] = value
        self.tracker.log(payload, step=self.global_step)

    def fit(self, *, resume: str | Path | None = None) -> dict:
        values = self.config.stage8
        # The freshly constructed model is always the unmodified Stage-3 source.
        self.baseline = self.evaluate()
        if resume:
            self.resume(resume)
            final_evaluation = self.evaluate()
        else:
            self.best_evaluation = self.baseline
            self.best_joint = self.baseline.joint_score
            self.best_epoch = 0
            self._capture_best_model()
            self._write_predictions(self.baseline)
            self.save(self.best_path, 0, self.baseline, "baseline", "best")
            self.save(self.last_path, 0, self.baseline, "baseline", "last")
            self._log_evaluation(0, self.baseline)
            final_evaluation = self.baseline
        final_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, values.max_epochs + 1):
            self.train_epoch(epoch)
            evaluation = self.evaluate()
            final_evaluation = evaluation
            final_epoch = epoch
            improved = evaluation.joint_score > self.best_joint + values.early_stopping_min_delta
            if improved:
                self.best_joint = evaluation.joint_score
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.best_evaluation = evaluation
                self._capture_best_model()
                self._write_predictions(evaluation)
                self.save(self.best_path, epoch, evaluation, "training", "best")
            else:
                self.bad_epochs += 1
            self.save(self.last_path, epoch, evaluation, "training", "last")
            self._log_evaluation(epoch, evaluation)
            if epoch >= values.min_epochs and self.bad_epochs >= values.early_stopping_patience:
                break
        self.save(
            self.last_path, final_epoch, final_evaluation, "complete", "last"
        )
        if self.best_model_state is None:
            raise RuntimeError("Stage-8 best model state was not initialized")
        self.model.load_state_dict(self.best_model_state, strict=True)
        best_evaluation = self.evaluate()
        self._write_predictions(best_evaluation)
        self.save(
            self.best_path, self.best_epoch, best_evaluation, "complete", "best"
        )
        best_metrics = best_evaluation.metrics
        baseline_metrics = self.baseline.metrics
        success = (
            best_evaluation.joint_score
            >= self.baseline.joint_score - values.joint_degradation_tolerance
            and best_metrics["violations"]["overall"]
            < self.baseline.metrics["violations"]["overall"]
        )
        summary = {
            "condition": self.condition.value,
            "source_stage3": self.source.lineage,
            "dataset_fingerprint": self.data_audit.get("dataset_fingerprint"),
            "baseline_joint_score": self.baseline.joint_score,
            "baseline_metrics": baseline_metrics,
            "best_epoch": self.best_epoch,
            "best_joint_score": best_evaluation.joint_score,
            "best_metrics": best_metrics,
            "joint_degradation_tolerance": values.joint_degradation_tolerance,
            "primary_success": success,
            "best_checkpoint": str(self.best_path),
            "last_checkpoint": str(self.last_path),
            "validation_predictions": str(self.prediction_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


class Stage8OnlineTrainer:
    """In-memory Stage8A/Stage8B training state for the online model.

    Checkpoint persistence is intentionally not part of this class; Task 3
    owns the V2 warmup/best/last artifact contract.  The exposed phase and
    round state lets that layer persist a self-contained lifecycle later.
    """

    def __init__(
        self,
        model,
        config: Stage8ExperimentConfig,
        train_loaders: Mapping[str, Any],
        val_loader,
        *,
        condition: Stage8Condition | str,
        tracker=None,
    ):
        validate_stage8_online_trainer_contract(config)
        required = {"phla", "ptcr", "joint"}
        if set(train_loaders) != required:
            raise ValueError("Stage-8 online training requires phla, ptcr, and joint loaders")
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loader = val_loader
        self.condition = Stage8Condition(condition)
        self.tracker = tracker
        self.device = next(model.parameters()).device
        self.phase: Stage8OnlinePhase = "stage8a"
        self.stage8a_epoch = 0
        self.stage8b_round = 0
        self.global_step = 0
        self.bad_rounds = 0
        self.best_joint = -float("inf")
        self.best_evaluation: Stage8Evaluation | None = None
        self.pass_history: list[tuple[str, int, str]] = []
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and config.stage8.mixed_precision == "fp16"),
        )
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage8a")

    def _autocast(self):
        if self.device.type != "cuda" or self.config.stage8.mixed_precision == "none":
            return nullcontext()
        dtype = torch.float16 if self.config.stage8.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)

    def _joint_samples(self) -> int:
        dataset = getattr(self.train_loaders["joint"], "dataset", None)
        if dataset is None:
            raise ValueError("Stage-8 online joint loader must expose its dataset")
        return len(dataset)

    def _replay_samples(self, task: str) -> int:
        loader = self.train_loaders[task]
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise ValueError(f"Stage-8 {task} replay loader must expose its dataset")
        return min(
            len(dataset),
            math.floor(self._joint_samples() * self.config.stage8.binary_replay_ratio),
        )

    def _replay_batches(self, task: str) -> int:
        samples = self._replay_samples(task)
        if samples == 0:
            return 0
        loader = self.train_loaders[task]
        batch_size = getattr(loader, "batch_size", None) or self.config.stage8.micro_batch_size
        return min(len(loader), math.ceil(samples / batch_size))

    def _planned_optimizer_steps(self, phase: Stage8OnlinePhase) -> int:
        accumulation = self.config.stage8.gradient_accumulation_steps
        joint_steps = math.ceil(len(self.train_loaders["joint"]) / accumulation)
        if phase == "stage8a":
            return max(1, joint_steps * self.config.stage8.stage8a_epochs)
        replay_steps = sum(
            math.ceil(self._replay_batches(task) / accumulation)
            for task in ("phla", "ptcr")
        )
        return max(1, (joint_steps + replay_steps) * self.config.stage8.max_rounds)

    def _build_optimization(self, phase: Stage8OnlinePhase) -> None:
        self.model.set_trainable_stage8("joint", phase)
        self.optimizer = build_stage8_online_optimizer(self.model, self.config, phase)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self._planned_optimizer_steps(phase),
            warmup_ratio=self.config.stage8.scheduler_warmup_ratio,
        )

    def _optimizer_step(self, trainable) -> float:
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, self.config.stage8.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return float(norm.detach().cpu())

    def _train_joint(
        self,
        *,
        phase: Stage8OnlinePhase,
        epoch_or_round: int,
    ) -> dict[str, Any]:
        self.model.set_trainable_stage8("joint", phase)
        self.pass_history.append((phase, epoch_or_round, "joint"))
        loader = self.train_loaders["joint"]
        total_batches = len(loader)
        accumulation = self.config.stage8.gradient_accumulation_steps
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        totals = {name: 0.0 for name in (
            "total", "supervised", "pht", "phla", "ptcr", "hierarchy", "pt_implication",
        )}
        samples = 0
        optimizer_steps = 0
        hierarchy_lambda = 0.0
        pt_implication_lambda = 0.0
        labels_seen = {task: [] for task in ("pht", "phla", "ptcr")}
        probabilities_seen = {task: [] for task in ("pht", "phla", "ptcr")}
        for batch_index, batch in enumerate(loader, start=1):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            labels = {
                key: batch[key].to(self.device, non_blocking=self.device.type == "cuda")
                for key in ("ph_labels", "pt_labels", "pht_labels")
            }
            with self._autocast():
                output = self.model.forward_all(
                    batch["peptide_sequences"], batch["hla_sequences"], batch["tcr_sequences"],
                )
                loss = compute_stage8_loss(
                    output.phla_logits, output.ptcr_logits, output.pht_logits,
                    labels["ph_labels"], labels["pt_labels"], labels["pht_labels"],
                    batch["label_types"],
                    condition=self.condition,
                    epoch=epoch_or_round,
                    phase=phase,
                    round_index=epoch_or_round if phase == "stage8b" else None,
                    margin=self.config.stage8.constraint_margin,
                    hierarchy_lambda=self.config.stage8.hierarchy_lambda,
                    pt_implication_lambda=self.config.stage8.pt_implication_lambda,
                    ramp_epochs=self.config.stage8.constraint_ramp_epochs,
                    pht_weight=self.config.stage8.pht_loss_weight,
                    phla_weight=0.0 if phase == "stage8a" else self.config.stage8.phla_loss_weight,
                    ptcr_weight=0.0 if phase == "stage8a" else self.config.stage8.ptcr_loss_weight,
                )
            for task, label_key, logits in (
                ("pht", "pht_labels", output.pht_logits),
                ("phla", "ph_labels", output.phla_logits),
                ("ptcr", "pt_labels", output.ptcr_logits),
            ):
                labels_seen[task].append(
                    labels[label_key].detach().cpu().numpy()
                )
                probabilities_seen[task].append(
                    torch.softmax(logits.detach(), dim=-1)[:, 1].float().cpu().numpy()
                )
            self.scaler.scale(loss.total / window_size).backward()
            count = int(labels["pht_labels"].shape[0])
            samples += count
            for name, value in (
                ("total", loss.total), ("supervised", loss.supervised),
                ("pht", loss.pht), ("phla", loss.phla), ("ptcr", loss.ptcr),
                ("hierarchy", loss.hierarchy), ("pt_implication", loss.pt_implication),
            ):
                totals[name] += float(value.detach().cpu()) * count
            hierarchy_lambda = loss.hierarchy_lambda
            pt_implication_lambda = loss.pt_implication_lambda
            if batch_index - window_start + 1 == window_size:
                self._optimizer_step(trainable)
                optimizer_steps += 1
        if not samples:
            raise RuntimeError("Stage-8 online joint loader is empty")
        task_metrics = {
            task: _online_training_metrics(
                labels_seen[task], probabilities_seen[task],
                totals[loss_name] / samples,
            )
            for task, loss_name in (
                ("pht", "pht"), ("phla", "phla"), ("ptcr", "ptcr")
            )
        }
        return {
            **{name: value / samples for name, value in totals.items()},
            "samples": samples,
            "optimizer_steps": optimizer_steps,
            "hierarchy_lambda": hierarchy_lambda,
            "pt_implication_lambda": pt_implication_lambda,
            "task_metrics": task_metrics,
        }

    def _train_replay(self, task: Literal["phla", "ptcr"], round_index: int) -> dict[str, Any]:
        self.model.set_trainable_stage8(task, "stage8b")
        self.pass_history.append(("stage8b", round_index, task))
        max_samples = self._replay_samples(task)
        if max_samples == 0:
            return {"loss": 0.0, "samples": 0, "optimizer_steps": 0}
        total_batches = self._replay_batches(task)
        accumulation = self.config.stage8.gradient_accumulation_steps
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        samples = 0
        loss_sum = 0.0
        optimizer_steps = 0
        labels_seen: list[np.ndarray] = []
        probabilities_seen: list[np.ndarray] = []
        for batch_index, batch in enumerate(
            _iter_capped_online_batches(self.train_loaders[task], max_samples), start=1
        ):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(self.device, non_blocking=self.device.type == "cuda")
            with self._autocast():
                output = self.model(
                    task, batch["peptide_sequences"], batch["receptor_sequences"]
                )
                loss = torch.nn.functional.cross_entropy(output.logits, labels)
            labels_seen.append(labels.detach().cpu().numpy())
            probabilities_seen.append(
                torch.softmax(output.logits.detach(), dim=-1)[:, 1].float().cpu().numpy()
            )
            self.scaler.scale(loss / window_size).backward()
            count = int(labels.shape[0])
            samples += count
            loss_sum += float(loss.detach().cpu()) * count
            if batch_index - window_start + 1 == window_size:
                self._optimizer_step(trainable)
                optimizer_steps += 1
        if not samples:
            raise RuntimeError(f"Stage-8 {task} replay loader is empty")
        metrics = _online_training_metrics(labels_seen, probabilities_seen, loss_sum / samples)
        return {
            **metrics,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
        }

    def train_stage8a_epoch(self, epoch: int) -> dict[str, Any]:
        if self.phase != "stage8a":
            raise RuntimeError("Stage8A has already completed")
        if epoch != self.stage8a_epoch + 1 or epoch > self.config.stage8.stage8a_epochs:
            raise ValueError("Stage8A epochs must run once in ascending order")
        metrics = self._train_joint(phase="stage8a", epoch_or_round=epoch)
        self.stage8a_epoch = epoch
        return metrics

    def fit_stage8a(self) -> list[dict[str, Any]]:
        return [
            self.train_stage8a_epoch(epoch)
            for epoch in range(self.stage8a_epoch + 1, self.config.stage8.stage8a_epochs + 1)
        ]

    def begin_stage8b(self) -> None:
        if self.phase == "stage8b":
            return
        if self.stage8a_epoch != self.config.stage8.stage8a_epochs:
            raise RuntimeError("Stage8B requires all Stage8A warmup epochs")
        self._build_optimization("stage8b")
        self.phase = "stage8b"

    def train_stage8b_round(self, round_index: int) -> dict[str, dict[str, Any]]:
        self.begin_stage8b()
        if round_index != self.stage8b_round + 1 or round_index > self.config.stage8.max_rounds:
            raise ValueError("Stage8B rounds must run once in ascending order within the budget")
        result = {
            "phla": self._train_replay("phla", round_index),
            "ptcr": self._train_replay("ptcr", round_index),
            "joint": self._train_joint(phase="stage8b", epoch_or_round=round_index),
        }
        self.stage8b_round = round_index
        return result

    def evaluate(self) -> Stage8Evaluation:
        return evaluate_online_stage8_model(
            self.model,
            self.val_loader,
            device=self.device,
            mixed_precision=self.config.stage8.mixed_precision,
            margin=self.config.stage8.constraint_margin,
            selection_weights=self.config.stage8.selection_weights,
        )

    def fit_stage8b(self) -> Stage8Evaluation:
        self.begin_stage8b()
        if self.best_evaluation is None:
            self.best_evaluation = self.evaluate()
            self.best_joint = self.best_evaluation.joint_score
        for round_index in range(self.stage8b_round + 1, self.config.stage8.max_rounds + 1):
            self.train_stage8b_round(round_index)
            evaluation = self.evaluate()
            if evaluation.joint_score > self.best_joint + self.config.stage8.early_stopping_min_delta:
                self.best_evaluation = evaluation
                self.best_joint = evaluation.joint_score
                self.bad_rounds = 0
            else:
                self.bad_rounds += 1
            if (
                round_index >= self.config.stage8.min_rounds
                and self.bad_rounds >= self.config.stage8.early_stopping_patience
            ):
                break
        if self.best_evaluation is None:  # pragma: no cover - defensive invariant.
            raise RuntimeError("Stage8B did not produce an evaluation")
        return self.best_evaluation

    def training_state(self) -> dict[str, int | float | str]:
        """State Task 3 can serialize without prescribing its checkpoint format."""

        state: dict[str, int | float | str] = {
            "phase": self.phase,
            "stage8a_epoch": self.stage8a_epoch,
            "stage8b_round": self.stage8b_round,
            "global_step": self.global_step,
            "bad_rounds": self.bad_rounds,
            "best_joint": self.best_joint,
        }
        run_id = getattr(self.tracker, "run_id", None)
        if run_id:
            state["swanlab_run_id"] = str(run_id)
        return state
