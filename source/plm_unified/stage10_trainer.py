"""Teacher-student training loop with Stage-10 retention gating."""

from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .stage10_checkpoint import (
    Stage3TeacherSource,
    atomic_torch_save,
    load_torch_checkpoint,
    stage10_teacher_policy,
    stage10_training_contract,
    validate_stage10_resume_checkpoint,
)
from .stage10_config import Stage10Condition, Stage10ExperimentConfig
from .stage10_evaluation import Stage10Evaluation, evaluate_stage10_models
from .stage10_loss import compute_stage10_loss
from .trainer import build_scheduler


def build_stage10_optimizer(
    teacher,
    student,
    config: Stage10ExperimentConfig,
) -> torch.optim.Optimizer:
    """Build the unchanged three Stage-8 groups for the Student only."""

    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    student_ids = {id(parameter) for parameter in student.parameters()}
    if teacher_ids & student_ids:
        raise RuntimeError("Stage-10 teacher/student parameter overlap detected")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Stage-10 teacher parameters must be frozen")

    values = config.stage10
    groups: list[dict[str, Any]] = []

    def add(name: str, modules, learning_rate: float) -> None:
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
        groups.append(
            {"name": name, "params": parameters, "lr": learning_rate}
        )

    add(
        "adapters",
        (student.peptide_adapter, student.hla_adapter, student.tcr_adapter),
        values.adapter_lr,
    )
    add(
        "pretrained_attention",
        (
            student.phla_cross_attention,
            student.ptcr_cross_attention,
            student.phla_pooling,
            student.ptcr_pooling,
            student.phla_classifier,
            student.ptcr_classifier,
        ),
        values.pretrained_attention_lr,
    )
    add(
        "triplet_modules",
        tuple(student.new_triplet_modules().values()),
        values.triplet_module_lr,
    )
    optimized_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    if len(optimized_ids) != len(set(optimized_ids)):
        raise RuntimeError("Stage-10 optimizer contains duplicate parameters")
    trainable_student_ids = {
        id(parameter)
        for parameter in student.parameters()
        if parameter.requires_grad
    }
    if set(optimized_ids) != trainable_student_ids:
        raise RuntimeError(
            "Stage-10 optimizer does not contain every trainable Student "
            "parameter exactly once"
        )
    if set(optimized_ids) & teacher_ids:
        raise RuntimeError("Stage-10 optimizer contains Teacher parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


def candidate_improves(
    evaluation: Stage10Evaluation,
    *,
    best_joint: float,
    min_delta: float,
) -> bool:
    return (
        evaluation.retention_eligible
        and evaluation.joint_score > best_joint + min_delta
    )


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


class Stage10Trainer:
    def __init__(
        self,
        models,
        config: Stage10ExperimentConfig,
        train_loader,
        val_loader,
        *,
        condition: Stage10Condition | str,
        source: Stage3TeacherSource,
        data_audit: dict,
        tracker=None,
        run_timestamp: str | None = None,
    ):
        self.models = models
        self.teacher = models.teacher
        self.student = models.student
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.condition = Stage10Condition(condition)
        self.source = source
        self.data_audit = data_audit
        self.tracker = tracker
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.device = next(self.student.parameters()).device
        if next(self.teacher.parameters()).device != self.device:
            raise RuntimeError("Stage-10 Teacher and Student devices differ")
        self.models.assert_teacher_frozen()
        self._teacher_initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.teacher.state_dict().items()
        }
        self.optimizer = build_stage10_optimizer(
            self.teacher,
            self.student,
            config,
        )
        values = config.stage10
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and values.mixed_precision == "fp16"
            ),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=max(1, values.max_epochs * len(train_loader)),
            warmup_ratio=values.scheduler_warmup_ratio,
        )
        self.global_step = 0
        self.start_epoch = 1
        self.best_epoch = 0
        self.best_joint = float("-inf")
        self.bad_epochs = 0
        self.baseline: Stage10Evaluation | None = None
        self.best_evaluation: Stage10Evaluation | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.teacher_policy = stage10_teacher_policy(config)
        self.training_contract = stage10_training_contract(
            config,
            data_audit.get("effective_data"),
        )
        self.run_dir = config.run_dir(self.condition)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = (
            self.run_dir / f"stage10_best_{self.run_timestamp}.pt"
        )
        self.last_path = (
            self.run_dir / f"stage10_last_{self.run_timestamp}.pt"
        )
        self.prediction_path = self.run_dir / (
            f"validation_predictions_{self.run_timestamp}.csv"
        )
        self.summary_path = self.run_dir / (
            f"summary_{self.run_timestamp}.json"
        )

    def _assert_teacher_unchanged(self) -> None:
        self.models.assert_teacher_frozen()
        current = self.teacher.state_dict()
        if set(current) != set(self._teacher_initial_state):
            raise RuntimeError("Stage-10 Teacher state keys changed")
        for name, tensor in current.items():
            if not torch.equal(
                tensor.detach().cpu(),
                self._teacher_initial_state[name],
            ):
                raise RuntimeError(
                    f"Stage-10 Teacher state changed at {name!r}"
                )

    def _move(self, batch: dict) -> dict:
        return {
            key: (
                value.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.student.train()
        self.teacher.eval()
        self._assert_teacher_unchanged()
        totals: dict[str, float] = {}
        samples = 0
        values = self.config.stage10
        use_autocast = (
            self.device.type == "cuda" and values.mixed_precision != "none"
        )
        autocast_dtype = (
            torch.float16
            if values.mixed_precision == "fp16"
            else torch.bfloat16
        )
        for batch in self.train_loader:
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with context:
                with torch.no_grad():
                    teacher_output = self.teacher.forward_legacy_heads(
                        batch["peptide_hidden"],
                        batch["peptide_mask"],
                        batch["hla_hidden"],
                        batch["hla_mask"],
                        batch["tcr_hidden"],
                        batch["tcr_mask"],
                    )
                self.models.assert_teacher_frozen()
                student_output = self.student.forward_all(
                    batch["peptide_hidden"],
                    batch["peptide_mask"],
                    batch["hla_hidden"],
                    batch["hla_mask"],
                    batch["tcr_hidden"],
                    batch["tcr_mask"],
                )
                loss = compute_stage10_loss(
                    student_output.phla_logits,
                    student_output.ptcr_logits,
                    student_output.pht_logits,
                    teacher_output.phla_logits,
                    teacher_output.ptcr_logits,
                    batch["ph_labels"],
                    batch["pt_labels"],
                    batch["pht_labels"],
                    batch["label_types"],
                    condition=self.condition,
                    epoch=epoch,
                    margin=values.constraint_margin,
                    hierarchy_lambda=values.hierarchy_lambda,
                    pt_implication_lambda=values.pt_implication_lambda,
                    ramp_epochs=values.constraint_ramp_epochs,
                    pht_weight=values.pht_loss_weight,
                    phla_weight=values.phla_loss_weight,
                    ptcr_weight=values.ptcr_loss_weight,
                    keep_loss_weight=values.keep_loss_weight,
                    phla_keep_weight=values.phla_keep_weight,
                    ptcr_keep_weight=values.ptcr_keep_weight,
                    distillation_temperature=(
                        values.distillation_temperature
                    ),
                )
            self.scaler.scale(loss.total).backward()
            self._assert_teacher_unchanged()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                values.gradient_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            self._assert_teacher_unchanged()

            count = int(batch["pht_labels"].shape[0])
            samples += count
            stage8_loss = loss.stage8
            scalars = {
                "total": loss.total,
                "stage8_total": stage8_loss.total,
                "supervised": stage8_loss.supervised,
                "pht": stage8_loss.pht,
                "phla": stage8_loss.phla,
                "ptcr": stage8_loss.ptcr,
                "hierarchy": stage8_loss.hierarchy,
                "pt_implication": stage8_loss.pt_implication,
                "keep": loss.keep,
                "keep_phla": loss.keep_phla,
                "keep_ptcr": loss.keep_ptcr,
            }
            for name, tensor in scalars.items():
                totals[name] = totals.get(name, 0.0) + (
                    float(tensor.detach().cpu()) * count
                )
            if self.tracker and (
                self.global_step == 1
                or self.global_step % values.log_every_steps == 0
            ):
                self.tracker.log(
                    {
                        **{
                            f"train/{name}": total / samples
                            for name, total in totals.items()
                        },
                        "train/gradient_norm": float(gradient_norm),
                        "train/epoch": epoch,
                        "train/hierarchy_lambda": (
                            stage8_loss.hierarchy_lambda
                        ),
                        "train/pt_implication_lambda": (
                            stage8_loss.pt_implication_lambda
                        ),
                        "train/keep_loss_weight": values.keep_loss_weight,
                    },
                    step=self.global_step,
                )
        if not samples:
            raise RuntimeError("Stage-10 training loader is empty")
        return {name: total / samples for name, total in totals.items()}

    def evaluate(self) -> Stage10Evaluation:
        self._assert_teacher_unchanged()
        evaluation = evaluate_stage10_models(
            self.teacher,
            self.student,
            self.val_loader,
            device=self.device,
            mixed_precision=self.config.stage10.mixed_precision,
            margin=self.config.stage10.constraint_margin,
            selection_weights=self.config.stage10.selection_weights,
            degradation_tolerance=(
                self.config.stage10.legacy_task_degradation_tolerance
            ),
        )
        self._assert_teacher_unchanged()
        return evaluation

    @staticmethod
    def _retention_payload(evaluation: Stage10Evaluation) -> dict:
        return {
            "eligible": evaluation.retention_eligible,
            **evaluation.retention,
        }

    def _checkpoint(
        self,
        epoch: int,
        evaluation: Stage10Evaluation,
        phase: str,
        artifact_role: str,
    ) -> dict:
        self._assert_teacher_unchanged()
        return {
            "format_version": 1,
            "checkpoint_type": "stage10_binding_model",
            "condition": self.condition.value,
            "phase": phase,
            "artifact_role": artifact_role,
            "epoch": epoch,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": getattr(self.tracker, "run_id", None),
            "config": self.config.to_dict(),
            "source_stage3": self.source.lineage,
            "teacher_policy": self.teacher_policy,
            "data_audit": self.data_audit,
            "training_contract": self.training_contract,
            "model": self.student.state_dict(),
            "best_model": self.best_model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng_state": _rng_state(),
            "best_epoch": self.best_epoch,
            "best_joint_score": self.best_joint,
            "bad_epochs": self.bad_epochs,
            "baseline_metrics": (
                self.baseline.student_metrics if self.baseline else None
            ),
            "baseline_teacher_metrics": (
                self.baseline.teacher_metrics if self.baseline else None
            ),
            "baseline_retention": (
                self._retention_payload(self.baseline)
                if self.baseline
                else None
            ),
            "validation_metrics": evaluation.student_metrics,
            "validation_teacher_metrics": evaluation.teacher_metrics,
            "validation_retention": self._retention_payload(evaluation),
            "validation_joint_score": evaluation.joint_score,
            "selected_pht_threshold": evaluation.selected_pht_threshold,
        }

    def save(
        self,
        path: Path,
        epoch: int,
        evaluation: Stage10Evaluation,
        phase: str,
        artifact_role: str,
    ) -> None:
        atomic_torch_save(
            self._checkpoint(epoch, evaluation, phase, artifact_role),
            path,
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(checkpoint_path, mmap=False)
        validate_stage10_resume_checkpoint(
            checkpoint,
            self.config,
            self.source.sha256,
            self.data_audit.get("effective_data"),
            condition=self.condition,
        )
        saved_fingerprint = checkpoint.get("data_audit", {}).get(
            "dataset_fingerprint"
        )
        current_fingerprint = self.data_audit.get("dataset_fingerprint")
        if saved_fingerprint != current_fingerprint:
            raise ValueError("Stage-10 resume dataset fingerprint mismatch")
        self.student.load_state_dict(checkpoint["model"], strict=True)
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
            raise ValueError(
                "Stage-10 resume checkpoint has no self-contained best Student"
            )
        self.best_model_state = best_model
        _restore_rng_state(checkpoint.get("rng_state"))
        self._assert_teacher_unchanged()

    def _capture_best_model(self) -> None:
        self.best_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.student.state_dict().items()
        }

    def _write_predictions(self, evaluation: Stage10Evaluation) -> None:
        arrays = evaluation.predictions
        with self.prediction_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_row",
                    "label_type",
                    "PH",
                    "PT",
                    "PHT",
                    "student_prob_PH",
                    "student_prob_PT",
                    "student_prob_PHT",
                    "teacher_prob_PH",
                    "teacher_prob_PT",
                    "PH_probability_drift",
                    "PT_probability_drift",
                ]
            )
            for index in range(len(arrays["pht_labels"])):
                student_ph = float(arrays["phla_prob"][index])
                student_pt = float(arrays["ptcr_prob"][index])
                teacher_ph = float(arrays["teacher_phla_prob"][index])
                teacher_pt = float(arrays["teacher_ptcr_prob"][index])
                writer.writerow(
                    [
                        int(arrays["source_rows"][index]),
                        arrays["label_types"][index],
                        int(arrays["ph_labels"][index]),
                        int(arrays["pt_labels"][index]),
                        int(arrays["pht_labels"][index]),
                        student_ph,
                        student_pt,
                        float(arrays["pht_prob"][index]),
                        teacher_ph,
                        teacher_pt,
                        student_ph - teacher_ph,
                        student_pt - teacher_pt,
                    ]
                )

    def _log_evaluation(
        self,
        epoch: int,
        evaluation: Stage10Evaluation,
    ) -> None:
        if not self.tracker:
            return
        payload = {
            "validation/epoch": epoch,
            "validation/joint_score": evaluation.joint_score,
            "validation/retention_eligible": float(
                evaluation.retention_eligible
            ),
        }
        for task in ("pht", "phla", "ptcr"):
            for name, value in evaluation.student_metrics[task].items():
                payload[f"validation/student/{task}/{name}"] = value
        for task in ("phla", "ptcr"):
            for name, value in evaluation.teacher_metrics[task].items():
                payload[f"validation/teacher/{task}/{name}"] = value
        for name, value in evaluation.student_metrics["violations"].items():
            payload[f"validation/violations/{name}"] = value
        for name, value in evaluation.retention.items():
            payload[f"validation/retention/{name}"] = value
        self.tracker.log(payload, step=self.global_step)

    def fit(self, *, resume: str | Path | None = None) -> dict:
        values = self.config.stage10
        self._assert_teacher_unchanged()
        # A newly built Student is the exact Stage-3 Teacher at epoch zero.
        self.baseline = self.evaluate()
        if not self.baseline.retention_eligible:
            raise RuntimeError(
                "Stage-10 epoch-zero Student does not pass the retention gate"
            )
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
            if candidate_improves(
                evaluation,
                best_joint=self.best_joint,
                min_delta=values.early_stopping_min_delta,
            ):
                self.best_joint = evaluation.joint_score
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.best_evaluation = evaluation
                self._capture_best_model()
                self._write_predictions(evaluation)
                self.save(
                    self.best_path,
                    epoch,
                    evaluation,
                    "training",
                    "best",
                )
            else:
                self.bad_epochs += 1
            self.save(
                self.last_path,
                epoch,
                evaluation,
                "training",
                "last",
            )
            self._log_evaluation(epoch, evaluation)
            if (
                epoch >= values.min_epochs
                and self.bad_epochs >= values.early_stopping_patience
            ):
                break

        self.save(
            self.last_path,
            final_epoch,
            final_evaluation,
            "complete",
            "last",
        )
        if self.best_model_state is None:
            raise RuntimeError("Stage-10 best Student state was not initialized")
        self.student.load_state_dict(self.best_model_state, strict=True)
        best_evaluation = self.evaluate()
        self.best_evaluation = best_evaluation
        self._write_predictions(best_evaluation)
        self.save(
            self.best_path,
            self.best_epoch,
            best_evaluation,
            "complete",
            "best",
        )
        self._assert_teacher_unchanged()

        best_metrics = best_evaluation.student_metrics
        baseline_metrics = self.baseline.student_metrics
        primary_success = (
            best_evaluation.retention_eligible
            and best_evaluation.joint_score
            >= self.baseline.joint_score
            - values.joint_degradation_tolerance
            and best_metrics["violations"]["overall"]
            < baseline_metrics["violations"]["overall"]
        )
        summary = {
            "condition": self.condition.value,
            "source_stage3": self.source.lineage,
            "teacher_policy": self.teacher_policy,
            "dataset_fingerprint": self.data_audit.get(
                "dataset_fingerprint"
            ),
            "baseline_joint_score": self.baseline.joint_score,
            "baseline_metrics": baseline_metrics,
            "baseline_teacher_metrics": self.baseline.teacher_metrics,
            "baseline_retention": self._retention_payload(self.baseline),
            "best_epoch": self.best_epoch,
            "best_joint_score": best_evaluation.joint_score,
            "best_metrics": best_metrics,
            "best_teacher_metrics": best_evaluation.teacher_metrics,
            "best_retention": self._retention_payload(best_evaluation),
            "joint_degradation_tolerance": (
                values.joint_degradation_tolerance
            ),
            "legacy_task_degradation_tolerance": (
                values.legacy_task_degradation_tolerance
            ),
            "primary_success": primary_success,
            "best_checkpoint": str(self.best_path),
            "last_checkpoint": str(self.last_path),
            "validation_predictions": str(self.prediction_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary
