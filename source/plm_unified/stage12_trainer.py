"""Online Stage12 lifecycle aligned with the Stage8 training schedule."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from .stage8_loss import Stage8Loss, compute_stage8_loss
from .stage8_trainer import (
    _iter_capped_online_batches,
    build_stage8_online_optimizer,
)
from .stage12_config import Stage12ExperimentConfig
from .stage12_data import replay_sample_count
from .stage12_evaluation import Stage12ValidationScores
from .trainer import build_scheduler


Stage12Phase = Literal["stage12a", "stage12b"]


def stage12_pass_order() -> tuple[str, str, str]:
    return "phla", "ptcr", "joint"


def stage12_replay_samples(
    task: Literal["phla", "ptcr"],
    *,
    joint_samples: int,
    task_samples: int,
    config: Stage12ExperimentConfig,
) -> int:
    ratio = (
        config.stage12.phla_replay_ratio
        if task == "phla" else config.stage12.ptcr_replay_ratio
    )
    return replay_sample_count(joint_samples, task_samples, ratio)


def compute_stage12_joint_loss(
    phla_logits: torch.Tensor,
    ptcr_logits: torch.Tensor,
    pht_logits: torch.Tensor,
    ph_labels: torch.Tensor,
    pt_labels: torch.Tensor,
    pht_labels: torch.Tensor,
    label_types: Sequence[str],
    *,
    config: Stage12ExperimentConfig,
    phase: Stage12Phase,
    round_or_epoch: int,
) -> Stage8Loss:
    if phase not in {"stage12a", "stage12b"}:
        raise ValueError(f"Unknown Stage12 phase: {phase}")
    values = config.stage12
    warmup = phase == "stage12a"
    return compute_stage8_loss(
        phla_logits,
        ptcr_logits,
        pht_logits,
        ph_labels,
        pt_labels,
        pht_labels,
        label_types,
        condition="full",
        epoch=round_or_epoch,
        phase="stage8a" if warmup else "stage8b",
        round_index=None if warmup else round_or_epoch,
        margin=values.constraint_margin,
        hierarchy_lambda=values.hierarchy_lambda,
        pt_implication_lambda=values.pt_implication_lambda,
        ramp_epochs=values.constraint_ramp_rounds,
        pht_weight=1.0 if warmup else values.pht_loss_weight,
        phla_weight=0.0 if warmup else values.phla_loss_weight,
        ptcr_weight=0.0 if warmup else values.ptcr_loss_weight,
    )


class Stage12OnlineTrainer:
    """Two-phase online trainer retaining the exact Stage8 model topology."""

    def __init__(
        self,
        model,
        config: Stage12ExperimentConfig,
        train_loaders: Mapping[str, Any],
        *,
        validation_fn: Callable[[Any], Stage12ValidationScores],
        tracker=None,
    ):
        config.validate()
        if set(train_loaders) != {"phla", "ptcr", "joint"}:
            raise ValueError("Stage12 training requires phla, ptcr, and joint loaders")
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.validation_fn = validation_fn
        self.tracker = tracker
        self.device = next(model.parameters()).device
        self.phase: Stage12Phase = "stage12a"
        self.stage12a_epoch = 0
        self.stage12b_round = 0
        self.global_step = 0
        self.bad_rounds = 0
        self.best_joint_score_seen = -float("inf")
        self.selection_state: dict[str, Any] = {}
        self.pass_history: list[tuple[str, int, str]] = []
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.stage12.mixed_precision == "fp16"
            ),
        )
        self._build_optimization("stage12a")

    def _autocast(self):
        if self.device.type != "cuda" or self.config.stage12.mixed_precision == "none":
            return nullcontext()
        dtype = (
            torch.float16
            if self.config.stage12.mixed_precision == "fp16"
            else torch.bfloat16
        )
        return torch.autocast("cuda", dtype=dtype)

    def _joint_samples(self) -> int:
        dataset = getattr(self.train_loaders["joint"], "dataset", None)
        if dataset is None:
            raise ValueError("Stage12 joint loader must expose its dataset")
        return len(dataset)

    def _replay_samples(self, task: Literal["phla", "ptcr"]) -> int:
        dataset = getattr(self.train_loaders[task], "dataset", None)
        if dataset is None:
            raise ValueError(f"Stage12 {task} loader must expose its dataset")
        return stage12_replay_samples(
            task,
            joint_samples=self._joint_samples(),
            task_samples=len(dataset),
            config=self.config,
        )

    def _replay_batches(self, task: Literal["phla", "ptcr"]) -> int:
        samples = self._replay_samples(task)
        if samples == 0:
            return 0
        loader = self.train_loaders[task]
        batch_size = getattr(loader, "batch_size", None) or self.config.stage12.micro_batch_size
        return min(len(loader), math.ceil(samples / batch_size))

    def _planned_optimizer_steps(self, phase: Stage12Phase) -> int:
        accumulation = self.config.stage12.gradient_accumulation_steps
        joint_steps = math.ceil(len(self.train_loaders["joint"]) / accumulation)
        if phase == "stage12a":
            return max(1, joint_steps * self.config.stage12.stage12a_epochs)
        replay_steps = sum(
            math.ceil(self._replay_batches(task) / accumulation)
            for task in ("phla", "ptcr")
        )
        return max(
            1,
            (joint_steps + replay_steps) * self.config.stage12.max_rounds,
        )

    def _build_optimization(self, phase: Stage12Phase) -> None:
        model_phase = "stage8a" if phase == "stage12a" else "stage8b"
        self.model.set_trainable_stage8("joint", model_phase)
        self.optimizer = build_stage8_online_optimizer(
            self.model, self.config, model_phase
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self._planned_optimizer_steps(phase),
            warmup_ratio=self.config.stage12.scheduler_warmup_ratio,
        )

    def _optimizer_step(self, trainable: Sequence[torch.nn.Parameter]) -> float:
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, self.config.stage12.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return float(norm.detach().cpu())

    def _train_joint(
        self, *, phase: Stage12Phase, round_or_epoch: int
    ) -> dict[str, float | int]:
        model_phase = "stage8a" if phase == "stage12a" else "stage8b"
        self.model.set_trainable_stage8("joint", model_phase)
        self.pass_history.append((phase, round_or_epoch, "joint"))
        loader = self.train_loaders["joint"]
        total_batches = len(loader)
        accumulation = self.config.stage12.gradient_accumulation_steps
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        totals = {name: 0.0 for name in (
            "total", "supervised", "pht", "phla", "ptcr", "hierarchy", "pt_implication",
        )}
        samples = 0
        optimizer_steps = 0
        hierarchy_lambda = 0.0
        pt_implication_lambda = 0.0
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
                    batch["peptide_sequences"],
                    batch["hla_sequences"],
                    batch["tcr_sequences"],
                )
                loss = compute_stage12_joint_loss(
                    output.phla_logits,
                    output.ptcr_logits,
                    output.pht_logits,
                    labels["ph_labels"],
                    labels["pt_labels"],
                    labels["pht_labels"],
                    batch["label_types"],
                    config=self.config,
                    phase=phase,
                    round_or_epoch=round_or_epoch,
                )
            self.scaler.scale(loss.total / window_size).backward()
            count = int(labels["pht_labels"].shape[0])
            samples += count
            for name, value in (
                ("total", loss.total), ("supervised", loss.supervised),
                ("pht", loss.pht), ("phla", loss.phla), ("ptcr", loss.ptcr),
                ("hierarchy", loss.hierarchy),
                ("pt_implication", loss.pt_implication),
            ):
                totals[name] += float(value.detach().cpu()) * count
            hierarchy_lambda = loss.hierarchy_lambda
            pt_implication_lambda = loss.pt_implication_lambda
            if batch_index - window_start + 1 == window_size:
                self._optimizer_step(trainable)
                optimizer_steps += 1
        if not samples:
            raise RuntimeError("Stage12 joint loader is empty")
        return {
            **{name: value / samples for name, value in totals.items()},
            "samples": samples,
            "optimizer_steps": optimizer_steps,
            "hierarchy_lambda": hierarchy_lambda,
            "pt_implication_lambda": pt_implication_lambda,
        }

    def _train_replay(
        self, task: Literal["phla", "ptcr"], round_index: int
    ) -> dict[str, float | int]:
        self.model.set_trainable_stage8(task, "stage8b")
        self.pass_history.append(("stage12b", round_index, task))
        max_samples = self._replay_samples(task)
        if max_samples == 0:
            return {"loss": 0.0, "samples": 0, "optimizer_steps": 0}
        total_batches = self._replay_batches(task)
        accumulation = self.config.stage12.gradient_accumulation_steps
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        samples = 0
        loss_sum = 0.0
        optimizer_steps = 0
        for batch_index, batch in enumerate(
            _iter_capped_online_batches(self.train_loaders[task], max_samples), start=1
        ):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            with self._autocast():
                output = self.model(
                    task, batch["peptide_sequences"], batch["receptor_sequences"]
                )
                loss = F.cross_entropy(output.logits, labels)
            self.scaler.scale(loss / window_size).backward()
            count = int(labels.shape[0])
            samples += count
            loss_sum += float(loss.detach().cpu()) * count
            if batch_index - window_start + 1 == window_size:
                self._optimizer_step(trainable)
                optimizer_steps += 1
        if not samples:
            raise RuntimeError(f"Stage12 {task} replay loader is empty")
        return {
            "loss": loss_sum / samples,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
        }

    def train_stage12a_epoch(self, epoch: int) -> dict[str, float | int]:
        if self.phase != "stage12a":
            raise RuntimeError("Stage12A has already completed")
        if epoch != self.stage12a_epoch + 1 or epoch > self.config.stage12.stage12a_epochs:
            raise ValueError("Stage12A epochs must run once in ascending order")
        result = self._train_joint(phase="stage12a", round_or_epoch=epoch)
        self.stage12a_epoch = epoch
        return result

    def begin_stage12b(self) -> None:
        if self.phase == "stage12b":
            return
        if self.stage12a_epoch != self.config.stage12.stage12a_epochs:
            raise RuntimeError("Stage12B requires the complete Stage12A warmup")
        self._build_optimization("stage12b")
        self.phase = "stage12b"

    def train_stage12b_round(
        self, round_index: int
    ) -> dict[str, dict[str, float | int]]:
        self.begin_stage12b()
        if round_index != self.stage12b_round + 1 or round_index > self.config.stage12.max_rounds:
            raise ValueError("Stage12B rounds must run once in ascending order")
        joint_sampler = getattr(self.train_loaders["joint"], "batch_sampler", None)
        if hasattr(joint_sampler, "set_epoch"):
            joint_sampler.set_epoch(round_index)
        result = {
            "phla": self._train_replay("phla", round_index),
            "ptcr": self._train_replay("ptcr", round_index),
            "joint": self._train_joint(
                phase="stage12b", round_or_epoch=round_index
            ),
        }
        self.stage12b_round = round_index
        return result

    def evaluate(self) -> Stage12ValidationScores:
        return self.validation_fn(self.model)

    def training_state(self) -> dict[str, int | float | str | None]:
        state: dict[str, int | float | str | None] = {
            "phase": self.phase,
            "stage12a_epoch": self.stage12a_epoch,
            "stage12b_round": self.stage12b_round,
            "global_step": self.global_step,
            "bad_rounds": self.bad_rounds,
            "best_joint_score_seen": self.best_joint_score_seen,
            "selection_state": self.selection_state,
        }
        run_id = getattr(self.tracker, "run_id", None)
        if run_id:
            state["swanlab_run_id"] = str(run_id)
        return state
