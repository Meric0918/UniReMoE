"""Two-phase PHT-only training for Stage PHT."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import torch

from .metrics import compute_binary_metrics
from .stage_adapter_evaluation import evaluate_stage_adapter as evaluate_stage_PHT
from .stage_adapter_loss import balanced_pht_cross_entropy
from .trainer import build_scheduler
from .stage_PHT_model import SEQUENCE_ENCODER_NAMES, INTERACTION_MODULE_NAMES

StagePHTPhase = Literal["warmup", "training"]


def build_stage_PHT_optimizer(model, config) -> torch.optim.Optimizer:
    model.set_trainable_pht()
    binding = model.binding_model
    values = config.stage_PHT
    groups = [
        {
            "name": "sequence_encoders",
            "params": [p for name in SEQUENCE_ENCODER_NAMES
                       for p in getattr(binding, name).parameters()],
            "lr": values.encoder_lr,
        },
        {
            "name": "pht_interaction_modules",
            "params": [p for name in INTERACTION_MODULE_NAMES
                       for p in getattr(binding, name).parameters()],
            "lr": values.pht_module_lr,
        },
    ]
    ids = [id(p) for group in groups for p in group["params"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Stage PHT optimizer contains duplicate parameters")
    if set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Stage PHT optimizer does not exactly cover trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


class StagePHTTrainer:
    def __init__(self, model, config, train_loader, val_loader, *, tracker=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tracker = tracker
        self.device = next(model.parameters()).device
        self.phase: StagePHTPhase = "warmup"
        self.warmup_epoch = 0
        self.training_round = 0
        self.global_step = 0
        self.bad_rounds = 0
        self.best_score = -float("inf")
        self.best_evaluation = None
        self.optimizer = build_stage_PHT_optimizer(model, config)
        self.scheduler = self._build_scheduler("warmup")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.stage_PHT.mixed_precision == "fp16"
            ),
        )

    def _planned_steps(self, phase: StagePHTPhase) -> int:
        per_pass = math.ceil(
            len(self.train_loader)
            / self.config.stage_PHT.gradient_accumulation_steps
        )
        count = (
            self.config.stage_PHT.warmup_epochs
            if phase == "warmup"
            else self.config.stage_PHT.max_rounds
        )
        return max(1, per_pass * count)

    def _build_scheduler(self, phase: StagePHTPhase):
        return build_scheduler(
            self.optimizer,
            total_steps=self._planned_steps(phase),
            warmup_ratio=self.config.stage_PHT.scheduler_warmup_ratio,
        )

    def _autocast(self):
        precision = self.config.stage_PHT.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)

    def _set_sampler_epoch(self, value: int) -> None:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(value)

    def train_pass(self, *, epoch_or_round: int) -> dict[str, Any]:
        self.model.set_trainable_pht()
        self._set_sampler_epoch(epoch_or_round)
        accumulation = self.config.stage_PHT.gradient_accumulation_steps
        total_batches = len(self.train_loader)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        labels_seen: list[np.ndarray] = []
        probabilities_seen: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        for batch_index, batch in enumerate(self.train_loader, start=1):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)
            labels = batch["pht_labels"].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            with self._autocast():
                output = self.model.forward_pht(
                    batch["peptide_sequences"],
                    batch["hla_sequences"],
                    batch["tcr_sequences"],
                )
                loss = balanced_pht_cross_entropy(
                    output.logits, labels, batch["label_types"]
                )
            self.scaler.scale(loss / window_size).backward()
            count = len(labels)
            samples += count
            loss_sum += float(loss.detach().cpu()) * count
            labels_seen.append(labels.detach().cpu().numpy())
            probabilities_seen.append(
                output.logits.detach().softmax(-1)[:, 1].float().cpu().numpy()
            )
            if batch_index - window_start + 1 == window_size:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    trainable, self.config.stage_PHT.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(norm.detach().cpu()))
        if not samples:
            raise RuntimeError("Stage PHT training loader is empty")
        average_loss = loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels_seen),
            np.concatenate(probabilities_seen),
            loss=average_loss,
            threshold=0.5,
        )
        return {
            "loss": average_loss,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
            "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "metrics": metrics.to_dict(),
        }

    def train_warmup_epoch(self, epoch: int) -> dict[str, Any]:
        if self.phase != "warmup":
            raise RuntimeError("Stage PHT warmup is complete")
        if epoch != self.warmup_epoch + 1:
            raise ValueError("Stage PHT warmup epochs must be consecutive")
        if epoch > self.config.stage_PHT.warmup_epochs:
            raise ValueError("Stage PHT warmup epoch budget exceeded")
        result = self.train_pass(epoch_or_round=epoch)
        self.warmup_epoch = epoch
        return result

    def begin_training(self) -> None:
        if self.phase == "training":
            return
        if self.warmup_epoch != self.config.stage_PHT.warmup_epochs:
            raise RuntimeError("Stage PHT training requires a complete warmup")
        self.optimizer = build_stage_PHT_optimizer(self.model, self.config)
        self.scheduler = self._build_scheduler("training")
        self.phase = "training"

    def train_round(self, round_index: int) -> dict[str, Any]:
        self.begin_training()
        if round_index != self.training_round + 1:
            raise ValueError("Stage PHT training rounds must be consecutive")
        if round_index > self.config.stage_PHT.max_rounds:
            raise ValueError("Stage PHT training round budget exceeded")
        result = self.train_pass(
            epoch_or_round=self.config.stage_PHT.warmup_epochs + round_index
        )
        self.training_round = round_index
        return result

    def evaluate(self):
        return evaluate_stage_PHT(
            self.model,
            self.val_loader,
            device=self.device,
            mixed_precision=self.config.stage_PHT.mixed_precision,
        )

    def training_state(self) -> dict[str, Any]:
        state = {
            "phase": self.phase,
            "warmup_epoch": self.warmup_epoch,
            "training_round": self.training_round,
            "global_step": self.global_step,
            "bad_rounds": self.bad_rounds,
            "best_score": self.best_score,
        }
        run_id = getattr(self.tracker, "run_id", None)
        if run_id:
            state["swanlab_run_id"] = str(run_id)
        return state
