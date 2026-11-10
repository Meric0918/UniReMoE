"""Two-phase PHT-only training for Stage Adapter."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import torch

from .metrics import compute_binary_metrics
from .stage_adapter_evaluation import evaluate_stage_adapter
from .stage_adapter_loss import balanced_pht_cross_entropy
from .trainer import build_scheduler

StageAdapterPhase = Literal["stage_adapter_a", "stage_adapter_b"]


def build_stage_adapter_optimizer(model, config) -> torch.optim.Optimizer:
    model.set_trainable_pht()
    values = config.stage_adapter
    binding = model.binding_model
    adapter_modules = (
        binding.pht_peptide_adapter,
        binding.pht_hla_adapter,
        binding.pht_tcr_adapter,
    )
    pht_modules = (
        binding.pht_phla_cross_attention,
        binding.pht_cross_attention,
        binding.pht_pooling,
        binding.pht_classifier,
    )
    groups = [
        {
            "name": "pht_residual_adapters",
            "params": [p for module in adapter_modules for p in module.parameters()],
            "lr": values.adapter_lr,
        },
        {
            "name": "pht_interaction_modules",
            "params": [p for module in pht_modules for p in module.parameters()],
            "lr": values.pht_module_lr,
        },
    ]
    if values.initialization != "checkpoint":
        groups.append({
            "name": "shared_adapters",
            "params": [p for name in binding.SHARED_ADAPTER_NAMES
                       for p in getattr(binding, name).parameters()],
            "lr": values.adapter_lr,
        })
    if values.initialization == "random":
        groups.append({
            "name": "encoders",
            "params": [p for encoder in (model.peptide_encoder, model.hla_encoder, model.tcr_encoder)
                       for p in encoder.parameters() if p.requires_grad],
            "lr": values.encoder_lr,
        })
    parameter_ids = [id(p) for group in groups for p in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Stage Adapter optimizer contains duplicate parameters")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if set(parameter_ids) != expected:
        raise RuntimeError("Stage Adapter optimizer does not exactly cover trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


class StageAdapterTrainer:
    def __init__(self, model, config, train_loader, val_loader, *, tracker=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tracker = tracker
        self.device = next(model.parameters()).device
        self.phase: StageAdapterPhase = "stage_adapter_a"
        self.stage_adapter_a_epoch = 0
        self.stage_adapter_b_round = 0
        self.global_step = 0
        self.bad_rounds = 0
        self.best_score = -float("inf")
        self.best_evaluation = None
        self.optimizer = build_stage_adapter_optimizer(model, config)
        self.scheduler = self._build_scheduler("stage_adapter_a")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.stage_adapter.mixed_precision == "fp16"
            ),
        )

    def _planned_steps(self, phase: StageAdapterPhase) -> int:
        per_pass = math.ceil(
            len(self.train_loader)
            / self.config.stage_adapter.gradient_accumulation_steps
        )
        count = (
            self.config.stage_adapter.stage_adapter_a_epochs
            if phase == "stage_adapter_a"
            else self.config.stage_adapter.max_rounds
        )
        return max(1, per_pass * count)

    def _build_scheduler(self, phase: StageAdapterPhase):
        return build_scheduler(
            self.optimizer,
            total_steps=self._planned_steps(phase),
            warmup_ratio=self.config.stage_adapter.scheduler_warmup_ratio,
        )

    def _autocast(self):
        precision = self.config.stage_adapter.mixed_precision
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
        accumulation = self.config.stage_adapter.gradient_accumulation_steps
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
                    trainable, self.config.stage_adapter.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(norm.detach().cpu()))
        if not samples:
            raise RuntimeError("Stage Adapter training loader is empty")
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

    def train_stage_adapter_a_epoch(self, epoch: int) -> dict[str, Any]:
        if self.phase != "stage_adapter_a":
            raise RuntimeError("StageAdapterA is complete")
        if epoch != self.stage_adapter_a_epoch + 1:
            raise ValueError("StageAdapterA epochs must be consecutive")
        if epoch > self.config.stage_adapter.stage_adapter_a_epochs:
            raise ValueError("StageAdapterA epoch budget exceeded")
        result = self.train_pass(epoch_or_round=epoch)
        self.stage_adapter_a_epoch = epoch
        return result

    def begin_stage_adapter_b(self) -> None:
        if self.phase == "stage_adapter_b":
            return
        if self.stage_adapter_a_epoch != self.config.stage_adapter.stage_adapter_a_epochs:
            raise RuntimeError("StageAdapterB requires a complete warmup")
        self.optimizer = build_stage_adapter_optimizer(self.model, self.config)
        self.scheduler = self._build_scheduler("stage_adapter_b")
        self.phase = "stage_adapter_b"

    def train_stage_adapter_b_round(self, round_index: int) -> dict[str, Any]:
        self.begin_stage_adapter_b()
        if round_index != self.stage_adapter_b_round + 1:
            raise ValueError("StageAdapterB rounds must be consecutive")
        if round_index > self.config.stage_adapter.max_rounds:
            raise ValueError("StageAdapterB round budget exceeded")
        result = self.train_pass(
            epoch_or_round=self.config.stage_adapter.stage_adapter_a_epochs + round_index
        )
        self.stage_adapter_b_round = round_index
        return result

    def evaluate(self):
        return evaluate_stage_adapter(
            self.model,
            self.val_loader,
            device=self.device,
            mixed_precision=self.config.stage_adapter.mixed_precision,
        )

    def training_state(self) -> dict[str, Any]:
        state = {
            "phase": self.phase,
            "stage_adapter_a_epoch": self.stage_adapter_a_epoch,
            "stage_adapter_b_round": self.stage_adapter_b_round,
            "global_step": self.global_step,
            "bad_rounds": self.bad_rounds,
            "best_score": self.best_score,
        }
        run_id = getattr(self.tracker, "run_id", None)
        if run_id:
            state["swanlab_run_id"] = str(run_id)
        return state
