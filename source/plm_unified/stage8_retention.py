"""Opt-in PT retention experiment; the archived Stage-8 trainer is unchanged.

Uses frozen pretrained encoders, a protected ternary update and a Stage-1 PT
teacher on original binary replay. PT labels from PHT remain diagnostics in
the protected joint pass: they do not update the pair representation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_binary_metrics
from .stage8_evaluation import stage8_task_score
from .stage8_model import ProgressiveStage8FineTuningModel
from .stage8_trainer import (
    Stage8OnlineTrainer, _iter_capped_online_batches, build_stage8_online_optimizer,
)
from .trainer import build_scheduler


@dataclass(frozen=True)
class PTRetentionPolicy:
    rounds: int = 10
    ph_replay_ratio: float = 0.25
    pt_replay_ratio: float = 1.0
    distillation_weight: float = 0.5
    temperature: float = 2.0
    label_smoothing: float = 0.03
    macro_tolerance: float = 0.01
    split_tolerance: float = 0.02
    pht_tolerance: float = 0.02

    def __post_init__(self):
        if not 1 <= self.rounds <= 100:
            raise ValueError("rounds must be between 1 and 100")
        if not all(math.isfinite(v) and v > 0 for v in (
            self.ph_replay_ratio, self.pt_replay_ratio, self.temperature,
        )):
            raise ValueError("Replay ratios and temperature must be finite and positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not all(math.isfinite(v) and v >= 0 for v in (
            self.distillation_weight, self.macro_tolerance,
            self.split_tolerance, self.pht_tolerance,
        )):
            raise ValueError("Weights and tolerances must be finite and nonnegative")


class ProtectedStage8Model(ProgressiveStage8FineTuningModel):
    def set_trainable_stage8(self, task, phase):
        super().set_trainable_stage8(task, phase)
        # Protect the complete encoder, including dropout state, on every pass.
        for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
            encoder.requires_grad_(False)
            encoder.set_active_training(False)
            encoder.eval()
        if task == "joint":
            self.binding_model.requires_grad_(False)
            self.binding_model.eval()
            for module in self.binding_model.new_triplet_modules().values():
                module.requires_grad_(True)
                module.train()
        elif task == "phla":
            # PH replay must not move the representation used by the PT head.
            self.binding_model.peptide_encoder.requires_grad_(False)
            self.binding_model.peptide_encoder.eval()


def distillation_loss(student_logits, teacher_logits, temperature):
    """KL(teacher || student), in FP32; no gradient enters the teacher."""
    return temperature ** 2 * F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=-1),
        F.softmax(teacher_logits.detach().float() / temperature, dim=-1),
        reduction="batchmean",
    )


def peptide_weights(rows):
    """Global inverse-sqrt frequencies, clipped like the saved Stage-1 run."""
    counts = Counter(p for p, _, _ in rows)
    raw = {p: n ** -0.5 for p, n in counts.items()}
    mean = sum(counts[p] * v for p, v in raw.items()) / len(rows)
    clipped = {p: min(4.0, max(0.25, v / mean)) for p, v in raw.items()}
    normalizer = sum(counts[p] * v for p, v in clipped.items()) / len(rows)
    return {p: v / normalizer for p, v in clipped.items()}


def retention_failures(candidate, reference, pht_score, initial_pht_score, policy):
    """Independent PT macro and per-split guard, plus PHT retention."""
    if not reference or set(candidate) != set(reference):
        raise ValueError("PT candidate/reference must contain the same nonempty splits")
    failures = []
    for metric in ("auroc", "aupr", "mcc"):
        current = [float(candidate[s][metric]) for s in reference]
        original = [float(reference[s][metric]) for s in reference]
        if not all(math.isfinite(v) for v in current + original):
            failures.append(f"nonfinite:{metric}")
            continue
        if np.mean(current) < np.mean(original) - policy.macro_tolerance:
            failures.append(f"macro:{metric}")
        for split, value, base in zip(reference, current, original):
            if value < base - policy.split_tolerance:
                failures.append(f"{split}:{metric}")
    if not math.isfinite(pht_score) or not math.isfinite(initial_pht_score):
        failures.append("nonfinite:pht")
    elif pht_score < initial_pht_score - policy.pht_tolerance:
        failures.append("pht:performance_avg")
    return failures


class PTRetentionTrainer(Stage8OnlineTrainer):
    def __init__(self, *args, teacher, pt_val_loaders, policy, **kwargs):
        self.policy = policy
        self.teacher = teacher.eval().requires_grad_(False)
        self.pt_val_loaders = pt_val_loaders
        super().__init__(*args, **kwargs)
        self.pt_weights = peptide_weights(self.train_loaders["ptcr"].dataset.rows)

    def _build_optimization(self, phase):
        self.model.set_trainable_stage8("joint", phase)
        if phase == "stage8b":
            # Register the union of all pass parameters; the protected joint
            # pass alone intentionally exposes only the ternary modules.
            self.model.binding_model.requires_grad_(True)
        self.optimizer = build_stage8_online_optimizer(self.model, self.config, phase)
        self.model.set_trainable_stage8("joint", phase)
        self.scheduler = build_scheduler(
            self.optimizer, total_steps=self._planned_optimizer_steps(phase),
            warmup_ratio=self.config.stage8.scheduler_warmup_ratio,
        )

    def _replay_samples(self, task):
        ratio = self.policy.pt_replay_ratio if task == "ptcr" else self.policy.ph_replay_ratio
        return min(len(self.train_loaders[task].dataset), math.floor(self._joint_samples() * ratio))

    def _planned_optimizer_steps(self, phase):
        total = super()._planned_optimizer_steps(phase)
        if phase == "stage8b":
            return total // self.config.stage8.max_rounds * self.policy.rounds
        return total

    def _train_joint(self, *, phase, epoch_or_round):
        sampler = self.train_loaders["joint"].batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_or_round)
        return super()._train_joint(phase=phase, epoch_or_round=epoch_or_round)

    def _pair_outputs(self, batch, task, *, with_teacher=False):
        # Both heads see the same original, frozen-PLM features. Reusing
        # fine-tuned Stage-8 features for a Stage-1 teacher would be invalid.
        with torch.no_grad():
            peptide = self.model.peptide_encoder(batch["peptide_sequences"])
            encoder = self.model.tcr_encoder if task == "ptcr" else self.model.hla_encoder
            receptor = encoder(batch["receptor_sequences"])
        args = (task, peptide.hidden, peptide.mask, receptor.hidden, receptor.mask)
        logits = self.model.binding_model(*args).logits
        teacher_logits = None
        if with_teacher:
            with torch.no_grad():
                teacher_logits = self.teacher(*args).logits
        return logits, teacher_logits

    def _train_replay(self, task, round_index):
        if task != "ptcr":
            return super()._train_replay(task, round_index)
        self.model.set_trainable_stage8(task, "stage8b")
        self.pass_history.append(("stage8b", round_index, task))
        loader = self.train_loaders[task]
        accumulation = self.config.stage8.gradient_accumulation_steps
        total_batches = self._replay_batches(task)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        samples = steps = 0
        ce_sum = kd_sum = 0.0
        cap = self._replay_samples(task)
        batch_size = loader.batch_size
        for index, batch in enumerate(_iter_capped_online_batches(loader, cap)):
            start = index // accumulation * accumulation
            if index == start:
                self.optimizer.zero_grad(set_to_none=True)
            # Sample weighting also handles the last short microbatch.
            window_samples = min(cap - start * batch_size, accumulation * batch_size)
            labels = batch["labels"].to(self.device)
            count = len(labels)
            weights = torch.tensor([self.pt_weights[p] for p in batch["peptide_sequences"]], device=self.device)
            with self._autocast():
                logits, teacher = self._pair_outputs(batch, task, with_teacher=True)
                ce = (F.cross_entropy(logits.float(), labels, reduction="none", label_smoothing=self.policy.label_smoothing) * weights).mean()
                kd = distillation_loss(logits, teacher, self.policy.temperature)
                loss = ce + self.policy.distillation_weight * kd
            self.scaler.scale(loss * count / window_samples).backward()
            samples += count
            ce_sum += float(ce.detach()) * count
            kd_sum += float(kd.detach()) * count
            if (index + 1) % accumulation == 0 or index + 1 == total_batches:
                self._optimizer_step(trainable)
                steps += 1
        if not samples:
            raise ValueError("PT replay is empty")
        return {"samples": samples, "optimizer_steps": steps,
                "ce": ce_sum / samples, "kd": kd_sum / samples}

    def train_stage8b_round(self, round_index):
        self.begin_stage8b()
        if round_index != self.stage8b_round + 1 or round_index > self.policy.rounds:
            raise ValueError("Retention rounds must be sequential and within budget")
        result = {
            "phla": self._train_replay("phla", round_index),
            "joint": self._train_joint(phase="stage8b", epoch_or_round=round_index),
            "ptcr": self._train_replay("ptcr", round_index),
        }
        self.stage8b_round = round_index
        return result

    @torch.inference_mode()
    def evaluate_pt(self):
        self.model.set_evaluation()
        self.teacher.eval()
        student, teacher = {}, {}
        for split, loader in self.pt_val_loaders.items():
            labels, scores, teacher_scores = [], [], []
            for batch in loader:
                with self._autocast():
                    logits, teacher_logits = self._pair_outputs(batch, "ptcr", with_teacher=True)
                labels.extend(batch["labels"].tolist())
                scores.extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
                teacher_scores.extend(teacher_logits.float().softmax(-1)[:, 1].cpu().tolist())
            for destination, probabilities in ((student, scores), (teacher, teacher_scores)):
                destination[split] = compute_binary_metrics(
                    np.asarray(labels), np.asarray(probabilities), loss=0.0,
                    strict_threshold=True,
                ).to_dict()
                # No separate CE accumulator in this evaluation path.
                destination[split]["loss"] = destination[split]["nll"]
        return student, teacher

    @staticmethod
    def selection_score(pt_metrics, pht_metrics):
        pt_score = np.mean([stage8_task_score(m) for m in pt_metrics.values()])
        return float(0.6 * pt_score + 0.3 * stage8_task_score(pht_metrics["pht"])
                     + 0.1 * stage8_task_score(pht_metrics["phla"]))
