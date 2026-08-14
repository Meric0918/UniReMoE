"""Stage-8 objectives plus teacher retention for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .stage8_config import Stage8Condition
from .stage8_loss import Stage8Loss, compute_stage8_loss
from .stage10_config import Stage10Condition


@dataclass
class Stage10Loss:
    total: torch.Tensor
    stage8: Stage8Loss
    keep: torch.Tensor
    keep_phla: torch.Tensor
    keep_ptcr: torch.Tensor
    keep_loss_weight: float


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student_log_probability = F.log_softmax(
        student_logits.float() / temperature,
        dim=-1,
    )
    teacher_probability = F.softmax(
        teacher_logits.detach().float() / temperature,
        dim=-1,
    )
    return F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * temperature**2


def compute_stage10_loss(
    phla_logits: torch.Tensor,
    ptcr_logits: torch.Tensor,
    pht_logits: torch.Tensor,
    teacher_phla_logits: torch.Tensor,
    teacher_ptcr_logits: torch.Tensor,
    ph_labels: torch.Tensor,
    pt_labels: torch.Tensor,
    pht_labels: torch.Tensor,
    label_types: Sequence[str],
    *,
    condition: Stage10Condition | str,
    epoch: int,
    margin: float = 0.05,
    hierarchy_lambda: float = 0.05,
    pt_implication_lambda: float = 0.03,
    ramp_epochs: int = 5,
    pht_weight: float = 1.0,
    phla_weight: float = 0.3,
    ptcr_weight: float = 0.2,
    supervised_weight: float = 1.0,
    keep_loss_weight: float = 1.0,
    phla_keep_weight: float = 1.0,
    ptcr_keep_weight: float = 1.0,
    distillation_temperature: float = 1.0,
) -> Stage10Loss:
    for name, value in (
        ("keep_loss_weight", keep_loss_weight),
        ("phla_keep_weight", phla_keep_weight),
        ("ptcr_keep_weight", ptcr_keep_weight),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    condition = Stage10Condition(condition)
    stage8 = compute_stage8_loss(
        phla_logits,
        ptcr_logits,
        pht_logits,
        ph_labels,
        pt_labels,
        pht_labels,
        label_types,
        condition=Stage8Condition(condition.value),
        epoch=epoch,
        margin=margin,
        hierarchy_lambda=hierarchy_lambda,
        pt_implication_lambda=pt_implication_lambda,
        ramp_epochs=ramp_epochs,
        pht_weight=pht_weight,
        phla_weight=phla_weight,
        ptcr_weight=ptcr_weight,
        supervised_weight=supervised_weight,
    )
    keep_phla = distillation_kl(
        phla_logits,
        teacher_phla_logits,
        temperature=distillation_temperature,
    )
    keep_ptcr = distillation_kl(
        ptcr_logits,
        teacher_ptcr_logits,
        temperature=distillation_temperature,
    )
    keep = phla_keep_weight * keep_phla + ptcr_keep_weight * keep_ptcr
    return Stage10Loss(
        total=stage8.total + keep_loss_weight * keep,
        stage8=stage8,
        keep=keep,
        keep_phla=keep_phla,
        keep_ptcr=keep_ptcr,
        keep_loss_weight=keep_loss_weight,
    )
