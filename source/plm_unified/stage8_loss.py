"""Supervised and directed consistency losses for Stage 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .stage8_config import Stage8Condition


Stage8LossPhase = Literal["stage8a", "stage8b"]


@dataclass
class Stage8Loss:
    total: torch.Tensor
    supervised: torch.Tensor
    pht: torch.Tensor
    phla: torch.Tensor
    ptcr: torch.Tensor
    hierarchy: torch.Tensor
    hierarchy_positive: torch.Tensor
    hierarchy_negative: torch.Tensor
    pt_implication: torch.Tensor
    pt_positive: torch.Tensor
    pt_negative: torch.Tensor
    hierarchy_lambda: float
    pt_implication_lambda: float


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def compute_stage8_loss(
    phla_logits: torch.Tensor,
    ptcr_logits: torch.Tensor,
    pht_logits: torch.Tensor,
    ph_labels: torch.Tensor,
    pt_labels: torch.Tensor,
    pht_labels: torch.Tensor,
    label_types: Sequence[str],
    *,
    condition: Stage8Condition | str,
    epoch: int,
    margin: float = 0.05,
    hierarchy_lambda: float = 0.05,
    pt_implication_lambda: float = 0.03,
    ramp_epochs: int = 5,
    pht_weight: float = 1.0,
    phla_weight: float = 0.3,
    ptcr_weight: float = 0.2,
    supervised_weight: float = 1.0,
    phase: Stage8LossPhase | None = None,
    round_index: int | None = None,
) -> Stage8Loss:
    condition = Stage8Condition(condition)
    if phase not in {None, "stage8a", "stage8b"}:
        raise ValueError(f"Unknown Stage-8 loss phase: {phase}")
    if epoch < 0 or ramp_epochs < 1:
        raise ValueError("epoch must be nonnegative and ramp_epochs positive")
    if round_index is not None and round_index < 0:
        raise ValueError("round_index must be nonnegative")
    device = pht_logits.device
    masks = {
        kind: torch.tensor([value == kind for value in label_types], device=device)
        for kind in ("111", "100", "010", "000")
    }
    ph_ce = F.cross_entropy(phla_logits, ph_labels, reduction="none")
    pt_ce = F.cross_entropy(ptcr_logits, pt_labels, reduction="none")
    pht_ce = F.cross_entropy(pht_logits, pht_labels, reduction="none")
    positive = masks["111"]
    negative = ~positive
    pht_loss = 0.5 * _masked_mean(pht_ce, positive) + 0.5 * _masked_mean(pht_ce, negative)
    phla_loss = ph_ce.mean()
    ptcr_loss = pt_ce.mean()
    supervised = pht_weight * pht_loss + phla_weight * phla_loss + ptcr_weight * ptcr_loss

    ph_prob = phla_logits.softmax(dim=-1)[:, 1]
    pt_prob = ptcr_logits.softmax(dim=-1)[:, 1]
    pht_prob = pht_logits.softmax(dim=-1)[:, 1]
    ph_zero = masks["000"] | masks["010"]
    pt_zero = masks["000"] | masks["100"]
    hierarchy_positive = _masked_mean(
        F.relu(pht_prob.detach() - ph_prob - margin).square(), positive
    )
    hierarchy_negative = _masked_mean(
        F.relu(pht_prob - ph_prob.detach() - margin).square(), ph_zero
    )
    hierarchy = hierarchy_positive + hierarchy_negative
    pt_positive = _masked_mean(
        F.relu(pht_prob.detach() - pt_prob - margin).square(), positive
    )
    pt_negative = _masked_mean(
        F.relu(pht_prob - pt_prob.detach() - margin).square(), pt_zero
    )
    pt_implication = pt_positive + pt_negative

    ramp_index = epoch if round_index is None else round_index
    ramp = min(float(ramp_index) / float(ramp_epochs), 1.0)
    constraints_enabled = phase != "stage8a"
    active_hierarchy = hierarchy_lambda * ramp if constraints_enabled and condition in {
        Stage8Condition.HIER_ONLY, Stage8Condition.FULL
    } else 0.0
    active_pt = pt_implication_lambda * ramp if constraints_enabled and condition in {
        Stage8Condition.PT_IMP_ONLY, Stage8Condition.FULL
    } else 0.0
    total = (
        supervised_weight * supervised
        + active_hierarchy * hierarchy
        + active_pt * pt_implication
    )
    return Stage8Loss(
        total, supervised, pht_loss, phla_loss, ptcr_loss,
        hierarchy, hierarchy_positive, hierarchy_negative,
        pt_implication, pt_positive, pt_negative,
        active_hierarchy, active_pt,
    )
