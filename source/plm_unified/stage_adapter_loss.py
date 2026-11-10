"""Balanced PHT-only supervision for Stage Adapter."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def balanced_pht_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_types: Sequence[str],
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("PHT logits must have shape [batch, 2]")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("PHT labels must align with logits")
    if len(label_types) != len(labels):
        raise ValueError("label_types must align with PHT labels")
    positive = torch.tensor(
        [value == "111" for value in label_types], device=logits.device
    )
    if not positive.any() or positive.all():
        raise ValueError("Each balanced PHT batch must contain positive and negative rows")
    losses = F.cross_entropy(logits, labels, reduction="none")
    return 0.5 * losses[positive].mean() + 0.5 * losses[~positive].mean()
