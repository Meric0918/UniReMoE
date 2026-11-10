"""Independent binary supervision, with indexing before BCE for missing labels."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class StageMOELoss:
    total: torch.Tensor
    supervised: torch.Tensor
    balance: torch.Tensor
    valid_labels: int


def masked_bce(logits, labels, mask):
    if logits.shape != labels.shape or logits.shape != mask.shape:
        raise ValueError('Logits, labels and validity masks must have identical shapes')
    valid = mask.bool() & torch.isfinite(labels)
    if not valid.any():
        return logits.reshape(-1)[:0].sum(), 0
    selected = labels[valid]
    if not ((selected == 0) | (selected == 1)).all():
        raise ValueError('Valid supervision must be binary')
    return F.binary_cross_entropy_with_logits(logits[valid].float(), selected.float()), int(valid.sum())


def compute_stage_MOE_loss(output, labels, masks, *, task_weights=None, balance_lambda=0.0):
    if not output.logits:
        raise ValueError('At least one predicted task is required')
    zero = next(iter(output.logits.values())).reshape(-1)[:0].sum()
    supervised, count = zero, 0
    active = []
    for task, logits in output.logits.items():
        if task not in labels or task not in masks:
            continue
        loss, n = masked_bce(logits, labels[task], masks[task])
        supervised = supervised + (task_weights or {}).get(task, 1.0) * loss
        count += n
        if n:
            active.append(task)
    # Each supervised branch is counted once. PH context in a PHT-only pass is
    # trained through the hierarchy, but does not receive duplicate PH regularization.
    balance = sum(((output.routing[t].mean(0) - 0.25).square().sum() for t in active), zero)
    return StageMOELoss(supervised + balance_lambda * balance, supervised, balance, count)
