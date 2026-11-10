"""Train-only group weights and exact effective-window CE/KD objectives."""

from __future__ import annotations

from collections import Counter
import math

import torch
import torch.nn.functional as F


def group_weights(rows):
    counts = Counter((p, int(y)) for p, _, y in rows)
    raw = {key: 1 / math.sqrt(n) for key, n in counts.items()}
    mean = sum(raw.values()) / len(raw)
    weights = {key: min(4.0, max(0.25, value / mean)) for key, value in raw.items()}
    return weights, [
        {"peptide": p, "label": y, "count": counts[p, y], "weight": weights[p, y]}
        for p, y in sorted(counts)
    ]


def sample_weights(peptides, labels, weight_map, device):
    if not weight_map:
        return torch.ones(len(peptides), device=device, dtype=torch.float32)
    return torch.tensor(
        [weight_map[p, int(y)] for p, y in zip(peptides, labels)],
        device=device,
        dtype=torch.float32,
    )


def window_pt_loss(
    logits,
    labels,
    weights,
    *,
    weight_sum,
    window_samples,
    smoothing=0.0,
    teacher_logits=None,
    alpha=0.0,
    temperature=2.0,
):
    if (
        weight_sum <= 0
        or window_samples < 1
        or not 0 <= smoothing < 1
        or not math.isfinite(alpha)
        or alpha < 0
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid PT loss configuration")
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights <= 0)
    ):
        raise ValueError("Nonfinite logits or invalid sample weights")
    ce = F.cross_entropy(
        logits.float(), labels, reduction="none", label_smoothing=smoothing
    )
    supervised = (ce * weights).sum() / weight_sum
    kd = logits.float().sum() * 0.0
    per_kd = None
    if alpha:
        if (
            teacher_logits is None
            or teacher_logits.shape != logits.shape
            or not torch.isfinite(teacher_logits).all()
        ):
            raise ValueError("Aligned finite teacher logits are required for KD")
        per_kd = temperature**2 * F.kl_div(
            F.log_softmax(logits.float() / temperature, dim=-1),
            F.softmax(teacher_logits.detach().float() / temperature, dim=-1),
            reduction="none",
        ).sum(-1)
        kd = per_kd.sum() / window_samples
    return supervised + alpha * kd, {
        "supervised": supervised,
        "kd": kd,
        "ce_per_sample": ce,
        "kd_per_sample": per_kd,
    }
