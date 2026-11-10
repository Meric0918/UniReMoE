"""Paired InfoNCE and joint Stage-9 classification loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .stage9_model import Stage9PairOutput


@dataclass
class Stage9LossOutput:
    total: torch.Tensor
    contrastive: torch.Tensor
    classification: torch.Tensor
    positive_classification: torch.Tensor
    negative_classification: torch.Tensor
    mean_positive_similarity: torch.Tensor
    mean_negative_similarity: torch.Tensor
    mean_similarity_gap: torch.Tensor
    ranking_correct: int
    samples: int


def paired_infonce_loss(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if positive_similarity.shape != negative_similarity.shape:
        raise ValueError("positive and negative similarities must have equal shape")
    return F.softplus(
        (negative_similarity - positive_similarity) / temperature
    ).mean()


def compute_stage9_pair_loss(
    output: Stage9PairOutput,
    *,
    temperature: float,
    contrastive_weight: float,
    classification_weight: float,
    include_classification: bool,
) -> Stage9LossOutput:
    contrastive = paired_infonce_loss(
        output.positive_similarity,
        output.negative_similarity,
        temperature=temperature,
    )
    samples = int(output.positive_similarity.shape[0])
    positive_labels = torch.ones(
        samples, dtype=torch.long, device=output.positive_logits.device
    )
    negative_labels = torch.zeros(
        samples, dtype=torch.long, device=output.negative_logits.device
    )
    if include_classification:
        positive_classification = F.cross_entropy(
            output.positive_logits, positive_labels
        )
        negative_classification = F.cross_entropy(
            output.negative_logits, negative_labels
        )
        classification = 0.5 * (
            positive_classification + negative_classification
        )
    else:
        zero = contrastive.new_zeros(())
        positive_classification = zero
        negative_classification = zero
        classification = zero
    total = (
        contrastive_weight * contrastive
        + classification_weight * classification
    )
    gap = output.positive_similarity - output.negative_similarity
    return Stage9LossOutput(
        total=total,
        contrastive=contrastive,
        classification=classification,
        positive_classification=positive_classification,
        negative_classification=negative_classification,
        mean_positive_similarity=output.positive_similarity.mean(),
        mean_negative_similarity=output.negative_similarity.mean(),
        mean_similarity_gap=gap.mean(),
        ranking_correct=int((gap > 0).sum().detach().cpu()),
        samples=samples,
    )
