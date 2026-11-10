"""Rectangular, multi-positive bidirectional CLIP and balanced PHT losses."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Stage11ContrastiveLoss:
    total: torch.Tensor
    valid_left: int
    valid_right: int
    positive_pairs: int
    negative_pairs: int
    positive_similarity: float
    negative_similarity: float

    @property
    def valid(self):
        return self.valid_left + self.valid_right > 0


def multi_positive_clip_loss(left, right, positive_mask, *, temperature=0.2):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Contrastive embeddings must have compatible [entities, dimensions] shapes")
    mask = positive_mask.to(device=left.device, dtype=torch.bool)
    if mask.shape != (len(left), len(right)) or not len(left) or not len(right):
        raise ValueError("Positive mask must align with nonempty entity axes")
    with torch.autocast(device_type=left.device.type, enabled=False):
        similarity = F.normalize(left.float(), dim=-1, eps=1e-8) @ F.normalize(right.float(), dim=-1, eps=1e-8).T
        logits = similarity / temperature

        def direction(scores, positives):
            counts = positives.sum(-1)
            valid = (counts > 0) & (counts < scores.shape[1])
            if not valid.any():
                return scores.sum() * 0.0, 0
            losses = -(scores.log_softmax(-1) * positives).sum(-1) / counts.clamp_min(1)
            return losses[valid].mean(), int(valid.sum().item())

        a, na = direction(logits, mask)
        b, nb = direction(logits.T, mask.T)
        # Average over available directions when one side has no valid anchors.
        total = (a + b) / max(int(na > 0) + int(nb > 0), 1)
    pos, neg = int(mask.sum().item()), int((~mask).sum().item())
    return Stage11ContrastiveLoss(
        total, na, nb, pos, neg,
        float(similarity.detach()[mask].mean().item()) if pos else 0.0,
        float(similarity.detach()[~mask].mean().item()) if neg else 0.0)


def balanced_pht_loss(logits, labels):
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    parts = [ce[labels == value].mean() for value in (0, 1) if (labels == value).any()]
    if not parts:
        raise ValueError("PHT batch cannot be empty")
    return torch.stack(parts).mean()
