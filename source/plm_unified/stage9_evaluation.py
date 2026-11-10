"""Pair-level ranking, classification, and embedding evaluation for Stage 9."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .metrics import BinaryMetrics, compute_binary_metrics
from .stage9_loss import compute_stage9_pair_loss
from .stage9_model import ProgressiveStage9FineTuningModel


@dataclass
class Stage9PairMetrics:
    samples: int
    contrastive_loss: float
    classification_loss: float
    ranking_correct: int
    ranking_accuracy: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    mean_similarity_gap: float
    median_similarity_gap: float
    classification: BinaryMetrics

    def to_dict(self) -> dict:
        values = asdict(self)
        values["classification"] = self.classification.to_dict()
        return values


@dataclass
class Stage9PairPredictionResult:
    metrics: Stage9PairMetrics
    pair_ids: list[str]
    source_rows: np.ndarray
    positive_similarities: np.ndarray
    negative_similarities: np.ndarray
    positive_probabilities: np.ndarray
    negative_probabilities: np.ndarray
    z_ph: np.ndarray | None = None
    z_t_positive: np.ndarray | None = None
    z_t_negative: np.ndarray | None = None


def compute_stage9_pair_metrics(
    *,
    positive_similarities,
    negative_similarities,
    positive_probabilities,
    negative_probabilities,
    contrastive_loss: float,
    classification_loss: float,
    threshold: float,
) -> Stage9PairMetrics:
    positive_similarities = np.asarray(positive_similarities, dtype=np.float64)
    negative_similarities = np.asarray(negative_similarities, dtype=np.float64)
    positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    negative_probabilities = np.asarray(negative_probabilities, dtype=np.float64)
    lengths = {
        len(positive_similarities),
        len(negative_similarities),
        len(positive_probabilities),
        len(negative_probabilities),
    }
    if len(lengths) != 1 or not positive_similarities.ndim == 1:
        raise ValueError("Stage-9 pair metric inputs must be aligned 1-D arrays")
    samples = len(positive_similarities)
    if samples == 0:
        raise ValueError("Stage-9 pair metrics require at least one pair")
    gaps = positive_similarities - negative_similarities
    labels = np.concatenate(
        [np.ones(samples, dtype=np.int64), np.zeros(samples, dtype=np.int64)]
    )
    probabilities = np.concatenate(
        [positive_probabilities, negative_probabilities]
    )
    classification = compute_binary_metrics(
        labels,
        probabilities,
        loss=classification_loss,
        threshold=threshold,
    )
    ranking_correct = int(np.sum(gaps > 0))
    return Stage9PairMetrics(
        samples=samples,
        contrastive_loss=float(contrastive_loss),
        classification_loss=float(classification_loss),
        ranking_correct=ranking_correct,
        ranking_accuracy=ranking_correct / samples,
        mean_positive_similarity=float(positive_similarities.mean()),
        mean_negative_similarity=float(negative_similarities.mean()),
        mean_similarity_gap=float(gaps.mean()),
        median_similarity_gap=float(np.median(gaps)),
        classification=classification,
    )


@torch.inference_mode()
def evaluate_stage9_pairs(
    model: ProgressiveStage9FineTuningModel,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
    temperature: float,
    threshold: float = 0.5,
    collect_embeddings: bool = False,
) -> Stage9PairPredictionResult:
    model.set_evaluation()
    positive_similarities: list[np.ndarray] = []
    negative_similarities: list[np.ndarray] = []
    positive_probabilities: list[np.ndarray] = []
    negative_probabilities: list[np.ndarray] = []
    pair_ids: list[str] = []
    source_rows: list[np.ndarray] = []
    embeddings: dict[str, list[np.ndarray]] = {
        "z_ph": [],
        "z_t_positive": [],
        "z_t_negative": [],
    }
    contrastive_sum = 0.0
    classification_sum = 0.0
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    for batch in loader:
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            output = model.forward_pair(
                batch["peptide_sequences"],
                batch["hla_sequences"],
                batch["positive_tcr_sequences"],
                batch["negative_tcr_sequences"],
            )
            losses = compute_stage9_pair_loss(
                output,
                temperature=temperature,
                contrastive_weight=1.0,
                classification_weight=1.0,
                include_classification=True,
            )
        batch_samples = losses.samples
        samples += batch_samples
        contrastive_sum += float(losses.contrastive.cpu()) * batch_samples
        classification_sum += float(losses.classification.cpu()) * batch_samples
        positive_similarities.append(output.positive_similarity.float().cpu().numpy())
        negative_similarities.append(output.negative_similarity.float().cpu().numpy())
        positive_probabilities.append(
            torch.softmax(output.positive_logits, dim=1)[:, 1].float().cpu().numpy()
        )
        negative_probabilities.append(
            torch.softmax(output.negative_logits, dim=1)[:, 1].float().cpu().numpy()
        )
        pair_ids.extend(batch["pair_ids"])
        source_rows.append(batch["source_rows"].cpu().numpy())
        if collect_embeddings:
            for name in embeddings:
                embeddings[name].append(getattr(output, name).float().cpu().numpy())
    if not samples:
        raise RuntimeError("Stage-9 pair evaluation loader is empty")
    positive_similarity_array = np.concatenate(positive_similarities)
    negative_similarity_array = np.concatenate(negative_similarities)
    positive_probability_array = np.concatenate(positive_probabilities)
    negative_probability_array = np.concatenate(negative_probabilities)
    metrics = compute_stage9_pair_metrics(
        positive_similarities=positive_similarity_array,
        negative_similarities=negative_similarity_array,
        positive_probabilities=positive_probability_array,
        negative_probabilities=negative_probability_array,
        contrastive_loss=contrastive_sum / samples,
        classification_loss=classification_sum / samples,
        threshold=threshold,
    )
    return Stage9PairPredictionResult(
        metrics=metrics,
        pair_ids=pair_ids,
        source_rows=np.concatenate(source_rows),
        positive_similarities=positive_similarity_array,
        negative_similarities=negative_similarity_array,
        positive_probabilities=positive_probability_array,
        negative_probabilities=negative_probability_array,
        **{
            name: np.concatenate(values) if values else None
            for name, values in embeddings.items()
        },
    )
