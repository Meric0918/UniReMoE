"""PHT-only evaluation and selection for Stage Adapter."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import BinaryMetrics, compute_binary_metrics, performance_avg
from .stage8_evaluation import select_mcc_threshold


@dataclass
class StageAdapterEvaluation:
    metrics: BinaryMetrics
    selected_threshold: float
    selected_metrics: BinaryMetrics
    labels: np.ndarray
    probabilities: np.ndarray
    source_rows: np.ndarray

    @property
    def selection_score(self) -> float:
        return performance_avg(self.metrics)


@torch.inference_mode()
def evaluate_stage_adapter(
    model,
    loader,
    *,
    device: torch.device,
    mixed_precision: str,
) -> StageAdapterEvaluation:
    model.set_evaluation()
    labels, probabilities, rows = [], [], []
    loss_sum = 0.0
    samples = 0
    use_autocast = device.type == "cuda" and mixed_precision != "none"
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    for batch in loader:
        batch_labels = batch["pht_labels"].to(
            device, non_blocking=device.type == "cuda"
        )
        context = torch.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()
        with context:
            output = model.forward_pht(
                batch["peptide_sequences"],
                batch["hla_sequences"],
                batch["tcr_sequences"],
            )
        count = len(batch_labels)
        samples += count
        loss_sum += float(
            F.cross_entropy(output.logits, batch_labels, reduction="sum").cpu()
        )
        labels.append(batch_labels.cpu().numpy())
        probabilities.append(
            output.logits.softmax(-1)[:, 1].float().cpu().numpy()
        )
        rows.append(batch["source_rows"].cpu().numpy())
    if not samples:
        raise RuntimeError("Stage Adapter evaluation loader is empty")
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    metrics = compute_binary_metrics(
        y_true, y_score, loss=loss_sum / samples, threshold=0.5
    )
    threshold, selected = select_mcc_threshold(y_true, y_score)
    selected.loss = metrics.loss
    return StageAdapterEvaluation(
        metrics, threshold, selected, y_true, y_score, np.concatenate(rows)
    )
