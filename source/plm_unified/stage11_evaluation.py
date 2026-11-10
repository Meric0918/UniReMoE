"""PHT-only evaluation at a fixed threshold, with per-negative-class errors."""

import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_binary_metrics, task_score
from .stage11_data import model_inputs, move_cached_batch


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@dataclass
class Stage11Evaluation:
    metrics: dict
    score: float
    predictions: dict


@torch.inference_mode()
def evaluate_stage11_model(model, loader, *, device, precision="none"):
    model.eval()
    labels, probabilities, losses, types, rows = [], [], [], [], []
    for raw in loader:
        batch = move_cached_batch(raw, device)
        with autocast_context(device, precision):
            output = model.forward_pht(*model_inputs(batch))
        logits = output.pht_logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite Stage11 evaluation logits")
        labels.extend(batch["pht_labels"].cpu().tolist())
        probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        losses.extend(F.cross_entropy(logits, batch["pht_labels"], reduction="none").cpu().tolist())
        types.extend(batch["label_types"])
        rows.extend(batch["source_rows"].cpu().tolist())
    if not labels:
        raise ValueError("Stage11 evaluation set is empty")
    y, p, ce, codes = np.array(labels), np.array(probabilities), np.array(losses), np.array(types)
    balanced = np.mean([ce[y == value].mean() for value in (0, 1) if (y == value).any()])
    metrics = compute_binary_metrics(y, p, loss=float(balanced), threshold=0.5).to_dict()
    metrics["unweighted_ce"] = float(ce.mean())
    metrics["threshold"] = 0.5
    for code in ("000", "010", "100"):
        selected = codes == code
        metrics[f"fpr_{code}"] = float((p[selected] >= 0.5).mean()) if selected.any() else None
        metrics[f"samples_{code}"] = int(selected.sum())
    return Stage11Evaluation(metrics, task_score(metrics), {
        "source_row": rows, "class_code": types, "label": labels, "probability": probabilities})


def write_predictions(path, evaluation, records):
    if len(records) != len(evaluation.predictions["label"]):
        raise ValueError("Prediction records are misaligned")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peptide", "HLA", "tcr", "source_file", "source_row", "class_code",
                         "label", "probability", "prediction"])
        for record, label, probability in zip(records, evaluation.predictions["label"], evaluation.predictions["probability"]):
            writer.writerow([record.peptide, record.hla, record.tcr, record.source_file,
                             record.source_row, record.label_type, label, probability, int(probability >= 0.5)])
