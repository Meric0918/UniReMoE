"""Validation-only threshold selection and native-task test metrics."""
from __future__ import annotations
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_binary_metrics
from .stage8_evaluation import select_mcc_threshold
from .stage_MOE_data import move_batch
from .stage_MOE_model import forward_batch


def autocast_context(device, precision):
    if device.type != 'cuda' or precision == 'none':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def write_json(path, value):
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class RoutingStatistics:
    def __init__(self):
        self.sums, self.entropies, self.counts = {}, {}, {}

    def update(self, routing):
        for task, gates in routing.items():
            gates = gates.detach().float()
            self.sums[task] = self.sums.get(task, 0) + gates.sum(0).cpu().numpy()
            self.entropies[task] = self.entropies.get(task, 0.0) + float(-(gates * gates.clamp_min(1e-12).log()).sum())
            self.counts[task] = self.counts.get(task, 0) + len(gates)

    def result(self):
        result = {}
        for task, count in self.counts.items():
            weights = self.sums[task] / count
            result[task] = {'samples': count, 'mean_weights': weights.tolist(),
                            'mean_entropy': self.entropies[task] / count,
                            'shared_fraction': float(weights[:2].sum()),
                            'private_fraction': float(weights[2:].sum())}
        return result


@torch.inference_mode()
def evaluate_task(model, loader, task, *, precision='none', threshold=None, select_threshold=False,
                  prediction_path=None):
    if select_threshold == (threshold is not None):
        raise ValueError('Select a threshold on validation OR provide a saved threshold')
    model.eval()
    device = next(model.parameters()).device
    labels, probabilities, rows = [], [], []
    stats = RoutingStatistics()
    loss_sum, count = 0.0, 0
    for raw in loader:
        batch = move_batch(raw, device)
        with autocast_context(device, precision):
            output = forward_batch(model, task, batch)
        logits = output.logits[task].float()
        valid = batch['masks'][task] & torch.isfinite(batch['labels'][task])
        if not valid.any():
            continue
        targets, selected = batch['labels'][task][valid], logits[valid]
        loss_sum += float(F.binary_cross_entropy_with_logits(selected, targets, reduction='sum'))
        count += len(targets)
        probs = selected.sigmoid().cpu().numpy()
        if not np.isfinite(probs).all():
            raise FloatingPointError('Non-finite prediction')
        labels.extend(targets.cpu().tolist())
        probabilities.extend(probs.tolist())
        stats.update({t: g[valid] for t, g in output.routing.items()})
        if prediction_path:
            gates = output.routing[task][valid].cpu().tolist()
            indices = valid.nonzero().flatten().cpu().tolist()
            for index, prob, gate in zip(indices, probs, gates):
                record = raw['records'][index]
                rows.append({'source_file': record.source_file, 'source_row': record.source_row,
                             'source': record.source, 'split_type': record.split_type,
                             'peptide': record.peptide, 'hla': record.hla, 'tcr': record.tcr,
                             'label': float(raw['labels'][task][index]), 'probability': float(prob),
                             **{f'expert_{j}': v for j, v in enumerate(gate)}})
    if not count:
        raise ValueError(f'{task}: no valid evaluation labels')
    if select_threshold:
        threshold, _ = select_mcc_threshold(labels, probabilities)
    metrics = compute_binary_metrics(labels, probabilities, loss=loss_sum / count, threshold=threshold).to_dict()
    metrics.update(positive_samples=int(sum(labels)), negative_samples=count - int(sum(labels)),
                   positive_fraction=sum(labels) / count)
    if prediction_path:
        target = Path(prediction_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[*rows[0], 'prediction', 'threshold'])
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'prediction': int(row['probability'] >= threshold), 'threshold': threshold})
    return {'metrics': metrics, 'threshold': float(threshold), 'routing': stats.result()}


def evaluate_validation(model, loaders, precision):
    tasks = {task: evaluate_task(model, loader, task, precision=precision, select_threshold=True)
             for task, loader in loaders.items()}
    score = sum(value['metrics']['aupr'] for value in tasks.values()) / 3
    if not math.isfinite(score):
        raise ValueError('All three validation tasks need finite AUPRC')
    return {'tasks': tasks, 'score': score, 'selection_metric': 'mean_task_pr_auc',
            'thresholds': {task: value['threshold'] for task, value in tasks.items()}}
