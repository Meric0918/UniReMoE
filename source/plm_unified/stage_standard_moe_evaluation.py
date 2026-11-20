"""Validation-only threshold selection and native-task test metrics."""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .stage_MOE_PT_evaluation import FusionStatistics
from .metrics import compute_binary_metrics
from .stage8_evaluation import select_mcc_threshold
from .stage_final_data import move_batch
from .stage_final_evaluation import autocast_context, write_json, triplet_diagnostics
from .stage_standard_moe_model import forward_batch


class RoutingStatistics:
    """Sample routing weights and selection rates; experts have no shared/private roles."""

    def __init__(self):
        self.sums, self.selections, self.entropies, self.counts = {}, {}, {}, {}

    def update(self, routing):
        for task, gates in routing.items():
            gates = gates.detach().float()
            self.sums[task] = self.sums.get(task, 0) + gates.sum(0).cpu().numpy()
            self.selections[task] = self.selections.get(task, 0) + (gates > 0).sum(0).cpu().numpy()
            self.entropies[task] = self.entropies.get(task, 0.0) + float(
                -(gates * gates.clamp_min(1e-12).log()).sum())
            self.counts[task] = self.counts.get(task, 0) + len(gates)

    def result(self):
        return {task: {'samples': count, 'mean_weights': (self.sums[task] / count).tolist(),
                       'selection_frequency': (self.selections[task] / count).tolist(),
                       'mean_entropy': self.entropies[task] / count}
                for task, count in self.counts.items()}


@torch.inference_mode()
def evaluate_task(model, loader, task, *, precision='none', threshold=None, select_threshold=False,
                  prediction_path=None):
    if select_threshold == (threshold is not None):
        raise ValueError('Select a threshold on validation OR provide a saved threshold')
    model.eval()
    device = next(model.parameters()).device
    labels, probabilities, rows = [], [], []
    diagnostics = []
    stats = RoutingStatistics()
    fusion = FusionStatistics()
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
        if output.fusion_beta is not None:
            fusion.update(output.fusion_beta[valid], batch['peptide_mask'][valid])
        details = {}
        if task == 'pht':
            indices = valid.nonzero().flatten().cpu().tolist()
            predictions = {t: output.logits[t].float().sigmoid().detach().cpu().tolist() for t in ('ph', 'pt', 'pht')}
            if any(not np.isfinite(v).all() for v in predictions.values()):
                raise FloatingPointError('Non-finite auxiliary prediction')
            for index in indices:
                record = raw['records'][index]
                complete = all(record.masks)
                code = record.class_code or (''.join(str(int(v)) for v in record.labels) if complete else '')
                detail = {'class_code': code, 'complete_labels': complete,
                          **{f'y_{t}': record.labels[j] if record.masks[j] else None
                             for j, t in enumerate(('ph', 'pt', 'pht'))},
                          **{f'p_{t}': predictions[t][index] for t in predictions}}
                for name, pair in (('hierarchy', 'ph'), ('pt_implication', 'pt')):
                    excess = max(detail['p_pht'] - detail[f'p_{pair}'] - model.consistency_margin, 0.0)
                    detail[f'{name}_excess'] = excess
                    detail[f'{name}_violation'] = excess > 0
                details[index] = detail
                diagnostics.append(detail)
        if prediction_path:
            gates = output.routing[task][valid].cpu().tolist()
            indices = valid.nonzero().flatten().cpu().tolist()
            for index, prob, gate in zip(indices, probs, gates):
                record = raw['records'][index]
                rows.append({'source_file': record.source_file, 'source_row': record.source_row,
                             'source': record.source, 'split_type': record.split_type,
                             'peptide': record.peptide, 'hla': record.hla, 'tcr': record.tcr,
                             'label': float(raw['labels'][task][index]), 'probability': float(prob),
                             'fusion_beta_mean': (float(output.fusion_beta[index][batch['peptide_mask'][index].bool()].mean())
                                                  if output.fusion_beta is not None else None),
                             **details.get(index, {}),
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
    result = {'metrics': metrics, 'threshold': float(threshold), 'routing': stats.result(), 'fusion': fusion.result()}
    if task == 'pht':
        result['consistency'] = triplet_diagnostics(diagnostics, threshold, model.consistency_margin)
    return result


def evaluate_validation(model, loaders, precision):
    tasks = {task: evaluate_task(model, loader, task, precision=precision, select_threshold=True)
             for task, loader in loaders.items()}
    score = sum(value['metrics']['aupr'] for value in tasks.values()) / 3
    if not math.isfinite(score):
        raise ValueError('All three validation tasks need finite AUPRC')
    return {'tasks': tasks, 'score': score, 'selection_metric': 'mean_task_pr_auc',
            'thresholds': {task: value['threshold'] for task, value in tasks.items()}}
