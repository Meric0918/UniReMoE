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

from .stage_MOE_PT_evaluation import FusionStatistics
from .stage_final_gated_statistics import GateStatistics
from .stage_final_gated_loss import compute_stage_final_gated_loss, LossStatistics
from .metrics import compute_binary_metrics
from .stage8_evaluation import select_mcc_threshold
from .stage_final_gated_data import move_batch
from .stage_final_gated_model import forward_batch


def triplet_diagnostics(rows, threshold, margin):
    """Probability diagnostics use all rows; directed masks require complete labels."""
    result = {'margin': margin, 'classes': {}, 'constraints': {}}
    for code in ('000', '010', '100', '111'):
        selected = [r for r in rows if r['class_code'] == code]
        rate = sum(r['p_pht'] >= threshold for r in selected) / len(selected) if selected else None
        result['classes'][code] = {'samples': len(selected),
                                  'recall' if code == '111' else 'false_positive_rate': rate}
    for name, pair, negative_codes in (('hierarchy', 'ph', ('000', '010')),
                                        ('pt_implication', 'pt', ('000', '100'))):
        scopes = {'all': rows,
                  'applicable': [r for r in rows if r['complete_labels'] and r['class_code'] in ('111', *negative_codes)],
                  'positive': [r for r in rows if r['complete_labels'] and r['class_code'] == '111'],
                  'negative': [r for r in rows if r['complete_labels'] and r['class_code'] in negative_codes]}
        result['constraints'][name] = {}
        for scope, selected in scopes.items():
            excess = [max(r['p_pht'] - r[f'p_{pair}'] - margin, 0.0) for r in selected]
            result['constraints'][name][scope] = {'samples': len(excess),
                'violation_rate': sum(v > 0 for v in excess) / len(excess) if excess else None,
                'mean_excess': sum(excess) / len(excess) if excess else None}
    return result


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
                  prediction_path=None, settings=None, epoch=0):
    if select_threshold == (threshold is not None):
        raise ValueError('Select a threshold on validation OR provide a saved threshold')
    model.eval()
    device = next(model.parameters()).device
    labels, probabilities, rows = [], [], []
    diagnostics = []
    stats = RoutingStatistics()
    fusion = FusionStatistics()
    sample_gates = GateStatistics()
    loss_statistics = LossStatistics()
    loss_sum, count = 0.0, 0
    for raw in loader:
        batch = move_batch(raw, device)
        with autocast_context(device, precision):
            output = forward_batch(model, task, batch)
        sample_gates.update(output, batch)
        if settings is not None:
            loss_statistics.update(task, compute_stage_final_gated_loss(
                output, batch['labels'], batch['masks'], task=task, settings=settings, epoch=epoch))
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
                             'residue_beta_mean': (float(output.residue_beta[index][batch['peptide_mask'][index].bool()].mean())
                                                   if output.residue_beta is not None else None),
                             'sample_fusion_gate': (float(output.sample_fusion_gate[index])
                                                    if output.sample_fusion_gate is not None else None),
                             'sample_constraint_gate': (float(output.sample_constraint_gate[index])
                                                        if output.sample_constraint_gate is not None else None),
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
    result = {'metrics': metrics, 'threshold': float(threshold), 'routing': stats.result(), 'fusion': fusion.result(), 'sample_gates': sample_gates.result()}
    if settings is not None:
        result['loss_details'] = loss_statistics.result()
    if task == 'pht':
        result['consistency'] = triplet_diagnostics(diagnostics, threshold, model.consistency_margin)
    return result


def evaluate_validation(model, loaders, precision, *, settings=None, epoch=0):
    tasks = {task: evaluate_task(model, loader, task, precision=precision, select_threshold=True,
                                settings=settings, epoch=epoch)
             for task, loader in loaders.items()}
    score = sum(value['metrics']['aupr'] for value in tasks.values()) / 3
    if not math.isfinite(score):
        raise ValueError('All three validation tasks need finite AUPRC')
    return {'tasks': tasks, 'score': score, 'selection_metric': 'mean_task_pr_auc',
            'thresholds': {task: value['threshold'] for task, value in tasks.items()}}
