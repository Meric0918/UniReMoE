"""Group-normalized sample weighting with the original directed probability gradients."""
import torch
import torch.nn.functional as F

from .stage_MOE_loss import masked_bce
from .stage_MOE_consis_loss import StageMOEConsisLoss as StageFinalGatedLoss, LossStatistics


def normalized_weights(gates):
    """Keep the denominator differentiable: constant rescaling cannot suppress a loss."""
    gates = gates.float()
    return gates / gates.mean() if gates.numel() else gates


def weighted_group(values, gates, mask):
    values = values.float()[mask]
    weights = normalized_weights(gates[mask])
    if not values.numel():
        zero = values.sum()
        return zero, zero, zero, weights
    return (values.mean(), (weights * values).mean(), (weights - 1).square().mean(), weights)


def compute_stage_final_gated_loss(output, labels, masks, *, task, settings, epoch):
    native, count = masked_bce(output.logits[task], labels[task], masks[task])
    supervised = getattr(settings, f'{task}_weight') * native
    zero = native * 0.0
    parts, counts = {'native_bce': native}, {'native_bce': count}
    ramp = settings.consistency_ramp(epoch)
    weights = {'hierarchy': ramp * settings.hierarchy_lambda,
               'pt_implication': ramp * settings.pt_implication_lambda}
    hierarchy = implication = regularization = zero
    if task == 'pht':
        gates = (output.sample_constraint_gate if settings.sample_constraint_enabled
                 else torch.ones_like(output.logits[task], dtype=torch.float32))
        if gates is None or gates.shape != output.logits[task].shape:
            raise ValueError('PHT supervision requires one constraint gate per sample')

        def group(name, values, mask, coefficient):
            nonlocal regularization
            # Disabled terms must not create zero gate gradients that would still
            # let AdamW decay the gate parameters during consistency warmup.
            active_gates = gates if coefficient > 0 else gates.detach()
            raw, weighted, dispersion, normalized = weighted_group(values, active_gates, mask)
            n = int(mask.sum())
            parts[name], counts[name] = weighted, n
            for suffix, value in (('raw', raw), ('gate_regularization', dispersion),
                                  ('weight_mean', normalized.mean() if n else zero),
                                  ('weight_min', normalized.min() if n else zero),
                                  ('weight_max', normalized.max() if n else zero)):
                parts[f'{name}_{suffix}'], counts[f'{name}_{suffix}'] = value, n
            if settings.sample_constraint_enabled and coefficient > 0:
                regularization = regularization + coefficient * settings.constraint_gate_regularization * dispersion
            return weighted

        for pair in ('ph', 'pt'):
            if output.logits[pair].shape != labels[pair].shape or labels[pair].shape != masks[pair].shape:
                raise ValueError('Logits, labels and validity masks must have identical shapes')
            valid_pair = masks[pair].bool() & torch.isfinite(labels[pair])
            if not ((labels[pair][valid_pair] == 0) | (labels[pair][valid_pair] == 1)).all():
                raise ValueError('Valid supervision must be binary')
            # Sanitize masked labels before BCE so missing labels cannot produce NaN gradients.
            target = torch.where(valid_pair, labels[pair], torch.zeros_like(labels[pair]))
            values = F.binary_cross_entropy_with_logits(output.logits[pair].float(), target.float(), reduction='none')
            coefficient = getattr(settings, f'aux_{pair}_weight')
            loss = group(f'aux_{pair}_bce', values, valid_pair, coefficient)
            supervised = supervised + coefficient * loss
        valid = torch.ones_like(masks['pht'], dtype=torch.bool)
        for t in ('ph', 'pt', 'pht'):
            valid &= masks[t].bool() & torch.isfinite(labels[t])
        ph, pt, pht = (labels[t] for t in ('ph', 'pt', 'pht'))
        positive = valid & (ph == 1) & (pt == 1) & (pht == 1)
        ph_zero = valid & (ph == 0) & (pht == 0)
        pt_zero = valid & (pt == 0) & (pht == 0)
        allowed = (((ph == 0) & (pt == 0) & (pht == 0)) |
                   ((ph == 0) & (pt == 1) & (pht == 0)) |
                   ((ph == 1) & (pt == 0) & (pht == 0)) |
                   ((ph == 1) & (pt == 1) & (pht == 1)))
        if (valid & ~allowed).any():
            raise ValueError('Consistency requires 000/010/100/111 labels')
        h, t, triple = (output.logits[k].float().sigmoid() for k in ('ph', 'pt', 'pht'))
        for name, left, right, mask, coefficient in (
            ('hierarchy_positive', triple.detach(), h, positive, weights['hierarchy']),
            ('hierarchy_negative', triple, h.detach(), ph_zero, weights['hierarchy']),
            ('pt_positive', triple.detach(), t, positive, weights['pt_implication']),
            ('pt_negative', triple, t.detach(), pt_zero, weights['pt_implication']),
        ):
            group(name, F.relu(left - right - settings.consistency_margin).square(), mask, coefficient)
        hierarchy = parts['hierarchy_positive'] + parts['hierarchy_negative']
        implication = parts['pt_positive'] + parts['pt_negative']
    balance = (output.routing[task].float().mean(0) - 0.25).square().sum() if count else zero
    total = supervised + weights['hierarchy'] * hierarchy + weights['pt_implication'] * implication
    total = total + settings.balance_lambda * balance + regularization
    parts.update(hierarchy=hierarchy, pt_implication=implication, supervised=supervised,
                 balance=balance, gate_regularization=regularization, total=total)
    return StageFinalGatedLoss(total, supervised, balance, count, parts, counts, weights)
