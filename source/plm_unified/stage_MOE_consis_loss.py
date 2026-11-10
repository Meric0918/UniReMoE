"""Label-aware directed consistency on four-type triplets, plus native BCE."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .stage_MOE_loss import masked_bce


@dataclass
class StageMOEConsisLoss:
    total: torch.Tensor
    supervised: torch.Tensor
    balance: torch.Tensor
    valid_labels: int  # Native task rows, never the sum of three labels.
    components: dict
    counts: dict
    weights: dict


def compute_stage_MOE_consis_loss(output, labels, masks, *, task, settings, epoch):
    native, count = masked_bce(output.logits[task], labels[task], masks[task])
    supervised = getattr(settings, f'{task}_weight') * native
    zero = native * 0.0
    parts = {'native_bce': native}
    counts = {'native_bce': count}
    ramp = settings.consistency_ramp(epoch)
    weights = {'hierarchy': ramp * settings.hierarchy_lambda,
               'pt_implication': ramp * settings.pt_implication_lambda}
    hierarchy = implication = zero
    if task == 'pht':
        for pair in ('ph', 'pt'):
            loss, n = masked_bce(output.logits[pair], labels[pair], masks[pair])
            parts[f'aux_{pair}_bce'], counts[f'aux_{pair}_bce'] = loss, n
            supervised = supervised + getattr(settings, f'aux_{pair}_weight') * loss
        valid = torch.ones_like(masks['pht'], dtype=torch.bool)
        for t in ('ph', 'pt', 'pht'):
            valid = valid & masks[t].bool() & torch.isfinite(labels[t])
        ph, pt, pht = (labels[t] for t in ('ph', 'pt', 'pht'))
        positive = valid & (ph == 1) & (pt == 1) & (pht == 1)
        ph_zero = valid & (ph == 0) & (pht == 0)
        pt_zero = valid & (pt == 0) & (pht == 0)
        allowed = ((ph == 0) & (pt == 0) & (pht == 0) |
                   (ph == 0) & (pt == 1) & (pht == 0) |
                   (ph == 1) & (pt == 0) & (pht == 0) |
                   (ph == 1) & (pt == 1) & (pht == 1))
        if (valid & ~allowed).any():
            raise ValueError('Consistency requires 000/010/100/111 labels')
        h, t, triple = (output.logits[k].float().sigmoid() for k in ('ph', 'pt', 'pht'))
        margin = settings.consistency_margin
        specifications = (
            ('hierarchy_positive', triple.detach(), h, positive),
            ('hierarchy_negative', triple, h.detach(), ph_zero),
            ('pt_positive', triple.detach(), t, positive),
            ('pt_negative', triple, t.detach(), pt_zero),
        )
        for name, left, right, mask in specifications:
            values = F.relu(left - right - margin).square()[mask]
            parts[name] = values.mean() if values.numel() else values.sum()
            counts[name] = int(mask.sum())
        hierarchy = parts['hierarchy_positive'] + parts['hierarchy_negative']
        implication = parts['pt_positive'] + parts['pt_negative']
    # Auxiliary classification does not add duplicate PH/PT routing penalties.
    balance = (output.routing[task].float().mean(0) - 0.25).square().sum() if count else zero
    total = supervised + weights['hierarchy'] * hierarchy + weights['pt_implication'] * implication
    total = total + settings.balance_lambda * balance
    parts.update(hierarchy=hierarchy, pt_implication=implication, supervised=supervised,
                 balance=balance, total=total)
    return StageMOEConsisLoss(total, supervised, balance, count, parts, counts, weights)


class LossStatistics:
    """Masked components use their own denominators; totals use native rows."""
    def __init__(self):
        self.sums, self.counts, self.weights = {}, {}, {}

    def update(self, task, result):
        self.weights = result.weights
        for name, value in result.components.items():
            key = f'{task}/{name}'
            n = result.counts.get(name, result.valid_labels)
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach()) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def result(self):
        return {'components': {key: {'mean': self.sums[key] / n if n else None, 'samples': n}
                               for key, n in self.counts.items()}, 'weights': dict(self.weights)}
