"""Bounded-memory, sample-weighted gate summaries; quantiles use 256 fixed bins."""
import numpy as np
import torch
from .stage_MOE_PT_evaluation import FusionStatistics


class ScalarGateStatistics:
    def __init__(self):
        self.count = self.total = self.squares = self.near_zero = self.near_one = 0
        self.histogram = np.zeros(256, dtype=np.int64)

    def update(self, values):
        values = values.detach().float().cpu().numpy().reshape(-1)
        self.count += len(values)
        self.total += float(values.sum(dtype=np.float64))
        self.squares += float(np.square(values.astype(np.float64)).sum())
        self.near_zero += int((values < .05).sum())
        self.near_one += int((values > .95).sum())
        self.histogram += np.histogram(values, bins=256, range=(0, 1))[0]

    def result(self):
        if not self.count:
            return {'samples': 0}
        cumulative = self.histogram.cumsum()
        return {'samples': self.count, 'mean': self.total / self.count,
                'std': max(self.squares / self.count - (self.total / self.count)**2, 0)**.5,
                'near_zero_fraction': self.near_zero / self.count,
                'near_one_fraction': self.near_one / self.count,
                'quantile_bin_width': 1 / 256,
                **{f'q{int(q*100):02d}': (int(np.searchsorted(cumulative, q * self.count)) + .5) / 256
                   for q in (.05, .25, .5, .75, .95)}}


class GateStatistics:
    def __init__(self):
        self.groups = {}
        self.residue = FusionStatistics()

    def update(self, output, batch):
        if output.sample_constraint_gate is None:
            return
        size = len(output.sample_constraint_gate)
        all_rows = torch.ones(size, dtype=torch.bool, device=output.sample_constraint_gate.device)
        masks = {'all': all_rows}
        valid = all_rows.clone()
        for task in ('ph', 'pt', 'pht'):
            valid &= batch['masks'][task].bool() & torch.isfinite(batch['labels'][task])
        for code in ('000', '010', '100', '111'):
            mask = valid.clone()
            for task, bit in zip(('ph', 'pt', 'pht'), code):
                mask &= batch['labels'][task] == int(bit)
            masks[code] = mask
        for name in ('sample_fusion_gate', 'sample_constraint_gate'):
            values = getattr(output, name)
            if values is not None:
                for code, mask in masks.items():
                    key = f'{name}/{code}'
                    self.groups.setdefault(key, ScalarGateStatistics()).update(values[mask])
        self.residue.update(output.residue_beta, batch['peptide_mask'])

    def result(self):
        return {'samples': {k: v.result() for k, v in self.groups.items()},
                'residue_beta': self.residue.result()}
