"""PH-supervised triplets with directed PH–PHT consistency and native PT training."""
from dataclasses import dataclass, field

import torch.nn.functional as F

from .config import SwanLabConfig
from .stage_MOE_loss import masked_bce
from .stage_final_config import StageFinalConfig, StageFinalTrainingSettings, load_stage_final_config
from .stage_final_loss import StageFinalLoss
from .stage_final_model import StageFinalBindingModel


@dataclass
class StageFinalNewSettings(StageFinalTrainingSettings):
    pt_fusion_enabled: bool = True
    fusion_hidden_dim: int = 128
    fusion_bias_init: float = -2.0
    aux_ph_weight: float = 0.1
    consistency_margin: float = 0.05
    hierarchy_lambda: float = 0.05


@dataclass
class StageFinalNewConfig(StageFinalConfig):
    stage_name = 'stage_final_new'
    architecture = 'stage_final_new_v1'
    ph_only_triplets = True
    objective_fields = ('aux_ph_weight', 'hierarchy_lambda', 'consistency_margin')
    settings_type = StageFinalNewSettings
    stage_final: StageFinalNewSettings = field(default_factory=StageFinalNewSettings)
    swanlab: SwanLabConfig = field(default_factory=lambda: SwanLabConfig(
        mode='disabled', group='plm-stage_final_new', experiment_prefix='PLM-Stage_final_new',
        tags=['ESMC-300M', 'TCR-BERT', 'stage_final_new']))

    @property
    def stage_final_new(self):
        return self.stage_final


def load_stage_final_new_config(path):
    return load_stage_final_config(path, config_type=StageFinalNewConfig)


def directed_penalty(left, right, mask, margin):
    excess = F.relu(left[mask] - right[mask] - margin).square()
    return (excess.mean() if excess.numel() else excess.sum()), excess.numel()


def compute_stage_final_new_loss(output, labels, masks, *, task, settings, epoch):
    native, count = masked_bce(output.logits[task], labels[task], masks[task])
    supervised = getattr(settings, f'{task}_weight') * native
    parts, counts = {'native_bce': native}, {'native_bce': count}
    weights = {'hierarchy': settings.consistency_ramp(epoch) * settings.hierarchy_lambda}
    total = supervised
    if task == 'pht':
        aux, counts['aux_ph_bce'] = masked_bce(output.logits['ph'], labels['ph'], masks['ph'])
        parts['aux_ph_bce'] = aux
        supervised = supervised + settings.aux_ph_weight * aux
        valid = masks['ph'].bool() & masks['pht'].bool()
        ph, pht = labels['ph'], labels['pht']
        positive = valid & (ph == 1) & (pht == 1)
        negative = valid & (ph == 0) & (pht == 0)
        h, triple = (output.logits[t].float().sigmoid() for t in ('ph', 'pht'))
        parts['hierarchy_positive'], counts['hierarchy_positive'] = directed_penalty(
            triple.detach(), h, positive, settings.consistency_margin)
        parts['hierarchy_negative'], counts['hierarchy_negative'] = directed_penalty(
            triple, h.detach(), negative, settings.consistency_margin)
        parts['hierarchy'] = parts['hierarchy_positive'] + parts['hierarchy_negative']
        counts['hierarchy'] = counts['hierarchy_positive'] + counts['hierarchy_negative']
        total = supervised + weights['hierarchy'] * parts['hierarchy']
    balance = (output.routing[task].float().mean(0) - 0.25).square().sum() if count else native * 0
    total = total + settings.balance_lambda * balance
    parts.update(supervised=supervised, balance=balance, total=total)
    return StageFinalLoss(total, supervised, balance, count, parts, counts, weights)


class StageFinalNewBindingModel(StageFinalBindingModel):
    architecture = StageFinalNewConfig.architecture
    compute_loss = staticmethod(compute_stage_final_new_loss)
