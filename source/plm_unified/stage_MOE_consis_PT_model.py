"""Sample-conditioned PT transfer, with legacy task gates for checkpoint evaluation."""
import torch
from torch import nn

from .stage_final_model import StageFinalBindingModel
from .stage_MOE_consis_PT_config import LegacyTaskGateConfig, StageMOEConsisPTConfig
from .stage_MOE_model import masked_mean
from .stage_MOE_PT_model import sample_gate


class LegacyTaskGateBindingModel(StageFinalBindingModel):
    architecture = LegacyTaskGateConfig.architecture

    def __init__(self, config):
        super().__init__(config)
        settings = config.stage_MOE_consis_PT
        self.task_gate_logit = None
        if settings.pt_fusion_enabled and settings.task_gate_enabled:
            self.task_gate_logit = nn.Parameter(torch.tensor(float(settings.task_gate_logit_init)))
        self.register_buffer('_unit_alpha', torch.tensor(1.0), persistent=False)

    def fusion_alpha(self, query, pt, hla_summary, peptide_mask):
        if self.task_gate_logit is None:
            return self._unit_alpha
        return self.task_gate_logit.float().sigmoid()


class StageMOEConsisPTBindingModel(StageFinalBindingModel):
    architecture = StageMOEConsisPTConfig.architecture

    def __init__(self, config):
        super().__init__(config)
        s = config.stage_MOE_consis_PT
        self.sample_gate = None
        if s.pt_fusion_enabled and s.sample_gate_enabled:
            self.sample_gate = sample_gate(config.model.model_dim, s.sample_gate_hidden_dim,
                                           s.sample_gate_logit_init)

    def fusion_alpha(self, query, pt, hla_summary, peptide_mask):
        if self.sample_gate is None:
            return query.new_ones(query.shape[0], dtype=torch.float32)
        context = torch.cat((masked_mean(self.fusion.query_norm(query), peptide_mask),
                             masked_mean(self.fusion.pt_norm(pt), peptide_mask), hla_summary), dim=-1)
        return self.sample_gate(context).float().sigmoid().squeeze(-1)
