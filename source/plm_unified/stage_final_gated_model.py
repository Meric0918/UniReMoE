"""Independent sample gates for residue PT fusion and auxiliary supervision."""
from dataclasses import dataclass
import math

import torch

from .stage_MOE_PT_model import StageMOEPTBindingModel, StageMOEPTOutput, sample_gate
from .stage_MOE_model import TASKS, masked_mean
from .stage_MOE_consis_model import forward_batch

ARCHITECTURE = 'stage_final_gated_v1'


@dataclass
class StageFinalGatedOutput(StageMOEPTOutput):
    residue_beta: torch.Tensor | None = None
    sample_fusion_gate: torch.Tensor | None = None
    sample_constraint_gate: torch.Tensor | None = None


class StageFinalGatedBindingModel(StageMOEPTBindingModel):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__(config)
        s = config.stage_final_gated
        self.consistency_margin = s.consistency_margin
        self.sample_fusion_enabled = s.sample_fusion_enabled
        self.sample_constraint_enabled = s.sample_constraint_enabled
        self.constraint_gate_floor = s.constraint_gate_floor
        self.sample_fusion = sample_gate(config.model.model_dim, s.sample_gate_hidden_dim,
                                         math.log(s.sample_fusion_init / (1 - s.sample_fusion_init)))
        self.sample_constraint = sample_gate(config.model.model_dim, s.sample_gate_hidden_dim, 0.0)

    def forward(self, *args, **kwargs):
        return StageFinalGatedOutput(**vars(super().forward(*args, **kwargs)))

    def _triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                 *, all_heads, return_attention=False, return_features=False):
        p = self.peptide_encoder(peptide_hidden, peptide_mask)
        h = self.hla_encoder(hla_hidden, hla_mask)
        t = self.tcr_encoder(tcr_hidden, tcr_mask)
        pm, hm, tm = masked_mean(p, peptide_mask), masked_mean(h, hla_mask), masked_mean(t, tcr_mask)
        output = StageFinalGatedOutput({}, {})
        zph = self._branch('ph', p, h, peptide_mask, hla_mask, [pm, hm], output,
                           return_attention, return_features, classify=all_heads)
        zpt = None
        if all_heads or self.pt_fusion_enabled or self.sample_constraint_enabled:
            zpt = self._branch('pt', p, t, peptide_mask, tcr_mask, [pm, tm], output,
                               return_attention, return_features, classify=all_heads)
        hidden, attention = self.cross_attention['pht'](zph, t, peptide_mask, tcr_mask, return_attention)
        context = None
        if self.sample_constraint_enabled or (self.pt_fusion_enabled and self.sample_fusion_enabled):
            context = torch.cat((masked_mean(self.fusion.query_norm(hidden), peptide_mask),
                                 masked_mean(self.fusion.pt_norm(zpt), peptide_mask), hm), -1)
        ones = torch.ones(p.shape[0], device=p.device, dtype=torch.float32)
        output.sample_constraint_gate = (self.constraint_gate_floor + (1 - self.constraint_gate_floor)
            * self.sample_constraint(context.detach()).float().sigmoid().squeeze(-1)
            if self.sample_constraint_enabled else ones)
        fused = hidden
        if self.pt_fusion_enabled:
            fused, output.residue_beta = self.fusion(hidden, zpt, hm, peptide_mask)
            output.sample_fusion_gate = (self.sample_fusion(context).float().sigmoid().squeeze(-1)
                                        if self.sample_fusion_enabled else ones)
            output.fusion_beta = output.residue_beta * output.sample_fusion_gate[:, None]
            if self.sample_fusion_enabled:
                residual = self.fusion.projection(self.fusion.pt_norm(zpt))
                fused = (hidden + output.fusion_beta[..., None].to(residual.dtype) * residual
                         ).masked_fill(~peptide_mask.bool().unsqueeze(-1), 0)
        route_context = torch.cat([masked_mean(fused, peptide_mask), pm, hm, tm], -1)
        sequence, gates = self.moe['pht'](fused, route_context, peptide_mask, self.shared_experts, self.temperature)
        output.routing['pht'] = gates
        pooled, pool_weights = self.pooling['pht'](sequence, peptide_mask)
        output.logits['pht'] = self.classifiers['pht'](pooled).squeeze(-1)
        if return_attention:
            output.attention['pht'] = attention
            output.pooling_weights['pht'] = pool_weights
        if return_features:
            output.features['pht'] = pooled
        return output
