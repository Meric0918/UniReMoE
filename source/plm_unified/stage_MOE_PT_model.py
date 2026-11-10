"""Document v2: residue-aligned, HLA-conditioned PT fusion before PHT MoE."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .stage_MOE_model import (
    StageMOEBindingModel, StageMOEOutput, TASKS, masked_mean, forward_batch,
)

ARCHITECTURE = 'stage_MOE_PT_v2'


@dataclass
class StageMOEPTOutput(StageMOEOutput):
    fusion_beta: torch.Tensor | None = None  # [B, peptide_length], zero on padding


class PTFusion(nn.Module):
    def __init__(self, dim, hidden_dim, bias_init):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.pt_norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(3 * dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.gate[-1].bias, bias_init)

    def forward(self, query, pt, hla_summary, peptide_mask):
        valid = peptide_mask.bool().unsqueeze(-1)
        query = query.masked_fill(~valid, 0)
        pt = pt.masked_fill(~valid, 0)
        normalized_pt = self.pt_norm(pt)
        context = torch.cat((self.query_norm(query), normalized_pt,
                             hla_summary[:, None, :].expand_as(query)), dim=-1)
        beta = self.gate(context).float().sigmoid().masked_fill(~valid, 0)
        residual = self.projection(normalized_pt)
        fused = (query + beta.to(residual.dtype) * residual).masked_fill(~valid, 0)
        return fused, beta.squeeze(-1)


class StageMOEPTBindingModel(StageMOEBindingModel):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__(config)
        s = config.stage_MOE_PT
        self.pt_fusion_enabled = s.pt_fusion_enabled
        self.fusion = PTFusion(config.model.model_dim, s.fusion_hidden_dim, s.fusion_bias_init)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return StageMOEPTOutput(**vars(result))

    def _triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                 *, all_heads, return_attention=False, return_features=False):
        if not self.pt_fusion_enabled:
            result = super()._triplet(peptide_hidden, peptide_mask, hla_hidden, hla_mask,
                                      tcr_hidden, tcr_mask, all_heads=all_heads,
                                      return_attention=return_attention, return_features=return_features)
            return StageMOEPTOutput(**vars(result))
        p = self.peptide_encoder(peptide_hidden, peptide_mask)
        h = self.hla_encoder(hla_hidden, hla_mask)
        t = self.tcr_encoder(tcr_hidden, tcr_mask)
        pm, hm, tm = masked_mean(p, peptide_mask), masked_mean(h, hla_mask), masked_mean(t, tcr_mask)
        output = StageMOEPTOutput({}, {})
        zph = self._branch('ph', p, h, peptide_mask, hla_mask, [pm, hm], output,
                           return_attention, return_features, classify=all_heads)
        zpt = self._branch('pt', p, t, peptide_mask, tcr_mask, [pm, tm], output,
                           return_attention, return_features, classify=all_heads)
        hidden, weights = self.cross_attention['pht'](zph, t, peptide_mask, tcr_mask, return_attention)
        fused, output.fusion_beta = self.fusion(hidden, zpt, hm, peptide_mask)
        context = torch.cat([masked_mean(fused, peptide_mask), pm, hm, tm], -1)
        sequence, gates = self.moe['pht'](fused, context, peptide_mask, self.shared_experts, self.temperature)
        output.routing['pht'] = gates
        pooled, pool_weights = self.pooling['pht'](sequence, peptide_mask)
        output.logits['pht'] = self.classifiers['pht'](pooled).squeeze(-1)
        if return_attention:
            output.attention['pht'] = weights
            output.pooling_weights['pht'] = pool_weights
        if return_features:
            output.features['pht'] = pooled
        return output
