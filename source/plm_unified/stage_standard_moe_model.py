"""Scheme A branch-local sparse MoE within the unchanged Full fusion topology."""
from __future__ import annotations

import math

import torch
from torch import nn

from .stage1_encoder_model import SequenceEncoder
from .stage_MOE_model import TASKS, AttentionResidual, AttentionPooling, expert, masked_mean
from .stage_MOE_PT_model import PTFusion, StageMOEPTOutput as StageStandardMoEOutput
from .stage_MOE_consis_model import forward_batch

ARCHITECTURE = 'stage_standard_moe_v1'


class StandardBranchMoE(nn.Module):
    """One linear router per branch; dispatch only selected samples to each expert."""

    def __init__(self, dim, settings):
        super().__init__()
        self.top_k = settings.top_k
        self.norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList(
            [expert(dim, settings.expert_hidden_dim) for _ in range(settings.num_experts)])
        self.router = nn.Linear(dim, settings.num_experts)
        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, hidden, mask, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be finite and positive')
        valid = mask.bool().unsqueeze(-1)
        hidden = hidden.masked_fill(~valid, 0)
        # Route on the pre-LN sequence, just as the Full model summarizes its MoE input.
        logits = self.router(masked_mean(hidden, mask)).float() / temperature
        # Softmax over selected logits is the stable renormalization of Top-k softmax weights.
        selected_logits, selected_experts = logits.topk(self.top_k, dim=-1)
        weights = selected_logits.softmax(-1)
        gates = torch.zeros_like(logits).scatter(1, selected_experts, weights)
        normalized = self.norm(hidden).masked_fill(~valid, 0)
        # Accumulate in fp32 under bf16/fp16 autocast; never evaluate unselected experts.
        mixed = torch.zeros_like(normalized, dtype=torch.float32)
        for index, module in enumerate(self.experts):
            rows, slots = torch.where(selected_experts == index)
            if rows.numel():
                values = module(normalized.index_select(0, rows))
                weighted = values.float() * weights[rows, slots, None, None]
                mixed = mixed.index_add(0, rows, weighted)
        return (hidden + self.dropout(mixed.to(hidden.dtype))).masked_fill(~valid, 0), gates


class StageStandardMoEBindingModel(nn.Module):
    """Retain PH→PHT, PT fusion and all heads; replace only the MoE submodules."""

    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__()
        config.validate()
        m, s = config.model, config.stage_standard_moe
        self.temperature = s.temperature_start
        self.consistency_margin = s.consistency_margin
        self.pt_fusion_enabled = s.pt_fusion_enabled
        for entity in ('peptide', 'hla', 'tcr'):
            setattr(self, f'{entity}_encoder', SequenceEncoder(
                getattr(m, f'{entity}_input_dim'), m.model_dim, m.num_attention_heads,
                m.feedforward_dim, s.sequence_encoder_num_layers, s.dropout))
        self.cross_attention = nn.ModuleDict({
            task: AttentionResidual(m.model_dim, m.num_attention_heads, s.dropout) for task in TASKS})
        self.moe = nn.ModuleDict({task: StandardBranchMoE(m.model_dim, s) for task in TASKS})
        self.pooling = nn.ModuleDict({
            task: AttentionPooling(m.model_dim, s.pooling_hidden_dim) for task in TASKS})
        self.classifiers = nn.ModuleDict({task: nn.Sequential(
            nn.Linear(m.model_dim, s.classifier_hidden_dim), nn.GELU(), nn.Dropout(s.dropout),
            nn.Linear(s.classifier_hidden_dim, 1)) for task in TASKS})
        self.fusion = PTFusion(m.model_dim, s.fusion_hidden_dim, s.fusion_bias_init)

    def _finish_branch(self, task, hidden, mask, output, attention, features, classify=True):
        sequence, output.routing[task] = self.moe[task](hidden, mask, self.temperature)
        if classify:
            pooled, weights = self.pooling[task](sequence, mask)
            output.logits[task] = self.classifiers[task](pooled).squeeze(-1)
            if attention:
                output.pooling_weights[task] = weights
            if features:
                output.features[task] = pooled
        return sequence

    def _branch(self, task, query, receptor, mask, receptor_mask, output,
                attention, features, classify=True):
        hidden, weights = self.cross_attention[task](query, receptor, mask, receptor_mask, attention)
        if attention:
            output.attention[task] = weights
        return self._finish_branch(task, hidden, mask, output, attention, features, classify)

    def forward(self, task, peptide_hidden, peptide_mask, receptor_hidden, receptor_mask,
                *, return_attention=False, return_features=False):
        task = {'phla': 'ph', 'ptcr': 'pt'}.get(task, task)
        if task not in ('ph', 'pt'):
            raise ValueError('Pair task must be ph or pt')
        peptide = self.peptide_encoder(peptide_hidden, peptide_mask)
        encoder = self.hla_encoder if task == 'ph' else self.tcr_encoder
        receptor = encoder(receptor_hidden, receptor_mask)
        output = StageStandardMoEOutput({}, {})
        self._branch(task, peptide, receptor, peptide_mask, receptor_mask, output,
                     return_attention, return_features)
        return output

    def _triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                 *, all_heads, return_attention=False, return_features=False):
        p = self.peptide_encoder(peptide_hidden, peptide_mask)
        h = self.hla_encoder(hla_hidden, hla_mask)
        t = self.tcr_encoder(tcr_hidden, tcr_mask)
        output = StageStandardMoEOutput({}, {})
        zph = self._branch('ph', p, h, peptide_mask, hla_mask, output,
                           return_attention, return_features, classify=all_heads)
        zpt = None
        if self.pt_fusion_enabled or all_heads:
            zpt = self._branch('pt', p, t, peptide_mask, tcr_mask, output,
                               return_attention, return_features, classify=all_heads)
        hidden, weights = self.cross_attention['pht'](
            zph, t, peptide_mask, tcr_mask, return_attention)
        if self.pt_fusion_enabled:
            hidden, output.fusion_beta = self.fusion(hidden, zpt, masked_mean(h, hla_mask), peptide_mask)
        if return_attention:
            output.attention['pht'] = weights
        self._finish_branch('pht', hidden, peptide_mask, output, return_attention, return_features)
        return output

    def forward_triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask, **kwargs):
        return self._triplet(peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                             all_heads=False, **kwargs)

    def forward_all(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask, **kwargs):
        return self._triplet(peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                             all_heads=True, **kwargs)
