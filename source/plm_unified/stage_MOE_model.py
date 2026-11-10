"""Stage8 topology with attention-only interaction blocks and dense residual MoE."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import nn

from .stage1_encoder_model import SequenceEncoder

ARCHITECTURE = 'stage_MOE_v1'
TASKS = ('ph', 'pt', 'pht')


def masked_mean(hidden, mask):
    mask = mask.bool()
    if (~mask.any(1)).any():
        raise ValueError('Every input sequence must contain a real residue')
    return hidden.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)


class AttentionResidual(nn.Module):
    """Document's pre-normalized query/key attention; no ordinary interaction FFN."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, query_mask, key_mask, return_attention=False):
        query_mask, key_mask = query_mask.bool(), key_mask.bool()
        if (~query_mask.any(1)).any() or (~key_mask.any(1)).any():
            raise ValueError('Attention inputs cannot contain empty sequences')
        q = self.query_norm(query.masked_fill(~query_mask.unsqueeze(-1), 0))
        k = self.key_norm(key.masked_fill(~key_mask.unsqueeze(-1), 0))
        update, weights = self.attention(q, k, k, key_padding_mask=~key_mask,
                                         need_weights=return_attention, average_attn_weights=False)
        return (query + self.dropout(update)).masked_fill(~query_mask.unsqueeze(-1), 0), weights


def expert(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class BranchMoE(nn.Module):
    def __init__(self, dim, context_dim, settings):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList([expert(dim, settings.expert_hidden_dim) for _ in range(2)])
        self.router = nn.Sequential(nn.Linear(context_dim, settings.gate_hidden_dim), nn.GELU(),
                                    nn.Linear(settings.gate_hidden_dim, 4))
        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, hidden, context, mask, shared_experts, temperature):
        if temperature <= 0:
            raise ValueError('Routing temperature must be positive')
        # Compute probabilities in fp32 even under autocast.
        gates = (self.router(context).float() / temperature).softmax(-1)
        normalized = self.norm(hidden)
        values = torch.stack([module(normalized) for module in (*shared_experts, *self.experts)], dim=2)
        mixed = (values * gates[:, None, :, None].to(values.dtype)).sum(2)
        return (hidden + self.dropout(mixed)).masked_fill(~mask.bool().unsqueeze(-1), 0), gates


class AttentionPooling(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, sequence, mask):
        if (~mask.bool().any(1)).any():
            raise ValueError('Cannot pool an empty sequence')
        scores = self.score(sequence).squeeze(-1).float().masked_fill(~mask.bool(), -torch.inf)
        weights = scores.softmax(-1)
        pooled = (sequence * weights.unsqueeze(-1).to(sequence.dtype)).sum(1)
        return pooled, weights


@dataclass
class StageMOEOutput:
    logits: dict[str, torch.Tensor]
    routing: dict[str, torch.Tensor]
    attention: dict[str, torch.Tensor] = field(default_factory=dict)
    pooling_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    features: dict[str, torch.Tensor] = field(default_factory=dict)


class StageMOEBindingModel(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__()
        m, s = config.model, config.stage_MOE
        self.temperature = s.temperature_start
        for entity in ('peptide', 'hla', 'tcr'):
            setattr(self, f'{entity}_encoder', SequenceEncoder(
                getattr(m, f'{entity}_input_dim'), m.model_dim, m.num_attention_heads,
                m.feedforward_dim, s.sequence_encoder_num_layers, s.dropout))
        self.shared_experts = nn.ModuleList([expert(m.model_dim, s.expert_hidden_dim) for _ in range(2)])
        self.cross_attention = nn.ModuleDict({t: AttentionResidual(m.model_dim, m.num_attention_heads, s.dropout) for t in TASKS})
        self.moe = nn.ModuleDict({t: BranchMoE(m.model_dim, m.model_dim * (4 if t == 'pht' else 3), s) for t in TASKS})
        self.pooling = nn.ModuleDict({t: AttentionPooling(m.model_dim, s.pooling_hidden_dim) for t in TASKS})
        self.classifiers = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(m.model_dim, s.classifier_hidden_dim), nn.GELU(), nn.Dropout(s.dropout),
            nn.Linear(s.classifier_hidden_dim, 1)) for t in TASKS})

    def _branch(self, task, query, receptor, peptide_mask, receptor_mask, summaries, output, attention, features, classify=True):
        hidden, weights = self.cross_attention[task](query, receptor, peptide_mask, receptor_mask, attention)
        context = torch.cat([masked_mean(hidden, peptide_mask), *summaries], -1)
        sequence, gates = self.moe[task](hidden, context, peptide_mask, self.shared_experts, self.temperature)
        output.routing[task] = gates
        if attention:
            output.attention[task] = weights
        if classify:
            pooled, pool_weights = self.pooling[task](sequence, peptide_mask)
            output.logits[task] = self.classifiers[task](pooled).squeeze(-1)
            if attention:
                output.pooling_weights[task] = pool_weights
            if features:
                output.features[task] = pooled
        return sequence

    def forward(self, task, peptide_hidden, peptide_mask, receptor_hidden, receptor_mask,
                *, return_attention=False, return_features=False):
        task = {'phla': 'ph', 'ptcr': 'pt'}.get(task, task)
        if task not in ('ph', 'pt'):
            raise ValueError('Pair task must be ph or pt')
        peptide = self.peptide_encoder(peptide_hidden, peptide_mask)
        encoder = self.hla_encoder if task == 'ph' else self.tcr_encoder
        receptor = encoder(receptor_hidden, receptor_mask)
        output = StageMOEOutput({}, {})
        self._branch(task, peptide, receptor, peptide_mask, receptor_mask,
                     [masked_mean(peptide, peptide_mask), masked_mean(receptor, receptor_mask)],
                     output, return_attention, return_features)
        return output

    def _triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                 *, all_heads, return_attention=False, return_features=False):
        p = self.peptide_encoder(peptide_hidden, peptide_mask)
        h = self.hla_encoder(hla_hidden, hla_mask)
        t = self.tcr_encoder(tcr_hidden, tcr_mask)
        pm, hm, tm = masked_mean(p, peptide_mask), masked_mean(h, hla_mask), masked_mean(t, tcr_mask)
        output = StageMOEOutput({}, {})
        zph = self._branch('ph', p, h, peptide_mask, hla_mask, [pm, hm], output,
                           return_attention, return_features, classify=all_heads)
        if all_heads:
            self._branch('pt', p, t, peptide_mask, tcr_mask, [pm, tm], output, return_attention, return_features)
        self._branch('pht', zph, t, peptide_mask, tcr_mask, [pm, hm, tm], output, return_attention, return_features)
        return output

    def forward_triplet(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask, **kwargs):
        return self._triplet(peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                             all_heads=False, **kwargs)

    def forward_all(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask, **kwargs):
        return self._triplet(peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask,
                             all_heads=True, **kwargs)


def forward_batch(model, task, batch):
    p = (batch['peptide_hidden'], batch['peptide_mask'])
    if task == 'pht':
        return model.forward_triplet(*p, batch['hla_hidden'], batch['hla_mask'], batch['tcr_hidden'], batch['tcr_mask'])
    entity = 'hla' if task == 'ph' else 'tcr'
    return model(task, *p, batch[f'{entity}_hidden'], batch[f'{entity}_mask'])
