"""Stage8's PHT path with two independent pre-attention contrastive branches."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import SequenceAdapter, CrossAttentionBlock, MaskedAttentionPooling, BindingClassifier

ARCHITECTURE = "stage11"


class ContrastiveProjection(nn.Module):
    def __init__(self, model_dim, projection_dim):
        super().__init__()
        self.pooling = MaskedAttentionPooling(model_dim)
        self.projection = nn.Linear(model_dim, projection_dim, bias=False)

    def forward(self, sequence, mask):
        pooled, _ = self.pooling(sequence, mask)
        # Normalize in float32 even under autocast, including zero-vector inputs.
        return F.normalize(self.projection(pooled).float(), dim=-1, eps=1e-8)


@dataclass
class Stage11ContrastiveOutput:
    left: torch.Tensor
    right: torch.Tensor


@dataclass
class Stage11Output:
    pht_logits: torch.Tensor
    ph_attention: torch.Tensor | None = None
    pht_attention: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None


class Stage11BindingModel(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config, *, projection_dim=256):
        super().__init__()
        for entity in ("peptide", "hla", "tcr"):
            setattr(self, f"{entity}_adapter", SequenceAdapter(
                getattr(config, f"{entity}_input_dim"), config.model_dim,
                config.adapter_hidden_dim, config.adapter_dropout))
        kwargs = dict(model_dim=config.model_dim, num_heads=config.num_attention_heads,
                      feedforward_dim=config.feedforward_dim, dropout=config.attention_dropout)
        self.phla_cross_attention = CrossAttentionBlock(**kwargs)
        self.triplet_cross_attention = CrossAttentionBlock(**kwargs)
        self.triplet_pooling = MaskedAttentionPooling(config.model_dim)
        self.triplet_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)
        for name in ("p", "h", "ph", "t"):
            setattr(self, f"{name}_contrastive", ContrastiveProjection(config.model_dim, projection_dim))

    def _ph(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, *, return_attention=False):
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        return self.phla_cross_attention(peptide, hla, peptide_mask, hla_mask,
                                         return_attention=return_attention)

    def forward_ph_contrastive(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask):
        return Stage11ContrastiveOutput(
            self.p_contrastive(self.peptide_adapter(peptide_hidden, peptide_mask), peptide_mask),
            self.h_contrastive(self.hla_adapter(hla_hidden, hla_mask), hla_mask))

    def forward_pht_contrastive(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask,
                                tcr_hidden, tcr_mask):
        # PH and T are independently deduplicated: their batch dimensions can differ.
        ph, _ = self._ph(peptide_hidden, peptide_mask, hla_hidden, hla_mask)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        return Stage11ContrastiveOutput(self.ph_contrastive(ph, peptide_mask),
                                         self.t_contrastive(tcr, tcr_mask))

    def forward_pht(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask,
                    tcr_hidden, tcr_mask, *, return_attention=False):
        ph, ph_attention = self._ph(peptide_hidden, peptide_mask, hla_hidden, hla_mask,
                                     return_attention=return_attention)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        pht, pht_attention = self.triplet_cross_attention(
            ph, tcr, peptide_mask, tcr_mask, return_attention=return_attention)
        pooled, weights = self.triplet_pooling(pht, peptide_mask, return_weights=return_attention)
        return Stage11Output(self.triplet_classifier(pooled), ph_attention, pht_attention, weights)

    def forward(self, *args, **kwargs):
        return self.forward_pht(*args, **kwargs)
