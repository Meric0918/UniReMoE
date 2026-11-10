"""Three random sequence encoders and the Stage Adapter PHT interaction path."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .attention import BindingClassifier, CrossAttentionBlock, MaskedAttentionPooling
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .stage1_encoder_model import SequenceEncoder

ARCHITECTURE = "stage_PHT"
SEQUENCE_ENCODER_NAMES = ("peptide_encoder", "hla_encoder", "tcr_encoder")
INTERACTION_MODULE_NAMES = (
    "pht_phla_cross_attention", "pht_cross_attention", "pht_pooling", "pht_classifier",
)


@dataclass
class StagePHTOutput:
    logits: torch.Tensor
    pooled_embedding: torch.Tensor


class StagePHTBindingModel(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__()
        self.model_config = asdict(config)
        for entity in ("peptide", "hla", "tcr"):
            setattr(self, f"{entity}_encoder", SequenceEncoder(
                getattr(config, f"{entity}_input_dim"),
                model_dim=config.model_dim,
                num_heads=config.num_attention_heads,
                feedforward_dim=config.feedforward_dim,
                num_layers=config.encoder_num_layers,
                dropout=config.encoder_dropout,
            ))
        attention = dict(
            model_dim=config.model_dim, num_heads=config.num_attention_heads,
            feedforward_dim=config.feedforward_dim, dropout=config.attention_dropout,
        )
        self.pht_phla_cross_attention = CrossAttentionBlock(**attention)
        self.pht_cross_attention = CrossAttentionBlock(**attention)
        self.pht_pooling = MaskedAttentionPooling(config.model_dim)
        self.pht_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)

    def forward_pht(self, peptide_hidden, peptide_mask, hla_hidden, hla_mask, tcr_hidden, tcr_mask):
        peptide = self.peptide_encoder(peptide_hidden, peptide_mask)
        hla = self.hla_encoder(hla_hidden, hla_mask)
        tcr = self.tcr_encoder(tcr_hidden, tcr_mask)
        ph_sequence, _ = self.pht_phla_cross_attention(peptide, hla, peptide_mask, hla_mask)
        pht_sequence, _ = self.pht_cross_attention(ph_sequence, tcr, peptide_mask, tcr_mask)
        pooled, _ = self.pht_pooling(pht_sequence, peptide_mask)
        return StagePHTOutput(self.pht_classifier(pooled), pooled)

    forward = forward_pht


class StagePHTModel(nn.Module):
    """Online frozen pretrained PLMs followed by the trainable PHT model."""

    architecture = ARCHITECTURE

    def __init__(self, binding_model, peptide_encoder, hla_encoder, tcr_encoder):
        super().__init__()
        self.binding_model = binding_model
        self.peptide_encoder = peptide_encoder
        self.hla_encoder = hla_encoder
        self.tcr_encoder = tcr_encoder
        self.set_trainable_pht()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep upstream dropout disabled even when a caller uses model.train().
        for name in SEQUENCE_ENCODER_NAMES:
            encoder = getattr(self, name)
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.set_active_training(False)
        return self

    def set_trainable_pht(self):
        self.train(True)
        self.binding_model.requires_grad_(True)

    def set_evaluation(self):
        self.eval()

    def encoder_metadata(self):
        return {entity: getattr(self, f"{entity}_encoder").metadata()
                for entity in ("peptide", "hla", "tcr")}

    def forward_pht(self, peptide_sequences, hla_sequences, tcr_sequences):
        with torch.no_grad():
            peptide = self.peptide_encoder(peptide_sequences)
            hla = self.hla_encoder(hla_sequences)
            tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_pht(
            peptide.hidden, peptide.mask, hla.hidden, hla.mask, tcr.hidden, tcr.mask,
        )

    forward = forward_pht


def build_stage_PHT_model(config, device: torch.device):
    dtype = {"none": "float32", "fp16": "float16", "bf16": "bfloat16"}[
        config.stage_PHT.mixed_precision
    ]
    if device.type != "cuda":
        dtype = "float32"
    kwargs = dict(device=device, dtype=dtype,
                  gradient_checkpointing=config.stage_PHT.gradient_checkpointing,
                  random_initialization=False)
    peptide = ESMCFineTuningEncoder(config.paths.esmc_model, config.model.peptide_max_length, **kwargs)
    hla = ESMCFineTuningEncoder(config.paths.esmc_model, config.model.hla_max_length, **kwargs)
    tcr = TCRBertFineTuningEncoder(config.paths.tcr_bert_model, config.model.tcr_max_length, **kwargs)
    for entity, encoder in (("peptide", peptide), ("hla", hla), ("tcr", tcr)):
        if encoder.hidden_size != getattr(config.model, f"{entity}_input_dim"):
            raise ValueError(f"{entity} PLM hidden size does not match config")
    return StagePHTModel(StagePHTBindingModel(config.model), peptide, hla, tcr).to(device)


def initialise_stage_PHT_random(model, *, config):
    model.set_trainable_pht()
    return {"kind": "random_binding", "seed": config.runtime.seed, "fold": config.runtime.fold}
