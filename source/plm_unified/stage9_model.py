"""Stage-9 PH--TCR contrastive heads layered on the Stage-3 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MaskedAttentionPooling
from .config import ModelConfig
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .stage9_config import Stage9ExperimentConfig
from .triplet_model import (
    ProgressiveTripletFineTuningModel,
    TripletBindingModel,
)


Stage9Phase = Literal["stage9a", "stage9b"]
Stage9Task = Literal["pair", "phla", "ptcr", "triplet"]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        projection_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(pooled), p=2, dim=-1)


@dataclass
class Stage9PairOutput:
    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    z_ph: torch.Tensor
    z_t_positive: torch.Tensor
    z_t_negative: torch.Tensor
    positive_similarity: torch.Tensor
    negative_similarity: torch.Tensor


@dataclass
class Stage9EmbeddingOutput:
    logits: torch.Tensor
    z_ph: torch.Tensor
    z_tcr: torch.Tensor
    similarity: torch.Tensor


class Stage9BindingModel(TripletBindingModel):
    """Stage-3 binding model with isolated PH and TCR projection heads."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        projection_dim: int = 128,
        projection_dropout: float = 0.1,
    ):
        super().__init__(config)
        self.contrastive_ph_pooling = MaskedAttentionPooling(config.model_dim)
        self.contrastive_tcr_pooling = MaskedAttentionPooling(config.model_dim)
        self.ph_projector = ProjectionHead(
            config.model_dim, projection_dim, projection_dropout
        )
        self.tcr_projector = ProjectionHead(
            config.model_dim, projection_dim, projection_dropout
        )

    def contrastive_modules(self) -> dict[str, nn.Module]:
        return {
            "contrastive_ph_pooling": self.contrastive_ph_pooling,
            "contrastive_tcr_pooling": self.contrastive_tcr_pooling,
            "ph_projector": self.ph_projector,
            "tcr_projector": self.tcr_projector,
        }

    def set_trainable_stage9a(self) -> None:
        self.requires_grad_(False)
        self.eval()
        for module in self.contrastive_modules().values():
            module.requires_grad_(True)
            module.train(True)

    def _ph_sequence(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        phla_sequence, _ = self.phla_cross_attention(
            peptide,
            hla,
            peptide_mask,
            hla_mask,
            return_attention=False,
        )
        phla_pooled, _ = self.contrastive_ph_pooling(
            phla_sequence, peptide_mask, return_weights=False
        )
        return phla_sequence, self.ph_projector(phla_pooled), peptide_mask

    def _adapt_and_project_tcr(
        self,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        pooled, _ = self.contrastive_tcr_pooling(
            tcr, tcr_mask, return_weights=False
        )
        return tcr, self.tcr_projector(pooled)

    def _triplet_logits(
        self,
        phla_sequence: torch.Tensor,
        peptide_mask: torch.Tensor,
        tcr: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> torch.Tensor:
        complex_sequence, _ = self.triplet_cross_attention(
            phla_sequence,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        pooled, _ = self.triplet_pooling(
            complex_sequence, peptide_mask, return_weights=False
        )
        return self.triplet_classifier(pooled)

    def forward_pair(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        positive_tcr_hidden: torch.Tensor,
        positive_tcr_mask: torch.Tensor,
        negative_tcr_hidden: torch.Tensor,
        negative_tcr_mask: torch.Tensor,
    ) -> Stage9PairOutput:
        phla_sequence, z_ph, peptide_mask = self._ph_sequence(
            peptide_hidden, peptide_mask, hla_hidden, hla_mask
        )
        positive_tcr, z_t_positive = self._adapt_and_project_tcr(
            positive_tcr_hidden, positive_tcr_mask
        )
        negative_tcr, z_t_negative = self._adapt_and_project_tcr(
            negative_tcr_hidden, negative_tcr_mask
        )
        positive_logits = self._triplet_logits(
            phla_sequence,
            peptide_mask,
            positive_tcr,
            positive_tcr_mask,
        )
        negative_logits = self._triplet_logits(
            phla_sequence,
            peptide_mask,
            negative_tcr,
            negative_tcr_mask,
        )
        return Stage9PairOutput(
            positive_logits=positive_logits,
            negative_logits=negative_logits,
            z_ph=z_ph,
            z_t_positive=z_t_positive,
            z_t_negative=z_t_negative,
            positive_similarity=(z_ph * z_t_positive).sum(dim=-1),
            negative_similarity=(z_ph * z_t_negative).sum(dim=-1),
        )

    def forward_embedding(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> Stage9EmbeddingOutput:
        phla_sequence, z_ph, peptide_mask = self._ph_sequence(
            peptide_hidden, peptide_mask, hla_hidden, hla_mask
        )
        tcr, z_tcr = self._adapt_and_project_tcr(tcr_hidden, tcr_mask)
        return Stage9EmbeddingOutput(
            logits=self._triplet_logits(
                phla_sequence, peptide_mask, tcr, tcr_mask
            ),
            z_ph=z_ph,
            z_tcr=z_tcr,
            similarity=(z_ph * z_tcr).sum(dim=-1),
        )


class ProgressiveStage9FineTuningModel(ProgressiveTripletFineTuningModel):
    """Online Stage-9 PLMs with frozen encoders and paired forward."""

    binding_model: Stage9BindingModel

    def set_trainable_stage9(
        self,
        task: Stage9Task,
        phase: Stage9Phase,
    ) -> None:
        if phase not in {"stage9a", "stage9b"}:
            raise ValueError(f"Unknown Stage-9 phase: {phase}")
        if phase == "stage9a" and task != "pair":
            raise ValueError("Stage 9A only trains the paired contrastive task")
        self._active_stage = phase  # type: ignore[assignment]
        self.requires_grad_(False)
        self.eval()
        for encoder in (
            self.peptide_encoder,
            self.hla_encoder,
            self.tcr_encoder,
        ):
            encoder.set_active_training(False)

        binding = self.binding_model
        if phase == "stage9a":
            active_modules = binding.contrastive_modules()
        elif task == "pair":
            active_modules = {
                "peptide_adapter": binding.peptide_adapter,
                "hla_adapter": binding.hla_adapter,
                "tcr_adapter": binding.tcr_adapter,
                "phla_cross_attention": binding.phla_cross_attention,
                "triplet_cross_attention": binding.triplet_cross_attention,
                "triplet_pooling": binding.triplet_pooling,
                "triplet_classifier": binding.triplet_classifier,
                **binding.contrastive_modules(),
            }
        elif task == "triplet":
            active_modules = binding.triplet_modules()
        elif task in {"phla", "ptcr"}:
            active_modules = binding.task_modules(task)
        else:
            raise ValueError(f"Unknown Stage-9 task: {task}")
        for module in active_modules.values():
            module.requires_grad_(True)
            module.train(True)

        # Stage 9 deliberately keeps all three PLM encoders frozen. The
        # encoder layer-count fields are retained only for compatibility with
        # older configs/checkpoints; no Stage 9 task activates encoder blocks.

    def forward_pair(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        positive_tcr_sequences: Sequence[str],
        negative_tcr_sequences: Sequence[str],
    ) -> Stage9PairOutput:
        batch_size = len(peptide_sequences)
        if not (
            len(hla_sequences)
            == len(positive_tcr_sequences)
            == len(negative_tcr_sequences)
            == batch_size
        ):
            raise ValueError("All Stage-9 pair sequence batches must have equal size")
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        combined_tcr = self.tcr_encoder(
            list(positive_tcr_sequences) + list(negative_tcr_sequences)
        )
        return self.binding_model.forward_pair(
            peptide.hidden,
            peptide.mask,
            hla.hidden,
            hla.mask,
            combined_tcr.hidden[:batch_size],
            combined_tcr.mask[:batch_size],
            combined_tcr.hidden[batch_size:],
            combined_tcr.mask[batch_size:],
        )

    def encode_triplet(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
    ) -> Stage9EmbeddingOutput:
        if not len(peptide_sequences) == len(hla_sequences) == len(tcr_sequences):
            raise ValueError("All Stage-9 embedding batches must have equal size")
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_embedding(
            peptide.hidden,
            peptide.mask,
            hla.hidden,
            hla.mask,
            tcr.hidden,
            tcr.mask,
        )


def build_stage9_model(
    config: Stage9ExperimentConfig,
    device: torch.device,
) -> ProgressiveStage9FineTuningModel:
    precision_to_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    values = config.stage9
    encoder_kwargs = {
        "device": device,
        "dtype": precision_to_dtype[values.mixed_precision],
        "gradient_checkpointing": values.gradient_checkpointing,
    }
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.peptide_max_length,
        **encoder_kwargs,
    )
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.hla_max_length,
        **encoder_kwargs,
    )
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model,
        config.model.tcr_max_length,
        **encoder_kwargs,
    )
    expected_dims = {
        "peptide": (peptide_encoder.hidden_size, config.model.peptide_input_dim),
        "hla": (hla_encoder.hidden_size, config.model.hla_input_dim),
        "tcr": (tcr_encoder.hidden_size, config.model.tcr_input_dim),
    }
    for entity, (actual, expected) in expected_dims.items():
        if actual != expected:
            raise ValueError(
                f"{entity} encoder hidden size {actual} does not match {expected}"
            )
    model = ProgressiveStage9FineTuningModel(
        Stage9BindingModel(
            config.model,
            projection_dim=values.projection_dim,
            projection_dropout=values.projection_dropout,
        ),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=values.peptide_unfrozen_layers,
        hla_unfrozen_layers=values.hla_unfrozen_layers,
        tcr_unfrozen_layers=values.tcr_unfrozen_layers,
    )
    return model.to(device)
