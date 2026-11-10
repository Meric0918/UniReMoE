"""Stage-5 model with frozen pretrained PLMs and random downstream modules."""

from __future__ import annotations

from typing import Literal

import torch

from .config import ExperimentConfig
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .triplet_model import (
    ProgressiveTripletFineTuningModel,
    TripletBindingModel,
)

Stage5Phase = Literal["stage5_1a", "stage5_1b"]


class Stage5FineTuningModel(ProgressiveTripletFineTuningModel):
    """Stage-3-compatible model whose three PLMs stay frozen in Stage 5."""

    binding_model: TripletBindingModel

    def fgm_modules(self) -> dict[str, torch.nn.Module]:
        """Return exactly the three adapters perturbed during Stage5-1B."""

        return {
            "peptide_adapter": self.binding_model.peptide_adapter,
            "hla_adapter": self.binding_model.hla_adapter,
            "tcr_adapter": self.binding_model.tcr_adapter,
        }

    def set_trainable_stage5(self, phase: Stage5Phase) -> None:
        if phase not in {"stage5_1a", "stage5_1b"}:
            raise ValueError(f"Unknown Stage-5 phase: {phase}")
        self._active_stage = phase
        if self._stage2a_hla_cache:
            self.clear_stage2a_hla_cache()
        self.requires_grad_(False)
        self.eval()
        for encoder in (
            self.peptide_encoder,
            self.hla_encoder,
            self.tcr_encoder,
        ):
            encoder.set_active_training(False)
        self.binding_model.requires_grad_(True)
        self.binding_model.train(True)


def build_stage5_model(
    config: ExperimentConfig,
    device: torch.device,
) -> Stage5FineTuningModel:
    precision_to_encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    dtype = precision_to_encoder_dtype[config.stage5.mixed_precision]
    encoder_kwargs = {
        "device": device,
        "dtype": dtype,
        # Frozen encoders run under no_grad, so activation checkpointing only
        # adds overhead and is intentionally disabled for Stage 5.
        "gradient_checkpointing": False,
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
    if peptide_encoder.hidden_size != config.model.peptide_input_dim:
        raise ValueError("Peptide ESM-C hidden size does not match model config")
    if hla_encoder.hidden_size != config.model.hla_input_dim:
        raise ValueError("HLA ESM-C hidden size does not match model config")
    if tcr_encoder.hidden_size != config.model.tcr_input_dim:
        raise ValueError("TCR-BERT hidden size does not match model config")
    model = Stage5FineTuningModel(
        TripletBindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=0,
        hla_unfrozen_layers=0,
        tcr_unfrozen_layers=0,
    )
    return model.to(device)
