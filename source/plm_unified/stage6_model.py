"""Stage-6 model with condition-specific downstream trainability."""

from __future__ import annotations

from typing import Literal

import torch

from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .stage6_config import Stage6ExperimentConfig
from .triplet_model import ProgressiveTripletFineTuningModel, TripletBindingModel

Stage6Phase = Literal["stage6_1a", "stage6_1b"]
Stage6TaskMode = Literal["joint", "phla_only"]


class Stage6FineTuningModel(ProgressiveTripletFineTuningModel):
    """Frozen PLMs plus either all downstream tasks or the PH branch only."""

    binding_model: TripletBindingModel

    def fgm_modules(self, task_mode: Stage6TaskMode) -> dict[str, torch.nn.Module]:
        modules = {
            "peptide_adapter": self.binding_model.peptide_adapter,
            "hla_adapter": self.binding_model.hla_adapter,
        }
        if task_mode == "joint":
            modules["tcr_adapter"] = self.binding_model.tcr_adapter
        return modules

    def set_trainable_stage6(
        self,
        phase: Stage6Phase,
        task_mode: Stage6TaskMode,
    ) -> None:
        if phase not in {"stage6_1a", "stage6_1b"}:
            raise ValueError(f"Unknown Stage-6 phase: {phase}")
        if task_mode not in {"joint", "phla_only"}:
            raise ValueError(f"Unknown Stage-6 task mode: {task_mode}")
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
        if task_mode == "joint":
            self.binding_model.requires_grad_(True)
            self.binding_model.train(True)
            return
        phla_modules = (
            self.binding_model.peptide_adapter,
            self.binding_model.hla_adapter,
            self.binding_model.phla_cross_attention,
            self.binding_model.phla_pooling,
            self.binding_model.phla_classifier,
        )
        for module in phla_modules:
            module.requires_grad_(True)
            module.train(True)


def build_stage6_model(
    config: Stage6ExperimentConfig,
    device: torch.device,
) -> Stage6FineTuningModel:
    precision_to_encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    dtype = precision_to_encoder_dtype[config.stage6.mixed_precision]
    encoder_kwargs = {
        "device": device,
        "dtype": dtype,
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
    model = Stage6FineTuningModel(
        TripletBindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=0,
        hla_unfrozen_layers=0,
        tcr_unfrozen_layers=0,
    )
    return model.to(device)
