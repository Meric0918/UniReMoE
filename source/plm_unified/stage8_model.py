"""Cached-feature Stage-8 model with a shared single-pass three-head forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .config import ModelConfig
from .finetuning import (
    ESMCFineTuningEncoder,
    ProgressiveFineTuningModel,
    TCRBertFineTuningEncoder,
)
from .triplet_model import (
    TripletBindingModel,
    initialise_binding_from_stage1,
)

if False:  # pragma: no cover - avoids a runtime configuration import cycle.
    from .stage8_config import Stage8ExperimentConfig


Stage8Phase = Literal["stage8a", "stage8b"]
Stage8Task = Literal["phla", "ptcr", "joint"]


@dataclass
class Stage8BindingOutput:
    phla_logits: torch.Tensor
    ptcr_logits: torch.Tensor
    pht_logits: torch.Tensor


class Stage8BindingModel(TripletBindingModel):
    """Stage-3 downstream modules trained against frozen residue caches."""

    def forward_all(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> Stage8BindingOutput:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)

        phla_sequence, _ = self.phla_cross_attention(
            peptide, hla, peptide_mask, hla_mask, return_attention=False
        )
        phla_pooled, _ = self.phla_pooling(
            phla_sequence, peptide_mask, return_weights=False
        )
        ptcr_sequence, _ = self.ptcr_cross_attention(
            peptide, tcr, peptide_mask, tcr_mask, return_attention=False
        )
        ptcr_pooled, _ = self.ptcr_pooling(
            ptcr_sequence, peptide_mask, return_weights=False
        )
        pht_sequence, _ = self.triplet_cross_attention(
            phla_sequence, tcr, peptide_mask, tcr_mask, return_attention=False
        )
        pht_pooled, _ = self.triplet_pooling(
            pht_sequence, peptide_mask, return_weights=False
        )
        return Stage8BindingOutput(
            phla_logits=self.phla_classifier(phla_pooled),
            ptcr_logits=self.ptcr_classifier(ptcr_pooled),
            pht_logits=self.triplet_classifier(pht_pooled),
        )


def build_stage8_binding_model(
    config: ModelConfig, stage3_binding_state: dict, *, device: torch.device
) -> Stage8BindingModel:
    model = Stage8BindingModel(config)
    model.load_state_dict(stage3_binding_state, strict=True)
    model.requires_grad_(True)
    return model.to(device)


class ProgressiveStage8FineTuningModel(ProgressiveFineTuningModel):
    """Online Stage-8 model with Stage8A/Stage8B trainability controls."""

    binding_model: Stage8BindingModel

    def initialise_triplet_from_ptcr(self) -> None:
        self.binding_model.initialise_triplet_from_ptcr()

    def set_trainable_stage8(self, task: Stage8Task, phase: Stage8Phase) -> None:
        if phase not in {"stage8a", "stage8b"}:
            raise ValueError(f"Unknown Stage-8 phase: {phase}")
        if task not in {"phla", "ptcr", "joint"}:
            raise ValueError(f"Unknown Stage-8 task: {task}")
        if phase == "stage8a" and task != "joint":
            raise ValueError("Stage 8A only trains the joint task")

        self._active_stage = phase
        self.requires_grad_(False)
        self.eval()
        for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
            encoder.set_active_training(False)

        if phase == "stage8a":
            active_modules = self.binding_model.new_triplet_modules()
        elif task == "phla":
            active_modules = self.binding_model.task_modules("phla")
        elif task == "ptcr":
            active_modules = self.binding_model.task_modules("ptcr")
        else:
            active_modules = dict(self.binding_model.named_children())
        for module in active_modules.values():
            module.requires_grad_(True)
            module.train(True)

        if phase == "stage8b":
            self._activate_encoder(
                self.peptide_encoder, self.peptide_unfrozen_layers
            )
            if task in {"phla", "joint"}:
                self._activate_encoder(self.hla_encoder, self.hla_unfrozen_layers)
            if task in {"ptcr", "joint"}:
                self._activate_encoder(self.tcr_encoder, self.tcr_unfrozen_layers)

    def forward_all(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
    ) -> Stage8BindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_all(
            peptide.hidden,
            peptide.mask,
            hla.hidden,
            hla.mask,
            tcr.hidden,
            tcr.mask,
        )


def initialise_stage8_from_stage1(
    model: ProgressiveStage8FineTuningModel,
    checkpoint_path: str,
    *,
    model_config: ModelConfig,
    fold: int,
    seed: int,
    expected_sha256: str,
) -> dict:
    """Initialise Stage 8 directly from its pinned Stage-1 best_joint source."""

    if expected_sha256 is None:
        raise ValueError("Stage-8 expected SHA256 is required")
    try:
        return initialise_binding_from_stage1(
            model.binding_model,
            checkpoint_path,
            model_config=model_config,
            fold=fold,
            seed=seed,
            encoder_metadata=model.encoder_metadata(),
            expected_sha256=expected_sha256,
            require_joint_checkpoint=True,
        )
    except (ValueError, RuntimeError) as error:
        message = str(error).replace("Stage-3", "Stage-8").replace(
            "Stage 3", "Stage-8"
        )
        raise type(error)(message) from error


def build_progressive_stage8_model(
    config: "Stage8ExperimentConfig",
    device: torch.device,
) -> ProgressiveStage8FineTuningModel:
    precision_to_encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    encoder_kwargs = {
        "device": device,
        "dtype": precision_to_encoder_dtype[config.stage8.mixed_precision],
        "gradient_checkpointing": config.stage8.gradient_checkpointing,
    }
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.peptide_max_length, **encoder_kwargs
    )
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.hla_max_length, **encoder_kwargs
    )
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model, config.model.tcr_max_length, **encoder_kwargs
    )
    dimensions = (
        ("Peptide ESM-C", peptide_encoder.hidden_size, config.model.peptide_input_dim),
        ("HLA ESM-C", hla_encoder.hidden_size, config.model.hla_input_dim),
        ("TCR-BERT", tcr_encoder.hidden_size, config.model.tcr_input_dim),
    )
    for name, actual, expected in dimensions:
        if actual != expected:
            raise ValueError(f"{name} hidden size does not match model config")
    model = ProgressiveStage8FineTuningModel(
        Stage8BindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=config.stage8.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.stage8.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.stage8.tcr_unfrozen_layers,
    )
    return model.to(device)
