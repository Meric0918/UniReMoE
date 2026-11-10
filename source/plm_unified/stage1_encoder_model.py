"""Stage1 with three sequence encoders over frozen PLM residue features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .attention import BindingClassifier, CrossAttentionBlock, MaskedAttentionPooling
from .model import Task, _masked_mean, extract_task_state_dict
from .stage1_encoder_config import Stage1EncoderModelConfig

ARCHITECTURE = "stage1_encoder"
ENCODER_UMAP_STAGES = ("plm", "sequence_encoder", "classifier_input")
ENCODER_TASK_MODULE_NAMES: dict[Task, tuple[str, ...]] = {
    "phla": (
        "peptide_encoder", "hla_encoder", "phla_cross_attention",
        "phla_pooling", "phla_classifier",
    ),
    "ptcr": (
        "peptide_encoder", "tcr_encoder", "ptcr_cross_attention",
        "ptcr_pooling", "ptcr_classifier",
    ),
}


class SequenceEncoder(nn.Module):
    """Project PLM features and contextualize residues using masked self-attention.

    PLM features already contain sequence position information. Each transformer
    layer uses pre-normalization, with a final normalization after the stack.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim), ("model_dim", model_dim),
            ("num_heads", num_heads), ("feedforward_dim", feedforward_dim),
            ("num_layers", num_layers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError(f"hidden must have shape [B, L, {self.input_dim}]")
        if mask.ndim != 2 or mask.shape != hidden.shape[:2]:
            raise ValueError("mask must have shape [B, L] matching hidden")
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("Every encoder sequence must contain at least one residue")
        padding = ~mask.unsqueeze(-1)
        output = self.projection(hidden.masked_fill(padding, 0.0))
        output = output.masked_fill(padding, 0.0)
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=~mask)
            output = output.masked_fill(padding, 0.0)
        return self.final_norm(output).masked_fill(padding, 0.0)


@dataclass
class EncoderBindingOutput:
    logits: torch.Tensor
    pooled_embedding: torch.Tensor
    cross_attention: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None
    plm_embedding: torch.Tensor | None = None
    sequence_encoder_embedding: torch.Tensor | None = None


class Stage1EncoderBindingModel(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config: Stage1EncoderModelConfig, *, _task: Task | None = None):
        super().__init__()
        if _task is not None and _task not in ENCODER_TASK_MODULE_NAMES:
            raise ValueError(f"Unknown task: {_task}")
        self.task = _task
        encoder_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "num_layers": config.encoder_num_layers,
            "dropout": config.encoder_dropout,
        }
        self.peptide_encoder = SequenceEncoder(config.peptide_input_dim, **encoder_kwargs)
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        for task in ("phla", "ptcr") if _task is None else (_task,):
            receptor = "hla" if task == "phla" else "tcr"
            setattr(self, f"{receptor}_encoder", SequenceEncoder(
                getattr(config, f"{receptor}_input_dim"), **encoder_kwargs,
            ))
            setattr(self, f"{task}_cross_attention", CrossAttentionBlock(**attention_kwargs))
            setattr(self, f"{task}_pooling", MaskedAttentionPooling(config.model_dim))
            setattr(self, f"{task}_classifier", BindingClassifier(
                config.model_dim, config.classifier_dropout,
            ))

    def forward(
        self,
        task: Task,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_hidden: torch.Tensor,
        receptor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
        return_features: bool = False,
    ) -> EncoderBindingOutput:
        self._validate_task(task)
        peptide = self.peptide_encoder(peptide_hidden, peptide_mask)
        receptor_name = "hla" if task == "phla" else "tcr"
        receptor = getattr(self, f"{receptor_name}_encoder")(receptor_hidden, receptor_mask)
        sequence, attention = getattr(self, f"{task}_cross_attention")(
            peptide, receptor, peptide_mask, receptor_mask,
            return_attention=return_attention,
        )
        pooled, weights = getattr(self, f"{task}_pooling")(
            sequence, peptide_mask, return_weights=return_attention,
        )
        plm_embedding = sequence_encoder_embedding = None
        if return_features:
            plm_embedding = torch.cat((
                _masked_mean(peptide_hidden, peptide_mask),
                _masked_mean(receptor_hidden, receptor_mask),
            ), dim=-1)
            sequence_encoder_embedding = torch.cat((
                _masked_mean(peptide, peptide_mask),
                _masked_mean(receptor, receptor_mask),
            ), dim=-1)
        return EncoderBindingOutput(
            logits=getattr(self, f"{task}_classifier")(pooled),
            pooled_embedding=pooled,
            cross_attention=attention,
            pooling_weights=weights,
            plm_embedding=plm_embedding,
            sequence_encoder_embedding=sequence_encoder_embedding,
        )

    def _validate_task(self, task: Task) -> None:
        if task not in ENCODER_TASK_MODULE_NAMES:
            raise ValueError(f"Unknown task: {task}")
        if self.task is not None and task != self.task:
            raise ValueError(f"This checkpoint is for task '{self.task}', not '{task}'")

    def task_modules(self, task: Task) -> dict[str, nn.Module]:
        self._validate_task(task)
        return {name: getattr(self, name) for name in ENCODER_TASK_MODULE_NAMES[task]}

    def fgm_modules(self, task: Task) -> dict[str, nn.Module]:
        modules = self.task_modules(task)
        return {name: module for name, module in modules.items() if name.endswith("_encoder")}

    def set_trainable_task(self, task: Task, *, freeze_shared: bool = False) -> None:
        active_modules = self.task_modules(task)
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, module in active_modules.items():
            if freeze_shared and name == "peptide_encoder":
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, value in self.named_parameters() if value.requires_grad]


class TaskSpecificStage1EncoderBindingModel(Stage1EncoderBindingModel):
    """A deployable encoder branch without the other task's private modules."""

    def __init__(self, config: Stage1EncoderModelConfig, task: Task):
        if task not in ENCODER_TASK_MODULE_NAMES:
            raise ValueError(f"Unknown task: {task}")
        super().__init__(config, _task=task)


def extract_encoder_task_state_dict(state_dict: dict, task: Task) -> dict:
    return extract_task_state_dict(state_dict, task, module_names=ENCODER_TASK_MODULE_NAMES)


def validate_encoder_checkpoint(checkpoint: Mapping, config) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected architecture=stage1_encoder checkpoint")
    if checkpoint.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("Checkpoint model configuration does not match current config")
