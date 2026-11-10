"""Stage15 pairwise binding model with one PH/PT Shared Hyper-PMT."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .attention import BindingClassifier, MaskedAttentionPooling, SequenceAdapter
from .config import ModelConfig
from .hyper_pmt import ENTITY_ORDER, EntityKind, HyperPMTFusionEncoder
from .model import BindingOutput, Task, _masked_mean
from .stage15_config import Stage15FusionConfig


STAGE15_TASK_MODULE_NAMES: dict[Task, tuple[str, ...]] = {
    "phla": (
        "peptide_adapter", "hla_adapter", "hypergraph_fusion",
        "phla_pooling", "phla_classifier",
    ),
    "ptcr": (
        "peptide_adapter", "tcr_adapter", "hypergraph_fusion",
        "ptcr_pooling", "ptcr_classifier",
    ),
}


class Stage15BindingModel(nn.Module):
    """Frozen-PLM downstream model trained from residue embedding caches."""

    def __init__(
        self,
        config: ModelConfig,
        fusion: Stage15FusionConfig | None = None,
    ):
        super().__init__()
        fusion = fusion or Stage15FusionConfig()
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        self.peptide_adapter = SequenceAdapter(
            config.peptide_input_dim, **adapter_kwargs
        )
        self.hla_adapter = SequenceAdapter(config.hla_input_dim, **adapter_kwargs)
        self.tcr_adapter = SequenceAdapter(config.tcr_input_dim, **adapter_kwargs)
        self.hypergraph_fusion = HyperPMTFusionEncoder(config, fusion)
        self.phla_pooling = MaskedAttentionPooling(config.model_dim)
        self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
        self.phla_classifier = BindingClassifier(
            config.model_dim, config.classifier_dropout
        )
        self.ptcr_classifier = BindingClassifier(
            config.model_dim, config.classifier_dropout
        )

    @staticmethod
    def _concat(
        states: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kinds = [kind for kind in ENTITY_ORDER if kind in states]
        return (
            torch.cat([states[kind] for kind in kinds], dim=1),
            torch.cat([masks[kind].bool() for kind in kinds], dim=1),
        )

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
    ) -> BindingOutput:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        if task == "phla":
            receptor_kind: EntityKind = "hla"
            receptor = self.hla_adapter(receptor_hidden, receptor_mask)
            pooling = self.phla_pooling
            classifier = self.phla_classifier
        elif task == "ptcr":
            receptor_kind = "tcr"
            receptor = self.tcr_adapter(receptor_hidden, receptor_mask)
            pooling = self.ptcr_pooling
            classifier = self.ptcr_classifier
        else:
            raise ValueError(f"Unknown task: {task}")
        entities: dict[EntityKind, torch.Tensor] = {
            "peptide": peptide,
            receptor_kind: receptor,
        }
        masks: dict[EntityKind, torch.Tensor] = {
            "peptide": peptide_mask.bool(),
            receptor_kind: receptor_mask.bool(),
        }
        encoding = self.hypergraph_fusion(entities, masks)
        sequence, sequence_mask = self._concat(encoding.states, masks)
        pooled, pooling_weights = pooling(
            sequence,
            sequence_mask,
            return_weights=return_attention,
        )
        encoder_embedding = adapter_embedding = None
        if return_features:
            encoder_embedding = torch.cat(
                (
                    _masked_mean(peptide_hidden, peptide_mask),
                    _masked_mean(receptor_hidden, receptor_mask),
                ),
                dim=-1,
            )
            adapter_embedding = torch.cat(
                (
                    _masked_mean(peptide, peptide_mask),
                    _masked_mean(receptor, receptor_mask),
                ),
                dim=-1,
            )
        return BindingOutput(
            logits=classifier(pooled),
            pooled_embedding=pooled,
            cross_attention=None,
            pooling_weights=pooling_weights,
            encoder_embedding=encoder_embedding,
            adapter_embedding=adapter_embedding,
        )

    def task_modules(self, task: Task) -> dict[str, nn.Module]:
        try:
            names = STAGE15_TASK_MODULE_NAMES[task]
        except KeyError as exc:
            raise ValueError(f"Unknown task: {task}") from exc
        return {name: getattr(self, name) for name in names}

    def fgm_modules(self, task: Task) -> dict[str, nn.Module]:
        if task == "phla":
            return {
                "peptide_adapter": self.peptide_adapter,
                "hla_adapter": self.hla_adapter,
            }
        if task == "ptcr":
            return {
                "peptide_adapter": self.peptide_adapter,
                "tcr_adapter": self.tcr_adapter,
            }
        raise ValueError(f"Unknown task: {task}")

    def set_trainable_task(self, task: Task, *, freeze_shared: bool = False) -> None:
        self.requires_grad_(False)
        for module in self.task_modules(task).values():
            module.requires_grad_(True)
        if freeze_shared:
            self.peptide_adapter.requires_grad_(False)
            self.hypergraph_fusion.requires_grad_(False)

    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]


class Stage15TaskBindingModel(nn.Module):
    """Deployable single-task Stage15 branch with the full shared fusion state."""

    def __init__(
        self,
        config: ModelConfig,
        fusion: Stage15FusionConfig,
        task: Task,
    ):
        super().__init__()
        if task not in STAGE15_TASK_MODULE_NAMES:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        self.peptide_adapter = SequenceAdapter(
            config.peptide_input_dim, **adapter_kwargs
        )
        self.hypergraph_fusion = HyperPMTFusionEncoder(config, fusion)
        if task == "phla":
            self.hla_adapter = SequenceAdapter(config.hla_input_dim, **adapter_kwargs)
            self.phla_pooling = MaskedAttentionPooling(config.model_dim)
            self.phla_classifier = BindingClassifier(
                config.model_dim, config.classifier_dropout
            )
        else:
            self.tcr_adapter = SequenceAdapter(config.tcr_input_dim, **adapter_kwargs)
            self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
            self.ptcr_classifier = BindingClassifier(
                config.model_dim, config.classifier_dropout
            )

    def forward(self, task: Task, *args, **kwargs) -> BindingOutput:
        if task != self.task:
            raise ValueError(f"This checkpoint is for task '{self.task}', not '{task}'")
        # Reuse the exact Stage15 forward implementation with this branch's
        # modules; it accesses only the task-specific attributes above.
        return Stage15BindingModel.forward(self, task, *args, **kwargs)

    _concat = staticmethod(Stage15BindingModel._concat)


def extract_stage15_task_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    task: Task,
) -> dict[str, torch.Tensor]:
    try:
        prefixes = tuple(
            f"{name}." for name in STAGE15_TASK_MODULE_NAMES[task]
        )
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task}") from exc
    selected = {
        name: value.detach().cpu()
        for name, value in state_dict.items()
        if name.startswith(prefixes)
    }
    if not selected:
        raise ValueError(f"No Stage15 parameters found for task '{task}'")
    return selected


__all__ = [
    "STAGE15_TASK_MODULE_NAMES",
    "Stage15BindingModel",
    "Stage15TaskBindingModel",
    "extract_stage15_task_state_dict",
]
