"""Stage13 model: heterogeneous GraphSAGE fusion before the Stage1 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .attention import (
    BindingClassifier,
    CrossAttentionBlock,
    MaskedAttentionPooling,
    SequenceAdapter,
)
from .config import ModelConfig
from .model import BindingOutput
from .stage13_config import Stage13GraphConfig
from .stage13_graph import RELATIONS, SampledGraph


Task = Literal["phla", "ptcr"]
NODE_TYPE_CODES = {"peptide": 0, "hla": 1, "tcr": 2}


class Stage13GraphLayer(nn.Module):
    """A relation-aware mean-aggregation GraphSAGE layer."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_transforms = nn.ModuleDict(
            {kind: nn.Linear(hidden_dim, hidden_dim) for kind in NODE_TYPE_CODES}
        )
        self.relation_transforms = nn.ModuleDict(
            {relation: nn.Linear(hidden_dim, hidden_dim, bias=False) for relation in RELATIONS}
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        child_features: torch.Tensor,
        relation_indices: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        output = torch.zeros_like(node_features)
        for code, kind in enumerate(NODE_TYPE_CODES):
            mask = node_types == code
            if mask.any():
                output[mask] = self.self_transforms[kind](node_features[mask]).to(output.dtype)
        for relation, indices in relation_indices.items():
            if indices.numel() == 0 or child_features.shape[0] == 0:
                continue
            valid = indices >= 0
            safe_indices = indices.clamp_min(0)
            gathered = child_features[safe_indices]
            counts = valid.sum(dim=1, keepdim=True).to(gathered.dtype)
            mean = (gathered * valid.unsqueeze(-1).to(gathered.dtype)).sum(dim=1)
            mean = mean / counts.clamp_min(1.0)
            message = self.relation_transforms[relation](mean)
            output = output + message.to(output.dtype) * (counts > 0).to(output.dtype)
        return self.dropout(self.activation(self.norm(output)))


class Stage13GraphSAGE(nn.Module):
    """Stacked sampled GraphSAGE operating on padded type-specific features."""

    def __init__(self, model: ModelConfig, graph: Stage13GraphConfig):
        super().__init__()
        self.hidden_dim = graph.hidden_dim
        self.input_projections = nn.ModuleDict(
            {
                "peptide": nn.Linear(model.peptide_input_dim, graph.hidden_dim),
                "hla": nn.Linear(model.hla_input_dim, graph.hidden_dim),
                "tcr": nn.Linear(model.tcr_input_dim, graph.hidden_dim),
            }
        )
        self.layers = nn.ModuleList(
            [Stage13GraphLayer(graph.hidden_dim, graph.dropout) for _ in range(graph.num_layers)]
        )

    def _project(self, features: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (features.shape[0], self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        dimensions = {0: "peptide", 1: "hla", 2: "tcr"}
        for code, kind in dimensions.items():
            mask = types == code
            if not mask.any():
                continue
            input_dim = self.input_projections[kind].in_features
            output[mask] = self.input_projections[kind](features[mask, :input_dim]).to(output.dtype)
        if (types < 0).any():
            raise ValueError("Unknown node type in Stage13 graph sample")
        return output

    def forward(self, graph: SampledGraph) -> torch.Tensor:
        if len(graph.ids_by_depth) != len(self.layers) + 1:
            raise ValueError("Sampled graph depth does not match GraphSAGE layers")
        child = self._project(graph.features_by_depth[-1], graph.types_by_depth[-1])
        for layer_index in reversed(range(len(self.layers))):
            parent = self._project(
                graph.features_by_depth[layer_index], graph.types_by_depth[layer_index]
            )
            child = self.layers[layer_index](
                parent,
                graph.types_by_depth[layer_index],
                child,
                graph.relation_indices[layer_index],
            )
        return child[graph.root_positions]


@dataclass
class Stage13Output(BindingOutput):
    graph_peptide_embedding: torch.Tensor | None = None
    graph_receptor_embedding: torch.Tensor | None = None


class Stage13BindingModel(nn.Module):
    """Stage1-style pHLA/pTCR heads with GraphSAGE graph-token fusion."""

    def __init__(self, config: ModelConfig, graph_config: Stage13GraphConfig):
        super().__init__()
        self.config = config
        self.graph_sage = Stage13GraphSAGE(config, graph_config)
        token_dim = graph_config.hidden_dim
        self.graph_token_projection = nn.ModuleDict(
            {
                "peptide": nn.Sequential(
                    nn.Linear(token_dim, config.peptide_input_dim),
                    nn.GELU(),
                    nn.Dropout(graph_config.projection_dropout),
                ),
                "hla": nn.Sequential(
                    nn.Linear(token_dim, config.hla_input_dim),
                    nn.GELU(),
                    nn.Dropout(graph_config.projection_dropout),
                ),
                "tcr": nn.Sequential(
                    nn.Linear(token_dim, config.tcr_input_dim),
                    nn.GELU(),
                    nn.Dropout(graph_config.projection_dropout),
                ),
            }
        )
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.peptide_adapter = SequenceAdapter(config.peptide_input_dim, **adapter_kwargs)
        self.hla_adapter = SequenceAdapter(config.hla_input_dim, **adapter_kwargs)
        self.tcr_adapter = SequenceAdapter(config.tcr_input_dim, **adapter_kwargs)
        self.phla_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.ptcr_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.phla_pooling = MaskedAttentionPooling(config.model_dim)
        self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
        self.phla_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)
        self.ptcr_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)

    def _prepend_token(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        graph_embedding: torch.Tensor,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.graph_token_projection[kind](graph_embedding).unsqueeze(1)
        token_mask = torch.ones((hidden.shape[0], 1), dtype=torch.bool, device=hidden.device)
        return torch.cat((token, hidden), dim=1), torch.cat((token_mask, mask.bool()), dim=1)

    def forward(
        self,
        task: Task,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_hidden: torch.Tensor,
        receptor_mask: torch.Tensor,
        graph: SampledGraph,
        *,
        return_attention: bool = False,
    ) -> Stage13Output:
        if task not in {"phla", "ptcr"}:
            raise ValueError(f"Unknown task: {task}")
        graph_embeddings = self.graph_sage(graph)
        batch_size = peptide_hidden.shape[0]
        if graph_embeddings.shape[0] != batch_size * 2:
            raise ValueError("Stage13 graph sample must contain peptide and receptor roots")
        graph_peptide = graph_embeddings[:batch_size]
        graph_receptor = graph_embeddings[batch_size:]
        peptide_hidden, peptide_mask = self._prepend_token(
            peptide_hidden, peptide_mask, graph_peptide, "peptide"
        )
        receptor_kind = "hla" if task == "phla" else "tcr"
        receptor_hidden, receptor_mask = self._prepend_token(
            receptor_hidden, receptor_mask, graph_receptor, receptor_kind
        )
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        if task == "phla":
            receptor = self.hla_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.phla_cross_attention
            pooling = self.phla_pooling
            classifier = self.phla_classifier
        else:
            receptor = self.tcr_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.ptcr_cross_attention
            pooling = self.ptcr_pooling
            classifier = self.ptcr_classifier
        sequence, attention_weights = cross_attention(
            peptide,
            receptor,
            peptide_mask,
            receptor_mask,
            return_attention=return_attention,
        )
        pooled, pooling_weights = pooling(
            sequence, peptide_mask, return_weights=return_attention
        )
        return Stage13Output(
            logits=classifier(pooled),
            pooled_embedding=pooled,
            cross_attention=attention_weights,
            pooling_weights=pooling_weights,
            graph_peptide_embedding=graph_peptide,
            graph_receptor_embedding=graph_receptor,
        )

    def task_modules(self, task: Task) -> dict[str, nn.Module]:
        if task == "phla":
            names = (
                "graph_sage",
                "graph_token_projection",
                "peptide_adapter",
                "hla_adapter",
                "phla_cross_attention",
                "phla_pooling",
                "phla_classifier",
            )
        elif task == "ptcr":
            names = (
                "graph_sage",
                "graph_token_projection",
                "peptide_adapter",
                "tcr_adapter",
                "ptcr_cross_attention",
                "ptcr_pooling",
                "ptcr_classifier",
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        return {name: getattr(self, name) for name in names}

    def fgm_modules(self, task: Task) -> dict[str, nn.Module]:
        if task == "phla":
            return {"peptide_adapter": self.peptide_adapter, "hla_adapter": self.hla_adapter}
        if task == "ptcr":
            return {"peptide_adapter": self.peptide_adapter, "tcr_adapter": self.tcr_adapter}
        raise ValueError(f"Unknown task: {task}")

    def set_trainable_task(self, task: Task) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.task_modules(task).values():
            for parameter in module.parameters():
                parameter.requires_grad = True
