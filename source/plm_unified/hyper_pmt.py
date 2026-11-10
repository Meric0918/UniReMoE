"""Shared heterogeneous Hyper-PMT fusion blocks.

The implementation is architecture-only so stages can share the exact same
fusion semantics without importing one another's training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


EntityKind = Literal["peptide", "hla", "tcr"]
ENTITY_ORDER: tuple[EntityKind, ...] = ("peptide", "hla", "tcr")
ENTITY_CODES = {kind: index for index, kind in enumerate(ENTITY_ORDER)}


class HyperPMTConfig(Protocol):
    num_layers: int
    feedforward_dim: int
    compatibility_hidden_dim: int
    dropout: float


@dataclass
class HyperedgeEncoding:
    pooled: torch.Tensor
    states: dict[EntityKind, torch.Tensor]
    static: dict[EntityKind, torch.Tensor] | None = None
    dynamic: dict[EntityKind, torch.Tensor] | None = None
    compatibility: dict[EntityKind, torch.Tensor] | None = None


class CrossEntityAttention(nn.Module):
    """Shared multi-head attention with ordered entity relation bias."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = float(dropout)
        self.relation_bias = nn.Parameter(torch.zeros(num_heads, 3, 3))

    def _project(self, value: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, length, _ = value.shape
        return (
            projection(value)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _bias(
        self,
        query_type: int,
        key_types: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.relation_bias[:, query_type, key_types]
        bias = bias.unsqueeze(0).unsqueeze(2).expand(key_mask.shape[0], -1, 1, -1)
        return bias.masked_fill(~key_mask[:, None, None, :].bool(), float("-inf"))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        *,
        query_type: int,
        key_types: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_mask = query_mask.bool()
        key_mask = key_mask.bool()
        if (~query_mask.any(dim=1)).any():
            raise ValueError("Every hypergraph entity needs at least one residue")
        if (~key_mask.any(dim=1)).any():
            raise ValueError("Every dynamic query needs a real source entity")
        if key_types.ndim != 1 or key_types.shape[0] != key_value.shape[1]:
            raise ValueError("key_types must align with key_value tokens")

        q = self._project(query, self.q_proj)
        k = self._project(key_value, self.k_proj)
        v = self._project(key_value, self.v_proj)
        bias = self._bias(query_type, key_types.to(query.device), key_mask)
        dropout_p = self.dropout if self.training else 0.0
        if return_attention:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            weights = torch.softmax(scores + bias, dim=-1)
            if dropout_p:
                weights = F.dropout(weights, p=dropout_p, training=True)
            attended = torch.matmul(weights, v)
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=dropout_p
            )
            weights = None
        attended = attended.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.model_dim
        )
        output = self.out_proj(attended)
        output = output.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        if weights is not None:
            weights = weights.masked_fill(~query_mask[:, None, :, None], 0.0)
        return output, weights


class HypergraphLayer(nn.Module):
    """One static/dynamic/compatibility update over a variable-size edge."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        compatibility_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.static_norm = nn.ModuleDict({
            kind: nn.LayerNorm(model_dim) for kind in ENTITY_ORDER
        })
        self.static = nn.ModuleDict({
            kind: nn.Sequential(
                nn.Linear(model_dim, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, model_dim),
            )
            for kind in ENTITY_ORDER
        })
        self.dynamic_attention = CrossEntityAttention(model_dim, num_heads, dropout)
        self.compatibility = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, compatibility_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compatibility_hidden_dim, model_dim),
        )
        self.gates = nn.ModuleDict({
            kind: nn.Linear(model_dim * 2, model_dim) for kind in ENTITY_ORDER
        })
        self.update_norm = nn.ModuleDict({
            kind: nn.LayerNorm(model_dim) for kind in ENTITY_ORDER
        })
        self.post_norm = nn.ModuleDict({
            kind: nn.LayerNorm(model_dim) for kind in ENTITY_ORDER
        })
        self.post_ffn = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _source_tokens(
        kind: EntityKind,
        static: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_kinds = [
            candidate
            for candidate in ENTITY_ORDER
            if candidate in static and candidate != kind
        ]
        if not source_kinds:
            raise ValueError("A hyperedge must contain at least two entities")
        values = torch.cat([static[candidate] for candidate in source_kinds], dim=1)
        mask = torch.cat([masks[candidate].bool() for candidate in source_kinds], dim=1)
        types = torch.cat([
            torch.full(
                (static[candidate].shape[1],),
                ENTITY_CODES[candidate],
                device=values.device,
                dtype=torch.long,
            )
            for candidate in source_kinds
        ])
        return values, mask, types

    def forward(
        self,
        states: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
        *,
        return_attention: bool = False,
    ) -> tuple[
        dict[EntityKind, torch.Tensor],
        dict[EntityKind, torch.Tensor],
        dict[EntityKind, torch.Tensor],
        dict[EntityKind, torch.Tensor],
        dict[EntityKind, torch.Tensor] | None,
    ]:
        static: dict[EntityKind, torch.Tensor] = {}
        dynamic: dict[EntityKind, torch.Tensor] = {}
        compatibility: dict[EntityKind, torch.Tensor] = {}
        attentions: dict[EntityKind, torch.Tensor] = {}
        for kind in ENTITY_ORDER:
            if kind in states:
                static[kind] = self.static[kind](self.static_norm[kind](states[kind]))
        for kind, value in static.items():
            source, source_mask, source_types = self._source_tokens(kind, static, masks)
            attended, weights = self.dynamic_attention(
                value,
                source,
                masks[kind],
                source_mask,
                query_type=ENTITY_CODES[kind],
                key_types=source_types,
                return_attention=return_attention,
            )
            dynamic[kind] = attended
            if weights is not None:
                attentions[kind] = weights
            features = torch.cat(
                (value, attended, value * attended, (value - attended).square()),
                dim=-1,
            )
            compatibility[kind] = self.compatibility(features)
        updated: dict[EntityKind, torch.Tensor] = {}
        for kind, value in static.items():
            dynamic_value = dynamic[kind]
            gate = torch.sigmoid(
                self.gates[kind](torch.cat((value, dynamic_value), dim=-1))
            )
            value = self.update_norm[kind](
                states[kind] + gate * compatibility[kind]
            )
            value = self.post_norm[kind](value + self.post_ffn(value))
            updated[kind] = value.masked_fill(
                ~masks[kind].bool().unsqueeze(-1), 0.0
            )
        return (
            updated,
            static,
            dynamic,
            compatibility,
            attentions if return_attention else None,
        )


class HyperPMTFusionEncoder(nn.Module):
    """Shared heterogeneous hyperedge encoder for pair or triplet entities."""

    def __init__(self, model: ModelConfig, config: HyperPMTConfig):
        super().__init__()
        self.model_dim = model.model_dim
        self.type_embeddings = nn.Parameter(torch.zeros(3, model.model_dim))
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.layers = nn.ModuleList([
            HypergraphLayer(
                model.model_dim,
                model.num_attention_heads,
                config.feedforward_dim,
                config.compatibility_hidden_dim,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        entities: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> HyperedgeEncoding:
        if len(entities) < 2:
            raise ValueError("HyperPMTFusionEncoder requires at least two entities")
        if set(entities) != set(masks):
            raise ValueError("HyperPMT entities and masks must have identical keys")
        if not set(entities).issubset(ENTITY_CODES):
            raise ValueError("HyperPMT received an unknown entity kind")
        batch_sizes = {value.shape[0] for value in entities.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All hypergraph entities must share a batch size")
        states = {
            kind: value + self.type_embeddings[ENTITY_CODES[kind]].view(1, 1, -1)
            for kind, value in entities.items()
        }
        for kind, value in states.items():
            if value.ndim != 3 or masks[kind].shape[:2] != value.shape[:2]:
                raise ValueError(f"Invalid {kind} tensor/mask shape")
            states[kind] = value.masked_fill(
                ~masks[kind].bool().unsqueeze(-1), 0.0
            )
        last_static = last_dynamic = last_compatibility = None
        for layer in self.layers:
            states, last_static, last_dynamic, last_compatibility, _ = layer(
                states, masks, return_attention=False
            )
        return HyperedgeEncoding(
            pooled=torch.cat(
                [states[kind] for kind in ENTITY_ORDER if kind in states], dim=1
            ),
            states=states,
            static=last_static if return_diagnostics else None,
            dynamic=last_dynamic if return_diagnostics else None,
            compatibility=last_compatibility if return_diagnostics else None,
        )


# Private aliases preserve Stage14's historical test/debug import surface.
_CrossEntityAttention = CrossEntityAttention
_HypergraphLayer = HypergraphLayer
_ENTITY_ORDER = ENTITY_ORDER
_ENTITY_CODES = ENTITY_CODES


__all__ = [
    "EntityKind",
    "HyperedgeEncoding",
    "HyperPMTFusionEncoder",
    "CrossEntityAttention",
    "HypergraphLayer",
]
