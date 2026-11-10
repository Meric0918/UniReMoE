"""Stage-14 heterogeneous Hyper-PMT fusion model.

The model keeps the residue-level adapters from Stage 1, but replaces the
sequential PH -> PHT path with one shared, variable-size hyperedge encoder.
Pairwise PH/PT edges and the ternary PHT hyperedge therefore use exactly the
same interaction parameters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    BindingClassifier,
    CrossAttentionBlock,
    MaskedAttentionPooling,
    SequenceAdapter,
)
from .config import ModelConfig
from .finetuning import (
    ESMCFineTuningEncoder,
    ProgressiveFineTuningModel,
    TCRBertFineTuningEncoder,
)
from .hyper_pmt import (
    ENTITY_ORDER as _ENTITY_ORDER,
    EntityKind,
    HyperPMTFusionEncoder,
)
from .stage14_config import Stage14ExperimentConfig, Stage14FusionConfig
from .stage8_checkpoint import checkpoint_sha256, load_torch_checkpoint
from .trainer import load_training_checkpoint


Stage14Phase = Literal["stage14a", "stage14b"]
Stage14Task = Literal["phla", "ptcr", "joint"]


@dataclass
class Stage14BindingOutput:
    phla_logits: torch.Tensor
    ptcr_logits: torch.Tensor
    pht_logits: torch.Tensor


class Stage14BindingModel(nn.Module):
    """Adapters, shared Hyper-PMT fusion, task pools/heads and PH prior."""

    def __init__(self, config: ModelConfig, fusion: Stage14FusionConfig | None = None):
        super().__init__()
        fusion = fusion or Stage14FusionConfig()
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        self.peptide_adapter = SequenceAdapter(config.peptide_input_dim, **adapter_kwargs)
        self.hla_adapter = SequenceAdapter(config.hla_input_dim, **adapter_kwargs)
        self.tcr_adapter = SequenceAdapter(config.tcr_input_dim, **adapter_kwargs)
        self.hypergraph_fusion = HyperPMTFusionEncoder(config, fusion)
        self.phla_pooling = MaskedAttentionPooling(config.model_dim)
        self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
        self.pht_pooling = MaskedAttentionPooling(config.model_dim)
        self.phla_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)
        self.ptcr_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)
        self.pht_classifier = BindingClassifier(config.model_dim, config.classifier_dropout)

        # This is deliberately a separate PH prior branch.  It is only used
        # for PHT, so pairwise supervision cannot turn the PHT path back into
        # the old sequential PH -> T representation.
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.phla_prior_attention = CrossAttentionBlock(**attention_kwargs)
        self.phla_prior_pooling = MaskedAttentionPooling(config.model_dim)
        self.phla_prior_projection = nn.Linear(config.model_dim, config.model_dim)
        self.phla_prior_gate = nn.Linear(config.model_dim * 2, config.model_dim)
        nn.init.eye_(self.phla_prior_projection.weight)
        nn.init.zeros_(self.phla_prior_projection.bias)
        nn.init.zeros_(self.phla_prior_gate.weight)
        nn.init.constant_(self.phla_prior_gate.bias, -2.0)

    @staticmethod
    def _concat(
        states: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kinds = [kind for kind in _ENTITY_ORDER if kind in states]
        sequence = torch.cat([states[kind] for kind in kinds], dim=1)
        mask = torch.cat([masks[kind].bool() for kind in kinds], dim=1)
        return sequence, mask

    def _adapted(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor | None = None,
        hla_mask: torch.Tensor | None = None,
        tcr_hidden: torch.Tensor | None = None,
        tcr_mask: torch.Tensor | None = None,
    ) -> tuple[dict[EntityKind, torch.Tensor], dict[EntityKind, torch.Tensor]]:
        entities: dict[EntityKind, torch.Tensor] = {
            "peptide": self.peptide_adapter(peptide_hidden, peptide_mask)
        }
        masks: dict[EntityKind, torch.Tensor] = {"peptide": peptide_mask.bool()}
        if (hla_hidden is None) != (hla_mask is None):
            raise ValueError("hla_hidden and hla_mask must be provided together")
        if (tcr_hidden is None) != (tcr_mask is None):
            raise ValueError("tcr_hidden and tcr_mask must be provided together")
        if hla_hidden is not None:
            entities["hla"] = self.hla_adapter(hla_hidden, hla_mask)
            masks["hla"] = hla_mask.bool()
        if tcr_hidden is not None:
            entities["tcr"] = self.tcr_adapter(tcr_hidden, tcr_mask)
            masks["tcr"] = tcr_mask.bool()
        return entities, masks

    def _edge(
        self,
        entities: Mapping[EntityKind, torch.Tensor],
        masks: Mapping[EntityKind, torch.Tensor],
        *,
        pooling: nn.Module,
        include_prior: bool = False,
    ) -> torch.Tensor:
        encoding = self.hypergraph_fusion(entities, masks)
        sequence, sequence_mask = self._concat(encoding.states, masks)
        pooled, _ = pooling(sequence, sequence_mask, return_weights=False)
        if include_prior:
            peptide = entities["peptide"]
            hla = entities["hla"]
            prior_sequence, _ = self.phla_prior_attention(
                peptide, hla, masks["peptide"], masks["hla"], return_attention=False
            )
            prior, _ = self.phla_prior_pooling(
                prior_sequence, masks["peptide"], return_weights=False
            )
            prior = self.phla_prior_projection(prior)
            gate = torch.sigmoid(self.phla_prior_gate(torch.cat((pooled, prior), dim=-1)))
            pooled = F.layer_norm(pooled + gate * prior, (pooled.shape[-1],))
        return pooled

    def forward(
        self,
        task: Literal["phla", "ptcr"],
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_hidden: torch.Tensor,
        receptor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        if task not in {"phla", "ptcr"}:
            raise ValueError(f"Unknown Stage-14 pair task: {task}")
        if task == "phla":
            entities, masks = self._adapted(
                peptide_hidden, peptide_mask, hla_hidden=receptor_hidden, hla_mask=receptor_mask
            )
            pooled = self._edge(entities, masks, pooling=self.phla_pooling)
            return _PairBindingOutput(self.phla_classifier(pooled), pooled)
        entities, masks = self._adapted(
            peptide_hidden, peptide_mask, tcr_hidden=receptor_hidden, tcr_mask=receptor_mask
        )
        pooled = self._edge(entities, masks, pooling=self.ptcr_pooling)
        return _PairBindingOutput(self.ptcr_classifier(pooled), pooled)

    def forward_all(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> Stage14BindingOutput:
        entities, masks = self._adapted(
            peptide_hidden,
            peptide_mask,
            hla_hidden=hla_hidden,
            hla_mask=hla_mask,
            tcr_hidden=tcr_hidden,
            tcr_mask=tcr_mask,
        )
        phla_pooled = self._edge(
            {"peptide": entities["peptide"], "hla": entities["hla"]},
            {"peptide": masks["peptide"], "hla": masks["hla"]},
            pooling=self.phla_pooling,
        )
        ptcr_pooled = self._edge(
            {"peptide": entities["peptide"], "tcr": entities["tcr"]},
            {"peptide": masks["peptide"], "tcr": masks["tcr"]},
            pooling=self.ptcr_pooling,
        )
        pht_pooled = self._edge(entities, masks, pooling=self.pht_pooling, include_prior=True)
        return Stage14BindingOutput(
            phla_logits=self.phla_classifier(phla_pooled),
            ptcr_logits=self.ptcr_classifier(ptcr_pooled),
            pht_logits=self.pht_classifier(pht_pooled),
        )

    def new_stage14_modules(self) -> dict[str, nn.Module]:
        return {
            "hypergraph_fusion": self.hypergraph_fusion,
            "pht_pooling": self.pht_pooling,
            "pht_classifier": self.pht_classifier,
            "phla_prior_projection": self.phla_prior_projection,
            "phla_prior_gate": self.phla_prior_gate,
        }

    def new_triplet_modules(self) -> dict[str, nn.Module]:
        """Compatibility alias used by generic downstream utilities."""

        return self.new_stage14_modules()

    def task_modules(self, task: Stage14Task) -> dict[str, nn.Module]:
        if task == "phla":
            return {
                "peptide_adapter": self.peptide_adapter,
                "hla_adapter": self.hla_adapter,
                "hypergraph_fusion": self.hypergraph_fusion,
                "phla_pooling": self.phla_pooling,
                "phla_classifier": self.phla_classifier,
            }
        if task == "ptcr":
            return {
                "peptide_adapter": self.peptide_adapter,
                "tcr_adapter": self.tcr_adapter,
                "hypergraph_fusion": self.hypergraph_fusion,
                "ptcr_pooling": self.ptcr_pooling,
                "ptcr_classifier": self.ptcr_classifier,
            }
        if task == "joint":
            return dict(self.named_children())
        raise ValueError(f"Unknown Stage-14 task: {task}")


@dataclass
class _PairBindingOutput:
    logits: torch.Tensor
    pooled_embedding: torch.Tensor


class ProgressiveStage14FineTuningModel(ProgressiveFineTuningModel):
    """Online PLMs with Stage14A/Stage14B trainability controls."""

    binding_model: Stage14BindingModel

    def set_trainable_stage14(self, task: Stage14Task, phase: Stage14Phase) -> None:
        if phase not in {"stage14a", "stage14b"}:
            raise ValueError(f"Unknown Stage-14 phase: {phase}")
        if task not in {"phla", "ptcr", "joint"}:
            raise ValueError(f"Unknown Stage-14 task: {task}")
        if phase == "stage14a" and task != "joint":
            raise ValueError("Stage 14A only trains the joint task")
        self._active_stage = phase
        self.requires_grad_(False)
        self.eval()
        for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
            encoder.set_active_training(False)
        if phase == "stage14a":
            active = self.binding_model.new_stage14_modules()
        else:
            active = self.binding_model.task_modules(task)
            if task == "joint":
                active = self.binding_model.task_modules("joint")
                # Prior attention/pooling are only useful on the joint PHT pass;
                # task_modules("joint") already includes them.
        for module in active.values():
            module.requires_grad_(True)
            module.train(True)
        if phase == "stage14b":
            self._activate_encoder(self.peptide_encoder, self.peptide_unfrozen_layers)
            if task in {"phla", "joint"}:
                self._activate_encoder(self.hla_encoder, self.hla_unfrozen_layers)
            if task in {"ptcr", "joint"}:
                self._activate_encoder(self.tcr_encoder, self.tcr_unfrozen_layers)

    def set_trainable_stage8(self, task: Stage14Task, phase: str) -> None:
        """Compatibility shim for shared Stage-8 training helpers."""

        mapped = {"stage8a": "stage14a", "stage8b": "stage14b"}.get(phase)
        if mapped is None:
            raise ValueError(f"Unknown Stage-8-compatible phase: {phase}")
        self.set_trainable_stage14(task, mapped)

    def forward(
        self,
        task: Literal["phla", "ptcr"],
        peptide_sequences: Sequence[str],
        receptor_sequences: Sequence[str],
    ):
        peptide = self.peptide_encoder(peptide_sequences)
        if task == "phla":
            receptor = self.hla_encoder(receptor_sequences)
        elif task == "ptcr":
            receptor = self.tcr_encoder(receptor_sequences)
        else:
            raise ValueError(f"Unknown Stage-14 task: {task}")
        return self.binding_model(
            task,
            peptide.hidden,
            peptide.mask,
            receptor.hidden,
            receptor.mask,
        )

    def forward_all(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
    ) -> Stage14BindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_all(
            peptide.hidden, peptide.mask,
            hla.hidden, hla.mask,
            tcr.hidden, tcr.mask,
        )


def build_progressive_stage14_model(
    config: Stage14ExperimentConfig,
    device: torch.device,
) -> ProgressiveStage14FineTuningModel:
    precision_to_encoder_dtype = {"none": "float32", "fp16": "float16", "bf16": "bfloat16"}
    encoder_kwargs = {
        "device": device,
        "dtype": precision_to_encoder_dtype[config.stage14.mixed_precision],
        "gradient_checkpointing": config.stage14.gradient_checkpointing,
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
    for name, actual, expected in (
        ("Peptide ESM-C", peptide_encoder.hidden_size, config.model.peptide_input_dim),
        ("HLA ESM-C", hla_encoder.hidden_size, config.model.hla_input_dim),
        ("TCR-BERT", tcr_encoder.hidden_size, config.model.tcr_input_dim),
    ):
        if actual != expected:
            raise ValueError(f"{name} hidden size does not match model config")
    return ProgressiveStage14FineTuningModel(
        Stage14BindingModel(config.model, config.fusion),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=config.stage14.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.stage14.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.stage14.tcr_unfrozen_layers,
    ).to(device)


def _asdict_model(config: ModelConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


def initialise_stage14_from_stage1(
    model: ProgressiveStage14FineTuningModel,
    checkpoint_path: str | Path,
    *,
    model_config: ModelConfig,
    fold: int,
    seed: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load Stage-1 weights and initialize the new hypergraph modules."""

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {source}")
    source_sha = checkpoint_sha256(source)
    if source_sha != expected_sha256:
        raise ValueError("Stage-1 checkpoint SHA256 does not match the configured source")
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    if not (
        checkpoint.get("format_version") in {3, 4}
        and checkpoint.get("checkpoint_type") == "stage1_joint_model"
        and checkpoint.get("stage") == "stage1b"
        and checkpoint.get("model_scope") == "joint_heads"
    ):
        raise ValueError("Stage14 requires a format_version=3 or 4 Stage-1 best_joint checkpoint")
    if checkpoint.get("config", {}).get("model") != _asdict_model(model_config):
        raise ValueError("Stage-1 checkpoint model configuration does not match Stage14")
    training = checkpoint.get("config", {}).get("training", {})
    for key, expected in (("fold", fold), ("seed", seed)):
        if training.get(key) != expected:
            raise ValueError(f"Stage-1 checkpoint {key} does not match Stage14")
    saved_cache = checkpoint.get("cache_metadata", {})
    fingerprints = {}
    current_metadata = model.encoder_metadata()
    for entity in _ENTITY_ORDER:
        saved = saved_cache.get(entity, {}).get("model", {}).get("checkpoint_fingerprint")
        if not saved:
            raise ValueError(f"Stage-1 checkpoint has no cached PLM fingerprint for {entity}")
        current = current_metadata[entity]["checkpoint_fingerprint"]
        if saved != current:
            raise ValueError(f"Stage-1 base-PLM fingerprint mismatch for {entity}")
        fingerprints[entity] = saved
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-1 checkpoint does not contain a model state")

    target = model.binding_model
    required = {
        "peptide_adapter.", "hla_adapter.", "tcr_adapter.",
        "phla_pooling.", "ptcr_pooling.",
        "phla_classifier.", "ptcr_classifier.",
    }
    missing = [prefix for prefix in required if not any(str(key).startswith(prefix) for key in state)]
    if missing:
        raise RuntimeError("Stage-1 checkpoint is missing required tensors: " + ", ".join(sorted(missing)))
    for target_name, source_name in (
        ("peptide_adapter", "peptide_adapter"),
        ("hla_adapter", "hla_adapter"),
        ("tcr_adapter", "tcr_adapter"),
        ("phla_pooling", "phla_pooling"),
        ("ptcr_pooling", "ptcr_pooling"),
        ("phla_classifier", "phla_classifier"),
        ("ptcr_classifier", "ptcr_classifier"),
    ):
        values = {
            key[len(source_name) + 1 :]: value
            for key, value in state.items()
            if str(key).startswith(source_name + ".")
        }
        getattr(target, target_name).load_state_dict(values, strict=True)
    prior_attention = {
        key[len("phla_cross_attention") + 1 :]: value
        for key, value in state.items()
        if str(key).startswith("phla_cross_attention.")
    }
    if not prior_attention:
        raise RuntimeError("Stage-1 checkpoint has no phla_cross_attention state")
    target.phla_prior_attention.load_state_dict(prior_attention, strict=True)
    target.phla_prior_pooling.load_state_dict(deepcopy(target.phla_pooling.state_dict()), strict=True)
    target.pht_pooling.load_state_dict(deepcopy(target.ptcr_pooling.state_dict()), strict=True)
    target.pht_classifier.load_state_dict(deepcopy(target.ptcr_classifier.state_dict()), strict=True)
    return {
        "kind": "stage1",
        "path": str(source),
        "sha256": source_sha,
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "format_version": checkpoint.get("format_version"),
        "stage": checkpoint.get("stage"),
        "model_scope": checkpoint.get("model_scope"),
        "best_round": checkpoint.get("best_round"),
        "round_score": checkpoint.get("round_score", checkpoint.get("best_round_score")),
        "run_timestamp": checkpoint.get("run_timestamp"),
        "source_seed": training.get("seed"),
        "source_fold": training.get("fold"),
        "encoder_fingerprints": fingerprints,
    }
