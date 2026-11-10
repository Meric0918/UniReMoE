"""Sequence-level HLA-peptide-TCR model initialised from PLM stage 1 or 2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn

from .attention import (
    BindingClassifier,
    CrossAttentionBlock,
    MaskedAttentionPooling,
)
from .config import ExperimentConfig, ModelConfig
from .finetuning import (
    ESMCFineTuningEncoder,
    ProgressiveFineTuningModel,
    TCRBertFineTuningEncoder,
)
from .model import UnifiedBindingModel
from .stage2_trainer import load_stage2_checkpoint
from .trainer import load_training_checkpoint

Stage3Phase = Literal["stage3a", "stage3b"]
Stage3Task = Literal["phla", "ptcr", "triplet"]


@dataclass
class TripletBindingOutput:
    logits: torch.Tensor
    phla_cross_attention: torch.Tensor | None = None
    triplet_cross_attention: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None


@dataclass
class TripletMultitaskOutput:
    phla_logits: torch.Tensor
    ptcr_logits: torch.Tensor
    pht_logits: torch.Tensor


class TripletBindingModel(UnifiedBindingModel):
    """Existing two binary branches plus a sequential ternary branch."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.triplet_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.triplet_pooling = MaskedAttentionPooling(config.model_dim)
        self.triplet_classifier = BindingClassifier(
            config.model_dim,
            config.classifier_dropout,
        )

    def initialise_triplet_from_ptcr(self) -> None:
        """Warm-start the new ternary path from the learned pTCR branch."""

        self.triplet_cross_attention.load_state_dict(
            deepcopy(self.ptcr_cross_attention.state_dict())
        )
        self.triplet_pooling.load_state_dict(
            deepcopy(self.ptcr_pooling.state_dict())
        )
        self.triplet_classifier.load_state_dict(
            deepcopy(self.ptcr_classifier.state_dict())
        )

    def new_triplet_modules(self) -> dict[str, nn.Module]:
        return {
            "triplet_cross_attention": self.triplet_cross_attention,
            "triplet_pooling": self.triplet_pooling,
            "triplet_classifier": self.triplet_classifier,
        }

    def triplet_modules(self) -> dict[str, nn.Module]:
        return {
            "peptide_adapter": self.peptide_adapter,
            "hla_adapter": self.hla_adapter,
            "tcr_adapter": self.tcr_adapter,
            "phla_cross_attention": self.phla_cross_attention,
            **self.new_triplet_modules(),
        }

    def forward_triplet(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> TripletBindingOutput:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        phla_sequence, phla_attention = self.phla_cross_attention(
            peptide,
            hla,
            peptide_mask,
            hla_mask,
            return_attention=return_attention,
        )
        complex_sequence, triplet_attention = self.triplet_cross_attention(
            phla_sequence,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=return_attention,
        )
        pooled, pooling_weights = self.triplet_pooling(
            complex_sequence,
            peptide_mask,
            return_weights=return_attention,
        )
        return TripletBindingOutput(
            logits=self.triplet_classifier(pooled),
            phla_cross_attention=phla_attention,
            triplet_cross_attention=triplet_attention,
            pooling_weights=pooling_weights,
        )

    def forward_all(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> TripletMultitaskOutput:
        """Return all Stage-3 heads from one shared residue encoding pass."""

        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        phla_sequence, _ = self.phla_cross_attention(
            peptide,
            hla,
            peptide_mask,
            hla_mask,
            return_attention=False,
        )
        phla_pooled, _ = self.phla_pooling(
            phla_sequence, peptide_mask, return_weights=False
        )
        ptcr_sequence, _ = self.ptcr_cross_attention(
            peptide,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        ptcr_pooled, _ = self.ptcr_pooling(
            ptcr_sequence, peptide_mask, return_weights=False
        )
        pht_sequence, _ = self.triplet_cross_attention(
            phla_sequence,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        pht_pooled, _ = self.triplet_pooling(
            pht_sequence, peptide_mask, return_weights=False
        )
        return TripletMultitaskOutput(
            phla_logits=self.phla_classifier(phla_pooled),
            ptcr_logits=self.ptcr_classifier(ptcr_pooled),
            pht_logits=self.triplet_classifier(pht_pooled),
        )


class ProgressiveTripletFineTuningModel(ProgressiveFineTuningModel):
    """Online PLMs with task-specific Stage-3 trainability controls."""

    binding_model: TripletBindingModel

    def set_trainable_stage3(
        self,
        task: Stage3Task,
        phase: Stage3Phase,
    ) -> None:
        if phase not in {"stage3a", "stage3b"}:
            raise ValueError(f"Unknown Stage-3 phase: {phase}")
        if phase == "stage3a" and task != "triplet":
            raise ValueError("Stage 3A only trains the triplet task")

        self._active_stage = phase
        if phase == "stage3b" and self._stage2a_hla_cache:
            self.clear_stage2a_hla_cache()
        self.requires_grad_(False)
        self.eval()
        for encoder in (
            self.peptide_encoder,
            self.hla_encoder,
            self.tcr_encoder,
        ):
            encoder.set_active_training(False)

        if phase == "stage3a":
            active_modules = self.binding_model.new_triplet_modules()
        elif task == "triplet":
            active_modules = self.binding_model.triplet_modules()
        elif task in {"phla", "ptcr"}:
            active_modules = self.binding_model.task_modules(task)
        else:
            raise ValueError(f"Unknown Stage-3 task: {task}")

        for module in active_modules.values():
            module.requires_grad_(True)
            module.train(True)

        if phase == "stage3b":
            self._activate_encoder(
                self.peptide_encoder,
                self.peptide_unfrozen_layers,
            )
            if task in {"phla", "triplet"}:
                self._activate_encoder(
                    self.hla_encoder,
                    self.hla_unfrozen_layers,
                )
            if task in {"ptcr", "triplet"}:
                self._activate_encoder(
                    self.tcr_encoder,
                    self.tcr_unfrozen_layers,
                )

    def forward_triplet(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
        *,
        return_attention: bool = False,
    ) -> TripletBindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        if self._active_stage == "stage3a":
            hla = self._encode_cached_stage2a_hla(hla_sequences)
        else:
            hla = self.hla_encoder(hla_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_triplet(
            peptide.hidden,
            peptide.mask,
            hla.hidden,
            hla.mask,
            tcr.hidden,
            tcr.mask,
            return_attention=return_attention,
        )

    def forward_all(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
    ) -> TripletMultitaskOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        if self._active_stage == "stage3a":
            hla = self._encode_cached_stage2a_hla(hla_sequences)
        else:
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


def build_progressive_triplet_model(
    config: ExperimentConfig,
    device: torch.device,
) -> ProgressiveTripletFineTuningModel:
    precision_to_encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    dtype = precision_to_encoder_dtype[config.triplet.mixed_precision]
    encoder_kwargs = {
        "device": device,
        "dtype": dtype,
        "gradient_checkpointing": config.triplet.gradient_checkpointing,
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
    model = ProgressiveTripletFineTuningModel(
        TripletBindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=config.triplet.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.triplet.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.triplet.tcr_unfrozen_layers,
    )
    return model.to(device)


def load_triplet_model_state(
    model: ProgressiveTripletFineTuningModel,
    state_dict: Mapping[str, Any],
) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [key for key in unexpected if not key.endswith("._extra_state")]
    if missing or unexpected:
        raise RuntimeError(
            "Stage-3 model state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def initialise_from_stage2(
    model: ProgressiveTripletFineTuningModel,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Load all Stage-2 tensors, then initialise only the new ternary modules."""

    source = Path(checkpoint_path).resolve()
    checkpoint = load_stage2_checkpoint(source, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("model") != config.to_dict()["model"]:
        raise ValueError("Stage-2 checkpoint model configuration does not match")
    source_training = checkpoint_config.get("training", {})
    if source_training.get("fold") != config.training.fold:
        raise ValueError(
            f"Stage-2 checkpoint fold={source_training.get('fold')} does not "
            f"match current fold={config.training.fold}"
        )
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get(
            "checkpoint_fingerprint"
        )
        current_fingerprint = current_metadata[entity][
            "checkpoint_fingerprint"
        ]
        if saved_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(
                f"Stage-2 base-PLM fingerprint mismatch for {entity}"
            )

    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=False,
    )
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [key for key in unexpected if not key.endswith("._extra_state")]
    allowed_prefixes = (
        "binding_model.triplet_cross_attention.",
        "binding_model.triplet_pooling.",
        "binding_model.triplet_classifier.",
    )
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Stage-2 to Stage-3 state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    model.binding_model.initialise_triplet_from_ptcr()
    return {
        "kind": "stage2",
        "path": str(source),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "format_version": checkpoint.get("format_version"),
        "stage": checkpoint.get("stage"),
        "stage2a_round": checkpoint.get("stage2a_round"),
        "stage2b_round": checkpoint.get("stage2b_round"),
        "global_step": checkpoint.get("global_step"),
        "run_timestamp": checkpoint.get("run_timestamp"),
        "source_seed": source_training.get("seed"),
        "source_fold": source_training.get("fold"),
        "encoder_fingerprints": {
            entity: saved_metadata.get(entity, {}).get(
                "checkpoint_fingerprint"
            )
            for entity in ("peptide", "hla", "tcr")
        },
        "encoder_metadata": saved_metadata,
    }


def _stage1_checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transfer_adapter_projection_to_sequence_encoders(
    state: Mapping[str, torch.Tensor],
    binding_model: TripletBindingModel,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Map the shape-compatible parts of Stage-1 adapters to sequence encoders."""

    transferred = dict(state)
    entities: list[str] = []
    parameter_map = (
        ("network.0.weight", "projection.0.weight"),
        ("network.0.bias", "projection.0.bias"),
        ("network.4.weight", "projection.1.weight"),
        ("network.4.bias", "projection.1.bias"),
    )
    for entity in ("peptide", "hla", "tcr"):
        encoder = getattr(binding_model, f"{entity}_encoder", None)
        if (
            encoder is None
            or not hasattr(encoder, "projection")
            or not hasattr(encoder, "layers")
        ):
            continue
        adapter_prefix = f"{entity}_adapter."
        adapter_keys = [
            key for key in transferred if key.startswith(adapter_prefix)
        ]
        if not adapter_keys:
            continue
        for old_suffix, new_suffix in parameter_map:
            old_key = f"{adapter_prefix}{old_suffix}"
            new_key = f"{entity}_encoder.{new_suffix}"
            if old_key not in transferred:
                raise ValueError(
                    f"Stage-1 {entity} adapter is missing transferable "
                    f"parameter {old_key}"
                )
            target = encoder.state_dict().get(new_suffix)
            if target is None or target.shape != transferred[old_key].shape:
                raise ValueError(
                    f"Stage-1 {entity} adapter parameter {old_key} does not match "
                    f"the Stage-8 sequence encoder shape at {new_key}"
                )
            transferred[new_key] = transferred[old_key]
        for key in adapter_keys:
            del transferred[key]
        entities.append(entity)
    return transferred, tuple(entities)


def initialise_binding_from_stage1(
    binding_model: TripletBindingModel,
    checkpoint_path: str | Path,
    *,
    model_config: ModelConfig,
    fold: int,
    seed: int,
    encoder_metadata: Mapping[str, Mapping[str, Any]],
    expected_sha256: str | None = None,
    require_joint_checkpoint: bool = False,
    transfer_sequence_encoder_projections: bool = False,
) -> dict[str, Any]:
    """Load a compatible Stage-1 source into a triplet-capable binding model."""

    source = Path(checkpoint_path).resolve()
    source_sha256 = _stage1_checkpoint_sha256(source)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError("Stage-1 checkpoint SHA256 does not match the configured source")
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    format_version = checkpoint.get("format_version")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type == "task_model":
        raise ValueError(
            "Stage 3 requires a full Stage-1 last checkpoint, not a task model"
        )

    is_completed_stage1 = (
        checkpoint_type is None
        and format_version in {1, 2, 3}
        and checkpoint.get("stage") == "complete"
    )
    is_stage1_joint = (
        format_version in {3, 4}
        and checkpoint_type == "stage1_joint_model"
        and checkpoint.get("stage") == "stage1b"
        and checkpoint.get("model_scope") == "joint_heads"
    )
    if require_joint_checkpoint and not is_stage1_joint:
        raise ValueError(
            "Stage-8 requires format_version=3 or 4 Stage-1 best_joint metadata"
        )
    if not is_completed_stage1 and not is_stage1_joint:
        if format_version not in {1, 2, 3}:
            raise ValueError("Unsupported Stage-1 checkpoint format")
        if checkpoint_type == "stage1_joint_model":
            raise ValueError(
                "Unsupported Stage-1 joint checkpoint format: expected "
                "format_version=3 or 4, stage='stage1b', "
                "model_scope='joint_heads'"
            )
        if checkpoint_type is not None:
            raise ValueError(
                "Stage 3 expected a Stage-1 training checkpoint, got "
                f"checkpoint_type={checkpoint_type!r}"
            )
        raise ValueError(
            "Stage 3 requires a completed Stage-1 checkpoint or a valid "
            "Stage-1 best_joint checkpoint"
        )

    checkpoint_config = checkpoint.get("config", {})
    source_model_config = checkpoint_config.get("model")
    expected_model_config = asdict(model_config)
    if (
        transfer_sequence_encoder_projections
        and isinstance(source_model_config, Mapping)
    ):
        # adapter_dropout has no role in the new sequence-encoder architecture.
        source_model_config = {
            key: value for key, value in source_model_config.items()
            if key != "adapter_dropout"
        }
        expected_model_config = {
            key: value for key, value in expected_model_config.items()
            if key != "adapter_dropout"
        }
    if source_model_config != expected_model_config:
        raise ValueError("Stage-1 checkpoint model configuration does not match")
    source_training = checkpoint_config.get("training", {})
    for key, expected in (("fold", fold), ("seed", seed)):
        if source_training.get(key) != expected:
            raise ValueError(
                f"Stage-1 checkpoint {key}={source_training.get(key)} "
                f"does not match current {key}={expected}"
            )

    saved_cache_metadata = checkpoint.get("cache_metadata", {})
    encoder_fingerprints: dict[str, str] = {}
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = (
            saved_cache_metadata.get(entity, {})
            .get("model", {})
            .get("checkpoint_fingerprint")
        )
        if not saved_fingerprint:
            raise ValueError(
                "Stage-1 checkpoint does not contain the cached base-PLM "
                f"fingerprint for {entity}"
            )
        current_fingerprint = encoder_metadata[entity][
            "checkpoint_fingerprint"
        ]
        if saved_fingerprint != current_fingerprint:
            raise ValueError(
                f"Stage-1 base-PLM fingerprint mismatch for {entity}"
            )
        encoder_fingerprints[entity] = saved_fingerprint

    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-1 checkpoint does not contain the unified model")
    transferred_entities: tuple[str, ...] = ()
    if transfer_sequence_encoder_projections:
        state, transferred_entities = _transfer_adapter_projection_to_sequence_encoders(
            state, binding_model
        )
    missing, unexpected = binding_model.load_state_dict(
        state,
        strict=False,
    )
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [
        key for key in unexpected if not key.endswith("._extra_state")
    ]
    allowed_prefixes = (
        "triplet_cross_attention.",
        "triplet_pooling.",
        "triplet_classifier.",
    ) + tuple(
        prefix
        for entity in transferred_entities
        for prefix in (
            f"{entity}_encoder.layers.",
            f"{entity}_encoder.final_norm.",
        )
    )
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Stage-1 to downstream-model state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )

    binding_model.initialise_triplet_from_ptcr()
    return {
        "kind": "stage1",
        "path": str(source),
        "sha256": source_sha256,
        "checkpoint_type": checkpoint_type,
        "format_version": format_version,
        "stage": checkpoint.get("stage"),
        "model_scope": checkpoint.get("model_scope"),
        "current_round": checkpoint.get("current_round"),
        "best_round": checkpoint.get("best_round"),
        "round_score": checkpoint.get(
            "round_score", checkpoint.get("best_round_score")
        ),
        "global_step": checkpoint.get("global_step"),
        "run_timestamp": checkpoint.get("run_timestamp"),
        "source_seed": source_training.get("seed"),
        "source_fold": source_training.get("fold"),
        "encoder_fingerprints": encoder_fingerprints,
        "sequence_encoder_initialization": (
            {
                "kind": "stage1_adapter_projection_transfer",
                "transferred_entities": list(transferred_entities),
                "new_self_attention_layers": "random_initialization",
                "new_final_norms": "random_initialization",
            }
            if transferred_entities
            else None
        ),
    }


def initialise_from_stage1(
    model: ProgressiveTripletFineTuningModel,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Stage-3 compatibility wrapper around the shared Stage-1 initializer."""

    return initialise_binding_from_stage1(
        model.binding_model,
        checkpoint_path,
        model_config=config.model,
        fold=config.training.fold,
        seed=config.training.seed,
        encoder_metadata=model.encoder_metadata(),
    )
