"""Task-routed Stage Adapter with checkpoint or random initialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn

from .attention import BindingClassifier, CrossAttentionBlock, MaskedAttentionPooling
from .config import ModelConfig
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .model import BindingOutput, UnifiedBindingModel
from .stage2_trainer import load_stage2_checkpoint
from .triplet_model import TripletBindingModel, initialise_binding_from_stage1

StageAdapterTask = Literal["phla", "ptcr", "pht"]


@dataclass
class StageAdapterPHTOutput:
    logits: torch.Tensor
    pooled_embedding: torch.Tensor


class ResidualTaskAdapter(nn.Module):
    """Randomly initialised bottleneck delta operating in Stage-1 model space."""

    def __init__(self, model_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.down = nn.Linear(model_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, model_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.activation(self.down(self.norm(hidden)))))
        output = hidden + delta
        return output.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)


class StageAdapterBindingModel(UnifiedBindingModel):
    """Frozen Stage-1 binary experts plus independently random PHT modules."""

    STAGE1_MODULE_NAMES = (
        "peptide_adapter", "hla_adapter", "tcr_adapter",
        "phla_cross_attention", "ptcr_cross_attention",
        "phla_pooling", "ptcr_pooling",
        "phla_classifier", "ptcr_classifier",
    )
    SHARED_ADAPTER_NAMES = ("peptide_adapter", "hla_adapter", "tcr_adapter")

    def __init__(
        self, config: ModelConfig, *, residual_adapter_dim: int = 64,
        train_shared_adapters: bool = False,
    ):
        super().__init__(config)
        self.train_shared_adapters = train_shared_adapters
        self.pht_peptide_adapter = ResidualTaskAdapter(
            config.model_dim, residual_adapter_dim, config.adapter_dropout
        )
        self.pht_hla_adapter = ResidualTaskAdapter(
            config.model_dim, residual_adapter_dim, config.adapter_dropout
        )
        self.pht_tcr_adapter = ResidualTaskAdapter(
            config.model_dim, residual_adapter_dim, config.adapter_dropout
        )
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.pht_phla_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.pht_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.pht_pooling = MaskedAttentionPooling(config.model_dim)
        self.pht_classifier = BindingClassifier(
            config.model_dim, config.classifier_dropout
        )
        self.freeze_stage1_modules()

    def stage1_modules(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name) for name in self.STAGE1_MODULE_NAMES}

    def pht_modules(self) -> dict[str, nn.Module]:
        return {
            "pht_peptide_adapter": self.pht_peptide_adapter,
            "pht_hla_adapter": self.pht_hla_adapter,
            "pht_tcr_adapter": self.pht_tcr_adapter,
            "pht_phla_cross_attention": self.pht_phla_cross_attention,
            "pht_cross_attention": self.pht_cross_attention,
            "pht_pooling": self.pht_pooling,
            "pht_classifier": self.pht_classifier,
        }

    def freeze_stage1_modules(self) -> None:
        for name, module in self.stage1_modules().items():
            if self.train_shared_adapters and name in self.SHARED_ADAPTER_NAMES:
                continue
            module.requires_grad_(False)
            module.eval()

    def set_pht_training(self, active: bool = True) -> None:
        self.requires_grad_(False)
        self.eval()
        if active:
            modules = self.pht_modules()
            if self.train_shared_adapters:
                modules.update({name: getattr(self, name) for name in self.SHARED_ADAPTER_NAMES})
            for module in modules.values():
                module.requires_grad_(True)
                module.train(True)
        self.freeze_stage1_modules()

    def forward_pht(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> StageAdapterPHTOutput:
        peptide_base = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla_base = self.hla_adapter(hla_hidden, hla_mask)
        tcr_base = self.tcr_adapter(tcr_hidden, tcr_mask)
        peptide = self.pht_peptide_adapter(peptide_base, peptide_mask)
        hla = self.pht_hla_adapter(hla_base, hla_mask)
        tcr = self.pht_tcr_adapter(tcr_base, tcr_mask)
        phla_sequence, _ = self.pht_phla_cross_attention(
            peptide, hla, peptide_mask, hla_mask, return_attention=False
        )
        pht_sequence, _ = self.pht_cross_attention(
            phla_sequence, tcr, peptide_mask, tcr_mask, return_attention=False
        )
        pooled, _ = self.pht_pooling(
            pht_sequence, peptide_mask, return_weights=False
        )
        return StageAdapterPHTOutput(self.pht_classifier(pooled), pooled)


class ProgressiveStageAdapterModel(nn.Module):
    """Online PLMs with training scope selected by initialization mode."""

    def __init__(
        self, binding_model, peptide_encoder, hla_encoder, tcr_encoder,
        *, initialization: str = "checkpoint",
    ):
        super().__init__()
        self.binding_model = binding_model
        self.peptide_encoder = peptide_encoder
        self.hla_encoder = hla_encoder
        self.tcr_encoder = tcr_encoder
        self.initialization = initialization
        self.set_trainable_pht()

    def _freeze_encoders(self) -> None:
        for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
            encoder.requires_grad_(False)
            encoder.set_active_training(False)

    def set_trainable_pht(self) -> None:
        self.requires_grad_(False)
        self.eval()
        self._freeze_encoders()
        self.binding_model.set_pht_training(True)
        if self.initialization == "random":
            for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
                encoder.set_full_training()

    def set_evaluation(self) -> None:
        self.eval()
        self._freeze_encoders()
        self.binding_model.set_pht_training(False)

    def encoder_metadata(self) -> dict:
        return {
            "peptide": self.peptide_encoder.metadata(),
            "hla": self.hla_encoder.metadata(),
            "tcr": self.tcr_encoder.metadata(),
        }

    def forward_pht(
        self,
        peptide_sequences: Sequence[str],
        hla_sequences: Sequence[str],
        tcr_sequences: Sequence[str],
    ) -> StageAdapterPHTOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model.forward_pht(
            peptide.hidden, peptide.mask, hla.hidden, hla.mask,
            tcr.hidden, tcr.mask,
        )

    def forward_ph(
        self, peptide_sequences: Sequence[str], hla_sequences: Sequence[str]
    ) -> BindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        hla = self.hla_encoder(hla_sequences)
        return self.binding_model(
            "phla", peptide.hidden, peptide.mask, hla.hidden, hla.mask
        )

    def forward_pt(
        self, peptide_sequences: Sequence[str], tcr_sequences: Sequence[str]
    ) -> BindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        tcr = self.tcr_encoder(tcr_sequences)
        return self.binding_model(
            "ptcr", peptide.hidden, peptide.mask, tcr.hidden, tcr.mask
        )

    def forward(
        self,
        task: StageAdapterTask,
        peptide_sequences: Sequence[str],
        *,
        hla_sequences: Sequence[str] | None = None,
        tcr_sequences: Sequence[str] | None = None,
    ):
        if task == "phla" and hla_sequences is not None and tcr_sequences is None:
            return self.forward_ph(peptide_sequences, hla_sequences)
        if task == "ptcr" and tcr_sequences is not None and hla_sequences is None:
            return self.forward_pt(peptide_sequences, tcr_sequences)
        if task == "pht" and hla_sequences is not None and tcr_sequences is not None:
            return self.forward_pht(peptide_sequences, hla_sequences, tcr_sequences)
        raise ValueError(f"Invalid input combination for task {task!r}")

    def frozen_binding_checksum(self) -> str:
        digest = hashlib.sha256()
        pht_prefixes = tuple(f"{name}." for name in self.binding_model.pht_modules())
        if self.binding_model.train_shared_adapters:
            pht_prefixes += tuple(f"{name}." for name in self.binding_model.SHARED_ADAPTER_NAMES)
        for name, tensor in sorted(self.binding_model.state_dict().items()):
            if name.startswith(pht_prefixes) or not torch.is_tensor(tensor):
                continue
            digest.update(name.encode("utf-8"))
            # Hash raw storage bytes so this also works for bfloat16 tensors,
            # whose NumPy conversion is unsupported by some PyTorch versions.
            digest.update(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()


def build_progressive_stage_adapter_model(config, device: torch.device):
    precision_to_dtype = {
        "none": "float32", "fp16": "float16", "bf16": "bfloat16"
    }
    random_encoders = config.stage_adapter.initialization == "random"
    kwargs = {
        "device": device,
        # Train random PLMs with float32 master weights and autocast forwards.
        "dtype": "float32" if random_encoders else precision_to_dtype[config.stage_adapter.mixed_precision],
        "gradient_checkpointing": config.stage_adapter.gradient_checkpointing,
        "random_initialization": random_encoders,
    }
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.peptide_max_length, **kwargs
    )
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model, config.model.hla_max_length, **kwargs
    )
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model, config.model.tcr_max_length, **kwargs
    )
    dimensions = (
        (peptide_encoder.hidden_size, config.model.peptide_input_dim, "peptide"),
        (hla_encoder.hidden_size, config.model.hla_input_dim, "HLA"),
        (tcr_encoder.hidden_size, config.model.tcr_input_dim, "TCR"),
    )
    for actual, expected, name in dimensions:
        if actual != expected:
            raise ValueError(f"{name} encoder hidden size does not match config")
    binding = StageAdapterBindingModel(
        config.model,
        residual_adapter_dim=config.stage_adapter.residual_adapter_dim,
        train_shared_adapters=config.stage_adapter.initialization != "checkpoint",
    )
    return ProgressiveStageAdapterModel(
        binding, peptide_encoder, hla_encoder, tcr_encoder,
        initialization=config.stage_adapter.initialization,
    ).to(device)


def initialise_stage_adapter_random(model, *, config) -> dict:
    """Record constructor initialization without loading any task checkpoint."""
    if config.stage_adapter.initialization not in {"random", "random_binding"}:
        raise ValueError("Random initialization requires a random training configuration")
    model.set_trainable_pht()
    return {
        "kind": config.stage_adapter.initialization,
        "seed": config.runtime.seed,
        "fold": config.runtime.fold,
        "frozen_binding_checksum": model.frozen_binding_checksum(),
    }


def initialise_stage_adapter_from_stage1(
    model: ProgressiveStageAdapterModel,
    checkpoint_path: str | Path,
    *,
    config,
) -> dict:
    """Validate Stage-1, then copy only its binary/shared modules."""

    source_model = TripletBindingModel(config.model)
    source = initialise_binding_from_stage1(
        source_model,
        checkpoint_path,
        model_config=config.model,
        fold=config.runtime.fold,
        seed=config.runtime.seed,
        encoder_metadata=model.encoder_metadata(),
        expected_sha256=config.paths.stage1_sha256,
        require_joint_checkpoint=False,
    )
    for name in StageAdapterBindingModel.STAGE1_MODULE_NAMES:
        getattr(model.binding_model, name).load_state_dict(
            getattr(source_model, name).state_dict(), strict=True
        )
    del source_model
    model.set_trainable_pht()
    source["frozen_binding_checksum"] = model.frozen_binding_checksum()
    return source


def initialise_stage_adapter_from_stage2(
    model: ProgressiveStageAdapterModel,
    checkpoint_path: str | Path,
    *,
    config,
    _loaded_checkpoint: Mapping | None = None,
) -> dict:
    """Load all Stage-2 joint weights and leave the new PHT branch random."""

    source = Path(checkpoint_path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    source_sha256 = digest.hexdigest()
    if source_sha256 != config.paths.stage1_sha256:
        raise ValueError("Stage-2 checkpoint SHA256 does not match configured source")

    checkpoint = (
        _loaded_checkpoint
        if _loaded_checkpoint is not None
        else load_stage2_checkpoint(source, map_location="cpu")
    )
    if (
        checkpoint.get("checkpoint_type") != "stage2_training"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Expected a full Stage-2 training checkpoint")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("model") != config.to_dict()["model"]:
        raise ValueError("Stage-2 checkpoint model configuration does not match")
    source_training = checkpoint_config.get("training", {})
    for key, expected in (
        ("fold", config.runtime.fold),
        ("seed", config.runtime.seed),
    ):
        if source_training.get(key) != expected:
            raise ValueError(
                f"Stage-2 checkpoint {key}={source_training.get(key)} "
                f"does not match current {key}={expected}"
            )

    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    fingerprints: dict[str, str] = {}
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get(
            "checkpoint_fingerprint"
        )
        current_fingerprint = current_metadata[entity].get(
            "checkpoint_fingerprint"
        )
        if not saved_fingerprint or saved_fingerprint != current_fingerprint:
            raise ValueError(
                f"Stage-2 base-PLM fingerprint mismatch for {entity}"
            )
        fingerprints[entity] = saved_fingerprint

    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-2 checkpoint does not contain the full model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [key for key in unexpected if not key.endswith("._extra_state")]
    pht_prefixes = tuple(
        f"binding_model.{name}." for name in model.binding_model.pht_modules()
    )
    disallowed_missing = [
        key for key in missing if not key.startswith(pht_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Stage-2 to Stage-Adapter state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )

    model.set_trainable_pht()
    return {
        "kind": "stage2",
        "path": str(source),
        "sha256": source_sha256,
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "stage": checkpoint.get("stage"),
        "stage2a_round": checkpoint.get("stage2a_round"),
        "stage2b_round": checkpoint.get("stage2b_round"),
        "best_joint_score": checkpoint.get("best_joint_scores", {}).get(
            "stage2b"
        ),
        "run_timestamp": checkpoint.get("run_timestamp"),
        "source_seed": source_training.get("seed"),
        "source_fold": source_training.get("fold"),
        "encoder_fingerprints": fingerprints,
        "frozen_binding_checksum": model.frozen_binding_checksum(),
    }


def initialise_stage_adapter_from_checkpoint(
    model: ProgressiveStageAdapterModel,
    checkpoint_path: str | Path,
    *,
    config,
) -> dict:
    """Initialise from a compatible Stage-1 or full Stage-2 checkpoint."""

    source = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(
            source, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(source, map_location="cpu")
    if checkpoint.get("checkpoint_type") == "stage2_training":
        return initialise_stage_adapter_from_stage2(
            model,
            source,
            config=config,
            _loaded_checkpoint=checkpoint,
        )
    del checkpoint
    return initialise_stage_adapter_from_stage1(
        model, source, config=config
    )
