"""Validate or precompute Stage-3 residue caches for Stage 10."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping

import torch

from .cache import EmbeddingCacheWriter, batched
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .stage8_data import Stage8AuditResult, open_stage8_caches
from .stage10_checkpoint import Stage3TeacherSource
from .stage10_config import Stage10ExperimentConfig


def _encoder_dtype(
    config: Stage10ExperimentConfig,
    device: torch.device,
) -> str:
    value = config.stage10.mixed_precision
    if value == "none" or device.type != "cuda":
        return "float32"
    return "float16" if value == "fp16" else "bfloat16"


def load_stage3_encoder(
    config: Stage10ExperimentConfig,
    checkpoint_state: Mapping[str, torch.Tensor],
    entity: str,
    *,
    device: torch.device,
):
    dtype = _encoder_dtype(config, device)
    if entity in {"peptide", "hla"}:
        max_length = (
            config.model.peptide_max_length
            if entity == "peptide"
            else config.model.hla_max_length
        )
        encoder = ESMCFineTuningEncoder(
            config.paths.esmc_model,
            max_length,
            device,
            dtype=dtype,
            gradient_checkpointing=False,
        )
    elif entity == "tcr":
        encoder = TCRBertFineTuningEncoder(
            config.paths.tcr_bert_model,
            config.model.tcr_max_length,
            device,
            dtype=dtype,
            gradient_checkpointing=False,
        )
    else:
        raise ValueError(f"Unknown Stage-10 cache entity: {entity}")
    prefix = f"{entity}_encoder."
    state = {
        name[len(prefix) :]: tensor
        for name, tensor in checkpoint_state.items()
        if name.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    missing = [
        name for name in missing if not name.endswith("._extra_state")
    ]
    unexpected = [
        name for name in unexpected if not name.endswith("._extra_state")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Stage-3 {entity} encoder mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    encoder.requires_grad_(False).eval()
    encoder.set_active_training(False)
    return encoder


def _unique_sequences(audit: Stage8AuditResult, entity: str) -> list[str]:
    return sorted(
        {
            getattr(record, entity)
            for record in (*audit.train_records, *audit.val_records)
        }
    )


def _validate_existing_cache(
    config: Stage10ExperimentConfig,
    source: Stage3TeacherSource,
    audit: Stage8AuditResult,
) -> Path:
    root = config.cache_dir(source.sha256)
    caches = open_stage8_caches(
        root,
        source_sha256=source.sha256,
        model=config.model,
        dataset_fingerprint=audit.dataset_fingerprint,
    )
    try:
        return root
    finally:
        for cache in caches.values():
            cache.close()


def precompute_stage10_caches(
    config: Stage10ExperimentConfig,
    source: Stage3TeacherSource,
    audit: Stage8AuditResult,
    *,
    device: torch.device,
    overwrite: bool = False,
) -> Path:
    """Reuse a valid shared cache or build it from the Stage-3 encoders."""

    audit.require_clean()
    root = config.cache_dir(source.sha256)
    if root.exists() and not overwrite:
        return _validate_existing_cache(config, source, audit)

    state = source.checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(
            "Stage-3 checkpoint model state is required to build residue caches"
        )
    max_lengths = {
        "peptide": config.model.peptide_max_length,
        "hla": config.model.hla_max_length,
        "tcr": config.model.tcr_max_length,
    }
    dimensions = {
        "peptide": config.model.peptide_input_dim,
        "hla": config.model.hla_input_dim,
        "tcr": config.model.tcr_input_dim,
    }
    for entity in ("peptide", "hla", "tcr"):
        encoder = load_stage3_encoder(
            config,
            state,
            entity,
            device=device,
        )
        if encoder.hidden_size != dimensions[entity]:
            raise ValueError(f"{entity} encoder hidden dimension mismatch")
        metadata = encoder.metadata()
        metadata.update(
            {
                "checkpoint_fingerprint": source.sha256,
                "stage3_checkpoint_sha256": source.sha256,
                "stage3_checkpoint_path": str(source.path),
                "dataset_fingerprint": audit.dataset_fingerprint,
            }
        )
        sequences = _unique_sequences(audit, entity)
        with EmbeddingCacheWriter(
            root,
            entity,
            hidden_dim=dimensions[entity],
            max_length=max_lengths[entity],
            model_metadata=metadata,
            dtype=config.cache.dtype,
            shard_size=config.cache.shard_size,
            chunk_rows=1,
            compression=config.cache.compression,
            overwrite=overwrite,
        ) as writer:
            for sequence_batch in batched(
                sequences,
                config.cache.precompute_batch_size,
            ):
                with torch.inference_mode():
                    encoded = encoder(sequence_batch)
                writer.add_batch(
                    sequence_batch,
                    encoded.hidden.detach().cpu().to(torch.float16).numpy(),
                    encoded.mask.detach().cpu().numpy(),
                    encoded.lengths.detach().cpu().numpy(),
                )
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _validate_existing_cache(config, source, audit)
