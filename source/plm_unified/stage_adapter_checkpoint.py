"""Self-contained checkpoints for Stage Adapter."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .stage8_checkpoint import atomic_torch_save, checkpoint_sha256
from .stage_adapter_config import StageAdapterTrainingConfig

FORMAT_VERSION = 2
CHECKPOINT_TYPE = "stage_adapter_v2_full_model"


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluation_payload(evaluation) -> dict[str, Any]:
    return {
        "pht": evaluation.metrics.to_dict(),
        "pht_selected": evaluation.selected_metrics.to_dict(),
        "selected_pht_threshold": evaluation.selected_threshold,
        "selection_score": evaluation.selection_score,
    }


def _base_payload(
    *,
    role: str,
    phase: str,
    model,
    config,
    source_stage1: Mapping[str, Any],
    source_warmup: Mapping[str, Any] | None,
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    checksum = model.frozen_binding_checksum()
    expected = source_stage1.get("frozen_binding_checksum")
    if expected and checksum != expected:
        raise RuntimeError("Frozen binary binding parameters changed during training")
    random_modules = sorted(model.binding_model.pht_modules())
    if config.stage_adapter.initialization != "checkpoint":
        random_modules = ["binding_model"]
        if config.stage_adapter.initialization == "random":
            random_modules.extend(["peptide_encoder", "hla_encoder", "tcr_encoder"])
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        "artifact_role": role,
        "phase": phase,
        "run_timestamp": run_timestamp,
        "source_stage1": dict(source_stage1),
        "source_initialization": dict(source_stage1),
        "source_warmup": dict(source_warmup) if source_warmup else None,
        "dataset_fingerprint": dataset_fingerprint,
        "effective_data": dict(effective_data),
        "random_initialization_seed": config.runtime.seed,
        "randomly_initialized_modules": random_modules,
        "frozen_binding_checksum": checksum,
        "encoder_metadata": model.encoder_metadata(),
        "config": config.to_dict(),
        "training_state": dict(training_state),
        "metrics": dict(metrics),
        "model": model.state_dict(),
    }


def save_stage_adapter_warmup(
    output_dir: str | Path,
    *, model, config, source_stage1, dataset_fingerprint: str,
    effective_data, run_timestamp: str, evaluation, training_state,
) -> Path:
    payload = _base_payload(
        role="warmup", phase="stage_adapter_a", model=model, config=config,
        source_stage1=source_stage1, source_warmup=None,
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=evaluation_payload(evaluation),
        training_state=training_state,
    )
    return atomic_torch_save(
        payload, Path(output_dir) / f"stage_adapter_warmup_{run_timestamp}.pt"
    )


def _warmup_source(path: str | Path) -> dict[str, Any]:
    value = load_stage_adapter_checkpoint(path, role="warmup")
    return {
        "path": str(Path(path).resolve()),
        "sha256": checkpoint_sha256(path),
        "run_timestamp": value["run_timestamp"],
    }


def save_stage_adapter_best(
    output_dir: str | Path,
    *, trainer, source_stage1, warmup_checkpoint, dataset_fingerprint: str,
    effective_data, run_timestamp: str, evaluation,
) -> Path:
    payload = _base_payload(
        role="best", phase="stage_adapter_b", model=trainer.model,
        config=trainer.config, source_stage1=source_stage1,
        source_warmup=_warmup_source(warmup_checkpoint),
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=evaluation_payload(evaluation),
        training_state=trainer.training_state(),
    )
    return atomic_torch_save(
        payload, Path(output_dir) / f"stage_adapter_best_{run_timestamp}.pt"
    )


def save_stage_adapter_last(
    output_dir: str | Path,
    *, trainer, source_stage1, warmup_checkpoint, best_checkpoint,
    dataset_fingerprint: str, effective_data, run_timestamp: str, evaluation,
) -> Path:
    payload = _base_payload(
        role="last", phase="stage_adapter_b", model=trainer.model,
        config=trainer.config, source_stage1=source_stage1,
        source_warmup=_warmup_source(warmup_checkpoint),
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=evaluation_payload(evaluation),
        training_state=trainer.training_state(),
    )
    payload.update({
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng_state": _rng_state(),
        "best_checkpoint": {
            "path": str(Path(best_checkpoint).resolve()),
            "sha256": checkpoint_sha256(best_checkpoint),
        },
    })
    return atomic_torch_save(
        payload, Path(output_dir) / f"stage_adapter_last_{run_timestamp}.pt"
    )


def load_stage_adapter_checkpoint(
    path: str | Path, *, role: str | None = None, mmap: bool = True
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    checkpoint = torch.load(source, **kwargs)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Stage Adapter checkpoint format")
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Not a Stage Adapter full-model checkpoint")
    if role is not None and checkpoint.get("artifact_role") != role:
        raise ValueError(f"Expected Stage Adapter {role} checkpoint")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("Stage Adapter checkpoint has no model state")
    return checkpoint


def validate_stage_adapter_checkpoint(
    checkpoint: Mapping[str, Any], *, config, dataset_fingerprint: str,
    effective_data: Mapping[str, Any], warmup_checkpoint: str | Path | None = None,
) -> None:
    source = checkpoint.get("source_stage1") or {}
    initialization = config.stage_adapter.initialization
    saved_config = checkpoint.get("config") or {}
    saved_initialization = saved_config.get("stage_adapter", {}).get("initialization", "checkpoint")
    if saved_initialization != initialization:
        raise ValueError("Stage Adapter initialization mode mismatch")
    if initialization == "checkpoint" and source.get("sha256") != config.paths.stage1_sha256:
        raise ValueError("Stage Adapter Stage-1 source mismatch")
    if initialization != "checkpoint" and (
        source.get("kind") != initialization or source.get("seed") != config.runtime.seed
        or source.get("fold") != config.runtime.fold
    ):
        raise ValueError("Stage Adapter random initialization source mismatch")
    if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage Adapter dataset fingerprint mismatch")
    if dict(checkpoint.get("effective_data") or {}) != dict(effective_data):
        raise ValueError("Stage Adapter effective-data mismatch")
    saved = checkpoint.get("config") or {}
    if saved.get("model") != config.to_dict().get("model"):
        raise ValueError("Stage Adapter model configuration mismatch")
    saved_training = dict(saved.get("stage_adapter") or {})
    # V2 checkpoints saved before these options used checkpoint initialization.
    defaults = StageAdapterTrainingConfig()
    saved_training.setdefault("initialization", defaults.initialization)
    saved_training.setdefault("encoder_lr", defaults.encoder_lr)
    if saved_training != config.to_dict().get("stage_adapter"):
        raise ValueError("Stage Adapter training configuration mismatch")
    if warmup_checkpoint is not None:
        warmup = checkpoint.get("source_warmup") or {}
        if warmup.get("sha256") != checkpoint_sha256(warmup_checkpoint):
            raise ValueError("Stage Adapter warmup SHA256 mismatch")


def restore_stage_adapter_model(checkpoint: Mapping[str, Any], model) -> None:
    """Restore weights only into a model with matching architecture and assets."""
    saved_config = checkpoint.get("config") or {}
    saved_mode = saved_config.get("stage_adapter", {}).get("initialization", "checkpoint")
    if saved_mode != model.initialization:
        raise ValueError("Stage Adapter initialization mode mismatch")
    saved_metadata = checkpoint.get("encoder_metadata") or {}
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        fingerprint = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        if fingerprint and fingerprint != current_metadata[entity].get("checkpoint_fingerprint"):
            raise ValueError(f"Stage Adapter encoder assets mismatch for {entity}")
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.frozen_binding_checksum() != checkpoint["frozen_binding_checksum"]:
        raise ValueError("Stage Adapter frozen binding checksum mismatch")


def resume_stage_adapter_last(
    checkpoint: Mapping[str, Any], *, trainer, config,
    dataset_fingerprint: str, effective_data: Mapping[str, Any],
) -> None:
    if checkpoint.get("artifact_role") != "last":
        raise ValueError("Stage Adapter resume requires a last checkpoint")
    validate_stage_adapter_checkpoint(
        checkpoint, config=config, dataset_fingerprint=dataset_fingerprint,
        effective_data=effective_data,
    )
    best = checkpoint.get("best_checkpoint") or {}
    best_path = Path(best.get("path", ""))
    if not best_path.is_file() or checkpoint_sha256(best_path) != best.get("sha256"):
        raise ValueError("Stage Adapter best companion is missing or changed")
    warmup = checkpoint.get("source_warmup") or {}
    warmup_path = Path(warmup.get("path", ""))
    if (
        not warmup_path.is_file()
        or checkpoint_sha256(warmup_path) != warmup.get("sha256")
    ):
        raise ValueError("Stage Adapter warmup companion is missing or changed")
    restore_stage_adapter_model(checkpoint, trainer.model)
    trainer.stage_adapter_a_epoch = config.stage_adapter.stage_adapter_a_epochs
    trainer.begin_stage_adapter_b()
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    trainer.scaler.load_state_dict(checkpoint.get("scaler", {}))
    state = checkpoint["training_state"]
    trainer.stage_adapter_b_round = int(state["stage_adapter_b_round"])
    trainer.global_step = int(state["global_step"])
    trainer.bad_rounds = int(state["bad_rounds"])
    trainer.best_score = float(state["best_score"])
    _restore_rng_state(checkpoint.get("rng_state"))
