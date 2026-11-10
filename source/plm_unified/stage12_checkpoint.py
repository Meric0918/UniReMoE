"""Checkpoint schema for independent Stage12 online models."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .artifacts import artifact_timestamp
from .stage8_checkpoint import atomic_torch_save, load_torch_checkpoint
from .stage12_config import Stage12ExperimentConfig
from .stage12_evaluation import Stage12ValidationScores

STAGE12_FORMAT_VERSION = 1
STAGE12_CHECKPOINT_TYPE = "stage12_online_model"
STAGE12_ARTIFACT_ROLES = {"warmup", "best", "last"}
FULL_MODEL_PREFIXES = (
    "peptide_encoder.", "hla_encoder.", "tcr_encoder.", "binding_model."
)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _validate_source(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage12 checkpoint has no {name} lineage")
    if not isinstance(value.get("path"), str) or not _valid_sha(value.get("sha256")):
        raise ValueError(f"Stage12 checkpoint has invalid {name} lineage")


def validate_stage12_checkpoint(
    checkpoint: Mapping[str, Any], *, artifact_role: str | None = None
) -> None:
    if checkpoint.get("format_version") != STAGE12_FORMAT_VERSION:
        raise ValueError("Unsupported Stage12 checkpoint format")
    if checkpoint.get("checkpoint_type") != STAGE12_CHECKPOINT_TYPE:
        raise ValueError("Expected a Stage12 online-model checkpoint")
    role = checkpoint.get("artifact_role")
    if role not in STAGE12_ARTIFACT_ROLES:
        raise ValueError("Stage12 checkpoint has an invalid artifact role")
    if artifact_role is not None and role != artifact_role:
        raise ValueError(f"Expected Stage12 {artifact_role} checkpoint, got {role!r}")
    _validate_source(checkpoint.get("source_stage1"), "Stage1 source")
    _validate_source(checkpoint.get("reference_stage8"), "Stage8 reference")
    if not isinstance(checkpoint.get("dataset_fingerprints"), Mapping):
        raise ValueError("Stage12 checkpoint has no dataset fingerprints")
    if not isinstance(checkpoint.get("effective_data"), Mapping):
        raise ValueError("Stage12 checkpoint has no effective data contract")
    if not isinstance(checkpoint.get("training_contract"), Mapping):
        raise ValueError("Stage12 checkpoint has no training contract")
    if not isinstance(checkpoint.get("training_state"), Mapping):
        raise ValueError("Stage12 checkpoint has no training state")
    if not isinstance(checkpoint.get("metrics"), Mapping):
        raise ValueError("Stage12 checkpoint has no metrics")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage12 checkpoint has no full online model")
    for prefix in FULL_MODEL_PREFIXES:
        if not any(str(name).startswith(prefix) for name in state):
            raise ValueError(f"Stage12 checkpoint model has no {prefix.rstrip('.')} state")


def stage12_training_contract(
    config: Stage12ExperimentConfig,
    effective_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = config.stage12
    return {
        "condition": values.condition,
        "stage12a_epochs": values.stage12a_epochs,
        "pass_order": ["phla", "ptcr", "joint"],
        "phla_replay_ratio": values.phla_replay_ratio,
        "ptcr_replay_ratio": values.ptcr_replay_ratio,
        "loss_weights": {
            "phla": values.phla_loss_weight,
            "ptcr": values.ptcr_loss_weight,
            "pht": values.pht_loss_weight,
        },
        "nominal_task_weights": [
            round(value, 12) for value in values.normalized_task_weights
        ],
        "constraints": {
            "hierarchy_lambda": values.hierarchy_lambda,
            "pt_implication_lambda": values.pt_implication_lambda,
            "margin": values.constraint_margin,
            "ramp_rounds": values.constraint_ramp_rounds,
        },
        "selection": {
            "threshold": values.fixed_threshold,
            "task_weights": list(values.task_selection_weights),
            "pht_head_weights": list(values.pht_joint_weights),
            "pt_splits": list(values.pt_selection_splits),
            "ph_splits": list(values.ph_selection_splits),
        },
        "effective_data": dict(effective_data or {}),
    }


def _capture_rng_state(trainer) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataloaders": {},
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    for name, loader in trainer.train_loaders.items():
        generator = getattr(loader, "generator", None)
        if generator is not None:
            state["dataloaders"][name] = generator.get_state()
    return state


def _scores_dict(scores: Stage12ValidationScores | Mapping[str, Any]) -> dict:
    return scores.to_dict() if isinstance(scores, Stage12ValidationScores) else dict(scores)


def save_stage12_checkpoint(
    path: str | Path,
    *,
    artifact_role: str,
    trainer,
    config: Stage12ExperimentConfig,
    source_stage1: Mapping[str, Any],
    reference_stage8: Mapping[str, Any],
    dataset_fingerprints: Mapping[str, str],
    effective_data: Mapping[str, Any],
    reference_scores: Stage12ValidationScores | Mapping[str, Any],
    current_scores: Stage12ValidationScores | Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Path:
    if artifact_role not in STAGE12_ARTIFACT_ROLES:
        raise ValueError(f"Unknown Stage12 artifact role: {artifact_role}")
    _validate_source(source_stage1, "Stage1 source")
    _validate_source(reference_stage8, "Stage8 reference")
    if source_stage1["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage12 Stage1 source SHA does not match config")
    if reference_stage8["sha256"] != config.paths.stage8_reference_sha256:
        raise ValueError("Stage12 Stage8 reference SHA does not match config")
    destination = Path(path)
    base_name = f"stage12_{artifact_role}.pt"
    timestamp = artifact_timestamp(destination, base_name)
    if timestamp is None:
        raise ValueError(f"Stage12 checkpoint filename must be timestamped from {base_name}")
    payload: dict[str, Any] = {
        "format_version": STAGE12_FORMAT_VERSION,
        "checkpoint_type": STAGE12_CHECKPOINT_TYPE,
        "artifact_role": artifact_role,
        "run_timestamp": timestamp,
        "source_stage1": dict(source_stage1),
        "reference_stage8": dict(reference_stage8),
        "dataset_fingerprints": dict(dataset_fingerprints),
        "effective_data": dict(effective_data),
        "training_contract": stage12_training_contract(config, effective_data),
        "config": config.to_dict(),
        "training_state": trainer.training_state(),
        "reference_scores": _scores_dict(reference_scores),
        "metrics": {
            "validation": _scores_dict(current_scores),
            "decision": dict(decision),
        },
        "model": trainer.model.state_dict(),
    }
    if artifact_role == "last":
        payload.update({
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "rng_state": _capture_rng_state(trainer),
        })
    validate_stage12_checkpoint(payload, artifact_role=artifact_role)
    return atomic_torch_save(payload, destination)


def load_stage12_checkpoint(
    path: str | Path, *, artifact_role: str | None = None
) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(path)
    validate_stage12_checkpoint(checkpoint, artifact_role=artifact_role)
    return checkpoint


def restore_stage12_last_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    trainer,
    config: Stage12ExperimentConfig,
    dataset_fingerprints: Mapping[str, str],
    effective_data: Mapping[str, Any],
) -> None:
    validate_stage12_checkpoint(checkpoint, artifact_role="last")
    if checkpoint["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage12 resume Stage1 source mismatch")
    if checkpoint["reference_stage8"]["sha256"] != config.paths.stage8_reference_sha256:
        raise ValueError("Stage12 resume Stage8 reference mismatch")
    if dict(checkpoint["dataset_fingerprints"]) != dict(dataset_fingerprints):
        raise ValueError("Stage12 resume dataset fingerprints mismatch")
    if dict(checkpoint["effective_data"]) != dict(effective_data):
        raise ValueError("Stage12 resume effective data mismatch")
    if checkpoint["training_contract"] != stage12_training_contract(config, effective_data):
        raise ValueError("Stage12 resume training contract mismatch")
    trainer.model.load_state_dict(checkpoint["model"], strict=True)
    state = checkpoint["training_state"]
    phase = state.get("phase")
    if phase == "stage12b":
        trainer.stage12a_epoch = int(state["stage12a_epoch"])
        trainer.begin_stage12b()
    elif phase != "stage12a":
        raise ValueError(f"Unknown Stage12 resume phase: {phase!r}")
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    trainer.phase = phase
    trainer.stage12a_epoch = int(state["stage12a_epoch"])
    trainer.stage12b_round = int(state["stage12b_round"])
    trainer.global_step = int(state["global_step"])
    trainer.bad_rounds = int(state["bad_rounds"])
    trainer.best_joint_score_seen = float(state["best_joint_score_seen"])
    trainer.selection_state = dict(state.get("selection_state", {}))
    rng = checkpoint.get("rng_state", {})
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    for name, generator_state in rng.get("dataloaders", {}).items():
        loader = trainer.train_loaders.get(name)
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.set_state(generator_state)
