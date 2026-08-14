"""Stage-3 teacher lineage and Stage-10 checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .stage10_config import Stage10Condition, Stage10ExperimentConfig


_TRAINING_ONLY_KEYS = {
    "best_model",
    "model",
    "optimizer",
    "rng_state",
    "scaler",
    "scheduler",
    "teacher",
}


@dataclass
class Stage3TeacherSource:
    """Validated Stage-3 source used to initialize both Stage-10 models."""

    path: Path
    sha256: str
    checkpoint: dict[str, Any]
    binding_state: dict[str, torch.Tensor]
    lineage: dict[str, Any]


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_checkpoint(
    path: str | Path,
    *,
    mmap: bool = True,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=mmap,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Torch checkpoint root must be a mapping")
    return dict(checkpoint)


def _require_stage3_encoder_state(
    state: Mapping[str, Any],
    entity: str,
) -> None:
    if not any(name.startswith(f"{entity}_encoder.") for name in state):
        raise ValueError(f"Stage-3 checkpoint has no {entity} encoder state")


def load_stage3_teacher_source(
    config: Stage10ExperimentConfig,
) -> Stage3TeacherSource:
    """Load and strictly validate the Stage-3 Teacher/Student source."""

    path = config.paths.stage3_checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 teacher checkpoint not found: {path}")
    checkpoint = load_torch_checkpoint(path)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Stage 10 requires Stage-3 checkpoint format_version=1")
    if checkpoint.get("checkpoint_type") != "stage3_multitask_model":
        raise ValueError(
            "Stage 10 requires a stage3_multitask_model checkpoint"
        )
    if set(checkpoint.get("tasks", ())) != {"phla", "ptcr", "triplet"}:
        raise ValueError(
            "Stage-3 checkpoint must contain phla, ptcr, and triplet tasks"
        )

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Stage-3 checkpoint has no configuration")
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Stage-3 model configuration does not match Stage 10")
    training = saved_config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Stage-3 checkpoint has no training configuration")
    for key, expected in (
        ("fold", config.runtime.fold),
        ("seed", config.runtime.seed),
    ):
        if training.get(key) != expected:
            raise ValueError(
                f"Stage-3 {key}={training.get(key)!r} does not match "
                f"Stage 10 {expected}"
            )

    upstream = checkpoint.get("initialization_source")
    if not isinstance(upstream, Mapping) or upstream.get("kind") != "stage1":
        raise ValueError("Stage 10 requires Stage 3 initialised from Stage 1")
    for key, expected in (
        ("source_fold", config.runtime.fold),
        ("source_seed", config.runtime.seed),
    ):
        if upstream.get(key) != expected:
            raise ValueError(
                f"Stage-3 lineage {key}={upstream.get(key)!r} does not "
                f"match Stage 10 {expected}"
            )

    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-3 checkpoint has no model state")
    for entity in ("peptide", "hla", "tcr"):
        _require_stage3_encoder_state(state, entity)
    prefix = "binding_model."
    binding_state = {
        name[len(prefix) :]: tensor
        for name, tensor in state.items()
        if name.startswith(prefix)
    }
    if not binding_state:
        raise ValueError("Stage-3 checkpoint has no binding_model state")

    sha256 = checkpoint_sha256(path)
    lineage = {
        "kind": "stage3",
        "role": "teacher_initialization",
        "path": str(path),
        "sha256": sha256,
        "checkpoint_type": checkpoint["checkpoint_type"],
        "source_fold": config.runtime.fold,
        "source_seed": config.runtime.seed,
        "upstream": dict(upstream),
    }
    return Stage3TeacherSource(
        path=path,
        sha256=sha256,
        checkpoint=checkpoint,
        binding_state=binding_state,
        lineage=lineage,
    )


def atomic_torch_save(
    payload: Mapping[str, Any],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination


def stage10_teacher_policy(
    config: Stage10ExperimentConfig,
) -> dict[str, Any]:
    values = config.stage10
    return {
        "source_checkpoint_type": "stage3_multitask_model",
        "source_binding_prefix": "binding_model.",
        "frozen": True,
        "evaluation_mode": True,
        "distilled_tasks": ["phla", "ptcr"],
        "distillation_direction": "teacher_to_student",
        "distillation_reduction": "batchmean",
        "temperature": values.distillation_temperature,
        "keep_loss_weight": values.keep_loss_weight,
        "phla_keep_weight": values.phla_keep_weight,
        "ptcr_keep_weight": values.ptcr_keep_weight,
        "embedded_in_checkpoint": False,
    }


def stage10_training_contract(
    config: Stage10ExperimentConfig,
    effective_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    serialised = config.to_dict()
    descriptor = {
        "model": serialised["model"],
        "stage10": serialised["stage10"],
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
        "effective_data": dict(effective_data or {}),
        "teacher_policy": stage10_teacher_policy(config),
    }
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "descriptor": descriptor,
    }


def _validate_student_only_state(checkpoint: Mapping[str, Any]) -> None:
    if "teacher" in checkpoint:
        raise ValueError("Stage-10 checkpoints cannot contain teacher state")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Stage-10 checkpoint has no student model state")
    forbidden_prefixes = (
        "binding_model.",
        "hla_encoder.",
        "peptide_encoder.",
        "tcr_encoder.",
        "teacher.",
    )
    if any(name.startswith(forbidden_prefixes) for name in state):
        raise ValueError(
            "Lean Stage-10 model must contain unprefixed student binding "
            "state only, without encoders or teacher"
        )


def validate_stage10_lean_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Stage10ExperimentConfig,
    source_sha256: str,
) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Stage-10 checkpoint format")
    if checkpoint.get("checkpoint_type") != "stage10_binding_model":
        raise ValueError("Expected a stage10_binding_model checkpoint")
    source = checkpoint.get("source_stage3")
    if not isinstance(source, Mapping) or source.get("sha256") != source_sha256:
        raise ValueError(
            "Stage-10 checkpoint references a different Stage-3 source"
        )
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Stage-10 checkpoint has no configuration")
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Stage-10 model configuration mismatch")
    _validate_student_only_state(checkpoint)


def validate_stage10_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Stage10ExperimentConfig,
    source_sha256: str,
    effective_data: Mapping[str, Any] | None,
    *,
    condition: Stage10Condition | str | None = None,
) -> None:
    validate_stage10_lean_checkpoint(checkpoint, config, source_sha256)
    if checkpoint.get("artifact_role") != "last":
        raise ValueError("Stage-10 resume requires a last checkpoint artifact")
    if condition is not None:
        expected_condition = Stage10Condition(condition).value
        if checkpoint.get("condition") != expected_condition:
            raise ValueError("Stage-10 resume condition mismatch")
    expected = stage10_training_contract(config, effective_data)
    saved = checkpoint.get("training_contract")
    if (
        not isinstance(saved, Mapping)
        or saved.get("sha256") != expected["sha256"]
        or saved.get("descriptor") != expected["descriptor"]
    ):
        raise ValueError(
            "Stage-10 resume training contract mismatch; hyperparameters, "
            "teacher policy, or effective data changed"
        )


def extract_stage10_binding_state(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    checkpoint_type = checkpoint.get("checkpoint_type")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-10 checkpoint has no model state")
    if checkpoint_type == "stage10_binding_model":
        _validate_student_only_state(checkpoint)
        return dict(state)
    if checkpoint_type == "stage10_multitask_model":
        prefix = "binding_model."
        binding_state = {
            name[len(prefix) :]: tensor
            for name, tensor in state.items()
            if name.startswith(prefix)
        }
        if not binding_state:
            raise ValueError(
                "Full Stage-10 checkpoint has no student binding model state"
            )
        if any("teacher" in name.lower() for name in state):
            raise ValueError("Full Stage-10 checkpoint contains teacher state")
        return binding_state
    raise ValueError(
        f"Unsupported Stage-10 checkpoint type: {checkpoint_type!r}"
    )


def validate_stage10_full_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Stage10ExperimentConfig,
    source_sha256: str,
) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported full Stage-10 checkpoint format")
    if checkpoint.get("checkpoint_type") != "stage10_multitask_model":
        raise ValueError("Expected a stage10_multitask_model checkpoint")
    source = checkpoint.get("source_stage3")
    if not isinstance(source, Mapping) or source.get("sha256") != source_sha256:
        raise ValueError(
            "Full Stage-10 checkpoint references a different Stage-3 source"
        )
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Full Stage-10 checkpoint has no configuration")
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Full Stage-10 model configuration mismatch")
    if "teacher" in checkpoint:
        raise ValueError("Full Stage-10 checkpoint contains teacher state")
    for key in ("optimizer", "scheduler", "scaler", "rng_state", "best_model"):
        if key in checkpoint:
            raise ValueError(
                f"Full Stage-10 checkpoint contains training-only key {key!r}"
            )
    extract_stage10_binding_state(checkpoint)


def export_stage10_full_checkpoint(
    source: Stage3TeacherSource,
    lean_checkpoint: Mapping[str, Any],
    destination: str | Path,
) -> Path:
    """Merge Stage-10 Student binding weights with the Stage-3 encoders."""

    if lean_checkpoint.get("checkpoint_type") != "stage10_binding_model":
        raise ValueError("Stage-10 full export requires a lean checkpoint")
    lean_source = lean_checkpoint.get("source_stage3")
    if (
        not isinstance(lean_source, Mapping)
        or lean_source.get("sha256") != source.sha256
    ):
        raise ValueError(
            "Stage-10 checkpoint references a different Stage-3 source"
        )
    _validate_student_only_state(lean_checkpoint)

    full = {
        key: value
        for key, value in source.checkpoint.items()
        if key not in _TRAINING_ONLY_KEYS
    }
    source_state = {
        name: tensor
        for name, tensor in source.checkpoint["model"].items()
        if not name.startswith("binding_model.")
        and "teacher" not in name.lower()
    }
    for name, tensor in lean_checkpoint["model"].items():
        source_state[f"binding_model.{name}"] = tensor

    training_metadata = {
        key: value
        for key, value in lean_checkpoint.items()
        if key not in _TRAINING_ONLY_KEYS
    }
    full.update(
        {
            "format_version": 1,
            "checkpoint_type": "stage10_multitask_model",
            "tasks": ["phla", "ptcr", "triplet"],
            "model": source_state,
            "config": lean_checkpoint["config"],
            "initialization_source": source.lineage,
            "source_stage3": source.lineage,
            "stage10_condition": lean_checkpoint.get("condition"),
            "teacher_policy": lean_checkpoint.get("teacher_policy"),
            "stage10_training": training_metadata,
            "phase": "complete",
            "global_step": lean_checkpoint.get("global_step", 0),
            "run_timestamp": lean_checkpoint.get("run_timestamp"),
            "validation_metrics": lean_checkpoint.get("validation_metrics"),
            "validation_retention": lean_checkpoint.get(
                "validation_retention"
            ),
            "joint_score": lean_checkpoint.get("validation_joint_score"),
            "selected_threshold": lean_checkpoint.get(
                "selected_pht_threshold"
            ),
        }
    )
    return atomic_torch_save(full, destination)
