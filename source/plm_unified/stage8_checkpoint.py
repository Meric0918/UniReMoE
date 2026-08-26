"""Stage-8 V2 checkpoints plus retained legacy compatibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .artifacts import artifact_timestamp, validate_artifact_timestamp
from .stage8_config import Stage8ExperimentConfig


STAGE8_V2_FORMAT_VERSION = 2
STAGE8_V2_CHECKPOINT_TYPE = "stage8_online_model"
_FULL_MODEL_PREFIXES = (
    "peptide_encoder.",
    "hla_encoder.",
    "tcr_encoder.",
    "binding_model.",
)


@dataclass
class Stage3Source:
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


def load_torch_checkpoint(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_stage3_source(config: Stage8ExperimentConfig) -> Stage3Source:
    path = config.paths.stage3_checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 checkpoint not found: {path}")
    checkpoint = load_torch_checkpoint(path)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Stage 8 requires Stage-3 checkpoint format_version=1")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in {"stage3_multitask_model", "stage3_joint_model"}:
        raise ValueError(
            "Stage 8 requires a stage3_multitask_model or stage3_joint_model checkpoint"
        )
    if checkpoint_type == "stage3_multitask_model":
        if set(checkpoint.get("tasks", ())) != {"phla", "ptcr", "triplet"}:
            raise ValueError(
                "Stage-3 checkpoint must contain phla, ptcr, and triplet tasks"
            )
    saved_config = checkpoint.get("config", {})
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Stage-3 model configuration does not match Stage 8")
    training = saved_config.get("training", {})
    for key, expected in (("fold", config.runtime.fold), ("seed", config.runtime.seed)):
        if training.get(key) != expected:
            raise ValueError(
                f"Stage-3 {key}={training.get(key)!r} does not match {expected}"
            )
    upstream = checkpoint.get("initialization_source")
    if not isinstance(upstream, Mapping) or upstream.get("kind") != "stage1":
        raise ValueError("Stage 8 requires Stage 3 initialised from Stage 1")
    for key, expected in (
        ("source_fold", config.runtime.fold), ("source_seed", config.runtime.seed)
    ):
        if upstream.get(key) != expected:
            raise ValueError(f"Stage-3 lineage {key} does not match Stage 8")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-3 checkpoint has no model state")
    if checkpoint_type == "stage3_joint_model":
        required_heads = (
            "binding_model.phla_classifier.",
            "binding_model.ptcr_classifier.",
            "binding_model.triplet_classifier.",
        )
        if not all(any(name.startswith(prefix) for name in state) for prefix in required_heads):
            raise ValueError(
                "Stage-3 joint checkpoint must contain phla, ptcr, and triplet heads"
            )
    prefix = "binding_model."
    binding_state = {
        name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)
    }
    if not binding_state:
        raise ValueError("Stage-3 checkpoint has no binding_model state")
    for entity in ("peptide", "hla", "tcr"):
        if not any(name.startswith(f"{entity}_encoder.") for name in state):
            raise ValueError(f"Stage-3 checkpoint has no {entity} encoder state")
    sha = checkpoint_sha256(path)
    lineage = {
        "kind": "stage3",
        "path": str(path),
        "sha256": sha,
        "checkpoint_type": checkpoint["checkpoint_type"],
        "source_fold": config.runtime.fold,
        "source_seed": config.runtime.seed,
        "upstream": dict(upstream),
    }
    return Stage3Source(path, sha, checkpoint, binding_state, lineage)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _stage8_v2_training_contract(
    config: Stage8ExperimentConfig,
) -> dict[str, Any]:
    descriptor = {
        "model": asdict(config.model),
        "stage8": asdict(config.stage8),
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
    }
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "descriptor": descriptor,
    }


def _normalise_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "validation_metrics",
        "validation_joint_score",
        "selected_pht_threshold",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(
            "Stage-8 checkpoint metrics are missing: " + ", ".join(missing)
        )
    return dict(metrics)


def _normalise_stage1_source(source: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(source)
    sha256 = value.get("sha256")
    if value.get("kind") != "stage1" or not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("Stage-8 V2 requires an exact Stage-1 source SHA256")
    return value


def _condition_value(condition: Any) -> str:
    return str(getattr(condition, "value", condition))


def _validate_full_model_state(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("Stage-8 V2 checkpoint has no full model state")
    for prefix in _FULL_MODEL_PREFIXES:
        if not any(name.startswith(prefix) for name in state):
            raise ValueError(
                f"Stage-8 V2 full model has no {prefix.removesuffix('.')} state"
            )
    return state


def _v2_base_checkpoint(
    *,
    model,
    config: Stage8ExperimentConfig,
    artifact_role: str,
    phase: str,
    source_stage1: Mapping[str, Any],
    source_stage8a: Mapping[str, Any] | None,
    dataset_fingerprint: str,
    condition: str | None,
    run_timestamp: str,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact_role not in {"warmup", "best", "last"}:
        raise ValueError(f"Unknown Stage-8 V2 artifact role: {artifact_role!r}")
    if phase not in {"stage8a", "stage8b"}:
        raise ValueError(f"Unknown Stage-8 V2 phase: {phase!r}")
    validate_artifact_timestamp(run_timestamp)
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError("Stage-8 V2 requires a data fingerprint")
    state = model.state_dict()
    _validate_full_model_state(state)
    source = _normalise_stage1_source(source_stage1)
    if source["sha256"] != config.paths.stage1_sha256:
        raise ValueError(
            "Stage-8 V2 Stage-1 source does not match the configured SHA256"
        )
    return {
        "format_version": STAGE8_V2_FORMAT_VERSION,
        "checkpoint_type": STAGE8_V2_CHECKPOINT_TYPE,
        "artifact_role": artifact_role,
        "phase": phase,
        "run_timestamp": run_timestamp,
        "condition": condition,
        "source_stage1": source,
        "source_stage8a": dict(source_stage8a) if source_stage8a else None,
        "dataset_fingerprint": dataset_fingerprint,
        "training_contract": _stage8_v2_training_contract(config),
        "config": config.to_dict(),
        "training_state": dict(training_state),
        "metrics": _normalise_metrics(metrics),
        "model": state,
    }


def _warmup_source(
    path: str | Path,
    *,
    source_stage1_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    checkpoint = load_stage8_v2_checkpoint(source_path, artifact_role="warmup")
    if (
        source_stage1_sha256 is not None
        and checkpoint["source_stage1"]["sha256"] != source_stage1_sha256
    ):
        raise ValueError("Stage-8 warmup references a different Stage-1 source")
    return {
        "path": str(source_path),
        "sha256": checkpoint_sha256(source_path),
        "run_timestamp": checkpoint["run_timestamp"],
        "checkpoint_type": checkpoint["checkpoint_type"],
    }


def save_stage8_warmup_checkpoint(
    output_dir: str | Path,
    *,
    model,
    config: Stage8ExperimentConfig,
    source_stage1: Mapping[str, Any],
    dataset_fingerprint: str,
    run_timestamp: str,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Write the shared encoder-inclusive Stage8A artifact."""

    payload = _v2_base_checkpoint(
        model=model,
        config=config,
        artifact_role="warmup",
        phase="stage8a",
        source_stage1=source_stage1,
        source_stage8a=None,
        dataset_fingerprint=dataset_fingerprint,
        condition=None,
        run_timestamp=run_timestamp,
        metrics=metrics,
        training_state=training_state or {},
    )
    destination = (
        Path(output_dir)
        / f"stage8a_best_pht_{run_timestamp}.pt"
    )
    return atomic_torch_save(payload, destination)


def save_stage8_best_checkpoint(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    run_timestamp: str,
    metrics: Mapping[str, Any],
) -> Path:
    """Write a lean-in-training-state but full-model Stage8B best artifact."""

    source = _normalise_stage1_source(source_stage1)
    warmup = _warmup_source(
        warmup_checkpoint, source_stage1_sha256=source["sha256"]
    )
    payload = _v2_base_checkpoint(
        model=trainer.model,
        config=trainer.config,
        artifact_role="best",
        phase="stage8b",
        source_stage1=source,
        source_stage8a=warmup,
        dataset_fingerprint=dataset_fingerprint,
        condition=_condition_value(trainer.condition),
        run_timestamp=run_timestamp,
        metrics=metrics,
        training_state=trainer.training_state(),
    )
    destination = Path(output_dir) / f"stage8_best_{run_timestamp}.pt"
    return atomic_torch_save(payload, destination)


def _capture_rng_state(trainer=None) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if trainer is not None:
        state["dataloaders"] = {
            name: loader.generator.get_state()
            for name, loader in trainer.train_loaders.items()
            if getattr(loader, "generator", None) is not None
        }
        state["sampler_epochs"] = {
            name: int(loader.batch_sampler.epoch)
            for name, loader in trainer.train_loaders.items()
            if hasattr(getattr(loader, "batch_sampler", None), "epoch")
        }
    return state


def _restore_rng_state(state: Mapping[str, Any] | None, trainer=None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if trainer is not None:
        for name, generator_state in state.get("dataloaders", {}).items():
            loader = trainer.train_loaders.get(name)
            if loader is not None and getattr(loader, "generator", None) is not None:
                loader.generator.set_state(generator_state)
        for name, epoch in state.get("sampler_epochs", {}).items():
            loader = trainer.train_loaders.get(name)
            sampler = getattr(loader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(epoch))


def save_stage8_last_checkpoint(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    run_timestamp: str,
    metrics: Mapping[str, Any],
    best_checkpoint: str | Path,
) -> Path:
    """Write the resumable Stage8B artifact without duplicating best weights."""

    source = _normalise_stage1_source(source_stage1)
    warmup = _warmup_source(
        warmup_checkpoint, source_stage1_sha256=source["sha256"]
    )
    best_path = Path(best_checkpoint).expanduser().resolve()
    expected_best = Path(output_dir).resolve() / f"stage8_best_{run_timestamp}.pt"
    if best_path != expected_best:
        raise ValueError(
            "Stage-8 last checkpoint requires its same-directory, "
            "same-timestamp best companion"
        )
    best = load_stage8_v2_checkpoint(best_path, artifact_role="best")
    payload = _v2_base_checkpoint(
        model=trainer.model,
        config=trainer.config,
        artifact_role="last",
        phase="stage8b",
        source_stage1=source,
        source_stage8a=warmup,
        dataset_fingerprint=dataset_fingerprint,
        condition=_condition_value(trainer.condition),
        run_timestamp=run_timestamp,
        metrics=metrics,
        training_state=trainer.training_state(),
    )
    if best.get("run_timestamp") != run_timestamp:
        raise ValueError("Stage-8 best companion has a different run timestamp")
    for key, expected in (
        ("condition", payload["condition"]),
        ("source_stage1", payload["source_stage1"]),
        ("source_stage8a", payload["source_stage8a"]),
        ("dataset_fingerprint", payload["dataset_fingerprint"]),
        ("training_contract", payload["training_contract"]),
    ):
        if best.get(key) != expected:
            raise ValueError(f"Stage-8 best companion {key} mismatch")
    payload.update({
        "best_companion": {
            "path": str(best_path),
            "sha256": checkpoint_sha256(best_path),
        },
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng_state": _capture_rng_state(trainer),
    })
    destination = Path(output_dir) / f"stage8_last_{run_timestamp}.pt"
    return atomic_torch_save(payload, destination)


def initialise_stage8b_checkpoints(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    run_timestamp: str,
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create the mandatory best/last pair before Stage8B round one."""

    state = trainer.training_state()
    if state.get("phase") != "stage8b" or state.get("stage8b_round") != 0:
        raise ValueError(
            "Stage-8 best companion initialization requires Stage8B round zero"
        )
    best = save_stage8_best_checkpoint(
        output_dir,
        trainer=trainer,
        source_stage1=source_stage1,
        warmup_checkpoint=warmup_checkpoint,
        dataset_fingerprint=dataset_fingerprint,
        run_timestamp=run_timestamp,
        metrics=metrics,
    )
    last = save_stage8_last_checkpoint(
        output_dir,
        trainer=trainer,
        source_stage1=source_stage1,
        warmup_checkpoint=warmup_checkpoint,
        dataset_fingerprint=dataset_fingerprint,
        run_timestamp=run_timestamp,
        metrics=metrics,
        best_checkpoint=best,
    )
    return best, last


def validate_stage8_v2_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    artifact_role: str | None = None,
) -> None:
    version = checkpoint.get("format_version")
    if version == 1:
        raise ValueError(
            "Stage-8 V1 checkpoints are not supported by the public Stage-8 flow"
        )
    if version != STAGE8_V2_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Stage-8 checkpoint format_version={version!r}; expected V2"
        )
    if checkpoint.get("checkpoint_type") != STAGE8_V2_CHECKPOINT_TYPE:
        raise ValueError("Expected a Stage-8 V2 online-model checkpoint")
    role = checkpoint.get("artifact_role")
    if role not in {"warmup", "best", "last"}:
        raise ValueError("Stage-8 V2 checkpoint has an invalid artifact role")
    if artifact_role is not None and role != artifact_role:
        raise ValueError(
            f"Expected Stage-8 {artifact_role} checkpoint, got {role!r}"
        )
    _validate_full_model_state(checkpoint.get("model"))
    _normalise_stage1_source(checkpoint.get("source_stage1", {}))
    if not checkpoint.get("dataset_fingerprint"):
        raise ValueError("Stage-8 V2 checkpoint has no data fingerprint")
    if not isinstance(checkpoint.get("training_contract"), Mapping):
        raise ValueError("Stage-8 V2 checkpoint has no training contract")
    if not isinstance(checkpoint.get("metrics"), Mapping):
        raise ValueError("Stage-8 V2 checkpoint has no metrics")
    validate_artifact_timestamp(str(checkpoint.get("run_timestamp")))


def load_stage8_v2_checkpoint(
    path: str | Path,
    *,
    artifact_role: str | None = None,
) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(path, mmap=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Torch checkpoint root must be a mapping")
    validate_stage8_v2_checkpoint(checkpoint, artifact_role=artifact_role)
    return dict(checkpoint)


def validate_stage8_warmup_for_run(
    checkpoint: Mapping[str, Any],
    *,
    config: Stage8ExperimentConfig,
    source_stage1_sha256: str,
    dataset_fingerprint: str,
) -> None:
    """Bind a shared warmup artifact to the current source/data/config."""

    validate_stage8_v2_checkpoint(checkpoint, artifact_role="warmup")
    if checkpoint["source_stage1"]["sha256"] != source_stage1_sha256:
        raise ValueError("Stage-8 warmup Stage-1 source SHA256 mismatch")
    if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage-8 warmup data fingerprint mismatch")
    expected = _stage8_v2_training_contract(config)
    if checkpoint.get("training_contract", {}).get("sha256") != expected["sha256"]:
        raise ValueError("Stage-8 warmup configuration mismatch")


def validate_stage8_v2_for_evaluation(
    checkpoint: Mapping[str, Any],
    *,
    config: Stage8ExperimentConfig,
    dataset_fingerprint: str | None = None,
) -> None:
    """Validate a public V2 artifact before direct online-model inference."""

    validate_stage8_v2_checkpoint(checkpoint)
    if checkpoint["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage-8 checkpoint Stage-1 source SHA256 mismatch")
    if (
        dataset_fingerprint is not None
        and checkpoint.get("dataset_fingerprint") != dataset_fingerprint
    ):
        raise ValueError("Stage-8 checkpoint data fingerprint mismatch")
    expected = _stage8_v2_training_contract(config)
    if checkpoint.get("training_contract", {}).get("sha256") != expected["sha256"]:
        raise ValueError("Stage-8 checkpoint configuration mismatch")


def _validate_resume_identity(
    last: Mapping[str, Any],
    best: Mapping[str, Any],
    *,
    config: Stage8ExperimentConfig,
    source_stage1_sha256: str,
    dataset_fingerprint: str,
    condition: Any,
) -> None:
    if last["source_stage1"]["sha256"] != source_stage1_sha256:
        raise ValueError("Stage-8 resume Stage-1 source SHA256 mismatch")
    condition_value = _condition_value(condition)
    if last.get("condition") != condition_value:
        raise ValueError("Stage-8 resume condition mismatch")
    if last.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage-8 resume data fingerprint mismatch")
    expected_contract = _stage8_v2_training_contract(config)
    if last.get("training_contract", {}).get("sha256") != expected_contract["sha256"]:
        raise ValueError("Stage-8 resume configuration mismatch")
    for key in (
        "run_timestamp", "condition", "source_stage1", "source_stage8a",
        "dataset_fingerprint", "training_contract",
    ):
        if best.get(key) != last.get(key):
            raise ValueError(f"Stage-8 best companion {key} mismatch")

    warmup_metadata = last.get("source_stage8a")
    if not isinstance(warmup_metadata, Mapping):
        raise ValueError("Stage-8 resume has no warmup source")
    warmup_path = Path(str(warmup_metadata.get("path", ""))).expanduser().resolve()
    if not warmup_path.is_file():
        raise FileNotFoundError(f"Stage-8 warmup checkpoint not found: {warmup_path}")
    actual_warmup_sha = checkpoint_sha256(warmup_path)
    if warmup_metadata.get("sha256") != actual_warmup_sha:
        raise ValueError("Stage-8 resume warmup checkpoint SHA256 mismatch")
    warmup = load_stage8_v2_checkpoint(warmup_path, artifact_role="warmup")
    if warmup["source_stage1"]["sha256"] != source_stage1_sha256:
        raise ValueError("Stage-8 resume warmup Stage-1 source mismatch")
    if warmup.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage-8 resume warmup data fingerprint mismatch")
    if warmup.get("training_contract", {}).get("sha256") != expected_contract["sha256"]:
        raise ValueError("Stage-8 resume warmup configuration mismatch")


def resume_stage8_last_checkpoint(
    checkpoint_path: str | Path,
    *,
    trainer,
    config: Stage8ExperimentConfig,
    source_stage1_sha256: str,
    dataset_fingerprint: str,
    condition: Any,
) -> dict[str, Any]:
    """Validate one same-run V2 last/best pair and restore online training."""

    last_path = Path(checkpoint_path).expanduser().resolve()
    timestamp = artifact_timestamp(last_path, "stage8_last.pt")
    if timestamp is None:
        raise ValueError("Stage-8 resume requires stage8_last_<timestamp>.pt")
    last = load_stage8_v2_checkpoint(last_path, artifact_role="last")
    if last.get("run_timestamp") != timestamp:
        raise ValueError("Stage-8 last filename and run timestamp mismatch")
    best_path = last_path.parent / f"stage8_best_{timestamp}.pt"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Stage-8 best companion not found: {best_path}"
        )
    companion = last.get("best_companion")
    if not isinstance(companion, Mapping):
        raise ValueError("Stage-8 last checkpoint has no best companion metadata")
    recorded_best = Path(str(companion.get("path", "")))
    if recorded_best.name != best_path.name:
        raise ValueError(
            "Stage-8 resume requires the same-directory, same-timestamp best companion"
        )
    if companion.get("sha256") != checkpoint_sha256(best_path):
        raise ValueError("Stage-8 resume best companion SHA256 mismatch")
    best = load_stage8_v2_checkpoint(best_path, artifact_role="best")
    _validate_resume_identity(
        last,
        best,
        config=config,
        source_stage1_sha256=source_stage1_sha256,
        dataset_fingerprint=dataset_fingerprint,
        condition=condition,
    )
    for key in ("optimizer", "scheduler", "scaler", "rng_state"):
        if key not in last:
            raise ValueError(f"Stage-8 last checkpoint has no {key} state")
    if "best_model" in last:
        raise ValueError("Stage-8 V2 last checkpoint must not duplicate best_model")

    trainer.model.load_state_dict(last["model"], strict=True)
    trainer.phase = "stage8a"
    trainer.stage8a_epoch = config.stage8.stage8a_epochs
    trainer._build_optimization("stage8b")
    trainer.phase = "stage8b"
    trainer.optimizer.load_state_dict(last["optimizer"])
    trainer.scheduler.load_state_dict(last["scheduler"])
    trainer.scaler.load_state_dict(last["scaler"])
    state = last.get("training_state", {})
    trainer.stage8a_epoch = int(state["stage8a_epoch"])
    trainer.stage8b_round = int(state["stage8b_round"])
    trainer.global_step = int(state["global_step"])
    trainer.bad_rounds = int(state["bad_rounds"])
    trainer.best_joint = float(state["best_joint"])
    _restore_rng_state(last["rng_state"], trainer)
    return last


def validate_stage8_lean_checkpoint(
    checkpoint: Mapping[str, Any], config: Stage8ExperimentConfig, source_sha256: str
) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Stage-8 checkpoint format")
    if checkpoint.get("checkpoint_type") != "stage8_binding_model":
        raise ValueError("Expected a stage8_binding_model checkpoint")
    if checkpoint.get("source_stage3", {}).get("sha256") != source_sha256:
        raise ValueError("Stage-8 checkpoint references a different Stage-3 source")
    if checkpoint.get("config", {}).get("model") != asdict(config.model):
        raise ValueError("Stage-8 model configuration mismatch")


def stage8_training_contract(
    config: Stage8ExperimentConfig,
    effective_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    descriptor = {
        "model": asdict(config.model),
        "stage8": asdict(config.stage8),
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
        "effective_data": dict(effective_data or {}),
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "descriptor": descriptor}


def validate_stage8_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Stage8ExperimentConfig,
    source_sha256: str,
    effective_data: Mapping[str, Any] | None,
) -> None:
    validate_stage8_lean_checkpoint(checkpoint, config, source_sha256)
    expected = stage8_training_contract(config, effective_data)
    saved = checkpoint.get("training_contract")
    if not isinstance(saved, Mapping) or saved.get("sha256") != expected["sha256"]:
        raise ValueError(
            "Stage-8 resume training contract mismatch; hyperparameters or "
            "effective data changed"
        )


def extract_stage8_binding_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    checkpoint_type = checkpoint.get("checkpoint_type")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-8 checkpoint has no model state")
    if checkpoint_type == "stage8_binding_model":
        return dict(state)
    if checkpoint_type in {"stage8_multitask_model", STAGE8_V2_CHECKPOINT_TYPE}:
        prefix = "binding_model."
        binding = {
            name[len(prefix):]: tensor
            for name, tensor in state.items() if name.startswith(prefix)
        }
        if not binding:
            raise ValueError("Full Stage-8 checkpoint has no binding model state")
        return binding
    raise ValueError(f"Unsupported Stage-8 checkpoint type: {checkpoint_type!r}")


def validate_stage8_full_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Stage8ExperimentConfig,
    source_sha256: str,
) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported full Stage-8 checkpoint format")
    if checkpoint.get("checkpoint_type") != "stage8_multitask_model":
        raise ValueError("Expected a stage8_multitask_model checkpoint")
    if checkpoint.get("source_stage3", {}).get("sha256") != source_sha256:
        raise ValueError("Full Stage-8 checkpoint references a different Stage-3 source")
    if checkpoint.get("config", {}).get("model") != asdict(config.model):
        raise ValueError("Full Stage-8 model configuration mismatch")
    extract_stage8_binding_state(checkpoint)


def export_stage8_full_checkpoint(
    source: Stage3Source,
    lean_checkpoint: Mapping[str, Any],
    destination: str | Path,
) -> Path:
    full = dict(source.checkpoint)
    source_state = {
        name: tensor
        for name, tensor in source.checkpoint["model"].items()
        if not name.startswith("binding_model.")
    }
    for name, tensor in lean_checkpoint["model"].items():
        source_state[f"binding_model.{name}"] = tensor
    training_metadata = {
        key: value for key, value in lean_checkpoint.items()
        if key not in {
            "model", "best_model", "optimizer", "scheduler", "scaler", "rng_state"
        }
    }
    full.update({
        "format_version": 1,
        "checkpoint_type": "stage8_multitask_model",
        "tasks": ["phla", "ptcr", "triplet"],
        "model": source_state,
        "config": lean_checkpoint["config"],
        "initialization_source": source.lineage,
        "stage8_condition": lean_checkpoint["condition"],
        "source_stage3": source.lineage,
        "stage8_training": training_metadata,
        "phase": "complete",
        "global_step": lean_checkpoint.get("global_step", 0),
        "run_timestamp": lean_checkpoint.get("run_timestamp"),
        "validation_metrics": lean_checkpoint.get("validation_metrics"),
        "joint_score": lean_checkpoint.get("validation_joint_score"),
        "selected_threshold": lean_checkpoint.get("selected_pht_threshold"),
    })
    return atomic_torch_save(full, destination)
