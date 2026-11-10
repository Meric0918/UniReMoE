"""Strict Stage-14 checkpoint and lineage handling."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .artifacts import validate_artifact_timestamp
from .stage14_config import Stage14Condition, Stage14ExperimentConfig
from .stage8_checkpoint import load_torch_checkpoint


STAGE14_FORMAT_VERSION = 1
STAGE14_CHECKPOINT_TYPE = "stage14_online_model"
_FULL_MODEL_PREFIXES = ("peptide_encoder.", "hla_encoder.", "tcr_encoder.", "binding_model.")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def stage14_training_contract(
    config: Stage14ExperimentConfig,
    effective_data: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = {
        "model": config.to_dict()["model"],
        "fusion": config.to_dict()["fusion"],
        "stage14": config.to_dict()["stage14"],
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
        "effective_data": dict(effective_data),
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "descriptor": descriptor}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _source(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage14 checkpoint has no valid Stage-1 source")
    source = dict(value)
    if source.get("kind") != "stage1" or not _is_sha256(source.get("sha256")):
        raise ValueError("Stage14 requires an exact Stage-1 source SHA256")
    return source


def _effective(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage14 checkpoint has no valid effective data")
    result = dict(value)
    required = {"limit_per_type", "train_samples", "validation_samples"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError("Stage14 effective data is missing: " + ", ".join(missing))
    limit = result["limit_per_type"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Stage14 effective limit_per_type must be positive or null")
    for name in ("train_samples", "validation_samples"):
        if not isinstance(result[name], int) or result[name] < 1:
            raise ValueError(f"Stage14 effective {name} must be positive")
    return result


def _metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage14 checkpoint has no valid metrics")
    required = {"validation_metrics", "validation_joint_score", "selected_pht_threshold"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("Stage14 checkpoint metrics are missing: " + ", ".join(missing))
    return dict(value)


def _validate_model_state(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("Stage14 checkpoint has no full model state")
    for prefix in _FULL_MODEL_PREFIXES:
        if not any(str(name).startswith(prefix) for name in state):
            raise ValueError(f"Stage14 model has no {prefix.removesuffix('.') } state")
    return state


def _base_payload(
    *,
    model,
    config: Stage14ExperimentConfig,
    artifact_role: str,
    phase: str,
    condition: str | None,
    source_stage1: Mapping[str, Any],
    source_stage14a: Mapping[str, Any] | None,
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact_role not in {"warmup", "best", "last"}:
        raise ValueError(f"Unknown Stage14 artifact role: {artifact_role}")
    if phase not in {"stage14a", "stage14b"}:
        raise ValueError(f"Unknown Stage14 phase: {phase}")
    validate_artifact_timestamp(run_timestamp)
    source = _source(source_stage1)
    if source["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage14 Stage-1 source does not match the configured SHA256")
    effective = _effective(effective_data)
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError("Stage14 requires a data fingerprint")
    _validate_model_state(model.state_dict())
    return {
        "format_version": STAGE14_FORMAT_VERSION,
        "checkpoint_type": STAGE14_CHECKPOINT_TYPE,
        "artifact_role": artifact_role,
        "phase": phase,
        "condition": condition,
        "run_timestamp": run_timestamp,
        "source_stage1": source,
        "source_stage14a": dict(source_stage14a) if source_stage14a else None,
        "dataset_fingerprint": dataset_fingerprint,
        "effective_data": effective,
        "training_contract": stage14_training_contract(config, effective),
        "config": config.to_dict(),
        "training_state": dict(training_state),
        "metrics": _metrics(metrics),
        "encoder_metadata": model.encoder_metadata(),
        "model": model.state_dict(),
    }


def _warmup_source(path: str | Path, source_sha256: str) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    checkpoint = load_stage14_checkpoint(source_path, artifact_role="warmup")
    if checkpoint["source_stage1"]["sha256"] != source_sha256:
        raise ValueError("Stage14 warmup references a different Stage-1 source")
    return {
        "path": str(source_path),
        "sha256": checkpoint_sha256(source_path),
        "run_timestamp": checkpoint["run_timestamp"],
        "checkpoint_type": checkpoint["checkpoint_type"],
    }


def save_stage14_warmup_checkpoint(
    output_dir: str | Path,
    *,
    model,
    config: Stage14ExperimentConfig,
    source_stage1: Mapping[str, Any],
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> Path:
    payload = _base_payload(
        model=model, config=config, artifact_role="warmup", phase="stage14a",
        condition=None, source_stage1=source_stage1, source_stage14a=None,
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=metrics, training_state=training_state,
    )
    return atomic_torch_save(
        payload,
        Path(output_dir) / f"stage14a_best_pht_{run_timestamp}.pt",
    )


def save_stage14_best_checkpoint(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
) -> Path:
    source = _source(source_stage1)
    payload = _base_payload(
        model=trainer.model, config=trainer.config, artifact_role="best", phase="stage14b",
        condition=trainer.condition.value, source_stage1=source,
        source_stage14a=_warmup_source(warmup_checkpoint, source["sha256"]),
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=metrics,
        training_state=trainer.training_state(),
    )
    return atomic_torch_save(payload, Path(output_dir) / f"stage14_best_{run_timestamp}.pt")


def _capture_rng_state(trainer) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
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


def _restore_rng_state(state: Mapping[str, Any], trainer) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, value in state.get("dataloaders", {}).items():
        loader = trainer.train_loaders.get(name)
        if loader is not None and getattr(loader, "generator", None) is not None:
            loader.generator.set_state(value)
    for name, epoch in state.get("sampler_epochs", {}).items():
        loader = trainer.train_loaders.get(name)
        sampler = getattr(loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))


def save_stage14_last_checkpoint(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
    best_checkpoint: str | Path,
) -> Path:
    source = _source(source_stage1)
    best_path = Path(best_checkpoint).expanduser().resolve()
    expected = Path(output_dir).expanduser().resolve() / f"stage14_best_{run_timestamp}.pt"
    if best_path != expected:
        raise ValueError("Stage14 last checkpoint requires the same-timestamp best companion")
    best = load_stage14_checkpoint(best_path, artifact_role="best")
    warmup = _warmup_source(warmup_checkpoint, source["sha256"])
    payload = _base_payload(
        model=trainer.model, config=trainer.config, artifact_role="last", phase="stage14b",
        condition=trainer.condition.value, source_stage1=source, source_stage14a=warmup,
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, metrics=metrics,
        training_state=trainer.training_state(),
    )
    for key in ("run_timestamp", "condition", "source_stage1", "source_stage14a",
                "dataset_fingerprint", "effective_data", "training_contract"):
        if best.get(key) != payload.get(key):
            raise ValueError(f"Stage14 best companion {key} mismatch")
    payload.update({
        "best_companion": {"path": str(best_path), "sha256": checkpoint_sha256(best_path)},
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng_state": _capture_rng_state(trainer),
    })
    return atomic_torch_save(payload, Path(output_dir) / f"stage14_last_{run_timestamp}.pt")


def initialise_stage14b_checkpoints(
    output_dir: str | Path,
    *,
    trainer,
    source_stage1: Mapping[str, Any],
    warmup_checkpoint: str | Path,
    dataset_fingerprint: str,
    effective_data: Mapping[str, Any],
    run_timestamp: str,
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    if trainer.training_state().get("phase") != "stage14b":
        raise ValueError("Stage14 best companion initialization requires Stage14B")
    best = save_stage14_best_checkpoint(
        output_dir, trainer=trainer, source_stage1=source_stage1,
        warmup_checkpoint=warmup_checkpoint, dataset_fingerprint=dataset_fingerprint,
        effective_data=effective_data, run_timestamp=run_timestamp, metrics=metrics,
    )
    last = save_stage14_last_checkpoint(
        output_dir, trainer=trainer, source_stage1=source_stage1,
        warmup_checkpoint=warmup_checkpoint, dataset_fingerprint=dataset_fingerprint,
        effective_data=effective_data, run_timestamp=run_timestamp, metrics=metrics,
        best_checkpoint=best,
    )
    return best, last


def validate_stage14_checkpoint(
    checkpoint: Mapping[str, Any], *, artifact_role: str | None = None
) -> None:
    if checkpoint.get("format_version") != STAGE14_FORMAT_VERSION:
        raise ValueError("Unsupported Stage14 checkpoint format")
    if checkpoint.get("checkpoint_type") != STAGE14_CHECKPOINT_TYPE:
        raise ValueError("Expected a Stage14 online-model checkpoint")
    role = checkpoint.get("artifact_role")
    if role not in {"warmup", "best", "last"}:
        raise ValueError("Stage14 checkpoint has an invalid artifact role")
    if artifact_role is not None and role != artifact_role:
        raise ValueError(f"Expected Stage14 {artifact_role} checkpoint, got {role!r}")
    _validate_model_state(checkpoint.get("model"))
    _source(checkpoint.get("source_stage1", {}))
    _effective(checkpoint.get("effective_data", {}))
    _metrics(checkpoint.get("metrics", {}))
    validate_artifact_timestamp(str(checkpoint.get("run_timestamp")))
    if not isinstance(checkpoint.get("training_contract"), Mapping):
        raise ValueError("Stage14 checkpoint has no training contract")
    if not isinstance(checkpoint.get("training_state"), Mapping):
        raise ValueError("Stage14 checkpoint has no training state")
    if role == "warmup":
        if checkpoint.get("phase") != "stage14a" or checkpoint.get("condition") is not None:
            raise ValueError("Stage14 warmup has invalid phase or condition")
        if checkpoint.get("source_stage14a") is not None:
            raise ValueError("Stage14 warmup cannot reference another warmup")
        for key in ("optimizer", "scheduler", "scaler", "rng_state", "best_companion"):
            if key in checkpoint:
                raise ValueError(f"Stage14 warmup cannot contain {key}")
    else:
        if checkpoint.get("phase") != "stage14b":
            raise ValueError("Stage14 best/last must be Stage14B artifacts")
        try:
            Stage14Condition(checkpoint.get("condition"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Stage14 checkpoint has an invalid condition") from exc
        warmup = checkpoint.get("source_stage14a")
        if not isinstance(warmup, Mapping) or not _is_sha256(warmup.get("sha256")):
            raise ValueError("Stage14 checkpoint has invalid warmup metadata")
        if role == "best":
            for key in ("optimizer", "scheduler", "scaler", "rng_state", "best_companion"):
                if key in checkpoint:
                    raise ValueError(f"Stage14 best cannot contain {key}")
        else:
            for key in ("optimizer", "scheduler", "scaler", "rng_state"):
                if not isinstance(checkpoint.get(key), Mapping):
                    raise ValueError(f"Stage14 last has no {key} state")
            if not isinstance(checkpoint.get("best_companion"), Mapping):
                raise ValueError("Stage14 last has no best companion")


def load_stage14_checkpoint(path: str | Path, *, artifact_role: str | None = None) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(path, mmap=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage14 checkpoint root must be a mapping")
    validate_stage14_checkpoint(checkpoint, artifact_role=artifact_role)
    return dict(checkpoint)


def validate_stage14_warmup_for_run(
    checkpoint: Mapping[str, Any], *, config: Stage14ExperimentConfig,
    dataset_fingerprint: str, effective_data: Mapping[str, Any]
) -> None:
    validate_stage14_checkpoint(checkpoint, artifact_role="warmup")
    if checkpoint["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage14 warmup Stage-1 source SHA256 mismatch")
    if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage14 warmup data fingerprint mismatch")
    expected = stage14_training_contract(config, effective_data)
    if checkpoint.get("training_contract", {}).get("sha256") != expected["sha256"]:
        raise ValueError("Stage14 warmup configuration mismatch")


def validate_stage14_for_evaluation(
    checkpoint: Mapping[str, Any], *, config: Stage14ExperimentConfig,
    dataset_fingerprint: str | None = None,
) -> None:
    validate_stage14_checkpoint(checkpoint)
    if checkpoint["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage14 checkpoint Stage-1 source SHA256 mismatch")
    if dataset_fingerprint is not None and checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage14 checkpoint data fingerprint mismatch")
    expected = stage14_training_contract(config, checkpoint["effective_data"])
    if checkpoint.get("training_contract", {}).get("sha256") != expected["sha256"]:
        raise ValueError("Stage14 checkpoint configuration mismatch")


def resume_stage14_last_checkpoint(
    checkpoint_path: str | Path, *, checkpoint: Mapping[str, Any] | None = None,
    trainer, config: Stage14ExperimentConfig, dataset_fingerprint: str,
    effective_data: Mapping[str, Any], condition: Stage14Condition | str,
) -> dict[str, Any]:
    last_path = Path(checkpoint_path).expanduser().resolve()
    last = dict(checkpoint) if checkpoint is not None else load_stage14_checkpoint(last_path, artifact_role="last")
    validate_stage14_checkpoint(last, artifact_role="last")
    if last["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage14 resume Stage-1 source SHA256 mismatch")
    best_path = last_path.parent / f"stage14_best_{last['run_timestamp']}.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Stage14 best companion not found: {best_path}")
    companion = last.get("best_companion", {})
    if Path(str(companion.get("path", ""))).expanduser().resolve() != best_path:
        raise ValueError("Stage14 resume requires the same-directory best companion")
    if companion.get("sha256") != checkpoint_sha256(best_path):
        raise ValueError("Stage14 resume best companion SHA256 mismatch")
    best = load_stage14_checkpoint(best_path, artifact_role="best")
    for key, expected in (
        ("condition", Stage14Condition(condition).value),
        ("source_stage1", last["source_stage1"]),
        ("source_stage14a", last["source_stage14a"]),
        ("dataset_fingerprint", dataset_fingerprint),
        ("effective_data", _effective(effective_data)),
        ("training_contract", stage14_training_contract(config, effective_data)),
    ):
        if last.get(key) != expected:
            raise ValueError(f"Stage14 resume {key} mismatch")
        if best.get(key) != expected:
            raise ValueError(f"Stage14 best companion {key} mismatch")
    warmup = Path(str(last["source_stage14a"]["path"])).expanduser().resolve()
    if not warmup.is_file() or checkpoint_sha256(warmup) != last["source_stage14a"]["sha256"]:
        raise ValueError("Stage14 resume warmup checkpoint is missing or changed")
    trainer.model.load_state_dict(last["model"], strict=True)
    trainer.phase = "stage8a"
    trainer.stage8a_epoch = config.stage14.stage8a_epochs
    trainer._build_optimization("stage8b")
    trainer.phase = "stage8b"
    trainer.optimizer.load_state_dict(last["optimizer"])
    trainer.scheduler.load_state_dict(last["scheduler"])
    trainer.scaler.load_state_dict(last["scaler"])
    state = last["training_state"]
    trainer.stage8a_epoch = int(state.get("stage14a_epoch", state.get("stage8a_epoch", config.stage14.stage8a_epochs)))
    trainer.stage8b_round = int(state.get("stage14b_round", state.get("stage8b_round", 0)))
    trainer.global_step = int(state["global_step"])
    trainer.bad_rounds = int(state["bad_rounds"])
    trainer.best_joint = float(state["best_joint"])
    _restore_rng_state(last["rng_state"], trainer)
    return last
