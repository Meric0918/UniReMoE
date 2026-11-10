"""Architecture-specific full checkpoints and exact Stage PHT continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .stage8_checkpoint import atomic_torch_save, checkpoint_sha256
from .stage_adapter_checkpoint import _rng_state, _restore_rng_state, evaluation_payload
from .stage_PHT_model import ARCHITECTURE

FORMAT_VERSION = 1
CHECKPOINT_TYPE = "stage_PHT_full_model"


def _require_architecture(checkpoint):
    if (checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("format_version") != FORMAT_VERSION
            or checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE):
        raise ValueError("Expected a stage_PHT full-model checkpoint")


def _companion(path, *, role):
    value = load_stage_PHT_checkpoint(path, role=role)
    return {"path": str(Path(path).resolve()), "sha256": checkpoint_sha256(path),
            "run_timestamp": value["run_timestamp"]}


def _base_payload(*, role, phase, model, config, source_initialization,
                  dataset_fingerprint, effective_data, run_timestamp, evaluation,
                  training_state, source_warmup=None):
    return {
        "architecture": ARCHITECTURE,
        "format_version": FORMAT_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        "artifact_role": role,
        "phase": phase,
        "run_timestamp": run_timestamp,
        "source_initialization": dict(source_initialization),
        "source_warmup": source_warmup,
        "dataset_fingerprint": dataset_fingerprint,
        "effective_data": dict(effective_data),
        "random_initialization_seed": config.runtime.seed,
        "randomly_initialized_modules": ["binding_model"],
        "encoder_metadata": model.encoder_metadata(),
        "config": config.to_dict(),
        "training_state": dict(training_state),
        "metrics": evaluation_payload(evaluation),
        "model": model.state_dict(),
    }


def save_stage_PHT_warmup(output_dir, *, model, config, source_initialization,
                          dataset_fingerprint, effective_data, run_timestamp,
                          evaluation, training_state):
    payload = _base_payload(
        role="warmup", phase="warmup", model=model, config=config,
        source_initialization=source_initialization,
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, evaluation=evaluation, training_state=training_state,
    )
    return atomic_torch_save(payload, Path(output_dir) / f"stage_PHT_warmup_{run_timestamp}.pt")


def _training_payload(*, role, trainer, source_initialization, warmup_checkpoint,
                      dataset_fingerprint, effective_data, run_timestamp, evaluation):
    return _base_payload(
        role=role, phase="training", model=trainer.model, config=trainer.config,
        source_initialization=source_initialization,
        source_warmup=_companion(warmup_checkpoint, role="warmup"),
        dataset_fingerprint=dataset_fingerprint, effective_data=effective_data,
        run_timestamp=run_timestamp, evaluation=evaluation,
        training_state=trainer.training_state(),
    )


def save_stage_PHT_best(output_dir, **kwargs):
    payload = _training_payload(role="best", **kwargs)
    return atomic_torch_save(payload, Path(output_dir) / f"stage_PHT_best_{payload['run_timestamp']}.pt")


def save_stage_PHT_last(output_dir, *, best_checkpoint, **kwargs):
    trainer = kwargs["trainer"]
    payload = _training_payload(role="last", **kwargs)
    payload.update({
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng_state": _rng_state(),
        "best_checkpoint": _companion(best_checkpoint, role="best"),
    })
    return atomic_torch_save(payload, Path(output_dir) / f"stage_PHT_last_{payload['run_timestamp']}.pt")


def load_stage_PHT_checkpoint(path, *, role=None, mmap=True):
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    checkpoint = torch.load(Path(path).expanduser().resolve(), **kwargs)
    _require_architecture(checkpoint)
    if role is not None and checkpoint.get("artifact_role") != role:
        raise ValueError(f"Expected Stage PHT {role} checkpoint")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("Stage PHT checkpoint has no model state")
    return checkpoint


def validate_stage_PHT_checkpoint(checkpoint, *, config, dataset_fingerprint,
                                  effective_data):
    _require_architecture(checkpoint)
    saved = checkpoint.get("config") or {}
    current = config.to_dict()
    for section in ("model", "stage_PHT"):
        if saved.get(section) != current[section]:
            raise ValueError(f"Stage PHT {section} configuration mismatch")
    source = checkpoint.get("source_initialization") or {}
    if source != {"kind": "random_binding", "seed": config.runtime.seed, "fold": config.runtime.fold}:
        raise ValueError("Stage PHT random initialization source mismatch")
    if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Stage PHT dataset fingerprint mismatch")
    if dict(checkpoint.get("effective_data") or {}) != dict(effective_data):
        raise ValueError("Stage PHT effective-data mismatch")


def restore_stage_PHT_model(checkpoint, model):
    _require_architecture(checkpoint)
    if checkpoint.get("config", {}).get("model") != model.binding_model.model_config:
        raise ValueError("Stage PHT model configuration mismatch")
    saved = checkpoint.get("encoder_metadata") or {}
    current = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        if entity not in saved:
            raise ValueError(f"Stage PHT missing PLM metadata for {entity}")
        for key in ("checkpoint_fingerprint", "encoder_class", "hidden_size", "max_length", "initialization"):
            if saved[entity].get(key) != current[entity].get(key):
                raise ValueError(f"Stage PHT PLM assets mismatch for {entity}: {key}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_evaluation()


def resume_stage_PHT_last(checkpoint, *, trainer, config, dataset_fingerprint, effective_data):
    if checkpoint.get("artifact_role") != "last":
        raise ValueError("Stage PHT resume requires a last checkpoint")
    validate_stage_PHT_checkpoint(checkpoint, config=config,
                                  dataset_fingerprint=dataset_fingerprint, effective_data=effective_data)
    for key in ("best_checkpoint", "source_warmup"):
        companion = checkpoint.get(key) or {}
        path = Path(companion.get("path", ""))
        if not path.is_file() or checkpoint_sha256(path) != companion.get("sha256"):
            raise ValueError(f"Stage PHT {key} companion is missing or changed")
    restore_stage_PHT_model(checkpoint, trainer.model)
    trainer.warmup_epoch = config.stage_PHT.warmup_epochs
    trainer.begin_training()
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    trainer.scaler.load_state_dict(checkpoint.get("scaler", {}))
    state = checkpoint["training_state"]
    trainer.training_round = int(state["training_round"])
    trainer.global_step = int(state["global_step"])
    trainer.bad_rounds = int(state["bad_rounds"])
    trainer.best_score = float(state["best_score"])
    _restore_rng_state(checkpoint.get("rng_state"))
