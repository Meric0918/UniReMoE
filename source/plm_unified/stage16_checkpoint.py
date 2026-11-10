"""Atomic artifacts, explicit provenance and complete resumable state."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .stage8_checkpoint import (
    atomic_torch_save,
    checkpoint_sha256,
    load_torch_checkpoint,
    load_stage8_v2_checkpoint,
)
from .stage16_config import config_from_dict


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n"
        )


def write_jsonl(path, values):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(
                    json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_training_state(model, loaders=()):
    rng = rng_state()
    modules = [(m, m.training) for m in model.modules()]
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    attrs = [
        (m, k, getattr(m, k))
        for m in model.modules()
        for k in ("_active_training", "_active_stage")
        if hasattr(m, k)
    ]
    generators = [
        (loader.generator, loader.generator.get_state())
        for loader in loaders
        if getattr(loader, "generator", None) is not None
    ]
    samplers = [
        (loader.batch_sampler, loader.batch_sampler.epoch)
        for loader in loaders
        if hasattr(loader.batch_sampler, "epoch")
    ]
    try:
        yield
    finally:
        for parameter, required in parameters:
            parameter.requires_grad_(required)
        for module, key, value in attrs:
            setattr(module, key, value)
        for module, training in modules:
            module.training = training
        for generator, state in generators:
            generator.set_state(state)
        for sampler, epoch in samplers:
            sampler.epoch = epoch
        restore_rng(rng)


def inference_payload(trainer, metrics, *, role, selectors=()):
    return {
        "format_version": 1,
        "checkpoint_type": f"stage{trainer.config.stage}_inference_model",
        "artifact_role": role,
        "selectors": list(selectors),
        "config": trainer.config.to_dict(),
        "contract": trainer.config.contract(),
        "manifest_sha256": digest(trainer.manifest),
        "manifest": trainer.manifest,
        "reference": trainer.reference,
        "round": trainer.completed_round,
        "global_step": trainer.global_step,
        "metrics": metrics,
        "debug": trainer.config.debug,
        "model": trainer.model.state_dict(),
        "swanlab_run_id": getattr(trainer, "swanlab_run_id", None),
    }


def save_training_checkpoint(trainer, path):
    payload = inference_payload(trainer, trainer.last_metrics, role="last")
    payload.update(
        {
            "checkpoint_type": f"stage{trainer.config.stage}_training_model",
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "rng": rng_state(),
            "trainer_state": trainer.state_dict(),
            "selector_states": trainer.selectors.state_dict(),
            "teacher_identity": trainer.teacher_identity,
            "swanlab_run_id": getattr(trainer, "swanlab_run_id", None),
        }
    )
    return atomic_torch_save(payload, path)


def restore_training_checkpoint(trainer, path):
    checkpoint = load_torch_checkpoint(path)
    if (
        checkpoint.get("checkpoint_type")
        != f"stage{trainer.config.stage}_training_model"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Only the matching stage training checkpoint can resume")
    if checkpoint["contract"] != trainer.config.contract():
        raise ValueError("Resume configuration/training contract changed")
    if checkpoint["manifest_sha256"] != digest(trainer.manifest):
        raise ValueError("Resume data/provenance manifest changed")
    if checkpoint.get("teacher_identity") != trainer.teacher_identity:
        raise ValueError("Resume teacher cache identity changed")
    if checkpoint.get("swanlab_run_id") and getattr(trainer, "swanlab_run_id", None):
        if checkpoint["swanlab_run_id"] != trainer.swanlab_run_id:
            raise ValueError("Resume SwanLab run identity changed")
    for name, state in checkpoint["selector_states"].items():
        if state["artifact"] is not None:
            companion = Path(path).parent / state["artifact"]
            if (
                not companion.is_file()
                or checkpoint_sha256(companion) != state["sha256"]
            ):
                raise ValueError(f"Invalid {name} best companion")
    trainer.model.load_state_dict(checkpoint["model"], strict=True)
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    trainer.reference = checkpoint["reference"]
    trainer.last_metrics = checkpoint["metrics"]
    trainer.load_state_dict(checkpoint["trainer_state"])
    trainer.selectors.load_state_dict(checkpoint["selector_states"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def load_inference_checkpoint(path, *, expected_stage=None):
    checkpoint = load_torch_checkpoint(path)
    config = config_from_dict(checkpoint["config"])
    if expected_stage is not None and config.stage != expected_stage:
        raise ValueError("Wrong stage checkpoint")
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("checkpoint_type") != f"stage{config.stage}_inference_model"
    ):
        raise ValueError(
            "Expected an inference checkpoint, not last or a legacy Stage8 artifact"
        )
    if checkpoint.get("contract") != config.contract():
        raise ValueError("Corrupt embedded configuration contract")
    if checkpoint.get("manifest_sha256") != digest(checkpoint["manifest"]):
        raise ValueError("Corrupt data manifest")
    return checkpoint, config


def load_common_warmup(path):
    checkpoint = load_torch_checkpoint(path)
    if checkpoint.get("checkpoint_type") == "stage16_18_common_warmup":
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("artifact_role") != "warmup"
        ):
            raise ValueError("Invalid common warmup artifact")
        if not all(
            key in checkpoint
            for key in (
                "model",
                "config",
                "source_stage1",
                "dataset_fingerprint",
                "effective_data",
                "training_state",
            )
        ):
            raise ValueError("Incomplete common warmup artifact")
        return checkpoint
    return load_stage8_v2_checkpoint(path, artifact_role="warmup")


def verify_original_encoders(model, warmup):
    """Check actual warmup weights, not merely declared encoder fingerprints."""
    for name, value in model.state_dict().items():
        if "_encoder." not in name:
            continue
        if not isinstance(value, torch.Tensor):
            if name.endswith("._extra_state"):
                continue  # Transformer Engine's process-local BytesIO metadata.
            raise ValueError(f"Unexpected non-tensor encoder state: {name}")
        expected = warmup["model"].get(name)
        if not isinstance(expected, torch.Tensor) or not torch.equal(
            value.cpu().to(expected.dtype), expected
        ):
            raise ValueError(
                f"Warmup encoder is not the original pretrained encoder: {name}"
            )
