"""Stage-3 source validation and Stage-9 checkpoint primitives."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .stage9_config import Stage9ExperimentConfig
from .stage9_model import ProgressiveStage9FineTuningModel


@dataclass
class Stage3Source:
    path: Path
    sha256: str
    checkpoint_type: str
    checkpoint: dict[str, Any]
    encoder_fingerprints: dict[str, str | None]
    lineage: dict[str, Any]


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_checkpoint(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_stage3_source(config: Stage9ExperimentConfig) -> Stage3Source:
    path = config.paths.stage3_checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 checkpoint not found: {path}")
    checkpoint = load_torch_checkpoint(path)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Stage 9 requires Stage-3 checkpoint format_version=1")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != "stage3_multitask_model":
        raise ValueError("Stage 9 requires a stage3_multitask_model checkpoint")
    if set(checkpoint.get("tasks", ())) != {"phla", "ptcr", "triplet"}:
        raise ValueError("Stage-3 checkpoint must contain phla, ptcr, and triplet tasks")
    saved_config = checkpoint.get("config", {})
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Stage-3 model configuration does not match Stage 9")
    saved_training = saved_config.get("training", {})
    for key, expected in (
        ("fold", config.runtime.fold),
        ("seed", config.runtime.seed),
    ):
        if saved_training.get(key) != expected:
            raise ValueError(
                f"Stage-3 {key}={saved_training.get(key)!r} does not match {expected}"
            )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-3 checkpoint has no model state")
    if not any(name.startswith("binding_model.") for name in state):
        raise ValueError("Stage-3 checkpoint has no binding_model state")
    for entity in ("peptide", "hla", "tcr"):
        if not any(name.startswith(f"{entity}_encoder.") for name in state):
            raise ValueError(f"Stage-3 checkpoint has no {entity} encoder state")
    metadata = checkpoint.get("encoder_metadata", {})
    fingerprints = {
        entity: metadata.get(entity, {}).get("checkpoint_fingerprint")
        for entity in ("peptide", "hla", "tcr")
    }
    missing_fingerprints = [
        entity for entity, fingerprint in fingerprints.items() if not fingerprint
    ]
    if missing_fingerprints:
        raise ValueError(
            "Stage-3 checkpoint is missing base-PLM fingerprint metadata for: "
            + ", ".join(missing_fingerprints)
        )
    sha256 = checkpoint_sha256(path)
    lineage = {
        "kind": "stage3",
        "path": str(path),
        "sha256": sha256,
        "checkpoint_type": checkpoint_type,
        "source_fold": config.runtime.fold,
        "source_seed": config.runtime.seed,
        "encoder_fingerprints": fingerprints,
    }
    return Stage3Source(
        path=path,
        sha256=sha256,
        checkpoint_type=checkpoint_type,
        checkpoint=checkpoint,
        encoder_fingerprints=fingerprints,
        lineage=lineage,
    )


def initialise_from_stage3(
    model: ProgressiveStage9FineTuningModel,
    source: Stage3Source,
) -> dict[str, Any]:
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved = source.encoder_fingerprints.get(entity)
        current = current_metadata[entity].get("checkpoint_fingerprint")
        if saved and saved != current:
            raise ValueError(
                f"Stage-3 base-PLM fingerprint mismatch for {entity}: "
                f"saved={saved}, current={current}"
            )
    state = source.checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-3 checkpoint has no model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("._extra_state")]
    unexpected = [name for name in unexpected if not name.endswith("._extra_state")]
    allowed_missing = (
        "binding_model.contrastive_ph_pooling.",
        "binding_model.contrastive_tcr_pooling.",
        "binding_model.ph_projector.",
        "binding_model.tcr_projector.",
    )
    disallowed_missing = [
        name for name in missing if not name.startswith(allowed_missing)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Stage-3 to Stage-9 state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    return dict(source.lineage)


def load_stage9_deploy_checkpoint(
    model: ProgressiveStage9FineTuningModel,
    path_or_checkpoint: str | Path | Mapping[str, Any],
    config: Stage9ExperimentConfig,
) -> dict[str, Any]:
    checkpoint = (
        path_or_checkpoint
        if isinstance(path_or_checkpoint, dict)
        else load_torch_checkpoint(path_or_checkpoint, mmap=False)
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Stage-9 checkpoint format")
    if checkpoint.get("checkpoint_type") != "stage9_multitask_model":
        raise ValueError("Expected a stage9_multitask_model checkpoint")
    if set(checkpoint.get("tasks", ())) != {"phla", "ptcr", "triplet", "pair"}:
        raise ValueError("Stage-9 checkpoint has an incomplete task set")
    saved_config = checkpoint.get("config", {})
    if saved_config.get("model") != asdict(config.model):
        raise ValueError("Stage-9 model configuration mismatch")
    saved_runtime = saved_config.get("runtime", {})
    for key, expected in (
        ("fold", config.runtime.fold),
        ("seed", config.runtime.seed),
    ):
        if saved_runtime.get(key) != expected:
            raise ValueError(f"Stage-9 checkpoint {key} mismatch")
    current_metadata = model.encoder_metadata()
    saved_metadata = checkpoint.get("encoder_metadata", {})
    for entity in ("peptide", "hla", "tcr"):
        saved = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        current = current_metadata[entity].get("checkpoint_fingerprint")
        if saved and saved != current:
            raise ValueError(f"Stage-9 base-PLM fingerprint mismatch for {entity}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-9 checkpoint has no model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("._extra_state")]
    unexpected = [name for name in unexpected if not name.endswith("._extra_state")]
    if missing or unexpected:
        raise RuntimeError(
            f"Stage-9 state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return checkpoint


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination
