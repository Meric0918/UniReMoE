"""Versioned Stage11 checkpoints and strict round-boundary resume contracts."""

from dataclasses import asdict
from pathlib import Path

import torch

from .stage8_checkpoint import atomic_torch_save
from .stage11_model import ARCHITECTURE


def training_contract(config, data_fingerprint, cache_metadata):
    training = asdict(config.stage11)
    training["class_dirs"] = list(training["class_dirs"])
    return {"model": asdict(config.model), "stage11": training,
            "fold": config.runtime.fold, "seed": config.runtime.seed,
            "data_fingerprint": data_fingerprint, "cache_metadata": cache_metadata,
            "initialization": "pretrained-frozen-PLM-random-downstream",
            "loss_version": "multi-positive-direction-mean-v1",
            "schedule": "ph-contrastive,pht-contrastive,pht-classification"}


def load_stage11_checkpoint(path, *, role=None):
    # Only load trusted local training artifacts: optimizer/RNG states require pickle.
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (value.get("architecture") != ARCHITECTURE or value.get("format_version") != 1
            or value.get("checkpoint_type") != "stage11_model"):
        raise ValueError("Expected a Stage11 v1 checkpoint")
    if role is not None and value.get("artifact_role") != role:
        raise ValueError(f"Expected a Stage11 {role} checkpoint")
    for name in ("model", "config", "training_contract", "cache_metadata", "run_timestamp", "round"):
        if name not in value:
            raise ValueError(f"Stage11 checkpoint missing {name}")
    return value
