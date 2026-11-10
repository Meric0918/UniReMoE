#!/usr/bin/env python3
"""Evaluate a Stage PHT checkpoint on its audited PHT validation set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.stage8_data import audit_stage8_class_dirs
from plm_unified.stage_PHT_checkpoint import (
    evaluation_payload,
    load_stage_PHT_checkpoint,
    restore_stage_PHT_model,
    validate_stage_PHT_checkpoint,
)
from plm_unified.stage_PHT_config import (
    apply_stage_PHT_overrides, load_stage_PHT_config,
)
from plm_unified.stage_adapter_data import make_stage_adapter_loaders as make_stage_PHT_loaders
from plm_unified.stage_adapter_evaluation import evaluate_stage_adapter as evaluate_stage_PHT
from plm_unified.stage_PHT_model import build_stage_PHT_model
from plm_unified.trainer import resolve_device


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/plm_stage_PHT.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = apply_stage_PHT_overrides(
        load_stage_PHT_config(args.config),
        device=args.device, num_workers=args.num_workers,
    )
    checkpoint = load_stage_PHT_checkpoint(args.checkpoint)
    limit_per_type = (checkpoint.get("effective_data") or {}).get(
        "limit_per_type"
    )
    values = config.stage_PHT
    audit = audit_stage8_class_dirs(
        config.data_dir(), values.class_dirs,
        values.train_filename, values.val_filename, config.model,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
    )
    audit.require_clean()
    train_loader, val_loader = make_stage_PHT_loaders(
        audit, micro_batch_size=values.micro_batch_size,
        eval_batch_size=values.eval_batch_size, seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        limit_per_type=limit_per_type,
    )
    effective = {
        "limit_per_type": limit_per_type,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
    }
    validate_stage_PHT_checkpoint(
        checkpoint, config=config, dataset_fingerprint=audit.dataset_fingerprint,
        effective_data=effective,
    )
    device = resolve_device(config.runtime.device)
    model = build_stage_PHT_model(config, device)
    restore_stage_PHT_model(checkpoint, model)
    evaluation = evaluate_stage_PHT(
        model, val_loader, device=device, mixed_precision=values.mixed_precision
    )
    payload = {
        "architecture": "stage_PHT",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_fingerprint": audit.dataset_fingerprint,
        **evaluation_payload(evaluation),
    }
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
