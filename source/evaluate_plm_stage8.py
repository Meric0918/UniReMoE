#!/usr/bin/env python3
"""Evaluate a full online Stage-8 V2 checkpoint on the strict validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.data import split_path
from plm_unified.stage8_checkpoint import (
    load_stage8_v2_checkpoint,
    validate_stage8_v2_for_evaluation,
)
from plm_unified.stage8_config import apply_stage8_overrides, load_stage8_config
from plm_unified.stage8_data import audit_stage8_class_dirs, make_stage8_online_loaders
from plm_unified.stage8_evaluation import evaluate_online_stage8_model
from plm_unified.stage8_model import build_progressive_stage8_model
from plm_unified.trainer import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage8.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = apply_stage8_overrides(
        load_stage8_config(args.config),
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate()
    audit = audit_stage8_class_dirs(
        config.data_dir(), config.stage8.class_dirs,
        config.stage8.train_filename, config.stage8.val_filename, config.model,
        reject_noncanonical_sequences=config.stage8.reject_noncanonical_sequences,
    )
    audit.require_clean()
    checkpoint = load_stage8_v2_checkpoint(args.checkpoint)
    validate_stage8_v2_for_evaluation(
        checkpoint, config=config, dataset_fingerprint=audit.dataset_fingerprint
    )
    _, val_loader = make_stage8_online_loaders(
        audit,
        phla_path=split_path(config.paths.data_root, "phla", "train", config.runtime.fold),
        ptcr_path=split_path(config.paths.data_root, "ptcr", "train", config.runtime.fold),
        model=config.model,
        micro_batch_size=config.stage8.micro_batch_size,
        eval_batch_size=config.stage8.eval_batch_size,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
    )
    device = resolve_device(config.runtime.device)
    model = build_progressive_stage8_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint.pop("model")
    result = evaluate_online_stage8_model(
        model, val_loader, device=device,
        mixed_precision=config.stage8.mixed_precision,
        margin=config.stage8.constraint_margin,
        selection_weights=config.stage8.selection_weights,
    )
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_format_version": checkpoint["format_version"],
        "source_stage1_sha256": checkpoint["source_stage1"]["sha256"],
        "source_stage8a_sha256": (
            checkpoint.get("source_stage8a") or {}
        ).get("sha256"),
        "condition": checkpoint.get("condition"),
        "dataset_fingerprint": audit.dataset_fingerprint,
        "joint_score": result.joint_score,
        "selected_pht_threshold": result.selected_pht_threshold,
        "metrics": result.metrics,
        "inference": "stage8_v2_full_online_model",
    }
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
