#!/usr/bin/env python3
"""Evaluate a Stage14 online checkpoint on the audited PHT validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.stage14_checkpoint import (
    load_stage14_checkpoint,
    validate_stage14_for_evaluation,
)
from plm_unified.stage14_config import apply_stage14_overrides, load_stage14_config
from plm_unified.stage14_model import build_progressive_stage14_model
from plm_unified.stage8_data import (
    OnlineStage8Dataset,
    audit_stage8_class_dirs,
    collate_online_stage8,
)
from plm_unified.stage8_evaluation import evaluate_online_stage8_model
from plm_unified.trainer import resolve_device
from torch.utils.data import DataLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "plm_stage14.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = apply_stage14_overrides(
        load_stage14_config(args.config), device=args.device,
        num_workers=args.num_workers,
    )
    checkpoint = load_stage14_checkpoint(args.checkpoint)
    device = resolve_device(config.runtime.device)
    audit = audit_stage8_class_dirs(
        config.data_dir(), config.stage14.class_dirs,
        config.stage14.train_filename, config.stage14.val_filename, config.model,
        reject_noncanonical_sequences=config.stage14.reject_noncanonical_sequences,
    )
    if audit.has_conflicts or audit.has_errors:
        raise ValueError("Stage14 validation data audit is not clean")
    val_loader = DataLoader(
        OnlineStage8Dataset(audit.val_records),
        batch_size=config.stage14.eval_batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.runtime.num_workers > 0,
        collate_fn=collate_online_stage8,
    )
    # The checkpoint contract carries the true training sample count; validation
    # only needs its own fingerprint and model configuration.
    validate_stage14_for_evaluation(
        checkpoint, config=config, dataset_fingerprint=audit.dataset_fingerprint
    )
    model = build_progressive_stage14_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = evaluate_online_stage8_model(
        model, val_loader, device=device,
        mixed_precision=config.stage14.mixed_precision,
        margin=config.stage14.constraint_margin,
        selection_weights=config.stage14.selection_weights,
    )
    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_type": checkpoint["checkpoint_type"],
        "format_version": checkpoint["format_version"],
        "condition": checkpoint.get("condition"),
        "source_stage1_sha256": checkpoint["source_stage1"]["sha256"],
        "dataset_fingerprint": audit.dataset_fingerprint,
        "joint_score": result.joint_score,
        "selected_pht_threshold": result.selected_pht_threshold,
        "metrics": result.metrics,
    }
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
