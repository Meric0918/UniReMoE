#!/usr/bin/env python3
"""Run paired Teacher/Student validation for a Stage-10 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from plm_unified.stage8_data import (
    audit_stage8_data,
    make_stage8_cached_loaders,
    open_stage8_caches,
    stage8_cache_hidden_dtype,
)
from plm_unified.stage10_checkpoint import (
    extract_stage10_binding_state,
    load_stage3_teacher_source,
    load_torch_checkpoint,
    validate_stage10_full_checkpoint,
    validate_stage10_lean_checkpoint,
)
from plm_unified.stage10_config import (
    apply_stage10_overrides,
    load_stage10_config,
)
from plm_unified.stage10_evaluation import evaluate_stage10_models
from plm_unified.stage10_model import (
    build_stage10_teacher_student_models,
)
from plm_unified.trainer import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/plm_stage10.yaml"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_stage10_overrides(
        load_stage10_config(args.config),
        device=args.device,
        num_workers=args.num_workers,
        stage3_checkpoint=args.stage3_checkpoint,
    )
    config.validate(require_files=True)
    audit = audit_stage8_data(
        config.train_path(),
        config.val_path(),
        config.model,
        reject_noncanonical_sequences=(
            config.stage10.reject_noncanonical_sequences
        ),
    )
    if audit.has_conflicts or audit.has_errors:
        raise SystemExit(
            "Stage-10 data audit failed; manual repair is required"
        )
    source = load_stage3_teacher_source(config)
    checkpoint = load_torch_checkpoint(args.checkpoint)
    if checkpoint.get("checkpoint_type") == "stage10_multitask_model":
        validate_stage10_full_checkpoint(
            checkpoint,
            config,
            source.sha256,
        )
    else:
        validate_stage10_lean_checkpoint(
            checkpoint,
            config,
            source.sha256,
        )

    caches = open_stage8_caches(
        config.cache_dir(source.sha256),
        source_sha256=source.sha256,
        model=config.model,
        dataset_fingerprint=audit.dataset_fingerprint,
    )
    device = resolve_device(config.runtime.device)
    _, val_loader = make_stage8_cached_loaders(
        audit,
        caches,
        batch_size=config.stage10.batch_size,
        eval_batch_size=config.stage10.eval_batch_size,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        hidden_dtype=stage8_cache_hidden_dtype(
            device,
            config.stage10.mixed_precision,
        ),
    )
    models = build_stage10_teacher_student_models(
        config.model,
        source.binding_state,
        device=device,
    )
    models.student.load_state_dict(
        extract_stage10_binding_state(checkpoint),
        strict=True,
    )
    source.checkpoint = {}
    source.binding_state = {}
    result = evaluate_stage10_models(
        models.teacher,
        models.student,
        val_loader,
        device=device,
        mixed_precision=config.stage10.mixed_precision,
        margin=config.stage10.constraint_margin,
        selection_weights=config.stage10.selection_weights,
        degradation_tolerance=(
            config.stage10.legacy_task_degradation_tolerance
        ),
    )
    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "condition": checkpoint.get(
            "condition",
            checkpoint.get("stage10_condition"),
        ),
        "source_stage3": source.lineage,
        "dataset_fingerprint": audit.dataset_fingerprint,
        "student_metrics": result.student_metrics,
        "teacher_metrics": result.teacher_metrics,
        "retention": result.retention,
        "retention_eligible": result.retention_eligible,
        "legacy_task_degradation_tolerance": (
            config.stage10.legacy_task_degradation_tolerance
        ),
        "joint_score": result.joint_score,
        "selected_pht_threshold": result.selected_pht_threshold,
    }
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
