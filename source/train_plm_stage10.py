#!/usr/bin/env python3
"""Train one Stage-10 Teacher/Student retention condition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage8_data import (
    audit_stage8_data,
    make_stage8_cached_loaders,
    open_stage8_caches,
    stage8_cache_hidden_dtype,
)
from plm_unified.stage10_checkpoint import (
    load_stage3_teacher_source,
    load_torch_checkpoint,
)
from plm_unified.stage10_config import (
    Stage10Condition,
    apply_stage10_overrides,
    load_stage10_config,
)
from plm_unified.stage10_model import (
    build_stage10_teacher_student_models,
)
from plm_unified.stage10_trainer import Stage10Trainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


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
    parser.add_argument(
        "--condition",
        choices=[value.value for value in Stage10Condition],
        required=True,
    )
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--limit-per-type",
        type=int,
        help="Debug only; applied after the complete strict audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resume_checkpoint = (
        load_torch_checkpoint(args.resume, mmap=False)
        if args.resume
        else None
    )
    config = apply_stage10_overrides(
        load_stage10_config(args.config),
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
        stage3_checkpoint=args.stage3_checkpoint,
    )
    config.validate(require_files=True)
    condition = Stage10Condition(args.condition)
    if (
        resume_checkpoint
        and resume_checkpoint.get("condition") != condition.value
    ):
        raise ValueError("--condition does not match --resume checkpoint")
    run_timestamp = args.run_timestamp or (
        resume_checkpoint.get("run_timestamp")
        if resume_checkpoint
        else None
    ) or create_artifact_timestamp()
    set_global_seed(config.runtime.seed)

    audit = audit_stage8_data(
        config.train_path(),
        config.val_path(),
        config.model,
        reject_noncanonical_sequences=(
            config.stage10.reject_noncanonical_sequences
        ),
    )
    audit_dir = config.run_dir(condition) / f"audit_{run_timestamp}"
    audit.write_reports(audit_dir)
    if audit.has_conflicts or audit.has_errors:
        print(
            "Stage-10 data audit failed; manual repair required. "
            f"Reports: {audit_dir}"
        )
        raise SystemExit(2)

    source = load_stage3_teacher_source(config)
    device = resolve_device(config.runtime.device)
    caches = open_stage8_caches(
        config.cache_dir(source.sha256),
        source_sha256=source.sha256,
        model=config.model,
        dataset_fingerprint=audit.dataset_fingerprint,
    )
    train_loader, val_loader = make_stage8_cached_loaders(
        audit,
        caches,
        batch_size=config.stage10.batch_size,
        eval_batch_size=config.stage10.eval_batch_size,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        limit_per_type=args.limit_per_type,
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
    data_audit = audit.to_dict()
    data_audit["effective_data"] = {
        "limit_per_type": args.limit_per_type,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
    }
    with SwanLabTracker(
        config,
        resume_id=(
            resume_checkpoint.get("swanlab_run_id")
            if resume_checkpoint
            else None
        ),
        run_timestamp=run_timestamp,
    ) as tracker:
        tracker.log(
            {
                "stage10/condition": condition.value,
                "stage10/source_stage3_sha256": source.sha256,
                "stage10/teacher_frozen": 1,
                "stage10/student_initialization": "stage3",
                "stage10/keep_loss_weight": (
                    config.stage10.keep_loss_weight
                ),
                "stage10/phla_keep_weight": (
                    config.stage10.phla_keep_weight
                ),
                "stage10/ptcr_keep_weight": (
                    config.stage10.ptcr_keep_weight
                ),
                "stage10/distillation_temperature": (
                    config.stage10.distillation_temperature
                ),
                "stage10/legacy_task_degradation_tolerance": (
                    config.stage10.legacy_task_degradation_tolerance
                ),
                "stage10/train_samples": len(train_loader.dataset),
                "stage10/validation_samples": len(val_loader.dataset),
            }
        )
        trainer = Stage10Trainer(
            models,
            config,
            train_loader,
            val_loader,
            condition=condition,
            source=source,
            data_audit=data_audit,
            tracker=tracker,
            run_timestamp=run_timestamp,
        )
        # Encoders and the original binding copy are no longer needed here.
        source.checkpoint = {}
        source.binding_state = {}
        summary = trainer.fit(resume=args.resume)
    print(
        f"Stage-10 {condition.value} complete: "
        f"{summary['best_checkpoint']}"
    )
    print(f"Primary success: {summary['primary_success']}")


if __name__ == "__main__":
    main()
