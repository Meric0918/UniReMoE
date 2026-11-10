#!/usr/bin/env python3
"""Fine-tune a Stage-3 model with paired PH--TCR contrastive learning."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, validate_artifact_timestamp
from plm_unified.stage9_checkpoint import (
    initialise_from_stage3,
    load_stage3_source,
    load_torch_checkpoint,
)
from plm_unified.stage9_config import (
    apply_stage9_overrides,
    load_stage9_config,
    restore_stage9_checkpoint_config,
)
from plm_unified.stage9_model import build_stage9_model
from plm_unified.stage9_pipeline import build_stage9_audit, build_stage9_loaders
from plm_unified.stage9_trainer import Stage9Trainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage9.yaml",
    )
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap every split after the complete data audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.stage3_checkpoint:
        raise ValueError("Use --resume or --stage3-checkpoint, not both")
    resume_checkpoint = (
        load_torch_checkpoint(args.resume, mmap=False) if args.resume else None
    )
    resume_source_path = (
        resume_checkpoint.get("source_stage3", {}).get("path")
        if resume_checkpoint
        else None
    )
    resume_config = (resume_checkpoint or {}).get("config", {})
    resume_runtime = resume_config.get("runtime", {})
    resume_stage9 = resume_config.get("stage9", {})
    resume_paths = resume_config.get("paths", {})
    resume_pair_path = None
    if resume_checkpoint and resume_paths.get("data_root"):
        resume_pair_path = (
            Path(resume_paths["data_root"])
            / resume_stage9.get("data_subdir", "data_stage9")
            / resume_stage9.get("pair_filename", "pht_pairs.csv")
        )
    config = load_stage9_config(args.config)
    if resume_checkpoint:
        restore_stage9_checkpoint_config(config, resume_config)
    config = apply_stage9_overrides(
        config,
        fold=args.fold if args.fold is not None else resume_runtime.get("fold"),
        seed=args.seed if args.seed is not None else resume_runtime.get("seed"),
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
        stage3_checkpoint=(
            args.stage3_checkpoint or resume_source_path
        ),
        pair_csv=args.pairs_csv or resume_pair_path,
    )
    config.validate(require_files=True)
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp
        or (resume_checkpoint or {}).get("run_timestamp")
        or create_artifact_timestamp()
    )
    set_global_seed(config.runtime.seed)
    audit = build_stage9_audit(config).require_clean()
    pair_loaders, legacy_train, legacy_val, audit_payload = build_stage9_loaders(
        config, audit, limit=args.limit
    )
    audit_path = config.run_dir() / f"data_audit_{run_timestamp}.json"
    audit.write_report(audit_path)
    source = load_stage3_source(config)
    device = resolve_device(config.runtime.device)
    model = build_stage9_model(config, device)
    initialise_from_stage3(model, source)
    source.checkpoint = {}
    gc.collect()
    resume_id = (resume_checkpoint or {}).get("swanlab_run_id")
    del resume_checkpoint
    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=run_timestamp
    ) as tracker:
        tracker.log(
            {
                "stage9/source_stage3_sha256": source.sha256,
                "stage9/pair_dataset_sha256": audit.dataset_fingerprint,
                "stage9/pair_train_rows": len(pair_loaders["train"].dataset),
                "stage9/pair_val_rows": len(pair_loaders["val"].dataset),
                "stage9/pair_test_rows": len(pair_loaders["test"].dataset),
                "stage9/temperature": config.stage9.temperature,
                "stage9/replay_ratio": config.stage9.replay_ratio,
            },
            step=0,
        )
        trainer = Stage9Trainer(
            model,
            config,
            pair_loaders["train"],
            pair_loaders["val"],
            legacy_train,
            legacy_val,
            source_stage3=source.lineage,
            data_audit=audit_payload,
            tracker=tracker,
            run_timestamp=run_timestamp,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/stage9b_round": summary["stage9b_round"],
                "training/selected_pair_threshold": summary[
                    "selected_pair_threshold"
                ],
            },
            step=trainer.global_step,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
