#!/usr/bin/env python3
"""Fine-tune a true HLA-peptide-TCR model from Stage 1 or Stage 2."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import (
    create_artifact_timestamp,
    validate_artifact_timestamp,
)
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_online_dataloader, split_path
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed
from plm_unified.triplet_data import (
    build_pht_stage3_loaders,
    build_strict_triplet_datasets,
    make_triplet_dataloader,
)
from plm_unified.triplet_model import (
    build_progressive_triplet_model,
    initialise_from_stage1,
    initialise_from_stage2,
)
from plm_unified.triplet_trainer import (
    Stage3Trainer,
    checkpoint_initialization_source,
    load_stage3_checkpoint,
)


def log_status(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "plm_triplet.yaml"
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Completed Stage-1 full last checkpoint used for a new run.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        help="Stage-2 full joint checkpoint used to initialise a new run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a full stage3_last_<timestamp>.pt checkpoint.",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument(
        "--run-timestamp",
        help="Optional shared artifact suffix in YYYYMMDD_HHMMSS_ffffff format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap returned rows per task and split after auditing.",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        help="PHT smoke only: retain at most this many rows per selected PHT type.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit configured data and exit before loading protein language models.",
    )
    return parser.parse_args()


def make_loaders(
    config,
    *,
    limit: int | None,
    limit_per_type: int | None = None,
):
    values = config.triplet
    if values.data_mode == "pht_multitask":
        if limit is not None:
            raise ValueError(
                "Use --limit-per-type, not --limit, with pht_multitask"
            )
        return build_pht_stage3_loaders(
            config,
            limit_per_type=limit_per_type,
        )
    if limit_per_type is not None:
        raise ValueError(
            "--limit-per-type is only valid with triplet.data_mode=pht_multitask"
        )
    bundle = build_strict_triplet_datasets(
        config.paths.data_root,
        data_subdir=values.data_subdir,
        merged_filename=values.merged_filename,
        train_filename=values.train_filename,
        val_filename=values.val_filename,
        test_filename=values.test_filename,
        verify_manifest=values.verify_merged_manifest,
        strict_split_cleanup=values.strict_split_cleanup,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        hla_max_length=config.model.hla_max_length,
        tcr_max_length=config.model.tcr_max_length,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
        limit=limit,
    )
    common = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    train_loaders = {
        task: make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "train",
                config.training.fold,
            ),
            task,
            batch_size=values.micro_batch_size,
            shuffle=True,
            seed=config.training.seed,
            **common,
        )
        for task in ("phla", "ptcr")
    }
    val_loaders = {
        task: make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "val",
                config.training.fold,
            ),
            task,
            batch_size=values.eval_batch_size,
            shuffle=False,
            seed=config.training.seed,
            **common,
        )
        for task in ("phla", "ptcr")
    }
    train_loaders["triplet"] = make_triplet_dataloader(
        bundle.train,
        batch_size=values.micro_batch_size,
        shuffle=True,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loaders["triplet"] = make_triplet_dataloader(
        bundle.val,
        batch_size=values.eval_batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    audit = {
        "manifest": bundle.manifest_audit,
        "strict_split_cleanup": values.strict_split_cleanup,
        "triplet_splits": {
            name: dataset.filter_stats
            for name, dataset in bundle.as_dict().items()
        },
        "binary_splits": {
            f"{split}/{task}": dict(loader.dataset.filter_stats)
            for split, loaders in (
                ("train", train_loaders),
                ("val", val_loaders),
            )
            for task, loader in loaders.items()
            if task in {"phla", "ptcr"}
        },
    }
    return train_loaders, val_loaders, audit


def main() -> None:
    args = parse_args()
    initialization_paths = [
        path
        for path in (args.stage1_checkpoint, args.stage2_checkpoint)
        if path is not None
    ]
    if args.resume and initialization_paths:
        raise ValueError(
            "Use --resume or one initialization checkpoint, not both"
        )
    if len(initialization_paths) > 1:
        raise ValueError(
            "Use either --stage1-checkpoint or --stage2-checkpoint, not both"
        )
    if not args.audit_only and not args.resume and not initialization_paths:
        raise ValueError(
            "A new run requires --stage1-checkpoint or --stage2-checkpoint; "
            "resuming requires --resume"
        )

    resume_metadata = None
    if args.resume:
        resume_metadata = load_stage3_checkpoint(
            args.resume,
            map_location="cpu",
        )
    resume_timestamp = (
        resume_metadata.get("run_timestamp") if resume_metadata else None
    )
    resume_id = (
        resume_metadata.get("swanlab_run_id") if resume_metadata else None
    )
    initialization_source = (
        checkpoint_initialization_source(resume_metadata)
        if resume_metadata
        else {}
    )
    del resume_metadata
    gc.collect()
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp
        or resume_timestamp
        or create_artifact_timestamp()
    )
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=not args.audit_only)
    set_global_seed(config.training.seed)
    log_status(
        f"[RUN START] fold={config.training.fold} "
        f"seed={config.training.seed} device={config.training.device} "
        f"timestamp={run_timestamp} limit={args.limit} "
        f"limit_per_type={args.limit_per_type}"
    )

    started_at = time.perf_counter()
    log_status("[DATA LOAD START] auditing and loading all three tasks")
    train_loaders, val_loaders, data_audit = make_loaders(
        config,
        limit=args.limit,
        limit_per_type=args.limit_per_type,
    )
    log_status(
        "[DATA LOAD END] "
        + ", ".join(
            f"{split}/{task}={len(loader.dataset):,}"
            for split, loaders in (
                ("train", train_loaders),
                ("val", val_loaders),
            )
            for task, loader in loaders.items()
        )
        + f" elapsed={time.perf_counter() - started_at:.1f}s"
    )
    if args.audit_only:
        print(json.dumps(data_audit, indent=2, sort_keys=True))
        return

    device = resolve_device(config.training.device)

    started_at = time.perf_counter()
    log_status("[MODEL LOAD START] loading two ESM-C encoders and TCR-BERT")
    model = build_progressive_triplet_model(config, device)
    log_status(
        "[MODEL LOAD END] "
        f"elapsed={time.perf_counter() - started_at:.1f}s"
    )
    if args.stage1_checkpoint:
        started_at = time.perf_counter()
        log_status(
            f"[STAGE1 LOAD START] checkpoint={args.stage1_checkpoint}"
        )
        initialization_source = initialise_from_stage1(
            model,
            args.stage1_checkpoint,
            config,
        )
        log_status(
            "[STAGE1 LOAD END] "
            f"stage={initialization_source.get('stage')} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )
    elif args.stage2_checkpoint:
        started_at = time.perf_counter()
        log_status(
            f"[STAGE2 LOAD START] checkpoint={args.stage2_checkpoint}"
        )
        initialization_source = initialise_from_stage2(
            model,
            args.stage2_checkpoint,
            config,
        )
        log_status(
            "[STAGE2 LOAD END] "
            f"stage={initialization_source.get('stage')} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )

    with SwanLabTracker(
        config,
        resume_id=resume_id,
        run_timestamp=run_timestamp,
    ) as tracker:
        input_log = {
                "input/mode": "online_stage3",
                "stage3/initialization_kind": initialization_source.get(
                    "kind", ""
                ),
                "stage3/source_checkpoint": initialization_source.get(
                    "path", ""
                ),
                "stage3/source_fold": initialization_source.get(
                    "source_fold", -1
                ),
                "stage3/source_seed": initialization_source.get(
                    "source_seed", -1
                ),
                "stage3/micro_batch_size": config.triplet.micro_batch_size,
                "stage3/eval_batch_size": config.triplet.eval_batch_size,
                "stage3/gradient_accumulation_steps": (
                    config.triplet.gradient_accumulation_steps
                ),
                "stage3/binary_replay_ratio": (
                    config.triplet.binary_replay_ratio
                ),
        }
        if config.triplet.data_mode == "pht_multitask":
            input_log.update(
                {
                    "input/mode": "online_stage3_pht_multitask",
                    "data/pht_label_types": ",".join(
                        data_audit["pht_label_types"]
                    ),
                    "data/dataset_fingerprint": data_audit[
                        "dataset_fingerprint"
                    ],
                    "data/limit_per_type": data_audit["limit_per_type"] or 0,
                    **{
                        f"data/train/{key}": value
                        for key, value in data_audit["train"].items()
                    },
                    **{
                        f"data/{split}/{key}": value
                        for split, stats in data_audit["validation"].items()
                        for key, value in stats.items()
                    },
                }
            )
        else:
            input_log.update(
                {
                    "data/manifest_matches": int(
                        bool(data_audit["manifest"].get("matches"))
                    ),
                    **{
                    f"data/triplet/{split}/{key}": value
                    for split, stats in data_audit[
                        "triplet_splits"
                    ].items()
                    for key, value in stats.items()
                },
                **{
                    f"data/{split_task}/{key}": value
                    for split_task, stats in data_audit[
                        "binary_splits"
                    ].items()
                    for key, value in stats.items()
                },
                }
            )
        tracker.log(input_log, step=0)
        trainer = Stage3Trainer(
            model,
            config,
            train_loaders,
            val_loaders,
            tracker,
            initialization_source=initialization_source,
            triplet_data_audit=data_audit,
            run_timestamp=run_timestamp,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/stage3b_round": summary["stage3b_round"],
                "training/best_joint_score": summary["best_joint_score"],
                "training/best_round": summary["best_round"],
                "training/best_round_score": summary["best_joint_score"],
                "training/selected_threshold": summary[
                    "selected_threshold"
                ],
            },
            step=trainer.global_step,
        )
    log_status(
        f"[RUN END] phase={summary['phase']} "
        f"global_step={summary['global_step']}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
