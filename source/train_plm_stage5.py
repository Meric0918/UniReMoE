#!/usr/bin/env python3
"""Train Stage 5 from random downstream parameters with frozen base PLMs."""

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
from plm_unified.stage5_model import build_stage5_model
from plm_unified.stage5_trainer import (
    Stage5Trainer,
    load_stage5_checkpoint,
    require_stage5_scratch_source,
    stage5_scratch_initialization_source,
)
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed
from plm_unified.triplet_data import (
    build_strict_triplet_datasets,
    make_triplet_dataloader,
)
from plm_unified.triplet_trainer import checkpoint_initialization_source


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
            / "plm_stage5.yaml"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap returned rows per task and split after auditing.",
    )
    return parser.parse_args()


def make_loaders(config, *, limit: int | None):
    values = config.stage5
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
            name: dataset.filter_stats for name, dataset in bundle.as_dict().items()
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
    resume_metadata = None
    if args.resume:
        resume_metadata = load_stage5_checkpoint(args.resume, map_location="cpu")
    resume_timestamp = (
        resume_metadata.get("run_timestamp") if resume_metadata else None
    )
    resume_id = resume_metadata.get("swanlab_run_id") if resume_metadata else None
    initialization_source = stage5_scratch_initialization_source()
    if resume_metadata:
        initialization_source = require_stage5_scratch_source(
            checkpoint_initialization_source(resume_metadata)
        )
    resume_training_config = (
        resume_metadata.get("config", {}).get("training", {})
        if resume_metadata
        else {}
    )
    del resume_metadata
    gc.collect()
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp or resume_timestamp or create_artifact_timestamp()
    )
    config = apply_overrides(
        load_config(args.config),
        fold=(
            args.fold
            if args.fold is not None
            else resume_training_config.get("fold")
        ),
        seed=(
            args.seed
            if args.seed is not None
            else resume_training_config.get("seed")
        ),
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    log_status(
        f"[RUN START] fold={config.training.fold} seed={config.training.seed} "
        f"device={device} timestamp={run_timestamp} limit={args.limit}"
    )

    started_at = time.perf_counter()
    log_status("[MODEL LOAD START] loading frozen ESM-C/ESM-C/TCR-BERT")
    model = build_stage5_model(config, device)
    log_status(f"[MODEL LOAD END] elapsed={time.perf_counter() - started_at:.1f}s")

    started_at = time.perf_counter()
    log_status("[DATA LOAD START] auditing and loading all three tasks")
    train_loaders, val_loaders, data_audit = make_loaders(
        config, limit=args.limit
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

    with SwanLabTracker(
        config,
        resume_id=resume_id,
        run_timestamp=run_timestamp,
    ) as tracker:
        encoder_metadata = model.encoder_metadata()
        tracker.log(
            {
                "input/mode": "online_stage5",
                "stage5/plms_frozen": 1,
                "stage5/inherits_stage1_or_stage2": 0,
                "stage5/initialization_kind": initialization_source.get(
                    "kind", ""
                ),
                "stage5/downstream_initialization": initialization_source.get(
                    "downstream_modules", ""
                ),
                "stage5/base_plms": initialization_source.get(
                    "base_plms", ""
                ),
                "stage5/micro_batch_size": config.stage5.micro_batch_size,
                "stage5/eval_batch_size": config.stage5.eval_batch_size,
                "stage5/gradient_accumulation_steps": (
                    config.stage5.gradient_accumulation_steps
                ),
                "stage5/loss_weight_pht": config.stage5.pht_loss_weight,
                "stage5/loss_weight_phla": config.stage5.phla_loss_weight,
                "stage5/loss_weight_ptcr": config.stage5.ptcr_loss_weight,
                "stage5/loss_reduction": "independent_task_batch_mean",
                "stage5/stage1a_fgm_enabled": 0,
                "stage5/stage1b_fgm_enabled": 1,
                "stage5/fgm_target": "peptide_hla_tcr_adapters",
                "stage5/fgm_epsilon": config.stage5.fgm_epsilon,
                "stage5/epoch_anchor_task": "triplet",
                "stage5/batches_per_joint_step/phla": 1,
                "stage5/batches_per_joint_step/ptcr": 1,
                "stage5/batches_per_joint_step/triplet": 1,
                **{
                    f"stage5/encoder/{entity}/checkpoint_fingerprint": (
                        encoder_metadata[entity]["checkpoint_fingerprint"]
                    )
                    for entity in ("peptide", "hla", "tcr")
                },
                "data/manifest_matches": int(
                    bool(data_audit["manifest"].get("matches"))
                ),
                **{
                    f"data/triplet/{split}/{key}": value
                    for split, stats in data_audit["triplet_splits"].items()
                    for key, value in stats.items()
                },
                **{
                    f"data/{split_task}/{key}": value
                    for split_task, stats in data_audit["binary_splits"].items()
                    for key, value in stats.items()
                },
            },
            step=0,
        )
        trainer = Stage5Trainer(
            model,
            config,
            train_loaders,
            val_loaders,
            tracker,
            initialization_source=initialization_source,
            data_audit=data_audit,
            run_timestamp=run_timestamp,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/stage1a_epoch": summary["stage1a_epoch"],
                "training/stage1b_epoch": summary["stage1b_epoch"],
                "training/best_joint_score": summary["best_joint_score"],
                "training/selected_threshold": summary["selected_threshold"],
            },
            step=trainer.global_step,
        )
    log_status(
        f"[RUN END] phase={summary['phase']} global_step={summary['global_step']}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
