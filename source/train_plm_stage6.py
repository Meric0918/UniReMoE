#!/usr/bin/env python3
"""Train one isolated Stage-6 PH coverage/task-conflict condition."""

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

from plm_unified.artifacts import create_artifact_timestamp, validate_artifact_timestamp
from plm_unified.data import make_online_dataloader, split_path
from plm_unified.stage6_config import (
    STAGE6_CONDITIONS,
    apply_stage6_overrides,
    load_stage6_config,
)
from plm_unified.stage6_model import build_stage6_model
from plm_unified.stage6_tracking import Stage6Tracker
from plm_unified.stage6_trainer import (
    Stage6Trainer,
    load_stage6_checkpoint,
    require_stage6_scratch_source,
    stage6_scratch_initialization_source,
)
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
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage6.yaml"
        ),
    )
    parser.add_argument("--condition", choices=STAGE6_CONDITIONS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--logical-steps", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap rows per task/split; also pass --logical-steps.",
    )
    return parser.parse_args()


def make_loaders(config, *, limit: int | None):
    values = config.stage6
    common = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    train_loaders = {
        "phla": make_online_dataloader(
            split_path(
                config.paths.data_root, "phla", "train", config.training.fold
            ),
            "phla",
            batch_size=values.micro_batch_size,
            shuffle=True,
            seed=config.training.seed,
            **common,
        )
    }
    val_loaders = {
        "phla": make_online_dataloader(
            split_path(
                config.paths.data_root, "phla", "val", config.training.fold
            ),
            "phla",
            batch_size=values.eval_batch_size,
            shuffle=False,
            seed=config.training.seed,
            **common,
        )
    }
    triplet_audit = {}
    manifest_audit = {}
    if values.task_mode == "joint":
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
        for split, dataset in (("train", bundle.train), ("val", bundle.val)):
            loader = make_triplet_dataloader(
                dataset,
                batch_size=(
                    values.micro_batch_size if split == "train" else values.eval_batch_size
                ),
                shuffle=split == "train",
                seed=config.training.seed,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
            )
            (train_loaders if split == "train" else val_loaders)["triplet"] = loader
        for split in ("train", "val"):
            loaders = train_loaders if split == "train" else val_loaders
            loaders["ptcr"] = make_online_dataloader(
                split_path(
                    config.paths.data_root,
                    "ptcr",
                    split,
                    config.training.fold,
                ),
                "ptcr",
                batch_size=(
                    values.micro_batch_size if split == "train" else values.eval_batch_size
                ),
                shuffle=split == "train",
                seed=config.training.seed,
                **common,
            )
        manifest_audit = bundle.manifest_audit
        triplet_audit = {
            name: dataset.filter_stats for name, dataset in bundle.as_dict().items()
        }
    audit = {
        "condition": values.condition,
        "task_mode": values.task_mode,
        "phla_coverage": values.phla_coverage,
        "logical_steps_per_epoch": values.logical_steps_per_epoch,
        "manifest": manifest_audit,
        "triplet_splits": triplet_audit,
        "binary_splits": {
            f"{split}/{task}": dict(loader.dataset.filter_stats)
            for split, loaders in (("train", train_loaders), ("val", val_loaders))
            for task, loader in loaders.items()
            if task in {"phla", "ptcr"}
        },
        "loader_batches": {
            f"{split}/{task}": len(loader)
            for split, loaders in (("train", train_loaders), ("val", val_loaders))
            for task, loader in loaders.items()
        },
        "loader_samples": {
            f"{split}/{task}": len(loader.dataset)
            for split, loaders in (("train", train_loaders), ("val", val_loaders))
            for task, loader in loaders.items()
        },
    }
    return train_loaders, val_loaders, audit


def main() -> None:
    args = parse_args()
    resume_metadata = None
    if args.resume:
        resume_metadata = load_stage6_checkpoint(args.resume, map_location="cpu")
    resume_config = resume_metadata.get("config", {}) if resume_metadata else {}
    resume_training = resume_config.get("training", {})
    resume_stage6 = resume_config.get("stage6", {})
    resume_timestamp = resume_metadata.get("run_timestamp") if resume_metadata else None
    resume_id = resume_metadata.get("swanlab_run_id") if resume_metadata else None
    initialization_source = stage6_scratch_initialization_source()
    if resume_metadata:
        initialization_source = require_stage6_scratch_source(
            checkpoint_initialization_source(resume_metadata)
        )
    del resume_metadata
    gc.collect()
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp or resume_timestamp or create_artifact_timestamp()
    )
    config = apply_stage6_overrides(
        load_stage6_config(args.config),
        condition=args.condition or resume_stage6.get("condition"),
        fold=args.fold if args.fold is not None else resume_training.get("fold"),
        seed=args.seed if args.seed is not None else resume_training.get("seed"),
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
        logical_steps=(
            args.logical_steps
            if args.logical_steps is not None
            else resume_stage6.get("logical_steps_per_epoch")
        ),
    )
    config.validate(require_models=True)
    if args.limit is not None and args.logical_steps is None and not args.resume:
        raise ValueError("Debug --limit runs must set --logical-steps explicitly")
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    log_status(
        f"[RUN START] condition={config.stage6.condition} "
        f"fold={config.training.fold} seed={config.training.seed} "
        f"device={device} timestamp={run_timestamp} limit={args.limit}"
    )
    started_at = time.perf_counter()
    log_status("[MODEL LOAD START] loading frozen ESM-C/ESM-C/TCR-BERT")
    model = build_stage6_model(config, device)
    log_status(f"[MODEL LOAD END] elapsed={time.perf_counter() - started_at:.1f}s")
    started_at = time.perf_counter()
    log_status("[DATA LOAD START] auditing Stage-6 active tasks")
    train_loaders, val_loaders, data_audit = make_loaders(config, limit=args.limit)
    log_status(
        "[DATA LOAD END] "
        + ", ".join(
            f"{split}/{task}={len(loader.dataset):,}"
            for split, loaders in (("train", train_loaders), ("val", val_loaders))
            for task, loader in loaders.items()
        )
        + f" elapsed={time.perf_counter() - started_at:.1f}s"
    )
    with Stage6Tracker(
        config, resume_id=resume_id, run_timestamp=run_timestamp
    ) as tracker:
        encoder_metadata = model.encoder_metadata()
        tracker.log(
            {
                "input/mode": "online_stage6",
                "stage6/condition": config.stage6.condition,
                "stage6/task_mode": config.stage6.task_mode,
                "stage6/phla_coverage": config.stage6.phla_coverage,
                "stage6/plms_frozen": 1,
                "stage6/initialization_kind": initialization_source["kind"],
                "stage6/logical_steps_per_epoch": (
                    config.stage6.logical_steps_per_epoch
                ),
                "stage6/phla_loader_batches": len(train_loaders["phla"]),
                "stage6/phla_loader_samples": len(train_loaders["phla"].dataset),
                "stage6/loss_weight_pht": config.stage6.pht_loss_weight,
                "stage6/loss_weight_phla": config.stage6.phla_loss_weight,
                "stage6/loss_weight_ptcr": config.stage6.ptcr_loss_weight,
                "stage6/loss_reduction": "active_task_sample_mean_per_logical_step",
                "stage6/stage1a_fgm_enabled": 0,
                "stage6/stage1b_fgm_enabled": 1,
                "stage6/fgm_epsilon": config.stage6.fgm_epsilon,
                **{
                    f"stage6/encoder/{entity}/checkpoint_fingerprint": (
                        encoder_metadata[entity]["checkpoint_fingerprint"]
                    )
                    for entity in ("peptide", "hla", "tcr")
                },
                **{
                    f"data/{name}/{key}": value
                    for name, stats in data_audit["binary_splits"].items()
                    for key, value in stats.items()
                },
            },
            step=0,
        )
        trainer = Stage6Trainer(
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
                "training/best_selection_score": summary["best_selection_score"],
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
