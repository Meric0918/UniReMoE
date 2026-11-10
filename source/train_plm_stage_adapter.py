#!/usr/bin/env python3
"""Train or resume the PHT-only Stage Adapter lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage8_data import audit_stage8_class_dirs
from plm_unified.stage_adapter_checkpoint import (
    evaluation_payload,
    load_stage_adapter_checkpoint,
    resume_stage_adapter_last,
    restore_stage_adapter_model,
    save_stage_adapter_best,
    save_stage_adapter_last,
    save_stage_adapter_warmup,
    validate_stage_adapter_checkpoint,
)
from plm_unified.stage_adapter_config import (
    apply_stage_adapter_overrides,
    load_stage_adapter_config,
)
from plm_unified.stage_adapter_data import make_stage_adapter_loaders
from plm_unified.stage_adapter_model import (
    build_progressive_stage_adapter_model,
    initialise_stage_adapter_from_checkpoint,
    initialise_stage_adapter_random,
)
from plm_unified.stage_adapter_trainer import StageAdapterTrainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/plm_stage_adapter.yaml"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--warmup-only", action="store_true")
    modes.add_argument("--warmup-checkpoint", type=Path)
    modes.add_argument("--resume", type=Path)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Load and freeze pretrained PLMs, randomly initialize binding modules, then train")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int)
    args = parser.parse_args(argv)
    if args.from_scratch and (args.resume or args.warmup_checkpoint or args.stage1_checkpoint):
        parser.error("--from-scratch cannot be combined with a source or resume checkpoint")
    if args.warmup_checkpoint is not None:
        if args.stage1_checkpoint is not None:
            parser.error("training from warmup does not accept --stage1-checkpoint")
    elif args.resume is not None:
        if args.stage1_checkpoint is not None or args.run_timestamp is not None:
            parser.error("resume does not accept source or timestamp overrides")
    return args


def _audit_and_load(config, *, timestamp: str, limit_per_type: int | None):
    values = config.stage_adapter
    audit = audit_stage8_class_dirs(
        config.data_dir(), values.class_dirs,
        values.train_filename, values.val_filename, config.model,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
    )
    audit_dir = config.run_dir() / f"audit_{timestamp}"
    audit.write_reports(audit_dir)
    audit.require_clean()
    loaders = make_stage_adapter_loaders(
        audit,
        micro_batch_size=values.micro_batch_size,
        eval_batch_size=values.eval_batch_size,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        limit_per_type=limit_per_type,
    )
    effective = {
        "limit_per_type": limit_per_type,
        "train_samples": len(loaders[0].dataset),
        "validation_samples": len(loaders[1].dataset),
    }
    return audit, loaders[0], loaders[1], effective


def _log(tracker, *, phase, progress, training, evaluation, step, optimizer):
    metrics = evaluation.metrics.to_dict()
    train_metrics = training["metrics"]
    payload = {
        "stage_adapter/phase": phase,
        "stage_adapter/progress": progress,
        "stage_adapter/train/loss": training["loss"],
        "stage_adapter/train/gradient_norm": training["gradient_norm"],
        "stage_adapter/validation/selection_score": evaluation.selection_score,
        "stage_adapter/validation/selected_threshold": evaluation.selected_threshold,
    }
    payload.update({f"stage_adapter/train/pht/{k}": v for k, v in train_metrics.items()})
    payload.update({f"stage_adapter/validation/pht/{k}": v for k, v in metrics.items()})
    for group in optimizer.param_groups:
        payload[f"stage_adapter/learning_rate/{group['name']}"] = group["lr"]
    tracker.log(payload, step=step)
    print(json.dumps({
        "phase": phase,
        "progress": progress,
        "global_step": step,
        "train": training,
        "validation": evaluation_payload(evaluation),
    }, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    resume_value = (
        load_stage_adapter_checkpoint(args.resume, role="last")
        if args.resume else None
    )
    effective_limit = args.limit_per_type
    if resume_value:
        saved_limit = (resume_value.get("effective_data") or {}).get(
            "limit_per_type"
        )
        if effective_limit is not None and effective_limit != saved_limit:
            raise ValueError(
                "--limit-per-type does not match the resumed checkpoint"
            )
        effective_limit = saved_limit
    timestamp = (
        resume_value["run_timestamp"] if resume_value
        else args.run_timestamp or create_artifact_timestamp()
    )
    config = load_stage_adapter_config(
        args.config, initialization="random_binding" if args.from_scratch else None
    )
    if args.from_scratch:
        config.stage_adapter.initialization = "random_binding"
    config = apply_stage_adapter_overrides(
        config,
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
    )
    if args.stage1_checkpoint is not None:
        config.paths.stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    new_run = args.resume is None and args.warmup_checkpoint is None
    if new_run and not args.warmup_only and config.stage_adapter.initialization == "checkpoint":
        raise ValueError("Use --from-scratch, --warmup-only, --warmup-checkpoint, or --resume")
    if config.stage_adapter.initialization != "checkpoint" and args.stage1_checkpoint:
        raise ValueError("Random initialization does not accept --stage1-checkpoint")
    config.validate(require_files=new_run)
    set_global_seed(config.runtime.seed)
    audit, train_loader, val_loader, effective = _audit_and_load(
        config, timestamp=timestamp, limit_per_type=effective_limit
    )
    device = resolve_device(config.runtime.device)
    model = build_progressive_stage_adapter_model(config, device)

    if new_run:
        if config.stage_adapter.initialization == "checkpoint":
            source_stage1 = initialise_stage_adapter_from_checkpoint(
                model, config.paths.stage1_checkpoint, config=config
            )
        else:
            source_stage1 = initialise_stage_adapter_random(model, config=config)
        with SwanLabTracker(
            config, run_timestamp=timestamp, experiment_suffix="warmup"
        ) as tracker:
            trainer = StageAdapterTrainer(
                model, config, train_loader, val_loader, tracker=tracker
            )
            best_path = None
            best_score = -float("inf")
            for epoch in range(1, config.stage_adapter.stage_adapter_a_epochs + 1):
                training = trainer.train_stage_adapter_a_epoch(epoch)
                evaluation = trainer.evaluate()
                _log(
                    tracker, phase="stage_adapter_a", progress=epoch,
                    training=training, evaluation=evaluation,
                    step=trainer.global_step,
                    optimizer=trainer.optimizer,
                )
                if evaluation.selection_score > best_score:
                    best_score = evaluation.selection_score
                    best_path = save_stage_adapter_warmup(
                        config.warmup_dir(), model=model, config=config,
                        source_stage1=source_stage1,
                        dataset_fingerprint=audit.dataset_fingerprint,
                        effective_data=effective, run_timestamp=timestamp,
                        evaluation=evaluation,
                        training_state=trainer.training_state(),
                    )
        if best_path is None:
            raise RuntimeError("StageAdapterA produced no checkpoint")
        print(f"StageAdapterA warmup complete: {best_path}")
        if args.warmup_only:
            return
        # Release warmup optimizer moments before allocating the next optimizer.
        del trainer
        args.warmup_checkpoint = best_path

    if resume_value:
        source_stage1 = dict(resume_value["source_stage1"])
        warmup_checkpoint = Path(resume_value["source_warmup"]["path"])
        output_dir = args.resume.resolve().parent
        resume_id = (resume_value.get("training_state") or {}).get("swanlab_run_id")
    else:
        warmup_checkpoint = args.warmup_checkpoint.expanduser().resolve()
        warmup = load_stage_adapter_checkpoint(warmup_checkpoint, role="warmup")
        validate_stage_adapter_checkpoint(
            warmup, config=config, dataset_fingerprint=audit.dataset_fingerprint,
            effective_data=effective,
        )
        restore_stage_adapter_model(warmup, model)
        source_stage1 = dict(warmup["source_stage1"])
        output_dir = config.run_dir()
        resume_id = (warmup.get("training_state") or {}).get("swanlab_run_id")
        del warmup

    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=timestamp,
        experiment_suffix="pht-only",
    ) as tracker:
        trainer = StageAdapterTrainer(
            model, config, train_loader, val_loader, tracker=tracker
        )
        if resume_value:
            resume_stage_adapter_last(
                resume_value, trainer=trainer, config=config,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective,
            )
            best_path = Path(resume_value["best_checkpoint"]["path"])
            start_round = trainer.stage_adapter_b_round + 1
        else:
            trainer.stage_adapter_a_epoch = config.stage_adapter.stage_adapter_a_epochs
            trainer.begin_stage_adapter_b()
            initial = trainer.evaluate()
            trainer.best_score = initial.selection_score
            trainer.best_evaluation = initial
            best_path = save_stage_adapter_best(
                output_dir, trainer=trainer, source_stage1=source_stage1,
                warmup_checkpoint=warmup_checkpoint,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective, run_timestamp=timestamp,
                evaluation=initial,
            )
            start_round = 1

        last_path = None
        for round_index in range(start_round, config.stage_adapter.max_rounds + 1):
            training = trainer.train_stage_adapter_b_round(round_index)
            evaluation = trainer.evaluate()
            improved = (
                evaluation.selection_score
                > trainer.best_score + config.stage_adapter.early_stopping_min_delta
            )
            if improved:
                trainer.best_score = evaluation.selection_score
                trainer.best_evaluation = evaluation
                trainer.bad_rounds = 0
                best_path = save_stage_adapter_best(
                    output_dir, trainer=trainer, source_stage1=source_stage1,
                    warmup_checkpoint=warmup_checkpoint,
                    dataset_fingerprint=audit.dataset_fingerprint,
                    effective_data=effective, run_timestamp=timestamp,
                    evaluation=evaluation,
                )
            else:
                trainer.bad_rounds += 1
            last_path = save_stage_adapter_last(
                output_dir, trainer=trainer, source_stage1=source_stage1,
                warmup_checkpoint=warmup_checkpoint,
                best_checkpoint=best_path,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective, run_timestamp=timestamp,
                evaluation=evaluation,
            )
            _log(
                tracker, phase="stage_adapter_b", progress=round_index,
                training=training, evaluation=evaluation,
                step=trainer.global_step,
                optimizer=trainer.optimizer,
            )
            if (
                round_index >= config.stage_adapter.min_rounds
                and trainer.bad_rounds >= config.stage_adapter.early_stopping_patience
            ):
                break
    print(json.dumps({
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_pht_score": trainer.best_score,
        "round": trainer.stage_adapter_b_round,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
