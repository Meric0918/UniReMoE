#!/usr/bin/env python3
"""Train or resume the PHT-only Stage PHT lifecycle."""

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
from plm_unified.stage_PHT_checkpoint import (
    evaluation_payload,
    load_stage_PHT_checkpoint,
    resume_stage_PHT_last,
    restore_stage_PHT_model,
    save_stage_PHT_best,
    save_stage_PHT_last,
    save_stage_PHT_warmup,
    validate_stage_PHT_checkpoint,
)
from plm_unified.stage_PHT_config import (
    apply_stage_PHT_overrides,
    load_stage_PHT_config,
)
from plm_unified.stage_adapter_data import make_stage_adapter_loaders
from plm_unified.stage_PHT_model import (
    build_stage_PHT_model,
    initialise_stage_PHT_random,
)
from plm_unified.stage_PHT_trainer import StagePHTTrainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/plm_stage_PHT.yaml"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--warmup-only", action="store_true")
    modes.add_argument("--warmup-checkpoint", type=Path)
    modes.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int)
    args = parser.parse_args(argv)
    if args.resume is not None and args.run_timestamp is not None:
        parser.error("resume does not accept timestamp overrides")
    return args


def _audit_and_load(config, *, timestamp: str, limit_per_type: int | None):
    values = config.stage_PHT
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
        "stage_PHT/phase": phase,
        "stage_PHT/progress": progress,
        "stage_PHT/train/loss": training["loss"],
        "stage_PHT/train/gradient_norm": training["gradient_norm"],
        "stage_PHT/validation/selection_score": evaluation.selection_score,
        "stage_PHT/validation/selected_threshold": evaluation.selected_threshold,
    }
    payload.update({f"stage_PHT/train/pht/{k}": v for k, v in train_metrics.items()})
    payload.update({f"stage_PHT/validation/pht/{k}": v for k, v in metrics.items()})
    for group in optimizer.param_groups:
        payload[f"stage_PHT/learning_rate/{group['name']}"] = group["lr"]
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
        load_stage_PHT_checkpoint(args.resume, role="last")
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
    config = load_stage_PHT_config(args.config)
    config = apply_stage_PHT_overrides(
        config,
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
    )
    new_run = args.resume is None and args.warmup_checkpoint is None
    config.validate(require_files=new_run)
    set_global_seed(config.runtime.seed)
    audit, train_loader, val_loader, effective = _audit_and_load(
        config, timestamp=timestamp, limit_per_type=effective_limit
    )
    device = resolve_device(config.runtime.device)
    model = build_stage_PHT_model(config, device)

    if new_run:
        source_initialization = initialise_stage_PHT_random(model, config=config)
        with SwanLabTracker(
            config, run_timestamp=timestamp, experiment_suffix="warmup"
        ) as tracker:
            trainer = StagePHTTrainer(
                model, config, train_loader, val_loader, tracker=tracker
            )
            best_path = None
            best_score = -float("inf")
            for epoch in range(1, config.stage_PHT.warmup_epochs + 1):
                training = trainer.train_warmup_epoch(epoch)
                evaluation = trainer.evaluate()
                _log(
                    tracker, phase="warmup", progress=epoch,
                    training=training, evaluation=evaluation,
                    step=trainer.global_step,
                    optimizer=trainer.optimizer,
                )
                if evaluation.selection_score > best_score:
                    best_score = evaluation.selection_score
                    best_path = save_stage_PHT_warmup(
                        config.warmup_dir(), model=model, config=config,
                        source_initialization=source_initialization,
                        dataset_fingerprint=audit.dataset_fingerprint,
                        effective_data=effective, run_timestamp=timestamp,
                        evaluation=evaluation,
                        training_state=trainer.training_state(),
                    )
        if best_path is None:
            raise RuntimeError("Stage PHT warmup produced no checkpoint")
        print(f"Stage PHT warmup complete: {best_path}")
        if args.warmup_only:
            return
        # Release warmup optimizer moments before allocating the next optimizer.
        del trainer
        args.warmup_checkpoint = best_path

    if resume_value:
        source_initialization = dict(resume_value["source_initialization"])
        warmup_checkpoint = Path(resume_value["source_warmup"]["path"])
        output_dir = args.resume.resolve().parent
        resume_id = (resume_value.get("training_state") or {}).get("swanlab_run_id")
    else:
        warmup_checkpoint = args.warmup_checkpoint.expanduser().resolve()
        warmup = load_stage_PHT_checkpoint(warmup_checkpoint, role="warmup")
        validate_stage_PHT_checkpoint(
            warmup, config=config, dataset_fingerprint=audit.dataset_fingerprint,
            effective_data=effective,
        )
        restore_stage_PHT_model(warmup, model)
        source_initialization = dict(warmup["source_initialization"])
        output_dir = config.run_dir()
        resume_id = (warmup.get("training_state") or {}).get("swanlab_run_id")
        del warmup

    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=timestamp,
        experiment_suffix="pht-only",
    ) as tracker:
        trainer = StagePHTTrainer(
            model, config, train_loader, val_loader, tracker=tracker
        )
        if resume_value:
            resume_stage_PHT_last(
                resume_value, trainer=trainer, config=config,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective,
            )
            best_path = Path(resume_value["best_checkpoint"]["path"])
            start_round = trainer.training_round + 1
        else:
            trainer.warmup_epoch = config.stage_PHT.warmup_epochs
            trainer.begin_training()
            initial = trainer.evaluate()
            trainer.best_score = initial.selection_score
            trainer.best_evaluation = initial
            best_path = save_stage_PHT_best(
                output_dir, trainer=trainer, source_initialization=source_initialization,
                warmup_checkpoint=warmup_checkpoint,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective, run_timestamp=timestamp,
                evaluation=initial,
            )
            start_round = 1

        last_path = args.resume.resolve() if args.resume else None
        for round_index in range(start_round, config.stage_PHT.max_rounds + 1):
            training = trainer.train_round(round_index)
            evaluation = trainer.evaluate()
            improved = (
                evaluation.selection_score
                > trainer.best_score + config.stage_PHT.early_stopping_min_delta
            )
            if improved:
                trainer.best_score = evaluation.selection_score
                trainer.best_evaluation = evaluation
                trainer.bad_rounds = 0
                best_path = save_stage_PHT_best(
                    output_dir, trainer=trainer, source_initialization=source_initialization,
                    warmup_checkpoint=warmup_checkpoint,
                    dataset_fingerprint=audit.dataset_fingerprint,
                    effective_data=effective, run_timestamp=timestamp,
                    evaluation=evaluation,
                )
            else:
                trainer.bad_rounds += 1
            last_path = save_stage_PHT_last(
                output_dir, trainer=trainer, source_initialization=source_initialization,
                warmup_checkpoint=warmup_checkpoint,
                best_checkpoint=best_path,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective, run_timestamp=timestamp,
                evaluation=evaluation,
            )
            _log(
                tracker, phase="training", progress=round_index,
                training=training, evaluation=evaluation,
                step=trainer.global_step,
                optimizer=trainer.optimizer,
            )
            if (
                round_index >= config.stage_PHT.min_rounds
                and trainer.bad_rounds >= config.stage_PHT.early_stopping_patience
            ):
                break
    print(json.dumps({
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_pht_score": trainer.best_score,
        "round": trainer.training_round,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
