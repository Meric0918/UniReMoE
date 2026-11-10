#!/usr/bin/env python3
"""Train or resume the independent Stage-14 Hyper-PMT lifecycle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.data import split_path
from plm_unified.stage14_checkpoint import (
    initialise_stage14b_checkpoints,
    load_stage14_checkpoint,
    resume_stage14_last_checkpoint,
    save_stage14_best_checkpoint,
    save_stage14_last_checkpoint,
    save_stage14_warmup_checkpoint,
    validate_stage14_warmup_for_run,
)
from plm_unified.stage14_config import (
    Stage14Condition,
    STAGE14_TRAINING_CONDITIONS,
    apply_stage14_overrides,
    load_stage14_config,
)
from plm_unified.stage14_model import (
    build_progressive_stage14_model,
    initialise_stage14_from_stage1,
)
from plm_unified.stage14_trainer import Stage14OnlineTrainer
from plm_unified.stage8_data import audit_stage8_class_dirs, make_stage8_online_loaders
from plm_unified.stage8_evaluation import stage8_task_score
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage14.yaml",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--warmup-only", action="store_true")
    modes.add_argument(
        "--condition",
        choices=[value.value for value in STAGE14_TRAINING_CONDITIONS],
    )
    modes.add_argument("--resume", type=Path)
    modes.add_argument("--all-ablations", action="store_true")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int, help="Debug only")
    args = parser.parse_args(argv)
    if args.warmup_only:
        if args.stage1_checkpoint is None:
            parser.error("--warmup-only requires --stage1-checkpoint")
        if args.warmup_checkpoint is not None:
            parser.error("--warmup-only does not accept --warmup-checkpoint")
    elif args.condition is not None:
        if args.warmup_checkpoint is None:
            parser.error("--condition requires --warmup-checkpoint")
        if args.stage1_checkpoint is not None:
            parser.error("--condition does not accept --stage1-checkpoint")
    elif args.resume is not None:
        if args.stage1_checkpoint is not None or args.warmup_checkpoint is not None:
            parser.error("--resume does not accept checkpoint overrides")
        if args.run_timestamp is not None:
            parser.error("--resume derives --run-timestamp from its last checkpoint")
    elif args.all_ablations:
        if args.stage1_checkpoint is None:
            parser.error("--all-ablations requires --stage1-checkpoint")
        if args.warmup_checkpoint is not None:
            parser.error("--all-ablations creates its own shared warmup")
    if args.limit_per_type is not None and args.limit_per_type < 1:
        parser.error("--limit-per-type must be positive")
    return args


def _append(command: list[str], option: str, value) -> None:
    if value is not None:
        command.extend((option, str(value)))


def _run_all_ablations(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_plm_stage14_ablations.py"),
        "--config", str(args.config),
        "--stage1-checkpoint", str(args.stage1_checkpoint),
    ]
    for option, value in (
        ("--device", args.device), ("--num-workers", args.num_workers),
        ("--swanlab-mode", args.swanlab_mode), ("--run-timestamp", args.run_timestamp),
        ("--limit-per-type", args.limit_per_type),
    ):
        _append(command, option, value)
    subprocess.run(command, check=True)


def _metrics(evaluation) -> dict:
    return {
        "validation_metrics": evaluation.metrics,
        "validation_joint_score": evaluation.joint_score,
        "selected_pht_threshold": evaluation.selected_pht_threshold,
    }


def _add_numeric_metrics(payload: dict, prefix: str, values: Mapping) -> None:
    for name, value in values.items():
        if isinstance(value, Real):
            payload[f"{prefix}/{name}"] = float(value)


def _log_stage14b_round(*, trainer, training: Mapping, evaluation, improved: bool, tracker) -> None:
    payload = {
        "stage14/phase": "stage14b",
        "stage14/condition": trainer.condition.value,
        "stage14/round": trainer.stage8b_round,
        "stage14/global_step": trainer.global_step,
        "stage14/improved": int(improved),
        "stage14/best_joint_score": trainer.best_joint,
        "stage14/bad_rounds": trainer.bad_rounds,
        "stage14/validation_joint_score": evaluation.joint_score,
        "stage14/validation/selected_pht_threshold": evaluation.selected_pht_threshold,
    }
    for task, values in training.items():
        if isinstance(values, Mapping):
            _add_numeric_metrics(payload, f"stage14/train/{task}", values)
    for task in ("pht", "phla", "ptcr"):
        values = evaluation.metrics.get(task, {})
        if isinstance(values, Mapping):
            _add_numeric_metrics(payload, f"stage14/validation/{task}", values)
    for group in ("violations", "constraint_components"):
        values = evaluation.metrics.get(group, {})
        if isinstance(values, Mapping):
            _add_numeric_metrics(payload, f"stage14/validation/{group}", values)
    tracker.log(payload, step=trainer.global_step)


def _audit_and_loaders(config, *, run_timestamp: str, limit_per_type: int | None):
    audit = audit_stage8_class_dirs(
        config.data_dir(), config.stage14.class_dirs,
        config.stage14.train_filename, config.stage14.val_filename, config.model,
        reject_noncanonical_sequences=config.stage14.reject_noncanonical_sequences,
    )
    audit_dir = config.run_dir() / f"audit_{run_timestamp}"
    audit.write_reports(audit_dir)
    if audit.has_conflicts or audit.has_errors:
        print(f"Stage-14 data audit failed; reports: {audit_dir}")
        raise SystemExit(2)
    train_loaders, val_loader = make_stage8_online_loaders(
        audit,
        phla_path=split_path(config.paths.data_root, "phla", "train", config.runtime.fold),
        ptcr_path=split_path(config.paths.data_root, "ptcr", "train", config.runtime.fold),
        model=config.model,
        micro_batch_size=config.stage14.micro_batch_size,
        eval_batch_size=config.stage14.eval_batch_size,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        limit_per_type=limit_per_type,
    )
    return audit, train_loaders, val_loader


def _effective_data(train_loaders, val_loader, limit_per_type: int | None) -> dict:
    return {
        "limit_per_type": limit_per_type,
        "train_samples": len(train_loaders["joint"].dataset),
        "validation_samples": len(val_loader.dataset),
    }


def _warmup_dir(config) -> Path:
    return config.run_dir() / "warmup"


def _train_warmup(*, config, model, train_loaders, val_loader, source_stage1,
                  dataset_fingerprint: str, effective_data, run_timestamp: str, tracker) -> Path:
    trainer = Stage14OnlineTrainer(
        model, config, train_loaders, val_loader,
        condition=Stage14Condition.BASELINE, tracker=tracker,
    )
    best_path: Path | None = None
    best_pht_score = -float("inf")
    for epoch in range(1, config.stage14.stage8a_epochs + 1):
        training = trainer.train_stage14a_epoch(epoch)
        evaluation = trainer.evaluate()
        pht_score = stage8_task_score(evaluation.metrics["pht"])
        tracker.log({
            "stage14/phase": "stage14a",
            "stage14/epoch": epoch,
            "stage14/train_total": training["total"],
            "stage14/validation_pht_score": pht_score,
            "stage14/validation_joint_score": evaluation.joint_score,
        }, step=trainer.global_step)
        if pht_score > best_pht_score:
            best_pht_score = pht_score
            best_path = save_stage14_warmup_checkpoint(
                _warmup_dir(config), model=model, config=config,
                source_stage1=source_stage1,
                dataset_fingerprint=dataset_fingerprint,
                effective_data=effective_data,
                run_timestamp=run_timestamp,
                metrics=_metrics(evaluation),
                training_state=trainer.training_state(),
            )
    if best_path is None:
        raise RuntimeError("Stage14A did not produce a warmup checkpoint")
    return best_path


def _train_stage14b_rounds(*, trainer, output_dir: Path, source_stage1,
                           warmup_checkpoint: Path, dataset_fingerprint: str,
                           effective_data, run_timestamp: str, best_path: Path,
                           start_round: int, tracker) -> tuple[Path, Path]:
    last_path = output_dir / f"stage14_last_{run_timestamp}.pt"
    for round_index in range(start_round, trainer.config.stage14.max_rounds + 1):
        training = trainer.train_stage14b_round(round_index)
        evaluation = trainer.evaluate()
        improved = evaluation.joint_score > trainer.best_joint + trainer.config.stage14.early_stopping_min_delta
        if improved:
            trainer.best_joint = evaluation.joint_score
            trainer.best_evaluation = evaluation
            trainer.bad_rounds = 0
            best_path = save_stage14_best_checkpoint(
                output_dir, trainer=trainer, source_stage1=source_stage1,
                warmup_checkpoint=warmup_checkpoint,
                dataset_fingerprint=dataset_fingerprint,
                effective_data=effective_data, run_timestamp=run_timestamp,
                metrics=_metrics(evaluation),
            )
        else:
            trainer.bad_rounds += 1
        last_path = save_stage14_last_checkpoint(
            output_dir, trainer=trainer, source_stage1=source_stage1,
            warmup_checkpoint=warmup_checkpoint,
            dataset_fingerprint=dataset_fingerprint,
            effective_data=effective_data, run_timestamp=run_timestamp,
            metrics=_metrics(evaluation), best_checkpoint=best_path,
        )
        _log_stage14b_round(
            trainer=trainer, training=training, evaluation=evaluation,
            improved=improved, tracker=tracker,
        )
        if (
            round_index >= trainer.config.stage14.min_rounds
            and trainer.bad_rounds >= trainer.config.stage14.early_stopping_patience
        ):
            break
    return best_path, last_path


def main() -> None:
    args = parse_args()
    if args.all_ablations:
        _run_all_ablations(args)
        return
    resume_value = load_stage14_checkpoint(args.resume, artifact_role="last") if args.resume else None
    if (
        resume_value is not None
        and resume_value.get("condition")
        not in {value.value for value in STAGE14_TRAINING_CONDITIONS}
    ):
        raise ValueError("Stage14 can resume only baseline or full checkpoints")
    resume_condition = (
        Stage14Condition(resume_value["condition"])
        if resume_value is not None else None
    )
    run_timestamp = (
        resume_value["run_timestamp"] if resume_value is not None
        else args.run_timestamp or create_artifact_timestamp()
    )
    config = apply_stage14_overrides(
        load_stage14_config(args.config), device=args.device,
        num_workers=args.num_workers, swanlab_mode=args.swanlab_mode,
    )
    if args.stage1_checkpoint is not None:
        config.paths.stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    config.validate()
    set_global_seed(config.runtime.seed)
    audit, train_loaders, val_loader = _audit_and_loaders(
        config, run_timestamp=run_timestamp, limit_per_type=args.limit_per_type,
    )
    effective_data = _effective_data(train_loaders, val_loader, args.limit_per_type)
    device = resolve_device(config.runtime.device)
    model = build_progressive_stage14_model(config, device)

    if args.warmup_only:
        source_stage1 = initialise_stage14_from_stage1(
            model, config.paths.stage1_checkpoint, model_config=config.model,
            fold=config.runtime.fold, seed=config.runtime.seed,
            expected_sha256=config.paths.stage1_sha256,
        )
        with SwanLabTracker(config, run_timestamp=run_timestamp, experiment_suffix="warmup") as tracker:
            warmup_path = _train_warmup(
                config=config, model=model, train_loaders=train_loaders,
                val_loader=val_loader, source_stage1=source_stage1,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective_data, run_timestamp=run_timestamp,
                tracker=tracker,
            )
        print(f"Stage14A warmup complete: {warmup_path}")
        return

    if resume_value is not None:
        condition = resume_condition
        source_stage1 = dict(resume_value["source_stage1"])
        warmup_checkpoint = Path(resume_value["source_stage14a"]["path"])
        output_dir = args.resume.expanduser().resolve().parent
        best_path = output_dir / f"stage14_best_{run_timestamp}.pt"
        resume_id = resume_value.get("training_state", {}).get("swanlab_run_id")
    else:
        condition = Stage14Condition(args.condition)
        warmup_checkpoint = args.warmup_checkpoint.expanduser().resolve()
        warmup = load_stage14_checkpoint(warmup_checkpoint, artifact_role="warmup")
        validate_stage14_warmup_for_run(
            warmup, config=config, dataset_fingerprint=audit.dataset_fingerprint,
            effective_data=effective_data,
        )
        model.load_state_dict(warmup["model"], strict=True)
        source_stage1 = dict(warmup["source_stage1"])
        output_dir = config.run_dir(condition)
        resume_id = None

    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=run_timestamp,
        experiment_suffix=condition.value,
    ) as tracker:
        trainer = Stage14OnlineTrainer(
            model, config, train_loaders, val_loader,
            condition=condition, tracker=tracker,
        )
        if resume_value is not None:
            resume_stage14_last_checkpoint(
                args.resume, checkpoint=resume_value, trainer=trainer, config=config,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective_data, condition=condition,
            )
            start_round = trainer.stage8b_round + 1
        else:
            trainer.stage8a_epoch = config.stage14.stage8a_epochs
            trainer.begin_stage14b()
            baseline = trainer.evaluate()
            trainer.best_evaluation = baseline
            trainer.best_joint = baseline.joint_score
            trainer.bad_rounds = 0
            best_path, _ = initialise_stage14b_checkpoints(
                output_dir, trainer=trainer, source_stage1=source_stage1,
                warmup_checkpoint=warmup_checkpoint,
                dataset_fingerprint=audit.dataset_fingerprint,
                effective_data=effective_data, run_timestamp=run_timestamp,
                metrics=_metrics(baseline),
            )
            start_round = 1
        best_path, last_path = _train_stage14b_rounds(
            trainer=trainer, output_dir=output_dir, source_stage1=source_stage1,
            warmup_checkpoint=warmup_checkpoint,
            dataset_fingerprint=audit.dataset_fingerprint,
            effective_data=effective_data, run_timestamp=run_timestamp,
            best_path=best_path, start_round=start_round, tracker=tracker,
        )
    print(f"Stage14 {condition.value} best: {best_path}")
    print(f"Stage14 {condition.value} last: {last_path}")


if __name__ == "__main__":
    main()
