#!/usr/bin/env python3
"""Train or resume Stage12 from the pinned Stage1 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.data import split_path
from plm_unified.stage8_checkpoint import (
    checkpoint_sha256,
    load_stage8_v2_checkpoint,
    validate_stage8_v2_checkpoint,
)
from plm_unified.stage8_data import audit_stage8_class_dirs
from plm_unified.stage8_model import (
    build_progressive_stage8_model,
    initialise_stage8_from_stage1,
)
from plm_unified.stage12_checkpoint import (
    load_stage12_checkpoint,
    restore_stage12_last_checkpoint,
    save_stage12_checkpoint,
)
from plm_unified.stage12_config import (
    apply_stage12_overrides,
    load_stage12_config,
)
from plm_unified.stage12_data import make_stage12_online_loaders
from plm_unified.stage12_evaluation import (
    Stage12BestTracker,
    Stage12SelectionDecision,
    Stage12ValidationScores,
    evaluate_stage12_validation,
)
from plm_unified.stage12_pipeline import (
    fingerprint_paths,
    make_selection_loaders,
    selection_paths,
)
from plm_unified.stage12_trainer import Stage12OnlineTrainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


@dataclass(frozen=True)
class _ResumeProgress:
    start_round: int
    exhausted: bool
    last_scores: Stage12ValidationScores
    last_decision: Stage12SelectionDecision


def _resume_progress(checkpoint: Mapping, config) -> _ResumeProgress:
    state = checkpoint["training_state"]
    metrics = checkpoint["metrics"]
    completed_round = int(state["stage12b_round"])
    start_round = completed_round + 1
    exhausted = start_round > config.stage12.max_rounds or (
        completed_round >= config.stage12.min_rounds
        and int(state["bad_rounds"]) >= config.stage12.early_stopping_patience
    )
    return _ResumeProgress(
        start_round=start_round,
        exhausted=exhausted,
        last_scores=Stage12ValidationScores.from_dict(metrics["validation"]),
        last_decision=Stage12SelectionDecision.from_dict(metrics["decision"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage12.yaml",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int)
    parser.add_argument("--limit-pair-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.audit_only and args.resume is not None:
        raise ValueError("--audit-only and --resume are mutually exclusive")
    if args.resume is not None and args.run_timestamp is not None:
        raise ValueError("--resume derives the run timestamp from the checkpoint")
    for name in ("limit_per_type", "limit_pair_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def _audit(config, timestamp: str):
    result = audit_stage8_class_dirs(
        config.data_dir(),
        config.stage12.class_dirs,
        config.stage12.train_filename,
        config.stage12.val_filename,
        config.model,
        reject_noncanonical_sequences=config.stage12.reject_noncanonical_sequences,
    )
    audit_dir = (
        config.paths.output_root
        / f"fold_{config.runtime.fold}"
        / f"seed_{config.runtime.seed}"
        / f"audit_{timestamp}"
    )
    result.write_reports(audit_dir)
    if result.has_conflicts or result.has_errors:
        raise RuntimeError(f"Stage12 data audit failed; reports: {audit_dir}")
    return result, audit_dir


def _dataset_fingerprints(config, audit) -> dict[str, str]:
    paths = selection_paths(config)
    validation_files = [path for task in paths.values() for path in task.values()]
    return {
        "pht": audit.dataset_fingerprint,
        "validation": fingerprint_paths(validation_files),
        "phla_train": fingerprint_paths([
            split_path(config.paths.data_root, "phla", "train", config.runtime.fold)
        ]),
        "ptcr_train": fingerprint_paths([
            split_path(config.paths.data_root, "ptcr", "train", config.runtime.fold)
        ]),
    }


def _effective_data(args, train_loaders, pht_loader, validation_loaders) -> dict:
    return {
        "limit_per_type": args.limit_per_type,
        "pair_limit": args.limit_pair_train,
        "validation_limit": args.limit_validation,
        "joint_train_samples": len(train_loaders["joint"].dataset),
        "phla_train_samples": len(train_loaders["phla"].dataset),
        "ptcr_train_samples": len(train_loaders["ptcr"].dataset),
        "pht_validation_samples": len(pht_loader.dataset),
        "selection_samples": {
            task: {split: len(loader.dataset) for split, loader in loaders.items()}
            for task, loaders in validation_loaders.items()
        },
    }


def _evaluate_reference(
    config,
    *,
    audit,
    pht_loader,
    validation_loaders,
    device: torch.device,
) -> tuple[Stage12ValidationScores, dict]:
    path = config.paths.stage8_reference_checkpoint.expanduser().resolve()
    actual_sha = checkpoint_sha256(path)
    if actual_sha != config.paths.stage8_reference_sha256:
        raise ValueError("Stage8 reference checkpoint SHA256 mismatch")
    checkpoint = load_stage8_v2_checkpoint(path, artifact_role="best")
    validate_stage8_v2_checkpoint(checkpoint, artifact_role="best")
    if checkpoint["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Stage8 reference uses a different Stage1 source")
    if checkpoint["dataset_fingerprint"] != audit.dataset_fingerprint:
        raise ValueError("Stage8 reference uses a different PHT dataset fingerprint")
    model = build_progressive_stage8_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        current = current_metadata[entity].get("checkpoint_fingerprint")
        if saved and saved != current:
            raise ValueError(f"Stage8 reference base-PLM mismatch for {entity}")
    model.load_state_dict(checkpoint["model"], strict=True)
    scores = evaluate_stage12_validation(
        model,
        pht_loader=pht_loader,
        ph_loaders=validation_loaders["phla"],
        pt_loaders=validation_loaders["ptcr"],
        config=config,
        device=device,
    )
    lineage = {
        "path": str(path),
        "sha256": actual_sha,
        "checkpoint_type": checkpoint["checkpoint_type"],
        "artifact_role": checkpoint["artifact_role"],
        "stage8b_round": checkpoint.get("training_state", {}).get("stage8b_round"),
    }
    del checkpoint, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, lineage


def _add_metrics(payload: dict, prefix: str, values: Mapping) -> None:
    for name, value in values.items():
        if isinstance(value, Real):
            payload[f"{prefix}/{name}"] = float(value)


def _round_log_payload(trainer, training, scores, update) -> dict:
    payload = {
        "stage12/phase": "stage12b",
        "stage12/round": trainer.stage12b_round,
        "stage12/global_step": trainer.global_step,
        "stage12/pt_macro": scores.pt_macro,
        "stage12/ph_macro": scores.ph_macro,
        "stage12/pht_joint": scores.pht_joint,
        "stage12/joint_score": scores.joint_score,
        "stage12/improved": int(update.improved),
        "stage12/success": int(update.improved),
        "stage12/joint_score_delta": update.decision.joint_score_delta,
        "stage12/pt_macro_delta": update.decision.pt_macro_delta,
        "stage12/ph_macro_delta": update.decision.ph_macro_delta,
        "stage12/pht_joint_delta": update.decision.pht_joint_delta,
        "stage12/bad_rounds": trainer.bad_rounds,
    }
    for task, values in training.items():
        _add_metrics(payload, f"stage12/train/{task}", values)
    for split, metrics in scores.pt_metrics.items():
        _add_metrics(payload, f"stage12/validation/pt/{split}", metrics)
        payload[f"stage12/validation/pt/{split}/score"] = scores.pt_splits[split]
    for split, metrics in scores.ph_metrics.items():
        _add_metrics(payload, f"stage12/validation/ph/{split}", metrics)
        payload[f"stage12/validation/ph/{split}/score"] = scores.ph_splits[split]
    return payload


def _tracker_from_state(reference, config, state: Mapping) -> Stage12BestTracker:
    tracker = Stage12BestTracker(
        reference,
        selection_weights=config.stage12.selection_weights,
        min_delta=config.stage12.early_stopping_min_delta,
    )
    best = state.get("best_scores")
    if best:
        tracker.best_scores = Stage12ValidationScores.from_dict(best)
        tracker.best_round = int(state["best_round"])
    return tracker


def _selection_state(tracker: Stage12BestTracker) -> dict:
    return {
        "best_round": tracker.best_round,
        "best_scores": tracker.best_scores.to_dict() if tracker.best_scores else None,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    resume_checkpoint = (
        load_stage12_checkpoint(args.resume, artifact_role="last")
        if args.resume else None
    )
    timestamp = (
        resume_checkpoint["run_timestamp"]
        if resume_checkpoint is not None
        else args.run_timestamp or create_artifact_timestamp()
    )
    config = apply_stage12_overrides(
        load_stage12_config(args.config),
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
    )
    config.validate(require_files=not args.audit_only)
    set_global_seed(config.runtime.seed)
    audit, audit_dir = _audit(config, timestamp)
    if args.audit_only:
        print(json.dumps({
            "status": "clean", "audit_dir": str(audit_dir),
            "dataset_fingerprint": audit.dataset_fingerprint,
        }, indent=2))
        return

    train_loaders, pht_loader = make_stage12_online_loaders(
        audit,
        phla_path=split_path(config.paths.data_root, "phla", "train", config.runtime.fold),
        ptcr_path=split_path(config.paths.data_root, "ptcr", "train", config.runtime.fold),
        model=config.model,
        micro_batch_size=config.stage12.micro_batch_size,
        eval_batch_size=config.stage12.eval_batch_size,
        phla_replay_ratio=config.stage12.phla_replay_ratio,
        ptcr_replay_ratio=config.stage12.ptcr_replay_ratio,
        seed=config.runtime.seed,
        num_workers=config.runtime.num_workers,
        limit_per_type=args.limit_per_type,
        pair_limit=args.limit_pair_train,
    )
    validation_loaders = make_selection_loaders(
        config, limit=args.limit_validation
    )
    dataset_fingerprints = _dataset_fingerprints(config, audit)
    effective_data = _effective_data(
        args, train_loaders, pht_loader, validation_loaders
    )
    device = resolve_device(config.runtime.device)

    if resume_checkpoint is None:
        reference_scores, reference_stage8 = _evaluate_reference(
            config,
            audit=audit,
            pht_loader=pht_loader,
            validation_loaders=validation_loaders,
            device=device,
        )
        model = build_progressive_stage8_model(config, device)
        source_stage1 = initialise_stage8_from_stage1(
            model,
            str(config.paths.stage1_checkpoint),
            model_config=config.model,
            fold=config.runtime.fold,
            seed=config.runtime.seed,
            expected_sha256=config.paths.stage1_sha256,
        )
        resume_id = None
    else:
        reference_scores = Stage12ValidationScores.from_dict(
            resume_checkpoint["reference_scores"]
        )
        reference_stage8 = dict(resume_checkpoint["reference_stage8"])
        source_stage1 = dict(resume_checkpoint["source_stage1"])
        model = build_progressive_stage8_model(config, device)
        resume_id = resume_checkpoint.get("training_state", {}).get("swanlab_run_id")
        resume_progress = _resume_progress(resume_checkpoint, config)

    output_dir = config.run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        role: output_dir / f"stage12_{role}_{timestamp}.pt"
        for role in ("warmup", "best", "last")
    }
    validation_fn = lambda value: evaluate_stage12_validation(
        value,
        pht_loader=pht_loader,
        ph_loaders=validation_loaders["phla"],
        pt_loaders=validation_loaders["ptcr"],
        config=config,
        device=device,
    )
    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=timestamp, experiment_suffix="full"
    ) as swanlab:
        trainer = Stage12OnlineTrainer(
            model, config, train_loaders, validation_fn=validation_fn, tracker=swanlab
        )
        swanlab.log({
            "stage12/reference/pt_macro": reference_scores.pt_macro,
            "stage12/reference/ph_macro": reference_scores.ph_macro,
            "stage12/reference/pht_joint": reference_scores.pht_joint,
            "stage12/config/phla_replay_ratio": config.stage12.phla_replay_ratio,
            "stage12/config/ptcr_replay_ratio": config.stage12.ptcr_replay_ratio,
            "stage12/config/phla_loss_weight": config.stage12.phla_loss_weight,
            "stage12/config/ptcr_loss_weight": config.stage12.ptcr_loss_weight,
            "stage12/config/pht_loss_weight": config.stage12.pht_loss_weight,
        }, step=0)
        if resume_checkpoint is None:
            warmup_scores = None
            for epoch in range(1, config.stage12.stage12a_epochs + 1):
                warmup_training = trainer.train_stage12a_epoch(epoch)
                warmup_scores = trainer.evaluate()
                swanlab.log({
                    "stage12/phase": "stage12a",
                    "stage12/epoch": epoch,
                    "stage12/train/joint/total": warmup_training["total"],
                    "stage12/validation/pt_macro": warmup_scores.pt_macro,
                    "stage12/validation/ph_macro": warmup_scores.ph_macro,
                    "stage12/validation/pht_joint": warmup_scores.pht_joint,
                    "stage12/validation/joint_score": warmup_scores.joint_score,
                }, step=trainer.global_step)
            assert warmup_scores is not None
            save_stage12_checkpoint(
                paths["warmup"],
                artifact_role="warmup",
                trainer=trainer,
                config=config,
                source_stage1=source_stage1,
                reference_stage8=reference_stage8,
                dataset_fingerprints=dataset_fingerprints,
                effective_data=effective_data,
                reference_scores=reference_scores,
                current_scores=warmup_scores,
                decision={"selection_excluded": True, "success": False},
            )
            tracker = _tracker_from_state(reference_scores, config, {})
            start_round = 1
        else:
            restore_stage12_last_checkpoint(
                resume_checkpoint,
                trainer=trainer,
                config=config,
                dataset_fingerprints=dataset_fingerprints,
                effective_data=effective_data,
            )
            tracker = _tracker_from_state(
                reference_scores, config, trainer.selection_state
            )
            start_round = resume_progress.start_round

        if resume_checkpoint is None:
            last_scores = None
            last_decision = None
            exhausted = False
        else:
            last_scores = resume_progress.last_scores
            last_decision = resume_progress.last_decision
            exhausted = resume_progress.exhausted
        round_range = () if exhausted else range(
            start_round, config.stage12.max_rounds + 1
        )
        for round_index in round_range:
            training = trainer.train_stage12b_round(round_index)
            scores = trainer.evaluate()
            update = tracker.consider(round_index, scores)
            if scores.joint_score > trainer.best_joint_score_seen + config.stage12.early_stopping_min_delta:
                trainer.best_joint_score_seen = scores.joint_score
                trainer.bad_rounds = 0
            else:
                trainer.bad_rounds += 1
            trainer.selection_state = _selection_state(tracker)
            if update.improved:
                save_stage12_checkpoint(
                    paths["best"],
                    artifact_role="best",
                    trainer=trainer,
                    config=config,
                    source_stage1=source_stage1,
                    reference_stage8=reference_stage8,
                    dataset_fingerprints=dataset_fingerprints,
                    effective_data=effective_data,
                    reference_scores=reference_scores,
                    current_scores=scores,
                    decision=update.decision.to_dict(),
                )
            save_stage12_checkpoint(
                paths["last"],
                artifact_role="last",
                trainer=trainer,
                config=config,
                source_stage1=source_stage1,
                reference_stage8=reference_stage8,
                dataset_fingerprints=dataset_fingerprints,
                effective_data=effective_data,
                reference_scores=reference_scores,
                current_scores=scores,
                decision=update.decision.to_dict(),
            )
            swanlab.log(
                _round_log_payload(trainer, training, scores, update),
                step=trainer.global_step,
            )
            last_scores = scores
            last_decision = update.decision
            if (
                round_index >= config.stage12.min_rounds
                and trainer.bad_rounds >= config.stage12.early_stopping_patience
            ):
                break

    success = tracker.best_round is not None
    summary = {
        "status": "success" if success else "no_eligible_success",
        "run_timestamp": timestamp,
        "source_stage1": source_stage1,
        "reference_stage8": reference_stage8,
        "reference_scores": reference_scores.to_dict(),
        "selection_state": _selection_state(tracker),
        "last_scores": last_scores.to_dict() if last_scores else None,
        "last_decision": last_decision.to_dict() if last_decision else None,
        "checkpoints": {
            role: str(path) if path.is_file() else None for role, path in paths.items()
        },
        "test_sets_accessed": False,
    }
    summary_path = output_dir / f"summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), **summary["checkpoints"]}, indent=2))


if __name__ == "__main__":
    main()
