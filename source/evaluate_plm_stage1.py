#!/usr/bin/env python3
"""Evaluate a unified PLM checkpoint on configured HLA/TCR splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import (
    create_artifact_timestamp,
    find_latest_artifact,
    find_latest_common_artifacts,
    timestamped_filename,
)
from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_cached_dataloader, make_online_dataloader, split_path
from plm_unified.evaluation import evaluate_model
from plm_unified.encoders import OnlineEncoderBundle, checkpoint_fingerprint
from plm_unified.model import TaskSpecificBindingModel, UnifiedBindingModel
from plm_unified.trainer import (
    BEST_ROUND_CHECKPOINT_FILENAME,
    TASK_CHECKPOINT_FILENAMES,
    load_training_checkpoint,
    resolve_device,
)


def _online_encoded_loader(loader, encoders, task):
    """Wrap an online string-batch loader into the hidden/mask dict evaluate_model expects."""
    for batch in loader:
        encoded = encoders.encode(
            task,
            batch["peptide_sequences"],
            batch["receptor_sequences"],
        )
        yield {
            "peptide_hidden": encoded["peptide_hidden"],
            "peptide_mask": encoded["peptide_mask"],
            "receptor_hidden": encoded["receptor_hidden"],
            "receptor_mask": encoded["receptor_mask"],
            "labels": batch["labels"],
            "group_ids": batch.get("group_ids"),
        }


def _validate_checkpoint_architecture(checkpoint, config, validator):
    if validator is not None:
        validator(checkpoint, config)
    elif checkpoint.get("architecture", "stage1") != "stage1":
        raise ValueError("Expected a Stage1 adapter checkpoint; use the architecture-specific evaluator")


def _joint_model(checkpoint, config, model_class=UnifiedBindingModel, validator=None):
    _validate_checkpoint_architecture(checkpoint, config, validator)
    model = model_class(config.model)
    model.load_state_dict(checkpoint["model"])
    return model


def _task_model(
    checkpoint, config, expected_task,
    model_class=TaskSpecificBindingModel, validator=None,
):
    _validate_checkpoint_architecture(checkpoint, config, validator)
    if checkpoint.get("checkpoint_type") != "task_model":
        raise ValueError("Expected a task_model checkpoint")
    checkpoint_task = checkpoint.get("task")
    if checkpoint_task != expected_task:
        raise ValueError(
            f"Checkpoint task '{checkpoint_task}' does not match '{expected_task}'"
        )
    saved_model_config = checkpoint.get("config", {}).get("model")
    if (
        saved_model_config is not None
        and saved_model_config != config.to_dict()["model"]
    ):
        raise ValueError(
            "Task checkpoint model configuration does not match current config"
        )
    model = model_class(config.model, expected_task)
    model.load_state_dict(checkpoint["model"])
    return model


def parse_args(argv=None, *, default_config="plm_stage1.yaml", default_split="val", default_tasks="phla,ptcr") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / default_config
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--splits", nargs="+", help="Evaluate multiple splits for one selected task.")
    parser.add_argument("--hla-split", default=default_split)
    parser.add_argument("--tcr-split", default=default_split)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Encode frozen PLMs online instead of reading the HDF5 cache. "
        "Use for splits not present in the cache (e.g. independent/external). "
        "Loads ESM-C and TCR-BERT on the device and encodes on the fly.",
    )
    parser.add_argument(
        "--tasks",
        default=default_tasks,
        help="Comma-separated subset of phla,ptcr to evaluate (default: both).",
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        help="Override training.online_batch_size for online encoding.",
    )
    return parser.parse_args(argv)


def main(
    argv=None,
    *,
    default_config="plm_stage1.yaml",
    default_split="val",
    config_loader=load_config,
    model_class=UnifiedBindingModel,
    task_model_class=TaskSpecificBindingModel,
    checkpoint_validator=None,
    default_tasks="phla,ptcr",
    allowed_tasks=("phla", "ptcr"),
    default_configured_splits=False,
    artifact_namespace=None,
) -> None:
    args = parse_args(argv, default_config=default_config, default_split=default_split, default_tasks=default_tasks)
    config = apply_overrides(
        config_loader(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.online_batch_size is not None:
        config.training.online_batch_size = args.online_batch_size
    if args.cache_root is not None:
        config.paths.cache_root = args.cache_root.expanduser().resolve()
    config.validate()
    splits = {"phla": args.hla_split, "ptcr": args.tcr_split}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in allowed_tasks]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Choose from: {sorted(allowed_tasks)}")
    if not tasks:
        raise ValueError("At least one task must be selected")

    selected_splits = args.splits
    if selected_splits is None and default_configured_splits:
        selected_splits = [
            name for name in config.cache.hla_splits
            if name == "val" or name.startswith("val_")
        ] if args.hla_split == default_split else [args.hla_split]
    if selected_splits is not None and (len(tasks) != 1 or not selected_splits):
        raise ValueError("--splits requires exactly one task and at least one split")

    models = {}
    checkpoint_paths = {}
    checkpoint_types = {}
    if args.checkpoint:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_type = checkpoint.get("checkpoint_type", "training")
        checkpoint_task = checkpoint.get("task")
        if checkpoint_type == "task_model":
            if tasks != [checkpoint_task]:
                raise ValueError(
                    f"Task checkpoint '{checkpoint_task}' requires "
                    f"--tasks {checkpoint_task}"
                )
            models[checkpoint_task] = _task_model(
                checkpoint,
                config,
                checkpoint_task,
                task_model_class, checkpoint_validator,
            )
        else:
            model = _joint_model(checkpoint, config, model_class, checkpoint_validator)
            models = {task: model for task in tasks}
        for task in tasks:
            checkpoint_paths[task] = str(checkpoint_path)
            checkpoint_types[task] = checkpoint_type
    else:
        try:
            joint_path = find_latest_artifact(
                config.run_dir() / "best_round",
                BEST_ROUND_CHECKPOINT_FILENAME,
            )
        except FileNotFoundError:
            joint_path = None
        if joint_path is not None:
            checkpoint = load_training_checkpoint(joint_path, map_location="cpu")
            model = _joint_model(checkpoint, config, model_class, checkpoint_validator)
            models = {task: model for task in tasks}
            for task in tasks:
                checkpoint_paths[task] = str(joint_path)
                checkpoint_types[task] = checkpoint.get(
                    "checkpoint_type",
                    "training",
                )
        else:
            resolved_checkpoints = find_latest_common_artifacts(
                config.run_dir(),
                {
                    task: TASK_CHECKPOINT_FILENAMES[task]
                    for task in tasks
                },
            )
            for task in tasks:
                checkpoint_path = resolved_checkpoints[task]
                checkpoint = load_training_checkpoint(
                    checkpoint_path,
                    map_location="cpu",
                )
                if checkpoint.get("checkpoint_type") == "task_model":
                    models[task] = _task_model(
                        checkpoint, config, task, task_model_class, checkpoint_validator,
                    )
                    checkpoint_types[task] = "task_model"
                else:
                    model = _joint_model(checkpoint, config, model_class, checkpoint_validator)
                    models[task] = model
                    checkpoint_types[task] = checkpoint.get(
                        "checkpoint_type",
                        "training",
                    )
                checkpoint_paths[task] = str(checkpoint_path)

    online_encoders = None
    if args.no_cache:
        config.validate(require_models=True)
        online_encoders = OnlineEncoderBundle(
            esmc_path=config.paths.esmc_model,
            tcr_bert_path=config.paths.tcr_bert_model,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            device=resolve_device(config.training.device),
            dtype=config.cache.dtype,
            **({"tasks": tuple(tasks)} if tuple(tasks) != ("phla", "ptcr") else {}),
        )
        if online_encoders.esmc.hidden_size != config.model.peptide_input_dim:
            raise ValueError("Online ESM-C hidden size does not match model config")
        if "ptcr" in tasks and online_encoders.tcr_bert.hidden_size != config.model.tcr_input_dim:
            raise ValueError("Online TCR-BERT hidden size does not match model config")

    cache_dir = config.cache_dir()
    esmc_fingerprint = (
        checkpoint_fingerprint(config.paths.esmc_model)
        if config.paths.esmc_model.is_dir()
        else None
    )
    tcr_fingerprint = (
        checkpoint_fingerprint(config.paths.tcr_bert_model)
        if "ptcr" in tasks and config.paths.tcr_bert_model.is_dir()
        else None
    )
    peptide_cache = None
    receptor_caches = {}
    if not args.no_cache:
        peptide_cache = EmbeddingCache(
            cache_dir / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
            expected_fingerprint=esmc_fingerprint,
        )
        if "phla" in tasks:
            receptor_caches["phla"] = EmbeddingCache(
                cache_dir / "hla",
                expected_hidden_dim=config.model.hla_input_dim,
                expected_fingerprint=esmc_fingerprint,
            )
        if "ptcr" in tasks:
            receptor_caches["ptcr"] = EmbeddingCache(
                cache_dir / "tcr",
                expected_hidden_dim=config.model.tcr_input_dim,
                expected_fingerprint=tcr_fingerprint,
            )

    online_batch_size = args.online_batch_size or config.training.online_batch_size
    # make_online_dataloader requires an int limit; None means "no cap".
    online_limit = args.limit if args.limit is not None else (2**31 - 1)

    results = {}
    for task in tasks:
        for split_name in (selected_splits or [splits[task]]):
            csv_path = split_path(
                config.paths.data_root,
                task,
                split_name,
                config.training.fold,
            )
            if args.no_cache:
                loader = make_online_dataloader(
                    csv_path,
                    task,
                    batch_size=online_batch_size,
                    shuffle=False,
                    seed=config.training.seed,
                    num_workers=config.training.num_workers,
                    pin_memory=config.training.pin_memory,
                    peptide_min_length=config.model.peptide_min_length,
                    peptide_max_length=config.model.peptide_max_length,
                    limit=online_limit,
                )
                loader = _online_encoded_loader(loader, online_encoders, task)
            else:
                loader = make_cached_dataloader(
                    csv_path,
                    task,
                    peptide_cache,
                    receptor_caches[task],
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    seed=config.training.seed,
                    num_workers=config.training.num_workers,
                    pin_memory=config.training.pin_memory,
                    persistent_workers=config.training.persistent_workers,
                    peptide_min_length=config.model.peptide_min_length,
                    peptide_max_length=config.model.peptide_max_length,
                    limit=args.limit,
                )
            metrics = evaluate_model(
                models[task],
                loader,
                task,
                config,
                strict_threshold=True,
            ).to_dict()
            if selected_splits is None:
                results[task] = metrics
            else:
                results.setdefault(task, {})[split_name] = metrics

    report_timestamp = create_artifact_timestamp()
    report_filename = timestamped_filename(
        f"evaluation-{artifact_namespace}.json" if artifact_namespace else
        f"evaluation-hla_{args.hla_split}-tcr_{args.tcr_split}.json",
        report_timestamp,
    )
    report_dir = (
        config.paths.project_root / "reports" / artifact_namespace
        / f"fold_{config.training.fold}" / f"seed_{config.training.seed}"
        if artifact_namespace else config.run_dir()
    )
    report_path = report_dir / report_filename
    report = {
        "artifact_timestamp": report_timestamp,
        "checkpoints": checkpoint_paths,
        "checkpoint_types": checkpoint_types,
        "fold": config.training.fold,
        "seed": config.training.seed,
        "splits": {tasks[0]: selected_splits} if selected_splits is not None else
        {task: splits[task] for task in tasks},
        "metrics": results,
        "report_path": str(report_path.resolve()),
    }
    architecture = getattr(model_class, "architecture", None)
    if architecture is not None:
        report["architecture"] = architecture
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Report written to: {report_path}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
