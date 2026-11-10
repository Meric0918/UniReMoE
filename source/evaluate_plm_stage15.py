#!/usr/bin/env python3
"""Evaluate Stage15 joint/training/task checkpoints on configured PH/PT splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, find_latest_artifact, find_latest_common_artifacts, timestamped_filename
from plm_unified.cache import EmbeddingCache
from plm_unified.data import make_cached_dataloader, make_online_dataloader, split_path
from plm_unified.encoders import OnlineEncoderBundle, checkpoint_fingerprint
from plm_unified.evaluation import evaluate_model
from plm_unified.stage15_config import apply_stage15_overrides, load_stage15_config
from plm_unified.stage15_model import Stage15BindingModel, Stage15TaskBindingModel
from plm_unified.stage15_trainer import STAGE15_BEST_ROUND_CHECKPOINT_FILENAME, STAGE15_TASK_CHECKPOINT_FILENAMES
from plm_unified.trainer import load_training_checkpoint, resolve_device


def _online_encoded_loader(loader, encoders, task):
    for batch in loader:
        encoded = encoders.encode(task, batch["peptide_sequences"], batch["receptor_sequences"])
        yield {
            "peptide_hidden": encoded["peptide_hidden"],
            "peptide_mask": encoded["peptide_mask"],
            "receptor_hidden": encoded["receptor_hidden"],
            "receptor_mask": encoded["receptor_mask"],
            "labels": batch["labels"],
            "group_ids": batch.get("group_ids"),
        }


def _validate_config(checkpoint, config) -> None:
    saved = checkpoint.get("config", {})
    current = config.to_dict()
    if saved.get("model") != current["model"]:
        raise ValueError("Stage15 checkpoint model configuration mismatch")
    if saved.get("fusion") != current["fusion"]:
        raise ValueError("Stage15 checkpoint fusion configuration mismatch")
    if checkpoint.get("architecture") != "shared_hyper_pmt_pair":
        raise ValueError("Stage15 checkpoint architecture mismatch")
    cache_metadata = checkpoint.get("cache_metadata", {})
    fingerprints = {
        "peptide": checkpoint_fingerprint(config.paths.esmc_model)
        if config.paths.esmc_model.is_dir() else None,
        "hla": checkpoint_fingerprint(config.paths.esmc_model)
        if config.paths.esmc_model.is_dir() else None,
        "tcr": checkpoint_fingerprint(config.paths.tcr_bert_model)
        if config.paths.tcr_bert_model.is_dir() else None,
    }
    for entity, current in fingerprints.items():
        saved = cache_metadata.get(entity, {}).get("model", {}).get(
            "checkpoint_fingerprint"
        )
        if saved and current and saved != current:
            raise ValueError(f"Stage15 base-PLM fingerprint mismatch for {entity}")


def _load_model(checkpoint, config, task=None):
    _validate_config(checkpoint, config)
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type == "stage15_task_model":
        if checkpoint.get("task") != task:
            raise ValueError(f"Stage15 task checkpoint does not match task '{task}'")
        model = Stage15TaskBindingModel(config.model, config.fusion, task)
    elif checkpoint_type in {"stage15_training_model", "stage15_joint_model"}:
        model = Stage15BindingModel(config.model, config.fusion)
    else:
        raise ValueError(f"Unsupported Stage15 checkpoint type: {checkpoint_type}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "plm_stage15.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--hla-split", default="val_unseen_pair")
    parser.add_argument("--tcr-split", default="val_unseen_pair")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tasks", default="phla,ptcr")
    parser.add_argument("--online-batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_stage15_overrides(
        load_stage15_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    splits = {"phla": args.hla_split, "ptcr": args.tcr_split}
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks or any(task not in splits for task in tasks):
        raise ValueError("--tasks must be a non-empty subset of phla,ptcr")

    models = {}
    checkpoint_paths = {}
    checkpoint_types = {}
    if args.checkpoint:
        path = args.checkpoint.expanduser().resolve()
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        if checkpoint.get("checkpoint_type") == "stage15_task_model":
            task = checkpoint.get("task")
            if tasks != [task]:
                raise ValueError(f"Task checkpoint requires --tasks {task}")
            models[task] = _load_model(checkpoint, config, task)
        else:
            model = _load_model(checkpoint, config)
            models = {task: model for task in tasks}
        checkpoint_paths = {task: str(path) for task in tasks}
        checkpoint_types = {task: checkpoint.get("checkpoint_type") for task in tasks}
    else:
        try:
            joint_path = find_latest_artifact(
                config.run_dir() / "best_round",
                STAGE15_BEST_ROUND_CHECKPOINT_FILENAME,
            )
        except FileNotFoundError:
            joint_path = None
        if joint_path:
            checkpoint = load_training_checkpoint(joint_path, map_location="cpu")
            model = _load_model(checkpoint, config)
            models = {task: model for task in tasks}
            checkpoint_paths = {task: str(joint_path) for task in tasks}
            checkpoint_types = {task: checkpoint.get("checkpoint_type") for task in tasks}
        else:
            paths = find_latest_common_artifacts(
                config.run_dir(),
                {task: STAGE15_TASK_CHECKPOINT_FILENAMES[task] for task in tasks},
            )
            for task, path in paths.items():
                checkpoint = load_training_checkpoint(path, map_location="cpu")
                models[task] = _load_model(checkpoint, config, task)
                checkpoint_paths[task] = str(path)
                checkpoint_types[task] = checkpoint.get("checkpoint_type")

    encoders = None
    if args.no_cache:
        config.validate(require_models=True)
        encoders = OnlineEncoderBundle(
            esmc_path=config.paths.esmc_model,
            tcr_bert_path=config.paths.tcr_bert_model,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            device=resolve_device(config.training.device),
            dtype=config.cache.dtype,
        )

    peptide_cache = None
    receptor_caches = {}
    if not args.no_cache:
        esmc_fingerprint = checkpoint_fingerprint(config.paths.esmc_model) if config.paths.esmc_model.is_dir() else None
        tcr_fingerprint = checkpoint_fingerprint(config.paths.tcr_bert_model) if config.paths.tcr_bert_model.is_dir() else None
        peptide_cache = EmbeddingCache(config.cache_dir() / "peptide", expected_hidden_dim=config.model.peptide_input_dim, expected_fingerprint=esmc_fingerprint)
        if "phla" in tasks:
            receptor_caches["phla"] = EmbeddingCache(config.cache_dir() / "hla", expected_hidden_dim=config.model.hla_input_dim, expected_fingerprint=esmc_fingerprint)
        if "ptcr" in tasks:
            receptor_caches["ptcr"] = EmbeddingCache(config.cache_dir() / "tcr", expected_hidden_dim=config.model.tcr_input_dim, expected_fingerprint=tcr_fingerprint)

    metrics = {}
    for task in tasks:
        csv_path = split_path(config.paths.data_root, task, splits[task], config.training.fold)
        if args.no_cache:
            loader = make_online_dataloader(
                csv_path, task,
                batch_size=args.online_batch_size or config.training.online_batch_size,
                shuffle=False, seed=config.training.seed,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=args.limit if args.limit is not None else 2**31 - 1,
            )
            loader = _online_encoded_loader(loader, encoders, task)
        else:
            loader = make_cached_dataloader(
                csv_path, task, peptide_cache, receptor_caches[task],
                batch_size=config.training.batch_size,
                shuffle=False, seed=config.training.seed,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                persistent_workers=config.training.persistent_workers,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=args.limit,
            )
        metrics[task] = evaluate_model(
            models[task], loader, task, config, strict_threshold=True
        ).to_dict()

    timestamp = create_artifact_timestamp()
    report_path = config.run_dir() / timestamped_filename(
        f"stage15_evaluation-hla_{args.hla_split}-tcr_{args.tcr_split}.json",
        timestamp,
    )
    report = {
        "stage": "stage15", "architecture": "shared_hyper_pmt_pair",
        "artifact_timestamp": timestamp, "checkpoints": checkpoint_paths,
        "checkpoint_types": checkpoint_types, "fold": config.training.fold,
        "seed": config.training.seed, "splits": splits, "metrics": metrics,
        "threshold": 0.5, "report_path": str(report_path.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
