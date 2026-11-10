#!/usr/bin/env python3
"""Train stage 1A/1B of the unified ESM-C/TCR-BERT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import (
    create_artifact_timestamp,
    validate_artifact_timestamp,
)
from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import (
    make_cached_dataloader,
    make_online_dataloader,
    make_sequential_dataloader,
    split_path,
)
from plm_unified.encoders import OnlineEncoderBundle, checkpoint_fingerprint
from plm_unified.model import UnifiedBindingModel
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import (
    Stage1Trainer,
    load_training_checkpoint,
    peek_swanlab_run_id,
    set_global_seed,
)


def parse_args(argv=None, *, default_config="plm_stage1.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / default_config
        ),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--batch-size", type=int, help="Override cached training batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader workers; use 0 to debug cache reads in-process.",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--run-timestamp",
        help="Optional shared artifact suffix in YYYYMMDD_HHMMSS_ffffff format.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Debug only: encode frozen PLMs online; requires --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: limit rows per train/validation split.",
    )
    umap_group = parser.add_mutually_exclusive_group()
    umap_group.add_argument(
        "--best-umap",
        dest="best_umap",
        action="store_true",
        help="Generate fixed-reference UMAP plots for every newly improved task model.",
    )
    umap_group.add_argument(
        "--no-best-umap",
        dest="best_umap",
        action="store_false",
        help="Disable best-model UMAP even when enabled in the YAML config.",
    )
    parser.set_defaults(best_umap=None)
    parser.add_argument(
        "--best-umap-output-dir",
        type=Path,
        help=(
            "Override the best-model UMAP output directory. The default is "
            "figures/<stage>/best_model_umap/fold_<fold>/seed_<seed>/<timestamp>."
        ),
    )
    parser.add_argument(
        "--best-umap-max-samples-per-split",
        type=int,
        help="Override the maximum fixed train/validation sample count per split.",
    )
    return parser.parse_args(argv)


def open_caches(config, *, tasks=("phla", "ptcr")):
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
    caches = {
        "peptide": EmbeddingCache(
            cache_dir / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
            expected_fingerprint=esmc_fingerprint,
        ),
    }
    if "phla" in tasks:
        caches["hla"] = EmbeddingCache(
            cache_dir / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
            expected_fingerprint=esmc_fingerprint,
        )
    if "ptcr" in tasks:
        caches["tcr"] = EmbeddingCache(
            cache_dir / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
            expected_fingerprint=tcr_fingerprint,
        )
    return caches


def make_loaders(config, caches, limit, *, online: bool, tasks=("phla", "ptcr")):
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "persistent_workers": config.training.persistent_workers,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    train_loaders = {}
    val_loaders = {}
    for task, receptor_name in (("phla", "hla"), ("ptcr", "tcr")):
        if task not in tasks:
            continue
        configured_splits = (
            config.cache.hla_splits if task == "phla" else config.cache.tcr_splits
        )
        validation_splits = [
            split
            for split in configured_splits
            if split == "val" or split.startswith("val_")
        ]
        if not validation_splits:
            raise ValueError(f"No validation splits configured for task {task}")
        train_path = split_path(
            config.paths.data_root,
            task,
            "train",
            config.training.fold,
        )
        validation_paths = {
            split: split_path(
                config.paths.data_root,
                task,
                split,
                config.training.fold,
            )
            for split in validation_splits
        }
        if online:
            online_common = {
                "batch_size": config.training.online_batch_size,
                "num_workers": config.training.num_workers,
                "pin_memory": config.training.pin_memory,
                "peptide_min_length": config.model.peptide_min_length,
                "peptide_max_length": config.model.peptide_max_length,
                "limit": limit,
            }
            train_loaders[task] = make_online_dataloader(
                train_path,
                task,
                shuffle=True,
                seed=config.training.seed,
                **online_common,
            )
            val_loaders[task] = {
                split: make_online_dataloader(
                    path,
                    task,
                    shuffle=False,
                    seed=config.training.seed,
                    **online_common,
                )
                for split, path in validation_paths.items()
            }
        else:
            train_loaders[task] = make_cached_dataloader(
                train_path,
                task,
                caches["peptide"],
                caches[receptor_name],
                shuffle=True,
                seed=config.training.seed,
                **common,
            )
            val_loaders[task] = {
                split: make_cached_dataloader(
                    path,
                    task,
                    caches["peptide"],
                    caches[receptor_name],
                    shuffle=False,
                    seed=config.training.seed,
                    **common,
                )
                for split, path in validation_paths.items()
            }
    return train_loaders, val_loaders


def main(
    argv=None,
    *,
    default_config="plm_stage1.yaml",
    config_loader=load_config,
    model_class=UnifiedBindingModel,
    trainer_class=Stage1Trainer,
    artifact_namespace="stage1",
    umap_stages=None,
    tasks=("phla", "ptcr"),
) -> None:
    args = parse_args(argv, default_config=default_config)
    resume_timestamp = None
    if args.resume and not args.run_timestamp:
        resume_timestamp = load_training_checkpoint(
            args.resume,
            map_location="cpu",
        ).get("run_timestamp")
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp or resume_timestamp or create_artifact_timestamp()
    )
    config = apply_overrides(
        config_loader(args.config),
        fold=args.fold,
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        config.validate()
    if args.cache_root is not None:
        config.paths.cache_root = args.cache_root.expanduser().resolve()
    if args.no_cache and args.limit is None:
        raise ValueError("--no-cache is a debug mode and requires an explicit --limit")
    set_global_seed(config.training.seed)
    online_encoders = None
    if args.no_cache:
        from plm_unified.trainer import resolve_device

        config.validate(require_models=True)
        online_encoders = OnlineEncoderBundle(
            esmc_path=config.paths.esmc_model,
            tcr_bert_path=config.paths.tcr_bert_model,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            device=resolve_device(config.training.device),
            dtype=config.cache.dtype,
            **({"tasks": tasks} if tasks != ("phla", "ptcr") else {}),
        )
        if online_encoders.esmc.hidden_size != config.model.peptide_input_dim:
            raise ValueError("Online ESM-C hidden size does not match model config")
        if "ptcr" in tasks and online_encoders.tcr_bert.hidden_size != config.model.tcr_input_dim:
            raise ValueError("Online TCR-BERT hidden size does not match model config")
        caches = {}
    else:
        caches = open_caches(config, tasks=tasks)
    train_loaders, val_loaders = make_loaders(
        config,
        caches,
        args.limit,
        online=args.no_cache,
        tasks=tasks,
    )
    model = model_class(config.model)
    best_umap_visualizer = None
    umap_train_loaders = {}
    best_umap_enabled = (
        config.training.best_umap_enabled
        if args.best_umap is None
        else args.best_umap
    )
    if best_umap_enabled:
        from plm_unified.stage1_umap import Stage1BestUMAPVisualizer

        umap_train_loaders = {
            task: make_sequential_dataloader(loader)
            for task, loader in train_loaders.items()
        }
        best_umap_output_dir = (
            args.best_umap_output_dir.expanduser().resolve()
            if args.best_umap_output_dir is not None
            else config.paths.project_root
            / "figures"
            / artifact_namespace
            / "best_model_umap"
            / f"fold_{config.training.fold}"
            / f"seed_{config.training.seed}"
            / run_timestamp
        )
        best_umap_visualizer = Stage1BestUMAPVisualizer(
            best_umap_output_dir,
            val_loaders,
            train_loaders=umap_train_loaders,
            max_samples_per_split=(
                args.best_umap_max_samples_per_split
                if args.best_umap_max_samples_per_split is not None
                else config.training.best_umap_max_samples_per_split
            ),
            random_state=config.training.best_umap_random_state,
            n_neighbors=config.training.best_umap_n_neighbors,
            min_dist=config.training.best_umap_min_dist,
            threshold=config.training.threshold,
            **({"feature_stages": umap_stages} if umap_stages is not None else {}),
        )

    resume_id = peek_swanlab_run_id(args.resume) if args.resume else None
    with SwanLabTracker(
        config,
        resume_id=resume_id,
        run_timestamp=run_timestamp,
    ) as tracker:
        data_log = {}
        for task, loader in train_loaders.items():
            for key, value in loader.dataset.filter_stats.items():
                data_log[f"data/{task}/train/{key}"] = value
        for task, split_loaders in val_loaders.items():
            for split_name, loader in split_loaders.items():
                for key, value in loader.dataset.filter_stats.items():
                    data_log[f"data/{task}/{split_name}/{key}"] = value
        for entity, cache in caches.items():
            data_log[f"cache/{entity}/count"] = cache.manifest["count"]
            data_log[f"cache/{entity}/hidden_dim"] = cache.manifest["hidden_dim"]
            data_log[f"cache/{entity}/max_length"] = cache.manifest["max_length"]
            data_log[f"cache/{entity}/checkpoint_fingerprint"] = cache.manifest[
                "model"
            ]["checkpoint_fingerprint"]
        if online_encoders is not None:
            data_log["input/mode"] = "online_debug"
            data_log["input/esmc_checkpoint_fingerprint"] = (
                online_encoders.esmc.checkpoint_fingerprint
            )
            if "ptcr" in tasks:
                data_log["input/tcr_bert_checkpoint_fingerprint"] = (
                    online_encoders.tcr_bert.checkpoint_fingerprint
                )
        tracker.log(data_log, step=0)
        trainer = trainer_class(
            model,
            config,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            tracker=tracker,
            cache_metadata={
                entity: cache.manifest for entity, cache in caches.items()
            }
            if caches
            else {"online": online_encoders.metadata()},
            online_encoders=online_encoders,
            input_mode="online_debug" if args.no_cache else "cache",
            run_timestamp=run_timestamp,
            best_umap_visualizer=best_umap_visualizer,
            umap_train_loaders=umap_train_loaders,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/final_round": summary["round"],
                **{
                    f"training/best_{task}_score": summary["best_task_scores"][task]
                    for task in tasks
                },
            },
            step=trainer.global_step,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
