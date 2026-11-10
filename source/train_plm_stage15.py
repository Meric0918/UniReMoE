#!/usr/bin/env python3
"""Train Stage15: Stage1 PH/PT lifecycle with Shared Hyper-PMT fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, validate_artifact_timestamp
from plm_unified.encoders import OnlineEncoderBundle
from plm_unified.stage15_config import apply_stage15_overrides, load_stage15_config
from plm_unified.stage15_model import Stage15BindingModel
from plm_unified.stage15_trainer import Stage15Trainer
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import load_training_checkpoint, peek_swanlab_run_id, resolve_device, set_global_seed
try:
    from train_plm_stage1 import make_loaders, open_caches
except ModuleNotFoundError:  # pragma: no cover - package import fallback.
    from source.train_plm_stage1 import make_loaders, open_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "plm_stage15.yaml"),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-timestamp")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int)
    umap = parser.add_mutually_exclusive_group()
    umap.add_argument("--best-umap", dest="best_umap", action="store_true")
    umap.add_argument("--no-best-umap", dest="best_umap", action="store_false")
    parser.set_defaults(best_umap=None)
    parser.add_argument("--best-umap-output-dir", type=Path)
    parser.add_argument("--best-umap-max-samples-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_timestamp = None
    if args.resume and not args.run_timestamp:
        resume_timestamp = load_training_checkpoint(
            args.resume, map_location="cpu"
        ).get("run_timestamp")
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp or resume_timestamp or create_artifact_timestamp()
    )
    config = apply_stage15_overrides(
        load_stage15_config(args.config),
        fold=args.fold,
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    if args.no_cache and args.limit is None:
        raise ValueError("--no-cache is a debug mode and requires --limit")
    set_global_seed(config.training.seed)
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
        )
        if online_encoders.esmc.hidden_size != config.model.peptide_input_dim:
            raise ValueError("Online ESM-C hidden size does not match model config")
        if online_encoders.tcr_bert.hidden_size != config.model.tcr_input_dim:
            raise ValueError("Online TCR-BERT hidden size does not match model config")
        caches = {}
    else:
        caches = open_caches(config)
    train_loaders, val_loaders = make_loaders(
        config, caches, args.limit, online=args.no_cache
    )
    model = Stage15BindingModel(config.model, config.fusion)

    best_umap_visualizer = None
    umap_train_loaders = {}
    best_umap_enabled = (
        config.training.best_umap_enabled
        if args.best_umap is None
        else args.best_umap
    )
    if best_umap_enabled:
        from plm_unified.data import make_sequential_dataloader
        from plm_unified.stage1_umap import Stage1BestUMAPVisualizer

        umap_train_loaders = {
            task: make_sequential_dataloader(loader)
            for task, loader in train_loaders.items()
        }
        output_dir = (
            args.best_umap_output_dir.expanduser().resolve()
            if args.best_umap_output_dir
            else config.paths.project_root / "figures" / "stage15"
            / "best_model_umap" / f"fold_{config.training.fold}"
            / f"seed_{config.training.seed}" / run_timestamp
        )
        best_umap_visualizer = Stage1BestUMAPVisualizer(
            output_dir,
            val_loaders,
            train_loaders=umap_train_loaders,
            max_samples_per_split=(
                args.best_umap_max_samples_per_split
                or config.training.best_umap_max_samples_per_split
            ),
            random_state=config.training.best_umap_random_state,
            n_neighbors=config.training.best_umap_n_neighbors,
            min_dist=config.training.best_umap_min_dist,
            threshold=config.training.threshold,
        )

    resume_id = peek_swanlab_run_id(args.resume) if args.resume else None
    with SwanLabTracker(
        config, resume_id=resume_id, run_timestamp=run_timestamp
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
            data_log[f"cache/{entity}/checkpoint_fingerprint"] = cache.manifest["model"]["checkpoint_fingerprint"]
        if online_encoders is not None:
            data_log["input/mode"] = "online_debug"
            data_log["input/esmc_checkpoint_fingerprint"] = online_encoders.esmc.checkpoint_fingerprint
            data_log["input/tcr_bert_checkpoint_fingerprint"] = online_encoders.tcr_bert.checkpoint_fingerprint
        tracker.log(data_log, step=0)
        trainer = Stage15Trainer(
            model,
            config,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            tracker=tracker,
            cache_metadata=(
                {entity: cache.manifest for entity, cache in caches.items()}
                if caches else {"online": online_encoders.metadata()}
            ),
            online_encoders=online_encoders,
            input_mode="online_debug" if args.no_cache else "cache",
            run_timestamp=run_timestamp,
            best_umap_visualizer=best_umap_visualizer,
            umap_train_loaders=umap_train_loaders,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log({
            "training/complete": 1,
            "training/final_round": summary["round"],
            "training/best_phla_score": summary["best_task_scores"]["phla"],
            "training/best_ptcr_score": summary["best_task_scores"]["ptcr"],
        }, step=trainer.global_step)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
