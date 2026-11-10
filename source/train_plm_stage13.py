#!/usr/bin/env python3
"""Train or resume the isolated Stage13 GraphSAGE-fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, validate_artifact_timestamp
from plm_unified.stage13_config import apply_stage13_overrides, load_stage13_config
from plm_unified.stage13_data import make_stage13_loaders
from plm_unified.stage13_graph import GraphArtifact
from plm_unified.stage13_model import Stage13BindingModel
from plm_unified.stage13_trainer import (
    Stage13Trainer,
    load_stage13_checkpoint,
    set_stage13_seed,
)
from plm_unified.stage13_tracking import Stage13Tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage13.yaml",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-train", type=int, help="Debug-only rows per training task")
    parser.add_argument("--limit-validation", type=int, help="Debug-only rows per validation split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.run_timestamp:
        raise ValueError("--resume derives the run timestamp; omit --run-timestamp")
    for name in ("limit_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    resume_checkpoint = load_stage13_checkpoint(args.resume, map_location="cpu") if args.resume else None
    saved_runtime = (resume_checkpoint or {}).get("config", {}).get("runtime", {})
    if args.fold is None and saved_runtime.get("fold") is not None:
        args.fold = int(saved_runtime["fold"])
    if args.seed is None and saved_runtime.get("seed") is not None:
        args.seed = int(saved_runtime["seed"])
    config = apply_stage13_overrides(
        load_stage13_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
    )
    timestamp = validate_artifact_timestamp(
        args.run_timestamp
        or (resume_checkpoint or {}).get("run_timestamp")
        or create_artifact_timestamp()
    )
    set_stage13_seed(config.runtime.seed)
    graph = GraphArtifact.load(config.graph_dir())
    train_loaders, val_loaders, _ = make_stage13_loaders(
        config,
        graph,
        limit_train=args.limit_train,
        limit_validation=args.limit_validation,
    )
    model = Stage13BindingModel(config.model, config.graph)
    with Stage13Tracker(config, run_timestamp=timestamp) as tracker:
        trainer = Stage13Trainer(
            model,
            config,
            graph,
            train_loaders,
            val_loaders,
            tracker,
            run_timestamp=timestamp,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
    summary_path = config.run_dir() / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
