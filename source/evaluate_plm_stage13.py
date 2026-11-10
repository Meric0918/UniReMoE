#!/usr/bin/env python3
"""Evaluate a Stage13 checkpoint on configured validation or test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from plm_unified.stage13_config import apply_stage13_overrides, load_stage13_config
from plm_unified.stage13_data import make_stage13_loaders
from plm_unified.stage13_graph import GraphArtifact
from plm_unified.stage13_model import Stage13BindingModel
from plm_unified.stage13_trainer import Stage13Trainer, load_stage13_checkpoint
from plm_unified.stage13_tracking import Stage13Tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage13.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = apply_stage13_overrides(
        load_stage13_config(args.config), device=args.device, num_workers=args.num_workers
    )
    checkpoint = load_stage13_checkpoint(args.checkpoint, map_location="cpu")
    graph = GraphArtifact.load(config.graph_dir())
    current_config = config.to_dict()
    if checkpoint.get("config", {}).get("model") != current_config["model"]:
        raise ValueError("Stage13 checkpoint model configuration mismatch")
    if checkpoint.get("config", {}).get("graph") != current_config["graph"]:
        raise ValueError("Stage13 checkpoint graph configuration mismatch")
    from plm_unified.stage13_graph import file_sha256

    if checkpoint.get("graph_manifest_sha256") != file_sha256(graph.root / "manifest.json"):
        raise ValueError("Stage13 checkpoint graph manifest mismatch")
    train_loaders, val_loaders, test_loaders = make_stage13_loaders(
        config, graph, limit_validation=args.limit, include_test=True
    )
    model = Stage13BindingModel(config.model, config.graph)
    model.load_state_dict(checkpoint["model"], strict=True)
    with Stage13Tracker(config, run_timestamp=checkpoint.get("run_timestamp", "20260901_000000_000000")) as tracker:
        trainer = Stage13Trainer(model, config, graph, train_loaders, val_loaders, tracker)
        loaders = val_loaders if args.split == "validation" else {
            "phla": {
                key.removeprefix("phla/"): value
                for key, value in test_loaders.items()
                if key.startswith("phla/")
            },
            "ptcr": {
                key.removeprefix("ptcr/"): value
                for key, value in test_loaders.items()
                if key.startswith("ptcr/")
            },
        }
        result: dict[str, dict[str, dict]] = {"phla": {}, "ptcr": {}}
        for task in ("phla", "ptcr"):
            for split, loader in loaders[task].items():
                result[task][split] = trainer.evaluate(task, loader).to_dict()
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "results": result,
    }
    output = args.output or args.checkpoint.with_name(args.checkpoint.stem + f"_{args.split}.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
