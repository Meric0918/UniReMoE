#!/usr/bin/env python3
"""Train Stage15 with the configured seeds and aggregate selected metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.stage15_config import apply_stage15_overrides, load_stage15_config
from plm_unified.stage15_trainer import STAGE15_BEST_ROUND_CHECKPOINT_FILENAME, STAGE15_TASK_CHECKPOINT_FILENAMES
from plm_unified.trainer import load_training_checkpoint


def aggregate_checkpoints(checkpoints: list[dict[str, Path]]) -> dict:
    def macro_metric(validation_metrics: dict, name: str) -> float:
        if name in validation_metrics:
            return float(validation_metrics[name])
        values = [float(metrics[name]) for metrics in validation_metrics.values()]
        if not values:
            raise ValueError(f"Stage15 checkpoint has no validation metric '{name}'")
        return statistics.fmean(values)

    runs = []
    for paths in checkpoints:
        task_states = {
            task: load_training_checkpoint(paths[task], map_location="cpu")
            for task in ("phla", "ptcr")
        }
        joint = load_training_checkpoint(paths["joint"], map_location="cpu")
        seeds = {state["config"]["training"]["seed"] for state in task_states.values()}
        if len(seeds) != 1:
            raise ValueError(f"Stage15 task checkpoints have different seeds: {seeds}")
        runs.append({
            "seed": seeds.pop(),
            "paths": {key: str(value) for key, value in paths.items()},
            "best_scores": {
                task: float(
                    state["best_score"]
                    if state.get("best_score") is not None
                    else statistics.fmean((
                        macro_metric(state["validation_metrics"], "auroc"),
                        macro_metric(state["validation_metrics"], "aupr"),
                    ))
                )
                for task, state in task_states.items()
            },
            "metrics": {task: state["validation_metrics"] for task, state in task_states.items()},
            "joint": {
                "best_round": int(joint.get("best_round", 0)),
                "round_score": float(joint.get("round_score", float("nan"))),
            },
        })
    aggregate = {}
    for task in ("phla", "ptcr"):
        aggregate[task] = {}
        for metric in ("auroc", "aupr", "accuracy", "mcc", "f1"):
            values = [
                macro_metric(run["metrics"][task], metric) for run in runs
            ]
            aggregate[task][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        values = [run["best_scores"][task] for run in runs]
        aggregate[task]["best_score"] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    joint_values = [run["joint"]["round_score"] for run in runs]
    return {
        "stage": "stage15", "architecture": "shared_hyper_pmt_pair",
        "runs": runs, "aggregate": aggregate,
        "joint_round_score": {
            "mean": statistics.fmean(joint_values),
            "std": statistics.stdev(joint_values) if len(joint_values) > 1 else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "plm_stage15.yaml"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    args = parser.parse_args()
    config = apply_stage15_overrides(load_stage15_config(args.config), fold=args.fold)
    seeds = args.seeds or config.training.seeds
    batch_timestamp = create_artifact_timestamp()
    checkpoints = []
    for seed in seeds:
        timestamp = create_artifact_timestamp()
        command = [
            sys.executable, str(Path(__file__).with_name("train_plm_stage15.py")),
            "--config", str(Path(args.config).resolve()), "--fold", str(config.training.fold),
            "--seed", str(seed), "--run-timestamp", timestamp,
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.swanlab_mode:
            command.extend(["--swanlab-mode", args.swanlab_mode])
        subprocess.run(command, check=True)
        run_dir = config.paths.output_root / f"fold_{config.training.fold}" / f"seed_{seed}"
        checkpoints.append({
            task: run_dir / timestamped_filename(filename, timestamp)
            for task, filename in STAGE15_TASK_CHECKPOINT_FILENAMES.items()
        } | {
            "joint": run_dir / "best_round" / timestamped_filename(
                STAGE15_BEST_ROUND_CHECKPOINT_FILENAME, timestamp
            )
        })
    summary = aggregate_checkpoints(checkpoints)
    path = config.paths.output_root / f"fold_{config.training.fold}" / timestamped_filename("stage15_seed_summary.json", batch_timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Seed summary: {path}")


if __name__ == "__main__":
    main()
