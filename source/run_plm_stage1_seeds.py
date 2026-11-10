#!/usr/bin/env python3
"""Run the confirmed fold across the configured three random seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.config import apply_overrides, load_config
from plm_unified.metrics import task_score
from plm_unified.trainer import (
    BEST_ROUND_CHECKPOINT_FILENAME,
    _macro_metric,
    _macro_validation_score,
    load_training_checkpoint,
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
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    return parser.parse_args(argv)


def main(
    argv=None,
    *,
    default_config="plm_stage1.yaml",
    config_loader=load_config,
    training_script_name="train_plm_stage1.py",
    tasks=("phla", "ptcr"),
) -> None:
    args = parse_args(argv, default_config=default_config)
    config = apply_overrides(config_loader(args.config), fold=args.fold)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        config.validate()
    seeds = args.seeds or config.training.seeds
    training_script = Path(__file__).with_name(training_script_name)
    batch_timestamp = create_artifact_timestamp()
    checkpoints = []
    for seed in seeds:
        run_timestamp = create_artifact_timestamp()
        command = [
            sys.executable,
            str(training_script),
            "--config",
            str(Path(args.config).resolve()),
            "--fold",
            str(config.training.fold),
            "--seed",
            str(seed),
            "--run-timestamp",
            run_timestamp,
        ]
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.device:
            command.extend(["--device", args.device])
        if args.cache_root:
            command.extend(["--cache-root", str(args.cache_root.expanduser().resolve())])
        if args.swanlab_mode:
            command.extend(["--swanlab-mode", args.swanlab_mode])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        run_dir = (
            config.paths.output_root
            / f"fold_{config.training.fold}"
            / f"seed_{seed}"
        )
        checkpoints.append(
            {
                **{
                    task: run_dir / timestamped_filename(f"best_{task}.pt", run_timestamp)
                    for task in tasks
                },
                "joint": run_dir
                / "best_round"
                / timestamped_filename(
                    BEST_ROUND_CHECKPOINT_FILENAME,
                    run_timestamp,
                ),
            }
        )
    summary = aggregate_checkpoints(checkpoints, tasks=tasks)
    summary_path = (
        config.paths.output_root
        / f"fold_{config.training.fold}"
        / timestamped_filename("seed_summary.json", batch_timestamp)
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Seed summary: {summary_path}")


def _best_checkpoint_score(checkpoint: dict) -> float:
    saved_score = checkpoint.get("best_score")
    if saved_score is not None:
        return float(saved_score)
    metrics = checkpoint["validation_metrics"]
    if "auroc" in metrics:
        return task_score(metrics)
    return _macro_validation_score(metrics)


def _validation_metric(metrics: dict, name: str) -> float:
    if name in metrics:
        return float(metrics[name])
    return _macro_metric(metrics, name)


def aggregate_checkpoints(checkpoints: list[dict[str, Path]], *, tasks=("phla", "ptcr")) -> dict:
    runs = []
    for task_paths in checkpoints:
        task_checkpoints = {
            task: load_training_checkpoint(task_paths[task], map_location="cpu")
            for task, path in task_paths.items()
            if task in tasks
        }
        if "joint" in task_paths:
            joint_checkpoint = load_training_checkpoint(
                task_paths["joint"],
                map_location="cpu",
            )
        else:
            joint_checkpoint = {
                "best_round": 0,
                "round_score": statistics.fmean(
                    [
                        float(checkpoint.get("best_score", 0.0))
                        for checkpoint in task_checkpoints.values()
                    ]
                ),
            }
        seeds = {
            checkpoint["config"]["training"]["seed"]
            for checkpoint in task_checkpoints.values()
        }
        if len(seeds) != 1:
            raise ValueError(f"Task checkpoints have different seeds: {seeds}")
        runs.append(
            {
                "paths": {
                    task: str(path) for task, path in task_paths.items()
                },
                "seed": seeds.pop(),
                "best_scores": {
                    task: _best_checkpoint_score(checkpoint)
                    for task, checkpoint in task_checkpoints.items()
                },
                "metrics": {
                    task: checkpoint["validation_metrics"]
                    for task, checkpoint in task_checkpoints.items()
                },
                "joint": {
                    "path": str(task_paths["joint"]),
                    "best_round": int(
                        joint_checkpoint.get(
                            "best_round",
                            joint_checkpoint.get("current_round", 0),
                        )
                    ),
                    "round_score": float(
                        joint_checkpoint.get(
                            "round_score",
                            joint_checkpoint.get("best_round_score", float("nan")),
                        )
                    ),
                },
            }
        )

    aggregate = {}
    metric_names = ("auroc", "aupr", "accuracy", "mcc", "f1")
    for task in tasks:
        aggregate[task] = {}
        for metric in metric_names:
            values = [_validation_metric(run["metrics"][task], metric) for run in runs]
            aggregate[task][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        score_values = [float(run["best_scores"][task]) for run in runs]
        aggregate[task]["best_score"] = {
            "mean": statistics.fmean(score_values),
            "std": statistics.stdev(score_values) if len(score_values) > 1 else 0.0,
            }
    joint_scores = [run["joint"]["round_score"] for run in runs]
    aggregate["best_round"] = {
        "round_score": {
            "mean": statistics.fmean(joint_scores),
            "std": statistics.stdev(joint_scores)
            if len(joint_scores) > 1
            else 0.0,
        },
        "round": {
            "mean": statistics.fmean(
                [run["joint"]["best_round"] for run in runs]
            ),
            "std": statistics.stdev(
                [run["joint"]["best_round"] for run in runs]
            )
            if len(runs) > 1
            else 0.0,
        },
    }
    return {"runs": runs, "aggregate": aggregate}


if __name__ == "__main__":
    main()
