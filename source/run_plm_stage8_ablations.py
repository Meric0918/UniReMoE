#!/usr/bin/env python3
"""Run one shared Stage8A warmup and four Stage8B V2 conditions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage8_config import Stage8Condition, load_stage8_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage8.yaml",
    )
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode", choices=("online", "local", "offline", "disabled")
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int)
    return parser.parse_args(argv)


def _append(command: list[str], option: str, value) -> None:
    if value is not None:
        command.extend((option, str(value)))


def build_ablation_commands(
    *,
    config_path: Path,
    stage1_checkpoint: Path,
    warmup_checkpoint: Path,
    run_timestamp: str,
    python_executable: Path,
    source_dir: Path,
    device: str | None,
    num_workers: int | None,
    swanlab_mode: str | None,
    limit_per_type: int | None,
) -> list[list[str]]:
    """Build the exact one-warmup/four-condition subprocess contract."""

    common = ["--config", str(config_path), "--run-timestamp", run_timestamp]
    for option, value in (
        ("--device", device),
        ("--num-workers", num_workers),
        ("--swanlab-mode", swanlab_mode),
        ("--limit-per-type", limit_per_type),
    ):
        _append(common, option, value)
    train_script = str(source_dir / "train_plm_stage8.py")
    commands = [[
        str(python_executable), train_script, *common,
        "--warmup-only", "--stage1-checkpoint", str(stage1_checkpoint),
    ]]
    for condition in Stage8Condition:
        commands.append([
            str(python_executable), train_script, *common,
            "--condition", condition.value,
            "--warmup-checkpoint", str(warmup_checkpoint),
        ])
    return commands


def main() -> None:
    args = parse_args()
    timestamp = args.run_timestamp or create_artifact_timestamp()
    config = load_stage8_config(args.config)
    warmup = (
        config.paths.output_root / f"fold_{config.runtime.fold}"
        / f"seed_{config.runtime.seed}" / "warmup"
        / f"stage8a_best_pht_{timestamp}.pt"
    ).resolve()
    commands = build_ablation_commands(
        config_path=args.config,
        stage1_checkpoint=args.stage1_checkpoint,
        warmup_checkpoint=warmup,
        run_timestamp=timestamp,
        python_executable=Path(sys.executable),
        source_dir=Path(__file__).resolve().parent,
        device=args.device,
        num_workers=args.num_workers,
        swanlab_mode=args.swanlab_mode,
        limit_per_type=args.limit_per_type,
    )
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
