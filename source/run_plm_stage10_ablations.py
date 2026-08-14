#!/usr/bin/env python3
"""Run all Stage-10 conditions sequentially with identical inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from plm_unified.stage10_config import Stage10Condition


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/plm_stage10.yaml"
        ),
    )
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument("--limit-per-type", type=int)
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args(argv)


def _append(command: list[str], option: str, value) -> None:
    if value is not None:
        command.extend((option, str(value)))


def build_commands(
    args: argparse.Namespace,
    *,
    python_executable: str | Path,
    source_dir: str | Path,
) -> list[list[str]]:
    source_dir = Path(source_dir)
    common = ["--config", str(args.config)]
    for option, value in (
        ("--stage3-checkpoint", args.stage3_checkpoint),
        ("--device", args.device),
    ):
        _append(common, option, value)
    commands: list[list[str]] = []
    if args.precompute:
        cache_command = [
            str(python_executable),
            str(source_dir / "precompute_plm_stage10_embeddings.py"),
            *common,
        ]
        if args.overwrite_cache:
            cache_command.append("--overwrite")
        commands.append(cache_command)
    for condition in Stage10Condition:
        command = [
            str(python_executable),
            str(source_dir / "train_plm_stage10.py"),
            *common,
            "--condition",
            condition.value,
        ]
        for option, value in (
            ("--num-workers", args.num_workers),
            ("--swanlab-mode", args.swanlab_mode),
            ("--limit-per-type", args.limit_per_type),
        ):
            _append(command, option, value)
        commands.append(command)
    return commands


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    commands = build_commands(
        args,
        python_executable=sys.executable,
        source_dir=Path(__file__).resolve().parent,
    )
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
