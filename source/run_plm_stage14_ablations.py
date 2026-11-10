#!/usr/bin/env python3
"""Run one shared Stage14A warmup and the baseline/full Stage14B conditions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage14_checkpoint import load_stage14_checkpoint
from plm_unified.stage14_config import (
    STAGE14_TRAINING_CONDITIONS,
    Stage14Condition,
    load_stage14_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage14.yaml",
    )
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--resume-baseline", type=Path)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--run-timestamp")
    parser.add_argument("--limit-per-type", type=int)
    args = parser.parse_args(argv)
    if args.stage1_checkpoint is None and args.warmup_checkpoint is None and args.resume_baseline is None:
        parser.error("ablation runs require --stage1-checkpoint, --warmup-checkpoint, or --resume-baseline")
    if args.resume_baseline is not None and args.limit_per_type is not None:
        parser.error("--resume-baseline cannot be combined with --limit-per-type")
    return args


def _append(command: list[str], option: str, value) -> None:
    if value is not None:
        command.extend((option, str(value)))


def build_condition_commands(
    *, config_path: Path, warmup_checkpoint: Path, run_timestamp: str,
    python_executable: Path, source_dir: Path, device: str | None,
    num_workers: int | None, swanlab_mode: str | None,
    limit_per_type: int | None, resume_baseline: Path | None = None,
) -> list[list[str]]:
    common = ["--config", str(config_path), "--run-timestamp", run_timestamp]
    for option, value in (("--device", device), ("--num-workers", num_workers),
                          ("--swanlab-mode", swanlab_mode), ("--limit-per-type", limit_per_type)):
        _append(common, option, value)
    train_script = str(source_dir / "train_plm_stage14.py")
    commands: list[list[str]] = []
    for condition in STAGE14_TRAINING_CONDITIONS:
        if condition == Stage14Condition.BASELINE and resume_baseline is not None:
            resume_common = ["--config", str(config_path)]
            for option, value in (("--device", device), ("--num-workers", num_workers),
                                  ("--swanlab-mode", swanlab_mode)):
                _append(resume_common, option, value)
            commands.append([str(python_executable), train_script, *resume_common,
                             "--resume", str(resume_baseline)])
        else:
            commands.append([str(python_executable), train_script, *common,
                             "--condition", condition.value,
                             "--warmup-checkpoint", str(warmup_checkpoint)])
    return commands


def _terminate(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        process.wait()


def run_parallel(commands: Sequence[Sequence[str]]) -> None:
    processes: list[subprocess.Popen] = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(list(command)))
        return_codes = [process.wait() for process in processes]
    except BaseException:
        _terminate(processes)
        raise
    for command, return_code in zip(commands, return_codes):
        if return_code:
            raise subprocess.CalledProcessError(return_code, list(command))


def main() -> None:
    args = parse_args()
    config = load_stage14_config(args.config)
    resume = None
    if args.resume_baseline is not None:
        resume = load_stage14_checkpoint(args.resume_baseline.expanduser().resolve(), artifact_role="last")
        if resume.get("condition") != Stage14Condition.BASELINE.value:
            raise ValueError("--resume-baseline must reference the baseline condition")
        if args.run_timestamp is not None and args.run_timestamp != resume["run_timestamp"]:
            raise ValueError("--run-timestamp does not match --resume-baseline")
        args.run_timestamp = resume["run_timestamp"]
        if args.warmup_checkpoint is None:
            args.warmup_checkpoint = Path(resume["source_stage14a"]["path"])
    timestamp = args.run_timestamp
    if args.warmup_checkpoint is not None:
        warmup = args.warmup_checkpoint.expanduser().resolve()
        metadata = load_stage14_checkpoint(warmup, artifact_role="warmup")
        if timestamp is None:
            timestamp = metadata["run_timestamp"]
        elif timestamp != metadata["run_timestamp"]:
            raise ValueError("--run-timestamp does not match --warmup-checkpoint")
    else:
        timestamp = timestamp or create_artifact_timestamp()
        warmup = (
            config.run_dir() / "warmup" / f"stage14a_best_pht_{timestamp}.pt"
        ).resolve()
        command = [
            str(Path(sys.executable)), str(Path(__file__).resolve().parent / "train_plm_stage14.py"),
            "--config", str(args.config), "--run-timestamp", timestamp,
            "--warmup-only", "--stage1-checkpoint", str(args.stage1_checkpoint),
        ]
        for option, value in (("--device", args.device), ("--num-workers", args.num_workers),
                              ("--swanlab-mode", args.swanlab_mode), ("--limit-per-type", args.limit_per_type)):
            _append(command, option, value)
        subprocess.run(command, check=True)
    commands = build_condition_commands(
        config_path=args.config, warmup_checkpoint=warmup, run_timestamp=timestamp,
        python_executable=Path(sys.executable), source_dir=Path(__file__).resolve().parent,
        device=args.device, num_workers=args.num_workers, swanlab_mode=args.swanlab_mode,
        limit_per_type=args.limit_per_type,
        resume_baseline=args.resume_baseline.expanduser().resolve() if args.resume_baseline else None,
    )
    if args.sequential:
        for command in commands:
            subprocess.run(command, check=True)
    else:
        run_parallel(commands)


if __name__ == "__main__":
    main()
