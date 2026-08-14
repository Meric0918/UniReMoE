#!/usr/bin/env python3
"""Merge a completed Stage-10 Student with its Stage-3 encoders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plm_unified.artifacts import validate_artifact_timestamp
from plm_unified.stage10_checkpoint import (
    export_stage10_full_checkpoint,
    load_stage3_teacher_source,
    load_torch_checkpoint,
    validate_stage10_lean_checkpoint,
)
from plm_unified.stage10_config import (
    apply_stage10_overrides,
    load_stage10_config,
)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_stage10_overrides(
        load_stage10_config(args.config),
        stage3_checkpoint=args.stage3_checkpoint,
    )
    source = load_stage3_teacher_source(config)
    lean = load_torch_checkpoint(args.checkpoint, mmap=False)
    validate_stage10_lean_checkpoint(lean, config, source.sha256)
    if lean.get("artifact_role") != "best" or lean.get("phase") != "complete":
        raise ValueError(
            "Stage-10 full export requires a completed best checkpoint"
        )
    timestamp = validate_artifact_timestamp(str(lean.get("run_timestamp")))
    output = args.output or args.checkpoint.with_name(
        f"stage10_full_{timestamp}.pt"
    )
    export_stage10_full_checkpoint(source, lean, output)
    print(f"Stage-10 full checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
