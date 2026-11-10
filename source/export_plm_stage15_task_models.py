#!/usr/bin/env python3
"""Export deployable PH/PT task branches from Stage15 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage15_trainer import export_stage15_task_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir and len(args.checkpoint) != 1:
        raise ValueError("--output-dir requires exactly one checkpoint")
    timestamp = create_artifact_timestamp()
    for checkpoint in args.checkpoint:
        outputs = export_stage15_task_checkpoints(
            checkpoint,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            artifact_timestamp=timestamp,
        )
        print(f"Source: {checkpoint.resolve()}")
        print(f"  pHLA: {outputs['phla']}")
        print(f"  pTCR: {outputs['ptcr']}")


if __name__ == "__main__":
    main()
