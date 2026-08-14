#!/usr/bin/env python3
"""Validate or build the shared Stage-3 residue cache for Stage 10."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plm_unified.stage8_data import audit_stage8_data
from plm_unified.stage10_checkpoint import load_stage3_teacher_source
from plm_unified.stage10_config import (
    apply_stage10_overrides,
    load_stage10_config,
)
from plm_unified.stage10_precompute import precompute_stage10_caches
from plm_unified.trainer import resolve_device, set_global_seed


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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_stage10_overrides(
        load_stage10_config(args.config),
        device=args.device,
        stage3_checkpoint=args.stage3_checkpoint,
    )
    config.validate(require_files=True)
    set_global_seed(config.runtime.seed)
    audit = audit_stage8_data(
        config.train_path(),
        config.val_path(),
        config.model,
        reject_noncanonical_sequences=(
            config.stage10.reject_noncanonical_sequences
        ),
    )
    audit_dir = (
        config.paths.output_root
        / "data_audits"
        / f"fold_{config.runtime.fold}"
    )
    audit.write_reports(audit_dir)
    if audit.has_conflicts or audit.has_errors:
        print(
            "Stage-10 data audit failed; manual repair required. "
            f"Reports: {audit_dir}"
        )
        raise SystemExit(2)
    source = load_stage3_teacher_source(config)
    root = precompute_stage10_caches(
        config,
        source,
        audit,
        device=resolve_device(config.runtime.device),
        overwrite=args.overwrite,
    )
    print(f"Stage-10 shared residue cache validated: {root}")


if __name__ == "__main__":
    main()
