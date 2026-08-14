#!/usr/bin/env python3
"""Audit Stage-10 four-type data without modifying source CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from plm_unified.stage8_data import audit_stage8_data
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
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_stage10_overrides(load_stage10_config(args.config))
    result = audit_stage8_data(
        config.train_path(),
        config.val_path(),
        config.model,
        reject_noncanonical_sequences=(
            config.stage10.reject_noncanonical_sequences
        ),
    )
    report_dir = args.report_dir or (
        config.paths.output_root
        / "data_audits"
        / f"fold_{config.runtime.fold}"
    )
    result.write_reports(report_dir)
    print(
        json.dumps(
            {
                "report": result.report,
                "conflict_count": len(result.conflicts),
                "error_count": len(result.errors),
                "dataset_fingerprint": result.dataset_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Stage-10 audit reports: {report_dir.resolve()}")
    if result.has_conflicts or result.has_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
