#!/usr/bin/env python3
"""Audit Stage-8 four-type train/validation data without modifying source CSVs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The audit entrypoint can be launched after a process or shell has selected
# Intel MKL while the environment provides GNU OpenMP (libgomp).  Normalize
# that incompatible combination before importing numpy through stage8_data.
if os.environ.get("MKL_THREADING_LAYER") == "INTEL":
    os.environ["MKL_THREADING_LAYER"] = "GNU"

from plm_unified.stage8_config import apply_stage8_overrides, load_stage8_config
from plm_unified.stage8_data import audit_stage8_class_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/plm_stage8.yaml")
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_stage8_overrides(load_stage8_config(args.config))
    result = audit_stage8_class_dirs(
        config.data_dir(), config.stage8.class_dirs,
        config.stage8.train_filename, config.stage8.val_filename,
        config.model,
        reject_noncanonical_sequences=config.stage8.reject_noncanonical_sequences,
    )
    report_dir = args.report_dir or (
        config.paths.output_root / "data_audits" / f"fold_{config.runtime.fold}"
    )
    result.write_reports(report_dir)
    print(json.dumps({
        "report": result.report,
        "conflict_count": len(result.conflicts),
        "error_count": len(result.errors),
        "dataset_fingerprint": result.dataset_fingerprint,
    }, indent=2, sort_keys=True))
    print(f"Stage-8 audit reports: {report_dir.resolve()}")
    if result.has_conflicts or result.has_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
