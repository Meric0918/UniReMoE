#!/usr/bin/env python3
"""Audit Stage14's four-type train/validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.stage14_config import apply_stage14_overrides, load_stage14_config
from plm_unified.stage8_data import audit_stage8_class_dirs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "plm_stage14.yaml")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    config = apply_stage14_overrides(load_stage14_config(args.config))
    result = audit_stage8_class_dirs(
        config.data_dir(), config.stage14.class_dirs,
        config.stage14.train_filename, config.stage14.val_filename,
        config.model,
        reject_noncanonical_sequences=config.stage14.reject_noncanonical_sequences,
    )
    report_dir = args.report_dir or config.run_dir() / "data_audit"
    result.write_reports(report_dir)
    print(json.dumps({
        "report": result.report,
        "conflict_count": len(result.conflicts),
        "error_count": len(result.errors),
        "dataset_fingerprint": result.dataset_fingerprint,
        "report_dir": str(report_dir.resolve()),
    }, indent=2, sort_keys=True))
    if result.has_conflicts or result.has_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
