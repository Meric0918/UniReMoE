#!/usr/bin/env python3
"""Audit Stage PHT four-type PHT data without modifying source CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.stage8_data import audit_stage8_class_dirs
from plm_unified.stage_PHT_config import load_stage_PHT_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/plm_stage_PHT.yaml"
    )
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    config = load_stage_PHT_config(args.config)
    values = config.stage_PHT
    result = audit_stage8_class_dirs(
        config.data_dir(), values.class_dirs,
        values.train_filename, values.val_filename, config.model,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
    )
    destination = args.report_dir or (
        config.paths.output_root / "data_audits" / f"fold_{config.runtime.fold}"
    )
    result.write_reports(destination)
    print(json.dumps({
        "report": result.report,
        "conflict_count": len(result.conflicts),
        "error_count": len(result.errors),
        "dataset_fingerprint": result.dataset_fingerprint,
    }, indent=2, sort_keys=True))
    print(f"Stage PHT audit reports: {destination.resolve()}")
    if result.has_conflicts or result.has_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
