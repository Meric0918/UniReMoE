#!/usr/bin/env python3
"""Audit and deterministically split paired Stage-9 PHT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.stage9_config import apply_stage9_overrides, load_stage9_config
from plm_unified.stage9_pipeline import build_stage9_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage9.yaml",
    )
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_stage9_overrides(
        load_stage9_config(args.config), pair_csv=args.pairs_csv
    )
    result = build_stage9_audit(config)
    report = args.report or (
        config.paths.output_root
        / "data_audits"
        / f"stage9_pairs_fold{config.runtime.fold}_seed{config.stage9.split_seed}.json"
    )
    result.write_report(report)
    report_summary = {
        key: value
        for key, value in result.report.items()
        if key != "split_manifest"
    }
    print(
        json.dumps(
            {
                "report": report_summary,
                "error_count": len(result.errors),
                "dataset_fingerprint": result.dataset_fingerprint,
                "report_path": str(report.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result.has_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
