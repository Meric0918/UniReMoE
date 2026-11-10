#!/usr/bin/env python3
"""Reanalyse one or more aligned Stage-7 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage7_analysis import (
    analyse_stage7_rows,
    average_aligned_runs,
    render_stage7_report,
)
from plm_unified.stage7_inference import SCORE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--permutation-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(run_dir: Path, manifest: dict) -> list[dict]:
    rows = []
    for split in manifest["splits"]:
        path = Path(manifest["artifacts"]["predictions"][split])
        if not path.is_file():
            path = run_dir / f"stage7_counterfactual_{split}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["label"] = int(row["label"]) if row["label"] else None
                row["source_row"] = (
                    int(row["source_row"]) if row["source_row"] else None
                )
                for column in SCORE_COLUMNS:
                    row[column] = float(row[column])
                rows.append(row)
    return rows


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    manifests = [_read_json(path / "stage7_manifest.json") for path in run_dirs]
    if any(manifest.get("stage") != "stage7" for manifest in manifests):
        raise ValueError("Every --run-dir must contain a Stage-7 manifest")
    if any(
        manifest.get("comparison") != "stage3_vs_stage1"
        or int(manifest.get("format_version", 0)) < 2
        for manifest in manifests
    ):
        raise ValueError(
            "Stage-7 analysis requires format-version-2 Stage3-vs-Stage1 runs"
        )
    folds = {manifest["fold"] for manifest in manifests}
    if len(folds) != 1:
        raise ValueError("Multi-run Stage-7 analysis requires the same fold")
    seeds = [manifest["seed"] for manifest in manifests]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Multi-run Stage-7 analysis requires distinct seeds")
    runs = [_read_rows(path, manifest) for path, manifest in zip(run_dirs, manifests)]
    rows = average_aligned_runs(runs)
    first_config = manifests[0]["config"]
    bootstrap_samples = (
        args.bootstrap_samples
        or int(first_config["stage7"]["bootstrap_samples"])
    )
    permutation_samples = (
        args.permutation_samples
        or int(first_config["stage7"]["permutation_samples"])
    )
    seed = args.seed if args.seed is not None else int(manifests[0]["seed"])
    if bootstrap_samples < 1 or permutation_samples < 1:
        raise ValueError("Analysis sample counts must be positive")
    audits = [_read_json(path / "stage7_data_audit.json") for path in run_dirs]
    summary = analyse_stage7_rows(
        rows,
        audits[0],
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
        run_count=len(runs),
    )
    aggregate_manifest = {
        "format_version": 2,
        "stage": "stage7",
        "comparison": "stage3_vs_stage1",
        "experiment_type": "stage7_aligned_score_average",
        "fold": next(iter(folds)),
        "seed": seeds,
        "source_run_dirs": [str(path) for path in run_dirs],
        "run_count": len(runs),
    }
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    elif len(run_dirs) == 1:
        output_dir = run_dirs[0]
    else:
        output_dir = run_dirs[0].parents[2] / f"aggregate_{create_artifact_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage7_summary.json"
    report_path = output_dir / "stage7_report.md"
    manifest_path = output_dir / "stage7_aggregate_manifest.json"
    _write_json(summary_path, summary)
    report_path.write_text(
        render_stage7_report(summary, aggregate_manifest), encoding="utf-8"
    )
    _write_json(manifest_path, aggregate_manifest)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "manifest": str(manifest_path),
                "conclusion": summary.get("conclusion"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
