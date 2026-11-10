#!/usr/bin/env python3
"""Build the cross-condition Stage-8 benchmark comparison Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from source.evaluate_plm_stage8_benchmarks import make_comparison_markdown
except ModuleNotFoundError:  # pragma: no cover - direct ``python source/...`` use
    from evaluate_plm_stage8_benchmarks import make_comparison_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        action="append",
        required=True,
        metavar="CONDITION=JSON",
        help="A per-condition evaluator JSON, repeat once for each model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected CONDITION=JSON, got {value!r}")
    condition, path = value.split("=", 1)
    condition = condition.strip()
    if not condition or not path.strip():
        raise ValueError(f"Expected non-empty CONDITION and JSON path: {value!r}")
    return condition, Path(path).expanduser().resolve()


def main() -> None:
    args = parse_args()
    reports = {}
    for value in args.input_json:
        condition, path = _parse_input(value)
        if condition in reports:
            raise ValueError(f"Duplicate condition: {condition}")
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("condition") not in {None, condition}:
            raise ValueError(
                f"Condition mismatch for {path}: "
                f"JSON has {report.get('condition')!r}, CLI has {condition!r}"
            )
        reports[condition] = report
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        make_comparison_markdown(reports, output) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "conditions": list(reports)}))


if __name__ == "__main__":
    main()
