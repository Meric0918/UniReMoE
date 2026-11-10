#!/usr/bin/env python3
"""Audit Stage11 training data without loading PLMs or reading test labels."""

import json

from plm_unified.stage11_cli import common_parser, config_from_args
from plm_unified.stage11_data import prepare_stage11_data


def main(argv=None):
    parser = common_parser(__doc__)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    data = prepare_stage11_data(config)
    output = config.run_dir() / "data_audit"
    data.audit.write_reports(output)
    report = {**data.report, "fingerprint": data.fingerprint,
              "cache_sequences": {k: len(v) for k, v in data.sequences().items()}}
    (output / "stage11_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Keep potentially lengthy rejected-row details in the audit artifact.
    print(json.dumps({k: v for k, v in report.items() if k != "ph_rejected_rows"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
