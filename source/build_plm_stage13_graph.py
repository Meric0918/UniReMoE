#!/usr/bin/env python3
"""Audit and build the Stage13 train-positive heterogeneous graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.stage13_config import load_stage13_config
from plm_unified.stage13_graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "plm_stage13.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug-only row limit per training CSV")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = load_stage13_config(args.config)
    artifact = build_graph(config, overwrite=args.overwrite, limit=args.limit)
    print(json.dumps(artifact.manifest, indent=2, sort_keys=True))
    print(f"Graph artifact: {artifact.root}")


if __name__ == "__main__":
    main()
