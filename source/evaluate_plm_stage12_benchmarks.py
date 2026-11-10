#!/usr/bin/env python3
"""Evaluate a successful Stage12 best checkpoint on PH/PT/PHT tests."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluate_plm_stage8_benchmarks import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        root = Path(__file__).resolve().parents[1]
        sys.argv[1:1] = ["--config", str(root / "configs" / "plm_stage12.yaml")]
    main()
