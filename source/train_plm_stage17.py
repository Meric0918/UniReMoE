#!/usr/bin/env python3
"""Run Stage17's audited training, dry-run or exact checkpoint resume."""
import os
from pathlib import Path
import sys

cuda_root = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and cuda_root.is_dir():
    os.environ["CUDA_HOME"] = str(cuda_root)

from plm_unified.stage16_pipeline import main

if __name__ == "__main__":
    main(17)
