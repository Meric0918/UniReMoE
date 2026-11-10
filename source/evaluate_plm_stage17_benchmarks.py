#!/usr/bin/env python3
"""Evaluate a locked Stage17 student checkpoint on PH/PT/PHT."""
import train_plm_stage17  # configure the local CUDA loader
from plm_unified.stage16_benchmarks import main

if __name__ == "__main__":
    main(17)
