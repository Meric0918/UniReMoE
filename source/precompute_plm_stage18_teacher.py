#!/usr/bin/env python3
"""Precompute pinned original-encoder Stage1 teacher logits; never on test."""
import train_plm_stage18  # configure local CUDA loader before constructing PLMs
from plm_unified.stage16_pipeline import precompute_main

if __name__ == "__main__":
    precompute_main()
