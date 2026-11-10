#!/usr/bin/env python3
"""Create an immutable validation-selection manifest before final testing."""
from plm_unified.stage16_benchmarks import lock_main

if __name__ == "__main__":
    lock_main()
