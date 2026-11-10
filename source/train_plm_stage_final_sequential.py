#!/usr/bin/env python3
"""Train/resume Stage Final with one PH, PT and PHT pass in that order per round."""
from plm_unified.stage_final_cli import main

if __name__ == '__main__':
    main('train-sequential')
