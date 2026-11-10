#!/usr/bin/env python3
"""Run the Stage1 encoder experiment across the configured seeds."""

from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from run_plm_stage1_seeds import main as stage1_main


def main(argv=None):
    stage1_main(
        argv,
        default_config="plm_stage1_encoder.yaml",
        config_loader=load_stage1_encoder_config,
        training_script_name="train_plm_stage1_encoder.py",
    )


if __name__ == "__main__":
    main()
