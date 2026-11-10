#!/usr/bin/env python3
"""Evaluate joint or exported Stage1 sequence-encoder checkpoints."""

from evaluate_plm_stage1 import main as stage1_main
from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from plm_unified.stage1_encoder_model import (
    Stage1EncoderBindingModel,
    TaskSpecificStage1EncoderBindingModel,
    validate_encoder_checkpoint,
)


def main(argv=None):
    stage1_main(
        argv,
        default_config="plm_stage1_encoder.yaml",
        default_split="val_unseen_pair",
        config_loader=load_stage1_encoder_config,
        model_class=Stage1EncoderBindingModel,
        task_model_class=TaskSpecificStage1EncoderBindingModel,
        checkpoint_validator=validate_encoder_checkpoint,
    )


if __name__ == "__main__":
    main()
