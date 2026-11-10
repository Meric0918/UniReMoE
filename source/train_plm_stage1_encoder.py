#!/usr/bin/env python3
"""Train or resume the independent Stage1 sequence-encoder experiment."""

# The common entrypoint imports NumPy before PyTorch, which also keeps MKL
# initialization consistent in children launched by the multi-seed runner.
from train_plm_stage1 import main as stage1_main
from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from plm_unified.stage1_encoder_model import ENCODER_UMAP_STAGES, Stage1EncoderBindingModel
from plm_unified.stage1_encoder_trainer import Stage1EncoderTrainer


def main(argv=None):
    stage1_main(
        argv,
        default_config="plm_stage1_encoder.yaml",
        config_loader=load_stage1_encoder_config,
        model_class=Stage1EncoderBindingModel,
        trainer_class=Stage1EncoderTrainer,
        artifact_namespace="stage1_encoder",
        umap_stages=ENCODER_UMAP_STAGES,
    )


if __name__ == "__main__":
    main()
