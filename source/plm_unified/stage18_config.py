"""Stage18 replay/regularisation/distillation condition configuration."""

from .stage16_config import Stage16ExperimentConfig as Stage18ExperimentConfig
from .stage16_config import load_stage16_config


def load_stage18_config(path):
    return load_stage16_config(path, expected_stage=18)
