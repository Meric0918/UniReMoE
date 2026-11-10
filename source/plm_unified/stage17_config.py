"""Stage17 configuration uses the shared strict schema and condition table."""

from .stage16_config import Stage16ExperimentConfig as Stage17ExperimentConfig
from .stage16_config import load_stage16_config


def load_stage17_config(path):
    return load_stage16_config(path, expected_stage=17)
