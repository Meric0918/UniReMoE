"""Stage18 inference artifacts include the fine-tuned encoders, never teacher state."""

from .stage16_checkpoint import save_training_checkpoint, restore_training_checkpoint
from .stage16_checkpoint import load_inference_checkpoint as _load


def load_stage18_checkpoint(path):
    return _load(path, expected_stage=18)
