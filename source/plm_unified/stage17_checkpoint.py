"""Stage17 typed checkpoints share the tested provenance/resume implementation."""

from .stage16_checkpoint import save_training_checkpoint, restore_training_checkpoint
from .stage16_checkpoint import load_inference_checkpoint as _load


def load_stage17_checkpoint(path):
    return _load(path, expected_stage=17)
