"""Independent MOE checkpoints, epoch-boundary RNG and strict resume contracts."""
from __future__ import annotations
import os
import random
import math
from pathlib import Path
import numpy as np
import torch

from .stage_final_model import ARCHITECTURE

MODEL_SETTINGS = ('pt_fusion_enabled', 'fusion_hidden_dim', 'fusion_bias_init', 'sequence_encoder_num_layers', 'dropout', 'expert_hidden_dim', 'gate_hidden_dim',
                  'pooling_hidden_dim', 'classifier_hidden_dim')


def model_contract(config):
    raw = config.to_dict()
    stage = config.stage_name
    fields = MODEL_SETTINGS + config.extra_model_settings
    return {'model': raw['model'], stage: {k: raw[stage][k] for k in fields}}


def training_contract(config, data_fingerprint, cache_fingerprints):
    settings = config.to_dict()[config.stage_name]
    for name in ('device', 'num_workers', 'eval_batch_size', 'log_every_steps', 'cache_batch_size', 'cache_shard_size'):
        settings.pop(name)
    return {'model': model_contract(config), 'training': settings,
            'data_fingerprint': data_fingerprint, 'cache_fingerprints': cache_fingerprints}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, config=None, role=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return validate_checkpoint(checkpoint, config=config, role=role)


def validate_checkpoint(checkpoint, *, config=None, role=None):
    expected = config.architecture if config is not None else ARCHITECTURE
    if not isinstance(checkpoint, dict) or checkpoint.get('architecture') != expected or checkpoint.get('format_version') != 1:
        raise ValueError(f'Expected a {expected} checkpoint. For older evaluation-only checkpoints, use evaluate or benchmark.')
    if checkpoint.get('role') not in ('best', 'last') or (role and checkpoint['role'] != role):
        raise ValueError(f'Expected {role or "best/last"} checkpoint')
    required = {'model', 'model_contract', 'config', 'temperature', 'thresholds', 'encoders', 'epoch'}
    if required - checkpoint.keys():
        raise ValueError('Incomplete stage_final checkpoint')
    if config and checkpoint['model_contract'] != model_contract(config):
        raise ValueError('Checkpoint model configuration mismatch')
    if set(checkpoint['thresholds']) != {'ph', 'pt', 'pht'}:
        raise ValueError('Checkpoint requires all three validation thresholds')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in checkpoint['thresholds'].values()):
        raise ValueError('Checkpoint thresholds must be finite probabilities')
    if not math.isfinite(checkpoint['temperature']) or checkpoint['temperature'] <= 0:
        raise ValueError('Checkpoint routing temperature must be finite and positive')
    if checkpoint['role'] == 'last':
        required_last = {'optimizer', 'scheduler', 'scaler', 'rng', 'contract', 'global_step',
                         'best_payload', 'best_score', 'bad_epochs', 'history'}
        if required_last - checkpoint.keys():
            raise ValueError('Incomplete last checkpoint: missing training/resume state')
    return checkpoint
