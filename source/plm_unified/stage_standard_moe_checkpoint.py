"""Independent MOE checkpoints, epoch-boundary RNG and strict resume contracts."""
from __future__ import annotations
import math
import torch

from .stage_standard_moe_model import ARCHITECTURE
from .stage_final_checkpoint import atomic_save, capture_rng, restore_rng

MODEL_SETTINGS = ('pt_fusion_enabled', 'fusion_hidden_dim', 'fusion_bias_init', 'sequence_encoder_num_layers', 'dropout', 'expert_hidden_dim', 'num_experts', 'top_k',
                  'pooling_hidden_dim', 'classifier_hidden_dim')


def model_contract(config):
    raw = config.to_dict()
    return {'architecture': ARCHITECTURE, 'router': 'linear_masked_mean_sample_topk_v1',
            'model': raw['model'], 'stage_standard_moe': {k: raw['stage_standard_moe'][k] for k in MODEL_SETTINGS}}


def training_contract(config, data_fingerprint, cache_fingerprints):
    settings = config.to_dict()['stage_standard_moe']
    for name in ('device', 'num_workers', 'eval_batch_size', 'log_every_steps', 'cache_batch_size', 'cache_shard_size'):
        settings.pop(name)
    return {'model': model_contract(config), 'training': settings,
            'data_fingerprint': data_fingerprint, 'cache_fingerprints': cache_fingerprints}


def load_checkpoint(path, *, config=None, role=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('format_version') != 1:
        raise ValueError('Expected a stage_standard_moe_v1 checkpoint; other stage checkpoints are incompatible')
    if checkpoint.get('training_mode') != 'sequential':
        raise ValueError('Expected sequential training mode')
    if checkpoint.get('role') not in ('best', 'last') or (role and checkpoint['role'] != role):
        raise ValueError(f'Expected {role or "best/last"} checkpoint')
    required = {'model', 'model_contract', 'config', 'temperature', 'thresholds', 'encoders', 'epoch'}
    if required - checkpoint.keys():
        raise ValueError('Incomplete stage_standard_moe checkpoint')
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
