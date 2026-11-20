"""Resolve the evaluation architecture and fusion settings from a saved Consis PT run."""
from pathlib import Path

import torch
import yaml

from .stage_MOE_consis_PT_config import LegacyTaskGateConfig, StageMOEConsisPTConfig
from .stage_MOE_consis_PT_model import LegacyTaskGateBindingModel, StageMOEConsisPTBindingModel
from .stage_final_checkpoint import validate_checkpoint
from .stage_final_config import parse_stage_final_config


def evaluation_setup(config_path, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    architectures = {
        LegacyTaskGateConfig.architecture: (LegacyTaskGateConfig, LegacyTaskGateBindingModel),
        StageMOEConsisPTConfig.architecture: (StageMOEConsisPTConfig, StageMOEConsisPTBindingModel),
    }
    if not isinstance(checkpoint, dict) or checkpoint.get('architecture') not in architectures:
        raise ValueError('Expected a stage_MOE_consis_PT v1 or v2 checkpoint')
    config_type, model_type = architectures[checkpoint['architecture']]
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text())
    stage = config_type.stage_name
    if not isinstance(raw, dict) or not isinstance(raw.get(stage, {}), dict):
        raise ValueError(f'Invalid {stage} evaluation configuration')
    settings = raw.setdefault(stage, {})
    for field in (*LegacyTaskGateConfig.extra_model_settings, *StageMOEConsisPTConfig.extra_model_settings):
        settings.pop(field, None)
    saved = checkpoint.get('config', {}).get(stage, {})
    for field in ('pt_fusion_enabled', 'fusion_hidden_dim', 'fusion_bias_init', *config_type.extra_model_settings):
        if field not in saved:
            raise ValueError(f'Missing checkpoint fusion setting {field}')
        settings[field] = saved[field]
    config = parse_stage_final_config(path, raw, config_type)
    return config, model_type, validate_checkpoint(checkpoint, config=config)
