"""Configuration for consistency training with sample-gated PT transfer."""
from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

from .config import SwanLabConfig
from .stage_final_config import StageFinalConfig, StageFinalSettings, parse_stage_final_config


@dataclass
class LegacyTaskGateSettings(StageFinalSettings):
    task_gate_enabled: bool = True
    task_gate_logit_init: float = 0.0


@dataclass
class StageMOEConsisPTSettings(StageFinalSettings):
    sample_gate_enabled: bool = True
    sample_gate_hidden_dim: int = 128
    sample_gate_logit_init: float = 0.0


@dataclass
class StageMOEConsisPTConfig(StageFinalConfig):
    stage_name = 'stage_MOE_consis_PT'
    architecture = 'stage_MOE_consis_PT_v2'
    settings_type = StageMOEConsisPTSettings
    extra_model_settings = ('sample_gate_enabled', 'sample_gate_hidden_dim', 'sample_gate_logit_init')

    stage_final: StageMOEConsisPTSettings = field(default_factory=StageMOEConsisPTSettings)
    swanlab: SwanLabConfig = field(default_factory=lambda: SwanLabConfig(
        mode='disabled', group='plm-stage_MOE_consis_PT', experiment_prefix='PLM-Stage_MOE_consis_PT',
        tags=['ESMC-300M', 'TCR-BERT', 'stage_MOE_consis_PT']))

    @property
    def stage_MOE_consis_PT(self):
        return self.stage_final

    def validate(self):
        super().validate()
        s = self.stage_MOE_consis_PT
        if not isinstance(s.sample_gate_enabled, bool):
            raise ValueError('sample_gate_enabled must be boolean')
        if type(s.sample_gate_hidden_dim) is not int or s.sample_gate_hidden_dim < 1:
            raise ValueError('sample_gate_hidden_dim must be a positive integer')
        if type(s.sample_gate_logit_init) not in (int, float) or not math.isfinite(s.sample_gate_logit_init):
            raise ValueError('sample_gate_logit_init must be a finite number')


@dataclass
class LegacyTaskGateConfig(StageMOEConsisPTConfig):
    architecture = 'stage_MOE_consis_PT_v1'
    settings_type = LegacyTaskGateSettings
    extra_model_settings = ('task_gate_enabled', 'task_gate_logit_init')
    stage_final: LegacyTaskGateSettings = field(default_factory=LegacyTaskGateSettings)

    def validate(self):
        StageFinalConfig.validate(self)
        s = self.stage_final
        if not isinstance(s.task_gate_enabled, bool):
            raise ValueError('task_gate_enabled must be boolean')
        if type(s.task_gate_logit_init) not in (int, float) or not math.isfinite(s.task_gate_logit_init):
            raise ValueError('task_gate_logit_init must be a finite number')


def load_stage_MOE_consis_PT_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    settings = raw.get(StageMOEConsisPTConfig.stage_name) if isinstance(raw, dict) else None
    if isinstance(settings, dict) and settings.keys() & set(LegacyTaskGateConfig.extra_model_settings):
        raise ValueError('This v1 task-gate configuration is for evaluate or benchmark. '
                         'Use the sample_gate settings for v2 training.')
    return parse_stage_final_config(path, raw, StageMOEConsisPTConfig)
