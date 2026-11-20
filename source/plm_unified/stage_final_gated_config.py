"""Standalone configuration for the frozen-PLM Stage Final Gated v1 experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml

from .config import ModelConfig, SwanLabConfig


@dataclass
class StageFinalGatedPaths:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    esmc_model: Path
    tcr_bert_model: Path


@dataclass
class StageFinalGatedSettings:
    sample_fusion_enabled: bool = True
    sample_constraint_enabled: bool = True
    sample_gate_hidden_dim: int = 128
    sample_fusion_init: float = 0.9
    constraint_gate_floor: float = 0.05
    constraint_gate_regularization: float = 1e-3
    pt_fusion_enabled: bool = True
    fusion_hidden_dim: int = 128
    fusion_bias_init: float = -2.0
    aux_ph_weight: float = 0.1
    aux_pt_weight: float = 0.1
    consistency_margin: float = 0.05
    hierarchy_lambda: float = 0.05
    pt_implication_lambda: float = 0.03
    consistency_warmup_epochs: int = 3
    consistency_ramp_epochs: int = 5
    sequence_encoder_num_layers: int = 3
    dropout: float = 0.1
    expert_hidden_dim: int = 512
    gate_hidden_dim: int = 128
    pooling_hidden_dim: int = 128
    classifier_hidden_dim: int = 128
    batch_size: int = 32
    eval_batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    learning_rate: float = 2e-4
    router_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_ratio: float = 0.05
    balance_lambda: float = 0.0
    temperature_start: float = 2.0
    temperature_end: float = 1.0
    temperature_epochs: int = 5
    ph_weight: float = 1.0
    pt_weight: float = 1.0
    pht_weight: float = 1.0
    mixed_precision: str = 'bf16'
    seed: int = 42
    device: str = 'auto'
    num_workers: int = 2
    limit_train_samples: int | None = None
    limit_eval_samples: int | None = None
    max_steps_per_epoch: int | None = None
    cache_batch_size: int = 32
    cache_shard_size: int = 16384
    log_every_steps: int = 50

    def temperature(self, epoch: int) -> float:
        if self.temperature_epochs == 1:
            return self.temperature_end
        progress = min(max(epoch - 1, 0) / max(self.temperature_epochs - 1, 1), 1.0)
        return self.temperature_start + progress * (self.temperature_end - self.temperature_start)

    def consistency_ramp(self, epoch: int) -> float:
        return min(max((epoch - self.consistency_warmup_epochs) / self.consistency_ramp_epochs, 0.0), 1.0)


@dataclass
class StageFinalGatedConfig:
    paths: StageFinalGatedPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    stage_final_gated: StageFinalGatedSettings = field(default_factory=StageFinalGatedSettings)
    swanlab: SwanLabConfig = field(default_factory=lambda: SwanLabConfig(
        mode='disabled', group='plm-stage_final_gated', experiment_prefix='PLM-Stage_final_gated',
        tags=['ESMC-300M', 'TCR-BERT', 'stage_final_gated']))

    @property
    def stage_MOE(self):
        """Configuration view used by the unchanged Stage MoE model."""
        return self.stage_final_gated

    @property
    def stage_MOE_PT(self):
        """Read-only view for the reused PT fusion model."""
        return self.stage_final_gated

    def validate(self):
        m, s = self.model, self.stage_final_gated
        for name in ('sample_fusion_enabled', 'sample_constraint_enabled'):
            if not isinstance(getattr(s, name), bool):
                raise ValueError(f'{name} must be boolean')
        if (not isinstance(s.sample_gate_hidden_dim, int) or isinstance(s.sample_gate_hidden_dim, bool)
                or s.sample_gate_hidden_dim < 1):
            raise ValueError('sample_gate_hidden_dim must be a positive integer')
        for name in ('sample_fusion_init', 'constraint_gate_floor'):
            value = getattr(s, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be in (0, 1)')
        if not math.isfinite(s.constraint_gate_regularization) or s.constraint_gate_regularization < 0:
            raise ValueError('constraint_gate_regularization must be finite and nonnegative')
        if not isinstance(s.pt_fusion_enabled, bool):
            raise ValueError("pt_fusion_enabled must be boolean")
        if not isinstance(s.fusion_hidden_dim, int) or isinstance(s.fusion_hidden_dim, bool) or s.fusion_hidden_dim < 1:
            raise ValueError("fusion_hidden_dim must be a positive integer")
        if not math.isfinite(s.fusion_bias_init):
            raise ValueError("fusion_bias_init must be finite")
        for name in ('aux_ph_weight', 'aux_pt_weight', 'hierarchy_lambda', 'pt_implication_lambda', 'consistency_margin'):
            value = getattr(s, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if s.consistency_margin >= 1:
            raise ValueError('consistency_margin must be less than 1')
        for name, minimum in (('consistency_warmup_epochs', 0), ('consistency_ramp_epochs', 1)):
            value = getattr(s, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.swanlab.mode not in ('disabled', 'online', 'offline', 'local'):
            raise ValueError('Invalid SwanLab mode')
        for name in ('model_dim', 'num_attention_heads', 'feedforward_dim', 'peptide_input_dim',
                     'hla_input_dim', 'tcr_input_dim', 'peptide_min_length', 'peptide_max_length',
                     'hla_max_length', 'tcr_max_length'):
            value = getattr(m, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'model.{name} must be a positive integer')
        if m.model_dim % m.num_attention_heads or m.peptide_min_length > m.peptide_max_length:
            raise ValueError('Invalid model dimensions or peptide length bounds')
        for name in ('sequence_encoder_num_layers', 'expert_hidden_dim', 'gate_hidden_dim',
                     'pooling_hidden_dim', 'classifier_hidden_dim', 'batch_size', 'eval_batch_size',
                     'max_epochs', 'patience', 'temperature_epochs', 'cache_batch_size',
                     'cache_shard_size', 'log_every_steps'):
            value = getattr(s, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'stage_final_gated.{name} must be a positive integer')
        for name in ('learning_rate', 'router_learning_rate', 'gradient_clip_norm',
                     'temperature_start', 'temperature_end', 'ph_weight', 'pt_weight', 'pht_weight'):
            value = getattr(s, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'stage_final_gated.{name} must be finite and positive')
        for name in ('min_delta', 'weight_decay', 'balance_lambda'):
            if not math.isfinite(getattr(s, name)) or getattr(s, name) < 0:
                raise ValueError(f'stage_final_gated.{name} must be finite and nonnegative')
        if not 0 <= s.dropout < 1 or not 0 <= s.warmup_ratio < 1:
            raise ValueError('dropout and warmup_ratio must be in [0, 1)')
        if s.mixed_precision not in ('none', 'bf16', 'fp16'):
            raise ValueError('mixed_precision must be none, bf16 or fp16')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (s.num_workers, s.seed)):
            raise ValueError('num_workers and seed must be nonnegative')
        for name in ('limit_train_samples', 'limit_eval_samples', 'max_steps_per_epoch'):
            value = getattr(s, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f'{name} must be a positive integer or null')
        # Keep all artifacts isolated from existing stages and source datasets.
        for name in ('cache_root', 'output_root'):
            root = getattr(self.paths, name).resolve()
            if not any('moe' in part.lower() or 'stage_final_gated' in part.lower() for part in root.parts):
                raise ValueError(f'{name} must use a dedicated MOE or stage_final_gated directory')
            for protected in (self.paths.data_root.resolve(), self.paths.esmc_model.resolve(),
                              self.paths.tcr_bert_model.resolve()):
                if root == protected or root in protected.parents or protected in root.parents:
                    raise ValueError(f'{name} overlaps data or pretrained models')
        cache, output = self.paths.cache_root.resolve(), self.paths.output_root.resolve()
        if not any('stage_final_gated' in part.lower() for part in output.parts):
            raise ValueError('output_root must use a dedicated stage_final_gated directory')
        if cache == output or cache in output.parents or output in cache.parents:
            raise ValueError('Cache and output directories must be separate')

    def to_dict(self):
        obj = asdict(self)
        obj['paths'] = {k: str(v) for k, v in obj['paths'].items()}
        return obj


def load_stage_final_gated_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - {'paths', 'model', 'stage_final_gated', 'swanlab'}:
        raise ValueError('Expected only paths, model, stage_final_gated and swanlab config sections')
    constructors = {'paths': StageFinalGatedPaths, 'model': ModelConfig, 'stage_final_gated': StageFinalGatedSettings}
    sections = {}
    for name, constructor in constructors.items():
        values = raw.get(name, {})
        if not isinstance(values, dict) or set(values) - constructor.__dataclass_fields__.keys():
            raise ValueError(f'Unknown fields in {name}')
        if name == 'paths':
            values = {k: (path.parent / Path(v).expanduser()).resolve() for k, v in values.items()}
        sections[name] = constructor(**values)
    if 'swanlab' in raw:
        values = raw['swanlab']
        if not isinstance(values, dict) or set(values) - SwanLabConfig.__dataclass_fields__.keys():
            raise ValueError('Unknown fields in swanlab')
        defaults = dict(mode='disabled', group='plm-stage_final_gated', experiment_prefix='PLM-Stage_final_gated',
                        tags=['ESMC-300M', 'TCR-BERT', 'stage_final_gated'])
        sections['swanlab'] = SwanLabConfig(**(defaults | values))
    config = StageFinalGatedConfig(**sections)
    config.validate()
    return config
