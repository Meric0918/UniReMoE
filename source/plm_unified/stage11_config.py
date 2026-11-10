"""Standalone configuration for frozen-PLM, alternating Stage11 training."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .config import CacheConfig, ModelConfig, PathsConfig, SwanLabConfig
from .stage8_config import Stage8RuntimeConfig, _resolve, _section, _serialise


@dataclass
class Stage11TrainingConfig:
    data_subdir: str = "PHT/classes"
    class_dirs: tuple[str, ...] = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
    ph_train_file: str = "PH/train.csv"
    reject_noncanonical_sequences: bool = True
    condition: str = "both"
    projection_dim: int = 256
    temperature: float = 0.2
    contrastive_batch_size: int = 256
    batch_size: int = 256
    eval_batch_size: int = 256
    min_rounds: int = 5
    max_rounds: int = 100
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    log_every_steps: int = 50


@dataclass
class Stage11ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        adapter_dropout=0.1, attention_dropout=0.1, classifier_dropout=0.2))
    cache: CacheConfig = field(default_factory=lambda: CacheConfig(hla_splits=[], tcr_splits=[]))
    runtime: Stage8RuntimeConfig = field(default_factory=Stage8RuntimeConfig)
    stage11: Stage11TrainingConfig = field(default_factory=Stage11TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=lambda: SwanLabConfig(
        group="plm-stage11-fold{fold}", experiment_prefix="PLM-Stage11",
        tags=["stage11", "frozen-PLM", "multi-positive-CLIP", "alternating"]))
    source_path: Path | None = None

    @property
    def training(self):
        return self.runtime

    def cache_dir(self, scope="train"):
        if scope not in {"train", "test"}:
            raise ValueError("Cache scope must be train or test")
        return self.paths.cache_root / f"fold_{self.runtime.fold}" / scope

    def run_dir(self):
        return (self.paths.output_root / f"fold_{self.runtime.fold}"
                / f"seed_{self.runtime.seed}" / self.stage11.condition)

    def to_dict(self):
        return _serialise(asdict(self))

    def validate(self):
        m, t = self.model, self.stage11
        for name in ("peptide_input_dim", "hla_input_dim", "tcr_input_dim", "adapter_hidden_dim",
                     "model_dim", "num_attention_heads", "feedforward_dim", "peptide_min_length",
                     "peptide_max_length", "hla_max_length", "tcr_max_length"):
            if getattr(m, name) < 1:
                raise ValueError(f"model.{name} must be positive")
        if m.model_dim % m.num_attention_heads or m.peptide_min_length > m.peptide_max_length:
            raise ValueError("Invalid attention dimensions or peptide length bounds")
        for name in ("adapter_dropout", "attention_dropout", "classifier_dropout"):
            if not 0 <= getattr(m, name) < 1:
                raise ValueError(f"model.{name} must be in [0,1)")
        if tuple(t.class_dirs) != ("PHT_000", "PHT_010", "PHT_100", "PHT_111"):
            raise ValueError("All four canonical PHT class directories are required")
        if t.condition not in {"none", "ph", "pht", "both"}:
            raise ValueError("condition must be none, ph, pht or both")
        if t.batch_size < 4 or t.batch_size % 4 or t.contrastive_batch_size < 2:
            raise ValueError("Classification batch must be divisible by 4; contrastive batch must be >=2")
        for name in ("projection_dim", "eval_batch_size", "min_rounds", "max_rounds",
                     "early_stopping_patience", "log_every_steps"):
            if getattr(t, name) < 1:
                raise ValueError(f"stage11.{name} must be positive")
        if t.min_rounds > t.max_rounds:
            raise ValueError("min_rounds cannot exceed max_rounds")
        for name in ("temperature", "learning_rate", "gradient_clip_norm"):
            if getattr(t, name) <= 0:
                raise ValueError(f"stage11.{name} must be positive")
        if t.weight_decay < 0 or t.early_stopping_min_delta < 0 or not 0 <= t.scheduler_warmup_ratio < 1:
            raise ValueError("Invalid optimizer or early stopping settings")
        if t.mixed_precision not in {"none", "bf16", "fp16"}:
            raise ValueError("mixed_precision must be none, bf16 or fp16")
        if self.runtime.fold < 1 or self.runtime.num_workers < 0:
            raise ValueError("Invalid fold or num_workers")
        if self.cache.dtype != "float16" or min(self.cache.shard_size, self.cache.precompute_batch_size) < 1:
            raise ValueError("Invalid float16 cache configuration")
        if self.swanlab.mode not in {"disabled", "offline", "online"}:
            raise ValueError("Invalid SwanLab mode")
        for name in ("data_subdir", "ph_train_file"):
            path = Path(getattr(t, name))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must stay within data_root")
        if Path(t.ph_train_file).name != "train.csv":
            raise ValueError("Supplementary PH data must be a train.csv file")


def config_from_dict(raw, *, base_dir=None):
    raw = dict(raw)
    raw.pop("source_path", None)
    sections = {"paths": PathsConfig, "model": ModelConfig, "cache": CacheConfig,
                "runtime": Stage8RuntimeConfig, "stage11": Stage11TrainingConfig, "swanlab": SwanLabConfig}
    if set(raw) - set(sections):
        raise ValueError("Unknown Stage11 configuration sections")
    values = {}
    for name, cls in sections.items():
        if name not in raw and name != "paths":
            continue
        section = _section(raw, name, cls.__dataclass_fields__)
        if name == "paths":
            section = {k: _resolve(v, base_dir or Path.cwd()) for k, v in section.items()}
        values[name] = cls(**section)
    config = Stage11ExperimentConfig(**values)
    config.validate()
    return config


def load_stage11_config(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage11 configuration must be a mapping")
    config = config_from_dict(raw, base_dir=path.parent)
    config.source_path = path
    return config
