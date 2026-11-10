"""Independent configuration for random downstream Stage PHT training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import SwanLabConfig
from .stage1_encoder_config import Stage1EncoderModelConfig
from .stage_adapter_config import StageAdapterRuntimeConfig, _resolve, _section, _serialise


@dataclass
class StagePHTPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path
    tcr_bert_model: Path


@dataclass
class StagePHTModelConfig(Stage1EncoderModelConfig):
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.2


@dataclass
class StagePHTTrainingConfig:
    data_subdir: str = "PHT/classes"
    class_dirs: tuple[str, ...] = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    reject_noncanonical_sequences: bool = True
    warmup_epochs: int = 2
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    min_rounds: int = 5
    max_rounds: int = 100
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    encoder_lr: float = 2.5e-5
    pht_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_steps: int = 50


@dataclass
class StagePHTExperimentConfig:
    paths: StagePHTPaths
    model: StagePHTModelConfig = field(default_factory=StagePHTModelConfig)
    runtime: StageAdapterRuntimeConfig = field(default_factory=StageAdapterRuntimeConfig)
    stage_PHT: StagePHTTrainingConfig = field(default_factory=StagePHTTrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Path | None = None

    @property
    def training(self):
        return self.runtime

    def data_dir(self) -> Path:
        return self.paths.data_root / self.stage_PHT.data_subdir

    def run_dir(self) -> Path:
        return self.paths.output_root / f"fold_{self.runtime.fold}" / f"seed_{self.runtime.seed}"

    def warmup_dir(self) -> Path:
        return self.run_dir() / "warmup"

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))

    def validate(self, *, require_files: bool = False) -> None:
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        for name in (
            "peptide_input_dim", "hla_input_dim", "tcr_input_dim", "model_dim",
            "num_attention_heads", "feedforward_dim", "encoder_num_layers",
            "peptide_min_length", "peptide_max_length", "hla_max_length", "tcr_max_length",
        ):
            value = getattr(self.model, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        for name in ("encoder_dropout", "attention_dropout", "classifier_dropout"):
            if not 0 <= getattr(self.model, name) < 1:
                raise ValueError(f"model.{name} must be in [0, 1)")
        values = self.stage_PHT
        if tuple(values.class_dirs) != ("PHT_000", "PHT_010", "PHT_100", "PHT_111"):
            raise ValueError("stage_PHT.class_dirs must list the four canonical PHT classes in order")
        for name in (
            "warmup_epochs", "micro_batch_size", "eval_batch_size",
            "gradient_accumulation_steps", "min_rounds", "max_rounds",
            "early_stopping_patience", "log_every_steps",
        ):
            value = getattr(values, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"stage_PHT.{name} must be a positive integer")
        if values.micro_batch_size % 4:
            raise ValueError("stage_PHT.micro_batch_size must be divisible by 4")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage_PHT.min_rounds cannot exceed max_rounds")
        for name in ("encoder_lr", "pht_module_lr", "gradient_clip_norm"):
            if not getattr(values, name) > 0:
                raise ValueError(f"stage_PHT.{name} must be positive")
        for name in ("early_stopping_min_delta", "weight_decay"):
            if not getattr(values, name) >= 0:
                raise ValueError(f"stage_PHT.{name} cannot be negative")
        if not 0 <= values.scheduler_warmup_ratio <= 1:
            raise ValueError("stage_PHT.scheduler_warmup_ratio must be in [0, 1]")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if require_files:
            for class_dir in values.class_dirs:
                for filename in (values.train_filename, values.val_filename):
                    path = self.data_dir() / class_dir / filename
                    if not path.is_file():
                        raise FileNotFoundError(f"Stage PHT CSV not found: {path}")


def load_stage_PHT_config(path: str | Path) -> StagePHTExperimentConfig:
    import yaml

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage PHT YAML root must be a mapping")
    unexpected = set(raw) - {"paths", "model", "runtime", "stage_PHT", "swanlab"}
    if unexpected:
        raise ValueError("Unknown Stage PHT config sections: " + ", ".join(sorted(unexpected)))
    paths = _section(raw, "paths", StagePHTPaths.__dataclass_fields__)
    missing = set(StagePHTPaths.__dataclass_fields__) - set(paths)
    if missing:
        raise ValueError("Missing Stage PHT paths: " + ", ".join(sorted(missing)))
    config = StagePHTExperimentConfig(
        paths=StagePHTPaths(**{key: _resolve(value, config_path.parent) for key, value in paths.items()}),
        model=StagePHTModelConfig(**_section(raw, "model", StagePHTModelConfig.__dataclass_fields__)),
        runtime=StageAdapterRuntimeConfig(**_section(raw, "runtime", StageAdapterRuntimeConfig.__dataclass_fields__)),
        stage_PHT=StagePHTTrainingConfig(**_section(raw, "stage_PHT", StagePHTTrainingConfig.__dataclass_fields__)),
        swanlab=SwanLabConfig(**_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage_PHT_overrides(config, *, device=None, num_workers=None, swanlab_mode=None):
    if device is not None:
        config.runtime.device = device
    if num_workers is not None:
        config.runtime.num_workers = num_workers
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    config.validate()
    return config
