"""Configuration for the isolated Stage-13 GraphSAGE pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import ModelConfig, SwanLabConfig


@dataclass
class Stage13Paths:
    project_root: Path
    data_root: Path
    residue_cache_root: Path
    graph_root: Path
    output_root: Path
    swanlog_root: Path
    test_residue_cache_root: Path | None = None
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class Stage13DataConfig:
    ph_train: str = "PH/train.csv"
    pt_train: str = "PT/train.csv"
    ph_validation_splits: list[str] = field(
        default_factory=lambda: [
            "val_unseen_hla",
            "val_unseen_pair",
            "val_unseen_peptide",
        ]
    )
    pt_validation_splits: list[str] = field(
        default_factory=lambda: [
            "val_unseen_pair",
            "val_unseen_peptide",
            "val_unseen_tcr",
        ]
    )
    ph_test_splits: list[str] = field(default_factory=list)
    pt_test_splits: list[str] = field(default_factory=list)

    def validate(self) -> None:
        for name in ("ph_train", "pt_train"):
            if not getattr(self, name).strip():
                raise ValueError(f"stage13.data.{name} cannot be empty")
        for name in (
            "ph_validation_splits",
            "pt_validation_splits",
            "ph_test_splits",
            "pt_test_splits",
        ):
            values = getattr(self, name)
            if name.endswith("validation_splits") and not values:
                raise ValueError(f"stage13.data.{name} cannot be empty")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"stage13.data.{name} contains an empty split")


@dataclass
class Stage13GraphConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    fanouts: list[int] = field(default_factory=lambda: [16, 8])
    dropout: float = 0.1
    projection_dropout: float = 0.1
    build_batch_size: int = 512
    sampling_seed: int = 20260901

    def validate(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("stage13.graph.hidden_dim must be positive")
        if self.num_layers < 1:
            raise ValueError("stage13.graph.num_layers must be positive")
        if len(self.fanouts) != self.num_layers or any(value < 1 for value in self.fanouts):
            raise ValueError("stage13.graph.fanouts must match num_layers and be positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.projection_dropout < 1:
            raise ValueError("stage13 graph dropouts must be in [0, 1)")
        if self.build_batch_size < 1:
            raise ValueError("stage13.graph.build_batch_size must be positive")


@dataclass
class Stage13RuntimeConfig:
    fold: int = 1
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class Stage13TrainingConfig:
    batch_size: int = 1024
    warmup_epochs_per_task: int = 2
    max_rounds: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    graph_lr: float = 3e-4
    fusion_lr: float = 3e-4
    adapter_lr: float = 3e-4
    cross_attention_lr: float = 2e-4
    pooling_lr: float = 3e-4
    classifier_lr: float = 3e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    fgm_epsilon: float = 1.0
    threshold: float = 0.5
    mixed_precision: str = "bf16"
    log_every_steps: int = 50

    def validate(self) -> None:
        for name in (
            "batch_size",
            "warmup_epochs_per_task",
            "max_rounds",
            "early_stopping_patience",
            "log_every_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"stage13.training.{name} must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("stage13.training.early_stopping_min_delta cannot be negative")
        for name in (
            "graph_lr",
            "fusion_lr",
            "adapter_lr",
            "cross_attention_lr",
            "pooling_lr",
            "classifier_lr",
            "gradient_clip_norm",
            "fgm_epsilon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"stage13.training.{name} must be positive")
        if self.weight_decay < 0 or not 0 <= self.scheduler_warmup_ratio < 1:
            raise ValueError("invalid Stage13 optimizer configuration")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("stage13.training.mixed_precision must be none, fp16, or bf16")


@dataclass
class Stage13ExperimentConfig:
    paths: Stage13Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    data: Stage13DataConfig = field(default_factory=Stage13DataConfig)
    graph: Stage13GraphConfig = field(default_factory=Stage13GraphConfig)
    runtime: Stage13RuntimeConfig = field(default_factory=Stage13RuntimeConfig)
    training: Stage13TrainingConfig = field(default_factory=Stage13TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Path | None = None

    def validate(self, *, require_models: bool = False) -> None:
        if self.runtime.fold < 1 or self.runtime.num_workers < 0:
            raise ValueError("invalid Stage13 runtime configuration")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        self.data.validate()
        self.graph.validate()
        self.training.validate()
        if self.swanlab.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError("invalid swanlab mode")
        if require_models:
            for name, path in (
                ("ESM-C", self.paths.esmc_model),
                ("TCR-BERT", self.paths.tcr_bert_model),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(f"{name} checkpoint directory not found: {path}")

    def graph_dir(self) -> Path:
        return self.paths.graph_root / f"fold_{self.runtime.fold}"

    def residue_cache_dir(self) -> Path:
        return self.paths.residue_cache_root / f"fold_{self.runtime.fold}"

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _section(raw: Mapping[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def load_stage13_config(path: str | Path) -> Stage13ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Stage13 YAML root must be a mapping")
    base_dir = config_path.parent
    path_values = _section(raw, "paths", set(Stage13Paths.__dataclass_fields__))
    required = {
        "project_root",
        "data_root",
        "residue_cache_root",
        "graph_root",
        "output_root",
        "swanlog_root",
    }
    missing = sorted(required - set(path_values))
    if missing:
        raise ValueError("Missing required paths: " + ", ".join(missing))
    path_values = {
        key: (None if value is None else _resolve_path(value, base_dir))
        for key, value in path_values.items()
    }
    config = Stage13ExperimentConfig(
        paths=Stage13Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", set(ModelConfig.__dataclass_fields__))),
        data=Stage13DataConfig(**_section(raw, "data", set(Stage13DataConfig.__dataclass_fields__))),
        graph=Stage13GraphConfig(**_section(raw, "graph", set(Stage13GraphConfig.__dataclass_fields__))),
        runtime=Stage13RuntimeConfig(**_section(raw, "runtime", set(Stage13RuntimeConfig.__dataclass_fields__))),
        training=Stage13TrainingConfig(**_section(raw, "training", set(Stage13TrainingConfig.__dataclass_fields__))),
        swanlab=SwanLabConfig(**_section(raw, "swanlab", set(SwanLabConfig.__dataclass_fields__))),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage13_overrides(
    config: Stage13ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
) -> Stage13ExperimentConfig:
    if fold is not None:
        config.runtime.fold = fold
    if seed is not None:
        config.runtime.seed = seed
    if device is not None:
        config.runtime.device = device
    if num_workers is not None:
        config.runtime.num_workers = num_workers
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    config.validate()
    return config
