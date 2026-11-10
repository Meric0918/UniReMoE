"""Strict configuration contract for Stage15 pairwise Shared Hyper-PMT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import CacheConfig, ModelConfig, SwanLabConfig, TrainingConfig


@dataclass
class Stage15Paths:
    project_root: Path
    data_root: Path
    benchmark_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class Stage15FusionConfig:
    num_layers: int = 2
    feedforward_dim: int = 512
    compatibility_hidden_dim: int = 512
    dropout: float = 0.1


@dataclass
class Stage15TrainingConfig(TrainingConfig):
    hyper_pmt_lr: float = 1e-4


@dataclass
class Stage15ExperimentConfig:
    paths: Stage15Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    fusion: Stage15FusionConfig = field(default_factory=Stage15FusionConfig)
    training: Stage15TrainingConfig = field(default_factory=Stage15TrainingConfig)
    swanlab: SwanLabConfig = field(
        default_factory=lambda: SwanLabConfig(
            group="plm-stage15-fold{fold}",
            experiment_prefix="PLM-Stage15",
            tags=[
                "ESMC-300M", "TCR-BERT", "stage15", "PH", "PT",
                "shared-hyper-pmt", "frozen-plm", "scratch-downstream",
            ],
        )
    )
    source_path: Optional[Path] = None

    def validate(self, *, require_models: bool = False) -> None:
        model = self.model
        training = self.training
        fusion = self.fusion
        if model.model_dim % model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if model.peptide_min_length > model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        for name in ("peptide_max_length", "hla_max_length", "tcr_max_length"):
            if getattr(model, name) < 1:
                raise ValueError(f"model.{name} must be positive")
        if self.cache.dtype != "float16":
            raise ValueError("cache.dtype must be float16 for the HDF5 cache")
        if self.cache.shard_size < 1 or self.cache.precompute_batch_size < 1:
            raise ValueError("cache sizes must be positive")
        if fusion.num_layers != 2:
            raise ValueError("fusion.num_layers must equal 2")
        if fusion.feedforward_dim < 1 or fusion.compatibility_hidden_dim < 1:
            raise ValueError("fusion hidden dimensions must be positive")
        if not 0.0 <= fusion.dropout < 1.0:
            raise ValueError("fusion.dropout must be in [0, 1)")
        for name in (
            "batch_size", "online_batch_size", "warmup_epochs_per_task",
            "max_rounds", "early_stopping_patience",
        ):
            if getattr(training, name) < 1:
                raise ValueError(f"training.{name} must be positive")
        if training.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        for name in (
            "shared_peptide_adapter_lr", "hla_adapter_lr", "tcr_adapter_lr",
            "phla_pooling_lr", "ptcr_pooling_lr", "phla_classifier_lr",
            "ptcr_classifier_lr", "hyper_pmt_lr",
        ):
            value = getattr(training, name)
            if value is None or value <= 0:
                raise ValueError(f"training.{name} must be positive for Stage15")
        if training.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if training.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("training.mixed_precision must be none, fp16, or bf16")
        if training.phla_group_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("Unsupported pHLA group weighting")
        if training.ptcr_group_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("Unsupported pTCR group weighting")
        if self.swanlab.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError("Unsupported SwanLab mode")
        if require_models:
            for name, path in (
                ("ESM-C", self.paths.esmc_model),
                ("TCR-BERT", self.paths.tcr_bert_model),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(f"{name} checkpoint directory not found: {path}")

    def cache_dir(self) -> Path:
        return self.paths.cache_root / f"fold_{self.training.fold}"

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.training.fold}"
            / f"seed_{self.training.seed}"
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


def _section(raw: Mapping[str, Any], name: str, fields) -> dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - set(fields))
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def _resolve(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_stage15_config(path: str | Path) -> Stage15ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load Stage15 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage15 YAML root must be a mapping")
    allowed = {"paths", "model", "cache", "fusion", "training", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("Unknown Stage15 config sections: " + ", ".join(unexpected))
    path_values = _section(raw, "paths", Stage15Paths.__dataclass_fields__)
    required = {
        "project_root", "data_root", "benchmark_root", "cache_root",
        "output_root", "swanlog_root",
    }
    missing = sorted(required - set(path_values))
    if missing:
        raise ValueError("Missing Stage15 paths: " + ", ".join(missing))
    path_values = {
        key: _resolve(value, config_path.parent)
        for key, value in path_values.items()
    }
    config = Stage15ExperimentConfig(
        paths=Stage15Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        cache=CacheConfig(**_section(raw, "cache", CacheConfig.__dataclass_fields__)),
        fusion=Stage15FusionConfig(**_section(
            raw, "fusion", Stage15FusionConfig.__dataclass_fields__
        )),
        training=Stage15TrainingConfig(**_section(
            raw, "training", Stage15TrainingConfig.__dataclass_fields__
        )),
        swanlab=SwanLabConfig(**_section(
            raw, "swanlab", SwanLabConfig.__dataclass_fields__
        )),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage15_overrides(
    config: Stage15ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    swanlab_mode: str | None = None,
    device: str | None = None,
    num_workers: int | None = None,
) -> Stage15ExperimentConfig:
    if fold is not None:
        config.training.fold = fold
    if seed is not None:
        config.training.seed = seed
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    if device is not None:
        config.training.device = device
    if num_workers is not None:
        config.training.num_workers = num_workers
    config.validate()
    return config


__all__ = [
    "Stage15ExperimentConfig",
    "Stage15FusionConfig",
    "Stage15Paths",
    "Stage15TrainingConfig",
    "apply_stage15_overrides",
    "load_stage15_config",
]
