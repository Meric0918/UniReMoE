"""Configuration for the PHT-only Stage Adapter lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .config import ModelConfig, SwanLabConfig


@dataclass
class StageAdapterPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path
    tcr_bert_model: Path
    stage1_checkpoint: Path | None = None
    stage1_sha256: str | None = None


@dataclass
class StageAdapterRuntimeConfig:
    fold: int = 1
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2


@dataclass
class StageAdapterTrainingConfig:
    initialization: str = "checkpoint"
    data_subdir: str = "PHT/classes"
    class_dirs: tuple[str, ...] = (
        "PHT_000", "PHT_010", "PHT_100", "PHT_111"
    )
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    reject_noncanonical_sequences: bool = True
    residual_adapter_dim: int = 64
    stage_adapter_a_epochs: int = 2
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    min_rounds: int = 5
    max_rounds: int = 100
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    adapter_lr: float = 2.5e-5
    pht_module_lr: float = 1e-4
    encoder_lr: float = 2.5e-5
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_steps: int = 50


@dataclass
class StageAdapterExperimentConfig:
    paths: StageAdapterPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: StageAdapterRuntimeConfig = field(
        default_factory=StageAdapterRuntimeConfig
    )
    stage_adapter: StageAdapterTrainingConfig = field(
        default_factory=StageAdapterTrainingConfig
    )
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Path | None = None

    @property
    def training(self) -> StageAdapterRuntimeConfig:
        return self.runtime

    def data_dir(self) -> Path:
        return self.paths.data_root / self.stage_adapter.data_subdir

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
        )

    def warmup_dir(self) -> Path:
        return self.run_dir() / "warmup"

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))

    def validate(self, *, require_files: bool = False) -> None:
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        values = self.stage_adapter
        if values.initialization not in {"checkpoint", "random", "random_binding"}:
            raise ValueError(
                "stage_adapter.initialization must be checkpoint, random, or random_binding"
            )
        if values.initialization == "checkpoint":
            digest = self.paths.stage1_sha256
            if (
                not self.paths.stage1_checkpoint
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
            ):
                raise ValueError(
                    "Checkpoint initialization requires stage1_checkpoint and a SHA256 hex digest"
                )
        expected_dirs = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
        if tuple(values.class_dirs) != expected_dirs:
            raise ValueError(
                "stage_adapter.class_dirs must be PHT_000, PHT_010, "
                "PHT_100, PHT_111 in canonical order"
            )
        for name in (
            "residual_adapter_dim", "stage_adapter_a_epochs",
            "micro_batch_size", "eval_batch_size",
            "gradient_accumulation_steps", "min_rounds", "max_rounds",
            "early_stopping_patience", "log_every_steps",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage_adapter.{name} must be positive")
        if values.micro_batch_size % 4:
            raise ValueError("stage_adapter.micro_batch_size must be divisible by 4")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage_adapter.min_rounds cannot exceed max_rounds")
        for name in (
            "early_stopping_min_delta", "adapter_lr", "pht_module_lr", "encoder_lr",
            "weight_decay", "scheduler_warmup_ratio", "gradient_clip_norm",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage_adapter.{name} cannot be negative")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if require_files:
            required = []
            if values.initialization == "checkpoint":
                required.append(("Stage-1 checkpoint", self.paths.stage1_checkpoint))
            for class_dir in values.class_dirs:
                root = self.data_dir() / class_dir
                required.extend(
                    (
                        (f"{class_dir} train CSV", root / values.train_filename),
                        (f"{class_dir} validation CSV", root / values.val_filename),
                    )
                )
            for label, path in required:
                if not path.is_file():
                    raise FileNotFoundError(f"{label} not found: {path}")


def _serialise(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
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


def _resolve(value, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_stage_adapter_config(
    path: str | Path, *, initialization: str | None = None
) -> StageAdapterExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load Stage Adapter config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage Adapter YAML root must be a mapping")
    allowed = {"paths", "model", "runtime", "stage_adapter", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(
            "Unknown Stage Adapter config sections: " + ", ".join(unexpected)
        )
    path_values = _section(raw, "paths", StageAdapterPaths.__dataclass_fields__)
    optional_paths = {"stage1_checkpoint", "stage1_sha256"}
    missing = set(StageAdapterPaths.__dataclass_fields__) - optional_paths - set(path_values)
    if missing:
        raise ValueError("Missing Stage Adapter paths: " + ", ".join(sorted(missing)))
    path_values = {
        key: value if key == "stage1_sha256" or value is None else _resolve(value, config_path.parent)
        for key, value in path_values.items()
    }
    training_values = _section(raw, "stage_adapter", StageAdapterTrainingConfig.__dataclass_fields__)
    if initialization is not None:
        training_values["initialization"] = initialization
    config = StageAdapterExperimentConfig(
        paths=StageAdapterPaths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        runtime=StageAdapterRuntimeConfig(
            **_section(raw, "runtime", StageAdapterRuntimeConfig.__dataclass_fields__)
        ),
        stage_adapter=StageAdapterTrainingConfig(**training_values),
        swanlab=SwanLabConfig(
            **_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)
        ),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage_adapter_overrides(
    config: StageAdapterExperimentConfig,
    *,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
) -> StageAdapterExperimentConfig:
    if device is not None:
        config.runtime.device = device
    if num_workers is not None:
        config.runtime.num_workers = num_workers
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    config.validate()
    return config
