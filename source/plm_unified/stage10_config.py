"""Standalone configuration for Stage-10 teacher-student retention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import CacheConfig, ModelConfig, SwanLabConfig


class Stage10Condition(str, Enum):
    BASELINE = "baseline"
    HIER_ONLY = "hier_only"
    PT_IMP_ONLY = "pt_imp_only"
    FULL = "full"


@dataclass
class Stage10Paths:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")
    stage3_checkpoint: Path = Path(
        "trained_model/PLM_Triplet/fold_4/seed_2026/"
        "triplet_best_20260802_215908_285880.pt"
    )


@dataclass
class Stage10RuntimeConfig:
    fold: int = 4
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 2


@dataclass
class Stage10TrainingConfig:
    data_subdir: str = "data_triple/fold4"
    train_filename: str = "train_merged_fold4.csv"
    val_filename: str = "val_merged_fold4.csv"
    reject_noncanonical_sequences: bool = True
    batch_size: int = 256
    eval_batch_size: int = 256
    max_epochs: int = 15
    min_epochs: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    pht_loss_weight: float = 1.0
    phla_loss_weight: float = 0.3
    ptcr_loss_weight: float = 0.2
    pht_score_weight: float = 0.4
    phla_score_weight: float = 0.2
    ptcr_score_weight: float = 0.4
    constraint_margin: float = 0.05
    hierarchy_lambda: float = 0.05
    pt_implication_lambda: float = 0.03
    constraint_ramp_epochs: int = 5
    joint_degradation_tolerance: float = 0.002
    adapter_lr: float = 2.5e-5
    pretrained_attention_lr: float = 5e-5
    triplet_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    log_every_steps: int = 50
    keep_loss_weight: float = 1.0
    phla_keep_weight: float = 1.0
    ptcr_keep_weight: float = 1.0
    distillation_temperature: float = 1.0
    legacy_task_degradation_tolerance: float = 0.002

    @property
    def selection_weights(self) -> tuple[float, float, float]:
        return (
            self.pht_score_weight,
            self.phla_score_weight,
            self.ptcr_score_weight,
        )

    @property
    def conditions(self) -> tuple[Stage10Condition, ...]:
        return tuple(Stage10Condition)


@dataclass
class Stage10ExperimentConfig:
    paths: Stage10Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    runtime: Stage10RuntimeConfig = field(default_factory=Stage10RuntimeConfig)
    stage10: Stage10TrainingConfig = field(default_factory=Stage10TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    @property
    def training(self) -> Stage10RuntimeConfig:
        """Compatibility view used by the shared experiment tracker."""

        return self.runtime

    def validate(self, *, require_files: bool = False) -> None:
        if self.runtime.fold != 4:
            raise ValueError("Stage 10 v1 is fixed to runtime.fold=4")
        if self.runtime.seed != 2026:
            raise ValueError("Stage 10 v1 is fixed to runtime.seed=2026")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")

        values = self.stage10
        approved_data = {
            "data_subdir": "data_triple/fold4",
            "train_filename": "train_merged_fold4.csv",
            "val_filename": "val_merged_fold4.csv",
        }
        for name, expected in approved_data.items():
            if getattr(values, name) != expected:
                raise ValueError(
                    f"Stage 10 v1 requires stage10.{name}={expected!r}"
                )
        if values.batch_size < 4 or values.batch_size % 4:
            raise ValueError(
                "stage10.batch_size must be positive and divisible by 4"
            )
        for name in (
            "eval_batch_size",
            "max_epochs",
            "min_epochs",
            "early_stopping_patience",
            "constraint_ramp_epochs",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage10.{name} must be positive")
        if values.min_epochs > values.max_epochs:
            raise ValueError("stage10.min_epochs cannot exceed max_epochs")
        for name in (
            "pht_loss_weight",
            "phla_loss_weight",
            "ptcr_loss_weight",
            "constraint_margin",
            "hierarchy_lambda",
            "pt_implication_lambda",
            "adapter_lr",
            "pretrained_attention_lr",
            "triplet_module_lr",
            "keep_loss_weight",
            "phla_keep_weight",
            "ptcr_keep_weight",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage10.{name} cannot be negative")
        if values.distillation_temperature <= 0:
            raise ValueError(
                "stage10.distillation_temperature must be positive"
            )
        if values.legacy_task_degradation_tolerance < 0:
            raise ValueError(
                "stage10 legacy task degradation tolerance cannot be negative"
            )
        if abs(sum(values.selection_weights) - 1.0) > 1e-8:
            raise ValueError("Stage-10 selection weights must sum to 1")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "stage10.mixed_precision must be none, fp16, or bf16"
            )
        if self.cache.dtype != "float16":
            raise ValueError("Stage-10 shared cache dtype must be float16")
        if require_files:
            for name, path in (
                ("Stage-3 teacher checkpoint", self.paths.stage3_checkpoint),
                ("Stage-10 train CSV", self.train_path()),
                ("Stage-10 validation CSV", self.val_path()),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"{name} not found: {path}")

    def train_path(self) -> Path:
        return (
            self.paths.data_root
            / self.stage10.data_subdir
            / self.stage10.train_filename
        )

    def val_path(self) -> Path:
        return (
            self.paths.data_root
            / self.stage10.data_subdir
            / self.stage10.val_filename
        )

    def cache_dir(self, source_sha256: str) -> Path:
        return (
            self.paths.cache_root
            / "plm_stage8"
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
            / source_sha256[:12]
        )

    def run_dir(self, condition: Stage10Condition | str) -> Path:
        value = Stage10Condition(condition).value
        return (
            self.paths.output_root
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
            / value
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


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
    return (
        (base_dir / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def load_stage10_config(path: str | Path) -> Stage10ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage-10 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage-10 YAML root must be a mapping")
    allowed = {"paths", "model", "cache", "runtime", "stage10", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(
            "Unknown Stage-10 config sections: " + ", ".join(unexpected)
        )
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage10Paths.__dataclass_fields__)
    required = {
        "project_root",
        "data_root",
        "cache_root",
        "output_root",
        "swanlog_root",
    }
    missing = required - set(path_values)
    if missing:
        raise ValueError(
            "Missing Stage-10 paths: " + ", ".join(sorted(missing))
        )
    path_values = {
        key: _resolve(value, base_dir) for key, value in path_values.items()
    }
    config = Stage10ExperimentConfig(
        paths=Stage10Paths(**path_values),
        model=ModelConfig(
            **_section(raw, "model", ModelConfig.__dataclass_fields__)
        ),
        cache=CacheConfig(
            **_section(raw, "cache", CacheConfig.__dataclass_fields__)
        ),
        runtime=Stage10RuntimeConfig(
            **_section(
                raw,
                "runtime",
                Stage10RuntimeConfig.__dataclass_fields__,
            )
        ),
        stage10=Stage10TrainingConfig(
            **_section(
                raw,
                "stage10",
                Stage10TrainingConfig.__dataclass_fields__,
            )
        ),
        swanlab=SwanLabConfig(
            **_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)
        ),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage10_overrides(
    config: Stage10ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
    stage3_checkpoint: str | Path | None = None,
) -> Stage10ExperimentConfig:
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
    if stage3_checkpoint is not None:
        config.paths.stage3_checkpoint = (
            Path(stage3_checkpoint).expanduser().resolve()
        )
    config.validate()
    return config
