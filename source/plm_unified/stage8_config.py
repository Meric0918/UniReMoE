"""Standalone configuration for Stage-8 constrained four-type training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import CacheConfig, ModelConfig, SwanLabConfig


class Stage8Condition(str, Enum):
    BASELINE = "baseline"
    HIER_ONLY = "hier_only"
    PT_IMP_ONLY = "pt_imp_only"
    FULL = "full"


@dataclass
class Stage8Paths:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")
    stage3_checkpoint: Path = Path(
        "trained_model/PLM_Triplet/fold_1/seed_42/best_round/"
        "stage3_best_joint_20260825_161220_330205.pt"
    )
    stage1_checkpoint: Path = Path(
        "trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/"
        "best_joint_20260824_174202_510837.pt"
    )
    stage1_sha256: str = (
        "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96"
    )


@dataclass
class Stage8RuntimeConfig:
    fold: int = 1
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2


@dataclass
class Stage8TrainingConfig:
    data_mode: str = "pht_class_dirs"
    data_subdir: str = "PHT/classes"
    class_dirs: tuple[str, ...] = (
        "PHT_000", "PHT_010", "PHT_100", "PHT_111"
    )
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    reject_noncanonical_sequences: bool = True
    stage8a_epochs: int = 2
    micro_batch_size: int = 4
    batch_size: int = 256
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    binary_replay_ratio: float = 0.25
    peptide_unfrozen_layers: int = 2
    hla_unfrozen_layers: int = 2
    tcr_unfrozen_layers: int = 2
    min_rounds: int = 5
    max_rounds: int = 15
    max_epochs: int = 15
    min_epochs: int = 5
    early_stopping_patience: int = 5
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
    gradient_checkpointing: bool = True
    log_every_steps: int = 50

    @property
    def selection_weights(self) -> tuple[float, float, float]:
        return (
            self.pht_score_weight,
            self.phla_score_weight,
            self.ptcr_score_weight,
        )

    @property
    def conditions(self) -> tuple[Stage8Condition, ...]:
        return tuple(Stage8Condition)


@dataclass
class Stage8ExperimentConfig:
    paths: Stage8Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    runtime: Stage8RuntimeConfig = field(default_factory=Stage8RuntimeConfig)
    stage8: Stage8TrainingConfig = field(default_factory=Stage8TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    @property
    def training(self) -> Stage8RuntimeConfig:
        """Compatibility view used by the shared experiment tracker."""
        return self.runtime

    def validate(self, *, require_files: bool = False) -> None:
        if self.runtime.fold != 1:
            raise ValueError("Stage 8 v1 is fixed to runtime.fold=1")
        if self.runtime.seed != 42:
            raise ValueError("Stage 8 v1 is fixed to runtime.seed=42")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        values = self.stage8
        if len(self.paths.stage1_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.paths.stage1_sha256.lower()
        ):
            raise ValueError("paths.stage1_sha256 must be a 64-character hexadecimal SHA256")
        if values.data_mode != "pht_class_dirs":
            raise ValueError("Stage 8 requires data_mode='pht_class_dirs'")
        if values.data_subdir != "PHT/classes":
            raise ValueError("Stage 8 requires data_subdir='PHT/classes'")
        expected_class_dirs = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
        class_dirs = tuple(values.class_dirs)
        if class_dirs != expected_class_dirs:
            raise ValueError(
                "Stage 8 requires all four class directories in canonical order: "
                "PHT_000, PHT_010, PHT_100, PHT_111"
            )
        if values.train_filename != "train.csv":
            raise ValueError("Stage 8 requires train_filename='train.csv'")
        if values.val_filename != "val.csv":
            raise ValueError("Stage 8 requires val_filename='val.csv'")
        binding_contract = {
            "stage8a_epochs": 2,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "binary_replay_ratio": 0.25,
            "peptide_unfrozen_layers": 2,
            "hla_unfrozen_layers": 2,
            "tcr_unfrozen_layers": 2,
        }
        for name, expected in binding_contract.items():
            if getattr(values, name) != expected:
                raise ValueError(f"stage8.{name} must equal {expected}")
        if values.batch_size < 4 or values.batch_size % 4:
            raise ValueError("stage8.batch_size must be positive and divisible by 4")
        for name in (
            "stage8a_epochs", "micro_batch_size", "eval_batch_size",
            "gradient_accumulation_steps", "peptide_unfrozen_layers",
            "hla_unfrozen_layers", "tcr_unfrozen_layers", "min_rounds",
            "max_rounds", "max_epochs", "min_epochs",
            "early_stopping_patience", "constraint_ramp_epochs",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage8.{name} must be positive")
        if values.min_epochs > values.max_epochs:
            raise ValueError("stage8.min_epochs cannot exceed max_epochs")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage8.min_rounds cannot exceed max_rounds")
        if not 0 < values.binary_replay_ratio <= 1:
            raise ValueError("stage8.binary_replay_ratio must be in (0, 1]")
        for name in (
            "pht_loss_weight", "phla_loss_weight", "ptcr_loss_weight",
            "constraint_margin", "hierarchy_lambda", "pt_implication_lambda",
            "adapter_lr", "pretrained_attention_lr", "triplet_module_lr",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage8.{name} cannot be negative")
        if abs(sum(values.selection_weights) - 1.0) > 1e-8:
            raise ValueError("Stage-8 selection weights must sum to 1")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("stage8.mixed_precision must be none, fp16, or bf16")
        if self.cache.dtype != "float16":
            raise ValueError("Stage-8 cache dtype must be float16")
        if require_files:
            required_files = [("Stage-1 checkpoint", self.paths.stage1_checkpoint)]
            required_files.extend(
                (f"Stage-8 train CSV ({class_dir})", path)
                for class_dir, path in zip(class_dirs, self.train_paths())
            )
            required_files.extend(
                (f"Stage-8 validation CSV ({class_dir})", path)
                for class_dir, path in zip(class_dirs, self.val_paths())
            )
            for name, path in required_files:
                if not path.is_file():
                    raise FileNotFoundError(f"{name} not found: {path}")

    def data_dir(self) -> Path:
        return self.paths.data_root / self.stage8.data_subdir

    def train_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir() / class_dir / self.stage8.train_filename
            for class_dir in self.stage8.class_dirs
        )

    def val_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir() / class_dir / self.stage8.val_filename
            for class_dir in self.stage8.class_dirs
        )

    def train_path(self) -> Path:
        """Return the first class path for legacy callers.

        Stage-8 execution must use ``train_paths()`` so that all four class
        directories are included.
        """
        return self.train_paths()[0]

    def val_path(self) -> Path:
        """Return the first class path for legacy callers."""
        return self.val_paths()[0]

    def cache_dir(self, source_sha256: str) -> Path:
        return (
            self.paths.cache_root / "plm_stage8" / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}" / source_sha256[:12]
        )

    def run_dir(self, condition: Stage8Condition | str) -> Path:
        value = Stage8Condition(condition).value
        return (
            self.paths.output_root / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}" / value
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
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_stage8_config(path: str | Path) -> Stage8ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage-8 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage-8 YAML root must be a mapping")
    allowed = {"paths", "model", "cache", "runtime", "stage8", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("Unknown Stage-8 config sections: " + ", ".join(unexpected))
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage8Paths.__dataclass_fields__)
    required = {"project_root", "data_root", "cache_root", "output_root", "swanlog_root"}
    missing = required - set(path_values)
    if missing:
        raise ValueError("Missing Stage-8 paths: " + ", ".join(sorted(missing)))
    path_values = {
        key: (
            value
            if key == "stage1_sha256"
            else _resolve(value, base_dir)
        )
        for key, value in path_values.items()
    }
    config = Stage8ExperimentConfig(
        paths=Stage8Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        cache=CacheConfig(**_section(raw, "cache", CacheConfig.__dataclass_fields__)),
        runtime=Stage8RuntimeConfig(**_section(raw, "runtime", Stage8RuntimeConfig.__dataclass_fields__)),
        stage8=Stage8TrainingConfig(**_section(raw, "stage8", Stage8TrainingConfig.__dataclass_fields__)),
        swanlab=SwanLabConfig(**_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage8_overrides(
    config: Stage8ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
    stage3_checkpoint: str | Path | None = None,
) -> Stage8ExperimentConfig:
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
        config.paths.stage3_checkpoint = Path(stage3_checkpoint).expanduser().resolve()
    config.validate()
    return config
