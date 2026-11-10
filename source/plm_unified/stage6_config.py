"""Standalone configuration for the Stage-6 PH coverage/conflict ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .config import (
    ModelConfig,
    PathsConfig,
    SwanLabConfig,
    TrainingConfig,
    _resolve_path,
    _serialise,
)

Stage6Condition = Literal[
    "stage6_phla_only_matched",
    "stage6_joint_phla_full",
    "stage6_phla_only_full",
]

STAGE6_CONDITIONS: tuple[Stage6Condition, ...] = (
    "stage6_phla_only_matched",
    "stage6_joint_phla_full",
    "stage6_phla_only_full",
)


@dataclass
class Stage6Config:
    """Controls one cell of the Stage-6 task-mode x PH-coverage design."""

    condition: Stage6Condition = "stage6_phla_only_matched"
    data_subdir: str = "data_triple"
    merged_filename: str = "merged_pseudo.csv"
    train_filename: str = "train_set.csv"
    val_filename: str = "val_set.csv"
    test_filename: str = "test_set.csv"
    verify_merged_manifest: bool = True
    strict_split_cleanup: bool = True
    reject_noncanonical_sequences: bool = True
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    logical_steps_per_epoch: int = 45_533
    stage1a_epochs: int = 5
    stage1b_epochs: int = 10
    pht_loss_weight: float = 1.0
    phla_loss_weight: float = 0.3
    ptcr_loss_weight: float = 0.2
    triplet_score_weight: float = 0.6
    phla_score_weight: float = 0.2
    ptcr_score_weight: float = 0.2
    selection_min_delta: float = 1e-4
    adapter_lr: float = 2.5e-5
    pretrained_attention_lr: float = 5e-5
    triplet_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    fgm_epsilon: float = 1.0
    mixed_precision: str = "bf16"
    log_every_optimizer_steps: int = 10

    @property
    def task_mode(self) -> Literal["joint", "phla_only"]:
        return "joint" if self.condition == "stage6_joint_phla_full" else "phla_only"

    @property
    def phla_coverage(self) -> Literal["matched", "full"]:
        return (
            "matched"
            if self.condition == "stage6_phla_only_matched"
            else "full"
        )


@dataclass
class Stage6ExperimentConfig:
    """Minimum experiment surface needed by Stage 6 and SwanLab."""

    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stage6: Stage6Config = field(default_factory=Stage6Config)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    def validate(self, require_models: bool = False) -> None:
        if self.model.model_dim % self.model.num_attention_heads != 0:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        if self.training.fold < 1:
            raise ValueError("fold must be a positive integer")
        if self.training.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.training.threshold < 0 or self.training.threshold > 1:
            raise ValueError("threshold must be in [0, 1]")
        values = self.stage6
        if values.condition not in STAGE6_CONDITIONS:
            raise ValueError(
                "stage6.condition must be one of: " + ", ".join(STAGE6_CONDITIONS)
            )
        for name in (
            "micro_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "logical_steps_per_epoch",
            "stage1a_epochs",
            "stage1b_epochs",
            "log_every_optimizer_steps",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage6.{name} must be positive")
        for name in (
            "pht_loss_weight",
            "phla_loss_weight",
            "ptcr_loss_weight",
            "selection_min_delta",
            "weight_decay",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage6.{name} cannot be negative")
        if values.phla_loss_weight <= 0:
            raise ValueError("stage6.phla_loss_weight must be positive")
        if values.task_mode == "joint" and values.pht_loss_weight <= 0:
            raise ValueError("joint Stage-6 runs require positive pht_loss_weight")
        score_weights = (
            values.triplet_score_weight,
            values.phla_score_weight,
            values.ptcr_score_weight,
        )
        if any(weight < 0 for weight in score_weights):
            raise ValueError("stage6 score weights cannot be negative")
        if abs(sum(score_weights) - 1.0) > 1e-8:
            raise ValueError("stage6 score weights must sum to 1")
        for name in (
            "adapter_lr",
            "pretrained_attention_lr",
            "triplet_module_lr",
            "gradient_clip_norm",
            "fgm_epsilon",
        ):
            if getattr(values, name) <= 0:
                raise ValueError(f"stage6.{name} must be positive")
        if not 0 <= values.scheduler_warmup_ratio < 1:
            raise ValueError("stage6.scheduler_warmup_ratio must be in [0, 1)")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "stage6.mixed_precision must be one of: none, fp16, bf16"
            )
        for name in (
            "data_subdir",
            "merged_filename",
            "train_filename",
            "val_filename",
            "test_filename",
        ):
            value = getattr(values, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"stage6.{name} must be a non-empty string")
        if self.swanlab.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError(
                "swanlab.mode must be one of: online, local, offline, disabled"
            )
        if require_models:
            for name, path in (
                ("ESM-C", self.paths.esmc_model),
                ("TCR-BERT", self.paths.tcr_bert_model),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(
                        f"{name} checkpoint directory not found: {path}"
                    )

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / self.stage6.condition
            / f"fold_{self.training.fold}"
            / f"seed_{self.training.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


def _section(raw: Mapping[str, Any], name: str, allowed) -> dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def load_stage6_config(path: str | Path) -> Stage6ExperimentConfig:
    """Load Stage-6 YAML without adding Stage-6 fields to shared config.py."""

    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage-6 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The root YAML value must be a mapping")
    allowed_root = {"paths", "model", "training", "stage6", "swanlab"}
    unexpected_root = sorted(set(raw) - allowed_root)
    if unexpected_root:
        raise ValueError(
            "Unknown Stage-6 config sections: " + ", ".join(unexpected_root)
        )
    base_dir = config_path.parent
    path_values = _section(raw, "paths", PathsConfig.__dataclass_fields__)
    required_paths = {
        "project_root",
        "data_root",
        "cache_root",
        "output_root",
        "swanlog_root",
    }
    missing_paths = sorted(required_paths - set(path_values))
    if missing_paths:
        raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
    path_values = {
        key: _resolve_path(value, base_dir) for key, value in path_values.items()
    }
    config = Stage6ExperimentConfig(
        paths=PathsConfig(**path_values),
        model=ModelConfig(
            **_section(raw, "model", ModelConfig.__dataclass_fields__)
        ),
        training=TrainingConfig(
            **_section(raw, "training", TrainingConfig.__dataclass_fields__)
        ),
        stage6=Stage6Config(
            **_section(raw, "stage6", Stage6Config.__dataclass_fields__)
        ),
        swanlab=SwanLabConfig(
            **_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)
        ),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage6_overrides(
    config: Stage6ExperimentConfig,
    *,
    condition: Optional[Stage6Condition] = None,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
    swanlab_mode: Optional[str] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    logical_steps: Optional[int] = None,
) -> Stage6ExperimentConfig:
    if condition is not None:
        config.stage6.condition = condition
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
    if logical_steps is not None:
        config.stage6.logical_steps_per_epoch = logical_steps
    config.validate()
    return config
