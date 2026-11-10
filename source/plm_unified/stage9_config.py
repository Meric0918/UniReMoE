"""Standalone configuration for Stage-9 paired contrastive fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ModelConfig, SwanLabConfig


@dataclass
class Stage9Paths:
    project_root: Path
    data_root: Path
    output_root: Path
    swanlog_root: Path
    stage3_checkpoint: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class Stage9RuntimeConfig:
    fold: int = 4
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class Stage9TrainingConfig:
    data_subdir: str = "data_stage9"
    pair_filename: str = "pht_pairs.csv"
    supplement_filename: str | None = None
    legacy_triplet_data_subdir: str = "data_triple"
    legacy_triplet_train_filename: str = "train_set.csv"
    legacy_triplet_val_filename: str = "val_set.csv"
    legacy_triplet_test_filename: str = "test_set.csv"
    reject_noncanonical_sequences: bool = True
    split_seed: int = 2026
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    ph_balanced_sampling: bool = True
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    stage9a_epochs: int = 2
    min_rounds: int = 3
    max_rounds: int = 10
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    replay_ratio: float = 0.25
    temperature: float = 0.07
    contrastive_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    legacy_degradation_tolerance: float = 0.01
    projection_dim: int = 128
    projection_dropout: float = 0.1
    # Retained for compatibility with older Stage-9 configs/checkpoints. The
    # current Stage-9 design keeps all three PLM encoders frozen.
    peptide_unfrozen_layers: int = 0
    hla_unfrozen_layers: int = 0
    tcr_unfrozen_layers: int = 0
    # Deprecated compatibility fields; Stage9 encoders are frozen and these
    # learning rates are intentionally unused by the Stage9 optimizer.
    plm_top_lr: float = 5e-6
    plm_lower_lr: float = 2.5e-6
    adapter_lr: float = 2.5e-5
    pretrained_module_lr: float = 5e-5
    contrastive_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_optimizer_steps: int = 10

    @property
    def split_ratios(self) -> tuple[float, float, float]:
        return self.train_fraction, self.val_fraction, self.test_fraction


@dataclass
class Stage9ExperimentConfig:
    paths: Stage9Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: Stage9RuntimeConfig = field(default_factory=Stage9RuntimeConfig)
    stage9: Stage9TrainingConfig = field(default_factory=Stage9TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    @property
    def training(self) -> Stage9RuntimeConfig:
        """Compatibility view consumed by the shared tracker."""

        return self.runtime

    def validate(self, *, require_files: bool = False) -> None:
        values = self.stage9
        if self.runtime.fold < 1:
            raise ValueError("runtime.fold must be positive")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if values.temperature <= 0:
            raise ValueError("stage9.temperature must be positive")
        if any(fraction <= 0 for fraction in values.split_ratios) or abs(
            sum(values.split_ratios) - 1.0
        ) > 1e-8:
            raise ValueError("stage9 split fractions must be positive and sum to 1")
        for name in (
            "micro_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "stage9a_epochs",
            "min_rounds",
            "max_rounds",
            "early_stopping_patience",
            "projection_dim",
            "log_every_optimizer_steps",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage9.{name} must be positive")
        for name in (
            "peptide_unfrozen_layers",
            "hla_unfrozen_layers",
            "tcr_unfrozen_layers",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage9.{name} cannot be negative")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage9.min_rounds cannot exceed max_rounds")
        for name in (
            "early_stopping_min_delta",
            "replay_ratio",
            "contrastive_loss_weight",
            "classification_loss_weight",
            "legacy_degradation_tolerance",
            "projection_dropout",
            "weight_decay",
        ):
            if getattr(values, name) < 0:
                raise ValueError(f"stage9.{name} cannot be negative")
        if values.contrastive_loss_weight == 0:
            raise ValueError("stage9.contrastive_loss_weight must be positive")
        for name in (
            "plm_top_lr",
            "plm_lower_lr",
            "adapter_lr",
            "pretrained_module_lr",
            "contrastive_module_lr",
            "gradient_clip_norm",
        ):
            if getattr(values, name) <= 0:
                raise ValueError(f"stage9.{name} must be positive")
        if not 0 <= values.scheduler_warmup_ratio < 1:
            raise ValueError("stage9.scheduler_warmup_ratio must be in [0, 1)")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("stage9.mixed_precision must be none, fp16, or bf16")
        if require_files:
            for name, path in (
                ("Stage-3 checkpoint", self.paths.stage3_checkpoint),
                ("Stage-9 pair CSV", self.pair_path()),
                ("Stage-3 train CSV", self.legacy_triplet_path("train")),
                ("ESM-C checkpoint", self.paths.esmc_model),
                ("TCR-BERT checkpoint", self.paths.tcr_bert_model),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"{name} not found: {path}")
            supplement = self.supplement_path()
            if supplement is not None and not supplement.exists():
                raise FileNotFoundError(
                    f"Stage-9 supplement CSV not found: {supplement}"
                )

    def pair_path(self) -> Path:
        return self.paths.data_root / self.stage9.data_subdir / self.stage9.pair_filename

    def supplement_path(self) -> Path | None:
        if not self.stage9.supplement_filename:
            return None
        return (
            self.paths.data_root
            / self.stage9.data_subdir
            / self.stage9.supplement_filename
        )

    def legacy_triplet_path(self, split: str) -> Path:
        names = {
            "train": self.stage9.legacy_triplet_train_filename,
            "val": self.stage9.legacy_triplet_val_filename,
            "test": self.stage9.legacy_triplet_test_filename,
        }
        if split not in names:
            raise ValueError(f"Unknown legacy triplet split: {split}")
        return self.paths.data_root / self.stage9.legacy_triplet_data_subdir / names[split]

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


def _section(raw: Mapping[str, Any], name: str, fields) -> dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - set(fields))
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def _resolve(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_stage9_config(path: str | Path) -> Stage9ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage-9 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage-9 YAML root must be a mapping")
    allowed = {"paths", "model", "runtime", "stage9", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("Unknown Stage-9 config sections: " + ", ".join(unexpected))
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage9Paths.__dataclass_fields__)
    required = {
        "project_root",
        "data_root",
        "output_root",
        "swanlog_root",
        "stage3_checkpoint",
    }
    missing = sorted(required - set(path_values))
    if missing:
        raise ValueError("Missing Stage-9 paths: " + ", ".join(missing))
    path_values = {
        key: _resolve(value, base_dir) for key, value in path_values.items()
    }
    config = Stage9ExperimentConfig(
        paths=Stage9Paths(**path_values),
        model=ModelConfig(
            **_section(raw, "model", ModelConfig.__dataclass_fields__)
        ),
        runtime=Stage9RuntimeConfig(
            **_section(raw, "runtime", Stage9RuntimeConfig.__dataclass_fields__)
        ),
        stage9=Stage9TrainingConfig(
            **_section(raw, "stage9", Stage9TrainingConfig.__dataclass_fields__)
        ),
        swanlab=SwanLabConfig(
            **_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)
        ),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage9_overrides(
    config: Stage9ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
    stage3_checkpoint: str | Path | None = None,
    pair_csv: str | Path | None = None,
) -> Stage9ExperimentConfig:
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
    if pair_csv is not None:
        pair_path = Path(pair_csv).expanduser().resolve()
        try:
            data_subdir = pair_path.parent.relative_to(config.paths.data_root)
        except ValueError:
            data_subdir = pair_path.parent
        config.stage9.data_subdir = str(data_subdir)
        config.stage9.pair_filename = pair_path.name
        config.stage9.supplement_filename = None
    config.validate()
    return config


def restore_stage9_checkpoint_config(
    config: Stage9ExperimentConfig,
    saved_config: Mapping[str, Any],
) -> Stage9ExperimentConfig:
    """Restore model/training semantics while keeping local runtime paths."""

    saved_model = saved_config.get("model")
    saved_stage9 = saved_config.get("stage9")
    saved_runtime = saved_config.get("runtime")
    if not isinstance(saved_model, Mapping) or not isinstance(
        saved_stage9, Mapping
    ) or not isinstance(saved_runtime, Mapping):
        raise ValueError("Stage-9 checkpoint has an incomplete saved config")
    config.model = ModelConfig(**dict(saved_model))
    config.stage9 = Stage9TrainingConfig(**dict(saved_stage9))
    config.runtime.fold = int(saved_runtime["fold"])
    config.runtime.seed = int(saved_runtime["seed"])
    config.validate()
    return config
