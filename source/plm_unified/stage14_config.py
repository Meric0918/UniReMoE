"""Configuration contract for the isolated Stage-14 hypergraph model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ModelConfig, SwanLabConfig
from .stage8_config import Stage8RuntimeConfig, Stage8TrainingConfig


STAGE14_STAGE1_SHA256 = (
    "2e870c1470144fc17d035763ab97eb154d58f1635925c8bf78dfe26361d95c5f"
)
STAGE14_APPROVED_STAGE1_SHA256S = frozenset({
    STAGE14_STAGE1_SHA256,
    "ea95033e9e228a9e8b514ee802bd27d1911b02b22447157473fcb477b9722d59",
})


class Stage14Condition(str, Enum):
    BASELINE = "baseline"
    FULL = "full"


STAGE14_TRAINING_CONDITIONS = (
    Stage14Condition.BASELINE,
    Stage14Condition.FULL,
)


@dataclass
class Stage14Paths:
    project_root: Path
    data_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")
    stage1_checkpoint: Path = Path(
        "trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/"
        "best_joint_20260920_151134_400917.pt"
    )
    stage1_sha256: str = STAGE14_STAGE1_SHA256


@dataclass
class Stage14FusionConfig:
    """Architecture-only settings for the shared PMT fusion encoder."""

    num_layers: int = 2
    feedforward_dim: int = 512
    compatibility_hidden_dim: int = 512
    dropout: float = 0.1


@dataclass
class Stage14TrainingConfig(Stage8TrainingConfig):
    """Stage-8-compatible data and optimization schedule for Stage14."""

    @property
    def conditions(self) -> tuple[Stage14Condition, ...]:
        return STAGE14_TRAINING_CONDITIONS


@dataclass
class Stage14ExperimentConfig:
    paths: Stage14Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: Stage14FusionConfig = field(default_factory=Stage14FusionConfig)
    runtime: Stage8RuntimeConfig = field(default_factory=Stage8RuntimeConfig)
    stage14: Stage14TrainingConfig = field(default_factory=Stage14TrainingConfig)
    swanlab: SwanLabConfig = field(
        default_factory=lambda: SwanLabConfig(
            group="plm-stage14-fold{fold}",
            experiment_prefix="PLM-Stage14",
            tags=[
                "ESMC-300M", "TCR-BERT", "stage14", "hypergraph",
                "hyper-pmt", "four-type", "constraints",
            ],
        )
    )
    source_path: Optional[Path] = None

    @property
    def training(self) -> Stage8RuntimeConfig:
        return self.runtime

    @property
    def stage8(self) -> Stage14TrainingConfig:
        """Compatibility view used by the shared Stage-8 data/loss helpers."""

        return self.stage14

    def validate(self, *, require_files: bool = False) -> None:
        if (self.runtime.fold, self.runtime.seed) != (1, 42):
            raise ValueError("Stage14 is fixed to runtime.fold=1 and runtime.seed=42")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.paths.stage1_sha256 not in STAGE14_APPROVED_STAGE1_SHA256S:
            raise ValueError("paths.stage1_sha256 is not an approved Stage1 source")
        if self.fusion.num_layers != 2:
            raise ValueError("fusion.num_layers must equal 2")
        if self.fusion.feedforward_dim < 1 or self.fusion.compatibility_hidden_dim < 1:
            raise ValueError("fusion hidden dimensions must be positive")
        if not 0.0 <= self.fusion.dropout < 1.0:
            raise ValueError("fusion.dropout must be in [0, 1)")
        values = self.stage14
        binding_contract = {
            "stage8a_epochs": 2,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "binary_replay_ratio": 0.25,
            "peptide_unfrozen_layers": 2,
            "hla_unfrozen_layers": 2,
            "tcr_unfrozen_layers": 2,
            "min_rounds": 5,
            "max_rounds": 100,
            "max_epochs": 100,
            "min_epochs": 5,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "constraint_ramp_epochs": 5,
            "pht_score_weight": 0.4,
            "phla_score_weight": 0.2,
            "ptcr_score_weight": 0.4,
        }
        for name, expected in binding_contract.items():
            if getattr(values, name) != expected:
                raise ValueError(f"stage14.{name} must equal {expected}")
        if values.data_mode != "pht_class_dirs":
            raise ValueError("Stage14 requires data_mode='pht_class_dirs'")
        if values.data_subdir != "PHT/classes":
            raise ValueError("Stage14 requires data_subdir='PHT/classes'")
        if tuple(values.class_dirs) != ("PHT_000", "PHT_010", "PHT_100", "PHT_111"):
            raise ValueError("Stage14 requires the canonical four PHT class directories")
        if values.train_filename != "train.csv" or values.val_filename != "val.csv":
            raise ValueError("Stage14 requires train.csv and val.csv")
        if values.batch_size % 4:
            raise ValueError("stage14.batch_size must be divisible by 4")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage14.min_rounds cannot exceed max_rounds")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("stage14.mixed_precision must be none, fp16, or bf16")
        if abs(sum(values.selection_weights) - 1.0) > 1e-8:
            raise ValueError("stage14 selection weights must sum to one")
        for name in (
            "eval_batch_size", "min_rounds", "max_rounds",
            "early_stopping_patience", "peptide_unfrozen_layers",
            "hla_unfrozen_layers", "tcr_unfrozen_layers",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage14.{name} must be positive")
        if require_files:
            required = {"Stage-1 checkpoint": self.paths.stage1_checkpoint}
            required.update({
                f"PHT train {kind}": path
                for kind, path in zip(values.class_dirs, self.train_paths())
            })
            required.update({
                f"PHT val {kind}": path
                for kind, path in zip(values.class_dirs, self.val_paths())
            })
            for name, path in required.items():
                if not path.is_file():
                    raise FileNotFoundError(f"{name} not found: {path}")

    def data_dir(self) -> Path:
        return self.paths.data_root / self.stage14.data_subdir

    def train_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir() / kind / self.stage14.train_filename
            for kind in self.stage14.class_dirs
        )

    def val_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.data_dir() / kind / self.stage14.val_filename
            for kind in self.stage14.class_dirs
        )

    def run_dir(self, condition: Stage14Condition | str | None = None) -> Path:
        root = (
            self.paths.output_root / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
        )
        return root if condition is None else root / Stage14Condition(condition).value

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


def _serialise(value):
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


def _resolve(value, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_stage14_config(path: str | Path) -> Stage14ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment contract.
        raise RuntimeError("PyYAML is required to load Stage-14 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage-14 YAML root must be a mapping")
    allowed = {"paths", "model", "fusion", "runtime", "stage14", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("Unknown Stage-14 config sections: " + ", ".join(unexpected))
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage14Paths.__dataclass_fields__)
    required = {"project_root", "data_root", "output_root", "swanlog_root"}
    missing = required - set(path_values)
    if missing:
        raise ValueError("Missing Stage-14 paths: " + ", ".join(sorted(missing)))
    path_values = {
        key: value if key == "stage1_sha256" else _resolve(value, base_dir)
        for key, value in path_values.items()
    }
    config = Stage14ExperimentConfig(
        paths=Stage14Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        fusion=Stage14FusionConfig(**_section(raw, "fusion", Stage14FusionConfig.__dataclass_fields__)),
        runtime=Stage8RuntimeConfig(**_section(raw, "runtime", Stage8RuntimeConfig.__dataclass_fields__)),
        stage14=Stage14TrainingConfig(**_section(raw, "stage14", Stage14TrainingConfig.__dataclass_fields__)),
        swanlab=SwanLabConfig(**_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage14_overrides(
    config: Stage14ExperimentConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
    stage1_checkpoint: str | Path | None = None,
) -> Stage14ExperimentConfig:
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
    if stage1_checkpoint is not None:
        config.paths.stage1_checkpoint = Path(stage1_checkpoint).expanduser().resolve()
    config.validate()
    return config
