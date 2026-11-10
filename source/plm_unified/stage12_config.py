"""Configuration contract for Stage-12 retraining aligned with Stage 8."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ModelConfig, SwanLabConfig


STAGE1_SHA256 = "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96"
STAGE8_REFERENCE_SHA256 = "575af59f232ed1e8d86aaba4286953eb0d8194243054f492092c1725effad624"


@dataclass
class Stage12Paths:
    project_root: Path
    data_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")
    stage1_checkpoint: Path = Path(
        "trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/"
        "best_joint_20260824_174202_510837.pt"
    )
    stage1_sha256: str = STAGE1_SHA256
    stage8_reference_checkpoint: Path = Path(
        "trained_model/PLM_Stage8/fold_1/seed_42/full/"
        "stage8_best_20260827_110610_283819.pt"
    )
    stage8_reference_sha256: str = STAGE8_REFERENCE_SHA256


@dataclass
class Stage12RuntimeConfig:
    fold: int = 1
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2


@dataclass
class Stage12TrainingConfig:
    data_subdir: str = "PHT/classes"
    class_dirs: tuple[str, ...] = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    reject_noncanonical_sequences: bool = True
    condition: str = "full"
    stage12a_epochs: int = 2
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    phla_replay_ratio: float = 0.25
    ptcr_replay_ratio: float = 0.25
    peptide_unfrozen_layers: int = 2
    hla_unfrozen_layers: int = 2
    tcr_unfrozen_layers: int = 2
    min_rounds: int = 5
    max_rounds: int = 15
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    phla_loss_weight: float = 0.3
    ptcr_loss_weight: float = 0.2
    pht_loss_weight: float = 1.0
    constraint_margin: float = 0.05
    hierarchy_lambda: float = 0.05
    pt_implication_lambda: float = 0.03
    constraint_ramp_rounds: int = 5
    adapter_lr: float = 0.000025
    pretrained_attention_lr: float = 0.00005
    triplet_module_lr: float = 0.0001
    weight_decay: float = 0.01
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_steps: int = 50
    fixed_threshold: float = 0.5
    pt_selection_splits: tuple[str, ...] = (
        "val_unseen_pair", "val_unseen_peptide", "val_unseen_tcr"
    )
    ph_selection_splits: tuple[str, ...] = (
        "val_unseen_hla", "val_unseen_pair", "val_unseen_peptide"
    )
    pht_joint_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    task_selection_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

    @property
    def normalized_task_weights(self) -> tuple[float, float, float]:
        """Return nominal PH/PT/PHT shares of the joint supervised loss."""

        total = self.phla_loss_weight + self.ptcr_loss_weight + self.pht_loss_weight
        return (
            self.phla_loss_weight / total,
            self.ptcr_loss_weight / total,
            self.pht_loss_weight / total,
        )

    @property
    def selection_weights(self) -> tuple[float, float, float]:
        """Weights for the overall Stage12 PHT/PH/PT selection score."""

        return self.task_selection_weights


@dataclass
class Stage12ExperimentConfig:
    paths: Stage12Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: Stage12RuntimeConfig = field(default_factory=Stage12RuntimeConfig)
    stage12: Stage12TrainingConfig = field(default_factory=Stage12TrainingConfig)
    swanlab: SwanLabConfig = field(
        default_factory=lambda: SwanLabConfig(
            group="plm-stage12-fold{fold}",
            experiment_prefix="PLM-Stage12",
            tags=["ESMC-300M", "TCR-BERT", "stage12", "stage8-aligned", "constraints"],
        )
    )
    source_path: Optional[Path] = None

    @property
    def training(self) -> Stage12RuntimeConfig:
        return self.runtime

    @property
    def stage8(self) -> Stage12TrainingConfig:
        """Compatibility view used by the unchanged Stage8 model builder."""

        return self.stage12

    def validate(self, *, require_files: bool = False) -> None:
        values = self.stage12
        exact_contract = {
            "condition": "full",
            "stage12a_epochs": 2,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "phla_replay_ratio": 0.25,
            "ptcr_replay_ratio": 0.25,
            "phla_loss_weight": 0.3,
            "ptcr_loss_weight": 0.2,
            "pht_loss_weight": 1.0,
            "pht_joint_weights": (1 / 3, 1 / 3, 1 / 3),
            "task_selection_weights": (1 / 3, 1 / 3, 1 / 3),
        }
        for name, expected in exact_contract.items():
            if getattr(values, name) != expected:
                raise ValueError(f"stage12.{name} must equal {expected}")
        if (self.runtime.fold, self.runtime.seed) != (1, 42):
            raise ValueError("Stage12 is fixed to fold=1 and seed=42")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.paths.stage1_sha256 != STAGE1_SHA256:
            raise ValueError("paths.stage1_sha256 does not match the registered source")
        if self.paths.stage8_reference_sha256 != STAGE8_REFERENCE_SHA256:
            raise ValueError("paths.stage8_reference_sha256 does not match the registered reference")
        if tuple(values.class_dirs) != ("PHT_000", "PHT_010", "PHT_100", "PHT_111"):
            raise ValueError("stage12.class_dirs must contain the canonical four PHT classes")
        if values.data_subdir != "PHT/classes":
            raise ValueError("stage12.data_subdir must equal PHT/classes")
        if values.min_rounds > values.max_rounds:
            raise ValueError("stage12.min_rounds cannot exceed max_rounds")
        for name in (
            "eval_batch_size", "min_rounds", "max_rounds", "early_stopping_patience",
            "peptide_unfrozen_layers", "hla_unfrozen_layers", "tcr_unfrozen_layers",
            "constraint_ramp_rounds",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage12.{name} must be positive")
        if self.model.model_dim % self.model.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if values.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("stage12.mixed_precision must be none, fp16, or bf16")
        if abs(sum(values.pht_joint_weights) - 1.0) > 1e-8:
            raise ValueError("stage12.pht_joint_weights must sum to one")
        if abs(sum(values.task_selection_weights) - 1.0) > 1e-8:
            raise ValueError("stage12.task_selection_weights must sum to one")
        if require_files:
            required = {
                "Stage1 checkpoint": self.paths.stage1_checkpoint,
                "Stage8 reference checkpoint": self.paths.stage8_reference_checkpoint,
            }
            required.update({f"PHT train {kind}": path for kind, path in zip(values.class_dirs, self.train_paths())})
            required.update({f"PHT val {kind}": path for kind, path in zip(values.class_dirs, self.val_paths())})
            for name, path in required.items():
                if not path.is_file():
                    raise FileNotFoundError(f"{name} not found: {path}")

    def data_dir(self) -> Path:
        return self.paths.data_root / self.stage12.data_subdir

    def train_paths(self) -> tuple[Path, ...]:
        return tuple(self.data_dir() / kind / self.stage12.train_filename for kind in self.stage12.class_dirs)

    def val_paths(self) -> tuple[Path, ...]:
        return tuple(self.data_dir() / kind / self.stage12.val_filename for kind in self.stage12.class_dirs)

    def run_dir(self) -> Path:
        return self.paths.output_root / f"fold_{self.runtime.fold}" / f"seed_{self.runtime.seed}" / "full"

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


def load_stage12_config(path: str | Path) -> Stage12ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load Stage12 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage12 YAML root must be a mapping")
    allowed = {"paths", "model", "runtime", "stage12", "swanlab"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("Unknown Stage12 config sections: " + ", ".join(unexpected))
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage12Paths.__dataclass_fields__)
    required = {"project_root", "data_root", "output_root", "swanlog_root"}
    missing = required - set(path_values)
    if missing:
        raise ValueError("Missing Stage12 paths: " + ", ".join(sorted(missing)))
    path_values = {
        key: value if key.endswith("sha256") else _resolve(value, base_dir)
        for key, value in path_values.items()
    }
    training_values = _section(raw, "stage12", Stage12TrainingConfig.__dataclass_fields__)
    for name in (
        "class_dirs", "pt_selection_splits", "ph_selection_splits",
        "pht_joint_weights", "task_selection_weights",
    ):
        if name in training_values:
            training_values[name] = tuple(training_values[name])
    config = Stage12ExperimentConfig(
        paths=Stage12Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        runtime=Stage12RuntimeConfig(**_section(raw, "runtime", Stage12RuntimeConfig.__dataclass_fields__)),
        stage12=Stage12TrainingConfig(**training_values),
        swanlab=SwanLabConfig(**_section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage12_overrides(
    config: Stage12ExperimentConfig,
    *,
    device: str | None = None,
    num_workers: int | None = None,
    swanlab_mode: str | None = None,
) -> Stage12ExperimentConfig:
    if device is not None:
        config.runtime.device = device
    if num_workers is not None:
        config.runtime.num_workers = num_workers
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    config.validate()
    return config
