"""Typed configuration for the unified PLM stage-1/stage-2 pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class PathsConfig:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class ModelConfig:
    peptide_input_dim: int = 960
    hla_input_dim: int = 960
    tcr_input_dim: int = 768
    adapter_hidden_dim: int = 256
    model_dim: int = 256
    num_attention_heads: int = 8
    feedforward_dim: int = 1024
    adapter_dropout: float = 0.15
    attention_dropout: float = 0.15
    classifier_dropout: float = 0.25
    peptide_min_length: int = 8
    peptide_max_length: int = 15
    hla_max_length: int = 34
    tcr_max_length: int = 34


@dataclass
class CacheConfig:
    dtype: str = "float16"
    shard_size: int = 16384
    precompute_batch_size: int = 32
    compression: Optional[str] = None
    hla_splits: list[str] = field(
        default_factory=lambda: ["train", "val"]
    )
    tcr_splits: list[str] = field(
        default_factory=lambda: ["train", "val"]
    )


@dataclass
class TrainingConfig:
    fold: int = 1
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 3407, 2026])
    batch_size: int = 1024
    online_batch_size: int = 32
    num_workers: int = 2
    warmup_epochs_per_task: int = 2
    max_rounds: int = 100
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    adapter_lr: float = 3e-4
    cross_attention_lr: float = 2e-4
    pooling_lr: float = 3e-4
    classifier_lr: float = 3e-4
    # Optional Stage1-v2 task-aware learning rates. When omitted, the legacy
    # module-level learning rates above are used for backwards compatibility.
    shared_peptide_adapter_lr: Optional[float] = None
    hla_adapter_lr: Optional[float] = None
    tcr_adapter_lr: Optional[float] = None
    phla_cross_attention_lr: Optional[float] = None
    ptcr_cross_attention_lr: Optional[float] = None
    phla_pooling_lr: Optional[float] = None
    ptcr_pooling_lr: Optional[float] = None
    phla_classifier_lr: Optional[float] = None
    ptcr_classifier_lr: Optional[float] = None
    weight_decay: float = 1e-2
    no_decay_bias_and_norm: bool = True
    scheduler_warmup_ratio: float = 0.05
    scheduler_total_rounds: Optional[int] = None
    scheduler_warmup_rounds: Optional[int] = None
    scheduler_min_lr_ratio: float = 0.0
    gradient_clip_norm: float = 1.0
    fgm_epsilon: float = 1.0
    phla_fgm_epsilon: Optional[float] = None
    ptcr_fgm_epsilon: Optional[float] = None
    # Keep FGM as a regulariser rather than letting it dominate the clean
    # supervised objective.  pTCR uses the smaller weight because its
    # unseen-pair validation is more sensitive to adversarial perturbations.
    phla_adversarial_loss_weight: float = 0.25
    ptcr_adversarial_loss_weight: float = 0.10
    adversarial_warmup_rounds: int = 0
    phla_label_smoothing: float = 0.0
    ptcr_label_smoothing: float = 0.0
    phla_group_weighting: str = "none"
    ptcr_group_weighting: str = "none"
    group_weight_min: float = 0.25
    group_weight_max: float = 4.0
    phla_early_stopping_patience: Optional[int] = None
    ptcr_early_stopping_patience: Optional[int] = None
    round_early_stopping_patience: Optional[int] = None
    task_step_ratio_enabled: bool = False
    phla_task_step_ratio: float = 1.0
    ptcr_task_step_ratio: float = 1.0
    retention_eval_enabled: bool = False
    retention_tolerance: float = 0.01
    # A discrimination improvement is accepted only when validation
    # calibration does not regress beyond these tolerances.  Keep this opt-in
    # for backwards-compatible loading of older Stage-1 configurations.
    selection_calibration_guard_enabled: bool = False
    selection_nll_tolerance: float = 0.05
    selection_ece_tolerance: float = 0.02
    threshold: float = 0.5
    mixed_precision: str = "bf16"
    log_every_steps: int = 50
    pin_memory: bool = True
    persistent_workers: bool = False
    device: str = "auto"
    best_umap_enabled: bool = False
    best_umap_max_samples_per_split: int = 1000
    best_umap_random_state: int = 42
    best_umap_n_neighbors: int = 30
    best_umap_min_dist: float = 0.1


@dataclass
class Stage2Config:
    """Progressive top-layer fine-tuning from chapter 15."""

    micro_batch_size: int = 8
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 16
    stage2a_rounds: int = 2
    stage2b_min_rounds: int = 5
    stage2b_max_rounds: int = 20
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    peptide_unfrozen_layers: int = 2
    hla_unfrozen_layers: int = 2
    tcr_unfrozen_layers: int = 2
    peptide_top_lr: float = 1e-5
    peptide_lower_lr: float = 5e-6
    hla_top_lr: float = 1e-5
    hla_lower_lr: float = 5e-6
    tcr_top_lr: float = 1e-5
    tcr_lower_lr: float = 5e-6
    adapter_lr: float = 5e-5
    cross_attention_lr: float = 1e-4
    pooling_lr: float = 1e-4
    classifier_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_optimizer_steps: int = 1


@dataclass
class TripletConfig:
    """Stage-3 HLA-peptide-TCR fine-tuning configuration."""

    data_mode: str = "legacy_triplet"
    data_subdir: str = "data_triple"
    merged_filename: str = "merged_pseudo.csv"
    train_filename: str = "train_set.csv"
    val_filename: str = "val_set.csv"
    test_filename: str = "test_set.csv"
    validation_subdir: str = "benchmarks"
    validation_splits: list[str] = field(default_factory=lambda: ["val"])
    selection_splits: list[str] = field(default_factory=lambda: ["val"])
    sampling_strategy: str = "legacy_shuffle"
    pht_label_types: list[str] = field(
        default_factory=lambda: ["111", "100", "010", "000"]
    )
    pht_class_dirs: list[str] = field(default_factory=list)
    verify_merged_manifest: bool = True
    strict_split_cleanup: bool = True
    reject_noncanonical_sequences: bool = True
    micro_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    head_warmup_epochs: int = 2
    min_rounds: int = 3
    max_rounds: int = 100
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    binary_replay_ratio: float = 0.25
    triplet_score_weight: float = 1.0 / 3.0
    phla_score_weight: float = 1.0 / 3.0
    ptcr_score_weight: float = 1.0 / 3.0
    peptide_unfrozen_layers: int = 2
    hla_unfrozen_layers: int = 2
    tcr_unfrozen_layers: int = 2
    plm_top_lr: float = 5e-6
    plm_lower_lr: float = 2.5e-6
    adapter_lr: float = 2.5e-5
    pretrained_attention_lr: float = 5e-5
    triplet_module_lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    log_every_optimizer_steps: int = 10


@dataclass
class Stage5Config:
    """Scratch downstream training with independently normalized task losses."""

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


@dataclass
class SwanLabConfig:
    project: str = "unifyimmun"
    workspace: Optional[str] = None
    mode: str = "online"
    group: str = "plm-stage1-fold1"
    experiment_prefix: str = "PLM-Stage1"
    tags: list[str] = field(
        default_factory=lambda: ["ESMC-300M", "TCR-BERT", "FGM", "stage1"]
    )


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    triplet: TripletConfig = field(default_factory=TripletConfig)
    stage5: Stage5Config = field(default_factory=Stage5Config)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    def validate(self, require_models: bool = False) -> None:
        if self.model.model_dim % self.model.num_attention_heads != 0:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        for name in ("peptide_max_length", "hla_max_length", "tcr_max_length"):
            if getattr(self.model, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.cache.dtype != "float16":
            raise ValueError("cache.dtype must be float16 for the HDF5 cache")
        if self.cache.shard_size < 1:
            raise ValueError("cache.shard_size must be positive")
        if self.training.fold < 1:
            raise ValueError("fold must be a positive integer")
        if (
            self.training.batch_size < 1
            or self.training.online_batch_size < 1
            or self.cache.precompute_batch_size < 1
        ):
            raise ValueError("batch sizes must be positive")
        if self.training.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.training.warmup_epochs_per_task < 0:
            raise ValueError("warmup_epochs_per_task cannot be negative")
        if self.training.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.training.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.training.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        optional_lrs = (
            "shared_peptide_adapter_lr",
            "hla_adapter_lr",
            "tcr_adapter_lr",
            "phla_cross_attention_lr",
            "ptcr_cross_attention_lr",
            "phla_pooling_lr",
            "ptcr_pooling_lr",
            "phla_classifier_lr",
            "ptcr_classifier_lr",
        )
        for name in optional_lrs:
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive when set")
        if self.training.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.training.scheduler_total_rounds is not None and self.training.scheduler_total_rounds < 1:
            raise ValueError("scheduler_total_rounds must be positive when set")
        if self.training.scheduler_warmup_rounds is not None and self.training.scheduler_warmup_rounds < 0:
            raise ValueError("scheduler_warmup_rounds cannot be negative")
        if not 0 <= self.training.scheduler_min_lr_ratio <= 1:
            raise ValueError("scheduler_min_lr_ratio must be in [0, 1]")
        scheduler_round_budget = (
            self.training.scheduler_total_rounds
            if self.training.scheduler_total_rounds is not None
            else self.training.max_rounds
        )
        if (
            self.training.scheduler_warmup_rounds is not None
            and self.training.scheduler_warmup_rounds > scheduler_round_budget
        ):
            raise ValueError("scheduler_warmup_rounds cannot exceed scheduler_total_rounds")
        for name in ("phla_fgm_epsilon", "ptcr_fgm_epsilon"):
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive when set")
        if self.training.phla_adversarial_loss_weight < 0 or self.training.ptcr_adversarial_loss_weight < 0:
            raise ValueError("adversarial loss weights cannot be negative")
        if self.training.adversarial_warmup_rounds < 0:
            raise ValueError("adversarial_warmup_rounds cannot be negative")
        for name in ("phla_label_smoothing", "ptcr_label_smoothing"):
            value = getattr(self.training, name)
            if not 0 <= value < 1:
                raise ValueError(f"training.{name} must be in [0, 1)")
        if self.training.phla_group_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("training.phla_group_weighting must be none or inverse_sqrt")
        if self.training.ptcr_group_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("training.ptcr_group_weighting must be none or inverse_sqrt")
        if self.training.group_weight_min <= 0 or self.training.group_weight_max < self.training.group_weight_min:
            raise ValueError("group weight bounds are invalid")
        for name in (
            "phla_early_stopping_patience",
            "ptcr_early_stopping_patience",
            "round_early_stopping_patience",
        ):
            value = getattr(self.training, name)
            if value is not None and value < 1:
                raise ValueError(f"training.{name} must be positive when set")
        for name in ("phla_task_step_ratio", "ptcr_task_step_ratio"):
            if getattr(self.training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.training.retention_tolerance < 0:
            raise ValueError("retention_tolerance cannot be negative")
        if self.training.selection_nll_tolerance < 0:
            raise ValueError("selection_nll_tolerance cannot be negative")
        if self.training.selection_ece_tolerance < 0:
            raise ValueError("selection_ece_tolerance cannot be negative")
        if self.training.best_umap_max_samples_per_split < 2:
            raise ValueError("best_umap_max_samples_per_split must be at least 2")
        if self.training.best_umap_n_neighbors < 2:
            raise ValueError("best_umap_n_neighbors must be at least 2")
        if self.training.best_umap_min_dist < 0:
            raise ValueError("best_umap_min_dist cannot be negative")
        if not 0 <= self.training.scheduler_warmup_ratio < 1:
            raise ValueError("scheduler_warmup_ratio must be in [0, 1)")
        if self.training.fgm_epsilon <= 0:
            raise ValueError("fgm_epsilon must be positive")
        if self.training.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: none, fp16, bf16")
        stage2 = self.stage2
        for name in (
            "micro_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "stage2a_rounds",
            "stage2b_min_rounds",
            "stage2b_max_rounds",
            "early_stopping_patience",
            "peptide_unfrozen_layers",
            "hla_unfrozen_layers",
            "tcr_unfrozen_layers",
            "log_every_optimizer_steps",
        ):
            if getattr(stage2, name) < 1:
                raise ValueError(f"stage2.{name} must be positive")
        if stage2.stage2b_min_rounds > stage2.stage2b_max_rounds:
            raise ValueError(
                "stage2.stage2b_min_rounds cannot exceed stage2.stage2b_max_rounds"
            )
        if not 0 <= stage2.scheduler_warmup_ratio < 1:
            raise ValueError("stage2.scheduler_warmup_ratio must be in [0, 1)")
        if stage2.early_stopping_min_delta < 0:
            raise ValueError("stage2.early_stopping_min_delta cannot be negative")
        for name in (
            "peptide_top_lr",
            "peptide_lower_lr",
            "hla_top_lr",
            "hla_lower_lr",
            "tcr_top_lr",
            "tcr_lower_lr",
            "adapter_lr",
            "cross_attention_lr",
            "pooling_lr",
            "classifier_lr",
            "gradient_clip_norm",
        ):
            if getattr(stage2, name) <= 0:
                raise ValueError(f"stage2.{name} must be positive")
        if stage2.weight_decay < 0:
            raise ValueError("stage2.weight_decay cannot be negative")
        if stage2.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "stage2.mixed_precision must be one of: none, fp16, bf16"
            )
        triplet = self.triplet
        for name in (
            "micro_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "head_warmup_epochs",
            "min_rounds",
            "max_rounds",
            "early_stopping_patience",
            "peptide_unfrozen_layers",
            "hla_unfrozen_layers",
            "tcr_unfrozen_layers",
            "log_every_optimizer_steps",
        ):
            if getattr(triplet, name) < 1:
                raise ValueError(f"triplet.{name} must be positive")
        if triplet.min_rounds > triplet.max_rounds:
            raise ValueError("triplet.min_rounds cannot exceed triplet.max_rounds")
        if triplet.early_stopping_min_delta < 0:
            raise ValueError(
                "triplet.early_stopping_min_delta cannot be negative"
            )
        if not 0 < triplet.binary_replay_ratio <= 1:
            raise ValueError("triplet.binary_replay_ratio must be in (0, 1]")
        score_weights = (
            triplet.triplet_score_weight,
            triplet.phla_score_weight,
            triplet.ptcr_score_weight,
        )
        if any(weight < 0 for weight in score_weights):
            raise ValueError("triplet score weights cannot be negative")
        if abs(sum(score_weights) - 1.0) > 1e-8:
            raise ValueError("triplet score weights must sum to 1")
        for name in (
            "plm_top_lr",
            "plm_lower_lr",
            "adapter_lr",
            "pretrained_attention_lr",
            "triplet_module_lr",
            "gradient_clip_norm",
        ):
            if getattr(triplet, name) <= 0:
                raise ValueError(f"triplet.{name} must be positive")
        if triplet.weight_decay < 0:
            raise ValueError("triplet.weight_decay cannot be negative")
        if not 0 <= triplet.scheduler_warmup_ratio < 1:
            raise ValueError(
                "triplet.scheduler_warmup_ratio must be in [0, 1)"
            )
        if triplet.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "triplet.mixed_precision must be one of: none, fp16, bf16"
            )
        if triplet.data_mode not in {"legacy_triplet", "pht_multitask"}:
            raise ValueError(
                "triplet.data_mode must be legacy_triplet or pht_multitask"
            )
        valid_pht_label_types = {"111", "100", "010", "000"}
        if not isinstance(triplet.pht_label_types, (list, tuple)):
            raise ValueError(
                "triplet.pht_label_types must be a non-empty list of strings"
            )
        if not triplet.pht_label_types:
            raise ValueError("triplet.pht_label_types cannot be empty")
        if len(set(triplet.pht_label_types)) != len(triplet.pht_label_types):
            raise ValueError("triplet.pht_label_types cannot contain duplicates")
        unknown_pht_types = sorted(
            set(triplet.pht_label_types) - valid_pht_label_types
        )
        if unknown_pht_types or any(
            not isinstance(label_type, str)
            or len(label_type) != 3
            or set(label_type) - {"0", "1"}
            for label_type in triplet.pht_label_types
        ):
            invalid_pht_types = sorted(
                set(unknown_pht_types)
                | {
                    label_type
                    for label_type in triplet.pht_label_types
                    if not isinstance(label_type, str)
                    or len(label_type) != 3
                    or set(label_type) - {"0", "1"}
                }
            )
            raise ValueError(
                "triplet.pht_label_types contains unsupported values: "
                + ", ".join(map(str, invalid_pht_types))
            )
        if triplet.sampling_strategy not in {
            "legacy_shuffle",
            "balanced_four_type",
            "balanced_selected_types",
        }:
            raise ValueError(
                "triplet.sampling_strategy must be legacy_shuffle or "
                "balanced_four_type or balanced_selected_types"
            )
        if triplet.data_mode == "pht_multitask":
            if not isinstance(triplet.pht_class_dirs, (list, tuple)):
                raise ValueError(
                    "triplet.pht_class_dirs must be a list of directory names"
                )
            if any(
                not isinstance(directory, str) or not directory.strip()
                for directory in triplet.pht_class_dirs
            ):
                raise ValueError(
                    "triplet.pht_class_dirs must contain non-empty strings"
                )
            if triplet.sampling_strategy not in {
                "balanced_four_type",
                "balanced_selected_types",
            }:
                raise ValueError(
                    "pht_multitask requires a balanced PHT sampling strategy"
                )
            if len(triplet.pht_label_types) < 2:
                raise ValueError(
                    "pht_multitask requires at least two pht_label_types"
                )
            if not triplet.validation_splits:
                raise ValueError("triplet.validation_splits cannot be empty")
            if not triplet.selection_splits:
                raise ValueError("triplet.selection_splits cannot be empty")
            unknown_selection = sorted(
                set(triplet.selection_splits) - set(triplet.validation_splits)
            )
            if unknown_selection:
                raise ValueError(
                    "triplet.selection_splits must be contained in "
                    "validation_splits: " + ", ".join(unknown_selection)
                )
        for name in (
            "data_subdir",
            "merged_filename",
            "train_filename",
            "val_filename",
            "test_filename",
            "validation_subdir",
        ):
            value = getattr(triplet, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"triplet.{name} must be a non-empty string")
        stage5 = self.stage5
        for name in (
            "micro_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "stage1a_epochs",
            "stage1b_epochs",
            "log_every_optimizer_steps",
        ):
            if getattr(stage5, name) < 1:
                raise ValueError(f"stage5.{name} must be positive")
        for name in (
            "pht_loss_weight",
            "phla_loss_weight",
            "ptcr_loss_weight",
            "selection_min_delta",
            "weight_decay",
        ):
            if getattr(stage5, name) < 0:
                raise ValueError(f"stage5.{name} cannot be negative")
        if stage5.pht_loss_weight == 0:
            raise ValueError("stage5.pht_loss_weight must be positive")
        score_weights = (
            stage5.triplet_score_weight,
            stage5.phla_score_weight,
            stage5.ptcr_score_weight,
        )
        if any(weight < 0 for weight in score_weights):
            raise ValueError("stage5 score weights cannot be negative")
        if abs(sum(score_weights) - 1.0) > 1e-8:
            raise ValueError("stage5 score weights must sum to 1")
        for name in (
            "adapter_lr",
            "pretrained_attention_lr",
            "triplet_module_lr",
            "gradient_clip_norm",
            "fgm_epsilon",
        ):
            if getattr(stage5, name) <= 0:
                raise ValueError(f"stage5.{name} must be positive")
        if not 0 <= stage5.scheduler_warmup_ratio < 1:
            raise ValueError(
                "stage5.scheduler_warmup_ratio must be in [0, 1)"
            )
        if stage5.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "stage5.mixed_precision must be one of: none, fp16, bf16"
            )
        for name in (
            "data_subdir",
            "merged_filename",
            "train_filename",
            "val_filename",
            "test_filename",
        ):
            value = getattr(stage5, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"stage5.{name} must be a non-empty string")
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
                    raise FileNotFoundError(f"{name} checkpoint directory not found: {path}")

    def cache_dir(self) -> Path:
        return self.paths.cache_root / f"fold_{self.training.fold}"

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.training.fold}"
            / f"seed_{self.training.seed}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return _serialise(asdict(self))


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _section(
    raw: Mapping[str, Any], name: str, allowed: Iterable[str]
) -> Dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def load_config(
    path: str | Path,
    *,
    config_type: type = ExperimentConfig,
    model_type: type = ModelConfig,
    training_type: type = TrainingConfig,
    swanlab_type: type = SwanLabConfig,
) -> ExperimentConfig:
    """Load a YAML config and resolve every relative path against that file."""

    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the experiment config. "
            "Install requirements-plm.txt."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The root YAML value must be a mapping")

    base_dir = config_path.parent
    path_values = _section(raw, "paths", PathsConfig.__dataclass_fields__)
    required_paths = {"project_root", "data_root", "cache_root", "output_root", "swanlog_root"}
    missing_paths = sorted(required_paths - set(path_values))
    if missing_paths:
        raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
    path_values = {
        key: _resolve_path(value, base_dir) for key, value in path_values.items()
    }

    model_values = _section(raw, "model", model_type.__dataclass_fields__)
    cache_values = _section(raw, "cache", CacheConfig.__dataclass_fields__)
    training_values = _section(raw, "training", training_type.__dataclass_fields__)
    stage2_values = _section(raw, "stage2", Stage2Config.__dataclass_fields__)
    triplet_values = _section(raw, "triplet", TripletConfig.__dataclass_fields__)
    stage5_values = _section(raw, "stage5", Stage5Config.__dataclass_fields__)
    swanlab_values = _section(raw, "swanlab", swanlab_type.__dataclass_fields__)

    config = config_type(
        paths=PathsConfig(**path_values),
        model=model_type(**model_values),
        cache=CacheConfig(**cache_values),
        training=training_type(**training_values),
        stage2=Stage2Config(**stage2_values),
        triplet=TripletConfig(**triplet_values),
        stage5=Stage5Config(**stage5_values),
        swanlab=swanlab_type(**swanlab_values),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_overrides(
    config: ExperimentConfig,
    *,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
    swanlab_mode: Optional[str] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> ExperimentConfig:
    """Apply common CLI overrides in place and validate the result."""

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
