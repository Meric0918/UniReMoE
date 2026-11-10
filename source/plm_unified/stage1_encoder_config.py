"""Independent configuration for Stage1 sequence-encoder experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .config import (
    ExperimentConfig, ModelConfig, TrainingConfig, SwanLabConfig,
    load_config as load_base_config,
)


@dataclass
class Stage1EncoderModelConfig:
    peptide_input_dim: int = 960
    hla_input_dim: int = 960
    tcr_input_dim: int = 768
    model_dim: int = 256
    num_attention_heads: int = 8
    feedforward_dim: int = 1024
    encoder_num_layers: int = 3
    encoder_dropout: float = 0.15
    attention_dropout: float = 0.15
    classifier_dropout: float = 0.25
    peptide_min_length: int = 8
    peptide_max_length: int = 15
    hla_max_length: int = 34
    tcr_max_length: int = 34


@dataclass
class Stage1EncoderTrainingConfig:
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
    encoder_lr: float = 3e-4
    cross_attention_lr: float = 2e-4
    pooling_lr: float = 3e-4
    classifier_lr: float = 3e-4
    # Task-specific rates override the module defaults when set.
    shared_peptide_encoder_lr: Optional[float] = None
    hla_encoder_lr: Optional[float] = None
    tcr_encoder_lr: Optional[float] = None
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
class Stage1EncoderSwanLabConfig(SwanLabConfig):
    group: str = "plm-stage1-encoder-fold{fold}"
    experiment_prefix: str = "PLM-Stage1-Encoder"
    tags: list[str] = field(default_factory=lambda: [
        "ESMC-300M", "TCR-BERT", "FGM", "stage1_encoder", "self-attention",
    ])


@dataclass
class Stage1EncoderExperimentConfig(ExperimentConfig):
    model: Stage1EncoderModelConfig = field(default_factory=Stage1EncoderModelConfig)
    training: Stage1EncoderTrainingConfig = field(default_factory=Stage1EncoderTrainingConfig)
    swanlab: Stage1EncoderSwanLabConfig = field(default_factory=Stage1EncoderSwanLabConfig)

    def validate(self, require_models: bool = False) -> None:
        for name in (
            "peptide_input_dim", "hla_input_dim", "tcr_input_dim", "model_dim",
            "num_attention_heads", "feedforward_dim", "encoder_num_layers",
        ):
            value = getattr(self.model, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        for name in ("encoder_dropout", "attention_dropout", "classifier_dropout"):
            if not 0 <= getattr(self.model, name) < 1:
                raise ValueError(f"model.{name} must be in [0, 1)")
        if self.training.encoder_lr <= 0:
            raise ValueError("training.encoder_lr must be positive")
        for name in ("shared_peptide_encoder_lr", "hla_encoder_lr", "tcr_encoder_lr"):
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive when set")
        # Reuse the full Stage1 validation contract through a temporary config.
        # Adapter fields exist only in this validation view, never in saved settings.
        model_values = {
            key: value for key, value in asdict(self.model).items()
            if key in ModelConfig.__dataclass_fields__
        }
        model_values["adapter_dropout"] = self.model.encoder_dropout
        training_values = {
            key.replace("encoder", "adapter"): value
            for key, value in asdict(self.training).items()
        }
        ExperimentConfig(
            paths=self.paths,
            model=ModelConfig(**model_values),
            cache=self.cache,
            training=TrainingConfig(**training_values),
            stage2=self.stage2,
            triplet=self.triplet,
            stage5=self.stage5,
            swanlab=self.swanlab,
            source_path=self.source_path,
        ).validate(require_models=require_models)

    def to_dict(self) -> dict:
        values = super().to_dict()
        for section in ("stage2", "triplet", "stage5"):
            values.pop(section, None)
        return values


def load_stage1_encoder_config(path: str | Path) -> Stage1EncoderExperimentConfig:
    return load_base_config(
        path,
        config_type=Stage1EncoderExperimentConfig,
        model_type=Stage1EncoderModelConfig,
        training_type=Stage1EncoderTrainingConfig,
        swanlab_type=Stage1EncoderSwanLabConfig,
    )
