"""Strict, self-contained experiment configuration for Stages 16--18."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping
from typing import get_type_hints, get_origin, get_args, Union
from types import UnionType

from .config import ModelConfig, SwanLabConfig


@dataclass
class ExperimentPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path
    esmc_model: Path
    tcr_bert_model: Path
    stage1_checkpoint: Path
    stage1_sha256: str
    warmup_checkpoint: Path
    warmup_sha256: str
    teacher_checkpoint: Path
    teacher_sha256: str
    swanlog_root: Path


@dataclass
class Runtime:
    fold: int = 1
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0


@dataclass
class Training:
    rounds: int = 35
    scheduler_rounds: int = 100
    scheduler_total_steps: int | None = None
    scheduler_warmup_ratio: float = 0.05
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    eval_batch_size: int = 32
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    peptide_unfrozen_layers: int = 2
    hla_unfrozen_layers: int = 2
    tcr_unfrozen_layers: int = 2
    adapter_lr: float = 2.5e-5
    pretrained_attention_lr: float = 5e-5
    triplet_module_lr: float = 1e-4
    plm_top_lr: float = 5e-6
    plm_lower_lr: float = 2.5e-6
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    pht_loss_weight: float = 1.0
    phla_loss_weight: float = 0.3
    ptcr_loss_weight: float = 0.2
    constraint_margin: float = 0.05
    hierarchy_lambda: float = 0.05
    pt_implication_lambda: float = 0.03
    constraint_ramp_epochs: int = 5
    constraint_condition: str = "full"
    sampler_policy: str = "controlled_v1"
    checkpoint_every_steps: int = 0
    log_every_steps: int = 50
    min_rounds: int = 5
    early_stopping_patience: int = 20
    deployment_early_stop: bool = False
    monitor_rounds: tuple[int, ...] = (0, 1, 5, 10, 20, 35)
    monitor_per_split: int = 2048
    gradient_diagnostics: bool = True
    independent_ph_evaluation: bool = False
    debug_pair_limit: int | None = None
    debug_limit_per_type: int | None = None


@dataclass
class Selection:
    macro_tolerance: float = 0.01
    split_tolerance: float = 0.02
    pht_tolerance: float = 0.02
    min_delta: float = 1e-4


@dataclass
class ReplayPolicy:
    freeze_plm: bool = False
    protect_joint: bool = False
    protect_ph_peptide: bool = False
    binary_replay: bool = True
    ph_replay_ratio: float = 0.25
    pt_replay_ratio: float = 0.25
    group_weighting: bool = False
    label_smoothing: float = 0.0
    distillation_weight: float = 0.0
    temperature: float = 2.0


def condition_policy(stage: int, condition: str, overrides: dict) -> ReplayPolicy:
    policy = ReplayPolicy()
    if stage == 16:
        if condition != "full":
            raise ValueError(
                "Stage16 condition must be full (all three selectors run together)"
            )
    elif stage == 17:
        if condition not in {"S17-U", "S17-F", "S17-J", "S17-P", "S17-Z"}:
            raise ValueError("Unknown Stage17 condition")
        policy.freeze_plm = condition != "S17-U"
        policy.protect_joint = condition in {"S17-J", "S17-P", "S17-Z"}
        policy.protect_ph_peptide = condition == "S17-P"
        policy.binary_replay = condition != "S17-Z"
    elif stage == 18:
        if condition not in {
            "S18-R0",
            "S18-R1",
            "S18-W",
            "S18-S",
            "S18-R2",
            "S18-R3",
            "S18-COMB",
            "S18-COMB-P",
        }:
            raise ValueError("Unknown Stage18 condition")
        policy.pt_replay_ratio = 0.25 if condition == "S18-R0" else 1.0
        policy.group_weighting = condition in {
            "S18-W",
            "S18-R2",
            "S18-R3",
            "S18-COMB",
            "S18-COMB-P",
        }
        policy.label_smoothing = (
            0.03
            if condition in {"S18-S", "S18-R2", "S18-R3", "S18-COMB", "S18-COMB-P"}
            else 0.0
        )
        policy.distillation_weight = (
            0.5 if condition in {"S18-R3", "S18-COMB", "S18-COMB-P"} else 0.0
        )
        policy.freeze_plm = policy.protect_joint = condition.startswith("S18-COMB")
        policy.protect_ph_peptide = condition == "S18-COMB-P"
    else:
        raise ValueError("stage must be 16, 17 or 18")
    allowed = {
        "pt_replay_ratio",
        "distillation_weight",
        "temperature",
        "label_smoothing",
        "group_weighting",
    }
    if overrides and (stage != 18 or set(overrides) - allowed):
        raise ValueError("Only explicit Stage18 replay/loss overrides are supported")
    for key, value in overrides.items():
        setattr(policy, key, value)
    for name in ("ph_replay_ratio", "pt_replay_ratio", "temperature"):
        if (
            type(getattr(policy, name)) not in (float, int)
            or not math.isfinite(getattr(policy, name))
            or getattr(policy, name) <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if (
        type(policy.distillation_weight) not in (float, int)
        or not math.isfinite(policy.distillation_weight)
        or policy.distillation_weight < 0
    ):
        raise ValueError("Invalid distillation weight")
    if (
        type(policy.label_smoothing) not in (float, int)
        or not 0 <= policy.label_smoothing < 1
        or not isinstance(policy.group_weighting, bool)
    ):
        raise ValueError("Invalid smoothing/group weighting")
    return policy


@dataclass
class Stage16ExperimentConfig:
    paths: ExperimentPaths
    stage: int = 16
    condition: str = "full"
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: Runtime = field(default_factory=Runtime)
    training: Training = field(default_factory=Training)
    selection: Selection = field(default_factory=Selection)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    policy_overrides: dict = field(default_factory=dict)
    source_path: str | None = None

    @property
    def stage8(self):
        # Only the original encoder factory's shape/precision settings are reused.
        return self.training

    @property
    def policy(self):
        return condition_policy(self.stage, self.condition, self.policy_overrides)

    @property
    def debug(self):
        return (
            self.training.debug_pair_limit is not None
            or self.training.debug_limit_per_type is not None
        )

    def validate(self):
        if type(self.stage) is not int or not isinstance(self.condition, str):
            raise ValueError("Stage must be integer and condition must be a string")
        self.policy
        if self.swanlab.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError("swanlab.mode must be online, local, offline or disabled")
        if not self.swanlab.project or not self.swanlab.experiment_prefix:
            raise ValueError("SwanLab project and experiment_prefix are required")
        t = self.training
        for name in (
            "rounds",
            "scheduler_rounds",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "eval_batch_size",
            "constraint_ramp_epochs",
            "log_every_steps",
            "early_stopping_patience",
            "monitor_per_split",
        ):
            if type(getattr(t, name)) is not int or getattr(t, name) < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if (
            t.micro_batch_size % 4
            or t.rounds > t.scheduler_rounds
            and t.scheduler_total_steps is None
        ):
            raise ValueError("Invalid batch size or scheduler horizon")
        for name in (
            "peptide_unfrozen_layers",
            "hla_unfrozen_layers",
            "tcr_unfrozen_layers",
            "checkpoint_every_steps",
            "min_rounds",
        ):
            if type(getattr(t, name)) is not int or getattr(t, name) < 0:
                raise ValueError(f"Invalid training.{name}")
        for name in (
            "debug_pair_limit",
            "debug_limit_per_type",
            "scheduler_total_steps",
        ):
            value = getattr(t, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"Invalid training.{name}")
        if t.mixed_precision not in {
            "none",
            "bf16",
            "fp16",
        } or t.sampler_policy not in {"controlled_v1", "legacy_order"}:
            raise ValueError("Invalid precision/sampler policy")
        if t.constraint_condition not in {
            "full",
            "baseline",
            "hier_only",
            "pt_imp_only",
        }:
            raise ValueError("Invalid constraint condition")
        for obj in (t, self.selection):
            for f in fields(obj):
                v = getattr(obj, f.name)
                if isinstance(v, float) and (not math.isfinite(v) or v < 0):
                    raise ValueError(f"Invalid {f.name}")
        if not 0 <= t.scheduler_warmup_ratio <= 1:
            raise ValueError("Invalid warmup ratio")
        for name in (
            "peptide_input_dim",
            "hla_input_dim",
            "tcr_input_dim",
            "adapter_hidden_dim",
            "model_dim",
            "num_attention_heads",
            "feedforward_dim",
            "peptide_min_length",
            "peptide_max_length",
            "hla_max_length",
            "tcr_max_length",
        ):
            if (
                type(getattr(self.model, name)) is not int
                or getattr(self.model, name) < 1
            ):
                raise ValueError(f"Invalid model.{name}")
        for name in ("adapter_dropout", "attention_dropout", "classifier_dropout"):
            if not 0 <= getattr(self.model, name) < 1:
                raise ValueError(f"Invalid model.{name}")
        if (
            self.model.model_dim % self.model.num_attention_heads
            or self.model.peptide_min_length > self.model.peptide_max_length
        ):
            raise ValueError("Invalid model dimensions/lengths")
        if (
            self.runtime.fold < 1
            or self.runtime.seed < 0
            or self.runtime.num_workers < 0
        ):
            raise ValueError("Invalid fold/seed/workers")
        if self.runtime.device not in {"cpu", "cuda"}:
            raise ValueError("runtime.device must be cpu or cuda")
        for name in ("stage1_sha256", "warmup_sha256", "teacher_sha256"):
            value = getattr(self.paths, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {name}")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=str))

    def contract(self):
        values = self.to_dict()
        values.pop("source_path", None)
        # Resource placement is not the scientific/training contract.
        values["runtime"].pop("device")
        values["runtime"].pop("num_workers")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _typed(cls, values):
    if not isinstance(values, Mapping) or set(values) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown/invalid fields for {cls.__name__}")

    def matches(value, kind):
        origin = get_origin(kind)
        if origin in (Union, UnionType):
            return any(matches(value, part) for part in get_args(kind))
        if kind is float:
            return type(value) in (float, int)
        if kind in (int, bool):
            return type(value) is kind
        if origin is tuple:
            return isinstance(value, (tuple, list)) and all(
                type(v) is int for v in value
            )
        if origin is list:
            item_type = get_args(kind)[0] if get_args(kind) else object
            return isinstance(value, list) and all(matches(item, item_type) for item in value)
        if origin is dict:
            key_type, value_type = get_args(kind) if get_args(kind) else (object, object)
            return isinstance(value, dict) and all(
                matches(key, key_type) and matches(item, value_type)
                for key, item in value.items()
            )
        return isinstance(value, kind)

    types = get_type_hints(cls)
    for name, value in values.items():
        if not matches(value, types[name]):
            raise ValueError(f"Invalid type for {cls.__name__}.{name}")
    return cls(**values)


def config_from_dict(raw: dict, *, base: Path | None = None) -> Stage16ExperimentConfig:
    if set(raw) - {f.name for f in fields(Stage16ExperimentConfig)}:
        raise ValueError("Unknown experiment configuration fields")
    values = dict(raw)
    paths = dict(values.pop("paths"))
    for key, value in paths.items():
        if not key.endswith("sha256"):
            path = Path(value).expanduser()
            paths[key] = (
                ((base or Path.cwd()) / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )
    config = Stage16ExperimentConfig(
        paths=_typed(ExperimentPaths, paths),
        model=_typed(ModelConfig, values.pop("model", {})),
        runtime=_typed(Runtime, values.pop("runtime", {})),
        training=_typed(Training, values.pop("training", {})),
        selection=_typed(Selection, values.pop("selection", {})),
        swanlab=_typed(SwanLabConfig, values.pop("swanlab", {})),
        **values,
    )
    config.training.monitor_rounds = tuple(config.training.monitor_rounds)
    config.validate()
    return config


def load_stage16_config(path, *, expected_stage=None):
    import yaml

    path = Path(path).resolve()
    config = config_from_dict(yaml.safe_load(path.read_text()), base=path.parent)
    config.source_path = str(path)
    if expected_stage is not None and config.stage != expected_stage:
        raise ValueError(f"Expected Stage{expected_stage} configuration")
    return config
