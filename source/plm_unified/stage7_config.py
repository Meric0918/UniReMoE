"""Standalone configuration for Stage-7 HLA counterfactual evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ModelConfig


@dataclass
class Stage7Paths:
    project_root: Path
    data_root: Path
    output_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class Stage7RuntimeConfig:
    fold: int = 4
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 0


@dataclass
class Stage7Config:
    """Controls derived datasets, inference, and Stage-7 statistics."""

    triplet_data_subdir: str = "data_triple"
    triplet_merged_filename: str = "merged_pseudo.csv"
    triplet_train_filename: str = "train_set.csv"
    triplet_val_filename: str = "val_set.csv"
    triplet_test_filename: str = "test_set.csv"
    phla_data_subdir: str = "data_HLA"
    phla_train_template: str = "train_fold_{fold}.csv"
    verify_merged_manifest: bool = True
    strict_split_cleanup: bool = True
    reject_noncanonical_sequences: bool = True
    metadata_mode: str = "pseudo_only"
    hla_metadata_csv: Optional[Path] = None
    max_decoys_per_anchor: int = 5
    require_decoy_hla_seen_in_triplet_train: bool = True
    include_all_tetrads: bool = True
    include_legacy_hla_shift: bool = True
    encoder_dtype: str = "float16"
    encoder_batch_size: int = 32
    score_batch_size: int = 256
    bootstrap_samples: int = 5000
    permutation_samples: int = 100000


@dataclass
class Stage7ExperimentConfig:
    paths: Stage7Paths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: Stage7RuntimeConfig = field(default_factory=Stage7RuntimeConfig)
    stage7: Stage7Config = field(default_factory=Stage7Config)
    source_path: Optional[Path] = None

    def validate(self, *, require_models: bool = False) -> None:
        if self.runtime.fold < 1:
            raise ValueError("runtime.fold must be positive")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")
        if self.model.model_dim % self.model.num_attention_heads != 0:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        values = self.stage7
        for name in (
            "max_decoys_per_anchor",
            "encoder_batch_size",
            "score_batch_size",
            "bootstrap_samples",
            "permutation_samples",
        ):
            if getattr(values, name) < 1:
                raise ValueError(f"stage7.{name} must be positive")
        if values.encoder_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "stage7.encoder_dtype must be float16, bfloat16, or float32"
            )
        if values.metadata_mode not in {"pseudo_only", "required"}:
            raise ValueError("stage7.metadata_mode must be pseudo_only or required")
        if values.metadata_mode == "required" and values.hla_metadata_csv is None:
            raise ValueError(
                "stage7.hla_metadata_csv is required when metadata_mode=required"
            )
        if values.hla_metadata_csv is not None and not values.hla_metadata_csv.is_file():
            raise FileNotFoundError(
                f"Stage-7 HLA metadata not found: {values.hla_metadata_csv}"
            )
        for name in (
            "triplet_data_subdir",
            "triplet_merged_filename",
            "triplet_train_filename",
            "triplet_val_filename",
            "triplet_test_filename",
            "phla_data_subdir",
            "phla_train_template",
        ):
            if not str(getattr(values, name)).strip():
                raise ValueError(f"stage7.{name} must be non-empty")
        try:
            values.phla_train_template.format(fold=self.runtime.fold)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "stage7.phla_train_template must support {fold}"
            ) from exc
        if require_models:
            for name, path in (
                ("ESM-C", self.paths.esmc_model),
                ("TCR-BERT", self.paths.tcr_bert_model),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    def phla_train_path(self) -> Path:
        return (
            self.paths.data_root
            / self.stage7.phla_data_subdir
            / self.stage7.phla_train_template.format(fold=self.runtime.fold)
        )

    def run_root(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.runtime.fold}"
            / f"seed_{self.runtime.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


def _serialise(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, list):
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


def load_stage7_config(path: str | Path) -> Stage7ExperimentConfig:
    """Load Stage 7 without adding fields to shared Stage-1/3 configs."""

    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage-7 config") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The Stage-7 YAML root must be a mapping")
    allowed_root = {"paths", "model", "runtime", "stage7"}
    unexpected = sorted(set(raw) - allowed_root)
    if unexpected:
        raise ValueError("Unknown Stage-7 config sections: " + ", ".join(unexpected))
    base_dir = config_path.parent
    path_values = _section(raw, "paths", Stage7Paths.__dataclass_fields__)
    missing = {"project_root", "data_root", "output_root"} - set(path_values)
    if missing:
        raise ValueError("Missing Stage-7 paths: " + ", ".join(sorted(missing)))
    path_values = {key: _resolve(value, base_dir) for key, value in path_values.items()}
    stage7_values = _section(raw, "stage7", Stage7Config.__dataclass_fields__)
    if stage7_values.get("hla_metadata_csv") is not None:
        stage7_values["hla_metadata_csv"] = _resolve(
            stage7_values["hla_metadata_csv"], base_dir
        )
    config = Stage7ExperimentConfig(
        paths=Stage7Paths(**path_values),
        model=ModelConfig(**_section(raw, "model", ModelConfig.__dataclass_fields__)),
        runtime=Stage7RuntimeConfig(
            **_section(raw, "runtime", Stage7RuntimeConfig.__dataclass_fields__)
        ),
        stage7=Stage7Config(**stage7_values),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_stage7_overrides(
    config: Stage7ExperimentConfig,
    *,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> Stage7ExperimentConfig:
    if fold is not None:
        config.runtime.fold = fold
    if seed is not None:
        config.runtime.seed = seed
    if device is not None:
        config.runtime.device = device
    if num_workers is not None:
        config.runtime.num_workers = num_workers
    config.validate()
    return config
