#!/usr/bin/env python3
"""Compare Stage-1 and Stage-2 PLM/entity and task-pair representations in 2D.

Entity projections use post-SequenceAdapter vectors by default; pass
``--entity-feature-space raw`` to project the original PLM vectors instead.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import shutil
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# Match the Stage-2 launcher: Transformer Engine discovers NVRTC through this
# CUDA overlay in the unifyimmun environment.
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

try:  # Direct ``python source/...`` execution.
    from plm_unified.cache import EmbeddingCache, batched
    from plm_unified.config import apply_overrides, load_config
    from plm_unified.data import Task, split_path
    from plm_unified.finetuning import (
        ESMCFineTuningEncoder,
        ProgressiveFineTuningModel,
        TCRBertFineTuningEncoder,
    )
    from plm_unified.metrics import compute_binary_metrics
    from plm_unified.model import UnifiedBindingModel
    from plm_unified.stage2_trainer import _load_progressive_model_state
    from plm_unified.trainer import (
        load_training_checkpoint,
        resolve_device,
        set_global_seed,
    )
    from plm_unified.visualization import (
        EntityMetadata,
        PairRecord,
        build_entity_metadata,
        fit_joint_projections,
        label_separation_summary,
        masked_mean,
        masked_mean_numpy,
        paired_shift_summary,
        read_pair_records,
        sample_pair_records,
        select_entity_sequences,
    )
except ModuleNotFoundError as exc:  # Import as ``source.visualize...`` in tests.
    if exc.name != "plm_unified":
        raise
    from source.plm_unified.cache import EmbeddingCache, batched
    from source.plm_unified.config import apply_overrides, load_config
    from source.plm_unified.data import Task, split_path
    from source.plm_unified.finetuning import (
        ESMCFineTuningEncoder,
        ProgressiveFineTuningModel,
        TCRBertFineTuningEncoder,
    )
    from source.plm_unified.metrics import compute_binary_metrics
    from source.plm_unified.model import UnifiedBindingModel
    from source.plm_unified.stage2_trainer import _load_progressive_model_state
    from source.plm_unified.trainer import (
        load_training_checkpoint,
        resolve_device,
        set_global_seed,
    )
    from source.plm_unified.visualization import (
        EntityMetadata,
        PairRecord,
        build_entity_metadata,
        fit_joint_projections,
        label_separation_summary,
        masked_mean,
        masked_mean_numpy,
        paired_shift_summary,
        read_pair_records,
        sample_pair_records,
        select_entity_sequences,
    )


STAGES = ("stage1", "stage2")


def log_status(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage2.yaml"
        ),
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        required=True,
        help="Full stage2b_best_joint or stage2_last training checkpoint.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Override checkpoint metadata's source_stage1_checkpoint.",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--tasks",
        default="phla,ptcr",
        help="Comma-separated subset of phla,ptcr.",
    )
    parser.add_argument(
        "--reducers",
        default="pca,umap",
        help="Comma-separated subset of pca,umap,tsne.",
    )
    parser.add_argument(
        "--entity-feature-space",
        "--entity-representation",
        dest="entity_feature_space",
        choices=("adapter", "raw"),
        default="adapter",
        help=(
            "Entity feature space for UMAP/PCA: adapter pools the masked "
            "SequenceAdapter output (config.model.model_dim, currently 256); "
            "raw pools the original PLM output (960/768)."
        ),
    )
    parser.add_argument("--max-pairs-per-task", type=int, default=20_000)
    parser.add_argument("--max-entity-sequences", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--metric-sample-size", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/plm_embeddings"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace analysis files previously generated in output-dir.",
    )
    parser.add_argument(
        "--save-high-dimensional",
        action="store_true",
        help="Also save pooled high-dimensional features as compressed NPZ files.",
    )
    return parser.parse_args()


def _parse_choices(raw: str, allowed: set[str], name: str) -> list[str]:
    values = list(dict.fromkeys(value.strip().lower() for value in raw.split(",")))
    values = [value for value in values if value]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}; choose from {sorted(allowed)}")
    if not values:
        raise ValueError(f"At least one {name} must be selected")
    return values


def _load_mmap_checkpoint(path: Path) -> dict[str, Any]:
    """Memory-map a trusted local checkpoint to avoid paging optimizer tensors."""

    source = path.expanduser().resolve()
    try:
        return torch.load(
            source,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(source, map_location="cpu", weights_only=False)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    """Create an output directory without silently mixing analysis runs."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite "
                "to replace this script's existing analysis files."
            )
        for filename in (
            "entity_coordinates.csv.gz",
            "pair_coordinates.csv.gz",
            "summary.json",
        ):
            target = path / filename
            if target.is_file():
                target.unlink()
        figures = path / "figures"
        if figures.is_dir():
            shutil.rmtree(figures)
        for pattern in ("entity_*_features.npz", "pair_*_features.npz"):
            for target in path.glob(pattern):
                if target.is_file():
                    target.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _validate_checkpoint_pair(
    config,
    stage2_checkpoint: Mapping[str, Any],
    stage1_path: Path,
) -> None:
    if stage2_checkpoint.get("checkpoint_type") != "stage2_training":
        raise ValueError("Expected a full stage2_training checkpoint")
    if stage2_checkpoint.get("stage") != "stage2b":
        raise ValueError("Stage-2 comparison requires a Stage-2B checkpoint")
    saved = stage2_checkpoint.get("config", {})
    if saved.get("model") != config.to_dict()["model"]:
        raise ValueError("Stage-2 checkpoint model configuration does not match")
    training = saved.get("training", {})
    for key in ("fold", "seed"):
        expected = getattr(config.training, key)
        if training.get(key) != expected:
            raise ValueError(
                f"Stage-2 checkpoint {key}={training.get(key)} "
                f"does not match requested {key}={expected}"
            )
    recorded_source = stage2_checkpoint.get("source_stage1_checkpoint")
    if recorded_source is None:
        raise ValueError("Stage-2 checkpoint does not record its Stage-1 source")
    if Path(recorded_source).resolve() != stage1_path.resolve():
        raise ValueError(
            "Stage-1 override is not the exact source recorded by Stage 2: "
            f"recorded={Path(recorded_source).resolve()} "
            f"requested={stage1_path.resolve()}"
        )


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _hidden_dtype(device: torch.device, precision: str) -> torch.dtype:
    """Return the tensor dtype expected by a model on ``device``."""

    if device.type != "cuda":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _extract_cached_entity_features(
    cache: EmbeddingCache,
    sequences: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for sequence_batch in batched(sequences, batch_size):
        locators = [cache.resolve(sequence) for sequence in sequence_batch]
        hidden, mask = cache.get_many(locators)
        chunks.append(masked_mean_numpy(hidden, mask))
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def _extract_cached_adapter_features(
    model: UnifiedBindingModel,
    cache: EmbeddingCache,
    sequences: Sequence[str],
    entity: str,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    """Pool the residue representations immediately after one SequenceAdapter.

    The cache stores the original PLM widths (960 for ESM-C and 768 for
    TCR-BERT).  Applying the model's adapter here makes the plotted entity
    vectors use the same common width as the interaction model, including the
    current 256-dimensional configuration.
    """

    adapter = getattr(model, f"{entity}_adapter")
    chunks: list[np.ndarray] = []
    hidden_dtype = _hidden_dtype(device, precision)
    model.to(device).eval()
    for sequence_batch in batched(sequences, batch_size):
        locators = [cache.resolve(sequence) for sequence in sequence_batch]
        hidden, mask = cache.get_many(locators)
        hidden_tensor = torch.from_numpy(hidden).to(device, dtype=hidden_dtype)
        mask_tensor = torch.from_numpy(mask).to(device, dtype=torch.bool)
        with _autocast_context(device, precision):
            adapted = adapter(hidden_tensor, mask_tensor)
            pooled = masked_mean(adapted, mask_tensor)
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def _extract_stage1_pair_features(
    model: UnifiedBindingModel,
    records: Sequence[PairRecord],
    task: Task,
    peptide_cache: EmbeddingCache,
    receptor_cache: EmbeddingCache,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.to(device).eval()
    hidden_dtype = _hidden_dtype(device, precision)
    for record_batch in batched(records, batch_size):
        peptide_hidden, peptide_mask = peptide_cache.get_many(
            [peptide_cache.resolve(record.peptide) for record in record_batch]
        )
        receptor_hidden, receptor_mask = receptor_cache.get_many(
            [receptor_cache.resolve(record.receptor) for record in record_batch]
        )
        with _autocast_context(device, precision):
            output = model(
                task,
                torch.from_numpy(peptide_hidden).to(device, dtype=hidden_dtype),
                torch.from_numpy(peptide_mask).to(device, dtype=torch.bool),
                torch.from_numpy(receptor_hidden).to(device, dtype=hidden_dtype),
                torch.from_numpy(receptor_mask).to(device, dtype=torch.bool),
            )
        features.append(output.pooled_embedding.float().cpu().numpy())
        probabilities.append(
            torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
        )
    return np.concatenate(features), np.concatenate(probabilities)


def _build_stage2_model(config, device: torch.device) -> ProgressiveFineTuningModel:
    dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }[config.stage2.mixed_precision]
    encoder_kwargs = {
        "device": device,
        "dtype": dtype,
        # Analysis is inference-only; activation checkpointing adds no value.
        "gradient_checkpointing": False,
    }
    log_status("Loading Stage-2 peptide ESM-C")
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.peptide_max_length,
        **encoder_kwargs,
    )
    log_status("Loading Stage-2 HLA ESM-C")
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.hla_max_length,
        **encoder_kwargs,
    )
    log_status("Loading Stage-2 TCR-BERT")
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model,
        config.model.tcr_max_length,
        **encoder_kwargs,
    )
    model = ProgressiveFineTuningModel(
        UnifiedBindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=config.stage2.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.stage2.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.stage2.tcr_unfrozen_layers,
    )
    return model.to(device)


@torch.inference_mode()
def _extract_stage2_entity_features(
    encoder,
    sequences: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for sequence_batch in batched(sequences, batch_size):
        with _autocast_context(device, precision):
            encoded = encoder(sequence_batch)
            pooled = masked_mean(encoded.hidden, encoded.mask)
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def _extract_stage2_adapter_features(
    model: ProgressiveFineTuningModel,
    encoder,
    sequences: Sequence[str],
    entity: str,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    """Pool online encoder states after the Stage-2 SequenceAdapter."""

    adapter = getattr(model.binding_model, f"{entity}_adapter")
    chunks: list[np.ndarray] = []
    for sequence_batch in batched(sequences, batch_size):
        with _autocast_context(device, precision):
            encoded = encoder(sequence_batch)
            adapted = adapter(encoded.hidden, encoded.mask)
            pooled = masked_mean(adapted, encoded.mask)
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def _encode_residue_lookup(
    encoder,
    sequences: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode a small repeated entity set once and retain it on CPU."""

    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for sequence_batch in batched(sequences, batch_size):
        with _autocast_context(device, precision):
            encoded = encoder(sequence_batch)
        for index, sequence in enumerate(sequence_batch):
            lookup[sequence] = (
                encoded.hidden[index].detach().cpu(),
                encoded.mask[index].detach().cpu(),
            )
    return lookup


@torch.inference_mode()
def _extract_stage2_pair_features(
    model: ProgressiveFineTuningModel,
    records: Sequence[PairRecord],
    task: Task,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract classifier-input pair vectors while caching the repeated side."""

    if task == "phla":
        repeated_sequences = sorted({record.receptor for record in records})
        repeated_lookup = _encode_residue_lookup(
            model.hla_encoder,
            repeated_sequences,
            batch_size=batch_size,
            device=device,
            precision=precision,
        )
    else:
        repeated_sequences = sorted({record.peptide for record in records})
        repeated_lookup = _encode_residue_lookup(
            model.peptide_encoder,
            repeated_sequences,
            batch_size=batch_size,
            device=device,
            precision=precision,
        )

    features: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for record_batch in batched(records, batch_size):
        if task == "phla":
            with _autocast_context(device, precision):
                peptide = model.peptide_encoder(
                    [record.peptide for record in record_batch]
                )
            receptor_hidden = torch.stack(
                [repeated_lookup[record.receptor][0] for record in record_batch]
            ).to(device)
            receptor_mask = torch.stack(
                [repeated_lookup[record.receptor][1] for record in record_batch]
            ).to(device)
            peptide_hidden = peptide.hidden
            peptide_mask = peptide.mask
        else:
            with _autocast_context(device, precision):
                receptor = model.tcr_encoder(
                    [record.receptor for record in record_batch]
                )
            peptide_hidden = torch.stack(
                [repeated_lookup[record.peptide][0] for record in record_batch]
            ).to(device)
            peptide_mask = torch.stack(
                [repeated_lookup[record.peptide][1] for record in record_batch]
            ).to(device)
            receptor_hidden = receptor.hidden
            receptor_mask = receptor.mask

        with _autocast_context(device, precision):
            output = model.binding_model(
                task,
                peptide_hidden,
                peptide_mask,
                receptor_hidden,
                receptor_mask,
            )
        features.append(output.pooled_embedding.float().cpu().numpy())
        probabilities.append(
            torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
        )
    return np.concatenate(features), np.concatenate(probabilities)


def _binary_summary(
    records: Sequence[PairRecord],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return compute_binary_metrics(
        labels,
        probabilities,
        loss=float(loss),
        threshold=threshold,
    ).to_dict()


def _outcomes(
    records: Sequence[PairRecord],
    probabilities: np.ndarray,
    threshold: float,
) -> list[str]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    names = {
        (1, 1): "TP",
        (0, 0): "TN",
        (0, 1): "FP",
        (1, 0): "FN",
    }
    return [
        names[(int(label), int(prediction))]
        for label, prediction in zip(labels, predictions)
    ]


def _write_csv_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_stage_comparison(
    output_prefix: Path,
    coordinates: Mapping[str, np.ndarray],
    colors: Mapping[str, Sequence[Any]],
    *,
    reducer: str,
    title: str,
    color_label: str,
    categorical: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate([coordinates[stage] for stage in STAGES], axis=0)
    x_pad = max(1e-6, float(np.ptp(combined[:, 0])) * 0.04)
    y_pad = max(1e-6, float(np.ptp(combined[:, 1])) * 0.04)
    x_limits = (float(combined[:, 0].min() - x_pad), float(combined[:, 0].max() + x_pad))
    y_limits = (float(combined[:, 1].min() - y_pad), float(combined[:, 1].max() + y_pad))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if categorical:
        categories = sorted(
            {str(value) for stage in STAGES for value in colors[stage]}
        )
        palette = plt.get_cmap("tab10")
        category_colors = {
            category: palette(index % 10)
            for index, category in enumerate(categories)
        }
        for axis, stage in zip(axes, STAGES):
            stage_colors = np.asarray([str(value) for value in colors[stage]])
            for category in categories:
                selected = stage_colors == category
                if selected.any():
                    axis.scatter(
                        coordinates[stage][selected, 0],
                        coordinates[stage][selected, 1],
                        s=7,
                        alpha=0.5,
                        linewidths=0,
                        rasterized=True,
                        color=category_colors[category],
                        label=category,
                    )
            axis.legend(
                title=color_label,
                markerscale=2,
                frameon=False,
                loc="best",
            )
    else:
        numeric = {
            stage: np.asarray(colors[stage], dtype=np.float32) for stage in STAGES
        }
        all_values = np.concatenate([numeric[stage] for stage in STAGES])
        value_min = float(np.nanmin(all_values))
        value_max = float(np.nanmax(all_values))
        scatter = None
        for axis, stage in zip(axes, STAGES):
            scatter = axis.scatter(
                coordinates[stage][:, 0],
                coordinates[stage][:, 1],
                c=numeric[stage],
                cmap="viridis",
                vmin=value_min,
                vmax=value_max,
                s=7,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
        figure.colorbar(scatter, ax=axes, label=color_label, shrink=0.85)

    for axis, stage in zip(axes, STAGES):
        axis.set(
            title=stage.upper(),
            xlabel="Component 1",
            ylabel="Component 2",
            xlim=x_limits,
            ylim=y_limits,
        )
        axis.grid(alpha=0.15, linewidth=0.5)
    figure.suptitle(title)
    # Keep the two file formats separate, and group each reducer beneath both
    # format directories:
    #   figures/png/{pca,umap}/<figure>.png
    #   figures/svg/{pca,umap}/<figure>.svg
    figure_name = output_prefix.name
    figures_root = output_prefix.parent
    for file_format in ("png", "svg"):
        target = figures_root / file_format / reducer / f"{figure_name}.{file_format}"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=300)
    plt.close(figure)


def _entity_rows(
    entity: str,
    sequences: Sequence[str],
    metadata: Mapping[str, EntityMetadata],
    projection,
    *,
    feature_space: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reducers = list(next(iter(projection.coordinates.values())))
    for stage in STAGES:
        for index, sequence in enumerate(sequences):
            item = metadata[sequence]
            row: dict[str, Any] = {
                "sample_id": f"{entity}:{sequence}",
                "entity": entity,
                "feature_space": feature_space,
                "sequence": sequence,
                "stage": stage,
                "sequence_length": item.sequence_length,
                "occurrence_count": item.occurrence_count,
                "positive_count": item.positive_count,
                "positive_fraction": item.positive_fraction,
                "source_tasks": item.source_tasks,
            }
            for reducer in reducers:
                row[f"{reducer}_1"] = float(
                    projection.coordinates[stage][reducer][index, 0]
                )
                row[f"{reducer}_2"] = float(
                    projection.coordinates[stage][reducer][index, 1]
                )
            rows.append(row)
    return rows


def _pair_rows(
    task: Task,
    records: Sequence[PairRecord],
    probabilities: Mapping[str, np.ndarray],
    outcomes: Mapping[str, Sequence[str]],
    projection,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reducers = list(next(iter(projection.coordinates.values())))
    receptor_name = "hla" if task == "phla" else "tcr"
    for stage in STAGES:
        predictions = (probabilities[stage] >= threshold).astype(np.int64)
        for index, record in enumerate(records):
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "task": task,
                "row_index": record.row_index,
                "peptide": record.peptide,
                receptor_name: record.receptor,
                "label": record.label,
                "stage": stage,
                "feature_space": "pooled_classifier_input",
                "probability": float(probabilities[stage][index]),
                "prediction": int(predictions[index]),
                "outcome": outcomes[stage][index],
                "peptide_length": len(record.peptide),
                f"{receptor_name}_length": len(record.receptor),
            }
            for reducer in reducers:
                row[f"{reducer}_1"] = float(
                    projection.coordinates[stage][reducer][index, 0]
                )
                row[f"{reducer}_2"] = float(
                    projection.coordinates[stage][reducer][index, 1]
                )
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.max_pairs_per_task < 2 or args.max_entity_sequences < 2:
        raise ValueError("Sample limits must be at least 2")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.metric_sample_size < 2:
        raise ValueError("metric-sample-size must be at least 2")
    tasks = _parse_choices(args.tasks, {"phla", "ptcr"}, "tasks")
    reducers = _parse_choices(args.reducers, {"pca", "umap", "tsne"}, "reducers")

    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    config.validate(require_models=True)
    set_global_seed(args.random_state)
    device = resolve_device(config.training.device)
    batch_size = args.batch_size or config.stage2.eval_batch_size
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    stage2_path = args.stage2_checkpoint.expanduser().resolve()
    log_status(f"Reading Stage-2 checkpoint metadata: {stage2_path}")
    stage2_checkpoint = _load_mmap_checkpoint(stage2_path)
    recorded_stage1 = stage2_checkpoint.get("source_stage1_checkpoint")
    stage1_path = (
        args.stage1_checkpoint.expanduser().resolve()
        if args.stage1_checkpoint
        else Path(recorded_stage1).resolve()
        if recorded_stage1
        else None
    )
    if stage1_path is None:
        raise ValueError(
            "No Stage-1 checkpoint was supplied or recorded by Stage 2"
        )
    _validate_checkpoint_pair(config, stage2_checkpoint, stage1_path)

    records_by_task: dict[Task, list[PairRecord]] = {}
    sampled_pairs: dict[Task, list[PairRecord]] = {}
    for task_index, task in enumerate(tasks):
        path = split_path(
            config.paths.data_root,
            task,
            args.split,
            config.training.fold,
        )
        records = read_pair_records(
            path,
            task,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        records_by_task[task] = records
        sampled_pairs[task] = sample_pair_records(
            records,
            args.max_pairs_per_task,
            random_state=args.random_state + task_index,
        )
        log_status(
            f"Sampled {len(sampled_pairs[task]):,}/{len(records):,} {task} rows"
        )

    entity_metadata = build_entity_metadata(records_by_task)
    entity_sequences = select_entity_sequences(
        sampled_pairs,
        entity_metadata,
        args.max_entity_sequences,
        random_state=args.random_state,
    )
    required_entities = {"peptide"}
    if "phla" in tasks:
        required_entities.add("hla")
    if "ptcr" in tasks:
        required_entities.add("tcr")
    entity_sequences = {
        entity: sequences
        for entity, sequences in entity_sequences.items()
        if entity in required_entities and sequences
    }
    log_status(
        "Entity samples: "
        + ", ".join(
            f"{entity}={len(sequences):,}"
            for entity, sequences in entity_sequences.items()
        )
    )

    cache_dir = config.cache_dir()
    encoder_metadata = stage2_checkpoint.get("encoder_metadata", {})
    caches = {
        entity: EmbeddingCache(
            cache_dir / entity,
            expected_hidden_dim=getattr(config.model, f"{entity}_input_dim"),
            expected_fingerprint=encoder_metadata.get(entity, {}).get(
                "checkpoint_fingerprint"
            ),
        )
        for entity in required_entities
    }

    log_status(f"Loading exact Stage-1 source: {stage1_path}")
    stage1_checkpoint = load_training_checkpoint(stage1_path, map_location="cpu")
    if stage1_checkpoint.get("stage") != "complete":
        raise ValueError("Stage-1 source checkpoint is not complete")
    if stage1_checkpoint.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("Stage-1 checkpoint model configuration does not match")
    stage1_model = UnifiedBindingModel(config.model)
    stage1_model.load_state_dict(stage1_checkpoint["model"])
    del stage1_checkpoint

    entity_features: dict[str, dict[str, np.ndarray]] = {}
    pair_features: dict[Task, dict[str, np.ndarray]] = {
        task: {} for task in tasks
    }
    pair_probabilities: dict[Task, dict[str, np.ndarray]] = {
        task: {} for task in tasks
    }
    for entity, sequences in entity_sequences.items():
        log_status(
            f"Extracting Stage-1 {entity} sequence features "
            f"({args.entity_feature_space})"
        )
        if args.entity_feature_space == "adapter":
            stage1_features = _extract_cached_adapter_features(
                stage1_model,
                caches[entity],
                sequences,
                entity,
                batch_size=batch_size,
                device=device,
                precision=config.training.mixed_precision,
            )
        else:
            stage1_features = _extract_cached_entity_features(
                caches[entity],
                sequences,
                batch_size=batch_size,
            )
        entity_features[entity] = {"stage1": stage1_features}
    for task in tasks:
        receptor_entity = "hla" if task == "phla" else "tcr"
        log_status(f"Extracting Stage-1 {task} pair features")
        features, probabilities = _extract_stage1_pair_features(
            stage1_model,
            sampled_pairs[task],
            task,
            caches["peptide"],
            caches[receptor_entity],
            batch_size=batch_size,
            device=device,
            precision=config.training.mixed_precision,
        )
        pair_features[task]["stage1"] = features
        pair_probabilities[task]["stage1"] = probabilities
    stage1_model.to("cpu")
    del stage1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log_status("Building Stage-2 online encoders")
    stage2_model = _build_stage2_model(config, device)
    _load_progressive_model_state(stage2_model, stage2_checkpoint["model"])
    stage2_model.set_evaluation()
    del stage2_checkpoint
    gc.collect()

    encoders = {
        "peptide": stage2_model.peptide_encoder,
        "hla": stage2_model.hla_encoder,
        "tcr": stage2_model.tcr_encoder,
    }
    for entity, sequences in entity_sequences.items():
        log_status(
            f"Extracting Stage-2 {entity} sequence features "
            f"({args.entity_feature_space})"
        )
        if args.entity_feature_space == "adapter":
            entity_features[entity]["stage2"] = _extract_stage2_adapter_features(
                stage2_model,
                encoders[entity],
                sequences,
                entity,
                batch_size=batch_size,
                device=device,
                precision=config.stage2.mixed_precision,
            )
        else:
            entity_features[entity]["stage2"] = _extract_stage2_entity_features(
                encoders[entity],
                sequences,
                batch_size=batch_size,
                device=device,
                precision=config.stage2.mixed_precision,
            )
    for task in tasks:
        log_status(f"Extracting Stage-2 {task} pair features")
        features, probabilities = _extract_stage2_pair_features(
            stage2_model,
            sampled_pairs[task],
            task,
            batch_size=batch_size,
            device=device,
            precision=config.stage2.mixed_precision,
        )
        pair_features[task]["stage2"] = features
        pair_probabilities[task]["stage2"] = probabilities

    stage2_model.to("cpu")
    del stage2_model
    for cache in caches.values():
        cache.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    summary: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "fold": config.training.fold,
        "seed": config.training.seed,
        "split": args.split,
        "tasks": tasks,
        "reducers": reducers,
        "feature_spaces": {
            "entity": args.entity_feature_space,
            "pair": "pooled_classifier_input",
        },
        "feature_dimensions": {
            "entity": {
                entity: (
                    config.model.model_dim
                    if args.entity_feature_space == "adapter"
                    else getattr(config.model, f"{entity}_input_dim")
                )
                for entity in sorted(required_entities)
            },
            "pair": config.model.model_dim,
        },
        "random_state": args.random_state,
        "stage1_checkpoint": str(stage1_path),
        "stage2_checkpoint": str(stage2_path),
        "sampling": {
            "max_pairs_per_task": args.max_pairs_per_task,
            "max_entity_sequences": args.max_entity_sequences,
            "full_pair_counts": {
                task: len(records_by_task[task]) for task in tasks
            },
            "sampled_pair_counts": {
                task: len(sampled_pairs[task]) for task in tasks
            },
            "entity_counts": {
                entity: len(sequences)
                for entity, sequences in entity_sequences.items()
            },
        },
        "entity_spaces": {},
        "pair_spaces": {},
        "interpretation_warning": (
            "Two-dimensional neighborhoods are descriptive projections, not "
            "evidence of causal residue contacts or biological subtypes."
        ),
    }

    entity_rows: list[dict[str, Any]] = []
    for entity, features_by_stage in entity_features.items():
        log_status(f"Fitting joint Stage-1/Stage-2 projections for {entity}")
        projection = fit_joint_projections(
            features_by_stage,
            reducers=reducers,
            feature_space=args.entity_feature_space,
            random_state=args.random_state,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            metric_sample_size=args.metric_sample_size,
        )
        projection.summary["stage_shift"] = paired_shift_summary(
            features_by_stage["stage1"],
            features_by_stage["stage2"],
        )
        summary["entity_spaces"][entity] = projection.summary
        entity_rows.extend(
            _entity_rows(
                entity,
                entity_sequences[entity],
                entity_metadata[entity],
                projection,
                feature_space=args.entity_feature_space,
            )
        )
        lengths = [
            entity_metadata[entity][sequence].sequence_length
            for sequence in entity_sequences[entity]
        ]
        positive_fractions = [
            entity_metadata[entity][sequence].positive_fraction
            for sequence in entity_sequences[entity]
        ]
        for reducer in reducers:
            coordinates = {
                stage: projection.coordinates[stage][reducer] for stage in STAGES
            }
            _plot_stage_comparison(
                output_dir / "figures" / f"entity_{entity}_{reducer}_length",
                coordinates,
                {stage: lengths for stage in STAGES},
                reducer=reducer,
                title=(
                    f"{entity.upper()} {args.entity_feature_space} representations "
                    f"({reducer.upper()})"
                ),
                color_label="Sequence length",
                categorical=False,
            )
            _plot_stage_comparison(
                output_dir
                / "figures"
                / f"entity_{entity}_{reducer}_positive_fraction",
                coordinates,
                {stage: positive_fractions for stage in STAGES},
                reducer=reducer,
                title=(
                    f"{entity.upper()} {args.entity_feature_space} representations "
                    f"({reducer.upper()})"
                ),
                color_label="Associated positive fraction",
                categorical=False,
            )
        if args.save_high_dimensional:
            np.savez_compressed(
                output_dir / f"entity_{entity}_features.npz",
                sample_ids=np.asarray(
                    [f"{entity}:{sequence}" for sequence in entity_sequences[entity]]
                ),
                **features_by_stage,
            )

    pair_rows: list[dict[str, Any]] = []
    for task, features_by_stage in pair_features.items():
        log_status(f"Fitting joint Stage-1/Stage-2 projections for {task}")
        projection = fit_joint_projections(
            features_by_stage,
            reducers=reducers,
            feature_space="pooled_classifier_input",
            random_state=args.random_state,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            metric_sample_size=args.metric_sample_size,
        )
        labels = [record.label for record in sampled_pairs[task]]
        outcomes = {
            stage: _outcomes(
                sampled_pairs[task],
                pair_probabilities[task][stage],
                config.training.threshold,
            )
            for stage in STAGES
        }
        task_summary: dict[str, Any] = dict(projection.summary)
        task_summary["stage_shift"] = paired_shift_summary(
            features_by_stage["stage1"],
            features_by_stage["stage2"],
        )
        task_summary["prediction_metrics"] = {
            stage: _binary_summary(
                sampled_pairs[task],
                pair_probabilities[task][stage],
                config.training.threshold,
            )
            for stage in STAGES
        }
        task_summary["label_separation"] = {
            stage: label_separation_summary(
                features_by_stage[stage],
                projection.coordinates[stage],
                labels,
                random_state=args.random_state,
                maximum=args.metric_sample_size,
            )
            for stage in STAGES
        }
        summary["pair_spaces"][task] = task_summary
        pair_rows.extend(
            _pair_rows(
                task,
                sampled_pairs[task],
                pair_probabilities[task],
                outcomes,
                projection,
                config.training.threshold,
            )
        )
        for reducer in reducers:
            coordinates = {
                stage: projection.coordinates[stage][reducer] for stage in STAGES
            }
            _plot_stage_comparison(
                output_dir / "figures" / f"pair_{task}_{reducer}_label",
                coordinates,
                {stage: labels for stage in STAGES},
                reducer=reducer,
                title=f"{task.upper()} pair representations ({reducer.upper()})",
                color_label="True label",
                categorical=True,
            )
            _plot_stage_comparison(
                output_dir / "figures" / f"pair_{task}_{reducer}_outcome",
                coordinates,
                outcomes,
                reducer=reducer,
                title=f"{task.upper()} prediction outcomes ({reducer.upper()})",
                color_label="Outcome",
                categorical=True,
            )
            _plot_stage_comparison(
                output_dir / "figures" / f"pair_{task}_{reducer}_probability",
                coordinates,
                pair_probabilities[task],
                reducer=reducer,
                title=f"{task.upper()} binding probability ({reducer.upper()})",
                color_label="Predicted probability",
                categorical=False,
            )
        if args.save_high_dimensional:
            np.savez_compressed(
                output_dir / f"pair_{task}_features.npz",
                sample_ids=np.asarray(
                    [record.sample_id for record in sampled_pairs[task]]
                ),
                labels=np.asarray(labels, dtype=np.int64),
                stage1_probabilities=pair_probabilities[task]["stage1"],
                stage2_probabilities=pair_probabilities[task]["stage2"],
                **features_by_stage,
            )

    _write_csv_gz(output_dir / "entity_coordinates.csv.gz", entity_rows)
    _write_csv_gz(output_dir / "pair_coordinates.csv.gz", pair_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_status(f"Analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
