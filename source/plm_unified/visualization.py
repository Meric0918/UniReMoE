"""Reusable sampling and dimensionality-reduction helpers for PLM analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeVar

import numpy as np
import torch

from .data import Task, normalise_sequence, task_columns


T = TypeVar("T")


@dataclass(frozen=True)
class PairRecord:
    """One filtered pHLA or pTCR CSV row with a stable sample identifier."""

    sample_id: str
    task: Task
    row_index: int
    peptide: str
    receptor: str
    label: int


@dataclass(frozen=True)
class EntityMetadata:
    """Validation-set metadata associated with one unique biological sequence."""

    entity: str
    sequence: str
    sequence_length: int
    occurrence_count: int
    positive_count: int
    positive_fraction: float
    source_tasks: str


@dataclass
class ProjectionResult:
    """Coordinates and diagnostics from one joint Stage-1/Stage-2 fit."""

    coordinates: dict[str, dict[str, np.ndarray]]
    summary: dict[str, Any]


def masked_mean(
    hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool real residue vectors while excluding padded positions."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, length, hidden_dim]")
    if mask.shape != hidden.shape[:2]:
        raise ValueError("mask shape must match hidden batch and length dimensions")
    mask = mask.bool()
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every sequence must contain at least one real residue")
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)


def masked_mean_numpy(
    hidden: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """NumPy equivalent of :func:`masked_mean`, accumulated in float32."""

    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, length, hidden_dim]")
    if mask.shape != hidden.shape[:2]:
        raise ValueError("mask shape must match hidden batch and length dimensions")
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every sequence must contain at least one real residue")
    return (
        (hidden * mask[..., None]).sum(axis=1)
        / counts[:, None].astype(np.float32)
    ).astype(np.float32, copy=False)


def read_pair_records(
    csv_path: str | Path,
    task: Task,
    *,
    peptide_min_length: int = 8,
    peptide_max_length: int = 15,
) -> list[PairRecord]:
    """Read and validate one pair split using the training pipeline's rules."""

    path = Path(csv_path)
    peptide_col, receptor_col, label_col = task_columns(task)
    records: list[PairRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {peptide_col, receptor_col, label_col}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row_index, row in enumerate(reader, start=2):
            peptide = normalise_sequence(row[peptide_col])
            if not peptide_min_length <= len(peptide) <= peptide_max_length:
                continue
            receptor = normalise_sequence(row[receptor_col])
            try:
                label = int(row[label_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_index}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0 or 1 at {path}:{row_index}")
            records.append(
                PairRecord(
                    sample_id=f"{task}:{path.stem}:{row_index}",
                    task=task,
                    row_index=row_index,
                    peptide=peptide,
                    receptor=receptor,
                    label=label,
                )
            )
    if not records:
        raise ValueError(f"No usable rows found in {path}")
    return records


def proportional_stratified_sample(
    values: Sequence[T],
    maximum: int,
    *,
    stratum: Callable[[T], Hashable],
    random_state: int,
) -> list[T]:
    """Deterministically sample at most ``maximum`` values by stratum.

    Stratum allocations follow the original proportions using the
    largest-remainder method. Values are returned in their original order.
    """

    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len(values) <= maximum:
        return list(values)

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[stratum(value)].append(index)

    exact = {
        key: maximum * len(indices) / len(values)
        for key, indices in grouped.items()
    }
    allocations = {key: int(np.floor(count)) for key, count in exact.items()}
    remaining = maximum - sum(allocations.values())
    ranked = sorted(
        grouped,
        key=lambda key: (-(exact[key] - allocations[key]), repr(key)),
    )
    for key in ranked[:remaining]:
        allocations[key] += 1

    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for key in sorted(grouped, key=repr):
        indices = np.asarray(grouped[key], dtype=np.int64)
        take = allocations[key]
        if take:
            selected.extend(
                int(index)
                for index in rng.choice(indices, size=take, replace=False)
            )
    selected.sort()
    if len(selected) != maximum:
        raise RuntimeError("Stratified sampler produced an unexpected sample count")
    return [values[index] for index in selected]


def sample_pair_records(
    records: Sequence[PairRecord],
    maximum: int,
    *,
    random_state: int,
) -> list[PairRecord]:
    """Sample pairs proportionally by true label."""

    return proportional_stratified_sample(
        records,
        maximum,
        stratum=lambda record: record.label,
        random_state=random_state,
    )


def build_entity_metadata(
    records_by_task: Mapping[Task, Sequence[PairRecord]],
) -> dict[str, dict[str, EntityMetadata]]:
    """Aggregate occurrence and label-association metadata by entity sequence."""

    counts: dict[str, dict[str, list[Any]]] = {
        "peptide": {},
        "hla": {},
        "tcr": {},
    }
    for task, records in records_by_task.items():
        receptor_entity = "hla" if task == "phla" else "tcr"
        for record in records:
            for entity, sequence in (
                ("peptide", record.peptide),
                (receptor_entity, record.receptor),
            ):
                state = counts[entity].setdefault(sequence, [0, 0, set()])
                state[0] += 1
                state[1] += record.label
                state[2].add(task)

    metadata: dict[str, dict[str, EntityMetadata]] = {}
    for entity, sequences in counts.items():
        metadata[entity] = {
            sequence: EntityMetadata(
                entity=entity,
                sequence=sequence,
                sequence_length=len(sequence),
                occurrence_count=int(state[0]),
                positive_count=int(state[1]),
                positive_fraction=float(state[1] / state[0]),
                source_tasks=",".join(sorted(state[2])),
            )
            for sequence, state in sequences.items()
        }
    return metadata


def select_entity_sequences(
    sampled_pairs: Mapping[Task, Sequence[PairRecord]],
    metadata: Mapping[str, Mapping[str, EntityMetadata]],
    maximum: int,
    *,
    random_state: int,
) -> dict[str, list[str]]:
    """Select unique entity sequences represented by the sampled pair rows."""

    candidates: dict[str, set[str]] = {
        "peptide": set(),
        "hla": set(),
        "tcr": set(),
    }
    for task, records in sampled_pairs.items():
        receptor_entity = "hla" if task == "phla" else "tcr"
        for record in records:
            candidates["peptide"].add(record.peptide)
            candidates[receptor_entity].add(record.receptor)

    selected: dict[str, list[str]] = {}
    offsets = {"peptide": 11, "hla": 23, "tcr": 37}
    for entity, values in candidates.items():
        ordered = sorted(values)
        selected[entity] = proportional_stratified_sample(
            ordered,
            maximum,
            stratum=lambda sequence, entity=entity: (
                metadata[entity][sequence].source_tasks,
                metadata[entity][sequence].sequence_length,
            ),
            random_state=random_state + offsets[entity],
        )
    return selected


def validate_feature_matrix(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Convert one feature matrix to finite float32 values."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two rows and two features")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def fit_joint_projections(
    stage_features: Mapping[str, np.ndarray],
    *,
    reducers: Sequence[str] = ("pca", "umap"),
    feature_space: str | None = None,
    random_state: int = 42,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    metric_sample_size: int = 5000,
) -> ProjectionResult:
    """Fit one coordinate system jointly across matched model stages.

    The feature width is inferred from the supplied matrices; no fixed 128
    dimensionality is assumed, so current 256-dimensional adapter outputs are
    handled without a separate reducer implementation.
    """

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import normalize

    if len(stage_features) < 2:
        raise ValueError("At least two stages are required for a joint projection")
    stages = list(stage_features)
    matrices = {
        stage: validate_feature_matrix(values, name=stage)
        for stage, values in stage_features.items()
    }
    row_counts = {matrix.shape[0] for matrix in matrices.values()}
    dimensions = {matrix.shape[1] for matrix in matrices.values()}
    if len(row_counts) != 1 or len(dimensions) != 1:
        raise ValueError("All stages must have matching rows and feature dimensions")

    requested = tuple(dict.fromkeys(value.lower() for value in reducers))
    unknown = sorted(set(requested) - {"pca", "umap", "tsne"})
    if unknown:
        raise ValueError(f"Unknown reducers: {unknown}")
    if not requested:
        raise ValueError("At least one reducer is required")

    normalized = {
        stage: normalize(matrix, norm="l2").astype(np.float32, copy=False)
        for stage, matrix in matrices.items()
    }
    combined = np.concatenate([normalized[stage] for stage in stages], axis=0)
    slices: dict[str, slice] = {}
    offset = 0
    for stage in stages:
        end = offset + normalized[stage].shape[0]
        slices[stage] = slice(offset, end)
        offset = end

    coordinates: dict[str, dict[str, np.ndarray]] = {
        stage: {} for stage in stages
    }
    summary: dict[str, Any] = {
        "sample_count_per_stage": next(iter(row_counts)),
        "feature_dimension": next(iter(dimensions)),
        "reducers": list(requested),
        "random_state": random_state,
    }
    if feature_space is not None:
        feature_space = str(feature_space).strip()
        if not feature_space:
            raise ValueError("feature_space cannot be empty")
        summary["feature_space"] = feature_space

    if "pca" in requested:
        pca2 = PCA(n_components=2, svd_solver="full")
        pca_coordinates = pca2.fit_transform(combined).astype(np.float32)
        for stage in stages:
            coordinates[stage]["pca"] = pca_coordinates[slices[stage]]
        summary["pca"] = {
            "explained_variance_ratio": [
                float(value) for value in pca2.explained_variance_ratio_
            ],
        }

    nonlinear = [name for name in requested if name in {"umap", "tsne"}]
    if nonlinear:
        pre_components = min(50, combined.shape[1], combined.shape[0] - 1)
        pca_pre = PCA(
            n_components=pre_components,
            svd_solver="randomized",
            random_state=random_state,
        )
        preprocessed = pca_pre.fit_transform(combined).astype(np.float32)
        preprocessed = normalize(preprocessed, norm="l2").astype(
            np.float32,
            copy=False,
        )
        summary["nonlinear_preprocessing"] = {
            "pca_components": pre_components,
            "explained_variance_ratio_sum": float(
                pca_pre.explained_variance_ratio_.sum()
            ),
        }

        if "umap" in requested:
            try:
                import umap
            except ImportError as exc:
                raise RuntimeError(
                    "UMAP requires 'umap-learn'. Install requirements-plm.txt."
                ) from exc
            effective_neighbors = min(
                umap_n_neighbors,
                max(2, combined.shape[0] - 1),
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=effective_neighbors,
                min_dist=umap_min_dist,
                metric="cosine",
                random_state=random_state,
                transform_seed=random_state,
            )
            umap_coordinates = reducer.fit_transform(preprocessed).astype(
                np.float32
            )
            for stage in stages:
                coordinates[stage]["umap"] = umap_coordinates[slices[stage]]
            summary["umap"] = {
                "n_neighbors": effective_neighbors,
                "min_dist": umap_min_dist,
                "metric": "cosine",
            }

        if "tsne" in requested:
            perplexity = min(30.0, max(2.0, (combined.shape[0] - 1) / 3))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                metric="euclidean",
                random_state=random_state,
            )
            tsne_coordinates = tsne.fit_transform(preprocessed).astype(
                np.float32
            )
            for stage in stages:
                coordinates[stage]["tsne"] = tsne_coordinates[slices[stage]]
            summary["tsne"] = {
                "perplexity": perplexity,
                "metric": "euclidean_after_pca",
            }

    diagnostics: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(random_state)
    for stage in stages:
        count = normalized[stage].shape[0]
        take = min(metric_sample_size, count)
        indices = (
            np.arange(count)
            if take == count
            else np.sort(rng.choice(count, size=take, replace=False))
        )
        neighbors = min(15, max(1, (take - 1) // 2))
        original_distances = pairwise_distances(
            normalized[stage][indices],
            metric="cosine",
        )
        diagnostics[stage] = {}
        for reducer_name, values in coordinates[stage].items():
            diagnostics[stage][f"{reducer_name}_trustworthiness"] = float(
                trustworthiness(
                    original_distances,
                    values[indices],
                    n_neighbors=neighbors,
                    metric="precomputed",
                )
            )
    summary["trustworthiness"] = diagnostics
    return ProjectionResult(coordinates=coordinates, summary=summary)


def paired_shift_summary(
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, float]:
    """Summarize matched high-dimensional representation movement."""

    from sklearn.preprocessing import normalize

    before = validate_feature_matrix(before, name="before")
    after = validate_feature_matrix(after, name="after")
    if before.shape != after.shape:
        raise ValueError("before and after feature matrices must have equal shape")
    cosine = np.sum(
        normalize(before, norm="l2") * normalize(after, norm="l2"),
        axis=1,
    )
    l2 = np.linalg.norm(after - before, axis=1)
    return {
        "cosine_similarity_mean": float(np.mean(cosine)),
        "cosine_similarity_median": float(np.median(cosine)),
        "cosine_similarity_q05": float(np.quantile(cosine, 0.05)),
        "cosine_similarity_q95": float(np.quantile(cosine, 0.95)),
        "l2_shift_mean": float(np.mean(l2)),
        "l2_shift_median": float(np.median(l2)),
    }


def label_separation_summary(
    features: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    labels: Sequence[int],
    *,
    random_state: int,
    maximum: int = 5000,
) -> dict[str, float | None]:
    """Compute descriptive true-label silhouette scores without overclaiming."""

    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    matrix = validate_feature_matrix(features, name="features")
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != (matrix.shape[0],):
        raise ValueError("labels must match the feature rows")
    if len(np.unique(label_array)) < 2:
        return {"high_dimensional": None}
    take = min(maximum, len(label_array))
    rng = np.random.default_rng(random_state)
    indices = (
        np.arange(take)
        if take == len(label_array)
        else np.sort(rng.choice(len(label_array), size=take, replace=False))
    )
    output: dict[str, float | None] = {
        "high_dimensional": float(
            silhouette_score(
                normalize(matrix[indices], norm="l2"),
                label_array[indices],
                metric="cosine",
            )
        )
    }
    for name, values in coordinates.items():
        values = np.asarray(values, dtype=np.float32)
        output[name] = float(
            silhouette_score(
                values[indices],
                label_array[indices],
                metric="euclidean",
            )
        )
    return output
