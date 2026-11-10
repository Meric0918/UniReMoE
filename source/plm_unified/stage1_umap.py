"""Train/validation UMAP snapshots for Stage-1 task-best checkpoints."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .model import Task
from .visualization import proportional_stratified_sample

UMAP_STAGES = ("encoder", "adapter", "classifier_input")
StageName = Literal["encoder", "adapter", "plm", "sequence_encoder", "classifier_input"]


@dataclass(frozen=True)
class UMAPArtifact:
    """Files emitted for one newly improved Stage-1 task checkpoint."""

    task: Task
    round_id: int
    score: float
    output_dir: Path
    figures: Mapping[StageName, Path]
    coordinates: Mapping[StageName, Path]
    summary: Path


def _dataset_labels(dataset: Any) -> np.ndarray:
    """Read labels without loading residue embeddings."""

    locations = getattr(dataset, "locations", None)
    if locations is not None:
        matrix = np.asarray(locations)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError("Cached pair locations must be a 2-D matrix")
        # Current cached rows are [peptide_shard, peptide_row, receptor_shard,
        # receptor_row, label, group_id]. Keep compatibility with old 5-column
        # datasets whose label occupied the final column.
        label_column = 4 if matrix.shape[1] >= 5 else -1
        return matrix[:, label_column].astype(np.int64, copy=False)
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        return np.asarray([int(row[2]) for row in rows], dtype=np.int64)
    raise TypeError(
        "Best-model UMAP requires a CachedPairDataset or OnlinePairDataset"
    )


def stratified_sample_indices(
    labels: Sequence[int] | np.ndarray,
    maximum: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Choose stable row indices proportionally by true label."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError("labels must be a non-empty one-dimensional array")
    if maximum < 2:
        raise ValueError("maximum must be at least 2")
    indices = list(range(len(values)))
    sampled = proportional_stratified_sample(
        indices,
        min(maximum, len(indices)),
        stratum=lambda index: int(values[index]),
        random_state=random_state,
    )
    return np.asarray(sampled, dtype=np.int64)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class Stage1BestUMAPVisualizer:
    """Plot fixed train/validation samples for task-best checkpoints.

    The first best checkpoint for each task and feature stage fits a reference
    PCA+UMAP reducer on the combined train/validation sample. Later task-best
    checkpoints transform those same rows through the fixed reducer, so the
    plots remain comparable across rounds within a task and stage.
    """

    def __init__(
        self,
        output_dir: str | Path,
        val_loaders: Mapping[Task, Mapping[str, Any]],
        *,
        train_loaders: Mapping[Task, Any] | None = None,
        max_samples_per_split: int = 1_000,
        random_state: int = 42,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        threshold: float = 0.5,
        feature_stages: Sequence[StageName] = UMAP_STAGES,
    ):
        if max_samples_per_split < 2:
            raise ValueError("max_samples_per_split must be at least 2")
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if min_dist < 0:
            raise ValueError("min_dist cannot be negative")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if not feature_stages or len(set(feature_stages)) != len(feature_stages):
            raise ValueError("feature_stages must be non-empty and unique")
        allowed_stages = {"encoder", "adapter", "plm", "sequence_encoder", "classifier_input"}
        if set(feature_stages) - allowed_stages:
            raise ValueError("Unknown UMAP feature stage")
        self.feature_stages = tuple(feature_stages)
        try:
            import joblib  # noqa: F401
            import matplotlib  # noqa: F401
            import sklearn  # noqa: F401
            import umap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Best-model UMAP requires matplotlib, scikit-learn, joblib, "
                "and umap-learn from requirements-plm.txt"
            ) from exc

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples_per_split = int(max_samples_per_split)
        self.random_state = int(random_state)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.threshold = float(threshold)
        self.train_loaders = dict(train_loaders or {})
        self.val_loaders = {
            task: dict(loaders)
            for task, loaders in val_loaders.items()
        }
        self.selected_indices: dict[tuple[Task, str], np.ndarray] = {}
        self.selected_labels: dict[tuple[Task, str], np.ndarray] = {}
        self._split_sizes: dict[tuple[Task, str], int] = {}
        self._split_offsets: dict[tuple[Task, str], int] = {}
        self._round_buffers: dict[
            tuple[Task, str],
            list[tuple[dict[StageName, np.ndarray], np.ndarray, np.ndarray]],
        ] = {}
        self.current_round: int | None = None

        split_offset = 0
        for task in ("phla", "ptcr"):
            train_loader = self.train_loaders.get(task)
            if train_loader is not None:
                self._register_split(
                    task,
                    "train",
                    train_loader,
                    random_state=self.random_state + split_offset,
                )
                split_offset += 1
            for split_name, loader in self.val_loaders.get(task, {}).items():
                self._register_split(
                    task,
                    split_name,
                    loader,
                    random_state=self.random_state + split_offset,
                )
                split_offset += 1
        if not self.selected_indices:
            raise ValueError("Best-model UMAP requires train or validation loaders")

    def _register_split(
        self,
        task: Task,
        split_name: str,
        loader: Any,
        *,
        random_state: int,
    ) -> None:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise TypeError(f"UMAP loader has no dataset: {task}/{split_name}")
        labels = _dataset_labels(dataset)
        key = (task, split_name)
        selected = stratified_sample_indices(
            labels,
            self.max_samples_per_split,
            random_state=random_state,
        )
        self.selected_indices[key] = selected
        self.selected_labels[key] = labels[selected]
        self._split_sizes[key] = int(len(labels))

    @staticmethod
    def _source_name(split_name: str) -> str:
        return "train" if split_name == "train" else "val"

    def start_round(self, round_id: int) -> None:
        if round_id < 1:
            raise ValueError("round_id must be positive")
        self.current_round = int(round_id)
        self._round_buffers = {key: [] for key in self.selected_indices}
        self._split_offsets = {key: 0 for key in self.selected_indices}

    def begin_split(self, task: Task, split_name: str) -> None:
        key = (task, split_name)
        if self.current_round is None:
            raise RuntimeError("start_round must be called before begin_split")
        if key not in self.selected_indices:
            raise KeyError(f"Unknown UMAP split: {task}/{split_name}")
        self._split_offsets[key] = 0
        self._round_buffers[key] = []

    def discard_split(self, task: Task, split_name: str) -> None:
        key = (task, split_name)
        if key in self.selected_indices:
            self._split_offsets[key] = 0
            self._round_buffers[key] = []

    def finish_split(self, task: Task, split_name: str) -> None:
        key = (task, split_name)
        if key not in self.selected_indices:
            raise KeyError(f"Unknown UMAP split: {task}/{split_name}")
        observed = self._split_offsets.get(key, 0)
        expected = self._split_sizes[key]
        if observed != expected:
            raise RuntimeError(
                f"Incomplete UMAP split {task}/{split_name}: "
                f"observed {observed}, expected {expected} rows"
            )

    @torch.inference_mode()
    def observe(
        self,
        task: Task,
        split_name: str,
        features: Mapping[StageName, torch.Tensor],
        labels: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> None:
        """Retain only fixed rows from one train/validation batch."""

        key = (task, split_name)
        if key not in self.selected_indices:
            return
        missing = [stage for stage in self.feature_stages if stage not in features]
        if missing:
            raise ValueError(f"Missing UMAP features: {missing}")
        if labels.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("UMAP labels and probabilities must be one-dimensional")
        if labels.shape[0] != probabilities.shape[0]:
            raise ValueError("UMAP labels and probabilities must have matching rows")
        batch_size = int(labels.shape[0])
        for stage in self.feature_stages:
            feature = features[stage]
            if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
                raise ValueError(
                    f"UMAP {stage} features must have shape [batch, dimension]"
                )
            if feature.shape[0] != batch_size:
                raise ValueError(
                    f"UMAP {stage} features and labels must have matching rows"
                )

        start = self._split_offsets[key]
        end = start + batch_size
        selected = self.selected_indices[key]
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, end, side="left"))
        if right > left:
            selected_features: dict[StageName, np.ndarray] = {}
            for stage in self.feature_stages:
                feature = features[stage]
                local = torch.as_tensor(
                    selected[left:right] - start,
                    dtype=torch.long,
                    device=feature.device,
                )
                selected_features[stage] = (
                    feature.index_select(0, local).detach().float().cpu().numpy()
                )
            local = torch.as_tensor(
                selected[left:right] - start,
                dtype=torch.long,
                device=labels.device,
            )
            selected_labels = labels.index_select(0, local).detach().long().cpu().numpy()
            selected_probabilities = (
                probabilities.index_select(0, local)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            self._round_buffers[key].append(
                (selected_features, selected_labels, selected_probabilities)
            )
        self._split_offsets[key] = end

    def _task_arrays(
        self,
        task: Task,
    ) -> tuple[
        dict[StageName, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        keys = [
            key
            for key in sorted(self.selected_indices)
            if key[0] == task
        ]
        if not keys:
            raise RuntimeError(f"No UMAP samples configured for task {task}")

        feature_parts: dict[StageName, list[np.ndarray]] = {
            stage: [] for stage in self.feature_stages
        }
        label_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        split_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        for key in keys:
            _, split_name = key
            chunks = self._round_buffers.get(key, [])
            if not chunks:
                raise RuntimeError(
                    f"No UMAP features were collected for {task}/{split_name}"
                )
            expected_rows = self.selected_indices[key]
            labels = np.concatenate([chunk[1] for chunk in chunks])
            probabilities = np.concatenate([chunk[2] for chunk in chunks])
            if len(labels) != len(expected_rows):
                raise RuntimeError(
                    f"Incomplete UMAP sample for {task}/{split_name}: "
                    f"collected {len(labels)}, expected {len(expected_rows)}"
                )
            if not np.array_equal(labels, self.selected_labels[key]):
                raise RuntimeError(
                    f"Dataset order or labels changed for UMAP split "
                    f"{task}/{split_name}"
                )
            for stage in self.feature_stages:
                features = np.concatenate(
                    [chunk[0][stage] for chunk in chunks],
                    axis=0,
                )
                if len(features) != len(expected_rows):
                    raise RuntimeError(
                        f"Incomplete {stage} UMAP sample for {task}/{split_name}"
                    )
                feature_parts[stage].append(
                    features.astype(np.float32, copy=False)
                )
            label_parts.append(labels)
            probability_parts.append(probabilities)
            source_parts.append(
                np.full(
                    len(labels),
                    self._source_name(split_name),
                    dtype=object,
                )
            )
            split_parts.append(np.full(len(labels), split_name, dtype=object))
            row_parts.append(expected_rows.copy())

        return (
            {
                stage: np.concatenate(feature_parts[stage]).astype(
                    np.float32,
                    copy=False,
                )
                for stage in self.feature_stages
            },
            np.concatenate(label_parts),
            np.concatenate(probability_parts),
            np.concatenate(source_parts),
            np.concatenate(split_parts),
            np.concatenate(row_parts),
        )

    @staticmethod
    def _sample_fingerprint(
        task: Task,
        stage: StageName,
        source_names: np.ndarray,
        split_names: np.ndarray,
        row_indices: np.ndarray,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(f"{task}:{stage}".encode("utf-8"))
        for source, split_name, row_index in zip(
            source_names,
            split_names,
            row_indices,
        ):
            digest.update(
                f"{source}:{split_name}:{int(row_index)}\n".encode("utf-8")
            )
        return digest.hexdigest()

    def _fit_or_transform(
        self,
        task: Task,
        stage: StageName,
        features: np.ndarray,
        *,
        sample_fingerprint: str,
    ) -> tuple[np.ndarray, str, int]:
        import joblib
        import umap
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize

        reference_path = self.output_dir / task / f"reference_reducer_{stage}.joblib"
        normalized = normalize(features, norm="l2").astype(np.float32, copy=False)
        if reference_path.is_file():
            state = joblib.load(reference_path)
            if state.get("sample_fingerprint") != sample_fingerprint:
                raise ValueError(
                    f"Existing UMAP reference sample does not match {task}/{stage}"
                )
            if int(state.get("feature_dimension", -1)) != features.shape[1]:
                raise ValueError(
                    f"Existing UMAP reference dimension does not match {task}/{stage}"
                )
            pca = state["pca"]
            reducer = state["umap"]
            preprocessed = normalize(
                pca.transform(normalized), norm="l2"
            ).astype(np.float32, copy=False)
            coordinates = reducer.transform(preprocessed).astype(np.float32)
            return coordinates, "reference_transform", int(pca.n_components_)

        components = min(50, features.shape[1], features.shape[0] - 1)
        if components < 2:
            raise ValueError("UMAP requires at least three samples and two features")
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        preprocessed = normalize(
            pca.fit_transform(normalized), norm="l2"
        ).astype(np.float32, copy=False)
        effective_neighbors = min(
            self.n_neighbors,
            max(2, features.shape[0] - 1),
        )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric="cosine",
            random_state=self.random_state,
            transform_seed=self.random_state,
            n_jobs=1,
        )
        coordinates = reducer.fit_transform(preprocessed).astype(np.float32)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = reference_path.with_suffix(".joblib.tmp")
        joblib.dump(
            {
                "pca": pca,
                "umap": reducer,
                "sample_fingerprint": sample_fingerprint,
                "feature_dimension": int(features.shape[1]),
                "task": task,
                "stage": stage,
                "random_state": self.random_state,
                "n_neighbors": effective_neighbors,
                "min_dist": self.min_dist,
            },
            temporary,
        )
        temporary.replace(reference_path)
        return coordinates, "reference_fit", components

    @staticmethod
    def _plot_train_val(
        path: Path,
        coordinates: np.ndarray,
        source_names: np.ndarray,
        labels: np.ndarray,
        *,
        title: str,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        source_palette = {
            "train": "#2563eb",
            "val": "#ea580c",
        }
        label_markers = {0: "o", 1: "^"}
        label_names = {0: "negative (0)", 1: "positive (1)"}
        figure, axis = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
        for source in ("train", "val"):
            for label in (0, 1):
                selected = (source_names == source) & (labels == label)
                if not np.any(selected):
                    continue
                axis.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    s=22,
                    alpha=0.64,
                    linewidths=0,
                    rasterized=True,
                    color=source_palette[source],
                    marker=label_markers[label],
                    label=f"{source} {label_names[label]}",
                )
        present_sources = [
            source
            for source in ("train", "val")
            if np.any(source_names == source)
        ]
        present_labels = [
            label for label in (0, 1) if np.any(labels == label)
        ]
        source_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=source_palette[source],
                markeredgecolor="none",
                label=source,
            )
            for source in present_sources
        ]
        label_handles = [
            Line2D(
                [],
                [],
                marker=label_markers[label],
                linestyle="None",
                markersize=7,
                color="#4a5568",
                markerfacecolor="#4a5568",
                markeredgecolor="none",
                label=label_names[label],
            )
            for label in present_labels
        ]
        if source_handles:
            source_legend = axis.legend(
                handles=source_handles,
                title="Source / color",
                loc="upper right",
                frameon=False,
            )
            axis.add_artist(source_legend)
        if label_handles:
            axis.legend(
                handles=label_handles,
                title="Label / marker",
                loc="lower right",
                frameon=False,
            )
        x_padding = max(1e-6, float(np.ptp(coordinates[:, 0])) * 0.04)
        y_padding = max(1e-6, float(np.ptp(coordinates[:, 1])) * 0.04)
        axis.set(
            title=title,
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            xlim=(
                float(coordinates[:, 0].min() - x_padding),
                float(coordinates[:, 0].max() + x_padding),
            ),
            ylim=(
                float(coordinates[:, 1].min() - y_padding),
                float(coordinates[:, 1].max() + y_padding),
            ),
        )
        axis.grid(alpha=0.15, linewidth=0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=240)
        plt.close(figure)

    @staticmethod
    def _write_coordinates(
        path: Path,
        *,
        task: Task,
        round_id: int,
        score: float,
        stage: StageName,
        source_names: np.ndarray,
        split_names: np.ndarray,
        row_indices: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        coordinates: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "task",
                    "round",
                    "score",
                    "stage",
                    "source",
                    "split",
                    "dataset_index",
                    "label",
                    "prediction_probability",
                    "outcome",
                    "umap_1",
                    "umap_2",
                ]
            )
            for index in range(len(labels)):
                writer.writerow(
                    [
                        task,
                        round_id,
                        score,
                        stage,
                        source_names[index],
                        split_names[index],
                        int(row_indices[index]),
                        int(labels[index]),
                        float(probabilities[index]),
                        outcomes[index],
                        float(coordinates[index, 0]),
                        float(coordinates[index, 1]),
                    ]
                )

    def _write_index(self) -> None:
        entries = []
        for path in sorted(self.output_dir.glob("*/round_*/summary.json")):
            with path.open("r", encoding="utf-8") as handle:
                entries.append(json.load(handle))
        entries.sort(key=lambda item: (item["task"], item["round"]))
        _atomic_json(
            self.output_dir / "index.json",
            {
                "projection_policy": (
                    "Each task/stage fits a first-best reference PCA+UMAP on "
                    "the combined fixed train/validation sample; later task-best "
                    "checkpoints transform into that same task/stage space."
                ),
                "snapshot_policy": "Render only when the task selection score improves.",
                "snapshots": entries,
            },
        )

    def render_task(self, task: Task, *, round_id: int, score: float) -> UMAPArtifact:
        if self.current_round != round_id:
            raise RuntimeError(
                f"UMAP round mismatch: collected {self.current_round}, requested {round_id}"
            )
        (
            feature_sets,
            labels,
            probabilities,
            source_names,
            split_names,
            row_indices,
        ) = self._task_arrays(task)
        outcomes = np.asarray(
            [
                "TP" if int(label) == 1 and probability > self.threshold else
                "FN" if int(label) == 1 else
                "FP" if probability > self.threshold else "TN"
                for label, probability in zip(labels, probabilities)
            ],
            dtype=object,
        )
        round_dir = self.output_dir / task / f"round_{round_id:04d}"
        figure_paths: dict[StageName, Path] = {}
        coordinate_paths: dict[StageName, Path] = {}
        stage_summaries: dict[str, dict[str, Any]] = {}
        for stage in self.feature_stages:
            features = feature_sets[stage]
            fingerprint = self._sample_fingerprint(
                task,
                stage,
                source_names,
                split_names,
                row_indices,
            )
            coordinates, projection_mode, pca_components = self._fit_or_transform(
                task,
                stage,
                features,
                sample_fingerprint=fingerprint,
            )
            figure_path = round_dir / f"umap_{stage}.png"
            coordinate_path = round_dir / f"coordinates_{stage}.csv.gz"
            task_title = "pHLA" if task == "phla" else "pTCR"
            self._plot_train_val(
                figure_path,
                coordinates,
                source_names,
                labels,
                title=(
                    f"{task_title} task-best | round {round_id} | "
                    f"score={score:.6f} | {stage}"
                ),
            )
            self._write_coordinates(
                coordinate_path,
                task=task,
                round_id=round_id,
                score=score,
                stage=stage,
                source_names=source_names,
                split_names=split_names,
                row_indices=row_indices,
                labels=labels,
                probabilities=probabilities,
                outcomes=outcomes,
                coordinates=coordinates,
            )
            figure_paths[stage] = figure_path
            coordinate_paths[stage] = coordinate_path
            stage_summaries[stage] = {
                "feature_dimension": int(features.shape[1]),
                "pca_components": int(pca_components),
                "projection_mode": projection_mode,
                "sample_fingerprint": fingerprint,
                "figure": str(figure_path),
                "coordinates": str(coordinate_path),
            }

        source_counts = {
            source: int(np.sum(source_names == source))
            for source in ("train", "val")
        }
        split_counts = {
            split_name: int(np.sum(split_names == split_name))
            for split_name in sorted(set(str(value) for value in split_names))
        }
        label_counts = {
            source: {
                str(label): int(
                    np.sum((source_names == source) & (labels == label))
                )
                for label in (0, 1)
            }
            for source in ("train", "val")
        }
        outcome_counts = {
            outcome: int(np.sum(outcomes == outcome))
            for outcome in ("TN", "FP", "FN", "TP")
        }
        summary = {
            "task": task,
            "round": int(round_id),
            "score": float(score),
            "sample_count": int(len(labels)),
            "source_counts": source_counts,
            "split_counts": split_counts,
            "label_counts": label_counts,
            "outcome_counts": outcome_counts,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "stages": stage_summaries,
            "coordinate_warning": (
                "Coordinates are comparable only within the same task and "
                "feature stage because each task/stage has its own first-best "
                "reference reducer."
            ),
        }
        summary_path = round_dir / "summary.json"
        _atomic_json(summary_path, summary)
        self._write_index()
        return UMAPArtifact(
            task=task,
            round_id=round_id,
            score=score,
            output_dir=round_dir,
            figures=figure_paths,
            coordinates=coordinate_paths,
            summary=summary_path,
        )
