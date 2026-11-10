"""Tests for Stage-1 multi-stage train/validation UMAP snapshots."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from source.plm_unified.config import (
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    SwanLabConfig,
    TrainingConfig,
)
from source.plm_unified.metrics import BinaryMetrics
from source.plm_unified.model import UnifiedBindingModel
from source.plm_unified.stage1_umap import (
    UMAP_STAGES,
    Stage1BestUMAPVisualizer,
    stratified_sample_indices,
)
from source.plm_unified.trainer import EpochResult, Stage1Trainer


class _Dataset:
    def __init__(self, labels):
        self.rows = [
            (f"PEPTIDE{index:03d}", f"RECEPTOR{index:03d}", int(label))
            for index, label in enumerate(labels)
        ]


class _Loader:
    def __init__(self, labels):
        self.dataset = _Dataset(labels)


def _metrics(score: float) -> BinaryMetrics:
    return BinaryMetrics(
        loss=1.0 - score,
        auroc=score,
        aupr=score,
        accuracy=score,
        mcc=score,
        f1=score,
        precision=score,
        recall=score,
        sensitivity=score,
        specificity=score,
        samples=4,
    )


def _epoch_result() -> EpochResult:
    return EpochResult(
        metrics=_metrics(0.5),
        clean_loss=0.5,
        adversarial_loss=0.0,
        total_loss=0.5,
        optimizer_steps=1,
        samples=4,
        attacked_parameter_count=0.0,
        fgm_mean_gradient_norm=0.0,
        fgm_max_gradient_norm=0.0,
        model_gradient_norm=0.0,
    )


def _feature_batch(start: int, stop: int) -> dict[str, torch.Tensor]:
    size = stop - start
    return {
        stage: torch.arange(
            start * 4,
            stop * 4,
            dtype=torch.float32,
        ).reshape(size, 4)
        for stage in UMAP_STAGES
    }


def test_stratified_sample_indices_are_balanced_and_reproducible():
    labels = np.asarray([0] * 80 + [1] * 20)
    first = stratified_sample_indices(labels, 50, random_state=42)
    second = stratified_sample_indices(labels, 50, random_state=42)
    np.testing.assert_array_equal(first, second)
    assert np.bincount(labels[first], minlength=2).tolist() == [40, 10]


def test_model_returns_three_pair_level_feature_spaces():
    config = ModelConfig(
        peptide_input_dim=12,
        hla_input_dim=12,
        tcr_input_dim=8,
        adapter_hidden_dim=16,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
    )
    model = UnifiedBindingModel(config).eval()
    peptide_hidden = torch.randn(2, 5, 12)
    peptide_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

    for task, receptor_dim in (("phla", 12), ("ptcr", 8)):
        receptor_hidden = torch.randn(2, 4, receptor_dim)
        receptor_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            dtype=torch.bool,
        )
        output = model(
            task,
            peptide_hidden,
            peptide_mask,
            receptor_hidden,
            receptor_mask,
            return_features=True,
        )
        expected_encoder = torch.cat(
            (
                (peptide_hidden * peptide_mask.unsqueeze(-1)).sum(1)
                / peptide_mask.sum(1, keepdim=True),
                (receptor_hidden * receptor_mask.unsqueeze(-1)).sum(1)
                / receptor_mask.sum(1, keepdim=True),
            ),
            dim=-1,
        )
        torch.testing.assert_close(output.encoder_embedding, expected_encoder)
        assert output.adapter_embedding.shape == (2, 16)
        assert output.pooled_embedding.shape == (2, 8)

        default_output = model(
            task,
            peptide_hidden,
            peptide_mask,
            receptor_hidden,
            receptor_mask,
        )
        assert default_output.encoder_embedding is None
        assert default_output.adapter_embedding is None


def test_visualizer_combines_train_and_all_val_splits_per_stage(tmp_path: Path):
    labels = np.asarray([0, 1] * 10)
    val_loaders = {
        "phla": {
            "val_unseen_hla": _Loader(labels),
            "val_unseen_pair": _Loader(labels),
        },
        "ptcr": {},
    }
    train_loaders = {"phla": _Loader(labels)}
    visualizer = Stage1BestUMAPVisualizer(
        tmp_path,
        val_loaders,
        train_loaders=train_loaders,
        max_samples_per_split=10,
        random_state=7,
        n_neighbors=3,
    )
    visualizer.start_round(1)
    for split_name in ("train", "val_unseen_hla", "val_unseen_pair"):
        visualizer.begin_split("phla", split_name)
        for start in (0, 7, 14):
            stop = min(20, start + 7)
            visualizer.observe(
                "phla",
                split_name,
                _feature_batch(start, stop),
                torch.from_numpy(labels[start:stop]),
                torch.linspace(0.05, 0.95, stop - start),
            )
        visualizer.finish_split("phla", split_name)

    projection_calls = []
    plot_calls = []

    def fake_projection(task, stage, values, *, sample_fingerprint):
        projection_calls.append((task, stage, values.shape, sample_fingerprint))
        return values[:, :2].astype(np.float32, copy=False), "test_projection", 2

    def fake_plot(path, coordinates, source_names, labels, *, title):
        plot_calls.append(
            (path, coordinates, source_names.copy(), labels.copy(), title)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")

    visualizer._fit_or_transform = fake_projection
    visualizer._plot_train_val = fake_plot
    artifact = visualizer.render_task("phla", round_id=1, score=0.7)

    assert set(artifact.figures) == set(UMAP_STAGES)
    assert set(artifact.coordinates) == set(UMAP_STAGES)
    assert len(projection_calls) == len(UMAP_STAGES)
    assert len(plot_calls) == len(UMAP_STAGES)
    for path, coordinates, sources, observed_labels, _ in plot_calls:
        assert path.is_file()
        assert coordinates.shape == (30, 2)
        assert set(sources.tolist()) == {"train", "val"}
        assert set(observed_labels.tolist()) == {0, 1}

    coordinate_path = artifact.coordinates["encoder"]
    with gzip.open(coordinate_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert {row["source"] for row in rows} == {"train", "val"}
    assert {row["split"] for row in rows} == {
        "train",
        "val_unseen_hla",
        "val_unseen_pair",
    }
    assert {row["stage"] for row in rows} == {"encoder"}

    summary = json.loads(artifact.summary.read_text())
    assert summary["sample_count"] == 30
    assert summary["source_counts"] == {"train": 10, "val": 20}
    assert summary["split_counts"] == {
        "train": 10,
        "val_unseen_hla": 10,
        "val_unseen_pair": 10,
    }
    assert set(summary["stages"]) == set(UMAP_STAGES)
    assert json.loads((tmp_path / "index.json").read_text())["snapshots"] == [
        summary
    ]


def test_trainer_collects_train_only_for_new_task_best(tmp_path: Path):
    config = ExperimentConfig(
        paths=PathsConfig(
            project_root=tmp_path,
            data_root=tmp_path,
            cache_root=tmp_path / "cache",
            output_root=tmp_path / "models",
            swanlog_root=tmp_path / "swanlog",
        ),
        model=ModelConfig(
            peptide_input_dim=12,
            hla_input_dim=12,
            tcr_input_dim=8,
            adapter_hidden_dim=16,
            model_dim=8,
            num_attention_heads=2,
            feedforward_dim=16,
        ),
        training=TrainingConfig(
            warmup_epochs_per_task=0,
            max_rounds=1,
            early_stopping_patience=1,
            mixed_precision="none",
            device="cpu",
        ),
        swanlab=SwanLabConfig(mode="disabled"),
    )

    class Tracker:
        run_id = None

        def __init__(self):
            self.records = []

        def log(self, values, *, step=None):
            self.records.append((dict(values), step))

    class Visualizer:
        def __init__(self):
            self.output_dir = tmp_path / "umap"
            self.rounds = []
            self.begun = []
            self.finished = []
            self.renders = []

        def start_round(self, round_id):
            self.rounds.append(round_id)

        def begin_split(self, task, split_name):
            self.begun.append((task, split_name))

        def finish_split(self, task, split_name):
            self.finished.append((task, split_name))

        def observe(self, task, split_name, features, labels, probabilities):
            del task, split_name, features, labels, probabilities

        def render_task(self, task, *, round_id, score):
            self.renders.append((task, round_id, score))
            return SimpleNamespace(
                figures={
                    stage: self.output_dir / task / f"{stage}.png"
                    for stage in UMAP_STAGES
                },
                coordinates={
                    stage: self.output_dir / task / f"{stage}.csv.gz"
                    for stage in UMAP_STAGES
                },
            )

    loaders = {"phla": [object()], "ptcr": [object()]}
    validation = {
        "phla": {"val": object()},
        "ptcr": {"val": object()},
    }
    tracker = Tracker()
    visualizer = Visualizer()
    trainer = Stage1Trainer(
        UnifiedBindingModel(config.model),
        config,
        loaders,
        validation,
        tracker,
        run_timestamp="20260914_120000_000001",
        best_umap_visualizer=visualizer,
    )
    trainer.stage = "stage1b"
    trainer.train_epoch = lambda *args, **kwargs: _epoch_result()
    trainer.evaluate = lambda task, loader, **kwargs: _metrics(
        0.7 if task == "phla" else 0.6
    )
    collected_train = []
    trainer._collect_umap_train = lambda task: collected_train.append(task)
    trainer.fit()

    assert visualizer.rounds == [1]
    assert collected_train == ["phla", "ptcr"]
    assert [(task, round_id) for task, round_id, _ in visualizer.renders] == [
        ("phla", 1),
        ("ptcr", 1),
    ]
    logged_keys = {key for values, _ in tracker.records for key in values}
    for task in ("phla", "ptcr"):
        for stage in UMAP_STAGES:
            assert (
                f"visualization/best_umap/{task}/{stage}_figure" in logged_keys
            )
            assert (
                f"visualization/best_umap/{task}/{stage}_coordinates" in logged_keys
            )
