"""Tests for the Stage1-v2 training safeguards and diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from source.plm_unified.config import (  # noqa: E402
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    SwanLabConfig,
    TrainingConfig,
)
from source.plm_unified.data import stable_group_id  # noqa: E402
from source.plm_unified.metrics import compute_binary_metrics  # noqa: E402
from source.plm_unified.model import UnifiedBindingModel  # noqa: E402
from source.plm_unified.trainer import (  # noqa: E402
    EpochResult,
    Stage1Trainer,
    TaskAwareCosineScheduler,
    _selection_calibration_guard,
    build_optimizer,
    build_scheduler,
)


class _Loader:
    def __init__(self, group_ids, labels):
        self.dataset = type(
            "Dataset",
            (),
            {
                "group_ids": np.asarray(group_ids, dtype=np.int64),
                "labels": np.asarray(labels, dtype=np.int64),
            },
        )()

    def __len__(self):
        return 1


class _Tracker:
    run_id = None

    def log(self, values, *, step=None):
        del values, step


def _config(tmp_path: Path, **training_values):
    return ExperimentConfig(
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
            max_rounds=2,
            mixed_precision="none",
            device="cpu",
            **training_values,
        ),
        swanlab=SwanLabConfig(mode="disabled"),
    )


def test_metrics_report_calibration_and_group_robustness():
    metrics = compute_binary_metrics(
        [0, 1, 0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
        loss=0.5,
        strict_threshold=True,
        group_ids=[1, 1, 2, 2, 3, 3],
    )

    assert metrics.nll > 0
    assert metrics.brier >= 0
    assert 0 <= metrics.ece <= 1
    assert metrics.group_count == 3
    assert metrics.worst_group_nll >= metrics.group_macro_nll


def test_selection_calibration_guard_requires_nll_and_ece_tolerances():
    assert _selection_calibration_guard(
        0.54,
        0.11,
        0.50,
        0.10,
        enabled=True,
        nll_tolerance=0.05,
        ece_tolerance=0.02,
    ) == (True, True, True)
    assert _selection_calibration_guard(
        0.56,
        0.11,
        0.50,
        0.10,
        enabled=True,
        nll_tolerance=0.05,
        ece_tolerance=0.02,
    ) == (False, True, False)
    assert _selection_calibration_guard(
        0.50,
        0.13,
        0.50,
        0.10,
        enabled=True,
        nll_tolerance=0.05,
        ece_tolerance=0.02,
    ) == (True, False, False)
    assert _selection_calibration_guard(
        10.0,
        10.0,
        0.50,
        0.10,
        enabled=False,
        nll_tolerance=0.0,
        ece_tolerance=0.0,
    ) == (True, True, True)


def test_epoch_result_logs_clean_adversarial_total_and_unweighted_metrics():
    metrics = compute_binary_metrics(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        loss=0.7,
    )
    result = EpochResult(
        metrics=metrics,
        clean_loss=0.7,
        adversarial_loss=0.9,
        total_loss=0.8,
        optimizer_steps=2,
        samples=4,
        attacked_parameter_count=3,
        fgm_mean_gradient_norm=0.2,
        fgm_max_gradient_norm=0.4,
        model_gradient_norm=0.3,
    )
    logged = result.to_log_dict("round/ptcr/train")
    assert logged["round/ptcr/train/clean_loss"] == pytest.approx(0.7)
    assert logged["round/ptcr/train/adversarial_loss"] == pytest.approx(0.9)
    assert logged["round/ptcr/train/total_loss"] == pytest.approx(0.8)
    assert logged["round/ptcr/train/unweighted_nll"] == pytest.approx(metrics.nll)
    assert logged["round/ptcr/train/unweighted_ece"] == pytest.approx(metrics.ece)


def test_task_aware_optimizer_has_separate_lr_and_no_decay_groups(tmp_path: Path):
    config = _config(
        tmp_path,
        shared_peptide_adapter_lr=5e-5,
        hla_adapter_lr=1e-4,
        ptcr_classifier_lr=7.5e-5,
        no_decay_bias_and_norm=True,
    )
    model = UnifiedBindingModel(config.model)
    optimizer = build_optimizer(model, config)
    names = {group["name"] for group in optimizer.param_groups}
    assert "shared_peptide_adapter/decay" in names
    assert "phla_classifier/no_decay" in names
    assert "ptcr_classifier/decay" in names
    learning_rates = {
        group["name"]: group["lr"] for group in optimizer.param_groups
    }
    assert learning_rates["shared_peptide_adapter/decay"] == pytest.approx(5e-5)
    assert learning_rates["ptcr_classifier/decay"] == pytest.approx(7.5e-5)
    assert all(
        group["weight_decay"] == 0
        for group in optimizer.param_groups
        if group["name"].endswith("/no_decay")
    )


def test_scheduler_uses_explicit_budget_and_minimum_lr():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0}])
    scheduler = build_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.05,
    )
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_task_scheduler_does_not_advance_frozen_private_branch(tmp_path: Path):
    config = _config(tmp_path)
    model = UnifiedBindingModel(config.model)
    optimizer = build_optimizer(model, config)
    scheduler = TaskAwareCosineScheduler(
        optimizer,
        total_steps=8,
        warmup_ratio=0.0,
        warmup_steps=0,
        min_lr_ratio=0.1,
        task_total_steps={"phla": 4, "ptcr": 4},
        task_warmup_steps={"phla": 0, "ptcr": 0},
    )

    scheduler.step("phla")
    steps_by_name = {
        group["name"]: scheduler.group_steps[index]
        for index, group in enumerate(optimizer.param_groups)
    }
    assert steps_by_name["shared_peptide_adapter/decay"] == 1
    assert steps_by_name["phla_classifier/decay"] == 1
    assert steps_by_name["ptcr_classifier/decay"] == 0

    scheduler.step("ptcr")
    steps_by_name = {
        group["name"]: scheduler.group_steps[index]
        for index, group in enumerate(optimizer.param_groups)
    }
    assert steps_by_name["shared_peptide_adapter/decay"] == 2
    assert steps_by_name["phla_classifier/decay"] == 1
    assert steps_by_name["ptcr_classifier/decay"] == 1


def test_inverse_sqrt_group_weights_are_deterministic(tmp_path: Path):
    config = _config(
        tmp_path,
        phla_group_weighting="inverse_sqrt",
        ptcr_group_weighting="inverse_sqrt",
    )
    loaders = {
        "phla": _Loader([1, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]),
        "ptcr": _Loader([4, 4, 5, 5], [0, 1, 0, 1]),
    }
    trainer = Stage1Trainer(
        UnifiedBindingModel(config.model),
        config,
        loaders,
        {"phla": {}, "ptcr": {}},
        _Tracker(),
    )
    weights = trainer._group_weight_maps["phla"]
    assert weights
    assert min(weights.values()) >= config.training.group_weight_min
    assert max(weights.values()) <= config.training.group_weight_max
    assert stable_group_id("PEPTIDE") == stable_group_id(" peptide ")
