"""Contract tests for Stage-12 retraining aligned with Stage 8."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from source import evaluate_plm_stage8_benchmarks as benchmark_evaluator
from source import train_plm_stage12 as stage12_cli

from source.plm_unified.stage12_config import (
    Stage12ExperimentConfig,
    Stage12Paths,
    load_stage12_config,
)
from source.plm_unified.stage12_data import (
    PeptideAwareBalancedSampler,
    make_stage12_online_loaders,
    make_pt_replay_loader,
    pt_row_sampling_weights,
    replay_sample_count,
)
from source.plm_unified.config import ModelConfig
from source.plm_unified.stage8_data import Stage8AuditResult, Stage8Record
from source.plm_unified.stage12_evaluation import (
    Stage12BestTracker,
    Stage12SelectionDecision,
    Stage12ValidationScores,
    evaluate_pair_validation_splits,
    evaluate_stage12_validation,
    task_score,
)
from source.plm_unified.stage12_checkpoint import (
    load_stage12_checkpoint,
    restore_stage12_last_checkpoint,
    save_stage12_checkpoint,
    validate_stage12_checkpoint,
)
from source.plm_unified.stage12_trainer import (
    Stage12OnlineTrainer,
    compute_stage12_joint_loss,
    stage12_pass_order,
    stage12_replay_samples,
)


def _config() -> Stage12ExperimentConfig:
    root = Path("/tmp/stage12-contract")
    return Stage12ExperimentConfig(
        paths=Stage12Paths(
            project_root=root,
            data_root=root / "dataset",
            output_root=root / "trained_model" / "PLM_Stage12",
            swanlog_root=root / "swanlog" / "PLM_Stage12",
        )
    )


def test_stage12_defaults_encode_the_approved_training_contract() -> None:
    config = _config()
    config.validate()

    assert (config.runtime.fold, config.runtime.seed) == (1, 42)
    assert config.stage12.stage12a_epochs == 2
    assert config.stage12.condition == "full"
    assert (
        config.stage12.phla_loss_weight,
        config.stage12.ptcr_loss_weight,
        config.stage12.pht_loss_weight,
    ) == (0.3, 0.2, 1.0)
    assert config.stage12.normalized_task_weights == pytest.approx((0.2, 0.1333333333, 0.6666666667))
    assert (
        config.stage12.phla_replay_ratio,
        config.stage12.ptcr_replay_ratio,
    ) == (0.25, 0.25)
    assert (
        config.stage12.hierarchy_lambda,
        config.stage12.pt_implication_lambda,
        config.stage12.constraint_margin,
        config.stage12.constraint_ramp_rounds,
    ) == (0.05, 0.03, 0.05, 5)
    assert config.stage12.pht_joint_weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert config.stage12.pt_selection_splits == (
        "val_unseen_pair",
        "val_unseen_peptide",
        "val_unseen_tcr",
    )
    assert config.stage12.ph_selection_splits == (
        "val_unseen_hla",
        "val_unseen_pair",
        "val_unseen_peptide",
    )


def test_repository_stage12_yaml_pins_stage1_and_stage8_reference() -> None:
    config = load_stage12_config(
        Path(__file__).resolve().parents[2] / "configs" / "plm_stage12.yaml"
    )

    assert config.paths.stage1_sha256 == (
        "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96"
    )
    assert config.paths.stage8_reference_sha256 == (
        "575af59f232ed1e8d86aaba4286953eb0d8194243054f492092c1725effad624"
    )
    assert config.paths.output_root.name == "PLM_Stage12"
    assert config.swanlab.mode == "online"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage12a_epochs", 1),
        ("condition", "baseline"),
        ("ptcr_replay_ratio", 0.75),
        ("pht_loss_weight", 0.6),
        ("pht_joint_weights", (0.4, 0.2, 0.4)),
        ("task_selection_weights", (0.4, 0.2, 0.4)),
    ],
)
def test_stage12_rejects_drift_from_the_registered_design(field: str, value) -> None:
    config = _config()
    setattr(config.stage12, field, value)

    with pytest.raises(ValueError, match=field):
        config.validate()


def test_pt_row_weights_give_each_peptide_sqrt_frequency_mass_per_label() -> None:
    rows = [
        ("COMMON", "T1", 0),
        ("COMMON", "T2", 0),
        ("COMMON", "T3", 0),
        ("COMMON", "T4", 0),
        ("RARE", "T5", 0),
        ("COMMON", "T6", 1),
        ("RARE", "T7", 1),
    ]

    weights = pt_row_sampling_weights(rows)

    assert weights[:4] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert weights[4:] == pytest.approx([1.0, 1.0, 1.0])
    # Four COMMON rows carry total mass 2=sqrt(4); RARE carries sqrt(1)=1.
    assert sum(weights[:4]) == pytest.approx(2.0)
    assert weights[4] == pytest.approx(1.0)


def test_pt_sampler_is_balanced_unique_and_deterministic_per_round() -> None:
    rows = [
        (f"P{index % 4}", f"T{index}", index % 2)
        for index in range(40)
    ]
    sampler = PeptideAwareBalancedSampler(rows, num_samples=18, seed=42)

    round_one = list(sampler)
    sampler.set_round(1)
    repeated_round_one = list(sampler)
    sampler.set_round(2)
    round_two = list(sampler)

    assert round_one == repeated_round_one
    assert round_two != round_one
    assert len(round_one) == len(set(round_one)) == 18
    assert Counter(rows[index][2] for index in round_one) == {0: 9, 1: 9}


def test_pt_sampler_rejects_a_request_larger_than_either_balanced_class() -> None:
    rows = [("P", f"T{index}", 0) for index in range(3)] + [
        ("P", f"U{index}", 1) for index in range(8)
    ]

    with pytest.raises(ValueError, match="without replacement"):
        PeptideAwareBalancedSampler(rows, num_samples=8, seed=42)


def test_pt_replay_loader_uses_the_registered_sampler(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "peptide,tcr,label\n"
        + "".join(
            f"PEPTIDEA,TCR{index},{index % 2}\n" for index in range(20)
        ),
        encoding="utf-8",
    )

    loader = make_pt_replay_loader(
        csv_path,
        micro_batch_size=4,
        num_samples=12,
        seed=42,
        num_workers=0,
        peptide_min_length=8,
        peptide_max_length=15,
    )
    labels = torch.cat([batch["labels"] for batch in loader]).tolist()

    assert isinstance(loader.sampler, PeptideAwareBalancedSampler)
    assert Counter(labels) == {0: 6, 1: 6}
    assert len(labels) == 12


def test_stage12_loader_bundle_keeps_four_type_balance_and_pt_cap(tmp_path: Path) -> None:
    records = [
        Stage8Record("PEPTIDEA", f"T{kind}", "HLA", int(kind[0]), int(kind[1]), int(kind[2]), index)
        for index, kind in enumerate(("000", "010", "100", "111"), start=2)
    ]
    audit = Stage8AuditResult(
        train_records=records,
        val_records=records,
        report={}, conflicts=[], errors=[], rejected_rows=[], dataset_fingerprint="fingerprint",
    )
    ph_path = tmp_path / "ph.csv"
    pt_path = tmp_path / "pt.csv"
    ph_path.write_text(
        "peptide,HLA,label\n" + "".join(
            f"PEPTIDEA,HLA{index},{index % 2}\n" for index in range(20)
        ), encoding="utf-8",
    )
    pt_path.write_text(
        "peptide,tcr,label\n" + "".join(
            f"PEPTIDEA,TCR{index},{index % 2}\n" for index in range(20)
        ), encoding="utf-8",
    )

    train, validation = make_stage12_online_loaders(
        audit,
        phla_path=ph_path,
        ptcr_path=pt_path,
        model=ModelConfig(),
        micro_batch_size=4,
        eval_batch_size=4,
        phla_replay_ratio=0.25,
        ptcr_replay_ratio=0.25,
        seed=42,
        num_workers=0,
    )

    assert set(train) == {"phla", "ptcr", "joint"}
    assert not isinstance(train["ptcr"].sampler, PeptideAwareBalancedSampler)
    assert len(train["ptcr"].dataset) == 20
    assert len(next(iter(train["joint"]))["label_types"]) == 4
    assert len(validation.dataset) == 4


def test_replay_sample_count_applies_separate_caps() -> None:
    assert replay_sample_count(72_190, 664_952, 0.75) == 54_142
    assert replay_sample_count(72_190, 664_952, 0.25) == 18_047
    assert replay_sample_count(100, 12, 0.75) == 12


def _metrics(value: float) -> dict[str, float]:
    return {
        "auroc": value,
        "aupr": value,
        "accuracy": value,
        "mcc": value,
        "f1": value,
        "loss": 1.0 - value,
    }


def _scores(
    pair: float,
    peptide: float,
    tcr: float,
    *,
    ph: float,
    pht: float,
) -> Stage12ValidationScores:
    return Stage12ValidationScores.from_metrics(
        pt_metrics={
            "val_unseen_pair": _metrics(pair),
            "val_unseen_peptide": _metrics(peptide),
            "val_unseen_tcr": _metrics(tcr),
        },
        ph_metrics={
            "val_unseen_hla": _metrics(ph),
            "val_unseen_pair": _metrics(ph),
            "val_unseen_peptide": _metrics(ph),
        },
        pht_joint=pht,
    )


def test_task_score_is_the_registered_five_metric_average() -> None:
    metrics = {
        "auroc": 0.9,
        "aupr": 0.8,
        "accuracy": 0.7,
        "mcc": 0.6,
        "f1": 0.5,
        "loss": 99.0,
    }

    assert task_score(metrics) == pytest.approx(0.7)


def test_validation_scores_use_equal_split_macros() -> None:
    scores = _scores(0.8, 0.5, 0.9, ph=0.7, pht=0.75)

    assert scores.pt_macro == pytest.approx((0.8 + 0.5 + 0.9) / 3)
    assert scores.ph_macro == pytest.approx(0.7)
    assert scores.pht_joint == pytest.approx(0.75)
    assert scores.joint_score == pytest.approx((scores.pt_macro + scores.ph_macro + scores.pht_joint) / 3)


def test_best_tracker_selects_equal_task_joint_without_pt_gate() -> None:
    reference = _scores(0.70, 0.60, 0.80, ph=0.75, pht=0.78)
    tracker = Stage12BestTracker(reference)

    with pytest.raises(ValueError, match="Stage12B"):
        tracker.consider(0, _scores(0.9, 0.9, 0.9, ph=0.9, pht=0.9))
    first = tracker.consider(1, _scores(0.702, 0.602, 0.802, ph=0.75, pht=0.78))
    pt_focused = tracker.consider(2, _scores(0.90, 0.90, 0.90, ph=0.40, pht=0.40))
    balanced = tracker.consider(3, _scores(0.80, 0.80, 0.80, ph=0.80, pht=0.80))

    assert first.improved is True
    assert pt_focused.improved is False
    assert balanced.improved is True
    assert tracker.best_round == 3
    assert tracker.best_scores is balanced.scores


def test_stage12_checkpoint_requires_full_online_model_and_registered_lineage() -> None:
    checkpoint = {
        "format_version": 1,
        "checkpoint_type": "stage12_online_model",
        "artifact_role": "best",
        "source_stage1": {"path": "/stage1.pt", "sha256": "a" * 64},
        "reference_stage8": {"path": "/stage8.pt", "sha256": "b" * 64},
        "dataset_fingerprints": {"pht": "pht", "validation": "val"},
        "effective_data": {"limit_per_type": None},
        "training_contract": {"condition": "full"},
        "training_state": {"stage12b_round": 3},
        "metrics": {"success": True},
        "model": {
            "peptide_encoder.layer": 1,
            "hla_encoder.layer": 1,
            "tcr_encoder.layer": 1,
            "binding_model.layer": 1,
        },
    }

    validate_stage12_checkpoint(checkpoint, artifact_role="best")
    checkpoint["model"].pop("tcr_encoder.layer")
    with pytest.raises(ValueError, match="tcr_encoder"):
        validate_stage12_checkpoint(checkpoint, artifact_role="best")


def test_stage12_checkpoint_roundtrip_separates_best_and_resumable_state(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _TinyStage12Model()
    pair_batch = {
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "receptor_sequences": ["TCR"] * 4,
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    joint_batch = {
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "hla_sequences": ["HLA"] * 4,
        "tcr_sequences": ["TCR"] * 4,
        "ph_labels": torch.tensor([0, 1, 0, 1]),
        "pt_labels": torch.tensor([0, 1, 1, 0]),
        "pht_labels": torch.tensor([0, 1, 0, 0]),
        "label_types": ["000", "111", "010", "100"],
    }
    trainer = Stage12OnlineTrainer(
        model,
        config,
        {
            "phla": _Loader(pair_batch, 8),
            "ptcr": _Loader(pair_batch, 8, sampler=_RoundSampler()),
            "joint": _Loader(joint_batch, 4, batch_sampler=_EpochBatchSampler()),
        },
        validation_fn=lambda value: _scores(0.7, 0.6, 0.8, ph=0.75, pht=0.78),
    )
    scores = _scores(0.708, 0.604, 0.808, ph=0.75, pht=0.78)
    lineage = {
        "source_stage1": {"path": "/stage1.pt", "sha256": config.paths.stage1_sha256},
        "reference_stage8": {
            "path": "/stage8.pt", "sha256": config.paths.stage8_reference_sha256,
        },
    }

    best_path = save_stage12_checkpoint(
        tmp_path / "stage12_best_20260828_120000_000001.pt",
        artifact_role="best",
        trainer=trainer,
        config=config,
        dataset_fingerprints={"pht": "pht", "validation": "validation"},
        effective_data={"limit_per_type": None, "pair_limit": None, "validation_limit": None},
        reference_scores=scores,
        current_scores=scores,
        decision={"success": True},
        **lineage,
    )
    last_path = save_stage12_checkpoint(
        tmp_path / "stage12_last_20260828_120000_000001.pt",
        artifact_role="last",
        trainer=trainer,
        config=config,
        dataset_fingerprints={"pht": "pht", "validation": "validation"},
        effective_data={"limit_per_type": None, "pair_limit": None, "validation_limit": None},
        reference_scores=scores,
        current_scores=scores,
        decision={"success": True},
        **lineage,
    )
    best = load_stage12_checkpoint(best_path, artifact_role="best")
    last = load_stage12_checkpoint(last_path, artifact_role="last")

    assert "optimizer" not in best
    assert {"optimizer", "scheduler", "scaler", "rng_state"}.issubset(last)
    assert last["training_contract"]["ptcr_replay_ratio"] == 0.25
    assert last["training_contract"]["nominal_task_weights"] == [0.2, 0.133333333333, 0.666666666667]
    assert last["training_contract"]["selection"]["task_weights"] == pytest.approx([1 / 3] * 3)
    assert last["training_contract"]["selection"]["pht_head_weights"] == pytest.approx([1 / 3] * 3)
    assert "pt_sampler" not in last["training_contract"]
    assert last["metrics"]["validation"]["joint_score"] == pytest.approx(scores.joint_score)
    assert last["effective_data"]["limit_per_type"] is None

    restored = Stage12OnlineTrainer(
        _TinyStage12Model(),
        config,
        {
            "phla": _Loader(pair_batch, 8),
            "ptcr": _Loader(pair_batch, 8, sampler=_RoundSampler()),
            "joint": _Loader(joint_batch, 4, batch_sampler=_EpochBatchSampler()),
        },
        validation_fn=lambda value: scores,
    )
    restore_stage12_last_checkpoint(
        last,
        trainer=restored,
        config=config,
        dataset_fingerprints={"pht": "pht", "validation": "validation"},
        effective_data={"limit_per_type": None, "pair_limit": None, "validation_limit": None},
    )
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    assert restored.training_state() == trainer.training_state()
    with pytest.raises(ValueError, match="effective data"):
        restore_stage12_last_checkpoint(
            last,
            trainer=restored,
            config=config,
            dataset_fingerprints={"pht": "pht", "validation": "validation"},
            effective_data={"limit_per_type": 2, "pair_limit": 128, "validation_limit": 128},
        )


def test_stage12_config_exposes_stage8_compatibility_view_without_new_model() -> None:
    config = _config()

    assert config.stage8 is config.stage12


class _PerfectPairModel:
    def set_evaluation(self) -> None:
        return None

    def __call__(self, task, peptide_sequences, receptor_sequences):
        assert task == "ptcr"
        assert len(peptide_sequences) == len(receptor_sequences) == 4
        return SimpleNamespace(
            logits=torch.tensor([[5.0, -5.0], [-5.0, 5.0], [4.0, -4.0], [-4.0, 4.0]])
        )


def test_pair_validation_uses_fixed_half_threshold_and_real_metrics() -> None:
    loader = [{
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "receptor_sequences": ["T1", "T2", "T3", "T4"],
        "labels": torch.tensor([0, 1, 0, 1]),
    }]

    metrics = evaluate_pair_validation_splits(
        _PerfectPairModel(),
        {"val_unseen_pair": loader},
        task="ptcr",
        device=torch.device("cpu"),
        mixed_precision="none",
        threshold=0.5,
    )["val_unseen_pair"]

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["aupr"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_combined_validation_returns_pt_ph_macros_and_original_pht_joint() -> None:
    config = _config()
    model = _TinyStage12Model()
    pair_loader = [{
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "receptor_sequences": ["TCR"] * 4,
        "labels": torch.tensor([0, 1, 0, 1]),
    }]
    joint_loader = [{
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "hla_sequences": ["HLA"] * 4,
        "tcr_sequences": ["TCR"] * 4,
        "ph_labels": torch.tensor([0, 1, 0, 1]),
        "pt_labels": torch.tensor([0, 1, 1, 0]),
        "pht_labels": torch.tensor([0, 1, 0, 1]),
        "label_types": ["000", "111", "010", "100"],
        "source_rows": torch.arange(4),
    }]

    scores = evaluate_stage12_validation(
        model,
        pht_loader=joint_loader,
        ph_loaders={name: pair_loader for name in config.stage12.ph_selection_splits},
        pt_loaders={name: pair_loader for name in config.stage12.pt_selection_splits},
        config=config,
        device=torch.device("cpu"),
    )

    assert set(scores.pt_splits) == set(config.stage12.pt_selection_splits)
    assert set(scores.ph_splits) == set(config.stage12.ph_selection_splits)
    assert 0.0 <= scores.pht_joint <= 1.0


def test_stage12_loss_keeps_warmup_pht_only_then_uses_registered_b_weights() -> None:
    config = _config()
    logits = torch.zeros(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    label_types = ["000", "111", "010", "100"]

    warmup = compute_stage12_joint_loss(
        logits, logits, logits, labels, labels, labels, label_types,
        config=config, phase="stage12a", round_or_epoch=1,
    )
    stage12b = compute_stage12_joint_loss(
        logits, logits, logits, labels, labels, labels, label_types,
        config=config, phase="stage12b", round_or_epoch=1,
    )

    assert warmup.supervised.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert warmup.hierarchy_lambda == 0.0
    assert warmup.pt_implication_lambda == 0.0
    assert stage12b.supervised.item() == pytest.approx(
        1.5 * torch.log(torch.tensor(2.0)).item()
    )
    assert stage12b.hierarchy_lambda == pytest.approx(0.01)
    assert stage12b.pt_implication_lambda == pytest.approx(0.006)


def test_stage12_round_schedule_and_replay_caps_are_task_specific() -> None:
    config = _config()

    assert stage12_pass_order() == ("phla", "ptcr", "joint")
    assert stage12_replay_samples(
        "phla", joint_samples=72_190, task_samples=664_952, config=config
    ) == 18_047
    assert stage12_replay_samples(
        "ptcr", joint_samples=72_190, task_samples=664_952, config=config
    ) == 18_047


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])

    def transformer_blocks(self):
        return self.blocks

    def set_active_training(self, active: bool) -> None:
        self.train(active)


class _TinyBinding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in (
            "peptide_adapter", "hla_adapter", "tcr_adapter",
            "phla_cross_attention", "ptcr_cross_attention",
            "phla_pooling", "ptcr_pooling",
        ):
            setattr(self, name, torch.nn.Linear(1, 1))
        self.phla_classifier = torch.nn.Linear(1, 2)
        self.ptcr_classifier = torch.nn.Linear(1, 2)
        self.triplet_cross_attention = torch.nn.Linear(1, 1)
        self.triplet_pooling = torch.nn.Linear(1, 1)
        self.triplet_classifier = torch.nn.Linear(1, 2)

    def new_triplet_modules(self):
        return {
            "triplet_cross_attention": self.triplet_cross_attention,
            "triplet_pooling": self.triplet_pooling,
            "triplet_classifier": self.triplet_classifier,
        }


class _TinyStage12Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.peptide_encoder = _TinyEncoder()
        self.hla_encoder = _TinyEncoder()
        self.tcr_encoder = _TinyEncoder()
        self.peptide_unfrozen_layers = 2
        self.hla_unfrozen_layers = 2
        self.tcr_unfrozen_layers = 2
        self.binding_model = _TinyBinding()

    def set_trainable_stage8(self, task: str, phase: str) -> None:
        self.requires_grad_(False)
        if phase == "stage8a":
            modules = self.binding_model.new_triplet_modules().values()
        else:
            modules = [self]
        for module in modules:
            module.requires_grad_(True)

    def set_evaluation(self) -> None:
        self.eval()

    @staticmethod
    def _features(count: int) -> torch.Tensor:
        return torch.ones(count, 1)

    def forward(self, task, peptide_sequences, receptor_sequences):
        features = self._features(len(peptide_sequences))
        classifier = (
            self.binding_model.phla_classifier
            if task == "phla" else self.binding_model.ptcr_classifier
        )
        return SimpleNamespace(logits=classifier(features))

    def forward_all(self, peptide_sequences, hla_sequences, tcr_sequences):
        features = self._features(len(peptide_sequences))
        return SimpleNamespace(
            phla_logits=self.binding_model.phla_classifier(features),
            ptcr_logits=self.binding_model.ptcr_classifier(features),
            pht_logits=self.binding_model.triplet_classifier(features),
        )


class _RoundSampler:
    def __init__(self) -> None:
        self.rounds: list[int] = []

    def set_round(self, value: int) -> None:
        self.rounds.append(value)


class _EpochBatchSampler:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, value: int) -> None:
        self.epochs.append(value)


class _Loader:
    def __init__(self, batch: dict, dataset_size: int, *, sampler=None, batch_sampler=None):
        self.batch = batch
        self.dataset = list(range(dataset_size))
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = None

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def test_online_trainer_executes_warmup_then_serial_stage12b_round() -> None:
    config = _config()
    pair_batch = {
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "receptor_sequences": ["TCR"] * 4,
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    joint_batch = {
        "peptide_sequences": ["PEPTIDEA"] * 4,
        "hla_sequences": ["HLA"] * 4,
        "tcr_sequences": ["TCR"] * 4,
        "ph_labels": torch.tensor([0, 1, 0, 1]),
        "pt_labels": torch.tensor([0, 1, 1, 0]),
        "pht_labels": torch.tensor([0, 1, 0, 0]),
        "label_types": ["000", "111", "010", "100"],
    }
    pt_sampler = _RoundSampler()
    joint_sampler = _EpochBatchSampler()
    loaders = {
        "phla": _Loader(pair_batch, 8),
        "ptcr": _Loader(pair_batch, 8, sampler=pt_sampler),
        "joint": _Loader(joint_batch, 4, batch_sampler=joint_sampler),
    }
    reference = _scores(0.7, 0.6, 0.8, ph=0.75, pht=0.78)
    trainer = Stage12OnlineTrainer(
        _TinyStage12Model(), config, loaders,
        validation_fn=lambda model: reference,
    )

    trainer.train_stage12a_epoch(1)
    trainer.train_stage12a_epoch(2)
    result = trainer.train_stage12b_round(1)

    assert tuple(result) == ("phla", "ptcr", "joint")
    assert trainer.pass_history == [
        ("stage12a", 1, "joint"),
        ("stage12a", 2, "joint"),
        ("stage12b", 1, "phla"),
        ("stage12b", 1, "ptcr"),
        ("stage12b", 1, "joint"),
    ]
    assert pt_sampler.rounds == []
    assert joint_sampler.epochs == [1]
    assert result["phla"]["samples"] == 1
    assert result["ptcr"]["samples"] == 1
    assert trainer.global_step == 5


def test_stage12_cli_supports_audit_train_and_resume_without_test_inputs() -> None:
    audit = stage12_cli.parse_args(["--audit-only", "--swanlab-mode", "disabled"])
    train = stage12_cli.parse_args(["--swanlab-mode", "disabled"])
    resume = stage12_cli.parse_args(["--resume", "/tmp/stage12_last.pt"])

    assert audit.audit_only is True
    assert train.audit_only is False and train.resume is None
    assert resume.resume == Path("/tmp/stage12_last.pt")
    assert not any("test" in action.dest for action in stage12_cli.build_parser()._actions)


def test_resume_state_preserves_last_observation_and_terminal_status() -> None:
    config = _config()
    reference = _scores(0.7, 0.6, 0.8, ph=0.75, pht=0.78)
    decision = Stage12SelectionDecision(
        improved=True,
        joint_score_delta=0.001,
        pt_macro_delta=0.001,
        ph_macro_delta=0.0,
        pht_joint_delta=0.0,
    )
    checkpoint = {
        "training_state": {
            "stage12b_round": config.stage12.min_rounds,
            "bad_rounds": config.stage12.early_stopping_patience,
        },
        "metrics": {
            "validation": reference.to_dict(),
            "decision": decision.to_dict(),
        },
    }

    progress = stage12_cli._resume_progress(checkpoint, config)

    assert progress.start_round == config.stage12.min_rounds + 1
    assert progress.exhausted is True
    assert progress.last_scores.to_dict() == reference.to_dict()
    assert progress.last_decision.to_dict() == decision.to_dict()


def test_dataset_fingerprints_cover_replay_and_selection_sources(tmp_path: Path) -> None:
    config = _config()
    config.paths.data_root = tmp_path
    for family in ("PH", "PT"):
        family_dir = tmp_path / family
        family_dir.mkdir()
        (family_dir / "train.csv").write_text("source,train\n", encoding="utf-8")
        benchmark_dir = family_dir / "benchmarks"
        benchmark_dir.mkdir()
        splits = (
            config.stage12.ph_selection_splits
            if family == "PH" else config.stage12.pt_selection_splits
        )
        for split in splits:
            (benchmark_dir / f"{split}.csv").write_text(
                "source,validation\n", encoding="utf-8"
            )

    fingerprints = stage12_cli._dataset_fingerprints(
        config, SimpleNamespace(dataset_fingerprint="pht")
    )

    assert set(fingerprints) == {
        "pht", "validation", "phla_train", "ptcr_train",
    }


def test_stage12_shell_exports_conda_cuda_home_for_transformer_engine(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s' \"${CUDA_HOME:-missing}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = dict(os.environ)
    environment.pop("CUDA_HOME", None)
    environment["PYTHON_BIN"] = str(fake_python)

    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[2] / "run_stage12.sh"), "audit"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )

    assert result.stdout == "/home/mjp/miniconda3/envs/unifyimmun/cuda"


def test_shared_benchmark_evaluator_reports_stage12_selection_metadata() -> None:
    checkpoint = {
        "checkpoint_type": "stage12_online_model",
        "training_state": {"stage12b_round": 7, "global_step": 123, "best_joint_score_seen": 0.88},
        "metrics": {
            "validation": {"joint_score": 0.73, "pt_macro": 0.73, "pht_joint": 0.88},
            "decision": {"success": True},
        },
    }

    summary = benchmark_evaluator._checkpoint_summary(checkpoint)

    assert benchmark_evaluator._stage_label(checkpoint) == "Stage 12"
    assert summary["best_round"] == 7
    assert summary["best_round_score"] == pytest.approx(0.73)
    assert summary["validation_joint_score"] == pytest.approx(0.73)
    assert summary["checkpoint_selected_threshold"] == 0.5


def test_shared_benchmark_evaluator_accepts_stage12_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        benchmark_evaluator,
        "validate_stage12_checkpoint",
        lambda checkpoint, artifact_role: calls.append(artifact_role),
    )

    benchmark_evaluator._validate_checkpoint_for_evaluation({
        "checkpoint_type": "stage12_online_model",
        "artifact_role": "last",
    })

    assert calls == ["last"]
