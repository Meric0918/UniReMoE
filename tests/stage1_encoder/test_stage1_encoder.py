"""Contract and lifecycle tests using synthetic residue features only."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from source.plm_unified.cache import EmbeddingCacheWriter
from source.plm_unified.config import PathsConfig, load_config
from source.plm_unified.evaluation import evaluate_model
from source.plm_unified.fgm import ModuleFGM
from source.plm_unified.stage1_encoder_config import (
    Stage1EncoderExperimentConfig,
    Stage1EncoderModelConfig,
    Stage1EncoderTrainingConfig,
    load_stage1_encoder_config,
)
from source.plm_unified.stage1_encoder_model import (
    ARCHITECTURE,
    ENCODER_UMAP_STAGES,
    SequenceEncoder,
    Stage1EncoderBindingModel,
    TaskSpecificStage1EncoderBindingModel,
    validate_encoder_checkpoint,
)
from source.plm_unified.stage1_encoder_trainer import (
    Stage1EncoderTrainer,
    build_encoder_optimizer,
)
from source.plm_unified.trainer import (
    TaskAwareCosineScheduler,
    build_task_checkpoint_state,
    export_task_checkpoints,
    load_training_checkpoint,
    set_global_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config(tmp_path):
    config = Stage1EncoderExperimentConfig(
        paths=PathsConfig(
            project_root=tmp_path, data_root=tmp_path / "data",
            cache_root=tmp_path / "cache", output_root=tmp_path / "models",
            swanlog_root=tmp_path / "swanlog",
            esmc_model=tmp_path / "no-esmc", tcr_bert_model=tmp_path / "no-tcr",
        ),
        model=Stage1EncoderModelConfig(
            peptide_input_dim=12, hla_input_dim=12, tcr_input_dim=8,
            model_dim=8, num_attention_heads=2, feedforward_dim=16,
            encoder_dropout=0.0, attention_dropout=0.0, classifier_dropout=0.0,
        ),
        training=Stage1EncoderTrainingConfig(
            warmup_epochs_per_task=1, max_rounds=1, batch_size=2,
            mixed_precision="none", device="cpu", num_workers=0,
            log_every_steps=100, retention_eval_enabled=True,
        ),
    )
    config.swanlab.mode = "disabled"
    config.validate()
    return config


def batch(task):
    generator = torch.Generator().manual_seed(7)
    receptor_dim = 12 if task == "phla" else 8
    return {
        "peptide_hidden": torch.randn(4, 5, 12, generator=generator),
        "peptide_mask": torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool),
        "receptor_hidden": torch.randn(4, 4, receptor_dim, generator=generator),
        "receptor_mask": torch.tensor([[1, 1, 1, 0]] * 4, dtype=torch.bool),
        "labels": torch.tensor([0, 1, 0, 1]),
        "group_ids": torch.tensor([10, 10, 20, 20]),
    }


def forward(model, task, values, **kwargs):
    return model(task, *[values[key] for key in (
        "peptide_hidden", "peptide_mask", "receptor_hidden", "receptor_mask",
    )], **kwargs)


class Loader:
    def __init__(self, task):
        self.values = batch(task)
        self.dataset = type("Dataset", (), {
            "rows": [("ACDEFGHI", "ACDE", label) for label in [0, 1, 0, 1]],
        })()
        self.generator = torch.Generator().manual_seed(12)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.values


class Tracker:
    run_id = None

    def log(self, values, *, step=None):
        pass


def trainer(config):
    loaders = {task: Loader(task) for task in ("phla", "ptcr")}
    return Stage1EncoderTrainer(
        Stage1EncoderBindingModel(config.model), config,
        loaders, {task: {"val": loader} for task, loader in loaders.items()},
        Tracker(), run_timestamp="20261013_010203_000001",
    )


def test_default_config_matches_plm_and_stage1_heads():
    encoder = load_stage1_encoder_config(PROJECT_ROOT / "configs/plm_stage1_encoder.yaml")
    original = load_config(PROJECT_ROOT / "configs/plm_stage1.yaml")
    assert encoder.cache_dir() == original.cache_dir()
    assert encoder.paths.data_root == original.paths.data_root
    assert encoder.run_dir() != original.run_dir()
    assert encoder.model.model_dim == 256
    assert encoder.model.num_attention_heads == 8
    assert encoder.model.encoder_num_layers == 3
    assert encoder.model.feedforward_dim == 1024
    assert encoder.training.shared_peptide_encoder_lr == 5e-5
    assert encoder.training.hla_encoder_lr == encoder.training.tcr_encoder_lr == 1e-4
    assert not any("adapter" in key for section in ("model", "training")
                   for key in encoder.to_dict()[section])
    assert encoder.swanlab.group != original.swanlab.group


@pytest.mark.parametrize("field,value", [
    ("encoder_num_layers", 0), ("encoder_num_layers", 1.5),
    ("num_attention_heads", 0), ("model_dim", 9),
    ("encoder_dropout", 1.0), ("tcr_input_dim", 0),
])
def test_invalid_model_config_is_rejected(config, field, value):
    setattr(config.model, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_padding_is_zero_and_does_not_influence_real_residues():
    encoder = SequenceEncoder(12, 8, 2, 16, dropout=0.0).eval()
    values = batch("phla")
    hidden, mask = values["peptide_hidden"], values["peptide_mask"]
    with torch.no_grad():
        expected = encoder(hidden, mask)
        changed = hidden.clone()
        changed[~mask] = float("nan")
        actual = encoder(changed, mask)
        unpadded = encoder(hidden[:, :3], mask[:, :3])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(expected[:, :3], unpadded, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(expected[~mask]) == 0
    assert torch.isfinite(expected).all()


def test_self_attention_propagates_information_between_residues():
    torch.manual_seed(9)
    encoder = SequenceEncoder(12, 8, 2, 16, dropout=0.0)
    hidden = torch.randn(1, 3, 12, requires_grad=True)
    output = encoder(hidden, torch.ones(1, 3, dtype=torch.bool))
    output[0, 0, 0].backward()
    assert hidden.grad[0, 1:].abs().sum() > 0
    with torch.no_grad():
        changed = hidden.clone()
        changed[0, 1] = torch.randn(12)
        assert not torch.allclose(output[0, 0], encoder(
            changed, torch.ones(1, 3, dtype=torch.bool),
        )[0, 0])


def test_empty_sequence_and_bad_shapes_are_rejected():
    encoder = SequenceEncoder(12, 8, 2, 16)
    with pytest.raises(ValueError, match="at least one residue"):
        encoder(torch.zeros(2, 3, 12), torch.tensor([[1, 1, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match="mask"):
        encoder(torch.zeros(2, 3, 12), torch.ones(2, 2))
    with pytest.raises(ValueError, match="hidden"):
        encoder(torch.zeros(2, 3, 8), torch.ones(2, 3))


@pytest.mark.parametrize("task", ["phla", "ptcr"])
def test_task_outputs_attention_gradients_and_freezing(config, task):
    model = Stage1EncoderBindingModel(config.model)
    model.set_trainable_task(task)
    values = batch(task)
    output = forward(model, task, values, return_attention=True, return_features=True)
    assert output.logits.shape == (4, 2)
    assert output.pooled_embedding.shape == (4, 8)
    assert output.plm_embedding.shape == (4, 24 if task == "phla" else 20)
    assert output.sequence_encoder_embedding.shape == (4, 16)
    assert output.cross_attention.shape == (4, 2, 5, 4)
    assert torch.count_nonzero(output.cross_attention[..., -1]) == 0
    assert torch.count_nonzero(output.pooling_weights[:, 3:]) == 0
    torch.nn.functional.cross_entropy(output.logits, values["labels"]).backward()
    active = model.task_modules(task)
    for module in active.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    inactive = "tcr_encoder" if task == "phla" else "hla_encoder"
    assert all(p.grad is None and not p.requires_grad for p in getattr(model, inactive).parameters())
    model.set_trainable_task(task, freeze_shared=True)
    assert not any(p.requires_grad for p in model.peptide_encoder.parameters())
    assert any(p.requires_grad for p in getattr(model, "hla_encoder" if task == "phla" else "tcr_encoder").parameters())
    default = forward(model, task, values)
    assert default.plm_embedding is default.sequence_encoder_embedding is None
    assert default.cross_attention is default.pooling_weights is None
    assert not any("adapter" in name for name, _ in model.named_modules())


def test_fgm_targets_active_encoders_and_restores_parameters(config):
    model = Stage1EncoderBindingModel(config.model)
    model.set_trainable_task("phla")
    values = batch("phla")
    torch.nn.functional.cross_entropy(forward(model, "phla", values).logits, values["labels"]).backward()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    fgm = ModuleFGM(model)
    stats = fgm.attack(model.fgm_modules("phla"), epsilon=0.5)
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert stats.attacked_parameter_count > 0
    assert any(name.startswith("peptide_encoder.layers.") for name in changed)
    assert any(name.startswith("hla_encoder.layers.") for name in changed)
    assert all(name.startswith(("peptide_encoder.", "hla_encoder.")) for name in changed)
    fgm.restore()
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())


def test_optimizer_covers_parameters_once_and_schedules_active_groups(config):
    config.training.shared_peptide_encoder_lr = 5e-5
    config.training.hla_encoder_lr = config.training.tcr_encoder_lr = 1e-4
    model = Stage1EncoderBindingModel(config.model)
    optimizer = build_encoder_optimizer(model, config)
    identities = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(p) for p in model.parameters()}
    for group in optimizer.param_groups:
        if group["name"].endswith("/no_decay"):
            assert group["weight_decay"] == 0
            assert all(p.ndim <= 1 for p in group["params"])
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["shared_peptide_encoder/decay"] == 5e-5
    assert rates["hla_encoder/decay"] == 1e-4
    scheduler = TaskAwareCosineScheduler(
        optimizer, total_steps=10, warmup_ratio=0, warmup_steps=None,
        min_lr_ratio=0, task_total_steps={"phla": 5, "ptcr": 5},
        task_warmup_steps={"phla": None, "ptcr": None},
    )
    scheduler.step("phla")
    steps = dict(zip([g["name"] for g in optimizer.param_groups], scheduler.group_steps))
    assert steps["shared_peptide_encoder/decay"] == 1
    assert steps["hla_encoder/decay"] == 1
    assert steps["tcr_encoder/decay"] == 0


@pytest.mark.parametrize("task", ["phla", "ptcr"])
def test_exported_task_model_matches_joint_predictions(config, task, tmp_path):
    model = Stage1EncoderBindingModel(config.model).eval()
    state = {"architecture": ARCHITECTURE, "model": model.state_dict(), "config": config.to_dict()}
    checkpoint = build_task_checkpoint_state(state, task, source_checkpoint=tmp_path / "joint.pt")
    assert checkpoint["architecture"] == ARCHITECTURE
    validate_encoder_checkpoint(checkpoint, config)
    exported = TaskSpecificStage1EncoderBindingModel(config.model, task).eval()
    exported.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        torch.testing.assert_close(forward(model, task, batch(task)).logits,
                                   forward(exported, task, batch(task)).logits)
    other = "ptcr" if task == "phla" else "phla"
    assert not any(name.startswith(other) for name in checkpoint["model"])
    assert not hasattr(exported, "tcr_encoder" if task == "phla" else "hla_encoder")
    with pytest.raises(ValueError, match="checkpoint is for task"):
        forward(exported, other, batch(other))


def test_checkpoint_validation_rejects_other_architectures_and_settings(config):
    with pytest.raises(ValueError, match="architecture"):
        validate_encoder_checkpoint({"config": config.to_dict()}, config)
    state = {"architecture": ARCHITECTURE, "config": config.to_dict()}
    state["config"]["model"]["encoder_num_layers"] = 1
    with pytest.raises(ValueError, match="configuration"):
        validate_encoder_checkpoint(state, config)
    with pytest.raises(ValueError, match="Unknown task"):
        forward(Stage1EncoderBindingModel(config.model), "unknown", batch("phla"))


@pytest.mark.parametrize("nested", [False, True])
def test_seed_summary_accepts_legacy_and_benchmark_metrics(tmp_path, monkeypatch, nested):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "source"))
    from run_plm_stage1_seeds import aggregate_checkpoints

    first = {"auroc": .8, "aupr": .6, "accuracy": .5, "mcc": .2, "f1": .4}
    second = {"auroc": .4, "aupr": .2, "accuracy": .5, "mcc": .0, "f1": .2}
    metrics = {"val_unseen_pair": first, "val_unseen_peptide": second} if nested else first
    paths = {}
    for task in ("phla", "ptcr"):
        paths[task] = tmp_path / f"{task}.pt"
        state = {"config": {"training": {"seed": 42}}, "validation_metrics": metrics}
        if task == "phla":
            state["best_score"] = .9
        torch.save(state, paths[task])
    paths["joint"] = tmp_path / "joint.pt"
    torch.save({"best_round": 1, "round_score": .8}, paths["joint"])
    summary = aggregate_checkpoints([paths])
    assert summary["runs"][0]["best_scores"]["phla"] == .9
    assert summary["aggregate"]["phla"]["auroc"]["mean"] == pytest.approx(.6 if nested else .8)
    assert summary["runs"][0]["best_scores"]["ptcr"] == pytest.approx(.38 if nested else .7)


def test_resume_restores_optimizer_scheduler_rng_and_shared_freeze(config, tmp_path):
    set_global_seed(13)
    first = trainer(config)
    first.train_epoch("phla", first.train_loaders["phla"], phase_name="stage1b", use_fgm=True)
    first.freeze_shared_sequence = True
    first.stage = "stage1b"
    path = tmp_path / "resume.pt"
    first.save_checkpoint(path)
    saved = load_training_checkpoint(path, map_location="cpu")
    assert saved["freeze_shared_encoder"] is True
    assert "freeze_shared_adapter" not in saved
    expected_random = torch.rand(3)
    expected = first.train_epoch("ptcr", first.train_loaders["ptcr"], phase_name="stage1b", use_fgm=True)
    second = trainer(config)
    second.load_checkpoint(path)
    torch.testing.assert_close(torch.rand(3), expected_random)
    actual = second.train_epoch("ptcr", second.train_loaders["ptcr"], phase_name="stage1b", use_fgm=True)
    assert second.freeze_shared_sequence
    assert second.global_step == first.global_step
    assert actual.clean_loss == pytest.approx(expected.clean_loss)
    assert second.scheduler.state_dict() == first.scheduler.state_dict()
    for name, parameter in first.model.named_parameters():
        torch.testing.assert_close(parameter, dict(second.model.named_parameters())[name])
    for name, values in first.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, second.optimizer.state_dict()["state"][name][key])
    saved["architecture"] = "stage1"
    torch.save(saved, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="architecture"):
        second.load_checkpoint(tmp_path / "wrong.pt")


def test_full_synthetic_lifecycle_saves_all_model_types_and_evaluates(config):
    run = trainer(config)
    summary = run.fit()
    assert summary["stage"] == "complete"
    assert run.warmup_completed == {"phla": 1, "ptcr": 1}
    joint = load_training_checkpoint(run.best_round_checkpoint_path, map_location="cpu")
    assert joint["architecture"] == ARCHITECTURE
    assert joint["checkpoint_type"] == "stage1_encoder_joint_model"
    assert run.validation_error_csv.output_dir.parts[-2] == "stage1_encoder"
    outputs = export_task_checkpoints(run.best_round_checkpoint_path, output_dir=run.run_dir / "exports")
    for task in ("phla", "ptcr"):
        assert run.task_checkpoint_paths[task].exists()
        assert run.calibration_checkpoint_paths[task].exists()
        assert run.robust_checkpoint_paths[task].exists()
        state = load_training_checkpoint(outputs[task], map_location="cpu")
        model = TaskSpecificStage1EncoderBindingModel(config.model, task)
        model.load_state_dict(state["model"])
        result = evaluate_model(model, Loader(task), task, config, strict_threshold=True)
        assert np.isfinite(result.loss)


def test_encoder_umap_uses_distinct_feature_stages(config, tmp_path):
    pytest.importorskip("umap")
    from source.plm_unified.stage1_umap import Stage1BestUMAPVisualizer

    loader = Loader("phla")
    visualizer = Stage1BestUMAPVisualizer(
        tmp_path / "umap", {"phla": {"val": loader}},
        train_loaders={"phla": loader}, feature_stages=ENCODER_UMAP_STAGES,
        n_neighbors=2, max_samples_per_split=4,
    )
    visualizer.start_round(1)
    run = trainer(config)
    features = run._feature_embeddings(forward(run.model, "phla", batch("phla"), return_features=True))
    for split in ("train", "val"):
        visualizer.begin_split("phla", split)
        visualizer.observe("phla", split, features, loader.values["labels"], torch.tensor([.1, .9, .3, .7]))
        visualizer.finish_split("phla", split)
    visualizer._fit_or_transform = lambda task, stage, values, **kw: (values[:, :2], "test", 2)
    def fake_plot(path, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")

    visualizer._plot_train_val = fake_plot
    artifact = visualizer.render_task("phla", round_id=1, score=.8)
    assert set(artifact.figures) == set(ENCODER_UMAP_STAGES)
    assert set(json.loads(artifact.summary.read_text())["stages"]) == set(ENCODER_UMAP_STAGES)


def test_training_and_evaluation_entrypoints_with_real_synthetic_cache(config, tmp_path):
    """Exercise the CLI, CSV readers, HDF5 cache, trainer and model loading together."""
    sequences = {
        "peptide": ["ACDEFGHI", "ACDEFGHIK"],
        "hla": ["ACDEF", "GHIKL"], "tcr": ["LMNPQ", "RSTVW"],
    }
    rng = np.random.default_rng(5)
    for entity, seqs in sequences.items():
        width, dim = (9, 12) if entity == "peptide" else (5, 8 if entity == "tcr" else 12)
        mask = np.zeros((2, width), dtype=np.uint8)
        for index, sequence in enumerate(seqs):
            mask[index, :len(sequence)] = 1
        with EmbeddingCacheWriter(
            config.cache_dir(), entity, hidden_dim=dim, max_length=width,
            model_metadata={"checkpoint_fingerprint": "synthetic"}, shard_size=2,
        ) as writer:
            writer.add_batch(seqs, rng.normal(size=(2, width, dim)), mask,
                             np.asarray([len(seq) for seq in seqs]))
    for task, entity in (("PH", "hla"), ("PT", "tcr")):
        root = config.paths.data_root / task
        (root / "benchmarks").mkdir(parents=True)
        for path in (root / "train.csv", root / "benchmarks/val_unseen_pair.csv"):
            with path.open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(["peptide", "HLA" if task == "PH" else "tcr", "label"])
                for peptide_index, receptor_index, label in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
                    writer.writerow([sequences["peptide"][peptide_index], sequences[entity][receptor_index], label])
    config.cache.hla_splits = config.cache.tcr_splits = ["train", "val_unseen_pair"]
    path = tmp_path / "encoder.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def cli(script, *args):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "source" / script), "--config", str(path), *args],
            cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    cli("train_plm_stage1_encoder.py", "--batch-size", "2", "--no-best-umap")
    last = next(config.run_dir().glob("last_*.pt"))
    cli("train_plm_stage1_encoder.py", "--resume", str(last), "--no-best-umap")
    cli("evaluate_plm_stage1_encoder.py")
    task_path = next(config.run_dir().glob("best_ptcr_[0-9]*.pt"))
    cli("evaluate_plm_stage1_encoder.py", "--checkpoint", str(task_path), "--tasks", "ptcr")
    reports = [json.loads(path.read_text()) for path in config.run_dir().glob("evaluation-*.json")]
    assert len(reports) == 2
    assert all(report["architecture"] == ARCHITECTURE for report in reports)
    assert {tuple(sorted(report["metrics"])) for report in reports} == {("phla", "ptcr"), ("ptcr",)}
    cli("run_plm_stage1_encoder_seeds.py", "--seeds", "42", "2026", "--batch-size", "2")
    summaries = list((config.paths.output_root / "fold_1").glob("seed_summary_*.json"))
    assert len(summaries) == 1
    assert (config.paths.output_root / "fold_1/seed_2026").is_dir()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("task", ["phla", "ptcr"])
def test_default_plm_dimensions_support_cuda_mixed_precision(task, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    model = Stage1EncoderBindingModel(Stage1EncoderModelConfig()).cuda()
    model.set_trainable_task(task)
    values = {
        "peptide_hidden": torch.randn(2, 15, 960, device="cuda", dtype=dtype),
        "peptide_mask": torch.tensor([[1] * 9 + [0] * 6] * 2, device="cuda", dtype=torch.bool),
        "receptor_hidden": torch.randn(2, 34, 960 if task == "phla" else 768, device="cuda", dtype=dtype),
        "receptor_mask": torch.tensor([[1] * 13 + [0] * 21] * 2, device="cuda", dtype=torch.bool),
    }
    with torch.autocast("cuda", dtype=dtype):
        output = forward(model, task, values)
        loss = torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert torch.isfinite(output.logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
