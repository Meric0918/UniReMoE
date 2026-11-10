"""Stage PHT model, frozen PLMs, continuation, and real CSV/CLI lifecycle."""

import copy
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from source.plm_unified.encoders import ResidueBatch
from source.plm_unified.stage1_encoder_model import SequenceEncoder
from source.plm_unified.stage_adapter_config import StageAdapterRuntimeConfig
from source.plm_unified.stage_adapter_data import collate_stage_adapter
from source.plm_unified.stage8_data import Stage8Record
from source.plm_unified.stage_PHT_config import (
    StagePHTPaths, StagePHTExperimentConfig, StagePHTModelConfig,
    StagePHTTrainingConfig, load_stage_PHT_config,
)
from source.plm_unified.stage_PHT_model import (
    StagePHTBindingModel, StagePHTModel, build_stage_PHT_model,
    initialise_stage_PHT_random, SEQUENCE_ENCODER_NAMES, INTERACTION_MODULE_NAMES,
)
from source.plm_unified.stage_PHT_checkpoint import (
    load_stage_PHT_checkpoint, save_stage_PHT_warmup, save_stage_PHT_best,
    save_stage_PHT_last, restore_stage_PHT_model, resume_stage_PHT_last,
    validate_stage_PHT_checkpoint,
)
from source.plm_unified.stage_PHT_trainer import StagePHTTrainer, build_stage_PHT_optimizer

ROOT = Path(__file__).resolve().parents[2]


class FrozenPLM(nn.Module):
    """Parameterized PLM stand-in that exposes accidental updates or dropout."""

    def __init__(self, hidden_size, max_length=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(32, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.active = False

    def set_active_training(self, active):
        self.active = active
        self.train(active)

    def metadata(self):
        return {"checkpoint_fingerprint": f"pretrained-{self.hidden_size}",
                "encoder_class": "FrozenPLM", "hidden_size": self.hidden_size,
                "max_length": self.max_length}

    def forward(self, sequences):
        lengths = torch.tensor([len(s) for s in sequences])
        ids = torch.zeros(len(sequences), self.max_length, dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, :len(seq)] = torch.tensor([ord(x) % 32 for x in seq])
        mask = torch.arange(self.max_length)[None, :] < lengths[:, None]
        return ResidueBatch(self.dropout(self.embedding(ids)), mask, lengths)


@pytest.fixture
def config(tmp_path):
    value = StagePHTExperimentConfig(
        paths=StagePHTPaths(tmp_path, tmp_path / "data", tmp_path / "models",
                            tmp_path / "logs", tmp_path / "esmc", tmp_path / "tcr"),
        model=StagePHTModelConfig(
            peptide_input_dim=6, hla_input_dim=6, tcr_input_dim=5,
            model_dim=8, num_attention_heads=2, feedforward_dim=16,
            encoder_dropout=0.1, attention_dropout=0.1, classifier_dropout=0.1,
            peptide_min_length=1, peptide_max_length=4, hla_max_length=4, tcr_max_length=4,
        ),
        runtime=StageAdapterRuntimeConfig(device="cpu", num_workers=0),
        stage_PHT=StagePHTTrainingConfig(
            micro_batch_size=4, eval_batch_size=4, gradient_accumulation_steps=2,
            min_rounds=1, max_rounds=2, mixed_precision="none",
        ),
    )
    value.swanlab.mode = "disabled"
    value.validate()
    return value


def make_model(config):
    return StagePHTModel(
        StagePHTBindingModel(config.model), FrozenPLM(config.model.peptide_input_dim),
        FrozenPLM(config.model.hla_input_dim), FrozenPLM(config.model.tcr_input_dim),
    )


def batch():
    return collate_stage_adapter([
        Stage8Record("AA", "CC", "DD", 1, 1, 1, 1),
        Stage8Record("AC", "CD", "DE", 1, 0, 0, 2),
        Stage8Record("AD", "CE", "DF", 0, 1, 0, 3),
        Stage8Record("AE", "CF", "DG", 0, 0, 0, 4),
    ])


def test_public_config_matches_adapter_training_and_encoder_structure():
    config = load_stage_PHT_config(ROOT / "configs/plm_stage_PHT.yaml")
    from source.plm_unified.stage_adapter_config import load_stage_adapter_config
    adapter = load_stage_adapter_config(ROOT / "configs/plm_stage_adapter_random.yaml")
    for key, value in vars(config.stage_PHT).items():
        if key not in {"encoder_lr", "warmup_epochs"}:
            assert value == getattr(adapter.stage_adapter, key)
    assert config.stage_PHT.encoder_lr == adapter.stage_adapter.adapter_lr
    assert config.stage_PHT.warmup_epochs == adapter.stage_adapter.stage_adapter_a_epochs
    assert config.model.encoder_num_layers == 3
    assert config.model.encoder_dropout == 0.15
    assert "stage_adapter" not in config.to_dict()
    assert "adapter_hidden_dim" not in config.to_dict()["model"]
    assert not hasattr(config.paths, "stage1_checkpoint")


def test_model_has_three_encoders_and_masks_padding(config):
    model = StagePHTBindingModel(config.model).eval()
    encoders = [module for module in model.modules() if isinstance(module, SequenceEncoder)]
    assert len(encoders) == 3
    assert all(len(encoder.layers) == 3 for encoder in encoders)
    assert not any("adapter" in name for name, _ in model.named_modules())
    masks = [torch.tensor([[True, True, False, False]] * 4)] * 3
    hidden = [torch.randn(4, 4, dim) for dim in (6, 6, 5)]
    args = [item for pair in zip(hidden, masks) for item in pair]
    with torch.no_grad():
        output = model(*args)
        changed = [h.masked_fill(~mask[..., None], float("nan")) for h, mask in zip(hidden, masks)]
        padded_output = model(*[item for pair in zip(changed, masks) for item in pair])
    assert output.logits.shape == (4, 2)
    assert output.pooled_embedding.shape == (4, 8)
    torch.testing.assert_close(output.logits, padded_output.logits, rtol=0, atol=0)
    masks[0] = torch.zeros_like(masks[0])
    with pytest.raises(ValueError, match="at least one residue"):
        model(*[item for pair in zip(hidden, masks) for item in pair])


def test_factory_loads_pretrained_plms_and_checks_dimensions(config):
    module = sys.modules[build_stage_PHT_model.__module__]
    def esmc(path, max_length, **kwargs):
        assert kwargs["random_initialization"] is False
        return FrozenPLM(6, max_length)
    def tcr(path, max_length, **kwargs):
        assert kwargs["random_initialization"] is False
        return FrozenPLM(5, max_length)
    with patch.object(module, "ESMCFineTuningEncoder", side_effect=esmc), \
         patch.object(module, "TCRBertFineTuningEncoder", side_effect=tcr):
        model = build_stage_PHT_model(config, torch.device("cpu"))
        assert all(not p.requires_grad for p in model.peptide_encoder.parameters())
        config.model.tcr_input_dim = 7
        with pytest.raises(ValueError, match="hidden size"):
            build_stage_PHT_model(config, torch.device("cpu"))


def test_training_updates_entire_pht_path_and_keeps_plms_frozen(config):
    model = make_model(config)
    optimizer = build_stage_PHT_optimizer(model, config)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.binding_model.parameters()}
    assert [g["lr"] for g in optimizer.param_groups] == [2.5e-5, 1e-4]
    trainer = StagePHTTrainer(model, config, [batch()] * 3, [batch()])
    with pytest.raises(RuntimeError, match="complete warmup"):
        trainer.begin_training()
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if not name.startswith("binding_model.")}
    for epoch in (1, 2):
        trainer.evaluate()
        before = {name: value.clone() for name, value in model.binding_model.state_dict().items()}
        result = trainer.train_warmup_epoch(epoch)
        assert result["optimizer_steps"] == 2  # Includes the final partial accumulation window.
        for name in SEQUENCE_ENCODER_NAMES + INTERACTION_MODULE_NAMES:
            module = getattr(model.binding_model, name)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            assert any(not torch.equal(value, before[f"{name}.{key}"])
                       for key, value in module.state_dict().items())
        for name in SEQUENCE_ENCODER_NAMES:
            encoder = getattr(model, name)
            assert not encoder.training and not encoder.active
            assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    first_optimizer = trainer.optimizer
    trainer.begin_training()
    assert trainer.optimizer is not first_optimizer
    assert not trainer.optimizer.state
    trainer.train_round(1)
    model.train()
    for name, value in frozen.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_resume_matches_continuous_training_with_dropout(config, tmp_path):
    model = make_model(config)
    trainer = StagePHTTrainer(model, config, [batch()], [batch()])
    for epoch in (1, 2):
        trainer.train_warmup_epoch(epoch)
    common = dict(source_initialization=initialise_stage_PHT_random(model, config=config),
                  dataset_fingerprint="data", effective_data={"train_samples": 4},
                  run_timestamp="test", evaluation=trainer.evaluate())
    warmup = save_stage_PHT_warmup(tmp_path, model=model, config=config,
                                   training_state=trainer.training_state(), **common)
    trainer.train_round(1)
    common["evaluation"] = trainer.evaluate()
    trainer.best_score = common["evaluation"].selection_score
    best = save_stage_PHT_best(tmp_path, trainer=trainer, warmup_checkpoint=warmup, **common)
    last = save_stage_PHT_last(tmp_path, trainer=trainer, warmup_checkpoint=warmup,
                               best_checkpoint=best, **common)
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    trainer.train_round(2)
    resumed = StagePHTTrainer(make_model(config), config, [batch()], [batch()])
    checkpoint = load_stage_PHT_checkpoint(last, role="last")
    resume_stage_PHT_last(checkpoint, trainer=resumed, config=config,
                          dataset_fingerprint="data", effective_data={"train_samples": 4})
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
    resumed.train_round(2)
    assert resumed.global_step == trainer.global_step
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    for key, state in trainer.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        validate_stage_PHT_checkpoint(checkpoint, config=config,
                                      dataset_fingerprint="other", effective_data={"train_samples": 4})
    wrong = copy.deepcopy(checkpoint)
    wrong["architecture"] = "stage1_encoder"
    torch.save(wrong, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="stage_PHT"):
        load_stage_PHT_checkpoint(tmp_path / "old.pt")
    wrong = copy.deepcopy(checkpoint)
    wrong["config"]["model"]["encoder_dropout"] = 0.2
    with pytest.raises(ValueError, match="model configuration"):
        restore_stage_PHT_model(wrong, make_model(config))
    wrong = copy.deepcopy(checkpoint)
    wrong["encoder_metadata"]["tcr"]["checkpoint_fingerprint"] = "different"
    with pytest.raises(ValueError, match="PLM assets"):
        restore_stage_PHT_model(wrong, make_model(config))


def write_data(config):
    columns = ["peptide", "HLA", "tcr", "PH", "PT", "PHT", "class_code"]
    for index, directory in enumerate(config.stage_PHT.class_dirs):
        code = directory[-3:]
        root = config.data_dir() / directory
        root.mkdir(parents=True)
        for split, prefix in (("train", "A"), ("val", "F")):
            with (root / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([prefix + "ACDE"[index], "CC", "DD", *map(int, code), code])


def test_real_csv_cli_warmup_training_evaluation_benchmark_and_resume(config, tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "source"))
    write_data(config)
    config_path = tmp_path / "stage_PHT.yaml"
    raw = config.to_dict()
    raw.pop("source_path")
    config_path.write_text(yaml.safe_dump(raw))
    entry = importlib.import_module("train_plm_stage_PHT")
    monkeypatch.setattr(entry, "build_stage_PHT_model", lambda c, d: make_model(c))
    common = ["--config", str(config_path), "--limit-per-type", "1"]
    def run(module, args):
        monkeypatch.setattr(sys, "argv", [module.__name__] + args)
        module.main()
    run(entry, common + ["--warmup-only", "--run-timestamp", "20261018_000000_000000"])
    warmup = next(config.warmup_dir().glob("stage_PHT_warmup_*.pt"))
    best_warmup_state = load_stage_PHT_checkpoint(warmup)["model"]
    original_restore = entry.restore_stage_PHT_model
    restored = []
    def capture_restore(checkpoint, model):
        original_restore(checkpoint, model)
        restored.append(True)
        for key, value in best_warmup_state.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    monkeypatch.setattr(entry, "restore_stage_PHT_model", capture_restore)
    run(entry, common + ["--warmup-checkpoint", str(warmup),
                         "--run-timestamp", "20261018_000001_000000"])
    assert restored == [True]
    last = next(config.run_dir().glob("stage_PHT_last_*.pt"))
    state = load_stage_PHT_checkpoint(last)
    assert state["training_state"]["training_round"] == 2
    assert state["effective_data"]["train_samples"] == 4
    best = Path(state["best_checkpoint"]["path"])
    evaluator = importlib.import_module("evaluate_plm_stage_PHT")
    monkeypatch.setattr(evaluator, "build_stage_PHT_model", lambda c, d: make_model(c))
    report = tmp_path / "eval.json"
    run(evaluator, ["--config", str(config_path), "--checkpoint", str(best), "--output", str(report)])
    assert json.loads(report.read_text())["pht"]["samples"] == 4
    # Make the checkpoint threshold distinctive so benchmark threshold reuse is observable.
    saved_best = load_stage_PHT_checkpoint(best)
    saved_best["metrics"]["selected_pht_threshold"] = 0.37
    benchmark_checkpoint = tmp_path / "benchmark_model.pt"
    torch.save(saved_best, benchmark_checkpoint)
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    with (benchmarks / "test_unseen_triple.csv").open("w") as target:
        for index, directory in enumerate(config.stage_PHT.class_dirs):
            lines = (config.data_dir() / directory / "val.csv").read_text().splitlines(True)
            target.writelines(lines if index == 0 else lines[1:])
    benchmark = importlib.import_module("evaluate_plm_stage_PHT_benchmarks")
    monkeypatch.setattr(benchmark, "build_stage_PHT_model", lambda c, d: make_model(c))
    thresholds = []
    compute = benchmark.compute_binary_metrics
    def capture_threshold(*args, **kwargs):
        thresholds.append(kwargs["threshold"])
        return compute(*args, **kwargs)
    monkeypatch.setattr(benchmark, "compute_binary_metrics", capture_threshold)
    report = tmp_path / "benchmark.json"
    run(benchmark, ["--config", str(config_path), "--checkpoint", str(benchmark_checkpoint),
                    "--benchmark-dir", str(benchmarks), "--output", str(report)])
    assert thresholds == [0.37]
    assert json.loads(report.read_text())["benchmarks"]["test_unseen_triple"]["validation_threshold"] == 0.37
    # Resuming a finished run should preserve the existing last artifact in the summary.
    run(entry, ["--config", str(config_path), "--resume", str(last)])
    assert str(last) in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [("encoder_num_layers", 0), ("num_attention_heads", 3),
                                          ("encoder_dropout", 1.0)])
def test_invalid_encoder_config_is_rejected(config, field, value):
    setattr(config.model, field, value)
    with pytest.raises(ValueError):
        config.validate()
