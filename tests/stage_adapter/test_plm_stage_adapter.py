"""Contract tests for the PHT-only Stage Adapter implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.plm_unified.config import ModelConfig, SwanLabConfig
from source.plm_unified.encoders import ResidueBatch
from source.plm_unified.metrics import compute_binary_metrics
from source.plm_unified.stage_adapter_checkpoint import (
    CHECKPOINT_TYPE,
    FORMAT_VERSION,
    load_stage_adapter_checkpoint,
    save_stage_adapter_warmup,
)
from source.plm_unified.stage_adapter_config import (
    StageAdapterExperimentConfig,
    StageAdapterPaths,
    StageAdapterRuntimeConfig,
    StageAdapterTrainingConfig,
    load_stage_adapter_config,
)
from source.plm_unified.stage_adapter_data import collate_stage_adapter
from source.plm_unified.stage_adapter_loss import balanced_pht_cross_entropy
from source.plm_unified.stage_adapter_model import (
    ProgressiveStageAdapterModel,
    StageAdapterBindingModel,
)
from source.plm_unified.stage_adapter_trainer import (
    StageAdapterTrainer,
    build_stage_adapter_optimizer,
)
from source.plm_unified.stage8_data import Stage8Record


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int, max_length: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.active = False

    def set_active_training(self, active: bool) -> None:
        self.active = active
        self.train(active)

    def metadata(self) -> dict:
        return {
            "encoder_class": type(self).__name__,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
        }

    def forward(self, sequences) -> ResidueBatch:
        batch_size = len(sequences)
        lengths = torch.tensor(
            [min(len(value), self.max_length) for value in sequences],
            dtype=torch.long,
        )
        mask = torch.arange(self.max_length).unsqueeze(0) < lengths.unsqueeze(1)
        hidden = torch.zeros(batch_size, self.max_length, self.hidden_size)
        for index, sequence in enumerate(sequences):
            for position, residue in enumerate(sequence[: self.max_length]):
                hidden[index, position] = (ord(residue) % 23 + 1) / 23.0
        return ResidueBatch(hidden, mask, lengths)


def _model_config() -> ModelConfig:
    return ModelConfig(
        peptide_input_dim=6,
        hla_input_dim=6,
        tcr_input_dim=5,
        adapter_hidden_dim=8,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
        adapter_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        peptide_min_length=1,
        peptide_max_length=4,
        hla_max_length=4,
        tcr_max_length=4,
    )


def _config(root: Path) -> StageAdapterExperimentConfig:
    return StageAdapterExperimentConfig(
        paths=StageAdapterPaths(
            project_root=root,
            data_root=root / "data",
            output_root=root / "models",
            swanlog_root=root / "logs",
            esmc_model=root / "esmc",
            tcr_bert_model=root / "tcrbert",
            stage1_checkpoint=root / "stage1.pt",
            stage1_sha256="0" * 64,
        ),
        model=_model_config(),
        runtime=StageAdapterRuntimeConfig(num_workers=0),
        stage_adapter=StageAdapterTrainingConfig(
            residual_adapter_dim=4,
            stage_adapter_a_epochs=2,
            micro_batch_size=4,
            eval_batch_size=4,
            gradient_accumulation_steps=1,
            min_rounds=1,
            max_rounds=2,
            mixed_precision="none",
            gradient_checkpointing=False,
        ),
        swanlab=SwanLabConfig(mode="disabled"),
    )


def _model(config: StageAdapterExperimentConfig) -> ProgressiveStageAdapterModel:
    return ProgressiveStageAdapterModel(
        StageAdapterBindingModel(
            config.model,
            residual_adapter_dim=config.stage_adapter.residual_adapter_dim,
        ),
        DummyEncoder(config.model.peptide_input_dim),
        DummyEncoder(config.model.hla_input_dim),
        DummyEncoder(config.model.tcr_input_dim),
    )


def _batch() -> dict:
    records = [
        Stage8Record("AA", "CC", "DD", 1, 1, 1, 1),
        Stage8Record("AC", "CD", "DE", 1, 0, 0, 2),
        Stage8Record("AD", "CE", "DF", 0, 1, 0, 3),
        Stage8Record("AE", "CF", "DG", 0, 0, 0, 4),
    ]
    return collate_stage_adapter(records)


class StageAdapterConfigTests(unittest.TestCase):
    def test_public_config_encodes_pht_only_design(self) -> None:
        path = Path(__file__).resolve().parents[2] / "configs/plm_stage_adapter.yaml"
        config = load_stage_adapter_config(path)
        self.assertEqual(config.model.model_dim, 256)
        self.assertEqual(config.stage_adapter.stage_adapter_a_epochs, 2)
        self.assertEqual(config.stage_adapter.min_rounds, 5)
        self.assertEqual(config.stage_adapter.max_rounds, 100)
        self.assertEqual(config.stage_adapter.early_stopping_patience, 20)
        self.assertEqual(
            config.paths.stage1_checkpoint.name,
            "last_20260831_174042_678934.pt",
        )
        self.assertEqual(
            config.paths.stage1_sha256,
            "77655f5ac9e20309f8cfccbfa5f259941851c8545b54d4ec8e7272088688c7bb",
        )
        self.assertEqual(
            tuple(config.stage_adapter.class_dirs),
            ("PHT_000", "PHT_010", "PHT_100", "PHT_111"),
        )


class StageAdapterLossTests(unittest.TestCase):
    def test_loss_weights_positive_and_three_negative_types_equally(self) -> None:
        logits = torch.tensor(
            [[0.1, 1.1], [1.2, 0.2], [0.8, 0.3], [1.4, 0.1]],
            requires_grad=True,
        )
        labels = torch.tensor([1, 0, 0, 0])
        actual = balanced_pht_cross_entropy(
            logits, labels, ["111", "100", "010", "000"]
        )
        row_losses = F.cross_entropy(logits, labels, reduction="none")
        expected = 0.5 * row_losses[0] + 0.5 * row_losses[1:].mean()
        torch.testing.assert_close(actual, expected)


class StageAdapterModelTests(unittest.TestCase):
    def test_only_pht_modules_are_trainable_and_routing_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _model(_config(Path(directory)))
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("binding_model.pht_") for name in trainable))
        self.assertFalse(model.peptide_encoder.active)
        self.assertFalse(model.hla_encoder.active)
        self.assertFalse(model.tcr_encoder.active)
        with self.assertRaises(ValueError):
            model("phla", ["AA"], tcr_sequences=["CC"])
        with self.assertRaises(ValueError):
            model("pht", ["AA"], hla_sequences=["DD"])

    def test_optimizer_exactly_covers_pht_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            model = _model(config)
            optimizer = build_stage_adapter_optimizer(model, config)
        optimised = {id(p) for group in optimizer.param_groups for p in group["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(optimised, trainable)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["pht_residual_adapters", "pht_interaction_modules"],
        )

    def test_one_train_pass_changes_pht_but_not_frozen_ph_expert(self) -> None:
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            model = _model(config)
            batch = _batch()
            model.set_evaluation()
            before_ph = model.forward_ph(
                batch["peptide_sequences"], batch["hla_sequences"]
            ).logits.detach().clone()
            before_checksum = model.frozen_binding_checksum()
            before_pht = {
                name: value.detach().clone()
                for name, value in model.binding_model.state_dict().items()
                if name.startswith("pht_")
            }
            trainer = StageAdapterTrainer(model, config, [batch], [batch])
            result = trainer.train_stage_adapter_a_epoch(1)
            model.set_evaluation()
            after_ph = model.forward_ph(
                batch["peptide_sequences"], batch["hla_sequences"]
            ).logits.detach()
        self.assertEqual(result["samples"], 4)
        self.assertEqual(before_checksum, model.frozen_binding_checksum())
        torch.testing.assert_close(before_ph, after_ph, rtol=0.0, atol=0.0)
        self.assertTrue(
            any(
                not torch.equal(value, model.binding_model.state_dict()[name])
                for name, value in before_pht.items()
            )
        )

    def test_frozen_checksum_supports_bfloat16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _model(_config(Path(directory))).bfloat16()
        self.assertEqual(len(model.frozen_binding_checksum()), 64)


class StageAdapterCheckpointTests(unittest.TestCase):
    def test_warmup_is_a_v2_full_model_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            model = _model(config)
            labels = [1, 0, 0, 0]
            probabilities = [0.9, 0.2, 0.3, 0.1]
            metrics = compute_binary_metrics(labels, probabilities, loss=0.2)
            evaluation = SimpleNamespace(
                metrics=metrics,
                selected_metrics=metrics,
                selected_threshold=0.5,
                selection_score=metrics.performance_avg,
            )
            source = {
                "sha256": config.paths.stage1_sha256,
                "frozen_binding_checksum": model.frozen_binding_checksum(),
            }
            path = save_stage_adapter_warmup(
                root,
                model=model,
                config=config,
                source_stage1=source,
                dataset_fingerprint="dataset",
                effective_data={"train_samples": 4, "validation_samples": 4},
                run_timestamp="test",
                evaluation=evaluation,
                training_state={"stage_adapter_a_epoch": 2},
            )
            checkpoint = load_stage_adapter_checkpoint(path, role="warmup")
        self.assertEqual(checkpoint["format_version"], FORMAT_VERSION)
        self.assertEqual(checkpoint["checkpoint_type"], CHECKPOINT_TYPE)
        self.assertIn("model", checkpoint)
        self.assertNotIn("optimizer", checkpoint)
        self.assertEqual(
            set(checkpoint["randomly_initialized_modules"]),
            set(model.binding_model.pht_modules()),
        )
        self.assertEqual(set(checkpoint["metrics"]), {
            "pht", "pht_selected", "selected_pht_threshold", "selection_score"
        })


if __name__ == "__main__":
    unittest.main()
