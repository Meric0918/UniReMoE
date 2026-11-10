"""Synthetic tests for Stage-5 three-task joint training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

if HAS_DEPS:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        HAS_DEPS = False


class RecordingTracker:
    def __init__(self):
        self.run_id = "synthetic-stage5-run"
        self.records = []

    def log(self, values, *, step=None):
        self.records.append((step, dict(values)))


@unittest.skipUnless(
    HAS_DEPS,
    "PyTorch and scikit-learn are required for Stage-5 tests",
)
class Stage5PipelineTests(unittest.TestCase):
    def _config(self, root: Path):
        from source.plm_unified.config import (
            ExperimentConfig,
            ModelConfig,
            PathsConfig,
            Stage5Config,
            SwanLabConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            paths=PathsConfig(
                project_root=root,
                data_root=root,
                cache_root=root / "cache",
                output_root=root / "models",
                swanlog_root=root / "swanlog",
            ),
            model=ModelConfig(
                peptide_input_dim=12,
                hla_input_dim=12,
                tcr_input_dim=8,
                adapter_hidden_dim=16,
                model_dim=8,
                num_attention_heads=2,
                feedforward_dim=16,
                adapter_dropout=0.0,
                attention_dropout=0.0,
                classifier_dropout=0.0,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            ),
            training=TrainingConfig(
                num_workers=0,
                mixed_precision="none",
                pin_memory=False,
                persistent_workers=False,
                device="cpu",
            ),
            stage5=Stage5Config(
                micro_batch_size=2,
                eval_batch_size=2,
                gradient_accumulation_steps=1,
                stage1a_epochs=1,
                stage1b_epochs=2,
                mixed_precision="none",
                log_every_optimizer_steps=1,
            ),
            swanlab=SwanLabConfig(mode="disabled"),
        )

    @staticmethod
    def _tiny_encoder(hidden_size: int, max_length: int, name: str):
        from source.plm_unified.encoders import ResidueBatch

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.max_length = max_length
                self.checkpoint_path = Path(f"/synthetic/{name}")
                self.checkpoint_fingerprint = f"fingerprint-{name}"
                self.gradient_checkpointing = False
                self.stem = nn.Linear(hidden_size, hidden_size)
                self.blocks = nn.ModuleList(
                    nn.Linear(hidden_size, hidden_size) for _ in range(2)
                )
                self.requires_grad_(False)

            def transformer_blocks(self):
                return self.blocks

            def set_active_training(self, active: bool):
                self.train(active)

            def metadata(self):
                return {
                    "checkpoint_path": str(self.checkpoint_path),
                    "checkpoint_fingerprint": self.checkpoint_fingerprint,
                    "hidden_size": self.hidden_size,
                    "max_length": self.max_length,
                    "encoder_class": type(self).__name__,
                    "transformer_layers": len(self.blocks),
                    "gradient_checkpointing": False,
                }

            def forward(self, sequences):
                values = torch.zeros(len(sequences), max_length, hidden_size)
                mask = torch.zeros(
                    len(sequences), max_length, dtype=torch.bool
                )
                lengths = torch.zeros(len(sequences), dtype=torch.long)
                for row, sequence in enumerate(sequences):
                    length = min(len(sequence), max_length)
                    values[row, :length, 0] = 1.0
                    mask[row, :length] = True
                    lengths[row] = length
                hidden = torch.tanh(self.stem(values))
                for block in self.blocks:
                    hidden = torch.tanh(block(hidden))
                return ResidueBatch(hidden, mask, lengths)

        return TinyEncoder()

    def _stage5_model(self, config):
        from source.plm_unified.stage5_model import Stage5FineTuningModel
        from source.plm_unified.triplet_model import TripletBindingModel

        return Stage5FineTuningModel(
            TripletBindingModel(config.model),
            self._tiny_encoder(12, 8, "peptide"),
            self._tiny_encoder(12, 8, "hla"),
            self._tiny_encoder(8, 8, "tcr"),
            peptide_unfrozen_layers=0,
            hla_unfrozen_layers=0,
            tcr_unfrozen_layers=0,
        )

    @staticmethod
    def _loaders():
        phla_batch = {
            "peptide_sequences": ["AAAA", "CCCC", "DDDD", "EEEE", "FFFF"],
            "receptor_sequences": ["GGGG", "TTTT", "VVVV", "WWWW", "YYYY"],
            "labels": torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        }
        ptcr_batch = {
            "peptide_sequences": ["AAAA", "CCCC", "DDDD"],
            "receptor_sequences": ["GGGG", "TTTT", "VVVV"],
            "labels": torch.tensor([0, 1, 0], dtype=torch.long),
        }
        triplet_batch = {
            "peptide_sequences": ["AAAA", "CCCC"],
            "hla_sequences": ["HHHH", "YYYY"],
            "tcr_sequences": ["GGGG", "TTTT"],
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
        return {
            "phla": [phla_batch] * 5,
            "ptcr": [ptcr_batch] * 3,
            "triplet": [triplet_batch, triplet_batch],
        }

    def test_downstream_modules_start_from_random_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            torch.manual_seed(1)
            first = self._stage5_model(config)
            torch.manual_seed(2)
            second = self._stage5_model(config)
            self.assertTrue(
                any(
                    not torch.equal(first_parameter, second_parameter)
                    for first_parameter, second_parameter in zip(
                        first.binding_model.parameters(),
                        second.binding_model.parameters(),
                    )
                )
            )
            first.eval()
            sequences = ["AAAA", "CCCC"]
            actual = first.forward_triplet(
                sequences, sequences, sequences
            )
            self.assertEqual(tuple(actual.logits.shape), (2, 2))

    def test_task_losses_are_independently_normalized(self):
        from source.plm_unified.stage5_trainer import (
            independently_normalized_stage5_loss,
        )

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            per_sample_losses = {
                "phla": torch.full((100,), 2.0, requires_grad=True),
                "ptcr": torch.full((10,), 3.0, requires_grad=True),
                "triplet": torch.full((2,), 5.0, requires_grad=True),
            }
            total, task_losses = independently_normalized_stage5_loss(
                per_sample_losses,
                config,
            )
            self.assertEqual(float(task_losses["phla"]), 2.0)
            self.assertEqual(float(task_losses["ptcr"]), 3.0)
            self.assertEqual(float(task_losses["triplet"]), 5.0)
            self.assertAlmostEqual(float(total), 6.2, places=6)
            total.backward()
            torch.testing.assert_close(
                per_sample_losses["phla"].grad,
                torch.full((100,), config.stage5.phla_loss_weight / 100),
            )
            torch.testing.assert_close(
                per_sample_losses["ptcr"].grad,
                torch.full((10,), config.stage5.ptcr_loss_weight / 10),
            )
            torch.testing.assert_close(
                per_sample_losses["triplet"].grad,
                torch.full((2,), config.stage5.pht_loss_weight / 2),
            )

    def test_plms_are_frozen_and_excluded_from_optimizer(self):
        from source.plm_unified.stage5_trainer import build_stage5_optimizer

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model = self._stage5_model(config)
            model.set_trainable_stage5("stage5_1a")
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in model.binding_model.parameters()
                )
            )
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for encoder in (
                        model.peptide_encoder,
                        model.hla_encoder,
                        model.tcr_encoder,
                    )
                    for parameter in encoder.parameters()
                )
            )
            optimizer = build_stage5_optimizer(model, config)
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(
                all(
                    id(parameter) not in optimized
                    for parameter in model.peptide_encoder.parameters()
                )
            )

    def test_fgm_targets_exactly_the_three_adapters(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model = self._stage5_model(config)
            targets = model.fgm_modules()
            self.assertEqual(
                set(targets),
                {"peptide_adapter", "hla_adapter", "tcr_adapter"},
            )
            self.assertIs(
                targets["peptide_adapter"], model.binding_model.peptide_adapter
            )
            self.assertIs(targets["hla_adapter"], model.binding_model.hla_adapter)
            self.assertIs(targets["tcr_adapter"], model.binding_model.tcr_adapter)
            targeted = {
                id(parameter)
                for module in targets.values()
                for parameter in module.parameters()
            }
            non_adapters = {
                id(parameter)
                for name, parameter in model.binding_model.named_parameters()
                if not name.startswith(
                    ("peptide_adapter.", "hla_adapter.", "tcr_adapter.")
                )
            }
            self.assertFalse(targeted & non_adapters)

    def test_stage1b_fgm_supports_gradient_accumulation(self):
        from source.plm_unified.stage5_trainer import (
            Stage5Trainer,
            stage5_scratch_initialization_source,
        )

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            config.stage5.gradient_accumulation_steps = 2
            loaders = self._loaders()
            trainer = Stage5Trainer(
                self._stage5_model(config),
                config,
                loaders,
                loaders,
                RecordingTracker(),
                initialization_source=stage5_scratch_initialization_source(),
                data_audit={},
                run_timestamp="20260804_120000_000002",
            )
            trainer._build_optimization("stage5_1b")
            result = trainer.train_joint_epoch(
                phase="stage5_1b",
                epoch_id=1,
            )
            self.assertEqual(result.joint_steps, 2)
            self.assertEqual(result.optimizer_steps, 1)
            self.assertGreater(result.losses["adversarial_total"], 0.0)
            self.assertGreater(result.fgm_attacked_parameter_count, 0.0)
            self.assertFalse(trainer.fgm._backup)

    def test_fixed_phase_transition_swanlog_and_resume(self):
        from source.plm_unified.stage5_trainer import (
            Stage5Trainer,
            load_stage5_checkpoint,
            stage5_scratch_initialization_source,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            loaders = self._loaders()
            tracker = RecordingTracker()
            trainer = Stage5Trainer(
                self._stage5_model(config),
                config,
                loaders,
                loaders,
                tracker,
                initialization_source=stage5_scratch_initialization_source(),
                data_audit={"manifest": {"matches": True}},
                run_timestamp="20260804_120000_000001",
            )
            summary = trainer.fit()
            self.assertEqual(summary["phase"], "complete")
            self.assertEqual(summary["stage1a_epoch"], 1)
            self.assertEqual(summary["stage1b_epoch"], 2)
            self.assertEqual(
                summary["initialization_source"]["kind"], "scratch"
            )
            self.assertTrue(trainer.stage1a_complete_path.is_file())
            self.assertTrue(trainer.stage1b_best_path.is_file())
            self.assertTrue(trainer.last_path.is_file())
            self.assertTrue(trainer.deploy_path.is_file())
            epoch_values = next(
                values
                for _, values in tracker.records
                if "stage5_1b/train/loss_total" in values
            )
            self.assertFalse(
                any("pt_imp" in key for key in epoch_values)
            )
            expected_clean_total = (
                config.stage5.pht_loss_weight
                * epoch_values["stage5_1b/train/loss_pht"]
                + config.stage5.phla_loss_weight
                * epoch_values["stage5_1b/train/loss_ph"]
                + config.stage5.ptcr_loss_weight
                * epoch_values["stage5_1b/train/loss_pt"]
            )
            self.assertAlmostEqual(
                epoch_values["stage5_1b/train/loss_clean_total"],
                expected_clean_total,
                places=6,
            )
            self.assertGreater(
                epoch_values["stage5_1b/train/loss_adversarial_total"], 0.0
            )
            self.assertAlmostEqual(
                epoch_values["stage5_1b/train/loss_total"],
                epoch_values["stage5_1b/train/loss_clean_total"]
                + epoch_values["stage5_1b/train/loss_adversarial_total"],
                places=6,
            )
            stage1a_values = next(
                values
                for _, values in tracker.records
                if "stage5_1a/train/loss_total" in values
            )
            self.assertEqual(
                stage1a_values["stage5_1a/train/loss_adversarial_total"], 0.0
            )
            self.assertEqual(
                stage1a_values["stage5_1a/train/fgm_attacked_parameters"], 0.0
            )
            self.assertGreater(
                epoch_values["stage5_1b/train/fgm_attacked_parameters"], 0.0
            )
            self.assertEqual(
                epoch_values["stage5_1b/train/joint_steps"],
                len(loaders["triplet"]),
            )
            checkpoint = load_stage5_checkpoint(
                trainer.last_path, map_location="cpu"
            )
            self.assertEqual(checkpoint["swanlab_run_id"], tracker.run_id)
            self.assertEqual(checkpoint["phase"], "complete")

            resumed = Stage5Trainer(
                self._stage5_model(config),
                config,
                loaders,
                loaders,
                RecordingTracker(),
                initialization_source=stage5_scratch_initialization_source(),
                data_audit={},
                run_timestamp="20260804_120000_000001",
            )
            resumed.load_checkpoint(trainer.last_path)
            self.assertEqual(resumed.phase, "complete")
            self.assertEqual(resumed.stage1b_epoch, 2)
            self.assertEqual(resumed.initialization_source["kind"], "scratch")

            checkpoint["initialization_source"] = {
                "kind": "stage2",
                "path": "/synthetic/stage2.pt",
            }
            torch.save(checkpoint, trainer.last_path)
            with self.assertRaisesRegex(ValueError, "inheritance is disabled"):
                resumed.load_checkpoint(trainer.last_path)


if __name__ == "__main__":
    unittest.main()
