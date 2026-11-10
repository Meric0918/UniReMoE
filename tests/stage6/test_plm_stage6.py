"""Synthetic tests for the isolated Stage-6 ablation pipeline."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn

    import sklearn  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class RecordingTracker:
    def __init__(self):
        self.run_id = "synthetic-stage6-run"
        self.records = []

    def log(self, values, *, step=None):
        self.records.append((step, dict(values)))


@unittest.skipUnless(
    HAS_DEPS,
    "PyTorch and scikit-learn are required for Stage-6 tests",
)
class Stage6PipelineTests(unittest.TestCase):
    def _config(self, root: Path, condition="stage6_phla_only_matched"):
        from source.plm_unified.config import (
            ModelConfig,
            PathsConfig,
            SwanLabConfig,
            TrainingConfig,
        )
        from source.plm_unified.stage6_config import (
            Stage6Config,
            Stage6ExperimentConfig,
        )

        return Stage6ExperimentConfig(
            paths=PathsConfig(
                project_root=root,
                data_root=root,
                cache_root=root / "cache",
                output_root=root / "PLM_Stage6",
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
                fold=4,
                seed=2026,
                num_workers=0,
                mixed_precision="none",
                pin_memory=False,
                persistent_workers=False,
                device="cpu",
            ),
            stage6=Stage6Config(
                condition=condition,
                micro_batch_size=2,
                eval_batch_size=2,
                gradient_accumulation_steps=1,
                logical_steps_per_epoch=2,
                stage1a_epochs=1,
                stage1b_epochs=1,
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
                mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
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

    def _model(self, config):
        from source.plm_unified.stage6_model import Stage6FineTuningModel
        from source.plm_unified.triplet_model import TripletBindingModel

        return Stage6FineTuningModel(
            TripletBindingModel(config.model),
            self._tiny_encoder(12, 8, "peptide"),
            self._tiny_encoder(12, 8, "hla"),
            self._tiny_encoder(8, 8, "tcr"),
            peptide_unfrozen_layers=0,
            hla_unfrozen_layers=0,
            tcr_unfrozen_layers=0,
        )

    @staticmethod
    def _phla_batches():
        batches = []
        for index in range(5):
            batches.append(
                {
                    "peptide_sequences": ["AAAA", "CCCC"],
                    "receptor_sequences": ["GGGG", "TTTT"],
                    "labels": torch.tensor([index % 2, (index + 1) % 2]),
                }
            )
        return batches

    @classmethod
    def _loaders(cls, joint=False):
        loaders = {"phla": cls._phla_batches()}
        if joint:
            loaders["ptcr"] = [
                {
                    "peptide_sequences": ["AAAA", "CCCC"],
                    "receptor_sequences": ["GGGG", "TTTT"],
                    "labels": torch.tensor([0, 1]),
                }
            ] * 3
            loaders["triplet"] = [
                {
                    "peptide_sequences": ["AAAA", "CCCC"],
                    "hla_sequences": ["HHHH", "YYYY"],
                    "tcr_sequences": ["GGGG", "TTTT"],
                    "labels": torch.tensor([0, 1]),
                }
            ] * 2
        return loaders

    def _trainer(self, root, condition):
        from source.plm_unified.stage6_trainer import (
            Stage6Trainer,
            stage6_scratch_initialization_source,
        )

        config = self._config(root, condition)
        loaders = self._loaders(joint=config.stage6.task_mode == "joint")
        return Stage6Trainer(
            self._model(config),
            config,
            loaders,
            loaders,
            RecordingTracker(),
            initialization_source=stage6_scratch_initialization_source(),
            data_audit={},
            run_timestamp="20260807_120000_000001",
        )

    def test_exact_full_phla_batch_allocation(self):
        from source.plm_unified.stage6_trainer import phla_batches_per_step

        schedule = phla_batches_per_step(200_762, 45_533, "full")
        self.assertEqual(len(schedule), 45_533)
        self.assertEqual(sum(schedule), 200_762)
        self.assertEqual(schedule.count(5), 18_630)
        self.assertEqual(schedule.count(4), 26_903)
        self.assertEqual(
            phla_batches_per_step(200_762, 45_533, "matched").count(1),
            45_533,
        )

    def test_multi_batch_phla_loss_is_sample_meaned_once(self):
        from source.plm_unified.stage6_trainer import stage6_weighted_loss

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(
                Path(temporary), "stage6_phla_only_full"
            )
            losses = torch.full((20,), 2.0, requires_grad=True)
            total, task_losses = stage6_weighted_loss(
                {"phla": losses}, config
            )
            self.assertEqual(float(task_losses["phla"]), 2.0)
            self.assertAlmostEqual(float(total), 0.6, places=6)
            total.backward()
            torch.testing.assert_close(
                losses.grad,
                torch.full((20,), config.stage6.phla_loss_weight / 20),
            )

    def test_matched_and_full_have_equal_logical_steps_but_different_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matched = self._trainer(root, "stage6_phla_only_matched")
            matched_result = matched.train_epoch(phase="stage6_1a", epoch_id=1)
            full = self._trainer(root, "stage6_phla_only_full")
            full_result = full.train_epoch(phase="stage6_1a", epoch_id=1)
            self.assertEqual(matched_result.logical_steps, 2)
            self.assertEqual(full_result.logical_steps, 2)
            self.assertEqual(matched_result.optimizer_steps, 2)
            self.assertEqual(full_result.optimizer_steps, 2)
            self.assertEqual(matched_result.task_batches["phla"], 2)
            self.assertEqual(full_result.task_batches["phla"], 5)
            self.assertEqual(matched_result.task_samples["phla"], 4)
            self.assertEqual(full_result.task_samples["phla"], 10)

    def test_phla_only_freezes_ptcr_triplet_and_tcr_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self._trainer(
                Path(temporary), "stage6_phla_only_matched"
            )
            binding = trainer.model.binding_model
            frozen_modules = (
                binding.tcr_adapter,
                binding.ptcr_cross_attention,
                binding.ptcr_pooling,
                binding.ptcr_classifier,
                *binding.new_triplet_modules().values(),
            )
            before = [
                parameter.detach().clone()
                for module in frozen_modules
                for parameter in module.parameters()
            ]
            trainer.train_epoch(phase="stage6_1a", epoch_id=1)
            after = [
                parameter.detach()
                for module in frozen_modules
                for parameter in module.parameters()
            ]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
            self.assertEqual(
                set(trainer.model.fgm_modules("phla_only")),
                {"peptide_adapter", "hla_adapter"},
            )

    def test_joint_full_tracks_all_tasks_and_stage6_artifact_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self._trainer(
                Path(temporary), "stage6_joint_phla_full"
            )
            result = trainer.train_epoch(phase="stage6_1b", epoch_id=1)
            self.assertEqual(set(result.metrics), {"phla", "ptcr", "triplet"})
            self.assertEqual(result.task_batches["phla"], 5)
            self.assertEqual(result.task_batches["ptcr"], 2)
            self.assertEqual(result.task_batches["triplet"], 2)
            self.assertGreater(result.fgm_attacked_parameter_count, 0)
            self.assertIn("PLM_Stage6", str(trainer.run_dir))
            self.assertIn("stage6_joint_phla_full", str(trainer.run_dir))
            self.assertTrue(trainer.last_path.name.startswith("stage6_last_"))
            self.assertNotIn("PLM_Stage5", str(trainer.last_path))

    def test_fit_checkpoint_and_resume_remain_stage6_local(self):
        from source.plm_unified.stage6_trainer import load_stage6_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self._trainer(root, "stage6_phla_only_matched")
            summary = trainer.fit()
            self.assertEqual(summary["phase"], "complete")
            self.assertTrue(trainer.stage1a_complete_path.is_file())
            self.assertTrue(trainer.stage1b_best_path.is_file())
            self.assertTrue(trainer.last_path.is_file())
            self.assertTrue(trainer.deploy_path.is_file())
            checkpoint = load_stage6_checkpoint(
                trainer.last_path, map_location="cpu"
            )
            self.assertEqual(checkpoint["condition"], "stage6_phla_only_matched")
            self.assertEqual(checkpoint["tasks"], ["phla"])
            self.assertEqual(checkpoint["checkpoint_type"], "stage6_training")
            resumed = self._trainer(root, "stage6_phla_only_matched")
            resumed.load_checkpoint(trainer.last_path)
            self.assertEqual(resumed.phase, "complete")
            self.assertEqual(resumed.initialization_source["stage"], "stage6")
            self.assertNotIn("PLM_Stage5", str(resumed.last_path))

    def test_external_cleanup_removes_overlap_ambiguity_and_duplicates(self):
        spec = importlib.util.spec_from_file_location(
            "stage6_eval",
            Path(__file__).resolve().parents[2]
            / "source"
            / "evaluate_plm_stage6_phla.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            external = root / "external.csv"
            for path, rows in (
                (train, [("AAAAAAAA", "H1", 0)]),
                (
                    external,
                    [
                        ("AAAAAAAA", "H1", 1),
                        ("CCCCCCCC", "H2", 0),
                        ("CCCCCCCC", "H2", 1),
                        ("DDDDDDDD", "H3", 1),
                        ("DDDDDDDD", "H3", 1),
                        ("EEEEEEEE", "H4", 0),
                    ],
                ),
            ):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("peptide", "HLA", "label"))
                    writer.writerows(rows)
            dataset = module.build_evaluation_dataset(
                external,
                train_path=train,
                peptide_min_length=8,
                peptide_max_length=15,
                clean_external=True,
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.filter_stats["train_overlap_rows_removed"], 1)
            self.assertEqual(dataset.filter_stats["ambiguous_rows_removed"], 2)
            self.assertEqual(dataset.filter_stats["duplicate_rows_removed"], 1)
            self.assertEqual(
                {(record.peptide, record.label) for record in dataset.records},
                {("DDDDDDDD", 1), ("EEEEEEEE", 0)},
            )

    def test_weighted_bootstrap_metric_matches_project_metric_and_is_reproducible(self):
        from source.plm_unified.metrics import compute_binary_metrics

        spec = importlib.util.spec_from_file_location(
            "stage6_analysis",
            Path(__file__).resolve().parents[2]
            / "source"
            / "analyze_plm_stage6.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        probabilities = np.asarray(
            [0.2, 0.7, 0.8, 0.6, 0.4, 0.1, 0.9, 0.3], dtype=np.float64
        )
        auroc, aupr = module._weighted_auc_aupr_many(
            labels,
            probabilities,
            np.ones((1, len(labels)), dtype=np.int32),
        )
        expected = compute_binary_metrics(labels, probabilities, loss=0.0)
        self.assertAlmostEqual(auroc[0], expected.auroc)
        self.assertAlmostEqual(aupr[0], expected.aupr)
        predictions = {
            condition: np.clip(probabilities + index * 0.01, 0, 1)
            for index, condition in enumerate(
                (
                    "stage5_joint_matched",
                    "stage6_phla_only_matched",
                    "stage6_joint_phla_full",
                    "stage6_phla_only_full",
                )
            )
        }
        first = module.paired_stratified_bootstrap(
            labels, predictions, samples=10, seed=2026, chunk_size=3
        )
        second = module.paired_stratified_bootstrap(
            labels, predictions, samples=10, seed=2026, chunk_size=3
        )
        for condition in predictions:
            np.testing.assert_array_equal(
                first[condition]["task_score"],
                second[condition]["task_score"],
            )


if __name__ == "__main__":
    unittest.main()
