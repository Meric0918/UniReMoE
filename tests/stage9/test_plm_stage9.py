"""Contract tests for Stage-9 paired PHT contrastive fine-tuning."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from source.plm_unified.config import ModelConfig


PAIR_HEADER = [
    "peptide",
    "hla",
    "tcr_positive",
    "tcr_negative",
    "pair_id",
]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _pair_row(index: int) -> list[str]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    first = alphabet[index % len(alphabet)]
    second = alphabet[(index // len(alphabet)) % len(alphabet)]
    peptide = (first + second + "ACDEFG")[:8]
    hla = (first + second + "Y" * 32)[:34]
    positive = f"CASS{first}{second}GQETQYF"
    negative = f"CASS{second}{first}IRSSYEQYF"
    return [peptide, hla, positive, negative, f"pair-{index}"]


class Stage9ConfigTests(unittest.TestCase):
    def test_defaults_encode_approved_training_design(self) -> None:
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )

        root = Path("/tmp/stage9-contract")
        config = Stage9ExperimentConfig(
            paths=Stage9Paths(
                project_root=root,
                data_root=root / "data",
                output_root=root / "models",
                swanlog_root=root / "logs",
                stage3_checkpoint=root / "stage3.pt",
            )
        )
        config.validate()
        self.assertEqual(config.stage9.split_seed, 2026)
        self.assertEqual(config.stage9.split_ratios, (0.8, 0.1, 0.1))
        self.assertEqual(config.stage9.temperature, 0.07)
        self.assertEqual(config.stage9.contrastive_loss_weight, 1.0)
        self.assertEqual(config.stage9.classification_loss_weight, 1.0)
        self.assertEqual(config.stage9.stage9a_epochs, 2)
        self.assertEqual(config.stage9.replay_ratio, 0.25)
        self.assertEqual(config.stage9.legacy_degradation_tolerance, 0.01)
        self.assertEqual(config.stage9.peptide_unfrozen_layers, 0)
        self.assertEqual(config.stage9.hla_unfrozen_layers, 0)
        self.assertEqual(config.stage9.tcr_unfrozen_layers, 0)

    def test_invalid_temperature_and_split_ratios_are_rejected(self) -> None:
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )

        root = Path("/tmp/stage9-contract")
        config = Stage9ExperimentConfig(
            paths=Stage9Paths(root, root, root, root, root / "stage3.pt")
        )
        config.stage9.temperature = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            config.validate()
        config.stage9.temperature = 0.07
        config.stage9.train_fraction = 0.7
        with self.assertRaisesRegex(ValueError, "split fractions"):
            config.validate()

    def test_repository_yaml_loads_and_resolves_paths(self) -> None:
        from source.plm_unified.stage9_config import load_stage9_config

        config = load_stage9_config(
            Path(__file__).resolve().parents[2] / "configs" / "plm_stage9.yaml"
        )
        self.assertTrue(config.paths.project_root.is_absolute())
        self.assertEqual(config.stage9.pair_filename, "pht_pairs_clean.csv")
        self.assertEqual(
            config.stage9.supplement_filename,
            "pht_pairs_full_valtest_supplement.csv",
        )
        self.assertEqual(config.stage9.peptide_unfrozen_layers, 0)
        self.assertEqual(config.stage9.hla_unfrozen_layers, 0)
        self.assertEqual(config.stage9.tcr_unfrozen_layers, 0)
        self.assertEqual(config.swanlab.group, "plm-stage9-fold{fold}")

    def test_checkpoint_training_values_are_restored_for_resume_and_evaluation(self) -> None:
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
            restore_stage9_checkpoint_config,
        )

        root = Path("/tmp/stage9-contract")
        source = Stage9ExperimentConfig(
            paths=Stage9Paths(root, root, root, root, root / "stage3.pt")
        )
        source.stage9.temperature = 0.2
        source.stage9.projection_dim = 64
        source.runtime.fold = 3
        source.runtime.seed = 42
        target = Stage9ExperimentConfig(
            paths=Stage9Paths(root, root, root, root, root / "stage3.pt")
        )

        restore_stage9_checkpoint_config(target, source.to_dict())

        self.assertEqual(target.stage9.temperature, 0.2)
        self.assertEqual(target.stage9.projection_dim, 64)
        self.assertEqual(target.runtime.fold, 3)
        self.assertEqual(target.runtime.seed, 42)


class Stage9DataTests(unittest.TestCase):
    def test_audit_rejects_tcr_role_conflicts_and_identical_pairs(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            shared = _pair_row(0)
            conflict = _pair_row(1)
            conflict[0] = shared[0]
            conflict[1] = shared[1]
            conflict[2] = shared[3]
            identical = _pair_row(2)
            identical[3] = identical[2]
            _write_csv(pairs, PAIR_HEADER, [shared, conflict, identical])
            _write_csv(stage3, ["peptide", "hla", "tcr", "label"], [])

            result = audit_stage9_pairs(pairs, stage3, ModelConfig())

            self.assertTrue(result.has_errors)
            reasons = " ".join(item["reason"] for item in result.errors)
            self.assertIn("positive and negative", reasons)
            self.assertIn("both positive and negative roles", reasons)
            with self.assertRaisesRegex(ValueError, "Stage-9 data audit failed"):
                result.require_clean()

    def test_duplicate_pairs_are_deduplicated_and_missing_pair_id_is_generated(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            rows = [_pair_row(index)[:-1] for index in range(80)]
            rows.append(list(rows[3]))
            _write_csv(pairs, PAIR_HEADER[:-1], rows)
            _write_csv(stage3, ["peptide", "hla", "tcr", "label"], [])

            result = audit_stage9_pairs(pairs, stage3, ModelConfig())

            result.require_clean()
            retained = result.train_records + result.val_records + result.test_records
            self.assertEqual(len(retained), 80)
            record = next(item for item in retained if item.peptide == rows[3][0])
            self.assertEqual(record.pair_id, "row:5")
            self.assertEqual(result.report["duplicate_pairs_removed"], 1)

    def test_common_case_insensitive_pair_header_aliases_are_accepted(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            _write_csv(
                pairs,
                ["Peptide", "HLA", "pos_TCR", "neg_TCR"],
                [_pair_row(index) for index in range(80)],
            )
            _write_csv(stage3, ["peptide", "tcr", "hla", "label"], [])

            result = audit_stage9_pairs(pairs, stage3, ModelConfig())

            result.require_clean()
            records = result.train_records + result.val_records + result.test_records
            self.assertEqual(len(records), 80)
            self.assertEqual(records[0].pair_id, "row:2")

    def test_supplement_can_assign_unseen_ph_groups_directly_to_val_and_test(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            supplement = root / "supplement.csv"
            stage3 = root / "stage3.csv"
            _write_csv(pairs, PAIR_HEADER, [_pair_row(index) for index in range(80)])
            with supplement.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(PAIR_HEADER + ["split"])
                for index in range(80, 100):
                    writer.writerow(
                        _pair_row(index) + ["val" if index < 90 else "test"]
                    )
            _write_csv(stage3, ["peptide", "hla", "tcr", "label"], [])

            result = audit_stage9_pairs(
                pairs,
                stage3,
                ModelConfig(),
                supplement_csv=supplement,
            ).require_clean()

            self.assertEqual(result.report["supplement_rows"], 20)
            self.assertEqual(result.report["supplement_split_rows"], {"val": 10, "test": 10})
            self.assertEqual(
                {record.pair_id for record in result.val_records}
                & {f"pair-{index}" for index in range(80, 90)},
                {f"pair-{index}" for index in range(80, 90)},
            )
            self.assertEqual(
                {record.pair_id for record in result.test_records}
                & {f"pair-{index}" for index in range(90, 100)},
                {f"pair-{index}" for index in range(90, 100)},
            )

    def test_stage3_overlap_forces_whole_ph_group_into_training(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            rows = [_pair_row(index) for index in range(80)]
            same_ph = list(rows[0])
            same_ph[2] = "CASSTTTGQETQYF"
            same_ph[3] = "CASSVVVIRSSYEQYF"
            same_ph[4] = "same-ph"
            rows.append(same_ph)
            _write_csv(pairs, PAIR_HEADER, rows)
            _write_csv(
                stage3,
                ["peptide", "hla", "tcr", "label"],
                [[rows[0][0], rows[0][1], rows[0][2], 1]],
            )

            result = audit_stage9_pairs(pairs, stage3, ModelConfig())

            result.require_clean()
            ph = (rows[0][0], rows[0][1])
            self.assertTrue(all(record.ph_key != ph for record in result.val_records))
            self.assertTrue(all(record.ph_key != ph for record in result.test_records))
            self.assertEqual(
                sum(record.ph_key == ph for record in result.train_records), 2
            )
            self.assertEqual(result.report["forced_train_ph_groups"], 1)

    def test_hash_split_is_reproducible_and_ph_disjoint(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            _write_csv(pairs, PAIR_HEADER, [_pair_row(index) for index in range(200)])
            _write_csv(stage3, ["peptide", "hla", "tcr", "label"], [])

            first = audit_stage9_pairs(pairs, stage3, ModelConfig()).require_clean()
            second = audit_stage9_pairs(pairs, stage3, ModelConfig()).require_clean()

            first_ids = {
                split: [record.pair_id for record in records]
                for split, records in first.split_records.items()
            }
            second_ids = {
                split: [record.pair_id for record in records]
                for split, records in second.split_records.items()
            }
            self.assertEqual(first_ids, second_ids)
            ph_sets = {
                split: {record.ph_key for record in records}
                for split, records in first.split_records.items()
            }
            self.assertTrue(ph_sets["train"].isdisjoint(ph_sets["val"]))
            self.assertTrue(ph_sets["train"].isdisjoint(ph_sets["test"]))
            self.assertTrue(ph_sets["val"].isdisjoint(ph_sets["test"]))
            self.assertTrue(ph_sets["val"])
            self.assertTrue(ph_sets["test"])

    def test_inverse_frequency_weights_give_each_ph_equal_total_weight(self) -> None:
        from source.plm_unified.stage9_data import (
            Stage9PairRecord,
            pair_sampling_weights,
        )

        records = [
            Stage9PairRecord("AAAAAAAA", "Y" * 34, "CASSP", "CASSN", "a", 2),
            Stage9PairRecord("AAAAAAAA", "Y" * 34, "CASSP2", "CASSN2", "b", 3),
            Stage9PairRecord("CCCCCCCC", "A" * 34, "CASSP3", "CASSN3", "c", 4),
        ]
        weights = pair_sampling_weights(records)
        self.assertEqual(weights, [0.5, 0.5, 1.0])

    def test_dataset_fingerprint_changes_when_stage3_overlap_manifest_changes(self) -> None:
        from source.plm_unified.stage9_data import audit_stage9_pairs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            stage3 = root / "stage3.csv"
            rows = [_pair_row(index) for index in range(100)]
            _write_csv(pairs, PAIR_HEADER, rows)
            _write_csv(stage3, ["peptide", "hla", "tcr", "label"], [])
            first = audit_stage9_pairs(pairs, stage3, ModelConfig())
            _write_csv(
                stage3,
                ["peptide", "hla", "tcr", "label"],
                [[rows[0][0], rows[0][1], rows[0][2], 1]],
            )
            second = audit_stage9_pairs(pairs, stage3, ModelConfig())

            self.assertNotEqual(first.dataset_fingerprint, second.dataset_fingerprint)
            self.assertEqual(first.report["pair_csv_sha256"], second.report["pair_csv_sha256"])
            self.assertNotEqual(
                first.report["stage3_train_sha256"],
                second.report["stage3_train_sha256"],
            )


class Stage9ModelAndLossTests(unittest.TestCase):
    def _model_config(self) -> ModelConfig:
        return ModelConfig(
            peptide_input_dim=12,
            hla_input_dim=12,
            tcr_input_dim=10,
            adapter_hidden_dim=16,
            model_dim=8,
            num_attention_heads=2,
            feedforward_dim=16,
            adapter_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
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
                    nn.Linear(hidden_size, hidden_size) for _ in range(3)
                )
                self.calls: list[list[str]] = []
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
                self.calls.append(list(sequences))
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

    def _progressive_model(self):
        from source.plm_unified.stage9_model import (
            ProgressiveStage9FineTuningModel,
            Stage9BindingModel,
        )

        config = self._model_config()
        return ProgressiveStage9FineTuningModel(
            Stage9BindingModel(config, projection_dim=6, projection_dropout=0.0),
            self._tiny_encoder(12, 8, "peptide"),
            self._tiny_encoder(12, 8, "hla"),
            self._tiny_encoder(10, 8, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    def test_pair_forward_returns_normalized_embeddings_and_two_logits(self) -> None:
        from source.plm_unified.stage9_model import Stage9BindingModel

        torch.manual_seed(7)
        model = Stage9BindingModel(
            self._model_config(), projection_dim=6, projection_dropout=0.0
        ).eval()
        batch = 3
        output = model.forward_pair(
            torch.randn(batch, 5, 12),
            torch.ones(batch, 5, dtype=torch.bool),
            torch.randn(batch, 7, 12),
            torch.ones(batch, 7, dtype=torch.bool),
            torch.randn(batch, 6, 10),
            torch.ones(batch, 6, dtype=torch.bool),
            torch.randn(batch, 6, 10),
            torch.ones(batch, 6, dtype=torch.bool),
        )

        self.assertEqual(output.positive_logits.shape, (batch, 2))
        self.assertEqual(output.negative_logits.shape, (batch, 2))
        self.assertEqual(output.z_ph.shape, (batch, 6))
        self.assertTrue(torch.allclose(output.z_ph.norm(dim=1), torch.ones(batch)))
        self.assertTrue(
            torch.allclose(output.z_t_positive.norm(dim=1), torch.ones(batch))
        )
        self.assertTrue(
            torch.allclose(output.z_t_negative.norm(dim=1), torch.ones(batch))
        )
        self.assertTrue(
            torch.allclose(
                output.positive_similarity,
                (output.z_ph * output.z_t_positive).sum(dim=1),
            )
        )

    def test_stage9a_freezes_stage3_and_trains_only_new_contrastive_modules(self) -> None:
        from source.plm_unified.stage9_model import Stage9BindingModel

        model = Stage9BindingModel(self._model_config())
        model.set_trainable_stage9a()
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(
                    (
                        "contrastive_ph_pooling.",
                        "contrastive_tcr_pooling.",
                        "ph_projector.",
                        "tcr_projector.",
                    )
                )
                for name in trainable
            )
        )

    def test_online_pair_forward_encodes_ph_once_and_tcr_as_one_double_batch(self) -> None:
        model = self._progressive_model().eval()
        output = model.forward_pair(
            ["AAAAAAAA", "CCCCCCCC"],
            ["YYYYYYYY", "FFFFFFFF"],
            ["CASSPOS", "CASSPOT"],
            ["CASSNEG", "CASSNET"],
        )

        self.assertEqual(output.positive_logits.shape, (2, 2))
        self.assertEqual(model.peptide_encoder.calls, [["AAAAAAAA", "CCCCCCCC"]])
        self.assertEqual(model.hla_encoder.calls, [["YYYYYYYY", "FFFFFFFF"]])
        self.assertEqual(
            model.tcr_encoder.calls,
            [["CASSPOS", "CASSPOT", "CASSNEG", "CASSNET"]],
        )

    def test_stage9b_trainability_is_task_scoped(self) -> None:
        model = self._progressive_model()
        model.set_trainable_stage9("pair", "stage9b")
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.binding_model.ph_projector.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.binding_model.ptcr_classifier.parameters()
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

        model.set_trainable_stage9("phla", "stage9b")
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.binding_model.phla_classifier.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.binding_model.ph_projector.parameters()
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

    def test_paired_infonce_matches_hand_calculation_and_penalizes_swapping(self) -> None:
        from source.plm_unified.stage9_loss import paired_infonce_loss

        positive = torch.tensor([0.8, 0.2])
        negative = torch.tensor([0.1, 0.4])
        actual = paired_infonce_loss(positive, negative, temperature=0.5)
        expected = torch.nn.functional.softplus((negative - positive) / 0.5).mean()
        swapped = paired_infonce_loss(negative, positive, temperature=0.5)

        self.assertTrue(torch.allclose(actual, expected))
        self.assertLess(float(actual), float(swapped))
        with self.assertRaisesRegex(ValueError, "temperature"):
            paired_infonce_loss(positive, negative, temperature=0.0)

    def test_joint_pair_loss_averages_positive_and_negative_classification(self) -> None:
        from source.plm_unified.stage9_loss import compute_stage9_pair_loss
        from source.plm_unified.stage9_model import Stage9PairOutput

        output = Stage9PairOutput(
            positive_logits=torch.tensor([[0.0, 2.0], [1.0, 0.0]]),
            negative_logits=torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
            z_ph=torch.zeros(2, 2),
            z_t_positive=torch.zeros(2, 2),
            z_t_negative=torch.zeros(2, 2),
            positive_similarity=torch.tensor([0.8, 0.6]),
            negative_similarity=torch.tensor([0.1, 0.2]),
        )
        result = compute_stage9_pair_loss(
            output,
            temperature=0.5,
            contrastive_weight=1.0,
            classification_weight=1.0,
            include_classification=True,
        )
        positive_labels = torch.ones(2, dtype=torch.long)
        negative_labels = torch.zeros(2, dtype=torch.long)
        expected_classification = 0.5 * (
            torch.nn.functional.cross_entropy(output.positive_logits, positive_labels)
            + torch.nn.functional.cross_entropy(output.negative_logits, negative_labels)
        )
        expected_contrastive = torch.nn.functional.softplus(
            (output.negative_similarity - output.positive_similarity) / 0.5
        ).mean()

        self.assertTrue(torch.allclose(result.classification, expected_classification))
        self.assertTrue(torch.allclose(result.contrastive, expected_contrastive))
        self.assertTrue(
            torch.allclose(result.total, expected_contrastive + expected_classification)
        )
        self.assertEqual(result.ranking_correct, 2)


class Stage9CheckpointTests(unittest.TestCase):
    def _config(self, root: Path):
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )

        config = Stage9ExperimentConfig(
            paths=Stage9Paths(root, root, root, root, root / "stage3.pt")
        )
        config.runtime.fold = 4
        config.runtime.seed = 2026
        return config

    def test_stage3_source_requires_deploy_type_matching_fold_and_seed(self) -> None:
        from source.plm_unified.stage9_checkpoint import load_stage3_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            valid = {
                "format_version": 1,
                "checkpoint_type": "stage3_multitask_model",
                "tasks": ["phla", "ptcr", "triplet"],
                "config": {
                    "model": asdict(config.model),
                    "training": {"fold": 4, "seed": 2026},
                },
                "model": {
                    "binding_model.placeholder": torch.tensor(1.0),
                    "peptide_encoder.placeholder": torch.tensor(1.0),
                    "hla_encoder.placeholder": torch.tensor(1.0),
                    "tcr_encoder.placeholder": torch.tensor(1.0),
                },
                "encoder_metadata": {
                    entity: {"checkpoint_fingerprint": f"{entity}-fp"}
                    for entity in ("peptide", "hla", "tcr")
                },
            }
            torch.save(valid, config.paths.stage3_checkpoint)
            source = load_stage3_source(config)
            self.assertEqual(source.checkpoint_type, "stage3_multitask_model")
            self.assertEqual(len(source.sha256), 64)

            invalid = dict(valid)
            invalid["checkpoint_type"] = "stage3_training"
            torch.save(invalid, config.paths.stage3_checkpoint)
            with self.assertRaisesRegex(ValueError, "stage3_multitask_model"):
                load_stage3_source(config)

            invalid = dict(valid)
            invalid["config"] = dict(valid["config"])
            invalid["config"]["training"] = {"fold": 1, "seed": 2026}
            torch.save(invalid, config.paths.stage3_checkpoint)
            with self.assertRaisesRegex(ValueError, "fold"):
                load_stage3_source(config)

            invalid = dict(valid)
            invalid["encoder_metadata"] = {}
            torch.save(invalid, config.paths.stage3_checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_stage3_source(config)

    def test_stage3_initialization_loads_old_weights_and_preserves_new_heads(self) -> None:
        from source.plm_unified.stage9_checkpoint import (
            initialise_from_stage3,
            load_stage3_source,
        )
        from tests.test_plm_stage9 import Stage9ModelAndLossTests

        factory = Stage9ModelAndLossTests()
        source_model = factory._progressive_model()
        target_model = factory._progressive_model()
        with torch.no_grad():
            source_model.binding_model.triplet_classifier.network[-1].weight.fill_(3.0)
        new_head_before = target_model.binding_model.ph_projector.network[0].weight.clone()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config.model = factory._model_config()
            state = {
                name: value
                for name, value in source_model.state_dict().items()
                if not name.startswith(
                    (
                        "binding_model.contrastive_ph_pooling.",
                        "binding_model.contrastive_tcr_pooling.",
                        "binding_model.ph_projector.",
                        "binding_model.tcr_projector.",
                    )
                )
            }
            torch.save(
                {
                    "format_version": 1,
                    "checkpoint_type": "stage3_multitask_model",
                    "tasks": ["phla", "ptcr", "triplet"],
                    "config": {
                        "model": asdict(config.model),
                        "training": {"fold": 4, "seed": 2026},
                    },
                    "model": state,
                    "encoder_metadata": source_model.encoder_metadata(),
                },
                config.paths.stage3_checkpoint,
            )
            source = load_stage3_source(config)
            initialise_from_stage3(target_model, source)

        torch.testing.assert_close(
            target_model.binding_model.triplet_classifier.network[-1].weight,
            source_model.binding_model.triplet_classifier.network[-1].weight,
        )
        torch.testing.assert_close(
            target_model.binding_model.ph_projector.network[0].weight,
            new_head_before,
        )

    def test_deploy_checkpoint_load_is_strict_and_restores_threshold(self) -> None:
        from source.plm_unified.stage9_checkpoint import load_stage9_deploy_checkpoint
        from source.plm_unified.stage9_config import Stage9ExperimentConfig, Stage9Paths

        factory = Stage9ModelAndLossTests()
        source_model = factory._progressive_model()
        target_model = factory._progressive_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Stage9ExperimentConfig(
                paths=Stage9Paths(root, root, root, root, root / "stage3.pt"),
                model=factory._model_config(),
            )
            checkpoint_path = root / "stage9.pt"
            torch.save(
                {
                    "format_version": 1,
                    "checkpoint_type": "stage9_multitask_model",
                    "tasks": ["phla", "ptcr", "triplet", "pair"],
                    "config": config.to_dict(),
                    "model": source_model.state_dict(),
                    "encoder_metadata": source_model.encoder_metadata(),
                    "selected_pair_threshold": 0.42,
                },
                checkpoint_path,
            )

            loaded = load_stage9_deploy_checkpoint(
                target_model, checkpoint_path, config
            )

            self.assertEqual(loaded["selected_pair_threshold"], 0.42)
            torch.testing.assert_close(
                target_model.binding_model.ph_projector.network[0].weight,
                source_model.binding_model.ph_projector.network[0].weight,
            )
            second_target = factory._progressive_model()
            loaded_from_memory = load_stage9_deploy_checkpoint(
                second_target, loaded, config
            )
            self.assertIs(loaded_from_memory, loaded)


class Stage9EvaluationAndTrainerTests(unittest.TestCase):
    def test_pair_metrics_report_explicit_ranking_and_classification(self) -> None:
        from source.plm_unified.stage9_evaluation import compute_stage9_pair_metrics

        result = compute_stage9_pair_metrics(
            positive_similarities=[0.8, 0.2],
            negative_similarities=[0.1, 0.4],
            positive_probabilities=[0.9, 0.7],
            negative_probabilities=[0.1, 0.3],
            contrastive_loss=0.25,
            classification_loss=0.4,
            threshold=0.5,
        )

        self.assertEqual(result.samples, 2)
        self.assertEqual(result.ranking_correct, 1)
        self.assertAlmostEqual(result.ranking_accuracy, 0.5)
        self.assertAlmostEqual(result.mean_similarity_gap, 0.25)
        self.assertAlmostEqual(result.classification.auroc, 1.0)
        self.assertAlmostEqual(result.classification.aupr, 1.0)

    def test_constrained_selection_rejects_forgetting_then_uses_lexicographic_order(self) -> None:
        from source.plm_unified.stage9_trainer import (
            Stage9SelectionMetrics,
            candidate_is_better,
            stage9a_candidate_is_better,
        )

        best = Stage9SelectionMetrics(0.70, 0.20, 0.80, 0.80)
        forgotten = Stage9SelectionMetrics(0.90, 0.50, 0.90, 0.789)
        self.assertFalse(
            candidate_is_better(
                forgotten,
                best,
                source_legacy_joint=0.80,
                degradation_tolerance=0.01,
                min_delta=1e-4,
            )
        )
        higher_ranking = Stage9SelectionMetrics(0.71, 0.10, 0.70, 0.79)
        self.assertTrue(
            candidate_is_better(
                higher_ranking,
                best,
                source_legacy_joint=0.80,
                degradation_tolerance=0.01,
                min_delta=1e-4,
            )
        )
        self.assertTrue(
            stage9a_candidate_is_better(
                contrastive_loss=0.2,
                ranking_accuracy=0.8,
                mean_similarity_gap=0.3,
                best_contrastive_loss=0.2,
                best_ranking_accuracy=0.7,
                best_mean_similarity_gap=0.4,
                min_delta=1e-4,
            )
        )
        self.assertFalse(
            stage9a_candidate_is_better(
                contrastive_loss=0.21,
                ranking_accuracy=1.0,
                mean_similarity_gap=1.0,
                best_contrastive_loss=0.2,
                best_ranking_accuracy=0.7,
                best_mean_similarity_gap=0.4,
                min_delta=1e-4,
            )
        )
        tie_better_gap = Stage9SelectionMetrics(0.70, 0.21, 0.70, 0.80)
        self.assertTrue(
            candidate_is_better(
                tie_better_gap,
                best,
                source_legacy_joint=0.80,
                degradation_tolerance=0.01,
                min_delta=1e-4,
            )
        )

    def test_optimizer_groups_are_phase_specific_and_replay_is_bounded(self) -> None:
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )
        from source.plm_unified.stage9_trainer import (
            bounded_replay_batches,
            build_stage9_optimizer,
            iter_capped_replay_batches,
            replay_sample_cap,
        )

        factory = Stage9ModelAndLossTests()
        model = factory._progressive_model()
        root = Path("/tmp/stage9-trainer")
        config = Stage9ExperimentConfig(
            paths=Stage9Paths(root, root, root, root, root / "stage3.pt"),
            model=factory._model_config(),
        )
        config.stage9.peptide_unfrozen_layers = 2
        config.stage9.hla_unfrozen_layers = 2
        config.stage9.tcr_unfrozen_layers = 2

        stage9a = build_stage9_optimizer(model, config, "stage9a")
        self.assertEqual(
            [group["name"] for group in stage9a.param_groups],
            ["contrastive_modules"],
        )
        stage9b = build_stage9_optimizer(model, config, "stage9b")
        self.assertEqual(
            {group["name"] for group in stage9b.param_groups},
            {
                "adapters",
                "pretrained_modules",
                "contrastive_modules",
            },
        )
        encoder_parameter_ids = {
            id(parameter)
            for encoder in (
                model.peptide_encoder,
                model.hla_encoder,
                model.tcr_encoder,
            )
            for parameter in encoder.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in stage9b.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(encoder_parameter_ids.isdisjoint(optimizer_parameter_ids))
        self.assertEqual(
            bounded_replay_batches(
                pair_samples=100,
                replay_ratio=0.25,
                replay_loader_batches=40,
                replay_batch_size=4,
            ),
            7,
        )
        self.assertEqual(
            bounded_replay_batches(
                pair_samples=1000,
                replay_ratio=0.25,
                replay_loader_batches=10,
                replay_batch_size=4,
            ),
            10,
        )
        self.assertEqual(replay_sample_cap(100, 0.25, 160), 25)
        self.assertEqual(replay_sample_cap(100, 0.0, 160), 0)
        self.assertEqual(
            bounded_replay_batches(
                pair_samples=100,
                replay_ratio=0.0,
                replay_loader_batches=40,
                replay_batch_size=4,
            ),
            0,
        )

        replay_loader = DataLoader(TensorDataset(torch.arange(100)), batch_size=4)
        capped_batches = list(iter_capped_replay_batches(replay_loader, 25))
        self.assertEqual(sum(len(batch[0]) for batch in capped_batches), 25)
        self.assertEqual(len(capped_batches), 7)
        self.assertEqual(len(capped_batches[-1][0]), 1)

    def test_full_two_stage_transition_saves_deploy_and_resume_checkpoints(self) -> None:
        from types import MethodType

        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )
        from source.plm_unified.stage9_evaluation import (
            Stage9PairMetrics,
            Stage9PairPredictionResult,
        )
        from source.plm_unified.stage9_trainer import Stage9Trainer

        factory = Stage9ModelAndLossTests()
        metrics = BinaryMetrics(
            loss=0.2,
            auroc=0.8,
            aupr=0.8,
            accuracy=0.75,
            mcc=0.5,
            f1=0.75,
            precision=0.75,
            recall=0.75,
            sensitivity=0.75,
            specificity=0.75,
            samples=4,
        )
        pair_metrics = Stage9PairMetrics(
            samples=2,
            contrastive_loss=0.2,
            classification_loss=0.3,
            ranking_correct=2,
            ranking_accuracy=1.0,
            mean_positive_similarity=0.8,
            mean_negative_similarity=0.1,
            mean_similarity_gap=0.7,
            median_similarity_gap=0.7,
            classification=metrics,
        )
        pair_result = Stage9PairPredictionResult(
            metrics=pair_metrics,
            pair_ids=["a", "b"],
            source_rows=np.array([2, 3]),
            positive_similarities=np.array([0.8, 0.7]),
            negative_similarities=np.array([0.1, 0.0]),
            positive_probabilities=np.array([0.9, 0.8]),
            negative_probabilities=np.array([0.1, 0.2]),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Stage9ExperimentConfig(
                paths=Stage9Paths(root, root, root / "models", root, root / "stage3.pt"),
                model=factory._model_config(),
            )
            config.runtime.device = "cpu"
            config.runtime.num_workers = 0
            config.stage9.mixed_precision = "none"
            config.stage9.stage9a_epochs = 1
            config.stage9.min_rounds = 1
            config.stage9.max_rounds = 1
            config.stage9.early_stopping_patience = 1
            config.stage9.gradient_accumulation_steps = 1
            config.stage9.peptide_unfrozen_layers = 2
            config.stage9.hla_unfrozen_layers = 2
            config.stage9.tcr_unfrozen_layers = 2
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            legacy = {"phla": loader, "ptcr": loader, "triplet": loader}
            trainer = Stage9Trainer(
                factory._progressive_model(),
                config,
                loader,
                loader,
                legacy,
                legacy,
                source_stage3={"path": "/synthetic/stage3.pt", "sha256": "abc"},
                data_audit={"dataset_fingerprint": "pairs-sha"},
                tracker=None,
                run_timestamp="20260813_120000_000001",
            )

            def fake_train_pair(self, *args, **kwargs):
                return {"total_loss": 0.2, "samples": 4}

            def fake_train_replay(self, *args, **kwargs):
                return {"loss": 0.2, "samples": 4}

            def fake_pair_evaluation(self):
                return pair_result

            def fake_legacy_evaluation(self):
                return ({"phla": metrics, "ptcr": metrics, "triplet": metrics}, 0.8)

            trainer.train_pair_epoch = MethodType(fake_train_pair, trainer)
            trainer.train_replay_epoch = MethodType(fake_train_replay, trainer)
            trainer.evaluate_pairs = MethodType(fake_pair_evaluation, trainer)
            trainer.evaluate_legacy = MethodType(fake_legacy_evaluation, trainer)
            summary = trainer.fit()

            self.assertEqual(summary["phase"], "complete")
            self.assertEqual(summary["stage9a_epoch"], 1)
            self.assertEqual(summary["stage9b_round"], 1)
            self.assertTrue(trainer.stage9a_best_path.is_file())
            self.assertTrue(trainer.stage9b_best_path.is_file())
            self.assertTrue(trainer.last_path.is_file())
            self.assertTrue(trainer.deploy_path.is_file())
            deploy = torch.load(trainer.deploy_path, map_location="cpu", weights_only=False)
            self.assertEqual(deploy["checkpoint_type"], "stage9_multitask_model")
            self.assertEqual(set(deploy["tasks"]), {"phla", "ptcr", "triplet", "pair"})
            self.assertEqual(deploy["selected_pair_threshold"], 0.5)

    def test_resume_rejects_changed_stage9_hyperparameters(self) -> None:
        from source.plm_unified.stage9_config import (
            Stage9ExperimentConfig,
            Stage9Paths,
        )
        from source.plm_unified.stage9_trainer import Stage9Trainer

        factory = Stage9ModelAndLossTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = Stage9Paths(root, root, root / "models", root, root / "stage3.pt")
            config = Stage9ExperimentConfig(paths=paths, model=factory._model_config())
            config.runtime.device = "cpu"
            config.stage9.mixed_precision = "none"
            config.stage9.peptide_unfrozen_layers = 2
            config.stage9.hla_unfrozen_layers = 2
            config.stage9.tcr_unfrozen_layers = 2
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            legacy = {"phla": loader, "ptcr": loader, "triplet": loader}
            trainer = Stage9Trainer(
                factory._progressive_model(), config, loader, loader, legacy, legacy,
                source_stage3={"sha256": "source-sha"},
                data_audit={"dataset_fingerprint": "data-sha"},
                run_timestamp="20260813_120000_000003",
            )
            trainer.save_checkpoint(trainer.last_path, artifact_role="last")

            changed = Stage9ExperimentConfig(paths=paths, model=factory._model_config())
            changed.runtime.device = "cpu"
            changed.stage9.mixed_precision = "none"
            changed.stage9.peptide_unfrozen_layers = 2
            changed.stage9.hla_unfrozen_layers = 2
            changed.stage9.tcr_unfrozen_layers = 2
            changed.stage9.temperature = 0.2
            resumed = Stage9Trainer(
                factory._progressive_model(), changed, loader, loader, legacy, legacy,
                source_stage3={"sha256": "source-sha"},
                data_audit={"dataset_fingerprint": "data-sha"},
                run_timestamp="20260813_120000_000003",
            )
            with self.assertRaisesRegex(ValueError, "training contract"):
                resumed.load_checkpoint(trainer.last_path)

    def test_completed_resume_does_not_run_additional_rounds(self) -> None:
        from types import MethodType

        from source.plm_unified.stage9_config import Stage9ExperimentConfig, Stage9Paths
        from source.plm_unified.stage9_trainer import Stage9Trainer

        factory = Stage9ModelAndLossTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Stage9ExperimentConfig(
                paths=Stage9Paths(root, root, root / "models", root, root / "stage3.pt"),
                model=factory._model_config(),
            )
            config.runtime.device = "cpu"
            config.stage9.mixed_precision = "none"
            config.stage9.max_rounds = 2
            config.stage9.peptide_unfrozen_layers = 2
            config.stage9.hla_unfrozen_layers = 2
            config.stage9.tcr_unfrozen_layers = 2
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            legacy = {"phla": loader, "ptcr": loader, "triplet": loader}

            def make_trainer():
                return Stage9Trainer(
                    factory._progressive_model(), config, loader, loader, legacy, legacy,
                    source_stage3={"sha256": "source-sha"},
                    data_audit={"dataset_fingerprint": "data-sha"},
                    run_timestamp="20260813_120000_000004",
                )

            original = make_trainer()
            original.phase = "complete"
            original.stage9b_round = 1
            original.source_legacy_joint = 0.8
            original.save_checkpoint(original.last_path, artifact_role="last")
            resumed = make_trainer()
            resumed.load_checkpoint(original.last_path)

            def fail_training(self, *args, **kwargs):
                raise AssertionError("completed run attempted additional training")

            resumed.train_pair_epoch = MethodType(fail_training, resumed)
            resumed.train_replay_epoch = MethodType(fail_training, resumed)
            summary = resumed.fit()
            self.assertEqual(summary["phase"], "complete")
            self.assertEqual(summary["stage9b_round"], 1)


if __name__ == "__main__":
    unittest.main()
