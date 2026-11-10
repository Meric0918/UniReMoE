"""Synthetic tests for Stage-3 triplet modelling and data hygiene."""

from __future__ import annotations

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import MethodType

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

if HAS_DEPS:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        HAS_DEPS = False


@unittest.skipUnless(
    HAS_DEPS,
    "PyTorch, NumPy, and scikit-learn are required for triplet tests",
)
class TripletPipelineTests(unittest.TestCase):
    def _config(self, root: Path):
        from source.plm_unified.config import (
            ExperimentConfig,
            ModelConfig,
            PathsConfig,
            SwanLabConfig,
            TrainingConfig,
            TripletConfig,
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
            triplet=TripletConfig(
                micro_batch_size=2,
                eval_batch_size=2,
                gradient_accumulation_steps=2,
                head_warmup_epochs=1,
                min_rounds=1,
                max_rounds=2,
                early_stopping_patience=1,
                binary_replay_ratio=0.25,
                peptide_unfrozen_layers=2,
                hla_unfrozen_layers=2,
                tcr_unfrozen_layers=2,
                mixed_precision="none",
                gradient_checkpointing=False,
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
                    nn.Linear(hidden_size, hidden_size) for _ in range(3)
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
                values = torch.zeros(
                    len(sequences),
                    max_length,
                    hidden_size,
                )
                mask = torch.zeros(
                    len(sequences),
                    max_length,
                    dtype=torch.bool,
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

    def _triplet_model(self, config):
        from source.plm_unified.triplet_model import (
            ProgressiveTripletFineTuningModel,
            TripletBindingModel,
        )

        return ProgressiveTripletFineTuningModel(
            TripletBindingModel(config.model),
            self._tiny_encoder(12, 8, "peptide"),
            self._tiny_encoder(12, 8, "hla"),
            self._tiny_encoder(8, 8, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    def _stage2_model(self, config):
        from source.plm_unified.finetuning import ProgressiveFineTuningModel
        from source.plm_unified.model import UnifiedBindingModel

        return ProgressiveFineTuningModel(
            UnifiedBindingModel(config.model),
            self._tiny_encoder(12, 8, "peptide"),
            self._tiny_encoder(12, 8, "hla"),
            self._tiny_encoder(8, 8, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    def test_strict_split_cleanup_and_manifest_audit(self):
        from source.plm_unified.triplet_data import (
            build_strict_triplet_datasets,
        )

        header = "peptide,tcr,hla,label\n"
        train = (
            "AAAAAAAA,CASSAAA,HHHHHHHH,1\n"
            "AAAAAAAA,CASSAAA,HHHHHHHH,1\n"
            "CCCCCCCC,CASSCCC,GGGGGGGG,0\n"
        )
        val = (
            "AAAAAAAA,CASSAAA,HHHHHHHH,1\n"
            "DDDDDDDD,CASSDDD,FFFFFFFF,0\n"
        )
        test = (
            "DDDDDDDD,CASSDDD,FFFFFFFF,0\n"
            "EEEEEEEE,CASSEEE,YYYYYYYY,1\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data_triple"
            data.mkdir()
            (data / "train_set.csv").write_text(
                header + train,
                encoding="utf-8",
            )
            (data / "val_set.csv").write_text(
                header + val,
                encoding="utf-8",
            )
            (data / "test_set.csv").write_text(
                header + test,
                encoding="utf-8",
            )
            (data / "merged_pseudo.csv").write_text(
                header + train + val + test,
                encoding="utf-8",
            )
            bundle = build_strict_triplet_datasets(
                root,
                data_subdir="data_triple",
                merged_filename="merged_pseudo.csv",
                train_filename="train_set.csv",
                val_filename="val_set.csv",
                test_filename="test_set.csv",
                verify_manifest=True,
                strict_split_cleanup=True,
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            )
            self.assertTrue(bundle.manifest_audit["matches"])
            self.assertEqual(
                (len(bundle.train), len(bundle.val), len(bundle.test)),
                (2, 1, 1),
            )
            self.assertEqual(
                bundle.train.filter_stats["within_split_duplicate"],
                1,
            )
            self.assertEqual(
                bundle.val.filter_stats["higher_priority_overlap"],
                1,
            )
            self.assertEqual(
                bundle.test.filter_stats["higher_priority_overlap"],
                1,
            )

    def test_pht_multitask_dataset_parses_uppercase_hla_and_task_labels(self):
        from source.plm_unified.triplet_data import (
            PHTMultitaskDataset,
            PHTTaskDataset,
            collate_pht_task,
        )

        rows = (
            "peptide,HLA,tcr,PH,PT,PHT,class_code\n"
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,1,1,111\n"
            "CCCCCCCC,GGGGGGGG,CASSCCC,1,0,0,100\n"
            "DDDDDDDD,FFFFFFFF,CASSDDD,0,1,0,010\n"
            "EEEEEEEE,YYYYYYYY,CASSEEE,0,0,0,000\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            path.write_text(rows, encoding="utf-8")
            dataset = PHTMultitaskDataset(
                path,
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            )

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset[0].hla, "HHHHHHHH")
            self.assertEqual(dataset.type_indices["100"], [1])
            ptcr = PHTTaskDataset(dataset, "ptcr")
            batch = collate_pht_task([ptcr[0], ptcr[1]], "ptcr")
            self.assertEqual(batch["peptide_sequences"], ["AAAAAAAA", "CCCCCCCC"])
            self.assertEqual(batch["receptor_sequences"], ["CASSAAA", "CASSCCC"])
            self.assertEqual(batch["labels"].tolist(), [1, 0])

    def test_pht_multitask_dataset_filters_selected_types_before_sequence_checks(self):
        from source.plm_unified.triplet_data import PHTMultitaskDataset

        rows = (
            "peptide,HLA,tcr,PH,PT,PHT\n"
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,1,1\n"
            "BBBBBBBB,IIIIIIII,CASSOOO,1,1,1\n"
            "CCCCCCCC,GGGGGGGG,CASSCCC,1,0,0\n"
            "DDDDDDDD,FFFFFFFF,CASSDDD,0,1,0\n"
            "EEEEEEEE,YYYYYYYY,CASSEEE,0,0,0\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            path.write_text(rows, encoding="utf-8")
            dataset = PHTMultitaskDataset(
                path,
                allowed_label_types=("000", "111"),
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.type_indices["111"], [0])
            self.assertEqual(dataset.type_indices["000"], [1])
            self.assertEqual(dataset.filter_stats["excluded_type_100"], 1)
            self.assertEqual(dataset.filter_stats["excluded_type_010"], 1)
            self.assertEqual(dataset.filter_stats["invalid_noncanonical"], 1)
            self.assertEqual(dataset.filter_stats["invalid_rows"], 1)

    def test_pht_balanced_sampler_keeps_all_111_and_rotates_other_types(self):
        from source.plm_unified.triplet_data import BalancedPHTSampler

        type_indices = {
            "111": [0, 1],
            "100": [2, 3, 4, 5],
            "010": [6, 7, 8, 9],
            "000": [10, 11, 12, 13],
        }
        sampler = BalancedPHTSampler(type_indices, seed=17)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)

        self.assertEqual(len(first), 8)
        self.assertEqual(set(first) & {0, 1}, {0, 1})
        for indices in type_indices.values():
            self.assertEqual(len(set(first) & set(indices)), 2)
        self.assertNotEqual(set(first) & {2, 3, 4, 5}, set(second) & {2, 3, 4, 5})
        duplicate = BalancedPHTSampler(type_indices, seed=17)
        duplicate.set_epoch(1)
        self.assertEqual(second, list(duplicate))

    def test_pht_balanced_sampler_accepts_selected_types(self):
        from source.plm_unified.triplet_data import BalancedPHTSampler

        type_indices = {
            "111": [3, 4],
            "100": [],
            "010": [],
            "000": [0, 1, 2],
        }
        sampler = BalancedPHTSampler(
            type_indices,
            label_types=("000", "111"),
            seed=17,
        )

        sampled = list(sampler)
        self.assertEqual(len(sampled), 4)
        self.assertEqual(len(set(sampled) & {0, 1, 2}), 2)
        self.assertEqual(len(set(sampled) & {3, 4}), 2)

    def test_pht_multitask_dataset_rejects_conflicting_duplicate_triplets(self):
        from source.plm_unified.triplet_data import PHTMultitaskDataset

        rows = (
            "peptide,HLA,tcr,PH,PT,PHT\n"
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,1,1\n"
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,0,0\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            path.write_text(rows, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting PH/PT/PHT labels"):
                PHTMultitaskDataset(
                    path,
                    peptide_min_length=8,
                    peptide_max_length=8,
                    hla_max_length=8,
                    tcr_max_length=8,
                )

    def test_pht_config_rejects_selection_splits_outside_validation(self):
        from source.plm_unified.config import TripletConfig

        values = TripletConfig(
            data_mode="pht_multitask",
            sampling_strategy="balanced_four_type",
            validation_splits=[
                "val_unseen_hla",
                "val_unseen_peptide",
                "val_unseen_tcr",
                "val_unseen_triple",
            ],
            selection_splits=["val_unseen_hla", "missing"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            config.triplet = values
            with self.assertRaisesRegex(ValueError, "selection_splits"):
                config.validate()

    def test_pht_config_accepts_selected_label_types_and_rejects_unknown_types(self):
        from source.plm_unified.config import TripletConfig

        values = TripletConfig(
            data_mode="pht_multitask",
            sampling_strategy="balanced_selected_types",
            pht_label_types=["000", "111"],
            validation_splits=["val"],
            selection_splits=["val"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            config.triplet = values
            config.validate()

            config.triplet.pht_label_types = ["000", "011"]
            with self.assertRaisesRegex(ValueError, "pht_label_types"):
                config.validate()

    def test_pht_class_directory_loader_merges_train_and_val(self):
        from source.plm_unified.triplet_data import build_pht_stage3_loaders

        header = "peptide,HLA,tcr,PH,PT,PHT\n"
        rows = {
            "PHT_000": (
                "AAAAAAAA,HHHHHHHH,CASSAAA,0,0,0\n"
                "CCCCCCCC,GGGGGGGG,CASSCCC,0,0,0\n"
            ),
            "PHT_111": (
                "DDDDDDDD,FFFFFFFF,CASSDDD,1,1,1\n"
                "EEEEEEEE,YYYYYYYY,CASSEEE,1,1,1\n"
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            classes = Path(temporary) / "classes"
            for class_name, content in rows.items():
                class_dir = classes / class_name
                class_dir.mkdir(parents=True)
                (class_dir / "train.csv").write_text(
                    header + content, encoding="utf-8"
                )
                (class_dir / "val.csv").write_text(
                    header + content.replace("A", "N").replace("D", "Q"),
                    encoding="utf-8",
                )

            config = self._config(Path(temporary))
            config.paths.data_root = classes
            config.triplet.data_mode = "pht_multitask"
            config.triplet.data_subdir = "."
            config.triplet.train_filename = "train.csv"
            config.triplet.val_filename = "val.csv"
            config.triplet.pht_class_dirs = ["PHT_000", "PHT_111"]
            config.triplet.validation_splits = ["val"]
            config.triplet.selection_splits = ["val"]
            config.triplet.sampling_strategy = "balanced_selected_types"
            config.triplet.pht_label_types = ["000", "111"]

            train_loaders, val_loaders, audit = build_pht_stage3_loaders(config)

            self.assertEqual(len(train_loaders["triplet"].dataset), 4)
            self.assertEqual(len(val_loaders["val"].dataset), 4)
            self.assertEqual(audit["pht_class_dirs"], ["PHT_000", "PHT_111"])
            self.assertEqual(audit["pht_label_types"], ["000", "111"])

    def test_stage3_joint_score_uses_equal_performance_average(self):
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.triplet_trainer import stage3_joint_score

        def metrics(auroc, aupr, accuracy, mcc, f1):
            return BinaryMetrics(
                loss=0.2,
                auroc=auroc,
                aupr=aupr,
                accuracy=accuracy,
                mcc=mcc,
                f1=f1,
                precision=0.5,
                recall=0.5,
                sensitivity=0.5,
                specificity=0.5,
                samples=4,
            )

        values = {
            "phla": metrics(0.9, 0.8, 0.7, 0.6, 0.5),
            "ptcr": metrics(0.5, 0.4, 0.3, 0.2, 0.1),
            "triplet": metrics(0.8, 0.7, 0.6, 0.5, 0.4),
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            expected = np.mean(
                [value.performance_avg for value in values.values()]
            )
            self.assertAlmostEqual(
                stage3_joint_score(values, config), expected
            )

    def test_stage3_evaluation_uses_strict_half_threshold(self):
        from types import SimpleNamespace

        from source.plm_unified.triplet_evaluation import evaluate_stage3_model

        class EqualProbabilityModel:
            def set_evaluation(self):
                pass

            def forward_triplet(
                self, peptide_sequences, hla_sequences, tcr_sequences
            ):
                return SimpleNamespace(
                    logits=torch.zeros(len(peptide_sequences), 2)
                )

        batch = {
            "labels": torch.tensor([0, 0, 1, 1]),
            "peptide_sequences": ["AAAAAAAA"] * 4,
            "hla_sequences": ["HHHHHHHH"] * 4,
            "tcr_sequences": ["CASSAAA"] * 4,
        }
        class Loader:
            dataset = [0, 1, 2, 3]

            def __iter__(self):
                return iter([batch])

        loader = Loader()

        result = evaluate_stage3_model(
            EqualProbabilityModel(),
            loader,
            "triplet",
            device=torch.device("cpu"),
            mixed_precision="none",
            threshold=0.5,
        )

        self.assertEqual(result.metrics.sensitivity, 0.0)
        self.assertEqual(result.metrics.specificity, 1.0)

    def test_stage3_defaults_match_stage1_early_stopping(self):
        from source.plm_unified.config import TripletConfig

        values = TripletConfig()

        self.assertEqual(values.early_stopping_patience, 5)
        self.assertAlmostEqual(values.early_stopping_min_delta, 0.001)

    def test_stage3_best_round_checkpoint_is_complete_joint_model(self):
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            loaders = {
                task: loader for task in ("phla", "ptcr", "triplet")
            }
            trainer = Stage3Trainer(
                self._triplet_model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                initialization_source={"kind": "stage1"},
                triplet_data_audit={},
                run_timestamp="20260825_120000_000001",
            )

            trainer.best_val_metrics = {"triplet": {"performance_avg": 0.8}}
            trainer.save_best_round_checkpoint(4, 0.8)
            checkpoint = torch.load(
                trainer.best_round_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(
                checkpoint["checkpoint_type"], "stage3_joint_model"
            )
            self.assertEqual(checkpoint["model_scope"], "joint_heads")
            self.assertEqual(checkpoint["best_round"], 4)
            self.assertAlmostEqual(checkpoint["round_score"], 0.8)
            self.assertIn("model", checkpoint)

    def test_triplet_shapes_attention_and_hla_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model = self._triplet_model(config).eval()
            binding = model.binding_model
            output = binding.forward_triplet(
                torch.randn(2, 5, 12),
                torch.ones(2, 5, dtype=torch.bool),
                torch.randn(2, 6, 12),
                torch.ones(2, 6, dtype=torch.bool),
                torch.randn(2, 7, 8),
                torch.ones(2, 7, dtype=torch.bool),
                return_attention=True,
            )
            self.assertEqual(tuple(output.logits.shape), (2, 2))
            self.assertEqual(
                tuple(output.phla_cross_attention.shape),
                (2, 2, 5, 6),
            )
            self.assertEqual(
                tuple(output.triplet_cross_attention.shape),
                (2, 2, 5, 7),
            )

            peptide = torch.randn(1, 5, 12)
            peptide_mask = torch.ones(1, 5, dtype=torch.bool)
            tcr = torch.randn(1, 7, 8)
            tcr_mask = torch.ones(1, 7, dtype=torch.bool)
            hla_mask = torch.ones(1, 6, dtype=torch.bool)
            first = binding.forward_triplet(
                peptide,
                peptide_mask,
                torch.zeros(1, 6, 12),
                hla_mask,
                tcr,
                tcr_mask,
            ).logits
            second = binding.forward_triplet(
                peptide,
                peptide_mask,
                torch.ones(1, 6, 12),
                hla_mask,
                tcr,
                tcr_mask,
            ).logits
            self.assertFalse(torch.allclose(first, second))

    def test_triplet_forward_all_returns_three_task_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._triplet_model(self._config(Path(temporary))).eval()
            output = model.binding_model.forward_all(
                torch.randn(3, 5, 12),
                torch.ones(3, 5, dtype=torch.bool),
                torch.randn(3, 6, 12),
                torch.ones(3, 6, dtype=torch.bool),
                torch.randn(3, 7, 8),
                torch.ones(3, 7, dtype=torch.bool),
            )

            self.assertEqual(tuple(output.phla_logits.shape), (3, 2))
            self.assertEqual(tuple(output.ptcr_logits.shape), (3, 2))
            self.assertEqual(tuple(output.pht_logits.shape), (3, 2))

    def test_pht_multitask_validation_collator_returns_all_labels(self):
        from source.plm_unified.triplet_data import (
            PHTMultitaskRecord,
            collate_pht_multitask,
        )

        records = [
            PHTMultitaskRecord("AAAAAAAA", "CASSAAA", "HHHHHHHH", 1, 1, 1, 2),
            PHTMultitaskRecord("CCCCCCCC", "CASSCCC", "GGGGGGGG", 1, 0, 0, 3),
        ]
        batch = collate_pht_multitask(records)

        self.assertEqual(batch["phla_labels"].tolist(), [1, 1])
        self.assertEqual(batch["ptcr_labels"].tolist(), [1, 0])
        self.assertEqual(batch["pht_labels"].tolist(), [1, 0])
        self.assertEqual(batch["source_rows"].tolist(), [2, 3])

    def test_progressive_triplet_forward_all_encodes_three_inputs_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._triplet_model(self._config(Path(temporary))).eval()
            output = model.forward_all(
                ["AAAAAAAA", "CCCCCCCC"],
                ["HHHHHHHH", "GGGGGGGG"],
                ["CASSAAA", "CASSCCC"],
            )

            self.assertEqual(tuple(output.phla_logits.shape), (2, 2))
            self.assertEqual(tuple(output.ptcr_logits.shape), (2, 2))
            self.assertEqual(tuple(output.pht_logits.shape), (2, 2))

    def test_composite_threshold_uses_mcc_f1_and_balanced_accuracy(self):
        from source.plm_unified.triplet_evaluation import (
            select_composite_threshold,
        )

        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probabilities = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        threshold, metrics, score = select_composite_threshold(
            labels, probabilities
        )

        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(metrics.mcc, 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_multitask_validation_reports_ph_pt_and_pht_metrics(self):
        from source.plm_unified.triplet_data import (
            PHTMultitaskRecord,
            collate_pht_multitask,
        )
        from source.plm_unified.triplet_evaluation import (
            evaluate_stage3_multitask,
        )

        records = [
            PHTMultitaskRecord("AAAAAAAA", "CASSAAA", "HHHHHHHH", 1, 1, 1, 2),
            PHTMultitaskRecord("CCCCCCCC", "CASSCCC", "GGGGGGGG", 1, 0, 0, 3),
            PHTMultitaskRecord("DDDDDDDD", "CASSDDD", "FFFFFFFF", 0, 1, 0, 4),
            PHTMultitaskRecord("EEEEEEEE", "CASSEEE", "YYYYYYYY", 0, 0, 0, 5),
        ]
        loader = DataLoader(records, batch_size=2, collate_fn=collate_pht_multitask)
        with tempfile.TemporaryDirectory() as temporary:
            model = self._triplet_model(self._config(Path(temporary))).eval()
            results = evaluate_stage3_multitask(
                model,
                loader,
                device=torch.device("cpu"),
                mixed_precision="none",
            )

        self.assertEqual(set(results), {"phla", "ptcr", "triplet"})
        self.assertEqual(results["phla"].metrics.samples, 4)
        self.assertEqual(results["ptcr"].labels.tolist(), [1, 0, 1, 0])
        self.assertEqual(results["triplet"].labels.tolist(), [1, 0, 0, 0])

    def test_pht_selection_score_excludes_monitor_only_unseen_triple(self):
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.triplet_trainer import pht_macro_selection_score

        def metrics(value):
            return BinaryMetrics(
                loss=0.1,
                auroc=value,
                aupr=value,
                accuracy=0.5,
                mcc=0.0,
                f1=0.0,
                precision=0.0,
                recall=0.0,
                sensitivity=0.0,
                specificity=1.0,
                samples=4,
            )

        values = {
            split: {
                task: metrics(value)
                for task in ("phla", "ptcr", "triplet")
            }
            for split, value in (
                ("val_unseen_hla", 0.6),
                ("val_unseen_peptide", 0.8),
                ("val_unseen_tcr", 1.0),
                ("val_unseen_triple", 0.0),
            )
        }
        score = pht_macro_selection_score(
            values,
            ["val_unseen_hla", "val_unseen_peptide", "val_unseen_tcr"],
        )

        self.assertAlmostEqual(score, 0.42)

    def test_stage3_trainer_evaluates_all_pht_benchmarks_and_heads(self):
        from functools import partial

        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_data import (
            PHTMultitaskDataset,
            PHTTaskDataset,
            collate_pht_multitask,
            collate_pht_task,
        )
        from source.plm_unified.triplet_trainer import Stage3Trainer

        rows = (
            "peptide,HLA,tcr,PH,PT,PHT\n"
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,1,1\n"
            "CCCCCCCC,GGGGGGGG,CASSCCC,1,0,0\n"
            "DDDDDDDD,FFFFFFFF,CASSDDD,0,1,0\n"
            "EEEEEEEE,YYYYYYYY,CASSEEE,0,0,0\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "pht.csv"
            path.write_text(rows, encoding="utf-8")
            dataset = PHTMultitaskDataset(
                path,
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            )
            train_loaders = {
                task: DataLoader(
                    PHTTaskDataset(dataset, task),
                    batch_size=2,
                    collate_fn=partial(collate_pht_task, task=task),
                )
                for task in ("phla", "ptcr", "triplet")
            }
            split_names = [
                "val_unseen_hla",
                "val_unseen_peptide",
                "val_unseen_tcr",
                "val_unseen_triple",
            ]
            val_loaders = {
                split: DataLoader(
                    dataset, batch_size=2, collate_fn=collate_pht_multitask
                )
                for split in split_names
            }
            config = self._config(root)
            config.triplet.data_mode = "pht_multitask"
            config.triplet.sampling_strategy = "balanced_four_type"
            config.triplet.validation_splits = split_names
            config.triplet.selection_splits = split_names[:3]
            trainer = Stage3Trainer(
                self._triplet_model(config),
                config,
                train_loaders,
                val_loaders,
                SwanLabTracker(config),
                initialization_source={"kind": "stage1", "path": "/stage1.pt"},
                triplet_data_audit={"dataset_fingerprint": "abc"},
                run_timestamp="20260821_120000_000001",
            )

            results, score, threshold, _, serialised = trainer._evaluate_all()

            self.assertEqual(set(results), set(split_names))
            self.assertEqual(set(results[split_names[0]]), {"phla", "ptcr", "triplet"})
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(threshold, 0.0)
            self.assertEqual(serialised["selection_splits"], split_names[:3])

    def test_pht_stage3_fit_uses_multibenchmark_evaluation_in_both_phases(self):
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.trainer import EpochResult
        from source.plm_unified.triplet_evaluation import PredictionResult
        from source.plm_unified.triplet_trainer import Stage3Trainer

        metrics = BinaryMetrics(
            loss=0.2, auroc=0.8, aupr=0.7, accuracy=0.5, mcc=0.0,
            f1=0.0, precision=0.0, recall=0.0, sensitivity=0.0,
            specificity=1.0, samples=4,
        )
        prediction = PredictionResult(
            metrics=metrics,
            labels=np.array([0, 0, 0, 1]),
            probabilities=np.array([0.1, 0.2, 0.3, 0.9]),
        )
        epoch_result = EpochResult(
            metrics=metrics, clean_loss=0.2, adversarial_loss=0.0,
            total_loss=0.2, optimizer_steps=1, samples=4,
            attacked_parameter_count=0.0, fgm_mean_gradient_norm=0.0,
            fgm_max_gradient_norm=0.0, model_gradient_norm=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config.triplet.data_mode = "pht_multitask"
            config.triplet.sampling_strategy = "balanced_four_type"
            config.triplet.validation_splits = ["hla", "peptide", "tcr", "triple"]
            config.triplet.selection_splits = ["hla", "peptide", "tcr"]
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            trainer = Stage3Trainer(
                self._triplet_model(config),
                config,
                {task: loader for task in ("phla", "ptcr", "triplet")},
                {split: loader for split in config.triplet.validation_splits},
                SwanLabTracker(config),
                initialization_source={"kind": "stage1", "path": "/stage1.pt"},
                triplet_data_audit={"dataset_fingerprint": "abc"},
                run_timestamp="20260821_120000_000002",
            )
            nested = {
                split: {task: prediction for task in ("phla", "ptcr", "triplet")}
                for split in config.triplet.validation_splits
            }
            serialised = {
                "selection_splits": config.triplet.selection_splits,
                "selection_score": 0.75,
            }
            calls = []

            def fake_train_epoch(self, *args, **kwargs):
                return epoch_result

            def fake_evaluate_all(self):
                calls.append(self.phase)
                return nested, 0.75, 0.4, metrics, serialised

            trainer.train_epoch = MethodType(fake_train_epoch, trainer)
            trainer._evaluate_all = MethodType(fake_evaluate_all, trainer)
            summary = trainer.fit()

            self.assertEqual(summary["phase"], "complete")
            self.assertGreaterEqual(len(calls), 2)
            self.assertAlmostEqual(summary["best_joint_score"], 0.75)

    def test_build_pht_stage3_loaders_uses_train_and_all_benchmarks(self):
        from source.plm_unified.triplet_data import build_pht_stage3_loaders

        header = "peptide,HLA,tcr,PH,PT,PHT\n"
        train = (
            "AAAAAAAA,HHHHHHHH,CASSAAA,1,1,1\n"
            "CCCCCCCC,GGGGGGGG,CASSCCC,1,0,0\n"
            "DDDDDDDD,FFFFFFFF,CASSDDD,0,1,0\n"
            "EEEEEEEE,YYYYYYYY,CASSEEE,0,0,0\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "PHT"
            benchmarks = data / "benchmarks"
            benchmarks.mkdir(parents=True)
            (data / "train.csv").write_text(header + train, encoding="utf-8")
            split_rows = {
                "val_unseen_hla": train.replace("HHHHHHHH", "IIIIIIII").replace("GGGGGGGG", "KKKKKKKK").replace("FFFFFFFF", "LLLLLLLL").replace("YYYYYYYY", "MMMMMMMM"),
                "val_unseen_peptide": train.replace("AAAAAAAA", "NNNNNNNN").replace("CCCCCCCC", "PPPPPPPP").replace("DDDDDDDD", "QQQQQQQQ").replace("EEEEEEEE", "RRRRRRRR"),
                "val_unseen_tcr": train.replace("CASSAAA", "SSSSSSS").replace("CASSCCC", "TTTTTTT").replace("CASSDDD", "VVVVVVV").replace("CASSEEE", "WWWWWWW"),
                "val_unseen_triple": (
                    train.replace("AAAAAAAA", "NNNNNNNN")
                    .replace("CCCCCCCC", "PPPPPPPP")
                    .replace("DDDDDDDD", "QQQQQQQQ")
                    .replace("EEEEEEEE", "RRRRRRRR")
                    .replace("HHHHHHHH", "IIIIIIII")
                    .replace("GGGGGGGG", "KKKKKKKK")
                    .replace("FFFFFFFF", "LLLLLLLL")
                    .replace("YYYYYYYY", "MMMMMMMM")
                    .replace("CASSAAA", "SSSSSSS")
                    .replace("CASSCCC", "TTTTTTT")
                    .replace("CASSDDD", "VVVVVVV")
                    .replace("CASSEEE", "WWWWWWW")
                ),
            }
            for split, rows in split_rows.items():
                (benchmarks / f"{split}.csv").write_text(
                    header + rows, encoding="utf-8"
                )
            config = self._config(root)
            config.paths.data_root = root
            config.triplet.data_mode = "pht_multitask"
            config.triplet.data_subdir = "PHT"
            config.triplet.train_filename = "train.csv"
            config.triplet.validation_subdir = "benchmarks"
            config.triplet.validation_splits = list(split_rows)
            config.triplet.selection_splits = list(split_rows)[:3]
            config.triplet.sampling_strategy = "balanced_four_type"
            config.triplet.pht_label_types = ["000", "111"]

            train_loaders, val_loaders, audit = build_pht_stage3_loaders(config)

            self.assertEqual(set(train_loaders), {"phla", "ptcr", "triplet"})
            self.assertEqual(set(val_loaders), set(split_rows))
            self.assertEqual(len(train_loaders["triplet"].sampler), 2)
            self.assertEqual(len(val_loaders["val_unseen_hla"].dataset), 2)
            self.assertEqual(audit["pht_label_types"], ["000", "111"])
            self.assertEqual(len(audit["dataset_fingerprint"]), 64)

    def test_stage3_cli_exposes_pht_smoke_and_audit_flags(self):
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, str(root / "source" / "train_plm_triplet.py"), "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--limit-per-type", completed.stdout)
        self.assertIn("--audit-only", completed.stdout)

    def test_pht_resume_rejects_changed_dataset_fingerprint(self):
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config.triplet.data_mode = "pht_multitask"
            config.triplet.sampling_strategy = "balanced_four_type"
            config.triplet.validation_splits = ["hla", "peptide", "tcr", "triple"]
            config.triplet.selection_splits = ["hla", "peptide", "tcr"]
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            train = {task: loader for task in ("phla", "ptcr", "triplet")}
            val = {split: loader for split in config.triplet.validation_splits}
            source = Stage3Trainer(
                self._triplet_model(config), config, train, val,
                SwanLabTracker(config),
                initialization_source={"kind": "stage1"},
                triplet_data_audit={"dataset_fingerprint": "first"},
                run_timestamp="20260821_120000_000003",
            )
            source.save_checkpoint(source.last_path)
            resumed = Stage3Trainer(
                self._triplet_model(config), config, train, val,
                SwanLabTracker(config),
                initialization_source={},
                triplet_data_audit={"dataset_fingerprint": "second"},
                run_timestamp="20260821_120000_000003",
            )

            with self.assertRaisesRegex(ValueError, "dataset fingerprint"):
                resumed.load_checkpoint(source.last_path)

    def test_stage3_sets_balanced_sampler_epoch_before_training(self):
        from torch.utils.data import Sampler

        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        class RecordingSampler(Sampler):
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

            def __iter__(self):
                return iter(range(4))

            def __len__(self):
                return 4

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            loader = DataLoader(
                TensorDataset(torch.arange(4)),
                batch_size=2,
                sampler=RecordingSampler(),
            )
            loaders = {task: loader for task in ("phla", "ptcr", "triplet")}
            trainer = Stage3Trainer(
                self._triplet_model(config), config, loaders, loaders,
                SwanLabTracker(config), initialization_source={},
                triplet_data_audit={}, run_timestamp="20260821_120000_000004",
            )

            trainer._set_train_sampler_epoch("triplet", 7)

            self.assertEqual(loader.sampler.epochs, [7])

    def test_pht_binary_replay_ratio_uses_balanced_epoch_size(self):
        from torch.utils.data import Sampler

        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        class FixedSampler(Sampler):
            def __iter__(self):
                return iter(range(20))

            def __len__(self):
                return 20

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config.triplet.data_mode = "pht_multitask"
            config.triplet.sampling_strategy = "balanced_four_type"
            config.triplet.validation_splits = ["hla"]
            config.triplet.selection_splits = ["hla"]
            dataset = TensorDataset(torch.arange(100))
            loaders = {
                task: DataLoader(
                    dataset, batch_size=2, sampler=FixedSampler()
                )
                for task in ("phla", "ptcr", "triplet")
            }
            trainer = Stage3Trainer(
                self._triplet_model(config), config, loaders,
                {"hla": loaders["triplet"]}, SwanLabTracker(config),
                initialization_source={}, triplet_data_audit={},
                run_timestamp="20260821_120000_000005",
            )

            self.assertEqual(trainer._binary_replay_batches("phla"), 3)

    def test_noncanonical_sequences_are_rejected_and_audited(self):
        from source.plm_unified.triplet_data import (
            build_strict_triplet_datasets,
        )

        header = "peptide,tcr,hla,label\n"
        train = (
            "AAAAAAAA,CASSAAA,HHHHHHHH,1\n"
            "CCCCCCCC,CASSOAA,GGGGGGGG,0\n"
        )
        val = "DDDDDDDD,CASSDDD,FFFFFFFF,0\n"
        test = "EEEEEEEE,CASSEEE,YYYYYYYY,1\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data_triple"
            data.mkdir()
            for name, content in (
                ("train_set.csv", train),
                ("val_set.csv", val),
                ("test_set.csv", test),
                ("merged_pseudo.csv", train + val + test),
            ):
                (data / name).write_text(
                    header + content,
                    encoding="utf-8",
                )

            bundle = build_strict_triplet_datasets(
                root,
                data_subdir="data_triple",
                merged_filename="merged_pseudo.csv",
                train_filename="train_set.csv",
                val_filename="val_set.csv",
                test_filename="test_set.csv",
                verify_manifest=True,
                strict_split_cleanup=True,
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            )

            self.assertEqual(len(bundle.train), 1)
            self.assertEqual(bundle.train.filter_stats["ambiguous_tcr"], 1)
            self.assertEqual(
                bundle.train.filter_stats["noncanonical_rejected"],
                1,
            )

    def test_stage3_trainability_and_ptcr_initialisation(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._triplet_model(self._config(Path(temporary)))
            with torch.no_grad():
                model.binding_model.ptcr_classifier.network[-1].weight.fill_(2.5)
            model.binding_model.initialise_triplet_from_ptcr()
            torch.testing.assert_close(
                model.binding_model.triplet_classifier.network[-1].weight,
                model.binding_model.ptcr_classifier.network[-1].weight,
            )

            model.set_trainable_stage3("triplet", "stage3a")
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in model.binding_model.triplet_classifier.parameters()
                )
            )
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for parameter in model.peptide_encoder.parameters()
                )
            )
            model.set_trainable_stage3("triplet", "stage3b")
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for block in model.hla_encoder.transformer_blocks()[-2:]
                    for parameter in block.parameters()
                )
            )
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for parameter in model.binding_model.ptcr_classifier.parameters()
                )
            )

    def test_synthetic_stage2_checkpoint_initialises_new_modules(self):
        from source.plm_unified.triplet_model import initialise_from_stage2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            source = self._stage2_model(config)
            with torch.no_grad():
                source.binding_model.ptcr_pooling.score.weight.fill_(1.25)
            checkpoint_path = root / "stage2.pt"
            torch.save(
                {
                    "format_version": 1,
                    "checkpoint_type": "stage2_training",
                    "model": source.state_dict(),
                    "config": config.to_dict(),
                    "encoder_metadata": source.encoder_metadata(),
                    "stage": "stage2b",
                    "stage2a_round": 2,
                    "stage2b_round": 1,
                    "global_step": 10,
                    "run_timestamp": "20260726_120000_000001",
                },
                checkpoint_path,
            )
            target = self._triplet_model(config)
            metadata = initialise_from_stage2(
                target,
                checkpoint_path,
                config,
            )
            self.assertEqual(metadata["kind"], "stage2")
            self.assertEqual(metadata["stage"], "stage2b")
            torch.testing.assert_close(
                target.binding_model.triplet_pooling.score.weight,
                target.binding_model.ptcr_pooling.score.weight,
            )
            self.assertTrue(
                torch.all(
                    target.binding_model.triplet_pooling.score.weight == 1.25
                )
            )

    def test_completed_stage1_checkpoint_initialises_binary_and_triplet_modules(self):
        from source.plm_unified.model import UnifiedBindingModel
        from source.plm_unified.triplet_model import initialise_from_stage1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            source = UnifiedBindingModel(config.model)
            with torch.no_grad():
                source.phla_classifier.network[-1].weight.fill_(0.75)
                source.ptcr_pooling.score.weight.fill_(1.5)
            checkpoint_path = root / "stage1_last.pt"
            torch.save(
                {
                    "format_version": 2,
                    "model": source.state_dict(),
                    "config": config.to_dict(),
                    "stage": "complete",
                    "current_round": 4,
                    "global_step": 12,
                    "cache_metadata": {
                        "peptide": {
                            "model": {
                                "checkpoint_fingerprint": "fingerprint-peptide"
                            }
                        },
                        "hla": {
                            "model": {
                                "checkpoint_fingerprint": "fingerprint-hla"
                            }
                        },
                        "tcr": {
                            "model": {
                                "checkpoint_fingerprint": "fingerprint-tcr"
                            }
                        },
                    },
                },
                checkpoint_path,
            )
            target = self._triplet_model(config)
            encoder_before = target.peptide_encoder.stem.weight.detach().clone()
            metadata = initialise_from_stage1(
                target,
                checkpoint_path,
                config,
            )

            self.assertEqual(metadata["kind"], "stage1")
            self.assertEqual(metadata["source_fold"], config.training.fold)
            self.assertEqual(metadata["source_seed"], config.training.seed)
            torch.testing.assert_close(
                target.binding_model.phla_classifier.network[-1].weight,
                source.phla_classifier.network[-1].weight,
            )
            torch.testing.assert_close(
                target.binding_model.triplet_pooling.score.weight,
                source.ptcr_pooling.score.weight,
            )
            torch.testing.assert_close(
                target.peptide_encoder.stem.weight,
                encoder_before,
            )

    def test_stage1_joint_checkpoint_v3_initialises_binary_and_triplet_modules(self):
        from source.plm_unified.model import UnifiedBindingModel
        from source.plm_unified.triplet_model import initialise_from_stage1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            source = UnifiedBindingModel(config.model)
            with torch.no_grad():
                source.phla_classifier.network[-1].weight.fill_(0.85)
                source.ptcr_pooling.score.weight.fill_(1.75)
            checkpoint_path = root / "stage1_best_joint.pt"
            torch.save(
                {
                    "format_version": 3,
                    "checkpoint_type": "stage1_joint_model",
                    "model_scope": "joint_heads",
                    "model": source.state_dict(),
                    "config": config.to_dict(),
                    "stage": "stage1b",
                    "best_round": 10,
                    "round_score": 0.68,
                    "cache_metadata": {
                        entity: {
                            "model": {
                                "checkpoint_fingerprint": f"fingerprint-{entity}"
                            }
                        }
                        for entity in ("peptide", "hla", "tcr")
                    },
                },
                checkpoint_path,
            )
            target = self._triplet_model(config)
            metadata = initialise_from_stage1(
                target,
                checkpoint_path,
                config,
            )

            self.assertEqual(metadata["kind"], "stage1")
            self.assertEqual(metadata["checkpoint_type"], "stage1_joint_model")
            self.assertEqual(metadata["format_version"], 3)
            self.assertEqual(metadata["stage"], "stage1b")
            self.assertEqual(metadata["best_round"], 10)
            torch.testing.assert_close(
                target.binding_model.phla_classifier.network[-1].weight,
                source.phla_classifier.network[-1].weight,
            )
            torch.testing.assert_close(
                target.binding_model.triplet_pooling.score.weight,
                source.ptcr_pooling.score.weight,
            )

    def test_stage1_initialization_rejects_invalid_sources(self):
        from source.plm_unified.model import UnifiedBindingModel
        from source.plm_unified.triplet_model import initialise_from_stage1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            base = {
                "format_version": 2,
                "model": UnifiedBindingModel(config.model).state_dict(),
                "config": config.to_dict(),
                "stage": "complete",
                "cache_metadata": {
                    entity: {
                        "model": {
                            "checkpoint_fingerprint": f"fingerprint-{entity}"
                        }
                    }
                    for entity in ("peptide", "hla", "tcr")
                },
            }
            cases = {
                "task_model": {
                    **base,
                    "checkpoint_type": "task_model",
                },
                "incomplete": {**base, "stage": "stage1b"},
                "seed_mismatch": {
                    **base,
                    "config": {
                        **base["config"],
                        "training": {
                            **base["config"]["training"],
                            "seed": config.training.seed + 1,
                        },
                    },
                },
                "fingerprint_mismatch": {
                    **base,
                    "cache_metadata": {
                        **base["cache_metadata"],
                        "tcr": {
                            "model": {
                                "checkpoint_fingerprint": "wrong-fingerprint"
                            }
                        },
                    },
                },
            }
            for name, state in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.pt"
                    torch.save(state, path)
                    with self.assertRaises(ValueError):
                        initialise_from_stage1(
                            self._triplet_model(config),
                            path,
                            config,
                        )

    def test_threshold_selection_and_joint_score(self):
        from source.plm_unified.triplet_evaluation import select_mcc_threshold
        from source.plm_unified.triplet_trainer import stage3_joint_score

        threshold, metrics = select_mcc_threshold(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.4, 0.35, 0.8]),
        )
        self.assertAlmostEqual(threshold, 0.35)
        self.assertGreater(metrics.mcc, 0)
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            score = stage3_joint_score(
                {
                    "triplet": {
                        "auroc": 0.8,
                        "aupr": 0.6,
                        "accuracy": 0.7,
                        "mcc": 0.5,
                        "f1": 0.4,
                    },
                    "phla": {
                        "auroc": 1.0,
                        "aupr": 0.8,
                        "accuracy": 0.9,
                        "mcc": 0.8,
                        "f1": 0.7,
                    },
                    "ptcr": {
                        "auroc": 0.9,
                        "aupr": 0.7,
                        "accuracy": 0.8,
                        "mcc": 0.7,
                        "f1": 0.6,
                    },
                },
                config,
            )
            self.assertAlmostEqual(score, (0.6 + 0.84 + 0.74) / 3)

    def test_threshold_selection_handles_large_confusion_count_products(self):
        from source.plm_unified.triplet_evaluation import select_mcc_threshold

        class_size = 100_000
        labels = np.concatenate(
            [
                np.ones(class_size, dtype=np.int64),
                np.zeros(class_size, dtype=np.int64),
            ]
        )
        probabilities = np.concatenate(
            [
                np.full(class_size, 0.9, dtype=np.float64),
                np.full(class_size, 0.1, dtype=np.float64),
            ]
        )

        threshold, metrics = select_mcc_threshold(labels, probabilities)

        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(metrics.mcc, 1.0)

    def test_binary_replay_is_capped_at_one_quarter_triplet_samples(self):
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            model = self._triplet_model(config)
            pair = DataLoader(TensorDataset(torch.arange(100)), batch_size=2)
            triplet = DataLoader(TensorDataset(torch.arange(20)), batch_size=2)
            train_loaders = {
                "phla": pair,
                "ptcr": pair,
                "triplet": triplet,
            }
            val_loaders = train_loaders
            trainer = Stage3Trainer(
                model,
                config,
                train_loaders,
                val_loaders,
                SwanLabTracker(config),
                initialization_source={
                    "kind": "stage2",
                    "path": "/synthetic/stage2.pt",
                },
                triplet_data_audit={},
                run_timestamp="20260726_120000_000001",
            )
            self.assertEqual(trainer._binary_replay_batches("phla"), 3)
            self.assertEqual(trainer._binary_replay_batches("ptcr"), 3)

    def test_stage1_lineage_uses_generic_checkpoint_metadata(self):
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.triplet_trainer import Stage3Trainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
            loaders = {
                "phla": loader,
                "ptcr": loader,
                "triplet": loader,
            }
            trainer = Stage3Trainer(
                self._triplet_model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                initialization_source={
                    "kind": "stage1",
                    "path": "/synthetic/stage1_last.pt",
                    "source_fold": 1,
                    "source_seed": 42,
                },
                triplet_data_audit={},
                run_timestamp="20260726_120000_000001",
            )

            training_state = trainer.checkpoint_state()
            self.assertEqual(
                training_state["initialization_source"]["kind"],
                "stage1",
            )
            self.assertNotIn("source_stage2", training_state)
            trainer.save_deploy_checkpoint()
            deploy = torch.load(
                trainer.deploy_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                deploy["initialization_source"]["path"],
                "/synthetic/stage1_last.pt",
            )
            self.assertNotIn("source_stage2", deploy)

    def test_full_stage3_transition_checkpoint_and_resume(self):
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.trainer import EpochResult
        from source.plm_unified.triplet_evaluation import PredictionResult
        from source.plm_unified.triplet_trainer import (
            Stage3Trainer,
            load_stage3_checkpoint,
        )

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
        prediction = PredictionResult(
            metrics=metrics,
            labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
        )
        epoch_result = EpochResult(
            metrics=metrics,
            clean_loss=0.2,
            adversarial_loss=0.0,
            total_loss=0.2,
            optimizer_steps=1,
            samples=4,
            attacked_parameter_count=0.0,
            fgm_mean_gradient_norm=0.0,
            fgm_max_gradient_norm=0.0,
            model_gradient_norm=0.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            loader = DataLoader(
                TensorDataset(torch.arange(8)),
                batch_size=2,
            )
            loaders = {
                "phla": loader,
                "ptcr": loader,
                "triplet": loader,
            }
            trainer = Stage3Trainer(
                self._triplet_model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                initialization_source={
                    "kind": "stage2",
                    "path": "/synthetic/stage2.pt",
                },
                triplet_data_audit={"manifest": {"matches": True}},
                run_timestamp="20260726_120000_000001",
            )

            def fake_train_epoch(self, *args, **kwargs):
                return epoch_result

            def fake_evaluate(self, *args, **kwargs):
                return prediction

            trainer.train_epoch = MethodType(fake_train_epoch, trainer)
            trainer.evaluate = MethodType(fake_evaluate, trainer)
            summary = trainer.fit()
            self.assertEqual(summary["phase"], "complete")
            self.assertEqual(summary["stage3a_epoch"], 1)
            self.assertEqual(summary["stage3b_round"], 1)
            self.assertTrue(trainer.stage3a_best_path.is_file())
            self.assertTrue(trainer.stage3b_best_path.is_file())
            self.assertTrue(trainer.last_path.is_file())
            self.assertTrue(trainer.deploy_path.is_file())

            last = load_stage3_checkpoint(
                trainer.last_path,
                map_location="cpu",
            )
            self.assertEqual(last["phase"], "complete")
            deploy = torch.load(
                trainer.deploy_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                deploy["checkpoint_type"],
                "stage3_multitask_model",
            )
            self.assertEqual(deploy["tasks"], ["phla", "ptcr", "triplet"])

            resumed = Stage3Trainer(
                self._triplet_model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                initialization_source={},
                triplet_data_audit={},
                run_timestamp="20260726_120000_000001",
            )
            resumed.load_checkpoint(trainer.last_path)
            self.assertEqual(resumed.phase, "complete")
            self.assertEqual(resumed.stage3b_round, 1)
            self.assertEqual(
                resumed.initialization_source["path"],
                "/synthetic/stage2.pt",
            )

            legacy_path = root / "legacy_stage3.pt"
            last.pop("initialization_source")
            torch.save(last, legacy_path)
            legacy_resumed = Stage3Trainer(
                self._triplet_model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                initialization_source={},
                triplet_data_audit={},
                run_timestamp="20260726_120000_000001",
            )
            legacy_resumed.load_checkpoint(legacy_path)
            self.assertEqual(
                legacy_resumed.initialization_source["kind"],
                "stage2",
            )


if __name__ == "__main__":
    unittest.main()
