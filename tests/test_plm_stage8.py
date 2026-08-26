"""Contract tests for the Stage-8 constrained four-type pipeline."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

from source.plm_unified.config import ModelConfig
from source.plm_unified.stage8_config import (
    Stage8Condition,
    Stage8ExperimentConfig,
    Stage8Paths,
    load_stage8_config,
)
from source.plm_unified.stage8_data import (
    BalancedFourTypeBatchSampler,
    CachedStage8Dataset,
    Stage8CacheCollator,
    Stage8DataError,
    Stage8Record,
    audit_stage8_data,
    audit_stage8_class_dirs,
    stage8_cache_hidden_dtype,
)
from source.plm_unified import stage8_checkpoint, stage8_data, stage8_model
from source.plm_unified.stage8_checkpoint import (
    Stage3Source, export_stage8_full_checkpoint, extract_stage8_binding_state,
    load_stage3_source, load_torch_checkpoint,
)
from source.plm_unified.stage8_evaluation import (
    evaluate_online_stage8_model,
    stage8_joint_score,
)
from source.plm_unified.stage8_loss import compute_stage8_loss
from source.plm_unified.stage8_model import (
    ProgressiveStage8FineTuningModel,
    Stage8BindingModel,
)
from source.plm_unified.stage8_trainer import (
    Stage8OnlineTrainer,
    Stage8Trainer,
    build_stage8_online_optimizer,
    build_stage8_optimizer,
)
from source.plm_unified.cache import CacheLocator, EmbeddingCacheWriter


HEADER = ["peptide", "tcr", "hla", "PH", "PT", "PHT"]


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


class Stage8ConfigTests(unittest.TestCase):
    def test_defaults_encode_approved_stage8_design(self) -> None:
        root = Path("/tmp/stage8-contract")
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(
                project_root=root,
                data_root=root / "data",
                cache_root=root / "cache",
                output_root=root / "models",
                swanlog_root=root / "logs",
            )
        )
        config.validate()
        self.assertEqual(config.runtime.fold, 1)
        self.assertEqual(config.runtime.seed, 42)
        self.assertEqual(config.stage8.data_mode, "pht_class_dirs")
        self.assertEqual(config.stage8.data_subdir, "PHT/classes")
        self.assertEqual(
            tuple(config.stage8.class_dirs),
            ("PHT_000", "PHT_010", "PHT_100", "PHT_111"),
        )
        self.assertEqual(
            [path.name for path in config.train_paths()],
            ["train.csv"] * 4,
        )
        self.assertEqual(config.stage8.batch_size, 256)
        self.assertEqual(config.stage8.pht_loss_weight, 1.0)
        self.assertEqual(config.stage8.phla_loss_weight, 0.3)
        self.assertEqual(config.stage8.ptcr_loss_weight, 0.2)
        self.assertEqual(config.stage8.constraint_margin, 0.05)
        self.assertEqual(config.stage8.hierarchy_lambda, 0.05)
        self.assertEqual(config.stage8.pt_implication_lambda, 0.03)
        self.assertEqual(config.stage8.constraint_ramp_epochs, 5)
        self.assertEqual(config.stage8.early_stopping_patience, 5)
        self.assertEqual(config.stage8.selection_weights, (0.4, 0.2, 0.4))
        self.assertEqual(config.stage8.conditions, tuple(Stage8Condition))
        self.assertEqual(config.swanlab.mode, "online")

    def test_v2_defaults_pin_the_exact_stage1_source_and_online_schedule(self) -> None:
        root = Path("/tmp/stage8-v2-contract")
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(root, root, root, root, root)
        )

        self.assertTrue(hasattr(config.paths, "stage1_checkpoint"))
        if not hasattr(config.paths, "stage1_checkpoint"):
            return
        self.assertEqual(
            config.paths.stage1_checkpoint,
            Path(
                "trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/"
                "best_joint_20260824_174202_510837.pt"
            ),
        )
        self.assertEqual(
            config.paths.stage1_sha256,
            "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96",
        )
        self.assertEqual(config.stage8.stage8a_epochs, 2)
        self.assertEqual(config.stage8.micro_batch_size, 4)
        self.assertEqual(config.stage8.gradient_accumulation_steps, 16)
        self.assertEqual(config.stage8.binary_replay_ratio, 0.25)
        self.assertEqual(config.stage8.peptide_unfrozen_layers, 2)
        self.assertEqual(config.stage8.hla_unfrozen_layers, 2)
        self.assertEqual(config.stage8.tcr_unfrozen_layers, 2)
        self.assertEqual(config.stage8.min_rounds, 5)
        self.assertEqual(config.stage8.max_rounds, 15)

    def test_batch_size_must_split_equally_across_four_types(self) -> None:
        root = Path("/tmp/stage8-contract")
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(root, root, root, root, root)
        )
        config.stage8.batch_size = 258
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            config.validate()

    def test_v2_binding_schedule_values_cannot_be_overridden(self) -> None:
        root = Path("/tmp/stage8-v2-contract")
        cases = {
            "stage8a_epochs": 3,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "binary_replay_ratio": 0.5,
            "peptide_unfrozen_layers": 1,
            "hla_unfrozen_layers": 1,
            "tcr_unfrozen_layers": 1,
        }
        for field, invalid_value in cases.items():
            with self.subTest(field=field):
                config = Stage8ExperimentConfig(
                    paths=Stage8Paths(root, root, root, root, root)
                )
                setattr(config.stage8, field, invalid_value)
                with self.assertRaisesRegex(ValueError, field):
                    config.validate()

    @staticmethod
    def _required_stage8_files(root: Path) -> None:
        for class_dir in ("PHT_000", "PHT_010", "PHT_100", "PHT_111"):
            directory = root / "PHT" / "classes" / class_dir
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "train.csv").touch()
            (directory / "val.csv").touch()

    def test_require_files_accepts_stage1_source_without_stage3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._required_stage8_files(root)
            stage1 = root / "stage1.pt"
            stage1.touch()
            config = Stage8ExperimentConfig(
                paths=Stage8Paths(
                    root, root, root / "cache", root / "models", root / "logs",
                    stage1_checkpoint=stage1,
                    stage3_checkpoint=root / "missing-stage3.pt",
                )
            )

            try:
                config.validate(require_files=True)
            except FileNotFoundError as exc:
                self.fail(f"Stage8 must not require Stage3: {exc}")

    def test_require_files_rejects_missing_stage1_even_if_stage3_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._required_stage8_files(root)
            stage3 = root / "stage3.pt"
            stage3.touch()
            config = Stage8ExperimentConfig(
                paths=Stage8Paths(
                    root, root, root / "cache", root / "models", root / "logs",
                    stage1_checkpoint=root / "missing-stage1.pt",
                    stage3_checkpoint=stage3,
                )
            )

            with self.assertRaisesRegex(FileNotFoundError, "Stage-1 checkpoint"):
                config.validate(require_files=True)

    def test_repository_stage8_yaml_enables_online_swanlab(self) -> None:
        config = load_stage8_config(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage8.yaml"
        )
        self.assertEqual(config.swanlab.mode, "online")
        self.assertEqual(config.runtime.fold, 1)
        self.assertEqual(config.runtime.seed, 42)
        self.assertEqual(config.stage8.data_mode, "pht_class_dirs")
        self.assertEqual(
            tuple(config.stage8.class_dirs),
            ("PHT_000", "PHT_010", "PHT_100", "PHT_111"),
        )
        self.assertEqual(
            config.paths.stage1_checkpoint.name,
            "best_joint_20260824_174202_510837.pt",
        )
        self.assertEqual(
            config.paths.stage1_sha256,
            "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96",
        )
        self.assertEqual(config.stage8.early_stopping_patience, 5)
        self.assertEqual(config.stage8.selection_weights, (0.4, 0.2, 0.4))

    def test_stage1_sha_is_loaded_as_metadata_not_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "stage8.yaml"
            config_path.write_text(
                "paths:\n"
                "  project_root: .\n"
                "  data_root: data\n"
                "  cache_root: cache\n"
                "  output_root: models\n"
                "  swanlog_root: logs\n"
                "  stage1_sha256: "
                "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96\n"
            )
            config = load_stage8_config(config_path)
            self.assertEqual(
                config.paths.stage1_sha256,
                "5cef56d3f5e392828d23659a53df20f7c8ec0f4678e5d639ec1b798ed02a5f96",
            )

    def test_v1_rejects_non_fold1_seed42_or_incomplete_class_dirs(self) -> None:
        root = Path("/tmp/stage8-contract")
        config = Stage8ExperimentConfig(paths=Stage8Paths(root, root, root, root, root))
        config.runtime.fold = 2
        with self.assertRaisesRegex(ValueError, "fold=1"):
            config.validate()
        config.runtime.fold = 1
        config.runtime.seed = 2026
        with self.assertRaisesRegex(ValueError, "seed=42"):
            config.validate()
        config.runtime.seed = 42
        config.stage8.class_dirs = ["PHT_000", "PHT_111"]
        with self.assertRaisesRegex(ValueError, "four class directories"):
            config.validate()


class Stage8DataTests(unittest.TestCase):
    @staticmethod
    def _write_class_csv(
        path: Path,
        class_code: str,
        rows: list[tuple[str, str, str]],
        split_type: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "peptide", "HLA", "tcr", "PH", "PT", "PHT",
            "class_code", "split_type", "source_file",
        ]
        labels = tuple(int(value) for value in class_code)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for peptide, hla, tcr in rows:
                writer.writerow([
                    peptide, hla, tcr, *labels, class_code,
                    split_type, path.name,
                ])

    @staticmethod
    def _write_four_class_dataset(root: Path) -> None:
        rows = {
            "000": [("AAAAAAAA", "Y" * 34, "CASSAAAA")],
            "010": [("CCCCCCCC", "F" * 34, "CASSCCCC")],
            "100": [("EEEEEEEE", "G" * 34, "CASSEEEE")],
            "111": [("GGGGGGGG", "H" * 34, "CASSGGGG")],
        }
        for class_code, class_rows in rows.items():
            class_dir = root / f"PHT_{class_code}"
            Stage8DataTests._write_class_csv(
                class_dir / "train.csv", class_code, class_rows, "train"
            )
            val_rows = [
                (peptide.replace("A", "T").replace("C", "T").replace("E", "T").replace("G", "T"), hla, tcr)
                for peptide, hla, tcr in class_rows
            ]
            Stage8DataTests._write_class_csv(
                class_dir / "val.csv", class_code, val_rows, "val_unseen_triple"
            )

    def test_audit_stage8_class_dirs_normalises_hla_and_merges_four_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            result = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            result.require_clean()
            self.assertEqual(len(result.train_records), 4)
            self.assertEqual(len(result.val_records), 4)
            self.assertEqual(
                result.report["label_type_counts"]["train"],
                {"111": 1, "100": 1, "010": 1, "000": 1},
            )
            self.assertEqual(
                result.report["source_files"]["train"],
                {"PHT_000": 1, "PHT_010": 1, "PHT_100": 1, "PHT_111": 1},
            )
            self.assertEqual(result.train_records[0].hla, "Y" * 34)

    def test_audit_stage8_class_dirs_rejects_class_code_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            mismatch = root / "PHT_010" / "train.csv"
            rows = [("CCCCCCCC", "F" * 34, "CASSCCCC")]
            self._write_class_csv(mismatch, "111", rows, "train")
            result = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            self.assertTrue(result.has_errors)
            self.assertTrue(any("class_code" in item["reason"] for item in result.errors))

    def test_audit_stage8_class_dirs_detects_cross_directory_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            shared = [("AAAAAAAA", "Y" * 34, "CASSAAAA")]
            self._write_class_csv(root / "PHT_111" / "train.csv", "111", shared, "train")
            result = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            self.assertTrue(result.has_conflicts)
            with self.assertRaises(Stage8DataError):
                result.require_clean()

    def test_audit_stage8_class_dirs_removes_same_label_train_val_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            shared = [("AAAAAAAA", "Y" * 34, "CASSAAAA")]
            val_rows = [
                ("TTTTTTTT", "Y" * 34, "CASSAAAA"),
                *shared,
            ]
            self._write_class_csv(
                root / "PHT_000" / "val.csv", "000", val_rows, "val"
            )
            result = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            result.require_clean()
            self.assertEqual(result.report["validation"]["train_overlap_removed"], 1)
            self.assertEqual(len(result.val_records), 4)

    def test_audit_deduplicates_same_label_and_removes_val_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            val = root / "val.csv"
            shared = ["AAAAAAAA", "CASSLGQETQYF", "Y" * 34, 1, 1, 1]
            _write_csv(
                train,
                [
                    shared,
                    shared,
                    ["CCCCCCCC", "CASSIRSSYEQYF", "A" * 34, 1, 0, 0],
                    ["EEEEEEEE", "CASSQETQYF", "C" * 34, 0, 1, 0],
                    ["FFFFFFFF", "CASSPGQETQYF", "D" * 34, 0, 0, 0],
                ],
            )
            _write_csv(
                val,
                [
                    shared,
                    ["DDDDDDDD", "CASSQETQYF", "C" * 34, 0, 1, 0],
                    ["GGGGGGGG", "CASSLGQETQYF", "E" * 34, 1, 1, 1],
                    ["HHHHHHHH", "CASSIRSSYEQYF", "F" * 34, 1, 0, 0],
                    ["IIIIIIII", "CASSIRSSYEQYF", "G" * 34, 0, 0, 0],
                ],
            )
            result = audit_stage8_data(train, val, ModelConfig())
            result.require_clean()
            self.assertEqual(len(result.train_records), 4)
            self.assertEqual(len(result.val_records), 4)
            self.assertEqual(result.report["train"]["same_label_duplicates"], 1)
            self.assertEqual(result.report["validation"]["train_overlap_removed"], 1)
            self.assertTrue(result.dataset_fingerprint)

    def test_conflicting_labels_are_reported_and_stop_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            val = root / "val.csv"
            key = ["AAAAAAAA", "CASSLGQETQYF", "Y" * 34]
            _write_csv(train, [key + [1, 1, 1], key + [1, 0, 0]])
            _write_csv(val, [["CCCCCCCC", "CASSIRSSYEQYF", "A" * 34, 0, 0, 0]])
            result = audit_stage8_data(train, val, ModelConfig())
            self.assertTrue(result.has_conflicts)
            self.assertEqual(len(result.conflicts), 1)
            with self.assertRaises(Stage8DataError):
                result.require_clean()

    def test_only_four_logically_supported_label_types_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            val = root / "val.csv"
            _write_csv(train, [["AAAAAAAA", "CASSLGQETQYF", "Y" * 34, 0, 0, 1]])
            _write_csv(val, [["CCCCCCCC", "CASSIRSSYEQYF", "A" * 34, 0, 0, 0]])
            result = audit_stage8_data(train, val, ModelConfig())
            self.assertTrue(result.has_errors)
            with self.assertRaisesRegex(Stage8DataError, "manual repair"):
                result.require_clean()

    def test_balanced_sampler_is_deterministic_and_oversamples_minor_types(self) -> None:
        type_indices = {"111": [0, 1, 2], "100": [3], "010": [4, 5], "000": [6]}
        sampler = BalancedFourTypeBatchSampler(type_indices, batch_size=8, seed=17)
        first = list(iter(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(iter(sampler)))
        self.assertEqual(len(first), 2)
        index_to_type = {
            index: label_type
            for label_type, indices in type_indices.items()
            for index in indices
        }
        for batch in first:
            self.assertEqual(Counter(index_to_type[index] for index in batch), {
                "111": 2, "100": 2, "010": 2, "000": 2,
            })

    def test_online_loaders_keep_ph_pt_replay_and_four_type_joint_batches(self) -> None:
        make_loaders = getattr(stage8_data, "make_stage8_online_loaders", None)
        self.assertIsNotNone(make_loaders)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            audit = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            phla_path = root / "phla.csv"
            ptcr_path = root / "ptcr.csv"
            with phla_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["peptide", "HLA", "label"])
                writer.writerows([["AAAAAAAA", "Y" * 34, 1], ["CCCCCCCC", "F" * 34, 0]])
            with ptcr_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["peptide", "tcr", "label"])
                writer.writerows([["AAAAAAAA", "CASSAAAA", 1], ["CCCCCCCC", "CASSCCCC", 0]])

            train, validation = make_loaders(
                audit,
                phla_path=phla_path,
                ptcr_path=ptcr_path,
                model=ModelConfig(),
                micro_batch_size=4,
                eval_batch_size=4,
                seed=17,
                num_workers=0,
                limit_per_type=1,
            )
            self.assertEqual(set(train), {"phla", "ptcr", "joint"})
            self.assertEqual(next(iter(train["phla"]))["labels"].tolist(), [0, 1])
            self.assertEqual(next(iter(train["ptcr"]))["labels"].tolist(), [1, 0])
            self.assertEqual(
                Counter(next(iter(train["joint"]))["label_types"]),
                {"111": 1, "100": 1, "010": 1, "000": 1},
            )
            self.assertEqual(len(next(iter(validation))["label_types"]), 4)

    def test_online_loader_requires_full_audit_before_applying_debug_limit(self) -> None:
        make_loaders = getattr(stage8_data, "make_stage8_online_loaders", None)
        self.assertIsNotNone(make_loaders)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_class_dataset(root)
            audit = audit_stage8_class_dirs(
                root,
                ["PHT_000", "PHT_010", "PHT_100", "PHT_111"],
                "train.csv",
                "val.csv",
                ModelConfig(),
            )
            audit.errors.append({"reason": "required label type 010 is absent"})
            with self.assertRaisesRegex(Stage8DataError, "manual repair"):
                make_loaders(
                    audit,
                    phla_path=root / "missing-phla.csv",
                    ptcr_path=root / "missing-ptcr.csv",
                    model=ModelConfig(),
                    micro_batch_size=4,
                    eval_batch_size=4,
                    seed=17,
                    num_workers=0,
                    limit_per_type=1,
                )


class Stage8ModelAndLossTests(unittest.TestCase):
    @staticmethod
    def _model_config() -> ModelConfig:
        return ModelConfig(
            peptide_input_dim=8,
            hla_input_dim=8,
            tcr_input_dim=6,
            adapter_hidden_dim=8,
            model_dim=8,
            num_attention_heads=2,
            feedforward_dim=16,
            adapter_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
            peptide_min_length=1,
            peptide_max_length=5,
            hla_max_length=7,
            tcr_max_length=6,
        )

    def test_forward_all_returns_three_heads_and_reuses_phla_attention(self) -> None:
        torch.manual_seed(7)
        model = Stage8BindingModel(self._model_config()).eval()
        calls = 0

        def count_call(_module, _args, _output):
            nonlocal calls
            calls += 1

        hook = model.phla_cross_attention.register_forward_hook(count_call)
        output = model.forward_all(
            torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool),
            torch.randn(3, 7, 8), torch.ones(3, 7, dtype=torch.bool),
            torch.randn(3, 6, 6), torch.ones(3, 6, dtype=torch.bool),
        )
        hook.remove()
        self.assertEqual(calls, 1)
        self.assertEqual(output.phla_logits.shape, (3, 2))
        self.assertEqual(output.ptcr_logits.shape, (3, 2))
        self.assertEqual(output.pht_logits.shape, (3, 2))

    @staticmethod
    def _tiny_encoder(hidden_size: int, name: str):
        import torch.nn as nn
        from source.plm_unified.encoders import ResidueBatch

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.checkpoint_fingerprint = f"fingerprint-{name}"
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
                    "checkpoint_fingerprint": self.checkpoint_fingerprint,
                    "hidden_size": self.hidden_size,
                }

            def forward(self, sequences):
                hidden = torch.zeros(len(sequences), 5, hidden_size)
                mask = torch.ones(len(sequences), 5, dtype=torch.bool)
                lengths = torch.full((len(sequences),), 5, dtype=torch.long)
                for block in self.blocks:
                    hidden = block(hidden)
                return ResidueBatch(hidden, mask, lengths)

        return TinyEncoder()

    def test_online_model_copies_ptcr_modules_and_switches_phase_trainability(self) -> None:
        progressive_type = getattr(
            stage8_model, "ProgressiveStage8FineTuningModel", None
        )
        self.assertIsNotNone(progressive_type)
        binding = Stage8BindingModel(self._model_config())
        with torch.no_grad():
            binding.ptcr_classifier.network[-1].weight.fill_(3.0)
        model = progressive_type(
            binding,
            self._tiny_encoder(8, "peptide"),
            self._tiny_encoder(8, "hla"),
            self._tiny_encoder(6, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

        model.initialise_triplet_from_ptcr()
        torch.testing.assert_close(
            model.binding_model.triplet_classifier.network[-1].weight,
            model.binding_model.ptcr_classifier.network[-1].weight,
        )
        model.set_trainable_stage8("joint", "stage8a")
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.binding_model.triplet_classifier.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.peptide_encoder.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.binding_model.ptcr_classifier.parameters()
        ))
        model.set_trainable_stage8("joint", "stage8b")
        self.assertTrue(all(
            parameter.requires_grad
            for encoder in (model.peptide_encoder, model.hla_encoder, model.tcr_encoder)
            for block in encoder.transformer_blocks()[-2:]
            for parameter in block.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.binding_model.phla_classifier.parameters()
        ))

    def test_stage1_initialisation_exactly_copies_all_ptcr_modules(self) -> None:
        initialise = getattr(stage8_model, "initialise_stage8_from_stage1", None)
        self.assertIsNotNone(initialise)
        from source.plm_unified.model import UnifiedBindingModel

        model_config = self._model_config()
        source = UnifiedBindingModel(model_config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage1_best_joint.pt"
            checkpoint = {
                "format_version": 3,
                "checkpoint_type": "stage1_joint_model",
                "stage": "stage1b",
                "model_scope": "joint_heads",
                "model": source.state_dict(),
                "config": {
                    "model": model_config.__dict__,
                    "training": {"fold": 1, "seed": 42},
                },
                "cache_metadata": {
                    entity: {"model": {"checkpoint_fingerprint": f"fingerprint-{entity}"}}
                    for entity in ("peptide", "hla", "tcr")
                },
            }
            torch.save(checkpoint, path)
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            target = SimpleNamespace(
                binding_model=Stage8BindingModel(model_config),
                encoder_metadata=lambda: {
                    entity: {"checkpoint_fingerprint": f"fingerprint-{entity}"}
                    for entity in ("peptide", "hla", "tcr")
                },
            )
            metadata = initialise(
                target,
                path,
                model_config=model_config,
                fold=1,
                seed=42,
                expected_sha256=expected_sha256,
            )
            self.assertEqual(metadata["kind"], "stage1")
            self.assertEqual(metadata["sha256"], expected_sha256)
            for triplet_name, ptcr_name in (
                ("triplet_cross_attention", "ptcr_cross_attention"),
                ("triplet_pooling", "ptcr_pooling"),
                ("triplet_classifier", "ptcr_classifier"),
            ):
                triplet_state = getattr(target.binding_model, triplet_name).state_dict()
                ptcr_state = getattr(source, ptcr_name).state_dict()
                self.assertEqual(set(triplet_state), set(ptcr_state))
                for name, value in ptcr_state.items():
                    torch.testing.assert_close(triplet_state[name], value)

    def test_stage1_initialisation_requires_exact_sha(self) -> None:
        initialise = getattr(stage8_model, "initialise_stage8_from_stage1", None)
        self.assertIsNotNone(initialise)
        from source.plm_unified.model import UnifiedBindingModel

        model_config = self._model_config()
        source = UnifiedBindingModel(model_config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage1_best_joint.pt"
            checkpoint = {
                "format_version": 3,
                "checkpoint_type": "stage1_joint_model",
                "stage": "stage1b",
                "model_scope": "joint_heads",
                "model": source.state_dict(),
                "config": {
                    "model": model_config.__dict__,
                    "training": {"fold": 1, "seed": 42},
                },
                "cache_metadata": {
                    entity: {"model": {"checkpoint_fingerprint": f"fingerprint-{entity}"}}
                    for entity in ("peptide", "hla", "tcr")
                },
            }
            torch.save(checkpoint, path)
            target = SimpleNamespace(
                binding_model=Stage8BindingModel(model_config),
                encoder_metadata=lambda: {
                    entity: {"checkpoint_fingerprint": f"fingerprint-{entity}"}
                    for entity in ("peptide", "hla", "tcr")
                },
            )

            with self.assertRaisesRegex(ValueError, "expected SHA256 is required"):
                initialise(
                    target,
                    path,
                    model_config=model_config,
                    fold=1,
                    seed=42,
                    expected_sha256=None,
                )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                initialise(
                    target, path, model_config=model_config, fold=1, seed=42,
                    expected_sha256="0" * 64,
                )
            checkpoint["config"] = {
                **checkpoint["config"], "training": {"fold": 2, "seed": 42}
            }
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "fold"):
                initialise(
                    target,
                    path,
                    model_config=model_config,
                    fold=1,
                    seed=42,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_stage8_initialisation_errors_use_stage8_terminology(self) -> None:
        model_config = self._model_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.pt"
            torch.save({
                "format_version": 3,
                "checkpoint_type": "task_model",
            }, path)
            target = SimpleNamespace(
                binding_model=Stage8BindingModel(model_config),
                encoder_metadata=lambda: {},
            )

            with self.assertRaises(ValueError) as raised:
                stage8_model.initialise_stage8_from_stage1(
                    target,
                    path,
                    model_config=model_config,
                    fold=1,
                    seed=42,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )

            message = str(raised.exception)
            self.assertIn("Stage-8", message)
            self.assertNotIn("Stage 3", message)
            self.assertNotIn("Stage-3", message)

    def test_condition_switches_and_linear_ramp(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]] * 2, requires_grad=True)
        labels = torch.tensor([1, 0, 1, 0])
        label_types = ["111", "000", "100", "010"]
        baseline = compute_stage8_loss(
            logits, logits, logits, labels, labels, labels, label_types,
            condition=Stage8Condition.BASELINE, epoch=3,
        )
        full = compute_stage8_loss(
            logits, logits, logits, labels, labels, labels, label_types,
            condition=Stage8Condition.FULL, epoch=3,
        )
        self.assertEqual(baseline.hierarchy_lambda, 0.0)
        self.assertEqual(baseline.pt_implication_lambda, 0.0)
        self.assertAlmostEqual(full.hierarchy_lambda, 0.03)
        self.assertAlmostEqual(full.pt_implication_lambda, 0.018)

    def test_positive_hierarchy_constraint_does_not_backpropagate_into_pht(self) -> None:
        phla = torch.tensor([[4.0, 0.0]], requires_grad=True)
        ptcr = torch.tensor([[0.0, 4.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 4.0]], requires_grad=True)
        labels = torch.tensor([1])
        result = compute_stage8_loss(
            phla, ptcr, pht, labels, labels, labels, ["111"],
            condition=Stage8Condition.HIER_ONLY, epoch=5,
            supervised_weight=0.0,
        )
        result.total.backward()
        self.assertGreater(float(phla.grad.abs().sum()), 0.0)
        self.assertEqual(float(pht.grad.abs().sum()), 0.0)

    def test_negative_hierarchy_constraint_does_not_backpropagate_into_ph(self) -> None:
        phla = torch.tensor([[4.0, 0.0]], requires_grad=True)
        ptcr = torch.tensor([[4.0, 0.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 4.0]], requires_grad=True)
        labels = torch.tensor([0])
        result = compute_stage8_loss(
            phla, ptcr, pht, labels, labels, labels, ["010"],
            condition=Stage8Condition.HIER_ONLY, epoch=5,
            supervised_weight=0.0,
        )
        result.total.backward()
        self.assertEqual(float(phla.grad.abs().sum()), 0.0)
        self.assertGreater(float(pht.grad.abs().sum()), 0.0)

    def test_pt_constraints_use_the_same_directed_stop_gradient(self) -> None:
        labels = torch.tensor([1])
        phla = torch.tensor([[0.0, 4.0]], requires_grad=True)
        ptcr = torch.tensor([[4.0, 0.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 4.0]], requires_grad=True)
        positive = compute_stage8_loss(
            phla, ptcr, pht, labels, labels, labels, ["111"],
            condition=Stage8Condition.PT_IMP_ONLY, epoch=5,
            supervised_weight=0.0,
        )
        positive.total.backward()
        self.assertGreater(float(ptcr.grad.abs().sum()), 0.0)
        self.assertEqual(float(pht.grad.abs().sum()), 0.0)

        phla = torch.tensor([[4.0, 0.0]], requires_grad=True)
        ptcr = torch.tensor([[4.0, 0.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 4.0]], requires_grad=True)
        zero = torch.tensor([0])
        negative = compute_stage8_loss(
            phla, ptcr, pht, zero, zero, zero, ["100"],
            condition=Stage8Condition.PT_IMP_ONLY, epoch=5,
            supervised_weight=0.0,
        )
        negative.total.backward()
        self.assertEqual(float(ptcr.grad.abs().sum()), 0.0)
        self.assertGreater(float(pht.grad.abs().sum()), 0.0)

    def test_pht_supervision_balances_positive_and_combined_negatives(self) -> None:
        phla = torch.zeros(4, 2, requires_grad=True)
        ptcr = torch.zeros(4, 2, requires_grad=True)
        pht = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
            requires_grad=True,
        )
        labels = torch.tensor([1, 0, 0, 0])
        result = compute_stage8_loss(
            phla, ptcr, pht,
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            labels,
            ["111", "000", "100", "010"],
            condition=Stage8Condition.BASELINE,
            epoch=0,
        )
        per_row = torch.nn.functional.cross_entropy(pht, labels, reduction="none")
        expected = 0.5 * per_row[0] + 0.5 * per_row[1:].mean()
        self.assertTrue(torch.allclose(result.pht, expected))

    def test_optimizer_has_approved_three_learning_rate_groups(self) -> None:
        model = Stage8BindingModel(self._model_config())
        root = Path("/tmp/stage8-contract")
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(root, root, root, root, root), model=self._model_config()
        )
        optimizer = build_stage8_optimizer(model, config)
        self.assertEqual(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            {
                "adapters": 2.5e-5,
                "pretrained_attention": 5e-5,
                "triplet_modules": 1e-4,
            },
        )
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(set(parameter_ids), {id(parameter) for parameter in model.parameters()})

    def test_joint_score_uses_only_predictive_metrics(self) -> None:
        metrics = {
            "pht": {
                "auroc": 0.8, "accuracy": 0.7, "mcc": 0.6,
                "f1": 0.5, "aupr": 0.4,
            },
            "phla": {
                "auroc": 0.7, "accuracy": 0.6, "mcc": 0.5,
                "f1": 0.4, "aupr": 0.3,
            },
            "ptcr": {
                "auroc": 0.9, "accuracy": 0.8, "mcc": 0.7,
                "f1": 0.6, "aupr": 0.5,
            },
            "violations": {"overall": 1.0},
        }
        self.assertAlmostEqual(stage8_joint_score(metrics), 0.62)


class _OnlineStage8Batches:
    """Small concrete iterable with DataLoader-like sample metadata."""

    def __init__(self, batches: list[dict], samples: int, batch_size: int = 4):
        self.batches = batches
        self.dataset = list(range(samples))
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class Stage8OnlineTrainingTests(unittest.TestCase):
    @staticmethod
    def _model_config() -> ModelConfig:
        return Stage8ModelAndLossTests._model_config()

    @staticmethod
    def _tiny_encoder(hidden_size: int, name: str):
        return Stage8ModelAndLossTests._tiny_encoder(hidden_size, name)

    def _model(self) -> ProgressiveStage8FineTuningModel:
        return ProgressiveStage8FineTuningModel(
            Stage8BindingModel(self._model_config()),
            self._tiny_encoder(8, "peptide"),
            self._tiny_encoder(8, "hla"),
            self._tiny_encoder(6, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    def _static_model(self) -> ProgressiveStage8FineTuningModel:
        """Concrete encoder fixture whose trainable blocks cannot alter outputs."""

        import torch.nn as nn
        from source.plm_unified.encoders import ResidueBatch

        class StaticEncoder(nn.Module):
            def __init__(self, hidden_size: int, name: str):
                super().__init__()
                self.hidden_size = hidden_size
                self.checkpoint_fingerprint = f"static-{name}"
                self.blocks = nn.ModuleList(
                    nn.Linear(hidden_size, hidden_size) for _ in range(2)
                )
                self.requires_grad_(False)

            def transformer_blocks(self):
                return self.blocks

            def set_active_training(self, active: bool):
                self.train(active)

            def forward(self, sequences):
                return ResidueBatch(
                    torch.zeros(len(sequences), 5, self.hidden_size),
                    torch.ones(len(sequences), 5, dtype=torch.bool),
                    torch.full((len(sequences),), 5, dtype=torch.long),
                )

        return ProgressiveStage8FineTuningModel(
            Stage8BindingModel(self._model_config()),
            StaticEncoder(8, "peptide"),
            StaticEncoder(8, "hla"),
            StaticEncoder(6, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    def _config(self, root: Path) -> Stage8ExperimentConfig:
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(root, root, root, root, root),
            model=self._model_config(),
        )
        config.stage8.mixed_precision = "none"
        return config

    @staticmethod
    def _joint_batch() -> dict:
        return {
            "peptide_sequences": ["PEP0", "PEP1", "PEP2", "PEP3"],
            "hla_sequences": ["HLA0", "HLA1", "HLA2", "HLA3"],
            "tcr_sequences": ["TCR0", "TCR1", "TCR2", "TCR3"],
            "ph_labels": torch.tensor([1, 1, 0, 0]),
            "pt_labels": torch.tensor([1, 0, 1, 0]),
            "pht_labels": torch.tensor([1, 0, 0, 0]),
            "label_types": ["111", "100", "010", "000"],
            "source_rows": torch.tensor([0, 1, 2, 3]),
        }

    @staticmethod
    def _replay_batch() -> dict:
        return {
            "peptide_sequences": ["PEP0", "PEP1", "PEP2", "PEP3"],
            "receptor_sequences": ["REC0", "REC1", "REC2", "REC3"],
            "labels": torch.tensor([1, 0, 1, 0]),
        }

    def test_stage8a_constraint_gate_ignores_full_condition(self) -> None:
        logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])

        loss = compute_stage8_loss(
            logits, logits, logits, labels, labels, labels, ["111", "000"],
            condition=Stage8Condition.FULL,
            epoch=5,
            phase="stage8a",
        )

        self.assertEqual(loss.hierarchy_lambda, 0.0)
        self.assertEqual(loss.pt_implication_lambda, 0.0)
        self.assertTrue(torch.equal(loss.total, loss.supervised))

    def test_online_trainer_rejects_every_mutable_binding_schedule_value(self) -> None:
        cases = {
            "stage8a_epochs": 3,
            "binary_replay_ratio": 0.5,
            "peptide_unfrozen_layers": 1,
            "hla_unfrozen_layers": 1,
            "tcr_unfrozen_layers": 1,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "min_rounds": 4,
            "max_rounds": 14,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 1e-3,
            "constraint_ramp_epochs": 4,
            "pht_score_weight": 0.5,
            "phla_score_weight": 0.3,
            "ptcr_score_weight": 0.3,
        }
        joint = _OnlineStage8Batches([self._joint_batch()], samples=4)
        replay = _OnlineStage8Batches([self._replay_batch()], samples=4)
        with tempfile.TemporaryDirectory() as temporary:
            for field, invalid_value in cases.items():
                with self.subTest(field=field):
                    config = self._config(Path(temporary))
                    setattr(config.stage8, field, invalid_value)
                    with self.assertRaisesRegex(ValueError, f"stage8.{field}"):
                        Stage8OnlineTrainer(
                            self._model(),
                            config,
                            {"phla": replay, "ptcr": replay, "joint": joint},
                            joint,
                            condition=Stage8Condition.FULL,
                        )

    def test_stage8b_stops_after_five_non_improving_rounds(self) -> None:
        joint = _OnlineStage8Batches([self._joint_batch()], samples=4)
        replay = _OnlineStage8Batches([self._replay_batch()], samples=4)
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            config.stage8.adapter_lr = 0.0
            config.stage8.pretrained_attention_lr = 0.0
            config.stage8.triplet_module_lr = 0.0
            trainer = Stage8OnlineTrainer(
                self._static_model(),
                config,
                {"phla": replay, "ptcr": replay, "joint": joint},
                joint,
                condition=Stage8Condition.FULL,
            )
            trainer.fit_stage8a()
            trainer.fit_stage8b()

        self.assertEqual(trainer.stage8b_round, 5)
        self.assertEqual(trainer.bad_rounds, 5)
        self.assertEqual(
            trainer.pass_history[-15:],
            [
                ("stage8b", round_index, task)
                for round_index in range(1, 6)
                for task in ("phla", "ptcr", "joint")
            ],
        )

    def test_stage8b_round_index_ramps_constraints_through_round_five(self) -> None:
        logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        weights = []
        for round_index in range(1, 7):
            loss = compute_stage8_loss(
                logits, logits, logits, labels, labels, labels, ["111", "000"],
                condition=Stage8Condition.FULL,
                epoch=0,
                phase="stage8b",
                round_index=round_index,
            )
            weights.append((loss.hierarchy_lambda, loss.pt_implication_lambda))

        expected = [
            (0.01, 0.006), (0.02, 0.012), (0.03, 0.018), (0.04, 0.024),
            (0.05, 0.03), (0.05, 0.03),
        ]
        for actual, expected_weight in zip(weights, expected):
            self.assertAlmostEqual(actual[0], expected_weight[0])
            self.assertAlmostEqual(actual[1], expected_weight[1])

    def test_online_optimizer_tracks_stage_trainability(self) -> None:
        model = self._model()
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model.set_trainable_stage8("joint", "stage8a")
            warmup = build_stage8_online_optimizer(model, config, "stage8a")
            warmup_ids = {
                id(parameter)
                for group in warmup.param_groups
                for parameter in group["params"]
            }
            self.assertEqual(
                warmup_ids,
                {
                    id(parameter)
                    for module in model.binding_model.new_triplet_modules().values()
                    for parameter in module.parameters()
                },
            )

            model.set_trainable_stage8("joint", "stage8b")
            online = build_stage8_online_optimizer(model, config, "stage8b")
            online_ids = {
                id(parameter)
                for group in online.param_groups
                for parameter in group["params"]
            }
            top_ids = {
                id(parameter)
                for encoder in (model.peptide_encoder, model.hla_encoder, model.tcr_encoder)
                for block in encoder.transformer_blocks()[-2:]
                for parameter in block.parameters()
            }
            self.assertTrue(top_ids <= online_ids)
            self.assertEqual(
                online_ids,
                {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
            )

    def test_stage8b_runs_capped_ph_then_pt_then_joint_passes(self) -> None:
        model = self._model()
        joint = _OnlineStage8Batches([self._joint_batch()], samples=4)
        replay = _OnlineStage8Batches([self._replay_batch()], samples=4)
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage8OnlineTrainer(
                model,
                self._config(Path(temporary)),
                {"phla": replay, "ptcr": replay, "joint": joint},
                joint,
                condition=Stage8Condition.FULL,
            )
            trainer.train_stage8a_epoch(1)
            trainer.train_stage8a_epoch(2)
            result = trainer.train_stage8b_round(1)

        self.assertEqual(trainer.pass_history[-3:], [
            ("stage8b", 1, "phla"),
            ("stage8b", 1, "ptcr"),
            ("stage8b", 1, "joint"),
        ])
        self.assertEqual(result["phla"]["samples"], 1)
        self.assertEqual(result["ptcr"]["samples"], 1)
        self.assertEqual(result["joint"]["samples"], 4)
        self.assertAlmostEqual(result["joint"]["hierarchy_lambda"], 0.01)
        self.assertAlmostEqual(result["joint"]["pt_implication_lambda"], 0.006)

    def test_online_evaluator_uses_sequences_and_preserves_stage8_score(self) -> None:
        model = self._model()
        joint = _OnlineStage8Batches([self._joint_batch()], samples=4)
        evaluation = evaluate_online_stage8_model(
            model,
            joint,
            device=torch.device("cpu"),
            mixed_precision="none",
        )

        self.assertEqual(evaluation.predictions["pht_labels"].tolist(), [1, 0, 0, 0])
        self.assertEqual(set(evaluation.metrics), {
            "phla", "ptcr", "pht", "pht_selected", "violations",
            "constraint_components", "by_type",
        })
        self.assertAlmostEqual(
            evaluation.joint_score,
            stage8_joint_score(evaluation.metrics),
        )


class _FakeCache:
    def __init__(self, width: int, dim: int):
        self.width = width
        self.dim = dim

    def resolve(self, sequence: str) -> CacheLocator:
        return CacheLocator(0, len(sequence), min(len(sequence), self.width))

    def get_many(self, locators):
        import numpy as np

        hidden = np.zeros((len(locators), self.width, self.dim), dtype=np.float16)
        mask = np.zeros((len(locators), self.width), dtype=np.bool_)
        for index, locator in enumerate(locators):
            hidden[index, :, :] = locator.row
            mask[index, : locator.length] = True
        return hidden, mask


class Stage8CacheAndCheckpointTests(unittest.TestCase):
    def test_cache_hidden_dtype_requires_active_cuda_autocast(self) -> None:
        self.assertEqual(
            stage8_cache_hidden_dtype(torch.device("cuda"), "none"),
            torch.float32,
        )
        self.assertEqual(
            stage8_cache_hidden_dtype(torch.device("cuda"), "bf16"),
            torch.float16,
        )
        self.assertEqual(
            stage8_cache_hidden_dtype(torch.device("cpu"), "bf16"),
            torch.float32,
        )
    def test_stage8_cache_writer_supports_row_sized_random_access_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with EmbeddingCacheWriter(
                root, "peptide", hidden_dim=2, max_length=3,
                model_metadata={"checkpoint_fingerprint": "sha"},
                shard_size=8, chunk_rows=1,
            ) as writer:
                writer.add_batch(
                    ["AAA", "CCC"],
                    np.zeros((2, 3, 2), dtype=np.float16),
                    np.ones((2, 3), dtype=np.uint8),
                    np.asarray([3, 3], dtype=np.int16),
                )
            import h5py

            with h5py.File(root / "peptide/shards/part-00000.h5", "r") as handle:
                self.assertEqual(handle["hidden"].chunks[0], 1)
            import json

            manifest = json.loads((root / "peptide/manifest.json").read_text())
            self.assertEqual(manifest["chunk_rows"], 1)

    def test_cached_dataset_and_collator_keep_labels_types_and_rows(self) -> None:
        records = [
            Stage8Record("AAAAAAAA", "CCCC", "D" * 10, 1, 1, 1, 12),
            Stage8Record("EEEEEEEE", "FFFF", "G" * 10, 1, 0, 0, 21),
        ]
        caches = {
            "peptide": _FakeCache(8, 3),
            "hla": _FakeCache(10, 3),
            "tcr": _FakeCache(6, 2),
        }
        dataset = CachedStage8Dataset(records, caches)
        batch = Stage8CacheCollator(caches)([dataset[0], dataset[1]])
        self.assertEqual(batch["peptide_hidden"].shape, (2, 8, 3))
        self.assertEqual(batch["hla_hidden"].shape, (2, 10, 3))
        self.assertEqual(batch["tcr_hidden"].shape, (2, 6, 2))
        self.assertEqual(batch["ph_labels"].tolist(), [1, 1])
        self.assertEqual(batch["pt_labels"].tolist(), [1, 0])
        self.assertEqual(batch["pht_labels"].tolist(), [1, 0])
        self.assertEqual(batch["label_types"], ["111", "100"])
        self.assertEqual(batch["source_rows"].tolist(), [12, 21])

    def test_stage3_source_validation_extracts_only_binding_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_config = Stage8ModelAndLossTests._model_config()
            binding = Stage8BindingModel(model_config)
            path = root / "source.pt"
            state = {f"binding_model.{name}": value for name, value in binding.state_dict().items()}
            state.update({
                "peptide_encoder.dummy": torch.tensor(1),
                "hla_encoder.dummy": torch.tensor(1),
                "tcr_encoder.dummy": torch.tensor(1),
            })
            torch.save({
                "format_version": 1,
                "checkpoint_type": "stage3_multitask_model",
                "tasks": ["phla", "ptcr", "triplet"],
                "config": {"model": model_config.__dict__, "training": {"fold": 4, "seed": 2026}},
                "initialization_source": {"kind": "stage1", "source_fold": 4, "source_seed": 2026},
                "model": state,
            }, path)
            config = Stage8ExperimentConfig(
                paths=Stage8Paths(root, root, root, root, root, stage3_checkpoint=path),
                model=model_config,
            )
            config.runtime.fold = 4
            config.runtime.seed = 2026
            source = load_stage3_source(config)
            self.assertEqual(set(source.binding_state), set(binding.state_dict()))
            self.assertEqual(len(source.sha256), 64)
            self.assertEqual(source.lineage["kind"], "stage3")

    def test_stage3_joint_checkpoint_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_config = Stage8ModelAndLossTests._model_config()
            binding = Stage8BindingModel(model_config)
            path = root / "stage3_joint.pt"
            state = {
                f"binding_model.{name}": value
                for name, value in binding.state_dict().items()
            }
            state.update({
                "peptide_encoder.dummy": torch.tensor(1),
                "hla_encoder.dummy": torch.tensor(1),
                "tcr_encoder.dummy": torch.tensor(1),
            })
            torch.save({
                "format_version": 1,
                "checkpoint_type": "stage3_joint_model",
                "config": {
                    "model": model_config.__dict__,
                    "training": {"fold": 1, "seed": 42},
                },
                "initialization_source": {
                    "kind": "stage1", "source_fold": 1, "source_seed": 42,
                },
                "model": state,
            }, path)
            config = Stage8ExperimentConfig(
                paths=Stage8Paths(
                    root, root, root, root, root,
                    stage3_checkpoint=path,
                ),
                model=model_config,
            )
            config.runtime.fold = 1
            config.runtime.seed = 42
            source = load_stage3_source(config)
            self.assertEqual(source.checkpoint["checkpoint_type"], "stage3_joint_model")
            self.assertEqual(set(source.binding_state), set(binding.state_dict()))

    def test_one_epoch_cpu_training_writes_lean_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_config = Stage8ModelAndLossTests._model_config()
            config = Stage8ExperimentConfig(
                paths=Stage8Paths(root, root, root, root / "models", root / "logs"),
                model=model_config,
            )
            config.stage8.batch_size = 4
            config.stage8.eval_batch_size = 4
            config.stage8.max_epochs = 1
            config.stage8.min_epochs = 1
            config.stage8.early_stopping_patience = 1
            config.stage8.mixed_precision = "none"
            config.validate()
            torch.manual_seed(11)
            batch = {
                "peptide_hidden": torch.randn(4, 5, 8),
                "peptide_mask": torch.ones(4, 5, dtype=torch.bool),
                "hla_hidden": torch.randn(4, 7, 8),
                "hla_mask": torch.ones(4, 7, dtype=torch.bool),
                "tcr_hidden": torch.randn(4, 6, 6),
                "tcr_mask": torch.ones(4, 6, dtype=torch.bool),
                "ph_labels": torch.tensor([1, 1, 0, 0]),
                "pt_labels": torch.tensor([1, 0, 1, 0]),
                "pht_labels": torch.tensor([1, 0, 0, 0]),
                "label_types": ["111", "100", "010", "000"],
                "source_rows": torch.tensor([2, 3, 4, 5]),
            }
            source = Stage3Source(
                root / "stage3.pt", "a" * 64, {}, {},
                {"kind": "stage3", "sha256": "a" * 64},
            )
            initial_model = Stage8BindingModel(model_config)
            source_binding_state = deepcopy(initial_model.state_dict())
            trainer = Stage8Trainer(
                initial_model, config, [batch], [batch],
                condition=Stage8Condition.FULL, source=source,
                data_audit={"dataset_fingerprint": "data-sha"},
                run_timestamp="20260809_000000_000001",
            )
            summary = trainer.fit()
            self.assertTrue(Path(summary["best_checkpoint"]).is_file())
            self.assertTrue(Path(summary["last_checkpoint"]).is_file())
            self.assertTrue(trainer.summary_path.is_file())
            self.assertTrue(trainer.prediction_path.is_file())
            checkpoint = load_torch_checkpoint(summary["best_checkpoint"], mmap=False)
            self.assertEqual(checkpoint["checkpoint_type"], "stage8_binding_model")
            self.assertFalse(any(name.startswith("peptide_encoder.") for name in checkpoint["model"]))
            self.assertEqual(checkpoint["source_stage3"]["sha256"], "a" * 64)
            self.assertIsInstance(checkpoint["best_model"], dict)

            source.checkpoint = {
                "model": {
                    "peptide_encoder.weight": torch.tensor([1.0]),
                    "binding_model.old": torch.tensor([0.0]),
                },
                "initialization_source": {"kind": "stage1"},
            }
            full_path = root / "full.pt"
            export_stage8_full_checkpoint(source, checkpoint, full_path)
            full = load_torch_checkpoint(full_path, mmap=False)
            self.assertEqual(full["checkpoint_type"], "stage8_multitask_model")
            self.assertIn("peptide_encoder.weight", full["model"])
            self.assertNotIn("binding_model.old", full["model"])
            self.assertTrue(any(name.startswith("binding_model.peptide_adapter") for name in full["model"]))
            self.assertEqual(full["initialization_source"]["kind"], "stage3")
            exported_binding = extract_stage8_binding_state(full)
            self.assertEqual(set(exported_binding), set(checkpoint["model"]))
            for name, tensor in checkpoint["model"].items():
                self.assertTrue(torch.equal(exported_binding[name], tensor))

            # A copied last checkpoint must be self-contained even if companion
            # best/prediction artifacts have disappeared.
            trainer.best_path.unlink()
            trainer.prediction_path.unlink()
            resume_model = Stage8BindingModel(model_config)
            resume_model.load_state_dict(source_binding_state)
            resumed = Stage8Trainer(
                resume_model, config, [batch], [batch],
                condition=Stage8Condition.FULL, source=source,
                data_audit={"dataset_fingerprint": "data-sha"},
                run_timestamp="20260809_000000_000001",
            )
            resumed.fit(resume=trainer.last_path)
            self.assertTrue(resumed.best_path.is_file())
            self.assertTrue(resumed.prediction_path.is_file())
            with self.assertRaisesRegex(ValueError, "requires.*last checkpoint"):
                resumed.resume(resumed.best_path)

            changed_config = deepcopy(config)
            changed_config.stage8.hierarchy_lambda = 0.06
            changed = Stage8Trainer(
                Stage8BindingModel(model_config), changed_config, [batch], [batch],
                condition=Stage8Condition.FULL, source=source,
                data_audit={"dataset_fingerprint": "data-sha"},
                run_timestamp="20260809_000000_000001",
            )
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                changed.resume(trainer.last_path)


class Stage8V2CheckpointTests(unittest.TestCase):
    TIMESTAMP = "20260826_120000_000001"

    @staticmethod
    def _model() -> ProgressiveStage8FineTuningModel:
        return ProgressiveStage8FineTuningModel(
            Stage8BindingModel(Stage8ModelAndLossTests._model_config()),
            Stage8ModelAndLossTests._tiny_encoder(8, "peptide"),
            Stage8ModelAndLossTests._tiny_encoder(8, "hla"),
            Stage8ModelAndLossTests._tiny_encoder(6, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=2,
            tcr_unfrozen_layers=2,
        )

    @staticmethod
    def _config(root: Path) -> Stage8ExperimentConfig:
        config = Stage8ExperimentConfig(
            paths=Stage8Paths(
                root, root, root / "cache", root / "models", root / "logs",
                stage1_checkpoint=root / "stage1.pt",
                stage1_sha256="a" * 64,
            ),
            model=Stage8ModelAndLossTests._model_config(),
        )
        config.stage8.mixed_precision = "none"
        return config

    @staticmethod
    def _metrics(score: float = 0.75) -> dict:
        return {
            "validation_metrics": {"pht": {"auroc": score}},
            "validation_joint_score": score,
            "selected_pht_threshold": 0.5,
        }

    @staticmethod
    def _source() -> dict:
        return {
            "kind": "stage1",
            "path": "/tmp/stage1.pt",
            "sha256": "a" * 64,
            "source_fold": 1,
            "source_seed": 42,
        }

    @staticmethod
    def _effective_data(limit_per_type=None) -> dict:
        return {
            "limit_per_type": limit_per_type,
            "train_samples": 4,
            "validation_samples": 4,
        }

    @staticmethod
    def _trainer(
        model: ProgressiveStage8FineTuningModel,
        config: Stage8ExperimentConfig,
        condition: Stage8Condition = Stage8Condition.FULL,
    ) -> Stage8OnlineTrainer:
        joint_batch = Stage8OnlineTrainingTests._joint_batch()
        replay_batch = Stage8OnlineTrainingTests._replay_batch()
        joint = _OnlineStage8Batches([joint_batch], samples=4)
        replay = _OnlineStage8Batches([replay_batch], samples=4)
        trainer = Stage8OnlineTrainer(
            model,
            config,
            {"phla": replay, "ptcr": replay, "joint": joint},
            joint,
            condition=condition,
        )
        trainer.stage8a_epoch = config.stage8.stage8a_epochs
        trainer.begin_stage8b()
        trainer.stage8b_round = 3
        trainer.global_step = 17
        trainer.bad_rounds = 2
        trainer.best_joint = 0.75
        return trainer

    def _write_run(self, root: Path):
        save_warmup = getattr(
            stage8_checkpoint, "save_stage8_warmup_checkpoint", None
        )
        save_best = getattr(
            stage8_checkpoint, "save_stage8_best_checkpoint", None
        )
        save_last = getattr(
            stage8_checkpoint, "save_stage8_last_checkpoint", None
        )
        self.assertIsNotNone(save_warmup)
        self.assertIsNotNone(save_best)
        self.assertIsNotNone(save_last)
        if save_warmup is None or save_best is None or save_last is None:
            raise AssertionError("Stage8 V2 checkpoint writers are unavailable")
        config = self._config(root)
        model = self._model()
        warmup = save_warmup(
            root / "warmup",
            model=model,
            config=config,
            source_stage1=self._source(),
            dataset_fingerprint="data-sha",
            effective_data=self._effective_data(),
            run_timestamp=self.TIMESTAMP,
            metrics=self._metrics(0.70),
        )
        trainer = self._trainer(model, config)
        run_dir = root / "full"
        best = save_best(
            run_dir,
            trainer=trainer,
            source_stage1=self._source(),
            warmup_checkpoint=warmup,
            dataset_fingerprint="data-sha",
            effective_data=self._effective_data(),
            run_timestamp=self.TIMESTAMP,
            metrics=self._metrics(),
        )
        last = save_last(
            run_dir,
            trainer=trainer,
            source_stage1=self._source(),
            warmup_checkpoint=warmup,
            dataset_fingerprint="data-sha",
            effective_data=self._effective_data(),
            run_timestamp=self.TIMESTAMP,
            metrics=self._metrics(0.73),
            best_checkpoint=best,
        )
        return config, model, trainer, warmup, best, last

    def test_v2_round_trip_contains_full_online_model_and_role_specific_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, model, _, warmup_path, best_path, last_path = self._write_run(root)
            warmup = load_torch_checkpoint(warmup_path, mmap=False)
            best = load_torch_checkpoint(best_path, mmap=False)
            last = load_torch_checkpoint(last_path, mmap=False)

            self.assertEqual(
                warmup_path.name,
                f"stage8a_best_pht_{self.TIMESTAMP}.pt",
            )
            self.assertEqual(best_path.name, f"stage8_best_{self.TIMESTAMP}.pt")
            self.assertEqual(last_path.name, f"stage8_last_{self.TIMESTAMP}.pt")
            for checkpoint in (warmup, best, last):
                self.assertEqual(checkpoint["format_version"], 2)
                self.assertEqual(checkpoint["checkpoint_type"], "stage8_online_model")
                self.assertEqual(checkpoint["source_stage1"]["sha256"], "a" * 64)
                self.assertEqual(checkpoint["dataset_fingerprint"], "data-sha")
                self.assertIn("training_contract", checkpoint)
                self.assertIn("metrics", checkpoint)
                for prefix in (
                    "peptide_encoder.", "hla_encoder.", "tcr_encoder.",
                    "binding_model.",
                ):
                    self.assertTrue(any(name.startswith(prefix) for name in checkpoint["model"]))
            for checkpoint in (warmup, best):
                self.assertNotIn("optimizer", checkpoint)
                self.assertNotIn("scheduler", checkpoint)
                self.assertNotIn("scaler", checkpoint)
                self.assertNotIn("rng_state", checkpoint)
            for key in ("optimizer", "scheduler", "scaler", "rng_state"):
                self.assertIn(key, last)
            self.assertNotIn("best_model", last)
            self.assertEqual(best["condition"], "full")
            self.assertEqual(
                best["source_stage8a"]["sha256"],
                hashlib.sha256(warmup_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                last["best_companion"]["sha256"],
                hashlib.sha256(best_path.read_bytes()).hexdigest(),
            )

            restored = self._model()
            restored.load_state_dict(best["model"], strict=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)
            extracted = extract_stage8_binding_state(best)
            self.assertEqual(
                set(extracted), set(model.binding_model.state_dict())
            )
            for name, value in model.binding_model.state_dict().items():
                torch.testing.assert_close(extracted[name], value)

    def test_public_v2_loader_rejects_v1_stage8_checkpoints_clearly(self) -> None:
        load_v2 = getattr(stage8_checkpoint, "load_stage8_v2_checkpoint", None)
        self.assertIsNotNone(load_v2)
        if load_v2 is None:
            return
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage8_best_old.pt"
            torch.save({
                "format_version": 1,
                "checkpoint_type": "stage8_binding_model",
                "model": {"weight": torch.ones(1)},
            }, path)
            with self.assertRaisesRegex(ValueError, "V1.*not supported"):
                load_v2(path)

    def test_v2_contract_records_effective_training_data_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, _, _, last_path = self._write_run(Path(temporary))
            last = load_torch_checkpoint(last_path, mmap=False)

            self.assertEqual(last.get("effective_data"), {
                "limit_per_type": None,
                "train_samples": 4,
                "validation_samples": 4,
            })
            self.assertEqual(
                last["training_contract"]["descriptor"].get("effective_data"),
                {
                    "limit_per_type": None,
                    "train_samples": 4,
                    "validation_samples": 4,
                },
            )

    def test_v2_validator_enforces_role_specific_metadata(self) -> None:
        validate = stage8_checkpoint.validate_stage8_v2_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, warmup_path, best_path, last_path = self._write_run(
                Path(temporary)
            )
            warmup = load_torch_checkpoint(warmup_path, mmap=False)
            best = load_torch_checkpoint(best_path, mmap=False)
            last = load_torch_checkpoint(last_path, mmap=False)
            cases = (
                ("warmup-phase", warmup, "phase", "stage8b", "stage8a"),
                ("warmup-condition", warmup, "condition", "baseline", "condition"),
                (
                    "warmup-source",
                    warmup,
                    "source_stage8a",
                    dict(best["source_stage8a"]),
                    "warmup source",
                ),
                ("warmup-optimizer", warmup, "optimizer", {}, "optimizer"),
                ("best-phase", best, "phase", "stage8a", "stage8b"),
                ("best-condition", best, "condition", "unknown", "condition"),
                ("best-source", best, "source_stage8a", None, "warmup source"),
                ("best-metrics", best, "metrics", {}, "metrics"),
                ("last-optimizer", last, "optimizer", None, "optimizer"),
                ("last-scheduler", last, "scheduler", None, "scheduler"),
                ("last-scaler", last, "scaler", None, "scaler"),
                ("last-rng", last, "rng_state", None, "rng_state"),
                ("last-best-model", last, "best_model", {}, "best_model"),
            )
            for name, original, key, value, message in cases:
                with self.subTest(name=name):
                    malformed = deepcopy(original)
                    if value is None:
                        malformed.pop(key, None)
                    else:
                        malformed[key] = value
                    with self.assertRaisesRegex(ValueError, message):
                        validate(malformed)

    def test_public_evaluation_validation_rejects_malformed_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _, _, best_path, _ = self._write_run(Path(temporary))
            best = load_torch_checkpoint(best_path, mmap=False)
            malformed_sources = (
                {**best["source_stage8a"], "sha256": "not-a-sha"},
                {**best["source_stage8a"], "path": None},
            )
            for source in malformed_sources:
                with self.subTest(source=source):
                    malformed = deepcopy(best)
                    malformed["source_stage8a"] = source
                    with self.assertRaisesRegex(ValueError, "warmup source"):
                        stage8_checkpoint.validate_stage8_v2_for_evaluation(
                            malformed, config=config
                        )

    def test_resume_restores_last_state_and_requires_exact_best_companion(self) -> None:
        resume = getattr(stage8_checkpoint, "resume_stage8_last_checkpoint", None)
        self.assertIsNotNone(resume)
        if resume is None:
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, saved_model, saved_trainer, _, _, last = self._write_run(root)
            restored_model = self._model()
            restored_trainer = self._trainer(restored_model, config)
            restored_trainer.stage8b_round = 0
            restored_trainer.global_step = 0
            restored_trainer.bad_rounds = 0
            restored_trainer.best_joint = -float("inf")

            checkpoint = resume(
                last,
                trainer=restored_trainer,
                config=config,
                source_stage1_sha256="a" * 64,
                dataset_fingerprint="data-sha",
                effective_data=self._effective_data(),
                condition=Stage8Condition.FULL,
            )

            self.assertEqual(checkpoint["artifact_role"], "last")
            self.assertEqual(restored_trainer.training_state(), saved_trainer.training_state())
            for name, value in saved_model.state_dict().items():
                torch.testing.assert_close(restored_model.state_dict()[name], value)

    def test_resume_reuses_preloaded_last_checkpoint_without_loading_it_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _, _, _, last_path = self._write_run(root)
            preloaded = load_torch_checkpoint(last_path, mmap=False)
            original_loader = stage8_checkpoint.load_stage8_v2_checkpoint

            def track_load(path, *, artifact_role=None):
                if Path(path).resolve() == last_path.resolve():
                    raise AssertionError("last checkpoint was loaded twice")
                return original_loader(path, artifact_role=artifact_role)

            with patch.object(
                stage8_checkpoint,
                "load_stage8_v2_checkpoint",
                side_effect=track_load,
            ):
                restored = stage8_checkpoint.resume_stage8_last_checkpoint(
                    last_path,
                    checkpoint=preloaded,
                    trainer=self._trainer(self._model(), config),
                    config=config,
                    source_stage1_sha256="a" * 64,
                    dataset_fingerprint="data-sha",
                    effective_data=self._effective_data(),
                    condition=Stage8Condition.FULL,
                )

            self.assertIs(restored, preloaded)

    def test_new_stage8b_run_initialises_best_companion_at_round_zero(self) -> None:
        initialise = getattr(
            stage8_checkpoint, "initialise_stage8b_checkpoints", None
        )
        self.assertIsNotNone(initialise)
        if initialise is None:
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            model = self._model()
            warmup_writer = stage8_checkpoint.save_stage8_warmup_checkpoint
            warmup = warmup_writer(
                root / "warmup",
                model=model,
                config=config,
                source_stage1=self._source(),
                dataset_fingerprint="data-sha",
                effective_data=self._effective_data(),
                run_timestamp=self.TIMESTAMP,
                metrics=self._metrics(0.70),
            )
            trainer = self._trainer(model, config)
            trainer.stage8b_round = 0
            best, last = initialise(
                root / "baseline",
                trainer=trainer,
                source_stage1=self._source(),
                warmup_checkpoint=warmup,
                dataset_fingerprint="data-sha",
                effective_data=self._effective_data(),
                run_timestamp=self.TIMESTAMP,
                metrics=self._metrics(0.70),
            )

            best_value = load_torch_checkpoint(best, mmap=False)
            last_value = load_torch_checkpoint(last, mmap=False)
            self.assertEqual(best_value["training_state"]["stage8b_round"], 0)
            self.assertEqual(last_value["training_state"]["stage8b_round"], 0)
            self.assertEqual(
                last_value["best_companion"]["sha256"],
                hashlib.sha256(best.read_bytes()).hexdigest(),
            )

    def test_resume_rejects_source_warmup_condition_data_and_config_mismatch(self) -> None:
        resume = getattr(stage8_checkpoint, "resume_stage8_last_checkpoint", None)
        self.assertIsNotNone(resume)
        if resume is None:
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _, warmup, _, last = self._write_run(root)
            cases = (
                ("source", {"source_stage1_sha256": "b" * 64}, config, "Stage-1 source"),
                ("condition", {"condition": Stage8Condition.BASELINE}, config, "condition"),
                ("data", {"dataset_fingerprint": "other-data"}, config, "data fingerprint"),
                (
                    "effective-limit",
                    {"effective_data": self._effective_data(limit_per_type=1)},
                    config,
                    "effective data",
                ),
                (
                    "effective-count",
                    {"effective_data": {
                        **self._effective_data(), "train_samples": 3,
                    }},
                    config,
                    "effective data",
                ),
            )
            for name, overrides, current_config, message in cases:
                with self.subTest(name=name):
                    kwargs = {
                        "trainer": self._trainer(self._model(), current_config),
                        "config": current_config,
                        "source_stage1_sha256": "a" * 64,
                        "dataset_fingerprint": "data-sha",
                        "effective_data": self._effective_data(),
                        "condition": Stage8Condition.FULL,
                    }
                    kwargs.update(overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        resume(last, **kwargs)

            changed_config = deepcopy(config)
            changed_config.stage8.hierarchy_lambda = 0.06
            with self.assertRaisesRegex(ValueError, "configuration"):
                resume(
                    last,
                    trainer=self._trainer(self._model(), config),
                    config=changed_config,
                    source_stage1_sha256="a" * 64,
                    dataset_fingerprint="data-sha",
                    effective_data=self._effective_data(),
                    condition=Stage8Condition.FULL,
                )

            original_warmup = warmup.read_bytes()
            torch.save({"format_version": 2, "changed": True}, warmup)
            with self.assertRaisesRegex(ValueError, "warmup.*SHA256"):
                resume(
                    last,
                    trainer=self._trainer(self._model(), config),
                    config=config,
                    source_stage1_sha256="a" * 64,
                    dataset_fingerprint="data-sha",
                    effective_data=self._effective_data(),
                    condition=Stage8Condition.FULL,
                )
            warmup.write_bytes(original_warmup)

    def test_resume_rejects_missing_wrong_sha_or_v1_best_companion(self) -> None:
        resume = getattr(stage8_checkpoint, "resume_stage8_last_checkpoint", None)
        self.assertIsNotNone(resume)
        if resume is None:
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _, _, best, last = self._write_run(root)
            kwargs = {
                "trainer": self._trainer(self._model(), config),
                "config": config,
                "source_stage1_sha256": "a" * 64,
                "dataset_fingerprint": "data-sha",
                "effective_data": self._effective_data(),
                "condition": Stage8Condition.FULL,
            }
            best_bytes = best.read_bytes()
            best.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "best companion"):
                resume(last, **kwargs)
            best.write_bytes(best_bytes)

            last_bytes = last.read_bytes()
            last_value = load_torch_checkpoint(last, mmap=False)
            last_value["best_companion"]["path"] = str(
                root / "other" / best.name
            )
            torch.save(last_value, last)
            with self.assertRaisesRegex(ValueError, "same-directory"):
                resume(last, **kwargs)
            last.write_bytes(last_bytes)

            last_value = load_torch_checkpoint(last, mmap=False)
            last_value["best_companion"]["sha256"] = "0" * 64
            torch.save(last_value, last)
            with self.assertRaisesRegex(ValueError, "best companion SHA256"):
                resume(last, **kwargs)

            torch.save({
                "format_version": 1,
                "checkpoint_type": "stage8_binding_model",
                "artifact_role": "last",
            }, last)
            with self.assertRaisesRegex(ValueError, "V1.*not supported"):
                resume(last, **kwargs)


class Stage8V2CliTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    PYTHON = Path(sys.executable)

    def _run_help(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.PYTHON), str(self.ROOT / "source" / script), "--help"],
            cwd=self.ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _run_train(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(self.PYTHON),
                str(self.ROOT / "source" / "train_plm_stage8.py"),
                *arguments,
            ],
            cwd=self.ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_train_cli_exposes_only_stage1_warmup_and_v2_modes(self) -> None:
        result = self._run_help("train_plm_stage8.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--warmup-only", result.stdout)
        self.assertIn("--stage1-checkpoint", result.stdout)
        self.assertIn("--warmup-checkpoint", result.stdout)
        self.assertIn("--all-ablations", result.stdout)
        self.assertNotIn("--stage3-checkpoint", result.stdout)

    def test_train_cli_modes_and_required_sources_are_enforced_by_parser(self) -> None:
        missing_stage1 = self._run_train("--warmup-only")
        self.assertEqual(missing_stage1.returncode, 2)
        self.assertIn("--warmup-only requires --stage1-checkpoint", missing_stage1.stderr)

        missing_warmup = self._run_train("--condition", "baseline")
        self.assertEqual(missing_warmup.returncode, 2)
        self.assertIn("--condition requires --warmup-checkpoint", missing_warmup.stderr)

        conflicting = self._run_train(
            "--condition", "baseline",
            "--resume", "stage8_last_20260826_120000_000001.pt",
        )
        self.assertEqual(conflicting.returncode, 2)
        self.assertIn("not allowed with argument", conflicting.stderr)

    def test_ablation_commands_share_one_warmup_timestamp_and_config(self) -> None:
        source_root = self.ROOT / "source"
        sys.path.insert(0, str(source_root))
        try:
            spec = importlib.util.spec_from_file_location(
                "stage8_ablation_cli_for_test",
                source_root / "run_plm_stage8_ablations.py",
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(source_root))
        build_commands = getattr(module, "build_ablation_commands", None)
        self.assertIsNotNone(build_commands)
        if build_commands is None:
            return
        config = Path("/tmp/stage8.yaml")
        stage1 = Path("/tmp/stage1.pt")
        warmup = Path("/tmp/stage8a_best_pht_20260826_120000_000001.pt")
        commands = build_commands(
            config_path=config,
            stage1_checkpoint=stage1,
            warmup_checkpoint=warmup,
            run_timestamp="20260826_120000_000001",
            python_executable=Path("/tmp/python"),
            source_dir=Path("/tmp/source"),
            device="cpu",
            num_workers=0,
            swanlab_mode="disabled",
            limit_per_type=1,
        )

        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0].count("--warmup-only"), 1)
        self.assertEqual(commands[0].count("--stage1-checkpoint"), 1)
        for command, condition in zip(commands[1:], Stage8Condition):
            self.assertIn("--condition", command)
            self.assertIn(condition.value, command)
            warmup_index = command.index("--warmup-checkpoint") + 1
            timestamp_index = command.index("--run-timestamp") + 1
            self.assertEqual(command[warmup_index], str(warmup))
            self.assertEqual(command[timestamp_index], "20260826_120000_000001")
            self.assertEqual(command[command.index("--config") + 1], str(config))

    def test_evaluation_clis_do_not_accept_stage3_sources(self) -> None:
        for script in ("evaluate_plm_stage8.py", "evaluate_plm_stage8_tcr_sets.py"):
            with self.subTest(script=script):
                result = self._run_help(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("--stage3-checkpoint", result.stdout)
                self.assertNotIn("--stage3-config", result.stdout)

    def test_run_stage8_exposes_audit_warmup_train_and_ablations_only(self) -> None:
        result = subprocess.run(
            ["bash", str(self.ROOT / "run_stage8.sh"), "unknown"],
            cwd=self.ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("{audit|warmup|train|ablations}", result.stderr)
        self.assertNotIn("cache", result.stderr)

    def test_clean_index_contains_public_stage8_config_and_audit_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean_root = Path(temporary) / "clean"
            clean_root.mkdir()
            export = subprocess.run(
                [
                    "git", "checkout-index", "--all",
                    f"--prefix={clean_root}/",
                ],
                cwd=self.ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, export.stderr)
            self.assertTrue((clean_root / "configs/plm_stage8.yaml").is_file())
            self.assertTrue((clean_root / "source/audit_plm_stage8_data.py").is_file())

    def test_clean_index_stage8_entrypoints_launch_without_uncommitted_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean_root = Path(temporary) / "clean"
            clean_root.mkdir()
            export = subprocess.run(
                [
                    "git", "checkout-index", "--all",
                    f"--prefix={clean_root}/",
                ],
                cwd=self.ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, export.stderr)
            for script in (
                "audit_plm_stage8_data.py",
                "train_plm_stage8.py",
                "run_plm_stage8_ablations.py",
                "evaluate_plm_stage8.py",
                "evaluate_plm_stage8_tcr_sets.py",
            ):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [
                            str(self.PYTHON),
                            str(clean_root / f"source/{script}"),
                            "--help",
                        ],
                        cwd=clean_root,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            environment = dict(os.environ)
            environment["PYTHON_BIN"] = str(self.PYTHON)
            audit = subprocess.run(
                ["bash", str(clean_root / "run_stage8.sh"), "audit", "--help"],
                cwd=clean_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(audit.returncode, 0, audit.stderr)



if __name__ == "__main__":
    unittest.main()
