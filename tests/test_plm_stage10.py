"""Contract tests for Stage-10 teacher-student retention training."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from source.plm_unified.cache import EmbeddingCacheWriter
from source.plm_unified.config import ModelConfig
from source.plm_unified.stage8_loss import compute_stage8_loss
from source.plm_unified.stage8_model import Stage8BindingModel
from source.plm_unified.stage8_evaluation import evaluate_stage8_model


try:
    stage10_config = importlib.import_module(
        "source.plm_unified.stage10_config"
    )
except ModuleNotFoundError:
    stage10_config = None

try:
    stage10_model = importlib.import_module(
        "source.plm_unified.stage10_model"
    )
    stage10_loss = importlib.import_module(
        "source.plm_unified.stage10_loss"
    )
except ModuleNotFoundError:
    stage10_model = None
    stage10_loss = None

try:
    stage10_evaluation = importlib.import_module(
        "source.plm_unified.stage10_evaluation"
    )
except ModuleNotFoundError:
    stage10_evaluation = None

try:
    stage10_checkpoint = importlib.import_module(
        "source.plm_unified.stage10_checkpoint"
    )
except ModuleNotFoundError:
    stage10_checkpoint = None

try:
    stage10_trainer = importlib.import_module(
        "source.plm_unified.stage10_trainer"
    )
except ModuleNotFoundError:
    stage10_trainer = None

try:
    stage10_external = importlib.import_module(
        "source.plm_unified.stage10_external"
    )
except ModuleNotFoundError:
    stage10_external = None


class Stage10ConfigTests(unittest.TestCase):
    def _require_module(self):
        self.assertIsNotNone(
            stage10_config,
            "source.plm_unified.stage10_config must exist",
        )
        return stage10_config

    def _config(self):
        module = self._require_module()
        root = Path("/tmp/stage10-contract")
        return module.Stage10ExperimentConfig(
            paths=module.Stage10Paths(
                project_root=root,
                data_root=root / "data",
                cache_root=root / "cache",
                output_root=root / "models",
                swanlog_root=root / "logs",
            )
        )

    def test_defaults_preserve_stage8_and_add_retention(self) -> None:
        module = self._require_module()
        config = self._config()
        config.validate()

        self.assertEqual(
            tuple(value.value for value in module.Stage10Condition),
            ("baseline", "hier_only", "pt_imp_only", "full"),
        )
        self.assertEqual(config.runtime.fold, 4)
        self.assertEqual(config.runtime.seed, 2026)
        self.assertEqual(config.stage10.batch_size, 256)
        self.assertEqual(config.stage10.pht_loss_weight, 1.0)
        self.assertEqual(config.stage10.phla_loss_weight, 0.3)
        self.assertEqual(config.stage10.ptcr_loss_weight, 0.2)
        self.assertEqual(config.stage10.constraint_margin, 0.05)
        self.assertEqual(config.stage10.hierarchy_lambda, 0.05)
        self.assertEqual(config.stage10.pt_implication_lambda, 0.03)
        self.assertEqual(config.stage10.constraint_ramp_epochs, 5)
        self.assertEqual(config.stage10.selection_weights, (0.6, 0.2, 0.2))
        self.assertEqual(config.stage10.keep_loss_weight, 1.0)
        self.assertEqual(config.stage10.phla_keep_weight, 1.0)
        self.assertEqual(config.stage10.ptcr_keep_weight, 1.0)
        self.assertEqual(config.stage10.distillation_temperature, 1.0)
        self.assertEqual(
            config.stage10.legacy_task_degradation_tolerance,
            0.002,
        )

    def test_cache_dir_reuses_stage8_stage3_feature_namespace(self) -> None:
        config = self._config()
        self.assertEqual(
            config.cache_dir("a" * 64),
            Path(
                "/tmp/stage10-contract/cache/plm_stage8/"
                "fold_4/seed_2026/aaaaaaaaaaaa"
            ),
        )

    def test_retention_values_are_validated(self) -> None:
        config = self._config()
        config.stage10.distillation_temperature = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            config.validate()

        config.stage10.distillation_temperature = 1.0
        config.stage10.keep_loss_weight = -0.1
        with self.assertRaisesRegex(ValueError, "keep_loss_weight"):
            config.validate()

        config.stage10.keep_loss_weight = 1.0
        config.stage10.legacy_task_degradation_tolerance = -0.001
        with self.assertRaisesRegex(ValueError, "degradation"):
            config.validate()

    def test_v1_locks_fold_seed_data_and_balanced_batch(self) -> None:
        config = self._config()
        config.runtime.fold = 5
        with self.assertRaisesRegex(ValueError, "fold=4"):
            config.validate()

        config.runtime.fold = 4
        config.runtime.seed = 42
        with self.assertRaisesRegex(ValueError, "seed=2026"):
            config.validate()

        config.runtime.seed = 2026
        config.stage10.train_filename = "other.csv"
        with self.assertRaisesRegex(ValueError, "train_merged_fold4"):
            config.validate()

        config.stage10.train_filename = "train_merged_fold4.csv"
        config.stage10.batch_size = 258
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            config.validate()

    def test_repository_yaml_uses_stage10_output_and_online_tracking(self) -> None:
        module = self._require_module()
        config = module.load_stage10_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "plm_stage10.yaml"
        )
        self.assertEqual(config.swanlab.mode, "online")
        self.assertEqual(config.paths.output_root.name, "PLM_Stage10")
        self.assertEqual(config.paths.swanlog_root.name, "PLM_Stage10")


class Stage10ModelAndLossTests(unittest.TestCase):
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

    def _require_modules(self):
        self.assertIsNotNone(
            stage10_model,
            "source.plm_unified.stage10_model must exist",
        )
        self.assertIsNotNone(
            stage10_loss,
            "source.plm_unified.stage10_loss must exist",
        )
        return stage10_model, stage10_loss

    @staticmethod
    def _inputs():
        return (
            torch.randn(3, 5, 8),
            torch.ones(3, 5, dtype=torch.bool),
            torch.randn(3, 7, 8),
            torch.ones(3, 7, dtype=torch.bool),
            torch.randn(3, 6, 6),
            torch.ones(3, 6, dtype=torch.bool),
        )

    def test_teacher_and_student_start_equal_and_teacher_stays_frozen(self) -> None:
        model_module, _ = self._require_modules()
        source = Stage8BindingModel(self._model_config()).state_dict()
        pair = model_module.build_stage10_teacher_student_models(
            self._model_config(),
            source,
            device=torch.device("cpu"),
        )

        pair.teacher.train(True)
        self.assertFalse(pair.teacher.training)
        self.assertFalse(
            any(parameter.requires_grad for parameter in pair.teacher.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in pair.student.parameters())
        )
        for name, tensor in pair.teacher.state_dict().items():
            self.assertTrue(torch.equal(tensor, pair.student.state_dict()[name]))
        pair.assert_teacher_frozen()

    def test_teacher_legacy_heads_match_initial_student_outputs(self) -> None:
        model_module, _ = self._require_modules()
        source = Stage8BindingModel(self._model_config()).state_dict()
        pair = model_module.build_stage10_teacher_student_models(
            self._model_config(),
            source,
            device=torch.device("cpu"),
        )
        pair.student.eval()

        inputs = self._inputs()
        teacher = pair.teacher.forward_legacy_heads(*inputs)
        student = pair.student.forward_all(*inputs)

        torch.testing.assert_close(teacher.phla_logits, student.phla_logits)
        torch.testing.assert_close(teacher.ptcr_logits, student.ptcr_logits)
        self.assertEqual(student.pht_logits.shape, (3, 2))

    def test_distillation_kl_is_zero_for_equal_logits(self) -> None:
        _, loss_module = self._require_modules()
        teacher = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        student = teacher.clone().requires_grad_(True)

        value = loss_module.distillation_kl(
            student,
            teacher,
            temperature=1.0,
        )

        self.assertEqual(value.dtype, torch.float32)
        self.assertAlmostEqual(float(value.detach()), 0.0, places=7)
        value.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_distillation_kl_is_positive_and_updates_only_student(self) -> None:
        _, loss_module = self._require_modules()
        teacher = torch.tensor([[3.0, -1.0], [-1.0, 3.0]])
        student = torch.tensor(
            [[-1.0, 3.0], [3.0, -1.0]],
            dtype=torch.float16,
            requires_grad=True,
        )

        value = loss_module.distillation_kl(
            student,
            teacher,
            temperature=1.0,
        )
        value.backward()

        self.assertGreater(float(value.detach()), 0.0)
        self.assertEqual(value.dtype, torch.float32)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_nonpositive_distillation_temperature_is_rejected(self) -> None:
        _, loss_module = self._require_modules()
        logits = torch.zeros(2, 2)
        with self.assertRaisesRegex(ValueError, "temperature"):
            loss_module.distillation_kl(logits, logits, temperature=0.0)

    def test_zero_keep_weight_is_exact_stage8_loss(self) -> None:
        _, loss_module = self._require_modules()
        config_module = self._require_config_module()
        phla = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
        ptcr = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        pht = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        label_types = ["111", "000"]
        stage8 = compute_stage8_loss(
            phla,
            ptcr,
            pht,
            labels,
            labels,
            labels,
            label_types,
            condition="full",
            epoch=3,
        )
        stage10 = loss_module.compute_stage10_loss(
            phla,
            ptcr,
            pht,
            phla.detach(),
            ptcr.detach(),
            labels,
            labels,
            labels,
            label_types,
            condition=config_module.Stage10Condition.FULL,
            epoch=3,
            keep_loss_weight=0.0,
        )

        self.assertTrue(torch.equal(stage10.stage8.total, stage8.total))
        self.assertTrue(torch.equal(stage10.total, stage8.total))

    @staticmethod
    def _require_config_module():
        if stage10_config is None:
            raise AssertionError("stage10_config must exist")
        return stage10_config


class Stage10CheckpointTests(unittest.TestCase):
    @staticmethod
    def _require_module():
        if stage10_checkpoint is None:
            raise AssertionError(
                "source.plm_unified.stage10_checkpoint must exist"
            )
        return stage10_checkpoint

    @staticmethod
    def _config(root: Path, checkpoint_path: Path):
        if stage10_config is None:
            raise AssertionError("stage10_config must exist")
        return stage10_config.Stage10ExperimentConfig(
            paths=stage10_config.Stage10Paths(
                project_root=root,
                data_root=root / "data",
                cache_root=root / "cache",
                output_root=root / "models",
                swanlog_root=root / "logs",
                stage3_checkpoint=checkpoint_path,
            ),
            model=Stage10ModelAndLossTests._model_config(),
        )

    @staticmethod
    def _stage3_checkpoint(config) -> dict:
        binding = Stage8BindingModel(config.model)
        state = {
            f"binding_model.{name}": tensor.detach().clone()
            for name, tensor in binding.state_dict().items()
        }
        state.update(
            {
                "peptide_encoder.dummy": torch.tensor([1.0]),
                "hla_encoder.dummy": torch.tensor([2.0]),
                "tcr_encoder.dummy": torch.tensor([3.0]),
            }
        )
        return {
            "format_version": 1,
            "checkpoint_type": "stage3_multitask_model",
            "tasks": ["phla", "ptcr", "triplet"],
            "config": {
                "model": config.to_dict()["model"],
                "training": {"fold": 4, "seed": 2026},
            },
            "initialization_source": {
                "kind": "stage1",
                "source_fold": 4,
                "source_seed": 2026,
            },
            "model": state,
        }

    @staticmethod
    def _lean_checkpoint(module, config, source, effective_data):
        contract = module.stage10_training_contract(config, effective_data)
        return {
            "format_version": 1,
            "checkpoint_type": "stage10_binding_model",
            "artifact_role": "last",
            "condition": "full",
            "model": {
                name: tensor.detach().clone()
                for name, tensor in source.binding_state.items()
            },
            "best_model": {
                name: tensor.detach().clone()
                for name, tensor in source.binding_state.items()
            },
            "config": config.to_dict(),
            "source_stage3": source.lineage,
            "teacher_policy": contract["descriptor"]["teacher_policy"],
            "training_contract": contract,
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 0},
            "scaler": {},
            "rng_state": {},
            "validation_metrics": {"pht": {"auroc": 0.8}},
            "validation_retention": {"eligible": True},
            "validation_joint_score": 0.8,
            "selected_pht_threshold": 0.5,
            "global_step": 1,
            "run_timestamp": "20260814_000000_000001",
        }

    def test_stage3_teacher_source_validates_lineage_and_extracts_binding(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage3.pt"
            config = self._config(root, path)
            payload = self._stage3_checkpoint(config)
            torch.save(payload, path)

            source = module.load_stage3_teacher_source(config)

            self.assertEqual(len(source.sha256), 64)
            self.assertEqual(source.lineage["kind"], "stage3")
            self.assertEqual(source.lineage["role"], "teacher_initialization")
            self.assertEqual(
                set(source.binding_state),
                {
                    name.removeprefix("binding_model.")
                    for name in payload["model"]
                    if name.startswith("binding_model.")
                },
            )
            self.assertFalse(
                any(name.endswith("encoder.dummy") for name in source.binding_state)
            )

    def test_stage3_teacher_source_rejects_invalid_contract_fields(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage3.pt"
            config = self._config(root, path)
            valid = self._stage3_checkpoint(config)
            invalid_payloads = {}

            changed = deepcopy(valid)
            changed["format_version"] = 2
            invalid_payloads["format"] = changed
            changed = deepcopy(valid)
            changed["checkpoint_type"] = "stage8_binding_model"
            invalid_payloads["type"] = changed
            changed = deepcopy(valid)
            changed["tasks"] = ["phla", "ptcr"]
            invalid_payloads["tasks"] = changed
            changed = deepcopy(valid)
            changed["config"]["model"]["model_dim"] = 16
            invalid_payloads["model"] = changed
            changed = deepcopy(valid)
            changed["config"]["training"]["fold"] = 5
            invalid_payloads["fold"] = changed
            changed = deepcopy(valid)
            changed["config"]["training"]["seed"] = 42
            invalid_payloads["seed"] = changed
            changed = deepcopy(valid)
            changed["initialization_source"]["kind"] = "stage2"
            invalid_payloads["upstream"] = changed
            changed = deepcopy(valid)
            del changed["model"]["peptide_encoder.dummy"]
            invalid_payloads["encoder"] = changed
            changed = deepcopy(valid)
            changed["model"] = {
                name: tensor
                for name, tensor in changed["model"].items()
                if not name.startswith("binding_model.")
            }
            invalid_payloads["binding"] = changed

            for case, payload in invalid_payloads.items():
                with self.subTest(case=case):
                    torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        module.load_stage3_teacher_source(config)

    def test_lean_and_resume_contracts_are_strict_and_teacher_free(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage3.pt"
            config = self._config(root, path)
            torch.save(self._stage3_checkpoint(config), path)
            source = module.load_stage3_teacher_source(config)
            effective_data = {
                "dataset_fingerprint": "data-sha",
                "train_samples": 400,
                "val_samples": 80,
            }
            lean = self._lean_checkpoint(
                module, config, source, effective_data
            )

            module.validate_stage10_lean_checkpoint(
                lean, config, source.sha256
            )
            module.validate_stage10_resume_checkpoint(
                lean,
                config,
                source.sha256,
                effective_data,
                condition="full",
            )
            contract = lean["training_contract"]["descriptor"]
            self.assertEqual(contract["stage10"], config.to_dict()["stage10"])
            self.assertEqual(contract["effective_data"], effective_data)
            self.assertTrue(contract["teacher_policy"]["frozen"])
            self.assertEqual(
                contract["teacher_policy"]["distilled_tasks"],
                ["phla", "ptcr"],
            )

            wrong_source = deepcopy(lean)
            wrong_source["source_stage3"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "different Stage-3"):
                module.validate_stage10_lean_checkpoint(
                    wrong_source, config, source.sha256
                )

            with_teacher = deepcopy(lean)
            with_teacher["teacher"] = source.binding_state
            with self.assertRaisesRegex(ValueError, "teacher"):
                module.validate_stage10_lean_checkpoint(
                    with_teacher, config, source.sha256
                )

            best = deepcopy(lean)
            best["artifact_role"] = "best"
            with self.assertRaisesRegex(ValueError, "last"):
                module.validate_stage10_resume_checkpoint(
                    best,
                    config,
                    source.sha256,
                    effective_data,
                    condition="full",
                )

            with self.assertRaisesRegex(ValueError, "condition"):
                module.validate_stage10_resume_checkpoint(
                    lean,
                    config,
                    source.sha256,
                    effective_data,
                    condition="baseline",
                )

            changed_data = dict(effective_data, train_samples=404)
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                module.validate_stage10_resume_checkpoint(
                    lean,
                    config,
                    source.sha256,
                    changed_data,
                    condition="full",
                )

    def test_full_export_contains_student_and_no_teacher_or_training_state(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage3.pt"
            destination = root / "stage10_full.pt"
            config = self._config(root, path)
            stage3_payload = self._stage3_checkpoint(config)
            stage3_payload["model"]["binding_model.obsolete"] = torch.tensor(
                [-1.0]
            )
            torch.save(stage3_payload, path)
            source = module.load_stage3_teacher_source(config)
            lean = self._lean_checkpoint(
                module,
                config,
                source,
                {"dataset_fingerprint": "data-sha"},
            )
            lean["model"].pop("obsolete")

            module.export_stage10_full_checkpoint(source, lean, destination)
            full = module.load_torch_checkpoint(destination, mmap=False)

            self.assertEqual(full["checkpoint_type"], "stage10_multitask_model")
            self.assertIn("peptide_encoder.dummy", full["model"])
            self.assertNotIn("binding_model.obsolete", full["model"])
            self.assertFalse(
                any("teacher" in name for name in full["model"])
            )
            for key in (
                "teacher",
                "optimizer",
                "scheduler",
                "scaler",
                "rng_state",
                "best_model",
            ):
                self.assertNotIn(key, full)
            module.validate_stage10_full_checkpoint(
                full, config, source.sha256
            )
            exported = module.extract_stage10_binding_state(full)
            self.assertEqual(set(exported), set(lean["model"]))
            for name, tensor in lean["model"].items():
                self.assertTrue(torch.equal(exported[name], tensor))


class Stage10TrainerTests(unittest.TestCase):
    @staticmethod
    def _require_module():
        if stage10_trainer is None:
            raise AssertionError(
                "source.plm_unified.stage10_trainer must exist"
            )
        return stage10_trainer

    @staticmethod
    def _config(root: Path):
        if stage10_config is None:
            raise AssertionError("stage10_config must exist")
        config = stage10_config.Stage10ExperimentConfig(
            paths=stage10_config.Stage10Paths(
                project_root=root,
                data_root=root / "data",
                cache_root=root / "cache",
                output_root=root / "models",
                swanlog_root=root / "logs",
                stage3_checkpoint=root / "stage3.pt",
            ),
            model=Stage10ModelAndLossTests._model_config(),
        )
        config.stage10.batch_size = 4
        config.stage10.eval_batch_size = 4
        config.stage10.max_epochs = 1
        config.stage10.min_epochs = 1
        config.stage10.early_stopping_patience = 1
        config.stage10.mixed_precision = "none"
        config.validate()
        return config

    @staticmethod
    def _pair_and_source(config):
        if stage10_model is None or stage10_checkpoint is None:
            raise AssertionError("Stage10 model and checkpoint modules must exist")
        torch.manual_seed(31)
        binding_state = Stage8BindingModel(config.model).state_dict()
        pair = stage10_model.build_stage10_teacher_student_models(
            config.model,
            binding_state,
            device=torch.device("cpu"),
        )
        source = stage10_checkpoint.Stage3TeacherSource(
            path=config.paths.stage3_checkpoint,
            sha256="a" * 64,
            checkpoint={},
            binding_state={
                name: tensor.detach().clone()
                for name, tensor in binding_state.items()
            },
            lineage={
                "kind": "stage3",
                "role": "teacher_initialization",
                "sha256": "a" * 64,
                "source_fold": 4,
                "source_seed": 2026,
            },
        )
        return pair, source

    @staticmethod
    def _batch():
        torch.manual_seed(37)
        return {
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

    @staticmethod
    def _data_audit():
        return {
            "dataset_fingerprint": "data-sha",
            "effective_data": {
                "dataset_fingerprint": "data-sha",
                "train_samples": 4,
                "val_samples": 4,
            },
        }

    def test_optimizer_has_every_student_parameter_and_no_teacher(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            pair, _ = self._pair_and_source(config)

            optimizer = module.build_stage10_optimizer(
                pair.teacher,
                pair.student,
                config,
            )
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            student = {id(parameter) for parameter in pair.student.parameters()}
            teacher = {id(parameter) for parameter in pair.teacher.parameters()}

            self.assertEqual(optimized, student)
            self.assertTrue(optimized.isdisjoint(teacher))
            self.assertEqual(
                [group["name"] for group in optimizer.param_groups],
                ["adapters", "pretrained_attention", "triplet_modules"],
            )
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                module.build_stage10_optimizer(
                    pair.student,
                    pair.student,
                    config,
                )

    def test_candidate_requires_retention_and_joint_improvement(self):
        module = self._require_module()
        self.assertFalse(
            module.candidate_improves(
                SimpleNamespace(retention_eligible=False, joint_score=0.99),
                best_joint=0.80,
                min_delta=0.0001,
            )
        )
        self.assertFalse(
            module.candidate_improves(
                SimpleNamespace(retention_eligible=True, joint_score=0.8001),
                best_joint=0.80,
                min_delta=0.0001,
            )
        )
        self.assertTrue(
            module.candidate_improves(
                SimpleNamespace(retention_eligible=True, joint_score=0.81),
                best_joint=0.80,
                min_delta=0.0001,
            )
        )

    def test_ineligible_high_joint_epoch_falls_back_to_epoch_zero(self):
        module = self._require_module()

        class IneligibleCandidateTrainer(module.Stage10Trainer):
            evaluation_calls = 0

            def train_epoch(self, epoch):
                return {}

            def evaluate(self):
                evaluation = super().evaluate()
                self.evaluation_calls += 1
                if self.evaluation_calls == 2:
                    return replace(
                        evaluation,
                        joint_score=evaluation.joint_score + 1.0,
                        retention_eligible=False,
                    )
                return evaluation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            pair, source = self._pair_and_source(config)
            trainer = IneligibleCandidateTrainer(
                pair,
                config,
                [self._batch()],
                [self._batch()],
                condition="full",
                source=source,
                data_audit=self._data_audit(),
                run_timestamp="20260814_010000_000001",
            )

            summary = trainer.fit()

            self.assertEqual(summary["best_epoch"], 0)
            best = stage10_checkpoint.load_torch_checkpoint(
                summary["best_checkpoint"], mmap=False
            )
            self.assertEqual(best["epoch"], 0)
            self.assertTrue(best["validation_retention"]["eligible"])

    def test_one_epoch_student_only_checkpoint_and_strict_resume(self):
        module = self._require_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            pair, source = self._pair_and_source(config)
            teacher_before = {
                name: tensor.detach().clone()
                for name, tensor in pair.teacher.state_dict().items()
            }
            timestamp = "20260814_020000_000001"
            trainer = module.Stage10Trainer(
                pair,
                config,
                [self._batch()],
                [self._batch()],
                condition="full",
                source=source,
                data_audit=self._data_audit(),
                run_timestamp=timestamp,
            )

            summary = trainer.fit()
            checkpoint = stage10_checkpoint.load_torch_checkpoint(
                summary["last_checkpoint"], mmap=False
            )

            self.assertEqual(
                checkpoint["checkpoint_type"], "stage10_binding_model"
            )
            self.assertNotIn("teacher", checkpoint)
            self.assertIn("teacher_policy", checkpoint)
            self.assertIn("validation_retention", checkpoint)
            self.assertIsInstance(checkpoint["best_model"], dict)
            self.assertFalse(
                any(name.startswith("binding_model.") for name in checkpoint["model"])
            )
            for name, tensor in pair.teacher.state_dict().items():
                self.assertTrue(torch.equal(tensor, teacher_before[name]))
            prediction_header = Path(
                summary["validation_predictions"]
            ).read_text(encoding="utf-8").splitlines()[0]
            for column in (
                "student_prob_PH",
                "student_prob_PT",
                "student_prob_PHT",
                "teacher_prob_PH",
                "teacher_prob_PT",
                "PH_probability_drift",
                "PT_probability_drift",
            ):
                self.assertIn(column, prediction_header)

            Path(summary["best_checkpoint"]).unlink()
            Path(summary["validation_predictions"]).unlink()
            resumed_pair, _ = self._pair_and_source(config)
            resumed = module.Stage10Trainer(
                resumed_pair,
                config,
                [self._batch()],
                [self._batch()],
                condition="full",
                source=source,
                data_audit=self._data_audit(),
                run_timestamp=timestamp,
            )
            resumed.fit(resume=summary["last_checkpoint"])
            self.assertTrue(resumed.best_path.is_file())
            self.assertTrue(resumed.prediction_path.is_file())
            resumed.models.assert_teacher_frozen()

            rejecting_pair, _ = self._pair_and_source(config)
            rejecting = module.Stage10Trainer(
                rejecting_pair,
                config,
                [self._batch()],
                [self._batch()],
                condition="full",
                source=source,
                data_audit=self._data_audit(),
                run_timestamp=timestamp,
            )
            with self.assertRaisesRegex(ValueError, "last"):
                rejecting.resume(resumed.best_path)

            changed_config = deepcopy(config)
            changed_config.stage10.keep_loss_weight = 0.9
            changed_pair, _ = self._pair_and_source(changed_config)
            changed = module.Stage10Trainer(
                changed_pair,
                changed_config,
                [self._batch()],
                [self._batch()],
                condition="full",
                source=source,
                data_audit=self._data_audit(),
                run_timestamp=timestamp,
            )
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                changed.resume(resumed.last_path)


class Stage10CliTests(unittest.TestCase):
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _load_script(cls, filename: str):
        path = cls._project_root() / "source" / filename
        if not path.is_file():
            raise AssertionError(f"Stage10 entrypoint must exist: {path}")
        source_root = str(path.parent)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        module_name = f"_stage10_test_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Cannot import Stage10 entrypoint: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_entrypoint_parsers_forward_all_supported_overrides(self):
        audit = self._load_script("audit_plm_stage10_data.py").parse_args(
            ["--config", "audit.yaml", "--report-dir", "reports"]
        )
        self.assertEqual(audit.config, Path("audit.yaml"))
        self.assertEqual(audit.report_dir, Path("reports"))

        cache = self._load_script(
            "precompute_plm_stage10_embeddings.py"
        ).parse_args(
            [
                "--config",
                "cache.yaml",
                "--stage3-checkpoint",
                "stage3.pt",
                "--device",
                "cpu",
                "--overwrite",
            ]
        )
        self.assertEqual(cache.stage3_checkpoint, Path("stage3.pt"))
        self.assertEqual(cache.device, "cpu")
        self.assertTrue(cache.overwrite)

        train = self._load_script("train_plm_stage10.py").parse_args(
            [
                "--config",
                "train.yaml",
                "--condition",
                "full",
                "--stage3-checkpoint",
                "stage3.pt",
                "--resume",
                "last.pt",
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--swanlab-mode",
                "disabled",
                "--run-timestamp",
                "20260814_030000_000001",
                "--limit-per-type",
                "2",
            ]
        )
        self.assertEqual(train.condition, "full")
        self.assertEqual(train.stage3_checkpoint, Path("stage3.pt"))
        self.assertEqual(train.resume, Path("last.pt"))
        self.assertEqual(train.device, "cpu")
        self.assertEqual(train.num_workers, 0)
        self.assertEqual(train.swanlab_mode, "disabled")
        self.assertEqual(train.run_timestamp, "20260814_030000_000001")
        self.assertEqual(train.limit_per_type, 2)

        evaluate = self._load_script("evaluate_plm_stage10.py").parse_args(
            [
                "--config",
                "eval.yaml",
                "--checkpoint",
                "best.pt",
                "--stage3-checkpoint",
                "stage3.pt",
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--output",
                "evaluation.json",
            ]
        )
        self.assertEqual(evaluate.checkpoint, Path("best.pt"))
        self.assertEqual(evaluate.output, Path("evaluation.json"))
        self.assertEqual(evaluate.num_workers, 0)

        export = self._load_script(
            "export_plm_stage10_full.py"
        ).parse_args(
            [
                "--config",
                "export.yaml",
                "--checkpoint",
                "best.pt",
                "--stage3-checkpoint",
                "stage3.pt",
                "--output",
                "full.pt",
            ]
        )
        self.assertEqual(export.checkpoint, Path("best.pt"))
        self.assertEqual(export.stage3_checkpoint, Path("stage3.pt"))
        self.assertEqual(export.output, Path("full.pt"))

    def test_ablation_commands_preserve_condition_order_and_options(self):
        module = self._load_script("run_plm_stage10_ablations.py")
        args = module.parse_args(
            [
                "--config",
                "stage10.yaml",
                "--stage3-checkpoint",
                "stage3.pt",
                "--device",
                "cuda:1",
                "--num-workers",
                "3",
                "--swanlab-mode",
                "online",
                "--limit-per-type",
                "5",
                "--precompute",
                "--overwrite-cache",
            ]
        )

        commands = module.build_commands(
            args,
            python_executable=Path("/opt/python"),
            source_dir=Path("/project/source"),
        )

        self.assertEqual(len(commands), 5)
        self.assertIn("precompute_plm_stage10_embeddings.py", commands[0][1])
        self.assertIn("--overwrite", commands[0])
        conditions = [
            command[command.index("--condition") + 1]
            for command in commands[1:]
        ]
        self.assertEqual(
            conditions,
            ["baseline", "hier_only", "pt_imp_only", "full"],
        )
        for command in commands[1:]:
            for option, value in (
                ("--config", "stage10.yaml"),
                ("--stage3-checkpoint", "stage3.pt"),
                ("--device", "cuda:1"),
                ("--num-workers", "3"),
                ("--swanlab-mode", "online"),
                ("--limit-per-type", "5"),
            ):
                self.assertEqual(command[command.index(option) + 1], value)

    def test_existing_shared_cache_is_validated_without_rewrite(self):
        module_path = (
            self._project_root()
            / "source/plm_unified/stage10_precompute.py"
        )
        if not module_path.is_file():
            raise AssertionError("stage10_precompute.py must exist")
        module = importlib.import_module(
            "source.plm_unified.stage10_precompute"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Stage10TrainerTests._config(root)
            source = stage10_checkpoint.Stage3TeacherSource(
                path=root / "stage3.pt",
                sha256="b" * 64,
                checkpoint={},
                binding_state={},
                lineage={"kind": "stage3", "sha256": "b" * 64},
            )
            audit = SimpleNamespace(
                dataset_fingerprint="cache-data-sha",
                require_clean=lambda: None,
            )
            cache_root = config.cache_dir(source.sha256)
            dimensions = {
                "peptide": config.model.peptide_input_dim,
                "hla": config.model.hla_input_dim,
                "tcr": config.model.tcr_input_dim,
            }
            lengths = {
                "peptide": config.model.peptide_max_length,
                "hla": config.model.hla_max_length,
                "tcr": config.model.tcr_max_length,
            }
            for entity in ("peptide", "hla", "tcr"):
                with EmbeddingCacheWriter(
                    cache_root,
                    entity,
                    hidden_dim=dimensions[entity],
                    max_length=lengths[entity],
                    model_metadata={
                        "checkpoint_fingerprint": source.sha256,
                        "dataset_fingerprint": audit.dataset_fingerprint,
                    },
                    chunk_rows=1,
                ) as writer:
                    writer.add_batch(
                        [entity.upper()],
                        np.zeros(
                            (1, lengths[entity], dimensions[entity]),
                            dtype=np.float16,
                        ),
                        np.ones((1, lengths[entity]), dtype=np.uint8),
                        np.asarray([lengths[entity]], dtype=np.int16),
                    )
            manifests = list(cache_root.glob("*/manifest.json"))
            before = {
                path: (path.stat().st_mtime_ns, path.read_bytes())
                for path in manifests
            }

            result = module.precompute_stage10_caches(
                config,
                source,
                audit,
                device=torch.device("cpu"),
                overwrite=False,
            )

            self.assertEqual(result, cache_root)
            self.assertEqual(
                before,
                {
                    path: (path.stat().st_mtime_ns, path.read_bytes())
                    for path in manifests
                },
            )

    def test_bash_dispatcher_rejects_unknown_action_with_stable_usage(self):
        script = self._project_root() / "run_stage10.sh"
        if not script.is_file():
            raise AssertionError("run_stage10.sh must exist")
        result = subprocess.run(
            ["bash", str(script), "not-an-action"],
            cwd=self._project_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "audit|cache|train|ablations|evaluate|evaluate-tcr|export",
            result.stderr,
        )


class Stage10ExternalEvaluationTests(unittest.TestCase):
    @staticmethod
    def _require_module():
        if stage10_external is None:
            raise AssertionError(
                "source.plm_unified.stage10_external must exist"
            )
        return stage10_external

    def test_comparison_report_has_signed_deltas_and_probability_summaries(self):
        module = self._require_module()
        labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
        probabilities = {
            "stage3": np.asarray([0.1, 0.8, 0.7, 0.2]),
            "stage8": np.asarray([0.2, 0.6, 0.4, 0.3]),
            "stage10": np.asarray([0.1, 0.75, 0.65, 0.2]),
        }

        report = module.build_external_comparison_report(
            labels,
            probabilities,
            threshold=0.5,
        )

        self.assertIn("stage10_minus_stage3", report["deltas"])
        self.assertIn("stage10_minus_stage8", report["deltas"])
        self.assertEqual(report["metrics"]["stage10"]["samples"], 4)
        self.assertEqual(
            report["metrics"]["stage10"]["positive_samples"], 2
        )
        self.assertAlmostEqual(
            report["probability_summary"]["stage10"]["positive_mean"],
            0.7,
        )
        self.assertAlmostEqual(
            report["probability_summary"]["stage10"]["negative_mean"],
            0.15,
        )
        expected = (
            report["metrics"]["stage10"]["accuracy"]
            - report["metrics"]["stage8"]["accuracy"]
        )
        self.assertAlmostEqual(
            report["deltas"]["stage10_minus_stage8"]["accuracy"],
            expected,
        )

    def test_comparison_without_stage8_only_reports_stage3_delta(self):
        module = self._require_module()
        report = module.build_external_comparison_report(
            np.asarray([0, 1], dtype=np.int64),
            {
                "stage3": np.asarray([0.2, 0.8]),
                "stage10": np.asarray([0.3, 0.7]),
            },
            threshold=0.5,
        )
        self.assertEqual(
            set(report["deltas"]),
            {"stage10_minus_stage3"},
        )
        self.assertNotIn("stage8", report["metrics"])

    def test_stage8_reference_must_share_exact_stage3_sha(self):
        module = self._require_module()
        valid = {
            "checkpoint_type": "stage8_binding_model",
            "source_stage3": {"sha256": "a" * 64},
        }
        module.validate_stage8_reference_source(valid, "a" * 64)
        invalid = deepcopy(valid)
        invalid["source_stage3"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different Stage-3"):
            module.validate_stage8_reference_source(invalid, "a" * 64)

    def test_multi_model_inference_preserves_row_order_and_fixed_threshold(self):
        module = self._require_module()

        class FakeBindingModel(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(
                self,
                task,
                peptide_hidden,
                peptide_mask,
                tcr_hidden,
                tcr_mask,
            ):
                if task != "ptcr":
                    raise AssertionError("Only pTCR inference is allowed")
                positive_logit = peptide_hidden[:, 0, 0] + self.offset
                logits = torch.stack(
                    (torch.zeros_like(positive_logit), positive_logit),
                    dim=-1,
                )
                return SimpleNamespace(logits=logits)

        encoded = module.EncodedPairSequences(
            peptide_index={"P1": 0, "P2": 1},
            tcr_index={"T1": 0, "T2": 1},
            peptide_hidden=np.asarray([[[1.0]], [[-1.0]]], dtype=np.float32),
            peptide_mask=np.ones((2, 1), dtype=np.uint8),
            tcr_hidden=np.zeros((2, 1, 1), dtype=np.float32),
            tcr_mask=np.ones((2, 1), dtype=np.uint8),
        )
        rows = [
            module.ExternalPairRow("P2", "T1", 1, 9),
            module.ExternalPairRow("P1", "T2", 0, 4),
        ]

        reports, probabilities = module.evaluate_binding_models_on_rows(
            {
                "stage10": FakeBindingModel(0.2),
                "stage3": FakeBindingModel(0.0),
            },
            rows,
            encoded,
            device=torch.device("cpu"),
            pair_batch_size=1,
            threshold=0.4,
        )

        expected_stage3 = 1.0 / (1.0 + np.exp(-np.asarray([-1.0, 1.0])))
        np.testing.assert_allclose(
            probabilities["stage3"],
            expected_stage3,
            rtol=1e-6,
        )
        self.assertEqual(reports["stage3"]["threshold_used"], 0.4)
        self.assertEqual(reports["stage10"]["metrics"]["samples"], 2)

    def test_row_csv_has_source_rows_and_available_model_columns(self):
        module = self._require_module()
        rows = {
            "set_b": [module.ExternalPairRow("P2", "T2", 0, 8)],
            "set_a": [
                module.ExternalPairRow("P1", "T1", 1, 3),
                module.ExternalPairRow("P3", "T3", 0, 7),
            ],
        }
        probabilities = {
            "set_a": {
                "stage3": np.asarray([0.8, 0.2]),
                "stage10": np.asarray([0.7, 0.3]),
            },
            "set_b": {
                "stage3": np.asarray([0.1]),
                "stage10": np.asarray([0.2]),
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "rows.csv"
            module.write_external_row_csv(destination, rows, probabilities)
            lines = destination.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            lines[0],
            "dataset,source_row,peptide,tcr,label,prob_stage3,prob_stage10",
        )
        self.assertTrue(lines[1].startswith("set_a,3,P1,T1,1,"))
        self.assertTrue(lines[2].startswith("set_a,7,P3,T3,0,"))
        self.assertTrue(lines[3].startswith("set_b,8,P2,T2,0,"))

    def test_external_cli_uses_fixed_threshold_and_optional_stage8(self):
        script = Stage10CliTests._load_script(
            "evaluate_plm_stage10_tcr_sets.py"
        )
        args = script.parse_args(
            [
                "--checkpoint",
                "stage10.pt",
                "--stage8-checkpoint",
                "stage8.pt",
                "--threshold",
                "0.5",
                "--rows-output",
                "rows.csv",
            ]
        )
        script.validate_args(args)
        self.assertEqual(args.checkpoint, Path("stage10.pt"))
        self.assertEqual(args.stage8_checkpoint, Path("stage8.pt"))
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.rows_output, Path("rows.csv"))
        for invalid in (0.0, 1.0):
            args.threshold = invalid
            with self.assertRaisesRegex(ValueError, "threshold"):
                script.validate_args(args)


class Stage10EvaluationTests(unittest.TestCase):
    @staticmethod
    def _require_module():
        if stage10_evaluation is None:
            raise AssertionError(
                "source.plm_unified.stage10_evaluation must exist"
            )
        return stage10_evaluation

    @staticmethod
    def _models_and_batch():
        if stage10_model is None:
            raise AssertionError("stage10_model must exist")
        config = Stage10ModelAndLossTests._model_config()
        torch.manual_seed(23)
        source = Stage8BindingModel(config).state_dict()
        models = stage10_model.build_stage10_teacher_student_models(
            config,
            source,
            device=torch.device("cpu"),
        )
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
        return models, batch

    @staticmethod
    def _score_metrics(value: float) -> dict[str, float]:
        return {"auroc": value, "aupr": value}

    def test_equal_models_have_zero_drift_and_are_retention_eligible(self) -> None:
        module = self._require_module()
        models, batch = self._models_and_batch()

        result = module.evaluate_stage10_models(
            models.teacher,
            models.student,
            [batch],
            device=torch.device("cpu"),
            mixed_precision="none",
            margin=0.05,
            selection_weights=(0.6, 0.2, 0.2),
            degradation_tolerance=0.002,
        )

        self.assertTrue(result.retention_eligible)
        self.assertLess(abs(result.retention["phla_kl"]), 1e-7)
        self.assertLess(abs(result.retention["ptcr_kl"]), 1e-7)
        self.assertLess(result.retention["phla_probability_mae"], 1e-7)
        self.assertLess(result.retention["ptcr_probability_mae"], 1e-7)
        self.assertEqual(result.retention["phla_agreement"], 1.0)
        self.assertEqual(result.retention["ptcr_agreement"], 1.0)
        self.assertEqual(
            result.predictions["source_rows"].tolist(),
            [2, 3, 4, 5],
        )

    def test_student_metrics_match_stage8_evaluation(self) -> None:
        module = self._require_module()
        models, batch = self._models_and_batch()
        stage10_result = module.evaluate_stage10_models(
            models.teacher,
            models.student,
            [batch],
            device=torch.device("cpu"),
            mixed_precision="none",
            margin=0.05,
            selection_weights=(0.6, 0.2, 0.2),
            degradation_tolerance=0.002,
        )
        stage8_result = evaluate_stage8_model(
            models.student,
            [batch],
            device=torch.device("cpu"),
            mixed_precision="none",
            margin=0.05,
            selection_weights=(0.6, 0.2, 0.2),
        )

        self.assertAlmostEqual(
            stage10_result.joint_score,
            stage8_result.joint_score,
            places=7,
        )
        self.assertAlmostEqual(
            stage10_result.selected_pht_threshold,
            stage8_result.selected_pht_threshold,
            places=7,
        )
        for task in ("phla", "ptcr", "pht"):
            for metric in (
                "accuracy",
                "aupr",
                "auroc",
                "f1",
                "loss",
                "mcc",
                "precision",
                "recall",
                "specificity",
            ):
                self.assertAlmostEqual(
                    stage10_result.student_metrics[task][metric],
                    stage8_result.metrics[task][metric],
                    places=7,
                )
        self.assertEqual(
            stage10_result.student_metrics["violations"],
            stage8_result.metrics["violations"],
        )

    def test_retention_gate_rejects_pt_drop_beyond_tolerance(self) -> None:
        module = self._require_module()
        teacher_metrics = {
            "phla": self._score_metrics(0.90),
            "ptcr": self._score_metrics(0.90),
        }
        student_metrics = {
            "phla": self._score_metrics(0.91),
            "ptcr": self._score_metrics(0.897),
        }

        self.assertFalse(
            module.retention_is_eligible(
                student_metrics,
                teacher_metrics,
                tolerance=0.002,
            )
        )

    def test_retention_gate_accepts_exact_tolerance_boundary(self) -> None:
        module = self._require_module()
        teacher_metrics = {
            "phla": self._score_metrics(0.90),
            "ptcr": self._score_metrics(0.90),
        }
        student_metrics = {
            "phla": self._score_metrics(0.898),
            "ptcr": self._score_metrics(0.898),
        }

        self.assertTrue(
            module.retention_is_eligible(
                student_metrics,
                teacher_metrics,
                tolerance=0.002,
            )
        )

    def test_retention_kl_stays_finite_for_extreme_logits(self) -> None:
        module = self._require_module()

        class ExtremeTeacher:
            def eval(self):
                return self

            def forward_legacy_heads(self, *inputs):
                logits = torch.tensor(
                    [[-1000.0, 1000.0]] * 4,
                    device=inputs[0].device,
                )
                return SimpleNamespace(
                    phla_logits=logits,
                    ptcr_logits=logits,
                )

        class ExtremeStudent:
            def eval(self):
                return self

            def forward_all(self, *inputs):
                logits = torch.tensor(
                    [[1000.0, -1000.0]] * 4,
                    device=inputs[0].device,
                )
                return SimpleNamespace(
                    phla_logits=logits,
                    ptcr_logits=logits,
                    pht_logits=torch.zeros(4, 2, device=inputs[0].device),
                )

        _, batch = self._models_and_batch()
        result = module.evaluate_stage10_models(
            ExtremeTeacher(),
            ExtremeStudent(),
            [batch],
            device=torch.device("cpu"),
            mixed_precision="none",
            margin=0.05,
            selection_weights=(0.6, 0.2, 0.2),
            degradation_tolerance=0.002,
        )

        self.assertTrue(np.isfinite(result.retention["phla_kl"]))
        self.assertTrue(np.isfinite(result.retention["ptcr_kl"]))


if __name__ == "__main__":
    unittest.main()
