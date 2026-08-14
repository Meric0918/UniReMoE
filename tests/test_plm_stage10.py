"""Contract tests for Stage-10 teacher-student retention training."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import torch

from source.plm_unified.config import ModelConfig
from source.plm_unified.stage8_loss import compute_stage8_loss
from source.plm_unified.stage8_model import Stage8BindingModel


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


if __name__ == "__main__":
    unittest.main()
