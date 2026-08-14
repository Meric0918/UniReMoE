"""Contract tests for Stage-10 teacher-student retention training."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import torch

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


if __name__ == "__main__":
    unittest.main()
