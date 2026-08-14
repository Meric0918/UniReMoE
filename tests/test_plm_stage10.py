"""Contract tests for Stage-10 teacher-student retention training."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


try:
    stage10_config = importlib.import_module(
        "source.plm_unified.stage10_config"
    )
except ModuleNotFoundError:
    stage10_config = None


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


if __name__ == "__main__":
    unittest.main()
