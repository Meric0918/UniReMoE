"""Synthetic tests for isolated Stage-7 counterfactual evaluation."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

try:
    import numpy as np
    import torch

    import networkx  # noqa: F401
    import sklearn  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "Stage-7 test dependencies are required")
class Stage7PipelineTests(unittest.TestCase):
    @staticmethod
    def _config(root: Path):
        from source.plm_unified.config import ModelConfig
        from source.plm_unified.stage7_config import (
            Stage7Config,
            Stage7ExperimentConfig,
            Stage7Paths,
            Stage7RuntimeConfig,
        )

        return Stage7ExperimentConfig(
            paths=Stage7Paths(
                project_root=root,
                data_root=root / "data",
                output_root=root / "trained_model" / "PLM_Stage7",
                esmc_model=root / "models" / "esmc",
                tcr_bert_model=root / "models" / "tcr",
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
                peptide_min_length=8,
                peptide_max_length=8,
                hla_max_length=8,
                tcr_max_length=8,
            ),
            runtime=Stage7RuntimeConfig(fold=4, seed=2026, device="cpu"),
            stage7=Stage7Config(
                max_decoys_per_anchor=2,
                include_all_tetrads=False,
                include_legacy_hla_shift=False,
                encoder_batch_size=2,
                score_batch_size=2,
                bootstrap_samples=100,
                permutation_samples=1000,
            ),
        )

    @staticmethod
    def _write_data(root: Path) -> None:
        triple = root / "data" / "data_triple"
        phla = root / "data" / "data_HLA"
        triple.mkdir(parents=True)
        phla.mkdir(parents=True)
        header = "peptide,tcr,hla,label\n"
        train = (
            "CCCCCCCC,CASSCCCC,GGGGGGGG,0\n"
            "DDDDDDDD,CASSTTTT,HHHHHHHH,1\n"
            "EEEEEEEE,CASSGGGG,YYYYYYYY,0\n"
        )
        val = (
            "AAAAAAAA,CASSAAAA,GGGGGGGG,1\n"
            "AAAAAAAA,CASSTTTT,HHHHHHHH,1\n"
            "FFFFFFFF,CASSFFFF,YYYYYYYY,0\n"
        )
        test = (
            "AAAAAAAA,CASSDDDD,GGGGGGGG,1\n"
            "AAAAAAAA,CASSSSSS,HHHHHHHH,1\n"
            "GGGGGGGG,CASSYYYY,YYYYYYYY,0\n"
        )
        (triple / "train_set.csv").write_text(header + train, encoding="utf-8")
        (triple / "val_set.csv").write_text(header + val, encoding="utf-8")
        (triple / "test_set.csv").write_text(header + test, encoding="utf-8")
        (triple / "merged_pseudo.csv").write_text(
            header + train + val + test, encoding="utf-8"
        )
        (phla / "train_fold_4.csv").write_text(
            "peptide,HLA,label\n"
            "AAAAAAAA,GGGGGGGG,1\n"
            "AAAAAAAA,HHHHHHHH,1\n"
            "AAAAAAAA,YYYYYYYY,1\n",
            encoding="utf-8",
        )

    def test_counterfactual_pairs_and_tetrads_are_audited(self):
        from source.plm_unified.stage7_counterfactual import (
            build_stage7_counterfactual_bundle,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_data(root)
            bundle = build_stage7_counterfactual_bundle(self._config(root))
            test_rows = bundle.members["test"]
            original_rows = [
                row for row in test_rows if row.experiment == "original_triplet"
            ]
            self.assertEqual(len(original_rows), 3)
            pair_rows = [
                row for row in test_rows if row.experiment == "phla_valid_pair"
            ]
            self.assertEqual(len({row.group_id for row in pair_rows}), 2)
            for group_id in {row.group_id for row in pair_rows}:
                group = [row for row in pair_rows if row.group_id == group_id]
                cognate = next(row for row in group if row.role == "cognate")
                decoys = [row for row in group if row.role.startswith("decoy_")]
                self.assertTrue(decoys)
                self.assertTrue(all(row.hla != cognate.hla for row in decoys))
                self.assertTrue(
                    all(row.evidence.startswith("phla_train_positive:") for row in decoys)
                )
            tetrads = [
                row for row in test_rows if row.experiment == "tetrad_disjoint"
            ]
            self.assertEqual(len({row.group_id for row in tetrads}), 1)
            self.assertEqual(
                {row.role for row in tetrads},
                {"diag_1", "diag_2", "offdiag_12", "offdiag_21"},
            )
            audit = bundle.audit["splits"]["test"]
            self.assertEqual(audit["original_triplets"]["rows"], 3)
            self.assertEqual(audit["phla_valid_pairs"]["anchors_with_decoy"], 2)
            self.assertEqual(audit["tetrads"]["disjoint_tetrads"], 1)

    def _write_checkpoints(self, root: Path, config):
        from source.plm_unified.encoders import checkpoint_fingerprint
        from source.plm_unified.model import TaskSpecificBindingModel
        from source.plm_unified.triplet_model import TripletBindingModel

        config.paths.esmc_model.mkdir(parents=True)
        config.paths.tcr_bert_model.mkdir(parents=True)
        (config.paths.esmc_model / "config.json").write_text("{}", encoding="utf-8")
        (config.paths.tcr_bert_model / "config.json").write_text("{}", encoding="utf-8")
        esmc = checkpoint_fingerprint(config.paths.esmc_model)
        tcr = checkpoint_fingerprint(config.paths.tcr_bert_model)
        saved_config = {
            "model": asdict(config.model),
            "training": {"fold": 4, "seed": 2026},
        }
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        paths = {}
        for task in ("phla", "ptcr"):
            model = TaskSpecificBindingModel(config.model, task)
            cache = {
                "peptide": {"model": {"checkpoint_fingerprint": esmc}},
                ("hla" if task == "phla" else "tcr"): {
                    "model": {
                        "checkpoint_fingerprint": esmc if task == "phla" else tcr
                    }
                },
            }
            value = {
                "checkpoint_type": "task_model",
                "task": task,
                "config": saved_config,
                "cache_metadata": cache,
                "model": model.state_dict(),
            }
            path = checkpoint_dir / f"{task}.pt"
            torch.save(value, path)
            paths[task] = path
        binding = TripletBindingModel(config.model)
        stage3 = {
            "checkpoint_type": "stage3_multitask_model",
            "config": saved_config,
            "initialization_source": {
                "kind": "stage1",
                "path": str(checkpoint_dir / "stage1_last.pt"),
                "source_fold": 4,
                "source_seed": 2026,
            },
            "encoder_metadata": {
                "peptide": {"checkpoint_fingerprint": esmc},
                "hla": {"checkpoint_fingerprint": esmc},
                "tcr": {"checkpoint_fingerprint": tcr},
            },
            "model": {
                f"binding_model.{name}": value
                for name, value in binding.state_dict().items()
            },
        }
        for entity in ("peptide", "hla", "tcr"):
            stage3["model"][f"{entity}_encoder.synthetic"] = torch.zeros(1)
        paths["stage3"] = checkpoint_dir / "stage3.pt"
        torch.save(stage3, paths["stage3"])
        return paths

    def test_checkpoint_loading_is_strict(self):
        from source.plm_unified.stage7_inference import load_stage7_models

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            paths = self._write_checkpoints(root, config)
            models = load_stage7_models(
                config,
                stage3_checkpoint=paths["stage3"],
                stage1_phla_checkpoint=paths["phla"],
                stage1_ptcr_checkpoint=paths["ptcr"],
                device=torch.device("cpu"),
            )
            self.assertEqual(models.stage1_phla.task, "phla")
            self.assertEqual(models.stage1_ptcr.task, "ptcr")
            self.assertEqual(len(models.checkpoint_manifest["stage3"]["sha256"]), 64)
            self.assertEqual(models.stage3_checkpoint_path, paths["stage3"])
            config.runtime.seed = 42
            with self.assertRaisesRegex(ValueError, "seed"):
                load_stage7_models(
                    config,
                    stage3_checkpoint=paths["stage3"],
                    stage1_phla_checkpoint=paths["phla"],
                    stage1_ptcr_checkpoint=paths["ptcr"],
                    device=torch.device("cpu"),
                )

    @staticmethod
    def _analysis_rows():
        from source.plm_unified.stage7_inference import SCORE_COLUMNS

        rows = []
        for split in ("val", "test"):
            for index in range(6):
                label = index % 2
                row = {
                    "split": split,
                    "experiment": "original_triplet",
                    "group_id": f"{split}:original:{index}",
                    "role": "original",
                    "peptide": f"PEPTIDE{index}",
                    "peptide_group": f"PEPTIDE{index}",
                    "tcr": f"TCR{index}",
                    "hla": f"HLA{index}",
                    "label": label,
                    "source_row": index + 2,
                    "evidence": "synthetic",
                }
                row.update({column: 0.0 for column in SCORE_COLUMNS})
                correct_margin = 2.0 if label else -2.0
                wrong_margin = -correct_margin
                row["stage1_ptcr_margin"] = wrong_margin
                row["stage1_late_margin"] = wrong_margin
                row["stage3_ptcr_margin"] = wrong_margin
                row["stage3_late_margin"] = 0.5 * correct_margin
                row["stage3_pht_margin"] = correct_margin
                rows.append(row)
        for index in range(6):
            common = {
                "split": "test",
                "peptide": f"PEPTIDE{index}",
                "peptide_group": f"PEPTIDE{index}",
                "tcr": f"TCR{index}",
                "label": 1,
                "source_row": index + 2,
                "evidence": "synthetic",
            }
            for role, hla, pht in (
                ("cognate", "H1", 2.0),
                ("decoy_1", "H2", 0.0),
            ):
                row = {
                    **common,
                    "experiment": "phla_valid_pair",
                    "group_id": f"test:pair:{index}",
                    "role": role,
                    "hla": hla,
                }
                row.update({column: 0.0 for column in SCORE_COLUMNS})
                row["stage3_pht_margin"] = pht
                rows.append(row)
            tetrad_values = (
                ("diag_1", "T1", "H1", 2.0),
                ("diag_2", "T2", "H2", 2.0),
                ("offdiag_12", "T1", "H2", 0.0),
                ("offdiag_21", "T2", "H1", 0.0),
            )
            for role, tcr, hla, pht in tetrad_values:
                row = {
                    **common,
                    "experiment": "tetrad_disjoint",
                    "group_id": f"test:tetrad:{index}",
                    "role": role,
                    "tcr": tcr,
                    "hla": hla,
                }
                row.update({column: 0.0 for column in SCORE_COLUMNS})
                row["stage3_pht_margin"] = pht
                rows.append(row)
        return rows

    def test_pair_bootstrap_tetrad_cancellation_and_conclusion(self):
        from source.plm_unified.stage7_analysis import analyse_stage7_rows

        rows = self._analysis_rows()
        summary = analyse_stage7_rows(
            rows,
            {"splits": {"test": {"shortcut_lookup": {}}}},
            bootstrap_samples=200,
            permutation_samples=5000,
            seed=2026,
        )
        pair = summary["splits"]["test"]["phla_valid_pairs"]
        self.assertEqual(
            pair["models"]["stage1_ptcr_margin"]["paired_win_rate"]["point"],
            0.5,
        )
        tetrad = summary["splits"]["test"]["tetrad_disjoint"]
        self.assertEqual(tetrad["additive_cancellation_max_abs"]["stage3_late_margin"], 0.0)
        self.assertGreater(
            tetrad["models"]["stage3_pht_margin"]["peptide_macro_d"]["ci95"][0],
            0,
        )
        self.assertTrue(summary["conclusion"]["stage3_pht_outperforms_stage1_ptcr"])
        self.assertTrue(summary["conclusion"]["non_additive_tcr_hla_compatibility"])
        self.assertTrue(summary["conclusion"]["preliminary"])

    def test_aligned_run_average_is_deterministic(self):
        from source.plm_unified.stage7_analysis import average_aligned_runs
        from source.plm_unified.stage7_inference import SCORE_COLUMNS

        first = self._analysis_rows()
        second = [{**row} for row in first]
        for row in second:
            for column in SCORE_COLUMNS:
                row[column] += 2.0
        averaged = average_aligned_runs([first, second])
        self.assertEqual(len(averaged), len(first))
        self.assertAlmostEqual(
            averaged[0]["stage1_phla_margin"],
            first[0]["stage1_phla_margin"] + 1.0,
        )

    def test_output_root_cannot_target_another_stage(self):
        from source.evaluate_plm_stage7 import _resolve_output_root

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self.assertEqual(
                _resolve_output_root(config, None),
                config.paths.output_root.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "another trained_model stage"):
                _resolve_output_root(
                    config,
                    root / "trained_model" / "PLM_Stage5" / "bad",
                )


if __name__ == "__main__":
    unittest.main()
