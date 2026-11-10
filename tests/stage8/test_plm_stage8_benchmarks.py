"""Unit tests for the Stage-8 benchmark evaluation/reporting helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from source import evaluate_plm_stage8_benchmarks as evaluator


def _metrics(auroc: float, *, samples: int = 4) -> dict[str, float | int]:
    return {
        "loss": 0.2,
        "auroc": auroc,
        "aupr": auroc - 0.01,
        "accuracy": auroc - 0.02,
        "mcc": auroc - 0.03,
        "f1": auroc - 0.04,
        "precision": auroc - 0.05,
        "recall": auroc - 0.06,
        "sensitivity": auroc - 0.06,
        "specificity": auroc - 0.07,
        "samples": samples,
        "positive_samples": samples // 2,
        "negative_samples": samples // 2,
        "positive_rate": 0.5,
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "threshold": 0.5,
        "performance_avg": auroc - 0.02,
    }


class Stage8BenchmarkHelperTests(unittest.TestCase):
    def test_finish_metrics_keeps_fixed_threshold_and_confusion_counts(self) -> None:
        result = evaluator.finish_metrics(
            [np.array([0, 1, 0, 1])],
            [np.array([0.1, 0.8, 0.6, 0.9])],
            loss_sum=0.8,
            samples=4,
            threshold=0.5,
        )

        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["positive_samples"], 2)
        self.assertEqual(result["negative_samples"], 2)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["threshold"], 0.5)

    def test_constraint_metrics_uses_stage8_directed_implications(self) -> None:
        result = evaluator.compute_constraint_metrics(
            np.array(["111", "100", "010", "000"]),
            phla=np.array([0.4, 0.4, 0.7, 0.4]),
            ptcr=np.array([0.3, 0.4, 0.4, 0.2]),
            pht=np.array([0.6, 0.8, 0.8, 0.4]),
            margin=0.05,
        )

        self.assertEqual(result["eligible"]["hierarchy_positive"], 1)
        self.assertEqual(result["eligible"]["hierarchy_negative"], 2)
        self.assertEqual(result["eligible"]["pt_positive"], 1)
        self.assertEqual(result["eligible"]["pt_negative"], 2)
        self.assertAlmostEqual(result["violations"]["hierarchy_positive"], 1.0)
        self.assertAlmostEqual(result["violations"]["hierarchy_negative"], 0.5)
        self.assertAlmostEqual(result["violations"]["pt_positive"], 1.0)
        self.assertAlmostEqual(result["violations"]["pt_negative"], 1.0)
        self.assertGreater(result["constraint_components"]["pt_positive"], 0.0)

    def test_single_report_markdown_mentions_condition_and_heads(self) -> None:
        report = {
            "condition": "full",
            "checkpoint": "/tmp/stage8_best_full.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_type": "stage8_online_model",
            "format_version": 2,
            "config": "/tmp/plm_stage8.yaml",
            "device": "cpu",
            "mixed_precision": "bf16",
            "encode_batch_size": 16,
            "row_batch_size": 32,
            "fixed_threshold": 0.5,
            "checkpoint_selected_threshold": 0.6,
            "best_round": 11,
            "best_round_score": 0.88,
            "validation_joint_score": 0.88,
            "constraint_margin": 0.05,
            "results": {
                "PH/test_unseen_hla": {
                    "family": "PH",
                    "benchmark": "test_unseen_hla",
                    "path": "/tmp/test_unseen_hla.csv",
                    "head": "phla",
                    "raw_rows": 4,
                    "samples": 4,
                    "filter_stats": {"total": 4, "kept": 4},
                    "metrics": {"fixed_0.5": _metrics(0.80)},
                },
                "PHT/test_unseen_triple": {
                    "family": "PHT",
                    "benchmark": "test_unseen_triple",
                    "path": "/tmp/test_unseen_triple.csv",
                    "head": "phla,ptcr,pht",
                    "raw_rows": 4,
                    "samples": 4,
                    "filter_stats": {"total": 4, "kept": 4},
                    "label_counts": {
                        "phla": {"positive": 2, "negative": 2},
                        "ptcr": {"positive": 2, "negative": 2},
                        "pht": {"positive": 2, "negative": 2},
                    },
                    "metrics": {
                        head: {
                            "fixed_0.5": _metrics(0.81),
                            "checkpoint_selected": _metrics(0.82),
                        }
                        for head in ("phla", "ptcr", "pht")
                    },
                    "constraints": {
                        "eligible": {
                            "hierarchy_positive": 1,
                            "hierarchy_negative": 2,
                            "pt_positive": 1,
                            "pt_negative": 2,
                        },
                        "violations": {
                            "hierarchy_positive": 0.1,
                            "hierarchy_negative": 0.1,
                            "pt_positive": 0.1,
                            "pt_negative": 0.1,
                            "overall": 0.1,
                        },
                        "constraint_components": {
                            "hierarchy_positive": 0.01,
                            "hierarchy_negative": 0.01,
                            "pt_positive": 0.01,
                            "pt_negative": 0.01,
                            "overall": 0.01,
                        },
                    },
                },
            },
            "totals": {"raw_rows": 8, "samples": 8, "filtered": 0},
            "unique_sequences": {"peptide": 3, "hla": 2, "tcr": 2},
            "outputs": {"json": "/tmp/full.json", "markdown": "/tmp/full.md"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            text = evaluator.make_markdown(report, Path(tmp) / "full.md")

        self.assertIn("Stage 8", text)
        self.assertIn("`full`", text)
        self.assertIn("`phla`", text)
        self.assertIn("`ptcr`", text)
        self.assertIn("`pht`", text)
        self.assertIn("约束", text)

    def test_comparison_markdown_reports_best_condition_and_signed_delta(self) -> None:
        def make_report(condition: str, score: float) -> dict:
            return {
                "condition": condition,
                "checkpoint": f"/tmp/{condition}.pt",
                "checkpoint_sha256": "b" * 64,
                "best_round": 11,
                "best_round_score": score,
                "validation_joint_score": score,
                "totals": {"raw_rows": 4, "samples": 4, "filtered": 0},
                "unique_sequences": {"peptide": 2, "hla": 1, "tcr": 1},
                "results": {
                    "PH/test_unseen_hla": {
                        "family": "PH",
                        "benchmark": "test_unseen_hla",
                        "head": "phla",
                        "raw_rows": 4,
                        "samples": 4,
                        "metrics": {"fixed_0.5": _metrics(score)},
                    }
                },
            }

        reports = {
            "baseline": make_report("baseline", 0.80),
            "full": make_report("full", 0.85),
        }
        text = evaluator.make_comparison_markdown(reports, Path("/tmp/compare.md"))

        self.assertIn("Stage 8", text)
        self.assertIn("baseline", text)
        self.assertIn("full", text)
        self.assertIn("最佳", text)
        self.assertIn("+0.050000", text)


if __name__ == "__main__":
    unittest.main()
