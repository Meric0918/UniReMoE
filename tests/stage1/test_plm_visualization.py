"""Synthetic tests for Stage-1/Stage-2 representation visualization."""

from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from source.plm_unified.config import ModelConfig
from source.plm_unified.model import UnifiedBindingModel
from source.visualize_plm_embeddings import _extract_cached_adapter_features
from source.plm_unified.visualization import (
    PairRecord,
    build_entity_metadata,
    fit_joint_projections,
    label_separation_summary,
    masked_mean,
    masked_mean_numpy,
    paired_shift_summary,
    read_pair_records,
    sample_pair_records,
    select_entity_sequences,
)


class PoolingTests(unittest.TestCase):
    def test_masked_mean_excludes_padding_in_torch_and_numpy(self):
        hidden = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
                [[2.0, 4.0], [200.0, 200.0], [300.0, 300.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        expected = torch.tensor([[2.0, 4.0], [2.0, 4.0]])
        self.assertTrue(torch.equal(masked_mean(hidden, mask), expected))
        np.testing.assert_allclose(
            masked_mean_numpy(hidden.numpy(), mask.numpy()),
            expected.numpy(),
        )

    def test_masked_mean_rejects_empty_sequence(self):
        with self.assertRaisesRegex(ValueError, "at least one real residue"):
            masked_mean(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, dtype=torch.bool),
            )

    def test_binding_output_exposes_exact_classifier_input(self):
        config = ModelConfig(
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
        )
        model = UnifiedBindingModel(config).eval()
        output = model(
            "phla",
            torch.randn(4, 5, 12),
            torch.ones(4, 5, dtype=torch.bool),
            torch.randn(4, 6, 12),
            torch.ones(4, 6, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.pooled_embedding.shape), (4, 8))
        self.assertTrue(
            torch.allclose(
                output.logits,
                model.phla_classifier(output.pooled_embedding),
            )
        )

    def test_cached_adapter_features_use_the_configured_256_width(self):
        class TinyCache:
            def resolve(self, sequence):
                return sequence

            def get_many(self, locators):
                hidden = np.zeros((len(locators), 3, 4), dtype=np.float32)
                hidden[:, :, 0] = 1.0
                mask = np.ones((len(locators), 3), dtype=np.bool_)
                return hidden, mask

        config = ModelConfig(
            peptide_input_dim=4,
            hla_input_dim=4,
            tcr_input_dim=4,
            adapter_hidden_dim=16,
            model_dim=256,
            num_attention_heads=8,
            feedforward_dim=32,
            adapter_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
        )
        model = UnifiedBindingModel(config).eval()
        features = _extract_cached_adapter_features(
            model,
            TinyCache(),
            ["AAA", "BBB"],
            "peptide",
            batch_size=2,
            device=torch.device("cpu"),
            precision="none",
        )
        self.assertEqual(features.shape, (2, 256))


class SamplingTests(unittest.TestCase):
    @staticmethod
    def _records(task: str, labels: list[int]) -> list[PairRecord]:
        return [
            PairRecord(
                sample_id=f"{task}:{index}",
                task=task,
                row_index=index + 2,
                peptide=f"PEPTIDE{index:03d}",
                receptor=f"RECEPTOR{index:03d}",
                label=label,
            )
            for index, label in enumerate(labels)
        ]

    def test_pair_sampling_is_proportional_and_reproducible(self):
        records = self._records("ptcr", [0] * 80 + [1] * 20)
        first = sample_pair_records(records, 50, random_state=42)
        second = sample_pair_records(records, 50, random_state=42)
        self.assertEqual(
            [record.sample_id for record in first],
            [record.sample_id for record in second],
        )
        self.assertEqual(Counter(record.label for record in first), {0: 40, 1: 10})
        self.assertEqual(len({record.sample_id for record in first}), 50)

    def test_csv_reader_uses_pipeline_filtering_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "val_fold_1.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["tcr", "peptide", "label"],
                )
                writer.writeheader()
                writer.writerow(
                    {"tcr": "CASSLG", "peptide": "ABCDEFGHI", "label": "1"}
                )
                writer.writerow(
                    {"tcr": "CASSXX", "peptide": "SHORT", "label": "0"}
                )
            records = read_pair_records(path, "ptcr")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sample_id, "ptcr:val_fold_1:2")
        self.assertEqual(records[0].peptide, "ABCDEFGHI")

    def test_entity_selection_is_shared_with_sampled_pairs(self):
        phla = [
            PairRecord(
                sample_id=f"phla:{index}",
                task="phla",
                row_index=index,
                peptide=f"PEPTIDE{index:02d}",
                receptor=f"HLA{index % 3}",
                label=index % 2,
            )
            for index in range(12)
        ]
        ptcr = [
            PairRecord(
                sample_id=f"ptcr:{index}",
                task="ptcr",
                row_index=index,
                peptide=f"ANTIGEN{index % 4:02d}",
                receptor=f"TCR{index:02d}",
                label=(index + 1) % 2,
            )
            for index in range(10)
        ]
        records = {"phla": phla, "ptcr": ptcr}
        metadata = build_entity_metadata(records)
        selected = select_entity_sequences(
            records,
            metadata,
            maximum=8,
            random_state=42,
        )
        self.assertEqual(len(selected["peptide"]), 8)
        self.assertEqual(set(selected["hla"]), {"HLA0", "HLA1", "HLA2"})
        self.assertEqual(len(selected["tcr"]), 8)
        candidate_peptides = {
            record.peptide for task_records in records.values() for record in task_records
        }
        self.assertTrue(set(selected["peptide"]).issubset(candidate_peptides))


class ProjectionTests(unittest.TestCase):
    def test_joint_projection_accepts_current_256_dimensional_features(self):
        rng = np.random.default_rng(256)
        stage1 = rng.normal(size=(32, 256)).astype(np.float32)
        stage2 = stage1 + 0.02 * rng.normal(size=stage1.shape).astype(np.float32)
        result = fit_joint_projections(
            {"stage1": stage1, "stage2": stage2},
            reducers=("pca",),
            feature_space="adapter",
            random_state=42,
            metric_sample_size=20,
        )
        self.assertEqual(result.summary["feature_dimension"], 256)
        self.assertEqual(result.summary["feature_space"], "adapter")
        self.assertEqual(result.coordinates["stage1"]["pca"].shape, (32, 2))
        self.assertEqual(result.coordinates["stage2"]["pca"].shape, (32, 2))

    def test_joint_pca_is_reproducible_and_reports_quality(self):
        rng = np.random.default_rng(42)
        stage1 = rng.normal(size=(60, 12)).astype(np.float32)
        stage2 = stage1 + 0.05 * rng.normal(size=stage1.shape).astype(np.float32)
        first = fit_joint_projections(
            {"stage1": stage1, "stage2": stage2},
            reducers=("pca",),
            random_state=42,
            metric_sample_size=40,
        )
        second = fit_joint_projections(
            {"stage1": stage1, "stage2": stage2},
            reducers=("pca",),
            random_state=42,
            metric_sample_size=40,
        )
        np.testing.assert_allclose(
            first.coordinates["stage1"]["pca"],
            second.coordinates["stage1"]["pca"],
        )
        self.assertEqual(first.coordinates["stage2"]["pca"].shape, (60, 2))
        self.assertIn(
            "pca_trustworthiness",
            first.summary["trustworthiness"]["stage1"],
        )
        self.assertEqual(len(first.summary["pca"]["explained_variance_ratio"]), 2)

    def test_shift_and_label_separation_summaries_are_finite(self):
        rng = np.random.default_rng(7)
        labels = np.asarray([0] * 30 + [1] * 30)
        features = rng.normal(size=(60, 8)).astype(np.float32)
        features[labels == 1, 0] += 2.0
        shifted = features + 0.01
        shift = paired_shift_summary(features, shifted)
        self.assertGreater(shift["cosine_similarity_mean"], 0.99)
        separation = label_separation_summary(
            features,
            {"pca": features[:, :2]},
            labels,
            random_state=42,
            maximum=50,
        )
        self.assertTrue(np.isfinite(separation["high_dimensional"]))
        self.assertTrue(np.isfinite(separation["pca"]))


if __name__ == "__main__":
    unittest.main()
