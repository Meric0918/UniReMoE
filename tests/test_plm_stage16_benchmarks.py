from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage16_18_helpers import fixture
from plm_unified.stage16_benchmarks import (
    cache_residues,
    lock_checkpoints,
    verify_lock,
    main,
)
from plm_unified.stage16_checkpoint import atomic_torch_save, inference_payload
from plm_unified.stage16_checkpoint import load_inference_checkpoint
from plm_unified.stage16_trainer import Stage16Trainer
from plm_unified.stage16_evaluation import predict_pairs, evaluate_joint
from plm_unified.stage16_analysis import paired_cluster_bootstrap


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_online_cached_outputs_match_and_bootstrap_alignment(self):
        c, m, d, r = fixture()
        dataset = d["pairs"]["ptcr"]
        online = predict_pairs(m, dataset, c)
        cached = cache_residues(
            m,
            {
                "peptide": {x[0] for x in dataset.rows},
                "tcr": {x[1] for x in dataset.rows},
            },
            c,
            8,
        )
        result = predict_pairs(cached, dataset, c)
        self.assertTrue(
            torch.allclose(
                torch.tensor(result["logits"]),
                torch.tensor(online["logits"]),
                atol=1e-6,
            )
        )
        bootstrap = paired_cluster_bootstrap(online, deepcopy(online), replicates=50)
        self.assertTrue(all(abs(v) < 1e-10 for v in bootstrap["difference"].values()))
        wrong = deepcopy(online)
        wrong["row_ids"][0] = "wrong"
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(online, wrong, replicates=5)

    def test_locked_cli_with_temporary_benchmarks_and_full_student_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, m, d, r = fixture()
            c.paths.data_root = root / "dataset"
            t = Stage16Trainer(m, c, d, reference=r)
            t.completed_round = 0
            artifact = root / "best.pt"
            payload = inference_payload(t, r, role="best", selectors=["guarded"])
            atomic_torch_save(payload, artifact)
            lock = root / "selection.json"
            lock_checkpoints([artifact], lock)
            self.assertEqual(verify_lock(artifact, lock), artifact)
            for family, receptor in (("PT", "tcr"), ("PH", "HLA")):
                target = root / "dataset" / family / "new"
                target.mkdir(parents=True)
                (target / "test_unseen_pair.csv").write_text(
                    f"peptide,{receptor},label\nACDE,IKLM,0\nCDEF,KLMN,1\n"
                )
            target = root / "dataset/PHT/benchmarks"
            target.mkdir(parents=True)
            (target / "test_unseen_triple.csv").write_text(
                "peptide,tcr,HLA,PH,PT,PHT\nACDE,IKLM,PQRS,0,0,0\nCDEF,KLMN,PQRS,0,1,0\nDEFG,LMNP,PQRS,1,0,0\nEFGH,MNPQ,PQRS,1,1,1\n"
            )
            with patch(
                "plm_unified.stage16_benchmarks.build_stage17_model", return_value=m
            ):
                report = main(
                    16,
                    [
                        "--checkpoint",
                        str(artifact),
                        "--selection-manifest",
                        str(lock),
                        "--output-prefix",
                        str(root / "results"),
                        "--device",
                        "cpu",
                        "--encode-batch-size",
                        "8",
                        "--row-batch-size",
                        "8",
                    ],
                )
            self.assertEqual(
                set(report["results"]),
                {
                    "PT/test_unseen_pair",
                    "PH/test_unseen_pair",
                    "PHT/test_unseen_triple",
                },
            )
            self.assertTrue((root / "results.predictions.pt").exists())
            payload["debug"] = True
            atomic_torch_save(payload, root / "debug.pt")
            with self.assertRaises(ValueError):
                lock_checkpoints([root / "debug.pt"], root / "debug-lock.json")
            with self.assertRaises(ValueError):
                lock_checkpoints([artifact], lock)

    def test_finetuned_encoder_and_all_heads_export_roundtrip(self):
        c, model, data, reference = fixture()
        before = deepcopy(model.state_dict())
        trainer = Stage16Trainer(model, c, data, reference=reference)
        trainer.train_until()
        self.assertTrue(
            any(
                not torch.equal(value, before[name])
                for name, value in model.state_dict().items()
                if "_encoder." in name
            )
        )
        pt = predict_pairs(model, data["pairs"]["ptcr"], c)["logits"]
        joint = evaluate_joint(model, data["joint_val"], c)[1]["probabilities"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected.pt"
            atomic_torch_save(
                inference_payload(
                    trainer, trainer.last_metrics, role="best", selectors=["legacy"]
                ),
                path,
            )
            checkpoint, loaded_config = load_inference_checkpoint(
                path, expected_stage=16
            )
            _, restored, _, _ = fixture()
            restored.load_state_dict(checkpoint["model"], strict=True)
            self.assertTrue(
                (
                    pt
                    == predict_pairs(restored, data["pairs"]["ptcr"], loaded_config)[
                        "logits"
                    ]
                ).all()
            )
            restored_joint = evaluate_joint(restored, data["joint_val"], loaded_config)[
                1
            ]["probabilities"]
            self.assertTrue(
                all((joint[head] == restored_joint[head]).all() for head in joint)
            )


if __name__ == "__main__":
    unittest.main()
