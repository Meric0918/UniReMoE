from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage16_18_helpers import fixture
from plm_unified.stage16_config import Selection, config_from_dict
from plm_unified.stage16_checkpoint import (
    restore_training_checkpoint,
    load_inference_checkpoint,
    rng_state,
)
from plm_unified.stage16_data import ReplayBatchSampler
from plm_unified.stage16_data import PairRows, collate_pairs
from plm_unified.stage16_evaluation import evaluate_all, summarize
from plm_unified.stage16_selection import retention_gate, ShadowSelectors, PT_SPLITS
from plm_unified.stage16_trainer import Stage16Trainer


class Stage16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gate_split_pht_nonfinite_and_counts(self):
        c, m, d, reference = fixture()
        candidate = deepcopy(reference)
        for s in PT_SPLITS:
            candidate["pt_independent"][s]["auroc"] = reference["pt_independent"][s][
                "auroc"
            ] = 0.8
        candidate["pt_independent"][PT_SPLITS[0]]["auroc"] = 0.77
        candidate["pt_independent"][PT_SPLITS[1]]["auroc"] = 0.83
        gate = retention_gate(candidate, reference, c.selection)
        self.assertIn(PT_SPLITS[0] + ":auroc", gate["failures"])
        self.assertNotIn("macro:auroc", gate["failures"])
        candidate["pt_independent"][PT_SPLITS[0]]["auroc"] = float("nan")
        candidate["pt_independent"][PT_SPLITS[1]]["samples"] += 1
        candidate["pht_val"]["pht"]["mcc"] = -1
        failures = retention_gate(candidate, reference, c.selection)["failures"]
        self.assertTrue(any("nonfinite" in k for k in failures))
        self.assertTrue(any("samples" in k for k in failures))
        self.assertIn("pht:performance_avg", failures)

    def test_selectors_keep_earlier_tie_and_allow_no_feasible(self):
        c, m, d, reference = fixture()
        selectors = ShadowSelectors(c.selection)
        decision = selectors.consider(reference, reference, 0)
        self.assertEqual(
            set(decision["improved"]), {"legacy", "pt_validation", "guarded"}
        )
        self.assertEqual(selectors.consider(reference, reference, 1)["improved"], [])
        self.assertEqual(selectors.states["guarded"].round, 0)
        broken = deepcopy(reference)
        for s in PT_SPLITS:
            broken["pt_independent"][s]["mcc"] = -1
        other = ShadowSelectors(c.selection)
        other.consider(broken, reference, 0)
        self.assertIsNone(other.states["guarded"].artifact)
        self.assertIsNone(other.states["guarded"].score)

    def test_eval_restores_modes_grad_flags_rng_and_next_update(self):
        c, m, d, reference = fixture()
        trainer = Stage16Trainer(m, c, d, reference=reference)
        m.set_trainable_pass("ptcr", c.policy)
        modes = [x.training for x in m.modules()]
        flags = [p.requires_grad for p in m.parameters()]
        state = rng_state()
        evaluate_all(m, d, c)
        self.assertEqual(modes, [x.training for x in m.modules()])
        self.assertEqual(flags, [p.requires_grad for p in m.parameters()])
        self.assertTrue(torch.equal(state["torch"], torch.get_rng_state()))
        self.assertEqual(state["python"], random.getstate())
        self.assertTrue(np.array_equal(state["numpy"][1], np.random.get_state()[1]))
        initial = deepcopy(m.state_dict())
        random_state = torch.get_rng_state()
        trainer.train_until(stop_after_steps=3)
        expected = deepcopy(m.state_dict())
        c2, m2, d2, r2 = fixture()
        m2.load_state_dict(initial)
        second = Stage16Trainer(m2, c2, d2, reference=r2)
        torch.set_rng_state(random_state)
        second.train_until(stop_after_steps=3)
        self.assertTrue(
            all(torch.equal(v, m2.state_dict()[k]) for k, v in expected.items())
        )

    def test_resume_mid_joint_exactly_matches_continuous(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture()
            a = Stage16Trainer(
                m, c, d, reference=r, output_dir=Path(temp) / "continuous"
            )
            a.train_until()
            expected = deepcopy(m.state_dict())
            c, m, d, r = fixture()
            b = Stage16Trainer(m, c, d, reference=r, output_dir=Path(temp) / "resume")
            self.assertFalse(b.train_until(stop_after_steps=3))
            self.assertEqual(b.cursor["pass_index"], 2)
            self.assertGreater(b.cursor["next_batch"], 0)
            c, m, d, r = fixture()
            resumed = Stage16Trainer(
                m, c, d, reference=r, output_dir=Path(temp) / "resume"
            )
            restore_training_checkpoint(resumed, Path(temp) / "resume/last.pt")
            self.assertTrue(resumed.train_until())
            self.assertTrue(
                all(torch.equal(v, m.state_dict()[k]) for k, v in expected.items())
            )
            self.assertEqual(a.global_step, resumed.global_step)
            self.assertEqual(a.scheduler.state_dict(), resumed.scheduler.state_dict())
            for key in a.selectors.states:
                self.assertEqual(
                    a.selectors.states[key].score, resumed.selectors.states[key].score
                )
            index = json.loads((Path(temp) / "resume/best_legacy.json").read_text())
            checkpoint, loaded_config = load_inference_checkpoint(
                Path(temp) / "resume" / index["path"]
            )
            self.assertEqual(loaded_config.contract(), c.contract())

    def test_nested_sampler_prefix_epoch_and_cursor(self):
        small = list(ReplayBatchSampler(40, 8, 4, 42, epoch=3))
        large = list(ReplayBatchSampler(40, 24, 4, 42, epoch=3))
        self.assertEqual(small, large[:2])
        self.assertEqual(
            list(ReplayBatchSampler(40, 24, 4, 42, epoch=3, start_batch=2)), large[2:]
        )
        self.assertNotEqual(large, list(ReplayBatchSampler(40, 24, 4, 42, epoch=4)))

    def test_strict_ties_and_unknown_config(self):
        metric = summarize([0, 1], [0.5, 0.5])
        self.assertEqual(metric["threshold_ties"], 2)
        self.assertEqual(
            metric["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 1, "tp": 0}
        )
        c, _, _, _ = fixture()
        raw = c.to_dict()
        raw["training"]["typo"] = 1
        with self.assertRaises(ValueError):
            config_from_dict(raw)

    def test_plain_ce_partial_window_matches_single_effective_batch(self):
        c, m, d, r = fixture(dropout=0)
        for module in m.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        a = Stage16Trainer(m, c, d, reference=r)
        m.set_trainable_pass("phla", c.policy)
        rows = [d["pairs"]["phla"][i] for i in range(7)]
        a._window(
            "phla",
            [collate_pairs(rows[:4]), collate_pairs(rows[4:])],
            {"samples": 0, "optimizer_steps": 0},
        )
        expected = deepcopy(m.state_dict())
        c, m, d, r = fixture(dropout=0)
        for module in m.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        b = Stage16Trainer(m, c, d, reference=r)
        m.set_trainable_pass("phla", c.policy)
        b._window("phla", [collate_pairs(rows)], {"samples": 0, "optimizer_steps": 0})
        self.assertTrue(
            all(
                torch.allclose(v, m.state_dict()[k], atol=1e-7)
                for k, v in expected.items()
            )
        )

    def test_bad_config_types_and_data_split_boundary(self):
        c, m, d, r = fixture()
        raw = c.to_dict()
        raw["training"]["rounds"] = True
        with self.assertRaises(ValueError):
            config_from_dict(raw)
        raw = c.to_dict()
        raw["training"]["mixed_precision"] = 7
        with self.assertRaises(ValueError):
            config_from_dict(raw)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test_unseen_pair.csv"
            path.write_text("peptide,tcr,label\nACDE,IKLM,1\n")
            with self.assertRaisesRegex(ValueError, "Test files"):
                PairRows(path, "ptcr", c.model)
            path = Path(temp) / "train.csv"
            path.write_text("peptide,tcr,label\nACDE,IKLM,1\nACDE,IKLM,0\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                PairRows(path, "ptcr", c.model)

    def test_resume_rejects_mutated_contract_and_null_best_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture()
            for split in PT_SPLITS:
                r["pt_independent"][split]["mcc"] = 1.0
            t = Stage16Trainer(m, c, d, reference=r, output_dir=temp)
            t.train_until(stop_after_steps=1)
            self.assertIsNone(t.selectors.states["guarded"].score)
            self.assertFalse((Path(temp) / "best_guarded.json").exists())
            c, m, d, r = fixture()
            u = Stage16Trainer(m, c, d, reference=r, output_dir=temp)
            restore_training_checkpoint(u, Path(temp) / "last.pt")
            self.assertIsNone(u.selectors.states["guarded"].score)
            u.config.training.adapter_lr *= 2
            with self.assertRaisesRegex(ValueError, "contract"):
                restore_training_checkpoint(u, Path(temp) / "last.pt")

    def test_terminal_early_stop_resume_does_not_train_extra_round(self):
        c, m, d, r = fixture()
        c.training.deployment_early_stop = True
        c.training.min_rounds = 1
        c.training.early_stopping_patience = 1
        t = Stage16Trainer(m, c, d, reference=r)
        t.completed_round = 1
        t.cursor = {"round": 2, "pass_index": 0, "next_batch": 0}
        t.selectors.states["guarded"].bad_rounds = 1
        before = deepcopy(m.state_dict())
        self.assertTrue(t.train_until())
        self.assertEqual(t.global_step, 0)
        self.assertTrue(
            all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
        )

    def test_other_seed_common_warmup_generation_and_loading(self):
        from plm_unified.stage16_warmup import generate_warmup
        from plm_unified.stage16_checkpoint import load_common_warmup, checkpoint_sha256

        c, m, d, r = fixture()
        d["manifest"]["pht_fingerprint"] = "f" * 64
        source_config = {
            "model": asdict(c.model),
            "training": {"seed": c.runtime.seed, "fold": c.runtime.fold},
        }
        metadata = {
            e: {"model": {"checkpoint_fingerprint": "fingerprint"}}
            for e in ("peptide", "hla", "tcr")
        }
        joint = {
            "checkpoint_type": "stage1_joint_model",
            "config": source_config,
            "cache_metadata": metadata,
            "model": {
                k: v.clone()
                for k, v in m.binding_model.state_dict().items()
                if not k.startswith("triplet_")
            },
        }
        teacher = {"task": "ptcr", "config": source_config, "cache_metadata": metadata}

        def checksum(path):
            if path == c.paths.stage1_checkpoint:
                return c.paths.stage1_sha256
            if path == c.paths.teacher_checkpoint:
                return c.paths.teacher_sha256
            return checkpoint_sha256(path)

        namespace = "plm_unified.stage16_warmup."
        with tempfile.TemporaryDirectory() as temp, patch(
            namespace + "checkpoint_sha256", side_effect=checksum
        ), patch(
            namespace + "checkpoint_fingerprint", return_value="fingerprint"
        ), patch(
            namespace + "load_torch_checkpoint", side_effect=[joint, teacher]
        ), patch(
            namespace + "audit_experiment_data", return_value=d
        ), patch(
            namespace + "build_stage17_model", return_value=m
        ):
            output = generate_warmup(c, Path(temp) / "warmup.pt")
            saved = load_common_warmup(output)
            self.assertIn(saved["training_state"]["stage8a_epoch"], (1, 2))
            self.assertTrue(
                all(
                    torch.equal(v, saved["model"]["binding_model." + k])
                    for k, v in joint["model"].items()
                )
            )
            self.assertTrue(output.with_suffix(".ready_config.json").is_file())

    def test_cache_hit_and_miss_leave_identical_training_trajectory(self):
        from plm_unified.stage16_pipeline import main
        from plm_unified.stage16_checkpoint import load_torch_checkpoint

        c, model, data, reference = fixture()
        c.training.rounds = 1
        namespace = "plm_unified.stage16_pipeline."
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for cache_miss in (True, False):

                def prepare(*args, **kwargs):
                    if cache_miss:
                        torch.randn(137)
                    return deepcopy(reference), None

                output = Path(temp) / str(cache_miss)
                with patch(namespace + "configured", return_value=deepcopy(c)), patch(
                    namespace + "audit_experiment_data", return_value=data
                ), patch(
                    namespace + "validate_sources",
                    return_value={"training_state": {"stage8a_epoch": 1}},
                ), patch(
                    namespace + "caches_and_reference", side_effect=prepare
                ), patch(
                    namespace + "build_student", return_value=deepcopy(model)
                ):
                    main(16, ["--output-dir", str(output)])
                outputs.append(load_torch_checkpoint(output / "last.pt")["model"])
            self.assertTrue(
                all(torch.equal(v, outputs[1][k]) for k, v in outputs[0].items())
            )


if __name__ == "__main__":
    unittest.main()
