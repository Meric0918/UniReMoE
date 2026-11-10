from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage16_18_helpers import fixture, pairs
from plm_unified.stage16_checkpoint import (
    atomic_torch_save,
    restore_training_checkpoint,
    write_json,
    checkpoint_sha256,
    load_torch_checkpoint,
    verify_original_encoders,
)
from plm_unified.stage18_teacher import teacher_identity, TeacherLogitCache
from plm_unified.stage18_loss import group_weights, window_pt_loss
from plm_unified.stage18_trainer import Stage18Trainer


def cache_for(c, d, path):
    identity = teacher_identity(c, d, "train")
    logits = torch.tensor([[0.0, 2.0] if r[2] else [2.0, 0.0] for r in d.rows])
    atomic_torch_save(
        {
            "identity": identity,
            "row_ids": d.row_ids,
            "input_keys": d.keys,
            "logits": logits,
        },
        path,
    )
    write_json(
        path.with_suffix(".json"),
        {"identity": identity, "sha256": checkpoint_sha256(path)},
    )
    return TeacherLogitCache(path, identity)


class Stage18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_weighted_smoothed_kd_window_matches_full_batch(self):
        torch.manual_seed(3)
        values = torch.randn(7, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0])
        weights = torch.tensor([0.25, 1.0, 4.0, 0.5, 1.0, 2.0, 0.25])
        teacher = torch.randn(7, 2, requires_grad=True)
        full = values.clone().requires_grad_()
        parts = values.clone().requires_grad_()
        loss, _ = window_pt_loss(
            full,
            labels,
            weights,
            weight_sum=float(weights.sum()),
            window_samples=7,
            smoothing=0.03,
            teacher_logits=teacher,
            alpha=0.5,
            temperature=2.0,
        )
        loss.backward()
        for ids in (slice(0, 4), slice(4, 6), slice(6, 7)):
            loss, _ = window_pt_loss(
                parts[ids],
                labels[ids],
                weights[ids],
                weight_sum=float(weights.sum()),
                window_samples=7,
                smoothing=0.03,
                teacher_logits=teacher[ids],
                alpha=0.5,
                temperature=2.0,
            )
            loss.backward()
        self.assertTrue(torch.allclose(full.grad, parts.grad, atol=1e-7))
        self.assertIsNone(teacher.grad)

    def test_kd_direction_temperature_and_alpha_zero(self):
        student = torch.zeros(1, 2, requires_grad=True)
        teacher = torch.tensor([[0.0, 3.0]])
        loss, parts = window_pt_loss(
            student,
            torch.tensor([1]),
            torch.ones(1),
            weight_sum=1,
            window_samples=1,
            teacher_logits=teacher,
            alpha=0.5,
            temperature=2.0,
        )
        expected = 4 * F.kl_div(
            F.log_softmax(student / 2, -1),
            F.softmax(teacher / 2, -1),
            reduction="batchmean",
        )
        self.assertTrue(torch.allclose(parts["kd"], expected))
        loss.backward()
        self.assertLess(student.grad[0, 1], 0)
        plain, _ = window_pt_loss(
            student,
            torch.tensor([1]),
            torch.ones(1),
            weight_sum=1,
            window_samples=1,
            alpha=0,
        )
        self.assertTrue(
            torch.allclose(plain, F.cross_entropy(student, torch.tensor([1])))
        )

    def test_group_label_counts_and_smoothing_target(self):
        rows = [("P", "T", 0)] * 16 + [("P", "T", 1)] + [("Q", "T", 0)]
        weights, stats = group_weights(rows)
        self.assertGreater(weights["P", 1], weights["P", 0])
        logits = torch.tensor([[0.0, 2.0]])
        loss, _ = window_pt_loss(
            logits,
            torch.tensor([1]),
            torch.ones(1),
            weight_sum=1,
            window_samples=1,
            smoothing=0.03,
        )
        expected = -(F.log_softmax(logits, -1) * torch.tensor([[0.015, 0.985]])).sum()
        self.assertTrue(torch.allclose(loss, expected))

    def test_cache_reorders_by_identity_and_rejects_role_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture(18)
            dataset = d["pairs"]["ptcr"]
            cache = cache_for(c, dataset, Path(temp) / "teacher.pt")
            ids = list(reversed(dataset.row_ids))
            keys = list(reversed(dataset.keys))
            self.assertTrue(torch.equal(cache.lookup(ids, keys), cache.logits.flip(0)))
            with self.assertRaises(ValueError):
                cache.lookup(ids, keys, role="reference")
            with self.assertRaises(ValueError):
                cache.lookup(ids, ["wrong"] * len(ids))
            wrong = {**cache.identity, "teacher_sha256": "x"}
            with self.assertRaises(ValueError):
                TeacherLogitCache(cache.path, wrong)

    def test_real_student_updates_teacher_cache_stays_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture(18)
            cache = cache_for(c, d["pairs"]["ptcr"], Path(temp) / "teacher.pt")
            before = deepcopy(m.state_dict())
            teacher = cache.logits.clone()
            trainer = Stage18Trainer(m, c, d, reference=r, teacher_cache=cache)
            trainer.train_until(stop_after_steps=trainer.steps_per_round())
            changed = [
                n for n, v in m.state_dict().items() if not torch.equal(v, before[n])
            ]
            self.assertTrue(any("_encoder." in n for n in changed))
            self.assertTrue(any("ptcr_classifier" in n for n in changed))
            self.assertTrue(any("triplet_" in n for n in changed))
            self.assertTrue(torch.equal(teacher, cache.logits))

    def test_kd_exact_resume_with_frozen_teacher_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture(18)
            cache = cache_for(c, d["pairs"]["ptcr"], Path(temp) / "teacher.pt")
            a = Stage18Trainer(
                m,
                c,
                d,
                reference=r,
                teacher_cache=cache,
                output_dir=Path(temp) / "full",
            )
            a.train_until()
            expected = deepcopy(m.state_dict())
            c, m, d, r = fixture(18)
            b = Stage18Trainer(
                m,
                c,
                d,
                reference=r,
                teacher_cache=cache,
                output_dir=Path(temp) / "resume",
            )
            b.train_until(stop_after_steps=2)
            c, m, d, r = fixture(18)
            resumed = Stage18Trainer(
                m,
                c,
                d,
                reference=r,
                teacher_cache=cache,
                output_dir=Path(temp) / "resume",
            )
            restore_training_checkpoint(resumed, Path(temp) / "resume/last.pt")
            resumed.train_until()
            self.assertTrue(
                all(torch.equal(v, m.state_dict()[k]) for k, v in expected.items())
            )

    def test_extreme_logits_equal_teacher_and_conditions(self):
        logits = torch.tensor(
            [[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True
        )
        loss, parts = window_pt_loss(
            logits,
            torch.tensor([0, 1]),
            torch.ones(2),
            weight_sum=2,
            window_samples=2,
            teacher_logits=logits.detach(),
            alpha=0.5,
            temperature=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(parts["kd"]), 0.0)
        for condition in (
            "S18-R0",
            "S18-R1",
            "S18-W",
            "S18-S",
            "S18-R2",
            "S18-R3",
            "S18-COMB",
            "S18-COMB-P",
        ):
            c, m, d, r = fixture(18, condition)
            self.assertEqual(c.policy.freeze_plm, condition.startswith("S18-COMB"))
            self.assertEqual(c.policy.protect_ph_peptide, condition == "S18-COMB-P")
        c, m, d, r = fixture(18, "S18-R1")
        c.policy_overrides = {"pt_replay_ratio": 2.0}
        trainer = Stage18Trainer(m, c, d, reference=r)
        self.assertEqual(trainer.cap("ptcr"), len(d["pairs"]["ptcr"]))

    def test_finite_but_modified_cache_logits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c, m, d, r = fixture(18)
            path = Path(temp) / "cache.pt"
            cache = cache_for(c, d["pairs"]["ptcr"], path)
            payload = load_torch_checkpoint(path)
            payload["logits"] = payload["logits"] + 0.1
            atomic_torch_save(payload, path)
            with self.assertRaisesRegex(ValueError, "integrity"):
                TeacherLogitCache(path, cache.identity)

    def test_teacher_rejects_finetuned_warmup_encoder_coordinates(self):
        c, model, data, reference = fixture(18)
        warmup = {"model": deepcopy(model.state_dict())}
        verify_original_encoders(model, warmup)
        warmup["model"]["peptide_encoder.blocks.0.weight"] += 0.1
        with self.assertRaisesRegex(ValueError, "original pretrained encoder"):
            verify_original_encoders(model, warmup)


if __name__ == "__main__":
    unittest.main()
