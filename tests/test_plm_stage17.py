from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage16_18_helpers import fixture
from plm_unified.stage16_data import joint_loader, pair_loader
from plm_unified.stage16_evaluation import predict_pairs
from plm_unified.stage16_trainer import Stage16Trainer, build_optimizer
from plm_unified.stage8_loss import compute_stage8_loss


class Stage17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_condition_permissions_and_optimizer_union(self):
        for condition in ("S17-U", "S17-F", "S17-J", "S17-P", "S17-Z"):
            c, m, d, r = fixture(17, condition)
            optimizer = build_optimizer(m, c)
            union = set()
            for task in ("phla", "ptcr", "joint"):
                m.set_trainable_pass(task, c.policy)
                active = {n for n, p in m.named_parameters() if p.requires_grad}
                union.update(id(p) for p in m.parameters() if p.requires_grad)
                if c.policy.freeze_plm:
                    self.assertFalse(any("_encoder." in n for n in active))
                if task == "joint" and c.policy.protect_joint:
                    self.assertTrue(
                        all(n.startswith("binding_model.triplet_") for n in active)
                    )
                if condition == "S17-P" and task == "phla":
                    self.assertFalse(any("peptide_adapter" in n for n in active))
            self.assertEqual(
                union,
                {id(p) for group in optimizer.param_groups for p in group["params"]},
            )

    def test_joint_preserves_pt_with_existing_optimizer_momentum(self):
        c, m, d, r = fixture(17, "S17-J")
        trainer = Stage16Trainer(m, c, d, reference=r)
        m.set_trainable_pass("ptcr", c.policy)
        batch = next(
            iter(pair_loader(d["pairs"]["ptcr"], c, cap=8, epoch=1, train=True))
        )
        stats = {"samples": 0, "optimizer_steps": 0}
        trainer._window("ptcr", [batch], stats)
        before = deepcopy(m.state_dict())
        pred = predict_pairs(m, d["pairs"]["ptcr"], c)["logits"]
        m.set_trainable_pass("joint", c.policy)
        batch = next(iter(joint_loader(d["joint_train"], c, train=True)))
        trainer._window("joint", [batch], {"samples": 0, "optimizer_steps": 0})
        changed = [
            n for n, v in m.state_dict().items() if not torch.equal(v, before[n])
        ]
        self.assertTrue(changed)
        self.assertTrue(all(n.startswith("binding_model.triplet_") for n in changed))
        self.assertTrue(
            (pred == predict_pairs(m, d["pairs"]["ptcr"], c)["logits"]).all()
        )

    def test_zero_unfrozen_layers_do_not_activate_whole_encoder(self):
        c, m, d, r = fixture(17, "S17-U")
        for entity in ("peptide", "hla", "tcr"):
            setattr(m, entity + "_unfrozen_layers", 0)
        m.set_trainable_pass("joint", c.policy)
        self.assertFalse(
            any(
                p.requires_grad
                for name, p in m.named_parameters()
                if "_encoder." in name
            )
        )

    def test_frozen_and_unfrozen_full_rounds(self):
        for condition in ("S17-U", "S17-F", "S17-J"):
            c, m, d, r = fixture(17, condition)
            before = deepcopy(m.state_dict())
            trainer = Stage16Trainer(m, c, d, reference=r)
            trainer.train_until(stop_after_steps=4)
            changed = [
                n for n, v in m.state_dict().items() if not torch.equal(v, before[n])
            ]
            self.assertEqual(
                any("_encoder." in n for n in changed), condition == "S17-U"
            )
            self.assertTrue(any("ptcr_classifier" in n for n in changed))

    def test_four_type_directed_constraint_gradients(self):
        for kind in ("000", "010", "100", "111"):
            ph = torch.tensor([[2.0, 0.0]], requires_grad=True)
            pt = torch.tensor([[2.0, 0.0]], requires_grad=True)
            pht = torch.tensor([[0.0, 2.0]], requires_grad=True)
            labels = [torch.tensor([int(v)]) for v in kind]
            loss = compute_stage8_loss(
                ph,
                pt,
                pht,
                *labels,
                [kind],
                condition="full",
                epoch=5,
                phase="stage8b",
                pht_weight=0.0,
                phla_weight=0.0,
                ptcr_weight=0.0
            )
            loss.total.backward()
            if kind == "111":
                self.assertLess(float(pt.grad[0, 1]), 0.0)
                self.assertLess(float(ph.grad[0, 1]), 0.0)
                self.assertEqual(float(pht.grad.abs().sum()), 0.0)
            else:
                self.assertGreater(float(pht.grad[0, 1]), 0.0)
                self.assertEqual(float(pt.grad.abs().sum()), 0.0)

    def test_protected_ph_and_no_replay_diagnostic_conditions(self):
        c, m, d, r = fixture(17, "S17-P")
        t = Stage16Trainer(m, c, d, reference=r)
        before = deepcopy(m.binding_model.peptide_adapter.state_dict())
        t.train_until(stop_after_steps=1)
        self.assertTrue(
            all(
                torch.equal(v, m.binding_model.peptide_adapter.state_dict()[k])
                for k, v in before.items()
            )
        )
        c, m, d, r = fixture(17, "S17-Z")
        t = Stage16Trainer(m, c, d, reference=r)
        before = predict_pairs(m, d["pairs"]["ptcr"], c)["logits"]
        t.train_until()
        self.assertTrue(
            (before == predict_pairs(m, d["pairs"]["ptcr"], c)["logits"]).all()
        )


if __name__ == "__main__":
    unittest.main()
