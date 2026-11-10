"""CPU checks of gradient routing, optimizer coverage and validation guards."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from plm_unified.config import ModelConfig
from plm_unified.data import _collate_online_pairs
from plm_unified.stage1_encoder_config import Stage1EncoderModelConfig
from plm_unified.stage1_encoder_model import Stage1EncoderBindingModel
from plm_unified.stage8_config import load_stage8_config
from plm_unified.stage8_data import (
    BalancedFourTypeBatchSampler, OnlineStage8Dataset, Stage8Record, collate_online_stage8,
)
from plm_unified.stage8_model import Stage8BindingModel
from plm_unified.stage8_retention import (
    ProtectedStage8Model, PTRetentionPolicy, PTRetentionTrainer,
    distillation_loss, peptide_weights, retention_failures,
)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.dropout = nn.Dropout(0.3)

    def transformer_blocks(self):
        return self.blocks

    def set_active_training(self, value):
        self.train(value)

    def forward(self, sequences):
        ids = torch.tensor([[ord(c) % 30 for c in s] for s in sequences])
        hidden = self.embedding(ids)
        for block in self.blocks:
            hidden = self.dropout(block(hidden))
        return SimpleNamespace(hidden=hidden, mask=torch.ones(ids.shape, dtype=torch.bool))


class ToyPairs(torch.utils.data.Dataset):
    def __init__(self):
        self.rows = [("ACDE", "FGHI", i % 2) for i in range(8)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def fixture():
    torch.manual_seed(4)
    config = load_stage8_config(Path(__file__).resolve().parents[1] / "configs/plm_stage8_20260920_best_joint.yaml")
    config.model = ModelConfig(peptide_input_dim=8, hla_input_dim=8, tcr_input_dim=8,
                               model_dim=8, adapter_hidden_dim=8, num_attention_heads=2,
                               feedforward_dim=16, adapter_dropout=0., attention_dropout=0., classifier_dropout=0.)
    config.stage8.mixed_precision = "none"
    config.stage8.sequence_encoder_dropout = 0.0
    binding = Stage8BindingModel(
        config.model,
        encoder_num_layers=config.stage8.sequence_encoder_num_layers,
        encoder_dropout=config.stage8.sequence_encoder_dropout,
    )
    teacher_config = Stage1EncoderModelConfig(
        peptide_input_dim=8,
        hla_input_dim=8,
        tcr_input_dim=8,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
        encoder_num_layers=config.stage8.sequence_encoder_num_layers,
        encoder_dropout=config.stage8.sequence_encoder_dropout,
        attention_dropout=0.0,
        classifier_dropout=0.0,
    )
    teacher = Stage1EncoderBindingModel(teacher_config, _task="ptcr")
    teacher.load_state_dict({k: v for k, v in binding.state_dict().items() if k in teacher.state_dict()})
    model = ProtectedStage8Model(binding, ToyEncoder(), ToyEncoder(), ToyEncoder())
    records = [Stage8Record("ACDE", "FGHI", "KLMN", int(kind[0]), int(kind[1]), int(kind[2]), i)
               for i, kind in enumerate(["000", "010", "100", "111"] * 2)]
    dataset = OnlineStage8Dataset(records)
    sampler = BalancedFourTypeBatchSampler(dataset.type_indices, batch_size=4, seed=42)
    joint = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_online_stage8)
    pair = DataLoader(ToyPairs(), batch_size=4, collate_fn=_collate_online_pairs)
    trainer = PTRetentionTrainer(model, config, {"phla": pair, "ptcr": pair, "joint": joint}, joint,
                                 condition="full", teacher=teacher, pt_val_loaders={"val": pair},
                                 policy=PTRetentionPolicy(rounds=2, ph_replay_ratio=1.))
    trainer.stage8a_epoch = 2
    trainer.begin_stage8b()
    return trainer


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_joint_update_preserves_every_pair_and_encoder_parameter(self):
        trainer = fixture()
        before = deepcopy(trainer.model.state_dict())
        trainer._train_joint(phase="stage8b", epoch_or_round=1)
        changed = [k for k, v in trainer.model.state_dict().items() if not torch.equal(v, before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith("binding_model.triplet_") for k in changed), changed)
        self.assertEqual(trainer.train_loaders["joint"].batch_sampler.epoch, 1)
        trainer._train_joint(phase="stage8b", epoch_or_round=2)
        self.assertEqual(trainer.train_loaders["joint"].batch_sampler.epoch, 2)

    def test_optimizer_union_and_complete_round(self):
        trainer = fixture()
        parameters = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        self.assertEqual(parameters, {id(p) for p in trainer.model.binding_model.parameters()})
        teacher_before = deepcopy(trainer.teacher.state_dict())
        encoder_before = deepcopy(trainer.model.peptide_encoder.state_dict())
        pt_before = deepcopy(trainer.model.binding_model.ptcr_classifier.state_dict())
        result = trainer.train_stage8b_round(1)
        self.assertEqual([x[2] for x in trainer.pass_history], ["phla", "joint", "ptcr"])
        self.assertEqual(result["ptcr"]["samples"], 8)
        self.assertTrue(any(not torch.equal(v, pt_before[k]) for k, v in trainer.model.binding_model.ptcr_classifier.state_dict().items()))
        self.assertTrue(all(torch.equal(v, teacher_before[k]) for k, v in trainer.teacher.state_dict().items()))
        self.assertTrue(all(torch.equal(v, encoder_before[k]) for k, v in trainer.model.peptide_encoder.state_dict().items()))
        student, teacher = trainer.evaluate_pt()
        self.assertEqual(student["val"]["samples"], 8)
        self.assertTrue(torch.isfinite(torch.tensor(teacher["val"]["auroc"])))

    def test_ph_replay_preserves_pt_shared_sequence_encoder(self):
        trainer = fixture()
        before = deepcopy(trainer.model.binding_model.peptide_encoder.state_dict())
        trainer._train_replay("phla", 1)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in trainer.model.binding_model.peptide_encoder.state_dict().items()))

    def test_kd_direction_and_teacher_stop_gradient(self):
        student = torch.tensor([[0., 0.]], requires_grad=True)
        teacher = torch.tensor([[0., 3.]], requires_grad=True)
        loss = distillation_loss(student, teacher, 2.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertLess(student.grad[0, 1], 0.)
        self.assertAlmostEqual(float(distillation_loss(teacher, teacher, 2.)), 0., places=6)

    def test_guard_catches_split_loss_hidden_by_macro(self):
        policy = PTRetentionPolicy()
        reference = {s: dict(auroc=.8, aupr=.8, mcc=.6) for s in ("a", "b", "c")}
        candidate = deepcopy(reference)
        candidate["a"]["auroc"] = .77
        candidate["b"]["auroc"] = .83
        failures = retention_failures(candidate, reference, .9, .9, policy)
        self.assertIn("a:auroc", failures)
        self.assertNotIn("macro:auroc", failures)
        self.assertEqual(retention_failures(reference, reference, .9, .9, policy), [])
        self.assertIn("pht:performance_avg", retention_failures(reference, reference, .8, .9, policy))

    def test_nonfinite_guard_and_policy(self):
        reference = {"a": dict(auroc=.8, aupr=.8, mcc=.6)}
        candidate = {"a": dict(auroc=float("nan"), aupr=.8, mcc=.6)}
        self.assertIn("nonfinite:auroc", retention_failures(candidate, reference, .9, .9, PTRetentionPolicy()))
        with self.assertRaises(ValueError):
            PTRetentionPolicy(temperature=0.)

    def test_group_weights_normalize_globally(self):
        rows = [("common", "T", 0)] * 20 + [("rare", "T", 1)]
        weights = peptide_weights(rows)
        self.assertGreater(weights["rare"], weights["common"])
        self.assertAlmostEqual(sum(weights[p] for p, _, _ in rows) / len(rows), 1.)

    def test_checkpoint_restore_uses_pinned_encoders_and_new_binding_weights(self):
        from evaluate_plm_stage8_pt_retention import restore_retention_model
        trainer = fixture()
        model = trainer.model
        original_state = deepcopy(model.state_dict())
        trained_binding = deepcopy(model.binding_model.state_dict())
        changed = next(iter(trained_binding))
        trained_binding[changed] = trained_binding[changed] + 1.0
        payload = {
            "checkpoint_type": "stage8_pt_retention_frozen_plm", "format_version": 1,
            "plan": {"warmup_checkpoint": "warmup.pt", "warmup_sha256": "pinned",
                     "base_config": {"source_path": "base.yaml", "model": asdict(trainer.config.model),
                                     "stage8": {"mixed_precision": "none", "constraint_margin": .05}}},
            "binding_model": trained_binding,
        }
        module = "evaluate_plm_stage8_pt_retention."
        with patch(module + "checkpoint_sha256", return_value="pinned"), \
             patch(module + "load_stage8_config", return_value=trainer.config), \
             patch(module + "load_stage8_v2_checkpoint", return_value={"model": original_state}), \
             patch(module + "build_progressive_stage8_model", return_value=model):
            restored, _ = restore_retention_model(payload, torch.device("cpu"))
        self.assertTrue(torch.equal(restored.binding_model.state_dict()[changed], trained_binding[changed]))
        self.assertTrue(all(torch.equal(v, original_state[k]) for k, v in restored.state_dict().items() if "_encoder." in k))
        with patch(module + "checkpoint_sha256", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                restore_retention_model(payload, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
