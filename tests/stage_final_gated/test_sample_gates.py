"""Sample gating invariants, nontrivial gradients, compatibility and precision."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from test_stage_final_gated import fixture, inputs, CODES, ROOT
from source.plm_unified.stage_final_gated_config import load_stage_final_gated_config
from source.plm_unified.stage_final_gated_model import StageFinalGatedBindingModel, StageFinalGatedOutput
from source.plm_unified.stage_final_gated_loss import compute_stage_final_gated_loss, weighted_group
from source.plm_unified.stage_final_gated_trainer import build_stage_final_gated_optimizer
from source.plm_unified.stage_final_gated_checkpoint import load_checkpoint, model_contract
from source.plm_unified.stage_final_gated_statistics import ScalarGateStatistics
from source.plm_unified.stage_final_model import StageFinalBindingModel
from source.plm_unified.stage_final_loss import compute_stage_final_loss
from types import SimpleNamespace


def supervision(codes=CODES, device='cpu'):
    labels = {t: torch.tensor([float(c[i]) for c in codes], device=device)
              for i, t in enumerate(('ph', 'pt', 'pht'))}
    return labels, {t: torch.ones(len(codes), dtype=torch.bool, device=device) for t in labels}


class SampleGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)
        self.temp = tempfile.TemporaryDirectory()
        self.config, _ = fixture(Path(self.temp.name))
        self.s = self.config.stage_final_gated
        self.model = StageFinalGatedBindingModel(self.config).eval()
        self.x = inputs(self.config, 4)

    def tearDown(self):
        self.temp.cleanup()

    def test_formula_initialization_padding_and_pair_isolation(self):
        captured = {}
        handle = self.model.fusion.register_forward_hook(lambda m, args, result: captured.update(query=args[0], pt=args[1]))
        output = self.model.forward_all(*self.x)
        handle.remove()
        torch.testing.assert_close(output.sample_fusion_gate, torch.full((4,), .9))
        torch.testing.assert_close(output.sample_constraint_gate, torch.full((4,), .525))
        torch.testing.assert_close(output.fusion_beta, .9 * output.residue_beta)
        self.assertEqual(int(torch.count_nonzero(output.fusion_beta[~self.x[1]])), 0)
        fused = captured['query'] + output.fusion_beta[..., None] * self.model.fusion.projection(self.model.fusion.pt_norm(captured['pt']))
        routed = {}
        hook = self.model.moe['pht'].register_forward_pre_hook(lambda m, args: routed.update(hidden=args[0]))
        self.model.forward_all(*self.x)
        hook.remove()
        torch.testing.assert_close(routed['hidden'], fused.masked_fill(~self.x[1][..., None], 0))
        poisoned = [v.clone() for v in self.x]
        for i in (0, 2, 4):
            poisoned[i][~poisoned[i+1]] = float('nan')
        other = self.model.forward_all(*poisoned)
        torch.testing.assert_close(other.logits['pht'], output.logits['pht'])
        for task, receptor in [('ph', self.x[2:4]), ('pt', self.x[4:])]:
            self.model.zero_grad(set_to_none=True)
            pair = self.model(task, *self.x[:2], *receptor)
            self.assertIsNone(pair.sample_fusion_gate)
            self.assertIsNone(pair.sample_constraint_gate)
            pair.logits[task].sum().backward()
            self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if n.startswith('sample_')))

    def test_both_disabled_exact_original_final_equivalence(self):
        self.s.sample_fusion_enabled = self.s.sample_constraint_enabled = False
        model = StageFinalGatedBindingModel(self.config).eval()
        old_config = SimpleNamespace(model=self.config.model, stage_MOE=self.s, stage_MOE_PT=self.s, stage_final=self.s)
        old = StageFinalBindingModel(old_config).eval()
        old.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('sample_')})
        for fusion_enabled in (True, False):
            model.pt_fusion_enabled = old.pt_fusion_enabled = fusion_enabled
            for method in ('forward_all', 'forward_triplet'):
                actual, expected = getattr(model, method)(*self.x), getattr(old, method)(*self.x)
                for task in expected.logits:
                    torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)
                if fusion_enabled:
                    torch.testing.assert_close(actual.fusion_beta, expected.fusion_beta, rtol=0, atol=0)
            labels, masks = supervision()
            actual = compute_stage_final_gated_loss(actual if 'ph' in actual.logits else model.forward_all(*self.x), labels, masks,
                                                    task='pht', settings=self.s, epoch=8)
            expected = compute_stage_final_loss(old.forward_all(*self.x), labels, masks, task='pht', settings=self.s, epoch=8)
            torch.testing.assert_close(actual.total, expected.total)

    def test_normalization_hand_computation_scaling_empty_singleton_and_gradient(self):
        values = torch.tensor([1., 3., 90.], requires_grad=True)
        gates = torch.tensor([.2, .6, .1], requires_grad=True)
        mask = torch.tensor([True, True, False])
        raw, loss, dispersion, weights = weighted_group(values, gates, mask)
        torch.testing.assert_close(weights, torch.tensor([.5, 1.5]))
        self.assertAlmostEqual(float(raw), 2.)
        self.assertAlmostEqual(float(loss), 2.5)
        self.assertAlmostEqual(float(dispersion), .25)
        torch.testing.assert_close(weighted_group(values, gates * .3, mask)[1], loss)
        loss.backward()
        torch.testing.assert_close(gates.grad, torch.tensor([-1.875, .625, 0.]))
        torch.testing.assert_close(values.grad, torch.tensor([.25, .75, 0.]))
        self.assertEqual(weighted_group(values, gates, torch.zeros_like(mask))[1].item(), 0.)
        singleton = weighted_group(values, gates, torch.tensor([True, False, False]))
        self.assertEqual(singleton[1].item(), 1.)
        self.assertEqual(singleton[2].item(), 0.)

    def test_gate_input_detached_but_gate_network_learns(self):
        # Break initial constant gate so first-layer gradients are meaningful too.
        with torch.no_grad():
            self.model.sample_constraint[-1].weight.fill_(.1)
        out = self.model.forward_all(*self.x)
        mask = torch.ones(4, dtype=torch.bool)
        loss = weighted_group(torch.tensor([.1, 2., .6, 4.]), out.sample_constraint_gate, mask)[1]
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.sample_constraint.parameters()), 0.)
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if not n.startswith('sample_constraint.')))
        self.model.zero_grad(set_to_none=True)
        out = self.model.forward_all(*self.x)
        out.logits['pht'].square().sum().backward()
        self.assertGreater(float(self.model.sample_fusion[-1].weight.grad.abs().sum()), 0.)

    def test_constant_gate_recovers_losses_and_regularizer_respects_disabled_terms(self):
        out = self.model.forward_all(*self.x)
        labels, masks = supervision()
        actual = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        original = compute_stage_final_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        torch.testing.assert_close(actual.total, original.total)
        out.sample_constraint_gate = torch.tensor([.1, .9, .2, .8], requires_grad=True)
        self.s.aux_ph_weight = self.s.aux_pt_weight = 0.
        self.s.consistency_warmup_epochs = 3
        off = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=1)
        self.assertEqual(float(off.components['gate_regularization']), 0.)
        torch.testing.assert_close(off.total, off.components['native_bce'])
        off.total.backward(retain_graph=True)
        self.assertIsNone(out.sample_constraint_gate.grad)
        active = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        self.assertGreater(float(active.components['gate_regularization']), 0.)

    def test_missing_labels_and_directed_gradient_with_nonconstant_gates(self):
        labels, masks = supervision(('111', '111', '000', '000'))
        logits = {t: torch.tensor([-.8, -.6, -.8, -.6], requires_grad=True) for t in labels}
        logits['pht'] = torch.tensor([1., 2., 1., 2.], requires_grad=True)
        out = StageFinalGatedOutput(logits, {'pht': torch.full((4, 4), .25)},
                                   sample_constraint_gate=torch.tensor([.2, .9, .6, .3], requires_grad=True))
        result = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        (result.components['hierarchy'] + result.components['pt_implication']).backward()
        self.assertTrue((logits['pht'].grad[:2] == 0).all())
        self.assertTrue((logits['pht'].grad[2:] > 0).all())
        self.assertTrue((logits['pt'].grad[:2] < 0).all())
        self.assertTrue((logits['pt'].grad[2:] == 0).all())
        labels['ph'][:] = float('nan'); masks['ph'][:] = False
        result = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(result.counts['aux_ph_bce'], 0)

    def test_configuration_contract_and_optimizer(self):
        optimizer = build_stage_final_gated_optimizer(self.model, self.s)
        rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(len(rates), len(list(self.model.parameters())))
        for n, p in self.model.named_parameters():
            if n.startswith('sample_'):
                self.assertEqual(rates[id(p)], self.s.router_learning_rate)
        for name, value in [('sample_fusion_enabled', False), ('sample_constraint_enabled', False),
                            ('sample_gate_hidden_dim', 17), ('constraint_gate_floor', .2)]:
            changed = deepcopy(self.config); setattr(changed.stage_final_gated, name, value)
            self.assertNotEqual(model_contract(changed), model_contract(self.config))
        for name, value in [('sample_fusion_enabled', 1), ('sample_gate_hidden_dim', True),
                            ('constraint_gate_floor', 0.), ('sample_fusion_init', 1.),
                            ('constraint_gate_regularization', float('nan'))]:
            changed = deepcopy(self.config); setattr(changed.stage_final_gated, name, value)
            with self.assertRaises(ValueError): changed.validate()
        path = Path(self.temp.name) / 'old.pt'
        torch.save({'architecture': 'stage_final_v1', 'format_version': 1}, path)
        with self.assertRaisesRegex(ValueError, 'stage_final_gated_v1'): load_checkpoint(path)

    def test_gate_disabling_excludes_unused_parameters_from_updates(self):
        self.s.sample_fusion_enabled = self.s.sample_constraint_enabled = False
        model = StageFinalGatedBindingModel(self.config).train()
        optimizer = build_stage_final_gated_optimizer(model, self.s)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('sample_')}
        out = model.forward_all(*self.x)
        labels, masks = supervision()
        loss = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=self.s, epoch=8)
        loss.total.backward()
        optimizer.step()
        for n, p in model.named_parameters():
            if n in before:
                self.assertIsNone(p.grad)
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_histogram_statistics(self):
        stats = ScalarGateStatistics()
        stats.update(torch.tensor([0., .25, .5, .75, 1.]))
        result = stats.result()
        self.assertEqual(result['samples'], 5)
        self.assertEqual(result['mean'], .5)
        self.assertAlmostEqual(result['q50'], .5, delta=1/256)
        self.assertEqual(result['near_zero_fraction'], .2)
        self.assertEqual(result['near_one_fraction'], .2)

    def test_default_dimensions_bf16_update(self):
        config = load_stage_final_gated_config(ROOT / 'configs/plm_stage_final_gated.yaml')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = StageFinalGatedBindingModel(config).to(device).train()
        values = []
        for entity in ('peptide', 'hla', 'tcr'):
            length = getattr(config.model, f'{entity}_max_length')
            hidden = torch.randn(4, length, getattr(config.model, f'{entity}_input_dim'), device=device)
            mask = torch.ones(4, length, dtype=torch.bool, device=device); mask[:, -2:] = False
            values.extend((hidden, mask))
        optimizer = build_stage_final_gated_optimizer(model, config.stage_final_gated)
        before = model.sample_fusion[-1].weight.detach().clone()
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model.forward_all(*values)
            labels, masks = supervision(device=device)
            loss = compute_stage_final_gated_loss(out, labels, masks, task='pht', settings=config.stage_final_gated, epoch=8)
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sample_fusion[-1].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
