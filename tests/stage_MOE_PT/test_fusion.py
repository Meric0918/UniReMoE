"""Behavioral tests specific to the v2 PT→PHT connection."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_stage_MOE_PT import small_config, inputs, ROOT
from source.plm_unified.stage_MOE_PT_config import load_stage_MOE_PT_config
from source.plm_unified.stage_MOE_model import StageMOEBindingModel
from source.plm_unified.stage_MOE_PT_model import StageMOEPTBindingModel
from source.plm_unified.stage_MOE_PT_trainer import build_stage_MOE_PT_optimizer
from source.plm_unified.stage_MOE_PT_evaluation import FusionStatistics
from source.plm_unified.stage_MOE_PT_checkpoint import load_checkpoint, model_contract


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = small_config(Path('/tmp/fusion_test'))
        self.model = StageMOEPTBindingModel(self.config).eval()
        self.x = inputs(self.config)

    def test_residue_fusion_formula_and_padding(self):
        fusion = self.model.fusion
        query, pt, hm = torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 8)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        fused, beta = fusion(query, pt, hm, mask)
        expected_beta = fusion.gate(torch.cat([
            fusion.query_norm(query), fusion.pt_norm(pt), hm[:, None].expand_as(query)], -1)).sigmoid().squeeze(-1)
        expected = query + expected_beta[..., None] * fusion.projection(fusion.pt_norm(pt))
        torch.testing.assert_close(beta[mask], expected_beta[mask])
        torch.testing.assert_close(fused[mask], expected[mask])
        self.assertEqual(int(torch.count_nonzero(beta[~mask])), 0)
        self.assertEqual(int(torch.count_nonzero(fused[~mask])), 0)
        self.assertTrue(((beta[mask] > 0) & (beta[mask] < 1)).all())
        self.assertEqual(float(fusion.gate[-1].bias), -2.0)

    def test_disabled_fusion_matches_v1_with_identical_weights(self):
        baseline = StageMOEBindingModel(self.config).eval()
        baseline.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                  if not k.startswith('fusion.')}, strict=True)
        self.model.pt_fusion_enabled = False
        for method in ('forward_all', 'forward_triplet'):
            expected = getattr(baseline, method)(*self.x)
            actual = getattr(self.model, method)(*self.x)
            self.assertIsNone(actual.fusion_beta)
            self.assertEqual(set(actual.routing), set(expected.routing))
            for task in expected.logits:
                torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)
        self.model.forward_triplet(*self.x).logits['pht'].sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.fusion.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.moe['pt'].parameters()))

    def test_beta_padding_and_forward_all_consistency(self):
        single = self.model.forward_triplet(*self.x)
        all_heads = self.model.forward_all(*self.x)
        torch.testing.assert_close(single.fusion_beta, all_heads.fusion_beta)
        self.assertEqual(single.fusion_beta.shape, self.x[1].shape)
        self.assertTrue((single.fusion_beta[~self.x[1]] == 0).all())
        changed = [x.clone() for x in self.x]
        for index in (0, 2, 4):
            changed[index][~changed[index + 1]] = float('nan')
        torch.testing.assert_close(single.fusion_beta, self.model.forward_triplet(*changed).fusion_beta)

    def test_hla_affects_pht_and_beta_but_not_pt(self):
        before = self.model.forward_all(*self.x)
        changed = list(self.x)
        changed[2] = torch.randn_like(changed[2]) * 3
        after = self.model.forward_all(*changed)
        torch.testing.assert_close(before.logits['pt'], after.logits['pt'], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before.logits['pht'], after.logits['pht']))
        self.assertFalse(torch.allclose(before.fusion_beta, after.fusion_beta))

    def test_fusion_lr_groups_and_pair_gradient_isolation(self):
        s = self.config.stage_MOE_PT
        optimizer = build_stage_MOE_PT_optimizer(self.model, s)
        rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        for p in self.model.fusion.gate.parameters():
            self.assertEqual(rates[id(p)], s.router_learning_rate)
        for module in (self.model.fusion.projection, self.model.fusion.pt_norm, self.model.fusion.query_norm):
            for p in module.parameters():
                self.assertEqual(rates[id(p)], s.learning_rate)
        for task, receptor in [('ph', self.x[2:4]), ('pt', self.x[4:])]:
            self.model.zero_grad(set_to_none=True)
            self.model(task, *self.x[:2], *receptor).logits[task].sum().backward()
            self.assertTrue(all(p.grad is None for p in self.model.fusion.parameters()))

    def test_statistics_exclude_padding_and_weight_by_residues(self):
        stats = FusionStatistics()
        stats.update(torch.tensor([[0., 1., .5], [.5, .9, .9]]),
                     torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool))
        stats.update(None, None)
        result = stats.result()
        self.assertEqual(result['residues'], 4)
        self.assertEqual(result['mean'], .5)
        self.assertAlmostEqual(result['std'], .125 ** .5)
        self.assertEqual(result['near_zero_fraction'], .25)
        self.assertEqual(result['near_one_fraction'], .25)

    def test_fusion_config_in_contract_and_invalid_values(self):
        original = model_contract(self.config)
        for field, value in [('pt_fusion_enabled', False), ('fusion_hidden_dim', 32), ('fusion_bias_init', -1.)]:
            config = deepcopy(self.config)
            setattr(config.stage_MOE_PT, field, value)
            self.assertNotEqual(original, model_contract(config))
        for field, value in [('pt_fusion_enabled', 'false'), ('fusion_hidden_dim', 0), ('fusion_bias_init', float('nan'))]:
            config = deepcopy(self.config)
            setattr(config.stage_MOE_PT, field, value)
            with self.assertRaises(ValueError):
                config.validate()

    def test_v1_checkpoint_is_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'v1.pt'
            torch.save({'architecture': 'stage_MOE_v1', 'format_version': 1}, path)
            with self.assertRaisesRegex(ValueError, 'stage_MOE_PT_v2'):
                load_checkpoint(path)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_default_size_bf16_backward_and_optimizer_step(self):
        config = load_stage_MOE_PT_config(ROOT / 'configs/plm_stage_MOE_PT.yaml')
        model = StageMOEPTBindingModel(config).cuda().train()
        values = []
        for entity in ('peptide', 'hla', 'tcr'):
            length = getattr(config.model, f'{entity}_max_length')
            hidden = torch.randn(2, length, getattr(config.model, f'{entity}_input_dim'), device='cuda')
            mask = torch.ones(2, length, device='cuda', dtype=torch.bool)
            mask[:, -2:] = False
            values.extend([hidden, mask])
        optimizer = build_stage_MOE_PT_optimizer(model, config.stage_MOE_PT)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.forward_all(*values)
            loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), torch.tensor([0., 1.], device='cuda')) for logits in output.logits.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
