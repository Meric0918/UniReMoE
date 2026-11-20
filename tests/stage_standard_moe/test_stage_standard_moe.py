"""Sparse dispatch, unchanged Full topology and real-cache sequential lifecycle."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/stage_final'))
from test_stage_final import fixture as full_fixture, inputs, CODES
from test_stage_MOE import small_config as base_config

from source.plm_unified.stage_final_config import load_stage_final_config
from source.plm_unified.stage_final_model import StageFinalBindingModel
from source.plm_unified.stage_final_loss import compute_stage_final_loss
from source.plm_unified.stage_final_data import audit_stage_final_data, make_loader, open_caches
from source.plm_unified.stage_final_checkpoint import load_checkpoint as load_full_checkpoint
from source.plm_unified.stage_standard_moe_config import (
    StageStandardMoEConfig, StageStandardMoEPaths, StageStandardMoESettings,
    load_stage_standard_moe_config)
from source.plm_unified.stage_standard_moe_model import (
    StandardBranchMoE, StageStandardMoEBindingModel, forward_batch)
from source.plm_unified.stage_standard_moe_loss import compute_stage_standard_moe_loss
from source.plm_unified.stage_standard_moe_checkpoint import load_checkpoint, model_contract
from source.plm_unified.stage_standard_moe_evaluation import RoutingStatistics, evaluate_task
from source.plm_unified.stage_standard_moe_trainer import (
    StageStandardMoESequentialTrainer, build_stage_standard_moe_optimizer)
from source.plm_unified.stage_standard_moe_cli import main, parser_for
from source.plm_unified.stage_standard_moe_tracking import StageStandardMoETracker
from source.plm_unified.trainer import set_global_seed


def convert(config, section):
    paths = asdict(config.paths)
    paths['output_root'] = config.paths.project_root / 'stage_standard_moe_output'
    values = asdict(getattr(config, section))
    values.pop('gate_hidden_dim')
    values['balance_lambda'] = 0.0
    return StageStandardMoEConfig(StageStandardMoEPaths(**paths), config.model,
                                 StageStandardMoESettings(**values))


def small_config(root):
    return convert(base_config(root), 'stage_MOE')


def fixture(root):
    full, _ = full_fixture(root)
    config = convert(full, 'stage_final')
    path = root / 'standard.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    return config, path


def supervision(device='cpu'):
    labels = {t: torch.tensor([float(c[i]) for c in CODES], device=device)
              for i, t in enumerate(('ph', 'pt', 'pht'))}
    return labels, {t: torch.ones(4, dtype=torch.bool, device=device) for t in labels}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = small_config(Path('/tmp/stage_standard_moe_unit'))
        self.model = StageStandardMoEBindingModel(self.config).eval()
        self.x = inputs(self.config, 4)

    def test_independent_experts_and_optimizer_groups(self):
        self.assertFalse(hasattr(self.model, 'shared_experts'))
        pools = [{id(p) for p in branch.experts.parameters()} for branch in self.model.moe.values()]
        self.assertEqual(len({id(e) for b in self.model.moe.values() for e in b.experts}), 12)
        self.assertTrue(all(not pools[i] & pools[j] for i in range(3) for j in range(i)))
        self.assertTrue(all(isinstance(branch.router, nn.Linear) for branch in self.model.moe.values()))
        optimizer = build_stage_standard_moe_optimizer(self.model, self.config.stage_standard_moe)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.model.parameters()})
        rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        for branch in self.model.moe.values():
            for p in branch.router.parameters():
                self.assertEqual(rates[id(p)], self.config.stage_standard_moe.router_learning_rate)
        for p in self.model.fusion.gate.parameters():
            self.assertEqual(rates[id(p)], self.config.stage_standard_moe.router_learning_rate)
        for p in self.model.fusion.projection.parameters():
            self.assertEqual(rates[id(p)], self.config.stage_standard_moe.learning_rate)

    def test_sparse_dispatch_formula_and_selected_gradients(self):
        branch = self.model.moe['ph']
        with torch.no_grad():
            branch.router.weight.zero_()
            branch.router.bias.copy_(torch.tensor([2., 1., -1., -2.]))
        hidden = torch.randn(3, 4, 8, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        calls = [0] * 4
        def record(index):
            def hook(module, args):
                calls[index] += args[0].shape[0]
            return hook
        handles = [e.register_forward_pre_hook(record(i)) for i, e in enumerate(branch.experts)]
        actual, gates = branch(hidden, mask, 1.5)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [3, 3, 0, 0])
        probs = (branch.router.bias / 1.5).softmax(-1)
        expected_gates = torch.cat([probs[:2] / probs[:2].sum(), torch.zeros(2)]).expand(3, -1)
        torch.testing.assert_close(gates, expected_gates)
        clean = hidden.masked_fill(~mask[..., None], 0)
        values = torch.stack([e(branch.norm(clean)) for e in branch.experts], 2)
        expected = (clean + (values * expected_gates[:, None, :, None]).sum(2)).masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertGreater(float(branch.router.weight.grad.abs().sum()), 0)
        for i, module in enumerate(branch.experts):
            if i < 2:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
            else:
                self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_sample_specific_dispatch_matches_dense_oracle(self):
        branch = self.model.moe['ph']
        hidden, mask = torch.randn(6, 4, 8), torch.ones(6, 4, dtype=torch.bool)
        with torch.no_grad():
            branch.router.weight.zero_()
            branch.router.bias.zero_()
            branch.router.weight[:, :4] = torch.eye(4)
            hidden[:, :, :4] = torch.tensor([
                [4, 3, 0, 0], [0, 4, 3, 0], [0, 0, 4, 3],
                [3, 0, 0, 4], [0, 4, 0, 3], [4, 0, 3, 0]])[:, None, :]
        seen = [[] for _ in range(4)]
        handles = [module.register_forward_pre_hook(
            lambda module, args, i=i: seen[i].append(args[0].detach().clone()))
            for i, module in enumerate(branch.experts)]
        result, gates = branch(hidden, mask, 1.)
        for handle in handles:
            handle.remove()
        normalized = branch.norm(hidden)
        for i in range(4):
            torch.testing.assert_close(seen[i][0], normalized[gates[:, i] > 0])
        dense = sum(e(normalized) * gates[:, None, i, None] for i, e in enumerate(branch.experts))
        torch.testing.assert_close(result, hidden + dense)
        self.assertTrue(((gates > 0).sum(-1) == 2).all())
        torch.testing.assert_close(gates.sum(-1), torch.ones(6))

    def test_padding_invariance_and_empty_input_rejection(self):
        changed = [v.clone() for v in self.x]
        for i in (0, 2, 4):
            changed[i][~changed[i + 1]] = float('nan')
        before = self.model.forward_all(*self.x)
        after = self.model.forward_all(*changed)
        for task in ('ph', 'pt', 'pht'):
            torch.testing.assert_close(before.logits[task], after.logits[task])
            torch.testing.assert_close(before.routing[task], after.routing[task])
        torch.testing.assert_close(before.fusion_beta, after.fusion_beta)
        self.assertEqual(int(torch.count_nonzero(after.fusion_beta[~self.x[1]])), 0)
        branch = self.model.moe['ph']
        with self.assertRaisesRegex(ValueError, 'real residue'):
            branch(torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool), 1.)
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'temperature'):
                branch(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool), temperature)

    def test_full_topology_equivalence_with_replaced_moe(self):
        # Full is the oracle for every surrounding operation, using the same new MoEs.
        full_config = SimpleNamespace(model=self.config.model, stage_MOE=self.config.stage_standard_moe,
            stage_MOE_PT=self.config.stage_standard_moe, stage_final=self.config.stage_standard_moe)
        # Full's constructor needs its old gate dimension before the MoEs are replaced.
        full_config.stage_final = deepcopy(full_config.stage_final)
        full_config.stage_final.gate_hidden_dim = 6
        full_config.stage_MOE = full_config.stage_MOE_PT = full_config.stage_final
        full = StageFinalBindingModel(full_config).eval()
        standard_state = self.model.state_dict()
        state = full.state_dict()
        for name in state:
            if not name.startswith(('moe.', 'shared_experts.')):
                state[name] = standard_state[name]
        full.load_state_dict(state)
        class Adapter(nn.Module):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch
            def forward(self, hidden, context, mask, shared_experts, temperature):
                return self.branch(hidden, mask, temperature)
        full.moe = nn.ModuleDict({task: Adapter(branch) for task, branch in self.model.moe.items()})
        for enabled in (True, False):
            full.pt_fusion_enabled = self.model.pt_fusion_enabled = enabled
            for method in ('forward_all', 'forward_triplet'):
                actual = getattr(self.model, method)(*self.x, return_attention=True, return_features=True)
                expected = getattr(full, method)(*self.x, return_attention=True, return_features=True)
                self.assertEqual(actual.logits.keys(), expected.logits.keys())
                for field in ('logits', 'routing', 'attention', 'features', 'pooling_weights'):
                    for task, value in getattr(actual, field).items():
                        torch.testing.assert_close(value, getattr(expected, field)[task], rtol=0, atol=0)
                if enabled:
                    torch.testing.assert_close(actual.fusion_beta, expected.fusion_beta, rtol=0, atol=0)
                else:
                    self.assertIsNone(actual.fusion_beta)

    def test_pht_router_uses_only_fused_sequence_mean(self):
        fused, context = [], []
        first = self.model.fusion.register_forward_hook(lambda module, args, out: fused.append(out[0]))
        second = self.model.moe['pht'].router.register_forward_pre_hook(lambda module, args: context.append(args[0]))
        self.model.forward_all(*self.x)
        first.remove()
        second.remove()
        mask = self.x[1]
        expected = fused[0].masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        torch.testing.assert_close(context[0], expected)
        self.assertEqual(context[0].shape, (4, 8))

    def test_pair_boundaries_and_optional_outputs(self):
        all_heads = self.model.forward_all(*self.x, return_attention=True, return_features=True)
        ph = self.model('phla', *self.x[:4])
        pt = self.model('ptcr', *self.x[:2], *self.x[4:])
        single = self.model.forward_triplet(*self.x)
        for task, out in [('ph', ph), ('pt', pt), ('pht', single)]:
            torch.testing.assert_close(out.logits[task], all_heads.logits[task])
            self.assertEqual(set(out.logits), {task})
            self.assertEqual(all_heads.features[task].shape, (4, 8))
            self.assertEqual(int(torch.count_nonzero(all_heads.pooling_weights[task][~self.x[1]])), 0)
        self.assertIsNone(ph.fusion_beta)
        self.assertIsNone(pt.fusion_beta)
        changed = list(self.x)
        changed[2] = torch.randn_like(changed[2]) * 3
        after = self.model.forward_all(*changed)
        torch.testing.assert_close(after.logits['pt'], all_heads.logits['pt'])
        self.assertFalse(torch.allclose(after.logits['pht'], all_heads.logits['pht']))

    def test_full_loss_equivalence_and_gradient_paths(self):
        settings = self.config.stage_standard_moe
        output = self.model.forward_all(*self.x)
        labels, masks = supervision()
        actual = compute_stage_standard_moe_loss(output, labels, masks, task='pht', settings=settings, epoch=8)
        expected = compute_stage_final_loss(output, labels, masks, task='pht', settings=settings, epoch=8)
        torch.testing.assert_close(actual.total, expected.total, rtol=0, atol=0)
        self.assertEqual(actual.counts, expected.counts)
        self.assertEqual(actual.counts['hierarchy_negative'], 2)
        self.assertEqual(actual.counts['pt_negative'], 2)
        actual.total.backward()
        for module in (self.model.fusion, self.model.moe['ph'], self.model.moe['pt'],
                       self.model.moe['pht'], *self.model.classifiers.values()):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_native_pht_alone_still_updates_ph_and_pt_features(self):
        settings = self.config.stage_standard_moe
        settings.aux_ph_weight = settings.aux_pt_weight = settings.hierarchy_lambda = settings.pt_implication_lambda = 0
        labels, masks = supervision()
        result = compute_stage_standard_moe_loss(self.model.forward_all(*self.x), labels, masks,
                                                 task='pht', settings=settings, epoch=8)
        result.total.backward()
        for module in (self.model.fusion, self.model.moe['ph'], self.model.moe['pt']):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        for task in ('ph', 'pt'):
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                                for p in self.model.classifiers[task].parameters()))

    def test_routing_statistics_and_nondefault_expert_balance(self):
        stats = RoutingStatistics()
        stats.update({'ph': torch.tensor([[.7, .3, 0, 0], [0, .6, .4, 0]])})
        values = stats.result()['ph']
        self.assertEqual(values['selection_frequency'], [.5, 1., .5, 0.])
        self.assertNotIn('shared_fraction', values)
        self.assertNotIn('private_fraction', values)
        self.config.stage_standard_moe.num_experts = 6
        self.config.stage_standard_moe.balance_lambda = .1
        output = StageStandardMoEBindingModel(self.config).forward_all(*self.x)
        labels, masks = supervision()
        result = compute_stage_standard_moe_loss(output, labels, masks, task='pht',
                                                 settings=self.config.stage_standard_moe, epoch=8)
        torch.testing.assert_close(result.balance, (output.routing['pht'].mean(0) - 1 / 6).square().sum())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_default_dimensions_cuda_bf16_step(self):
        config = load_stage_standard_moe_config(ROOT / 'configs/plm_stage_standard_moe.yaml')
        model = StageStandardMoEBindingModel(config).cuda().train()
        values = []
        for entity in ('peptide', 'hla', 'tcr'):
            length = getattr(config.model, f'{entity}_max_length')
            hidden = torch.randn(4, length, getattr(config.model, f'{entity}_input_dim'), device='cuda')
            mask = torch.ones(4, length, dtype=torch.bool, device='cuda')
            mask[:, -2:] = False
            values.extend([hidden, mask])
        optimizer = build_stage_standard_moe_optimizer(model, config.stage_standard_moe)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.forward_all(*values)
            labels, masks = supervision('cuda')
            loss = compute_stage_standard_moe_loss(output, labels, masks, task='pht',
                settings=config.stage_standard_moe, epoch=8).total
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        for branch in model.moe.values():
            self.assertGreater(float(branch.router.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


class ConfigTests(unittest.TestCase):
    def test_default_matches_full_training_settings(self):
        config = load_stage_standard_moe_config(ROOT / 'configs/plm_stage_standard_moe.yaml')
        full = load_stage_final_config(ROOT / 'configs/plm_stage_final.yaml')
        for name, value in asdict(full.stage_final).items():
            if name not in ('gate_hidden_dim', 'device'):
                self.assertEqual(getattr(config.stage_standard_moe, name), value, name)
        self.assertEqual(config.stage_standard_moe.num_experts, 4)
        self.assertEqual(config.stage_standard_moe.top_k, 2)
        self.assertEqual(config.stage_standard_moe.seed, 42)
        self.assertEqual(config.paths.data_root, ROOT / 'training_logs/stage_final_20261110_160519/data')
        self.assertEqual(config.paths.cache_root, ROOT / 'embedding_cache/plm_stage_MOE')
        self.assertEqual(set(config.to_dict()), {'paths', 'model', 'stage_standard_moe', 'swanlab'})

    def test_config_roundtrip_validation_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            config = small_config(Path(temp))
            path = Path(temp) / 'standard.yaml'
            path.write_text(yaml.safe_dump(config.to_dict()))
            self.assertEqual(load_stage_standard_moe_config(path).to_dict(), config.to_dict())
            for name, value in [('num_experts', 0), ('num_experts', True), ('top_k', 0), ('top_k', 5)]:
                changed = deepcopy(config)
                setattr(changed.stage_standard_moe, name, value)
                with self.assertRaises(ValueError):
                    changed.validate()
            for name, value in [('num_experts', 6), ('top_k', 1), ('pt_fusion_enabled', False)]:
                changed = deepcopy(config)
                setattr(changed.stage_standard_moe, name, value)
                self.assertNotEqual(model_contract(changed), model_contract(config))
            raw = config.to_dict()
            raw['stage_standard_moe']['gate_hidden_dim'] = 128
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, 'Unknown fields'):
                load_stage_standard_moe_config(path)
            config.paths.output_root = Path(temp) / 'PLM_Stage_final'
            with self.assertRaisesRegex(ValueError, 'dedicated stage_standard_moe'):
                config.validate()
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            parser_for('train')


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.path = fixture(self.root)

    def resources(self):
        datasets, report = audit_stage_final_data(self.config)
        caches = open_caches(self.config)
        for cache in caches.values():
            self.addCleanup(cache.close)
        return datasets, report, caches

    def create(self, name, resources, config=None, tracker=None):
        datasets, report, caches = resources
        config = config or self.config
        set_global_seed(config.stage_standard_moe.seed)
        return StageStandardMoESequentialTrainer(StageStandardMoEBindingModel(config), config,
            datasets, caches, report['dataset_fingerprint'], self.root / name, tracker=tracker)

    def test_exact_resume_counts_thresholds_and_contract_rejections(self):
        resources = self.resources()
        full = self.create('full', resources)
        full.fit()
        partial = self.create('partial', resources)
        partial.fit(stop_after_epoch=1)
        resumed = self.create('partial', resources)
        resumed.fit(resume=partial.run_dir / 'last.pt')
        for key, value in full.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(full.global_step, resumed.global_step)
        self.assertEqual(full.scheduler.state_dict(), resumed.scheduler.state_dict())
        training = full.history[0]['training']
        self.assertEqual(training['sampled_rows'], dict.fromkeys(('ph', 'pt', 'pht'), 8))
        self.assertEqual(training['task_optimizer_steps'], dict.fromkeys(('ph', 'pt', 'pht'), 2))
        self.assertEqual(training['task_order'], ['ph', 'pt', 'pht'])
        self.assertEqual(training['optimizer_steps'], 6)
        self.assertEqual(training['fusion']['residues'], 16)
        self.assertEqual(training['loss_details']['components']['pht/aux_ph_bce']['samples'], 8)
        best = load_checkpoint(full.run_dir / 'best.pt', config=self.config, role='best')
        self.assertEqual(set(best['thresholds']), {'ph', 'pt', 'pht'})
        self.assertEqual(best['thresholds'], best['validation']['thresholds'])
        self.assertEqual(best['model_contract']['router'], 'linear_masked_mean_sample_topk_v1')
        with self.assertRaisesRegex(ValueError, 'stage_final_v1'):
            load_full_checkpoint(full.run_dir / 'best.pt')
        changed = deepcopy(self.config)
        changed.stage_standard_moe.hierarchy_lambda = .9
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            self.create('wrong_loss', resources, changed).resume(partial.run_dir / 'last.pt')
        changed = deepcopy(self.config)
        changed.stage_standard_moe.top_k = 1
        with self.assertRaisesRegex(ValueError, 'model configuration mismatch'):
            self.create('wrong_topk', resources, changed).resume(partial.run_dir / 'last.pt')
        wrong = self.create('wrong_data', resources)
        wrong.contract['data_fingerprint'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            wrong.resume(partial.run_dir / 'last.pt')
        wrong = self.create('wrong_cache', resources)
        wrong.contract['cache_fingerprints'] = {}
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            wrong.resume(partial.run_dir / 'last.pt')

    def test_tail_batches_and_disabled_fusion(self):
        self.config.stage_standard_moe.batch_size = 3
        self.config.stage_standard_moe.max_steps_per_epoch = None
        self.config.stage_standard_moe.pt_fusion_enabled = False
        trainer = self.create('tail', self.resources())
        result = trainer.train_epoch(1)
        self.assertEqual(result['sampled_rows'], dict.fromkeys(('ph', 'pt', 'pht'), 8))
        self.assertEqual(result['optimizer_steps'], 9)
        self.assertEqual(result['fusion'], {'residues': 0})
        self.assertTrue(all(p.grad is None for p in trainer.model.fusion.parameters()))

    def test_cli_audit_train_resume_evaluate_and_benchmark(self):
        def run(action, *args, expected=0):
            result = subprocess.run(['bash', str(ROOT / 'run_stage_standard_moe.sh'), action,
                '--config', str(self.path), *map(str, args)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, expected, result.stdout + '\n' + result.stderr)
            return result
        run('audit', '--include-benchmarks', '--output-dir', self.root / 'audit')
        output = self.root / 'trained'
        run('train-sequential', '--swanlab-mode', 'disabled', '--output-dir', output)
        run('resume-sequential', '--resume', output / 'last.pt')
        run('resume-sequential', expected=2)
        run('train', expected=2)
        checkpoint = load_checkpoint(output / 'best.pt')
        for action in ('evaluate', 'benchmark'):
            target = self.root / action
            run(action, '--checkpoint', output / 'best.pt', '--output-dir', target)
            result = json.loads((target / 'metrics.json').read_text())
            self.assertEqual(result['architecture'], 'stage_standard_moe_v1')
            self.assertEqual(result['active_experts_per_sample_per_branch'], 2)
            for key, value in result['results'].items():
                self.assertEqual(value['threshold'], checkpoint['thresholds'][key.split('/')[0]])
                self.assertIn('accuracy', value['metrics'])
                for branch in value['routing'].values():
                    self.assertAlmostEqual(sum(branch['selection_frequency']), 2)
                    self.assertNotIn('shared_fraction', branch)
            self.assertEqual(result['results'][next(k for k in result['results'] if k.startswith('pht/'))]
                             ['consistency']['classes']['111']['samples'], 1)
            with next(target.glob('pht_*predictions.csv')).open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                self.assertEqual(sum(float(row[f'expert_{i}']) > 0 for i in range(4)), 2)
                self.assertIn('p_ph', row)
                self.assertIn('fusion_beta_mean', row)
        # Non-empty run directories cannot be reused for a fresh training run.
        run('train-sequential', '--output-dir', output, expected=1)

    def test_evaluation_uses_saved_margin_and_accepts_missing_auxiliary_labels(self):
        trainer = self.create('margin', self.resources())
        trainer.fit(stop_after_epoch=1)
        self.config.stage_standard_moe.consistency_margin = .4
        self.path.write_text(yaml.safe_dump(self.config.to_dict()))
        result = main('evaluate', ['--config', str(self.path), '--checkpoint', str(trainer.run_dir / 'best.pt'),
                                   '--output-dir', str(self.root / 'eval_margin')])
        self.assertEqual(result['results']['pht/test']['consistency']['margin'], .05)
        records = self.resources()[0]['val']['pht']
        from dataclasses import replace
        records = [replace(r, labels=(float('nan'), float('nan'), r.labels[2]), masks=(False, False, True))
                   for r in records]
        caches = self.resources()[2]
        values = evaluate_task(trainer.model, make_loader(records, 'pht', caches, self.config.stage_standard_moe),
                               'pht', threshold=.5)
        self.assertEqual(values['metrics']['samples'], 4)
        self.assertEqual(values['consistency']['constraints']['hierarchy']['applicable']['samples'], 0)

    def test_tracking_has_sparse_statistics_fusion_and_consistency(self):
        self.config.swanlab.mode = 'online'
        self.config.stage_standard_moe.log_every_steps = 1
        run = SimpleNamespace(id='fake-standard', url='https://example.invalid', log=Mock(), finish=Mock())
        swanlab = SimpleNamespace(init=Mock(return_value=run))
        resources = self.resources()
        with patch.dict('sys.modules', {'swanlab': swanlab}):
            tracker = StageStandardMoETracker(self.config, 'sequential', self.root / 'tracked')
            trainer = self.create('tracked', resources, tracker=tracker)
            trainer.fit(stop_after_epoch=1)
            keys = {key for call in run.log.call_args_list for key in call.args[0]}
            self.assertIn('train/routing/ph/selection_frequency/0', keys)
            self.assertIn('train/fusion/mean', keys)
            self.assertIn('train/loss_details/components/pht/hierarchy_positive/mean', keys)
            self.assertIn('val/tasks/pht/consistency/constraints/hierarchy/all/violation_rate', keys)
            self.assertFalse(any('shared_fraction' in key for key in keys))
            self.assertEqual(load_checkpoint(trainer.run_dir / 'last.pt')['swanlab_run_id'], 'fake-standard')
            self.assertEqual(swanlab.init.call_args.kwargs['group'], 'plm-stage_standard_moe')
            tracker.finish()
            run.finish.assert_called_once()

    def test_checkpoint_rejects_other_architectures_modes_and_incomplete_state(self):
        path = self.root / 'wrong.pt'
        for architecture in ('stage_final_v1', 'stage_MOE_v1', 'stage_MOE_consis_v1'):
            torch.save({'architecture': architecture, 'format_version': 1}, path)
            with self.assertRaisesRegex(ValueError, 'stage_standard_moe_v1'):
                load_checkpoint(path)
        trainer = self.create('checkpoint', self.resources())
        trainer.fit(stop_after_epoch=1)
        checkpoint = load_checkpoint(trainer.run_dir / 'last.pt')
        for key, value, message in [('training_mode', 'joint', 'sequential'),
                                    ('temperature', float('nan'), 'temperature'),
                                    ('thresholds', {'ph': .5}, 'three validation')]:
            changed = {**checkpoint, key: value}
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, message):
                load_checkpoint(path)
        checkpoint.pop('rng')
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, 'missing training/resume'):
            load_checkpoint(path)

    def test_cache_rebuild_guard_preserves_existing_manifests(self):
        manifests = {p: p.read_bytes() for p in self.config.paths.cache_root.rglob('manifest.json')}
        with self.assertRaisesRegex(FileExistsError, 'cache exists'):
            main('cache', ['--config', str(self.path), '--output-dir', str(self.root / 'cache_guard')])
        self.assertTrue(manifests)
        for path, content in manifests.items():
            self.assertEqual(content, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
