"""CPU contract and end-to-end tests using real HDF5 caches, without loading PLMs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np
import torch
import yaml

from source.plm_unified.cache import EmbeddingCacheWriter
from source.plm_unified.config import ModelConfig
from source.plm_unified.encoders import checkpoint_fingerprint
from source.plm_unified.stage_MOE_PT_config import StageMOEPTConfig, StageMOEPTPaths, StageMOEPTSettings, load_stage_MOE_PT_config
from source.plm_unified.stage_MOE_PT_model import StageMOEPTBindingModel, StageMOEPTOutput, TASKS
from source.plm_unified.stage_MOE_loss import masked_bce, compute_stage_MOE_loss
from source.plm_unified.stage_MOE_data import (audit_stage_MOE_data, collect_sequences, open_caches,
    read_task_records, make_loader, file_sha256)
from source.plm_unified.stage_MOE_PT_trainer import StageMOEPTTrainer, build_stage_MOE_PT_optimizer
from source.plm_unified.stage_MOE_PT_checkpoint import load_checkpoint
from source.plm_unified.stage_MOE_PT_evaluation import evaluate_task
from source.plm_unified.trainer import set_global_seed

ROOT = Path(__file__).resolve().parents[2]


def small_config(root, dropout=0.0):
    return StageMOEPTConfig(
        StageMOEPTPaths(root, root / 'data', root / 'MOE_PT_cache', root / 'MOE_PT_output', root / 'esmc', root / 'tcr'),
        ModelConfig(peptide_input_dim=6, hla_input_dim=5, tcr_input_dim=7, model_dim=8,
                    num_attention_heads=2, feedforward_dim=16, peptide_min_length=2,
                    peptide_max_length=4, hla_max_length=4, tcr_max_length=4),
        StageMOEPTSettings(sequence_encoder_num_layers=1, dropout=dropout, expert_hidden_dim=12,
            gate_hidden_dim=6, pooling_hidden_dim=6, classifier_hidden_dim=6, batch_size=4,
            eval_batch_size=4, max_epochs=2, patience=5, mixed_precision='none', device='cpu',
            num_workers=0, max_steps_per_epoch=2, seed=23, balance_lambda=1e-3))


def inputs(config, batch_size=3):
    values = []
    for entity in ('peptide', 'hla', 'tcr'):
        hidden = torch.randn(batch_size, 4, getattr(config.model, f'{entity}_input_dim'))
        mask = torch.tensor([[1, 1, 0, 0]] * batch_size, dtype=torch.bool)
        values.extend([hidden, mask])
    return values


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fixture(root, dropout=0.1):
    config = small_config(root, dropout)
    for path in (config.paths.esmc_model, config.paths.tcr_bert_model):
        path.mkdir()
        (path / 'config.json').write_text('{}')
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    for task in TASKS:
        for split, offset, size in [('train', 0, 8), ('val', 8, 4), ('test', 12, 4)]:
            rows = []
            for index in range(offset, offset + size):
                p, label = 'A' + alphabet[index], index % 2
                if task == 'ph':
                    row = [p, 'CC', label, split, 'native']
                elif task == 'pt':
                    row = [p, 'DD', label, split, 'native']
                else:
                    row = [p, 'CC', 'DD', 1 - label, 1 - label, label, split, 'synthetic-test']
                rows.append(row)
            header = {'ph': ['peptide', 'HLA', 'label', 'split_type', 'source'],
                      'pt': ['peptide', 'tcr', 'label', 'split_type', 'source'],
                      'pht': ['peptide', 'HLA', 'tcr', 'PH', 'PT', 'PHT', 'split_type', 'source_file']}[task]
            path = config.paths.data_root / task.upper() / f'{split}.csv'
            write_csv(path, header, rows)
            if split == 'test':
                write_csv(path.parent / 'benchmarks/test_unseen_peptide.csv', header, rows)
    datasets, report = audit_stage_MOE_data(config, ('train', 'val', 'test'))
    sequences = collect_sequences(datasets)
    rng = np.random.default_rng(42)
    for entity, values in sequences.items():
        dim = getattr(config.model, f'{entity}_input_dim')
        source = config.paths.tcr_bert_model if entity == 'tcr' else config.paths.esmc_model
        metadata = {'checkpoint_fingerprint': checkpoint_fingerprint(source), 'checkpoint_path': str(source),
                    'hidden_size': dim, 'max_length': 4, 'encoder_class': 'FixtureOnly'}
        seqs = sorted(values)
        hidden = rng.normal(size=(len(seqs), 4, dim)).astype(np.float16)
        mask = np.zeros((len(seqs), 4), dtype=bool)
        mask[:, :2] = True
        hidden[:, 2:] = 0
        with EmbeddingCacheWriter(config.paths.cache_root, entity, hidden_dim=dim, max_length=4,
                                  model_metadata=metadata, chunk_rows=1) as writer:
            writer.add_batch(seqs, hidden, mask, np.full(len(seqs), 2, dtype=np.int64))
    config_path = root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    return config, config_path


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = small_config(Path('/tmp/MOE_test'))
        torch.manual_seed(9)
        self.model = StageMOEPTBindingModel(self.config).eval()
        self.x = inputs(self.config)

    def test_shared_experts_and_optimizer_exact_coverage(self):
        experts = [*self.model.shared_experts, *(e for b in self.model.moe.values() for e in b.experts)]
        self.assertEqual(len(experts), 8)
        self.assertEqual(len({id(e) for e in experts}), 8)
        self.assertFalse(torch.equal(experts[0][0].weight, experts[1][0].weight))
        optimizer = build_stage_MOE_PT_optimizer(self.model, self.config.stage_MOE_PT)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.model.parameters()})
        self.assertFalse(any('feedforward' in key for key in self.model.cross_attention.state_dict()))

    def test_forward_parity_boundaries_and_routing(self):
        x = self.x
        with torch.no_grad():
            all_out = self.model.forward_all(*x, return_attention=True, return_features=True)
            ph = self.model('ph', *x[:4])
            pt = self.model('pt', *x[:2], *x[4:])
            pht = self.model.forward_triplet(*x)
            changed_t = list(x)
            changed_t[4] = torch.randn_like(x[4])
            changed_h = list(x)
            changed_h[2] = torch.randn_like(x[2])
            torch.testing.assert_close(all_out.logits['ph'], self.model.forward_all(*changed_t).logits['ph'])
            torch.testing.assert_close(all_out.logits['pt'], self.model.forward_all(*changed_h).logits['pt'])
        for task, out in [('ph', ph), ('pt', pt), ('pht', pht)]:
            torch.testing.assert_close(out.logits[task], all_out.logits[task])
            self.assertEqual(out.logits[task].shape, (3,))
            torch.testing.assert_close(all_out.routing[task].sum(-1), torch.ones(3))
            self.assertEqual(all_out.routing[task].shape, (3, 4))
            self.assertEqual(all_out.features[task].shape, (3, 8))
            self.assertEqual(torch.count_nonzero(all_out.pooling_weights[task][:, 2:]), 0)
        self.assertEqual(set(pht.routing), {'ph', 'pt', 'pht'})
        self.assertEqual(set(pht.logits), {'pht'})

    def test_padding_values_do_not_change_output(self):
        changed = [v.clone() for v in self.x]
        for index in (0, 2, 4):
            changed[index][:, 2:] = float('nan')
        with torch.no_grad():
            before, after = self.model.forward_all(*self.x), self.model.forward_all(*changed)
        for task in TASKS:
            torch.testing.assert_close(before.logits[task], after.logits[task])
            torch.testing.assert_close(before.routing[task], after.routing[task])
        changed[1].zero_()
        with self.assertRaises(ValueError):
            self.model.forward_all(*changed)

    def test_pht_gradient_reaches_ph_and_pt(self):
        out = self.model.forward_triplet(*self.x)
        out.logits['pht'].sum().backward()
        for module in (self.model.shared_experts, self.model.moe['ph'], self.model.cross_attention['ph'],
                       self.model.moe['pht'], self.model.hla_encoder, self.model.moe['pt'],
                       self.model.cross_attention['pt'], self.model.fusion):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        for module in (self.model.classifiers['pt'], self.model.classifiers['ph']):
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_dense_mixture_matches_manual_computation(self):
        block = self.model.moe['ph']
        h, c = torch.randn(2, 3, 8), torch.randn(2, 24)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        result, gates = block(h, c, mask, self.model.shared_experts, 1.5)
        expected = h + sum(gates[:, j, None, None] * module(block.norm(h))
                           for j, module in enumerate([*self.model.shared_experts, *block.experts]))
        expected = expected.masked_fill(~mask.unsqueeze(-1), 0)
        torch.testing.assert_close(result, expected)

    def test_masked_bce_excludes_nan_and_unsupervised_gradients(self):
        logits = torch.tensor([0.1, 0.4, -0.3], requires_grad=True)
        loss, count = masked_bce(logits, torch.tensor([1., float('nan'), 0.]), torch.tensor([True, True, False]))
        self.assertEqual(count, 1)
        loss.backward()
        self.assertEqual(logits.grad[1:].abs().sum(), 0)
        zero, count = masked_bce(logits, torch.full((3,), float('nan')), torch.zeros(3, dtype=torch.bool))
        self.assertEqual(count, 0)
        self.assertEqual(zero.item(), 0)
        self.assertTrue(torch.isfinite(zero))
        with self.assertRaises(ValueError):
            masked_bce(logits, torch.tensor([2., 0., 0.]), torch.ones(3, dtype=torch.bool))

    def test_loss_masking_and_balance(self):
        logits = torch.zeros(2, requires_grad=True)
        gates = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]], requires_grad=True)
        out = StageMOEPTOutput({'ph': logits}, {'ph': gates})
        result = compute_stage_MOE_loss(out, {'ph': torch.tensor([0., 1.])},
                                       {'ph': torch.ones(2, dtype=torch.bool)}, balance_lambda=0.01)
        self.assertAlmostEqual(result.balance.item(), 0.25)
        result.total.backward()
        self.assertIsNotNone(gates.grad)
        empty = compute_stage_MOE_loss(out, {'ph': torch.full((2,), float('nan'))},
                                      {'ph': torch.zeros(2, dtype=torch.bool)}, balance_lambda=1.)
        self.assertEqual(empty.total.item(), 0)

    def test_default_config_and_rejection(self):
        config = load_stage_MOE_PT_config(ROOT / 'configs/plm_stage_MOE_PT.yaml')
        self.assertEqual(config.stage_MOE_PT.temperature(1), 2.)
        self.assertEqual(config.stage_MOE_PT.temperature(5), 1.)
        self.assertEqual(config.stage_MOE_PT.temperature(30), 1.)
        config.stage_MOE_PT.learning_rate = float('nan')
        with self.assertRaises(ValueError):
            config.validate()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stage_MOE_PT_')
        self.root = Path(self.temp.name)
        self.config, self.path = fixture(self.root)
        self.caches = open_caches(self.config)
        self.data, self.audit = audit_stage_MOE_data(self.config)

    def tearDown(self):
        for cache in self.caches.values():
            cache.close()
        self.temp.cleanup()

    def test_native_labels_and_optional_input(self):
        for task in TASKS:
            for r in self.data['train'][task]:
                self.assertEqual(sum(r.masks), 1)
                self.assertTrue(r.masks[TASKS.index(task)])
                self.assertTrue(np.isfinite(r.labels[TASKS.index(task)]))
        self.assertIsNone(self.data['train']['ph'][0].tcr)
        self.assertIsNone(self.data['train']['pt'][0].hla)
        row = self.data['train']['pht'][0]
        self.assertEqual(row.labels[2], 0)
        self.assertFalse(row.masks[0])
        self.assertTrue(np.isnan(row.labels[0]))

    def test_duplicate_conflict_and_split_overlap(self):
        path = self.config.paths.data_root / 'PH/train.csv'
        with path.open('a') as handle:
            handle.write('AA,CC,1,train,native\n')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            read_task_records(path, 'ph', self.config.model)
        write_csv(path, ['peptide', 'HLA', 'label'], [['AA', 'CC', 0], ['AA', 'CC', 0], ['AC', 'CC', 1]])
        records, stats = read_task_records(path, 'ph', self.config.model)
        self.assertEqual(len(records), 2)
        self.assertEqual(stats['duplicates'], 1)
        val = self.config.paths.data_root / 'PH/val.csv'
        write_csv(val, ['peptide', 'HLA', 'label'], [['AA', 'CC', 0], ['AD', 'CC', 1]])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit_stage_MOE_data(self.config)

    def test_missing_cache_and_model_fingerprint(self):
        records = deepcopy(self.data['train']['ph'])
        from dataclasses import replace
        records[0] = replace(records[0], peptide='YY')
        with self.assertRaises(KeyError):
            make_loader(records, 'ph', self.caches, self.config.stage_MOE_PT)
        (self.config.paths.esmc_model / 'config.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            open_caches(self.config)

    def test_invalid_sequences_are_reported_without_rewriting_csv(self):
        path = self.config.paths.data_root / 'PHT/train.csv'
        with path.open('a') as handle:
            handle.write('AA,CC,DX,1,1,0,train,invalid-source\n')
        before = file_sha256(path)
        records, stats = read_task_records(path, 'pht', self.config.model)
        self.assertEqual(len(records), 8)
        self.assertEqual(stats['invalid_sequence_rows'], 1)
        self.assertEqual(stats['rejected_rows'][0]['invalid'][0]['entity'], 'tcr')
        self.assertEqual(before, file_sha256(path))

    def test_benchmark_audit_checks_other_tasks_and_supervised_overlap(self):
        from source.plm_unified.stage_MOE_PT_cli import cold_report, collect_benchmarks
        records, _ = read_task_records(self.config.paths.data_root / 'PT/test.csv', 'pt', self.config.model)
        # TCR DD is present in the PHT training pool, even with PT removed.
        training = {'ph': self.data['train']['ph'], 'pht': self.data['train']['pht']}
        report = cold_report(records, Path('test_unseen_tcr.csv'), training)
        self.assertFalse(report['tcr']['strict_cold_eligible'])
        path = self.config.paths.data_root / 'PH/benchmarks/test_unseen_peptide.csv'
        write_csv(path, ['peptide', 'HLA', 'label'], [['AA', 'CC', 0]])
        with self.assertRaisesRegex(ValueError, 'overlap train'):
            collect_benchmarks(self.config, self.data, self.audit)

    def trainer(self, name):
        return StageMOEPTTrainer(StageMOEPTBindingModel(self.config), self.config, self.data, self.caches,
                               self.audit['dataset_fingerprint'], self.root / name)

    def test_epoch_resume_matches_uninterrupted_training(self):
        set_global_seed(self.config.stage_MOE_PT.seed)
        continuous = self.trainer('continuous')
        continuous.fit()
        set_global_seed(self.config.stage_MOE_PT.seed)
        first = self.trainer('restart')
        first.fit(stop_after_epoch=1)
        resumed = self.trainer('restart')
        resumed.fit(resume=self.root / 'restart/last.pt')
        self.assertEqual(resumed.global_step, continuous.global_step)
        self.assertEqual(resumed.scheduler.state_dict(), continuous.scheduler.state_dict())
        for key, value in continuous.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
        # Unavailable group metrics are NaN; compare serialization so NaN != NaN
        # does not obscure an otherwise exact deterministic validation result.
        self.assertEqual(json.dumps(resumed.history[-1]['validation'], sort_keys=True),
                         json.dumps(continuous.history[-1]['validation'], sort_keys=True))
        checkpoint = load_checkpoint(self.root / 'restart/best.pt', config=self.config)
        self.assertEqual(set(checkpoint['thresholds']), set(TASKS))
        records, _ = read_task_records(self.config.paths.data_root / 'PHT/test.csv', 'pht', self.config.model)
        model = StageMOEPTBindingModel(self.config)
        model.load_state_dict(checkpoint['model'])
        model.temperature = checkpoint['temperature']
        loader = make_loader(records, 'pht', self.caches, self.config.stage_MOE_PT)
        result = evaluate_task(model, loader, 'pht', threshold=checkpoint['thresholds']['pht'],
                               prediction_path=self.root / 'prediction.csv')
        self.assertEqual(result['metrics']['samples'], 4)
        self.assertEqual(result['threshold'], checkpoint['thresholds']['pht'])
        self.assertTrue((self.root / 'prediction.csv').is_file())
        self.config.stage_MOE_PT.balance_lambda = 0.5
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            self.trainer('bad').resume(self.root / 'restart/last.pt')
        torch.save({'architecture': 'stage8'}, self.root / 'stage8.pt')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            load_checkpoint(self.root / 'stage8.pt')

    def test_balanced_task_microbatches_and_cpu_updates(self):
        # PT is the longest task. PH/PHT must cycle rather than disappear.
        self.data['train']['ph'] = self.data['train']['ph'][:4]
        self.data['train']['pht'] = self.data['train']['pht'][:4]
        trainer = self.trainer('cycling')
        initial = trainer.model.shared_experts[0][0].weight.detach().clone()
        stats = trainer.train_epoch(1)
        self.assertEqual(stats['sampled_rows'], {'ph': 8, 'pt': 8, 'pht': 8})
        self.assertEqual(stats['optimizer_steps'], 2)
        self.assertFalse(torch.equal(initial, trainer.model.shared_experts[0][0].weight))
        self.assertTrue(all(np.isfinite(v) for v in stats['loss'].values()))

    def test_cli_train_resume_test_and_benchmarks(self):
        def run(script, *args):
            result = subprocess.run([sys.executable, str(ROOT / 'source' / script), '--config', str(self.path), *map(str, args)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60,
                                    env={**__import__('os').environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result
        run('audit_plm_stage_MOE_PT_data.py', '--include-benchmarks', '--output-dir', self.root / 'audit')
        run('train_plm_stage_MOE_PT.py', '--output-dir', self.root / 'cli_train')
        checkpoint = self.root / 'cli_train/best.pt'
        checkpoint_hash = file_sha256(checkpoint)
        run('train_plm_stage_MOE_PT.py', '--resume', self.root / 'cli_train/last.pt')
        self.assertEqual(checkpoint_hash, file_sha256(checkpoint))
        run('evaluate_plm_stage_MOE_PT.py', '--checkpoint', checkpoint, '--output-dir', self.root / 'eval')
        run('evaluate_plm_stage_MOE_PT_benchmarks.py', '--checkpoint', checkpoint, '--output-dir', self.root / 'bench')
        result = json.loads((self.root / 'bench/metrics.json').read_text())
        self.assertEqual(len(result['results']), 3)
        self.assertTrue(all(value['cold_audit']['peptide']['strict_cold_eligible'] for value in result['results'].values()))
        self.assertTrue((self.root / 'bench/metrics.md').is_file())


if __name__ == '__main__':
    unittest.main()
