"""Directed-gradient, data boundary and real-cache lifecycle regression tests."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/stage_MOE'))
from test_stage_MOE import fixture as original_fixture, inputs, write_csv
from source.plm_unified.stage_MOE_model import StageMOEOutput
from source.plm_unified.stage_MOE_consis_config import (StageMOEConsisConfig, StageMOEConsisPaths,
    StageMOEConsisSettings, load_stage_MOE_consis_config)
from source.plm_unified.stage_MOE_consis_model import StageMOEConsisBindingModel, forward_batch
from source.plm_unified.stage_MOE_consis_loss import compute_stage_MOE_consis_loss
from source.plm_unified.stage_MOE_consis_data import audit_stage_MOE_consis_data, read_task_records, open_caches
from source.plm_unified.stage_MOE_consis_trainer import StageMOEConsisTrainer
from source.plm_unified.stage_MOE_consis_sequential_trainer import StageMOEConsisSequentialTrainer
from source.plm_unified.stage_MOE_consis_checkpoint import load_checkpoint
from source.plm_unified.stage_MOE_consis_evaluation import evaluate_task, triplet_diagnostics
from source.plm_unified.stage_MOE_checkpoint import load_checkpoint as load_old_checkpoint
from source.plm_unified.trainer import set_global_seed

CODES = ('000', '010', '100', '111')


def fixture(root):
    original, _ = original_fixture(root)
    paths = asdict(original.paths)
    paths['output_root'] = root / 'MOE_consis_output'
    config = StageMOEConsisConfig(StageMOEConsisPaths(**paths), original.model,
        StageMOEConsisSettings(**asdict(original.stage_MOE), consistency_warmup_epochs=0,
                              consistency_ramp_epochs=1))
    for task in ('ph', 'pt', 'pht'):
        for path in (config.paths.data_root / task.upper()).rglob('*.csv'):
            with path.open() as handle:
                reader = csv.DictReader(handle)
                headers, rows = list(reader.fieldnames), list(reader)
            if task == 'pht':
                headers.append('class_code')
            for i, row in enumerate(rows):
                code = CODES[i % 4]
                if task == 'pht':
                    row.update(PH=code[0], PT=code[1], PHT=code[2], class_code=code)
                else:
                    row['label'] = code[0 if task == 'ph' else 1]
            write_csv(path, headers, [[row[h] for h in headers] for row in rows])
    path = root / 'consis.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    return config, path


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings = StageMOEConsisSettings(consistency_warmup_epochs=0, consistency_ramp_epochs=1)

    def calculate(self, codes=CODES, probabilities=None, epoch=1, task='pht'):
        n = len(codes)
        probabilities = probabilities or {'ph': [0.2]*n, 'pt': [0.3]*n, 'pht': [0.8]*n}
        logits = {t: torch.logit(torch.tensor(p)).requires_grad_() for t, p in probabilities.items()}
        labels = {t: torch.tensor([float(c[i]) for c in codes]) for i, t in enumerate(('ph','pt','pht'))}
        masks = {t: torch.ones(n, dtype=torch.bool) for t in labels}
        output = StageMOEOutput(logits, {t: torch.full((n, 4), 0.25) for t in logits})
        result = compute_stage_MOE_consis_loss(output, labels, masks, task=task, settings=self.settings, epoch=epoch)
        return result, logits

    def test_four_type_values_and_normalization(self):
        result, _ = self.calculate()
        self.assertAlmostEqual(float(result.components['hierarchy']), 2 * .55**2, places=6)
        self.assertAlmostEqual(float(result.components['pt_implication']), 2 * .45**2, places=6)
        self.assertEqual(result.valid_labels, 4)
        self.assertEqual(result.counts['hierarchy_negative'], 2)
        self.assertEqual(result.counts['pt_negative'], 2)
        expected = result.supervised + .05 * 2 * .55**2 + .03 * 2 * .45**2
        torch.testing.assert_close(result.total, expected)

    def test_directed_gradients_each_class(self):
        for code in CODES:
            with self.subTest(code=code):
                result, logits = self.calculate((code,))
                (result.components['hierarchy'] + result.components['pt_implication']).backward()
                grads = {t: 0 if z.grad is None else float(z.grad) for t, z in logits.items()}
                if code == '111':
                    self.assertEqual(grads['pht'], 0)
                    self.assertLess(grads['ph'], 0)
                    self.assertLess(grads['pt'], 0)
                else:
                    self.assertGreater(grads['pht'], 0)
                    self.assertEqual(grads['ph'], 0)
                    self.assertEqual(grads['pt'], 0)
                if code == '100':
                    self.assertEqual(float(result.components['hierarchy']), 0)
                if code == '010':
                    self.assertEqual(float(result.components['pt_implication']), 0)

    def test_margin_and_empty_groups(self):
        result, logits = self.calculate(('111',), {'ph':[.7], 'pt':[.8], 'pht':[.72]})
        self.assertEqual(float(result.components['hierarchy']), 0)
        self.assertEqual(float(result.components['pt_implication']), 0)
        result.total.backward()
        self.assertTrue(all(torch.isfinite(z.grad).all() for z in logits.values()))

    def test_ramp_zero_weights_and_native_pair(self):
        s = self.settings
        s.consistency_warmup_epochs, s.consistency_ramp_epochs = 3, 5
        self.assertEqual([s.consistency_ramp(e) for e in (1,3,4,8,9)], [0,0,.2,1,1])
        s.aux_ph_weight = s.aux_pt_weight = s.hierarchy_lambda = s.pt_implication_lambda = 0
        result, _ = self.calculate()
        torch.testing.assert_close(result.total, result.components['native_bce'])
        result, _ = self.calculate(task='ph')
        self.assertNotIn('aux_pt_bce', result.components)
        torch.testing.assert_close(result.total, result.components['native_bce'])

    def test_only_pht_routing_regularized(self):
        s = self.settings
        s.balance_lambda = 1
        labels = {t: torch.ones(2) for t in ('ph','pt','pht')}
        masks = {t: torch.ones(2, dtype=torch.bool) for t in labels}
        routing = {t: torch.tensor([[1.,0,0,0]]*2, requires_grad=True) for t in labels}
        routing['pht'] = torch.full((2,4), .25, requires_grad=True)
        out = StageMOEOutput({t:torch.zeros(2, requires_grad=True) for t in labels}, routing)
        result = compute_stage_MOE_consis_loss(out, labels, masks, task='pht', settings=s, epoch=1)
        self.assertEqual(float(result.balance), 0)
        result.total.backward()
        self.assertIsNone(routing['ph'].grad)
        self.assertIsNone(routing['pt'].grad)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.path = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_and_all_head_backward(self):
        config = load_stage_MOE_consis_config(self.path)
        model = StageMOEConsisBindingModel(config)
        output = model.forward_all(*inputs(config, 4))
        labels = {t:torch.tensor([float(c[i]) for c in CODES]) for i,t in enumerate(('ph','pt','pht'))}
        masks = {t:torch.ones(4,dtype=torch.bool) for t in labels}
        loss = compute_stage_MOE_consis_loss(output, labels, masks, task='pht', settings=config.stage_MOE_consis, epoch=1)
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.architecture, 'stage_MOE_consis_v1')

    def test_audit_requires_labels_and_detects_conflicts_before_limits(self):
        datasets, report = audit_stage_MOE_consis_data(self.config)
        self.assertTrue(all(all(r.masks) for r in datasets['train']['pht']))
        self.assertEqual(report['binary_supervision_audit']['ph']['training_pairs'], 8)
        path = self.config.paths.data_root / 'PHT/train.csv'
        content = path.read_text()
        path.write_text(content.replace('class_code','unused_code'))
        self.config.stage_MOE_consis.limit_train_samples = 1
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            audit_stage_MOE_consis_data(self.config)
        path.write_text(content)
        with path.open('a') as handle:
            handle.write('AA,CC,DD,1,1,1,train,conflict,111\n')
        with self.assertRaisesRegex(ValueError, 'conflicting pht'):
            audit_stage_MOE_consis_data(self.config)

    def test_binary_conflict_and_heldout_leakage(self):
        path = self.config.paths.data_root / 'PH/train.csv'
        content = path.read_text()
        path.write_text(content.replace('AA,CC,0', 'AA,CC,1'))
        with self.assertRaisesRegex(ValueError, 'binary supervision conflict'):
            audit_stage_MOE_consis_data(self.config)
        path.write_text(content)
        # Different TCR preserves triple identity, but the heldout PH pair leaks.
        with (self.config.paths.data_root/'PHT/train.csv').open('a') as handle:
            handle.write('AK,CC,EE,0,0,0,train,leak,000\n')
        with self.assertRaisesRegex(ValueError, 'binary supervision leakage'):
            audit_stage_MOE_consis_data(self.config)
        from source.plm_unified.stage_MOE_consis_cli import audit_with_report
        with self.assertRaisesRegex(ValueError, 'binary supervision leakage'):
            audit_with_report(self.config, self.root/'failed_audit')
        report=json.loads((self.root/'failed_audit/audit_failure.json').read_text())
        self.assertGreater(report['binary_supervision_audit']['ph']['overlap_rows'],0)

    def test_label_mismatch_and_missing_evaluation_auxiliaries(self):
        path = self.root/'eval.csv'
        write_csv(path, ['peptide','HLA','tcr','PH','PT','PHT','class_code'], [['AA','CC','DD',0,0,0,'111']])
        with self.assertRaisesRegex(ValueError, 'class_code/labels mismatch'):
            read_task_records(path,'pht',self.config.model,expected_split='test')
        write_csv(path, ['peptide','HLA','tcr','PHT'], [['AA','CC','DD',0],['AC','CC','DD',1]])
        records, _ = read_task_records(path,'pht',self.config.model,expected_split='test')
        self.assertEqual(records[0].masks, (False, False, True))
        from source.plm_unified.stage_MOE_consis_data import make_loader
        caches = open_caches(self.config)
        try:
            result = evaluate_task(StageMOEConsisBindingModel(self.config),
                make_loader(records,'pht',caches,self.config.stage_MOE_consis), 'pht', threshold=.5)
            self.assertEqual(result['metrics']['samples'],2)
            scope = result['consistency']['constraints']['hierarchy']
            self.assertEqual(scope['all']['samples'],2)
            self.assertIsNone(scope['applicable']['violation_rate'])
        finally:
            for cache in caches.values(): cache.close()

    def test_joint_and_sequential_exact_resume_and_counts(self):
        datasets, report = audit_stage_MOE_consis_data(self.config)
        caches = open_caches(self.config)
        try:
            for cls in (StageMOEConsisTrainer, StageMOEConsisSequentialTrainer):
                with self.subTest(mode=cls.training_mode):
                    def create(name):
                        set_global_seed(23)
                        return cls(StageMOEConsisBindingModel(self.config), self.config, datasets, caches,
                                   report['dataset_fingerprint'], self.root/name)
                    full = create(f'{cls.training_mode}_full')
                    full.fit()
                    partial = create(f'{cls.training_mode}_partial')
                    partial.fit(stop_after_epoch=1)
                    resumed = create(f'{cls.training_mode}_partial')
                    resumed.fit(resume=partial.run_dir/'last.pt')
                    for k, v in full.model.state_dict().items():
                        torch.testing.assert_close(v, resumed.model.state_dict()[k], rtol=0, atol=0)
                    self.assertEqual(full.global_step,resumed.global_step)
                    training=full.history[0]['training']
                    self.assertEqual(training['sampled_rows'],dict.fromkeys(('ph','pt','pht'),8))
                    self.assertEqual(training['optimizer_steps'],2 if cls.training_mode=='joint' else 6)
                    details=training['loss_details']['components']
                    self.assertEqual(details['pht/aux_ph_bce']['samples'],8)
                    self.assertNotIn('ph/aux_ph_bce',details)
                    with self.assertRaisesRegex(ValueError,'stage_MOE_v1'):
                        load_old_checkpoint(full.run_dir/'best.pt')
                    opposite = StageMOEConsisSequentialTrainer if cls is StageMOEConsisTrainer else StageMOEConsisTrainer
                    wrong = opposite(StageMOEConsisBindingModel(self.config),self.config,datasets,caches,
                                     report['dataset_fingerprint'],self.root/'wrong')
                    with self.assertRaisesRegex(ValueError,'mode mismatch'):
                        wrong.resume(full.run_dir/'last.pt')
                    wrong_config=deepcopy(self.config)
                    wrong_config.stage_MOE_consis.hierarchy_lambda=.9
                    wrong=cls(StageMOEConsisBindingModel(wrong_config),wrong_config,datasets,caches,
                              report['dataset_fingerprint'],self.root/'changed')
                    with self.assertRaisesRegex(ValueError,'contract mismatch'):
                        wrong.resume(full.run_dir/'last.pt')
        finally:
            for cache in caches.values(): cache.close()

    def test_cli_lifecycle_both_modes(self):
        def run(action,*args):
            completed = subprocess.run(['bash', str(ROOT/'run_stage_MOE_consis.sh'), action,
                '--config',str(self.path),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
        run('audit','--include-benchmarks','--output-dir',self.root/'audit')
        for action, resume in (('train','resume'),('train-sequential','resume-sequential')):
            output=self.root/action
            run(action,'--output-dir',output)
            run(resume,'--resume',output/'last.pt')
            checkpoint=load_checkpoint(output/'best.pt')
            for evaluation in ('evaluate','benchmark'):
                target=self.root/f'{action}_{evaluation}'
                run(evaluation,'--checkpoint',output/'best.pt','--output-dir',target)
                result=json.loads((target/'metrics.json').read_text())
                for key,value in result['results'].items():
                    self.assertEqual(value['threshold'],checkpoint['thresholds'][key.split('/')[0]])
                    if key.startswith('pht/'):
                        self.assertEqual(value['consistency']['classes']['111']['samples'],1)
                csv_path=next(target.glob('pht_*predictions.csv'))
                self.assertIn('p_ph',csv_path.read_text().splitlines()[0])

    def test_tracking_includes_components_and_validation(self):
        from source.plm_unified.stage_MOE_consis_tracking import StageMOEConsisTracker
        self.config.swanlab.mode='online'
        self.config.stage_MOE_consis.log_every_steps=1
        run=SimpleNamespace(id='fake-consis',url='https://example.invalid',log=Mock(),finish=Mock())
        swanlab=SimpleNamespace(init=Mock(return_value=run))
        datasets,report=audit_stage_MOE_consis_data(self.config)
        caches=open_caches(self.config)
        try:
            with patch.dict('sys.modules',{'swanlab':swanlab}):
                tracker=StageMOEConsisTracker(self.config,'joint',self.root/'tracked')
                trainer=StageMOEConsisTrainer(StageMOEConsisBindingModel(self.config),self.config,datasets,
                    caches,report['dataset_fingerprint'],self.root/'tracked',tracker=tracker)
                trainer.fit(stop_after_epoch=1)
                keys={key for call in run.log.call_args_list for key in call.args[0]}
                self.assertIn('train/loss_details/components/pht/hierarchy_positive/mean',keys)
                self.assertIn('val/tasks/pht/consistency/constraints/hierarchy/all/violation_rate',keys)
                self.assertEqual(load_checkpoint(trainer.run_dir/'last.pt')['swanlab_run_id'],'fake-consis')
                tracker.finish()
                run.finish.assert_called_once()
        finally:
            for cache in caches.values(): cache.close()

    def test_cache_reuse_and_explicit_rebuild_guard(self):
        from source.plm_unified.stage_MOE_consis_cli import main
        manifests={p:p.read_bytes() for p in self.config.paths.cache_root.rglob('manifest.json')}
        self.assertTrue(manifests)
        with self.assertRaisesRegex(FileExistsError,'cache exists'):
            main('cache',['--config',str(self.path),'--output-dir',str(self.root/'cache_audit')])
        for path,content in manifests.items():
            self.assertEqual(content,path.read_bytes())

    def test_diagnostic_denominators_and_empty_scopes(self):
        rows=[{'class_code':c,'complete_labels':True,'p_ph':.2,'p_pt':.3,'p_pht':.8} for c in CODES]
        result=triplet_diagnostics(rows,.5,.05)
        self.assertEqual(result['classes']['111']['recall'],1)
        self.assertEqual(result['constraints']['hierarchy']['all']['samples'],4)
        self.assertEqual(result['constraints']['hierarchy']['applicable']['samples'],3)
        self.assertAlmostEqual(result['constraints']['hierarchy']['applicable']['mean_excess'],.55)
        self.assertIsNone(triplet_diagnostics([], .5, .05)['classes']['111']['recall'])


if __name__ == '__main__':
    unittest.main()
