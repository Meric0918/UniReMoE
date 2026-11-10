"""Order, per-task updates, scheduler sizing and round-boundary recovery."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_stage_MOE import fixture, ROOT
from source.plm_unified.stage_MOE_checkpoint import load_checkpoint
from source.plm_unified.stage_MOE_data import open_caches, audit_stage_MOE_data
from source.plm_unified.stage_MOE_model import StageMOEBindingModel
from source.plm_unified.stage_MOE_trainer import StageMOETrainer
from source.plm_unified import stage_MOE_sequential_trainer as sequential
from source.plm_unified.trainer import set_global_seed


class SequentialTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stage_MOE_sequential_')
        self.root = Path(self.temp.name)
        self.config, self.config_path = fixture(self.root)
        self.caches = open_caches(self.config)
        self.data, self.audit = audit_stage_MOE_data(self.config)

    def tearDown(self):
        for cache in self.caches.values():
            cache.close()
        self.temp.cleanup()

    def trainer(self, name, cls=sequential.StageMOESequentialTrainer):
        return cls(StageMOEBindingModel(self.config), self.config, self.data, self.caches,
                   self.audit['dataset_fingerprint'], self.root / name)

    def test_full_task_order_each_batch_update_and_no_repetition(self):
        self.data['train']['ph'] = self.data['train']['ph'][:5]
        self.data['train']['pht'] = self.data['train']['pht'][:3]
        trainer = self.trainer('order')
        self.assertEqual(trainer.steps_per_epoch, 5)
        events, updated = [], []
        original_forward, original_update = sequential.forward_batch, trainer._optimizer_step

        def forward(model, task, batch):
            events.append(task)
            return original_forward(model, task, batch)

        def update():
            task = events[-1]
            # Only parameters reached by the current task get updates/weight decay.
            if task != 'pt':
                self.assertTrue(all(p.grad is None for p in trainer.model.moe['pt'].parameters()))
            if task == 'pht':
                self.assertTrue(any(p.grad is not None for p in trainer.model.moe['ph'].parameters()))
            before = {t: trainer.model.moe[t].experts[0][0].weight.detach().clone() for t in ('ph', 'pt', 'pht')}
            result = original_update()
            if task == 'ph':
                torch.testing.assert_close(before['pt'], trainer.model.moe['pt'].experts[0][0].weight, rtol=0, atol=0)
            if task == 'pt':
                torch.testing.assert_close(before['ph'], trainer.model.moe['ph'].experts[0][0].weight, rtol=0, atol=0)
            if task == 'pht':
                torch.testing.assert_close(before['pt'], trainer.model.moe['pt'].experts[0][0].weight, rtol=0, atol=0)
            updated.append(task)
            return result

        with patch.object(sequential, 'forward_batch', side_effect=forward), patch.object(trainer, '_optimizer_step', side_effect=update):
            stats = trainer.train_epoch(1)
        self.assertEqual(events, ['ph', 'ph', 'pt', 'pt', 'pht'])
        self.assertEqual(updated, events)
        self.assertEqual(stats['sampled_rows'], {'ph': 5, 'pt': 8, 'pht': 3})
        self.assertEqual(stats['task_optimizer_steps'], {'ph': 2, 'pt': 2, 'pht': 1})
        self.assertEqual(trainer.global_step, 5)
        self.assertEqual(trainer.scheduler.last_epoch, 5)

    def test_debug_step_cap_applies_to_all_three_tasks(self):
        self.config.stage_MOE.max_steps_per_epoch = 1
        trainer = self.trainer('limited')
        self.assertEqual(trainer.steps_per_epoch, 3)
        stats = trainer.train_epoch(1)
        self.assertEqual(stats['task_optimizer_steps'], {'ph': 1, 'pt': 1, 'pht': 1})
        self.assertEqual(stats['sampled_rows'], {'ph': 4, 'pt': 4, 'pht': 4})

    def test_sequential_resume_is_exact_and_modes_cannot_mix(self):
        set_global_seed(self.config.stage_MOE.seed)
        continuous = self.trainer('continuous')
        continuous.fit()
        set_global_seed(self.config.stage_MOE.seed)
        interrupted = self.trainer('interrupted')
        interrupted.fit(stop_after_epoch=1)
        checkpoint = load_checkpoint(self.root / 'interrupted/last.pt')
        self.assertEqual(checkpoint['training_mode'], 'sequential')
        self.assertEqual(checkpoint['contract']['schedule']['task_order'], ['ph', 'pt', 'pht'])
        resumed = self.trainer('interrupted')
        resumed.fit(resume=self.root / 'interrupted/last.pt')
        for key, value in continuous.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
        self.assertEqual(continuous.global_step, resumed.global_step)
        self.assertEqual(json.dumps(continuous.history[-1]['validation'], sort_keys=True),
                         json.dumps(resumed.history[-1]['validation'], sort_keys=True))
        joint = self.trainer('joint', StageMOETrainer)
        with self.assertRaisesRegex(ValueError, 'training mode mismatch'):
            joint.resume(self.root / 'interrupted/last.pt')
        joint.fit(stop_after_epoch=1)
        with self.assertRaisesRegex(ValueError, 'training mode mismatch'):
            resumed.resume(self.root / 'joint/last.pt')

    def test_entrypoint_resume_and_existing_evaluator(self):
        def run(action, *args):
            result = subprocess.run(['bash', str(ROOT / 'run_stage_MOE.sh'), action,
                                     '--config', str(self.config_path), *map(str, args)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60,
                                    env={**os.environ, 'PYTHON_BIN': __import__('sys').executable,
                                         'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result
        output = self.root / 'cli'
        run('train-sequential', '--output-dir', output)
        run('resume-sequential', '--resume', output / 'last.pt')
        run('evaluate', '--checkpoint', output / 'best.pt', '--output-dir', self.root / 'evaluation')
        history = json.loads((output / 'history.json').read_text())
        self.assertTrue(all(row['training']['task_order'] == ['ph', 'pt', 'pht'] for row in history))
        self.assertEqual(len(history), 2)
        self.assertTrue((self.root / 'evaluation/metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
