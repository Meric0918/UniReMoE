"""Online tracking uses separate task modes and remains optional in CPU tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from test_stage_MOE import small_config, fixture
from source.plm_unified.stage_MOE_tracking import StageMOETracker, scalar_metrics
from source.plm_unified.stage_MOE_trainer import StageMOETrainer
from source.plm_unified.stage_MOE_model import StageMOEBindingModel
from source.plm_unified.stage_MOE_data import audit_stage_MOE_data, open_caches
from source.plm_unified.stage_MOE_checkpoint import load_checkpoint


class TrackingTests(unittest.TestCase):
    def test_flatten_metrics_filters_nonfinite_and_includes_gate_weights(self):
        values = scalar_metrics({'val': {'aupr': .8, 'undefined': float('nan')},
                                 'routing': {'weights': [.1, .2, .3, .4]}, 'mode': 'joint'})
        self.assertEqual(values['val/aupr'], .8)
        self.assertEqual(values['routing/weights/3'], .4)
        self.assertNotIn('val/undefined', values)
        self.assertNotIn('mode', values)

    def test_online_init_mode_identity_and_resume(self):
        with TemporaryDirectory(prefix='stage_MOE_tracking_') as temp:
            root = Path(temp)
            config = small_config(root)
            config.swanlab.mode = 'online'
            run = SimpleNamespace(id='example', url='https://example.invalid/run', log=Mock(), finish=Mock())
            with patch.dict('sys.modules', {'swanlab': SimpleNamespace(init=Mock(return_value=run))}):
                import swanlab
                tracker = StageMOETracker(config, 'sequential', root, resume_id='example')
                self.assertEqual(swanlab.init.call_args.kwargs['resume'], 'must')
                self.assertEqual(swanlab.init.call_args.kwargs['mode'], 'online')
                self.assertEqual(swanlab.init.call_args.kwargs['config']['training_mode'], 'sequential')
                tracker.log({'train': {'loss': .5}}, step=10)
                run.log.assert_called_once_with({'train/loss': .5}, step=10)
                tracker.finish('crashed')
                run.finish.assert_called_once_with(state='crashed')
                self.assertTrue((root / 'swanlab.json').is_file())

    def test_offline_does_not_read_online_only_url(self):
        with TemporaryDirectory(prefix='stage_MOE_tracking_') as temp:
            root = Path(temp)
            config = small_config(root)
            config.swanlab.mode = 'offline'
            run = SimpleNamespace(id='offline', log=Mock(), finish=Mock())
            with patch.dict('sys.modules', {'swanlab': SimpleNamespace(init=Mock(return_value=run))}):
                tracker = StageMOETracker(config, 'joint', root)
                self.assertIsNone(tracker.url)
                tracker.finish()

    def test_training_logs_validation_and_checkpoints_run_id(self):
        import torch
        torch.set_num_threads(1)
        with TemporaryDirectory(prefix='stage_MOE_tracking_') as temp:
            root = Path(temp)
            config, _ = fixture(root)
            config.stage_MOE.log_every_steps = 1
            data, report = audit_stage_MOE_data(config)
            caches = open_caches(config)
            try:
                tracker = SimpleNamespace(run_id='test-run', log=Mock())
                trainer = StageMOETrainer(StageMOEBindingModel(config), config, data, caches,
                    report['dataset_fingerprint'], root / 'run', tracker=tracker)
                trainer.fit(stop_after_epoch=1)
                calls = tracker.log.call_args_list
                self.assertTrue(any('train_step' in call.args[0] for call in calls))
                self.assertTrue(any('val' in call.args[0] for call in calls))
                last = load_checkpoint(root / 'run/last.pt')
                best = load_checkpoint(root / 'run/best.pt')
                self.assertEqual(last['swanlab_run_id'], 'test-run')
                self.assertEqual(best['swanlab_run_id'], 'test-run')
            finally:
                for cache in caches.values():
                    cache.close()


if __name__ == '__main__':
    unittest.main()
