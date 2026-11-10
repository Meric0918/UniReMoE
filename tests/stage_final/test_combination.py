"""Checks at the boundary between fusion and consistency supervision."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from test_fusion import small_config
from test_stage_final import inputs
from source.plm_unified.stage_final_config import load_stage_final_config
from source.plm_unified.stage_final_model import StageFinalBindingModel
from source.plm_unified.stage_MOE_PT_model import StageMOEPTBindingModel
from source.plm_unified.stage_MOE_consis_model import StageMOEConsisBindingModel
from source.plm_unified.stage_final_loss import compute_stage_final_loss
from source.plm_unified.stage_MOE_loss import compute_stage_MOE_loss
from source.plm_unified.stage_final_checkpoint import load_checkpoint


class CombinationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.config = small_config(Path('/tmp/stage_final_combination'))
        self.model = StageFinalBindingModel(self.config).eval()
        self.x = inputs(self.config, 4)

    def test_pt_equivalence_and_consis_ablation(self):
        pt = StageMOEPTBindingModel(self.config).eval()
        pt.load_state_dict(self.model.state_dict())
        actual, expected = self.model.forward_all(*self.x), pt.forward_all(*self.x)
        for task in ('ph', 'pt', 'pht'):
            torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)
        torch.testing.assert_close(actual.fusion_beta, expected.fusion_beta, rtol=0, atol=0)
        # Consis only needs a settings view, without adding old public config keys.
        from types import SimpleNamespace
        config = SimpleNamespace(model=self.config.model, stage_MOE=self.config.stage_final,
                                 stage_MOE_consis=self.config.stage_final)
        consis = StageMOEConsisBindingModel(config).eval()
        consis.load_state_dict({k: v for k, v in self.model.state_dict().items() if not k.startswith('fusion.')})
        self.model.pt_fusion_enabled = False
        actual, expected = self.model.forward_all(*self.x), consis.forward_all(*self.x)
        for task in ('ph', 'pt', 'pht'):
            torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)

    def test_native_pht_loss_updates_fusion_and_pt_features_only(self):
        s = self.config.stage_final
        s.aux_ph_weight = s.aux_pt_weight = s.hierarchy_lambda = s.pt_implication_lambda = 0
        output = self.model.forward_all(*self.x)
        codes = ('000', '010', '100', '111')
        labels = {t: torch.tensor([float(c[i]) for c in codes]) for i, t in enumerate(('ph','pt','pht'))}
        masks = {t: torch.ones(4, dtype=torch.bool) for t in labels}
        loss = compute_stage_final_loss(output, labels, masks, task='pht', settings=s, epoch=8)
        native = compute_stage_MOE_loss(output, {'pht': labels['pht']}, {'pht': masks['pht']},
                                       balance_lambda=s.balance_lambda)
        torch.testing.assert_close(loss.total, native.total)
        loss.total.backward()
        for module in (self.model.fusion, self.model.cross_attention['pt'], self.model.moe['pt']):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        for p in self.model.classifiers['pt'].parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)

    def test_config_serializes_only_final_and_rejects_other_checkpoints(self):
        self.config.validate()
        self.assertEqual(set(self.config.to_dict()), {'paths', 'model', 'stage_final', 'swanlab'})
        with tempfile.TemporaryDirectory() as temp:
            import yaml
            path = Path(temp) / 'config.yaml'
            path.write_text(yaml.safe_dump(self.config.to_dict()))
            self.assertEqual(load_stage_final_config(path).to_dict(), self.config.to_dict())
            for arch in ('stage_MOE_v1', 'stage_MOE_PT_v2', 'stage_MOE_consis_v1'):
                checkpoint = Path(temp) / 'old.pt'
                torch.save({'architecture': arch, 'format_version': 1}, checkpoint)
                with self.assertRaisesRegex(ValueError, 'stage_final_v1'):
                    load_checkpoint(checkpoint)
