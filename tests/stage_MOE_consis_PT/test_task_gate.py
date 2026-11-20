import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/stage_MOE_consis'))
from test_stage_MOE_consis import fixture, inputs
from source.plm_unified.stage_MOE_consis_model import StageMOEConsisBindingModel
from source.plm_unified.stage_MOE_consis_PT_config import LegacyTaskGateConfig, load_stage_MOE_consis_PT_config
from source.plm_unified.stage_MOE_consis_PT_evaluation import evaluation_setup
from source.plm_unified.stage_MOE_consis_PT_model import LegacyTaskGateBindingModel, StageMOEConsisPTBindingModel
from source.plm_unified.stage_MOE_PT_model import StageMOEPTBindingModel
from source.plm_unified.stage_MOE_PT_evaluation import FusionStatistics
from source.plm_unified.stage_final_checkpoint import load_checkpoint, model_contract
from source.plm_unified.stage_final_config import load_stage_final_config
from source.plm_unified.stage_final_trainer import StageFinalTrainer, build_stage_final_optimizer
from source.plm_unified.stage_final_sequential_trainer import StageFinalSequentialTrainer
from source.plm_unified.stage_final_data import audit_stage_final_data, open_caches
from source.plm_unified.trainer import set_global_seed


@pytest.fixture
def experiment(tmp_path):
    baseline, path = fixture(tmp_path)
    raw = baseline.to_dict()
    raw['paths']['output_root'] = str(tmp_path / 'stage_MOE_consis_PT_output')
    raw['stage_MOE_consis_PT'] = raw.pop('stage_MOE_consis')
    path.write_text(yaml.safe_dump(raw))
    return baseline, load_stage_MOE_consis_PT_config(path), path


def test_sample_gate_and_gradient(experiment):
    _, config, _ = experiment
    torch.manual_seed(13)
    model = StageMOEConsisPTBindingModel(config).eval()
    values = inputs(config)
    values[1][0, 2] = True
    captured = {}

    def capture_fusion(module, args, output):
        query, pt, _, mask = args
        pt.retain_grad()
        captured.update(query=query, pt=pt, mask=mask, fused=output[0])

    with model.fusion.register_forward_hook(capture_fusion):
        output = model.forward_all(*values)
    assert output.fusion_alpha.shape == (values[0].shape[0],)
    assert (output.fusion_alpha == 0.5).all()
    pt, mask = captured['pt'], captured['mask']
    residual = model.fusion.projection(model.fusion.pt_norm(pt))
    expected = captured['query'] + 0.5 * output.fusion_beta[..., None] * residual
    torch.testing.assert_close(captured['fused'], expected.masked_fill(~mask[..., None], 0))
    assert not output.fusion_beta[~mask].any()

    output.logits['pht'].sum().backward()
    for grad in (pt.grad, model.sample_gate[-1].weight.grad, model.fusion.gate[-1].weight.grad):
        assert grad.isfinite().all() and grad.abs().sum() > 0
    settings = config.stage_MOE_consis_PT
    optimizer = build_stage_final_optimizer(model, settings)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    for param in [*model.sample_gate.parameters(), *model.fusion.gate.parameters()]:
        assert rates[id(param)] == settings.router_learning_rate

    optimizer.step()
    with torch.no_grad():
        updated = model.forward_all(*values)
        assert updated.fusion_alpha.std() > 0
        # Recompute using the updated encoders and projection.
        def verify(module, args, result):
            query, pt, _, mask = args
            transfer = module.projection(module.pt_norm(pt))
            expected = query + updated.fusion_alpha[:, None, None] * result[1][..., None] * transfer
            torch.testing.assert_close(result[0], expected.masked_fill(~mask[..., None], 0))
        with model.fusion.register_forward_hook(verify):
            model.forward_all(*values)
        for i in range(len(values[0])):
            single = model.forward_all(*(value[i:i+1] for value in values))
            torch.testing.assert_close(single.fusion_alpha, updated.fusion_alpha[i:i+1])
            torch.testing.assert_close(single.logits['pht'], updated.logits['pht'][i:i+1])


@pytest.mark.parametrize('transfer', [False, True])
def test_ablations(experiment, transfer):
    baseline, config, _ = experiment
    settings = config.stage_MOE_consis_PT
    settings.pt_fusion_enabled = transfer
    settings.sample_gate_enabled = False
    model = StageMOEConsisPTBindingModel(config).eval()
    ref = StageMOEPTBindingModel(config) if transfer else StageMOEConsisBindingModel(baseline)
    ref.load_state_dict({k: model.state_dict()[k] for k in ref.state_dict()})
    values = inputs(config)
    expected = ref.eval().forward_all(*values)
    actual = model.forward_all(*values)
    for task in expected.logits:
        torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)
    if transfer:
        assert (actual.fusion_alpha == 1).all()
    else:
        assert actual.fusion_beta is None


def test_sample_and_residue_weighted_statistics():
    stats = FusionStatistics()
    stats.update(torch.tensor([[.5, .5], [1., .75]]), torch.tensor([[1, 1], [1, 0]]),
                 alpha=torch.tensor([.25, .75]))
    stats.update(torch.tensor([[.75]]), torch.tensor([[1]]), alpha=torch.tensor([.5]))
    result = stats.result()
    assert result['samples'] == 3 and result['residues'] == 4
    assert result['alpha_mean'] == .5
    assert result['alpha_std'] == pytest.approx((.125 / 3)**.5)
    assert result['beta_mean'] == 2.75 / 4
    assert result['effective_mean'] == (.25 + .75 + .375) / 4


def test_effective_mean_weights_each_valid_residue():
    stats = FusionStatistics()
    stats.update(torch.tensor([[0.25, 0.75, 0.9]]), torch.tensor([[1, 1, 0]]), alpha=torch.tensor(0.5))
    stats.update(torch.tensor([[0.75]]), torch.tensor([[1]]), alpha=torch.tensor(1.0))
    result = stats.result()
    assert result['alpha'] == 1.0
    assert result['beta_mean'] == pytest.approx(1.75 / 3)
    assert result['effective_mean'] == pytest.approx(1.25 / 3)


@pytest.mark.parametrize('suffix', ['', '-sequential'])
def test_cli_lifecycle(experiment, tmp_path, suffix):
    _, config, path = experiment
    run_dir = tmp_path / 'train'

    def run(action, *args):
        subprocess.run([str(ROOT / 'run_stage_MOE_consis_PT.sh'), action, '--config', str(path),
                        *map(str, args)], check=True)

    run('train' + suffix, '--output-dir', run_dir)
    last = load_checkpoint(run_dir / 'last.pt', config=config)
    assert last['architecture'] == 'stage_MOE_consis_PT_v2'
    fusion = last['history'][0]['training']['fusion']
    assert 0 < fusion['effective_mean'] < fusion['beta_mean'] < 1
    assert fusion['alpha_std'] > 0
    assert last['model']['sample_gate.2.weight'].abs().sum() > 0
    with pytest.raises(ValueError, match='stage_final_v1'):
        load_checkpoint(run_dir / 'best.pt')
    partial_dir = tmp_path / 'partial'
    datasets, audit = audit_stage_final_data(config)
    caches = open_caches(config)
    try:
        set_global_seed(config.stage_MOE_consis_PT.seed)
        trainer_type = StageFinalSequentialTrainer if suffix else StageFinalTrainer
        trainer = trainer_type(StageMOEConsisPTBindingModel(config), config, datasets, caches,
                               audit['dataset_fingerprint'], partial_dir)
        trainer.fit(stop_after_epoch=1)
    finally:
        for cache in caches.values():
            cache.close()
    run('resume' + suffix, '--resume', partial_dir / 'last.pt')
    resumed = load_checkpoint(partial_dir / 'last.pt', config=config)
    assert resumed['epoch'] == last['epoch'] == 2
    for name, value in last['model'].items():
        torch.testing.assert_close(resumed['model'][name], value, rtol=0, atol=0)
    for action in ('evaluate', 'benchmark'):
        run(action, '--checkpoint', run_dir / 'best.pt', '--output-dir', tmp_path / action)
        report = json.loads((tmp_path / action / 'metrics.json').read_text())
        for name, result in report['results'].items():
            if name.startswith('pht/'):
                assert result['consistency']['classes']['111']['samples'] == 1
                with next((tmp_path / action).glob('pht*predictions.csv')).open() as stream:
                    alphas = torch.tensor([float(row['fusion_alpha']) for row in csv.DictReader(stream)])
                assert result['fusion']['alpha_mean'] == pytest.approx(alphas.mean().item())
                assert result['fusion']['alpha_std'] == pytest.approx(alphas.std(unbiased=False).item(), abs=1e-7)
    config.stage_MOE_consis_PT.sample_gate_enabled = False
    with pytest.raises(ValueError, match='model configuration mismatch'):
        load_checkpoint(run_dir / 'best.pt', config=config)


def test_evaluate_legacy_checkpoint_with_sample_gate_config(experiment, tmp_path):
    _, config, path = experiment
    legacy = load_stage_final_config(path, config_type=LegacyTaskGateConfig)
    legacy.stage_final.task_gate_logit_init = 1.25
    legacy_model = LegacyTaskGateBindingModel(legacy).eval()
    checkpoint_path = tmp_path / 'v1.pt'
    torch.save(dict(architecture=legacy.architecture, format_version=1, role='best',
                    model=legacy_model.state_dict(), model_contract=model_contract(legacy),
                    config=legacy.to_dict(), temperature=1., encoders={}, epoch=1,
                    thresholds=dict.fromkeys(('ph', 'pt', 'pht'), .5)), checkpoint_path)
    path.write_text(yaml.safe_dump(config.to_dict()))
    restored, model_type, checkpoint = evaluation_setup(path, checkpoint_path)
    model = model_type(restored).eval()
    model.load_state_dict(checkpoint['model'])
    values = inputs(config)
    actual = model.forward_all(*values)
    torch.testing.assert_close(actual.fusion_alpha, torch.tensor(1.25).sigmoid())
    torch.testing.assert_close(actual.logits['pht'], legacy_model.forward_all(*values).logits['pht'],
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match='evaluate or benchmark'):
        load_checkpoint(checkpoint_path, config=config, role='last')
