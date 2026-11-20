"""PH-only triplet supervision with the Stage Final binding model."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/stage_final'))
from test_stage_final import fixture, inputs
from source.plm_unified.stage_MOE_model import StageMOEOutput
from source.plm_unified.stage_final_checkpoint import load_checkpoint
from source.plm_unified.stage_final_data import audit_stage_final_data, open_caches, read_task_records
from source.plm_unified.stage_final_model import StageFinalBindingModel
from source.plm_unified.stage_final_new import (
    StageFinalNewBindingModel, StageFinalNewSettings, compute_stage_final_new_loss, load_stage_final_new_config)
from source.plm_unified.stage_final_sequential_trainer import StageFinalSequentialTrainer
from source.plm_unified.stage_final_trainer import StageFinalTrainer
from source.plm_unified.trainer import set_global_seed


@pytest.fixture
def experiment(tmp_path):
    baseline, path = fixture(tmp_path)
    raw = baseline.to_dict()
    settings = raw.pop('stage_final')
    del settings['aux_pt_weight'], settings['pt_implication_lambda']
    raw['stage_final_new'] = settings
    raw['paths']['output_root'] = str(tmp_path / 'stage_final_new_output')
    path.write_text(yaml.safe_dump(raw))
    return baseline, load_stage_final_new_config(path), path


def objective(codes):
    n = len(codes)
    logits = {task: torch.logit(torch.full((n,), p)).requires_grad_()
              for task, p in [('ph', .2), ('pht', .8), ('pt', .3)]}
    labels = {task: torch.tensor([float(c[i]) for c in codes]) for i, task in enumerate(('ph', 'pt', 'pht'))}
    masks = {task: torch.ones(n, dtype=torch.bool) for task in labels}
    settings = StageFinalNewSettings(consistency_warmup_epochs=0, consistency_ramp_epochs=1)
    output = StageMOEOutput(logits, {'pht': torch.full((n, 4), .25)})

    def calculate():
        return compute_stage_final_new_loss(output, labels, masks, task='pht', settings=settings, epoch=1)

    result = calculate()
    labels['pt'].fill_(float('nan'))
    masks['pt'].fill_(False)
    output.logits['pt'] = torch.full((n,), float('nan'))
    torch.testing.assert_close(calculate().total, result.total)
    expected_bce = F.binary_cross_entropy_with_logits(logits['pht'], labels['pht'])
    expected_bce += .1 * F.binary_cross_entropy_with_logits(logits['ph'], labels['ph'])
    torch.testing.assert_close(result.supervised, expected_bce)
    return result, logits


def test_groups_and_gradients():
    result, logits = objective(['111', '000', '010', '100'])
    hierarchy = result.components['hierarchy']
    assert hierarchy.item() == pytest.approx(2 * .55**2)
    assert result.counts['hierarchy_positive'] == 1 and result.counts['hierarchy_negative'] == 2
    assert result.counts['aux_ph_bce'] == result.counts['native_bce'] == 4
    torch.testing.assert_close(result.total, result.supervised + .05 * hierarchy)
    dh, dt = torch.autograd.grad(hierarchy, (logits['ph'], logits['pht']))
    assert dh[0] < 0 and torch.equal(dh[1:], torch.zeros(3))
    assert dt[1] > 0 and dt[2] > 0 and dt[0] == dt[3] == 0
    smaller, _ = objective(['111', '000', '010'])
    torch.testing.assert_close(hierarchy, smaller.components['hierarchy'])
    assert smaller.counts['hierarchy'] == result.counts['hierarchy'] == 3


def test_100_only_trains_both_bces():
    result, logits = objective(['100'])
    assert result.components['hierarchy'] == result.counts['hierarchy'] == 0
    assert result.total == result.supervised
    result.total.backward()
    assert logits['ph'].grad < 0 and logits['pht'].grad > 0


def test_model_equivalence(experiment):
    baseline, config, _ = experiment
    model = StageFinalNewBindingModel(config).eval()
    ref = StageFinalBindingModel(baseline).eval()
    ref.load_state_dict(model.state_dict(), strict=True)
    values = inputs(config)
    expected, actual = ref.forward_all(*values), model.forward_all(*values)
    torch.testing.assert_close(expected.logits, actual.logits, rtol=0, atol=0)
    torch.testing.assert_close(expected.routing, actual.routing, rtol=0, atol=0)
    torch.testing.assert_close(expected.fusion_beta, actual.fusion_beta, rtol=0, atol=0)


def test_optional_triplet_columns(experiment, tmp_path):
    _, config, _ = experiment
    path = tmp_path / 'triplets.csv'
    path.write_text('peptide,HLA,tcr,PH,PHT\nAA,CC,DD,1,0\n')
    records, _ = read_task_records(path, 'pht', config.model, expected_split='train', ph_only_triplets=True)
    assert records[0].masks == (True, False, True)
    path.write_text('peptide,HLA,tcr,PHT\nAA,CC,DD,0\n')
    with pytest.raises(ValueError, match='missing columns'):
        read_task_records(path, 'pht', config.model, expected_split='train', ph_only_triplets=True)


@pytest.mark.parametrize('mode, trainer_type', [('joint', StageFinalTrainer), ('sequential', StageFinalSequentialTrainer)])
def test_training_resume_and_evaluation(experiment, tmp_path, mode, trainer_type):
    _, config, path = experiment
    run_dir = tmp_path / 'full'
    suffix = '' if mode == 'joint' else '-sequential'

    def run(action, *args):
        subprocess.run([str(ROOT / 'run_stage_final_new.sh'), action, '--config', str(path), *map(str, args)],
                       check=True)

    run('train' + suffix, '--output-dir', run_dir)
    last = load_checkpoint(run_dir / 'last.pt', config=config)
    assert last['architecture'] == 'stage_final_new_v1'
    assert last['training_mode'] == mode
    assert not any(key in json.dumps(last['history']) for key in ('pt_implication', 'aux_pt_bce', 'pt_negative', 'pt_positive'))
    for epoch in last['history']:
        assert set(epoch['training']['loss_details']['weights']) == {'hierarchy'}
        assert 'pt/native_bce' in epoch['training']['loss_details']['components']

    datasets, audit = audit_stage_final_data(config)
    assert all(not rec.masks[1] for rec in datasets['train']['pht'])
    assert audit['binary_supervision_audit']['pt']['training_pairs'] == len(datasets['train']['pt'])
    caches = open_caches(config)
    partial = tmp_path / 'partial'
    try:
        set_global_seed(config.stage_final.seed)
        trainer = trainer_type(StageFinalNewBindingModel(config), config, datasets, caches,
                               audit['dataset_fingerprint'], partial)
        trainer.fit(stop_after_epoch=1)
    finally:
        for c in caches.values():
            c.close()
    run('resume' + suffix, '--resume', partial / 'last.pt')
    resumed = load_checkpoint(partial / 'last.pt', config=config)
    torch.testing.assert_close(last['model'], resumed['model'], rtol=0, atol=0)

    # Exercise both fully labelled and PHT-only heldout rows.
    test_file = config.paths.data_root / 'PHT/test.csv'
    with test_file.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row['PH'] = row['PT'] = row['class_code'] = ''
    with test_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for action in ('evaluate', 'benchmark'):
        dest = tmp_path / action
        run(action, '--checkpoint', run_dir / 'best.pt', '--output-dir', dest)
        report = json.loads((dest / 'metrics.json').read_text())
        assert report['label_policy'] == 'triplet_ph_supervision'
        assert any(k.startswith('pt/') for k in report['results'])
        pht = next(v for k, v in report['results'].items() if k.startswith('pht/'))
        constraints = pht['consistency']['constraints']
        assert set(constraints) == {'hierarchy'}
        applicable = constraints['hierarchy']['applicable']
        with next(dest.glob('pht*predictions.csv')).open() as handle:
            predictions = list(csv.DictReader(handle))
        assert 'pt_implication_excess' not in predictions[0]
        if action == 'benchmark':
            assert pht['consistency']['classes']['100']['samples'] == 1
            assert applicable['samples'] == 3
            assert next(r for r in predictions if r['class_code'] == '100')['hierarchy_violation'] == ''
        else:
            assert applicable['samples'] == 0 and applicable['violation_rate'] is None
            assert all(r['hierarchy_excess'] == '' for r in predictions)


def test_audit_distinguishes_native_and_triplet_pt_pairs(experiment):
    _, config, _ = experiment
    root = config.paths.data_root
    with (root / 'PT/test.csv').open() as handle:
        heldout = next(csv.DictReader(handle))
    path = root / 'PHT/train.csv'
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields, row = reader.fieldnames, next(reader)
    row.update(peptide=heldout['peptide'], tcr=heldout['tcr'], HLA='EE', PH='1', PHT='0', class_code='100')
    with path.open('a') as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)
    _, report = audit_stage_final_data(config)
    assert report['binary_supervision_audit']['pt']['overlap_rows'] == 0
    path = root / 'PT/train.csv'
    heldout['split_type'] = 'train'
    with path.open() as handle:
        fields = csv.DictReader(handle).fieldnames
    with path.open('a') as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(heldout)
    with pytest.raises(ValueError, match='binary supervision leakage'):
        audit_stage_final_data(config)
