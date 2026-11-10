"""Command-line lifecycle for the independent Stage MOE v1 pipeline."""
from __future__ import annotations
import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Match the repository launchers' Transformer Engine/NVRTC environment setup.
_cuda_home = Path(sys.executable).resolve().parent.parent / 'cuda'
if 'CUDA_HOME' not in os.environ and _cuda_home.is_dir():
    os.environ['CUDA_HOME'] = str(_cuda_home)

import torch

from .stage_MOE_config import load_stage_MOE_config
from .stage_MOE_data import (TASKS, audit_stage_MOE_data, collect_sequences, open_caches,
                            read_task_records, make_loader)
from .stage_MOE_model import StageMOEBindingModel
from .stage_MOE_checkpoint import load_checkpoint
from .stage_MOE_evaluation import evaluate_task, write_json
from .stage_MOE_trainer import StageMOETrainer
from .stage_MOE_sequential_trainer import StageMOESequentialTrainer
from .trainer import resolve_device, set_global_seed


def parser_for(action):
    parser = argparse.ArgumentParser(description=f'Stage MOE v1: {action}')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/plm_stage_MOE.yaml')
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--limit-train-samples', type=int)
    parser.add_argument('--limit-eval-samples', type=int)
    parser.add_argument('--max-steps-per-epoch', type=int)
    parser.add_argument('--output-dir', type=Path)
    if action in ('audit', 'cache'):
        parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'),
                            default=['train', 'val', 'test'] if action == 'audit' else ['train', 'val'])
        parser.add_argument('--include-benchmarks', action='store_true')
    if action == 'cache':
        parser.add_argument('--overwrite', action='store_true')
    if action in ('train', 'train-sequential'):
        parser.add_argument('--resume', type=Path)
        parser.add_argument('--swanlab-mode', choices=('online', 'offline', 'local', 'disabled'))
    if action == 'train-sequential':
        parser.epilog = ('Each round runs one full PH pass, then PT, then PHT; every batch updates parameters. '
                         'Validation/early stopping occur after all three tasks. '
                         '--max-steps-per-epoch caps each task separately in this mode.')
    if action in ('evaluate', 'benchmark'):
        parser.add_argument('--checkpoint', type=Path, required=True)
        parser.add_argument('--split', choices=('val', 'test'), default='test')
        parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    return parser


def benchmark_paths(config):
    result = {}
    for task in TASKS:
        paths = sorted((config.paths.data_root / task.upper() / 'benchmarks').glob('test_*.csv'))
        if not paths:
            raise FileNotFoundError(f'No {task} benchmark test files')
        result[task] = paths
    return result


def collect_benchmarks(config, datasets, report):
    for task, paths in benchmark_paths(config).items():
        for path in paths:
            records, stats = read_task_records(path, task, config.model, expected_split='test')
            heldout = {r.key(task) for r in records}
            for split in ('train', 'val'):
                if split in datasets:
                    overlap = heldout & {r.key(task) for r in datasets[split][task]}
                    if overlap:
                        raise ValueError(f'{path}: {len(overlap)} supervised records overlap {split}')
            datasets[f'benchmark/{task}/{path.stem}'] = {task: records}
            report['files'][f'benchmark/{task}/{path.stem}'] = stats


def cold_report(records, path, training):
    result = {}
    for entity in ('peptide', 'hla', 'tcr'):
        values = {getattr(r, entity) for r in records if getattr(r, entity)}
        known = {getattr(r, entity) for rows in training.values() for r in rows if getattr(r, entity)}
        if values:
            overlap = len(values & known)
            result[entity] = {'unique_entities': len(values), 'seen_in_any_training_task': overlap,
                              'strict_cold_eligible': overlap == 0}
    result['claimed_cold_entity'] = next((e for e in ('peptide', 'hla', 'tcr') if f'unseen_{e}' in path.stem), None)
    return result


def precompute(config, datasets, overwrite):
    from .cache import EmbeddingCacheWriter, batched
    from .encoders import ESMCResidueEncoder, TCRBertResidueEncoder, encoder_metadata
    sequences = collect_sequences(datasets)
    for entity in sequences:
        if (config.paths.cache_root / entity).exists() and not overwrite:
            raise FileExistsError(f'{entity} cache exists; use --overwrite explicitly to rebuild the MOE cache')
    device, s = resolve_device(config.stage_MOE.device), config.stage_MOE
    encoder = None
    try:
        for entity in ('peptide', 'hla', 'tcr'):
            if entity in ('peptide', 'tcr'):
                if encoder is not None:
                    del encoder
                    encoder = None
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                cls = TCRBertResidueEncoder if entity == 'tcr' else ESMCResidueEncoder
                path = config.paths.tcr_bert_model if entity == 'tcr' else config.paths.esmc_model
                encoder = cls(path, max_length=getattr(config.model, f'{entity}_max_length'), device=device, dtype='float16')
            encoder.max_length = getattr(config.model, f'{entity}_max_length')
            if encoder.hidden_size != getattr(config.model, f'{entity}_input_dim'):
                raise ValueError(f'{entity}: PLM hidden dimension mismatch')
            ordered = sorted(sequences[entity])
            with EmbeddingCacheWriter(config.paths.cache_root, entity, hidden_dim=encoder.hidden_size,
                max_length=encoder.max_length, model_metadata=encoder_metadata(encoder), dtype='float16',
                shard_size=s.cache_shard_size, chunk_rows=1, overwrite=overwrite) as writer:
                for index, batch in enumerate(batched(ordered, s.cache_batch_size), 1):
                    with torch.inference_mode():
                        encoded = encoder.encode(batch)
                    writer.add_batch(batch, encoded.hidden.cpu().to(torch.float16).numpy(),
                                     encoded.mask.cpu().numpy(), encoded.lengths.cpu().numpy())
                    if index == 1 or index % 100 == 0:
                        print(f'cache {entity}: {min(index * s.cache_batch_size, len(ordered))}/{len(ordered)}', flush=True)
    finally:
        del encoder
    return {entity: len(values) for entity, values in sequences.items()}


def evaluate(config, args, action, caches, output):
    checkpoint = load_checkpoint(args.checkpoint, config=config)
    for entity, cache in caches.items():
        if checkpoint['encoders'][entity]['checkpoint_fingerprint'] != cache.manifest['model']['checkpoint_fingerprint']:
            raise ValueError(f'{entity}: checkpoint/cache encoder mismatch')
    model = StageMOEBindingModel(config).to(resolve_device(config.stage_MOE.device))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.temperature = checkpoint['temperature']
    datasets, audit = audit_stage_MOE_data(config, splits=('train', 'val'))
    paths = benchmark_paths(config) if action == 'benchmark' else {
        task: [config.paths.data_root / task.upper() / f'{args.split}.csv'] for task in args.tasks}
    result = {'architecture': model.architecture, 'checkpoint': str(args.checkpoint.resolve()),
              'training_mode': checkpoint.get('training_mode', 'joint'),
              'checkpoint_epoch': checkpoint['epoch'], 'temperature': model.temperature,
              'threshold_source': 'checkpoint_validation', 'label_policy': 'native_task_only', 'results': {}}
    for task in args.tasks:
        for path in paths[task]:
            records, stats = read_task_records(path, task, config.model,
                                               expected_split='test' if action == 'benchmark' else args.split)
            # Test records cannot coincide with training/validation supervision.
            if action == 'benchmark' or args.split == 'test':
                heldout = {r.key(task) for r in records}
                for split in ('train', 'val'):
                    overlap = heldout & {r.key(task) for r in datasets[split][task]}
                    if overlap:
                        raise ValueError(f'{path}: {len(overlap)} supervised records overlap {split}')
            loader = make_loader(records, task, caches, config.stage_MOE)
            key = f'{task}/{path.stem}'
            values = evaluate_task(model, loader, task, precision=config.stage_MOE.mixed_precision,
                                   threshold=checkpoint['thresholds'][task],
                                   prediction_path=output / f'{task}_{path.stem}_predictions.csv')
            values['data'] = stats
            values['cold_audit'] = cold_report(records, path, datasets['train'])
            result['results'][key] = values
    write_json(output / 'metrics.json', result)
    lines = ['# Stage MOE evaluation', '', f'Checkpoint: `{args.checkpoint}`', '',
             '| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for key, value in result['results'].items():
        m = value['metrics']
        entity = value['cold_audit']['claimed_cold_entity']
        cold = (f'{entity}: {value["cold_audit"][entity]["seen_in_any_training_task"]} seen in training'
                if entity else 'Mixed / combination split')
        lines.append(f'| {key} | {m["samples"]} | {m["auroc"]:.6f} | {m["aupr"]:.6f} | {m["mcc"]:.6f} | {m["f1"]:.6f} | {value["threshold"]:.6f} | {cold} |')
    (output / 'metrics.md').write_text('\n'.join(lines) + '\n')
    return result


def main(action, argv=None):
    args = parser_for(action).parse_args(argv)
    config = load_stage_MOE_config(args.config)
    if getattr(args, 'swanlab_mode', None) is not None:
        config.swanlab.mode = args.swanlab_mode
    for name in ('device', 'seed', 'num_workers', 'limit_train_samples', 'limit_eval_samples', 'max_steps_per_epoch'):
        value = getattr(args, name)
        if value is not None:
            setattr(config.stage_MOE, name, value)
    config.validate()
    set_global_seed(config.stage_MOE.seed)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output_dir or config.paths.output_root / f'seed_{config.stage_MOE.seed}' / f'{action}_{stamp}'
    is_training = action in ('train', 'train-sequential')
    if is_training and args.resume and args.output_dir is None:
        output = args.resume.resolve().parent
    output = Path(output).resolve()
    if is_training and not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Training output directory is not empty: {output}')
    if action in ('audit', 'cache'):
        datasets, report = audit_stage_MOE_data(config, splits=tuple(dict.fromkeys(args.splits)))
        if args.include_benchmarks:
            collect_benchmarks(config, datasets, report)
            if 'train' in datasets:
                report['benchmark_cold_audit'] = {
                    key: cold_report(next(iter(tasks.values())), Path(key.split('/')[-1]), datasets['train'])
                    for key, tasks in datasets.items() if key.startswith('benchmark/')}
        report['sequence_counts'] = {k: len(v) for k, v in collect_sequences(datasets).items()}
        write_json(output / 'audit.json', report)
        print(json.dumps({'audit': str(output / 'audit.json'), 'sequence_counts': report['sequence_counts']}), flush=True)
        if action == 'cache':
            precompute(config, datasets, args.overwrite)
            write_json(config.paths.cache_root / 'coverage.json', report)
        return report
    caches = open_caches(config)
    try:
        if is_training:
            datasets, report = audit_stage_MOE_data(config)
            model = StageMOEBindingModel(config).to(resolve_device(config.stage_MOE.device))
            trainer_class = StageMOESequentialTrainer if action == 'train-sequential' else StageMOETrainer
            trainer = trainer_class(model, config, datasets, caches, report['dataset_fingerprint'], output)
            if args.resume:
                trainer.resume(args.resume)
            write_json(output / 'audit.json', report)
            write_json(output / 'config.json', config.to_dict())
            from .stage_MOE_tracking import StageMOETracker
            tracker = StageMOETracker(config, trainer.training_mode, output, resume_id=trainer.swanlab_run_id)
            trainer.tracker = tracker
            try:
                tracker.log({'run': {'started': 1, 'resume_epoch': trainer.epoch,
                                    'max_epochs': config.stage_MOE.max_epochs,
                                    'patience': config.stage_MOE.patience,
                                    'parameters': sum(p.numel() for p in model.parameters())}}, step=trainer.global_step)
                trainer.fit()
            except BaseException:
                tracker.finish('crashed')
                raise
            else:
                tracker.finish()
            print(f'Artifacts: {output}', flush=True)
            return trainer.history
        output.mkdir(parents=True, exist_ok=True)
        return evaluate(config, args, action, caches, output)
    finally:
        for cache in caches.values():
            cache.close()
