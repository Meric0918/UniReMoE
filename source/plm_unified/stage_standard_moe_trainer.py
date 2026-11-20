"""Independent Standard MoE lifecycle with PH, PT, PHT passes per round."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import torch

from .stage_standard_moe_checkpoint import (atomic_save, capture_rng, restore_rng, load_checkpoint,
                                   model_contract, training_contract)
from .stage_final_data import make_loader, move_batch, cache_identity
from .stage_standard_moe_evaluation import autocast_context, RoutingStatistics, evaluate_validation, write_json
from .stage_MOE_PT_evaluation import FusionStatistics
from .stage_standard_moe_loss import compute_stage_standard_moe_loss, LossStatistics
from .stage_standard_moe_model import TASKS, ARCHITECTURE, forward_batch
from .stage_final_trainer import build_stage_final_optimizer as build_stage_standard_moe_optimizer


class StageStandardMoESequentialTrainer:
    training_mode = 'sequential'

    def __init__(self, model, config, datasets, caches, data_fingerprint, run_dir, *, tracker=None):
        config.validate()
        self.model, self.config, self.datasets, self.caches = model, config, datasets, caches
        self.settings = s = config.stage_standard_moe
        self.tracker = tracker
        self.swanlab_run_id = None
        self.device = next(model.parameters()).device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.contract = training_contract(config, data_fingerprint, cache_identity(caches))
        self.contract['schedule'] = {
            'mode': self.training_mode, 'task_order': list(TASKS),
            'passes_per_round': 1, 'max_steps_per_epoch_scope': 'per_task',
        }
        self.optimizer = build_stage_standard_moe_optimizer(model, s)
        self.steps_per_epoch = self._epoch_step_budget()
        total = self.steps_per_epoch * s.max_epochs
        warmup = int(total * s.warmup_ratio)
        def multiplier(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total - step) / max(total - warmup, 1))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, multiplier)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda' and s.mixed_precision == 'fp16')
        self.epoch = self.global_step = self.bad_epochs = 0
        self.best_score = -math.inf
        self.best_payload = None
        self.history = []
        self.validation_loaders = {t: make_loader(datasets['val'][t], t, caches, s) for t in TASKS}

    def _epoch_step_budget(self):
        s = self.settings
        self.task_step_budgets = {}
        for task in TASKS:
            size = len(self.datasets['train'][task])
            size = min(size, s.limit_train_samples or size)
            steps = math.ceil(size / s.batch_size)
            self.task_step_budgets[task] = min(steps, s.max_steps_per_epoch or steps)
        return sum(self.task_step_budgets.values())

    def _optimizer_step(self):
        """Apply accumulated gradients; advance the schedule only on a real update."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm,
                                      error_if_nonfinite=not self.scaler.is_enabled())
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= before:
            self.scheduler.step()
            self.global_step += 1
            return True
        return False

    def _log_tracking(self, values):
        if self.tracker is not None:
            self.tracker.log(values, step=self.global_step)

    def train_epoch(self, epoch):
        s = self.settings
        self.model.train()
        self.model.temperature = s.temperature(epoch)
        samples, sums = dict.fromkeys(TASKS, 0), dict.fromkeys(TASKS, 0.0)
        task_steps = dict.fromkeys(TASKS, 0)
        routing = RoutingStatistics()
        fusion = FusionStatistics()
        balance_sum = 0.0
        loss_statistics = LossStatistics()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.monotonic()
        for task in TASKS:
            loader = make_loader(self.datasets['train'][task], task, self.caches, s,
                                 training=True, epoch=epoch)
            print(json.dumps({'epoch': epoch, 'training_mode': self.training_mode,
                              'task': task, 'event': 'start', 'steps': self.task_step_budgets[task]}), flush=True)
            iterator = iter(loader)
            for step in range(self.task_step_budgets[task]):
                raw = next(iterator)
                self.optimizer.zero_grad(set_to_none=True)
                batch = move_batch(raw, self.device)
                with autocast_context(self.device, s.mixed_precision):
                    output = forward_batch(self.model, task, batch)
                    result = compute_stage_standard_moe_loss(
                        output, batch['labels'], batch['masks'],
                        task=task, settings=s, epoch=epoch)
                if not torch.isfinite(result.total):
                    raise FloatingPointError(f'Non-finite {task} loss at epoch {epoch}, step {step}')
                if result.valid_labels:
                    self.scaler.scale(result.total).backward()
                    task_steps[task] += int(self._optimizer_step())
                samples[task] += result.valid_labels
                sums[task] += float(result.supervised.detach()) * result.valid_labels
                balance_sum += float(result.balance.detach())
                routing.update(output.routing)
                fusion.update(output.fusion_beta, batch['peptide_mask'])
                loss_statistics.update(task, result)
                if (step + 1) % s.log_every_steps == 0:
                    print(json.dumps({'epoch': epoch, 'task': task, 'step': step + 1,
                                      'steps': self.task_step_budgets[task],
                                      'loss': sums[task] / max(samples[task], 1)}), flush=True)
                    self._log_tracking({'epoch': epoch, 'train_step': {
                        'loss': {task: sums[task] / max(samples[task], 1)},
                        'loss_details': loss_statistics.result(), 'fusion': fusion.result(),
                        'temperature': self.model.temperature,
                        'lr': {g['name']: g['lr'] for g in self.optimizer.param_groups}}})
            # Release workers/readers before starting the next task's pass.
            del iterator, loader
            print(json.dumps({'epoch': epoch, 'task': task, 'event': 'complete',
                              'sampled_rows': samples[task], 'optimizer_steps': task_steps[task]}), flush=True)
            self._log_tracking({'epoch': epoch, 'train_pass': {task: {
                'loss': sums[task] / max(samples[task], 1), 'samples': samples[task],
                'optimizer_steps': task_steps[task]}}})
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - start
        return {'training_mode': self.training_mode, 'task_order': list(TASKS),
                'loss': {t: sums[t] / max(samples[t], 1) for t in TASKS}, 'sampled_rows': samples,
                'task_optimizer_steps': task_steps, 'optimizer_steps': sum(task_steps.values()),
                'loss_details': loss_statistics.result(), 'fusion': fusion.result(),
                'balance_loss': balance_sum / self.steps_per_epoch, 'routing': routing.result(),
                'temperature': self.model.temperature, 'seconds': elapsed,
                'samples_per_second': sum(samples.values()) / max(elapsed, 1e-9),
                'cuda_peak_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'expert_parameters': sum(p.numel() for branch in self.model.moe.values() for p in branch.experts.parameters())}

    def _payload(self, evaluation, role):
        return {'architecture': ARCHITECTURE, 'format_version': 1, 'role': role,
                'training_mode': self.training_mode,
                'swanlab_run_id': getattr(self.tracker, 'run_id', None) or self.swanlab_run_id,
                'model': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                'model_contract': model_contract(self.config), 'config': self.config.to_dict(),
                'contract': self.contract, 'epoch': self.epoch, 'global_step': self.global_step,
                'temperature': self.model.temperature, 'thresholds': evaluation['thresholds'],
                'validation': evaluation, 'encoders': {e: c.manifest['model'] for e, c in self.caches.items()},
                'data_fingerprint': self.contract['data_fingerprint']}

    def resume(self, path):
        checkpoint = load_checkpoint(path, config=self.config, role='last')
        if checkpoint.get('training_mode') != self.training_mode:
            raise ValueError('Resume training mode mismatch: expected sequential')
        if checkpoint.get('contract') != self.contract:
            raise ValueError('Resume training/data/cache contract mismatch')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.temperature = checkpoint['temperature']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.epoch, self.global_step = checkpoint['epoch'], checkpoint['global_step']
        self.best_score, self.bad_epochs = checkpoint['best_score'], checkpoint['bad_epochs']
        self.best_payload, self.history = checkpoint['best_payload'], checkpoint['history']
        self.swanlab_run_id = checkpoint.get('swanlab_run_id')
        best_path = self.run_dir / 'best.pt'
        existing = load_checkpoint(best_path, config=self.config, role='best') if best_path.exists() else None
        same_best = (existing is not None and existing['epoch'] == self.best_payload['epoch']
                     and existing['thresholds'] == self.best_payload['thresholds']
                     and existing['contract'] == self.best_payload['contract']
                     and all(torch.equal(existing['model'][k], v) for k, v in self.best_payload['model'].items()))
        if not same_best:
            atomic_save(self.best_payload, best_path)
        restore_rng(checkpoint['rng'])

    def fit(self, *, resume=None, stop_after_epoch=None):
        if resume:
            self.resume(resume)
        if self.bad_epochs >= self.settings.patience:
            return self.history
        end = min(self.settings.max_epochs, stop_after_epoch or self.settings.max_epochs)
        for epoch in range(self.epoch + 1, end + 1):
            training = self.train_epoch(epoch)
            evaluation = evaluate_validation(self.model, self.validation_loaders, self.settings.mixed_precision)
            self.epoch = epoch
            if evaluation['score'] > self.best_score + self.settings.min_delta:
                self.best_score, self.bad_epochs = evaluation['score'], 0
                self.best_payload = self._payload(evaluation, 'best')
                atomic_save(self.best_payload, self.run_dir / 'best.pt')
            else:
                self.bad_epochs += 1
            self.history.append({'epoch': epoch, 'training': training, 'validation': evaluation})
            self._log_tracking({'epoch': epoch, 'train': training, 'val': evaluation,
                                'early_stopping': {'bad_epochs': self.bad_epochs, 'best_aupr': self.best_score}})
            last = self._payload(evaluation, 'last')
            last.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                        scaler=self.scaler.state_dict(), rng=capture_rng(), best_score=self.best_score,
                        bad_epochs=self.bad_epochs, best_payload=self.best_payload, history=self.history)
            atomic_save(last, self.run_dir / 'last.pt')
            write_json(self.run_dir / 'history.json', self.history)
            print(json.dumps({'epoch': epoch, 'validation_aupr': evaluation['score'],
                              'best_aupr': self.best_score, 'bad_epochs': self.bad_epochs}), flush=True)
            if self.bad_epochs >= self.settings.patience:
                break
        return self.history
