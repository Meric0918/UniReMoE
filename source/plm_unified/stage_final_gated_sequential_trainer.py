"""Each round trains one PH pass, then one PT pass, then one PHT pass."""
from __future__ import annotations

import json
import math
import time

import torch

from .stage_final_gated_data import make_loader, move_batch
from .stage_final_gated_evaluation import autocast_context, RoutingStatistics
from .stage_MOE_PT_evaluation import FusionStatistics
from .stage_final_gated_statistics import GateStatistics
from .stage_final_gated_loss import compute_stage_final_gated_loss, LossStatistics
from .stage_final_gated_model import TASKS, forward_batch
from .stage_final_gated_trainer import StageFinalGatedTrainer


class StageFinalGatedSequentialTrainer(StageFinalGatedTrainer):
    """One optimizer per run; validate and early-stop only after all three passes."""

    training_mode = 'sequential'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract['schedule'] = {
            'mode': self.training_mode, 'task_order': list(TASKS),
            'passes_per_round': 1, 'max_steps_per_epoch_scope': 'per_task',
        }

    def _epoch_step_budget(self):
        s = self.settings
        self.task_step_budgets = {}
        for task in TASKS:
            size = len(self.datasets['train'][task])
            size = min(size, s.limit_train_samples or size)
            steps = math.ceil(size / s.batch_size)
            self.task_step_budgets[task] = min(steps, s.max_steps_per_epoch or steps)
        return sum(self.task_step_budgets.values())

    def train_epoch(self, epoch):
        s = self.settings
        self.model.train()
        self.model.temperature = s.temperature(epoch)
        samples, sums = dict.fromkeys(TASKS, 0), dict.fromkeys(TASKS, 0.0)
        task_steps = dict.fromkeys(TASKS, 0)
        routing = RoutingStatistics()
        fusion = FusionStatistics()
        sample_gates = GateStatistics()
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
                    result = compute_stage_final_gated_loss(
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
                sample_gates.update(output, batch)
                loss_statistics.update(task, result)
                if (step + 1) % s.log_every_steps == 0:
                    print(json.dumps({'epoch': epoch, 'task': task, 'step': step + 1,
                                      'steps': self.task_step_budgets[task],
                                      'loss': sums[task] / max(samples[task], 1)}), flush=True)
                    self._log_tracking({'epoch': epoch, 'train_step': {
                        'loss': {task: sums[task] / max(samples[task], 1)},
                        'loss_details': loss_statistics.result(), 'fusion': fusion.result(), 'sample_gates': sample_gates.result(),
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
                'loss_details': loss_statistics.result(), 'fusion': fusion.result(), 'sample_gates': sample_gates.result(),
                'balance_loss': balance_sum / self.steps_per_epoch, 'routing': routing.result(),
                'temperature': self.model.temperature, 'seconds': elapsed,
                'samples_per_second': sum(samples.values()) / max(elapsed, 1e-9),
                'cuda_peak_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'expert_parameters': sum(p.numel() for p in self.model.shared_experts.parameters()) +
                    sum(p.numel() for branch in self.model.moe.values() for p in branch.experts.parameters())}
