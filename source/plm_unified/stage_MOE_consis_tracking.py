"""SwanLab scalar logging for joint and sequential MOE runs."""
from __future__ import annotations

import math
from numbers import Real

from .stage_MOE_consis_evaluation import write_json


def scalar_metrics(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(scalar_metrics(item, f'{prefix}/{key}' if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(scalar_metrics(item, f'{prefix}/{index}'))
    elif isinstance(value, Real) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


class StageMOEConsisTracker:
    def __init__(self, config, training_mode, run_dir, *, resume_id=None):
        self.run = None
        self.run_id = None
        self.url = None
        if config.swanlab.mode == 'disabled':
            return
        import swanlab
        settings = config.swanlab
        name = f'{settings.experiment_prefix}-{training_mode}-seed{config.stage_MOE_consis.seed}-{run_dir.name}'
        self.run = swanlab.init(
            project=settings.project, workspace=settings.workspace,
            name=name, mode=settings.mode, group=settings.group,
            tags=[*settings.tags, training_mode], job_type='train',
            description=f'Stage MOE Consis v1 {training_mode} PH/PT/PHT training with frozen PLM residue caches.',
            config={**config.to_dict(), 'training_mode': training_mode},
            log_dir=str(run_dir / 'swanlog'),
            **({'id': resume_id, 'resume': 'must'} if resume_id else {'resume': 'never'}))
        self.run_id = self.run.id
        self.url = self.run.url if settings.mode == 'online' else None
        info = {'mode': settings.mode, 'project': settings.project, 'training_mode': training_mode,
                'run_id': self.run_id, 'url': self.url}
        write_json(run_dir / 'swanlab.json', info)
        print(f'SwanLab [{training_mode}]: {self.url} (run_id={self.run_id})', flush=True)

    def log(self, values, *, step):
        if self.run is not None:
            self.run.log(scalar_metrics(values), step=step)

    def finish(self, state='success'):
        if self.run is not None:
            self.run.finish(state=state)
            self.run = None
