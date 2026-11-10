"""Minimal Stage13 tracker kept independent of the Stage1 tracker contract."""

from __future__ import annotations

from typing import Any, Mapping

from .stage13_config import Stage13ExperimentConfig


class Stage13Tracker:
    def __init__(self, config: Stage13ExperimentConfig, *, run_timestamp: str):
        self.mode = config.swanlab.mode
        self.run = None
        self.run_id = None
        self.text_factory = None
        if self.mode == "disabled":
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab is required for Stage13 logging; use --swanlab-mode disabled"
            ) from exc
        self.text_factory = swanlab.Text
        config.paths.swanlog_root.mkdir(parents=True, exist_ok=True)
        group = config.swanlab.group.format(fold=config.runtime.fold)
        self.run = swanlab.init(
            project=config.swanlab.project,
            workspace=config.swanlab.workspace,
            experiment_name=(
                f"{config.swanlab.experiment_prefix}-fold{config.runtime.fold}"
                f"-seed{config.runtime.seed}-{run_timestamp}"
            ),
            description=(
                "Stage-13 train-positive heterogeneous GraphSAGE fusion before "
                "the frozen-PLM Stage-1 adapters."
            ),
            job_type="train",
            group=group,
            tags=config.swanlab.tags,
            config=config.to_dict(),
            logdir=str(config.paths.swanlog_root),
            mode=self.mode,
            resume="never",
        )
        self.run_id = getattr(self.run, "id", None)

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        payload = {
            key: self.text_factory(value)
            if isinstance(value, str) and self.text_factory is not None
            else value
            for key, value in values.items()
        }
        if step is not None:
            payload.setdefault("global_step", step)
        try:
            self.run.log(payload, step=step)
        except TypeError:
            self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None

    def __enter__(self) -> "Stage13Tracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
