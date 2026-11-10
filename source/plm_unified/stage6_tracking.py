"""SwanLab tracking adapter with Stage-6-specific run metadata."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .stage6_config import Stage6ExperimentConfig


class Stage6Tracker:
    def __init__(
        self,
        config: Stage6ExperimentConfig,
        *,
        resume_id: Optional[str] = None,
        run_timestamp: Optional[str] = None,
    ):
        self.mode = config.swanlab.mode
        self.run = None
        self.run_id: Optional[str] = None
        self.text_factory = None
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        if self.mode == "disabled":
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is required or use --swanlab-mode disabled"
            ) from exc
        self.text_factory = swanlab.Text
        config.paths.swanlog_root.mkdir(parents=True, exist_ok=True)
        experiment_name = (
            f"{config.swanlab.experiment_prefix}-{config.stage6.condition}"
            f"-fold{config.training.fold}-seed{config.training.seed}"
            f"-{self.run_timestamp}"
        )
        init_kwargs: dict[str, Any] = {
            "project": config.swanlab.project,
            "workspace": config.swanlab.workspace,
            "experiment_name": experiment_name,
            "description": (
                "Stage-6 PH coverage and multi-task conflict ablation with "
                "frozen PLMs and fixed logical optimizer steps."
            ),
            "job_type": "train",
            "group": config.swanlab.group.format(fold=config.training.fold),
            "tags": [*config.swanlab.tags, config.stage6.condition],
            "config": config.to_dict(),
            "logdir": str(config.paths.swanlog_root),
            "mode": self.mode,
            "resume": "must" if resume_id else "never",
        }
        if resume_id:
            init_kwargs["id"] = resume_id
        self.run = swanlab.init(**init_kwargs)
        self.run_id = getattr(self.run, "id", None)

    def log(self, values: Mapping[str, Any], *, step: Optional[int] = None) -> None:
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

    def __enter__(self) -> "Stage6Tracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
