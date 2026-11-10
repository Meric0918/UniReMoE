"""Stage1 training lifecycle specialised for the Stage15 architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .artifacts import create_artifact_timestamp, timestamped_filename, validate_artifact_timestamp
from .model import Task
from .stage15_config import Stage15ExperimentConfig
from .stage15_model import Stage15BindingModel, extract_stage15_task_state_dict
from .trainer import (
    Stage1Trainer,
    _atomic_torch_save,
    build_task_checkpoint_state,
    load_training_checkpoint,
)


STAGE15_TASK_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "stage15_best_phla.pt",
    "ptcr": "stage15_best_ptcr.pt",
}
STAGE15_CALIBRATION_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "stage15_best_phla_calibration.pt",
    "ptcr": "stage15_best_ptcr_calibration.pt",
}
STAGE15_ROBUST_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "stage15_best_phla_robust.pt",
    "ptcr": "stage15_best_ptcr_robust.pt",
}
STAGE15_BEST_ROUND_CHECKPOINT_FILENAME = "stage15_best_joint.pt"


def build_stage15_optimizer(
    model: Stage15BindingModel,
    config: Stage15ExperimentConfig,
) -> torch.optim.Optimizer:
    training = config.training
    specs = (
        ("shared_peptide_adapter", model.peptide_adapter, training.shared_peptide_adapter_lr, "shared"),
        ("shared_hyper_pmt", model.hypergraph_fusion, training.hyper_pmt_lr, "shared"),
        ("hla_adapter", model.hla_adapter, training.hla_adapter_lr, "phla"),
        ("tcr_adapter", model.tcr_adapter, training.tcr_adapter_lr, "ptcr"),
        ("phla_pooling", model.phla_pooling, training.phla_pooling_lr, "phla"),
        ("ptcr_pooling", model.ptcr_pooling, training.ptcr_pooling_lr, "ptcr"),
        ("phla_classifier", model.phla_classifier, training.phla_classifier_lr, "phla"),
        ("ptcr_classifier", model.ptcr_classifier, training.ptcr_classifier_lr, "ptcr"),
    )
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for module_name, module, learning_rate, scheduler_task in specs:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                raise ValueError(f"Duplicate Stage15 optimizer parameter in {module_name}.{name}")
            seen.add(id(parameter))
            is_no_decay = training.no_decay_bias_and_norm and (
                parameter.ndim <= 1
                or name.endswith("bias")
                or "norm" in name.lower()
            )
            (no_decay if is_no_decay else decay).append(parameter)
        for suffix, parameters, weight_decay in (
            ("decay", decay, training.weight_decay),
            ("no_decay", no_decay, 0.0),
        ):
            if parameters:
                groups.append({
                    "name": f"{module_name}/{suffix}",
                    "params": parameters,
                    "lr": float(learning_rate),
                    "weight_decay": weight_decay,
                    "scheduler_task": scheduler_task,
                })
    expected = {id(parameter) for parameter in model.parameters()}
    if seen != expected:
        raise ValueError("Stage15 optimizer does not cover the model exactly")
    return torch.optim.AdamW(groups, weight_decay=0.0)


def build_stage15_task_checkpoint_state(
    training_checkpoint: Mapping[str, Any],
    task: Task,
    *,
    source_checkpoint: str | Path,
    selection_role: str = "discrimination",
    selection_score: float | None = None,
    selection_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if training_checkpoint.get("checkpoint_type") not in {
        "stage15_training_model", "stage15_joint_model"
    }:
        raise ValueError("Expected a Stage15 training or joint checkpoint")
    state = build_task_checkpoint_state(
        training_checkpoint,
        task,
        source_checkpoint=source_checkpoint,
        selection_role=selection_role,
        selection_score=selection_score,
        selection_metrics=selection_metrics,
        state_extractor=extract_stage15_task_state_dict,
    )
    state.update({
        "checkpoint_type": "stage15_task_model",
        "architecture": "shared_hyper_pmt_pair",
        "model": extract_stage15_task_state_dict(training_checkpoint["model"], task),
    })
    return state


class Stage15Trainer(Stage1Trainer):
    display_name = "Stage 15"
    checkpoint_architecture = "shared_hyper_pmt_pair"

    def _build_optimizer(self, model, config):
        return build_stage15_optimizer(model, config)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_error_csv.output_dir = (
            self.config.paths.project_root / "csv" / "stage15" / self.run_timestamp
        )
        self.last_checkpoint_path = self.run_dir / timestamped_filename(
            "stage15_last.pt", self.run_timestamp
        )
        self.warmup_checkpoint_path = self.run_dir / timestamped_filename(
            "stage15_warmup_last.pt", self.run_timestamp
        )
        self.task_checkpoint_paths = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in STAGE15_TASK_CHECKPOINT_FILENAMES.items()
        }
        self.calibration_checkpoint_paths = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in STAGE15_CALIBRATION_CHECKPOINT_FILENAMES.items()
        }
        self.robust_checkpoint_paths = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in STAGE15_ROBUST_CHECKPOINT_FILENAMES.items()
        }
        self.best_round_checkpoint_path = (
            self.run_dir / "best_round" / timestamped_filename(
                STAGE15_BEST_ROUND_CHECKPOINT_FILENAME, self.run_timestamp
            )
        )

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state.update({
            "checkpoint_type": "stage15_training_model",
            "architecture": "shared_hyper_pmt_pair",
        })
        return state

    def _task_state(
        self,
        task: Task,
        *,
        role: str = "discrimination",
        score: float | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_stage15_task_checkpoint_state(
            self.checkpoint_state(),
            task,
            source_checkpoint=self.last_checkpoint_path,
            selection_role=role,
            selection_score=score,
            selection_metrics=metrics,
        )

    def save_best_task_checkpoint(self, task: Task) -> None:
        _atomic_torch_save(self._task_state(task), self.task_checkpoint_paths[task])

    def save_selection_checkpoint(
        self,
        task: Task,
        *,
        role: str,
        score: float,
        metrics: Mapping[str, Any],
    ) -> None:
        if role == "calibration":
            target = self.calibration_checkpoint_paths[task]
        elif role == "robust":
            target = self.robust_checkpoint_paths[task]
        else:
            raise ValueError(f"Unknown Stage15 selection role: {role}")
        _atomic_torch_save(
            self._task_state(task, role=role, score=score, metrics=metrics),
            target,
        )

    def save_best_round_checkpoint(self, round_id: int, round_score: float) -> None:
        state = self.checkpoint_state()
        state.update({
            "checkpoint_type": "stage15_joint_model",
            "best_round": int(round_id),
            "round_score": float(round_score),
            "model_scope": "joint_heads",
        })
        _atomic_torch_save(state, self.best_round_checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        if checkpoint.get("checkpoint_type") != "stage15_training_model":
            raise ValueError("Resume requires a stage15_training_model checkpoint")
        if checkpoint.get("architecture") != "shared_hyper_pmt_pair":
            raise ValueError("Stage15 checkpoint architecture mismatch")
        saved_fusion = checkpoint.get("config", {}).get("fusion")
        if saved_fusion != self.config.to_dict()["fusion"]:
            raise ValueError("Checkpoint fusion configuration does not match current config")
        super().load_checkpoint(path)


def export_stage15_task_checkpoints(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    artifact_timestamp: str | None = None,
) -> dict[Task, Path]:
    source = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    if checkpoint.get("checkpoint_type") not in {
        "stage15_training_model", "stage15_joint_model"
    }:
        raise ValueError("Expected a Stage15 training or joint checkpoint")
    destination = Path(output_dir).resolve() if output_dir else source.parent
    timestamp = validate_artifact_timestamp(
        artifact_timestamp or create_artifact_timestamp()
    )
    targets = {
        task: destination / timestamped_filename(filename, timestamp)
        for task, filename in STAGE15_TASK_CHECKPOINT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Stage15 task checkpoint already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for task, target in targets.items():
        _atomic_torch_save(
            build_stage15_task_checkpoint_state(
                checkpoint, task, source_checkpoint=source
            ),
            target,
        )
    return targets


__all__ = [
    "STAGE15_BEST_ROUND_CHECKPOINT_FILENAME",
    "STAGE15_TASK_CHECKPOINT_FILENAMES",
    "Stage15Trainer",
    "build_stage15_optimizer",
    "build_stage15_task_checkpoint_state",
    "export_stage15_task_checkpoints",
]
