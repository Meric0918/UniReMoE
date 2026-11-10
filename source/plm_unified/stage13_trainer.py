"""Stage13 alternating trainer with query-edge-safe GraphSAGE sampling."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .artifacts import create_artifact_timestamp, timestamped_filename, validate_artifact_timestamp
from .fgm import ModuleFGM
from .metrics import BinaryMetrics, compute_binary_metrics, performance_avg, prefixed_metrics
from .stage13_config import Stage13ExperimentConfig
from .stage13_graph import GraphArtifact, GraphSampler, SampledGraph, file_sha256
from .stage13_model import Stage13BindingModel
from .stage13_tracking import Stage13Tracker


TASKS = ("phla", "ptcr")


def resolve_stage13_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_stage13_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(state: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)


def load_stage13_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class Stage13EpochResult:
    metrics: BinaryMetrics
    clean_loss: float
    adversarial_loss: float
    optimizer_steps: int
    samples: int
    graph_nodes: float

    def to_log_dict(self, prefix: str) -> dict[str, float | int]:
        values = prefixed_metrics(prefix, self.metrics)
        values.update(
            {
                f"{prefix}/clean_loss": self.clean_loss,
                f"{prefix}/adversarial_loss": self.adversarial_loss,
                f"{prefix}/optimizer_steps": self.optimizer_steps,
                f"{prefix}/samples": self.samples,
                f"{prefix}/sampled_graph_nodes": self.graph_nodes,
            }
        )
        return values


class Stage13Trainer:
    def __init__(
        self,
        model: Stage13BindingModel,
        config: Stage13ExperimentConfig,
        graph: GraphArtifact,
        train_loaders: Mapping[str, Any],
        val_loaders: Mapping[str, Mapping[str, Any]],
        tracker: Stage13Tracker,
        *,
        run_timestamp: str | None = None,
    ):
        self.model = model
        self.config = config
        self.graph = graph
        self.train_loaders = {task: train_loaders[task] for task in TASKS}
        self.val_loaders = {
            task: dict(val_loaders.get(task, {})) for task in TASKS
        }
        if any(not self.val_loaders[task] for task in TASKS):
            raise ValueError("Stage13 requires at least one validation split per task")
        self.tracker = tracker
        self.device = resolve_stage13_device(config.runtime.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.sampler = GraphSampler(
            graph, config.graph.fanouts, config.graph.sampling_seed
        )
        self.optimizer = self._build_optimizer()
        steps_per_round = sum(len(loader) for loader in self.train_loaders.values())
        total_steps = steps_per_round * (
            config.training.warmup_epochs_per_task + config.training.max_rounds
        )
        self.scheduler = self._build_scheduler(total_steps)
        scaler_enabled = self.device.type == "cuda" and config.training.mixed_precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.fgm = ModuleFGM(model)
        self.stage = "warmup"
        self.warmup_completed = {task: 0 for task in TASKS}
        self.current_round = 0
        self.best_task_scores = {task: -float("inf") for task in TASKS}
        self.best_val_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
        self.no_improve_rounds = {task: 0 for task in TASKS}
        self.stopped_tasks = {task: False for task in TASKS}
        self.best_round_id = 0
        self.best_round_score = -float("inf")
        self.round_no_improve_rounds = 0
        self.round_stopped = False
        self.global_step = 0
        self.run_dir = config.run_dir()
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.last_checkpoint_path = self.run_dir / timestamped_filename(
            "last.pt", self.run_timestamp
        )
        self.warmup_checkpoint_path = self.run_dir / timestamped_filename(
            "warmup_last.pt", self.run_timestamp
        )
        self.task_checkpoint_paths = {
            task: self.run_dir / timestamped_filename(f"best_{task}.pt", self.run_timestamp)
            for task in TASKS
        }
        self.best_round_checkpoint_path = (
            self.run_dir / "best_round" / timestamped_filename("best_joint.pt", self.run_timestamp)
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        training = self.config.training
        groups = [
            {"name": "graph_sage", "params": self.model.graph_sage.parameters(), "lr": training.graph_lr},
            {"name": "graph_fusion", "params": self.model.graph_token_projection.parameters(), "lr": training.fusion_lr},
            {
                "name": "adapters",
                "params": list(self.model.peptide_adapter.parameters())
                + list(self.model.hla_adapter.parameters())
                + list(self.model.tcr_adapter.parameters()),
                "lr": training.adapter_lr,
            },
            {
                "name": "cross_attention",
                "params": list(self.model.phla_cross_attention.parameters())
                + list(self.model.ptcr_cross_attention.parameters()),
                "lr": training.cross_attention_lr,
            },
            {
                "name": "pooling",
                "params": list(self.model.phla_pooling.parameters())
                + list(self.model.ptcr_pooling.parameters()),
                "lr": training.pooling_lr,
            },
            {
                "name": "classifiers",
                "params": list(self.model.phla_classifier.parameters())
                + list(self.model.ptcr_classifier.parameters()),
                "lr": training.classifier_lr,
            },
        ]
        return torch.optim.AdamW(groups, weight_decay=training.weight_decay)

    def _build_scheduler(self, total_steps: int):
        warmup_steps = max(1, round(total_steps * self.config.training.scheduler_warmup_ratio))

        def multiplier(step: int) -> float:
            if step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, multiplier)

    def _autocast(self):
        precision = self.config.training.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _pooled_numpy(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        values = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
        valid = mask.detach().cpu().numpy().astype(np.float32, copy=False)[..., None]
        return (values * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)

    @staticmethod
    def _pad_features(values: np.ndarray, width: int = 960) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] > width:
            raise ValueError("Unexpected Stage13 pooled feature shape")
        output = np.zeros((values.shape[0], width), dtype=np.float32)
        output[:, : values.shape[1]] = values
        return output

    def _make_graph(
        self,
        task: str,
        cpu_batch: Mapping[str, torch.Tensor],
        batch_index: int,
        *,
        deterministic: bool,
    ) -> SampledGraph:
        peptide_ids = cpu_batch["peptide_node_ids"].tolist()
        receptor_ids = cpu_batch["receptor_node_ids"].tolist()
        batch_size = len(peptide_ids)
        root_ids = [int(value) for value in peptide_ids + receptor_ids]
        root_features = np.concatenate(
            (
                self._pad_features(
                    self._pooled_numpy(cpu_batch["peptide_hidden"], cpu_batch["peptide_mask"])
                ),
                self._pad_features(
                    self._pooled_numpy(cpu_batch["receptor_hidden"], cpu_batch["receptor_mask"])
                ),
            ),
            axis=0,
        )
        receptor_code = 1 if task == "phla" else 2
        root_types = [0] * batch_size + [receptor_code] * batch_size
        excluded = [
            (int(left), int(right))
            for left, right in zip(peptide_ids, receptor_ids)
            if int(left) >= 0 and int(right) >= 0
        ]
        seed = self.config.graph.sampling_seed + (
            1000003 * batch_index if deterministic else self.global_step * 1000003 + batch_index
        )
        return self.sampler.sample(
            root_ids,
            root_features,
            root_types=root_types,
            excluded_edges=excluded,
            seed=seed,
        ).to(self.device)

    def _move_hidden(self, value: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32
        if self.device.type == "cuda":
            if self.config.training.mixed_precision == "fp16":
                dtype = torch.float16
            elif self.config.training.mixed_precision == "bf16":
                dtype = torch.bfloat16
        return value.to(self.device, dtype=dtype, non_blocking=self.device.type == "cuda")

    def _move_batch(self, cpu_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "peptide_hidden": self._move_hidden(cpu_batch["peptide_hidden"]),
            "peptide_mask": cpu_batch["peptide_mask"].to(self.device, dtype=torch.bool),
            "receptor_hidden": self._move_hidden(cpu_batch["receptor_hidden"]),
            "receptor_mask": cpu_batch["receptor_mask"].to(self.device, dtype=torch.bool),
            "labels": cpu_batch["labels"].to(self.device, dtype=torch.long),
        }

    def _forward(self, task: str, batch: Mapping[str, torch.Tensor], graph: SampledGraph):
        return self.model(
            task,
            batch["peptide_hidden"],
            batch["peptide_mask"],
            batch["receptor_hidden"],
            batch["receptor_mask"],
            graph,
        )

    def train_epoch(self, task: str, loader: Any, *, phase_name: str, use_fgm: bool) -> Stage13EpochResult:
        self.model.set_trainable_task(task)
        self.model.train()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        clean_loss_sum = 0.0
        adversarial_loss_sum = 0.0
        samples = 0
        graph_nodes: list[int] = []
        for batch_index, cpu_batch in enumerate(loader, start=1):
            graph = self._make_graph(task, cpu_batch, batch_index, deterministic=False)
            batch = self._move_batch(cpu_batch)
            batch_samples = int(batch["labels"].shape[0])
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                clean_output = self._forward(task, batch, graph)
                clean_loss = self.criterion(clean_output.logits, batch["labels"])
            self.scaler.scale(clean_loss).backward()
            adversarial_loss_value = 0.0
            if use_fgm:
                self.fgm.attack(self.model.fgm_modules(task), epsilon=self.config.training.fgm_epsilon)
                try:
                    with self._autocast():
                        adversarial_output = self._forward(task, batch, graph)
                        adversarial_loss = self.criterion(adversarial_output.logits, batch["labels"])
                    self.scaler.scale(adversarial_loss).backward()
                    adversarial_loss_value = float(adversarial_loss.detach().cpu())
                finally:
                    self.fgm.restore()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, self.config.training.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            clean_value = float(clean_loss.detach().cpu())
            clean_loss_sum += clean_value * batch_samples
            adversarial_loss_sum += adversarial_loss_value * batch_samples
            samples += batch_samples
            graph_nodes.append(sum(len(values) for values in graph.ids_by_depth))
            probabilities.append(torch.softmax(clean_output.logits.detach(), dim=1)[:, 1].float().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
            if batch_index % self.config.training.log_every_steps == 0 or batch_index == len(loader):
                self.tracker.log(
                    {
                        f"{phase_name}/{task}/batch_clean_loss": clean_value,
                        f"{phase_name}/{task}/batch_adversarial_loss": adversarial_loss_value,
                        "graph/query_edge_exclusion": 1,
                        "graph/sampled_nodes": graph_nodes[-1],
                        **{f"learning_rate/{group.get('name', index)}": group["lr"] for index, group in enumerate(self.optimizer.param_groups)},
                    },
                    step=self.global_step,
                )
        average_clean = clean_loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=average_clean,
            threshold=self.config.training.threshold,
            strict_threshold=True,
        )
        return Stage13EpochResult(
            metrics=metrics,
            clean_loss=average_clean,
            adversarial_loss=adversarial_loss_sum / samples,
            optimizer_steps=len(loader),
            samples=samples,
            graph_nodes=float(np.mean(graph_nodes)) if graph_nodes else 0.0,
        )

    @torch.inference_mode()
    def evaluate(self, task: str, loader: Any) -> BinaryMetrics:
        self.model.eval()
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        for batch_index, cpu_batch in enumerate(loader, start=1):
            graph = self._make_graph(task, cpu_batch, batch_index, deterministic=True)
            batch = self._move_batch(cpu_batch)
            with self._autocast():
                output = self._forward(task, batch, graph)
                loss = self.criterion(output.logits, batch["labels"])
            count = int(batch["labels"].shape[0])
            samples += count
            loss_sum += float(loss.detach().cpu()) * count
            probabilities.append(torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
        return compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=loss_sum / samples,
            threshold=self.config.training.threshold,
            strict_threshold=True,
        )

    def _task_score(self, task: str, metrics: Mapping[str, BinaryMetrics]) -> float:
        return float(np.mean([performance_avg(value) for value in metrics.values()]))

    def fit(self) -> dict[str, Any]:
        if self.stage == "complete":
            return self.summary()
        if self.stage == "warmup":
            for task in TASKS:
                for epoch in range(self.warmup_completed[task] + 1, self.config.training.warmup_epochs_per_task + 1):
                    result = self.train_epoch(task, self.train_loaders[task], phase_name="warmup", use_fgm=False)
                    self.warmup_completed[task] = epoch
                    self.tracker.log({"stage": "warmup", "task": task, "warmup_epoch": epoch, **result.to_log_dict(f"warmup/{task}/train")}, step=self.global_step)
                    self.save_checkpoint(self.last_checkpoint_path)
            self.stage = "stage13b"
            self.save_checkpoint(self.warmup_checkpoint_path)
            self.save_checkpoint(self.last_checkpoint_path)
        for round_id in range(self.current_round + 1, self.config.training.max_rounds + 1):
            active = [task for task in TASKS if not self.stopped_tasks[task]]
            if not active:
                break
            train_results = {
                task: self.train_epoch(task, self.train_loaders[task], phase_name=f"round_{round_id}", use_fgm=True)
                for task in active
            }
            val_metrics = {
                task: {split: self.evaluate(task, loader) for split, loader in self.val_loaders[task].items()}
                for task in active
            }
            scores: dict[str, float] = {}
            improved: dict[str, bool] = {}
            newly_stopped: list[str] = []
            for task in active:
                scores[task] = self._task_score(task, val_metrics[task])
                improved[task] = scores[task] > self.best_task_scores[task] + self.config.training.early_stopping_min_delta
                if improved[task]:
                    self.best_task_scores[task] = scores[task]
                    self.best_val_metrics[task] = {split: value.to_dict() for split, value in val_metrics[task].items()}
                    self.no_improve_rounds[task] = 0
                else:
                    self.no_improve_rounds[task] += 1
                    if self.no_improve_rounds[task] >= self.config.training.early_stopping_patience:
                        self.stopped_tasks[task] = True
                        newly_stopped.append(task)
            self.current_round = round_id
            round_score = float(np.mean([self.best_task_scores[task] for task in TASKS]))
            round_improved = round_score > self.best_round_score + self.config.training.early_stopping_min_delta
            if round_improved:
                self.best_round_id = round_id
                self.best_round_score = round_score
                self.round_no_improve_rounds = 0
            else:
                self.round_no_improve_rounds += 1
                if self.round_no_improve_rounds >= self.config.training.early_stopping_patience:
                    self.round_stopped = True
            logs: dict[str, Any] = {
                "stage": "stage13b",
                "round": round_id,
                "early_stopping/round_score": round_score,
                "early_stopping/round_best_score": self.best_round_score,
                "early_stopping/round_no_improve_rounds": self.round_no_improve_rounds,
                "early_stopping/round_stopped": int(self.round_stopped),
            }
            for task in active:
                logs.update(train_results[task].to_log_dict(f"round/{task}/train"))
                for split, metrics in val_metrics[task].items():
                    logs.update(prefixed_metrics(f"round/{task}/val/{split}", metrics))
                logs[f"early_stopping/{task}/score"] = scores[task]
                logs[f"early_stopping/{task}/best_score"] = self.best_task_scores[task]
                logs[f"early_stopping/{task}/no_improve_rounds"] = self.no_improve_rounds[task]
            self.tracker.log(logs, step=self.global_step)
            self.save_checkpoint(self.last_checkpoint_path)
            for task in active:
                if improved[task]:
                    self.save_best_task_checkpoint(task)
            if round_improved:
                self.save_best_round_checkpoint()
            if self.round_stopped or all(self.stopped_tasks.values()):
                break
        self.stage = "complete"
        self.save_checkpoint(self.last_checkpoint_path)
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "checkpoint_type": "stage13_training",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "warmup_completed": dict(self.warmup_completed),
            "current_round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_val_metrics": self.best_val_metrics,
            "no_improve_rounds": dict(self.no_improve_rounds),
            "stopped_tasks": dict(self.stopped_tasks),
            "best_round": self.best_round_id,
            "best_round_score": self.best_round_score,
            "round_no_improve_rounds": self.round_no_improve_rounds,
            "round_stopped": self.round_stopped,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "graph_manifest": self.graph.manifest,
            "graph_manifest_sha256": file_sha256(self.graph.root / "manifest.json"),
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": self.tracker.run_id,
            "dataloader_rng_state": {
                task: loader.generator.get_state()
                for task, loader in self.train_loaders.items()
                if getattr(loader, "generator", None) is not None
            },
            "rng_state": _capture_rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        _atomic_save(self.checkpoint_state(), path)

    def save_best_task_checkpoint(self, task: str) -> None:
        state = self.checkpoint_state()
        state.update({"checkpoint_type": "stage13_task_model", "task": task})
        _atomic_save(state, self.task_checkpoint_paths[task])

    def save_best_round_checkpoint(self) -> None:
        state = self.checkpoint_state()
        state.update({"checkpoint_type": "stage13_joint_model", "model_scope": "joint_heads"})
        _atomic_save(state, self.best_round_checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_stage13_checkpoint(path, map_location="cpu")
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported Stage13 checkpoint format")
        saved_config = checkpoint.get("config", {})
        if saved_config.get("model") != self.config.to_dict()["model"]:
            raise ValueError("Stage13 checkpoint model configuration mismatch")
        if saved_config.get("graph") != self.config.to_dict()["graph"]:
            raise ValueError("Stage13 checkpoint graph configuration mismatch")
        if checkpoint.get("graph_manifest_sha256") != file_sha256(self.graph.root / "manifest.json"):
            raise ValueError("Stage13 graph manifest mismatch")
        saved_runtime = saved_config.get("runtime", {})
        for key in ("fold", "seed"):
            if saved_runtime.get(key) != getattr(self.config.runtime, key):
                raise ValueError(f"Stage13 checkpoint {key} does not match current config")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.stage = checkpoint["stage"]
        self.warmup_completed = {task: int(checkpoint["warmup_completed"][task]) for task in TASKS}
        self.current_round = int(checkpoint["current_round"])
        self.best_task_scores = {task: float(checkpoint["best_task_scores"][task]) for task in TASKS}
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        self.no_improve_rounds = {task: int(checkpoint["no_improve_rounds"][task]) for task in TASKS}
        self.stopped_tasks = {task: bool(checkpoint["stopped_tasks"][task]) for task in TASKS}
        self.best_round_id = int(checkpoint.get("best_round", 0))
        self.best_round_score = float(checkpoint.get("best_round_score", -float("inf")))
        self.round_no_improve_rounds = int(checkpoint.get("round_no_improve_rounds", 0))
        self.round_stopped = bool(checkpoint.get("round_stopped", False))
        self.global_step = int(checkpoint["global_step"])
        for task, state in checkpoint.get("dataloader_rng_state", {}).items():
            generator = getattr(self.train_loaders.get(task), "generator", None)
            if generator is not None:
                generator.set_state(state)
        _restore_rng_state(checkpoint.get("rng_state"))

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fold": self.config.runtime.fold,
            "seed": self.config.runtime.seed,
            "round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_val_metrics": self.best_val_metrics,
            "stopped_tasks": dict(self.stopped_tasks),
            "best_round": self.best_round_id,
            "best_round_score": self.best_round_score,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "run_dir": str(self.run_dir),
            "last_checkpoint": str(self.last_checkpoint_path),
            "warmup_checkpoint": str(self.warmup_checkpoint_path),
            "task_checkpoints": {task: str(path) for task, path in self.task_checkpoint_paths.items()},
            "best_round_checkpoint": str(self.best_round_checkpoint_path),
        }
