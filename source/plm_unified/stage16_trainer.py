"""Controlled PH -> PT -> PHT training with exact window-boundary restart."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from itertools import islice
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .stage8_loss import compute_stage8_loss
from .trainer import build_scheduler
from .stage16_checkpoint import (
    atomic_torch_save,
    checkpoint_sha256,
    digest,
    inference_payload,
    save_training_checkpoint,
    write_json,
    write_jsonl,
)
from .stage16_data import pair_loader, joint_loader, stratified_indices
from .stage16_evaluation import autocast, evaluate_all, evaluate_pairs
from .stage16_selection import ShadowSelectors
from .stage18_loss import group_weights, sample_weights, window_pt_loss
from .artifacts import create_artifact_timestamp


def build_optimizer(model, config):
    policy = config.policy
    t = config.training
    # Union over all passes, not the parameters active only during joint.
    active = set()
    for task in ("phla", "ptcr", "joint"):
        model.set_trainable_pass(task, policy)
        active.update(id(p) for p in model.parameters() if p.requires_grad)
    groups = []
    seen = set()

    def add(name, parameters, lr):
        params = [p for p in parameters if id(p) in active]
        if not params:
            return
        if any(id(p) in seen for p in params):
            raise ValueError("Duplicate optimizer parameters")
        seen.update(id(p) for p in params)
        groups.append({"name": name, "params": params, "lr": lr})

    binding = model.binding_model
    for name, module in binding.named_children():
        lr = (
            t.adapter_lr
            if name.endswith("adapter")
            else (
                t.triplet_module_lr
                if name.startswith("triplet_")
                else t.pretrained_attention_lr
            )
        )
        add(name, module.parameters(), lr)
    for entity in ("peptide", "hla", "tcr"):
        encoder = getattr(model, entity + "_encoder")
        blocks = encoder.transformer_blocks()
        count = getattr(t, entity + "_unfrozen_layers")
        for index in range(max(0, len(blocks) - count), len(blocks)):
            add(
                f"{entity}_plm_{index}",
                blocks[index].parameters(),
                t.plm_top_lr if index == len(blocks) - 1 else t.plm_lower_lr,
            )
    if seen != active:
        raise ValueError(
            "Optimizer does not cover the complete allowed parameter union"
        )
    model.set_trainable_pass("joint", policy)
    return torch.optim.AdamW(groups, weight_decay=t.weight_decay)


def gradient_report(losses, parameters):
    parameters = [p for p in parameters if p.requires_grad]
    result = {}
    vectors = {}
    for name, loss in losses.items():
        grads = (
            torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            if parameters and loss.requires_grad
            else [None] * len(parameters)
        )
        vectors[name] = grads
        result[name + "_norm"] = math.sqrt(
            sum(
                float(g.detach().float().square().sum()) for g in grads if g is not None
            )
        )
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            dot = sum(
                float((a.detach().float() * b.detach().float()).sum())
                for a, b in zip(vectors[left], vectors[right])
                if a is not None and b is not None
            )
            norm = result[left + "_norm"] * result[right + "_norm"]
            result[left + "_vs_" + right] = dot / norm if norm else None
    return result


class Stage16Trainer:
    def __init__(
        self,
        model,
        config,
        data,
        *,
        reference=None,
        teacher_cache=None,
        output_dir=None,
        tracker=None,
    ):
        config.validate()
        self.model, self.config, self.data = model, config, data
        self.device = next(model.parameters()).device
        self.manifest = data["manifest"]
        self.reference = reference
        self.teacher_cache = teacher_cache
        self.teacher_identity = (
            {"identity": teacher_cache.identity, "sha256": teacher_cache.sha256}
            if teacher_cache
            else None
        )
        if config.policy.distillation_weight and teacher_cache is None:
            raise ValueError(
                "KD requires an independent original-encoder teacher cache"
            )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.tracker = tracker
        self.swanlab_run_id = getattr(tracker, "run_id", None)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weight_map, self.group_stats = (
            group_weights(data["pairs"]["ptcr"].rows)
            if config.policy.group_weighting
            else ({}, [])
        )
        self.optimizer = build_optimizer(model, config)
        planned = (
            config.training.scheduler_total_steps
            or self.steps_per_round() * config.training.scheduler_rounds
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=planned,
            warmup_ratio=config.training.scheduler_warmup_ratio,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda"
            and config.training.mixed_precision == "fp16",
        )
        self.selectors = ShadowSelectors(config.selection)
        self.completed_round = -1
        self.global_step = 0
        self.cursor = {"round": 1, "pass_index": 0, "next_batch": 0}
        self.last_metrics = None
        self.round_history = []
        self.decision_history = []
        self.pass_history = []
        self.monitor_history = []
        self.step_history = []
        self.current_stats = {}
        self.seen_pt_indices = set()
        self.pt_exposures = 0
        self.pt_indices = {
            row: i for i, row in enumerate(data["pairs"]["ptcr"].row_ids)
        }
        self.monitor_data = {
            s: d.subset(
                stratified_indices(
                    d.rows, config.training.monitor_per_split, config.runtime.seed
                )
            )
            for s, d in data["pt_val"].items()
        }
        self.monitor_previous = {}
        self.initial_parameters = (
            {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            if config.training.gradient_diagnostics
            else {}
        )
        self._interrupt = False

    def cap(self, task):
        if not self.config.policy.binary_replay:
            return 0
        ratio = (
            self.config.policy.pt_replay_ratio
            if task == "ptcr"
            else self.config.policy.ph_replay_ratio
        )
        return min(
            len(self.data["pairs"][task]),
            math.floor(len(self.data["joint_train"]) * ratio),
        )

    def batches(self, task):
        if task == "joint":
            counts = Counter(r.label_type for r in self.data["joint_train"])
            return math.ceil(
                max(counts.values()) / (self.config.training.micro_batch_size // 4)
            )
        return math.ceil(self.cap(task) / self.config.training.micro_batch_size)

    def steps_per_round(self):
        return sum(
            math.ceil(
                self.batches(task) / self.config.training.gradient_accumulation_steps
            )
            for task in ("phla", "ptcr", "joint")
        )

    def state_dict(self):
        return {
            "completed_round": self.completed_round,
            "global_step": self.global_step,
            "cursor": self.cursor,
            "current_stats": self.current_stats,
            "round_history": self.round_history,
            "decision_history": self.decision_history,
            "pass_history": self.pass_history,
            "monitor_history": self.monitor_history,
            "step_history": self.step_history,
            "seen_pt_indices": sorted(self.seen_pt_indices),
            "pt_exposures": self.pt_exposures,
            "monitor_previous": self.monitor_previous,
            "swanlab_run_id": self.swanlab_run_id,
        }

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.seen_pt_indices = set(self.seen_pt_indices)
        self._interrupt = False

    @staticmethod
    def _numeric_metrics(value, prefix=""):
        output = {}
        if isinstance(value, dict):
            for key, item in value.items():
                output.update(Stage16Trainer._numeric_metrics(item, f"{prefix}/{key}"))
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            output[prefix.lstrip("/")] = float(value)
        return output

    def _swanlab_log(self, values, *, step=None):
        if self.tracker is not None:
            payload = self._numeric_metrics(values)
            if payload:
                self.tracker.log(payload, step=step if step is not None else self.global_step)

    def _shared_parameters(self):
        return [
            p
            for n, p in self.model.named_parameters()
            if any(
                prefix in n
                for prefix in (
                    "peptide_encoder.",
                    "tcr_encoder.",
                    "peptide_adapter.",
                    "tcr_adapter.",
                )
            )
        ]

    def _monitor(self, label, round_index):
        if (
            not self.config.training.gradient_diagnostics
            or round_index not in self.config.training.monitor_rounds
        ):
            return
        key = f"{round_index}:{label}"
        if any(item["key"] == key for item in self.monitor_history):
            return
        metrics = {}
        differences = {}
        for split, dataset in self.monitor_data.items():
            metrics[split], pred = evaluate_pairs(self.model, dataset, self.config)
            previous = self.monitor_previous.get(split)
            if previous is not None:
                delta = pred["probabilities"] - previous
                differences[split] = {
                    "mean": float(delta.mean()),
                    "abs_p50": float(np.quantile(abs(delta), 0.5)),
                    "abs_p95": float(np.quantile(abs(delta), 0.95)),
                    "abs_max": float(abs(delta).max()),
                }
            self.monitor_previous[split] = pred["probabilities"]
        sums = {}
        for name, param in self.model.named_parameters():
            base = self.initial_parameters[name].float()
            now = param.detach().cpu().float()
            prefix = (
                ".".join(name.split(".")[:2])
                if name.startswith("binding_model.")
                else name.split(".")[0]
            )
            delta, norm = sums.get(prefix, (0.0, 0.0))
            sums[prefix] = (
                delta + float((now - base).square().sum()),
                norm + float(base.square().sum()),
            )
        self.monitor_history.append(
            {
                "key": key,
                "round": round_index,
                "pass": label,
                "metrics": metrics,
                "probability_change": differences,
                "relative_to_warmup_l2": {
                    k: math.sqrt(d / max(n, 1e-30)) for k, (d, n) in sums.items()
                },
            }
        )

    def _record_evaluation(self):
        metrics, predictions = evaluate_all(self.model, self.data, self.config)
        if self.reference is None:
            raise ValueError("Independent PT reference is missing")
        if self.completed_round < 0:
            self.reference = {**self.reference, "pht_val": metrics["pht_val"]}
            self.completed_round = 0
        decision = self.selectors.consider(
            metrics, self.reference, self.completed_round
        )
        self.last_metrics = metrics
        self.round_history.append(
            {
                "round": self.completed_round,
                "global_step": self.global_step,
                "metrics": metrics,
            }
        )
        self.decision_history.append(decision)
        self._swanlab_log({"validation": metrics, "selection": decision}, step=self.global_step)
        if self.output_dir:
            if self.completed_round == 0:
                atomic_torch_save(
                    inference_payload(self, metrics, role="initial"),
                    self.output_dir / "initial.pt",
                )
            if decision["improved"]:
                name = f"selected_round_{self.completed_round:03d}.pt"
                path = self.output_dir / name
                atomic_torch_save(
                    inference_payload(
                        self, metrics, role="best", selectors=decision["improved"]
                    ),
                    path,
                )
                checksum = checkpoint_sha256(path)
                for selector in decision["improved"]:
                    self.selectors.states[selector].artifact = name
                    self.selectors.states[selector].sha256 = checksum
                    write_json(
                        self.output_dir / f"best_{selector}.json",
                        {
                            "path": name,
                            "sha256": checksum,
                            "selector": selector,
                            "round": self.completed_round,
                        },
                    )
            # Per-round predictions enable same-trajectory selector and clustered-bootstrap analysis.
            atomic_torch_save(
                predictions,
                self.output_dir
                / f"validation_predictions_round_{self.completed_round:03d}.pt",
            )
            self.flush_logs()
            save_training_checkpoint(self, self.output_dir / "last.pt")
        print(
            f"Stage{self.config.stage} {self.config.condition} round={self.completed_round} step={self.global_step} guard={decision['gate']['pass']} improved={decision['improved']}",
            flush=True,
        )
        return decision

    def flush_logs(self):
        if not self.output_dir:
            return
        for name, state in self.selectors.states.items():
            index_path = self.output_dir / f"best_{name}.json"
            if state.artifact is not None:
                write_json(
                    index_path,
                    {
                        "path": state.artifact,
                        "sha256": state.sha256,
                        "selector": name,
                        "round": state.round,
                    },
                )
            elif index_path.exists():
                # A crash may publish an index just before last.pt commits. Keep
                # the orphan for inspection, but never expose it as a resumed best.
                index_path.rename(
                    self.output_dir
                    / f"uncommitted_{name}_{create_artifact_timestamp()}.json"
                )
        for name, rows in (
            ("round_metrics", self.round_history),
            ("selection_decisions", self.decision_history),
            ("pass_metrics", self.pass_history),
            ("monitor_metrics", self.monitor_history),
            ("training_steps", self.step_history),
        ):
            write_jsonl(self.output_dir / f"{name}.jsonl", rows)
        write_json(self.output_dir / "reference_metrics.json", self.reference)
        write_json(
            self.output_dir / "group_weights.json",
            {"groups": self.group_stats, "sha256": digest(self.group_stats)},
        )
        write_json(
            self.output_dir / "teacher_cache_manifest.json", self.teacher_identity
        )
        exposures = []
        cumulative = 0
        for record in self.pass_history:
            if record["task"] == "ptcr":
                cumulative += record["samples"]
                exposures.append(
                    {
                        "round": record["round"],
                        "rows": record["samples"],
                        "cumulative_rows": cumulative,
                        "unique_rows": record.get("unique_rows"),
                        "unique_peptides": record.get("unique_peptides"),
                        "unique_tcrs": record.get("unique_tcrs"),
                    }
                )
        write_jsonl(self.output_dir / "exposure_stats.jsonl", exposures)
        status = (
            "no_feasible_checkpoint"
            if self.selectors.states["guarded"].score is None
            else "guarded_candidate_available"
        )
        lines = [
            f"# Stage {self.config.stage}: {self.config.condition}",
            "",
            f"Status: {status}",
            f"Completed rounds: {self.completed_round}",
            f"Optimizer steps: {self.global_step}",
            f"Debug subset: {self.config.debug}",
            "",
            "| Selector | Round | Score | Artifact |",
            "|---|---:|---:|---|",
        ]
        for name, state in self.selectors.states.items():
            lines.append(
                f"| {name} | {state.round} | {state.score} | {state.artifact} |"
            )
        (self.output_dir / "training_summary.md").write_text("\n".join(lines) + "\n")

    def _window(self, task, batches, stats):
        t = self.config.training
        policy = self.config.policy
        size = sum(
            len(batch["pht_labels"] if task == "joint" else batch["labels"])
            for batch in batches
        )
        weight_sum = (
            sum(
                float(
                    sample_weights(
                        batch["peptide_sequences"],
                        batch["labels"],
                        self.weight_map,
                        "cpu",
                    ).sum()
                )
                for batch in batches
            )
            if task == "ptcr"
            else size
        )
        self.optimizer.zero_grad(set_to_none=True)
        diag = (
            t.gradient_diagnostics
            and self.cursor["round"] in t.monitor_rounds
            and self.cursor["next_batch"] == 0
        )
        for batch in batches:
            with autocast(self.config, self.device):
                if task == "joint":
                    out = self.model.forward_all(
                        batch["peptide_sequences"],
                        batch["hla_sequences"],
                        batch["tcr_sequences"],
                    )
                    labels = {
                        k: batch[k].to(self.device)
                        for k in ("ph_labels", "pt_labels", "pht_labels")
                    }
                    loss = compute_stage8_loss(
                        out.phla_logits.float(),
                        out.ptcr_logits.float(),
                        out.pht_logits.float(),
                        labels["ph_labels"],
                        labels["pt_labels"],
                        labels["pht_labels"],
                        batch["label_types"],
                        condition=t.constraint_condition,
                        epoch=self.cursor["round"],
                        phase="stage8b",
                        round_index=self.cursor["round"],
                        margin=t.constraint_margin,
                        hierarchy_lambda=t.hierarchy_lambda,
                        pt_implication_lambda=t.pt_implication_lambda,
                        ramp_epochs=t.constraint_ramp_epochs,
                        pht_weight=t.pht_loss_weight,
                        phla_weight=t.phla_loss_weight,
                        ptcr_weight=t.ptcr_loss_weight,
                    )
                    count = len(labels["pht_labels"])
                    objective = loss.total * count / size
                    constant = (
                        (
                            t.phla_loss_weight * loss.phla
                            + t.ptcr_loss_weight * loss.ptcr
                            + loss.hierarchy_lambda * loss.hierarchy_positive
                            + loss.pt_implication_lambda * loss.pt_positive
                        )
                        if policy.protect_joint
                        else 0.0
                    )
                    components = {
                        "reported_total_loss": float(loss.total.detach()) * count,
                        "effective_train_loss": float((loss.total - constant).detach())
                        * count,
                    }
                    for name in ("pht", "phla", "ptcr", "hierarchy", "pt_implication"):
                        components[name] = float(getattr(loss, name).detach()) * count
                    grad_losses = {
                        "pht": loss.pht,
                        "pt_aux": loss.ptcr,
                        "constraints": loss.hierarchy_lambda * loss.hierarchy
                        + loss.pt_implication_lambda * loss.pt_implication,
                    }
                else:
                    labels = batch["labels"].to(self.device)
                    count = len(labels)
                    out = self.model(
                        task, batch["peptide_sequences"], batch["receptor_sequences"]
                    )
                    if task == "ptcr":
                        weights = sample_weights(
                            batch["peptide_sequences"],
                            batch["labels"],
                            self.weight_map,
                            self.device,
                        )
                        teacher = (
                            self.teacher_cache.lookup(
                                batch["row_ids"],
                                batch["input_keys"],
                                device=self.device,
                            )
                            if policy.distillation_weight
                            else None
                        )
                        objective, parts = window_pt_loss(
                            out.logits,
                            labels,
                            weights,
                            weight_sum=weight_sum,
                            window_samples=size,
                            smoothing=policy.label_smoothing,
                            teacher_logits=teacher,
                            alpha=policy.distillation_weight,
                            temperature=policy.temperature,
                        )
                        components = {
                            "weighted_ce_sum": float(
                                (parts["ce_per_sample"].detach() * weights).sum()
                            ),
                            "weight_sum": float(weights.sum()),
                            "weight_squares_sum": float(weights.square().sum()),
                            "kd_sum": float(parts["kd"].detach()) * size,
                            "reported_total_loss": float(objective.detach()) * size,
                        }
                        stats["sample_weight_min"] = min(
                            stats.get("sample_weight_min", float("inf")),
                            float(weights.min()),
                        )
                        stats["sample_weight_max"] = max(
                            stats.get("sample_weight_max", 0.0), float(weights.max())
                        )
                        contributions = stats.setdefault("group_loss_contributions", {})
                        for peptide, label, value, weight in zip(
                            batch["peptide_sequences"],
                            batch["labels"].tolist(),
                            parts["ce_per_sample"].detach().tolist(),
                            weights.tolist(),
                        ):
                            key = f"{peptide}:{label}"
                            group = contributions.setdefault(
                                key,
                                {
                                    "samples": 0,
                                    "weighted_ce_sum": 0.0,
                                    "weight_sum": 0.0,
                                },
                            )
                            group["samples"] += 1
                            group["weighted_ce_sum"] += value * weight
                            group["weight_sum"] += weight
                        grad_losses = {
                            "supervised": parts["supervised"],
                            "kd": parts["kd"] * policy.distillation_weight,
                        }
                        if teacher is not None:
                            teacher_prob = teacher.float().softmax(-1)[:, 1]
                            student_prob = out.logits.detach().float().softmax(-1)[:, 1]
                            disagreement = (teacher_prob > 0.5) != labels.bool()
                            components.update(
                                {
                                    "teacher_student_abs_sum": float(
                                        (teacher_prob - student_prob).abs().sum()
                                    ),
                                    "teacher_label_disagreements": int(
                                        disagreement.sum()
                                    ),
                                    "student_correct_on_teacher_disagreement": int(
                                        (
                                            ((student_prob > 0.5) == labels.bool())
                                            & disagreement
                                        ).sum()
                                    ),
                                }
                            )
                        self.seen_pt_indices.update(
                            self.pt_indices[row] for row in batch["row_ids"]
                        )
                        self.pt_exposures += count
                    else:
                        ce = F.cross_entropy(
                            out.logits.float(), labels, reduction="sum"
                        )
                        objective = ce / size
                        components = {"reported_total_loss": float(ce.detach())}
                        grad_losses = {}
            if diag and grad_losses:
                stats["gradient_diagnostics"] = gradient_report(
                    grad_losses, self._shared_parameters()
                )
                diag = False
            self.scaler.scale(objective).backward()
            stats["samples"] += count
            for key, value in components.items():
                stats[key] = stats.get(key, 0.0) + value
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            t.gradient_clip_norm,
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        stats["optimizer_steps"] += 1
        self.cursor["next_batch"] += len(batches)
        if self.global_step % t.log_every_steps == 0:
            self.step_history.append(
                {
                    "round": self.cursor["round"],
                    "task": task,
                    "global_step": self.global_step,
                    "gradient_norm": float(norm),
                    "lr": {g["name"]: g["lr"] for g in self.optimizer.param_groups},
                }
            )
            print(
                f"stage{self.config.stage} round={self.cursor['round']} pass={task} step={self.global_step}",
                flush=True,
            )

    def train_until(self, *, stop_after_steps=None):
        """Resume at an optimizer-window boundary; no model RNG is consumed by skipping rows."""
        if self.completed_round < 0:
            self._record_evaluation()
            self._monitor("initial", 0)
        if (
            self.config.training.deployment_early_stop
            and self.completed_round >= self.config.training.min_rounds
            and self.selectors.states["guarded"].bad_rounds
            >= self.config.training.early_stopping_patience
        ):
            return True
        if stop_after_steps is not None and self.global_step >= stop_after_steps:
            return False
        tasks = ("phla", "ptcr", "joint")
        while self.cursor["round"] <= self.config.training.rounds:
            round_index = self.cursor["round"]
            if self.cursor["pass_index"] == 0 and self.cursor["next_batch"] == 0:
                self._monitor("before_phla", round_index)
            while self.cursor["pass_index"] < 3:
                task = tasks[self.cursor["pass_index"]]
                self.model.set_trainable_pass(task, self.config.policy)
                if not self.current_stats:
                    self.current_stats = {
                        "round": round_index,
                        "task": task,
                        "samples": 0,
                        "optimizer_steps": 0,
                    }
                stats = self.current_stats
                loader = (
                    joint_loader(
                        self.data["joint_train"],
                        self.config,
                        epoch=round_index,
                        start_batch=self.cursor["next_batch"],
                        train=True,
                    )
                    if task == "joint"
                    else pair_loader(
                        self.data["pairs"][task],
                        self.config,
                        cap=self.cap(task),
                        epoch=round_index,
                        start_batch=self.cursor["next_batch"],
                        train=True,
                    )
                )
                iterator = iter(loader)
                start = time.monotonic()
                while batches := list(
                    islice(iterator, self.config.training.gradient_accumulation_steps)
                ):
                    self._window(task, batches, stats)
                    checkpoint_due = (
                        self.config.training.checkpoint_every_steps
                        and self.global_step
                        % self.config.training.checkpoint_every_steps
                        == 0
                    )
                    interrupted = (
                        self._interrupt
                        or stop_after_steps is not None
                        and self.global_step >= stop_after_steps
                    )
                    if self.output_dir and (checkpoint_due or interrupted):
                        self.flush_logs()
                        save_training_checkpoint(self, self.output_dir / "last.pt")
                    if interrupted:
                        return False
                stats["seconds"] = stats.get("seconds", 0.0) + time.monotonic() - start
                if task == "ptcr":
                    rows = [
                        self.data["pairs"]["ptcr"].rows[i] for i in self.seen_pt_indices
                    ]
                    stats.update(
                        {
                            "unique_rows": len(rows),
                            "unique_peptides": len({r[0] for r in rows}),
                            "unique_tcrs": len({r[1] for r in rows}),
                        }
                    )
                    if stats["samples"]:
                        stats.update(
                            {
                                "supervised_CE": stats["weighted_ce_sum"]
                                / stats["weight_sum"],
                                "KD_T2": stats["kd_sum"] / stats["samples"],
                                "alpha_KD": self.config.policy.distillation_weight
                                * stats["kd_sum"]
                                / stats["samples"],
                                "effective_weighted_samples": stats["weight_sum"] ** 2
                                / stats["weight_squares_sum"],
                            }
                        )
                        if self.teacher_cache:
                            stats["teacher_student_probability_MAE"] = (
                                stats["teacher_student_abs_sum"] / stats["samples"]
                            )
                            stats["teacher_label_disagreement_fraction"] = (
                                stats["teacher_label_disagreements"] / stats["samples"]
                            )
                            stats["student_accuracy_on_teacher_disagreement"] = (
                                stats["student_correct_on_teacher_disagreement"]
                                / stats["teacher_label_disagreements"]
                                if stats["teacher_label_disagreements"]
                                else None
                            )
                self.pass_history.append(stats)
                self._swanlab_log(
                    {f"train/{task}": {k: v for k, v in stats.items() if k not in {"group_loss_contributions", "gradient_diagnostics"}}},
                    step=self.global_step,
                )
                self.current_stats = {}
                self.cursor["pass_index"] += 1
                self.cursor["next_batch"] = 0
                self._monitor("after_" + task, round_index)
            self.completed_round = round_index
            self.cursor = {"round": round_index + 1, "pass_index": 0, "next_batch": 0}
            self._record_evaluation()
            if (
                self.config.training.deployment_early_stop
                and self.completed_round >= self.config.training.min_rounds
                and self.selectors.states["guarded"].bad_rounds
                >= self.config.training.early_stopping_patience
            ):
                break
        return True
