"""Three-pass alternating optimization with exact round-boundary continuation."""

import json
import math

import torch

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .stage8_trainer import _rng_state, _restore_rng_state
from .stage11_checkpoint import atomic_torch_save, load_stage11_checkpoint, training_contract
from .stage11_data import model_inputs, move_cached_batch
from .stage11_evaluation import autocast_context, evaluate_stage11_model, write_predictions
from .stage11_loss import balanced_pht_loss, multi_positive_clip_loss
from .stage11_model import ARCHITECTURE
from .trainer import build_scheduler


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Stage11Trainer:
    def __init__(self, model, config, data, loaders, cache_metadata, *, tracker=None, run_timestamp=None):
        config.validate()
        self.model, self.config, self.data, self.loaders = model, config, data, loaders
        self.settings, self.cache_metadata = config.stage11, cache_metadata
        self.device = next(model.parameters()).device
        self.contract = training_contract(config, data.fingerprint, cache_metadata)
        self.tracker = tracker
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.settings.learning_rate,
                                           weight_decay=self.settings.weight_decay)
        passes = int(self.settings.condition in {"ph", "both"}) + int(self.settings.condition in {"pht", "both"})
        self.contrastive_steps = math.ceil(len(data.pht_pairs) / self.settings.contrastive_batch_size)
        self.steps_per_round = len(loaders.classification) + passes * self.contrastive_steps
        self.scheduler = build_scheduler(self.optimizer, total_steps=self.settings.max_rounds * self.steps_per_round,
                                         warmup_ratio=self.settings.scheduler_warmup_ratio,
                                         warmup_steps=0 if self.settings.scheduler_warmup_ratio == 0 else None)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.settings.mixed_precision == "fp16")
        self.run_timestamp = validate_artifact_timestamp(run_timestamp or create_artifact_timestamp())
        self.round = self.global_step = self.bad_rounds = self.best_round = 0
        self.best_score = float("-inf")
        self.best_state = self.best_evaluation = self.latest_evaluation = None
        self.output_dir = config.run_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path(self, role, suffix="pt"):
        return self.output_dir / f"stage11_{role}_{self.run_timestamp}.{suffix}"

    def _update(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage11 loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm, error_if_nonfinite=True)
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= before
        if updated:
            self.scheduler.step()
            self.global_step += 1
        return int(updated)

    def _pass(self, batches, task):
        self.model.train()
        totals = {"planned_batches": len(batches), "batches": 0, "optimizer_steps": 0,
                  "skipped_batches": 0, "sampled_relations": 0, "samples": 0,
                  "valid_left": 0, "valid_right": 0, "positive_pairs": 0, "negative_pairs": 0,
                  "known_negative_pairs": 0, "unlabeled_pairs": 0}
        loss_sum = positive_sum = negative_sum = 0.0
        for raw in batches:
            batch = move_cached_batch(raw, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            totals["batches"] += 1
            with autocast_context(self.device, self.settings.mixed_precision):
                if task == "classification":
                    output = self.model.forward_pht(*model_inputs(batch))
                    loss = balanced_pht_loss(output.pht_logits, batch["pht_labels"])
                    totals["samples"] += len(batch["pht_labels"])
                else:
                    output = getattr(self.model, f"forward_{task}_contrastive")(*model_inputs(batch, task=task))
                    result = multi_positive_clip_loss(output.left, output.right, batch["positive_mask"],
                                                       temperature=self.settings.temperature)
                    loss = result.total
                    for name in ("valid_left", "valid_right", "positive_pairs", "negative_pairs"):
                        totals[name] += getattr(result, name)
                    for name in ("known_negative_pairs", "unlabeled_pairs", "sampled_relations"):
                        totals[name] += batch[name]
                    positive_sum += result.positive_similarity * result.positive_pairs
                    negative_sum += result.negative_similarity * result.negative_pairs
                    if not result.valid:
                        totals["skipped_batches"] += 1
                        continue
            totals["optimizer_steps"] += self._update(loss)
            loss_sum += float(loss.detach().cpu())
            if self.tracker and self.global_step % self.settings.log_every_steps == 0:
                self.tracker.log({f"train/{task}/loss": float(loss.detach().cpu()), "round": self.round + 1},
                                 step=self.global_step)
        totals["loss"] = loss_sum / max(totals["batches"] - totals["skipped_batches"], 1)
        if task != "classification":
            totals["positive_similarity"] = positive_sum / max(totals["positive_pairs"], 1)
            totals["negative_similarity"] = negative_sum / max(totals["negative_pairs"], 1)
            totals["positive_fraction"] = totals["positive_pairs"] / max(totals["positive_pairs"] + totals["negative_pairs"], 1)
        return totals

    def train_round(self, round_index):
        if round_index != self.round + 1 or round_index > self.settings.max_rounds:
            raise ValueError("Stage11 rounds must be consecutive and within max_rounds")
        training = {}
        for task in ("ph", "pht"):
            if self.settings.condition in {task, "both"}:
                print(f"stage11 round={round_index} phase={task}_contrastive", flush=True)
                training[task] = self._pass(self.loaders.contrastive(task, round_index), task)
        self.loaders.classification.batch_sampler.set_epoch(round_index)
        print(f"stage11 round={round_index} phase=classification", flush=True)
        training["classification"] = self._pass(self.loaders.classification, "classification")
        self.round = round_index
        evaluation = evaluate_stage11_model(self.model, self.loaders.validation, device=self.device,
                                             precision=self.settings.mixed_precision)
        if not math.isfinite(evaluation.score):
            raise ValueError("Stage11 validation must contain both labels and a finite score")
        self.latest_evaluation = {"metrics": evaluation.metrics, "score": evaluation.score}
        if evaluation.score > self.best_score + self.settings.early_stopping_min_delta:
            self.best_score, self.best_round, self.bad_rounds = evaluation.score, self.round, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_evaluation = self.latest_evaluation
            self.save_best()
            write_predictions(self.path("validation_predictions", "csv"), evaluation,
                              self.loaders.validation.dataset.records)
        else:
            self.bad_rounds += 1
        self.save_last()
        record = {"round": self.round, "global_step": self.global_step,
                  "training": training, "validation": self.latest_evaluation}
        with self.path("history", "jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        if self.tracker:
            metrics = {"round": self.round, "validation/score": evaluation.score}
            metrics.update({f"validation/{k}": v for k, v in evaluation.metrics.items() if isinstance(v, (float, int)) and math.isfinite(v)})
            for task, values in training.items():
                metrics.update({f"train/{task}/{k}": v for k, v in values.items()})
            self.tracker.log(metrics, step=self.global_step)
        print(f"stage11 round={self.round} score={evaluation.score:.6f} "
              f"AUROC={evaluation.metrics['auroc']:.6f} AUPR={evaluation.metrics['aupr']:.6f}", flush=True)
        return record

    def _base(self, role, state, evaluation):
        return {"architecture": ARCHITECTURE, "format_version": 1, "checkpoint_type": "stage11_model",
                "artifact_role": role, "config": self.config.to_dict(), "model": state,
                "training_contract": self.contract, "cache_metadata": self.cache_metadata,
                "run_timestamp": self.run_timestamp, "round": self.round, "global_step": self.global_step,
                "evaluation": evaluation, "swanlab_run_id": getattr(self.tracker, "run_id", None)}

    def save_best(self):
        value = self._base("best", self.best_state, self.best_evaluation)
        value["round"] = self.best_round
        atomic_torch_save(value, self.path("best"))

    def save_last(self):
        value = self._base("last", self.model.state_dict(), self.latest_evaluation)
        value.update({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                      "scaler": self.scaler.state_dict(), "rng_state": _rng_state(),
                      "loaders": self.loaders.state_dict(), "best_model": self.best_state,
                      "best_evaluation": self.best_evaluation, "best_score": self.best_score,
                      "best_round": self.best_round, "bad_rounds": self.bad_rounds})
        atomic_torch_save(value, self.path("last"))

    def resume(self, path):
        value = load_stage11_checkpoint(path, role="last")
        if value["training_contract"] != self.contract:
            raise ValueError("Stage11 resume data, configuration or PLM/cache contract mismatch")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.scheduler.load_state_dict(value["scheduler"])
        self.scaler.load_state_dict(value["scaler"])
        self.loaders.load_state_dict(value["loaders"])
        self.round, self.global_step, self.bad_rounds = value["round"], value["global_step"], value["bad_rounds"]
        self.best_score, self.best_round = value["best_score"], value["best_round"]
        self.best_state, self.best_evaluation = value["best_model"], value["best_evaluation"]
        self.latest_evaluation = value["evaluation"]
        self.run_timestamp = validate_artifact_timestamp(value["run_timestamp"])
        _restore_rng_state(value["rng_state"])
        if not self.path("best").exists():
            self.save_best()

    def fit(self):
        for index in range(self.round + 1, self.settings.max_rounds + 1):
            if self.round >= self.settings.min_rounds and self.bad_rounds >= self.settings.early_stopping_patience:
                break
            self.train_round(index)
        summary = {"round": self.round, "best_round": self.best_round, "best_score": self.best_score,
                   "condition": self.settings.condition, "best": str(self.path("best")),
                   "last": str(self.path("last")), "global_step": self.global_step,
                   "planned_steps_per_round": self.steps_per_round, "data": self.data.report}
        self.path("summary", "json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
