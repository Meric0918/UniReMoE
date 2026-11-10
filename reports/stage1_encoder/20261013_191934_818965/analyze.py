"""Reproduce the frozen task-best analysis; run from the project root."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch

ROOT = Path.cwd()
STAMP = "20261013_191934_818965"
OUT = ROOT / "reports/stage1_encoder" / STAMP
sys.path.insert(0, str(ROOT / "source"))

from plm_unified.cache import EmbeddingCache
from plm_unified.data import make_cached_dataloader, normalise_sequence
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from plm_unified.stage1_encoder_model import TaskSpecificStage1EncoderBindingModel, validate_encoder_checkpoint
from plm_unified.trainer import load_training_checkpoint


SPLITS = {
    "phla": ("test_unseen_hla", "test_unseen_pair", "test_unseen_peptide"),
    "ptcr": ("test_unseen_pair", "test_unseen_peptide", "test_unseen_tcr"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro(metrics):
    names = ("auroc", "aupr", "accuracy", "mcc", "f1", "recall", "precision", "nll", "brier", "ece", "performance_avg")
    return {name: float(np.mean([values[name] for values in metrics.values()])) for name in names}


def read_pairs(path, task):
    receptor_column = "HLA" if task == "phla" else "tcr"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            peptide = normalise_sequence(row["peptide"])
            if not 8 <= len(peptide) <= 15:
                continue
            yield peptide, normalise_sequence(row[receptor_column]), int(row["label"])


def wilson(successes, count):
    z = 1.959963984540054
    fraction = successes / count
    denom = 1 + z * z / count
    centre = (fraction + z * z / (2 * count)) / denom
    half = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denom
    return [float(centre - half), float(centre + half)]


def diagnostics(labels, probs, groups):
    pred = probs > .5
    positive = labels == 1
    cm = {
        "TP": int(np.sum(pred & positive)), "FN": int(np.sum(~pred & positive)),
        "TN": int(np.sum(~pred & ~positive)), "FP": int(np.sum(pred & ~positive)),
    }
    correct = pred == positive
    confident = np.maximum(probs, 1 - probs)
    high_error = (~correct) & (np.abs(labels - probs) > .8)
    bins = []
    for indices in np.array_split(np.argsort(probs, kind="mergesort"), 15):
        if len(indices):
            bins.append({"samples": len(indices), "probability": float(np.mean(probs[indices])),
                         "positive_rate": float(np.mean(labels[indices])),
                         "confidence": float(np.mean(confident[indices])),
                         "accuracy": float(np.mean(correct[indices]))})
    return {
        "confusion": cm, "positive": int(positive.sum()), "negative": int((~positive).sum()),
        "predicted_positive_rate": float(pred.mean()), "positive_rate": float(positive.mean()),
        "mean_probability_positive": float(probs[positive].mean()),
        "mean_probability_negative": float(probs[~positive].mean()),
        "high_confidence_errors": int(high_error.sum()),
        "high_confidence_FN": int(np.sum(high_error & positive)),
        "high_confidence_FP": int(np.sum(high_error & ~positive)),
        "accuracy_wilson_95": wilson(int(correct.sum()), len(labels)),
        "calibration_bins": bins,
    }


@torch.inference_mode()
def predict(model, loader, task):
    labels, scores, groups = [], [], []
    loss_sum = samples = 0
    start = time.monotonic()
    for index, batch in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                task, batch["peptide_hidden"].cuda().to(torch.bfloat16),
                batch["peptide_mask"].cuda(), batch["receptor_hidden"].cuda().to(torch.bfloat16),
                batch["receptor_mask"].cuda(),
            )
            loss = torch.nn.functional.cross_entropy(output.logits, batch["labels"].cuda())
        count = len(batch["labels"])
        loss_sum += float(loss.cpu()) * count
        samples += count
        labels.append(batch["labels"].numpy())
        scores.append(output.logits.softmax(dim=1)[:, 1].float().cpu().numpy())
        groups.append(batch["group_ids"].numpy())
        if index % 50 == 0 or index == len(loader):
            print(f"  batches {index}/{len(loader)} samples={samples} elapsed={time.monotonic()-start:.1f}s", flush=True)
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(groups), loss_sum / samples


def parse_training_log(path):
    rounds = []
    current = None
    for line in path.read_text().splitlines():
        match = re.match(r"\[Stage 1 EncoderB\].*\[round=(\d+)/\d+\]", line)
        if match:
            current = {"round": int(match[1]), "phla": {}, "ptcr": {}}
            rounds.append(current)
        if current is None:
            continue
        match = re.match(r"\s+(pHLA|pTCR) (val_\w+|train): (.*)", line)
        if match:
            task = "phla" if match[1] == "pHLA" else "ptcr"
            values = {k: float(v) for k, v in re.findall(r"([A-Za-z_]+)=(-?\d+\.\d+)", match[3])}
            current[task][match[2]] = values
        match = re.match(r"\s+Joint round: score=(\S+) best=(\S+) improved=(\w+)", line)
        if match:
            current["joint"] = {"score": float(match[1]), "best": float(match[2]), "improved": match[3]}
    return rounds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_stage1_encoder_config(ROOT / f"training_logs/stage1_encoder_{STAMP}.yaml")
    train_sets = {}
    for task, folder in (("phla", "PH"), ("ptcr", "PT")):
        rows = list(read_pairs(config.paths.data_root / folder / "train.csv", task))
        train_sets[task] = {
            "peptides": {p for p, r, y in rows}, "receptors": {r for p, r, y in rows},
            "pairs": {(p, r) for p, r, y in rows},
            "positive": sum(y for p, r, y in rows), "samples": len(rows),
        }
    shared_peptides = train_sets["phla"]["peptides"] | train_sets["ptcr"]["peptides"]
    root = config.run_dir()
    results = {
        "run_timestamp": STAMP, "seed": 42, "fold": 1, "encoder_layers": 3,
        "threshold_rule": "probability > 0.5", "precision": "bf16",
        "analyzed_at": datetime.now().isoformat(), "config_snapshot": str(config.source_path),
        "training_rounds": parse_training_log(ROOT / f"training_logs/stage1_encoder_{STAMP}.log"),
        "train": {task: {"samples": values["samples"], "positive": values["positive"],
                           "peptides": len(values["peptides"]), "receptors": len(values["receptors"])}
                  for task, values in train_sets.items()},
        "tasks": {},
    }
    last = load_training_checkpoint(root / f"last_{STAMP}.pt", map_location="cpu")
    results["completion"] = {key: last[key] for key in (
        "stage", "current_round", "global_step", "best_round", "best_round_score",
        "round_stopped", "stopped_tasks", "freeze_shared_encoder", "retention_violations",
    )}
    for task in ("phla", "ptcr"):
        path = root / f"best_{task}_{STAMP}.pt"
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        validate_encoder_checkpoint(checkpoint, config)
        item = {
            "checkpoint": str(path), "sha256": sha256(path),
            "best_round": checkpoint["current_round"], "score": checkpoint["best_score"],
            "parameters": sum(v.numel() for v in checkpoint["model"].values()),
            "validation": checkpoint["validation_metrics"],
            "validation_macro": macro(checkpoint["validation_metrics"]),
            "alternatives": {}, "test": {}, "audits": {},
        }
        for role in ("calibration", "robust"):
            alternative = load_training_checkpoint(root / f"best_{task}_{role}_{STAMP}.pt", map_location="cpu")
            metrics = alternative["selection_metrics"]
            item["alternatives"][role] = {
                "round": alternative["current_round"], "selection_score": alternative["selection_score"],
                "macro": macro(metrics), "metrics": metrics,
            }
        folder = "PH" if task == "phla" else "PT"
        receptor = "hla" if task == "phla" else "tcr"
        caches = {}
        for entity in ("peptide", receptor):
            cache_path = ROOT / "embedding_cache/plm_stage1_ph_pt_benchmark_test/fold_1" / entity
            manifest = checkpoint["cache_metadata"][entity]
            caches[entity] = EmbeddingCache(
                cache_path, expected_hidden_dim=manifest["hidden_dim"],
                expected_fingerprint=manifest["model"]["checkpoint_fingerprint"],
            )
        model = TaskSpecificStage1EncoderBindingModel(config.model, task).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        for split in SPLITS[task]:
            csv_path = config.paths.data_root / folder / "benchmarks" / f"{split}.csv"
            loader = make_cached_dataloader(
                csv_path, task, caches["peptide"], caches[receptor], batch_size=1024,
                shuffle=False, seed=42, num_workers=2, pin_memory=True,
                persistent_workers=False, peptide_min_length=8, peptide_max_length=15,
            )
            print(f"Evaluating {task}/{split}: {len(loader.dataset)} rows", flush=True)
            labels, scores, groups, loss = predict(model, loader, task)
            metrics = compute_binary_metrics(labels, scores, loss=loss, threshold=.5,
                                             strict_threshold=True, group_ids=groups).to_dict()
            details = diagnostics(labels, scores, groups)
            rows = list(read_pairs(csv_path, task))
            assert [y for p, r, y in rows] == labels.tolist()
            peptide_counts = Counter(p for p, r, y in rows)
            receptor_counts = Counter(r for p, r, y in rows)
            pair_counts = Counter((p, r) for p, r, y in rows)
            seen = {}
            conflicts = 0
            for p, r, y in rows:
                if (p, r) in seen and seen[p, r] != y:
                    conflicts += 1
                seen[p, r] = y
            audit = {
                "data": str(csv_path), "sha256": sha256(csv_path),
                "samples": len(rows), "unique_peptides": len(peptide_counts),
                "unique_receptors": len(receptor_counts),
                "duplicate_pair_rows": len(rows) - len(pair_counts), "conflicting_pair_rows": conflicts,
                "pairs_in_task_train": sum(count for pair, count in pair_counts.items() if pair in train_sets[task]["pairs"]),
                "peptide_rows_seen_in_task_train": sum(count for p, count in peptide_counts.items() if p in train_sets[task]["peptides"]),
                "peptide_rows_seen_in_any_task_train": sum(count for p, count in peptide_counts.items() if p in shared_peptides),
                "receptor_rows_seen_in_task_train": sum(count for r, count in receptor_counts.items() if r in train_sets[task]["receptors"]),
            }
            item["test"][split] = {"metrics": metrics, **details}
            item["audits"][split] = audit
            pred_path = OUT / f"{task}_{split}_predictions.csv.gz"
            with gzip.open(pred_path, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["peptide", receptor, "label", "probability", "prediction", "group_id"])
                for (p, r, y), probability, group in zip(rows, scores, groups):
                    writer.writerow([p, r, y, float(probability), int(probability > .5), int(group)])
            print(json.dumps({"task": task, "split": split,
                              **{key: round(metrics[key], 6) for key in ("auroc", "aupr", "performance_avg", "recall", "ece")}}, ensure_ascii=False), flush=True)
            results["tasks"][task] = item
            (OUT / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
        item["test_macro"] = macro({split: values["metrics"] for split, values in item["test"].items()})
        for cache in caches.values():
            cache.close()
        del model
        torch.cuda.empty_cache()
        results["tasks"][task] = item
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Complete: {OUT / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
