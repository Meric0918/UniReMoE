#!/usr/bin/env python3
"""Evaluate the joint-best and task-best Stage1 Encoder checkpoints."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from plm_unified.cache import EmbeddingCache
from plm_unified.data import make_cached_dataloader, normalise_sequence, stable_group_id
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage1_encoder_config import load_stage1_encoder_config
from plm_unified.stage1_encoder_model import (
    Stage1EncoderBindingModel,
    TaskSpecificStage1EncoderBindingModel,
    extract_encoder_task_state_dict,
    validate_encoder_checkpoint,
)
from plm_unified.trainer import load_training_checkpoint

STAMP = "20261013_191934_818965"
MODEL_DIR = ROOT / "trained_model/PLM_Stage1_Encoder/fold_1/seed_42"
CACHE_DIR = ROOT / "embedding_cache/plm_stage1_ph_pt_benchmark_test/fold_1"
CONFIG = ROOT / f"training_logs/stage1_encoder_{STAMP}.yaml"
SPLITS = {
    "phla": ("test_unseen_hla", "test_unseen_pair", "test_unseen_peptide"),
    "ptcr": ("test_unseen_pair", "test_unseen_peptide", "test_unseen_tcr"),
}
VAL_SPLITS = {
    "phla": ("val_unseen_hla", "val_unseen_pair", "val_unseen_peptide"),
    "ptcr": ("val_unseen_pair", "val_unseen_peptide", "val_unseen_tcr"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, task):
    receptor_column = "HLA" if task == "phla" else "tcr"
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for number, row in enumerate(csv.DictReader(handle), 2):
            peptide = normalise_sequence(row["peptide"])
            if not 8 <= len(peptide) <= 15:
                raise ValueError(f"Unexpected peptide length in {path}:{number}")
            rows.append((number, peptide, normalise_sequence(row[receptor_column]), int(row["label"])))
    return rows


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(results):
    (OUT / "metrics.json").write_text(
        json.dumps(clean(results), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def predict(models, loader, task):
    labels, groups = [], []
    scores = {role: [] for role in models}
    loss_sum = {role: 0.0 for role in models}
    count = 0
    started = time.monotonic()
    for index, batch in enumerate(loader, 1):
        peptide = batch["peptide_hidden"].cuda().to(torch.bfloat16)
        peptide_mask = batch["peptide_mask"].cuda()
        receptor = batch["receptor_hidden"].cuda().to(torch.bfloat16)
        receptor_mask = batch["receptor_mask"].cuda()
        target = batch["labels"].cuda()
        n = len(target)
        for role, model in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(task, peptide, peptide_mask, receptor, receptor_mask)
                loss = F.cross_entropy(output.logits, target)
            loss_sum[role] += float(loss.cpu()) * n
            scores[role].append(output.logits.softmax(dim=1)[:, 1].float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        groups.append(batch["group_ids"].numpy())
        count += n
        if index % 50 == 0 or index == len(loader):
            print(f"  {task} batch {index}/{len(loader)} rows={count} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return (np.concatenate(labels), np.concatenate(groups),
            {role: np.concatenate(parts) for role, parts in scores.items()},
            {role: loss_sum[role] / count for role in models})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the Stage1 Encoder bf16 evaluation")
    config = load_stage1_encoder_config(CONFIG)
    joint_path = MODEL_DIR / f"best_round/best_joint_{STAMP}.pt"
    joint = load_training_checkpoint(joint_path, map_location="cpu")
    validate_encoder_checkpoint(joint, config)
    results = {
        "evaluation_started": datetime.now().isoformat(),
        "architecture": "stage1_encoder", "fold": 1, "seed": 42,
        "precision": "bf16", "batch_size": 1024, "workers": 2,
        "threshold": 0.5, "threshold_rule": "probability > 0.5",
        "config_snapshot": str(CONFIG),
        "checkpoints": {"joint": {"path": str(joint_path), "sha256": sha256(joint_path),
                                     "round": joint["current_round"], "validation_score": joint["round_score"]}},
        "tasks": {},
    }
    train = {}
    for task, folder in (("phla", "PH"), ("ptcr", "PT")):
        train_rows = read_rows(config.paths.data_root / folder / "train.csv", task)
        val_rows = []
        for split in VAL_SPLITS[task]:
            val_rows.extend(read_rows(config.paths.data_root / folder / "benchmarks" / f"{split}.csv", task))
        train[task] = {
            "pairs": {(p, r) for _, p, r, _ in train_rows},
            "peptides": {p for _, p, _, _ in train_rows},
            "receptors": {r for _, _, r, _ in train_rows},
            "val_pairs": {(p, r) for _, p, r, _ in val_rows},
        }
        del train_rows, val_rows
    shared_peptides = train["phla"]["peptides"] | train["ptcr"]["peptides"]

    for task, folder, receptor_entity in (("phla", "PH", "hla"), ("ptcr", "PT", "tcr")):
        task_path = MODEL_DIR / f"best_{task}_{STAMP}.pt"
        checkpoint = load_training_checkpoint(task_path, map_location="cpu")
        validate_encoder_checkpoint(checkpoint, config)
        if checkpoint.get("task") != task or checkpoint.get("checkpoint_type") != "task_model":
            raise ValueError(f"Wrong task checkpoint: {task_path}")
        expected = extract_encoder_task_state_dict(joint["model"], task)
        identical = set(expected) == set(checkpoint["model"]) and all(
            torch.equal(expected[k], checkpoint["model"][k]) for k in expected
        )
        results["checkpoints"][task] = {
            "path": str(task_path), "sha256": sha256(task_path),
            "round": checkpoint["current_round"], "validation_score": checkpoint["best_score"],
            "same_task_weights_as_joint": identical,
        }
        caches = {}
        joint_model = task_model = None
        try:
            for entity in ("peptide", receptor_entity):
                for saved in (joint, checkpoint):
                    if saved["cache_metadata"][entity]["model"]["checkpoint_fingerprint"] != checkpoint["cache_metadata"][entity]["model"]["checkpoint_fingerprint"]:
                        raise ValueError(f"{task}: checkpoint encoder fingerprint mismatch for {entity}")
                manifest = checkpoint["cache_metadata"][entity]
                caches[entity] = EmbeddingCache(
                    CACHE_DIR / entity, expected_hidden_dim=manifest["hidden_dim"],
                    expected_fingerprint=manifest["model"]["checkpoint_fingerprint"],
                )
            joint_model = Stage1EncoderBindingModel(config.model).cuda().eval()
            joint_model.load_state_dict(joint["model"], strict=True)
            task_model = TaskSpecificStage1EncoderBindingModel(config.model, task).cuda().eval()
            task_model.load_state_dict(checkpoint["model"], strict=True)
            results["tasks"][task] = {}
            for split in SPLITS[task]:
                path = config.paths.data_root / folder / "benchmarks" / f"{split}.csv"
                rows = read_rows(path, task)
                loader = make_cached_dataloader(
                    path, task, caches["peptide"], caches[receptor_entity], batch_size=1024,
                    shuffle=False, seed=42, num_workers=2, pin_memory=True,
                    persistent_workers=False, peptide_min_length=8, peptide_max_length=15,
                )
                if len(rows) != len(loader.dataset) or loader.dataset.filter_stats["kept"] != len(rows):
                    raise ValueError(f"Unexpected row filtering in {path}")
                expected_labels = np.asarray([y for _, _, _, y in rows])
                expected_groups = np.asarray([stable_group_id(p if task == "ptcr" else r) for _, p, r, _ in rows])
                if not np.array_equal(loader.dataset.labels, expected_labels) or not np.array_equal(loader.dataset.group_ids, expected_groups):
                    raise ValueError(f"Dataset row order does not match {path}")
                print(f"Evaluating {task}/{split}: {len(rows)} rows with both checkpoints", flush=True)
                labels, groups, scores, losses = predict({"joint": joint_model, "task_best": task_model}, loader, task)
                if not np.array_equal(labels, expected_labels) or not np.array_equal(groups, expected_groups):
                    raise ValueError(f"Predictions are not aligned with {path}")
                if identical and not np.array_equal(scores["joint"], scores["task_best"]):
                    raise ValueError(f"Identical weights produced differing results for {path}")
                test_pairs = {(p, r) for _, p, r, _ in rows}
                peptides = {p for _, p, _, _ in rows}
                receptors = {r for _, _, r, _ in rows}
                audit = {
                    "source_file": str(path), "sha256": sha256(path), "input_rows": len(rows),
                    "filter_stats": loader.dataset.filter_stats,
                    "same_task_train_pair_overlap": len(test_pairs & train[task]["pairs"]),
                    "same_task_val_pair_overlap": len(test_pairs & train[task]["val_pairs"]),
                    "unique_peptides": len(peptides),
                    "peptides_in_ph_or_pt_train": len(peptides & shared_peptides),
                    "unique_receptors": len(receptors),
                    "receptors_in_same_task_train": len(receptors & train[task]["receptors"]),
                }
                result = {"data": audit, "models": {}}
                for role, probability in scores.items():
                    metrics = compute_binary_metrics(labels, probability, loss=losses[role], threshold=0.5,
                                                     strict_threshold=True, group_ids=groups).to_dict()
                    prediction = probability > 0.5
                    name = f"{role}_{task}_{split}_predictions.csv.gz"
                    with gzip.open(OUT / name, "wt", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["source_row", "peptide", receptor_entity, "label", "probability", "prediction", "group_id"])
                        for (row_number, p, r, y), prob, pred, group in zip(rows, probability, prediction, groups):
                            writer.writerow([row_number, p, r, y, float(prob), int(pred), int(group)])
                    result["models"][role] = {
                        "metrics": metrics, "positive_samples": int(labels.sum()),
                        "negative_samples": len(labels) - int(labels.sum()), "prediction_file": name,
                        "confusion": {"tp": int(np.sum(prediction & (labels == 1))),
                                      "tn": int(np.sum(~prediction & (labels == 0))),
                                      "fp": int(np.sum(prediction & (labels == 0))),
                                      "fn": int(np.sum(~prediction & (labels == 1)))},
                    }
                    print(json.dumps({"task": task, "split": split, "model": role, "n": len(labels),
                                      "auroc": metrics["auroc"], "aupr": metrics["aupr"],
                                      "mcc": metrics["mcc"]}), flush=True)
                results["tasks"][task][split] = result
                save(results)
        finally:
            for cache in caches.values():
                cache.close()
            del joint_model, task_model
            torch.cuda.empty_cache()
    results["evaluation_finished"] = datetime.now().isoformat()
    save(results)
    print(f"Complete: {OUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
