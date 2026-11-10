#!/usr/bin/env python3
"""Independently verify prediction rows and metrics against benchmark CSVs."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from plm_unified.data import normalise_sequence, stable_group_id


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(labels, probabilities):
    pred = probabilities > 0.5
    tn, fp, fn, tp = map(int, confusion_matrix(labels, pred, labels=[0, 1]).ravel())
    prec, recall, _ = precision_recall_curve(labels, probabilities)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    order = np.argsort(clipped, kind="mergesort")
    ece = 0.0
    for idx in np.array_split(order, min(15, len(labels))):
        confidence = np.maximum(clipped[idx], 1 - clipped[idx])
        accuracy = np.mean((clipped[idx] >= 0.5) == labels[idx])
        ece += len(idx) / len(labels) * abs(float(np.mean(confidence)) - float(accuracy))
    return {
        "samples": len(labels), "auroc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(auc(recall, prec)), "accuracy": float(accuracy_score(labels, pred)),
        "mcc": float(matthews_corrcoef(labels, pred)), "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)), "specificity": tn / (tn + fp),
        "nll": float(np.mean(-np.log(np.where(labels == 1, clipped, 1 - clipped)))),
        "brier": float(np.mean((clipped - labels) ** 2)), "ece": ece,
    }, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    result = json.loads((OUT / "metrics.json").read_text())
    assert result.get("evaluation_finished") and len(result["tasks"]) == 2
    checked = {"all_checks_passed": False, "datasets": {}}
    for task, splits in result["tasks"].items():
        for split, entry in splits.items():
            source = Path(entry["data"]["source_file"])
            if sha256(source) != entry["data"]["sha256"]:
                raise ValueError(f"Dataset hash changed: {source}")
            expected_receptor = "HLA" if task == "phla" else "tcr"
            output_receptor = "hla" if task == "phla" else "tcr"
            for role, model in entry["models"].items():
                prediction_file = OUT / model["prediction_file"]
                labels, scores = [], []
                with source.open(newline="", encoding="utf-8-sig") as sf, gzip.open(
                    prediction_file, "rt", newline="", encoding="utf-8"
                ) as pf:
                    for rownum, (s, p) in enumerate(zip_longest(csv.DictReader(sf), csv.DictReader(pf)), 2):
                        if s is None or p is None:
                            raise ValueError(f"{task}/{split}/{role}: row count mismatch")
                        peptide = normalise_sequence(s["peptide"])
                        receptor = normalise_sequence(s[expected_receptor])
                        label, probability = int(s["label"]), float(p["probability"])
                        group = stable_group_id(peptide if task == "ptcr" else receptor)
                        if (int(p["source_row"]) != rownum or p["peptide"] != peptide or
                            p[output_receptor] != receptor or int(p["label"]) != label or
                            int(p["group_id"]) != group or not 0 <= probability <= 1 or
                            int(p["prediction"]) != int(probability > 0.5)):
                            raise ValueError(f"{task}/{split}/{role}: mismatch at source row {rownum}")
                        labels.append(label)
                        scores.append(probability)
                y = np.asarray(labels, dtype=np.int64)
                p = np.asarray(scores, dtype=np.float64)
                metrics, confusion = independent_metrics(y, p)
                for name, actual in metrics.items():
                    expected = model["metrics"][name]
                    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"{task}/{split}/{role}: {name} differs: {actual} vs {expected}")
                if confusion != model["confusion"] or len(y) != entry["data"]["input_rows"]:
                    raise ValueError(f"{task}/{split}/{role}: confusion matrix or count mismatch")
                if int(y.sum()) != model["positive_samples"]:
                    raise ValueError(f"{task}/{split}/{role}: positive count mismatch")
                checked["datasets"][f"{task}/{split}/{role}"] = {
                    "rows": len(y), "source_sha256": entry["data"]["sha256"],
                    "prediction_sha256": sha256(prediction_file),
                    "metrics_checked": sorted(metrics), "confusion_checked": True,
                }
                print(f"{task}/{split}/{role}: verified {len(y)} rows and metrics", flush=True)
    assert len(checked["datasets"]) == 12
    checked["all_checks_passed"] = True
    (OUT / "verification.json").write_text(json.dumps(checked, indent=2) + "\n")
    print("All checks passed", flush=True)


if __name__ == "__main__":
    main()
