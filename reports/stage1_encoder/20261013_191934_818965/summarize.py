"""Build diagnostic figures and the branch analysis from frozen predictions."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from analyze import OUT, ROOT, STAMP, read_pairs, wilson

LABELS = {
    "unseen_hla": "New HLA", "unseen_pair": "Both unseen",
    "unseen_peptide": "New peptide", "unseen_tcr": "New TCR",
}
COLORS = ["#2563eb", "#ea580c", "#059669"]


def short(split):
    return LABELS[split.split("_", 1)[1]]


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def load_predictions(task, split):
    with gzip.open(OUT / f"{task}_{split}_predictions.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    return rows, np.asarray([int(row["label"]) for row in rows]), np.asarray([float(row["probability"]) for row in rows])


def add_validation_details(results):
    for task, info in results["tasks"].items():
        folder = "PH" if task == "phla" else "PT"
        info["validation_diagnostics"] = {}
        error_dir = "pHLA" if task == "phla" else "pTCR"
        path = ROOT / "csv/stage1_encoder" / STAMP / error_dir / f"{error_dir}_round_{info['best_round']}.csv"
        high_errors = defaultdict(Counter)
        for row in csv.DictReader(path.open(newline="")):
            high_errors[row["source"]]["FN" if int(row["label"]) else "FP"] += 1
        for split, metrics in info["validation"].items():
            data = ROOT / "dataset" / folder / "benchmarks" / f"{split}.csv"
            rows = list(read_pairs(data, task))
            positive = sum(y for p, r, y in rows)
            negative = len(rows) - positive
            tp = round(positive * metrics["recall"])
            tn = round(negative * metrics["specificity"])
            assert len(rows) == metrics["samples"]
            assert abs((tp + tn) / len(rows) - metrics["accuracy"]) < 1e-9
            info["validation_diagnostics"][split] = {
                "positive": positive, "negative": negative,
                "confusion": {"TP": tp, "TN": tn, "FP": negative - tn, "FN": positive - tp},
                "high_confidence_FN": high_errors[split]["FN"],
                "high_confidence_FP": high_errors[split]["FP"],
            }


def branch_figure(task, info, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    splits = list(info["test"])
    x = np.arange(len(splits))
    for axis, metric, title in zip(axes[0], ["auroc", "performance_avg", "recall"], ["AUROC", "Five-metric score", "Positive recall at p > 0.5"]):
        validation = [info["validation"][split.replace("test_", "val_")][metric] for split in splits]
        test = [info["test"][split]["metrics"][metric] for split in splits]
        axis.bar(x - .18, validation, .36, color="#94a3b8", label="Validation")
        bars = axis.bar(x + .18, test, .36, color=COLORS, label="Test")
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set(xticks=x, xticklabels=[short(s) for s in splits], ylim=(0, 1.06), title=title)
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="lower right", fontsize=8)
    for split, color in zip(splits, COLORS):
        labels, probs = predictions[split]
        metric = info["test"][split]["metrics"]
        fpr, tpr, _ = roc_curve(labels, probs)
        axes[1, 0].plot(fpr, tpr, color=color, label=f"{short(split)} ({metric['auroc']:.3f})")
        precision, recall, _ = precision_recall_curve(labels, probs)
        axes[1, 1].plot(recall, precision, color=color, label=f"{short(split)} ({metric['aupr']:.3f})")
        order = np.argsort(probs, kind="mergesort")
        bins = [chunk for chunk in np.array_split(order, 15) if len(chunk)]
        axes[1, 2].plot([probs[b].mean() for b in bins], [labels[b].mean() for b in bins],
                        "o-", color=color, markersize=4, label=f"{short(split)} (ECE {metric['ece']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#9ca3af")
    axes[1, 1].axhline(.5, linestyle="--", color="#9ca3af")
    axes[1, 2].plot([0, 1], [0, 1], "--", color="#9ca3af")
    for axis, title, xlabel, ylabel in zip(axes[1], ["Test ROC", "Test precision-recall", "Test probability calibration"],
                                         ["False positive rate", "Recall", "Mean positive probability"],
                                         ["True positive rate", "Precision", "Observed positive fraction"]):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=(-.02, 1.02))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"{'pHLA' if task == 'phla' else 'pTCR'} task-best: round {info['best_round']}, seed 42, three encoder layers", fontsize=14)
    fig.savefig(OUT / f"{task}_diagnostics.png", dpi=180)
    plt.close(fig)


def training_figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, task in enumerate(("phla", "ptcr")):
        history = results["training_rounds"]
        rounds = [item["round"] for item in history]
        splits = list(results["tasks"][task]["validation"])
        axes[0, column].plot(rounds, [item[task]["train"]["AUROC"] for item in history], "o-", color="#111827", label="Train (epoch aggregate)")
        axes[1, column].plot(rounds, [item[task]["train"]["NLL"] for item in history], "o-", color="#111827", label="Train NLL")
        for split, color in zip(splits, COLORS):
            axes[0, column].plot(rounds, [item[task][split]["AUROC"] for item in history], "o-", color=color, label=short(split))
            axes[1, column].plot(rounds, [item[task][split]["NLL"] for item in history], "o-", color=color, label=short(split))
        for row in (0, 1):
            axis = axes[row, column]
            axis.axvline(results["tasks"][task]["best_round"], linestyle="--", color="#7c3aed", label="Task-best round")
            axis.set(xlabel="Alternating training round", ylabel="AUROC" if row == 0 else "NLL",
                     title=f"{'pHLA' if task == 'phla' else 'pTCR'}: {'ranking' if row == 0 else 'calibration / loss'}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.savefig(OUT / "training_dynamics.png", dpi=180)
    plt.close(fig)


def umap_figure(results):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    base = ROOT / "figures/stage1_encoder/best_model_umap/fold_1/seed_42" / STAMP
    for row_index, task in enumerate(("phla", "ptcr")):
        best = results["tasks"][task]["best_round"]
        for column, stage in enumerate(("plm", "sequence_encoder", "classifier_input")):
            path = base / task / f"round_{best:04d}" / f"coordinates_{stage}.csv.gz"
            with gzip.open(path, "rt") as handle:
                rows = list(csv.DictReader(handle))
            axis = axes[row_index, column]
            for source, marker in (("train", "o"), ("val", "^")):
                for label, color in (("0", "#2563eb"), ("1", "#dc2626")):
                    selected = [r for r in rows if r["source"] == source and r["label"] == label]
                    axis.scatter([float(r["umap_1"]) for r in selected], [float(r["umap_2"]) for r in selected],
                                 marker=marker, color=color, s=8, alpha=.38,
                                 label=f"{source} {'positive' if label == '1' else 'negative'}", rasterized=True)
            axis.set(title=f"{'pHLA' if task == 'phla' else 'pTCR'} round {best}: {stage}", xlabel="UMAP 1", ylabel="UMAP 2")
            axis.legend(fontsize=7, markerscale=2)
    fig.suptitle("Fixed train / validation samples; separate reference space for every task and feature stage", fontsize=12)
    fig.savefig(OUT / "feature_umap.png", dpi=180)
    plt.close(fig)


def group_errors(task, split, rows, labels, probs):
    key = "hla" if task == "phla" else "peptide"
    groups = defaultdict(Counter)
    for row, label, prob in zip(rows, labels, probs):
        counts = groups[row[key]]
        counts["samples"] += 1
        counts["positive"] += int(label == 1)
        counts["negative"] += int(label == 0)
        counts["FN"] += int(label == 1 and prob <= .5)
        counts["FP"] += int(label == 0 and prob > .5)
        counts["high_confidence_FN"] += int(label == 1 and prob < .2)
        counts["high_confidence_FP"] += int(label == 0 and prob > .8)
    summary = [{"group": name, **dict(counts), "FN_rate": counts["FN"] / counts["positive"] if counts["positive"] else None,
                "FP_rate": counts["FP"] / counts["negative"] if counts["negative"] else None} for name, counts in groups.items()]
    summary.sort(key=lambda values: values["FN"], reverse=True)
    path = OUT / f"{task}_{split}_groups.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary[:10]


def main():
    results = json.loads((OUT / "evaluation.json").read_text())
    add_validation_details(results)
    for task, info in results["tasks"].items():
        predictions = {}
        info["top_error_groups"] = {}
        for split, entry in info["test"].items():
            rows, labels, probs = load_predictions(task, split)
            predictions[split] = (labels, probs)
            info["top_error_groups"][split] = group_errors(task, split, rows, labels, probs)
            if task == "phla" and split == "test_unseen_pair":
                rng = np.random.default_rng(42)
                positive, negative = np.flatnonzero(labels), np.flatnonzero(1 - labels)
                estimates = []
                for _ in range(2000):
                    indices = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
                    estimates.append(roc_auc_score(labels[indices], probs[indices]))
                entry["auroc_stratified_row_bootstrap_95"] = np.quantile(estimates, [.025, .975]).tolist()
        branch_figure(task, info, predictions)
    training_figure(results)
    umap_figure(results)
    (OUT / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Figures and analysis.json written", flush=True)


if __name__ == "__main__":
    main()
