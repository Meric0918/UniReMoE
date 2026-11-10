#!/usr/bin/env python
"""Evaluate the selected pMTnet PHT checkpoint across benchmark CSV files."""

import glob
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from train_pht_from_scratch import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    build_model,
    encode_split,
    load_atchley_factors,
    load_blosum50,
)


DATA_DIR = Path("/home/mjp/Project/PLM/UnifyImmun/dataset/PHT/benchmarks")
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "training_output" / "PHT_scratch_20261010_211230"
TARGET = "PHT"
THRESHOLD = 0.5


def load_benchmark_csv(path):
    # Preserve leading zeroes in class codes in the exported predictions.
    frame = pd.read_csv(str(path), dtype={"class_code": str})
    required = {"peptide", "HLA", "tcr", TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("{} missing required columns: {}".format(path, missing))
    if frame.empty:
        raise ValueError("{} contains no rows".format(path))

    for column in ("peptide", "HLA", "tcr"):
        if frame[column].isnull().any():
            raise ValueError("{} has missing values in {}".format(path, column))
        frame[column] = (frame[column].astype(str).str.replace("\xa0", "", regex=False)
                         .str.strip().str.upper())
        if (frame[column] == "").any():
            raise ValueError("{} has empty sequences in {}".format(path, column))

    invalid = frame[(frame["peptide"].str.len() > 15) |
                    (frame["tcr"].str.len() > 80) |
                    (~frame["HLA"].str.len().isin((33, 34)))]
    if not invalid.empty:
        raise ValueError("{} has invalid encoder input lengths at CSV row {}".format(
            path, invalid.index[0] + 2))

    labels = pd.to_numeric(frame[TARGET], errors="raise")
    if not set(labels.unique().tolist()).issubset({0, 1}):
        raise ValueError("{} must contain only binary {} labels".format(path, TARGET))
    if len(set(labels.unique().tolist())) != 2:
        raise ValueError("{} must contain both {} classes".format(path, TARGET))
    frame[TARGET] = labels.astype(np.int8)
    frame["benchmark_file"] = path.name
    frame["partition"] = "test_unseen" if path.name.startswith("test_") else "val_unseen"
    return frame


def calculate_metrics(frame):
    labels = frame[TARGET].values.astype(np.int8)
    logits = frame["PHT_logit"].values.astype(np.float64)
    probabilities = frame["PHT_probability"].values.astype(np.float64)
    predictions = (probabilities >= THRESHOLD).astype(np.int8)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    specificity = tn / float(tn + fp) if tn + fp else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "rows": int(len(labels)),
        "positive_rows": int(np.sum(labels == 1)),
        "negative_rows": int(np.sum(labels == 0)),
        "accuracy": float((tp + tn) / float(len(labels))),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc_score(labels, logits)),
        "average_precision": float(average_precision_score(labels, logits)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def test_table_row(name, metrics):
    return "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
        name, metrics["roc_auc"], metrics["average_precision"], metrics["accuracy"],
        metrics["mcc"], metrics["f1"], metrics["precision"], metrics["recall"],
        metrics["specificity"])


def row_counter(frame):
    normalized = frame.copy()
    for column in ("peptide", "HLA", "tcr"):
        normalized[column] = (normalized[column].astype(str).str.replace("\xa0", "", regex=False)
                              .str.strip().str.upper())
    labels = pd.to_numeric(normalized[TARGET], errors="raise").astype(np.int8).values
    return Counter((peptide, hla, tcr, int(label)) for peptide, hla, tcr, label in zip(
        normalized["peptide"].values, normalized["HLA"].values,
        normalized["tcr"].values, labels))


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError("No CSV files in {}".format(DATA_DIR))

    model_metrics_path = MODEL_DIR / "metrics.json"
    with open(str(model_metrics_path), "r") as model_metrics_file:
        model_metrics = json.load(model_metrics_file)
    best_epoch = int(model_metrics["best_epoch"])
    best_validation_loss = float(model_metrics["best_validation_loss"])

    factors = load_atchley_factors(PROJECT_ROOT / "library" / "Atchley_factors.csv")
    blosum = load_blosum50()
    model = build_model()
    weights_path = MODEL_DIR / "full_model_weights.h5"
    model.load_weights(str(weights_path))

    prediction_frames = []
    for file_path in files:
        path = Path(file_path)
        frame = load_benchmark_csv(path)
        encoded = encode_split(frame, factors, blosum)
        logits = model.predict(list(encoded), batch_size=512).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        predictions = (probabilities >= THRESHOLD).astype(np.int8)

        result = frame.copy()
        result["PHT_logit"] = logits
        result["PHT_probability"] = probabilities
        result["PHT_prediction"] = predictions
        result["correct"] = result[TARGET].values == predictions
        prediction_frames.append(result)
        print("Evaluated {}: {} rows".format(path.name, len(result)))
        del encoded

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    duplicates = int(all_predictions.duplicated(subset=["peptide", "HLA", "tcr"]).sum())
    prediction_path = OUTPUT_DIR / "predictions.csv"
    all_predictions.to_csv(str(prediction_path), index=False)

    by_file = {}
    for file_name, frame in all_predictions.groupby("benchmark_file", sort=True):
        by_file[file_name] = calculate_metrics(frame)
    by_partition = {}
    for partition, frame in all_predictions.groupby("partition", sort=True):
        by_partition[partition] = calculate_metrics(frame)

    partition_alignment = {}
    for partition, original_split in (("val_unseen", "val"), ("test_unseen", "test")):
        benchmark_rows = all_predictions[all_predictions["partition"] == partition]
        original_path = DEFAULT_DATA_DIR / (original_split + ".csv")
        original_frame = pd.read_csv(str(original_path))
        benchmark_counter = row_counter(benchmark_rows)
        original_counter = row_counter(original_frame)
        partition_alignment[partition] = {
            "original_file": str(original_path),
            "benchmark_rows": int(len(benchmark_rows)),
            "original_rows": int(len(original_frame)),
            "exact_match_by_peptide_hla_tcr_pht": benchmark_counter == original_counter,
            "benchmark_only_rows": int(sum((benchmark_counter - original_counter).values())),
            "original_only_rows": int(sum((original_counter - benchmark_counter).values())),
        }

    metrics = {
        "evaluation_date": str(date.today()),
        "target": TARGET,
        "threshold": THRESHOLD,
        "model_selection": "lowest original validation loss",
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "dataset_dir": str(DATA_DIR),
        "input_files": files,
        "weights": str(weights_path),
        "architecture": str(MODEL_DIR / "model_architecture.json"),
        "rows_across_files": int(len(all_predictions)),
        "duplicate_triplets_across_files": duplicates,
        "by_partition": by_partition,
        "partition_alignment": partition_alignment,
        "by_file": by_file,
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(str(metrics_path), "w") as output_file:
        json.dump(metrics, output_file, indent=2, sort_keys=True)

    columns = "| 数据集 | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity |"
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# pMTnet 最佳模型 PHT Test 集指标",
        "",
        "模型为第 {} 轮最佳 checkpoint；PHT 概率阈值为 0.5。AUPR 使用 average precision 计算；Specificity = TN / (TN + FP)。".format(
            best_epoch),
        "",
        columns,
        separator,
        test_table_row("test_unseen（总体）", by_partition["test_unseen"]),
    ]
    for file_name in sorted(by_file):
        if file_name.startswith("test_"):
            lines.append(test_table_row(file_name, by_file[file_name]))

    report_path = OUTPUT_DIR / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {}".format(report_path))
    print("Wrote {}".format(metrics_path))
    print("Wrote {}".format(prediction_path))
    print(json.dumps({"by_partition": by_partition, "by_file": by_file},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
