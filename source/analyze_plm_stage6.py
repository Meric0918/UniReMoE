#!/usr/bin/env python3
"""Compare the four Stage-6 design cells with paired stratified bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.metrics import compute_binary_metrics, task_score

CONDITIONS = (
    "stage5_joint_matched",
    "stage6_phla_only_matched",
    "stage6_joint_phla_full",
    "stage6_phla_only_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="CONDITION=CSV",
        help="Repeat once for each of the four predefined conditions.",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stage5-evaluation", type=Path)
    parser.add_argument("--joint-full-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("stage6_analysis"))
    return parser.parse_args()


def _parse_prediction_args(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        condition, separator, raw_path = value.partition("=")
        if not separator or not condition or not raw_path:
            raise ValueError("--prediction must use CONDITION=CSV")
        if condition in paths:
            raise ValueError(f"Duplicate prediction condition: {condition}")
        paths[condition] = Path(raw_path).expanduser().resolve()
    if set(paths) != set(CONDITIONS):
        missing = sorted(set(CONDITIONS) - set(paths))
        extra = sorted(set(paths) - set(CONDITIONS))
        raise ValueError(f"Prediction conditions mismatch; missing={missing}, extra={extra}")
    return paths


def _read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "label", "phla_probability"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"]
            if not sample_id:
                raise ValueError(f"Empty sample_id at {path}:{row_number}")
            sample_ids.append(sample_id)
            labels.append(int(row["label"]))
            probabilities.append(float(row["phla_probability"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_id values in {path}")
    label_values = np.asarray(labels, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if len(label_values) == 0 or set(np.unique(label_values)) != {0, 1}:
        raise ValueError(f"{path} must contain both binary classes")
    if np.any(~np.isfinite(probability_values)) or np.any(
        (probability_values < 0) | (probability_values > 1)
    ):
        raise ValueError(f"Invalid probability in {path}")
    return sample_ids, label_values, probability_values


def _binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )


def _point_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = compute_binary_metrics(
        labels,
        probabilities,
        loss=_binary_cross_entropy(labels, probabilities),
        threshold=0.5,
    )
    return {**metrics.to_dict(), "task_score": task_score(metrics)}


def _weighted_auc_aupr_many(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    ranking=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact weighted ROC/PR trapezoids for many bootstrap replicates."""

    if ranking is None:
        order = np.argsort(-probabilities, kind="mergesort")
        sorted_scores = probabilities[order]
        endpoints = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )
    else:
        order, endpoints = ranking
    sorted_labels = labels[order].astype(np.float64)
    sorted_weights = weights[:, order].astype(np.float64, copy=False)
    true_positive = np.cumsum(sorted_weights * sorted_labels, axis=1)
    false_positive = np.cumsum(sorted_weights * (1.0 - sorted_labels), axis=1)
    true_positive = true_positive[:, endpoints]
    false_positive = false_positive[:, endpoints]
    positive_total = true_positive[:, -1:]
    negative_total = false_positive[:, -1:]
    true_positive_rate = true_positive / positive_total
    false_positive_rate = false_positive / negative_total
    previous_tpr = np.concatenate(
        [np.zeros((len(weights), 1)), true_positive_rate[:, :-1]], axis=1
    )
    previous_fpr = np.concatenate(
        [np.zeros((len(weights), 1)), false_positive_rate[:, :-1]], axis=1
    )
    auroc = np.sum(
        (false_positive_rate - previous_fpr)
        * (true_positive_rate + previous_tpr)
        * 0.5,
        axis=1,
    )
    denominator = true_positive + false_positive
    precision = np.divide(
        true_positive,
        denominator,
        out=np.ones_like(true_positive),
        where=denominator != 0,
    )
    recall = true_positive_rate
    previous_precision = np.concatenate(
        [np.ones((len(weights), 1)), precision[:, :-1]], axis=1
    )
    previous_recall = np.concatenate(
        [np.zeros((len(weights), 1)), recall[:, :-1]], axis=1
    )
    aupr = np.sum(
        (recall - previous_recall) * (precision + previous_precision) * 0.5,
        axis=1,
    )
    return auroc, aupr


def paired_stratified_bootstrap(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 16,
) -> dict[str, dict[str, np.ndarray]]:
    if samples < 1 or chunk_size < 1:
        raise ValueError("Bootstrap samples and chunk size must be positive")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    rankings = {}
    for condition, probabilities in predictions.items():
        order = np.argsort(-probabilities, kind="mergesort")
        sorted_scores = probabilities[order]
        endpoints = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )
        rankings[condition] = (order, endpoints)
    results = {
        condition: {
            "auroc": np.empty(samples, dtype=np.float64),
            "aupr": np.empty(samples, dtype=np.float64),
            "task_score": np.empty(samples, dtype=np.float64),
        }
        for condition in predictions
    }
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        rows = stop - start
        weights = np.zeros((rows, len(labels)), dtype=np.int32)
        for row in range(rows):
            positive_draw = positive[
                rng.integers(0, len(positive), size=len(positive))
            ]
            negative_draw = negative[
                rng.integers(0, len(negative), size=len(negative))
            ]
            np.add.at(weights[row], positive_draw, 1)
            np.add.at(weights[row], negative_draw, 1)
        for condition, probabilities in predictions.items():
            auroc, aupr = _weighted_auc_aupr_many(
                labels,
                probabilities,
                weights,
                ranking=rankings[condition],
            )
            results[condition]["auroc"][start:stop] = auroc
            results[condition]["aupr"][start:stop] = aupr
            results[condition]["task_score"][start:stop] = (auroc + aupr) / 2
    return results


def _interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "bootstrap_mean": float(np.mean(values)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def _interpret_improvement(point: float, lower: float) -> str:
    if point >= 0.01 and lower > 0:
        return "clear_improvement"
    if point >= 0.005:
        return "directional_improvement"
    if abs(point) < 0.005:
        return "negligible"
    if point < 0:
        return "regression"
    return "small_improvement"


def _load_joint_retention(stage5_path: Path, full_path: Path) -> dict:
    with stage5_path.open("r", encoding="utf-8") as handle:
        stage5 = json.load(handle)
    with full_path.open("r", encoding="utf-8") as handle:
        full = json.load(handle)
    output = {}
    for task in ("ptcr", "triplet"):
        baseline = stage5.get("joint_retention", {}).get(task, {}).get("task_score")
        current = full.get("joint_retention", {}).get(task, {}).get("task_score")
        if baseline is None or current is None:
            raise ValueError(f"Evaluation JSON is missing joint_retention/{task}")
        delta = float(current) - float(baseline)
        output[task] = {
            "stage5_joint_matched": float(baseline),
            "stage6_joint_phla_full": float(current),
            "delta": delta,
            "status": (
                "acceptable"
                if delta >= -0.005
                else "unacceptable"
                if delta < -0.01
                else "warning"
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    paths = _parse_prediction_args(args.prediction)
    loaded = {condition: _read_predictions(path) for condition, path in paths.items()}
    reference_ids, labels, _ = loaded["stage5_joint_matched"]
    predictions = {}
    for condition in CONDITIONS:
        sample_ids, condition_labels, probabilities = loaded[condition]
        if sample_ids != reference_ids:
            raise ValueError(f"Sample alignment mismatch for {condition}")
        if not np.array_equal(condition_labels, labels):
            raise ValueError(f"Label alignment mismatch for {condition}")
        predictions[condition] = probabilities
    point = {
        condition: _point_metrics(labels, probabilities)
        for condition, probabilities in predictions.items()
    }
    bootstrap = paired_stratified_bootstrap(
        labels,
        predictions,
        samples=args.bootstrap_samples,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    contrast_definitions = {
        "coverage_effect": (
            {"stage6_joint_phla_full": 1.0, "stage5_joint_matched": -1.0}
        ),
        "task_conflict_effect": (
            {"stage6_phla_only_matched": 1.0, "stage5_joint_matched": -1.0}
        ),
        "interaction_effect": {
            "stage6_joint_phla_full": 1.0,
            "stage5_joint_matched": -1.0,
            "stage6_phla_only_full": -1.0,
            "stage6_phla_only_matched": 1.0,
        },
    }
    contrasts = {}
    for name, coefficients in contrast_definitions.items():
        metrics = {}
        for metric in ("auroc", "aupr", "task_score"):
            values = sum(
                coefficient * bootstrap[condition][metric]
                for condition, coefficient in coefficients.items()
            )
            point_delta = sum(
                coefficient * point[condition][metric]
                for condition, coefficient in coefficients.items()
            )
            metrics[metric] = {"point_delta": point_delta, **_interval(values)}
        primary = metrics["task_score"]
        metrics["task_score"]["interpretation"] = _interpret_improvement(
            primary["point_delta"], primary["ci95_lower"]
        )
        contrasts[name] = metrics
    payload = {
        "split": args.split,
        "samples": len(labels),
        "positive_samples": int(labels.sum()),
        "negative_samples": int(len(labels) - labels.sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "primary_metric": "task_score=(AUROC+AUPR)/2",
        "prediction_sources": {key: str(value) for key, value in paths.items()},
        "condition_metrics": point,
        "contrasts": contrasts,
    }
    if bool(args.stage5_evaluation) != bool(args.joint_full_evaluation):
        raise ValueError(
            "Use --stage5-evaluation and --joint-full-evaluation together"
        )
    if args.stage5_evaluation:
        payload["joint_retention"] = _load_joint_retention(
            args.stage5_evaluation.resolve(), args.joint_full_evaluation.resolve()
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / timestamped_filename(
        f"stage6_comparison_{args.split}.json", create_artifact_timestamp()
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_path)
    print(json.dumps({"analysis": str(output_path), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
