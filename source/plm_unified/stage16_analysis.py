"""Ablation summaries and paired cluster bootstrap on locked predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .stage16_checkpoint import load_torch_checkpoint, write_json
from .stage16_selection import PT_SPLITS, task_score


def _weighted_scores(labels, probabilities, weights):
    precision, recall, _ = precision_recall_curve(
        labels, probabilities, sample_weight=weights
    )
    pred = probabilities > 0.5
    tp = float(weights[pred & (labels == 1)].sum())
    tn = float(weights[~pred & (labels == 0)].sum())
    fp = float(weights[pred & (labels == 0)].sum())
    fn = float(weights[~pred & (labels == 1)].sum())
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "auroc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "aupr": float(auc(recall, precision)),
        "mcc": (tp * tn - fp * fn) / np.sqrt(denominator) if denominator else 0.0,
    }


def paired_cluster_bootstrap(
    baseline, candidate, *, cluster="peptides", replicates=2000, seed=42
):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if len(set(candidate["row_ids"])) != len(candidate["row_ids"]):
        raise ValueError("Duplicate candidate prediction row IDs")
    index = {row: i for i, row in enumerate(candidate["row_ids"])}
    if set(index) != set(baseline["row_ids"]):
        raise ValueError("Predictions have different row identities")
    order = [index[row] for row in baseline["row_ids"]]
    labels = np.asarray(baseline["labels"], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(candidate["labels"])[order]):
        raise ValueError("Prediction labels differ")
    if [candidate["input_keys"][i] for i in order] != baseline["input_keys"]:
        raise ValueError("Prediction input identities differ")
    if [candidate[cluster][i] for i in order] != baseline[cluster]:
        raise ValueError("Cluster identities differ")
    a = np.asarray(baseline["probabilities"], dtype=np.float64)
    b = np.asarray(candidate["probabilities"], dtype=np.float64)[order]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite predictions")
    groups, ids = np.unique(np.asarray(baseline[cluster]), return_inverse=True)
    if len(np.unique(labels)) < 2:
        raise ValueError("Bootstrap requires both labels")
    generator = np.random.default_rng(seed)
    deltas = {key: [] for key in ("auroc", "aupr", "mcc")}
    for _ in range(replicates):
        counts = np.bincount(
            generator.integers(len(groups), size=len(groups)), minlength=len(groups)
        )
        weights = counts[ids].astype(np.float64)
        if not weights[labels == 0].sum() or not weights[labels == 1].sum():
            continue
        left, right = _weighted_scores(labels, a, weights), _weighted_scores(
            labels, b, weights
        )
        for key in deltas:
            deltas[key].append(right[key] - left[key])
    left, right = _weighted_scores(labels, a, np.ones(len(labels))), _weighted_scores(
        labels, b, np.ones(len(labels))
    )
    return {
        "cluster": cluster,
        "groups": len(groups),
        "requested_replicates": replicates,
        "valid_replicates": len(deltas["auroc"]),
        "seed": seed,
        "difference": {key: right[key] - left[key] for key in deltas},
        "ci95": {
            key: np.quantile(values, [0.025, 0.975]).tolist() if values else None
            for key, values in deltas.items()
        },
    }


def ablation_summary(run_dirs, output_prefix):
    rows = []
    for directory in map(Path, run_dirs):
        run = json.loads((directory / "run_manifest.json").read_text())
        history = [
            json.loads(line)
            for line in (directory / "round_metrics.jsonl").read_text().splitlines()
        ]
        decisions = [
            json.loads(line)
            for line in (directory / "selection_decisions.jsonl")
            .read_text()
            .splitlines()
        ]
        for selector in ("legacy", "pt_validation", "guarded"):
            index = directory / f"best_{selector}.json"
            selected = (
                json.loads(index.read_text())["round"] if index.is_file() else None
            )
            entry = next((v for v in history if v["round"] == selected), None)
            metrics = entry["metrics"] if entry else None
            rows.append(
                {
                    "run": str(directory),
                    "stage": run["config"]["stage"],
                    "condition": run["config"]["condition"],
                    "policy_overrides": run["config"].get("policy_overrides", {}),
                    "effective_policy": run.get("effective_policy"),
                    "seed": run["config"]["runtime"]["seed"],
                    "debug": run["debug"],
                    "selector": selector,
                    "selected_round": selected,
                    "guard_feasible_fraction": sum(x["gate"]["pass"] for x in decisions)
                    / len(decisions),
                    "pt_macro_S": (
                        sum(task_score(metrics["pt_independent"][s]) for s in PT_SPLITS)
                        / 3
                        if metrics
                        else None
                    ),
                    "pht_S": task_score(metrics["pht_val"]["pht"]) if metrics else None,
                    "metrics": metrics,
                    "optimizer_steps": run.get("global_step"),
                    "seconds_this_invocation": run.get("seconds_this_invocation"),
                    "total_wall_seconds": run.get("total_wall_seconds"),
                }
            )
    prefix = Path(output_prefix)
    write_json(prefix.with_suffix(".json"), {"runs": rows})
    lines = [
        "# Stage16/17/18 ablation summary",
        "",
        "Validation-only; no automatic test-based configuration selection.",
        "",
        "| Stage | Condition | Seed | Debug | Selector | Round | PT macro S | PHT S | Feasible fraction |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in (
                    "stage",
                    "condition",
                    "seed",
                    "debug",
                    "selector",
                    "selected_round",
                    "pt_macro_S",
                    "pht_S",
                    "guard_feasible_fraction",
                )
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--candidate-predictions", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.runs:
        ablation_summary(args.runs, args.output_prefix)
    if bool(args.baseline_predictions) != bool(args.candidate_predictions):
        raise ValueError("Both paired prediction files are required")
    if args.baseline_predictions:
        left = load_torch_checkpoint(args.baseline_predictions)
        right = load_torch_checkpoint(args.candidate_predictions)
        if "pt_independent" in left:
            left, right = left["pt_independent"], right["pt_independent"]
        else:
            left = {k: v["ptcr"] for k, v in left.items() if k.startswith("PT/")}
            right = {k: v["ptcr"] for k, v in right.items() if k.startswith("PT/")}
        if set(left) != set(right):
            raise ValueError("Prediction split names differ")
        results = {}
        for split in left:
            results[split] = {
                "peptide": paired_cluster_bootstrap(
                    left[split], right[split], replicates=args.bootstrap, seed=args.seed
                )
            }
            if split.endswith("tcr"):
                results[split]["tcr_sensitivity"] = paired_cluster_bootstrap(
                    left[split],
                    right[split],
                    cluster="receptors",
                    replicates=args.bootstrap,
                    seed=args.seed,
                )
        write_json(args.output_prefix.with_suffix(".bootstrap.json"), results)
    if not args.runs and not args.baseline_predictions:
        raise ValueError("Provide runs or paired predictions")
