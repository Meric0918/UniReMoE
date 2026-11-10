"""Paired and peptide-clustered statistics for Stage-7 predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .metrics import compute_binary_metrics
from .stage7_inference import SCORE_COLUMNS
from .triplet_evaluation import select_mcc_threshold


CLASSIFICATION_COMPARISONS = {
    "stage3_pht_vs_stage1_ptcr": (
        "stage3_pht_margin",
        "stage1_ptcr_margin",
    ),
    "stage3_pht_vs_stage1_late": (
        "stage3_pht_margin",
        "stage1_late_margin",
    ),
    "stage3_pht_vs_stage3_ptcr": (
        "stage3_pht_margin",
        "stage3_ptcr_margin",
    ),
    "stage3_pht_vs_stage3_late": (
        "stage3_pht_margin",
        "stage3_late_margin",
    ),
}


def _confidence_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def bootstrap_column_means(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 32,
) -> np.ndarray:
    """Bootstrap row means without allocating samples x rows all at once."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("Bootstrap values must be a non-empty 1-D/2-D array")
    rng = np.random.default_rng(seed)
    output = np.empty((samples, array.shape[1]), dtype=np.float64)
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        output[start : start + count] = array[indices].mean(axis=1)
    return output


def sign_flip_pvalue(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 4096,
) -> float:
    """One-sided Monte-Carlo randomisation test for mean(values) > 0."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Sign-flip values must be a non-empty 1-D array")
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        signs = rng.integers(0, 2, size=(count, len(array)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_means = (signs * array).mean(axis=1)
        extreme += int(np.sum(null_means >= observed))
    return (extreme + 1) / (samples + 1)


def _group_rows(rows: Sequence[Mapping], experiment: str) -> dict[str, list[Mapping]]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == experiment:
            groups[row["group_id"]].append(row)
    return dict(groups)


def analyse_phla_valid_pairs(
    rows: Sequence[Mapping],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    groups = _group_rows(rows, "phla_valid_pair")
    if not groups:
        return {"groups": 0, "status": "no_pairs"}
    by_model: dict[str, list[list[float]]] = {column: [] for column in SCORE_COLUMNS}
    peptides = set()
    group_peptides = []
    decoy_total = 0
    for group_id in sorted(groups):
        group = groups[group_id]
        cognate = [row for row in group if row["role"] == "cognate"]
        decoys = [row for row in group if str(row["role"]).startswith("decoy_")]
        if len(cognate) != 1 or not decoys:
            raise ValueError(f"Malformed pHLA-valid group: {group_id}")
        peptide = str(cognate[0]["peptide_group"])
        peptides.add(peptide)
        group_peptides.append(peptide)
        decoy_total += len(decoys)
        for column in SCORE_COLUMNS:
            positive = float(cognate[0][column])
            negative = np.asarray([float(row[column]) for row in decoys])
            differences = positive - negative
            wins = np.where(differences > 0, 1.0, np.where(differences < 0, 0.0, 0.5))
            greater = int(np.sum(negative > positive))
            equal = int(np.sum(negative == positive))
            average_rank = 1.0 + greater + 0.5 * equal
            by_model[column].append(
                [
                    float(differences.mean()),
                    float(wins.mean()),
                    1.0 / average_rank,
                    float(greater == 0 and equal == 0),
                ]
            )
    metrics = ("mean_delta", "paired_win_rate", "mrr", "hits_at_1")
    result = {
        "groups": len(groups),
        "peptides": len(peptides),
        "decoys": decoy_total,
        "models": {},
    }
    arrays = {}
    for model_index, (column, values) in enumerate(by_model.items()):
        array = np.asarray(values, dtype=np.float64)
        arrays[column] = array
        bootstrap = bootstrap_column_means(
            array,
            samples=bootstrap_samples,
            seed=seed + model_index,
        )
        result["models"][column] = {
            metric: {
                "point": float(array[:, index].mean()),
                "ci95": _confidence_interval(bootstrap[:, index]),
            }
            for index, metric in enumerate(metrics)
        }
        peptide_arrays = np.asarray(
            [
                array[
                    [
                        index
                        for index, value in enumerate(group_peptides)
                        if value == peptide
                    ]
                ].mean(axis=0)
                for peptide in sorted(peptides)
            ],
            dtype=np.float64,
        )
        peptide_bootstrap = bootstrap_column_means(
            peptide_arrays,
            samples=bootstrap_samples,
            seed=seed + 1000 + model_index,
        )
        result["models"][column].update(
            {
                f"peptide_macro_{metric}": {
                    "point": float(peptide_arrays[:, index].mean()),
                    "ci95": _confidence_interval(peptide_bootstrap[:, index]),
                }
                for index, metric in enumerate(metrics)
            }
        )
    difference = arrays["stage3_pht_margin"][:, 0] - arrays["stage3_late_margin"][:, 0]
    difference_bootstrap = bootstrap_column_means(
        difference,
        samples=bootstrap_samples,
        seed=seed + 100,
    )[:, 0]
    result["stage3_pht_minus_late_delta"] = {
        "point": float(difference.mean()),
        "ci95": _confidence_interval(difference_bootstrap),
    }
    return result


def _tetrad_value(group: Sequence[Mapping], column: str) -> float:
    roles = {row["role"]: row for row in group}
    expected = {"diag_1", "diag_2", "offdiag_12", "offdiag_21"}
    if set(roles) != expected:
        raise ValueError(
            f"Malformed tetrad roles; expected={sorted(expected)}, got={sorted(roles)}"
        )
    return (
        float(roles["diag_1"][column])
        + float(roles["diag_2"][column])
        - float(roles["offdiag_12"][column])
        - float(roles["offdiag_21"][column])
    ) / 2.0


def analyse_tetrads(
    rows: Sequence[Mapping],
    experiment: str,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict:
    groups = _group_rows(rows, experiment)
    if not groups:
        return {"groups": 0, "status": "no_tetrads"}
    group_peptide = {group_id: group[0]["peptide_group"] for group_id, group in groups.items()}
    peptide_values: dict[str, dict[str, list[float]]] = {
        column: defaultdict(list) for column in SCORE_COLUMNS
    }
    for group_id in sorted(groups):
        for column in SCORE_COLUMNS:
            peptide_values[column][group_peptide[group_id]].append(
                _tetrad_value(groups[group_id], column)
            )
    result = {
        "groups": len(groups),
        "peptides": len(set(group_peptide.values())),
        "models": {},
    }
    model_peptide_arrays = {}
    for model_index, column in enumerate(SCORE_COLUMNS):
        values = np.asarray(
            [
                np.mean(peptide_values[column][peptide])
                for peptide in sorted(peptide_values[column])
            ],
            dtype=np.float64,
        )
        model_peptide_arrays[column] = values
        wins = np.where(values > 0, 1.0, np.where(values < 0, 0.0, 0.5))
        stacked = np.column_stack([values, wins])
        bootstrap = bootstrap_column_means(
            stacked,
            samples=bootstrap_samples,
            seed=seed + model_index,
        )
        result["models"][column] = {
            "peptide_macro_d": {
                "point": float(values.mean()),
                "ci95": _confidence_interval(bootstrap[:, 0]),
            },
            "peptide_win_rate": {
                "point": float(wins.mean()),
                "ci95": _confidence_interval(bootstrap[:, 1]),
            },
            "sign_flip_pvalue_one_sided": sign_flip_pvalue(
                values,
                samples=permutation_samples,
                seed=seed + 1000 + model_index,
            ),
            "max_abs_peptide_d": float(np.max(np.abs(values))),
        }
    difference = (
        model_peptide_arrays["stage3_pht_margin"]
        - model_peptide_arrays["stage3_late_margin"]
    )
    difference_bootstrap = bootstrap_column_means(
        difference,
        samples=bootstrap_samples,
        seed=seed + 2000,
    )[:, 0]
    result["stage3_pht_minus_late_peptide_macro_d"] = {
        "point": float(difference.mean()),
        "ci95": _confidence_interval(difference_bootstrap),
        "sign_flip_pvalue_one_sided": sign_flip_pvalue(
            difference,
            samples=permutation_samples,
            seed=seed + 3000,
        ),
    }
    result["additive_cancellation_max_abs"] = {
        column: result["models"][column]["max_abs_peptide_d"]
        for column in (
            "stage1_phla_margin",
            "stage1_ptcr_margin",
            "stage1_late_margin",
            "stage3_phla_margin",
            "stage3_ptcr_margin",
            "stage3_late_margin",
        )
    }
    return result


def _sigmoid(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _binary_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )


def analyse_original_triplets(
    rows: Sequence[Mapping],
    *,
    thresholds: Mapping[str, float] | None,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Compare Stage-3 PHT with HLA-free and additive Stage-1 baselines."""

    groups = _group_rows(rows, "original_triplet")
    if not groups:
        return {"groups": 0, "status": "no_original_triplets"}
    ordered = []
    for group_id in sorted(groups):
        group = groups[group_id]
        if len(group) != 1 or group[0]["role"] != "original":
            raise ValueError(f"Malformed original-triplet group: {group_id}")
        ordered.append(group[0])
    labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return {
            "groups": len(ordered),
            "status": "insufficient_classes",
        }
    probabilities = {
        column: _sigmoid([float(row[column]) for row in ordered])
        for column in SCORE_COLUMNS
    }
    selected_thresholds = {}
    model_results = {}
    for column in SCORE_COLUMNS:
        if thresholds is None:
            threshold, _ = select_mcc_threshold(labels, probabilities[column])
            threshold_source = "selected_on_validation"
        else:
            if column not in thresholds:
                raise ValueError(f"Missing validation threshold for {column}")
            threshold = float(thresholds[column])
            threshold_source = "validation"
        selected_thresholds[column] = threshold
        metrics = compute_binary_metrics(
            labels,
            probabilities[column],
            loss=_binary_loss(labels, probabilities[column]),
            threshold=threshold,
        ).to_dict()
        model_results[column] = {
            "threshold": threshold,
            "threshold_source": threshold_source,
            "metrics": metrics,
        }

    peptide_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        peptide_indices[str(row["peptide_group"])].append(index)
    comparisons = {}
    metric_names = ("auroc", "aupr", "accuracy", "mcc", "f1")
    for comparison_index, (name, (candidate, baseline)) in enumerate(
        CLASSIFICATION_COMPARISONS.items()
    ):
        candidate_metrics = model_results[candidate]["metrics"]
        baseline_metrics = model_results[baseline]["metrics"]
        candidate_correct = (
            probabilities[candidate] >= selected_thresholds[candidate]
        ) == labels
        baseline_correct = (
            probabilities[baseline] >= selected_thresholds[baseline]
        ) == labels
        peptide_gain = np.asarray(
            [
                np.mean(
                    candidate_correct[indices].astype(np.float64)
                    - baseline_correct[indices].astype(np.float64)
                )
                for _, indices in sorted(peptide_indices.items())
            ],
            dtype=np.float64,
        )
        bootstrap = bootstrap_column_means(
            peptide_gain,
            samples=bootstrap_samples,
            seed=seed + comparison_index,
        )[:, 0]
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "metric_delta": {
                metric: float(candidate_metrics[metric] - baseline_metrics[metric])
                for metric in metric_names
            },
            "peptide_macro_accuracy_gain": {
                "point": float(peptide_gain.mean()),
                "ci95": _confidence_interval(bootstrap),
                "peptides": len(peptide_gain),
            },
        }
    return {
        "groups": len(ordered),
        "peptides": len(peptide_indices),
        "selected_thresholds": selected_thresholds,
        "models": model_results,
        "comparisons": comparisons,
    }


def analyse_legacy_hla_shift(rows: Sequence[Mapping]) -> dict:
    groups = _group_rows(rows, "legacy_hla_shift")
    if not groups:
        return {"groups": 0, "status": "disabled"}
    labels = []
    original = []
    shifted = []
    changed = 0
    changed_by_label = defaultdict(lambda: [0, 0])
    for group_id in sorted(groups):
        roles = {row["role"]: row for row in groups[group_id]}
        if set(roles) != {"original", "shifted"}:
            raise ValueError(f"Malformed legacy HLA-shift group: {group_id}")
        label = int(roles["original"]["label"])
        labels.append(label)
        original.append(float(roles["original"]["stage3_pht_margin"]))
        shifted.append(float(roles["shifted"]["stage3_pht_margin"]))
        is_changed = roles["original"]["hla"] != roles["shifted"]["hla"]
        changed += int(is_changed)
        changed_by_label[label][0] += int(is_changed)
        changed_by_label[label][1] += 1
    label_values = np.asarray(labels, dtype=np.int64)
    changed_result = {
        "groups": len(groups),
        "changed_fraction": changed / len(groups),
        "changed_fraction_label_0": (
            changed_by_label[0][0] / changed_by_label[0][1]
            if changed_by_label[0][1]
            else None
        ),
        "changed_fraction_label_1": (
            changed_by_label[1][0] / changed_by_label[1][1]
            if changed_by_label[1][1]
            else None
        ),
    }
    if len(np.unique(label_values)) < 2:
        return {**changed_result, "status": "insufficient_classes"}
    original_probability = _sigmoid(original)
    shifted_probability = _sigmoid(shifted)
    original_metrics = compute_binary_metrics(
        label_values,
        original_probability,
        loss=_binary_loss(label_values, original_probability),
    ).to_dict()
    shifted_metrics = compute_binary_metrics(
        label_values,
        shifted_probability,
        loss=_binary_loss(label_values, shifted_probability),
    ).to_dict()
    return {
        **changed_result,
        "original_metrics": original_metrics,
        "shifted_metrics": shifted_metrics,
        "metric_delta_shifted_minus_original": {
            name: float(shifted_metrics[name] - original_metrics[name])
            for name in ("auroc", "aupr", "accuracy", "mcc", "f1")
        },
    }


def _conclusion(test_result: Mapping, *, run_count: int) -> dict:
    classification = test_result.get("original_triplet_classification", {})
    comparisons = classification.get("comparisons", {})

    def supported_gain(name: str) -> bool:
        comparison = comparisons.get(name, {})
        metric_delta = comparison.get("metric_delta", {})
        accuracy_ci = comparison.get("peptide_macro_accuracy_gain", {}).get(
            "ci95", [-math.inf, math.inf]
        )
        return (
            metric_delta.get("auroc", -math.inf) > 0
            and metric_delta.get("aupr", -math.inf) > 0
            and accuracy_ci[0] > 0
        )

    stage3_over_stage1 = supported_gain("stage3_pht_vs_stage1_ptcr")
    within_stage3_hla_gain = supported_gain("stage3_pht_vs_stage3_ptcr")
    pair = test_result.get("phla_valid_pairs", {})
    tetrad = test_result.get("tetrad_disjoint", {})
    pair_model = pair.get("models", {}).get("stage3_pht_margin", {})
    pair_delta = pair_model.get("peptide_macro_mean_delta", {}).get(
        "ci95", [-math.inf, math.inf]
    )
    pair_win = pair_model.get("peptide_macro_paired_win_rate", {}).get(
        "ci95", [-math.inf, math.inf]
    )
    hla_sensitive = pair_delta[0] > 0 and pair_win[0] > 0.5
    tetrad_model = tetrad.get("models", {}).get("stage3_pht_margin", {})
    tetrad_d = tetrad_model.get("peptide_macro_d", {}).get(
        "ci95", [-math.inf, math.inf]
    )
    pvalue = tetrad_model.get("sign_flip_pvalue_one_sided", 1.0)
    versus_late = tetrad.get("stage3_pht_minus_late_peptide_macro_d", {}).get(
        "ci95", [-math.inf, math.inf]
    )
    conditional = tetrad_d[0] > 0 and pvalue < 0.05 and versus_late[0] > 0
    if stage3_over_stage1 and within_stage3_hla_gain and hla_sensitive:
        interpretation = "supports_stage3_hla_conditioned_ptcr_improvement"
    elif stage3_over_stage1 and hla_sensitive:
        interpretation = "supports_stage3_over_stage1_with_partial_hla_attribution"
    elif stage3_over_stage1:
        interpretation = "supports_stage3_over_stage1_without_hla_attribution"
    elif hla_sensitive:
        interpretation = "supports_hla_sensitivity_without_ptcr_improvement"
    else:
        interpretation = "does_not_support_stage3_hla_conditioned_improvement"
    return {
        "stage3_pht_outperforms_stage1_ptcr": stage3_over_stage1,
        "stage3_pht_outperforms_stage3_ptcr_ablation": within_stage3_hla_gain,
        "hla_or_phla_sensitivity": hla_sensitive,
        "non_additive_tcr_hla_compatibility": conditional,
        "interpretation": interpretation,
        "preliminary": run_count < 3,
        "run_count": run_count,
        "claim_limit": (
            "Stage3 received target PHT supervision that Stage1 did not; the "
            "comparison therefore combines HLA conditioning, architecture, "
            "fine-tuning, and target-task training effects. Synthetic decoys "
            "are not experimentally validated non-binding triples."
        ),
    }


def analyse_stage7_rows(
    rows: Sequence[Mapping],
    data_audit: Mapping,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    run_count: int = 1,
) -> dict:
    split_rows: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)
    result = {
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "analysis_seed": seed,
        "run_count": run_count,
        "splits": {},
    }
    if "test" in split_rows and "val" not in split_rows:
        raise ValueError(
            "Stage-7 test classification requires validation rows for thresholds"
        )
    validation_classification = None
    validation_thresholds = None
    if "val" in split_rows:
        validation_classification = analyse_original_triplets(
            split_rows["val"],
            thresholds=None,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 90000,
        )
        validation_thresholds = validation_classification.get(
            "selected_thresholds"
        )
    for split_index, split in enumerate(sorted(split_rows)):
        current = split_rows[split]
        split_seed = seed + split_index * 10000
        if split == "val":
            classification = validation_classification
        else:
            classification = analyse_original_triplets(
                current,
                thresholds=validation_thresholds,
                bootstrap_samples=bootstrap_samples,
                seed=split_seed + 90000,
            )
        result["splits"][split] = {
            "shortcut_lookup": data_audit.get("splits", {})
            .get(split, {})
            .get("shortcut_lookup", {}),
            "original_triplet_classification": classification,
            "phla_valid_pairs": analyse_phla_valid_pairs(
                current,
                bootstrap_samples=bootstrap_samples,
                seed=split_seed,
            ),
            "tetrad_disjoint": analyse_tetrads(
                current,
                "tetrad_disjoint",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=split_seed + 100,
            ),
            "tetrad_all": analyse_tetrads(
                current,
                "tetrad_all",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=split_seed + 200,
            ),
            "legacy_hla_shift": analyse_legacy_hla_shift(current),
        }
    if "test" in result["splits"]:
        result["conclusion"] = _conclusion(
            result["splits"]["test"], run_count=run_count
        )
    return result


def average_aligned_runs(runs: Sequence[Sequence[Mapping]]) -> list[dict]:
    """Average score columns after requiring identical derived rows across runs."""

    if not runs:
        raise ValueError("At least one Stage-7 run is required")
    identity_fields = (
        "split",
        "experiment",
        "group_id",
        "role",
        "peptide",
        "tcr",
        "hla",
        "label",
        "source_row",
        "evidence",
        "peptide_group",
    )
    def identity(row):
        return tuple(row[field] for field in identity_fields)

    indexed = [{identity(row): row for row in run} for run in runs]
    reference = set(indexed[0])
    if any(set(current) != reference for current in indexed[1:]):
        raise ValueError("Stage-7 run rows are not aligned")
    averaged = []
    for key in sorted(reference):
        row = {field: value for field, value in zip(identity_fields, key)}
        for column in SCORE_COLUMNS:
            row[column] = float(np.mean([float(current[key][column]) for current in indexed]))
        averaged.append(row)
    return averaged


def render_stage7_report(summary: Mapping, manifest: Mapping) -> str:
    lines = [
        "# PLM Stage 7：Stage 3 与 Stage 1 对比结果",
        "",
        f"- 运行数：{summary.get('run_count', 1)}",
        f"- Fold：{manifest.get('fold')}",
        f"- Seed：{manifest.get('seed')}",
        "- 合成 decoy 为未观测组合，不等同于实验真阴性。",
        "",
    ]
    for split, result in summary.get("splits", {}).items():
        lines.extend([f"## {split}", ""])
        classification = result["original_triplet_classification"]
        pair = result["phla_valid_pairs"]
        tetrad = result["tetrad_disjoint"]
        if classification.get("groups"):
            stage1 = classification["models"]["stage1_ptcr_margin"]["metrics"]
            stage1_late = classification["models"]["stage1_late_margin"][
                "metrics"
            ]
            stage3_ptcr = classification["models"]["stage3_ptcr_margin"][
                "metrics"
            ]
            stage3 = classification["models"]["stage3_pht_margin"]["metrics"]
            gain = classification["comparisons"][
                "stage3_pht_vs_stage1_ptcr"
            ]
            accuracy_gain = gain["peptide_macro_accuracy_gain"]
            lines.extend(
                [
                    f"- 原始三元组：{classification['groups']}；"
                    f"peptides：{classification['peptides']}",
                    "- Stage1 pTCR："
                    f"AUROC={stage1['auroc']:.4f}，AUPR={stage1['aupr']:.4f}，"
                    f"MCC={stage1['mcc']:.4f}",
                    "- Stage1 pHLA+pTCR late fusion："
                    f"AUROC={stage1_late['auroc']:.4f}，"
                    f"AUPR={stage1_late['aupr']:.4f}，"
                    f"MCC={stage1_late['mcc']:.4f}",
                    "- Stage3 pTCR path ablation："
                    f"AUROC={stage3_ptcr['auroc']:.4f}，"
                    f"AUPR={stage3_ptcr['aupr']:.4f}，"
                    f"MCC={stage3_ptcr['mcc']:.4f}",
                    "- Stage3 PHT："
                    f"AUROC={stage3['auroc']:.4f}，AUPR={stage3['aupr']:.4f}，"
                    f"MCC={stage3['mcc']:.4f}",
                    "- Stage3−Stage1 peptide-macro accuracy gain："
                    f"{accuracy_gain['point']:.4f} "
                    f"(95% CI {accuracy_gain['ci95'][0]:.4f}–"
                    f"{accuracy_gain['ci95'][1]:.4f})",
                ]
            )
        lines.append(
            f"- pHLA-valid anchors：{pair.get('groups', 0)}；"
            f"decoys：{pair.get('decoys', 0)}"
        )
        if pair.get("groups"):
            pht = pair["models"]["stage3_pht_margin"]
            lines.append(
                "- Stage3 PHT peptide-macro paired win rate："
                f"{pht['peptide_macro_paired_win_rate']['point']:.4f} "
                "(95% CI "
                f"{pht['peptide_macro_paired_win_rate']['ci95'][0]:.4f}–"
                f"{pht['peptide_macro_paired_win_rate']['ci95'][1]:.4f})"
            )
        lines.append(
            f"- 独立 tetrads：{tetrad.get('groups', 0)}；"
            f"peptides：{tetrad.get('peptides', 0)}"
        )
        if tetrad.get("groups"):
            pht = tetrad["models"]["stage3_pht_margin"]
            lines.append(
                "- Stage3 PHT peptide-macro D："
                f"{pht['peptide_macro_d']['point']:.4f} "
                f"(95% CI {pht['peptide_macro_d']['ci95'][0]:.4f}–"
                f"{pht['peptide_macro_d']['ci95'][1]:.4f}, "
                f"p={pht['sign_flip_pvalue_one_sided']:.6g})"
            )
        lines.append("")
    conclusion = summary.get("conclusion")
    if conclusion:
        lines.extend(
            [
                "## 结论分级",
                "",
                f"- 判定：`{conclusion['interpretation']}`",
                "- Stage3 PHT 优于 Stage1 pTCR："
                f"{conclusion['stage3_pht_outperforms_stage1_ptcr']}",
                "- Stage3 PHT 优于同 checkpoint 的 pTCR 消融："
                f"{conclusion['stage3_pht_outperforms_stage3_ptcr_ablation']}",
                f"- HLA 反事实敏感：{conclusion['hla_or_phla_sensitivity']}",
                f"- 初步结果：{conclusion['preliminary']}",
                f"- 限制：{conclusion['claim_limit']}",
                "",
            ]
        )
    return "\n".join(lines)
