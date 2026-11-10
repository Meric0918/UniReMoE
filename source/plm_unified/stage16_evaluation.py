"""Full-split, strict-threshold evaluation without training-state side effects."""

from __future__ import annotations

from contextlib import nullcontext
import time
import warnings
from collections import Counter

import numpy as np
import torch

from .data import stable_group_id
from .metrics import compute_binary_metrics
from .stage16_checkpoint import preserve_training_state
from .stage16_data import pair_loader, joint_loader


def autocast(config, device):
    precision = config.training.mixed_precision
    return (
        torch.autocast(
            "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
        )
        if device.type == "cuda" and precision != "none"
        else nullcontext()
    )


def summarize(labels, probabilities, *, groups=None):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        not len(labels)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Empty/invalid evaluation predictions")
    with warnings.catch_warnings():
        # Four-type diagnostics intentionally contain single-class subgroups.
        warnings.filterwarnings(
            "ignore", message="A single label was found.*", category=UserWarning
        )
        metrics = compute_binary_metrics(
            labels,
            probabilities,
            loss=0.0,
            strict_threshold=True,
            group_ids=(
                np.array([stable_group_id(g) for g in groups])
                if groups is not None
                else None
            ),
        ).to_dict()
    metrics["loss"] = metrics["nll"]
    positive = probabilities > 0.5
    # Keep the project's equal-frequency confidence-ECE, with the SAME strict
    # tie convention as classification rather than metrics.py's legacy >= tie.
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    chunks = np.array_split(np.argsort(clipped, kind="mergesort"), min(15, len(labels)))
    confidence = np.maximum(clipped, 1.0 - clipped)
    metrics["ece"] = sum(
        len(ids)
        / len(labels)
        * abs(
            float(confidence[ids].mean()) - float((positive[ids] == labels[ids]).mean())
        )
        for ids in chunks
    )
    metrics.update(
        {
            "positive_samples": int(labels.sum()),
            "negative_samples": int((1 - labels).sum()),
            "threshold_ties": int(np.sum(probabilities == 0.5)),
            "predicted_positive_fraction": float(positive.mean()),
            "confusion_matrix": {
                "tn": int(np.sum(~positive & (labels == 0))),
                "fp": int(np.sum(positive & (labels == 0))),
                "fn": int(np.sum(~positive & (labels == 1))),
                "tp": int(np.sum(positive & (labels == 1))),
            },
        }
    )
    if groups is not None:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        indices = {}
        for i, g in enumerate(groups):
            indices.setdefault(g, []).append(i)
        aucs, auprs, covered = [], [], 0
        for ids in indices.values():
            y, p = labels[ids], probabilities[ids]
            if len(np.unique(y)) == 2:
                aucs.append(float(roc_auc_score(y, p)))
                precision, recall, _ = precision_recall_curve(y, p)
                auprs.append(float(auc(recall, precision)))
                covered += len(ids)
        metrics.update(
            {
                "group_macro_auroc": float(np.mean(aucs)) if aucs else None,
                "group_macro_aupr": float(np.mean(auprs)) if auprs else None,
                "two_class_group_count": len(aucs),
                "two_class_group_row_coverage": covered / len(labels),
            }
        )
    return metrics


@torch.inference_mode()
def predict_pairs(model, dataset, config):
    loader = pair_loader(dataset, config)
    device = next(model.parameters()).device
    logits = []
    row_ids = []
    input_keys = []
    with preserve_training_state(model, [loader]):
        model.set_evaluation()
        model.requires_grad_(False)
        for batch in loader:
            with autocast(config, device):
                values = model(
                    dataset.task,
                    batch["peptide_sequences"],
                    batch["receptor_sequences"],
                ).logits
            logits.append(values.float().cpu().numpy())
            row_ids.extend(batch["row_ids"])
            input_keys.extend(batch["input_keys"])
    values = np.concatenate(logits)
    probabilities = (
        torch.from_numpy(values).softmax(-1)[:, 1].numpy().astype(np.float64)
    )
    if row_ids != dataset.row_ids or input_keys != dataset.keys:
        raise ValueError("Evaluation row ordering/identity changed")
    return {
        "logits": values,
        "probabilities": probabilities,
        "labels": np.array([r[2] for r in dataset.rows]),
        "row_ids": row_ids,
        "input_keys": input_keys,
        "peptides": [r[0] for r in dataset.rows],
        "receptors": [r[1] for r in dataset.rows],
    }


def evaluate_pairs(model, dataset, config):
    predictions = predict_pairs(model, dataset, config)
    groups = (
        predictions["peptides"] if dataset.task == "ptcr" else predictions["receptors"]
    )
    return (
        summarize(predictions["labels"], predictions["probabilities"], groups=groups),
        predictions,
    )


@torch.inference_mode()
def evaluate_joint(model, records, config):
    loader = joint_loader(records, config)
    device = next(model.parameters()).device
    logits = {h: [] for h in ("phla", "ptcr", "pht")}
    with preserve_training_state(model, [loader]):
        model.set_evaluation()
        model.requires_grad_(False)
        for batch in loader:
            with autocast(config, device):
                output = model.forward_all(
                    batch["peptide_sequences"],
                    batch["hla_sequences"],
                    batch["tcr_sequences"],
                )
            for head in logits:
                logits[head].append(getattr(output, head + "_logits").float().cpu())
    probabilities = {
        h: torch.cat(chunks).softmax(-1)[:, 1].numpy().astype(np.float64)
        for h, chunks in logits.items()
    }
    labels = {
        h: np.array([getattr(r, k) for r in records])
        for h, k in (("phla", "ph"), ("ptcr", "pt"), ("pht", "pht"))
    }
    metrics = {h: summarize(labels[h], probabilities[h]) for h in logits}
    types = np.array([r.label_type for r in records])
    splits = np.array([r.split_type for r in records])
    for name, groups in (("by_type", types), ("by_split_type", splits)):
        metrics[name] = {
            str(g): {
                h: summarize(labels[h][groups == g], probabilities[h][groups == g])
                for h in logits
            }
            for g in np.unique(groups)
        }
    margin = config.training.constraint_margin
    eligible = {
        "hierarchy_positive": types == "111",
        "hierarchy_negative": np.isin(types, ["000", "010"]),
        "pt_positive": types == "111",
        "pt_negative": np.isin(types, ["000", "100"]),
    }
    constraints = {}
    for key, mask in eligible.items():
        diff = (
            probabilities["pht"]
            - probabilities["phla" if key.startswith("hierarchy") else "ptcr"]
            - margin
        )
        constraints[key] = {
            "eligible": int(mask.sum()),
            "violations": int(np.sum((diff > 0) & mask)),
            "rate": float(np.mean(diff[mask] > 0)) if mask.any() else None,
            "hinge": (
                float(np.mean(np.maximum(diff[mask], 0) ** 2)) if mask.any() else None
            ),
        }
    metrics["constraints"] = constraints
    return metrics, {
        "probabilities": probabilities,
        "labels": labels,
        "types": types,
        "split_types": splits,
        "peptides": [r.peptide for r in records],
        "tcrs": [r.tcr for r in records],
        "row_ids": [f"{r.source_file}:{r.source_row}" for r in records],
    }


def evaluate_all(model, data, config):
    start = time.monotonic()
    pt, predictions = {}, {}
    for split, dataset in data["pt_val"].items():
        pt[split], predictions[split] = evaluate_pairs(model, dataset, config)
    if config.stage == 18:
        frequencies = Counter(p for p, _, _ in data["pairs"]["ptcr"].rows)
        threshold = float(np.median(list(frequencies.values())))
        for split, pred in predictions.items():
            counts = np.array([frequencies[p] for p in pred["peptides"]])
            masks = {
                "unseen": counts == 0,
                "rare": (counts > 0) & (counts <= threshold),
                "frequent": counts > threshold,
            }
            pt[split]["training_frequency_strata"] = {
                "threshold": threshold,
                "definition": "train-only median peptide frequency",
                "metrics": {
                    k: summarize(pred["labels"][mask], pred["probabilities"][mask])
                    for k, mask in masks.items()
                    if mask.any()
                },
            }
    pht, pht_predictions = evaluate_joint(model, data["joint_val"], config)
    result = {
        "pt_independent": pt,
        "pht_val": pht,
        "evaluation_seconds": time.monotonic() - start,
    }
    if config.training.independent_ph_evaluation:
        result["ph_independent"] = {
            s: evaluate_pairs(model, d, config)[0] for s, d in data["ph_val"].items()
        }
    return result, {"pt_independent": predictions, "pht_val": pht_predictions}
