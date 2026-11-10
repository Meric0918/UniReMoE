#!/usr/bin/env python3
"""Evaluate Stage-9 ranking, paired classification, and legacy retention."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from plm_unified.stage9_checkpoint import (
    initialise_from_stage3,
    load_stage3_source,
    load_stage9_deploy_checkpoint,
    load_torch_checkpoint,
)
from plm_unified.stage9_config import (
    apply_stage9_overrides,
    load_stage9_config,
    restore_stage9_checkpoint_config,
)
from plm_unified.stage9_evaluation import evaluate_stage9_pairs
from plm_unified.stage9_model import build_stage9_model
from plm_unified.stage9_pipeline import build_stage9_audit, build_stage9_loaders
from plm_unified.stage9_trainer import stage9_legacy_joint_score
from plm_unified.trainer import resolve_device
from plm_unified.triplet_evaluation import evaluate_stage3_model, select_mcc_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage9.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path)
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--no-source-comparison", action="store_true")
    return parser.parse_args()


def _legacy_metrics(model, loaders, config, device):
    results = {
        task: evaluate_stage3_model(
            model,
            loaders[task],
            task,
            device=device,
            mixed_precision=config.stage9.mixed_precision,
            threshold=0.5,
        ).metrics
        for task in ("phla", "ptcr", "triplet")
    }
    return results, stage9_legacy_joint_score(results)


def _selected_threshold(result):
    samples = result.metrics.samples
    labels = np.concatenate(
        [np.ones(samples, dtype=np.int64), np.zeros(samples, dtype=np.int64)]
    )
    probabilities = np.concatenate(
        [result.positive_probabilities, result.negative_probabilities]
    )
    return select_mcc_threshold(labels, probabilities)


def main() -> None:
    args = parse_args()
    if args.select_threshold and args.split == "test":
        raise ValueError("Threshold selection is forbidden on the Stage-9 test split")
    checkpoint_metadata = load_torch_checkpoint(args.checkpoint)
    saved_config = checkpoint_metadata.get("config", {})
    saved_runtime = saved_config.get("runtime", {})
    saved_stage9 = saved_config.get("stage9", {})
    saved_paths = saved_config.get("paths", {})
    saved_pair_path = None
    if saved_paths.get("data_root"):
        saved_pair_path = (
            Path(saved_paths["data_root"])
            / saved_stage9.get("data_subdir", "data_stage9")
            / saved_stage9.get("pair_filename", "pht_pairs.csv")
        )
    saved_source_path = checkpoint_metadata.get("source_stage3", {}).get("path")
    config = load_stage9_config(args.config)
    restore_stage9_checkpoint_config(config, saved_config)
    config = apply_stage9_overrides(
        config,
        fold=args.fold if args.fold is not None else saved_runtime.get("fold"),
        seed=args.seed if args.seed is not None else saved_runtime.get("seed"),
        device=args.device,
        num_workers=args.num_workers,
        stage3_checkpoint=args.stage3_checkpoint or saved_source_path,
        pair_csv=args.pairs_csv or saved_pair_path,
    )
    for name, path in (
        ("Stage-9 checkpoint", args.checkpoint),
        ("Stage-9 pair CSV", config.pair_path()),
        ("ESM-C checkpoint", config.paths.esmc_model),
        ("TCR-BERT checkpoint", config.paths.tcr_bert_model),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    audit = build_stage9_audit(config).require_clean()
    pair_loaders, _, legacy_val, audit_payload = build_stage9_loaders(config, audit)
    device = resolve_device(config.runtime.device)
    model = build_stage9_model(config, device)
    checkpoint = load_stage9_deploy_checkpoint(model, checkpoint_metadata, config)
    checkpoint = {key: value for key, value in checkpoint.items() if key != "model"}
    del checkpoint_metadata
    gc.collect()
    threshold = float(checkpoint.get("selected_pair_threshold", 0.5))
    result = evaluate_stage9_pairs(
        model,
        pair_loaders[args.split],
        device=device,
        mixed_precision=config.stage9.mixed_precision,
        temperature=config.stage9.temperature,
        threshold=threshold,
        collect_embeddings=args.save_embeddings,
    )
    selected_metrics = None
    if args.select_threshold:
        threshold, selected_metrics = _selected_threshold(result)
    legacy_metrics, legacy_joint = _legacy_metrics(model, legacy_val, config, device)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_type": checkpoint["checkpoint_type"],
        "split": args.split,
        "dataset_fingerprint": audit.dataset_fingerprint,
        "source_stage3": checkpoint.get("source_stage3"),
        "threshold": threshold,
        "threshold_source": "selected_on_validation" if args.select_threshold else "checkpoint_validation",
        "pair_metrics": result.metrics.to_dict(),
        "selected_threshold_metrics": (
            selected_metrics.to_dict() if selected_metrics else None
        ),
        "legacy_validation": {
            "joint_score": legacy_joint,
            "source_joint_score": checkpoint.get("source_legacy_joint"),
            "forgetting": (
                legacy_joint - float(checkpoint["source_legacy_joint"])
                if checkpoint.get("source_legacy_joint") is not None
                else None
            ),
            "tasks": {
                task: metrics.to_dict() for task, metrics in legacy_metrics.items()
            },
        },
        "data_audit": {
            "dataset_fingerprint": audit_payload["dataset_fingerprint"],
            "report": {
                key: value
                for key, value in audit_payload["report"].items()
                if key != "split_manifest"
            },
            "effective_data": audit_payload["effective_data"],
        },
    }
    if not args.no_source_comparison:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        source = load_stage3_source(config)
        source_model = build_stage9_model(config, device)
        initialise_from_stage3(source_model, source)
        source.checkpoint = {}
        gc.collect()
        source_result = evaluate_stage9_pairs(
            source_model,
            pair_loaders[args.split],
            device=device,
            mixed_precision=config.stage9.mixed_precision,
            temperature=config.stage9.temperature,
            threshold=0.5,
        )
        source_probability_gap = (
            source_result.positive_probabilities
            - source_result.negative_probabilities
        )
        stage9_probability_gap = (
            result.positive_probabilities - result.negative_probabilities
        )
        payload["source_stage3_pair_comparison"] = {
            "source_checkpoint_sha256": source.sha256,
            "source_classification": source_result.metrics.classification.to_dict(),
            "source_probability_ranking_accuracy": float(
                np.mean(source_probability_gap > 0)
            ),
            "stage9_probability_ranking_accuracy": float(
                np.mean(stage9_probability_gap > 0)
            ),
            "mean_probability_gap_delta": float(
                stage9_probability_gap.mean() - source_probability_gap.mean()
            ),
        }

    output_dir = args.output_dir or args.checkpoint.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"evaluation_stage9_{args.split}_{checkpoint.get('run_timestamp', 'unknown')}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"predictions_stage9_{args.split}_{checkpoint.get('run_timestamp', 'unknown')}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    records = audit.split_records[args.split]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row", "pair_id", "peptide", "hla",
                "tcr_positive", "tcr_negative", "similarity_positive",
                "similarity_negative", "similarity_gap", "ranking_correct",
                "probability_positive", "probability_negative",
                "prediction_positive", "prediction_negative",
            ]
        )
        for index, record in enumerate(records):
            positive_similarity = float(result.positive_similarities[index])
            negative_similarity = float(result.negative_similarities[index])
            positive_probability = float(result.positive_probabilities[index])
            negative_probability = float(result.negative_probabilities[index])
            writer.writerow(
                [
                    record.source_row, record.pair_id, record.peptide, record.hla,
                    record.tcr_positive, record.tcr_negative, positive_similarity,
                    negative_similarity, positive_similarity - negative_similarity,
                    int(positive_similarity > negative_similarity), positive_probability,
                    negative_probability, int(positive_probability >= threshold),
                    int(negative_probability >= threshold),
                ]
            )
    if args.save_embeddings:
        np.savez_compressed(
            output_dir / f"embeddings_stage9_{args.split}_{checkpoint.get('run_timestamp', 'unknown')}.npz",
            pair_ids=np.asarray(result.pair_ids),
            source_rows=result.source_rows,
            z_ph=result.z_ph,
            z_t_positive=result.z_t_positive,
            z_t_negative=result.z_t_negative,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Predictions: {csv_path}")


if __name__ == "__main__":
    main()
