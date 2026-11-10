#!/usr/bin/env python3
"""Evaluate a Stage-5 checkpoint and export PHT predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.config import apply_overrides, load_config
from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage5_model import build_stage5_model
from plm_unified.trainer import load_training_checkpoint, resolve_device
from plm_unified.triplet_data import (
    HlaShiftedTripletDataset,
    build_strict_triplet_datasets,
    make_triplet_dataloader,
)
from plm_unified.triplet_evaluation import (
    evaluate_stage3_model,
    select_mcc_threshold,
)
from plm_unified.triplet_model import load_triplet_model_state
from plm_unified.triplet_trainer import checkpoint_initialization_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "plm_stage5.yaml"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--hla-shuffle", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_predictions(
    path: Path,
    dataset,
    result,
    threshold: float,
) -> None:
    if result.source_rows is None:
        raise ValueError("Prediction result does not contain source rows")
    if not (
        len(dataset) == len(result.probabilities) == len(result.source_rows)
    ):
        raise ValueError("Prediction count does not match the dataset")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_row",
                "peptide",
                "tcr",
                "hla",
                "label",
                "pht_probability",
                "prediction",
            ),
        )
        writer.writeheader()
        for record, source_row, pht_probability in zip(
            dataset.records,
            result.source_rows,
            result.probabilities,
        ):
            if record.source_row != int(source_row):
                raise RuntimeError("Prediction order no longer matches source rows")
            writer.writerow(
                {
                    "source_row": record.source_row,
                    "peptide": record.peptide,
                    "tcr": record.tcr,
                    "hla": record.hla,
                    "label": record.label,
                    "pht_probability": f"{float(pht_probability):.10g}",
                    "prediction": int(pht_probability >= threshold),
                }
            )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.select_threshold and args.split != "val":
        raise ValueError("--select-threshold may only be used with --split val")
    if args.select_threshold and args.threshold is not None:
        raise ValueError("Use either --select-threshold or --threshold, not both")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in {"stage5_training", "stage5_multitask_model"}:
        raise ValueError(
            "Expected a stage5_training or stage5_multitask_model checkpoint"
        )
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Stage-5 checkpoint format")
    saved_config = checkpoint.get("config", {})
    current_config = config.to_dict()
    if saved_config.get("model") != current_config["model"]:
        raise ValueError("Checkpoint model configuration does not match")
    saved_training = saved_config.get("training", {})
    for key in ("fold", "seed"):
        if saved_training.get(key) != current_config["training"][key]:
            raise ValueError(f"Checkpoint {key} does not match evaluation config")
    saved_stage5 = dict(saved_config.get("stage5", {}))
    current_stage5 = dict(current_config["stage5"])
    for legacy_key in (
        "consistency_ramp_epochs",
        "pt_imp_max_weight",
        "pt_imp_margin",
    ):
        saved_stage5.pop(legacy_key, None)
    for runtime_key in ("eval_batch_size", "log_every_optimizer_steps"):
        saved_stage5.pop(runtime_key, None)
        current_stage5.pop(runtime_key, None)
    if saved_stage5 != current_stage5:
        raise ValueError("Checkpoint Stage-5 configuration does not match")

    device = resolve_device(config.training.device)
    model = build_stage5_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get(
            "checkpoint_fingerprint"
        )
        current_fingerprint = current_metadata[entity]["checkpoint_fingerprint"]
        if saved_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(
                f"Checkpoint base-PLM fingerprint mismatch for {entity}"
            )
    model_state = checkpoint.pop("model")
    load_triplet_model_state(model, model_state)
    del model_state
    model.to(device)

    values = config.stage5
    bundle = build_strict_triplet_datasets(
        config.paths.data_root,
        data_subdir=values.data_subdir,
        merged_filename=values.merged_filename,
        train_filename=values.train_filename,
        val_filename=values.val_filename,
        test_filename=values.test_filename,
        verify_manifest=values.verify_merged_manifest,
        strict_split_cleanup=values.strict_split_cleanup,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        hla_max_length=config.model.hla_max_length,
        tcr_max_length=config.model.tcr_max_length,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
        limit=args.limit,
    )
    dataset = bundle.val if args.split == "val" else bundle.test
    loader = make_triplet_dataloader(
        dataset,
        batch_size=values.eval_batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    checkpoint_threshold = float(checkpoint.get("selected_threshold", 0.5))
    evaluation_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else checkpoint_threshold
    )
    if not 0 <= evaluation_threshold <= 1:
        raise ValueError("Checkpoint threshold must be in [0, 1]")
    result = evaluate_stage3_model(
        model,
        loader,
        "triplet",
        device=device,
        mixed_precision=values.mixed_precision,
        threshold=evaluation_threshold,
        collect_source_rows=True,
    )
    if args.select_threshold:
        evaluation_threshold, selected_metrics = select_mcc_threshold(
            result.labels, result.probabilities
        )
        selected_metrics.loss = result.metrics.loss
    else:
        selected_metrics = result.metrics
    default_metrics = compute_binary_metrics(
        result.labels,
        result.probabilities,
        loss=result.metrics.loss,
        threshold=0.5,
    )
    initialization_source = checkpoint_initialization_source(checkpoint)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": checkpoint_type,
        "split": args.split,
        "samples": len(dataset),
        "filter_stats": dataset.filter_stats,
        "manifest_audit": bundle.manifest_audit,
        "checkpoint_threshold": checkpoint_threshold,
        "evaluation_threshold": evaluation_threshold,
        "default_threshold_metrics": default_metrics.to_dict(),
        "selected_threshold_metrics": selected_metrics.to_dict(),
        "initialization_source": initialization_source,
        "validation_metrics": checkpoint.get(
            "validation_metrics", checkpoint.get("best_val_metrics", {})
        ),
    }
    if args.hla_shuffle:
        shuffled_dataset = HlaShiftedTripletDataset(dataset)
        shuffled_loader = make_triplet_dataloader(
            shuffled_dataset,
            batch_size=values.eval_batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        shuffled_result = evaluate_stage3_model(
            model,
            shuffled_loader,
            "triplet",
            device=device,
            mixed_precision=values.mixed_precision,
            threshold=evaluation_threshold,
        )
        payload["hla_shuffle"] = {
            "changed_fraction": shuffled_dataset.changed_fraction,
            "metrics": shuffled_result.metrics.to_dict(),
        }

    output_dir = (
        args.output_dir.resolve() if args.output_dir else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = create_artifact_timestamp()
    json_path = output_dir / timestamped_filename(
        f"evaluation_stage5_triplet_{args.split}.json", timestamp
    )
    _write_json(json_path, payload)
    prediction_path = None
    if not args.no_predictions:
        prediction_path = output_dir / timestamped_filename(
            f"predictions_stage5_triplet_{args.split}.csv", timestamp
        )
        _write_predictions(
            prediction_path,
            dataset,
            result,
            evaluation_threshold,
        )
    print(
        json.dumps(
            {
                "evaluation": str(json_path),
                "predictions": str(prediction_path) if prediction_path else None,
                "threshold": evaluation_threshold,
                "metrics": selected_metrics.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
