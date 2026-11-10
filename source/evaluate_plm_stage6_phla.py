#!/usr/bin/env python3
"""Evaluate Stage-5/Stage-6 checkpoints on PH validation and external sets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.data import make_online_dataloader, normalise_sequence, split_path
from plm_unified.metrics import task_score
from plm_unified.stage6_config import apply_stage6_overrides, load_stage6_config
from plm_unified.stage6_model import build_stage6_model
from plm_unified.trainer import load_training_checkpoint, resolve_device
from plm_unified.triplet_data import (
    build_strict_triplet_datasets,
    make_triplet_dataloader,
)
from plm_unified.triplet_evaluation import evaluate_stage3_model
from plm_unified.triplet_model import load_triplet_model_state


@dataclass(frozen=True)
class PHLARecord:
    sample_id: str
    peptide: str
    hla: str
    label: int
    source_row: int


class AuditedPHLADataset(Dataset):
    def __init__(self, records: list[PHLARecord], audit: dict[str, int]):
        if not records:
            raise ValueError("No PH samples remain after Stage-6 evaluation cleanup")
        self.records = records
        self.filter_stats = audit

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PHLARecord:
        return self.records[index]


def _collate_phla(records: list[PHLARecord]):
    return {
        "peptide_sequences": [record.peptide for record in records],
        "receptor_sequences": [record.hla for record in records],
        "labels": torch.tensor([record.label for record in records], dtype=torch.long),
    }


def _read_pair_labels(
    path: Path,
    *,
    peptide_min_length: int,
    peptide_max_length: int,
) -> dict[tuple[str, str], set[int]]:
    labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "HLA", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            peptide = normalise_sequence(row["peptide"])
            if not peptide_min_length <= len(peptide) <= peptide_max_length:
                continue
            hla = normalise_sequence(row["HLA"])
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0 or 1 at {path}:{row_number}")
            labels[(peptide, hla)].add(label)
    return labels


def build_evaluation_dataset(
    path: str | Path,
    *,
    train_path: str | Path,
    peptide_min_length: int,
    peptide_max_length: int,
    clean_external: bool,
    limit: int | None = None,
) -> AuditedPHLADataset:
    """Remove train overlap, conflicting pairs and duplicate external pairs."""

    source = Path(path).resolve()
    train = Path(train_path).resolve()
    train_pairs = (
        set(
            _read_pair_labels(
                train,
                peptide_min_length=peptide_min_length,
                peptide_max_length=peptide_max_length,
            )
        )
        if clean_external
        else set()
    )
    candidate_labels = _read_pair_labels(
        source,
        peptide_min_length=peptide_min_length,
        peptide_max_length=peptide_max_length,
    )
    ambiguous_pairs = {
        pair for pair, labels in candidate_labels.items() if len(labels) > 1
    }
    audit = {
        "source_rows": 0,
        "eligible_rows": 0,
        "peptide_too_short": 0,
        "peptide_too_long": 0,
        "train_overlap_rows_removed": 0,
        "train_overlap_pairs": 0,
        "ambiguous_rows_removed": 0,
        "ambiguous_pairs": len(ambiguous_pairs) if clean_external else 0,
        "duplicate_rows_removed": 0,
        "kept_rows": 0,
    }
    records: list[PHLARecord] = []
    seen: set[tuple[str, str]] = set()
    overlap_pairs: set[tuple[str, str]] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "HLA", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{source} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            audit["source_rows"] += 1
            peptide = normalise_sequence(row["peptide"])
            if len(peptide) < peptide_min_length:
                audit["peptide_too_short"] += 1
                continue
            if len(peptide) > peptide_max_length:
                audit["peptide_too_long"] += 1
                continue
            hla = normalise_sequence(row["HLA"])
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {source}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0 or 1 at {source}:{row_number}")
            audit["eligible_rows"] += 1
            pair = (peptide, hla)
            if clean_external and pair in train_pairs:
                audit["train_overlap_rows_removed"] += 1
                overlap_pairs.add(pair)
                continue
            if clean_external and pair in ambiguous_pairs:
                audit["ambiguous_rows_removed"] += 1
                continue
            if clean_external and pair in seen:
                audit["duplicate_rows_removed"] += 1
                continue
            seen.add(pair)
            records.append(
                PHLARecord(
                    sample_id=f"{source.name}:{row_number}",
                    peptide=peptide,
                    hla=hla,
                    label=label,
                    source_row=row_number,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    audit["train_overlap_pairs"] = len(overlap_pairs)
    audit["kept_rows"] = len(records)
    return AuditedPHLADataset(records, audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage6.yaml"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("val", "independent", "external", "all"), default="all"
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-clean-external", action="store_true")
    parser.add_argument("--joint-val-metrics", action="store_true")
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


def _write_predictions(path: Path, dataset, result, condition: str) -> None:
    if len(dataset) != len(result.labels):
        raise ValueError("Prediction count does not match PH dataset")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "sample_id",
                "source_row",
                "peptide",
                "HLA",
                "label",
                "phla_probability",
            ),
        )
        writer.writeheader()
        for record, label, probability in zip(
            dataset.records, result.labels, result.probabilities
        ):
            if int(label) != record.label:
                raise RuntimeError("Prediction order no longer matches PH records")
            writer.writerow(
                {
                    "condition": condition,
                    "sample_id": record.sample_id,
                    "source_row": record.source_row,
                    "peptide": record.peptide,
                    "HLA": record.hla,
                    "label": record.label,
                    "phla_probability": f"{float(probability):.10g}",
                }
            )
    temporary.replace(path)


def _checkpoint_identity(checkpoint: dict) -> tuple[str, str]:
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type in {"stage5_training", "stage5_multitask_model"}:
        return "stage5_joint_matched", "stage5"
    if checkpoint_type in {"stage6_training", "stage6_model"}:
        condition = checkpoint.get("condition") or checkpoint.get("config", {}).get(
            "stage6", {}
        ).get("condition")
        if not condition:
            raise ValueError("Stage-6 checkpoint does not identify its condition")
        return str(condition), "stage6"
    raise ValueError("Expected a Stage-5 or Stage-6 checkpoint")


def _joint_validation_metrics(model, config, threshold: float, limit: int | None):
    values = config.stage6
    ptcr_loader = make_online_dataloader(
        split_path(config.paths.data_root, "ptcr", "val", config.training.fold),
        "ptcr",
        batch_size=values.eval_batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        limit=limit,
    )
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
        limit=limit,
    )
    triplet_loader = make_triplet_dataloader(
        bundle.val,
        batch_size=values.eval_batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    results = {}
    for task, loader in (("ptcr", ptcr_loader), ("triplet", triplet_loader)):
        result = evaluate_stage3_model(
            model,
            loader,
            task,
            device=resolve_device(config.training.device),
            mixed_precision=values.mixed_precision,
            threshold=threshold,
        )
        results[task] = {
            "metrics": result.metrics.to_dict(),
            "task_score": task_score(result.metrics),
        }
    return results


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    condition, stage = _checkpoint_identity(checkpoint)
    saved_config = checkpoint.get("config", {})
    config = apply_stage6_overrides(
        load_stage6_config(args.config),
        fold=(
            args.fold
            if args.fold is not None
            else saved_config.get("training", {}).get("fold")
        ),
        seed=(
            args.seed
            if args.seed is not None
            else saved_config.get("training", {}).get("seed")
        ),
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    if saved_config.get("model") != config.to_dict()["model"]:
        raise ValueError("Checkpoint model configuration does not match Stage-6 config")
    device = resolve_device(config.training.device)
    model = build_stage6_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        fingerprint = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        if fingerprint and fingerprint != current_metadata[entity][
            "checkpoint_fingerprint"
        ]:
            raise ValueError(f"Checkpoint base-PLM fingerprint mismatch for {entity}")
    load_triplet_model_state(model, checkpoint["model"])
    model.to(device)
    checkpoint_tasks = set(checkpoint.get("tasks", ("phla", "ptcr", "triplet")))
    splits = (
        ("val", "independent", "external")
        if args.split == "all"
        else (args.split,)
    )
    hla_root = config.paths.data_root / "data_HLA"
    split_paths = {
        "val": hla_root / f"val_fold_{config.training.fold}.csv",
        "independent": hla_root / "independent_set.csv",
        "external": hla_root / "external_set.csv",
    }
    train_path = hla_root / f"train_fold_{config.training.fold}.csv"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / "stage6_phla_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = create_artifact_timestamp()
    outputs = {}
    for split in splits:
        clean_external = split != "val" and not args.no_clean_external
        dataset = build_evaluation_dataset(
            split_paths[split],
            train_path=train_path,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            clean_external=clean_external,
            limit=args.limit,
        )
        generator = torch.Generator().manual_seed(config.training.seed)
        loader = DataLoader(
            dataset,
            batch_size=config.stage6.eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=False,
            collate_fn=_collate_phla,
            generator=generator,
        )
        result = evaluate_stage3_model(
            model,
            loader,
            "phla",
            device=device,
            mixed_precision=config.stage6.mixed_precision,
            threshold=args.threshold,
        )
        payload = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_stage": stage,
            "condition": condition,
            "split": split,
            "source": str(split_paths[split]),
            "train_overlap_source": str(train_path),
            "external_cleanup_enabled": clean_external,
            "dataset_audit": dataset.filter_stats,
            "threshold": args.threshold,
            "metrics": result.metrics.to_dict(),
            "task_score": task_score(result.metrics),
        }
        if (
            split == "val"
            and args.joint_val_metrics
            and condition in {"stage5_joint_matched", "stage6_joint_phla_full"}
            and {"ptcr", "triplet"}.issubset(checkpoint_tasks)
        ):
            payload["joint_retention"] = _joint_validation_metrics(
                model, config, args.threshold, args.limit
            )
        json_path = output_dir / timestamped_filename(
            f"evaluation_{condition}_phla_{split}.json", timestamp
        )
        _write_json(json_path, payload)
        prediction_path = None
        if not args.no_predictions:
            prediction_path = output_dir / timestamped_filename(
                f"predictions_{condition}_phla_{split}.csv", timestamp
            )
            _write_predictions(prediction_path, dataset, result, condition)
        outputs[split] = {
            "evaluation": str(json_path),
            "predictions": str(prediction_path) if prediction_path else None,
            "samples": len(dataset),
            "task_score": task_score(result.metrics),
        }
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
