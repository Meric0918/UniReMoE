#!/usr/bin/env python3
"""Evaluate the pTCR branch of a Stage-5 checkpoint on external pair sets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sibling evaluator and the local ``plm_unified`` package importable both
# when this file is executed directly and when it is imported from the project.
_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from evaluate_plm_triplet_tcr_sets import _encode_unique, _evaluate_rows
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import OnlinePairDataset
from plm_unified.stage5_model import build_stage5_model
from plm_unified.trainer import load_training_checkpoint, resolve_device
from plm_unified.triplet_model import load_triplet_model_state


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "plm_stage5.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "data_TCR" / "test_data_set",
    )
    parser.add_argument(
        "--files",
        help="Comma-separated CSV filenames; default is every CSV in --data-dir.",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _resolve_files(data_dir: Path, value: str | None) -> list[Path]:
    if value:
        paths = [data_dir / item.strip() for item in value.split(",") if item.strip()]
    else:
        paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset file not found: {missing[0]}")
    return paths


def _validate_checkpoint(checkpoint: dict, config, model) -> None:
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in {"stage5_training", "stage5_multitask_model"}:
        raise ValueError(
            "Expected a stage5_training or stage5_multitask_model checkpoint, "
            f"got {checkpoint_type!r}"
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
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved_fingerprint = saved_metadata.get(entity, {}).get(
            "checkpoint_fingerprint"
        )
        current_fingerprint = current_metadata[entity]["checkpoint_fingerprint"]
        if saved_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(f"Checkpoint base-PLM fingerprint mismatch for {entity}")


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if args.encode_batch_size < 1 or args.pair_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    config.validate(require_models=True)
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    dataset_paths = _resolve_files(data_dir, args.files)
    device = resolve_device(config.training.device)

    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    model = build_stage5_model(config, device)
    _validate_checkpoint(checkpoint, config, model)
    model_state = checkpoint.pop("model")
    load_triplet_model_state(model, model_state)
    del model_state
    gc.collect()
    model.set_evaluation()

    print(
        f"[INIT] checkpoint={checkpoint_path} device={device} "
        f"threshold={args.threshold}",
        flush=True,
    )

    dataset_rows = {}
    filter_stats = {}
    names = []
    for path in dataset_paths:
        name = path.stem
        if name in dataset_rows:
            raise ValueError(f"Duplicate dataset name: {name}")
        dataset = OnlinePairDataset(
            path,
            "ptcr",
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        dataset_rows[name] = dataset.rows
        filter_stats[name] = dict(dataset.filter_stats)
        names.append(name)
        print(
            f"[{name}] rows={len(dataset.rows):,} "
            f"filter_stats={dataset.filter_stats}",
            flush=True,
        )

    all_peptides = sorted(
        {row[0] for rows in dataset_rows.values() for row in rows},
        key=lambda value: (-len(value), value),
    )
    all_tcrs = sorted(
        {row[1] for rows in dataset_rows.values() for row in rows},
        key=lambda value: (-len(value), value),
    )
    if any(len(value) > config.model.tcr_max_length for value in all_tcrs):
        raise ValueError("At least one TCR exceeds the configured maximum length")
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}
    print(
        f"[UNIQUE] peptides={len(all_peptides):,} tcrs={len(all_tcrs):,}",
        flush=True,
    )

    peptide_hidden, peptide_mask = _encode_unique(
        "peptide",
        model.peptide_encoder,
        all_peptides,
        config.model.peptide_max_length,
        args.encode_batch_size,
        device,
    )
    tcr_hidden, tcr_mask = _encode_unique(
        "tcr",
        model.tcr_encoder,
        all_tcrs,
        config.model.tcr_max_length,
        args.encode_batch_size,
        device,
    )

    results = {}
    for name in names:
        metrics = _evaluate_rows(
            model,
            dataset_rows[name],
            peptide_index,
            tcr_index,
            peptide_hidden,
            peptide_mask,
            tcr_hidden,
            tcr_mask,
            device=device,
            pair_batch_size=args.pair_batch_size,
            threshold=args.threshold,
        )
        results[name] = {
            "source_file": str((data_dir / f"{name}.csv").resolve()),
            "filter_stats": filter_stats[name],
            "metrics": metrics,
            "threshold_used": args.threshold,
            "inference": "full_stage5_checkpoint_ptcr_branch",
        }
        print(f"[{name}] DONE {json.dumps(metrics, sort_keys=True)}", flush=True)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if args.output:
        report_path = args.output.resolve()
    else:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else checkpoint_path.parent
        )
        report_path = output_dir / (
            f"evaluation_stage5_ptcr_test_data_set_{timestamp}.json"
        )
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "branch": "ptcr",
        "data_directory": str(data_dir),
        "datasets": results,
        "unique_sequences": {
            "peptide": len(all_peptides),
            "tcr": len(all_tcrs),
        },
        "threshold_used": args.threshold,
        "checkpoint_selected_threshold": checkpoint.get("selected_threshold"),
        "notes": [
            "The input CSV files contain peptide,tcr,label and no HLA, so the Stage-5 pTCR branch was evaluated.",
            "The full Stage-5 checkpoint encoders were run once per unique sequence.",
            "Rows outside the configured peptide length range [8, 15] were excluded.",
            "Metrics use threshold 0.5 unless overridden; no threshold was selected on these evaluation datasets.",
        ],
        "report_timestamp": timestamp,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
