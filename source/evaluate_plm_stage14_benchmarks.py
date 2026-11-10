#!/usr/bin/env python3
"""Evaluate a Stage14 checkpoint on PH/PT/PHT benchmark test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

_SOURCE_DIR = str(Path(__file__).resolve().parent)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

try:  # Works both as ``python source/...`` and as an imported package.
    from evaluate_plm_stage8_benchmarks import (
        _PHT_HEADS,
        _PHT_LABEL_ATTRIBUTES,
        encode_unique,
        evaluate_pair,
        evaluate_pht,
        load_benchmark_data,
        make_markdown,
    )
except ModuleNotFoundError:  # pragma: no cover - defensive import fallback.
    from source.evaluate_plm_stage8_benchmarks import (
        _PHT_HEADS,
        _PHT_LABEL_ATTRIBUTES,
        encode_unique,
        evaluate_pair,
        evaluate_pht,
        load_benchmark_data,
        make_markdown,
    )
from plm_unified.stage14_checkpoint import (
    checkpoint_sha256,
    load_stage14_checkpoint,
    validate_stage14_for_evaluation,
)
from plm_unified.stage14_config import apply_stage14_overrides, load_stage14_config
from plm_unified.stage14_model import build_progressive_stage14_model
from plm_unified.stage8_evaluation import stage8_joint_score
from plm_unified.trainer import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "plm_stage14.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = apply_stage14_overrides(
        load_stage14_config(args.config), device=args.device, num_workers=0
    )
    checkpoint = load_stage14_checkpoint(checkpoint_path)
    validate_stage14_for_evaluation(checkpoint, config=config)
    device = resolve_device(config.runtime.device)
    benchmarks, all_peptides, all_hlas, all_tcrs = load_benchmark_data(root, config, args.limit)
    model = build_progressive_stage14_model(config, device)
    saved_metadata = checkpoint.get("encoder_metadata", {})
    current_metadata = model.encoder_metadata()
    for entity in ("peptide", "hla", "tcr"):
        saved = saved_metadata.get(entity, {}).get("checkpoint_fingerprint")
        current = current_metadata[entity].get("checkpoint_fingerprint")
        if saved and saved != current:
            raise ValueError(f"Stage14 checkpoint base-PLM fingerprint mismatch for {entity}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_evaluation()
    mixed_precision = config.stage14.mixed_precision
    peptide_hidden, peptide_mask = encode_unique(
        "peptide", model.peptide_encoder, all_peptides, config.model.peptide_max_length,
        args.encode_batch_size, device, mixed_precision,
    )
    hla_hidden, hla_mask = encode_unique(
        "HLA", model.hla_encoder, all_hlas, config.model.hla_max_length,
        args.encode_batch_size, device, mixed_precision,
    )
    tcr_hidden, tcr_mask = encode_unique(
        "TCR", model.tcr_encoder, all_tcrs, config.model.tcr_max_length,
        args.encode_batch_size, device, mixed_precision,
    )
    peptide_index = {value: index for index, value in enumerate(all_peptides)}
    hla_index = {value: index for index, value in enumerate(all_hlas)}
    tcr_index = {value: index for index, value in enumerate(all_tcrs)}
    results: dict[str, dict] = {}
    for key, item in sorted(benchmarks.items()):
        if item["family"] == "PHT":
            continue
        receptor_index = hla_index if item["task"] == "phla" else tcr_index
        receptor_hidden = hla_hidden if item["task"] == "phla" else tcr_hidden
        receptor_mask = hla_mask if item["task"] == "phla" else tcr_mask
        metrics = evaluate_pair(
            model, item["rows"], peptide_index, receptor_index,
            peptide_hidden, peptide_mask, receptor_hidden, receptor_mask,
            item["task"], device, mixed_precision, args.row_batch_size, 0.5,
        )
        results[key] = {
            "family": item["family"], "benchmark": item["benchmark"],
            "path": str(item["path"].resolve()), "head": item["head"],
            "raw_rows": item["raw_rows"], "samples": len(item["rows"]),
            "filter_stats": item["filter_stats"], "metrics": {"fixed_0.5": metrics},
        }
    selected_threshold = float(checkpoint["metrics"]["selected_pht_threshold"])
    for key, item in sorted(benchmarks.items()):
        if item["family"] != "PHT":
            continue
        evaluated = evaluate_pht(
            model, item["records"], peptide_index, hla_index, tcr_index,
            peptide_hidden, peptide_mask, hla_hidden, hla_mask,
            tcr_hidden, tcr_mask, device, mixed_precision, args.row_batch_size,
            {"fixed_0.5": 0.5, "checkpoint_selected": selected_threshold},
            config.stage14.constraint_margin,
        )
        label_counts = {}
        for task, attribute in _PHT_LABEL_ATTRIBUTES.items():
            positive = sum(int(getattr(record, attribute)) for record in item["records"])
            label_counts[task] = {"positive": positive, "negative": len(item["records"]) - positive}
        results[key] = {
            "family": "PHT", "benchmark": item["benchmark"],
            "path": str(item["path"].resolve()), "head": ",".join(_PHT_HEADS),
            "raw_rows": item["raw_rows"], "samples": len(item["records"]),
            "filter_stats": item["filter_stats"], "label_counts": label_counts,
            "metrics": evaluated["metrics"], "constraints": evaluated["constraints"],
            "joint_score_fixed_0.5": stage8_joint_score(
                {task: evaluated["metrics"][task]["fixed_0.5"] for task in _PHT_HEADS},
                config.stage14.selection_weights,
            ),
        }
    totals = {
        "raw_rows": sum(int(item["raw_rows"]) for item in results.values()),
        "samples": sum(int(item["samples"]) for item in results.values()),
    }
    totals["filtered"] = totals["raw_rows"] - totals["samples"]
    report = {
        "stage_label": "Stage 14", "condition": checkpoint.get("condition"),
        "checkpoint": str(checkpoint_path), "config": str(Path(args.config).resolve()),
        "checkpoint_type": checkpoint["checkpoint_type"],
        "format_version": checkpoint["format_version"],
        "artifact_role": checkpoint["artifact_role"],
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "best_round": checkpoint.get("training_state", {}).get("stage14b_round"),
        "best_round_score": checkpoint.get("metrics", {}).get("validation_joint_score"),
        "validation_joint_score": checkpoint.get("metrics", {}).get("validation_joint_score"),
        "checkpoint_selected_threshold": selected_threshold,
        "checkpoint_validation_metrics": checkpoint.get("metrics", {}).get("validation_metrics", {}),
        "device": str(device), "mixed_precision": mixed_precision,
        "encode_batch_size": args.encode_batch_size, "row_batch_size": args.row_batch_size,
        "fixed_threshold": 0.5, "constraint_margin": config.stage14.constraint_margin,
        "results": results, "totals": totals,
        "unique_sequences": {"peptide": len(all_peptides), "hla": len(all_hlas), "tcr": len(all_tcrs)},
        "outputs": {"json": str(args.output_json.resolve()), "markdown": str(args.output_markdown.resolve())},
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_markdown.write_text(make_markdown(report, output_markdown), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
