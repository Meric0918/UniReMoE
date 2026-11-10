#!/usr/bin/env python3
"""Evaluate only the PHT head on all PHT benchmark test splits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

import torch
from torch.utils.data import DataLoader

from plm_unified.metrics import compute_binary_metrics
from plm_unified.stage_PHT_checkpoint import load_stage_PHT_checkpoint, restore_stage_PHT_model
from plm_unified.stage_PHT_config import apply_stage_PHT_overrides, load_stage_PHT_config
from plm_unified.stage_adapter_data import collate_stage_adapter as collate_stage_PHT
from plm_unified.stage_adapter_evaluation import evaluate_stage_adapter as evaluate_stage_PHT
from plm_unified.stage_PHT_model import build_stage_PHT_model
from plm_unified.trainer import resolve_device
from plm_unified.triplet_data import PHTMultitaskDataset


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/plm_stage_PHT.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=root / "dataset/PHT/benchmarks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = apply_stage_PHT_overrides(
        load_stage_PHT_config(args.config),
        device=args.device, num_workers=args.num_workers,
    )
    checkpoint = load_stage_PHT_checkpoint(args.checkpoint)
    device = resolve_device(config.runtime.device)
    model = build_stage_PHT_model(config, device)
    restore_stage_PHT_model(checkpoint, model)
    selected_threshold = float(checkpoint["metrics"]["selected_pht_threshold"])
    results = {}
    for path in sorted(args.benchmark_dir.glob("test_*.csv")):
        dataset = PHTMultitaskDataset(
            path,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            reject_noncanonical_sequences=config.stage_PHT.reject_noncanonical_sequences,
        )
        if args.limit is not None:
            dataset.records = dataset.records[:args.limit]
        loader = DataLoader(
            dataset, batch_size=config.stage_PHT.eval_batch_size,
            shuffle=False, num_workers=config.runtime.num_workers,
            pin_memory=torch.cuda.is_available(), collate_fn=collate_stage_PHT,
        )
        evaluation = evaluate_stage_PHT(
            model, loader, device=device,
            mixed_precision=config.stage_PHT.mixed_precision,
        )
        selected = compute_binary_metrics(
            evaluation.labels, evaluation.probabilities,
            loss=evaluation.metrics.loss, threshold=selected_threshold,
        )
        results[path.stem] = {
            "samples": evaluation.metrics.samples,
            "fixed_0.5": evaluation.metrics.to_dict(),
            "validation_threshold": selected_threshold,
            "at_validation_threshold": selected.to_dict(),
        }
    if not results:
        raise FileNotFoundError(f"No test_*.csv files found in {args.benchmark_dir}")
    payload = {
        "architecture": "stage_PHT",
        "checkpoint": str(args.checkpoint.resolve()),
        "head": "pht",
        "benchmarks": results,
    }
    payload = _json_safe(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
