#!/usr/bin/env python3
"""Evaluate the pTCR branch of a full Stage-3 checkpoint on ``*_set.csv`` files.

The external files under ``data/data_TCR`` contain peptide/TCR pairs rather
than HLA-peptide-TCR triples.  This evaluator therefore uses the pTCR branch
stored in the Stage-3 multitask checkpoint.  It encodes each unique peptide
and TCR once with the checkpoint's fine-tuned PLM encoders, then reuses those
representations for all rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

_CONDA_CUDA_HOME = Path(__file__).resolve().parents[1] / ".." / "miniconda3"

import numpy as np
import torch
import torch.nn as nn

from plm_unified.config import load_config
from plm_unified.data import OnlinePairDataset
from plm_unified.metrics import compute_binary_metrics
from plm_unified.trainer import load_training_checkpoint, resolve_device
from plm_unified.triplet_model import (
    build_progressive_triplet_model,
    load_triplet_model_state,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "plm_triplet.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "data_TCR",
    )
    parser.add_argument(
        "--sets",
        default="covid,independent,triple",
        help="Comma-separated *_set stems.",
    )
    parser.add_argument(
        "--files",
        help="Comma-separated CSV filenames; overrides --sets and supports arbitrary filenames.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _set_cuda_home() -> None:
    if "CUDA_HOME" in os.environ:
        return
    candidate = Path(os.sys.executable).resolve().parent.parent / "cuda"
    if candidate.is_dir():
        os.environ["CUDA_HOME"] = str(candidate)


def _encode_unique(
    label: str,
    encoder,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.empty(
        (len(sequences), max_length, encoder.hidden_size),
        dtype=np.float16,
    )
    mask = np.empty((len(sequences), max_length), dtype=np.uint8)
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(sequences), batch_size),
            start=1,
        ):
            end = min(start + batch_size, len(sequences))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                encoded = encoder(sequences[start:end])
            hidden[start:end] = (
                encoded.hidden.detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
            mask[start:end] = (
                encoded.mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
            )
            if batch_id == 1 or batch_id % 50 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                print(
                    f"[ENCODE {label}] {batch_id:,}/{total_batches:,} "
                    f"sequences={end:,}/{len(sequences):,} "
                    f"rate={end / max(elapsed, 1e-9):.1f}/s "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
    return hidden, mask


def _evaluate_rows(
    model,
    rows: list[tuple[str, str, int]],
    peptide_index: dict[str, int],
    tcr_index: dict[str, int],
    peptide_hidden: np.ndarray,
    peptide_mask: np.ndarray,
    tcr_hidden: np.ndarray,
    tcr_mask: np.ndarray,
    *,
    device: torch.device,
    pair_batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    criterion = nn.CrossEntropyLoss()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    total_batches = (len(rows) + pair_batch_size - 1) // pair_batch_size
    started = time.time()
    with torch.inference_mode():
        for batch_id, start in enumerate(
            range(0, len(rows), pair_batch_size),
            start=1,
        ):
            batch_rows = rows[start : start + pair_batch_size]
            peptide_locations = np.fromiter(
                (peptide_index[row[0]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            tcr_locations = np.fromiter(
                (tcr_index[row[1]] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            batch_labels_cpu = np.fromiter(
                (row[2] for row in batch_rows),
                dtype=np.int64,
                count=len(batch_rows),
            )
            peptide_batch = torch.from_numpy(
                peptide_hidden[peptide_locations]
            ).to(device, dtype=torch.bfloat16, non_blocking=True)
            peptide_mask_batch = torch.from_numpy(
                peptide_mask[peptide_locations]
            ).to(device, dtype=torch.bool, non_blocking=True)
            tcr_batch = torch.from_numpy(tcr_hidden[tcr_locations]).to(
                device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            tcr_mask_batch = torch.from_numpy(tcr_mask[tcr_locations]).to(
                device,
                dtype=torch.bool,
                non_blocking=True,
            )
            batch_labels = torch.from_numpy(batch_labels_cpu).to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model.binding_model(
                    "ptcr",
                    peptide_batch,
                    peptide_mask_batch,
                    tcr_batch,
                    tcr_mask_batch,
                )
                loss = criterion(output.logits, batch_labels)
            batch_samples = len(batch_rows)
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch_labels_cpu)
            if batch_id == 1 or batch_id % 100 == 0 or batch_id == total_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                print(
                    f"[EVAL] {batch_id:,}/{total_batches:,} "
                    f"rows={samples:,}/{len(rows):,} "
                    f"rate={samples / max(elapsed, 1e-9):.1f}/s "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    metrics = compute_binary_metrics(
        y_true,
        y_score,
        loss=loss_sum / samples,
        threshold=threshold,
    ).to_dict()
    metrics["positive_samples"] = int(y_true.sum())
    metrics["negative_samples"] = int((y_true == 0).sum())
    return metrics


def main() -> None:
    _set_cuda_home()
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if args.encode_batch_size < 1 or args.pair_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.files:
        filenames = tuple(
            filename.strip() for filename in args.files.split(",") if filename.strip()
        )
        names = tuple(Path(filename).stem for filename in filenames)
    else:
        names = tuple(name.strip() for name in args.sets.split(",") if name.strip())
        filenames = tuple(f"{name}_set.csv" for name in names)
    if not names:
        raise ValueError("At least one dataset stem is required")

    config = load_config(args.config)
    config.validate(require_models=True)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    dataset_specs = tuple(zip(names, filenames))
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "stage3_multitask_model":
        raise ValueError(
            "Expected stage3_multitask_model, got "
            f"{checkpoint.get('checkpoint_type')!r}"
        )
    if "ptcr" not in checkpoint.get("tasks", []):
        raise ValueError("Checkpoint does not contain the pTCR branch")
    if checkpoint.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("Checkpoint model config does not match the config")
    checkpoint_selected_threshold = checkpoint.get("selected_threshold", 0.5)

    print(f"[INIT] checkpoint={checkpoint_path}", flush=True)
    print(
        f"[INIT] device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"encode_batch={args.encode_batch_size} "
        f"pair_batch={args.pair_batch_size} threshold={args.threshold}",
        flush=True,
    )

    dataset_rows: dict[str, list[tuple[str, str, int]]] = {}
    filter_stats: dict[str, dict[str, int]] = {}
    dataset_paths = {}
    for name, filename in dataset_specs:
        path = data_dir / filename
        dataset_paths[name] = path
        dataset = OnlinePairDataset(
            path,
            "ptcr",
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        dataset_rows[name] = dataset.rows
        filter_stats[name] = dict(dataset.filter_stats)
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

    started = time.time()
    model = build_progressive_triplet_model(config, device)
    load_triplet_model_state(model, checkpoint["model"])
    del checkpoint
    gc.collect()
    model.set_evaluation()
    print(f"[INIT] model loaded in {time.time() - started:.1f}s", flush=True)

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
            "source_file": str(dataset_paths[name].resolve()),
            "filter_stats": filter_stats[name],
            "metrics": metrics,
            "threshold_used": args.threshold,
            "inference": "full_stage3_checkpoint_ptcr_branch",
        }
        print(
            f"[{name}] DONE {json.dumps(metrics, sort_keys=True)}",
            flush=True,
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    report_path = (
        args.output.resolve()
        if args.output
        else checkpoint_path.parent
        / f"evaluation_ptcr_data_TCR_sets_{timestamp}.json"
    )
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": "stage3_multitask_model",
        "branch": "ptcr",
        "data_directory": str(data_dir),
        "datasets": results,
        "unique_sequences": {
            "peptide": len(all_peptides),
            "tcr": len(all_tcrs),
        },
        "threshold_used": args.threshold,
        "checkpoint_selected_threshold": checkpoint_selected_threshold,
        "notes": [
            "The *_set.csv files contain peptide,tcr,label and no HLA, so the retained pTCR branch was evaluated.",
            "The full Stage-3 checkpoint encoders were run once per unique sequence; the Stage-1 embedding cache was not used.",
            "Rows outside the configured peptide length range [8, 15] were excluded.",
            "Metrics use threshold 0.5; no threshold was selected on these evaluation datasets.",
        ],
        "report_timestamp": timestamp,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] report={report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
