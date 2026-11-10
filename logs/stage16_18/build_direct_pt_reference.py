#!/usr/bin/env python3
"""Build a full-split PT teacher cache from direct pair-level inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

import train_plm_stage18  # noqa: F401  # configure local CUDA loading first
import numpy as np
import torch

from plm_unified.stage16_data import audit_experiment_data
from plm_unified.stage16_evaluation import predict_pairs
from plm_unified.stage16_pipeline import configured, parse_args, validate_sources
from plm_unified.stage18_teacher import build_teacher, teacher_identity
from plm_unified.stage16_checkpoint import (
    atomic_torch_save,
    checkpoint_sha256,
    digest,
    write_json,
)


def main() -> None:
    config_path = ROOT / "configs/plm_stage18.yaml"
    args = parse_args(18, ["--config", str(config_path)])
    config = configured(args, 18)
    torch.set_num_threads(2)
    data = audit_experiment_data(config)
    validate_sources(config, data)
    dataset = data["pt_val"]["val_unseen_tcr"]
    identity = teacher_identity(config, dataset, "reference")
    path = config.paths.cache_root / "reference" / (digest(identity) + ".pt")
    if path.is_file():
        print(f"Existing reference cache: {path}", flush=True)
        return

    print(
        f"Direct teacher inference: {dataset.path.name}, rows={len(dataset)}",
        flush=True,
    )
    teacher = build_teacher(config, torch.device(config.runtime.device))
    predictions = predict_pairs(teacher, dataset, config)

    # Preserve each full prediction batch when making the independent check.
    # Direct subset inference can otherwise change BF16 padding and batch layout.
    rng = np.random.default_rng(config.runtime.seed)
    sampled = sorted(
        rng.choice(len(dataset), size=min(1024, len(dataset)), replace=False).tolist()
    )
    batch_size = config.training.eval_batch_size
    check_indices = sorted(
        {
            j
            for i in sampled
            for j in range(
                (i // batch_size) * batch_size,
                min(((i // batch_size) + 1) * batch_size, len(dataset)),
            )
        }
    )
    check_dataset = dataset.subset(check_indices)
    check = predict_pairs(teacher, check_dataset, config)
    check_by_row = {
        row_id: logits for row_id, logits in zip(check["row_ids"], check["logits"])
    }
    full_by_row = {
        row_id: logits
        for row_id, logits in zip(predictions["row_ids"], predictions["logits"])
    }
    compared_rows = [dataset.row_ids[i] for i in sampled]
    expected = np.stack([full_by_row[row_id] for row_id in compared_rows])
    observed = np.stack([check_by_row[row_id] for row_id in compared_rows])
    tolerance = 0.1 if config.training.mixed_precision != "none" else 1e-4
    error = float(np.max(np.abs(expected - observed)))
    if not np.allclose(observed, expected, atol=tolerance, rtol=tolerance):
        raise ValueError(f"Direct teacher cache check failed: max logit error={error}")

    verification = {
        "samples": len(sampled),
        "row_ids": compared_rows,
        "batch_preserving_rows": len(check_indices),
        "max_abs_logit_error": error,
        "atol": tolerance,
        "rtol": tolerance,
        "method": "direct_full_split_teacher_forward_same_pair_batches_v1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "identity": identity,
            "row_ids": dataset.row_ids,
            "input_keys": dataset.keys,
            "logits": torch.from_numpy(predictions["logits"]),
            "verification": verification,
            "generation_method": "direct_full_split_teacher_forward_v1",
        },
        path,
    )
    sha256 = checkpoint_sha256(path)
    write_json(
        path.with_suffix(".json"),
        {
            "identity": identity,
            "verification": verification,
            "generation_method": "direct_full_split_teacher_forward_v1",
            "sha256": sha256,
        },
    )
    print(
        f"DIRECT CACHE PASSED: rows={len(dataset)}, max_error={error}, sha256={sha256}, path={path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
