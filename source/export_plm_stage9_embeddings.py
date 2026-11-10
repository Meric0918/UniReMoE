#!/usr/bin/env python3
"""Export Stage-9 PH/TCR embeddings and binding scores for PHT rows."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from plm_unified.data import normalise_sequence
from plm_unified.stage9_checkpoint import (
    load_stage9_deploy_checkpoint,
    load_torch_checkpoint,
)
from plm_unified.stage9_config import (
    apply_stage9_overrides,
    load_stage9_config,
    restore_stage9_checkpoint_config,
)
from plm_unified.stage9_model import build_stage9_model
from plm_unified.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/plm_stage9.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _read_rows(path: Path, config) -> list[dict[str, str | int]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "hla", "tcr"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            peptide = normalise_sequence(row["peptide"])
            hla = normalise_sequence(row["hla"])
            tcr = normalise_sequence(row["tcr"])
            if not config.model.peptide_min_length <= len(peptide) <= config.model.peptide_max_length:
                raise ValueError(f"Invalid peptide length at {path}:{row_number}")
            if not 1 <= len(hla) <= config.model.hla_max_length:
                raise ValueError(f"Invalid HLA length at {path}:{row_number}")
            if not 1 <= len(tcr) <= config.model.tcr_max_length:
                raise ValueError(f"Invalid TCR length at {path}:{row_number}")
            rows.append({
                "source_row": row_number,
                "peptide": peptide,
                "hla": hla,
                "tcr": tcr,
            })
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    checkpoint_metadata = load_torch_checkpoint(args.checkpoint)
    saved_config = checkpoint_metadata.get("config", {})
    saved_runtime = saved_config.get("runtime", {})
    config = load_stage9_config(args.config)
    restore_stage9_checkpoint_config(config, saved_config)
    config = apply_stage9_overrides(
        config,
        fold=args.fold if args.fold is not None else saved_runtime.get("fold"),
        seed=args.seed if args.seed is not None else saved_runtime.get("seed"),
        device=args.device,
    )
    rows = _read_rows(args.input_csv, config)
    device = resolve_device(config.runtime.device)
    model = build_stage9_model(config, device)
    checkpoint = load_stage9_deploy_checkpoint(model, checkpoint_metadata, config)
    checkpoint = {key: value for key, value in checkpoint.items() if key != "model"}
    del checkpoint_metadata
    model.set_evaluation()
    arrays = {name: [] for name in ("z_ph", "z_tcr", "similarity", "probability")}
    use_autocast = device.type == "cuda" and config.stage9.mixed_precision != "none"
    dtype = torch.float16 if config.stage9.mixed_precision == "fp16" else torch.bfloat16
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            context = torch.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()
            with context:
                output = model.encode_triplet(
                    [row["peptide"] for row in batch],
                    [row["hla"] for row in batch],
                    [row["tcr"] for row in batch],
                )
            arrays["z_ph"].append(output.z_ph.float().cpu().numpy())
            arrays["z_tcr"].append(output.z_tcr.float().cpu().numpy())
            arrays["similarity"].append(output.similarity.float().cpu().numpy())
            arrays["probability"].append(
                torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
            )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        source_rows=np.asarray([row["source_row"] for row in rows]),
        peptide=np.asarray([row["peptide"] for row in rows]),
        hla=np.asarray([row["hla"] for row in rows]),
        tcr=np.asarray([row["tcr"] for row in rows]),
        **{name: np.concatenate(values) for name, values in arrays.items()},
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_type": checkpoint["checkpoint_type"],
        "rows": len(rows),
        "projection_dim": config.stage9.projection_dim,
        "selected_pair_threshold": checkpoint.get("selected_pair_threshold"),
        "output": str(args.output_npz.resolve()),
    }
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
