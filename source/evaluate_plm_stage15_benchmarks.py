#!/usr/bin/env python3
"""Evaluate Stage15 on all PH/PT benchmark test files at threshold 0.5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_SOURCE_DIR = str(Path(__file__).resolve().parent)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from evaluate_plm_stage8_benchmarks import encode_unique, evaluate_pair
from plm_unified.data import OnlinePairDataset
from plm_unified.encoders import OnlineEncoderBundle
from plm_unified.stage15_config import apply_stage15_overrides, load_stage15_config
from plm_unified.stage15_model import Stage15BindingModel, Stage15TaskBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


class _BindingWrapper:
    def __init__(self, binding_model):
        self.binding_model = binding_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint, config):
    if checkpoint.get("architecture") != "shared_hyper_pmt_pair":
        raise ValueError("Stage15 checkpoint architecture mismatch")
    saved = checkpoint.get("config", {})
    if saved.get("model") != config.to_dict()["model"] or saved.get("fusion") != config.to_dict()["fusion"]:
        raise ValueError("Stage15 checkpoint configuration mismatch")
    checkpoint_type = checkpoint.get("checkpoint_type")
    task = checkpoint.get("task")
    if checkpoint_type == "stage15_task_model":
        model = Stage15TaskBindingModel(config.model, config.fusion, task)
        tasks = [task]
    elif checkpoint_type in {"stage15_training_model", "stage15_joint_model"}:
        model = Stage15BindingModel(config.model, config.fusion)
        tasks = ["phla", "ptcr"]
    else:
        raise ValueError(f"Unsupported Stage15 checkpoint type: {checkpoint_type}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, tasks


def _markdown(report: dict) -> str:
    lines = [
        "# Stage15 PH/PT benchmark results",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- SHA-256: `{report['checkpoint_sha256']}`",
        f"- Fixed threshold: `{report['fixed_threshold']}`",
        f"- Samples: **{report['totals']['samples']}**",
        "",
        "| Dataset | Head | Samples | AUROC | AUPR | Accuracy | MCC | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(report["results"].items()):
        metric = item["metrics"]["fixed_0.5"]
        lines.append(
            f"| `{key}` | `{item['head']}` | {metric['samples']} | "
            f"{metric['auroc']:.6f} | {metric['aupr']:.6f} | "
            f"{metric['accuracy']:.6f} | {metric['mcc']:.6f} | {metric['f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "plm_stage15.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = apply_stage15_overrides(
        load_stage15_config(args.config), device=args.device, num_workers=0
    )
    config.validate(require_models=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    binding_model, tasks = _load_model(checkpoint, config)
    device = resolve_device(config.training.device)
    binding_model.to(device).eval()

    specs = {"phla": ("PH", "hla"), "ptcr": ("PT", "tcr")}
    benchmarks = {}
    peptides: set[str] = set()
    receptors: dict[str, set[str]] = {"phla": set(), "ptcr": set()}
    for task in tasks:
        family, _ = specs[task]
        paths = sorted((config.paths.benchmark_root / family / "benchmarks").glob("test_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No Stage15 {family} benchmark files found")
        receptor_limit = config.model.hla_max_length if task == "phla" else config.model.tcr_max_length
        for path in paths:
            dataset = OnlinePairDataset(
                path, task,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=args.limit,
            )
            rows = dataset.rows
            if any(len(row[1]) > receptor_limit for row in rows):
                raise ValueError(f"{path} contains receptors over the configured limit")
            key = f"{family}/{path.stem}"
            benchmarks[key] = {
                "family": family, "task": task, "head": task, "path": path,
                "rows": rows, "raw_rows": int(dataset.filter_stats["total"]),
                "filter_stats": dict(dataset.filter_stats),
            }
            peptides.update(row[0] for row in rows)
            receptors[task].update(row[1] for row in rows)

    encoders = OnlineEncoderBundle(
        esmc_path=config.paths.esmc_model,
        tcr_bert_path=config.paths.tcr_bert_model,
        peptide_max_length=config.model.peptide_max_length,
        hla_max_length=config.model.hla_max_length,
        tcr_max_length=config.model.tcr_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    saved_cache = checkpoint.get("cache_metadata", {})
    current_fingerprints = {
        "peptide": encoders.esmc.checkpoint_fingerprint,
        "hla": encoders.esmc.checkpoint_fingerprint,
        "tcr": encoders.tcr_bert.checkpoint_fingerprint,
    }
    for entity, current in current_fingerprints.items():
        saved = saved_cache.get(entity, {}).get("model", {}).get(
            "checkpoint_fingerprint"
        )
        if saved and saved != current:
            raise ValueError(f"Stage15 base-PLM fingerprint mismatch for {entity}")
    peptide_values = sorted(peptides, key=lambda value: (-len(value), value))
    encoders.esmc.max_length = config.model.peptide_max_length
    peptide_hidden, peptide_mask = encode_unique(
        "peptide", encoders.esmc, peptide_values, config.model.peptide_max_length,
        args.encode_batch_size, device, config.training.mixed_precision,
    )
    encoded_receptors = {}
    for task in tasks:
        values = sorted(receptors[task], key=lambda value: (-len(value), value))
        if task == "phla":
            encoder = encoders.esmc
            maximum = config.model.hla_max_length
            encoder.max_length = maximum
            label = "HLA"
        else:
            encoder = encoders.tcr_bert
            maximum = config.model.tcr_max_length
            label = "TCR"
        hidden, mask = encode_unique(
            label, encoder, values, maximum, args.encode_batch_size,
            device, config.training.mixed_precision,
        )
        encoded_receptors[task] = (values, hidden, mask)

    peptide_index = {value: index for index, value in enumerate(peptide_values)}
    wrapper = _BindingWrapper(binding_model)
    results = {}
    for key, item in sorted(benchmarks.items()):
        values, hidden, mask = encoded_receptors[item["task"]]
        metrics = evaluate_pair(
            wrapper, item["rows"], peptide_index,
            {value: index for index, value in enumerate(values)},
            peptide_hidden, peptide_mask, hidden, mask, item["task"], device,
            config.training.mixed_precision, args.row_batch_size, 0.5,
        )
        results[key] = {
            "family": item["family"], "benchmark": Path(item["path"]).stem,
            "path": str(Path(item["path"]).resolve()), "head": item["head"],
            "raw_rows": item["raw_rows"], "samples": len(item["rows"]),
            "filter_stats": item["filter_stats"],
            "metrics": {"fixed_0.5": metrics},
        }
    totals = {
        "raw_rows": sum(item["raw_rows"] for item in results.values()),
        "samples": sum(item["samples"] for item in results.values()),
    }
    totals["filtered"] = totals["raw_rows"] - totals["samples"]
    report = {
        "stage_label": "Stage 15", "architecture": "shared_hyper_pmt_pair",
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(args.config.resolve()), "device": str(device),
        "mixed_precision": config.training.mixed_precision,
        "fixed_threshold": 0.5, "results": results, "totals": totals,
        "unique_sequences": {
            "peptide": len(peptide_values),
            "hla": len(receptors["phla"]), "tcr": len(receptors["ptcr"]),
        },
        "outputs": {
            "json": str(args.output_json.resolve()),
            "markdown": str(args.output_markdown.resolve()),
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
