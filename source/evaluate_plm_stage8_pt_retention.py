"""Final fixed-threshold benchmark evaluation of a locked retention model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

# Reuse the original Stage-8 row inference and unique-sequence encoding.
from evaluate_plm_stage8_benchmarks import (
    encode_unique, evaluate_pair, evaluate_pht, fixed_metric_rows,
    load_benchmark_data, _macro_rows,
)
from train_plm_stage8_pt_retention import _CUDA_DIRECTORY  # CUDA loader setup
from plm_unified.stage8_checkpoint import checkpoint_sha256, load_stage8_v2_checkpoint
from plm_unified.stage8_config import load_stage8_config
from plm_unified.stage8_model import build_progressive_stage8_model


def restore_retention_model(checkpoint, device):
    if checkpoint.get("checkpoint_type") != "stage8_pt_retention_frozen_plm" or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a frozen-PLM PT retention checkpoint")
    plan = checkpoint["plan"]
    warmup_path = Path(plan["warmup_checkpoint"])
    if checkpoint_sha256(warmup_path) != plan["warmup_sha256"]:
        raise ValueError("Frozen encoder warmup SHA256 has changed")
    config = load_stage8_config(plan["base_config"]["source_path"])
    if asdict(config.model) != plan["base_config"]["model"]:
        raise ValueError("Base model configuration has changed")
    config.stage8.mixed_precision = plan["base_config"]["stage8"]["mixed_precision"]
    config.stage8.constraint_margin = plan["base_config"]["stage8"]["constraint_margin"]
    warmup = load_stage8_v2_checkpoint(warmup_path, artifact_role="warmup")
    model = build_progressive_stage8_model(config, device)
    model.load_state_dict(warmup["model"], strict=True)
    model.binding_model.load_state_dict(checkpoint["binding_model"], strict=True)
    model.set_evaluation()
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    args = parser.parse_args()
    if min(args.encode_batch_size, args.row_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("artifact_role") != "best" or checkpoint.get("metrics", {}).get("guard_failures"):
        raise ValueError("Final evaluation requires a guard-passing, validation-selected checkpoint")
    device = torch.device(args.device)
    model, config = restore_retention_model(checkpoint, device)
    root = config.paths.project_root
    data_dirs = {"PH": config.paths.data_root / "PH/new", "PT": config.paths.data_root / "PT/new",
                 "PHT": config.paths.data_root / "PHT/benchmarks"}
    benchmarks, peptides, hlas, tcrs = load_benchmark_data(
        root, config, ph_data_dir=data_dirs["PH"], pt_data_dir=data_dirs["PT"], pht_data_dir=data_dirs["PHT"],
    )
    sequences = {"peptide": peptides, "hla": hlas, "tcr": tcrs}
    indices, features = {}, {}
    precision = config.stage8.mixed_precision
    for entity, seqs in sequences.items():
        indices[entity] = {s: i for i, s in enumerate(seqs)}
        features[entity] = encode_unique(
            entity, getattr(model, f"{entity}_encoder"), seqs,
            getattr(config.model, f"{entity}_max_length"), args.encode_batch_size, device, precision,
        )
    results = {}
    for key, item in sorted(benchmarks.items()):
        summary = {k: item[k] for k in ("family", "benchmark", "head", "raw_rows") if k in item}
        if item["family"] == "PHT":
            metrics = evaluate_pht(
                model, item["records"], indices["peptide"], indices["hla"], indices["tcr"],
                *features["peptide"], *features["hla"], *features["tcr"],
                device, precision, args.row_batch_size, {"fixed_0.5": 0.5}, config.stage8.constraint_margin,
            )
            summary.update(metrics)
        else:
            receptor = "hla" if item["task"] == "phla" else "tcr"
            metrics = evaluate_pair(
                model, item["rows"], indices["peptide"], indices[receptor],
                *features["peptide"], *features[receptor], item["task"], device, precision, args.row_batch_size, 0.5,
            )
            summary["metrics"] = {"fixed_0.5": metrics}
        results[key] = summary
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_type": checkpoint["checkpoint_type"], "training_plan": checkpoint["plan"],
        "validation_selection": checkpoint["metrics"], "results": results,
        "data_directories": {k: str(v) for k, v in data_dirs.items()},
        "fixed_threshold": 0.5, "mixed_precision": precision,
    }
    columns = ("auroc", "aupr", "accuracy", "mcc", "f1", "precision", "recall", "specificity")
    lines = ["# PT retention：锁定模型的最终测试", "",
             "固定阈值 0.5；复用原 Stage 8 benchmark 推理和指标实现。没有在 test 上选择阈值。", "",
             "| Benchmark | Head | " + " | ".join(columns) + " |",
             "|---|---|" + "---:|" * len(columns)]
    for key, head, metric in fixed_metric_rows(report):
        lines.append(f"| {key} | {head} | " + " | ".join(f"{metric[c]:.6f}" for c in columns) + " |")
    lines.extend(["", "各 benchmark 等权宏平均（不是 pooled metric）：", ""])
    for (family, head), values in _macro_rows(report).items():
        lines.append(f"- {family}/{head}: " + ", ".join(f"{k}={v:.6f}" for k, v in values.items()))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
