"""Locked final PH/PT/PHT benchmarks and self-contained student restoration."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .stage8_data import Stage8Record
from .triplet_data import PHTMultitaskDataset
from .stage16_checkpoint import (
    checkpoint_sha256,
    load_inference_checkpoint,
    write_json,
    atomic_torch_save,
)
from .stage16_data import PairRows
from .stage16_evaluation import evaluate_pairs, evaluate_joint, autocast
from .stage17_model import build_stage17_model


def resolve_checkpoint(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        index = json.loads(path.read_text())
        target = (path.parent / index["path"]).resolve()
        if checkpoint_sha256(target) != index["sha256"]:
            raise ValueError("Best checkpoint index SHA mismatch")
        return target
    return path


def lock_checkpoints(paths, output):
    output = Path(output)
    if output.exists():
        raise ValueError(
            "A selection lock is immutable; use a new path for a new protocol"
        )
    entries = []
    for path in paths:
        path = resolve_checkpoint(path)
        checkpoint, config = load_inference_checkpoint(path)
        if checkpoint["debug"] or config.debug:
            raise ValueError(
                "Debug/subset checkpoints cannot be locked for final testing"
            )
        if checkpoint["artifact_role"] not in {"initial", "best"}:
            raise ValueError(
                "Only validation-selected or initial controls may be locked"
            )
        entries.append(
            {
                "path": str(path),
                "sha256": checkpoint_sha256(path),
                "contract": config.contract(),
                "stage": config.stage,
                "condition": config.condition,
                "round": checkpoint["round"],
                "selectors": checkpoint["selectors"],
            }
        )
    if not entries:
        raise ValueError("At least one checkpoint must be locked")
    write_json(
        output,
        {
            "version": 1,
            "status": "locked_before_test",
            "checkpoints": entries,
            "threshold": 0.5,
            "strict_threshold": True,
        },
    )
    return entries


def verify_lock(path, lock):
    path = resolve_checkpoint(path)
    manifest = json.loads(Path(lock).read_text())
    if (
        manifest.get("status") != "locked_before_test"
        or manifest.get("threshold") != 0.5
        or not manifest.get("strict_threshold")
    ):
        raise ValueError("Invalid final evaluation selection lock")
    matches = [
        entry for entry in manifest["checkpoints"] if Path(entry["path"]) == path
    ]
    if len(matches) != 1 or matches[0]["sha256"] != checkpoint_sha256(path):
        raise ValueError("Checkpoint absent from lock or modified after selection")
    return path


class CachedInferenceModel(nn.Module):
    """Residue cache is per checkpoint; never reused across student updates."""

    def __init__(self, model, features, indices, precision):
        super().__init__()
        self.binding_model = model.binding_model
        self.features = features
        self.indices = indices
        self.precision = precision

    def set_evaluation(self):
        self.eval()

    def _features(self, entity, sequences):
        device = next(self.parameters()).device
        ids = [self.indices[entity][s] for s in sequences]
        hidden, mask = self.features[entity]
        dtype = (
            torch.float32
            if device.type == "cpu" or self.precision == "none"
            else torch.bfloat16 if self.precision == "bf16" else torch.float16
        )
        return hidden[ids].to(device, dtype=dtype), mask[ids].to(device)

    def forward(self, task, peptides, receptors):
        return self.binding_model(
            task,
            *self._features("peptide", peptides),
            *self._features("hla" if task == "phla" else "tcr", receptors),
        )

    def forward_all(self, peptides, hlas, tcrs):
        return self.binding_model.forward_all(
            *self._features("peptide", peptides),
            *self._features("hla", hlas),
            *self._features("tcr", tcrs),
        )


@torch.inference_mode()
def cache_residues(model, sequences, config, batch_size):
    device = next(model.parameters()).device
    model.set_evaluation()
    features = {}
    indices = {}
    for entity, values in sequences.items():
        values = sorted(values)
        encoder = getattr(model, entity + "_encoder")
        maximum = getattr(config.model, entity + "_max_length")
        hidden = torch.empty(
            (len(values), maximum, encoder.hidden_size), dtype=torch.float32
        )
        masks = torch.empty((len(values), maximum), dtype=torch.bool)
        for start in range(0, len(values), batch_size):
            with autocast(config, device):
                encoded = encoder(values[start : start + batch_size])
            hidden[start : start + len(encoded.hidden)] = encoded.hidden.float().cpu()
            masks[start : start + len(encoded.hidden)] = encoded.mask.cpu()
        features[entity] = (hidden, masks)
        indices[entity] = {s: i for i, s in enumerate(values)}
    return CachedInferenceModel(
        model, features, indices, config.training.mixed_precision
    )


def load_benchmarks(config):
    pair_sets = {}
    joint_sets = {}
    sequences = {e: set() for e in ("peptide", "hla", "tcr")}
    manifest = {}
    for family, task, receptor in (("PH", "phla", "hla"), ("PT", "ptcr", "tcr")):
        paths = sorted((config.paths.data_root / family / "new").glob("test_*.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing {family}/new benchmarks")
        for path in paths:
            key = family + "/" + path.stem
            dataset = PairRows(path, task, config.model, allow_test=True)
            pair_sets[key] = dataset
            manifest[key] = dataset.manifest
            sequences["peptide"].update(p for p, _, _ in dataset.rows)
            sequences[receptor].update(r for _, r, _ in dataset.rows)
    paths = sorted((config.paths.data_root / "PHT/benchmarks").glob("test_*.csv"))
    if not paths:
        raise FileNotFoundError("Missing PHT benchmarks")
    for path in paths:
        dataset = PHTMultitaskDataset(
            path,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            reject_noncanonical_sequences=True,
            allowed_label_types=("111", "100", "010", "000"),
        )
        records = [
            Stage8Record(
                r.peptide,
                r.tcr,
                r.hla,
                r.ph,
                r.pt,
                r.pht,
                r.source_row,
                str(path),
                path.stem,
            )
            for r in dataset.records
        ]
        key = "PHT/" + path.stem
        joint_sets[key] = records
        manifest[key] = {
            "path": str(path),
            "sha256": checkpoint_sha256(path),
            "filter_stats": dataset.filter_stats,
            "samples": len(records),
        }
        for entity in sequences:
            sequences[entity].update(getattr(r, entity) for r in records)
    return pair_sets, joint_sets, sequences, manifest


def main(stage, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Locked Stage{stage} final benchmarks, threshold > 0.5"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    args = parser.parse_args(argv)
    if min(args.encode_batch_size, args.row_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    torch.set_num_threads(2)
    path = verify_lock(args.checkpoint, args.selection_manifest)
    checkpoint, config = load_inference_checkpoint(path, expected_stage=stage)
    if checkpoint["debug"]:
        raise ValueError("Debug models cannot access final test")
    config.runtime.device = args.device
    config.training.eval_batch_size = args.row_batch_size
    pairs, joints, sequences, manifest = load_benchmarks(config)
    model = build_stage17_model(config, torch.device(args.device))
    model.load_state_dict(checkpoint["model"], strict=True)
    cached = cache_residues(model, sequences, config, args.encode_batch_size)
    del model, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = {}
    predictions = {}
    for key, dataset in pairs.items():
        metric, pred = evaluate_pairs(cached, dataset, config)
        results[key] = {dataset.task: metric}
        predictions[key] = {dataset.task: pred}
        print(
            f"{key}: AUROC={metric['auroc']:.6f} AUPR={metric['aupr']:.6f}", flush=True
        )
    for key, records in joints.items():
        metric, pred = evaluate_joint(cached, records, config)
        results[key] = metric
        predictions[key] = pred
    grouped = defaultdict(list)
    for key, heads in results.items():
        for head, metric in heads.items():
            if head in ("phla", "ptcr", "pht"):
                grouped[key.split("/")[0] + "/" + head].append(metric)
    columns = (
        "auroc",
        "aupr",
        "accuracy",
        "mcc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "nll",
        "ece",
    )
    macro = {
        group: {
            m: float(np.mean([v[m] for v in items]))
            for m in (*columns, "performance_avg")
        }
        for group, items in grouped.items()
    }
    report = {
        "checkpoint": str(path),
        "sha256": checkpoint_sha256(path),
        "selection_lock_sha256": checkpoint_sha256(args.selection_manifest),
        "threshold": 0.5,
        "strict_threshold": True,
        "evaluation_device": args.device,
        "autocast_precision": (
            config.training.mixed_precision if args.device == "cuda" else "float32_cpu"
        ),
        "encode_batch_size": args.encode_batch_size,
        "row_batch_size": args.row_batch_size,
        "data_manifest": manifest,
        "results": results,
        "macro": macro,
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix(".json"), report)
    atomic_torch_save(predictions, prefix.with_suffix(".predictions.pt"))
    lines = [
        f"# Stage{stage}: locked final benchmarks",
        "",
        "Threshold: score > 0.5; macro averages weight benchmarks equally.",
        "",
        "| Dataset | Head | " + " | ".join(columns) + " |",
        "|---|---|" + "---:|" * len(columns),
    ]
    for key, heads in results.items():
        for head, metric in heads.items():
            if head in ("phla", "ptcr", "pht"):
                lines.append(
                    f"| {key} | {head} | "
                    + " | ".join(f"{metric[m]:.6f}" for m in columns)
                    + " |"
                )
    lines += ["", "Macro averages:", ""] + [
        f"- {name}: " + ", ".join(f"{k}={v:.6f}" for k, v in values.items())
        for name, values in macro.items()
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


def lock_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lock all selected controls/candidates before opening test data"
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lock_checkpoints(args.checkpoints, args.output)
