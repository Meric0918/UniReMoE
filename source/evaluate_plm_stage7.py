#!/usr/bin/env python3
"""Compare Stage 3 with Stage 1 and test HLA-conditioned pTCR gains."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.stage7_analysis import analyse_stage7_rows, render_stage7_report
from plm_unified.stage7_config import apply_stage7_overrides, load_stage7_config
from plm_unified.stage7_counterfactual import build_stage7_counterfactual_bundle
from plm_unified.stage7_inference import (
    SCORE_COLUMNS,
    load_stage7_models,
    score_stage7_members,
)
from plm_unified.trainer import resolve_device, set_global_seed


MEMBER_COLUMNS = (
    "split",
    "experiment",
    "group_id",
    "role",
    "peptide",
    "tcr",
    "hla",
    "label",
    "source_row",
    "evidence",
    "peptide_group",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage7.yaml"
        ),
    )
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-phla-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-ptcr-checkpoint", type=Path, required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--limit-tetrads", type=int)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MEMBER_COLUMNS + SCORE_COLUMNS,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in writer.fieldnames})
    temporary.replace(path)


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("--splits must be a comma-separated subset of val,test")
    if "test" in splits and "val" not in splits:
        raise ValueError("Stage-7 test inference requires val for late-fusion normalisation")
    return splits


def _resolve_output_root(config, override: Path | None) -> Path:
    output = (
        config.paths.output_root.resolve()
        if override is None
        else override.expanduser().resolve()
    )
    project = config.paths.project_root.resolve()
    trained_model = (project / "trained_model").resolve()
    try:
        relative = output.relative_to(trained_model)
    except ValueError:
        return output
    if relative.parts and relative.parts[0] == "PLM_Stage7":
        return output
    raise ValueError("Stage 7 refuses to write into another trained_model stage")


def main() -> None:
    args = parse_args()
    config = apply_stage7_overrides(
        load_stage7_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    for name in ("limit_pairs", "limit_tetrads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    splits = _parse_splits(args.splits)
    set_global_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)

    counterfactual = build_stage7_counterfactual_bundle(
        config,
        splits=splits,
        limit_pairs=args.limit_pairs,
        limit_tetrads=args.limit_tetrads,
    )
    members = [member for split in splits for member in counterfactual.members[split]]
    debug_limits = [
        value for value in (args.limit_pairs, args.limit_tetrads) if value is not None
    ]
    validation_records = counterfactual.triplets.val.records
    if debug_limits:
        validation_records = validation_records[: max(64, *debug_limits)]
        counterfactual.audit["debug_validation_normalisation_rows"] = len(
            validation_records
        )
    models = load_stage7_models(
        config,
        stage3_checkpoint=args.stage3_checkpoint,
        stage1_phla_checkpoint=args.stage1_phla_checkpoint,
        stage1_ptcr_checkpoint=args.stage1_ptcr_checkpoint,
        device=device,
    )
    scored, inference_audit = score_stage7_members(
        config,
        models,
        members,
        validation_records,
        device=device,
    )
    counterfactual.audit["inference"] = inference_audit

    timestamp = create_artifact_timestamp()
    output_root = _resolve_output_root(config, args.output_root)
    run_dir = (
        output_root
        / f"fold_{config.runtime.fold}"
        / f"seed_{config.runtime.seed}"
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = {}
    for split in splits:
        path = run_dir / f"stage7_counterfactual_{split}.csv"
        split_rows = [row for row in scored if row["split"] == split]
        _write_predictions(path, split_rows)
        predictions[split] = str(path.resolve())
    audit_path = run_dir / "stage7_data_audit.json"
    _write_json(audit_path, counterfactual.audit)
    manifest = {
        "format_version": 2,
        "stage": "stage7",
        "comparison": "stage3_vs_stage1",
        "experiment_type": "stage3_hla_conditioned_ptcr_evaluation",
        "run_timestamp": timestamp,
        "run_dir": str(run_dir.resolve()),
        "fold": config.runtime.fold,
        "seed": config.runtime.seed,
        "splits": list(splits),
        "device": str(device),
        "debug_limits": {
            "pairs": args.limit_pairs,
            "tetrads": args.limit_tetrads,
            "is_limited": args.limit_pairs is not None or args.limit_tetrads is not None,
        },
        "config": config.to_dict(),
        "checkpoints": models.checkpoint_manifest,
        "artifacts": {
            "data_audit": str(audit_path.resolve()),
            "predictions": predictions,
        },
        "source_stage_isolation": {
            "training_performed": False,
            "checkpoint_written": False,
            "swanlab_used": False,
            "stage3_files_modified": False,
            "stage5_files_modified": False,
        },
    }
    summary = analyse_stage7_rows(
        scored,
        counterfactual.audit,
        bootstrap_samples=config.stage7.bootstrap_samples,
        permutation_samples=config.stage7.permutation_samples,
        seed=config.runtime.seed,
    )
    summary_path = run_dir / "stage7_summary.json"
    report_path = run_dir / "stage7_report.md"
    manifest_path = run_dir / "stage7_manifest.json"
    manifest["artifacts"].update(
        {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        }
    )
    _write_json(summary_path, summary)
    _write_text(report_path, render_stage7_report(summary, manifest))
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "manifest": str(manifest_path.resolve()),
                "summary": str(summary_path.resolve()),
                "report": str(report_path.resolve()),
                "conclusion": summary.get("conclusion"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
