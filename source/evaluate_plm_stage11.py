#!/usr/bin/env python3
"""Evaluate a selected Stage11 model on PHT main and unseen-entity test sets."""

import json
from pathlib import Path

from plm_unified.stage11_checkpoint import load_stage11_checkpoint
from plm_unified.stage11_cli import common_parser, config_from_args
from plm_unified.stage11_data import (classification_loader, open_stage11_caches, prepare_stage11_data,
                                    read_stage11_test, require_coverage, sequences_for_records, test_paths)
from plm_unified.stage11_evaluation import evaluate_stage11_model, write_predictions
from plm_unified.stage11_model import Stage11BindingModel
from plm_unified.stage11_trainer import json_safe
from plm_unified.trainer import resolve_device


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test", "benchmarks", "all"), default="all")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    saved = load_stage11_checkpoint(args.checkpoint)
    config = config_from_args(args, saved=saved)
    data = prepare_stage11_data(config)
    if data.fingerprint != saved["training_contract"]["data_fingerprint"]:
        raise ValueError("Evaluation data audit differs from the checkpoint training data")
    train_keys = {r.key for r in data.audit.train_records}
    datasets = {}
    if args.split == "val":
        datasets["val"] = (data.audit.val_records, data.audit.report["validation"], "train")
    else:
        for name, path in test_paths(config).items():
            if args.split == "test" and name != "test" or args.split == "benchmarks" and name == "test":
                continue
            records, report = read_stage11_test(path, config)
            datasets[name] = (records, report, "test")
    output = args.report_dir or config.run_dir() / f"evaluation_{saved['run_timestamp']}_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.runtime.device)
    model = Stage11BindingModel(config.model, projection_dim=config.stage11.projection_dim).to(device)
    model.load_state_dict(saved["model"], strict=True)
    results = {}
    scope = "train" if args.split == "val" else "test"
    caches, _ = open_stage11_caches(config, scope=scope, expected_metadata=saved["cache_metadata"])
    try:
        for name, (records, report, _) in datasets.items():
            require_coverage(caches, sequences_for_records(records))
            loader = classification_loader(config, records, caches)
            try:
                evaluation = evaluate_stage11_model(model, loader, device=device, precision=config.stage11.mixed_precision)
            finally:
                for cache in loader.collate_fn.caches.values():
                    cache.close()
            write_predictions(output / f"{name}_predictions.csv", evaluation, records)
            overlap = sum(r.key in train_keys for r in records)
            results[name] = {"metrics": evaluation.metrics, "score": evaluation.score,
                             "audit": report, "training_triple_overlap": overlap}
            print(f"{name}: AUROC={evaluation.metrics['auroc']:.6f} AUPR={evaluation.metrics['aupr']:.6f}", flush=True)
    finally:
        for cache in caches.values():
            cache.close()
    report = {"checkpoint": str(args.checkpoint.resolve()), "round": saved["round"], "threshold": 0.5,
              "condition": config.stage11.condition, "results": results}
    (output / "summary.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage11 PHT evaluation", "", f"Checkpoint: `{args.checkpoint}`", "", "Threshold: 0.5", "",
             "| Split | Samples | AUROC | AUPR | MCC | F1 | Train overlap |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        m = result["metrics"]
        lines.append(f"| {name} | {m['samples']} | {m['auroc']:.6f} | {m['aupr']:.6f} | {m['mcc']:.6f} | {m['f1']:.6f} | {result['training_triple_overlap']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
