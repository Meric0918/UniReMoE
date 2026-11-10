#!/usr/bin/env python3
"""Train Stage11 with PH CLIP, PH/T CLIP and PHT classification passes."""

import json
from pathlib import Path

from plm_unified.stage11_checkpoint import load_stage11_checkpoint
from plm_unified.stage11_cli import common_parser, config_from_args
from plm_unified.stage11_data import Stage11Loaders, open_stage11_caches, prepare_stage11_data, require_coverage
from plm_unified.stage11_model import Stage11BindingModel
from plm_unified.stage11_trainer import Stage11Trainer, json_safe
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--condition", choices=("none", "ph", "pht", "both"))
    for name in ("batch-size", "contrastive-batch-size", "eval-batch-size", "max-rounds", "min-rounds"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--run-timestamp")
    args = parser.parse_args(argv)
    saved = load_stage11_checkpoint(args.resume, role="last") if args.resume else None
    if saved and args.run_timestamp:
        parser.error("--resume retains the checkpoint timestamp")
    config = config_from_args(args, saved=saved)
    set_global_seed(config.runtime.seed)
    data = prepare_stage11_data(config)
    caches, metadata = open_stage11_caches(config, expected_metadata=saved["cache_metadata"] if saved else None)
    loaders = None
    try:
        require_coverage(caches, data.sequences())
        loaders = Stage11Loaders(config, data, caches)
        model = Stage11BindingModel(config.model, projection_dim=config.stage11.projection_dim).to(resolve_device(config.runtime.device))
        trainer = Stage11Trainer(model, config, data, loaders, metadata,
                                 run_timestamp=saved["run_timestamp"] if saved else args.run_timestamp)
        if saved:
            trainer.resume(args.resume)
        elif trainer.path("last").exists() or trainer.path("best").exists():
            raise FileExistsError("Stage11 run timestamp already exists; use --resume or a new timestamp")
        audit_dir = trainer.output_dir / f"data_audit_{trainer.run_timestamp}"
        data.audit.write_reports(audit_dir)
        (audit_dir / "stage11_audit.json").write_text(json.dumps(data.report, indent=2, ensure_ascii=False), encoding="utf-8")
        trainer.path("config", "json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        with SwanLabTracker(config, resume_id=saved.get("swanlab_run_id") if saved else None,
                            run_timestamp=trainer.run_timestamp, experiment_suffix=config.stage11.condition) as tracker:
            trainer.tracker = tracker
            summary = trainer.fit()
        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        if loaders:
            loaders.close()
        for cache in caches.values():
            cache.close()


if __name__ == "__main__":
    main()
