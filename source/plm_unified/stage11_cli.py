"""Shared Stage11 CLI configuration and explicit runtime overrides."""

import argparse
from pathlib import Path

from .stage11_config import config_from_dict, load_stage11_config


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/plm_stage11.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"))
    return parser


def config_from_args(args, *, saved=None):
    config = config_from_dict(saved["config"]) if saved else load_stage11_config(args.config)
    for name in ("fold", "seed", "device", "num_workers"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.runtime, name, value)
    for name in ("cache_root", "output_root"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.paths, name, value.expanduser().resolve())
    if args.swanlab_mode is not None:
        config.swanlab.mode = args.swanlab_mode
    for name in ("condition", "batch_size", "contrastive_batch_size", "eval_batch_size", "max_rounds", "min_rounds"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.stage11, name, value)
    config.validate()
    return config
