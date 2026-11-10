"""Data and validation orchestration helpers for Stage12."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .data import make_online_dataloader, split_path
from .stage12_config import Stage12ExperimentConfig


def fingerprint_paths(paths: list[Path] | tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def selection_paths(config: Stage12ExperimentConfig) -> dict[str, dict[str, Path]]:
    result = {
        "phla": {
            split: split_path(config.paths.data_root, "phla", split, config.runtime.fold)
            for split in config.stage12.ph_selection_splits
        },
        "ptcr": {
            split: split_path(config.paths.data_root, "ptcr", split, config.runtime.fold)
            for split in config.stage12.pt_selection_splits
        },
    }
    if any("test" in split.lower() for values in result.values() for split in values):
        raise ValueError("Stage12 training selection cannot access test splits")
    return result


def make_selection_loaders(
    config: Stage12ExperimentConfig, *, limit: int | None = None
) -> dict[str, dict[str, object]]:
    paths = selection_paths(config)
    result: dict[str, dict[str, object]] = {"phla": {}, "ptcr": {}}
    for task_index, task in enumerate(("phla", "ptcr")):
        for split_index, (split, path) in enumerate(paths[task].items()):
            result[task][split] = make_online_dataloader(
                path,
                task,
                batch_size=config.stage12.eval_batch_size,
                shuffle=False,
                seed=config.runtime.seed + 100 + task_index * 10 + split_index,
                num_workers=config.runtime.num_workers,
                pin_memory=torch.cuda.is_available(),
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=limit,
            )
    return result
