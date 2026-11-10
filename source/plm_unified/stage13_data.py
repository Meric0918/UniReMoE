"""Cache-backed pair datasets for the isolated Stage13 graph pipeline."""

from __future__ import annotations

import csv
import random
from array import array
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import CacheLocator, EmbeddingCache
from .data import normalise_sequence, resolve_task_columns
from .stage13_config import Stage13ExperimentConfig
from .stage13_graph import GraphArtifact


def stage13_split_path(data_root: str | Path, task: str, split: str) -> Path:
    if task not in {"phla", "ptcr"}:
        raise ValueError(f"Unknown Stage13 task: {task}")
    directory = Path(data_root) / ("PH" if task == "phla" else "PT")
    if split.startswith(("val_", "test_")):
        path = directory / "benchmarks" / f"{split}.csv"
    else:
        path = directory / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Stage13 split not found: {path}")
    return path


class Stage13PairDataset(Dataset):
    """Compact locations plus graph node IDs resolved once from a CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        task: str,
        peptide_cache: EmbeddingCache,
        receptor_cache: EmbeddingCache,
        graph: GraphArtifact,
        *,
        peptide_min_length: int,
        peptide_max_length: int,
        limit: int | None = None,
    ):
        if task not in {"phla", "ptcr"}:
            raise ValueError(f"Unknown Stage13 task: {task}")
        self.csv_path = Path(csv_path)
        self.task = task
        receptor_kind = "hla" if task == "phla" else "tcr"
        values = array("q")
        self.filter_stats = {
            "total": 0,
            "kept": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
            "graph_unknown_peptide": 0,
            "graph_unknown_receptor": 0,
        }
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            peptide_col, receptor_col, label_col = resolve_task_columns(
                "phla" if task == "phla" else "ptcr", reader.fieldnames
            )
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{self.csv_path} must contain columns: {sorted(required)}")
            for row_number, row in enumerate(reader, start=2):
                if limit is not None and self.filter_stats["kept"] >= limit:
                    break
                self.filter_stats["total"] += 1
                peptide = normalise_sequence(row[peptide_col])
                if len(peptide) < peptide_min_length:
                    self.filter_stats["peptide_too_short"] += 1
                    continue
                if len(peptide) > peptide_max_length:
                    self.filter_stats["peptide_too_long"] += 1
                    continue
                receptor = normalise_sequence(row[receptor_col])
                try:
                    label = int(row[label_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid label at {self.csv_path}:{row_number}") from exc
                if label not in {0, 1}:
                    raise ValueError(f"Label must be 0 or 1 at {self.csv_path}:{row_number}")
                try:
                    peptide_location = peptide_cache.resolve(peptide)
                    receptor_location = receptor_cache.resolve(receptor)
                except KeyError as exc:
                    raise KeyError(f"Cache miss at {self.csv_path}:{row_number}: {exc}") from exc
                peptide_id = graph.lookup("peptide", peptide)
                receptor_id = graph.lookup(receptor_kind, receptor)
                if peptide_id < 0:
                    self.filter_stats["graph_unknown_peptide"] += 1
                if receptor_id < 0:
                    self.filter_stats["graph_unknown_receptor"] += 1
                values.extend(
                    (
                        peptide_location.shard,
                        peptide_location.row,
                        receptor_location.shard,
                        receptor_location.row,
                        label,
                        peptide_id,
                        receptor_id,
                    )
                )
                self.filter_stats["kept"] += 1
        self.locations = np.frombuffer(values, dtype=np.int64).reshape(-1, 7).copy()
        if not len(self.locations):
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.locations[index]


class Stage13PairCollator:
    def __init__(self, peptide_cache: EmbeddingCache, receptor_cache: EmbeddingCache):
        self.peptide_cache = peptide_cache
        self.receptor_cache = receptor_cache

    def __call__(self, rows: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        matrix = np.asarray(rows, dtype=np.int64)
        peptide_locations = [CacheLocator(int(row[0]), int(row[1]), 0) for row in matrix]
        receptor_locations = [CacheLocator(int(row[2]), int(row[3]), 0) for row in matrix]
        peptide_hidden, peptide_mask = self.peptide_cache.get_many(peptide_locations)
        receptor_hidden, receptor_mask = self.receptor_cache.get_many(receptor_locations)
        return {
            "peptide_hidden": torch.from_numpy(peptide_hidden),
            "peptide_mask": torch.from_numpy(peptide_mask),
            "receptor_hidden": torch.from_numpy(receptor_hidden),
            "receptor_mask": torch.from_numpy(receptor_mask),
            "labels": torch.from_numpy(matrix[:, 4].copy()).long(),
            "peptide_node_ids": torch.from_numpy(matrix[:, 5].copy()).long(),
            "receptor_node_ids": torch.from_numpy(matrix[:, 6].copy()).long(),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    config: Stage13ExperimentConfig,
    graph: GraphArtifact,
    task: str,
    path: Path,
    peptide_cache: EmbeddingCache,
    receptor_cache: EmbeddingCache,
    *,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> DataLoader:
    dataset = Stage13PairDataset(
        path,
        task,
        peptide_cache,
        receptor_cache,
        graph,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        limit=limit,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=Stage13PairCollator(
            peptide_cache.worker_reader(), receptor_cache.worker_reader()
        ),
        generator=generator,
        worker_init_fn=_seed_worker,
        multiprocessing_context="spawn" if config.runtime.num_workers > 0 else None,
    )


def open_stage13_caches(config: Stage13ExperimentConfig) -> dict[str, EmbeddingCache]:
    return {
        "peptide": EmbeddingCache(
            config.residue_cache_dir() / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
        ),
        "hla": EmbeddingCache(
            config.residue_cache_dir() / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
        ),
        "tcr": EmbeddingCache(
            config.residue_cache_dir() / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
        ),
    }


def _open_caches_at_root(
    config: Stage13ExperimentConfig, root: Path
) -> dict[str, EmbeddingCache]:
    return {
        "peptide": EmbeddingCache(
            root / f"fold_{config.runtime.fold}" / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
        ),
        "hla": EmbeddingCache(
            root / f"fold_{config.runtime.fold}" / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
        ),
        "tcr": EmbeddingCache(
            root / f"fold_{config.runtime.fold}" / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
        ),
    }


def make_stage13_loaders(
    config: Stage13ExperimentConfig,
    graph: GraphArtifact,
    *,
    limit_train: int | None = None,
    limit_validation: int | None = None,
    include_test: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, dict[str, DataLoader]], dict[str, DataLoader]]:
    caches = open_stage13_caches(config)
    test_caches: dict[str, EmbeddingCache] | None = None
    train: dict[str, DataLoader] = {}
    validation: dict[str, dict[str, DataLoader]] = {"phla": {}, "ptcr": {}}
    test: dict[str, DataLoader] = {}
    try:
        ph_path = Path(config.paths.data_root) / config.data.ph_train
        pt_path = Path(config.paths.data_root) / config.data.pt_train
        train["phla"] = _make_loader(
            config, graph, "phla", ph_path, caches["peptide"], caches["hla"],
            shuffle=True, seed=config.runtime.seed + 11, limit=limit_train
        )
        train["ptcr"] = _make_loader(
            config, graph, "ptcr", pt_path, caches["peptide"], caches["tcr"],
            shuffle=True, seed=config.runtime.seed + 17, limit=limit_train
        )
        for task, splits, receptor in (
            ("phla", config.data.ph_validation_splits, "hla"),
            ("ptcr", config.data.pt_validation_splits, "tcr"),
        ):
            for index, split in enumerate(splits):
                validation[task][split] = _make_loader(
                    config, graph, task, stage13_split_path(config.paths.data_root, task, split),
                    caches["peptide"], caches[receptor], shuffle=False,
                    seed=config.runtime.seed + 101 + index, limit=limit_validation,
                )
        if include_test:
            if config.paths.test_residue_cache_root is not None:
                test_caches = _open_caches_at_root(
                    config, config.paths.test_residue_cache_root
                )
            else:
                test_caches = caches
            for task, splits, receptor in (
                ("phla", config.data.ph_test_splits, "hla"),
                ("ptcr", config.data.pt_test_splits, "tcr"),
            ):
                for index, split in enumerate(splits):
                    test[f"{task}/{split}"] = _make_loader(
                        config, graph, task, stage13_split_path(config.paths.data_root, task, split),
                        test_caches["peptide"], test_caches[receptor], shuffle=False,
                        seed=config.runtime.seed + 201 + index, limit=limit_validation,
                    )
    except Exception:
        for cache in caches.values():
            cache.close()
        if test_caches is not None and test_caches is not caches:
            for cache in test_caches.values():
                cache.close()
        raise
    return train, validation, test
