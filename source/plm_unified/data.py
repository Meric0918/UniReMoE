"""CSV discovery and cache-backed pair datasets."""

from __future__ import annotations

import csv
import hashlib
import random
from array import array
from functools import partial
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .cache import CacheLocator, EmbeddingCache

Task = Literal["phla", "ptcr"]


def stable_group_id(sequence: str) -> int:
    """Return a reproducible non-negative integer id for one entity group."""

    digest = hashlib.blake2b(
        normalise_sequence(sequence).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def normalise_sequence(sequence: str) -> str:
    value = "".join(str(sequence).split()).replace("-", "").upper()
    if not value:
        raise ValueError("Encountered an empty sequence")
    return value


def split_path(
    data_root: str | Path,
    task: Task,
    split: str,
    fold: int,
) -> Path:
    root = Path(data_root)
    dataset_task_dir = root / ("PH" if task == "phla" else "PT")
    if dataset_task_dir.is_dir():
        if split.startswith(("val_", "test_")):
            path = dataset_task_dir / "benchmarks" / f"{split}.csv"
        else:
            path = dataset_task_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        return path

    # Some task-specific runs point data_root directly at a prepared train/val
    # directory, such as dataset/PHT for the Stage PH binary head.
    direct_path = root / f"{split}.csv"
    if direct_path.is_file():
        return direct_path

    task_dir = root / ("data_HLA" if task == "phla" else "data_TCR")
    filename = (
        f"{split}_fold_{fold}.csv"
        if split in {"train", "val"}
        else f"{split}_set.csv"
    )
    path = task_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return path


def task_columns(task: Task) -> tuple[str, str, str]:
    if task == "phla":
        return "peptide", "HLA", "label"
    if task == "ptcr":
        return "peptide", "tcr", "label"
    raise ValueError(f"Unknown task: {task}")


def resolve_task_columns(
    task: Task,
    fieldnames: Sequence[str] | None,
) -> tuple[str, str, str]:
    """Resolve task columns, accepting PHT's ``PH`` label and ``cdr3`` alias."""

    peptide_col, receptor_col, label_col = task_columns(task)
    available = set(fieldnames or ())
    if task == "ptcr" and receptor_col not in available and "cdr3" in available:
        receptor_col = "cdr3"
    if task == "phla" and label_col not in available and "PH" in available:
        label_col = "PH"
    return peptide_col, receptor_col, label_col


def iter_column(path: str | Path, column: str) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found in {path}")
        for row_number, row in enumerate(reader, start=2):
            raw = row.get(column)
            if raw is None or not str(raw).strip():
                raise ValueError(f"Missing {column} at {path}:{row_number}")
            yield normalise_sequence(raw)


def collect_cache_sequences(
    data_root: str | Path,
    fold: int,
    hla_splits: Sequence[str],
    tcr_splits: Sequence[str],
    peptide_min_length: int,
    peptide_max_length: int,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Collect unique peptide/HLA/TCR strings required by configured splits."""

    entities: dict[str, set[str]] = {
        "peptide": set(),
        "hla": set(),
        "tcr": set(),
    }
    stats: dict[str, dict[str, int]] = {}
    for task, splits in (("phla", hla_splits), ("ptcr", tcr_splits)):
        peptide_col, receptor_col, _ = task_columns(task)
        for split in splits:
            path = split_path(data_root, task, split, fold)
            receptor_entity = "hla" if task == "phla" else "tcr"
            split_stats = {
                "total": 0,
                "kept": 0,
                "peptide_too_short": 0,
                "peptide_too_long": 0,
            }
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {peptide_col, receptor_col}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} must contain columns: {sorted(required)}")
                for row_number, row in enumerate(reader, start=2):
                    split_stats["total"] += 1
                    peptide = normalise_sequence(row[peptide_col])
                    if len(peptide) < peptide_min_length:
                        split_stats["peptide_too_short"] += 1
                        continue
                    if len(peptide) > peptide_max_length:
                        split_stats["peptide_too_long"] += 1
                        continue
                    receptor = normalise_sequence(row[receptor_col])
                    entities["peptide"].add(peptide)
                    entities[receptor_entity].add(receptor)
                    split_stats["kept"] += 1
            stats[f"{task}/{split}"] = split_stats
    return entities, stats


class CachedPairDataset(Dataset):
    """Compact pair locations resolved once from a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        task: Task,
        peptide_cache: EmbeddingCache,
        receptor_cache: EmbeddingCache,
        *,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        limit: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.task = task
        values = array("q")
        self.filter_stats = {
            "total": 0,
            "kept": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
        }

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            peptide_col, receptor_col, label_col = resolve_task_columns(
                task, reader.fieldnames
            )
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {sorted(required)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if limit is not None and len(values) // 6 >= limit:
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
                    peptide_location = peptide_cache.resolve(peptide)
                    receptor_location = receptor_cache.resolve(receptor)
                except KeyError as exc:
                    raise KeyError(
                        f"Cache miss while loading {self.csv_path}:{row_number}: {exc}"
                    ) from exc
                try:
                    label = int(row[label_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label at {self.csv_path}:{row_number}"
                    ) from exc
                if label not in {0, 1}:
                    raise ValueError(
                        f"Label must be 0 or 1 at {self.csv_path}:{row_number}"
                    )
                values.extend(
                    (
                        peptide_location.shard,
                        peptide_location.row,
                        receptor_location.shard,
                        receptor_location.row,
                        label,
                        stable_group_id(peptide if task == "ptcr" else receptor),
                    )
                )
                self.filter_stats["kept"] += 1
        self.locations = np.frombuffer(values, dtype=np.int64).reshape(-1, 6).copy()
        self.group_ids = self.locations[:, 5].copy()
        self.labels = self.locations[:, 4].copy()
        if not len(self.locations):
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.locations[index]


class CachedPairCollator:
    """Load one batch with grouped HDF5 reads from both entity caches."""

    def __init__(
        self,
        peptide_cache: EmbeddingCache,
        receptor_cache: EmbeddingCache,
        task: Task | None = None,
    ):
        self.peptide_cache = peptide_cache
        self.receptor_cache = receptor_cache
        self.task = task

    def __call__(self, rows: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        matrix = np.asarray(rows, dtype=np.int64)
        peptide_locations = [
            CacheLocator(int(row[0]), int(row[1]), 0) for row in matrix
        ]
        receptor_locations = [
            CacheLocator(int(row[2]), int(row[3]), 0) for row in matrix
        ]
        peptide_hidden, peptide_mask = self.peptide_cache.get_many(peptide_locations)
        receptor_hidden, receptor_mask = self.receptor_cache.get_many(
            receptor_locations
        )
        return {
            "peptide_hidden": torch.from_numpy(peptide_hidden),
            "peptide_mask": torch.from_numpy(peptide_mask),
            "receptor_hidden": torch.from_numpy(receptor_hidden),
            "receptor_mask": torch.from_numpy(receptor_mask),
            "labels": torch.from_numpy(matrix[:, 4].copy()).long(),
            "group_ids": torch.from_numpy(matrix[:, 5].copy()).long(),
        }


class OnlinePairDataset(Dataset):
    """Small debug dataset that keeps raw sequences for online frozen-PLM encoding."""

    def __init__(
        self,
        csv_path: str | Path,
        task: Task,
        *,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        limit: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.task = task
        self.rows: list[tuple[str, str, int]] = []
        self.group_ids: list[int] = []
        self.filter_stats = {
            "total": 0,
            "kept": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
        }
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            peptide_col, receptor_col, label_col = resolve_task_columns(
                task, reader.fieldnames
            )
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {sorted(required)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if limit is not None and len(self.rows) >= limit:
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
                    raise ValueError(
                        f"Invalid label at {self.csv_path}:{row_number}"
                    ) from exc
                if label not in {0, 1}:
                    raise ValueError(
                        f"Label must be 0 or 1 at {self.csv_path}:{row_number}"
                    )
                self.rows.append((peptide, receptor, label))
                self.group_ids.append(
                    stable_group_id(peptide if task == "ptcr" else receptor)
                )
                self.filter_stats["kept"] += 1
        if not self.rows:
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return self.rows[index]


class ConcatenatedPairDataset(ConcatDataset):
    """Concatenate pair CSV datasets while retaining aggregate filter stats."""

    def __init__(self, datasets: Sequence[OnlinePairDataset]):
        if not datasets:
            raise ValueError("At least one pair dataset is required")
        super().__init__(datasets)
        self.filter_stats = {
            key: sum(dataset.filter_stats[key] for dataset in datasets)
            for key in datasets[0].filter_stats
        }
        self.csv_paths = tuple(dataset.csv_path for dataset in datasets)


def _collate_online_pairs(
    rows: Sequence[tuple[str, str, int]],
    *,
    task: Task | None = None,
) -> dict[str, list[str] | torch.Tensor]:
    resolved_task = task or "ptcr"
    return {
        "peptide_sequences": [row[0] for row in rows],
        "receptor_sequences": [row[1] for row in rows],
        "labels": torch.tensor([row[2] for row in rows], dtype=torch.long),
        "group_ids": torch.tensor(
            [
                stable_group_id(row[0] if resolved_task == "ptcr" else row[1])
                for row in rows
            ],
            dtype=torch.long,
        ),
    }


def make_cached_dataloader(
    csv_path: str | Path,
    task: Task,
    peptide_cache: EmbeddingCache,
    receptor_cache: EmbeddingCache,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    peptide_min_length: int = 8,
    peptide_max_length: int = 15,
    limit: int | None = None,
) -> DataLoader:
    dataset = CachedPairDataset(
        csv_path,
        task,
        peptide_cache,
        receptor_cache,
        peptide_min_length=peptide_min_length,
        peptide_max_length=peptide_max_length,
        limit=limit,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator = CachedPairCollator(
        peptide_cache.worker_reader(),
        receptor_cache.worker_reader(),
        task=task,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
        collate_fn=collator,
        generator=generator,
        worker_init_fn=_seed_worker,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def make_online_dataloader(
    csv_path: str | Path | Sequence[str | Path],
    task: Task,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    peptide_min_length: int,
    peptide_max_length: int,
    limit: int | None = None,
) -> DataLoader:
    paths = [csv_path] if isinstance(csv_path, (str, Path)) else list(csv_path)
    if not paths:
        raise ValueError("At least one CSV path is required")
    datasets = [
        OnlinePairDataset(
            path,
            task,
            peptide_min_length=peptide_min_length,
            peptide_max_length=peptide_max_length,
            limit=limit,
        )
        for path in paths
    ]
    dataset = (
        datasets[0]
        if len(datasets) == 1
        else ConcatenatedPairDataset(datasets)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=partial(_collate_online_pairs, task=task),
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def make_sequential_dataloader(
    loader: DataLoader,
    *,
    batch_size: int | None = None,
) -> DataLoader:
    """Build a deterministic, single-process view of an existing loader.

    Stage-1 best-model UMAP needs to revisit the same training rows after a
    round has finished.  The optimization loaders intentionally shuffle, so a
    separate sequential view keeps dataset indices stable without changing the
    training loader or its random-number state.  A single worker also avoids
    reopening cache readers just for this diagnostic pass.
    """

    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a torch.utils.data.DataLoader")
    effective_batch_size = batch_size or loader.batch_size
    if effective_batch_size is None:
        raise ValueError("loader must expose a batch size")
    return DataLoader(
        loader.dataset,
        batch_size=int(effective_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=loader.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
        worker_init_fn=None,
        persistent_workers=False,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
