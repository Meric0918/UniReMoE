"""Strict raw-sequence datasets for HLA-peptide-TCR stage-3 training."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import normalise_sequence

TripletKey = tuple[str, str, str]
_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
PHT_LABEL_TYPES = ("111", "100", "010", "000")


def _normalise_csv_paths(
    csv_path: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(csv_path, (str, Path)):
        paths = (Path(csv_path),)
    else:
        paths = tuple(Path(path) for path in csv_path)
    if not paths:
        raise ValueError("At least one PHT CSV path is required")
    return paths


def _validate_pht_label_types(
    label_types: Sequence[str] | None,
    *,
    parameter_name: str,
) -> tuple[str, ...]:
    if label_types is None:
        return PHT_LABEL_TYPES
    if isinstance(label_types, str):
        raise ValueError(f"{parameter_name} must be a sequence of label types")
    selected = tuple(label_types)
    if not selected:
        raise ValueError(f"{parameter_name} cannot be empty")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{parameter_name} cannot contain duplicates")
    unsupported = [kind for kind in selected if kind not in PHT_LABEL_TYPES]
    if unsupported:
        raise ValueError(
            f"{parameter_name} contains unsupported label types: "
            + ", ".join(map(str, unsupported))
        )
    return selected


@dataclass(frozen=True)
class TripletRecord:
    peptide: str
    tcr: str
    hla: str
    label: int
    source_row: int

    @property
    def key(self) -> TripletKey:
        return self.peptide, self.tcr, self.hla


@dataclass(frozen=True)
class PHTMultitaskRecord:
    peptide: str
    tcr: str
    hla: str
    ph: int
    pt: int
    pht: int
    source_row: int

    @property
    def key(self) -> TripletKey:
        return self.peptide, self.tcr, self.hla

    @property
    def label_type(self) -> str:
        return f"{self.ph}{self.pt}{self.pht}"


class PHTMultitaskDataset(Dataset):
    """Filtered PH/PT/PHT rows shared by all three Stage-3 task views."""

    def __init__(
        self,
        csv_path: str | Path | Sequence[str | Path],
        *,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        hla_max_length: int = 34,
        tcr_max_length: int = 34,
        reject_noncanonical_sequences: bool = True,
        limit_per_type: int | None = None,
        allowed_label_types: Sequence[str] | None = None,
    ):
        self.csv_paths = _normalise_csv_paths(csv_path)
        # Keep the singular attribute for callers that use the legacy loader.
        self.csv_path = self.csv_paths[0]
        self.allowed_label_types = _validate_pht_label_types(
            allowed_label_types,
            parameter_name="allowed_label_types",
        )
        if limit_per_type is not None and limit_per_type < 1:
            raise ValueError("limit_per_type must be positive")
        self.records: list[PHTMultitaskRecord] = []
        self.type_indices: dict[str, list[int]] = {
            kind: [] for kind in PHT_LABEL_TYPES
        }
        self.filter_stats: dict[str, int] = {
            "total": 0,
            "returned": 0,
            "same_label_duplicates": 0,
            **{f"type_{kind}": 0 for kind in PHT_LABEL_TYPES},
            **{f"excluded_type_{kind}": 0 for kind in PHT_LABEL_TYPES},
            "invalid_rows": 0,
            "invalid_peptide_length": 0,
            "invalid_hla_length": 0,
            "invalid_tcr_length": 0,
            "invalid_noncanonical": 0,
            "invalid_empty_sequence": 0,
        }
        digest = hashlib.sha256()
        digest.update(
            ("allowed_label_types:" + ",".join(self.allowed_label_types) + "\n")
            .encode("utf-8")
        )
        labels_by_key: dict[TripletKey, tuple[int, int, int]] = {}
        source_row_offset = 0
        for source_path in self.csv_paths:
            digest.update((f"source_path:{source_path}\n").encode("utf-8"))
            file_rows = 0
            with source_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"peptide", "HLA", "tcr", "PH", "PT", "PHT"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        f"{source_path} must contain columns: {sorted(required)}"
                    )
                for row_number, row in enumerate(reader, start=2):
                    file_rows = row_number - 1
                    self.filter_stats["total"] += 1
                    source_row = source_row_offset + row_number
                    try:
                        ph, pt, pht = (
                            int(row[name]) for name in ("PH", "PT", "PHT")
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid PH/PT/PHT labels at {source_path}:{row_number}"
                        ) from exc
                    if any(value not in {0, 1} for value in (ph, pt, pht)):
                        raise ValueError(
                            f"PH/PT/PHT labels must be binary at "
                            f"{source_path}:{row_number}"
                        )
                    label_type = f"{ph}{pt}{pht}"
                    if label_type not in PHT_LABEL_TYPES:
                        raise ValueError(
                            f"Unsupported PH/PT/PHT label type {label_type} at "
                            f"{source_path}:{row_number}"
                        )
                    if label_type not in self.allowed_label_types:
                        self.filter_stats[f"excluded_type_{label_type}"] += 1
                        continue
                    try:
                        peptide = normalise_sequence(row["peptide"])
                        hla = normalise_sequence(row["HLA"])
                        tcr = normalise_sequence(row["tcr"])
                    except ValueError:
                        self.filter_stats["invalid_empty_sequence"] += 1
                        self.filter_stats["invalid_rows"] += 1
                        continue
                    if not peptide_min_length <= len(peptide) <= peptide_max_length:
                        self.filter_stats["invalid_peptide_length"] += 1
                        self.filter_stats["invalid_rows"] += 1
                        continue
                    if not 1 <= len(hla) <= hla_max_length:
                        self.filter_stats["invalid_hla_length"] += 1
                        self.filter_stats["invalid_rows"] += 1
                        continue
                    if not 1 <= len(tcr) <= tcr_max_length:
                        self.filter_stats["invalid_tcr_length"] += 1
                        self.filter_stats["invalid_rows"] += 1
                        continue
                    if reject_noncanonical_sequences and any(
                        not set(sequence).issubset(_CANONICAL_AMINO_ACIDS)
                        for sequence in (peptide, hla, tcr)
                    ):
                        self.filter_stats["invalid_noncanonical"] += 1
                        self.filter_stats["invalid_rows"] += 1
                        continue
                    key = (peptide, tcr, hla)
                    previous_labels = labels_by_key.get(key)
                    current_labels = (ph, pt, pht)
                    if previous_labels is not None:
                        if previous_labels != current_labels:
                            raise ValueError(
                                "Conflicting PH/PT/PHT labels for triplet "
                                f"{key!r} in {source_path}"
                            )
                        self.filter_stats["same_label_duplicates"] += 1
                        continue
                    labels_by_key[key] = current_labels
                    digest.update(
                        (
                            f"{peptide}\t{hla}\t{tcr}\t{label_type}\n"
                        ).encode("utf-8")
                    )
                    self.filter_stats[f"type_{label_type}"] += 1
                    if (
                        limit_per_type is not None
                        and len(self.type_indices[label_type]) >= limit_per_type
                    ):
                        continue
                    record = PHTMultitaskRecord(
                        peptide=peptide,
                        tcr=tcr,
                        hla=hla,
                        ph=ph,
                        pt=pt,
                        pht=pht,
                        source_row=source_row,
                    )
                    self.type_indices[label_type].append(len(self.records))
                    self.records.append(record)
            source_row_offset += file_rows + 1
        if not self.records:
            raise ValueError(
                "No usable rows found in "
                + ", ".join(str(path) for path in self.csv_paths)
            )
        self.filter_stats["returned"] = len(self.records)
        self.dataset_fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PHTMultitaskRecord:
        return self.records[index]


class PHTTaskDataset(Dataset):
    """A task-specific label view over shared PHT records."""

    def __init__(self, dataset: PHTMultitaskDataset, task: str):
        if task not in {"phla", "ptcr", "triplet"}:
            raise ValueError(f"Unknown PHT task: {task}")
        self.dataset = dataset
        self.task = task

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> PHTMultitaskRecord:
        return self.dataset[index]


def collate_pht_task(
    records: Sequence[PHTMultitaskRecord],
    task: str,
) -> dict[str, list[str] | torch.Tensor]:
    labels = {
        "phla": [record.ph for record in records],
        "ptcr": [record.pt for record in records],
        "triplet": [record.pht for record in records],
    }
    if task not in labels:
        raise ValueError(f"Unknown PHT task: {task}")
    output: dict[str, list[str] | torch.Tensor] = {
        "peptide_sequences": [record.peptide for record in records],
        "labels": torch.tensor(labels[task], dtype=torch.long),
        "source_rows": torch.tensor(
            [record.source_row for record in records], dtype=torch.long
        ),
    }
    if task == "phla":
        output["receptor_sequences"] = [record.hla for record in records]
    elif task == "ptcr":
        output["receptor_sequences"] = [record.tcr for record in records]
    else:
        output["hla_sequences"] = [record.hla for record in records]
        output["tcr_sequences"] = [record.tcr for record in records]
    return output


def collate_pht_multitask(
    records: Sequence[PHTMultitaskRecord],
) -> dict[str, list[str] | torch.Tensor]:
    return {
        "peptide_sequences": [record.peptide for record in records],
        "hla_sequences": [record.hla for record in records],
        "tcr_sequences": [record.tcr for record in records],
        "phla_labels": torch.tensor(
            [record.ph for record in records], dtype=torch.long
        ),
        "ptcr_labels": torch.tensor(
            [record.pt for record in records], dtype=torch.long
        ),
        "pht_labels": torch.tensor(
            [record.pht for record in records], dtype=torch.long
        ),
        "source_rows": torch.tensor(
            [record.source_row for record in records], dtype=torch.long
        ),
    }


class BalancedPHTSampler(Sampler[int]):
    """Equal selected PHT-type epochs with deterministic rotation."""

    def __init__(
        self,
        type_indices: Mapping[str, Sequence[int]],
        *,
        seed: int,
        label_types: Sequence[str] | None = None,
    ):
        self.label_types = _validate_pht_label_types(
            label_types,
            parameter_name="label_types",
        )
        missing = [kind for kind in self.label_types if not type_indices.get(kind)]
        if missing:
            raise ValueError(
                "Balanced PHT sampling requires non-empty selected types: "
                + ", ".join(missing)
            )
        self.type_indices = {
            kind: list(type_indices[kind]) for kind in self.label_types
        }
        self.target_per_type = min(
            len(self.type_indices[kind]) for kind in self.label_types
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.target_per_type * len(self.label_types)

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        selected: dict[str, list[int]] = {}
        for kind_index, kind in enumerate(self.label_types):
            source = self.type_indices[kind]
            base_generator = torch.Generator().manual_seed(
                self.seed + 97_409 * kind_index
            )
            order = torch.randperm(
                len(source), generator=base_generator
            ).tolist()
            offset = (self.epoch * self.target_per_type) % len(source)
            rotated = order[offset:] + order[:offset]
            selected[kind] = [source[index] for index in rotated[: self.target_per_type]]
        for offset in range(self.target_per_type):
            group = [selected[kind][offset] for kind in self.label_types]
            order = torch.randperm(len(group), generator=generator).tolist()
            for index in order:
                yield group[index]


def _require_multitask_classes(
    dataset: PHTMultitaskDataset,
    *,
    split_name: str,
) -> None:
    for task, attribute in (
        ("phla", "ph"),
        ("ptcr", "pt"),
        ("triplet", "pht"),
    ):
        labels = {getattr(record, attribute) for record in dataset.records}
        if labels != {0, 1}:
            raise ValueError(
                f"{split_name}/{task} must contain both binary classes"
            )


def _validate_entity_holdout(
    train: PHTMultitaskDataset,
    validation: PHTMultitaskDataset,
    split_name: str,
) -> None:
    suffix_to_attributes = {
        "val_unseen_hla": ("hla",),
        "val_unseen_peptide": ("peptide",),
        "val_unseen_tcr": ("tcr",),
        "val_unseen_triple": ("peptide", "hla", "tcr"),
    }
    attributes = suffix_to_attributes.get(split_name)
    if attributes is None:
        return
    for attribute in attributes:
        train_values = {getattr(record, attribute) for record in train.records}
        overlap = {
            getattr(record, attribute)
            for record in validation.records
            if getattr(record, attribute) in train_values
        }
        if overlap:
            raise ValueError(
                f"{split_name} is not entity-disjoint for {attribute}: "
                f"overlap_count={len(overlap)}"
            )


def build_pht_stage3_loaders(
    config,
    *,
    limit_per_type: int | None = None,
):
    """Build same-source PH/PT/PHT training views and benchmark loaders."""

    values = config.triplet
    root = config.paths.data_root / values.data_subdir
    label_types = _validate_pht_label_types(
        values.pht_label_types,
        parameter_name="triplet.pht_label_types",
    )
    common = {
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "hla_max_length": config.model.hla_max_length,
        "tcr_max_length": config.model.tcr_max_length,
        "reject_noncanonical_sequences": values.reject_noncanonical_sequences,
        "limit_per_type": limit_per_type,
        "allowed_label_types": label_types,
    }

    def class_csv_paths(filename: str) -> list[Path] | None:
        if not values.pht_class_dirs:
            return None
        return [
            root / class_dir / filename
            for class_dir in values.pht_class_dirs
        ]

    train_paths = class_csv_paths(values.train_filename)
    train_dataset = PHTMultitaskDataset(
        train_paths if train_paths is not None else root / values.train_filename,
        **common,
    )
    _require_multitask_classes(train_dataset, split_name="train")
    train_loaders = {}
    for task in ("phla", "ptcr", "triplet"):
        sampler = BalancedPHTSampler(
            train_dataset.type_indices,
            seed=config.training.seed,
            label_types=label_types,
        )
        train_loaders[task] = DataLoader(
            PHTTaskDataset(train_dataset, task),
            batch_size=values.micro_batch_size,
            sampler=sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=False,
            drop_last=False,
            collate_fn=partial(collate_pht_task, task=task),
            worker_init_fn=_seed_worker,
        )
    validation_datasets = {}
    val_loaders = {}
    for split_name in values.validation_splits:
        validation_paths = (
            class_csv_paths(values.val_filename)
            if values.pht_class_dirs and split_name == "val"
            else None
        )
        dataset = PHTMultitaskDataset(
            (
                validation_paths
                if validation_paths is not None
                else root / values.validation_subdir / f"{split_name}.csv"
            ),
            **common,
        )
        _require_multitask_classes(dataset, split_name=split_name)
        _validate_entity_holdout(train_dataset, dataset, split_name)
        validation_datasets[split_name] = dataset
        val_loaders[split_name] = DataLoader(
            dataset,
            batch_size=values.eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_pht_multitask,
            worker_init_fn=_seed_worker,
        )
    digest = hashlib.sha256()
    digest.update(f"train:{train_dataset.dataset_fingerprint}\n".encode())
    for split_name in values.validation_splits:
        digest.update(
            (
                f"{split_name}:"
                f"{validation_datasets[split_name].dataset_fingerprint}\n"
            ).encode()
        )
    audit = {
        "data_mode": "pht_multitask",
        "dataset_fingerprint": digest.hexdigest(),
        "limit_per_type": limit_per_type,
        "pht_class_dirs": list(values.pht_class_dirs),
        "train_sources": [
            str(path)
            for path in (
                train_paths
                if train_paths is not None
                else [root / values.train_filename]
            )
        ],
        "validation_sources": {
            split_name: [
                str(path)
                for path in (
                    class_csv_paths(values.val_filename)
                    if values.pht_class_dirs and split_name == "val"
                    else [
                        root
                        / values.validation_subdir
                        / f"{split_name}.csv"
                    ]
                )
            ]
            for split_name in values.validation_splits
        },
        "pht_label_types": list(label_types),
        "train": dict(train_dataset.filter_stats),
        "validation": {
            split: dict(dataset.filter_stats)
            for split, dataset in validation_datasets.items()
        },
        "validation_splits": list(values.validation_splits),
        "selection_splits": list(values.selection_splits),
    }
    return train_loaders, val_loaders, audit


@dataclass(frozen=True)
class TripletDatasetBundle:
    train: "OnlineTripletDataset"
    val: "OnlineTripletDataset"
    test: "OnlineTripletDataset"
    manifest_audit: dict[str, int | bool | str]

    def as_dict(self) -> dict[str, "OnlineTripletDataset"]:
        return {"train": self.train, "val": self.val, "test": self.test}


def triplet_data_paths(
    data_root: str | Path,
    *,
    data_subdir: str = "data_triple",
    merged_filename: str = "merged_pseudo.csv",
    train_filename: str = "train_set.csv",
    val_filename: str = "val_set.csv",
    test_filename: str = "test_set.csv",
) -> dict[str, Path]:
    root = Path(data_root) / data_subdir
    paths = {
        "merged": root / merged_filename,
        "train": root / train_filename,
        "val": root / val_filename,
        "test": root / test_filename,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Triplet dataset files not found: " + ", ".join(missing)
        )
    return paths


def _raw_counter(path: Path) -> Counter[tuple[str, str, str, str]]:
    values: Counter[tuple[str, str, str, str]] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "tcr", "hla", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                key = (
                    str(row["peptide"]).strip(),
                    str(row["tcr"]).strip(),
                    str(row["hla"]).strip(),
                    str(row["label"]).strip(),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Malformed triplet row at {path}:{row_number}"
                ) from exc
            values[key] += 1
    return values


def verify_merged_manifest(
    merged_path: str | Path,
    split_paths: Sequence[str | Path],
) -> dict[str, int | bool | str]:
    """Verify that merged_pseudo is exactly the multiset union of its splits."""

    merged_path = Path(merged_path)
    merged = _raw_counter(merged_path)
    split_union: Counter[tuple[str, str, str, str]] = Counter()
    split_rows = 0
    for path_value in split_paths:
        current = _raw_counter(Path(path_value))
        split_union.update(current)
        split_rows += sum(current.values())
    merged_only = sum((merged - split_union).values())
    splits_only = sum((split_union - merged).values())
    matches = merged == split_union
    audit: dict[str, int | bool | str] = {
        "merged_path": str(merged_path.resolve()),
        "merged_rows": sum(merged.values()),
        "split_rows": split_rows,
        "merged_only_rows": merged_only,
        "splits_only_rows": splits_only,
        "matches": matches,
    }
    if not matches:
        raise ValueError(
            "Triplet merged manifest does not equal train+val+test: "
            f"merged_only={merged_only}, splits_only={splits_only}"
        )
    return audit


class OnlineTripletDataset(Dataset):
    """Validated triplets with deterministic duplicate and leakage removal."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        excluded_labels: Mapping[TripletKey, int] | None = None,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        hla_max_length: int = 34,
        tcr_max_length: int = 34,
        strict_split_cleanup: bool = True,
        reject_noncanonical_sequences: bool = True,
        limit: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.records: list[TripletRecord] = []
        self.all_labels_by_key: dict[TripletKey, int] = {}
        excluded_labels = dict(excluded_labels or {})
        self.filter_stats: dict[str, int] = {
            "total": 0,
            "eligible": 0,
            "returned": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
            "hla_too_long": 0,
            "tcr_too_long": 0,
            "within_split_duplicate": 0,
            "higher_priority_overlap": 0,
            "ambiguous_peptide": 0,
            "ambiguous_hla": 0,
            "ambiguous_tcr": 0,
            "noncanonical_rejected": 0,
            "limit_dropped": 0,
            "eligible_label_0": 0,
            "eligible_label_1": 0,
            "returned_label_0": 0,
            "returned_label_1": 0,
        }

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"peptide", "tcr", "hla", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {sorted(required)}"
                )
            for row_number, row in enumerate(reader, start=2):
                self.filter_stats["total"] += 1
                try:
                    peptide = normalise_sequence(row["peptide"])
                    tcr = normalise_sequence(row["tcr"])
                    hla = normalise_sequence(row["hla"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid sequence at {self.csv_path}:{row_number}"
                    ) from exc

                if len(peptide) < peptide_min_length:
                    self.filter_stats["peptide_too_short"] += 1
                    continue
                if len(peptide) > peptide_max_length:
                    self.filter_stats["peptide_too_long"] += 1
                    continue
                if len(hla) > hla_max_length:
                    self.filter_stats["hla_too_long"] += 1
                    continue
                if len(tcr) > tcr_max_length:
                    self.filter_stats["tcr_too_long"] += 1
                    continue
                self.filter_stats["ambiguous_peptide"] += int(
                    not set(peptide).issubset(_CANONICAL_AMINO_ACIDS)
                )
                self.filter_stats["ambiguous_hla"] += int(
                    not set(hla).issubset(_CANONICAL_AMINO_ACIDS)
                )
                self.filter_stats["ambiguous_tcr"] += int(
                    not set(tcr).issubset(_CANONICAL_AMINO_ACIDS)
                )
                noncanonical = (
                    not set(peptide).issubset(_CANONICAL_AMINO_ACIDS)
                    or not set(hla).issubset(_CANONICAL_AMINO_ACIDS)
                    or not set(tcr).issubset(_CANONICAL_AMINO_ACIDS)
                )
                if reject_noncanonical_sequences and noncanonical:
                    self.filter_stats["noncanonical_rejected"] += 1
                    continue
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label at {self.csv_path}:{row_number}"
                    ) from exc
                if label not in {0, 1}:
                    raise ValueError(
                        f"Label must be 0 or 1 at {self.csv_path}:{row_number}"
                    )

                key = (peptide, tcr, hla)
                previous = self.all_labels_by_key.get(key)
                if previous is not None:
                    if previous != label:
                        raise ValueError(
                            "Conflicting labels for triplet "
                            f"{key!r} in {self.csv_path}"
                        )
                    if strict_split_cleanup:
                        self.filter_stats["within_split_duplicate"] += 1
                        continue
                else:
                    self.all_labels_by_key[key] = label

                excluded_label = excluded_labels.get(key)
                if excluded_label is not None:
                    if excluded_label != label:
                        raise ValueError(
                            "Conflicting labels across triplet splits for "
                            f"{key!r}: higher={excluded_label}, "
                            f"{self.csv_path.name}={label}"
                        )
                    if strict_split_cleanup:
                        self.filter_stats["higher_priority_overlap"] += 1
                        continue

                self.filter_stats["eligible"] += 1
                self.filter_stats[f"eligible_label_{label}"] += 1
                self.records.append(
                    TripletRecord(
                        peptide=peptide,
                        tcr=tcr,
                        hla=hla,
                        label=label,
                        source_row=row_number,
                    )
                )

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.filter_stats["limit_dropped"] = max(
                0, len(self.records) - limit
            )
            self.records = self.records[:limit]
        self.filter_stats["returned"] = len(self.records)
        for record in self.records:
            self.filter_stats[f"returned_label_{record.label}"] += 1
        if not self.records:
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TripletRecord:
        return self.records[index]


def build_strict_triplet_datasets(
    data_root: str | Path,
    *,
    data_subdir: str,
    merged_filename: str,
    train_filename: str,
    val_filename: str,
    test_filename: str,
    verify_manifest: bool,
    strict_split_cleanup: bool,
    peptide_min_length: int,
    peptide_max_length: int,
    hla_max_length: int,
    tcr_max_length: int,
    reject_noncanonical_sequences: bool = True,
    limit: int | None = None,
) -> TripletDatasetBundle:
    paths = triplet_data_paths(
        data_root,
        data_subdir=data_subdir,
        merged_filename=merged_filename,
        train_filename=train_filename,
        val_filename=val_filename,
        test_filename=test_filename,
    )
    if verify_manifest:
        manifest = verify_merged_manifest(
            paths["merged"],
            (paths["train"], paths["val"], paths["test"]),
        )
    else:
        manifest = {
            "merged_path": str(paths["merged"].resolve()),
            "matches": False,
            "verification_disabled": True,
        }

    common = {
        "peptide_min_length": peptide_min_length,
        "peptide_max_length": peptide_max_length,
        "hla_max_length": hla_max_length,
        "tcr_max_length": tcr_max_length,
        "strict_split_cleanup": strict_split_cleanup,
        "reject_noncanonical_sequences": reject_noncanonical_sequences,
        "limit": limit,
    }
    train = OnlineTripletDataset(paths["train"], **common)
    val = OnlineTripletDataset(
        paths["val"],
        excluded_labels=train.all_labels_by_key,
        **common,
    )
    higher_priority = dict(train.all_labels_by_key)
    for key, label in val.all_labels_by_key.items():
        existing = higher_priority.get(key)
        if existing is not None and existing != label:
            raise ValueError(f"Conflicting train/val labels for {key!r}")
        higher_priority[key] = label
    test = OnlineTripletDataset(
        paths["test"],
        excluded_labels=higher_priority,
        **common,
    )
    return TripletDatasetBundle(
        train=train,
        val=val,
        test=test,
        manifest_audit=manifest,
    )


def collate_triplets(
    records: Sequence[TripletRecord],
) -> dict[str, list[str] | torch.Tensor]:
    return {
        "peptide_sequences": [record.peptide for record in records],
        "hla_sequences": [record.hla for record in records],
        "tcr_sequences": [record.tcr for record in records],
        "labels": torch.tensor(
            [record.label for record in records],
            dtype=torch.long,
        ),
        "source_rows": torch.tensor(
            [record.source_row for record in records],
            dtype=torch.long,
        ),
    }


def make_triplet_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
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
        collate_fn=collate_triplets,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


class HlaShiftedTripletDataset(Dataset):
    """Deterministic HLA-shuffle ablation retaining all other row fields."""

    def __init__(self, dataset: OnlineTripletDataset):
        if len(dataset) < 2:
            raise ValueError("HLA shuffle requires at least two records")
        self.dataset = dataset
        self.offset = max(1, len(dataset) // 2)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TripletRecord:
        original = self.dataset[index]
        shifted = self.dataset[(index + self.offset) % len(self.dataset)]
        return TripletRecord(
            peptide=original.peptide,
            tcr=original.tcr,
            hla=shifted.hla,
            label=original.label,
            source_row=original.source_row,
        )

    @property
    def changed_fraction(self) -> float:
        changed = sum(
            self.dataset[index].hla
            != self.dataset[(index + self.offset) % len(self.dataset)].hla
            for index in range(len(self.dataset))
        )
        return changed / len(self.dataset)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
