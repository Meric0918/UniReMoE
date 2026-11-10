"""Strict audit, PH-level splitting, and loading for Stage-9 pairs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import ModelConfig
from .data import normalise_sequence


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Stage9PairRecord:
    peptide: str
    hla: str
    tcr_positive: str
    tcr_negative: str
    pair_id: str
    source_row: int

    @property
    def ph_key(self) -> tuple[str, str]:
        return self.peptide, self.hla

    @property
    def pair_key(self) -> tuple[str, str, str, str]:
        return self.peptide, self.hla, self.tcr_positive, self.tcr_negative


class Stage9PairDataset(Dataset):
    def __init__(self, records: Sequence[Stage9PairRecord]):
        if not records:
            raise ValueError("Stage-9 pair dataset cannot be empty")
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Stage9PairRecord:
        return self.records[index]


@dataclass
class Stage9AuditResult:
    train_records: list[Stage9PairRecord]
    val_records: list[Stage9PairRecord]
    test_records: list[Stage9PairRecord]
    report: dict
    errors: list[dict]
    dataset_fingerprint: str

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def split_records(self) -> dict[str, list[Stage9PairRecord]]:
        return {
            "train": self.train_records,
            "val": self.val_records,
            "test": self.test_records,
        }

    def require_clean(self) -> "Stage9AuditResult":
        if self.has_errors:
            raise ValueError("Stage-9 data audit failed; inspect audit errors")
        return self

    def to_dict(self) -> dict:
        return {
            "report": self.report,
            "errors": self.errors,
            "dataset_fingerprint": self.dataset_fingerprint,
            "splits": {
                split: [
                    {
                        "pair_id": record.pair_id,
                        "peptide": record.peptide,
                        "hla": record.hla,
                        "source_row": record.source_row,
                    }
                    for record in records
                ]
                for split, records in self.split_records.items()
            },
        }

    def write_report(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_error(record: Stage9PairRecord, model: ModelConfig, reject: bool) -> str | None:
    if not model.peptide_min_length <= len(record.peptide) <= model.peptide_max_length:
        return "peptide length outside configured range"
    if not 1 <= len(record.hla) <= model.hla_max_length:
        return "HLA length outside configured range"
    if not 1 <= len(record.tcr_positive) <= model.tcr_max_length:
        return "positive TCR length outside configured range"
    if not 1 <= len(record.tcr_negative) <= model.tcr_max_length:
        return "negative TCR length outside configured range"
    if reject and any(
        not set(sequence).issubset(CANONICAL_AMINO_ACIDS)
        for sequence in (
            record.peptide,
            record.hla,
            record.tcr_positive,
            record.tcr_negative,
        )
    ):
        return "noncanonical amino acid"
    return None


def _read_stage3_triples(path: Path) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(
            reader.fieldnames or (),
            {
                "peptide": {"peptide", "p"},
                "hla": {"hla", "h"},
                "tcr": {"tcr", "tcr_sequence", "receptor", "receptor_sequence"},
            },
            path=path,
        )
        for row_number, row in enumerate(reader, start=2):
            try:
                triples.add(
                    (
                        normalise_sequence(row[columns["peptide"]]),
                        normalise_sequence(row[columns["hla"]]),
                        normalise_sequence(row[columns["tcr"]]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid Stage-3 row at {path}:{row_number}") from exc
    return triples


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _resolve_columns(
    fieldnames: Sequence[str],
    aliases: dict[str, set[str]],
    *,
    path: Path,
    optional: set[str] | None = None,
) -> dict[str, str]:
    optional = optional or set()
    normalised = {
        field: _normalise_column_name(field)
        for field in fieldnames
        if field is not None
    }
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        matches = [
            field for field, normalised_name in normalised.items()
            if normalised_name in candidates
        ]
        if len(matches) > 1:
            raise ValueError(
                f"{path} has multiple columns for '{canonical}': {', '.join(matches)}"
            )
        if matches:
            resolved[canonical] = matches[0]
        elif canonical not in optional:
            raise ValueError(
                f"{path} missing Stage-9 columns: {canonical}"
            )
    return resolved


def _ph_fraction(ph_key: tuple[str, str], split_seed: int) -> float:
    payload = f"{split_seed}\0{ph_key[0]}\0{ph_key[1]}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def audit_stage9_pairs(
    pair_csv: str | Path,
    stage3_train_csv: str | Path,
    model: ModelConfig,
    *,
    supplement_csv: str | Path | None = None,
    split_seed: int = 2026,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    reject_noncanonical_sequences: bool = True,
) -> Stage9AuditResult:
    pair_path = Path(pair_csv)
    stage3_path = Path(stage3_train_csv)
    supplement_path = Path(supplement_csv) if supplement_csv is not None else None
    errors: list[dict] = []
    records: list[Stage9PairRecord] = []
    supplement_records: list[Stage9PairRecord] = []
    supplement_split_by_ph: dict[tuple[str, str], str] = {}
    seen_pairs: set[tuple[str, str, str, str]] = set()
    duplicate_pairs = 0

    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(
            reader.fieldnames or (),
            {
                "peptide": {"peptide", "p"},
                "hla": {"hla", "h"},
                "tcr_positive": {
                    "tcr_positive",
                    "positive_tcr",
                    "tcr_pos",
                    "pos_tcr",
                    "positive",
                },
                "tcr_negative": {
                    "tcr_negative",
                    "negative_tcr",
                    "tcr_neg",
                    "neg_tcr",
                    "negative",
                },
                "pair_id": {"pair_id", "pair", "id"},
            },
            path=pair_path,
            optional={"pair_id"},
        )
        for row_number, row in enumerate(reader, start=2):
            try:
                record = Stage9PairRecord(
                    peptide=normalise_sequence(row[columns["peptide"]]),
                    hla=normalise_sequence(row[columns["hla"]]),
                    tcr_positive=normalise_sequence(row[columns["tcr_positive"]]),
                    tcr_negative=normalise_sequence(row[columns["tcr_negative"]]),
                    pair_id=(row.get(columns.get("pair_id", "")) or "").strip()
                    or f"row:{row_number}",
                    source_row=row_number,
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"row": row_number, "reason": f"invalid sequence: {exc}"})
                continue
            if record.tcr_positive == record.tcr_negative:
                errors.append(
                    {
                        "row": row_number,
                        "reason": "positive and negative TCR must differ",
                    }
                )
                continue
            sequence_error = _sequence_error(
                record, model, reject_noncanonical_sequences
            )
            if sequence_error:
                errors.append({"row": row_number, "reason": sequence_error})
                continue
            if record.pair_key in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(record.pair_key)
            records.append(record)

    base_ph = {record.ph_key for record in records}
    stage3_triples = _read_stage3_triples(stage3_path)
    stage3_ph = {(peptide, hla) for peptide, hla, _ in stage3_triples}

    if supplement_path is not None:
        with supplement_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _resolve_columns(
                reader.fieldnames or (),
                {
                    "peptide": {"peptide", "p"},
                    "hla": {"hla", "h"},
                    "tcr_positive": {
                        "tcr_positive",
                        "positive_tcr",
                        "tcr_pos",
                        "pos_tcr",
                        "positive",
                    },
                    "tcr_negative": {
                        "tcr_negative",
                        "negative_tcr",
                        "tcr_neg",
                        "neg_tcr",
                        "negative",
                    },
                    "pair_id": {"pair_id", "pair", "id"},
                    "split": {"split", "dataset_split", "partition"},
                },
                path=supplement_path,
                optional={"pair_id"},
            )
            for row_number, row in enumerate(reader, start=2):
                split_value = (row.get(columns["split"]) or "").strip().lower()
                if split_value not in {"val", "test"}:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": "supplement split must be val or test",
                        }
                    )
                    continue
                try:
                    record = Stage9PairRecord(
                        peptide=normalise_sequence(row[columns["peptide"]]),
                        hla=normalise_sequence(row[columns["hla"]]),
                        tcr_positive=normalise_sequence(
                            row[columns["tcr_positive"]]
                        ),
                        tcr_negative=normalise_sequence(
                            row[columns["tcr_negative"]]
                        ),
                        pair_id=(row.get(columns.get("pair_id", "")) or "").strip()
                        or f"supplement-row:{row_number}",
                        source_row=row_number,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": f"invalid supplement sequence: {exc}",
                        }
                    )
                    continue
                if record.tcr_positive == record.tcr_negative:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": "positive and negative TCR must differ",
                        }
                    )
                    continue
                sequence_error = _sequence_error(
                    record, model, reject_noncanonical_sequences
                )
                if sequence_error:
                    errors.append({"row": row_number, "reason": sequence_error})
                    continue
                previous_split = supplement_split_by_ph.get(record.ph_key)
                if previous_split is not None and previous_split != split_value:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": "supplement PH group has multiple requested splits",
                            "peptide": record.peptide,
                            "hla": record.hla,
                        }
                    )
                supplement_split_by_ph.setdefault(record.ph_key, split_value)
                if record.ph_key in base_ph:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": "supplement PH group overlaps base Stage-9 pairs",
                            "peptide": record.peptide,
                            "hla": record.hla,
                        }
                    )
                if record.ph_key in stage3_ph:
                    errors.append(
                        {
                            "row": row_number,
                            "reason": "supplement PH group overlaps Stage-3 train",
                            "peptide": record.peptide,
                            "hla": record.hla,
                        }
                    )
                if record.pair_key in seen_pairs:
                    duplicate_pairs += 1
                    continue
                seen_pairs.add(record.pair_key)
                records.append(record)
                supplement_records.append(record)

    roles: dict[tuple[str, str, str], set[str]] = {}
    role_rows: dict[tuple[str, str, str], list[int]] = {}
    for record in records:
        for tcr, role in (
            (record.tcr_positive, "positive"),
            (record.tcr_negative, "negative"),
        ):
            key = (record.peptide, record.hla, tcr)
            roles.setdefault(key, set()).add(role)
            role_rows.setdefault(key, []).append(record.source_row)
    for key, observed in roles.items():
        if len(observed) > 1:
            errors.append(
                {
                    "rows": role_rows[key],
                    "reason": "same (P,H,T) appears in both positive and negative roles",
                    "peptide": key[0],
                    "hla": key[1],
                    "tcr": key[2],
                }
            )

    forced_ph = {
        record.ph_key
        for record in records
        if (record.peptide, record.hla, record.tcr_positive) in stage3_triples
        or (record.peptide, record.hla, record.tcr_negative) in stage3_triples
    }
    splits: dict[str, list[Stage9PairRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    ph_assignments: dict[tuple[str, str], str] = {}
    for record in records:
        split = ph_assignments.get(record.ph_key)
        if split is None:
            requested_split = supplement_split_by_ph.get(record.ph_key)
            if requested_split is not None:
                split = requested_split
            else:
                fraction = _ph_fraction(record.ph_key, split_seed)
                if record.ph_key in forced_ph or fraction < train_fraction:
                    split = "train"
                elif fraction < train_fraction + val_fraction:
                    split = "val"
                else:
                    split = "test"
            ph_assignments[record.ph_key] = split
        splits[split].append(record)

    for split in ("train", "val", "test"):
        if not splits[split]:
            errors.append({"reason": f"derived {split} split is empty"})
    pair_sha256 = _file_sha256(pair_path)
    stage3_sha256 = _file_sha256(stage3_path)
    supplement_sha256 = (
        _file_sha256(supplement_path) if supplement_path is not None else None
    )
    split_manifest = [
        {
            "peptide": ph_key[0],
            "hla": ph_key[1],
            "split": split,
        }
        for ph_key, split in sorted(ph_assignments.items())
    ]
    fingerprint_payload = {
        "pair_csv_sha256": pair_sha256,
        "stage3_train_sha256": stage3_sha256,
        "supplement_csv_sha256": supplement_sha256,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "split_manifest": split_manifest,
    }
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "pair_csv": str(pair_path.resolve()),
        "stage3_train_csv": str(stage3_path.resolve()),
        "pair_csv_sha256": pair_sha256,
        "stage3_train_sha256": stage3_sha256,
        "supplement_csv": (
            str(supplement_path.resolve()) if supplement_path is not None else None
        ),
        "supplement_csv_sha256": supplement_sha256,
        "supplement_rows": len(supplement_records),
        "supplement_ph_groups": len(supplement_split_by_ph),
        "supplement_split_rows": {
            split: sum(
                1
                for record in supplement_records
                if supplement_split_by_ph.get(record.ph_key) == split
            )
            for split in ("val", "test")
        },
        "input_rows": len(records) + duplicate_pairs + sum(
            1 for error in errors if "row" in error
        ),
        "retained_pairs": len(records),
        "duplicate_pairs_removed": duplicate_pairs,
        "forced_train_ph_groups": len(forced_ph),
        "ph_groups": len(ph_assignments),
        "split_seed": split_seed,
        "split_rows": {name: len(values) for name, values in splits.items()},
        "split_ph_groups": dict(Counter(ph_assignments.values())),
        "split_manifest": split_manifest,
    }
    return Stage9AuditResult(
        train_records=splits["train"],
        val_records=splits["val"],
        test_records=splits["test"],
        report=report,
        errors=errors,
        dataset_fingerprint=dataset_fingerprint,
    )


def pair_sampling_weights(records: Sequence[Stage9PairRecord]) -> list[float]:
    counts = Counter(record.ph_key for record in records)
    return [1.0 / counts[record.ph_key] for record in records]


def collate_stage9_pairs(
    records: Sequence[Stage9PairRecord],
) -> dict[str, list[str] | torch.Tensor]:
    return {
        "peptide_sequences": [record.peptide for record in records],
        "hla_sequences": [record.hla for record in records],
        "positive_tcr_sequences": [record.tcr_positive for record in records],
        "negative_tcr_sequences": [record.tcr_negative for record in records],
        "pair_ids": [record.pair_id for record in records],
        "source_rows": torch.tensor(
            [record.source_row for record in records], dtype=torch.long
        ),
    }


def make_stage9_pair_loader(
    records: Sequence[Stage9PairRecord],
    *,
    batch_size: int,
    training: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    ph_balanced_sampling: bool = True,
) -> DataLoader:
    dataset = Stage9PairDataset(records)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = training
    if training and ph_balanced_sampling:
        sampler = WeightedRandomSampler(
            pair_sampling_weights(records),
            num_samples=len(records),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_stage9_pairs,
        generator=generator if sampler is None else None,
    )
    loader.stage9_generator = generator  # type: ignore[attr-defined]
    return loader
