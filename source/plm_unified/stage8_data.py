"""Strict data audit and deterministic four-type sampling for Stage 8."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .cache import CacheLocator, EmbeddingCache
from .config import ModelConfig
from .data import make_online_dataloader


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
LABEL_TYPES = ("111", "100", "010", "000")


class Stage8DataError(ValueError):
    pass


def stage8_cache_hidden_dtype(
    device: torch.device,
    mixed_precision: str,
) -> torch.dtype:
    """Keep fp16 cache payloads only when CUDA autocast will consume them."""

    return (
        torch.float16
        if device.type == "cuda" and mixed_precision != "none"
        else torch.float32
    )


@dataclass(frozen=True)
class Stage8Record:
    peptide: str
    tcr: str
    hla: str
    ph: int
    pt: int
    pht: int
    source_row: int
    source_file: str = ""
    split_type: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return self.peptide, self.tcr, self.hla

    @property
    def labels(self) -> tuple[int, int, int]:
        return self.ph, self.pt, self.pht

    @property
    def label_type(self) -> str:
        return f"{self.ph}{self.pt}{self.pht}"


@dataclass
class Stage8AuditResult:
    train_records: list[Stage8Record]
    val_records: list[Stage8Record]
    report: dict
    conflicts: list[dict]
    errors: list[dict]
    rejected_rows: list[dict]
    dataset_fingerprint: str

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def require_clean(self) -> "Stage8AuditResult":
        if self.has_conflicts or self.has_errors:
            raise Stage8DataError(
                "Stage-8 data audit failed; inspect the conflict/error report and "
                "perform manual repair before caching or training"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "report": self.report,
            "conflicts": self.conflicts,
            "errors": self.errors,
            "rejected_rows": self.rejected_rows,
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    def write_reports(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "audit.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        for filename, rows in (
            ("conflicts.csv", self.conflicts),
            ("errors.csv", self.errors),
            ("rejected_rows.csv", self.rejected_rows),
        ):
            keys = sorted({key for row in rows for key in row})
            with (root / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
                writer.writeheader()
                if rows:
                    writer.writerows(rows)


_COLUMN_ALIASES = {
    "peptide": "peptide",
    "hla": "hla",
    "tcr": "tcr",
    "ph": "PH",
    "pt": "PT",
    "pht": "PHT",
    "class_code": "class_code",
    "split_type": "split_type",
    "source_file": "source_file",
}


def _canonical_field_map(fieldnames: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_name in fieldnames:
        raw = str(raw_name).lstrip("\ufeff").strip()
        canonical = _COLUMN_ALIASES.get(raw.lower(), raw.lower())
        if canonical in mapping:
            raise Stage8DataError(
                f"duplicate columns after normalization: {canonical}"
            )
        mapping[canonical] = raw_name
    return mapping


def _canonical_row(
    row: Mapping[str, str], field_map: Mapping[str, str]
) -> dict[str, str]:
    return {
        canonical: row.get(raw_name, "")
        for canonical, raw_name in field_map.items()
    }


def _normalise_row(
    row: Mapping[str, str],
    row_number: int,
    *,
    source_file: str = "",
) -> Stage8Record:
    def field(name: str) -> str:
        value = row.get(name)
        if value is None:
            raise Stage8DataError(f"missing column {name}")
        return value.strip().upper()

    try:
        labels = tuple(int(field(name)) for name in ("PH", "PT", "PHT"))
    except ValueError as exc:
        raise Stage8DataError("labels must be integers") from exc
    if any(value not in {0, 1} for value in labels):
        raise Stage8DataError("labels must be binary")
    return Stage8Record(
        field("peptide"), field("tcr"), field("hla"), *labels, row_number,
        source_file, row.get("split_type", "").strip(),
    )


def _read_split(
    path: Path,
    split: str,
    model: ModelConfig,
    reject_noncanonical: bool,
    *,
    expected_class_code: str | None = None,
    source_label: str | None = None,
) -> tuple[list[Stage8Record], dict, list[dict], list[dict], list[dict]]:
    source_name = source_label or path.name
    stats = {
        "input_rows": 0,
        "retained_rows": 0,
        "same_label_duplicates": 0,
        "conflicting_keys": 0,
        "invalid_sequence_rows": 0,
        "invalid_label_type_rows": 0,
        "class_code_mismatch_rows": 0,
        "source_files": {source_name: 0},
        "split_type_counts": {},
    }
    errors: list[dict] = []
    rejected_rows: list[dict] = []
    by_key: dict[tuple[str, str, str], Stage8Record] = {}
    conflicts: dict[tuple[str, str, str], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_map = _canonical_field_map(reader.fieldnames or [])
        required = {"peptide", "tcr", "hla", "PH", "PT", "PHT"}
        if expected_class_code is not None:
            required.add("class_code")
        missing = required - set(field_map)
        if missing:
            raise Stage8DataError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            stats["input_rows"] += 1
            row = _canonical_row(row, field_map)
            try:
                record = _normalise_row(
                    row, row_number, source_file=source_name
                )
            except Stage8DataError as exc:
                errors.append({
                    "split": split, "source_file": source_name,
                    "row": row_number, "reason": str(exc),
                })
                continue
            if expected_class_code is not None:
                class_code = row.get("class_code", "").strip().upper()
                if class_code != expected_class_code or record.label_type != expected_class_code:
                    stats["class_code_mismatch_rows"] += 1
                    errors.append({
                        "split": split, "source_file": source_name,
                        "row": row_number,
                        "reason": (
                            f"class_code {class_code!r} and label type "
                            f"{record.label_type!r} do not match "
                            f"directory type {expected_class_code!r}"
                        ),
                    })
                    continue
            invalid_reason = _sequence_error(record, model, reject_noncanonical)
            if invalid_reason:
                stats["invalid_sequence_rows"] += 1
                rejected_rows.append({
                    "split": split, "source_file": source_name,
                    "row": row_number,
                    "reason": invalid_reason,
                    "peptide": record.peptide,
                    "tcr": record.tcr,
                    "hla": record.hla,
                })
                continue
            if record.label_type not in LABEL_TYPES:
                stats["invalid_label_type_rows"] += 1
                errors.append({
                    "split": split, "source_file": source_name,
                    "row": row_number,
                    "reason": f"unsupported label type {record.label_type}",
                })
                continue
            stats["source_files"][source_name] += 1
            if record.split_type:
                split_counts = stats["split_type_counts"]
                split_counts[record.split_type] = split_counts.get(record.split_type, 0) + 1
            previous = by_key.get(record.key)
            if previous is None:
                by_key[record.key] = record
            elif previous.labels == record.labels:
                stats["same_label_duplicates"] += 1
            else:
                item = conflicts.setdefault(record.key, {
                    "split": split,
                    "source_file": source_name,
                    "peptide": record.peptide,
                    "tcr": record.tcr,
                    "hla": record.hla,
                    "first_row": previous.source_row,
                    "first_source_file": previous.source_file,
                    "first_labels": previous.label_type,
                    "conflicting_rows": [],
                    "conflicting_source_files": [],
                    "conflicting_labels": [],
                })
                item["conflicting_rows"].append(row_number)
                item["conflicting_source_files"].append(source_name)
                item["conflicting_labels"].append(record.label_type)
    stats["conflicting_keys"] = len(conflicts)
    stats["retained_rows"] = len(by_key)
    serialised_conflicts = []
    for item in conflicts.values():
        item = dict(item)
        item["conflicting_rows"] = ";".join(map(str, item["conflicting_rows"]))
        item["conflicting_source_files"] = ";".join(item["conflicting_source_files"])
        item["conflicting_labels"] = ";".join(item["conflicting_labels"])
        serialised_conflicts.append(item)
    return list(by_key.values()), stats, serialised_conflicts, errors, rejected_rows


def _sequence_error(
    record: Stage8Record, model: ModelConfig, reject_noncanonical: bool
) -> str | None:
    if not model.peptide_min_length <= len(record.peptide) <= model.peptide_max_length:
        return "peptide length outside configured range"
    if not 1 <= len(record.hla) <= model.hla_max_length:
        return "HLA length outside configured range"
    if not 1 <= len(record.tcr) <= model.tcr_max_length:
        return "TCR length outside configured range"
    if reject_noncanonical and any(
        not set(sequence).issubset(CANONICAL_AMINO_ACIDS)
        for sequence in record.key
    ):
        return "noncanonical amino acid"
    return None


def _merge_split_parts(
    parts: Sequence[
        tuple[list[Stage8Record], dict, list[dict], list[dict], list[dict]]
    ],
    split: str,
) -> tuple[list[Stage8Record], dict, list[dict], list[dict], list[dict]]:
    stats = {
        "input_rows": 0,
        "retained_rows": 0,
        "same_label_duplicates": 0,
        "conflicting_keys": 0,
        "invalid_sequence_rows": 0,
        "invalid_label_type_rows": 0,
        "class_code_mismatch_rows": 0,
        "source_files": {},
        "split_type_counts": {},
    }
    errors: list[dict] = []
    rejected_rows: list[dict] = []
    conflicts: list[dict] = []
    by_key: dict[tuple[str, str, str], Stage8Record] = {}
    for records, part_stats, part_conflicts, part_errors, part_rejected in parts:
        for name in (
            "input_rows", "same_label_duplicates", "invalid_sequence_rows",
            "invalid_label_type_rows", "class_code_mismatch_rows",
        ):
            stats[name] += part_stats.get(name, 0)
        for source_name, count in part_stats.get("source_files", {}).items():
            stats["source_files"][source_name] = (
                stats["source_files"].get(source_name, 0) + count
            )
        for split_type, count in part_stats.get("split_type_counts", {}).items():
            stats["split_type_counts"][split_type] = (
                stats["split_type_counts"].get(split_type, 0) + count
            )
        conflicts.extend(part_conflicts)
        errors.extend(part_errors)
        rejected_rows.extend(part_rejected)
        for record in records:
            previous = by_key.get(record.key)
            if previous is None:
                by_key[record.key] = record
            elif previous.labels == record.labels:
                stats["same_label_duplicates"] += 1
            else:
                conflicts.append({
                    "split": split,
                    "source_file": record.source_file,
                    "peptide": record.peptide,
                    "tcr": record.tcr,
                    "hla": record.hla,
                    "first_row": previous.source_row,
                    "first_source_file": previous.source_file,
                    "first_labels": previous.label_type,
                    "conflicting_rows": str(record.source_row),
                    "conflicting_source_files": record.source_file,
                    "conflicting_labels": record.label_type,
                })
    stats["conflicting_keys"] = len(conflicts)
    stats["retained_rows"] = len(by_key)
    return list(by_key.values()), stats, conflicts, errors, rejected_rows


def _finalise_audit(
    train: list[Stage8Record],
    train_stats: dict,
    train_conflicts: list[dict],
    train_errors: list[dict],
    train_rejected: list[dict],
    val: list[Stage8Record],
    val_stats: dict,
    val_conflicts: list[dict],
    val_errors: list[dict],
    val_rejected: list[dict],
) -> Stage8AuditResult:
    conflicts = list(train_conflicts) + list(val_conflicts)
    errors = list(train_errors) + list(val_errors)
    rejected_rows = list(train_rejected) + list(val_rejected)
    ambiguous_keys = {
        (item.get("peptide"), item.get("tcr"), item.get("hla"))
        for item in conflicts
    }
    train_by_key = {record.key: record for record in train}
    retained_val: list[Stage8Record] = []
    overlap_removed = 0
    overlap_conflicts = 0
    for record in val:
        if record.key in ambiguous_keys:
            retained_val.append(record)
            continue
        train_record = train_by_key.get(record.key)
        if train_record is None:
            retained_val.append(record)
        elif train_record.labels == record.labels:
            overlap_removed += 1
        else:
            overlap_conflicts += 1
            conflicts.append({
                "split": "train_validation",
                "source_file": record.source_file,
                "peptide": record.peptide,
                "tcr": record.tcr,
                "hla": record.hla,
                "first_row": train_record.source_row,
                "first_source_file": train_record.source_file,
                "first_labels": train_record.label_type,
                "conflicting_rows": str(record.source_row),
                "conflicting_source_files": record.source_file,
                "conflicting_labels": record.label_type,
            })
    val_stats["train_overlap_removed"] = overlap_removed
    val_stats["train_overlap_conflicts"] = overlap_conflicts
    val_stats["retained_rows"] = len(retained_val)
    report = {
        "train": train_stats,
        "validation": val_stats,
        "label_type_counts": {
            "train": _type_counts(train),
            "validation": _type_counts(retained_val),
        },
        "source_files": {
            "train": train_stats.get("source_files", {}),
            "validation": val_stats.get("source_files", {}),
        },
        "split_type_counts": {
            "train": train_stats.get("split_type_counts", {}),
            "validation": val_stats.get("split_type_counts", {}),
        },
    }
    for split, records in (("train", train), ("validation", retained_val)):
        counts = _type_counts(records)
        for kind, count in counts.items():
            if count == 0:
                errors.append({
                    "split": split,
                    "row": "",
                    "reason": f"required label type {kind} is absent after cleanup",
                })
    return Stage8AuditResult(
        train, retained_val, report, conflicts, errors, rejected_rows,
        _fingerprint(train, retained_val),
    )


def audit_stage8_data(
    train_path: str | Path,
    val_path: str | Path,
    model: ModelConfig,
    *,
    reject_noncanonical_sequences: bool = True,
) -> Stage8AuditResult:
    train, train_stats, conflicts, errors, rejected_rows = _read_split(
        Path(train_path), "train", model, reject_noncanonical_sequences
    )
    val, val_stats, val_conflicts, val_errors, val_rejected = _read_split(
        Path(val_path), "validation", model, reject_noncanonical_sequences
    )
    return _finalise_audit(
        train, train_stats, conflicts, errors, rejected_rows,
        val, val_stats, val_conflicts, val_errors, val_rejected,
    )


def audit_stage8_class_dirs(
    class_root: str | Path,
    class_dirs: Sequence[str],
    train_filename: str,
    val_filename: str,
    model: ModelConfig,
    *,
    reject_noncanonical_sequences: bool = True,
) -> Stage8AuditResult:
    """Audit four PHT class directories as one Stage-8 dataset.

    Each class CSV must contain the regular sequence/label columns plus
    ``class_code``.  The directory name, class_code, and derived PH/PT/PHT
    label type must agree before a row is admitted to the unified dataset.
    """
    root = Path(class_root)
    directories = tuple(str(value) for value in class_dirs)
    expected_dirs = {f"PHT_{kind}" for kind in LABEL_TYPES}
    if set(directories) != expected_dirs or len(directories) != len(expected_dirs):
        raise Stage8DataError(
            "Stage-8 class_dirs must contain PHT_000, PHT_010, PHT_100, and PHT_111"
        )

    train_parts = []
    val_parts = []
    for class_dir in directories:
        class_code = class_dir.removeprefix("PHT_")
        train_parts.append(_read_split(
            root / class_dir / train_filename,
            "train",
            model,
            reject_noncanonical_sequences,
            expected_class_code=class_code,
            source_label=class_dir,
        ))
        val_parts.append(_read_split(
            root / class_dir / val_filename,
            "validation",
            model,
            reject_noncanonical_sequences,
            expected_class_code=class_code,
            source_label=class_dir,
        ))
    train = _merge_split_parts(train_parts, "train")
    val = _merge_split_parts(val_parts, "validation")
    return _finalise_audit(*train, *val)


def _type_counts(records: Sequence[Stage8Record]) -> dict[str, int]:
    return {kind: sum(record.label_type == kind for record in records) for kind in LABEL_TYPES}


def _fingerprint(train: Sequence[Stage8Record], val: Sequence[Stage8Record]) -> str:
    digest = hashlib.sha256()
    for split, records in (("train", train), ("validation", val)):
        for record in records:
            digest.update(
                (split + "\t" + "\t".join(record.key) + "\t" + record.label_type + "\n").encode()
            )
    return digest.hexdigest()


class BalancedFourTypeBatchSampler(Sampler[list[int]]):
    """Equal quarters from 111/100/010/000, cycling minority types."""

    def __init__(
        self, type_indices: Mapping[str, Sequence[int]], *, batch_size: int, seed: int
    ):
        if batch_size < 4 or batch_size % 4:
            raise ValueError("batch_size must be divisible by 4")
        missing = set(LABEL_TYPES) - set(type_indices)
        empty = [kind for kind in LABEL_TYPES if not type_indices.get(kind)]
        if missing or empty:
            raise ValueError("all four Stage-8 label types must contain samples")
        self.type_indices = {kind: list(type_indices[kind]) for kind in LABEL_TYPES}
        self.batch_size = batch_size
        self.per_type = batch_size // 4
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(max(map(len, self.type_indices.values())) / self.per_type)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        streams: dict[str, list[int]] = {}
        needed = len(self) * self.per_type
        for kind in LABEL_TYPES:
            source = self.type_indices[kind]
            stream: list[int] = []
            while len(stream) < needed:
                order = torch.randperm(len(source), generator=generator).tolist()
                stream.extend(source[index] for index in order)
            streams[kind] = stream[:needed]
        for batch_index in range(len(self)):
            start = batch_index * self.per_type
            batch = []
            for kind in LABEL_TYPES:
                batch.extend(streams[kind][start : start + self.per_type])
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


class OnlineStage8Dataset(Dataset[Stage8Record]):
    """Audited Stage-8 records retained as raw sequences for online encoders."""

    def __init__(self, records: Sequence[Stage8Record]):
        self.records = list(records)
        self.type_indices = {
            kind: [
                index
                for index, record in enumerate(self.records)
                if record.label_type == kind
            ]
            for kind in LABEL_TYPES
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Stage8Record:
        return self.records[index]


def collate_online_stage8(records: Sequence[Stage8Record]) -> dict:
    return {
        "peptide_sequences": [record.peptide for record in records],
        "hla_sequences": [record.hla for record in records],
        "tcr_sequences": [record.tcr for record in records],
        "ph_labels": torch.tensor([record.ph for record in records], dtype=torch.long),
        "pt_labels": torch.tensor([record.pt for record in records], dtype=torch.long),
        "pht_labels": torch.tensor([record.pht for record in records], dtype=torch.long),
        "label_types": [record.label_type for record in records],
        "source_rows": torch.tensor(
            [record.source_row for record in records], dtype=torch.long
        ),
    }


def make_stage8_online_loaders(
    audit: Stage8AuditResult,
    *,
    phla_path: str | Path,
    ptcr_path: str | Path,
    model: ModelConfig,
    micro_batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
    limit_per_type: int | None = None,
) -> tuple[dict[str, DataLoader], DataLoader]:
    """Build deterministic online PH/PT replay and balanced four-type loaders.

    The audit gate intentionally precedes the optional debug limit so a small
    smoke subset can never conceal a missing or conflicting four-type source.
    """

    audit.require_clean()
    if micro_batch_size < 4 or micro_batch_size % 4:
        raise ValueError("micro_batch_size must be positive and divisible by 4")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")

    train_dataset = OnlineStage8Dataset(
        _limit_by_type(audit.train_records, limit_per_type)
    )
    val_dataset = OnlineStage8Dataset(
        _limit_by_type(audit.val_records, limit_per_type)
    )
    joint_sampler = BalancedFourTypeBatchSampler(
        train_dataset.type_indices, batch_size=micro_batch_size, seed=seed
    )
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_online_stage8,
    }
    joint_loader = DataLoader(train_dataset, batch_sampler=joint_sampler, **common)
    validation_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, **common
    )
    pair_kwargs = {
        "batch_size": micro_batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "peptide_min_length": model.peptide_min_length,
        "peptide_max_length": model.peptide_max_length,
    }
    train = {
        "phla": make_online_dataloader(
            phla_path, "phla", seed=seed, **pair_kwargs
        ),
        "ptcr": make_online_dataloader(
            ptcr_path, "ptcr", seed=seed + 1, **pair_kwargs
        ),
        "joint": joint_loader,
    }
    return train, validation_loader


@dataclass(frozen=True)
class CachedStage8Sample:
    peptide_locator: CacheLocator
    hla_locator: CacheLocator
    tcr_locator: CacheLocator
    ph: int
    pt: int
    pht: int
    label_type: str
    source_row: int


class CachedStage8Dataset(Dataset[CachedStage8Sample]):
    """Compact sequence-to-cache locator mapping constructed once per run."""

    def __init__(
        self,
        records: Sequence[Stage8Record],
        caches: Mapping[str, EmbeddingCache],
    ):
        self.records = list(records)
        self.samples = [
            CachedStage8Sample(
                caches["peptide"].resolve(record.peptide),
                caches["hla"].resolve(record.hla),
                caches["tcr"].resolve(record.tcr),
                record.ph,
                record.pt,
                record.pht,
                record.label_type,
                record.source_row,
            )
            for record in self.records
        ]
        self.type_indices = {
            kind: [index for index, sample in enumerate(self.samples) if sample.label_type == kind]
            for kind in LABEL_TYPES
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CachedStage8Sample:
        return self.samples[index]


class Stage8CacheCollator:
    def __init__(
        self,
        caches: Mapping[str, EmbeddingCache],
        *,
        hidden_dtype: torch.dtype = torch.float32,
    ):
        self.caches = dict(caches)
        self.hidden_dtype = hidden_dtype

    def __call__(self, samples: Sequence[CachedStage8Sample]) -> dict:
        import numpy as np

        output: dict = {}
        for entity, attribute in (
            ("peptide", "peptide_locator"),
            ("hla", "hla_locator"),
            ("tcr", "tcr_locator"),
        ):
            hidden, mask = self.caches[entity].get_many(
                [getattr(sample, attribute) for sample in samples]
            )
            hidden_tensor = torch.from_numpy(np.asarray(hidden, dtype=np.float16))
            output[f"{entity}_hidden"] = hidden_tensor.to(self.hidden_dtype)
            output[f"{entity}_mask"] = torch.from_numpy(
                np.asarray(mask, dtype=np.bool_)
            )
        for name in ("ph", "pt", "pht"):
            output[f"{name}_labels"] = torch.tensor(
                [getattr(sample, name) for sample in samples], dtype=torch.long
            )
        output["label_types"] = [sample.label_type for sample in samples]
        output["source_rows"] = torch.tensor(
            [sample.source_row for sample in samples], dtype=torch.long
        )
        return output


def open_stage8_caches(
    cache_root: str | Path,
    *,
    source_sha256: str,
    model: ModelConfig,
    dataset_fingerprint: str | None = None,
) -> dict[str, EmbeddingCache]:
    root = Path(cache_root)
    dimensions = {
        "peptide": model.peptide_input_dim,
        "hla": model.hla_input_dim,
        "tcr": model.tcr_input_dim,
    }
    caches: dict[str, EmbeddingCache] = {}
    try:
        for entity, dimension in dimensions.items():
            caches[entity] = EmbeddingCache(
                root / entity,
                expected_hidden_dim=dimension,
                expected_fingerprint=source_sha256,
            )
            if caches[entity].manifest.get("chunk_rows") != 1:
                raise ValueError(
                    f"{entity} Stage-8 cache uses an obsolete layout; "
                    "regenerate it with row-sized chunks"
                )
    except Exception:
        for cache in caches.values():
            cache.close()
        raise
    if dataset_fingerprint is not None:
        for entity, cache in caches.items():
            cached = cache.manifest.get("model", {}).get("dataset_fingerprint")
            if cached != dataset_fingerprint:
                for value in caches.values():
                    value.close()
                raise ValueError(
                    f"{entity} Stage-8 cache dataset fingerprint mismatch"
                )
    return caches


def _limit_by_type(
    records: Sequence[Stage8Record], limit_per_type: int | None
) -> list[Stage8Record]:
    if limit_per_type is None:
        return list(records)
    if limit_per_type < 1:
        raise ValueError("limit_per_type must be positive")
    counts = {kind: 0 for kind in LABEL_TYPES}
    retained = []
    for record in records:
        if counts[record.label_type] < limit_per_type:
            retained.append(record)
            counts[record.label_type] += 1
    return retained


def make_stage8_cached_loaders(
    audit: Stage8AuditResult,
    caches: Mapping[str, EmbeddingCache],
    *,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
    limit_per_type: int | None = None,
    hidden_dtype: torch.dtype = torch.float32,
) -> tuple[DataLoader, DataLoader]:
    audit.require_clean()
    train_dataset = CachedStage8Dataset(
        _limit_by_type(audit.train_records, limit_per_type), caches
    )
    val_dataset = CachedStage8Dataset(
        _limit_by_type(audit.val_records, limit_per_type), caches
    )
    sampler = BalancedFourTypeBatchSampler(
        train_dataset.type_indices, batch_size=batch_size, seed=seed
    )
    collator = Stage8CacheCollator(caches, hidden_dtype=hidden_dtype)
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collator,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, **common
    )
    return train_loader, val_loader
