"""Audited datasets for Stage-3 versus Stage-1 Stage-7 evaluation."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import normalise_sequence
from .stage7_config import Stage7ExperimentConfig
from .triplet_data import (
    OnlineTripletDataset,
    TripletDatasetBundle,
    TripletRecord,
    build_strict_triplet_datasets,
)


@dataclass(frozen=True)
class Stage7Member:
    split: str
    experiment: str
    group_id: str
    role: str
    peptide: str
    tcr: str
    hla: str
    label: int | None
    source_row: int | None
    evidence: str
    peptide_group: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HlaMetadata:
    species: str
    mhc_class: str


@dataclass
class Stage7CounterfactualBundle:
    triplets: TripletDatasetBundle
    members: dict[str, list[Stage7Member]]
    audit: dict


def _original_triplet_members(
    split: str,
    dataset: OnlineTripletDataset,
    *,
    limit: int | None = None,
) -> list[Stage7Member]:
    """Return each labelled triplet once for the primary performance test."""

    records = dataset.records[:limit] if limit is not None else dataset.records
    return [
        Stage7Member(
            split=split,
            experiment="original_triplet",
            group_id=f"{split}:original:{record.source_row}",
            role="original",
            peptide=record.peptide,
            tcr=record.tcr,
            hla=record.hla,
            label=record.label,
            source_row=record.source_row,
            evidence=f"{dataset.csv_path.name}:{record.source_row}:label_{record.label}",
            peptide_group=record.peptide,
        )
        for record in records
    ]


def load_hla_metadata(path: str | Path) -> dict[str, HlaMetadata]:
    """Read Stage-7-owned pseudo-sequence metadata when explicitly supplied."""

    source = Path(path).resolve()
    result: dict[str, HlaMetadata] = {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"hla", "species", "mhc_class"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{source} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            hla = normalise_sequence(row["hla"])
            metadata = HlaMetadata(
                species=str(row["species"]).strip(),
                mhc_class=str(row["mhc_class"]).strip().upper(),
            )
            if not metadata.species or not metadata.mhc_class:
                raise ValueError(f"Empty HLA metadata at {source}:{row_number}")
            previous = result.get(hla)
            if previous is not None and previous != metadata:
                raise ValueError(f"Conflicting HLA metadata for {hla!r}")
            result[hla] = metadata
    return result


def _read_positive_phla_pairs(path: Path) -> tuple[dict[str, dict[str, str]], dict]:
    """Return unambiguous positive HLA candidates keyed by peptide and HLA."""

    labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    first_source: dict[tuple[str, str], str] = {}
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", "HLA", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            peptide = normalise_sequence(row["peptide"])
            hla = normalise_sequence(row["HLA"])
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid pHLA label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"pHLA label must be 0/1 at {path}:{row_number}")
            key = (peptide, hla)
            labels[key].add(label)
            first_source.setdefault(key, f"{path.name}:{row_number}")
    positives: dict[str, dict[str, str]] = defaultdict(dict)
    ambiguous = 0
    for (peptide, hla), values in labels.items():
        if values == {1}:
            positives[peptide][hla] = first_source[(peptide, hla)]
        elif len(values) > 1:
            ambiguous += 1
    return dict(positives), {
        "path": str(path.resolve()),
        "rows": rows,
        "unique_pairs": len(labels),
        "positive_pairs": sum(len(values) for values in positives.values()),
        "ambiguous_pairs_excluded": ambiguous,
    }


def _metadata_compatible(
    cognate: str,
    decoy: str,
    metadata: Mapping[str, HlaMetadata] | None,
) -> bool:
    if metadata is None:
        return True
    if cognate not in metadata or decoy not in metadata:
        return False
    return metadata[cognate] == metadata[decoy]


def _counterfactual_pairs(
    split: str,
    dataset: OnlineTripletDataset,
    *,
    phla_positives: Mapping[str, Mapping[str, str]],
    triplet_train_hlas: set[str],
    triplet_train_hla_frequency: Counter[str],
    known_positive_triplets: set[tuple[str, str, str]],
    max_decoys: int,
    require_seen_hla: bool,
    metadata: Mapping[str, HlaMetadata] | None,
    limit_pairs: int | None,
) -> tuple[list[Stage7Member], dict]:
    members: list[Stage7Member] = []
    audit = Counter()
    anchors = [record for record in dataset.records if record.label == 1]
    if limit_pairs is not None:
        anchors = anchors[:limit_pairs]
    audit["positive_anchors_considered"] = len(anchors)
    for record in anchors:
        candidates = []
        for hla, source in phla_positives.get(record.peptide, {}).items():
            audit["candidate_pairs_seen"] += 1
            if hla == record.hla:
                audit["same_hla_excluded"] += 1
                continue
            if require_seen_hla and hla not in triplet_train_hlas:
                audit["unseen_triplet_train_hla_excluded"] += 1
                continue
            if (record.peptide, record.tcr, hla) in known_positive_triplets:
                audit["known_positive_triplet_excluded"] += 1
                continue
            if not _metadata_compatible(record.hla, hla, metadata):
                audit["metadata_mismatch_excluded"] += 1
                continue
            frequency_distance = abs(
                math.log1p(triplet_train_hla_frequency[hla])
                - math.log1p(triplet_train_hla_frequency[record.hla])
            )
            candidates.append((frequency_distance, hla, source))
        candidates.sort(key=lambda item: (item[0], item[1]))
        candidates = candidates[:max_decoys]
        if not candidates:
            audit["anchors_without_decoy"] += 1
            continue
        audit["anchors_with_decoy"] += 1
        audit["selected_decoys"] += len(candidates)
        group_id = f"{split}:pair:{record.source_row}"
        members.append(
            Stage7Member(
                split=split,
                experiment="phla_valid_pair",
                group_id=group_id,
                role="cognate",
                peptide=record.peptide,
                tcr=record.tcr,
                hla=record.hla,
                label=1,
                source_row=record.source_row,
                evidence=f"{dataset.csv_path.name}:{record.source_row}:label_1",
                peptide_group=record.peptide,
            )
        )
        for decoy_rank, (_, hla, source) in enumerate(candidates, start=1):
            members.append(
                Stage7Member(
                    split=split,
                    experiment="phla_valid_pair",
                    group_id=group_id,
                    role=f"decoy_{decoy_rank}",
                    peptide=record.peptide,
                    tcr=record.tcr,
                    hla=hla,
                    label=None,
                    source_row=record.source_row,
                    evidence=f"phla_train_positive:{source}",
                    peptide_group=record.peptide,
                )
            )
    return members, dict(audit)


def _valid_tetrad_edges(
    records: Sequence[TripletRecord],
    known_positive_triplets: set[tuple[str, str, str]],
) -> list[tuple[int, int]]:
    edges = []
    for first, second in combinations(range(len(records)), 2):
        left = records[first]
        right = records[second]
        if left.tcr == right.tcr or left.hla == right.hla:
            continue
        offdiag = (
            (left.peptide, left.tcr, right.hla),
            (left.peptide, right.tcr, left.hla),
        )
        if any(key in known_positive_triplets for key in offdiag):
            continue
        edges.append((first, second))
    return edges


def _maximum_disjoint_edges(
    records: Sequence[TripletRecord],
    edges: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Stage 7 requires networkx>=3.1; install requirements-stage7.txt"
        ) from exc
    graph = nx.Graph()
    graph.add_nodes_from(range(len(records)))
    ordered = sorted(
        edges,
        key=lambda edge: (
            min(records[edge[0]].hla, records[edge[1]].hla),
            max(records[edge[0]].hla, records[edge[1]].hla),
            records[edge[0]].source_row,
            records[edge[1]].source_row,
        ),
    )
    edge_count = len(ordered)
    for rank, (first, second) in enumerate(ordered):
        graph.add_edge(first, second, weight=edge_count - rank)
    matching = nx.algorithms.matching.max_weight_matching(
        graph,
        maxcardinality=True,
        weight="weight",
    )
    return sorted(
        (min(first, second), max(first, second)) for first, second in matching
    )


def _tetrad_members(
    split: str,
    peptide: str,
    records: Sequence[TripletRecord],
    edges: Sequence[tuple[int, int]],
    *,
    experiment: str,
) -> list[Stage7Member]:
    members: list[Stage7Member] = []
    for first_index, second_index in edges:
        first = records[first_index]
        second = records[second_index]
        source_low, source_high = sorted((first.source_row, second.source_row))
        group_id = f"{split}:{experiment}:{source_low}:{source_high}"
        definitions = (
            ("diag_1", first.tcr, first.hla, first.source_row, 1, "triplet_label_1"),
            ("diag_2", second.tcr, second.hla, second.source_row, 1, "triplet_label_1"),
            (
                "offdiag_12",
                first.tcr,
                second.hla,
                None,
                None,
                "unobserved_cross_hla_decoy",
            ),
            (
                "offdiag_21",
                second.tcr,
                first.hla,
                None,
                None,
                "unobserved_cross_hla_decoy",
            ),
        )
        for role, tcr, hla, source_row, label, evidence in definitions:
            members.append(
                Stage7Member(
                    split=split,
                    experiment=experiment,
                    group_id=group_id,
                    role=role,
                    peptide=peptide,
                    tcr=tcr,
                    hla=hla,
                    label=label,
                    source_row=source_row,
                    evidence=evidence,
                    peptide_group=peptide,
                )
            )
    return members


def _counterfactual_tetrads(
    split: str,
    dataset: OnlineTripletDataset,
    *,
    known_positive_triplets: set[tuple[str, str, str]],
    include_all: bool,
    limit_tetrads: int | None,
) -> tuple[list[Stage7Member], dict]:
    positive_by_peptide: dict[str, list[TripletRecord]] = defaultdict(list)
    for record in dataset.records:
        if record.label == 1:
            positive_by_peptide[record.peptide].append(record)
    members: list[Stage7Member] = []
    audit = Counter()
    for peptide in sorted(positive_by_peptide):
        records = sorted(
            positive_by_peptide[peptide], key=lambda record: record.source_row
        )
        if len({record.hla for record in records}) < 2:
            continue
        audit["peptides_with_multiple_positive_hla"] += 1
        edges = _valid_tetrad_edges(records, known_positive_triplets)
        if not edges:
            continue
        audit["peptides_with_valid_tetrads"] += 1
        audit["all_valid_tetrads"] += len(edges)
        disjoint = _maximum_disjoint_edges(records, edges)
        audit["disjoint_tetrads"] += len(disjoint)
        members.extend(
            _tetrad_members(
                split,
                peptide,
                records,
                disjoint,
                experiment="tetrad_disjoint",
            )
        )
        if include_all:
            members.extend(
                _tetrad_members(
                    split,
                    peptide,
                    records,
                    edges,
                    experiment="tetrad_all",
                )
            )
    if limit_tetrads is not None:
        keep = set()
        returned_by_experiment = {}
        for experiment in ("tetrad_disjoint", "tetrad_all"):
            selected_groups = []
            seen = set()
            for member in members:
                if member.experiment != experiment or member.group_id in seen:
                    continue
                seen.add(member.group_id)
                selected_groups.append(member.group_id)
            chosen = selected_groups[:limit_tetrads]
            keep.update(chosen)
            returned_by_experiment[experiment] = len(chosen)
        members = [member for member in members if member.group_id in keep]
        audit["debug_limit_tetrads"] = limit_tetrads
        audit["returned_tetrads"] = returned_by_experiment
    else:
        audit["returned_tetrads"] = {
            experiment: len(
                {
                    member.group_id
                    for member in members
                    if member.experiment == experiment
                }
            )
            for experiment in ("tetrad_disjoint", "tetrad_all")
        }
    return members, dict(audit)


def _legacy_shift_members(
    split: str,
    dataset: OnlineTripletDataset,
    *,
    limit: int | None = None,
) -> tuple[list[Stage7Member], dict]:
    records = dataset.records[:limit] if limit is not None else dataset.records
    if len(records) < 2:
        raise ValueError("Legacy HLA shift requires at least two records")
    offset = max(1, len(records) // 2)
    members = []
    changed = 0
    changed_by_label = Counter()
    label_counts = Counter(record.label for record in records)
    for index, original in enumerate(records):
        shifted_hla = records[(index + offset) % len(records)].hla
        is_changed = shifted_hla != original.hla
        changed += int(is_changed)
        changed_by_label[original.label] += int(is_changed)
        group_id = f"{split}:legacy_shift:{original.source_row}"
        for role, hla, evidence in (
            ("original", original.hla, "original_triplet"),
            ("shifted", shifted_hla, "deterministic_half_offset_hla"),
        ):
            members.append(
                Stage7Member(
                    split=split,
                    experiment="legacy_hla_shift",
                    group_id=group_id,
                    role=role,
                    peptide=original.peptide,
                    tcr=original.tcr,
                    hla=hla,
                    label=original.label,
                    source_row=original.source_row,
                    evidence=evidence,
                    peptide_group=original.peptide,
                )
            )
    return members, {
        "rows": len(records),
        "debug_limit": limit,
        "offset": offset,
        "changed": changed,
        "changed_fraction": changed / len(records),
        "changed_fraction_label_0": (
            changed_by_label[0] / label_counts[0] if label_counts[0] else None
        ),
        "changed_fraction_label_1": (
            changed_by_label[1] / label_counts[1] if label_counts[1] else None
        ),
    }


def shortcut_lookup_audit(
    train: OnlineTripletDataset,
    dataset: OnlineTripletDataset,
) -> dict[str, dict[str, float | int]]:
    """Measure simple training-frequency shortcuts on one evaluation split."""

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for Stage-7 audit") from exc
    definitions = {
        "hla": lambda record: record.hla,
        "peptide": lambda record: record.peptide,
        "phla": lambda record: (record.peptide, record.hla),
    }
    labels = np.asarray([record.label for record in dataset.records], dtype=np.int64)
    result = {}
    for name, key_fn in definitions.items():
        counts: dict[object, list[int]] = defaultdict(lambda: [0, 0])
        for record in train.records:
            counts[key_fn(record)][record.label] += 1
        scores = []
        seen = 0
        for record in dataset.records:
            key = key_fn(record)
            if key in counts:
                seen += 1
            negative, positive = counts.get(key, (0, 0))
            scores.append((positive + 1) / (negative + positive + 2))
        values = np.asarray(scores, dtype=np.float64)
        result[name] = {
            "auroc": float(roc_auc_score(labels, values)),
            "aupr": float(average_precision_score(labels, values)),
            "seen_rows": seen,
            "samples": len(labels),
            "seen_fraction": seen / len(labels),
        }
    return result


def build_stage7_counterfactual_bundle(
    config: Stage7ExperimentConfig,
    *,
    splits: Iterable[str] = ("val", "test"),
    limit_pairs: int | None = None,
    limit_tetrads: int | None = None,
) -> Stage7CounterfactualBundle:
    """Build all Stage-7 derived datasets without modifying source CSV files."""

    requested = tuple(dict.fromkeys(splits))
    if not requested or any(split not in {"val", "test"} for split in requested):
        raise ValueError("Stage-7 splits must be a non-empty subset of val,test")
    values = config.stage7
    triplets = build_strict_triplet_datasets(
        config.paths.data_root,
        data_subdir=values.triplet_data_subdir,
        merged_filename=values.triplet_merged_filename,
        train_filename=values.triplet_train_filename,
        val_filename=values.triplet_val_filename,
        test_filename=values.triplet_test_filename,
        verify_manifest=values.verify_merged_manifest,
        strict_split_cleanup=values.strict_split_cleanup,
        peptide_min_length=config.model.peptide_min_length,
        peptide_max_length=config.model.peptide_max_length,
        hla_max_length=config.model.hla_max_length,
        tcr_max_length=config.model.tcr_max_length,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
    )
    phla_positives, phla_audit = _read_positive_phla_pairs(
        config.phla_train_path()
    )
    metadata = None
    if values.hla_metadata_csv is not None:
        metadata = load_hla_metadata(values.hla_metadata_csv)
        if values.metadata_mode == "required":
            all_hlas = {
                record.hla
                for dataset in triplets.as_dict().values()
                for record in dataset.records
            }
            all_hlas.update(
                hla for candidates in phla_positives.values() for hla in candidates
            )
            missing = sorted(all_hlas - set(metadata))
            if missing:
                raise ValueError(
                    "Required HLA metadata is incomplete; missing "
                    f"{len(missing)} pseudo-sequences"
                )
    known_positive_triplets = {
        record.key
        for dataset in triplets.as_dict().values()
        for record in dataset.records
        if record.label == 1
    }
    train_hlas = {record.hla for record in triplets.train.records}
    train_hla_frequency = Counter(record.hla for record in triplets.train.records)
    members_by_split: dict[str, list[Stage7Member]] = {}
    audit = {
        "manifest": triplets.manifest_audit,
        "triplet_filter_stats": {
            name: dataset.filter_stats for name, dataset in triplets.as_dict().items()
        },
        "phla_candidates": phla_audit,
        "metadata_mode": values.metadata_mode,
        "metadata_path": str(values.hla_metadata_csv) if values.hla_metadata_csv else None,
        "splits": {},
    }
    for split in requested:
        dataset = triplets.val if split == "val" else triplets.test
        debug_limits = [
            value for value in (limit_pairs, limit_tetrads) if value is not None
        ]
        original_limit = max(debug_limits) if debug_limits else None
        original_members = _original_triplet_members(
            split,
            dataset,
            limit=original_limit,
        )
        pair_members, pair_audit = _counterfactual_pairs(
            split,
            dataset,
            phla_positives=phla_positives,
            triplet_train_hlas=train_hlas,
            triplet_train_hla_frequency=train_hla_frequency,
            known_positive_triplets=known_positive_triplets,
            max_decoys=values.max_decoys_per_anchor,
            require_seen_hla=values.require_decoy_hla_seen_in_triplet_train,
            metadata=metadata,
            limit_pairs=limit_pairs,
        )
        tetrad_members, tetrad_audit = _counterfactual_tetrads(
            split,
            dataset,
            known_positive_triplets=known_positive_triplets,
            include_all=values.include_all_tetrads,
            limit_tetrads=limit_tetrads,
        )
        legacy_members: list[Stage7Member] = []
        legacy_audit = None
        if values.include_legacy_hla_shift:
            legacy_members, legacy_audit = _legacy_shift_members(
                split,
                dataset,
                limit=original_limit,
            )
        members_by_split[split] = (
            original_members + pair_members + tetrad_members + legacy_members
        )
        audit["splits"][split] = {
            "shortcut_lookup": shortcut_lookup_audit(triplets.train, dataset),
            "original_triplets": {
                "rows": len(original_members),
                "debug_limit": original_limit,
            },
            "phla_valid_pairs": pair_audit,
            "tetrads": tetrad_audit,
            "legacy_hla_shift": legacy_audit,
            "prediction_rows": len(members_by_split[split]),
        }
    return Stage7CounterfactualBundle(
        triplets=triplets,
        members=members_by_split,
        audit=audit,
    )
