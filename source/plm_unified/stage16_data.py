"""Audited row identities and restartable, nested replay sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import normalise_sequence, resolve_task_columns
from .stage8_data import (
    CANONICAL_AMINO_ACIDS,
    OnlineStage8Dataset,
    BalancedFourTypeBatchSampler,
    audit_stage8_class_dirs,
    collate_online_stage8,
)
from .stage16_checkpoint import checkpoint_sha256, digest
from .stage16_selection import PT_SPLITS


class PairRows(Dataset):
    def __init__(self, path, task, model, *, allow_test=False):
        self.path, self.task = Path(path).resolve(), task
        if self.path.stem.startswith("test") and not allow_test:
            raise ValueError(
                "Test files cannot enter training/validation/cache preparation"
            )
        self.file_sha256 = checkpoint_sha256(self.path)
        self.rows, self.row_ids, self.keys, self.rejected = [], [], [], []
        seen = defaultdict(set)
        aliases = defaultdict(set)
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            pcol, rcol, ycol = resolve_task_columns(task, reader.fieldnames)
            if not {pcol, rcol, ycol} <= set(reader.fieldnames or ()):
                raise ValueError(f"Invalid header: {self.path}")
            total = 0
            for line, row in enumerate(reader, 2):
                total += 1
                try:
                    p, r, y = (
                        normalise_sequence(row[pcol]),
                        normalise_sequence(row[rcol]),
                        int(row[ycol]),
                    )
                    if y not in (0, 1):
                        raise ValueError("nonbinary label")
                    if (
                        not model.peptide_min_length
                        <= len(p)
                        <= model.peptide_max_length
                    ):
                        raise ValueError("peptide length")
                    if len(r) > (
                        model.hla_max_length if task == "phla" else model.tcr_max_length
                    ):
                        raise ValueError("receptor length")
                    if set(p + r) - CANONICAL_AMINO_ACIDS:
                        raise ValueError("noncanonical sequence")
                except (ValueError, TypeError) as error:
                    self.rejected.append({"source_row": line, "reason": str(error)})
                    continue
                seen[p, r].add(y)
                aliases[p, r].add((row[pcol], row[rcol]))
                self.rows.append((p, r, y))
                self.row_ids.append(f"{self.file_sha256}:{line}")
                self.keys.append(digest([p, r]))
        if not self.rows:
            raise ValueError(f"Empty usable dataset: {self.path}")
        conflicts = [key for key, labels in seen.items() if len(labels) > 1]
        if conflicts:
            raise ValueError(
                f"Conflicting pair labels in {self.path}: {len(conflicts)} keys"
            )
        self.manifest = {
            "path": str(self.path),
            "sha256": self.file_sha256,
            "task": task,
            "raw_rows": total,
            "samples": len(self.rows),
            "positive_samples": sum(row[2] for row in self.rows),
            "negative_samples": sum(1 - row[2] for row in self.rows),
            "rejected": self.rejected,
            "duplicate_pairs": len(self.rows) - len(seen),
            "row_ids": self.row_ids,
            "normalized_input_aliases": sum(
                len(values) > 1 for values in aliases.values()
            ),
            "truncation_policy": "reject_overlength_no_truncation",
            "input_keys_sha256": digest(self.keys),
            "normalization": "canonical-upper-no-gaps-v1",
        }

    def subset(self, indices):
        result = object.__new__(PairRows)
        result.__dict__ = dict(self.__dict__)
        result.rows = [self.rows[i] for i in indices]
        result.row_ids = [self.row_ids[i] for i in indices]
        result.keys = [self.keys[i] for i in indices]
        result.manifest = {
            **self.manifest,
            "parent_manifest_sha256": digest(self.manifest),
            "samples": len(result.rows),
            "positive_samples": sum(r[2] for r in result.rows),
            "negative_samples": sum(1 - r[2] for r in result.rows),
            "row_ids": result.row_ids,
            "input_keys_sha256": digest(result.keys),
        }
        return result

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        p, r, y = self.rows[index]
        return {
            "peptide": p,
            "receptor": r,
            "label": y,
            "row_id": self.row_ids[index],
            "input_key": self.keys[index],
        }


def collate_pairs(rows):
    return {
        "peptide_sequences": [r["peptide"] for r in rows],
        "receptor_sequences": [r["receptor"] for r in rows],
        "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        "row_ids": [r["row_id"] for r in rows],
        "input_keys": [r["input_key"] for r in rows],
    }


def stratified_indices(rows, limit, seed):
    generator = np.random.default_rng(seed)
    labels = np.array([row[2] for row in rows])
    groups = [np.flatnonzero(labels == k) for k in (0, 1)]
    if limit >= len(rows):
        return list(range(len(rows)))
    quotas = [min(len(groups[0]), limit // 2), min(len(groups[1]), limit - limit // 2)]
    remaining = limit - sum(quotas)
    for k in (0, 1):
        add = min(remaining, len(groups[k]) - quotas[k])
        quotas[k] += add
        remaining -= add
    return sorted(
        int(i) for k in (0, 1) for i in generator.permutation(groups[k])[: quotas[k]]
    )


class ReplayBatchSampler(Sampler):
    def __init__(self, size, cap, batch_size, seed, *, epoch=0, start_batch=0):
        self.size, self.cap, self.batch_size, self.seed = (
            size,
            min(size, cap),
            batch_size,
            seed,
        )
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        order = torch.randperm(
            self.size,
            generator=torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch),
        ).tolist()[: self.cap]
        for start in range(
            self.start_batch * self.batch_size, self.cap, self.batch_size
        ):
            yield order[start : start + self.batch_size]

    def __len__(self):
        return max(0, math.ceil(self.cap / self.batch_size) - self.start_batch)


class OffsetBatchSampler(Sampler):
    def __init__(self, sampler, start_batch=0):
        self.sampler, self.start_batch = sampler, start_batch
        self.epoch = sampler.epoch

    def __iter__(self):
        for i, batch in enumerate(self.sampler):
            if i >= self.start_batch:
                yield batch

    def __len__(self):
        return max(0, len(self.sampler) - self.start_batch)


def collate_joint(records):
    result = collate_online_stage8(records)
    result["row_ids"] = [f"{r.source_file}:{r.source_row}" for r in records]
    result["split_types"] = [r.split_type for r in records]
    return result


def pair_loader(dataset, config, *, cap=None, epoch=0, start_batch=0, train=False):
    generator = torch.Generator().manual_seed(config.runtime.seed + 77)
    kwargs = dict(
        num_workers=config.runtime.num_workers,
        collate_fn=collate_pairs,
        generator=generator,
        persistent_workers=False,
        pin_memory=config.runtime.device.startswith("cuda"),
    )
    if train:
        sampler = ReplayBatchSampler(
            len(dataset),
            cap,
            config.training.micro_batch_size,
            config.runtime.seed + (1 if dataset.task == "ptcr" else 0),
            epoch=epoch,
            start_batch=start_batch,
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(
        dataset, batch_size=config.training.eval_batch_size, shuffle=False, **kwargs
    )


def joint_loader(records, config, *, epoch=0, start_batch=0, train=False):
    dataset = OnlineStage8Dataset(records)
    kwargs = dict(
        num_workers=config.runtime.num_workers,
        collate_fn=collate_joint,
        generator=torch.Generator().manual_seed(config.runtime.seed + 78),
        persistent_workers=False,
    )
    if train:
        sampler = BalancedFourTypeBatchSampler(
            dataset.type_indices,
            batch_size=config.training.micro_batch_size,
            seed=config.runtime.seed,
        )
        sampler.set_epoch(
            0 if config.training.sampler_policy == "legacy_order" else epoch
        )
        return DataLoader(
            dataset, batch_sampler=OffsetBatchSampler(sampler, start_batch), **kwargs
        )
    return DataLoader(
        dataset, batch_size=config.training.eval_batch_size, shuffle=False, **kwargs
    )


def audit_experiment_data(config):
    root, model = config.paths.data_root, config.model
    classes = ("PHT_000", "PHT_010", "PHT_100", "PHT_111")
    audit = audit_stage8_class_dirs(
        root / "PHT/classes",
        classes,
        "train.csv",
        "val.csv",
        model,
        reject_noncanonical_sequences=True,
    )
    audit.require_clean()
    pairs = {
        task: PairRows(root / family / "train.csv", task, model)
        for task, family in (("phla", "PH"), ("ptcr", "PT"))
    }
    pt_val = {
        s: PairRows(root / "PT/benchmarks" / f"{s}.csv", "ptcr", model)
        for s in PT_SPLITS
    }
    ph_splits = ("val_unseen_hla", "val_unseen_pair", "val_unseen_peptide")
    ph_val = {
        s: PairRows(root / "PH/benchmarks" / f"{s}.csv", "phla", model)
        for s in ph_splits
    }
    overlap = {}
    # Audit pair-level leakage against ALL training tasks, not only the PHT table.
    pht_pt = {(r.peptide, r.tcr) for r in audit.train_records}
    pht_ph = {(r.peptide, r.hla) for r in audit.train_records}
    all_peptides = {p for data in pairs.values() for p, _, _ in data.rows} | {
        r.peptide for r in audit.train_records
    }
    violations = []
    cross_labels = {}
    for task, attribute in (("ptcr", "tcr"), ("phla", "hla")):
        pair_labels = {(p, r): y for p, r, y in pairs[task].rows}
        projection = defaultdict(set)
        for record in audit.train_records:
            key = (record.peptide, getattr(record, attribute))
            label = record.pt if task == "ptcr" else record.ph
            projection[key].add(label)
        internal = sum(len(labels) > 1 for labels in projection.values())
        cross = sum(
            key in pair_labels and pair_labels[key] not in labels
            for key, labels in projection.items()
        )
        cross_labels[task] = {
            "pht_projected_conflicting_keys": internal,
            "conflicting_with_binary_train": cross,
        }
        if internal or cross:
            violations.append(f"{task}:projected_label_conflict")
    for task, validations, pht_pairs in (
        ("ptcr", pt_val, pht_pt),
        ("phla", ph_val, pht_ph),
    ):
        train_pairs = {(p, r) for p, r, _ in pairs[task].rows} | pht_pairs
        train_receptors = {r for _, r in train_pairs}
        for split, data in validations.items():
            values = {
                "pair_rows": sum((p, r) in train_pairs for p, r, _ in data.rows),
                "peptide_rows": sum(p in all_peptides for p, _, _ in data.rows),
                "receptor_rows": sum(r in train_receptors for _, r, _ in data.rows),
            }
            overlap[f"{task}/{split}"] = values
            if values["pair_rows"]:
                violations.append(f"{task}/{split}:training_pair_overlap")
            if split.endswith("peptide") and values["peptide_rows"]:
                # Cross-task exposure is reported explicitly; it is not a pair label leak.
                values["cross_task_unseen_warning"] = True
            if split.endswith(("tcr", "hla")) and values["receptor_rows"]:
                violations.append(f"{task}/{split}:heldout_receptor_overlap")
    # PHT holdouts are also checked against the binary replay sources. A
    # triple holdout permits known constituent entities, but not a known triple.
    train_entities = {
        "peptide": all_peptides,
        "tcr": {r for _, r, _ in pairs["ptcr"].rows}
        | {r.tcr for r in audit.train_records},
        "hla": {r for _, r, _ in pairs["phla"].rows}
        | {r.hla for r in audit.train_records},
    }
    train_triples = {r.key for r in audit.train_records}
    for split in sorted({r.split_type for r in audit.val_records}):
        records = [r for r in audit.val_records if r.split_type == split]
        values = {
            entity + "_rows": sum(getattr(r, entity) in entities for r in records)
            for entity, entities in train_entities.items()
        }
        values["triple_rows"] = sum(r.key in train_triples for r in records)
        overlap["pht/" + split] = values
        if values["triple_rows"]:
            violations.append(f"pht/{split}:training_triple_overlap")
        heldout = split.removeprefix("val_unseen_")
        if heldout in train_entities and values[heldout + "_rows"]:
            values["cross_task_unseen_warning"] = True
    manifest = {
        "version": 1,
        "model_input": asdict(model),
        "pht_fingerprint": audit.dataset_fingerprint,
        "pht_audit": audit.report,
        "pht_rejected": audit.rejected_rows,
        "pht_files": {
            str(root / "PHT/classes" / c / f): checkpoint_sha256(
                root / "PHT/classes" / c / f
            )
            for c in classes
            for f in ("train.csv", "val.csv")
        },
        "pairs": {k: v.manifest for k, v in pairs.items()},
        "pt_validation": {k: v.manifest for k, v in pt_val.items()},
        "ph_validation": {k: v.manifest for k, v in ph_val.items()},
        "overlap": overlap,
        "projected_label_audit": cross_labels,
        "violations": violations,
        "debug": config.debug,
    }
    if violations:
        raise ValueError("Dataset leakage audit failed: " + ", ".join(violations))

    # Apply debug limits only after the full audit; the manifest retains full provenance.
    def subset(data):
        return (
            data.subset(
                stratified_indices(
                    data.rows, config.training.debug_pair_limit, config.runtime.seed
                )
            )
            if config.training.debug_pair_limit
            else data
        )

    pairs = {k: subset(v) for k, v in pairs.items()}
    pt_val = {k: subset(v) for k, v in pt_val.items()}
    ph_val = {k: subset(v) for k, v in ph_val.items()}
    train_records, val_records = audit.train_records, audit.val_records
    if config.training.debug_limit_per_type:

        def limit(records):
            counts = Counter()
            out = []
            for r in records:
                if counts[r.label_type] < config.training.debug_limit_per_type:
                    out.append(r)
                    counts[r.label_type] += 1
            return out

        train_records, val_records = limit(train_records), limit(val_records)
    manifest["effective"] = {
        "pair_rows": {k: v.row_ids for k, v in pairs.items()},
        "pt_val_rows": {k: v.row_ids for k, v in pt_val.items()},
        "joint_train": len(train_records),
        "joint_val": len(val_records),
    }
    return {
        "pairs": pairs,
        "pt_val": pt_val,
        "ph_val": ph_val,
        "joint_train": train_records,
        "joint_val": val_records,
        "manifest": manifest,
    }
