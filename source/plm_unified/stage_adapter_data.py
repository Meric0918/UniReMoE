"""PHT-only online loaders backed by the strict Stage-8 four-type audit."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .stage8_data import BalancedFourTypeBatchSampler, Stage8AuditResult, Stage8Record


class StageAdapterDataset(Dataset[Stage8Record]):
    def __init__(self, records: Sequence[Stage8Record]):
        self.records = list(records)
        self.type_indices = {
            kind: [
                index for index, record in enumerate(self.records)
                if record.label_type == kind
            ]
            for kind in ("111", "100", "010", "000")
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Stage8Record:
        return self.records[index]


def collate_stage_adapter(records: Sequence[Stage8Record]) -> dict:
    return {
        "peptide_sequences": [record.peptide for record in records],
        "hla_sequences": [record.hla for record in records],
        "tcr_sequences": [record.tcr for record in records],
        "pht_labels": torch.tensor(
            [record.pht for record in records], dtype=torch.long
        ),
        "label_types": [record.label_type for record in records],
        "source_rows": torch.tensor(
            [record.source_row for record in records], dtype=torch.long
        ),
    }


def _limit_by_type(records, limit_per_type: int | None):
    if limit_per_type is None:
        return list(records)
    if limit_per_type < 1:
        raise ValueError("limit_per_type must be positive")
    counts = {kind: 0 for kind in ("111", "100", "010", "000")}
    selected = []
    for record in records:
        if counts[record.label_type] < limit_per_type:
            selected.append(record)
            counts[record.label_type] += 1
    return selected


def make_stage_adapter_loaders(
    audit: Stage8AuditResult,
    *,
    micro_batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
    limit_per_type: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    audit.require_clean()
    if micro_batch_size < 4 or micro_batch_size % 4:
        raise ValueError("micro_batch_size must be positive and divisible by 4")
    train = StageAdapterDataset(_limit_by_type(audit.train_records, limit_per_type))
    val = StageAdapterDataset(_limit_by_type(audit.val_records, limit_per_type))
    sampler = BalancedFourTypeBatchSampler(
        train.type_indices, batch_size=micro_batch_size, seed=seed
    )
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_stage_adapter,
    }
    return (
        DataLoader(train, batch_sampler=sampler, **common),
        DataLoader(val, batch_size=eval_batch_size, shuffle=False, **common),
    )
