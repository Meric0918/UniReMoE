"""Stage12 replay sampling and online loader helpers."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from .config import ModelConfig
from .data import (
    OnlinePairDataset,
    _collate_online_pairs,
    _seed_worker,
    make_online_dataloader,
)
from .stage8_data import (
    BalancedFourTypeBatchSampler,
    OnlineStage8Dataset,
    Stage8AuditResult,
    _limit_by_type,
    collate_online_stage8,
)


PairRow = tuple[str, str, int]


def replay_sample_count(joint_samples: int, dataset_samples: int, ratio: float) -> int:
    if joint_samples < 0 or dataset_samples < 0:
        raise ValueError("sample counts cannot be negative")
    if not 0 < ratio <= 1:
        raise ValueError("replay ratio must be in (0, 1]")
    return min(dataset_samples, math.floor(joint_samples * ratio))


def pt_row_sampling_weights(rows: Sequence[PairRow]) -> list[float]:
    """Give peptide-label groups mass proportional to sqrt(group frequency)."""

    frequencies = Counter((str(row[0]), int(row[2])) for row in rows)
    return [1.0 / math.sqrt(frequencies[(str(row[0]), int(row[2]))]) for row in rows]


class PeptideAwareBalancedSampler(Sampler[int]):
    """Balanced PT sampler with sqrt-peptide weighting and no replacement."""

    def __init__(
        self,
        rows: Sequence[PairRow],
        *,
        num_samples: int,
        seed: int,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.rows = rows
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.round = 1
        self.indices = {
            label: [index for index, row in enumerate(rows) if int(row[2]) == label]
            for label in (0, 1)
        }
        self.weights = pt_row_sampling_weights(rows)
        quotas = self._quotas()
        if any(quotas[label] > len(self.indices[label]) for label in (0, 1)):
            raise ValueError(
                "balanced PT replay cannot draw requested samples without replacement"
            )

    def _quotas(self) -> dict[int, int]:
        return {0: self.num_samples // 2, 1: self.num_samples - self.num_samples // 2}

    def set_round(self, round_index: int) -> None:
        if round_index < 1:
            raise ValueError("round_index must be positive")
        self.round = int(round_index)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.round
        )
        selected: list[int] = []
        for label, quota in self._quotas().items():
            label_indices = self.indices[label]
            label_weights = torch.tensor(
                [self.weights[index] for index in label_indices], dtype=torch.double
            )
            offsets = torch.multinomial(
                label_weights, quota, replacement=False, generator=generator
            ).tolist()
            selected.extend(label_indices[offset] for offset in offsets)
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(selected[index] for index in order)


def make_pt_replay_loader(
    csv_path: str | Path,
    *,
    micro_batch_size: int,
    num_samples: int,
    seed: int,
    num_workers: int,
    peptide_min_length: int,
    peptide_max_length: int,
) -> DataLoader:
    dataset = OnlinePairDataset(
        csv_path,
        "ptcr",
        peptide_min_length=peptide_min_length,
        peptide_max_length=peptide_max_length,
    )
    sampler = PeptideAwareBalancedSampler(
        dataset.rows, num_samples=num_samples, seed=seed
    )
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
        collate_fn=_collate_online_pairs,
        worker_init_fn=_seed_worker,
    )


def make_stage12_online_loaders(
    audit: Stage8AuditResult,
    *,
    phla_path: str | Path,
    ptcr_path: str | Path,
    model: ModelConfig,
    micro_batch_size: int,
    eval_batch_size: int,
    phla_replay_ratio: float,
    ptcr_replay_ratio: float,
    seed: int,
    num_workers: int,
    limit_per_type: int | None = None,
    pair_limit: int | None = None,
) -> tuple[dict[str, DataLoader], DataLoader]:
    audit.require_clean()
    if micro_batch_size < 4 or micro_batch_size % 4:
        raise ValueError("micro_batch_size must be divisible by four")
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
    phla_loader = make_online_dataloader(
        phla_path,
        "phla",
        batch_size=micro_batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        peptide_min_length=model.peptide_min_length,
        peptide_max_length=model.peptide_max_length,
        limit=pair_limit,
    )
    pt_loader = make_online_dataloader(
        ptcr_path,
        "ptcr",
        batch_size=micro_batch_size,
        shuffle=True,
        seed=seed + 1,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        peptide_min_length=model.peptide_min_length,
        peptide_max_length=model.peptide_max_length,
        limit=pair_limit,
    )
    # Validate the independent ratios; both loaders remain full-sized and are
    # capped by Stage12OnlineTrainer on every round, as in Stage8.
    replay_sample_count(len(train_dataset), len(phla_loader.dataset), phla_replay_ratio)
    replay_sample_count(len(train_dataset), len(pt_loader.dataset), ptcr_replay_ratio)
    return {"phla": phla_loader, "ptcr": pt_loader, "joint": joint_loader}, validation_loader
