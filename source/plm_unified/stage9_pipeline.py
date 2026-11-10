"""Shared Stage-9 audit and loader assembly used by every CLI."""

from __future__ import annotations

from dataclasses import replace

from .data import make_online_dataloader, split_path
from .stage9_config import Stage9ExperimentConfig
from .stage9_data import (
    Stage9AuditResult,
    audit_stage9_pairs,
    make_stage9_pair_loader,
)
from .triplet_data import OnlineTripletDataset, make_triplet_dataloader


def build_stage9_audit(config: Stage9ExperimentConfig) -> Stage9AuditResult:
    values = config.stage9
    return audit_stage9_pairs(
        config.pair_path(),
        config.legacy_triplet_path("train"),
        config.model,
        supplement_csv=config.supplement_path(),
        split_seed=values.split_seed,
        train_fraction=values.train_fraction,
        val_fraction=values.val_fraction,
        reject_noncanonical_sequences=values.reject_noncanonical_sequences,
    )


def _limited_audit(
    audit: Stage9AuditResult,
    limit: int | None,
) -> Stage9AuditResult:
    if limit is None:
        return audit
    if limit < 1:
        raise ValueError("limit must be positive")
    limited = replace(
        audit,
        train_records=audit.train_records[:limit],
        val_records=audit.val_records[:limit],
        test_records=audit.test_records[:limit],
        report=dict(audit.report),
    )
    limited.report["debug_limit_per_split"] = limit
    limited.report["effective_split_rows"] = {
        name: len(records) for name, records in limited.split_records.items()
    }
    return limited


def build_stage9_loaders(
    config: Stage9ExperimentConfig,
    audit: Stage9AuditResult,
    *,
    limit: int | None = None,
):
    audit = _limited_audit(audit.require_clean(), limit)
    values = config.stage9
    common_pair = {
        "seed": config.runtime.seed,
        "num_workers": config.runtime.num_workers,
        "pin_memory": config.runtime.pin_memory,
    }
    pair_loaders = {
        "train": make_stage9_pair_loader(
            audit.train_records,
            batch_size=values.micro_batch_size,
            training=True,
            ph_balanced_sampling=values.ph_balanced_sampling,
            **common_pair,
        ),
        "val": make_stage9_pair_loader(
            audit.val_records,
            batch_size=values.eval_batch_size,
            training=False,
            ph_balanced_sampling=False,
            **common_pair,
        ),
        "test": make_stage9_pair_loader(
            audit.test_records,
            batch_size=values.eval_batch_size,
            training=False,
            ph_balanced_sampling=False,
            **common_pair,
        ),
    }
    common_binary = {
        "num_workers": config.runtime.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    legacy_train = {
        task: make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "train",
                config.runtime.fold,
            ),
            task,
            batch_size=values.micro_batch_size,
            shuffle=True,
            seed=config.runtime.seed + index + 11,
            **common_binary,
        )
        for index, task in enumerate(("phla", "ptcr"))
    }
    legacy_val = {
        task: make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "val",
                config.runtime.fold,
            ),
            task,
            batch_size=values.eval_batch_size,
            shuffle=False,
            seed=config.runtime.seed + index + 21,
            **common_binary,
        )
        for index, task in enumerate(("phla", "ptcr"))
    }
    triplet_common = {
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "hla_max_length": config.model.hla_max_length,
        "tcr_max_length": config.model.tcr_max_length,
        "strict_split_cleanup": True,
        "reject_noncanonical_sequences": values.reject_noncanonical_sequences,
        "limit": limit,
    }
    triplet_train = OnlineTripletDataset(
        config.legacy_triplet_path("train"), **triplet_common
    )
    triplet_val = OnlineTripletDataset(
        config.legacy_triplet_path("val"),
        excluded_labels=triplet_train.all_labels_by_key,
        **triplet_common,
    )
    legacy_train["triplet"] = make_triplet_dataloader(
        triplet_train,
        batch_size=values.micro_batch_size,
        shuffle=True,
        seed=config.runtime.seed + 31,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
    )
    legacy_val["triplet"] = make_triplet_dataloader(
        triplet_val,
        batch_size=values.eval_batch_size,
        shuffle=False,
        seed=config.runtime.seed + 41,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
    )
    audit_payload = audit.to_dict()
    audit_payload["effective_data"] = {
        "limit_per_split": limit,
        "pair_rows": {
            name: len(loader.dataset) for name, loader in pair_loaders.items()
        },
        "legacy_train_rows": {
            task: len(loader.dataset) for task, loader in legacy_train.items()
        },
        "legacy_val_rows": {
            task: len(loader.dataset) for task, loader in legacy_val.items()
        },
    }
    return pair_loaders, legacy_train, legacy_val, audit_payload
