#!/usr/bin/env python3
"""Export Stage-1 test rows with large label/probability discrepancies.

The script evaluates the task-specific Stage-1 best checkpoints on all PH/PT
benchmark test splits.  It writes rows satisfying
``abs(label - prediction_score) > discrepancy_threshold`` to one CSV per task.
Existing Stage-1 entity caches are reused; only test entities absent from the
cache are encoded online.
"""

from __future__ import annotations

import argparse
import csv
import gc
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import normalise_sequence, resolve_task_columns, split_path
from plm_unified.encoders import (
    ESMCResidueEncoder,
    TCRBertResidueEncoder,
    checkpoint_fingerprint,
)
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


Task = Literal["phla", "ptcr"]

TEST_SPLITS: dict[Task, tuple[str, ...]] = {
    "phla": ("test_unseen_hla", "test_unseen_pair", "test_unseen_peptide"),
    "ptcr": ("test_unseen_pair", "test_unseen_peptide", "test_unseen_tcr"),
}


@dataclass(frozen=True)
class TestRow:
    peptide: str
    receptor: str
    label: int
    source: str


@dataclass
class EntityRepresentations:
    sequences: list[str]
    hidden: np.ndarray
    mask: np.ndarray

    def indices(self, values: Sequence[str]) -> np.ndarray:
        index = {sequence: position for position, sequence in enumerate(self.sequences)}
        return np.asarray([index[value] for value in values], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "plm_stage1_ph_pt.yaml"
        ),
    )
    parser.add_argument("--phla-checkpoint", type=Path, required=True)
    parser.add_argument("--ptcr-checkpoint", type=Path, required=True)
    parser.add_argument("--phla-round", type=int, required=True)
    parser.add_argument("--ptcr-round", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("csv/stage1/test"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--discrepancy-threshold", type=float, default=0.8)
    return parser.parse_args()


def read_test_rows(
    path: Path,
    task: Task,
    *,
    data_root: Path,
    peptide_min_length: int,
    peptide_max_length: int,
) -> list[TestRow]:
    peptide_col, receptor_col, label_col = resolve_task_columns(task, None)
    rows: list[TestRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        peptide_col, receptor_col, label_col = resolve_task_columns(
            task, reader.fieldnames
        )
        required = {peptide_col, receptor_col, label_col}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        source = path.relative_to(data_root).as_posix()
        for row_number, row in enumerate(reader, start=2):
            peptide = normalise_sequence(row[peptide_col])
            if not peptide_min_length <= len(peptide) <= peptide_max_length:
                continue
            receptor = normalise_sequence(row[receptor_col])
            try:
                label = int(row[label_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0 or 1 at {path}:{row_number}")
            rows.append(TestRow(peptide, receptor, label, source))
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return sorted(set(values))


def _load_cache(
    cache_dir: Path,
    entity: str,
    *,
    expected_hidden_dim: int,
    expected_fingerprint: str,
) -> EmbeddingCache | None:
    entity_dir = cache_dir / entity
    if not (entity_dir / "manifest.json").is_file():
        return None
    return EmbeddingCache(
        entity_dir,
        expected_hidden_dim=expected_hidden_dim,
        expected_fingerprint=expected_fingerprint,
    )


def _fill_from_cache(
    sequences: list[str],
    *,
    cache: EmbeddingCache | None,
    hidden: np.ndarray,
    mask: np.ndarray,
) -> list[str]:
    if cache is None:
        return sequences
    cached_positions: list[int] = []
    locators = []
    missing: list[str] = []
    for position, sequence in enumerate(sequences):
        try:
            locator = cache.resolve(sequence)
        except KeyError:
            missing.append(sequence)
            continue
        cached_positions.append(position)
        locators.append(locator)
    if locators:
        cached_hidden, cached_mask = cache.get_many(locators)
        hidden[np.asarray(cached_positions, dtype=np.int64)] = cached_hidden
        mask[np.asarray(cached_positions, dtype=np.int64)] = cached_mask
    return missing


def build_representations(
    values: Sequence[str],
    *,
    entity: str,
    hidden_dim: int,
    max_length: int,
    cache: EmbeddingCache | None,
    encoder,
    encode_batch_size: int,
    encoder_factory=None,
) -> EntityRepresentations:
    sequences = _ordered_unique(values)
    sequence_to_position = {
        sequence: position for position, sequence in enumerate(sequences)
    }
    hidden = np.empty(
        (len(sequences), max_length, hidden_dim),
        dtype=np.float16,
    )
    mask = np.empty((len(sequences), max_length), dtype=np.bool_)
    missing = _fill_from_cache(
        sequences,
        cache=cache,
        hidden=hidden,
        mask=mask,
    )
    print(
        f"[{entity}] unique={len(sequences):,} "
        f"cached={len(sequences) - len(missing):,} "
        f"online={len(missing):,}",
        flush=True,
    )
    if missing:
        if encoder is None and encoder_factory is not None:
            encoder = encoder_factory()
        if encoder is None:
            raise RuntimeError(
                f"{entity} has {len(missing):,} uncached sequences, but no encoder "
                "was provided"
            )
        for start in range(0, len(missing), encode_batch_size):
            batch = missing[start : start + encode_batch_size]
            encoded = encoder.encode(batch)
            positions = np.asarray(
                [sequence_to_position[sequence] for sequence in batch],
                dtype=np.int64,
            )
            hidden[positions] = encoded.hidden.detach().cpu().numpy().astype(
                np.float16,
                copy=False,
            )
            mask[positions] = encoded.mask.detach().cpu().numpy()
            if start == 0 or (start + len(batch)) % (encode_batch_size * 100) == 0:
                print(
                    f"[{entity}] encoded {start + len(batch):,}/{len(missing):,} "
                    "missing entities",
                    flush=True,
                )
    return EntityRepresentations(sequences, hidden, mask)


def load_task_model(
    checkpoint_path: Path,
    config,
    task: Task,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = load_training_checkpoint(checkpoint_path.resolve(), map_location="cpu")
    if checkpoint.get("checkpoint_type") != "task_model":
        raise ValueError(f"Expected task_model checkpoint: {checkpoint_path}")
    if checkpoint.get("task") != task:
        raise ValueError(
            f"Checkpoint task {checkpoint.get('task')!r} does not match {task!r}"
        )
    model_config = checkpoint.get("config", {}).get("model")
    if model_config is not None and model_config != config.to_dict()["model"]:
        raise ValueError(f"Model configuration mismatch: {checkpoint_path}")
    model = TaskSpecificBindingModel(config.model, task)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_scores(
    model: torch.nn.Module,
    task: Task,
    rows: Sequence[TestRow],
    peptide: EntityRepresentations,
    receptor: EntityRepresentations,
    *,
    device: torch.device,
    config,
    batch_size: int,
) -> np.ndarray:
    peptide_indices = peptide.indices([row.peptide for row in rows])
    receptor_indices = receptor.indices([row.receptor for row in rows])
    scores = np.empty(len(rows), dtype=np.float64)
    use_autocast = device.type == "cuda" and config.training.mixed_precision != "none"
    autocast_dtype = (
        torch.float16
        if config.training.mixed_precision == "fp16"
        else torch.bfloat16
    )
    input_dtype = autocast_dtype if use_autocast else torch.float32
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        peptide_hidden = torch.from_numpy(peptide.hidden[peptide_indices[start:end]]).to(
            device=device,
            dtype=input_dtype,
        )
        peptide_mask = torch.from_numpy(peptide.mask[peptide_indices[start:end]]).to(
            device=device,
            dtype=torch.bool,
        )
        receptor_hidden = torch.from_numpy(
            receptor.hidden[receptor_indices[start:end]]
        ).to(device=device, dtype=input_dtype)
        receptor_mask = torch.from_numpy(
            receptor.mask[receptor_indices[start:end]]
        ).to(device=device, dtype=torch.bool)
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            output = model(
                task,
                peptide_hidden,
                peptide_mask,
                receptor_hidden,
                receptor_mask,
            )
            batch_scores = torch.softmax(output.logits, dim=1)[:, 1]
        scores[start:end] = batch_scores.float().cpu().numpy()
        if start == 0 or end == len(rows) or end % (batch_size * 100) == 0:
            print(
                f"[{task}] predicted {end:,}/{len(rows):,} rows",
                flush=True,
            )
    return scores


def write_discrepancies(
    output_path: Path,
    task: Task,
    rows: Sequence[TestRow],
    scores: np.ndarray,
    threshold: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["hla", "peptide", "label", "prediction_score", "source"]
        if task == "phla"
        else ["peptide", "tcr", "label", "prediction_score", "source"]
    )
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score in zip(rows, scores):
            if abs(float(row.label) - float(score)) <= threshold:
                continue
            payload = {
                "peptide": row.peptide,
                "label": row.label,
                "prediction_score": f"{float(score):.10f}",
                "source": row.source,
            }
            if task == "phla":
                payload["hla"] = row.receptor
            else:
                payload["tcr"] = row.receptor
            writer.writerow(payload)
            count += 1
    return count


def evaluate_task(
    *,
    task: Task,
    round_id: int,
    checkpoint_path: Path,
    config,
    device: torch.device,
    output_dir: Path,
    encode_batch_size: int,
    prediction_batch_size: int,
    discrepancy_threshold: float,
) -> None:
    data_root = config.paths.data_root.resolve()
    split_rows: list[TestRow] = []
    for split in TEST_SPLITS[task]:
        path = split_path(data_root, task, split, config.training.fold)
        rows = read_test_rows(
            path,
            task,
            data_root=data_root,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        print(f"[{task}] {path.relative_to(data_root)} rows={len(rows):,}", flush=True)
        split_rows.extend(rows)
    print(f"[{task}] total rows={len(split_rows):,}", flush=True)

    esmc_fingerprint = checkpoint_fingerprint(config.paths.esmc_model)
    tcr_fingerprint = checkpoint_fingerprint(config.paths.tcr_bert_model)
    cache_dir = config.cache_dir()
    peptide_cache = _load_cache(
        cache_dir,
        "peptide",
        expected_hidden_dim=config.model.peptide_input_dim,
        expected_fingerprint=esmc_fingerprint,
    )
    if task == "phla":
        receptor_cache = _load_cache(
            cache_dir,
            "hla",
            expected_hidden_dim=config.model.hla_input_dim,
            expected_fingerprint=esmc_fingerprint,
        )
        esmc_encoder_holder = {"value": None}
        esmc_max_length = {"value": config.model.peptide_max_length}

        def get_esmc_encoder():
            encoder = esmc_encoder_holder["value"]
            if encoder is None:
                encoder = ESMCResidueEncoder(
                    config.paths.esmc_model,
                    max_length=esmc_max_length["value"],
                    device=device,
                    dtype=config.cache.dtype,
                )
                esmc_encoder_holder["value"] = encoder
            else:
                encoder.max_length = esmc_max_length["value"]
            return encoder

        peptide = build_representations(
            [row.peptide for row in split_rows],
            entity="phla/peptide",
            hidden_dim=config.model.peptide_input_dim,
            max_length=config.model.peptide_max_length,
            cache=peptide_cache,
            encoder=None,
            encoder_factory=get_esmc_encoder,
            encode_batch_size=encode_batch_size,
        )
        esmc_max_length["value"] = config.model.hla_max_length
        receptor = build_representations(
            [row.receptor for row in split_rows],
            entity="phla/hla",
            hidden_dim=config.model.hla_input_dim,
            max_length=config.model.hla_max_length,
            cache=receptor_cache,
            encoder=None,
            encoder_factory=get_esmc_encoder,
            encode_batch_size=encode_batch_size,
        )
    else:
        receptor_cache = _load_cache(
            cache_dir,
            "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
            expected_fingerprint=tcr_fingerprint,
        )
        peptide_encoder_holder = {"value": None}

        def get_peptide_encoder():
            encoder = peptide_encoder_holder["value"]
            if encoder is None:
                encoder = ESMCResidueEncoder(
                    config.paths.esmc_model,
                    max_length=config.model.peptide_max_length,
                    device=device,
                    dtype=config.cache.dtype,
                )
                peptide_encoder_holder["value"] = encoder
            return encoder

        peptide = build_representations(
            [row.peptide for row in split_rows],
            entity="ptcr/peptide",
            hidden_dim=config.model.peptide_input_dim,
            max_length=config.model.peptide_max_length,
            cache=peptide_cache,
            encoder=None,
            encoder_factory=get_peptide_encoder,
            encode_batch_size=encode_batch_size,
        )
        peptide_encoder_holder["value"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tcr_encoder_holder = {"value": None}

        def get_tcr_encoder():
            encoder = tcr_encoder_holder["value"]
            if encoder is None:
                encoder = TCRBertResidueEncoder(
                    config.paths.tcr_bert_model,
                    max_length=config.model.tcr_max_length,
                    device=device,
                    dtype=config.cache.dtype,
                )
                tcr_encoder_holder["value"] = encoder
            return encoder

        receptor = build_representations(
            [row.receptor for row in split_rows],
            entity="ptcr/tcr",
            hidden_dim=config.model.tcr_input_dim,
            max_length=config.model.tcr_max_length,
            cache=receptor_cache,
            encoder=None,
            encoder_factory=get_tcr_encoder,
            encode_batch_size=encode_batch_size,
        )

    model = load_task_model(checkpoint_path, config, task, device)
    scores = predict_scores(
        model,
        task,
        split_rows,
        peptide,
        receptor,
        device=device,
        config=config,
        batch_size=prediction_batch_size,
    )
    output_path = output_dir / ("pHLA" if task == "phla" else "pTCR") / (
        f"{'pHLA' if task == 'phla' else 'pTCR'}_round_{round_id}.csv"
    )
    count = write_discrepancies(
        output_path,
        task,
        split_rows,
        scores,
        discrepancy_threshold,
    )
    print(
        f"[{task}] discrepancy_rows={count:,} "
        f"threshold>{discrepancy_threshold} output={output_path}",
        flush=True,
    )
    del model, peptide, receptor, split_rows, scores
    if peptide_cache is not None:
        peptide_cache.close()
    if receptor_cache is not None:
        receptor_cache.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        device=args.device,
    )
    config.validate(require_models=True)
    if args.encode_batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.discrepancy_threshold < 0:
        raise ValueError("discrepancy threshold cannot be negative")
    device = resolve_device(config.training.device)
    output_dir = args.output_root.resolve() / create_artifact_timestamp()
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    evaluate_task(
        task="phla",
        round_id=args.phla_round,
        checkpoint_path=args.phla_checkpoint,
        config=config,
        device=device,
        output_dir=output_dir,
        encode_batch_size=args.encode_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        discrepancy_threshold=args.discrepancy_threshold,
    )
    evaluate_task(
        task="ptcr",
        round_id=args.ptcr_round,
        checkpoint_path=args.ptcr_checkpoint,
        config=config,
        device=device,
        output_dir=output_dir,
        encode_batch_size=args.encode_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        discrepancy_threshold=args.discrepancy_threshold,
    )
    print(f"Finished. Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
