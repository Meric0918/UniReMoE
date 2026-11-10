"""Checkpoint validation and Stage-3 versus Stage-1 inference for Stage 7."""

from __future__ import annotations

import gc
import hashlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .encoders import OnlineEncoderBundle, checkpoint_fingerprint
from .finetuning import ESMCFineTuningEncoder, TCRBertFineTuningEncoder
from .model import TaskSpecificBindingModel
from .stage7_config import Stage7ExperimentConfig
from .stage7_counterfactual import Stage7Member
from .triplet_data import TripletRecord
from .triplet_model import TripletBindingModel


SCORE_COLUMNS = (
    "stage1_phla_margin",
    "stage1_ptcr_margin",
    "stage1_late_margin",
    "stage3_phla_margin",
    "stage3_ptcr_margin",
    "stage3_late_margin",
    "stage3_pht_margin",
)


@dataclass
class EncodedStore:
    names: tuple[str, ...]
    hidden: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        self.index = {name: index for index, name in enumerate(self.names)}

    def gather(
        self,
        values: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.tensor([self.index[value] for value in values], dtype=torch.long)
        hidden = self.hidden.index_select(0, indices).to(
            device,
            dtype=torch.float32 if device.type == "cpu" else self.hidden.dtype,
            non_blocking=device.type == "cuda",
        )
        mask = self.mask.index_select(0, indices).to(
            device,
            non_blocking=device.type == "cuda",
        )
        return hidden, mask


@dataclass
class Stage7Models:
    stage1_phla: TaskSpecificBindingModel
    stage1_ptcr: TaskSpecificBindingModel
    stage3: TripletBindingModel
    stage3_checkpoint_path: Path
    checkpoint_manifest: dict


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _saved_training(checkpoint: Mapping) -> Mapping:
    return checkpoint.get("config", {}).get("training", {})


def _stage1_fingerprint(checkpoint: Mapping, entity: str) -> str | None:
    metadata = checkpoint.get("cache_metadata", {}).get(entity, {})
    return metadata.get("model", {}).get("checkpoint_fingerprint")


def _validate_common_checkpoint_fields(
    checkpoint: Mapping,
    config: Stage7ExperimentConfig,
    *,
    label: str,
) -> None:
    if checkpoint.get("config", {}).get("model") != asdict(config.model):
        raise ValueError(f"{label} model configuration does not match Stage 7")
    training = _saved_training(checkpoint)
    for key, expected in (
        ("fold", config.runtime.fold),
        ("seed", config.runtime.seed),
    ):
        if training.get(key) != expected:
            raise ValueError(
                f"{label} {key}={training.get(key)!r} does not match {expected}"
            )


def load_stage7_models(
    config: Stage7ExperimentConfig,
    *,
    stage3_checkpoint: str | Path,
    stage1_phla_checkpoint: str | Path,
    stage1_ptcr_checkpoint: str | Path,
    device: torch.device,
) -> Stage7Models:
    """Load downstream modules and validate Stage-3 encoder provenance."""

    paths = {
        "stage3": Path(stage3_checkpoint).expanduser().resolve(),
        "stage1_phla": Path(stage1_phla_checkpoint).expanduser().resolve(),
        "stage1_ptcr": Path(stage1_ptcr_checkpoint).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    stage3_checkpoint_value = _load_checkpoint(paths["stage3"])
    stage1_phla_value = _load_checkpoint(paths["stage1_phla"])
    stage1_ptcr_value = _load_checkpoint(paths["stage1_ptcr"])
    if stage3_checkpoint_value.get("checkpoint_type") != "stage3_multitask_model":
        raise ValueError("Stage 7 requires a stage3_multitask_model checkpoint")
    lineage = stage3_checkpoint_value.get("initialization_source")
    if not isinstance(lineage, Mapping) or lineage.get("kind") != "stage1":
        raise ValueError("Stage 7 requires Stage3 initialised from Stage1")
    for key, expected in (
        ("source_fold", config.runtime.fold),
        ("source_seed", config.runtime.seed),
    ):
        if lineage.get(key) != expected:
            raise ValueError(
                f"Stage3 lineage {key}={lineage.get(key)!r} does not match "
                f"{expected}"
            )
    for task, checkpoint in (
        ("phla", stage1_phla_value),
        ("ptcr", stage1_ptcr_value),
    ):
        if checkpoint.get("checkpoint_type") != "task_model":
            raise ValueError(f"Stage1 {task} checkpoint must be a task_model")
        if checkpoint.get("task") != task:
            raise ValueError(f"Stage1 {task} checkpoint has task={checkpoint.get('task')}")
    _validate_common_checkpoint_fields(stage3_checkpoint_value, config, label="Stage3")
    _validate_common_checkpoint_fields(
        stage1_phla_value, config, label="Stage1 pHLA"
    )
    _validate_common_checkpoint_fields(
        stage1_ptcr_value, config, label="Stage1 pTCR"
    )

    local_esmc = checkpoint_fingerprint(config.paths.esmc_model)
    local_tcr = checkpoint_fingerprint(config.paths.tcr_bert_model)
    stage3_metadata = stage3_checkpoint_value.get("encoder_metadata", {})
    expected_fingerprints = {
        "stage3_peptide": stage3_metadata.get("peptide", {}).get(
            "checkpoint_fingerprint"
        ),
        "stage3_hla": stage3_metadata.get("hla", {}).get("checkpoint_fingerprint"),
        "stage3_tcr": stage3_metadata.get("tcr", {}).get("checkpoint_fingerprint"),
        "stage1_phla_peptide": _stage1_fingerprint(stage1_phla_value, "peptide"),
        "stage1_phla_hla": _stage1_fingerprint(stage1_phla_value, "hla"),
        "stage1_ptcr_peptide": _stage1_fingerprint(stage1_ptcr_value, "peptide"),
        "stage1_ptcr_tcr": _stage1_fingerprint(stage1_ptcr_value, "tcr"),
    }
    for label, fingerprint in expected_fingerprints.items():
        if not fingerprint:
            raise ValueError(f"Missing base-PLM fingerprint in {label}")
        expected = local_tcr if label.endswith("_tcr") else local_esmc
        if fingerprint != expected:
            raise ValueError(f"Base-PLM fingerprint mismatch for {label}")

    stage1_phla = TaskSpecificBindingModel(config.model, "phla")
    stage1_phla.load_state_dict(stage1_phla_value["model"])
    stage1_ptcr = TaskSpecificBindingModel(config.model, "ptcr")
    stage1_ptcr.load_state_dict(stage1_ptcr_value["model"])
    stage3 = TripletBindingModel(config.model)
    prefix = "binding_model."
    binding_state = {
        name[len(prefix) :]: tensor
        for name, tensor in stage3_checkpoint_value["model"].items()
        if name.startswith(prefix)
    }
    if not binding_state:
        raise ValueError("Stage3 checkpoint does not contain binding_model weights")
    stage3.load_state_dict(binding_state)
    for entity in ("peptide", "hla", "tcr"):
        encoder_prefix = f"{entity}_encoder."
        if not any(
            name.startswith(encoder_prefix)
            for name in stage3_checkpoint_value["model"]
        ):
            raise ValueError(
                f"Stage3 checkpoint does not contain {entity}_encoder weights"
            )
    for model in (stage1_phla, stage1_ptcr, stage3):
        model.eval().requires_grad_(False)
        model.to(device)

    manifest = {
        label: {
            "path": str(path),
            "sha256": checkpoint_sha256(path),
            "checkpoint_type": (
                stage3_checkpoint_value.get("checkpoint_type")
                if label == "stage3"
                else (
                    stage1_phla_value.get("checkpoint_type")
                    if label == "stage1_phla"
                    else stage1_ptcr_value.get("checkpoint_type")
                )
            ),
        }
        for label, path in paths.items()
    }
    manifest["base_plm_fingerprints"] = {
        "esmc": local_esmc,
        "tcr_bert": local_tcr,
    }
    manifest["stage3"]["fine_tuned_encoder_weights_used"] = True
    manifest["stage3"]["initialization_source"] = {
        key: lineage.get(key)
        for key in ("kind", "path", "source_fold", "source_seed")
    }
    del stage3_checkpoint_value, stage1_phla_value, stage1_ptcr_value
    gc.collect()
    return Stage7Models(
        stage1_phla,
        stage1_ptcr,
        stage3,
        paths["stage3"],
        manifest,
    )


def _chunks(values: Sequence, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def _encode_entity(
    encoder,
    sequences: Iterable[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    storage_dtype: torch.dtype,
) -> EncodedStore:
    names = tuple(sorted(set(sequences)))
    encoder.max_length = max_length
    hidden = []
    masks = []
    for batch in _chunks(names, batch_size):
        encoded = (
            encoder.encode(batch) if hasattr(encoder, "encode") else encoder(batch)
        )
        hidden.append(encoded.hidden.to(device="cpu", dtype=storage_dtype))
        masks.append(encoded.mask.cpu())
    if not hidden:
        raise ValueError("Cannot encode an empty Stage-7 entity set")
    return EncodedStore(names, torch.cat(hidden), torch.cat(masks))


def _storage_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu" or name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported Stage-7 encoder dtype: {name}")


def _entity_sequences(
    members: Iterable[Stage7Member],
    validation_records: Sequence[TripletRecord],
) -> dict[str, list[str]]:
    members = list(members)
    values = {
        "peptide": [member.peptide for member in members],
        "hla": [member.hla for member in members],
        "tcr": [member.tcr for member in members],
    }
    values["peptide"].extend(record.peptide for record in validation_records)
    values["hla"].extend(record.hla for record in validation_records)
    values["tcr"].extend(record.tcr for record in validation_records)
    return values


def encode_stage1_entities(
    config: Stage7ExperimentConfig,
    members: Iterable[Stage7Member],
    validation_records: Sequence[TripletRecord],
    *,
    device: torch.device,
) -> tuple[dict[str, EncodedStore], dict]:
    """Encode Stage-1 inputs with the original frozen base PLMs."""

    values = _entity_sequences(members, validation_records)
    storage_dtype = _storage_dtype(config.stage7.encoder_dtype, device)
    encoders = OnlineEncoderBundle(
        esmc_path=config.paths.esmc_model,
        tcr_bert_path=config.paths.tcr_bert_model,
        peptide_max_length=config.model.peptide_max_length,
        hla_max_length=config.model.hla_max_length,
        tcr_max_length=config.model.tcr_max_length,
        device=device,
        dtype=config.stage7.encoder_dtype,
    )
    stores = {
        "peptide": _encode_entity(
            encoders.esmc,
            values["peptide"],
            max_length=config.model.peptide_max_length,
            batch_size=config.stage7.encoder_batch_size,
            device=device,
            storage_dtype=storage_dtype,
        ),
        "hla": _encode_entity(
            encoders.esmc,
            values["hla"],
            max_length=config.model.hla_max_length,
            batch_size=config.stage7.encoder_batch_size,
            device=device,
            storage_dtype=storage_dtype,
        ),
        "tcr": _encode_entity(
            encoders.tcr_bert,
            values["tcr"],
            max_length=config.model.tcr_max_length,
            batch_size=config.stage7.encoder_batch_size,
            device=device,
            storage_dtype=storage_dtype,
        ),
    }
    audit = {
        "mode": "stage1_frozen_base_plm_hidden",
        "storage_dtype": str(stores["peptide"].hidden.dtype),
        "unique_sequences": {
            entity: len(store.names) for entity, store in stores.items()
        },
    }
    del encoders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stores, audit


def _load_stage3_encoder(
    config: Stage7ExperimentConfig,
    checkpoint_state: Mapping[str, torch.Tensor],
    entity: str,
    *,
    device: torch.device,
):
    if entity in {"peptide", "hla"}:
        max_length = (
            config.model.peptide_max_length
            if entity == "peptide"
            else config.model.hla_max_length
        )
        encoder = ESMCFineTuningEncoder(
            config.paths.esmc_model,
            max_length,
            device,
            dtype=config.stage7.encoder_dtype,
            gradient_checkpointing=False,
        )
    elif entity == "tcr":
        encoder = TCRBertFineTuningEncoder(
            config.paths.tcr_bert_model,
            config.model.tcr_max_length,
            device,
            dtype=config.stage7.encoder_dtype,
            gradient_checkpointing=False,
        )
    else:
        raise ValueError(f"Unknown Stage-3 encoder entity: {entity}")
    prefix = f"{entity}_encoder."
    state = {
        name[len(prefix) :]: tensor
        for name, tensor in checkpoint_state.items()
        if name.startswith(prefix)
    }
    if not state:
        raise ValueError(f"Stage3 checkpoint has no {entity}_encoder state")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("._extra_state")]
    unexpected = [name for name in unexpected if not name.endswith("._extra_state")]
    if missing or unexpected:
        raise RuntimeError(
            f"Stage3 {entity} encoder state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.set_active_training(False)
    return encoder


def encode_stage3_entities(
    config: Stage7ExperimentConfig,
    models: Stage7Models,
    members: Iterable[Stage7Member],
    validation_records: Sequence[TripletRecord],
    *,
    device: torch.device,
) -> tuple[dict[str, EncodedStore], dict]:
    """Encode inputs with the fine-tuned encoders stored in Stage 3."""

    values = _entity_sequences(members, validation_records)
    storage_dtype = _storage_dtype(config.stage7.encoder_dtype, device)
    checkpoint = _load_checkpoint(models.stage3_checkpoint_path)
    checkpoint_state = checkpoint["model"]
    stores = {}
    max_lengths = {
        "peptide": config.model.peptide_max_length,
        "hla": config.model.hla_max_length,
        "tcr": config.model.tcr_max_length,
    }
    for entity in ("peptide", "hla", "tcr"):
        encoder = _load_stage3_encoder(
            config,
            checkpoint_state,
            entity,
            device=device,
        )
        stores[entity] = _encode_entity(
            encoder,
            values[entity],
            max_length=max_lengths[entity],
            batch_size=config.stage7.encoder_batch_size,
            device=device,
            storage_dtype=storage_dtype,
        )
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    audit = {
        "mode": "stage3_checkpoint_fine_tuned_plm_hidden",
        "storage_dtype": str(stores["peptide"].hidden.dtype),
        "unique_sequences": {
            entity: len(store.names) for entity, store in stores.items()
        },
    }
    del checkpoint_state, checkpoint
    gc.collect()
    return stores, audit


def _autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


@torch.inference_mode()
def _score_binary_pairs(
    model,
    task: str,
    pairs: Sequence[tuple[str, str]],
    peptide_store: EncodedStore,
    receptor_store: EncodedStore,
    *,
    batch_size: int,
    device: torch.device,
    dtype: str,
) -> dict[tuple[str, str], float]:
    result = {}
    for batch in _chunks(pairs, batch_size):
        peptides = [pair[0] for pair in batch]
        receptors = [pair[1] for pair in batch]
        peptide_hidden, peptide_mask = peptide_store.gather(peptides, device)
        receptor_hidden, receptor_mask = receptor_store.gather(receptors, device)
        with _autocast_context(device, dtype):
            output = model(
                task,
                peptide_hidden,
                peptide_mask,
                receptor_hidden,
                receptor_mask,
            )
        margins = (output.logits[:, 1] - output.logits[:, 0]).float().cpu().tolist()
        result.update(zip(batch, margins))
    return result


@torch.inference_mode()
def _score_triplets(
    model: TripletBindingModel,
    triplets: Sequence[tuple[str, str, str]],
    stores: Mapping[str, EncodedStore],
    *,
    batch_size: int,
    device: torch.device,
    dtype: str,
) -> dict[tuple[str, str, str], float]:
    result = {}
    for batch in _chunks(triplets, batch_size):
        peptides = [value[0] for value in batch]
        tcrs = [value[1] for value in batch]
        hlas = [value[2] for value in batch]
        peptide_hidden, peptide_mask = stores["peptide"].gather(peptides, device)
        hla_hidden, hla_mask = stores["hla"].gather(hlas, device)
        tcr_hidden, tcr_mask = stores["tcr"].gather(tcrs, device)
        with _autocast_context(device, dtype):
            output = model.forward_triplet(
                peptide_hidden,
                peptide_mask,
                hla_hidden,
                hla_mask,
                tcr_hidden,
                tcr_mask,
            )
        margins = (output.logits[:, 1] - output.logits[:, 0]).float().cpu().tolist()
        result.update(zip(batch, margins))
    return result


def _normalisation(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std())
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Stage-7 validation margin standard deviation must be positive")
    return {"mean": float(array.mean()), "std": standard_deviation}


def _standardise(value: float, statistics: Mapping[str, float]) -> float:
    return (value - statistics["mean"]) / statistics["std"]


def score_stage7_members(
    config: Stage7ExperimentConfig,
    models: Stage7Models,
    members: Sequence[Stage7Member],
    validation_records: Sequence[TripletRecord],
    *,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Score Stage 1, release its stores, then score fine-tuned Stage 3."""

    phla_pairs = {
        (member.peptide, member.hla) for member in members
    } | {(record.peptide, record.hla) for record in validation_records}
    ptcr_pairs = {
        (member.peptide, member.tcr) for member in members
    } | {(record.peptide, record.tcr) for record in validation_records}
    triplets = {
        (member.peptide, member.tcr, member.hla) for member in members
    }
    phla_pairs = sorted(phla_pairs)
    ptcr_pairs = sorted(ptcr_pairs)
    triplets = sorted(triplets)
    kwargs = {
        "batch_size": config.stage7.score_batch_size,
        "device": device,
        "dtype": config.stage7.encoder_dtype,
    }
    stage1_stores, stage1_encoding = encode_stage1_entities(
        config,
        members,
        validation_records,
        device=device,
    )
    stage1_phla = _score_binary_pairs(
        models.stage1_phla,
        "phla",
        phla_pairs,
        stage1_stores["peptide"],
        stage1_stores["hla"],
        **kwargs,
    )
    stage1_ptcr = _score_binary_pairs(
        models.stage1_ptcr,
        "ptcr",
        ptcr_pairs,
        stage1_stores["peptide"],
        stage1_stores["tcr"],
        **kwargs,
    )
    stage1_normalisation = {
        "stage1_phla": _normalisation(
            [stage1_phla[(record.peptide, record.hla)] for record in validation_records]
        ),
        "stage1_ptcr": _normalisation(
            [stage1_ptcr[(record.peptide, record.tcr)] for record in validation_records]
        ),
    }
    del stage1_stores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage3_stores, stage3_encoding = encode_stage3_entities(
        config,
        models,
        members,
        validation_records,
        device=device,
    )
    stage3_phla = _score_binary_pairs(
        models.stage3,
        "phla",
        phla_pairs,
        stage3_stores["peptide"],
        stage3_stores["hla"],
        **kwargs,
    )
    stage3_ptcr = _score_binary_pairs(
        models.stage3,
        "ptcr",
        ptcr_pairs,
        stage3_stores["peptide"],
        stage3_stores["tcr"],
        **kwargs,
    )
    stage3_pht = _score_triplets(
        models.stage3,
        triplets,
        stage3_stores,
        **kwargs,
    )
    stage3_normalisation = {
        "stage3_phla": _normalisation(
            [stage3_phla[(record.peptide, record.hla)] for record in validation_records]
        ),
        "stage3_ptcr": _normalisation(
            [stage3_ptcr[(record.peptide, record.tcr)] for record in validation_records]
        ),
    }
    normalisation = {**stage1_normalisation, **stage3_normalisation}
    del stage3_stores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    scored = []
    for member in members:
        phla_key = (member.peptide, member.hla)
        ptcr_key = (member.peptide, member.tcr)
        triplet_key = (member.peptide, member.tcr, member.hla)
        values = member.to_dict()
        values.update(
            {
                "stage1_phla_margin": stage1_phla[phla_key],
                "stage1_ptcr_margin": stage1_ptcr[ptcr_key],
                "stage3_phla_margin": stage3_phla[phla_key],
                "stage3_ptcr_margin": stage3_ptcr[ptcr_key],
                "stage3_pht_margin": stage3_pht[triplet_key],
            }
        )
        values["stage1_late_margin"] = _standardise(
            values["stage1_phla_margin"], normalisation["stage1_phla"]
        ) + _standardise(
            values["stage1_ptcr_margin"], normalisation["stage1_ptcr"]
        )
        values["stage3_late_margin"] = _standardise(
            values["stage3_phla_margin"], normalisation["stage3_phla"]
        ) + _standardise(
            values["stage3_ptcr_margin"], normalisation["stage3_ptcr"]
        )
        scored.append(values)
    inference_audit = {
        "unique_phla_pairs": len(phla_pairs),
        "unique_ptcr_pairs": len(ptcr_pairs),
        "unique_triplets": len(triplets),
        "encoding": {
            "stage1": stage1_encoding,
            "stage3": stage3_encoding,
        },
        "late_fusion_validation_normalisation": normalisation,
    }
    return scored, inference_audit
