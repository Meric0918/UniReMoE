"""Immutable, row-addressed Stage1 teacher logits on ORIGINAL frozen encoders."""

from __future__ import annotations

from pathlib import Path
import time
import json

import numpy as np
import torch
from torch import nn
from .encoders import checkpoint_fingerprint

from .model import TaskSpecificBindingModel
from .stage16_checkpoint import (
    atomic_torch_save,
    checkpoint_sha256,
    digest,
    load_torch_checkpoint,
    write_json,
    load_common_warmup,
    verify_original_encoders,
)
from .stage16_evaluation import predict_pairs
from .stage17_model import build_stage17_model


class PTTeacher(nn.Module):
    def __init__(self, binding, peptide_encoder, tcr_encoder):
        super().__init__()
        self.binding_model = binding
        self.peptide_encoder = peptide_encoder
        self.tcr_encoder = tcr_encoder
        self.requires_grad_(False)
        self.set_evaluation()

    def set_evaluation(self):
        self.eval()
        self.peptide_encoder.set_active_training(False)
        self.tcr_encoder.set_active_training(False)

    def forward(self, task, peptides, receptors):
        if task != "ptcr":
            raise ValueError("PT teacher only supports ptcr")
        peptide = self.peptide_encoder(peptides)
        tcr = self.tcr_encoder(receptors)
        return self.binding_model(
            task, peptide.hidden, peptide.mask, tcr.hidden, tcr.mask
        )


def build_teacher(config, device):
    base = build_stage17_model(config, device)
    warmup = load_common_warmup(config.paths.warmup_checkpoint)
    verify_original_encoders(base, warmup)
    # Loading the pinned full warmup ensures original PLM inputs, never student inputs.
    base.load_state_dict(warmup["model"], strict=True)
    teacher = load_torch_checkpoint(config.paths.teacher_checkpoint)
    binding = TaskSpecificBindingModel(config.model, "ptcr").to(device)
    binding.load_state_dict(teacher["model"], strict=True)
    return PTTeacher(binding, base.peptide_encoder, base.tcr_encoder)


def teacher_identity(config, dataset, role):
    if role not in {"train", "reference"}:
        raise ValueError("Teacher cache roles are train/reference only")
    if dataset.path.stem.startswith("test"):
        raise ValueError("Test teacher caches are prohibited")
    if (role == "train") != (dataset.path.stem == "train"):
        raise ValueError("Teacher cache role does not match dataset split")
    return {
        "schema": 3,
        "role": role,
        "teacher_sha256": config.paths.teacher_sha256,
        "encoder_warmup_sha256": config.paths.warmup_sha256,
        "stage1_sha256": config.paths.stage1_sha256,
        "encoder_fingerprints": {
            "peptide": checkpoint_fingerprint(config.paths.esmc_model),
            "tcr": checkpoint_fingerprint(config.paths.tcr_bert_model),
        },
        "encoding_protocol": "unique_original_residue_fp32_storage_v1",
        "model": config.to_dict()["model"],
        "compute_device": config.runtime.device,
        "precision": (
            config.training.mixed_precision
            if config.runtime.device == "cuda"
            else "float32_cpu"
        ),
        "dataset_sha256": dataset.file_sha256,
        "row_ids_sha256": digest(dataset.row_ids),
        "input_keys_sha256": digest(dataset.keys),
        "labels_sha256": digest([r[2] for r in dataset.rows]),
        "normalization": "canonical-upper-no-gaps-v1",
        "samples": len(dataset),
    }


class TeacherLogitCache:
    def __init__(self, path, identity):
        self.path = Path(path)
        sidecar = json.loads(self.path.with_suffix(".json").read_text())
        self.sha256 = checkpoint_sha256(path)
        if sidecar.get("identity") != identity or sidecar.get("sha256") != self.sha256:
            raise ValueError("Teacher cache sidecar/bytes integrity mismatch")
        saved = load_torch_checkpoint(self.path)
        if saved.get("identity") != identity:
            raise ValueError("Teacher cache identity changed")
        self.identity = identity
        self.logits = saved["logits"].float()
        self.row_ids = saved["row_ids"]
        self.keys = saved["input_keys"]
        if (
            self.logits.shape != (identity["samples"], 2)
            or not torch.isfinite(self.logits).all()
        ):
            raise ValueError("Invalid teacher logits")
        if (
            digest(self.row_ids) != identity["row_ids_sha256"]
            or digest(self.keys) != identity["input_keys_sha256"]
        ):
            raise ValueError("Teacher cache row/input key corruption")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("Duplicate teacher row ID")
        self.index = {row: i for i, row in enumerate(self.row_ids)}

    def lookup(self, row_ids, input_keys, *, device="cpu", role="train"):
        if role != self.identity["role"]:
            raise ValueError("Teacher cache role misuse")
        if len(row_ids) != len(input_keys):
            raise ValueError("Mismatched teacher lookup lengths")
        indices = []
        for row, key in zip(row_ids, input_keys):
            if row not in self.index:
                raise ValueError("Missing teacher row ID")
            index = self.index[row]
            if self.keys[index] != key:
                raise ValueError("Teacher input key mismatch")
            indices.append(index)
        return self.logits[indices].to(device)


def prepare_cache(config, dataset, role, *, teacher=None, verify_samples=1024):
    if verify_samples < 1:
        raise ValueError("Teacher cache verification requires a positive sample count")
    identity = teacher_identity(config, dataset, role)
    path = config.paths.cache_root / role / (digest(identity) + ".pt")
    if path.is_file():
        return TeacherLogitCache(path, identity)
    if teacher is None:
        raise FileNotFoundError(
            f"Teacher cache is missing; run precompute first: {path}"
        )
    start = time.monotonic()
    from .stage16_benchmarks import cache_residues

    sequences = {
        "peptide": {r[0] for r in dataset.rows},
        "tcr": {r[1] for r in dataset.rows},
    }
    cached = cache_residues(teacher, sequences, config, config.training.eval_batch_size)
    predictions = predict_pairs(cached, dataset, config)
    del cached
    generator = np.random.default_rng(config.runtime.seed)
    indices = sorted(
        generator.choice(
            len(dataset), size=min(verify_samples, len(dataset)), replace=False
        ).tolist()
    )
    check = predict_pairs(teacher, dataset.subset(indices), config)["logits"]
    # BF16 batch-layout changes can change logits slightly, so also record actual error.
    # Cached residue features are stored in FP32 but consumed by the student
    # under BF16 autocast. The fixed BF16 logit tolerance is deliberately
    # larger than FP32's tolerance; the observed error remains recorded in the
    # sidecar and is not used to tune the model.
    tolerance = 0.1 if config.training.mixed_precision != "none" else 1e-4
    error = float(np.max(np.abs(check - predictions["logits"][indices])))
    if not np.allclose(
        check, predictions["logits"][indices], atol=tolerance, rtol=tolerance
    ):
        raise ValueError(f"Teacher cache spot-check failed: max logit error={error}")
    verification = {
        "samples": len(indices),
        "row_ids": [dataset.row_ids[i] for i in indices],
        "max_abs_logit_error": error,
        "atol": tolerance,
        "rtol": tolerance,
        "seconds": time.monotonic() - start,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "identity": identity,
            "row_ids": dataset.row_ids,
            "input_keys": dataset.keys,
            "logits": torch.from_numpy(predictions["logits"]),
            "verification": verification,
        },
        path,
    )
    write_json(
        path.with_suffix(".json"),
        {
            "identity": identity,
            "verification": verification,
            "sha256": checkpoint_sha256(path),
        },
    )
    return TeacherLogitCache(path, identity)
