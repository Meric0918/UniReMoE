"""Run the protected PT retention experiment, or validate it with --dry-run.

Starts from a Stage8A checkpoint, never a Stage8B fine-tuned encoder. All model
selection uses train/validation paths. The experimental checkpoint contains
binding weights and frozen encoder provenance; it is not a Stage-8 V2 export.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

_CUDA_DIRECTORY = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CUDA_DIRECTORY.is_dir():
    os.environ["CUDA_HOME"] = str(_CUDA_DIRECTORY)

import torch
import yaml

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.data import make_online_dataloader
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.stage8_checkpoint import (
    atomic_torch_save, checkpoint_sha256, load_stage8_v2_checkpoint,
)
from plm_unified.stage8_config import load_stage8_config
from plm_unified.stage8_data import audit_stage8_class_dirs, make_stage8_online_loaders
from plm_unified.stage8_evaluation import stage8_task_score
from plm_unified.stage8_model import build_progressive_stage8_model
from plm_unified.stage8_retention import (
    ProtectedStage8Model, PTRetentionPolicy, PTRetentionTrainer, retention_failures,
)
from plm_unified.trainer import set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/plm_stage8_pt_retention_20260924.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Audit inputs and plan, without allocating PLMs or training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = args.config.resolve()
    experiment = yaml.safe_load(path.read_text())
    required = {"base_config", "warmup_checkpoint", "pt_teacher_checkpoint", "output_root", "policy"}
    if set(experiment) != required:
        raise ValueError(f"Experiment requires exactly {sorted(required)}")
    resolve = lambda key: (path.parent / experiment[key]).resolve()
    config = load_stage8_config(resolve("base_config"))
    policy = PTRetentionPolicy(**experiment["policy"])
    audit = audit_stage8_class_dirs(
        config.data_dir(), config.stage8.class_dirs, config.stage8.train_filename,
        config.stage8.val_filename, config.model,
        reject_noncanonical_sequences=config.stage8.reject_noncanonical_sequences,
    )
    audit.require_clean()
    warmup = load_stage8_v2_checkpoint(resolve("warmup_checkpoint"), artifact_role="warmup")
    if warmup["source_stage1"]["sha256"] != config.paths.stage1_sha256:
        raise ValueError("Warmup and base config Stage-1 lineage disagree")
    if warmup["dataset_fingerprint"] != audit.dataset_fingerprint:
        raise ValueError("Warmup and current PHT data fingerprints disagree")
    if warmup["effective_data"]["limit_per_type"] is not None:
        raise ValueError("A debug-subset warmup cannot start a full retention experiment")
    teacher_path = resolve("pt_teacher_checkpoint")
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False, mmap=True)
    if teacher_checkpoint.get("task") != "ptcr" or teacher_checkpoint.get("input_mode") != "cache":
        raise ValueError("Teacher must be a cache-trained Stage-1 ptcr task checkpoint")
    if teacher_checkpoint["config"]["model"] != asdict(config.model):
        raise ValueError("Teacher model dimensions/config do not match")
    for name in ("esmc_model", "tcr_bert_model"):
        if Path(teacher_checkpoint["config"]["paths"][name]).resolve() != getattr(config.paths, name).resolve():
            raise ValueError(f"Teacher and student encoder source differs: {name}")
    config.runtime.device = args.device
    config.runtime.num_workers = 0
    loaders, val_loader = make_stage8_online_loaders(
        audit, phla_path=config.paths.data_root / "PH/train.csv",
        ptcr_path=config.paths.data_root / "PT/train.csv", model=config.model,
        micro_batch_size=config.stage8.micro_batch_size,
        eval_batch_size=config.stage8.eval_batch_size, seed=config.runtime.seed, num_workers=0,
    )
    pt_val_loaders = {
        split: make_online_dataloader(
            config.paths.data_root / "PT/benchmarks" / f"{split}.csv", "ptcr",
            batch_size=config.stage8.eval_batch_size, shuffle=False,
            seed=config.runtime.seed, num_workers=0, pin_memory=False,
            peptide_min_length=config.model.peptide_min_length,
            peptide_max_length=config.model.peptide_max_length,
        )
        for split in ("val_unseen_pair", "val_unseen_peptide", "val_unseen_tcr")
    }
    plan = {
        "base_config": config.to_dict(), "policy": asdict(policy),
        "dataset_fingerprint": audit.dataset_fingerprint,
        "warmup_checkpoint": str(resolve("warmup_checkpoint")),
        "warmup_sha256": checkpoint_sha256(resolve("warmup_checkpoint")),
        "teacher_checkpoint": str(teacher_path), "teacher_sha256": checkpoint_sha256(teacher_path),
        "pt_validation_rows": {k: len(v.dataset) for k, v in pt_val_loaders.items()},
        "pt_replay_rows_per_round": min(len(loaders["ptcr"].dataset), int(len(audit.train_records) * policy.pt_replay_ratio)),
        "encoder_policy": "frozen_original_plm",
        "joint_update": "triplet_modules_only",
        "pass_order": ["phla", "joint", "ptcr"],
        "selection": "0.6*independent_PT_macro+0.3*PHT+0.1*PHT_PH; guarded against PT teacher and initial PHT",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN PASSED; no PLMs allocated and no training performed", flush=True)
        return
    set_global_seed(config.runtime.seed)
    device = torch.device(args.device)
    original = build_progressive_stage8_model(config, device)
    # Enforce teacher feature compatibility before restoring the warmup state.
    for name, value in original.state_dict().items():
        saved = warmup["model"][name]
        if "_encoder." in name and not torch.equal(value.cpu().to(saved.dtype), saved):
            raise ValueError(f"Warmup encoder differs from original pretrained encoder: {name}")
    original.load_state_dict(warmup["model"], strict=True)
    model = ProtectedStage8Model(original.binding_model, original.peptide_encoder, original.hla_encoder, original.tcr_encoder)
    del original, warmup
    teacher = TaskSpecificBindingModel(config.model, "ptcr").to(device)
    teacher.load_state_dict(teacher_checkpoint["model"], strict=True)
    del teacher_checkpoint
    trainer = PTRetentionTrainer(
        model, config, loaders, val_loader, condition="full",
        teacher=teacher, pt_val_loaders=pt_val_loaders, policy=policy,
    )
    trainer.stage8a_epoch = config.stage8.stage8a_epochs
    trainer.begin_stage8b()
    output = resolve("output_root") / create_artifact_timestamp()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Evaluating initial model on PHT and independent PT validation: {output}", flush=True)
    initial = trainer.evaluate()
    initial_pht = stage8_task_score(initial.metrics["pht"])
    initial_pt, reference_pt = trainer.evaluate_pt()
    best_score = -float("inf")
    history = []
    for round_index in range(policy.rounds + 1):
        if round_index == 0:
            evaluation, pt_metrics, training = initial, initial_pt, {}
        else:
            print(f"Training retention round {round_index}/{policy.rounds}", flush=True)
            training = trainer.train_stage8b_round(round_index)
            evaluation = trainer.evaluate()
            pt_metrics, _ = trainer.evaluate_pt()
        failures = retention_failures(
            pt_metrics, reference_pt, stage8_task_score(evaluation.metrics["pht"]), initial_pht, policy,
        )
        score = trainer.selection_score(pt_metrics, evaluation.metrics)
        record = {
            "round": round_index, "score": score, "guard_failures": failures,
            "pt_validation": pt_metrics, "pht_validation": evaluation.metrics,
            "teacher_pt_validation": reference_pt, "training": training,
        }
        history.append(record)
        print(json.dumps({"round": round_index, "score": score, "guard_failures": failures}), flush=True)
        payload = {
            "checkpoint_type": "stage8_pt_retention_frozen_plm", "format_version": 1,
            "plan": plan, "metrics": record,
            "binding_model": {k: v.detach().cpu().clone() for k, v in model.binding_model.state_dict().items()},
        }
        if round_index == 0:
            atomic_torch_save({**payload, "artifact_role": "initial"}, output / "initial.pt")
        if not failures and score > best_score + config.stage8.early_stopping_min_delta:
            best_score = score
            atomic_torch_save({**payload, "artifact_role": "best"}, output / "best.pt")
        atomic_torch_save({**payload, "artifact_role": "last"}, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Finished: {output}; guard-passing best exists: {(output / 'best.pt').exists()}", flush=True)


if __name__ == "__main__":
    main()
