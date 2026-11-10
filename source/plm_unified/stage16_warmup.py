"""Generate a two-epoch, same-seed common warmup for independent seed repeats."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch

from .encoders import checkpoint_fingerprint
from .trainer import set_global_seed, build_scheduler
from .stage16_checkpoint import (
    atomic_torch_save,
    checkpoint_sha256,
    load_torch_checkpoint,
    write_json,
)
from .stage16_config import load_stage16_config, ReplayPolicy
from .stage16_data import audit_experiment_data, joint_loader
from .stage16_evaluation import autocast, evaluate_joint
from .stage16_selection import task_score
from .stage17_model import build_stage17_model
from .stage8_loss import compute_stage8_loss


def generate_warmup(config, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Warmup artifacts are immutable; choose a new output path")
    if config.debug:
        raise ValueError("A common multi-seed warmup must use full audited data")
    set_global_seed(config.runtime.seed)
    sources = {}
    for name in ("stage1", "teacher"):
        path = getattr(config.paths, name + "_checkpoint")
        if checkpoint_sha256(path) != getattr(config.paths, name + "_sha256"):
            raise ValueError(f"{name} SHA mismatch")
        source = load_torch_checkpoint(path)
        sources[name] = source
        if (
            source["config"]["training"]["seed"] != config.runtime.seed
            or source["config"]["training"]["fold"] != config.runtime.fold
        ):
            raise ValueError("Warmup requires same-seed/fold Stage1 sources")
        if source["config"]["model"] != asdict(config.model):
            raise ValueError("Source model config mismatch")
    if (
        sources["stage1"].get("checkpoint_type") != "stage1_joint_model"
        or sources["teacher"].get("task") != "ptcr"
    ):
        raise ValueError("Expected joint source and separate PT expert")
    fingerprints = {
        entity: checkpoint_fingerprint(
            config.paths.tcr_bert_model if entity == "tcr" else config.paths.esmc_model
        )
        for entity in ("peptide", "hla", "tcr")
    }
    for source in sources.values():
        for entity, fingerprint in fingerprints.items():
            if (
                source["cache_metadata"][entity]["model"]["checkpoint_fingerprint"]
                != fingerprint
            ):
                raise ValueError("Original PLM/tokenizer fingerprint mismatch")
    data = audit_experiment_data(config)
    device = torch.device(config.runtime.device)
    model = build_stage17_model(config, device)
    missing, unexpected = model.binding_model.load_state_dict(
        sources["stage1"]["model"], strict=False
    )
    if unexpected or any(not key.startswith("triplet_") for key in missing):
        raise ValueError("Stage1 binary state mismatch")
    model.binding_model.initialise_triplet_from_ptcr()
    policy = ReplayPolicy(freeze_plm=True, protect_joint=True, binary_replay=False)
    model.set_trainable_pass("joint", policy)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.triplet_module_lr,
        weight_decay=config.training.weight_decay,
    )
    total_steps = 2 * (
        (
            len(joint_loader(data["joint_train"], config, train=True))
            + config.training.gradient_accumulation_steps
            - 1
        )
        // config.training.gradient_accumulation_steps
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=config.training.scheduler_warmup_ratio,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.training.mixed_precision == "fp16",
    )
    best = -float("inf")
    history = []
    global_step = 0
    for epoch in (1, 2):
        model.set_trainable_pass("joint", policy)
        iterator = iter(
            joint_loader(data["joint_train"], config, epoch=epoch, train=True)
        )
        while batches := list(
            islice(iterator, config.training.gradient_accumulation_steps)
        ):
            optimizer.zero_grad(set_to_none=True)
            size = sum(len(b["pht_labels"]) for b in batches)
            for batch in batches:
                with autocast(config, device):
                    out = model.forward_all(
                        batch["peptide_sequences"],
                        batch["hla_sequences"],
                        batch["tcr_sequences"],
                    )
                    labels = [
                        batch[k].to(device)
                        for k in ("ph_labels", "pt_labels", "pht_labels")
                    ]
                    loss = compute_stage8_loss(
                        out.phla_logits.float(),
                        out.ptcr_logits.float(),
                        out.pht_logits.float(),
                        *labels,
                        batch["label_types"],
                        condition="baseline",
                        epoch=epoch,
                        phase="stage8a",
                        phla_weight=0.0,
                        ptcr_weight=0.0,
                    )
                scaler.scale(loss.total * len(labels[0]) / size).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.training.gradient_clip_norm,
            )
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite warmup gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
        metrics, _ = evaluate_joint(model, data["joint_val"], config)
        score = task_score(metrics["pht"])
        history.append({"epoch": epoch, "score": score, "metrics": metrics})
        if score > best:
            best = score
            payload = {
                "checkpoint_type": "stage16_18_common_warmup",
                "format_version": 1,
                "artifact_role": "warmup",
                "config": config.to_dict(),
                "model": model.state_dict(),
                "source_stage1": {
                    "kind": "stage1",
                    "path": str(config.paths.stage1_checkpoint),
                    "sha256": config.paths.stage1_sha256,
                    "encoder_fingerprints": fingerprints,
                },
                "dataset_fingerprint": data["manifest"]["pht_fingerprint"],
                "effective_data": {
                    "limit_per_type": None,
                    "train_samples": len(data["joint_train"]),
                    "validation_samples": len(data["joint_val"]),
                },
                "training_state": {"stage8a_epoch": epoch, "global_step": global_step},
                "metrics": {"validation_metrics": metrics},
            }
            atomic_torch_save(payload, output)
        print(f"Common warmup epoch={epoch} PHT S={score}", flush=True)
    ready = config.to_dict()
    ready["paths"]["warmup_checkpoint"] = str(output.resolve())
    ready["paths"]["warmup_sha256"] = checkpoint_sha256(output)
    write_json(output.with_suffix(".ready_config.json"), ready)
    write_json(output.with_suffix(".history.json"), history)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    generate_warmup(load_stage16_config(args.config), args.output)
