"""Shared audited lifecycle/CLI for the three independent experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import signal
import time
import platform
import json

import torch

from .artifacts import create_artifact_timestamp
from .trainer import set_global_seed
from .encoders import checkpoint_fingerprint
from .stage8_checkpoint import load_stage8_v2_checkpoint, load_torch_checkpoint
from .stage16_checkpoint import (
    checkpoint_sha256,
    config_from_dict,
    digest,
    write_json,
    restore_training_checkpoint,
    load_common_warmup,
    verify_original_encoders,
)
from .stage16_config import load_stage16_config
from .stage16_data import audit_experiment_data
from .stage16_evaluation import summarize
from .stage16_trainer import Stage16Trainer
from .stage17_model import build_stage17_model
from .stage17_trainer import Stage17Trainer
from .tracking import SwanLabTracker
from .stage18_trainer import Stage18Trainer
from .stage18_teacher import (
    build_teacher,
    prepare_cache,
    teacher_identity,
    TeacherLogitCache,
)


def validate_sources(config, data):
    paths = config.paths
    for name in ("stage1", "warmup", "teacher"):
        path = getattr(paths, name + "_checkpoint")
        if checkpoint_sha256(path) != getattr(paths, name + "_sha256"):
            raise ValueError(f"{name} source SHA256 mismatch")
    warmup = load_common_warmup(paths.warmup_checkpoint)
    teacher = load_torch_checkpoint(paths.teacher_checkpoint)
    joint = load_torch_checkpoint(paths.stage1_checkpoint)
    if warmup["source_stage1"]["sha256"] != paths.stage1_sha256:
        raise ValueError("Warmup and Stage1 lineage mismatch")
    if warmup["dataset_fingerprint"] != data["manifest"]["pht_fingerprint"]:
        raise ValueError("PHT data fingerprint changed since warmup")
    if warmup["effective_data"]["limit_per_type"] is not None:
        raise ValueError("The common warmup must not be a debug-subset artifact")
    if teacher.get("task") != "ptcr" or teacher.get("input_mode") != "cache":
        raise ValueError("Teacher must be a frozen-cache Stage1 PT export")
    actual_fingerprints = {
        "peptide": checkpoint_fingerprint(paths.esmc_model),
        "hla": checkpoint_fingerprint(paths.esmc_model),
        "tcr": checkpoint_fingerprint(paths.tcr_bert_model),
    }
    for entity, fingerprint in actual_fingerprints.items():
        if (
            warmup["source_stage1"].get("encoder_fingerprints", {}).get(entity)
            != fingerprint
        ):
            raise ValueError(
                f"Original encoder/tokenizer fingerprint changed: {entity}"
            )
        if (
            teacher["cache_metadata"][entity]["model"]["checkpoint_fingerprint"]
            != fingerprint
        ):
            raise ValueError(f"Teacher feature coordinate mismatch: {entity}")
    for checkpoint in (warmup, teacher, joint):
        if checkpoint["config"]["model"] != asdict(config.model):
            raise ValueError("Source model shape/configuration mismatch")
        runtime = checkpoint["config"].get(
            "runtime", checkpoint["config"].get("training", {})
        )
        if (
            runtime.get("seed") != config.runtime.seed
            or runtime.get("fold") != config.runtime.fold
        ):
            raise ValueError(
                "Source fold/seed mismatch; do not relabel a fixed initialization as a new training seed"
            )
        for key in ("esmc_model", "tcr_bert_model"):
            if (
                Path(checkpoint["config"]["paths"][key]).resolve()
                != getattr(paths, key).resolve()
            ):
                raise ValueError("Source base encoder path mismatch")
    if any(
        not torch.equal(value, warmup["model"]["binding_model." + key])
        for key, value in joint["model"].items()
    ):
        raise ValueError("Warmup binary weights differ from pinned Stage1 joint")
    data["manifest"]["sources"] = {
        k: {
            "path": str(getattr(paths, k + "_checkpoint")),
            "sha256": getattr(paths, k + "_sha256"),
        }
        for k in ("stage1", "warmup", "teacher")
    }
    return warmup


def caches_and_reference(config, data, *, include_train=False):
    needed = [("reference", d) for d in data["pt_val"].values()]
    if include_train:
        needed.append(("train", data["pairs"]["ptcr"]))
    caches = {}
    missing = []
    for role, dataset in needed:
        identity = teacher_identity(config, dataset, role)
        path = config.paths.cache_root / role / (digest(identity) + ".pt")
        if path.is_file():
            caches[role, dataset.path.stem] = TeacherLogitCache(path, identity)
        else:
            missing.append((role, dataset))
    if missing:
        print(
            f"Preparing {len(missing)} independent teacher cache(s) on original frozen PLMs",
            flush=True,
        )
        teacher = build_teacher(config, torch.device(config.runtime.device))
        for role, dataset in missing:
            print(
                f"Teacher {role}: {dataset.path.name}, rows={len(dataset)}", flush=True
            )
            caches[role, dataset.path.stem] = prepare_cache(
                config, dataset, role, teacher=teacher
            )
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    reference = {"pt_independent": {}}
    reference["teacher_cache_identities"] = {}
    for split, dataset in data["pt_val"].items():
        cache = caches["reference", split]
        logits = cache.lookup(dataset.row_ids, dataset.keys, role="reference")
        reference["pt_independent"][split] = summarize(
            [r[2] for r in dataset.rows],
            logits.softmax(-1)[:, 1].numpy(),
            groups=[r[0] for r in dataset.rows],
        )
        reference["teacher_cache_identities"][split] = {
            "identity": cache.identity,
            "sha256": cache.sha256,
        }
    return reference, caches.get(("train", "train"))


def build_student(config, warmup):
    model = build_stage17_model(config, torch.device(config.runtime.device))
    verify_original_encoders(model, warmup)
    model.load_state_dict(warmup["model"], strict=True)
    return model


def parse_args(stage, argv=None, *, precompute=False):
    parser = argparse.ArgumentParser(
        description=f"Stage {stage}: audited controlled training and evaluation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / f"configs/plm_stage{stage}.yaml",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--condition")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--debug-pair-limit", type=int)
    parser.add_argument("--debug-limit-per-type", type=int)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Checkpoint and stop at an optimizer boundary; useful for smoke/resume validation",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-gradient-diagnostics", action="store_true")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    if precompute:
        parser.add_argument(
            "--role", choices=("train", "reference", "both"), default="both"
        )
    return parser.parse_args(argv)


def configured(args, stage):
    if args.resume:
        if (
            any(
                v is not None
                for v in (
                    args.condition,
                    args.rounds,
                    args.debug_pair_limit,
                    args.debug_limit_per_type,
                )
            )
            or args.no_gradient_diagnostics
        ):
            raise ValueError(
                "Resume uses embedded configuration; scientific overrides are prohibited"
            )
        saved = load_torch_checkpoint(args.resume)
        config = config_from_dict(saved["config"])
        if config.stage != stage:
            raise ValueError("Resume stage mismatch")
    else:
        config = load_stage16_config(args.config, expected_stage=stage)
        if args.condition:
            config.condition = args.condition
        if args.rounds is not None:
            config.training.rounds = args.rounds
        if args.debug_pair_limit is not None:
            config.training.debug_pair_limit = args.debug_pair_limit
        if args.debug_limit_per_type is not None:
            config.training.debug_limit_per_type = args.debug_limit_per_type
        if args.no_gradient_diagnostics:
            config.training.gradient_diagnostics = False
    if args.device:
        if args.resume and args.device != config.runtime.device:
            raise ValueError(
                "Exact training resume must retain the original CPU/CUDA device type; inference may use either"
            )
        config.runtime.device = args.device
    if args.swanlab_mode:
        if args.resume and args.swanlab_mode != config.swanlab.mode:
            raise ValueError("Resume must keep the saved SwanLab mode")
        config.swanlab.mode = args.swanlab_mode
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop-after-steps must be positive")
    config.validate()
    return config


def main(stage, argv=None):
    invocation_start = time.monotonic()
    args = parse_args(stage, argv)
    config = configured(args, stage)
    torch.set_num_threads(2)
    set_global_seed(config.runtime.seed)
    print(
        f"Stage{stage}/{config.condition}: auditing train/validation and source lineage",
        flush=True,
    )
    data = audit_experiment_data(config)
    warmup = validate_sources(config, data)
    if args.resume:
        output = args.resume.resolve().parent
        resume_checkpoint = load_torch_checkpoint(args.resume)
        resume_run_id = resume_checkpoint.get("swanlab_run_id") or resume_checkpoint.get("trainer_state", {}).get("swanlab_run_id")
        previous_run = (
            json.loads((output / "run_manifest.json").read_text())
            if (output / "run_manifest.json").is_file()
            else {}
        )
        if args.output_dir and args.output_dir.resolve() != output:
            raise ValueError("Resume must keep its artifact directory")
    else:
        previous_run = {}
        resume_checkpoint = None
        resume_run_id = None
        output = (
            args.output_dir
            or config.paths.output_root
            / f"fold_{config.runtime.fold}"
            / f"seed_{config.runtime.seed}"
            / config.condition
            / create_artifact_timestamp()
        )
        if output.exists() and any(output.iterdir()):
            raise ValueError("New run requires an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
    run = {
        "config": config.to_dict(),
        "effective_policy": asdict(config.policy),
        "contract": config.contract(),
        "manifest_sha256": digest(data["manifest"]),
        "source_warmup_selected_epoch": warmup["training_state"]["stage8a_epoch"],
        "implementation_version": "controlled_v1",
        "status": "audit_passed",
        "debug": config.debug,
    }
    run["implementation_files"] = {
        p.name: checkpoint_sha256(p)
        for p in Path(__file__).parent.glob("stage1[678]_*.py")
    }
    run["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if args.audit_only or args.dry_run:
        if args.resume:
            raise ValueError(
                "Use the original config for audit-only; resume never rewrites artifacts before verification"
            )
        write_json(output / "data_manifest.json", data["manifest"])
        write_json(output / "run_manifest.json", run)
        print(f"AUDIT PASSED: {output}; no PLM allocation or training", flush=True)
        return output
    tracker = SwanLabTracker(
        config,
        resume_id=resume_run_id,
        run_timestamp=output.name,
        experiment_suffix=config.condition,
    )
    run["swanlab_run_id"] = tracker.run_id
    tracker.log({"run/status": "started", "run/stage": stage, "run/debug": config.debug}, step=0)
    reference, cache = caches_and_reference(
        config, data, include_train=bool(config.policy.distillation_weight)
    )
    model = build_student(config, warmup)
    del warmup
    gc.collect()
    # A cache miss constructs a teacher and consumes initialization RNG; a hit
    # does not. All student weights were strictly restored from the common
    # warmup, so start the SAME training RNG stream after either preparation path.
    set_global_seed(config.runtime.seed)
    cls = {16: Stage16Trainer, 17: Stage17Trainer, 18: Stage18Trainer}[stage]
    trainer = cls(
        model, config, data, reference=reference, teacher_cache=cache, output_dir=output,
        tracker=tracker
    )
    if args.resume:
        checkpoint = restore_training_checkpoint(trainer, args.resume)
        # The immutable reference caches must still match the resumed run.
        if (
            checkpoint["reference"]["teacher_cache_identities"]
            != reference["teacher_cache_identities"]
        ):
            raise ValueError("Reference teacher caches changed across resume")
        del checkpoint
        gc.collect()
        trainer.flush_logs()
    write_json(output / "data_manifest.json", data["manifest"])
    write_json(output / "run_manifest.json", run)
    previous_handlers = {}

    def request_stop(signum, frame):
        trainer._interrupt = True
        print(
            "Stop requested; saving at the next optimizer-window boundary", flush=True
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    start = time.monotonic()
    run["preparation_wall_seconds"] = start - invocation_start
    try:
        finished = trainer.train_until(stop_after_steps=args.stop_after_steps)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    run.update(
        {
            "status": "finished" if finished else "checkpointed_stop",
            "completed_round": trainer.completed_round,
            "global_step": trainer.global_step,
            "seconds_this_invocation": time.monotonic() - start,
            "total_wall_seconds": previous_run.get(
                "total_wall_seconds", previous_run.get("seconds_this_invocation", 0.0)
            )
            + time.monotonic()
            - invocation_start,
            "training_wall_seconds_total": previous_run.get(
                "training_wall_seconds_total",
                previous_run.get("seconds_this_invocation", 0.0),
            )
            + time.monotonic()
            - start,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if config.runtime.device == "cuda"
                else 0
            ),
        }
    )
    write_json(output / "run_manifest.json", run)
    print(f"Stage{stage}: {run['status']}; artifacts={output}", flush=True)
    tracker.log({"run/status": run["status"], "run/global_step": trainer.global_step}, step=trainer.global_step)
    tracker.finish()
    return output


def precompute_main(argv=None):
    args = parse_args(18, argv, precompute=True)
    config = configured(args, 18)
    torch.set_num_threads(2)
    set_global_seed(config.runtime.seed)
    data = audit_experiment_data(config)
    validate_sources(config, data)
    if args.audit_only or args.dry_run:
        print("Teacher cache inputs and lineage validated; no logits generated")
        return
    # Reference caches are always prepared separately from the training role.
    if args.role in {"reference", "both"}:
        caches_and_reference(config, data, include_train=args.role == "both")
    else:
        teacher = build_teacher(config, torch.device(config.runtime.device))
        prepare_cache(config, data["pairs"]["ptcr"], "train", teacher=teacher)
