"""Small SwanLab adapter that keeps tests and disabled runs dependency-free."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .config import ExperimentConfig


class SwanLabTracker:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        resume_id: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        experiment_suffix: Optional[str] = None,
    ):
        self.mode = config.swanlab.mode
        self.run = None
        self.run_id: Optional[str] = None
        self.text_factory = None
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        if self.mode == "disabled":
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is required for this run. "
                "Install requirements-plm.txt or pass --swanlab-mode disabled."
            ) from exc
        self.text_factory = swanlab.Text

        logdir = config.paths.swanlog_root
        logdir.mkdir(parents=True, exist_ok=True)
        fold = getattr(config.training, "fold", None)
        seed = getattr(config.training, "seed", None)
        runtime = getattr(config, "runtime", None)
        if runtime is not None:
            fold = getattr(runtime, "fold", fold)
            seed = getattr(runtime, "seed", seed)
        if fold is None or seed is None:
            raise ValueError("SwanLab-compatible config must expose fold and seed")
        group = config.swanlab.group.format(fold=fold)
        experiment_name = (
            f"{config.swanlab.experiment_prefix}"
            f"-fold{fold}-seed{seed}"
            f"-{self.run_timestamp}"
        )
        if experiment_suffix:
            experiment_name = f"{experiment_name}-{experiment_suffix}"
        experiment_prefix = config.swanlab.experiment_prefix.lower()
        if "stage-adapter" in experiment_prefix or "stage_adapter" in experiment_prefix:
            initialization = getattr(getattr(config, "stage_adapter", None), "initialization", "checkpoint")
            if initialization == "random":
                description = "Stage Adapter PHT training from random PLMs and binding modules, with the full PHT path trainable."
            elif initialization == "random_binding":
                description = "Stage Adapter PHT training from random shared adapters and PHT modules with frozen pretrained PLMs."
            else:
                description = (
                    "Stage Adapter PHT-only training with frozen PLM encoders and "
                    "pHLA/pTCR experts, and trainable PHT-specific modules."
                )
        elif "stage11" in experiment_prefix:
            description = (
                "Stage11 frozen ESM-C/TCR-BERT PHT training with alternating "
                "multi-positive peptide/HLA and PH/TCR contrastive passes."
            )
        elif "stage15" in experiment_prefix:
            description = (
                "Stage-15 from-scratch downstream PH/PT training on frozen PLM "
                "caches with one Shared Hyper-PMT fusion encoder."
            )
        elif "stage12" in experiment_prefix:
            description = (
                "Stage-12 Stage-1-initialised online-PLM retraining with "
                "PT-focused replay and validation-gated PH/PHT retention."
            )
        elif "stage14" in experiment_prefix:
            description = (
                "Stage-14 Stage-1-initialised Hyper-PMT heterogeneous hypergraph "
                "fusion with shared PH/PT/PHT interaction and gated PH prior."
            )
        elif "stage8" in experiment_prefix:
            description = (
                "Stage-8 Stage-1-initialised, online-PLM four-type training with "
                "directed hierarchy and pTCR implication constraints."
            )
        elif "stage5" in experiment_prefix:
            description = (
                "Stage-5 random-downstream, frozen-PLM pHLA/pTCR/PHT joint "
                "training with independently normalized task losses."
            )
        elif "triplet" in experiment_prefix or "stage3" in experiment_prefix:
            description = (
                "Stage-3 HLA-peptide-TCR sequential cross-attention "
                "fine-tuning with bounded pHLA/pTCR replay."
            )
        elif "stage-pht" in experiment_prefix or "stage_pht" in experiment_prefix:
            description = (
                "Stage PHT training with frozen pretrained ESM-C/TCR-BERT, three "
                "random sequence encoders and sequential PH/TCR cross-attention."
            )
        elif "stage1-encoder" in experiment_prefix or "stage1_encoder" in experiment_prefix:
            description = (
                "Stage1 training on frozen PLM residue features with shared Peptide "
                "and independent HLA/TCR self-attention encoders and module-targeted FGM."
            )
        elif "stage2" in experiment_prefix:
            description = (
                "Unified ESM-C/TCR-BERT progressive stage-2 fine-tuning with "
                "alternating pHLA/pTCR training and task-specific checkpoints."
            )
        else:
            description = (
                "Unified ESM-C/TCR-BERT stage-1 training with shared Peptide "
                "Adapter and module-targeted FGM."
            )
        init_kwargs: dict[str, Any] = {
            "project": config.swanlab.project,
            "workspace": config.swanlab.workspace,
            "experiment_name": experiment_name,
            "description": description,
            "job_type": "train",
            "group": group,
            "tags": config.swanlab.tags,
            "config": config.to_dict(),
            "logdir": str(logdir),
            "mode": self.mode,
        }
        if resume_id:
            init_kwargs.update({"id": resume_id, "resume": "must"})
        else:
            init_kwargs["resume"] = "never"
        self.run = swanlab.init(**init_kwargs)
        self.run_id = getattr(self.run, "id", None)

    def log(self, values: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if self.run is None:
            return
        payload = {
            key: self.text_factory(value)
            if isinstance(value, str) and self.text_factory is not None
            else value
            for key, value in values.items()
        }
        if step is not None:
            payload.setdefault("global_step", step)
        try:
            self.run.log(payload, step=step)
        except TypeError:
            self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None

    def __enter__(self) -> "SwanLabTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
