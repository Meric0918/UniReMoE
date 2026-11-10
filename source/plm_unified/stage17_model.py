"""Shared Stage16/17/18 forward with explicit pass-local gradient permissions."""

from __future__ import annotations

from .stage8_model import (
    ProgressiveStage8FineTuningModel,
    build_progressive_stage8_model,
)


class Stage17Model(ProgressiveStage8FineTuningModel):
    def set_trainable_pass(self, task, policy):
        if task not in {"phla", "ptcr", "joint"}:
            raise ValueError(f"Unknown pass: {task}")
        self.zero_grad(set_to_none=True)
        self.requires_grad_(False)
        self.eval()
        self._active_stage = task
        for encoder in (self.peptide_encoder, self.hla_encoder, self.tcr_encoder):
            encoder.set_active_training(False)
        if task == "joint":
            modules = (
                self.binding_model.new_triplet_modules()
                if policy.protect_joint
                else dict(self.binding_model.named_children())
            )
        else:
            modules = (
                self.binding_model.task_modules(task) if policy.binary_replay else {}
            )
            if task == "phla" and policy.protect_ph_peptide:
                modules = {k: v for k, v in modules.items() if k != "peptide_adapter"}
        for module in modules.values():
            module.requires_grad_(True)
            module.train()
        if not policy.freeze_plm and (task == "joint" or policy.binary_replay):
            for entity in ("peptide", "hla", "tcr"):
                if (
                    entity == "hla"
                    and task == "ptcr"
                    or entity == "tcr"
                    and task == "phla"
                ):
                    continue
                encoder = getattr(self, f"{entity}_encoder")
                count = getattr(self, f"{entity}_unfrozen_layers")
                if count:  # [-0:] would otherwise unfreeze ALL blocks.
                    for block in encoder.transformer_blocks()[-count:]:
                        block.requires_grad_(True)
                    encoder.set_active_training(True)


def build_stage17_model(config, device):
    base = build_progressive_stage8_model(config, device)
    return Stage17Model(
        base.binding_model,
        base.peptide_encoder,
        base.hla_encoder,
        base.tcr_encoder,
        peptide_unfrozen_layers=config.training.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.training.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.training.tcr_unfrozen_layers,
    )
