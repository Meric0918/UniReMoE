"""Stage-14 training adapters and optimizer for the Hyper-PMT model.

The numerical loop is intentionally inherited from the battle-tested Stage-8
online trainer.  Stage14 owns its trainability contract and optimizer groups;
the inherited loop only supplies deterministic replay, accumulation and
constraint-loss bookkeeping.
"""

from __future__ import annotations

from typing import Any

import torch

from .stage8_config import Stage8Condition
from .stage8_trainer import (
    Stage8OnlineTrainer,
    validate_stage8_online_trainer_contract,
)
from .stage14_config import Stage14Condition, Stage14ExperimentConfig
from .trainer import build_scheduler


def validate_stage14_trainer_contract(config: Stage14ExperimentConfig) -> None:
    """Apply the fixed Stage14 schedule before any optimizer is created."""

    config.validate()
    # The shared validation checks the exact replay/constraint schedule via
    # config.stage8 compatibility.  Keep this explicit so a future change to
    # Stage8 does not silently alter Stage14.
    validate_stage8_online_trainer_contract(config)  # type: ignore[arg-type]


def build_stage14_online_optimizer(
    model,
    config: Stage14ExperimentConfig,
    phase: str,
) -> torch.optim.Optimizer:
    if phase not in {"stage8a", "stage8b"}:
        raise ValueError(f"Unknown Stage14 optimizer phase: {phase}")
    values = config.stage14
    groups: list[dict[str, Any]] = []

    def add(name: str, parameters, lr: float) -> None:
        active = [parameter for parameter in parameters if parameter.requires_grad]
        if active:
            groups.append({"name": name, "params": active, "lr": lr})

    binding = model.binding_model
    if phase == "stage8a":
        add(
            "hypergraph_new_modules",
            (
                parameter
                for module in binding.new_stage14_modules().values()
                for parameter in module.parameters()
            ),
            values.triplet_module_lr,
        )
    else:
        def add_encoder(name: str, encoder, layer_count: int) -> None:
            blocks = encoder.transformer_blocks()
            add(f"{name}_plm_top", blocks[-1].parameters(), 5e-6)
            if layer_count > 1:
                add(
                    f"{name}_plm_lower",
                    (
                        parameter
                        for block in blocks[-layer_count:-1]
                        for parameter in block.parameters()
                    ),
                    2.5e-6,
                )

        add_encoder("peptide", model.peptide_encoder, model.peptide_unfrozen_layers)
        add_encoder("hla", model.hla_encoder, model.hla_unfrozen_layers)
        add_encoder("tcr", model.tcr_encoder, model.tcr_unfrozen_layers)
        add(
            "adapters",
            (
                parameter
                for module in (binding.peptide_adapter, binding.hla_adapter, binding.tcr_adapter)
                for parameter in module.parameters()
            ),
            values.adapter_lr,
        )
        add(
            "pretrained_attention_and_heads",
            (
                parameter
                for module in (
                    binding.phla_prior_attention,
                    binding.phla_prior_pooling,
                    binding.phla_pooling,
                    binding.ptcr_pooling,
                    binding.phla_classifier,
                    binding.ptcr_classifier,
                )
                for parameter in module.parameters()
            ),
            values.pretrained_attention_lr,
        )
        add(
            "hypergraph_modules",
            (
                parameter
                for module in (
                    binding.hypergraph_fusion,
                    binding.pht_pooling,
                    binding.pht_classifier,
                    binding.phla_prior_projection,
                    binding.phla_prior_gate,
                )
                for parameter in module.parameters()
            ),
            values.triplet_module_lr,
        )
    parameter_ids = [
        id(parameter) for group in groups for parameter in group["params"]
    ]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Stage14 optimizer contains duplicate parameters")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if set(parameter_ids) != trainable_ids:
        raise RuntimeError("Stage14 optimizer does not cover exactly trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


class Stage14OnlineTrainer(Stage8OnlineTrainer):
    """Stage14-named facade over the deterministic Stage8 online loop."""

    def __init__(
        self,
        model,
        config: Stage14ExperimentConfig,
        train_loaders,
        val_loader,
        *,
        condition: Stage14Condition | str,
        tracker=None,
    ):
        validate_stage14_trainer_contract(config)
        super().__init__(
            model,
            config,  # the config exposes .stage8 for the shared loop
            train_loaders,
            val_loader,
            condition=Stage8Condition(Stage14Condition(condition).value),
            tracker=tracker,
        )

    def _build_optimization(self, phase: str) -> None:
        mapped = {"stage8a": "stage14a", "stage8b": "stage14b"}.get(phase)
        if mapped is None:
            raise ValueError(f"Unknown Stage14 phase: {phase}")
        self.model.set_trainable_stage14("joint", mapped)
        self.optimizer = build_stage14_online_optimizer(self.model, self.config, phase)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self._planned_optimizer_steps(phase),
            warmup_ratio=self.config.stage14.scheduler_warmup_ratio,
        )

    def train_stage14a_epoch(self, epoch: int):
        return self.train_stage8a_epoch(epoch)

    def fit_stage14a(self):
        return self.fit_stage8a()

    def begin_stage14b(self) -> None:
        self.begin_stage8b()

    def train_stage14b_round(self, round_index: int):
        return self.train_stage8b_round(round_index)

    def fit_stage14b(self):
        return self.fit_stage8b()

    def training_state(self) -> dict[str, int | float | str]:
        state = dict(super().training_state())
        state["phase"] = str(state["phase"]).replace("stage8", "stage14")
        state["stage14a_epoch"] = state.pop("stage8a_epoch")
        state["stage14b_round"] = state.pop("stage8b_round")
        return state
