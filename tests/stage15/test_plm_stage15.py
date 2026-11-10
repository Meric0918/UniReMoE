"""Contract tests for Stage15 pairwise Shared Hyper-PMT."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from source.plm_unified.config import CacheConfig, ModelConfig, SwanLabConfig
from source.plm_unified.hyper_pmt import HyperPMTFusionEncoder
from source.plm_unified.stage14_config import Stage14FusionConfig
from source.plm_unified.stage14_model import HyperPMTFusionEncoder as Stage14HyperPMT
from source.plm_unified.stage15_config import (
    Stage15ExperimentConfig,
    Stage15FusionConfig,
    Stage15Paths,
    Stage15TrainingConfig,
    load_stage15_config,
)
from source.plm_unified.stage15_model import (
    Stage15BindingModel,
    Stage15TaskBindingModel,
    extract_stage15_task_state_dict,
)
from source.plm_unified.stage15_trainer import (
    Stage15Trainer,
    build_stage15_optimizer,
    build_stage15_task_checkpoint_state,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        peptide_input_dim=8,
        hla_input_dim=8,
        tcr_input_dim=6,
        adapter_hidden_dim=8,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
        adapter_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        peptide_min_length=1,
        peptide_max_length=5,
        hla_max_length=7,
        tcr_max_length=6,
    )


def _fusion() -> Stage15FusionConfig:
    return Stage15FusionConfig(
        feedforward_dim=16,
        compatibility_hidden_dim=16,
        dropout=0.0,
    )


def _config(root: Path) -> Stage15ExperimentConfig:
    training = Stage15TrainingConfig(
        batch_size=2,
        online_batch_size=2,
        num_workers=0,
        warmup_epochs_per_task=1,
        max_rounds=1,
        early_stopping_patience=1,
        shared_peptide_adapter_lr=5e-5,
        hla_adapter_lr=1e-4,
        tcr_adapter_lr=1e-4,
        phla_pooling_lr=1e-4,
        ptcr_pooling_lr=1e-4,
        phla_classifier_lr=7.5e-5,
        ptcr_classifier_lr=7.5e-5,
        hyper_pmt_lr=1e-4,
        scheduler_warmup_ratio=0.0,
        scheduler_warmup_rounds=0,
        mixed_precision="none",
        device="cpu",
        pin_memory=False,
        phla_group_weighting="none",
        ptcr_group_weighting="none",
        best_umap_enabled=False,
    )
    return Stage15ExperimentConfig(
        paths=Stage15Paths(
            root, root, root, root / "cache", root / "models", root / "logs"
        ),
        model=_model_config(),
        cache=CacheConfig(),
        fusion=_fusion(),
        training=training,
        swanlab=SwanLabConfig(mode="disabled"),
    )


def _batch(task: str):
    receptor_dim, receptor_length = (8, 7) if task == "phla" else (6, 6)
    return {
        "peptide_hidden": torch.randn(2, 5, 8),
        "peptide_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool),
        "receptor_hidden": torch.randn(2, receptor_length, receptor_dim),
        "receptor_mask": torch.ones(2, receptor_length, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
        "group_ids": torch.tensor([1, 2]),
    }


class _Loader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = type("Dataset", (), {"filter_stats": {"kept": 2}})()
        self.generator = None

    def __iter__(self):
        return iter([self.batch])

    def __len__(self):
        return 1


class _Tracker:
    run_id = None

    def log(self, values, *, step=None):
        del values, step


def test_repository_config_matches_stage1_control_and_stage14_fusion():
    root = Path(__file__).resolve().parents[2]
    stage15 = load_stage15_config(root / "configs" / "plm_stage15.yaml")
    assert stage15.fusion.num_layers == 2
    assert stage15.fusion.feedforward_dim == 512
    assert stage15.training.hyper_pmt_lr == pytest.approx(1e-4)
    assert stage15.paths.output_root.name == "PLM_Stage15"
    assert stage15.paths.cache_root.name == "plm_stage1_ph_pt"
    assert stage15.training.seeds == [42, 3407, 2026]


def test_stage14_reexports_the_neutral_hyper_pmt_implementation():
    assert Stage14HyperPMT is HyperPMTFusionEncoder
    model = Stage14HyperPMT(
        _model_config(),
        Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16),
    )
    assert "layers.0.dynamic_attention.relation_bias" in model.state_dict()


def test_pair_forward_uses_all_entity_tokens_and_has_no_pht_branch():
    model = Stage15BindingModel(_model_config(), _fusion()).eval()
    ph = _batch("phla")
    pt = _batch("ptcr")
    ph_out = model("phla", **{key: value for key, value in ph.items() if key not in {"labels", "group_ids"}}, return_attention=True, return_features=True)
    pt_out = model("ptcr", **{key: value for key, value in pt.items() if key not in {"labels", "group_ids"}}, return_attention=True, return_features=True)
    assert ph_out.logits.shape == pt_out.logits.shape == (2, 2)
    assert ph_out.pooling_weights.shape == (2, 12)
    assert pt_out.pooling_weights.shape == (2, 11)
    assert ph_out.cross_attention is None
    assert ph_out.encoder_embedding.shape == (2, 16)
    assert pt_out.encoder_embedding.shape == (2, 14)
    assert ph_out.adapter_embedding.shape == pt_out.adapter_embedding.shape == (2, 16)
    assert not hasattr(model, "pht_classifier")
    assert not hasattr(model, "phla_prior_attention")
    assert not hasattr(model, "phla_cross_attention")


def test_hyper_pmt_excludes_the_query_entity_and_respects_padding():
    encoder = HyperPMTFusionEncoder(_model_config(), _fusion())
    layer = encoder.layers[0]
    states = {
        "peptide": torch.randn(1, 2, 8),
        "hla": torch.randn(1, 3, 8),
        "tcr": torch.randn(1, 4, 8),
    }
    masks = {kind: torch.ones(1, value.shape[1], dtype=torch.bool) for kind, value in states.items()}
    updated, _, _, _, attention = layer(states, masks, return_attention=True)
    assert attention["peptide"].shape[-1] == 7
    assert attention["hla"].shape[-1] == 6
    assert attention["tcr"].shape[-1] == 5
    assert updated["peptide"].shape == states["peptide"].shape


def test_trainability_optimizer_scheduler_and_fgm_contract(tmp_path: Path):
    config = _config(tmp_path)
    model = Stage15BindingModel(config.model, config.fusion)
    optimizer = build_stage15_optimizer(model, config)
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in optimized}) == len(optimized)
    assert {id(parameter) for parameter in optimized} == {id(parameter) for parameter in model.parameters()}
    hyper_groups = [group for group in optimizer.param_groups if group["name"].startswith("shared_hyper_pmt/")]
    assert hyper_groups and all(group["scheduler_task"] == "shared" for group in hyper_groups)
    assert all(group["lr"] == pytest.approx(1e-4) for group in hyper_groups)

    model.set_trainable_task("phla")
    assert all(parameter.requires_grad for parameter in model.hypergraph_fusion.parameters())
    assert not any(parameter.requires_grad for parameter in model.tcr_adapter.parameters())
    model.set_trainable_task("ptcr", freeze_shared=True)
    assert not any(parameter.requires_grad for parameter in model.peptide_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in model.hypergraph_fusion.parameters())
    assert set(model.fgm_modules("ptcr")) == {"peptide_adapter", "tcr_adapter"}


def test_task_checkpoint_is_strict_and_matches_joint_logits(tmp_path: Path):
    config = _config(tmp_path)
    model = Stage15BindingModel(config.model, config.fusion).eval()
    state = {
        "format_version": 4,
        "checkpoint_type": "stage15_joint_model",
        "architecture": "shared_hyper_pmt_pair",
        "model": model.state_dict(),
        "config": config.to_dict(),
        "best_val_metrics": {"phla": {"val": {"auroc": 0.8, "aupr": 0.7}}},
        "best_task_scores": {"phla": 0.75, "ptcr": 0.5},
    }
    exported = build_stage15_task_checkpoint_state(
        state, "phla", source_checkpoint=tmp_path / "joint.pt"
    )
    assert exported["checkpoint_type"] == "stage15_task_model"
    assert any(name.startswith("hypergraph_fusion.") for name in exported["model"])
    assert not any(name.startswith("tcr_adapter.") for name in exported["model"])
    task_model = Stage15TaskBindingModel(config.model, config.fusion, "phla").eval()
    task_model.load_state_dict(exported["model"], strict=True)
    batch = _batch("phla")
    inputs = {key: value for key, value in batch.items() if key not in {"labels", "group_ids"}}
    assert torch.equal(model("phla", **inputs).logits, task_model("phla", **inputs).logits)
    with pytest.raises(ValueError, match="not 'ptcr'"):
        task_model("ptcr", **inputs)


def test_stage15_trainer_runs_cached_epoch_and_emits_stage15_state(tmp_path: Path):
    config = _config(tmp_path)
    loaders = {task: _Loader(_batch(task)) for task in ("phla", "ptcr")}
    trainer = Stage15Trainer(
        Stage15BindingModel(config.model, config.fusion),
        config,
        train_loaders=loaders,
        val_loaders={task: {"val": loader} for task, loader in loaders.items()},
        tracker=_Tracker(),
        run_timestamp="20260916_120000_000000",
    )
    ph_result = trainer.train_epoch("phla", loaders["phla"], phase_name="warmup", use_fgm=False)
    pt_result = trainer.train_epoch("ptcr", loaders["ptcr"], phase_name="round_1", use_fgm=True)
    assert ph_result.samples == pt_result.samples == 2
    state = trainer.checkpoint_state()
    assert state["checkpoint_type"] == "stage15_training_model"
    assert state["architecture"] == "shared_hyper_pmt_pair"
    assert trainer.last_checkpoint_path.name.startswith("stage15_last_")
    assert "csv/stage15" in str(trainer.validation_error_csv.output_dir)
    steps = {
        group["name"]: trainer.scheduler.group_steps[index]
        for index, group in enumerate(trainer.optimizer.param_groups)
    }
    assert steps["shared_hyper_pmt/decay"] == 2
    assert steps["phla_classifier/decay"] == 1
    assert steps["ptcr_classifier/decay"] == 1

    trainer.save_checkpoint(trainer.last_checkpoint_path)
    resumed = Stage15Trainer(
        Stage15BindingModel(config.model, config.fusion),
        config,
        train_loaders=loaders,
        val_loaders={task: {"val": loader} for task, loader in loaders.items()},
        tracker=_Tracker(),
        run_timestamp="20260916_120000_000000",
    )
    resumed.load_checkpoint(trainer.last_checkpoint_path)
    assert resumed.global_step == trainer.global_step
    assert resumed.scheduler.state_dict()["group_steps"] == trainer.scheduler.state_dict()["group_steps"]

    trainer.save_best_round_checkpoint(1, 0.75)
    with pytest.raises(ValueError, match="stage15_training_model"):
        resumed.load_checkpoint(trainer.best_round_checkpoint_path)


def test_stage15_full_warmup_and_one_round_smoke(tmp_path: Path):
    config = _config(tmp_path)
    loaders = {task: _Loader(_batch(task)) for task in ("phla", "ptcr")}
    trainer = Stage15Trainer(
        Stage15BindingModel(config.model, config.fusion),
        config,
        train_loaders=loaders,
        val_loaders={task: {"val": loader} for task, loader in loaders.items()},
        tracker=_Tracker(),
        run_timestamp="20260916_130000_000000",
    )
    summary = trainer.fit()
    assert summary["stage"] == "complete"
    assert summary["round"] == 1
    assert trainer.warmup_completed == {"phla": 1, "ptcr": 1}
    assert trainer.last_checkpoint_path.is_file()
    assert trainer.warmup_checkpoint_path.is_file()
    assert trainer.best_round_checkpoint_path.is_file()
    assert all(path.is_file() for path in trainer.task_checkpoint_paths.values())
