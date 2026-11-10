"""Contract tests for the Stage-14 Hyper-PMT branch."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from source.plm_unified.config import ModelConfig
from source.plm_unified.encoders import ResidueBatch
from source.plm_unified.model import UnifiedBindingModel
from source.plm_unified.stage14_checkpoint import (
    initialise_stage14b_checkpoints,
    load_stage14_checkpoint,
    resume_stage14_last_checkpoint,
    save_stage14_warmup_checkpoint,
    validate_stage14_checkpoint,
)
from source.plm_unified.stage14_config import (
    STAGE14_STAGE1_SHA256,
    Stage14ExperimentConfig,
    Stage14FusionConfig,
    Stage14Paths,
    load_stage14_config,
)
from source.plm_unified.stage14_model import (
    HyperPMTFusionEncoder,
    ProgressiveStage14FineTuningModel,
    Stage14BindingModel,
    initialise_stage14_from_stage1,
)
from source.plm_unified.stage14_trainer import (
    Stage14OnlineTrainer,
    build_stage14_online_optimizer,
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


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int, name: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.checkpoint_fingerprint = f"tiny-{name}"
        self.requires_grad_(False)

    def transformer_blocks(self):
        return self.blocks

    def set_active_training(self, active: bool):
        self.train(active)

    def metadata(self):
        return {
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "hidden_size": self.hidden_size,
        }

    def forward(self, sequences):
        hidden = torch.ones(len(sequences), 5, self.hidden_size)
        for block in self.blocks:
            hidden = block(hidden)
        return ResidueBatch(
            hidden,
            torch.ones(len(sequences), 5, dtype=torch.bool),
            torch.full((len(sequences),), 5, dtype=torch.long),
        )


def _config(root: Path) -> Stage14ExperimentConfig:
    config = Stage14ExperimentConfig(
        paths=Stage14Paths(root, root, root / "models", root / "logs"),
        model=_model_config(),
        fusion=Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16),
    )
    config.stage14.mixed_precision = "none"
    return config


def _model() -> ProgressiveStage14FineTuningModel:
    model_config = _model_config()
    return ProgressiveStage14FineTuningModel(
        Stage14BindingModel(
            model_config,
            Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16),
        ),
        _TinyEncoder(8, "peptide"),
        _TinyEncoder(8, "hla"),
        _TinyEncoder(6, "tcr"),
        peptide_unfrozen_layers=2,
        hla_unfrozen_layers=2,
        tcr_unfrozen_layers=2,
    )


class _Loader:
    def __init__(self, batch, samples=4):
        self.batch = batch
        self.dataset = list(range(samples))
        self.batch_size = 4
        self.generator = None

    def __iter__(self):
        return iter([self.batch])

    def __len__(self):
        return 1


def _joint_batch():
    return {
        "peptide_sequences": ["A"] * 4,
        "hla_sequences": ["A"] * 4,
        "tcr_sequences": ["A"] * 4,
        "ph_labels": torch.tensor([1, 0, 0, 1]),
        "pt_labels": torch.tensor([1, 0, 1, 0]),
        "pht_labels": torch.tensor([1, 0, 0, 1]),
        "label_types": ["111", "000", "010", "100"],
        "source_rows": torch.arange(4),
    }


def _pair_batch():
    return {
        "peptide_sequences": ["A"] * 4,
        "receptor_sequences": ["A"] * 4,
        "labels": torch.tensor([1, 0, 1, 0]),
    }


class Stage14ConfigTests(unittest.TestCase):
    def test_repository_config_is_independent_and_registered(self):
        config = load_stage14_config(Path(__file__).resolve().parents[2] / "configs" / "plm_stage14.yaml")
        self.assertEqual(config.paths.stage1_sha256, STAGE14_STAGE1_SHA256)
        self.assertEqual(config.model.model_dim, 256)
        self.assertEqual(config.fusion.num_layers, 2)
        self.assertEqual(config.stage14.selection_weights, (0.4, 0.2, 0.4))
        self.assertEqual(config.run_dir("full").name, "full")

    def test_contract_rejects_noncanonical_schedule(self):
        config = _config(Path("/tmp/stage14-test"))
        config.stage14.gradient_accumulation_steps = 8
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            config.validate()


class Stage14FusionTests(unittest.TestCase):
    def test_pair_and_triple_edges_share_shape_contract(self):
        model = Stage14BindingModel(
            _model_config(), Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16)
        ).eval()
        p = torch.randn(2, 5, 8)
        h = torch.randn(2, 7, 8)
        t = torch.randn(2, 6, 6)
        masks = [torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 6, dtype=torch.bool)]
        output = model.forward_all(p, masks[0], h, masks[1], t, masks[2])
        self.assertEqual(output.phla_logits.shape, (2, 2))
        self.assertEqual(output.ptcr_logits.shape, (2, 2))
        self.assertEqual(output.pht_logits.shape, (2, 2))
        self.assertEqual(model("phla", p, masks[0], h, masks[1]).logits.shape, (2, 2))
        self.assertEqual(model("ptcr", p, masks[0], t, masks[2]).logits.shape, (2, 2))

    def test_dynamic_attention_kv_excludes_query_entity(self):
        encoder = HyperPMTFusionEncoder(
            _model_config(), Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16)
        )
        layer = encoder.layers[0]
        states = {
            "peptide": torch.randn(1, 2, 8),
            "hla": torch.randn(1, 3, 8),
            "tcr": torch.randn(1, 4, 8),
        }
        masks = {kind: torch.ones(1, value.shape[1], dtype=torch.bool) for kind, value in states.items()}
        _, _, _, _, attentions = layer(states, masks, return_attention=True)
        self.assertEqual(attentions["peptide"].shape[-1], 7)
        self.assertEqual(attentions["hla"].shape[-1], 6)
        self.assertEqual(attentions["tcr"].shape[-1], 5)

    def test_padding_is_zero_and_empty_entity_is_rejected(self):
        model = Stage14BindingModel(
            _model_config(), Stage14FusionConfig(feedforward_dim=16, compatibility_hidden_dim=16)
        ).eval()
        p = torch.randn(1, 5, 8)
        h = torch.randn(1, 7, 8)
        pm = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        hm = torch.ones(1, 7, dtype=torch.bool)
        output = model("phla", p, pm, h, hm).pooled_embedding
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(ValueError, "at least one"):
            model("phla", p, torch.zeros_like(pm), h, hm)


class Stage14TrainingTests(unittest.TestCase):
    def test_trainability_and_optimizer_cover_exact_parameters(self):
        model = _model()
        config = _config(Path("/tmp/stage14-test"))
        model.set_trainable_stage14("joint", "stage14a")
        self.assertTrue(all(p.requires_grad for p in model.binding_model.new_stage14_modules()["hypergraph_fusion"].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.peptide_encoder.parameters()))
        optimizer = build_stage14_online_optimizer(model, config, "stage8a")
        self.assertEqual(
            {id(p) for group in optimizer.param_groups for p in group["params"]},
            {id(p) for p in model.parameters() if p.requires_grad},
        )
        model.set_trainable_stage14("joint", "stage14b")
        self.assertTrue(all(p.requires_grad for p in model.peptide_encoder.transformer_blocks()[-2].parameters()))
        optimizer = build_stage14_online_optimizer(model, config, "stage8b")
        self.assertEqual(
            {id(p) for group in optimizer.param_groups for p in group["params"]},
            {id(p) for p in model.parameters() if p.requires_grad},
        )

    def test_online_loop_runs_stage14a_and_stage14b(self):
        model = _model()
        config = _config(Path("/tmp/stage14-test"))
        joint = _Loader(_joint_batch())
        pair = _Loader(_pair_batch())
        trainer = Stage14OnlineTrainer(
            model, config, {"phla": pair, "ptcr": pair, "joint": joint}, joint,
            condition="full",
        )
        self.assertEqual(trainer.training_state()["phase"], "stage14a")
        trainer.train_stage14a_epoch(1)
        trainer.train_stage14a_epoch(2)
        trainer.begin_stage14b()
        result = trainer.train_stage14b_round(1)
        self.assertIn("joint", result)
        self.assertEqual(trainer.training_state()["phase"], "stage14b")


class Stage14CheckpointTests(unittest.TestCase):
    def test_stage1_mapping_and_warmup_checkpoint(self):
        model_config = _model_config()
        source_model = UnifiedBindingModel(model_config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "stage1.pt"
            torch.save({
                "format_version": 3,
                "checkpoint_type": "stage1_joint_model",
                "stage": "stage1b",
                "model_scope": "joint_heads",
                "config": {"model": model_config.__dict__, "training": {"fold": 1, "seed": 42}},
                "cache_metadata": {
                    kind: {"model": {"checkpoint_fingerprint": f"tiny-{kind}"}}
                    for kind in ("peptide", "hla", "tcr")
                },
                "model": source_model.state_dict(),
            }, source_path)
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            model = _model()
            lineage = initialise_stage14_from_stage1(
                model, source_path, model_config=model_config,
                fold=1, seed=42, expected_sha256=source_sha,
            )
            self.assertEqual(lineage["kind"], "stage1")
            torch.testing.assert_close(
                model.binding_model.pht_classifier.network[0].weight,
                source_model.ptcr_classifier.network[0].weight,
            )

            config = _config(root)
            config.paths.stage1_sha256 = STAGE14_STAGE1_SHA256
            trainer = Stage14OnlineTrainer(
                model, config,
                {"phla": _Loader(_pair_batch()), "ptcr": _Loader(_pair_batch()), "joint": _Loader(_joint_batch())},
                _Loader(_joint_batch()), condition="baseline",
            )
            metrics = trainer.evaluate()
            payload_source = {"kind": "stage1", "sha256": STAGE14_STAGE1_SHA256}
            path = save_stage14_warmup_checkpoint(
                root / "warmup", model=model, config=config,
                source_stage1=payload_source, dataset_fingerprint="data",
                effective_data={"limit_per_type": 1, "train_samples": 4, "validation_samples": 4},
                run_timestamp="20260901_000000_000001",
                metrics={"validation_metrics": metrics.metrics, "validation_joint_score": metrics.joint_score,
                         "selected_pht_threshold": metrics.selected_pht_threshold},
                training_state=trainer.training_state(),
            )
            checkpoint = load_stage14_checkpoint(path, artifact_role="warmup")
            validate_stage14_checkpoint(checkpoint, artifact_role="warmup")
            self.assertEqual(checkpoint["checkpoint_type"], "stage14_online_model")

            trainer.stage8a_epoch = config.stage14.stage8a_epochs
            trainer.begin_stage14b()
            validation = trainer.evaluate()
            trainer.best_evaluation = validation
            trainer.best_joint = validation.joint_score
            best_path, last_path = initialise_stage14b_checkpoints(
                root / "full", trainer=trainer, source_stage1=payload_source,
                warmup_checkpoint=path, dataset_fingerprint="data",
                effective_data={"limit_per_type": 1, "train_samples": 4, "validation_samples": 4},
                run_timestamp="20260901_000000_000001",
                metrics={"validation_metrics": validation.metrics,
                         "validation_joint_score": validation.joint_score,
                         "selected_pht_threshold": validation.selected_pht_threshold},
            )
            resumed = Stage14OnlineTrainer(
                _model(), config,
                {"phla": _Loader(_pair_batch()), "ptcr": _Loader(_pair_batch()), "joint": _Loader(_joint_batch())},
                _Loader(_joint_batch()), condition="baseline",
            )
            resume_stage14_last_checkpoint(
                last_path, trainer=resumed, config=config,
                dataset_fingerprint="data",
                effective_data={"limit_per_type": 1, "train_samples": 4, "validation_samples": 4},
                condition="baseline",
            )
            self.assertEqual(resumed.training_state()["phase"], "stage14b")
            self.assertEqual(resumed.training_state()["stage14b_round"], 0)


if __name__ == "__main__":
    unittest.main()
