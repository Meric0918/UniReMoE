"""Random initialization, full PHT gradients, and checkpoint continuation."""

import copy
from contextlib import redirect_stdout
import io
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from test_plm_stage_adapter import _batch, _config, DummyEncoder
from source.plm_unified.encoders import ResidueBatch
from source.plm_unified.stage_adapter_config import load_stage_adapter_config
from source.plm_unified.stage_adapter_model import (
    ProgressiveStageAdapterModel, StageAdapterBindingModel,
    build_progressive_stage_adapter_model, initialise_stage_adapter_random,
)
from source.plm_unified.stage_adapter_checkpoint import (
    load_stage_adapter_checkpoint, restore_stage_adapter_model,
    resume_stage_adapter_last, save_stage_adapter_best,
    save_stage_adapter_last, save_stage_adapter_warmup,
    validate_stage_adapter_checkpoint,
)
from source.plm_unified.stage_adapter_trainer import StageAdapterTrainer

cuda_root = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and cuda_root.is_dir():
    os.environ["CUDA_HOME"] = str(cuda_root)


class TrainableEncoder(DummyEncoder):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def set_full_training(self):
        self.requires_grad_(True)
        self.set_active_training(True)

    def forward(self, sequences):
        ids = torch.zeros(len(sequences), self.max_length, dtype=torch.long)
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor([ord(value) % 32 for value in sequence])
        mask = torch.arange(self.max_length).unsqueeze(0) < lengths.unsqueeze(1)
        return ResidueBatch(self.embedding(ids), mask, lengths)


def random_model(config):
    return ProgressiveStageAdapterModel(
        StageAdapterBindingModel(config.model, residual_adapter_dim=4,
                                 train_shared_adapters=True),
        TrainableEncoder(config.model.peptide_input_dim),
        TrainableEncoder(config.model.hla_input_dim),
        TrainableEncoder(config.model.tcr_input_dim),
        initialization=config.stage_adapter.initialization,
    )


class RandomStageAdapterTests(unittest.TestCase):
    def test_random_config_needs_no_stage_checkpoint_or_digest(self):
        path = Path(__file__).resolve().parents[2] / "configs/plm_stage_adapter_random.yaml"
        config = load_stage_adapter_config(path)
        self.assertEqual(config.stage_adapter.initialization, "random_binding")
        self.assertIsNone(config.paths.stage1_checkpoint)
        self.assertIsNone(config.paths.stage1_sha256)
        config.stage_adapter.initialization = "checkpoint"
        with self.assertRaises(ValueError):
            config.validate()

    def test_random_training_updates_encoders_and_shared_adapters_after_evaluation(self):
        for mode in ("random", "random_binding"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = _config(Path(directory))
                config.stage_adapter.initialization = mode
                model = random_model(config)
                trainer = StageAdapterTrainer(model, config, [_batch()], [_batch()])
                optimized = [id(p) for g in trainer.optimizer.param_groups for p in g["params"]]
                self.assertEqual(len(optimized), len(set(optimized)))
                self.assertEqual(set(optimized), {id(p) for p in model.parameters() if p.requires_grad})
                for epoch in (1, 2):
                    trainer.evaluate()
                    shared_before = model.binding_model.peptide_adapter.state_dict()
                    shared_before = {k: v.clone() for k, v in shared_before.items()}
                    encoder_before = model.peptide_encoder.embedding.weight.detach().clone()
                    frozen_before = model.frozen_binding_checksum()
                    trainer.train_stage_adapter_a_epoch(epoch)
                    self.assertTrue(any(not torch.equal(value, model.binding_model.peptide_adapter.state_dict()[key])
                                        for key, value in shared_before.items()))
                    self.assertEqual(model.frozen_binding_checksum(), frozen_before)
                    changed = not torch.equal(encoder_before, model.peptide_encoder.embedding.weight)
                    self.assertEqual(changed, mode == "random")
                    self.assertEqual(model.peptide_encoder.active, mode == "random")
                trainer.train_stage_adapter_b_round(1)

    def test_random_checkpoints_resume_exact_optimizer_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            config.stage_adapter.initialization = "random"
            config.paths.stage1_checkpoint = None
            config.paths.stage1_sha256 = None
            config.validate()
            model = random_model(config)
            source = initialise_stage_adapter_random(model, config=config)
            trainer = StageAdapterTrainer(model, config, [_batch()], [_batch()])
            for epoch in (1, 2):
                trainer.train_stage_adapter_a_epoch(epoch)
            evaluation = trainer.evaluate()
            common = dict(source_stage1=source, dataset_fingerprint="data",
                          effective_data={"train_samples": 4}, run_timestamp="test",
                          evaluation=evaluation)
            warmup = save_stage_adapter_warmup(directory, model=model, config=config,
                                              training_state=trainer.training_state(), **common)
            payload = load_stage_adapter_checkpoint(warmup)
            self.assertIn("peptide_encoder", payload["randomly_initialized_modules"])
            other_mode = copy.deepcopy(config)
            other_mode.stage_adapter.initialization = "random_binding"
            with self.assertRaisesRegex(ValueError, "initialization mode mismatch"):
                validate_stage_adapter_checkpoint(payload, config=other_mode,
                                                  dataset_fingerprint="data", effective_data={"train_samples": 4})
            trainer.train_stage_adapter_b_round(1)
            evaluation = trainer.evaluate()
            trainer.best_score = evaluation.selection_score
            common["evaluation"] = evaluation
            best = save_stage_adapter_best(directory, trainer=trainer, warmup_checkpoint=warmup, **common)
            last = save_stage_adapter_last(directory, trainer=trainer, warmup_checkpoint=warmup,
                                          best_checkpoint=best, **common)
            resumed = StageAdapterTrainer(random_model(config), config, [_batch()], [_batch()])
            resume_stage_adapter_last(load_stage_adapter_checkpoint(last), trainer=resumed,
                                      config=config, dataset_fingerprint="data",
                                      effective_data={"train_samples": 4})
            self.assertEqual(resumed.global_step, trainer.global_step)
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
            trainer.train_stage_adapter_b_round(2)
            resumed.train_stage_adapter_b_round(2)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)

    def test_old_v2_checkpoint_config_keeps_default_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            saved_config = config.to_dict()
            saved_config["stage_adapter"].pop("initialization")
            saved_config["stage_adapter"].pop("encoder_lr")
            payload = {"config": saved_config, "source_stage1": {"sha256": config.paths.stage1_sha256},
                       "dataset_fingerprint": "data", "effective_data": {}}
            validate_stage_adapter_checkpoint(payload, config=config,
                                              dataset_fingerprint="data", effective_data={})

    def test_real_random_plms_need_only_tokenizers_and_config_and_receive_gradients(self):
        from transformers import BertConfig, BertTokenizer, ESMCConfig, ESMCTokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            config.stage_adapter.initialization = "random"
            config.stage_adapter.gradient_checkpointing = True
            config.model.peptide_input_dim = config.model.hla_input_dim = config.model.tcr_input_dim = 8
            ESMCConfig(d_model=8, n_heads=2, n_layers=1).save_pretrained(config.paths.esmc_model)
            ESMCTokenizer().save_pretrained(config.paths.esmc_model)
            BertConfig(hidden_size=8, num_attention_heads=2, num_hidden_layers=1,
                       intermediate_size=16, vocab_size=25).save_pretrained(config.paths.tcr_bert_model)
            vocab = config.paths.tcr_bert_model / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("ACDEFGHIKLMNPQRSTVWY")) + "\n")
            BertTokenizer(vocab_file=str(vocab), do_lower_case=False).save_pretrained(config.paths.tcr_bert_model)
            with patch("safetensors.torch.load_file", side_effect=AssertionError("Pretrained weights loaded")), \
                 patch("source.plm_unified.finetuning._load_trusted_state_dict", side_effect=AssertionError("Pretrained weights loaded")), \
                 patch("transformers.models.esmc.modeling_esmc._te_available", False):
                # Use ESM-C's real PyTorch implementation for this CPU check.
                model = build_progressive_stage_adapter_model(config, torch.device("cpu"))
            trainer = StageAdapterTrainer(model, config, [_batch()], [_batch()])
            # Accelerated attention kernels require CUDA; exercise real SDPA on CPU.
            with patch("transformers.models.esmc.modeling_esmc._xformers_available", False):
                trainer.evaluate()
                trainer.train_stage_adapter_a_epoch(1)
            for encoder in (model.peptide_encoder, model.hla_encoder, model.tcr_encoder):
                self.assertEqual(encoder.metadata()["initialization"], "random")
                trainable = [p for p in encoder.parameters() if p.requires_grad]
                self.assertTrue(trainable)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
            self.assertFalse(model.tcr_encoder.model.pooler.dense.weight.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in model.peptide_encoder.model.lm_head.parameters()))

    def test_cli_from_scratch_runs_warmup_and_training_without_source_checkpoint(self):
        source_path = str(Path(__file__).resolve().parents[2] / "source")
        with patch.object(sys, "path", [source_path] + sys.path):
            entry = importlib.import_module("train_plm_stage_adapter")
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            config.paths.stage1_checkpoint = None
            config.paths.stage1_sha256 = None
            for class_dir in config.stage_adapter.class_dirs:
                path = config.data_dir() / class_dir
                path.mkdir(parents=True)
                (path / "train.csv").touch()
                (path / "val.csv").touch()
            argv = ["--from-scratch", "--swanlab-mode", "disabled",
                    "--run-timestamp", "20261010_000000_000000"]
            args = entry.parse_args(argv)
            data = (SimpleNamespace(dataset_fingerprint="data"), [_batch()], [_batch()],
                    {"train_samples": 4, "validation_samples": 4})
            with patch.object(entry, "parse_args", return_value=args), \
                 patch.object(entry, "load_stage_adapter_config", return_value=config), \
                 patch.object(entry, "_audit_and_load", return_value=data), \
                 patch.object(entry, "build_progressive_stage_adapter_model", side_effect=lambda c, d: random_model(c)), \
                 patch.object(entry, "initialise_stage_adapter_from_checkpoint", side_effect=AssertionError("Stage checkpoint loaded")), \
                 redirect_stdout(io.StringIO()):
                entry.main()
            warmups = list(config.warmup_dir().glob("*.pt"))
            lasts = list(config.run_dir().glob("stage_adapter_last_*.pt"))
            self.assertEqual(len(warmups), 1)
            self.assertEqual(len(lasts), 1)
            payload = load_stage_adapter_checkpoint(lasts[0])
            self.assertEqual(payload["source_stage1"]["kind"], "random_binding")
            self.assertEqual(payload["training_state"]["stage_adapter_b_round"], 2)


if __name__ == "__main__":
    unittest.main()
