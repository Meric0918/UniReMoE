"""Stage11 multi-positive targets, phase isolation, cache workflows and exact resume."""

import copy
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from source.plm_unified.cache import EmbeddingCacheWriter
from source.plm_unified.config import ModelConfig, PathsConfig
from source.plm_unified.encoders import checkpoint_fingerprint
from source.plm_unified.stage8_config import Stage8RuntimeConfig
from source.plm_unified.stage11_config import Stage11ExperimentConfig, Stage11TrainingConfig, load_stage11_config
from source.plm_unified.stage11_data import (ContrastiveCollator, CyclingRelationSampler, RelationDataset,
    Stage11Loaders, model_inputs, move_cached_batch, open_stage11_caches, prepare_stage11_data)
from source.plm_unified.stage11_loss import multi_positive_clip_loss
from source.plm_unified.stage11_model import Stage11BindingModel
from source.plm_unified.stage11_trainer import Stage11Trainer
from source.plm_unified.trainer import set_global_seed

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def experiment(tmp_path):
    config = Stage11ExperimentConfig(
        paths=PathsConfig(project_root=tmp_path, data_root=tmp_path / "dataset", cache_root=tmp_path / "cache",
                          output_root=tmp_path / "models", swanlog_root=tmp_path / "logs",
                          esmc_model=tmp_path / "esmc", tcr_bert_model=tmp_path / "tcrbert"),
        model=ModelConfig(peptide_input_dim=12, hla_input_dim=12, tcr_input_dim=6, model_dim=8,
                          adapter_hidden_dim=8, num_attention_heads=2, feedforward_dim=16,
                          peptide_min_length=8, peptide_max_length=8, hla_max_length=4, tcr_max_length=4),
        runtime=Stage8RuntimeConfig(seed=17, device="cpu", num_workers=0),
        stage11=Stage11TrainingConfig(projection_dim=8, batch_size=4, contrastive_batch_size=3,
                                      eval_batch_size=4, min_rounds=1, max_rounds=2,
                                      mixed_precision="none", scheduler_warmup_ratio=0))
    config.swanlab.mode = "disabled"
    for name in ("esmc_model", "tcr_bert_model"):
        path = getattr(config.paths, name)
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model": name}))
    aa = "ACDEFGHIKLMNPQRSTVWY"
    header = ["peptide", "HLA", "tcr", "PH", "PT", "PHT", "class_code", "split_type"]
    validation = []
    for kind, code in enumerate(("000", "010", "100", "111")):
        train_rows = [[aa[kind] * 7 + aa[i], aa[i] * 4, aa[i + 4] * 4,
                       *map(int, code), code, "train"] for i in range(4)]
        val_row = [aa[kind + 5] * 8, "WYAC", "DEFG", *map(int, code), code, "val"]
        validation.append(val_row)
        for split, rows in (("train", train_rows), ("val", [val_row])):
            write_csv(config.paths.data_root / "PHT/classes" / f"PHT_{code}" / f"{split}.csv", header, rows)
    # One retained PH-only positive, one known negative, one excluded validation pair.
    write_csv(config.paths.data_root / "PH/train.csv", ["peptide", "HLA", "label", "split_type"],
              [["PPPPPPPP", "ACDE", 1, "train"], ["QQQQQQQQ", "ACDE", 0, "train"],
               [validation[3][0], "WYAC", 1, "train"]])
    for name in ("test.csv", *(f"benchmarks/test_unseen_{v}.csv" for v in ("peptide", "hla", "tcr", "triple"))):
        write_csv(config.paths.data_root / "PHT" / name, header,
                  [row[:-1] + ["test"] for row in validation])
    data = prepare_stage11_data(config)
    rng = np.random.default_rng(17)
    for entity, sequences in data.sequences().items():
        strings = sorted(sequences)
        dim, length = getattr(config.model, f"{entity}_input_dim"), getattr(config.model, f"{entity}_max_length")
        base = config.paths.tcr_bert_model if entity == "tcr" else config.paths.esmc_model
        with EmbeddingCacheWriter(config.cache_dir(), entity, hidden_dim=dim, max_length=length,
                model_metadata={"checkpoint_fingerprint": checkpoint_fingerprint(base)}, shard_size=8, chunk_rows=1) as writer:
            writer.add_batch(strings, rng.standard_normal((len(strings), length, dim)),
                             np.ones((len(strings), length), dtype=bool), np.array([len(s) for s in strings]))
    shutil.copytree(config.cache_dir(), config.cache_dir("test"))
    raw = config.to_dict()
    raw.pop("source_path")
    config.source_path = tmp_path / "config.yaml"
    config.source_path.write_text(yaml.safe_dump(raw))
    return config, data


def make_trainer(experiment, timestamp="20261018_120000_000001"):
    config, data = experiment
    set_global_seed(config.runtime.seed)
    caches, metadata = open_stage11_caches(config)
    loaders = Stage11Loaders(config, data, caches)
    model = Stage11BindingModel(config.model, projection_dim=config.stage11.projection_dim)
    trainer = Stage11Trainer(model, config, data, loaders, metadata, run_timestamp=timestamp)
    return trainer, caches


def close(trainer, caches):
    trainer.loaders.close()
    for cache in caches.values():
        cache.close()


def test_single_positive_matches_clip_and_rectangular_multi_positive():
    left, right = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    score = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / 0.2
    expected = (F.cross_entropy(score, torch.arange(4)) + F.cross_entropy(score.T, torch.arange(4))) / 2
    result = multi_positive_clip_loss(left, right, torch.eye(4, dtype=torch.bool))
    torch.testing.assert_close(result.total, expected)
    mask = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    left, right = torch.randn(2, 8, requires_grad=True), torch.randn(3, 8, requires_grad=True)
    score = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / 0.2
    a = -(score.log_softmax(-1) * mask).sum(-1) / mask.sum(-1)
    b = -(score.T.log_softmax(-1) * mask.T).sum(-1) / mask.T.sum(-1)
    result = multi_positive_clip_loss(left, right, mask)
    torch.testing.assert_close(result.total, (a.mean() + b.mean()) / 2)
    result.total.backward()
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()


@pytest.mark.parametrize("shape", [(1, 1), (3, 2)])
def test_no_negatives_and_zero_vectors_are_finite(shape):
    left, right = torch.zeros(shape[0], 8, requires_grad=True), torch.zeros(shape[1], 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = multi_positive_clip_loss(left, right, torch.ones(shape, dtype=torch.bool))
    assert not result.valid and result.total.item() == 0 and result.total.dtype == torch.float32
    result.total.backward()
    assert torch.isfinite(left.grad).all()
    if shape == (3, 2):
        result = multi_positive_clip_loss(left, right, torch.tensor([[1, 0], [0, 1], [0, 0]]))
        assert result.valid_left == 2 and result.valid_right == 2
        assert torch.isfinite(result.total)


def test_data_merge_validation_exclusion_and_conflicts(experiment):
    config, data = experiment
    assert data.report["ph_positive_validation_overlap_excluded"] == 1
    assert ("PPPPPPPP", "ACDE") in data.ph_pairs
    assert not any((r.peptide, r.hla) in data.ph_labels for r in data.audit.val_records)
    assert len(data.pht_pairs) == 4
    assert all(key in {((r.peptide, r.hla), r.tcr) for r in data.audit.train_records}
               for key in data.pht_labels)
    record = next(r for r in data.audit.train_records if r.ph == 0)
    with (config.paths.data_root / "PH/train.csv").open("a") as handle:
        csv.writer(handle).writerow([record.peptide, record.hla, 1, "train"])
    with pytest.raises(ValueError, match="Cross-task PH label conflict"):
        prepare_stage11_data(config)


def test_deduplication_includes_unsampled_known_positive(experiment):
    config, data = experiment
    caches, _ = open_stage11_caches(config)
    try:
        pairs = data.ph_pairs[:3]
        dataset = RelationDataset(pairs, caches, "ph")
        labels = dict(data.ph_labels)
        labels[(pairs[0][0], pairs[1][1])] = 1
        collator = ContrastiveCollator(caches, labels, "ph")
        samples = [dataset[0], dataset[1], dataset[0]]
        batch = collator(samples)
        assert batch["peptide_hidden"].shape[0] == len(set(p[0] for p in pairs[:2]))
        assert batch["hla_hidden"].shape[0] == len(set(p[1] for p in pairs[:2]))
        assert batch["positive_mask"][0, 1]
        assert batch["sampled_relations"] == 3
        for cache in collator.caches.values():
            cache.close()
    finally:
        for cache in caches.values():
            cache.close()


@pytest.mark.parametrize("task,active", [
    ("ph", {"peptide_adapter", "hla_adapter", "p_contrastive", "h_contrastive"}),
    ("pht", {"peptide_adapter", "hla_adapter", "tcr_adapter", "phla_cross_attention", "ph_contrastive", "t_contrastive"}),
    ("classification", {"peptide_adapter", "hla_adapter", "tcr_adapter", "phla_cross_attention",
                        "triplet_cross_attention", "triplet_pooling", "triplet_classifier"}),
])
def test_phase_gradients_and_inactive_adamw_parameters(experiment, task, active):
    trainer, caches = make_trainer(experiment)
    try:
        loader = trainer.loaders.classification if task == "classification" else trainer.loaders.contrastive(task, 1)
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        result = trainer._pass([next(iter(loader))], task)
        assert result["optimizer_steps"] == 1
        modules = {name.split(".")[0] for name, p in trainer.model.named_parameters() if p.grad is not None}
        assert modules == active
        for name, value in trainer.model.state_dict().items():
            if name.split(".")[0] not in active:
                assert torch.equal(value, before[name])
        raw = next(iter(trainer.loaders.classification))
        raw["peptide_hidden"] = raw["peptide_hidden"].float().requires_grad_(True)
        moved = move_cached_batch(raw, torch.device("cpu"))
        assert not moved["peptide_hidden"].requires_grad
    finally:
        close(trainer, caches)


def test_masking_attention_direction_and_inference_skips_projections(experiment):
    trainer, caches = make_trainer(experiment)
    try:
        batch = move_cached_batch(next(iter(trainer.loaders.classification)), torch.device("cpu"))
        batch["peptide_mask"][:, -1] = False
        batch["hla_mask"][:, -1] = False
        batch["tcr_mask"][:, -1] = False
        model = trainer.model.eval()
        handles = [getattr(model, f"{name}_contrastive").register_forward_hook(
            lambda *_: pytest.fail("Inference invoked a contrastive head")) for name in ("p", "h", "ph", "t")]
        output = model.forward_pht(*model_inputs(batch), return_attention=True)
        assert output.pht_logits.shape == (4, 2)
        assert output.ph_attention.shape == (4, 2, 8, 4)
        assert output.pht_attention.shape == (4, 2, 8, 4)
        for entity in ("peptide", "hla", "tcr"):
            batch[f"{entity}_hidden"][:, -1] = 1e4
        altered = model.forward_pht(*model_inputs(batch)).pht_logits
        torch.testing.assert_close(output.pht_logits, altered)
        for handle in handles:
            handle.remove()
    finally:
        close(trainer, caches)


def test_ph_rotation_and_state_restore():
    sampler = CyclingRelationSampler(11, 5)
    first = sampler.take(7)
    state = sampler.state_dict()
    rest = sampler.take(4)
    assert len(set(first + rest)) == 11
    restored = CyclingRelationSampler(11, 999)
    restored.load_state_dict(state)
    assert restored.take(4) == rest
    assert restored.take(23) == sampler.take(23)


def test_round_resume_exact_and_contract_rejection(experiment, tmp_path):
    trainer, caches = make_trainer(experiment)
    first = trainer.train_round(1)
    checkpoint = tmp_path / "round1.pt"
    shutil.copyfile(trainer.path("last"), checkpoint)
    second = trainer.train_round(2)
    expected = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    expected_step = trainer.global_step
    close(trainer, caches)
    resumed, caches = make_trainer(experiment, timestamp="20261018_120000_000002")
    try:
        resumed.resume(checkpoint)
        actual = resumed.train_round(2)
        assert first["training"]["ph"]["planned_batches"] == first["training"]["pht"]["planned_batches"]
        assert actual["training"] == second["training"]
        assert resumed.global_step == expected_step
        for key, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        for key, state in resumed.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_optimizer["state"][key][name], rtol=0, atol=0)
        resumed.contract["data_fingerprint"] = "changed"
        with pytest.raises(ValueError, match="contract mismatch"):
            resumed.resume(checkpoint)
    finally:
        close(resumed, caches)


@pytest.mark.parametrize("condition,tasks", [("none", {"classification"}), ("ph", {"ph", "classification"}),
                                            ("pht", {"pht", "classification"})])
def test_ablation_phases(experiment, condition, tasks):
    config, _ = experiment
    config.stage11.condition = condition
    trainer, caches = make_trainer(experiment)
    try:
        result = trainer.train_round(1)
        assert set(result["training"]) == tasks
    finally:
        close(trainer, caches)


def test_spawned_workers_and_end_to_end_cli(experiment):
    import sys
    config, _ = experiment
    config.runtime.num_workers = 1
    trainer, caches = make_trainer(experiment)
    try:
        batch = next(iter(trainer.loaders.contrastive("pht", 1)))
        assert batch["positive_mask"].ndim == 2
    finally:
        close(trainer, caches)
    common = ["--config", str(config.source_path), "--swanlab-mode", "disabled"]
    for script, args in (
        ("audit_plm_stage11_data.py", []),
        ("precompute_plm_stage11_embeddings.py", ["--audit-only"]),
        ("train_plm_stage11.py", ["--max-rounds", "1", "--min-rounds", "1"]),
    ):
        result = subprocess.run([sys.executable, str(ROOT / "source" / script), *common, *args],
                                cwd=ROOT, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
    checkpoint = next(config.run_dir().glob("stage11_best_*.pt"))
    result = subprocess.run([sys.executable, str(ROOT / "source/evaluate_plm_stage11.py"),
                             *common, "--checkpoint", str(checkpoint), "--split", "all"],
                            cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    summary = next(config.run_dir().glob("evaluation_*/summary.json"))
    assert len(json.loads(summary.read_text())["results"]) == 5


def test_config_defaults_and_invalid_condition():
    config = load_stage11_config(ROOT / "configs/plm_stage11.yaml")
    assert config.stage11.temperature == 0.2 and config.stage11.condition == "both"
    config.stage11.condition = "joint"
    with pytest.raises(ValueError, match="condition"):
        config.validate()
