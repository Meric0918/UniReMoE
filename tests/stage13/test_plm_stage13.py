from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from source.plm_unified.config import ModelConfig
from source.plm_unified.cache import CacheLocator
from source.plm_unified.stage13_config import (
    Stage13ExperimentConfig,
    Stage13GraphConfig,
    Stage13Paths,
    load_stage13_config,
)
from source.plm_unified.stage13_graph import GraphArtifact, GraphSampler, build_graph
from source.plm_unified.stage13_model import Stage13BindingModel


def _artifact(tmp_path: Path) -> GraphArtifact:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"manifest_version": 1}), encoding="utf-8"
    )
    entities = {"peptide": ["P1", "P2"], "hla": ["H1"], "tcr": ["T1"]}
    for kind, values in entities.items():
        (tmp_path / f"nodes_{kind}.json").write_text(json.dumps(values), encoding="utf-8")
        np.save(tmp_path / f"features_{kind}.npy", np.ones((len(values), 960), dtype=np.float16))
    adjacency = {
        "p_h": ([0, 1, 1], [0]),
        "h_p": ([0, 2], [0, 1]),
        "p_t": ([0, 1, 2], [0]),
        "t_p": ([0, 1], [0]),
    }
    for relation, (indptr, indices) in adjacency.items():
        np.save(tmp_path / f"adj_{relation}_indptr.npy", np.asarray(indptr, dtype=np.int64))
        np.save(tmp_path / f"adj_{relation}_indices.npy", np.asarray(indices, dtype=np.int64))
    return GraphArtifact.load(tmp_path)


def test_stage13_config_loads_without_touching_stage1_config():
    config = load_stage13_config(Path(__file__).parents[2] / "configs" / "plm_stage13.yaml")
    assert config.graph.fanouts == [16, 8]
    assert config.data.ph_validation_splits


def test_sampler_excludes_both_directions_of_query_edge(tmp_path: Path):
    artifact = _artifact(tmp_path)
    sampler = GraphSampler(artifact, [2, 2], seed=7)
    peptide = artifact.global_id("peptide", 0)
    tcr = artifact.global_id("tcr", 0)
    sampled = sampler.sample(
        [peptide, tcr],
        np.zeros((2, 960), dtype=np.float32),
        root_types=[0, 2],
        excluded_edges=[(peptide, tcr)],
        seed=3,
    )
    assert sampled.relation_indices[0]["p_t"][0].eq(-1).all()
    assert sampled.relation_indices[0]["t_p"][1].eq(-1).all()


def test_unknown_root_is_an_isolated_inductive_node(tmp_path: Path):
    artifact = _artifact(tmp_path)
    sampled = GraphSampler(artifact, [2, 2], seed=7).sample(
        [-1, -1],
        np.zeros((2, 960), dtype=np.float32),
        root_types=[0, 2],
        seed=3,
    )
    assert sampled.root_positions.tolist() == [0, 1]
    assert all(value.numel() == 0 or value.eq(-1).all() for value in sampled.relation_indices[0].values())


def test_graph_token_fusion_preserves_stage1_residue_pipeline(tmp_path: Path):
    artifact = _artifact(tmp_path)
    sampled = GraphSampler(artifact, [2, 2], seed=7).sample(
        [artifact.global_id("peptide", 0), artifact.global_id("hla", 0)],
        np.zeros((2, 960), dtype=np.float32),
        root_types=[0, 1],
        seed=3,
    )
    model_config = ModelConfig(
        adapter_hidden_dim=8,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
    )
    model = Stage13BindingModel(
        model_config,
        Stage13GraphConfig(hidden_dim=8, num_layers=2, fanouts=[2, 2]),
    )
    output = model(
        "phla",
        torch.randn(1, 15, 960),
        torch.ones(1, 15, dtype=torch.bool),
        torch.randn(1, 34, 960),
        torch.ones(1, 34, dtype=torch.bool),
        sampled,
    )
    assert output.logits.shape == (1, 2)
    assert output.pooled_embedding.shape == (1, 8)


def test_build_graph_deduplicates_positive_edges_and_ignores_negatives(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "dataset"
    (data_root / "PH").mkdir(parents=True)
    (data_root / "PT").mkdir(parents=True)
    (data_root / "PH" / "train.csv").write_text(
        "peptide,HLA,label\nP1,H1,1\nP1,H1,1\nP1,H2,0\nP2,H1,1\n",
        encoding="utf-8",
    )
    (data_root / "PT" / "train.csv").write_text(
        "peptide,tcr,label\nP1,T1,1\nP1,T2,0\nP2,T1,1\n",
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache" / "fold_1"
    for kind in ("peptide", "hla", "tcr"):
        path = cache_root / kind
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeCache:
        def __init__(self, entity_dir, *, expected_hidden_dim):
            self.kind = Path(entity_dir).name
            self.hidden_dim = expected_hidden_dim
            self.max_length = 4
            self.index = {
                value: CacheLocator(0, index, 1)
                for index, value in enumerate(
                    {"peptide": ["P1", "P2"], "hla": ["H1"], "tcr": ["T1"]}[self.kind]
                )
            }

        def resolve(self, sequence):
            return self.index[sequence]

        def get_many(self, locators):
            hidden = np.zeros((len(locators), 4, self.hidden_dim), dtype=np.float16)
            mask = np.ones((len(locators), 4), dtype=bool)
            return hidden, mask

        def close(self):
            pass

    monkeypatch.setattr("source.plm_unified.stage13_graph.EmbeddingCache", FakeCache)
    config = Stage13ExperimentConfig(
        paths=Stage13Paths(
            project_root=tmp_path,
            data_root=data_root,
            residue_cache_root=tmp_path / "cache",
            graph_root=tmp_path / "graphs",
            output_root=tmp_path / "output",
            swanlog_root=tmp_path / "logs",
        ),
        graph=Stage13GraphConfig(build_batch_size=2),
    )
    artifact = build_graph(config)
    assert artifact.manifest["positive_edges"] == {"p_h": 2, "p_t": 2}
    assert artifact.counts == {"peptide": 2, "hla": 1, "tcr": 1}
    assert artifact.relation_global_neighbors(artifact.global_id("peptide", 0), "p_h").tolist() == [2]
