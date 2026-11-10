"""Train-only heterogeneous graph artifacts and deterministic GraphSAGE sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .cache import EmbeddingCache
from .data import normalise_sequence
from .stage13_config import Stage13ExperimentConfig


GRAPH_MANIFEST_VERSION = 1
NODE_TYPES = ("peptide", "hla", "tcr")
RELATIONS = ("p_h", "h_p", "p_t", "t_p")
TYPE_CODE = {name: index for index, name in enumerate(NODE_TYPES)}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_binary_edges(
    path: str | Path,
    receptor_column: str,
    *,
    limit: int | None = None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    positives: set[tuple[str, str]] = set()
    negatives: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", receptor_column, "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and row_number - 1 > limit:
                break
            peptide = normalise_sequence(row["peptide"])
            receptor = normalise_sequence(row[receptor_column])
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"Label must be 0 or 1 at {path}:{row_number}")
            (positives if label else negatives).add((peptide, receptor))
    conflict = positives & negatives
    if conflict:
        example = next(iter(sorted(conflict)))
        raise ValueError(f"Conflicting train edge in {path}: {example}")
    return positives, negatives


def _csr(adjacency: Sequence[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(adjacency) + 1, dtype=np.int64)
    values: list[int] = []
    for index, neighbors in enumerate(adjacency):
        values.extend(sorted(neighbors))
        indptr[index + 1] = len(values)
    return indptr, np.asarray(values, dtype=np.int64)


def _masked_mean(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    valid = np.asarray(mask, dtype=np.float32)[..., None]
    denominator = np.maximum(valid.sum(axis=1), 1.0)
    return (values * valid).sum(axis=1) / denominator


@dataclass
class GraphArtifact:
    """Memory-mappable train graph and pooled frozen-encoder node features."""

    root: Path
    manifest: dict
    sequences: dict[str, tuple[str, ...]]
    features: dict[str, np.ndarray]
    adjacency: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def load(cls, root: str | Path) -> "GraphArtifact":
        root_path = Path(root).resolve()
        manifest_path = root_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Graph manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manifest_version") != GRAPH_MANIFEST_VERSION:
            raise ValueError("Unsupported Stage13 graph manifest version")
        sequences = {
            kind: tuple(json.loads((root_path / f"nodes_{kind}.json").read_text(encoding="utf-8")))
            for kind in NODE_TYPES
        }
        features = {
            kind: np.load(root_path / f"features_{kind}.npy", mmap_mode="r")
            for kind in NODE_TYPES
        }
        adjacency = {
            relation: (
                np.load(root_path / f"adj_{relation}_indptr.npy", mmap_mode="r"),
                np.load(root_path / f"adj_{relation}_indices.npy", mmap_mode="r"),
            )
            for relation in RELATIONS
        }
        for kind in NODE_TYPES:
            if len(sequences[kind]) != len(features[kind]):
                raise ValueError(f"Graph node/feature count mismatch for {kind}")
        return cls(root_path, manifest, sequences, features, adjacency)

    @property
    def counts(self) -> dict[str, int]:
        return {kind: len(self.sequences[kind]) for kind in NODE_TYPES}

    @property
    def offsets(self) -> dict[str, int]:
        peptide_count = self.counts["peptide"]
        hla_count = self.counts["hla"]
        return {"peptide": 0, "hla": peptide_count, "tcr": peptide_count + hla_count}

    @property
    def total_nodes(self) -> int:
        return sum(self.counts.values())

    def global_id(self, kind: str, local_id: int) -> int:
        if kind not in TYPE_CODE:
            raise ValueError(f"Unknown graph node type: {kind}")
        if not 0 <= local_id < self.counts[kind]:
            raise IndexError(local_id)
        return self.offsets[kind] + local_id

    def decode_id(self, global_id: int) -> tuple[str, int]:
        if not 0 <= int(global_id) < self.total_nodes:
            raise IndexError(global_id)
        value = int(global_id)
        offsets = self.offsets
        if value < offsets["hla"]:
            return "peptide", value
        if value < offsets["tcr"]:
            return "hla", value - offsets["hla"]
        return "tcr", value - offsets["tcr"]

    def type_code_for_id(self, global_id: int) -> int:
        if int(global_id) < 0:
            return -1
        return TYPE_CODE[self.decode_id(int(global_id))[0]]

    def lookup(self, kind: str, sequence: str) -> int:
        # The graph is loaded once per process; a dict is faster than scanning JSON.
        mapping = getattr(self, "_lookup_cache", None)
        if mapping is None:
            mapping = {
                node_kind: {sequence: index for index, sequence in enumerate(values)}
                for node_kind, values in self.sequences.items()
            }
            setattr(self, "_lookup_cache", mapping)
        local = mapping[kind].get(normalise_sequence(sequence))
        return -1 if local is None else self.global_id(kind, local)

    def feature_for_ids(self, global_ids: Sequence[int]) -> np.ndarray:
        """Return padded float32 [N, max_input_dim] pooled encoder features."""

        result = np.zeros((len(global_ids), 960), dtype=np.float32)
        for row, global_id in enumerate(global_ids):
            if int(global_id) < 0:
                continue
            kind, local_id = self.decode_id(int(global_id))
            feature = np.asarray(self.features[kind][local_id], dtype=np.float32)
            result[row, : feature.shape[-1]] = feature
        return result

    def neighbors(self, global_id: int, relation: str) -> np.ndarray:
        kind, local_id = self.decode_id(int(global_id))
        expected = {
            "p_h": "peptide",
            "h_p": "hla",
            "p_t": "peptide",
            "t_p": "tcr",
        }[relation]
        if kind != expected:
            return np.empty(0, dtype=np.int64)
        indptr, indices = self.adjacency[relation]
        return np.asarray(indices[int(indptr[local_id]) : int(indptr[local_id + 1])])

    def relation_global_neighbors(self, global_id: int, relation: str) -> np.ndarray:
        """Return global IDs for one relation's local adjacency row."""

        if int(global_id) < 0:
            return np.empty(0, dtype=np.int64)
        local_neighbors = self.neighbors(global_id, relation)
        target_kind = {"p_h": "hla", "h_p": "peptide", "p_t": "tcr", "t_p": "peptide"}[relation]
        offset = self.offsets[target_kind]
        return local_neighbors.astype(np.int64, copy=False) + offset


def _write_features(
    cache: EmbeddingCache,
    sequences: Sequence[str],
    path: Path,
    *,
    batch_size: int,
    padded_dim: int = 960,
) -> None:
    values = np.zeros((len(sequences), padded_dim), dtype=np.float16)
    locators = [cache.resolve(sequence) for sequence in sequences]
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        hidden, mask = cache.get_many(locators[start:end])
        pooled = _masked_mean(hidden, mask).astype(np.float16, copy=False)
        values[start:end, : pooled.shape[1]] = pooled
    np.save(path, values)


def build_graph(
    config: Stage13ExperimentConfig,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> GraphArtifact:
    """Build a train-positive-only graph from the two Stage13 binary datasets."""

    data_root = config.paths.data_root
    ph_path = data_root / config.data.ph_train
    pt_path = data_root / config.data.pt_train
    if not ph_path.is_file() or not pt_path.is_file():
        raise FileNotFoundError(f"Stage13 training files not found under {data_root}")
    ph_positive, _ = _read_binary_edges(ph_path, "HLA", limit=limit)
    pt_positive, _ = _read_binary_edges(pt_path, "tcr", limit=limit)
    for task, train_edges, splits, receptor_column in (
        (
            "phla",
            ph_positive,
            config.data.ph_validation_splits + config.data.ph_test_splits,
            "HLA",
        ),
        (
            "ptcr",
            pt_positive,
            config.data.pt_validation_splits + config.data.pt_test_splits,
            "tcr",
        ),
    ):
        for split in splits:
            holdout = (
                Path(data_root)
                / ("PH" if task == "phla" else "PT")
                / "benchmarks"
                / f"{split}.csv"
            )
            if not holdout.is_file():
                continue
            holdout_positive, holdout_negative = _read_binary_edges(
                holdout, receptor_column
            )
            overlap = train_edges & (holdout_positive | holdout_negative)
            if overlap:
                example = next(iter(sorted(overlap)))
                raise ValueError(f"Train/holdout pair overlap in {holdout}: {example}")
    peptides = sorted({p for p, _ in ph_positive | pt_positive})
    hlas = sorted({h for _, h in ph_positive})
    tcrs = sorted({t for _, t in pt_positive})
    if not peptides or not hlas or not tcrs:
        raise ValueError("Stage13 graph needs positive PH and PT edges")
    p_index = {value: index for index, value in enumerate(peptides)}
    h_index = {value: index for index, value in enumerate(hlas)}
    t_index = {value: index for index, value in enumerate(tcrs)}
    adjacency_sets = {
        "p_h": [set() for _ in peptides],
        "h_p": [set() for _ in hlas],
        "p_t": [set() for _ in peptides],
        "t_p": [set() for _ in tcrs],
    }
    for peptide, hla in ph_positive:
        p_id, h_id = p_index[peptide], h_index[hla]
        adjacency_sets["p_h"][p_id].add(h_id)
        adjacency_sets["h_p"][h_id].add(p_id)
    for peptide, tcr in pt_positive:
        p_id, t_id = p_index[peptide], t_index[tcr]
        adjacency_sets["p_t"][p_id].add(t_id)
        adjacency_sets["t_p"][t_id].add(p_id)

    output = config.graph_dir()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Graph artifact already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    caches = {
        "peptide": EmbeddingCache(
            config.residue_cache_dir() / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
        ),
        "hla": EmbeddingCache(
            config.residue_cache_dir() / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
        ),
        "tcr": EmbeddingCache(
            config.residue_cache_dir() / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
        ),
    }
    try:
        for kind, values in (("peptide", peptides), ("hla", hlas), ("tcr", tcrs)):
            for sequence in values:
                try:
                    caches[kind].resolve(sequence)
                except KeyError as exc:
                    raise KeyError(f"Graph node is missing from {kind} cache: {sequence}") from exc
            (output / f"nodes_{kind}.json").write_text(
                json.dumps(values, ensure_ascii=False), encoding="utf-8"
            )
            _write_features(
                caches[kind],
                values,
                output / f"features_{kind}.npy",
                batch_size=config.graph.build_batch_size,
            )
    finally:
        for cache in caches.values():
            cache.close()
    for relation, rows in adjacency_sets.items():
        indptr, indices = _csr(rows)
        np.save(output / f"adj_{relation}_indptr.npy", indptr)
        np.save(output / f"adj_{relation}_indices.npy", indices)
    manifest = {
        "manifest_version": GRAPH_MANIFEST_VERSION,
        "stage": "stage13",
        "edge_policy": "positive_train_only",
        "node_counts": {"peptide": len(peptides), "hla": len(hlas), "tcr": len(tcrs)},
        "positive_edges": {"p_h": len(ph_positive), "p_t": len(pt_positive)},
        "relations": list(RELATIONS),
        "graph_hidden_dim": config.graph.hidden_dim,
        "num_layers": config.graph.num_layers,
        "fanouts": list(config.graph.fanouts),
        "source_files": {
            "ph_train": {"path": str(ph_path), "sha256": file_sha256(ph_path)},
            "pt_train": {"path": str(pt_path), "sha256": file_sha256(pt_path)},
        },
        "cache_manifests": {
            kind: file_sha256(config.residue_cache_dir() / kind / "manifest.json")
            for kind in NODE_TYPES
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return GraphArtifact.load(output)


@dataclass
class SampledGraph:
    """A deduplicated multi-hop neighborhood represented as tensor blocks."""

    ids_by_depth: list[torch.Tensor]
    features_by_depth: list[torch.Tensor]
    relation_indices: list[dict[str, torch.Tensor]]
    root_positions: torch.Tensor
    types_by_depth: list[torch.Tensor]

    def to(self, device: torch.device) -> "SampledGraph":
        return SampledGraph(
            ids_by_depth=[value.to(device) for value in self.ids_by_depth],
            features_by_depth=[value.to(device) for value in self.features_by_depth],
            relation_indices=[
                {key: value.to(device) for key, value in relation.items()}
                for relation in self.relation_indices
            ],
            root_positions=self.root_positions.to(device),
            types_by_depth=[value.to(device) for value in self.types_by_depth],
        )


class GraphSampler:
    """Deterministic, relation-aware neighbor sampler with query-edge masking."""

    def __init__(self, artifact: GraphArtifact, fanouts: Sequence[int], seed: int):
        if not fanouts or any(int(value) < 1 for value in fanouts):
            raise ValueError("fanouts must contain positive values")
        self.artifact = artifact
        self.fanouts = tuple(int(value) for value in fanouts)
        self.seed = int(seed)

    def _sample_row(
        self,
        parent: int,
        relation: str,
        fanout: int,
        rng: np.random.Generator,
        excluded: Mapping[int, set[int]],
    ) -> list[int]:
        if int(parent) < 0:
            return []
        candidates = self.artifact.relation_global_neighbors(parent, relation)
        blocked = excluded.get(int(parent), set())
        if blocked:
            candidates = np.asarray([value for value in candidates if int(value) not in blocked])
        if len(candidates) > fanout:
            candidates = rng.choice(candidates, size=fanout, replace=False)
        return [int(value) for value in candidates]

    def sample(
        self,
        root_ids: Sequence[int],
        root_features: np.ndarray | torch.Tensor,
        *,
        root_types: Sequence[int] | None = None,
        excluded_edges: Iterable[tuple[int, int]] = (),
        seed: int | None = None,
    ) -> SampledGraph:
        # Give each unknown query entity its own negative sentinel.  A single -1
        # cannot be deduplicated because peptide and receptor roots may both be
        # absent from the train graph in the same batch.
        roots = [int(value) if int(value) >= 0 else -(index + 1) for index, value in enumerate(root_ids)]
        if not roots:
            raise ValueError("Cannot sample a graph with no roots")
        excluded: dict[int, set[int]] = {}
        for left, right in excluded_edges:
            if int(left) < 0 or int(right) < 0:
                continue
            excluded.setdefault(int(left), set()).add(int(right))
            excluded.setdefault(int(right), set()).add(int(left))
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        unique_roots = list(dict.fromkeys(roots))
        root_positions = torch.tensor(
            [unique_roots.index(value) for value in roots], dtype=torch.long
        )
        if isinstance(root_features, torch.Tensor):
            root_values = root_features.detach().cpu().numpy()
        else:
            root_values = np.asarray(root_features)
        if root_values.shape[0] != len(roots):
            raise ValueError("root_features and root_ids have different row counts")
        if root_types is None:
            root_types = [self.artifact.type_code_for_id(value) for value in root_ids]
        if len(root_types) != len(roots):
            raise ValueError("root_types and root_ids have different row counts")
        # If a root occurs multiple times in a batch, its encoder feature is identical;
        # use the first occurrence for the deduplicated root block.
        root_feature_map = {value: root_values[index] for index, value in enumerate(roots)}
        root_type_map = {value: int(root_types[index]) for index, value in enumerate(roots)}
        depth_ids: list[list[int]] = [unique_roots]
        depth_types: list[list[int]] = [[root_type_map[value] for value in unique_roots]]
        depth_features: list[np.ndarray] = [
            np.asarray([root_feature_map[value] for value in unique_roots], dtype=np.float32)
        ]
        transitions: list[dict[str, torch.Tensor]] = []
        for layer, fanout in enumerate(self.fanouts):
            parents = depth_ids[-1]
            children_by_relation: dict[str, list[list[int]]] = {}
            all_children: list[int] = []
            for relation in RELATIONS:
                rows = [
                    self._sample_row(parent, relation, fanout, rng, excluded)
                    for parent in parents
                ]
                children_by_relation[relation] = rows
                all_children.extend(value for row in rows for value in row)
            children = list(dict.fromkeys(all_children))
            child_index = {value: index for index, value in enumerate(children)}
            relation_indices: dict[str, torch.Tensor] = {}
            for relation, rows in children_by_relation.items():
                index_rows = [
                    [child_index[value] for value in row] for row in rows
                ]
                width = max((len(row) for row in index_rows), default=0)
                width = max(width, fanout)
                matrix = torch.full((len(parents), width), -1, dtype=torch.long)
                for row_index, row in enumerate(index_rows):
                    if row:
                        matrix[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
                relation_indices[relation] = matrix
            depth_ids.append(children)
            depth_types.append([self.artifact.type_code_for_id(value) for value in children])
            depth_features.append(self.artifact.feature_for_ids(children))
            transitions.append(relation_indices)
        return SampledGraph(
            ids_by_depth=[torch.tensor(values, dtype=torch.long) for values in depth_ids],
            features_by_depth=[torch.from_numpy(values) for values in depth_features],
            relation_indices=transitions,
            root_positions=root_positions,
            types_by_depth=[torch.tensor(values, dtype=torch.long) for values in depth_types],
        )
