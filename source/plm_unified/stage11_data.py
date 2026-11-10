"""Training-only positive relations, frozen caches and deterministic Stage11 batches."""

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .cache import EmbeddingCache
from .encoders import checkpoint_fingerprint
from .stage8_data import (BalancedFourTypeBatchSampler, CANONICAL_AMINO_ACIDS,
                          CachedStage8Dataset, Stage8AuditResult, _read_split,
                          audit_stage8_class_dirs)
class Stage11CacheCollator:
    def __init__(self, caches):
        self.caches = {entity: cache.worker_reader() for entity, cache in caches.items()}

    def __call__(self, samples):
        batch = {}
        for entity, cache in self.caches.items():
            hidden, mask = cache.get_many([getattr(sample, f"{entity}_locator") for sample in samples])
            batch[f"{entity}_hidden"] = torch.from_numpy(hidden)
            batch[f"{entity}_mask"] = torch.from_numpy(mask)
        for name in ("ph", "pht"):
            batch[f"{name}_labels"] = torch.tensor([getattr(sample, name) for sample in samples], dtype=torch.long)
        batch["label_types"] = [sample.label_type for sample in samples]
        batch["source_rows"] = torch.tensor([sample.source_row for sample in samples])
        return batch


def move_cached_batch(batch, device):
    # Float32 also supports CPU and CUDA runs without autocast; cache stays float16 on disk.
    return {key: value.to(device, dtype=torch.float32 if value.is_floating_point() else value.dtype,
                          non_blocking=device.type == "cuda").detach()
            if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@dataclass
class Stage11Data:
    audit: Stage8AuditResult
    ph_labels: dict
    pht_labels: dict
    ph_pairs: list
    pht_pairs: list
    fingerprint: str
    report: dict

    def sequences(self):
        result = sequences_for_records(self.audit.train_records + self.audit.val_records)
        for peptide, hla in self.ph_pairs:
            result["peptide"].add(peptide)
            result["hla"].add(hla)
        return result


def _insert_label(mapping, key, label, source):
    if key in mapping and mapping[key] != label:
        raise ValueError(f"Conflicting {source} labels for {key}")
    mapping[key] = label


def prepare_stage11_data(config):
    t, m = config.stage11, config.model
    audit = audit_stage8_class_dirs(config.paths.data_root / t.data_subdir, t.class_dirs,
                                    "train.csv", "val.csv", m,
                                    reject_noncanonical_sequences=t.reject_noncanonical_sequences)
    audit.require_clean()
    ph, pht = {}, {}
    for record in audit.train_records:
        _insert_label(ph, (record.peptide, record.hla), record.ph, "PHT-derived PH")
        _insert_label(pht, ((record.peptide, record.hla), record.tcr), record.pht, "PHT")
    validation_ph = {(r.peptide, r.hla) for r in audit.val_records}
    supplemental = {}
    rejected = []
    path = config.paths.data_root / t.ph_train_file
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"peptide", "HLA", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("PH train.csv requires peptide,HLA,label columns")
        rows = 0
        for number, row in enumerate(reader, start=2):
            rows += 1
            peptide, hla = row["peptide"].strip().upper(), row["HLA"].strip().upper()
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid PH label at {path}:{number}") from error
            if label not in {0, 1}:
                raise ValueError(f"Nonbinary PH label at {path}:{number}")
            if row.get("split_type", "train").strip() != "train":
                raise ValueError(f"Non-training row in PH train.csv:{number}")
            valid = m.peptide_min_length <= len(peptide) <= m.peptide_max_length and 1 <= len(hla) <= m.hla_max_length
            if t.reject_noncanonical_sequences:
                valid = valid and set(peptide + hla).issubset(CANONICAL_AMINO_ACIDS)
            if not valid:
                rejected.append({"row": number, "peptide": peptide, "HLA": hla})
                continue
            key = (peptide, hla)
            _insert_label(supplemental, key, label, "supplemental PH")
            if key in ph and ph[key] != label:
                raise ValueError(f"Cross-task PH label conflict at {path}:{number}: {key}")
    excluded = {key for key in supplemental if key in validation_ph}
    for key, label in supplemental.items():
        if key not in excluded:
            _insert_label(ph, key, label, "merged PH")
    ph_pairs = sorted(key for key, label in ph.items() if label == 1)
    pht_pairs = sorted(key for key, label in pht.items() if label == 1)
    if not ph_pairs or not pht_pairs:
        raise ValueError("Both contrastive tasks require training positives")
    digest = hashlib.sha256(audit.dataset_fingerprint.encode())
    digest.update(path.read_bytes())
    digest.update(json.dumps(sorted(ph.items()), separators=(",", ":")).encode())
    report = {"ph_input_rows": rows, "ph_invalid_rows": len(rejected),
              "ph_rejected_rows": rejected, "ph_unique_pairs": len(supplemental),
              "ph_validation_overlap_excluded": len(excluded),
              "ph_positive_validation_overlap_excluded": sum(supplemental[k] for k in excluded),
              "ph_positive_pairs": len(ph_pairs), "pht_positive_pairs": len(pht_pairs),
              "classification_train_rows": len(audit.train_records),
              "classification_val_rows": len(audit.val_records),
              "negative_policy": "in-batch non-positive; known negative and unlabeled counted separately"}
    return Stage11Data(audit, ph, pht, ph_pairs, pht_pairs, digest.hexdigest(), report)


def sequences_for_records(records):
    return {entity: {getattr(r, entity) for r in records} for entity in ("peptide", "hla", "tcr")}


def test_paths(config):
    root = config.paths.data_root / "PHT"
    return {"test": root / "test.csv", **{
        f"test_unseen_{entity}": root / "benchmarks" / f"test_unseen_{entity}.csv"
        for entity in ("peptide", "hla", "tcr", "triple")}}


def read_stage11_test(path, config):
    records, stats, conflicts, errors, rejected = _read_split(
        Path(path), "test", config.model, config.stage11.reject_noncanonical_sequences)
    if conflicts or errors:
        raise ValueError(f"Invalid test data {path}: {conflicts[:2]} {errors[:2]}")
    if not records:
        raise ValueError(f"No valid test samples in {path}")
    return records, {**stats, "rejected_rows": rejected}


def open_stage11_caches(config, *, scope="train", expected_metadata=None):
    caches = {}
    try:
        esmc = checkpoint_fingerprint(config.paths.esmc_model)
        fingerprints = {"peptide": esmc, "hla": esmc,
                        "tcr": checkpoint_fingerprint(config.paths.tcr_bert_model)}
        metadata = {}
        for entity, fingerprint in fingerprints.items():
            cache = EmbeddingCache(config.cache_dir(scope) / entity,
                                   expected_hidden_dim=getattr(config.model, f"{entity}_input_dim"),
                                   expected_fingerprint=fingerprint)
            caches[entity] = cache
            if cache.max_length != getattr(config.model, f"{entity}_max_length") or cache.manifest["dtype"] != config.cache.dtype:
                raise ValueError(f"Incompatible {entity} cache dimensions or dtype")
            metadata[entity] = {key: cache.manifest[key] for key in ("hidden_dim", "max_length", "dtype")}
            metadata[entity]["checkpoint_fingerprint"] = fingerprint
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError("Stage11 cache/PLM metadata differs from checkpoint")
        return caches, metadata
    except Exception:
        for cache in caches.values():
            cache.close()
        raise


def require_coverage(caches, sequences):
    for entity, values in sequences.items():
        for sequence in values:
            try:
                caches[entity].resolve(sequence)
            except KeyError as error:
                raise ValueError(f"Missing {entity} sequence; run Stage11 precompute for this scope") from error


class RelationDataset:
    """Locators are resolved in the parent process, avoiding worker SQLite lookups."""
    def __init__(self, pairs, caches, task):
        self.pairs = pairs
        self.samples = []
        for left, right in pairs:
            if task == "ph":
                locators = (caches["peptide"].resolve(left), caches["hla"].resolve(right))
            else:
                locators = (caches["peptide"].resolve(left[0]), caches["hla"].resolve(left[1]),
                            caches["tcr"].resolve(right))
            self.samples.append((left, right, locators))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ContrastiveCollator:
    def __init__(self, caches, labels, task):
        self.caches = {k: v.worker_reader() for k, v in caches.items() if task == "pht" or k != "tcr"}
        self.labels, self.task = labels, task

    def __call__(self, samples):
        left, right = {}, {}
        for a, b, locators in samples:
            left.setdefault(a, locators[:-1])
            right.setdefault(b, locators[-1])
        batch = {}
        entity_locators = {"peptide": [v[0] for v in left.values()]}
        if self.task == "ph":
            entity_locators["hla"] = list(right.values())
        else:
            entity_locators["hla"] = [v[1] for v in left.values()]
            entity_locators["tcr"] = list(right.values())
        for entity, locators in entity_locators.items():
            hidden, mask = self.caches[entity].get_many(locators)
            batch[f"{entity}_hidden"] = torch.from_numpy(hidden)
            batch[f"{entity}_mask"] = torch.from_numpy(mask)
        labels = torch.tensor([[self.labels.get((a, b), -1) for b in right] for a in left])
        batch["positive_mask"] = labels == 1
        batch["known_negative_pairs"] = int((labels == 0).sum())
        batch["unlabeled_pairs"] = int((labels == -1).sum())
        batch["sampled_relations"] = len(samples)
        return batch


class CyclingRelationSampler:
    """No-replacement PH traversal with a checkpointable permutation and cursor."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("PH relation pool cannot be empty")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor = 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.cursor = 0
            end = min(self.size, self.cursor + count - len(result))
            result.extend(self.order[self.cursor:end])
            self.cursor = end
        return result

    def state_dict(self):
        return {"size": self.size, "order": list(self.order), "cursor": self.cursor,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size or sorted(state["order"]) != list(range(self.size)):
            raise ValueError("PH sampler pool mismatch")
        if not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid PH sampler cursor")
        self.order, self.cursor = list(state["order"]), state["cursor"]
        self.generator.set_state(state["generator"].cpu())


def _loader(config, dataset, collator, *, seed, **kwargs):
    workers = config.runtime.num_workers
    if workers:
        # Spawned workers import torch and NumPy independently. MKL's default
        # Intel threading layer can abort when this environment loads libgomp.
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    return DataLoader(dataset, collate_fn=collator, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False,
                      multiprocessing_context="spawn" if workers else None,
                      generator=torch.Generator().manual_seed(seed), **kwargs)


def classification_loader(config, records, caches, *, training=False):
    dataset = CachedStage8Dataset(records, caches)
    kwargs = {"batch_sampler": BalancedFourTypeBatchSampler(
        dataset.type_indices, batch_size=config.stage11.batch_size, seed=config.runtime.seed)} if training else {
        "batch_size": config.stage11.eval_batch_size, "shuffle": False}
    return _loader(config, dataset, Stage11CacheCollator(caches), seed=config.runtime.seed + 10, **kwargs)


class Stage11Loaders:
    def __init__(self, config, data, caches):
        self.config = config
        self.classification = classification_loader(config, data.audit.train_records, caches, training=True)
        self.validation = classification_loader(config, data.audit.val_records, caches)
        self.datasets = {task: RelationDataset(getattr(data, f"{task}_pairs"), caches, task) for task in ("ph", "pht")}
        self.collators = {task: ContrastiveCollator(caches, getattr(data, f"{task}_labels"), task) for task in ("ph", "pht")}
        self.ph_sampler = CyclingRelationSampler(len(data.ph_pairs), config.runtime.seed + 100)

    def contrastive(self, task, round_index):
        config = self.config
        size = config.stage11.contrastive_batch_size
        # Match the PHT pass's batch sizes, including its short final batch.
        count = len(self.datasets["pht"])
        if task == "ph":
            indices = self.ph_sampler.take(count)
        else:
            generator = torch.Generator().manual_seed(config.runtime.seed + 1_000_003 * round_index)
            indices = torch.randperm(count, generator=generator).tolist()
        batches = [indices[i:i + size] for i in range(0, count, size)]
        return _loader(config, self.datasets[task], self.collators[task],
                       seed=config.runtime.seed + round_index, batch_sampler=batches)

    def state_dict(self):
        return {"ph_sampler": self.ph_sampler.state_dict(),
                "classification_rng": self.classification.generator.get_state(),
                "validation_rng": self.validation.generator.get_state()}

    def load_state_dict(self, state):
        self.ph_sampler.load_state_dict(state["ph_sampler"])
        self.classification.generator.set_state(state["classification_rng"].cpu())
        self.validation.generator.set_state(state["validation_rng"].cpu())

    def close(self):
        for collator in (*self.collators.values(), self.classification.collate_fn, self.validation.collate_fn):
            for cache in collator.caches.values():
                cache.close()


def model_inputs(batch, *, task="pht"):
    entities = ("peptide", "hla") if task == "ph" else ("peptide", "hla", "tcr")
    return tuple(batch[f"{entity}_{part}"] for entity in entities for part in ("hidden", "mask"))
